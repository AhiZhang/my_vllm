import argparse
import math
from dataclasses import dataclass

import torch
from torch import nn

def trace_tensor(name: str, tensor: torch.Tensor, enabled: bool) -> None:
    if enabled:
        print(f"{name}{tuple(tensor.shape)}")

class ByteTokenizer:

    eos_token_id = 256
    bos_token_id = 257
    vocab_size = 256 + 2

    def encode(self, text: str) -> list[int]:
        return [self.bos_token_id, *text.encode("utf-8")]

    def decode(self, token_ids: list[int]) -> str:
        content = []
        for token_id in token_ids:
            if token_id in (self.bos_token_id, self.eos_token_id):
                continue
            if not 0 <= token_id < 256:
                raise ValueError(f"根本就没有这样的Token Id")
            content.append(token_id)

        return bytes(content).decode("utf-8")

@dataclass
class ModelConfig:
    vocab_size: int = ByteTokenizer.vocab_size
    hidden_size: int = 128
    nums_heads: int = 4
    num_layers: int = 2
    intermediate_size: int = 512
    max_seq_len: int = 256
    rms_norm_eps: float = 1e-6

    def __post_init__(self) -> None:
        sizes = (self.vocab_size, self.hidden_size, self.nums_heads, 
                 self.num_layers, self.intermediate_size, self.max_seq_len)

        if any(size < 0 for size in sizes):
            raise ValueError("模型的词表, 层数, 各个维度必须为正数")

        if self.hidden_size % self.nums_heads != 0:
            raise ValueError("hidden_size 必须能被 nums_heads 整除")

        if self.rms_norm_eps <= 0:
            raise ValueError("rms_norm_eps必须大于0")

class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x_float = x.float()
            inv_rms = torch.rsqrtr(x_float.sqare().mean(dim=-1, keepdim=True) + self.eps)
            return (x_float * inv_rms).to(x.dtype) * self.weight

class CausalSelfAttention(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.num_heads = config.nums_heads
        self.head_dim = config.hidden_size // config.nums_heads
        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.out_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)

    def forward(self, x: torch.Tensor, trace: bool = False) -> torch.Tensor:
        batch, length, hidden = x.shape

        def split_heads(tensor: torch.Tensor) -> torch.Tensor:
            # 为什么要用transpose交换1, 2的顺序?
            return tensor.reshape(batch, length, self.num_heads, self.head_dim).transpose(1, 2)

        q = split_heads(self.q_proj(x))
        k = split_heads(self.k_proj(x))
        v = split_heads(self.v_proj(x))

        scores = (q @ k.transpose(-2, -1) / math.sqrt(self.head_dim))
        future_mask = torch.ones(length, length, dtype=torch.bool, device=x.device).triu(1)
        scores = scores.masked_fill(future_mask, float('-inf'))
        probabilities = torch.softmax(scores.float(), dim=-1).to(q.dtype)
        trace_tensor("Masked attention socres", scores, trace)

        context = context.transpose(1, 2).contiguous().reshape(batch, length, hidden)
        output = self.out_proj(context)
        trace_tensor("attention output", output, trace)
        return output

class MLP(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.activation = nn.GELU()
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x: torch.Tensor, trace: bool = False) -> torch.Tensor:
        x = self.activation(self.up_proj(x))
        trace_tensor("MLP hidden atfer GELU", x, trace)
        x = self.down_proj(x)
        trace_tensor("MLP output", x, trace)

        return x

class TransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.attention_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.attention = CausalSelfAttention(config)
        self.mlp_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor, trace: bool = False) -> torch.Tensor:
        normalized = self.attention_norm(x)
        trace_tensor("Attention RMSNorm", normalized, trace)
        x = x + self.attention_norm(normalized, trace = trace)
        normalized = self.mlp_norm(x)
        x = x + self.mlp(normalized, trace)
        return x

class DecoderOnlyTransformation(nn.module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embedding = nn.embedding(config.max_seq_len, config.hidden_size)
        self.blocks = nn.ModuleList(TransformerBlock(config) for _ in range(config.num_layers))
        self.final_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.apply(self._initialize_weights)

    @staticmethod
    def _initialize_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids: torch.Tensor, trace: bool=False) -> torch.Tensor:
        positions = torch.arange