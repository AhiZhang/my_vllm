"""一个用于学习的 Decoder-only Transformer，仅依赖 PyTorch。

运行示例（WSL）：
    python /mnt/e/ai_infra/my_vllm/reference/decoder_only.py --trace
    python /mnt/e/ai_infra/my_vllm/reference/decoder_only.py --no-kv-cache

权重随机初始化，输出仅用于验证计算流程，没有语言能力。这是采用可学习位置
embedding 的教学模型；后续加载预训练权重时，需要匹配其模型结构与 tokenizer。
生成默认启用逐层 KV cache：先 prefill 整段输入，再每轮只计算新 token。
缓存通过参数传入并返回，不放在模型全局状态中；训练 forward 仍默认返回 logits。
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass

import torch
from torch import nn


KVCache = tuple[torch.Tensor, torch.Tensor]  # 每层 K/V：[B, H, cached_tokens, Dh]


def trace_tensor(name: str, tensor: torch.Tensor, enabled: bool) -> None:
    if enabled:
        print(f"  {name:<35} {tuple(tensor.shape)}")


class ByteTokenizer:
    """UTF-8 字节对应 ID 0..255，另外保留 EOS 和 BOS，无需下载词表。"""

    eos_token_id = 256
    bos_token_id = 257
    vocab_size = 258

    def encode(self, text: str) -> list[int]:
        # BOS 也使空字符串具有一个有效输入 token。
        return [self.bos_token_id, *text.encode("utf-8")]

    def decode(self, token_ids: list[int]) -> str:
        content = []
        for token_id in token_ids:
            if token_id in (self.bos_token_id, self.eos_token_id):
                continue
            if not 0 <= token_id < 256:
                raise ValueError(f"无效的 token ID：{token_id}")
            content.append(token_id)
        # 随机模型可能生成不完整或无效的 UTF-8 字节，用替代字符显示。
        return bytes(content).decode("utf-8", errors="replace")


@dataclass
class ModelConfig:
    vocab_size: int = ByteTokenizer.vocab_size
    hidden_size: int = 128
    num_heads: int = 4
    num_layers: int = 2
    intermediate_size: int = 512
    max_seq_len: int = 256
    rms_norm_eps: float = 1e-6

    def __post_init__(self) -> None:
        sizes = (self.vocab_size, self.hidden_size, self.num_heads,
                 self.num_layers, self.intermediate_size, self.max_seq_len)
        if any(size <= 0 for size in sizes):
            raise ValueError("模型的词表、层数和各个维度必须为正数。")
        if self.hidden_size % self.num_heads != 0:
            raise ValueError("hidden_size 必须能被 num_heads 整除。")
        if self.rms_norm_eps <= 0:
            raise ValueError("rms_norm_eps 必须大于 0。")


class RMSNorm(nn.Module):
    """RMSNorm(x) = weight * x / sqrt(mean(x**2) + eps)。"""

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_float = x.float()
        inv_rms = torch.rsqrt(x_float.square().mean(dim=-1, keepdim=True) + self.eps)
        return (x_float * inv_rms).to(x.dtype) * self.weight


class CausalSelfAttention(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.num_heads = config.num_heads
        self.head_dim = config.hidden_size // config.num_heads
        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.out_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)

    def forward(self, x: torch.Tensor, trace: bool = False, *,
                past_key_value: KVCache | None = None, use_cache: bool = False
                ) -> torch.Tensor | tuple[torch.Tensor, KVCache]:
        batch, length, hidden = x.shape
        # [B, T, D] -> [B, H, T, Dh]，其中 D = H * Dh。
        def split_heads(tensor: torch.Tensor) -> torch.Tensor:
            return tensor.reshape(batch, length, self.num_heads, self.head_dim).transpose(1, 2)

        q = split_heads(self.q_proj(x))
        k = split_heads(self.k_proj(x))
        v = split_heads(self.v_proj(x))
        past_length = 0
        if past_key_value is not None:
            past_k, past_v = past_key_value
            if past_k.dtype != k.dtype or past_v.dtype != v.dtype:
                raise ValueError("缓存与本次 K/V 的 dtype 不同，请在相同精度下创建和使用缓存。")
            past_length = past_k.size(2)
            k = torch.cat((past_k, k), dim=2)
            v = torch.cat((past_v, v), dim=2)
        for name, tensor in (("Q", q), ("K", k), ("V", v)):
            trace_tensor(name, tensor, trace)

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        # query 的绝对位置从 past_length 起算，key 包含历史和本次输入。
        # 不能直接对 [new_tokens, total_tokens] 使用普通 triu(1)。
        query_positions = past_length + torch.arange(length, device=x.device)
        key_positions = torch.arange(k.size(2), device=x.device)
        future_mask = key_positions[None, :] > query_positions[:, None]
        scores = scores.masked_fill(future_mask, float("-inf"))
        probabilities = torch.softmax(scores.float(), dim=-1).to(q.dtype)
        trace_tensor("masked attention scores", scores, trace)
        trace_tensor("attention probabilities", probabilities, trace)

        context = probabilities @ v
        trace_tensor("attention context", context, trace)
        # [B, H, T, Dh] -> [B, T, D]。
        context = context.transpose(1, 2).contiguous().reshape(batch, length, hidden)
        output = self.out_proj(context)
        trace_tensor("attention output", output, trace)
        return (output, (k, v)) if use_cache else output


class MLP(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.activation = nn.GELU()
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x: torch.Tensor, trace: bool = False) -> torch.Tensor:
        x = self.activation(self.up_proj(x))
        trace_tensor("MLP hidden (after GELU)", x, trace)
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

    def forward(self, x: torch.Tensor, trace: bool = False, *,
                past_key_value: KVCache | None = None, use_cache: bool = False
                ) -> torch.Tensor | tuple[torch.Tensor, KVCache]:
        normalized = self.attention_norm(x)
        trace_tensor("attention RMSNorm", normalized, trace)
        attended = self.attention(normalized, trace=trace,
                                  past_key_value=past_key_value, use_cache=use_cache)
        if use_cache:
            attended, present = attended
        x = x + attended
        trace_tensor("attention residual", x, trace)
        normalized = self.mlp_norm(x)
        trace_tensor("MLP RMSNorm", normalized, trace)
        x = x + self.mlp(normalized, trace=trace)
        trace_tensor("MLP residual", x, trace)
        return (x, present) if use_cache else x


class DecoderOnlyTransformer(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        # Attention 本身不知道顺序，所以需要提供位置信息。
        self.position_embedding = nn.Embedding(config.max_seq_len, config.hidden_size)
        self.blocks = nn.ModuleList(TransformerBlock(config) for _ in range(config.num_layers))
        self.final_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.apply(self._initialize_weights)

    @staticmethod
    def _initialize_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids: torch.Tensor, trace: bool = False, *,
                past_key_values: list[KVCache] | None = None, use_cache: bool = False
                ) -> torch.Tensor | tuple[torch.Tensor, list[KVCache]]:
        """默认返回 logits；use_cache=True 时返回 (logits, 每层 K/V)。

        有缓存时 input_ids 只含尚未处理的新 token，位置编码接着历史长度继续。
        缓存仅供同一请求在 eval 模式使用，不改变模型参数或 checkpoint 格式。
        """
        if input_ids.ndim != 2 or input_ids.dtype != torch.long:
            raise ValueError("input_ids 必须是形状 [B, T] 的 torch.long 张量。")
        if input_ids.size(0) == 0 or not 1 <= input_ids.size(1) <= self.config.max_seq_len:
            raise ValueError(f"需要非空 batch，且序列长度在 1..{self.config.max_seq_len} 之间。")
        if past_key_values is not None and not use_cache:
            raise ValueError("传入 past_key_values 时必须开启 use_cache。")
        if use_cache and self.training:
            raise ValueError("KV cache 仅用于推理，请先调用 model.eval()。")
        past_length = 0
        if past_key_values is not None:
            if len(past_key_values) != len(self.blocks):
                raise ValueError("缓存层数必须与 Transformer block 数量一致。")
            for index, (key, value) in enumerate(past_key_values):
                if key.ndim != 4 or key.shape != value.shape:
                    raise ValueError("每层缓存必须是形状相同的四维 K/V 张量。")
                if index == 0:
                    past_length = key.size(2)
                expected = (input_ids.size(0), self.config.num_heads, past_length,
                            self.config.hidden_size // self.config.num_heads)
                if key.shape != expected or key.device != input_ids.device or value.device != input_ids.device:
                    raise ValueError("缓存的 batch、head、历史长度、head_dim 或设备不匹配。")
        if past_length + input_ids.size(1) > self.config.max_seq_len:
            raise ValueError("历史缓存加新输入超出 max_seq_len。")

        trace_tensor("input IDs", input_ids, trace)
        positions = torch.arange(past_length, past_length + input_ids.size(1),
                                 device=input_ids.device).unsqueeze(0)
        token_embeddings = self.token_embedding(input_ids)
        position_embeddings = self.position_embedding(positions)
        trace_tensor("token embeddings", token_embeddings, trace)
        trace_tensor("position embeddings", position_embeddings, trace)
        x = token_embeddings + position_embeddings
        present_key_values = []
        for index, block in enumerate(self.blocks):
            if trace:
                print(f"Transformer block {index}:")
            result = block(x, trace=trace, use_cache=use_cache,
                           past_key_value=past_key_values[index] if past_key_values is not None else None)
            if use_cache:
                x, present = result
                present_key_values.append(present)
            else:
                x = result
        x = self.final_norm(x)
        trace_tensor("final RMSNorm", x, trace)
        logits = self.lm_head(x)
        trace_tensor("logits", logits, trace)
        return (logits, present_key_values) if use_cache else logits


@torch.inference_mode()
def generate(
    model: DecoderOnlyTransformer,
    input_ids: torch.Tensor,
    max_new_tokens: int = 32,
    eos_token_id: int | None = ByteTokenizer.eos_token_id,
    trace: bool = False,
    use_cache: bool = True,
) -> torch.Tensor:
    """单请求 greedy decoding，返回包含原输入和生成部分的 [1, T_out]。"""
    if input_ids.ndim != 2 or input_ids.size(0) != 1 or input_ids.dtype != torch.long:
        raise ValueError("第一版生成只接受形状 [1, T] 的 torch.long 输入。")
    if not 1 <= input_ids.size(1) <= model.config.max_seq_len:
        raise ValueError("输入长度超出模型支持的上下文范围。")
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens 不能为负数。")

    model.eval()
    generated = input_ids.clone()
    past_key_values = None
    for step in range(max_new_tokens):
        if generated.size(1) >= model.config.max_seq_len:
            break
        # prefill 输入整个 prompt；之后 decode 每次只输入上轮生成的一个 token。
        current_ids = generated[:, -1:] if past_key_values is not None else generated
        if use_cache:
            logits, past_key_values = model(current_ids, trace=trace and step < 2,
                                            past_key_values=past_key_values, use_cache=True)
        else:
            logits = model(current_ids, trace=trace and step == 0)
        last_logits = logits[:, -1, :]  # [1, V]，用于预测下一个位置。
        next_token = last_logits.argmax(dim=-1, keepdim=True)  # [1, 1]。
        trace_tensor("last position logits", last_logits, trace and step == 0)
        trace_tensor("greedy next token", next_token, trace and step == 0)
        generated = torch.cat((generated, next_token), dim=1)
        token_id = next_token.item()
        if trace:
            cached = past_key_values[0][0].size(2) if past_key_values is not None else 0
            print(f"Step {step + 1:02d}: input_tokens={current_ids.size(1)}, cached_tokens={cached}, "
                  f"next_token_id={token_id}, sequence_length={generated.size(1)}")
        if eos_token_id is not None and token_id == eos_token_id:
            break
    return generated


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--text", default="The capital of France is")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--max-seq-len", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--trace", action="store_true", help="打印 prefill、首次 decode 的形状和每步 token")
    parser.add_argument("--no-kv-cache", dest="use_cache", action="store_false",
                        help="关闭 KV cache，每轮重算完整上下文以供对照")
    args = parser.parse_args()
    if args.max_new_tokens < 0 or args.max_seq_len <= 0:
        parser.error("max-new-tokens 必须非负，max-seq-len 必须为正。")

    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA 不可用；可使用 --device cpu。")

    tokenizer = ByteTokenizer()
    token_ids = tokenizer.encode(args.text)
    if len(token_ids) > args.max_seq_len:
        parser.error(f"输入含 BOS 共 {len(token_ids)} 个 token，超过 max-seq-len。")
    torch.manual_seed(args.seed)
    config = ModelConfig(vocab_size=tokenizer.vocab_size, max_seq_len=args.max_seq_len)
    model = DecoderOnlyTransformer(config).to(device)
    input_ids = torch.tensor([token_ids], dtype=torch.long, device=device)
    print("随机初始化模型：输出仅用于验证实现，不具有语言能力。")
    print(f"Device: {device}; parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Input text: {args.text!r}")
    print(f"Input token IDs: {token_ids}")
    print(f"KV cache: {args.use_cache}")
    output_ids = generate(model, input_ids, args.max_new_tokens, tokenizer.eos_token_id,
                          trace=args.trace, use_cache=args.use_cache)
    output_tokens = output_ids[0].tolist()
    new_tokens = output_tokens[len(token_ids):]
    print(f"Generated token IDs: {new_tokens}")
    print(f"Generated token count: {len(new_tokens)}")
    if new_tokens and new_tokens[-1] == tokenizer.eos_token_id:
        reason = "EOS"
    elif len(output_tokens) >= config.max_seq_len:
        reason = "max_seq_len"
    else:
        reason = "max_new_tokens"
    print(f"Stop reason: {reason}")
    # repr 保留文本内容，也避免随机字节中的控制字符影响终端显示。
    print(f"Output text: {tokenizer.decode(output_tokens)!r}")


if __name__ == "__main__":
    main()
