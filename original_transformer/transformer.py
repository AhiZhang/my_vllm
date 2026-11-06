"""《Attention Is All You Need》的 Encoder–Decoder Transformer 教学实现。

只依赖 PyTorch；显式实现多头注意力，不使用 nn.Transformer / nn.MultiheadAttention。
结构遵循原论文：正弦位置编码、缩放 embedding、Post-LN、ReLU FFN、残差 dropout，
并在共享词表下共享源/目标 embedding 和输出投影权重。

WSL 运行（使用已有 nanovllm-ref 环境）：
    python /mnt/e/ai_infra/my_vllm/original_transformer/transformer.py --trace
    python /mnt/e/ai_infra/my_vllm/original_transformer/transformer.py --preset base --trace
    python /mnt/e/ai_infra/my_vllm/original_transformer/transformer.py --demo-train-steps 300 --text hello

默认 small：2 层 encoder + 2 层 decoder，d_model=128，4 heads，d_ff=512。
base：论文的 6+6 层、d_model=512、8 heads、d_ff=2048、dropout=0.1。
为免下载数据，tokenizer 使用 UTF-8 字节而非论文的 BPE，因此不是论文实验复现，
也不兼容 decoder_only.py 的词表和 checkpoint。默认随机权重没有翻译能力。
可选训练仅记忆四个固定句对，学习率固定，不保存权重，不代表泛化翻译能力。
生成默认缓存 decoder 历史 self-attention K/V 和每层固定的 cross-attention K/V；
--no-kv-cache 可关闭缓存进行对照。Encoder 仍一次性处理完整源文本。

阅读顺序：Tokenizer -> PositionalEncoding -> MultiHeadAttention -> FeedForward
          -> EncoderLayer / DecoderLayer -> Transformer -> generate / main。
论文：https://arxiv.org/abs/1706.03762
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


KVCache = tuple[torch.Tensor, torch.Tensor]  # [B, H, tokens, Dh]


@dataclass
class DecoderLayerCache:
    self_key_value: KVCache
    cross_key_value: KVCache


@dataclass
class DecoderCache:
    layers: list[DecoderLayerCache]
    target_padding_mask: torch.Tensor  # [B, past_tokens]，True 表示 PAD


def trace_tensor(name: str, tensor: torch.Tensor, enabled: bool) -> None:
    if enabled:
        print(f"  {name:<35} {tuple(tensor.shape)}")


class ByteTokenizer:
    """源语言和目标语言共用字节词表；PAD / BOS / EOS 与普通字节分开。"""

    pad_token_id = 256
    bos_token_id = 257
    eos_token_id = 258
    vocab_size = 259

    def encode(self, text: str) -> list[int]:
        return [self.bos_token_id, *text.encode("utf-8"), self.eos_token_id]

    def decode(self, token_ids: list[int]) -> str:
        content = []
        for token_id in token_ids:
            if token_id == self.eos_token_id:
                break
            if token_id in (self.pad_token_id, self.bos_token_id):
                continue
            if not 0 <= token_id < 256:
                raise ValueError(f"无效的 token ID：{token_id}")
            content.append(token_id)
        return bytes(content).decode("utf-8", errors="replace")


def tokenize_batch(texts: list[str], device: torch.device) -> torch.Tensor:
    """将不同长度文本右侧补 PAD，得到 [B, T]。"""
    if not texts:
        raise ValueError("texts 不能为空。")
    tokenizer = ByteTokenizer()
    sequences = [tokenizer.encode(text) for text in texts]
    result = torch.full((len(texts), max(map(len, sequences))),
                        tokenizer.pad_token_id, dtype=torch.long, device=device)
    for row, ids in enumerate(sequences):
        result[row, :len(ids)] = torch.tensor(ids, dtype=torch.long, device=device)
    return result


@dataclass
class TransformerConfig:
    vocab_size: int = ByteTokenizer.vocab_size
    d_model: int = 128
    num_heads: int = 4
    num_encoder_layers: int = 2
    num_decoder_layers: int = 2
    d_ff: int = 512
    max_seq_len: int = 256
    dropout: float = 0.1
    layer_norm_eps: float = 1e-6
    pad_token_id: int = ByteTokenizer.pad_token_id

    def __post_init__(self) -> None:
        sizes = (self.vocab_size, self.d_model, self.num_heads,
                 self.num_encoder_layers, self.num_decoder_layers,
                 self.d_ff, self.max_seq_len)
        if any(size <= 0 for size in sizes):
            raise ValueError("词表、层数和各个维度必须为正数。")
        if self.d_model % self.num_heads:
            raise ValueError("d_model 必须能被 num_heads 整除。")
        if not 0 <= self.dropout < 1 or not self.layer_norm_eps > 0:
            raise ValueError("需要 0 <= dropout < 1 且 layer_norm_eps > 0。")
        if not 0 <= self.pad_token_id < self.vocab_size:
            raise ValueError("pad_token_id 必须在词表中。")


class PositionalEncoding(nn.Module):
    """PE(pos,2i)=sin(pos/10000**(2i/D))，奇数维使用 cos；没有可训练参数。"""

    def __init__(self, config: TransformerConfig):
        super().__init__()
        positions = torch.arange(config.max_seq_len, dtype=torch.float32)[:, None]
        frequencies = torch.exp(torch.arange(0, config.d_model, 2).float()
                                * (-math.log(10000.0) / config.d_model))
        angles = positions * frequencies
        encoding = torch.zeros(config.max_seq_len, config.d_model)
        encoding[:, 0::2] = angles.sin()
        encoding[:, 1::2] = angles[:, :config.d_model // 2].cos()
        # buffer 会随 model.to(device) 移动，但不会被 optimizer 更新。
        self.register_buffer("encoding", encoding[None, :, :])
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor, offset: int = 0) -> torch.Tensor:
        if offset < 0 or offset + x.size(1) > self.encoding.size(1):
            raise ValueError("位置编码偏移加新序列长度超出 max_seq_len。")
        return self.dropout(x + self.encoding[:, offset:offset + x.size(1)].to(x.dtype))


class MultiHeadAttention(nn.Module):
    """三种 attention 共用实现；mask 的 True 一律表示“禁止关注”。"""

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.num_heads = config.num_heads
        self.head_dim = config.d_model // config.num_heads
        self.q_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.k_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.v_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.out_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                mask: torch.Tensor, trace: bool = False, *,
                past_key_value: KVCache | None = None, use_cache: bool = False,
                static_kv: bool = False) -> torch.Tensor | tuple[torch.Tensor, KVCache]:
        batch, query_len, hidden = query.shape

        def split_heads(x: torch.Tensor) -> torch.Tensor:
            # Q 的长度是 Tq，K/V 的长度是 Tk；交叉注意力时两者可以不同。
            return x.reshape(batch, -1, self.num_heads, self.head_dim).transpose(1, 2)

        q = split_heads(self.q_proj(query))   # [B, H, Tq, Dh]
        if static_kv and past_key_value is not None:
            # Encoder memory 不变：后续轮次连 K/V 的线性投影也可以跳过。
            k, v = past_key_value
        else:
            k = split_heads(self.k_proj(key))     # [B, H, Tk, Dh]
            v = split_heads(self.v_proj(value))
            if past_key_value is not None:
                past_k, past_v = past_key_value
                if past_k.dtype != k.dtype or past_v.dtype != v.dtype:
                    raise ValueError("缓存与当前计算精度不同，请重新创建缓存。")
                k = torch.cat((past_k, k), dim=2)
                v = torch.cat((past_v, v), dim=2)
        if k.dtype != q.dtype or v.dtype != q.dtype:
            raise ValueError("缓存与当前 Q 的精度不同，请重新创建缓存。")
        for name, tensor in (("Q", q), ("K", k), ("V", v), ("mask (True = blocked)", mask)):
            trace_tensor(name, tensor, trace)

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        scores = scores.masked_fill(mask, float("-inf"))  # mask 广播到 head 维
        probabilities = torch.softmax(scores.float(), dim=-1).to(q.dtype)
        trace_tensor("attention probabilities", probabilities, trace)
        context = self.dropout(probabilities) @ v        # [B, H, Tq, Dh]
        context = context.transpose(1, 2).contiguous().reshape(batch, query_len, hidden)
        output = self.out_proj(context)
        trace_tensor("attention output", output, trace)
        return (output, (k, v)) if use_cache else output


class FeedForward(nn.Module):
    """每个 token 独立做相同的两层变换：max(0, xW1+b1)W2+b2。"""

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.up_proj = nn.Linear(config.d_model, config.d_ff)
        self.down_proj = nn.Linear(config.d_ff, config.d_model)

    def forward(self, x: torch.Tensor, trace: bool = False) -> torch.Tensor:
        x = F.relu(self.up_proj(x))
        trace_tensor("FFN hidden (after ReLU)", x, trace)
        return self.down_proj(x)


class EncoderLayer(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.self_attention = MultiHeadAttention(config)
        self.ffn = FeedForward(config)
        self.norm1 = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)
        self.norm2 = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor, source_mask: torch.Tensor,
                trace: bool = False) -> torch.Tensor:
        # Encoder 无 causal mask，任何非 PAD 位置都可以互相关注。
        attended = self.self_attention(x, x, x, source_mask, trace)
        x = self.norm1(x + self.dropout(attended))        # Post-LN: LN(x + sublayer(x))
        trace_tensor("encoder self-attention Add & Norm", x, trace)
        x = self.norm2(x + self.dropout(self.ffn(x, trace)))
        trace_tensor("encoder FFN Add & Norm", x, trace)
        return x


class DecoderLayer(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.self_attention = MultiHeadAttention(config)
        self.cross_attention = MultiHeadAttention(config)
        self.ffn = FeedForward(config)
        self.norm1 = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)
        self.norm2 = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)
        self.norm3 = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor, memory: torch.Tensor,
                target_mask: torch.Tensor, source_mask: torch.Tensor,
                trace: bool = False, *, past_key_value: DecoderLayerCache | None = None,
                use_cache: bool = False) -> torch.Tensor | tuple[torch.Tensor, DecoderLayerCache]:
        if trace:
            print("  [decoder masked self-attention]")
        attended = self.self_attention(x, x, x, target_mask, trace,
                                      past_key_value=past_key_value.self_key_value if past_key_value else None,
                                      use_cache=use_cache)
        if use_cache:
            attended, self_kv = attended
        x = self.norm1(x + self.dropout(attended))
        trace_tensor("decoder self-attention Add & Norm", x, trace)
        if trace:
            print("  [decoder cross-attention: Q=decoder, K/V=encoder]")
        attended = self.cross_attention(x, memory, memory, source_mask, trace,
                                       past_key_value=past_key_value.cross_key_value if past_key_value else None,
                                       use_cache=use_cache, static_kv=True)
        if use_cache:
            attended, cross_kv = attended
        x = self.norm2(x + self.dropout(attended))
        trace_tensor("decoder cross-attention Add & Norm", x, trace)
        x = self.norm3(x + self.dropout(self.ffn(x, trace)))
        trace_tensor("decoder FFN Add & Norm", x, trace)
        return (x, DecoderLayerCache(self_kv, cross_kv)) if use_cache else x


class Transformer(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.d_model,
                                      padding_idx=config.pad_token_id)
        self.position_encoding = PositionalEncoding(config)
        self.encoder_layers = nn.ModuleList(EncoderLayer(config)
                                           for _ in range(config.num_encoder_layers))
        self.decoder_layers = nn.ModuleList(DecoderLayer(config)
                                           for _ in range(config.num_decoder_layers))
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.apply(self._initialize_weights)
        # 共用同一个 Parameter，而非只把初值复制过去。
        self.lm_head.weight = self.embedding.weight

    @staticmethod
    def _initialize_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=module.embedding_dim ** -0.5)
            with torch.no_grad():
                module.weight[module.padding_idx].zero_()

    def _check_ids(self, ids: torch.Tensor, name: str, allow_initial_pad: bool = False) -> None:
        if ids.ndim != 2 or ids.dtype != torch.long:
            raise ValueError(f"{name} 必须是 [B, T] 的 torch.long 张量。")
        if ids.size(0) == 0 or not 1 <= ids.size(1) <= self.config.max_seq_len:
            raise ValueError(f"{name} 需要非空 batch，长度须在 1..{self.config.max_seq_len}。")
        # 使用右侧 padding；首 token 有效，保证任何 query 至少能关注一个 key。
        if not allow_initial_pad and (ids[:, 0] == self.config.pad_token_id).any():
            raise ValueError(f"{name} 每行必须以有效 token 开头，请使用右侧 padding。")

    def _embed(self, ids: torch.Tensor, trace: bool, offset: int = 0) -> torch.Tensor:
        x = self.embedding(ids) * math.sqrt(self.config.d_model)
        trace_tensor("embedding * sqrt(d_model)", x, trace)
        x = self.position_encoding(x, offset=offset)
        trace_tensor("embedding + sinusoidal PE + dropout", x, trace)
        return x

    def source_mask(self, source_ids: torch.Tensor) -> torch.Tensor:
        return (source_ids == self.config.pad_token_id)[:, None, None, :]  # [B,1,1,S]

    def encode(self, source_ids: torch.Tensor, trace: bool = False) -> torch.Tensor:
        self._check_ids(source_ids, "source_ids")
        x = self._embed(source_ids, trace)
        mask = self.source_mask(source_ids)
        for index, layer in enumerate(self.encoder_layers, 1):
            if trace:
                print(f"[Encoder layer {index}]")
            x = layer(x, mask, trace)
        trace_tensor("encoder memory", x, trace)
        return x

    def decode(self, target_ids: torch.Tensor, memory: torch.Tensor,
               source_mask: torch.Tensor, trace: bool = False, *,
               cache: DecoderCache | None = None, use_cache: bool = False
               ) -> torch.Tensor | tuple[torch.Tensor, DecoderCache]:
        """训练默认返回 logits；缓存推理返回 (新 token logits, DecoderCache)。

        每个 cache 只能对应同一条源文本/同一批请求；换源文本时传入 cache=None。
        """
        if cache is not None and not use_cache:
            raise ValueError("传入 cache 时必须开启 use_cache。")
        if use_cache and self.training:
            raise ValueError("KV cache 仅用于推理，请先调用 model.eval()。")
        past_length = 0
        if cache is not None:
            padding = cache.target_padding_mask
            if padding.ndim != 2 or padding.dtype != torch.bool or padding.size(0) != target_ids.size(0):
                raise ValueError("历史 target padding mask 必须是 [B, past_tokens] 的布尔张量。")
            if padding.device != target_ids.device or len(cache.layers) != len(self.decoder_layers):
                raise ValueError("缓存设备或层数与本次请求不匹配。")
            past_length = padding.size(1)
            if past_length and padding[:, 0].any():
                raise ValueError("缓存中的首个目标 token 必须有效。")
            for layer in cache.layers:
                for kv, tokens in ((layer.self_key_value, past_length),
                                   (layer.cross_key_value, memory.size(1))):
                    key, value = kv
                    expected = (target_ids.size(0), self.config.num_heads, tokens,
                                self.config.d_model // self.config.num_heads)
                    if key.shape != expected or value.shape != expected:
                        raise ValueError("缓存的 batch、head、token 长度或 head_dim 不匹配。")
                    if key.device != target_ids.device or value.device != target_ids.device:
                        raise ValueError("缓存与输入设备不一致。")
        self._check_ids(target_ids, "target_ids", allow_initial_pad=past_length > 0)
        length = target_ids.size(1)
        if past_length + length > self.config.max_seq_len:
            raise ValueError("历史缓存加新输入超出 max_seq_len。")
        query_positions = past_length + torch.arange(length, device=target_ids.device)
        key_positions = torch.arange(past_length + length, device=target_ids.device)
        future_mask = key_positions[None, :] > query_positions[:, None]
        padding_mask = target_ids == self.config.pad_token_id
        if cache is not None:
            padding_mask = torch.cat((cache.target_padding_mask, padding_mask), dim=1)
        target_mask = padding_mask[:, None, None, :] | future_mask[None, None, :, :]
        x = self._embed(target_ids, trace, offset=past_length)
        present_layers = []
        for index, layer in enumerate(self.decoder_layers, 1):
            if trace:
                print(f"[Decoder layer {index}]")
            result = layer(x, memory, target_mask, source_mask, trace, use_cache=use_cache,
                           past_key_value=cache.layers[index - 1] if cache is not None else None)
            if use_cache:
                x, present = result
                present_layers.append(present)
            else:
                x = result
        # 各子层已做 Post-LN，不再额外添加一次 final norm。
        logits = self.lm_head(x)
        trace_tensor("LM head logits", logits, trace)
        return (logits, DecoderCache(present_layers, padding_mask)) if use_cache else logits

    def forward(self, source_ids: torch.Tensor, target_ids: torch.Tensor,
                trace: bool = False) -> torch.Tensor:
        """source_ids [B,S] + 右移的目标 target_ids [B,T] -> logits [B,T,V]。"""
        if source_ids.size(0) != target_ids.size(0):
            raise ValueError("源序列和目标序列的 batch 大小必须一致。")
        memory = self.encode(source_ids, trace)
        return self.decode(target_ids, memory, self.source_mask(source_ids), trace)


@torch.inference_mode()
def generate(model: Transformer, text: str, max_new_tokens: int = 32,
             temperature: float = 0.0, seed: int = 42, use_cache: bool = True,
             trace: bool = False) -> list[int]:
    """单条源文本 -> 从 BOS 开始自回归生成；t=0 greedy，t>0 温度采样。"""
    if max_new_tokens < 0 or not math.isfinite(temperature) or temperature < 0:
        raise ValueError("需要 max_new_tokens >= 0 且 temperature 为有限非负数。")
    device = next(model.parameters()).device
    tokenizer = ByteTokenizer()
    source = tokenize_batch([text], device)
    output = torch.tensor([[tokenizer.bos_token_id]], dtype=torch.long, device=device)
    rng = torch.Generator(device=device).manual_seed(seed)
    was_training = model.training
    model.eval()
    try:
        memory = model.encode(source)                # 整个生成过程只算一次 encoder
        source_mask = model.source_mask(source)
        cache = None
        for step in range(min(max_new_tokens, model.config.max_seq_len - 1)):
            current_ids = output[:, -1:] if cache is not None else output
            if use_cache:
                logits, cache = model.decode(current_ids, memory, source_mask,
                                              trace=trace and step < 2, cache=cache, use_cache=True)
            else:
                logits = model.decode(current_ids, memory, source_mask, trace=trace and step == 0)
            logits = logits[:, -1, :].float()
            # PAD 和 BOS 是输入结构标记，不作为生成内容。
            logits[:, [tokenizer.pad_token_id, tokenizer.bos_token_id]] = -float("inf")
            if temperature == 0:
                next_token = logits.argmax(dim=-1, keepdim=True)
            else:
                # 先减最大值，避免很小的 t 在缩放时令正 logits 溢出。
                limits = torch.finfo(torch.float32)
                scores = logits.double() if not limits.tiny <= temperature <= limits.max else logits
                scores = (scores - scores.amax(dim=-1, keepdim=True)) / temperature
                next_token = torch.multinomial(scores.softmax(dim=-1), 1, generator=rng)
            output = torch.cat((output, next_token), dim=1)
            if trace:
                cached = cache.target_padding_mask.size(1) if cache is not None else 0
                print(f"Step {step + 1}: input_tokens={current_ids.size(1)}, "
                      f"cached_tokens={cached}, next_token_id={next_token.item()}")
            if next_token.item() == tokenizer.eos_token_id:
                break
        return output[0].tolist()
    finally:
        model.train(was_training)


def demo_train(model: Transformer, steps: int, device: torch.device) -> None:
    """四个句对的 teacher forcing 演示；只用于跑通训练与生成。"""
    pairs = [("hello", "你好"), ("good morning", "早上好"),
             ("thank you", "谢谢"), ("good night", "晚安")]
    source = tokenize_batch([source for source, _ in pairs], device)
    target = tokenize_batch([target for _, target in pairs], device)
    # target:  [BOS, y1, y2, ..., EOS, PAD]
    # input:   [BOS, y1, y2, ...]     每个位置预测其下一个 token
    # labels:  [y1,  y2, ..., EOS]     PAD 位置不计入 loss
    decoder_input, labels = target[:, :-1], target[:, 1:]
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4,
                                 betas=(0.9, 0.98), eps=1e-9)
    model.train()
    for step in range(1, steps + 1):
        optimizer.zero_grad(set_to_none=True)
        logits = model(source, decoder_input)
        loss = F.cross_entropy(logits.reshape(-1, model.config.vocab_size),
                               labels.reshape(-1), ignore_index=model.config.pad_token_id,
                               label_smoothing=0.1)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        if step == 1 or step % 50 == 0 or step == steps:
            print(f"demo step {step:>4}/{steps} | loss {loss.item():.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--preset", choices=("small", "base"), default="small")
    parser.add_argument("--text", default="hello", help="encoder 的源文本")
    parser.add_argument("--target", default="你好", help="用于演示 teacher forcing 的目标文本")
    parser.add_argument("--trace", action="store_true", help="逐层打印张量形状")
    parser.add_argument("--no-kv-cache", dest="use_cache", action="store_false",
                        help="关闭 decoder self/cross-attention KV cache 以供对照")
    parser.add_argument("--demo-train-steps", type=int, default=0,
                        help="可选：在四个固定句对上短暂训练，不保存权重")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--temperature", "--t", "-t", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    if args.demo_train_steps < 0 or args.max_new_tokens < 0:
        parser.error("训练步数和生成 token 上限不能为负数。")
    if not math.isfinite(args.temperature) or args.temperature < 0:
        parser.error("temperature 必须是有限非负数。")
    torch.manual_seed(args.seed)
    torch.set_num_threads(2)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else "cpu" if args.device == "auto" else args.device)
    config = TransformerConfig() if args.preset == "small" else TransformerConfig(
        d_model=512, num_heads=8, num_encoder_layers=6, num_decoder_layers=6, d_ff=2048)
    if max(len(ByteTokenizer().encode(args.text)),
           len(ByteTokenizer().encode(args.target))) > config.max_seq_len:
        parser.error(f"文本含 BOS/EOS 后的字节 token 数不能超过 {config.max_seq_len}。")
    model = Transformer(config).to(device)
    print(f"Device: {device} | Preset: {args.preset} | Parameters: {sum(p.numel() for p in model.parameters()):,}")
    if args.demo_train_steps:
        print("仅在四个固定句对上演示训练；本次权重不保存。")
        demo_train(model, args.demo_train_steps, device)
    else:
        print("随机初始化权重：以下输出只验证计算过程，没有翻译能力。")
    model.eval()
    source = tokenize_batch([args.text], device)
    target = tokenize_batch([args.target], device)
    print(f"Source IDs: {source[0].tolist()}")
    print(f"Decoder input IDs: {target[0, :-1].tolist()}")
    print(f"Next-token labels: {target[0, 1:].tolist()}")
    with torch.inference_mode():
        logits = model(source, target[:, :-1], trace=args.trace)
        loss = F.cross_entropy(logits.reshape(-1, config.vocab_size),
                               target[:, 1:].reshape(-1), ignore_index=config.pad_token_id)
    print(f"Logits: {tuple(logits.shape)} | Teacher-forcing loss: {loss.item():.4f}")
    print(f"KV cache: {args.use_cache}")
    output = generate(model, args.text, args.max_new_tokens, args.temperature, args.seed,
                      use_cache=args.use_cache, trace=args.trace)
    print(f"Source: {args.text!r}")
    print(f"Output (t={args.temperature}): {ByteTokenizer().decode(output)!r}")
    print(f"Generated IDs: {output}")


if __name__ == "__main__":
    main()
