# 手写 Transformer 学习代码

```text
my_vllm/
├── reference/                 # Codex 的 Decoder-only 参考实现
│   ├── decoder_only.py       # 模型、显式 attention、KV cache、随机权重演示
│   ├── train.py              # 训练 / 续训 / 加载已有权重生成
│   ├── prepare_poetry.py     # 唐诗语料准备
│   ├── TRAINING.md
│   └── POETRY_TRAINING.md
├── original_transformer/      # 论文结构的 Encoder–Decoder Transformer
│   └── transformer.py        # 正弦 PE、Post-LN、交叉注意力、decoder KV cache
├── mine/                      # 你自己的实现，源码原样移动
│   └── decoder_only_mine.py
├── data/                      # 已有公共语料，路径不变
└── runs/                      # 已有 checkpoint 和训练记录，路径不变
```

以下命令在 WSL 中执行：

```bash
source ~/.venvs/nanovllm-ref/bin/activate
cd /mnt/e/ai_infra/my_vllm
```

## Decoder-only：加载已有模型

```bash
python reference/train.py generate \
  --checkpoint runs/tinyshakespeare/best.pt \
  --prompt 'ROMEO:' --t 0.8 --seed 42 --max-new-tokens 64

python reference/train.py generate \
  --checkpoint runs/tang-poetry/best.pt \
  --prompt '春' --t 0.8 --seed 42 --max-new-tokens 64
```

两种 tokenizer 都由 checkpoint 自动恢复。KV cache 默认开启，添加 `--no-kv-cache` 可对照每轮重算完整上下文的路径，添加 `--trace` 可看到 prefill、首次 decode 和缓存长度。已有权重无需重新训练。

仅查看随机小模型的计算过程：

```bash
python reference/decoder_only.py --text hello --max-new-tokens 4 --trace
```

训练和续训命令见 [TRAINING.md](reference/TRAINING.md)，唐诗模型见 [POETRY_TRAINING.md](reference/POETRY_TRAINING.md)。

## 原版 Encoder–Decoder Transformer

```bash
python original_transformer/transformer.py --trace
python original_transformer/transformer.py --demo-train-steps 300 --text hello
```

`--preset base` 使用论文的层数和隐藏维度；词表仍是教学用的 UTF-8 字节词表。`--t 0.8` 开启温度采样；默认 `t=0` 为 greedy。默认随机权重没有翻译能力，四句对训练演示只用于验证流程，权重不保存。这里也可以使用 `--no-kv-cache` 关闭缓存。

## KV cache 的阅读要点

- **Prefill**：Decoder-only 一次处理整个 prompt，得到每层 `[B, H, prompt_tokens, Dh]` 的 K/V。
- **Decode**：每轮只处理上轮生成的一个 token。新 Q 关注历史与当前的 K/V；新 K/V 追加到历史末尾。
- **位置偏移**：新 token 的位置从缓存长度开始，因果 mask 也按绝对位置构造，支持一次追加多个 token。
- **Encoder–Decoder**：encoder 对整段源文本只计算一次；decoder 每层还缓存由 encoder memory 投影得到的固定 cross-attention K/V。
- **隔离请求**：cache 显式传入、显式返回。新请求从 `None` 开始，换源文本、batch 或模型权重后也要重建缓存。

模型的默认训练 forward 仍返回完整 logits，缓存仅供 `eval()` 下的推理使用。缓存不是可训练参数，不写入 checkpoint。本版用 `torch.cat` 追加 K/V，便于阅读，会复制历史缓存；尚未实现预分配缓存、PagedAttention 或连续批处理。

Decoder-only 底层调用示例（模型需先加载权重并 `eval()`）：

```python
with torch.inference_mode():
    logits, cache = model(prompt_ids, use_cache=True)
    next_id = logits[:, -1].argmax(dim=-1, keepdim=True)
    logits, cache = model(next_id, past_key_values=cache, use_cache=True)
```

Encoder–Decoder 对应先 `memory = model.encode(source_ids)`，然后调用 `model.decode(target_ids, memory, source_mask, use_cache=True)`；后续只传新的目标 token 和返回的 `cache`。
