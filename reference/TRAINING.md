# Decoder-only Transformer 训练说明

独立训练入口为 `reference/train.py`，导入同目录的 `decoder_only.py`。自己的学习实现在 `mine/decoder_only_mine.py`，保持原样。下列命令均在 `my_vllm` 根目录执行，语料和已有权重仍在公共 `data/`、`runs/` 目录。

唐诗语料、字符词表和新模型的运行方式见 [POETRY_TRAINING.md](POETRY_TRAINING.md)。本文保留原 TinyShakespeare 训练记录；两次训练使用独立的 checkpoint 目录。

## 1. 进入 WSL 环境

```bash
source ~/.venvs/nanovllm-ref/bin/activate
cd /mnt/e/ai_infra/my_vllm
```

## 2. 准备语料并划分集合

首次训练自动下载 [TinyShakespeare](https://github.com/karpathy/char-rnn/tree/master/data/tinyshakespeare)，保存为 `data/tinyshakespeare.txt`，约 1.1 MB。也可以用 `--data /path/to/corpus.txt` 指定自己的 UTF-8 语料，各段之间用空行分隔。

程序按空行拆段，去除完全相同的段落，再按原始顺序将前 90% 段落作为训练集、后 10% 作为验证集。划分发生在切窗之前，两个集合不会共享同一个段落。

本次数据 SHA256：

```text
86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed
```

实际共有 7148 个去重段落：训练 6433 个、验证 715 个。编码后分别为 1,038,438 和 83,058 个 token。

## 3. 构造输入与右移标签

每个段落编码为 `BOS + UTF-8 字节 + 换行 + EOS`，然后分别拼成训练和验证 token 流。

一次随机采样 32 个窗口，每个窗口取 257 个 token：

```python
x = block[:, :-1]  # [32, 256]：输入
y = block[:, 1:]   # [32, 256]：下一个 token 的正确答案
logits = model(x)  # [32, 256, 258]
loss = cross_entropy(logits.reshape(-1, 258), y.reshape(-1))
```

训练使用全部位置的原始 logits，不先做 softmax。训练调用 `model.train()` 和 forward；训练结束后由 `train.py` 的 `sample()` 生成示例。采样温度只影响生成，不参与交叉熵训练。

## 4. 反向传播与更新参数

训练循环的核心顺序为：

```python
optimizer.zero_grad(set_to_none=True)
loss = next_token_loss(model, block)
loss.backward()
clip_grad_norm_(model.parameters(), 1.0)
optimizer.step()
```

本次使用原模型默认配置：492,672 个参数，两层、hidden size 128、4 个 heads、最大位置256。使用 FP32、AdamW、初始峰值学习率 `3e-4`、100步 warmup，之后余弦衰减到 `3e-5`，最多5000步。

每步输入长度也是256，覆盖全部可学习位置 embedding。

## 5. 验证与保存

每100步记录近期平均训练 loss，每250步顺序遍历整个验证 token 流，包括最后不足256的短窗口。验证 loss 按目标 token 数加权。

训练输出位于 `runs/tinyshakespeare/`：

- `best.pt`：验证 loss 最低时的 checkpoint。
- `last.pt`：最近一次验证时的 checkpoint，供中断后续训。
- `metrics.jsonl`：训练和验证指标。
- `run_config.json`：模型配置、训练配方、数据 SHA256 和模型源文件 SHA256。
- `summary.json`：最终结果及从 best.pt 重新加载后的生成样本。

checkpoint 包含模型配置、tokenizer 约定、权重、优化器状态、当前步数及随机数状态，使用 `weights_only=True` 加载。保存先写临时文件，再替换目标文件。

已有训练目录不会被新训练覆盖。要重新从随机权重训练一次，请指定新目录：

```bash
python reference/train.py train \
  --output-dir runs/tinyshakespeare-new \
  --steps 5000 --batch-size 32 --seq-len 256 --lr 0.0003 \
  --eval-every 250 --log-every 100 --device cuda
```

## 6. 加载训练权重并生成

本次已完成的模型可以直接运行：

```bash
python reference/train.py generate \
  --checkpoint runs/tinyshakespeare/best.pt \
  --temperature 0.8 --max-new-tokens 96 --device cuda
```

默认提示为 `ROMEO:` 后接换行。也可以指定：

```bash
python reference/train.py generate \
  --checkpoint runs/tinyshakespeare/best.pt \
  --prompt "First Citizen:" --t 1.0 --seed 42 --max-new-tokens 96
```

温度 `t` 的含义：

- `t=0`：Greedy Decoding，每次选择最大 logit。
- `0<t<1`：概率分布更集中，输出更保守，低温时仍可能重复。
- `t=1`：按模型本身的 softmax 概率分布采样。
- `t>1`：分布更平缓、输出更多样，也更容易出现拼写或语法问题。

默认 `t=0.8`；`--temperature`、`--t` 和 `-t` 是同一个参数。温度必须为有限的非负数。

核心计算是：

```python
probabilities = torch.softmax(last_logits / t, dim=-1)
next_token = torch.multinomial(probabilities, num_samples=1)
```

实际代码先减去当前行最大 logit，以提高数值稳定性。仅缩放 logits 后继续 `argmax` 不会改变选择，因此 `t>0` 必须按概率抽样。[PyTorch multinomial](https://docs.pytorch.org/docs/2.9/generated/torch.multinomial.html)

不传 `--seed` 时每次调用可以得到不同输出；指定相同 seed 可在相同设备和环境下复现采样。`t=0` 不受 seed 影响。EOS 和长度上限仍会终止生成。

训练结束时的示例温度可用 `train --sample-temperature 0.8` 设置；示例采样种子使用训练的 `--seed`。已有 `summary.json` 保留原训练时产生的 greedy 样本，新的 `generate` 命令按本次指定的温度运行。

调用 `decoder_only.py` 自身的原入口仍会创建随机权重；读取训练结果请用上面的独立入口。

生成默认开启 KV cache，先 prefill 整段提示，再逐 token decode。缓存只在本次生成中使用；训练仍执行不带缓存的完整 forward，已有 checkpoint 无需重训。

```bash
python reference/train.py generate \
  --checkpoint runs/tinyshakespeare/best.pt \
  --prompt 'ROMEO:' --t 0 --max-new-tokens 4 --trace
```

在同一命令中添加 `--no-kv-cache` 可对比完整重算路径。`--trace` 会显示首轮输入整个 prompt，之后开启缓存的每轮 `input_tokens=1`，以及每层缓存的 K/V 形状和累计 token 数。实现用 `torch.cat` 追加缓存，尚未使用预分配或分页缓存；目录说明和底层 API 见 [README.md](../README.md)。

## 7. 恢复或延长训练

```bash
python reference/train.py train \
  --resume runs/tinyshakespeare/last.pt \
  --steps 10000 --device cuda
```

`--steps` 是目标总步数：从5000步恢复到10000步会再训练5000步。恢复时沿用 checkpoint 的 batch、长度、学习率、warmup、裁剪和 seed，继续写入同一个目录；语料哈希变化会报错。

如果是中断后恢复，保持原来的目标总步数即可延续原学习率曲线；主动扩大总步数时，余弦曲线会按新的上限重新计算。

## 本次训练结果（2026-10-30）

- 已完成5000步，处理40,960,000个目标 token。
- 初始验证 loss：5.5776；最佳验证 loss：1.6730，出现在第5000步。
- 最后100步的平均训练 loss：约1.4978。
- 验证集 byte-level perplexity：约5.328，不能与词级或 BPE token 的 perplexity 直接比较。
- 训练循环约98.70秒，包含循环中的验证和存盘，不包含下载与初始准备。
- 峰值 PyTorch 已分配显存约282.86 MiB，不含 CUDA 上下文及其他程序占用。
- 当时已从 best.pt 独立重建模型、加载权重并验证 greedy 生成；现在也可直接用同一权重进行温度采样。

模型已学到一些英文词形和语料模式，但输出仍有明显重复、不通顺。本次并未训练通用问答能力。例如 `The capital of France is` 不属于该语料的目标任务，不能期待它给出事实性答案。

训练前还用临时小数据验证了 checkpoint 保存 / 加载、优化器步数恢复、生成、覆盖保护和语料哈希保护；临时文件随检查自动清除。
