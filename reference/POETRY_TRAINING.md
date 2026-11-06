# 唐诗模型训练与使用

本任务从随机权重训练独立的唐诗模型，使用本地 `data/chinese-poetry` 仓库中的 `全唐诗/poet.tang.*.json`。训练和模型代码现位于 `reference/`，自己的学习代码原样保存在 `mine/`。下列命令均在 `my_vllm` 根目录执行，语料和已有权重路径不变。

## 语料准备

进入已经配置的 WSL 环境：

```bash
source ~/.venvs/nanovllm-ref/bin/activate
cd /mnt/e/ai_infra/my_vllm
python reference/prepare_poetry.py
```

`prepare_poetry.py` 读取 58 个唐诗 JSON 文件，只使用正文 `paragraphs`，不使用作者传记、排名、注解，也不读取 `poet.song.*.json`。保留繁体原文和逐联换行，去除行内空白并做 Unicode NFC 规范化；过滤缺字方框、损坏字符、控制字符以及含 ASCII 字母/数字的条目。忽略标点和空白后正文相同的诗只保留一首。不同异文、繁简或近似重复仍可能存在。

本次原始 57,607 首，清洗后保留 55,788 首。生成文件位于 `data/tang_poetry/`：

- `poems.txt`：每首正文一个文档，各首之间用空行分隔。
- `manifest.json`：原始仓库 commit、每个源文件的 SHA256、清洗规则及数量、生成语料 SHA256。
- `SOURCE_LICENSE.txt`：原库 MIT 许可证。

原始仓库 commit 为 `b8594f81a89752241442f2ce267d6f66f96704ee`，本次生成语料 SHA256 为 `2dcd2b376ec893128c71e4076b139c011215f86fed84066cef03cc01f6b47db7`。原始 clone 不会被写入。

## 划分与字符词表

先用 seed 42 打乱整首诗，再将前 90% 用于训练、后 10% 用于验证。训练 50,209 首，验证 5,579 首；划分发生在建立词表及切 token 窗口之前。

字符词表只从训练集建立，按 Unicode 码点排序，另加 UNK、EOS、BOS。一汉字或标点对应一个 token；这与原英文模型的 UTF-8 字节词表不同。验证集独有字符记为 UNK，本次为 229 个，占验证 token 约 0.060%。推理时从 checkpoint 恢复词表，并明确拒绝词表外的提示字符。

每首正文编码为 `BOS + 正文 + 换行 + EOS`。训练和验证分别拼成 token 流，再分别切窗；窗口可能跨越 EOS/BOS，长诗也可能被拆为多个窗口。训练使用全部位置的右移标签与交叉熵。BOS 和 UNK 在生成时屏蔽，EOS 可以正常终止生成。

## 从头训练

本次运行命令：

```bash
python reference/train.py train \
  --data data/tang_poetry/poems.txt --output-dir runs/tang-poetry \
  --tokenizer characters --split-seed 42 \
  --hidden-size 256 --num-heads 4 --num-layers 4 --intermediate-size 1024 \
  --max-seq-len 128 --seq-len 128 --batch-size 32 \
  --steps 10000 --lr 0.0006 --warmup-steps 300 --precision bf16 \
  --eval-every 1000 --log-every 100 --sample-temperature 0.8 --device cuda
```

本模型有 7,890,176 个参数，字符词表大小为 9,198；4 层、hidden size 256、4 个 attention heads、MLP 1024、128 token 上下文。使用 BF16 自动混合精度，模型及优化器主权重仍为 FP32；AdamW、weight decay 0.01、梯度裁剪 1.0，学习率 warmup 后余弦衰减。`bf16` 训练需使用支持 BF16 的 CUDA GPU。

训练集 3,484,473 个 token，验证集 380,113 个 token。10,000 步处理 40,960,000 个目标 token，相当于约 11.8 倍训练流长度；因为随机有放回采样窗口，这不是严格逐轮遍历。

已有结果目录有覆盖保护。再次从头训练请换成 `--output-dir runs/tang-poetry-new`。

## 加载与生成

```bash
python reference/train.py generate \
  --checkpoint runs/tang-poetry/best.pt \
  --prompt "春" --t 0.8 --seed 42 --max-new-tokens 96 --device cuda
```

不传 `--prompt` 时，从 BOS 开始自由生成一首。也可以试 `--prompt "明月"`、`--prompt "山"`。原文为繁体，提示建议同样使用繁体；例如有些简体字不在词表内，会被明确报错。

`--t 0` 使用 greedy，`0<t<1` 分布较集中，`t=1` 按原始分布抽样，`t>1` 更随机；默认 0.8。固定 `--seed` 可在同一设备和环境下复现采样。生成在遇到 EOS、达到新 token 上限或总长达到 128 token 时结束。本模型用于学习古诗语言模式，不保证平仄、押韵、主题连贯或原创性。

加载时必须使用新模型自身的字符词表。`train.py generate` 会自动识别 checkpoint 中的 tokenizer；原莎士比亚 checkpoint 仍可用同一入口加载。

生成默认启用 KV cache，已有唐诗权重无需重新训练。可添加 `--trace` 查看逐 token 解码过程，或添加 `--no-kv-cache` 与完整重算对比。语料和 checkpoint 仍保存在 `my_vllm/data/` 与 `my_vllm/runs/`。

## 结果文件与续训

输出位于 `runs/tang-poetry/`：`best.pt`、`last.pt`、`tokenizer.json`、`run_config.json`、`metrics.jsonl`、`summary.json`。本次终端训练日志额外保存在 `console.log`。

每 1,000 步在整个验证集上计算按目标 token 数加权的 loss，保存最近权重及最优权重。结束后重新从 `best.pt` 加载并生成示例，结果写入 `summary.json`。字符级 loss / perplexity 与之前英文模型的字节级指标不能直接比较。

中断后恢复，或将目标总步数延长到 20,000：

```bash
python reference/train.py train \
  --resume runs/tang-poetry/last.pt --steps 20000 --device cuda
```

续训恢复模型配置、词表、batch、长度、精度、学习率、warmup、seed、整诗划分方式、优化器及随机状态，并检查语料 SHA256；扩大总步数会按新上限重新计算余弦曲线。

## 本次已完成的结果（2026-10-30）

- 完成 10,000 步，处理 40,960,000 个目标 token。
- 初始验证 loss 9.1911；最佳验证 loss 4.0685，来自第 7,000 步；最后验证 loss 4.0885。
- 最佳字符级 perplexity 58.47；最后 100 步平均训练 loss 3.3192。
- 训练循环 269.75 秒，约 4 分 30 秒，包含循环内的验证及 checkpoint 保存，不包含语料准备、初始验证和结束后生成。
- 峰值 PyTorch 已分配显存 727.27 MiB，不包含 CUDA 上下文及其他程序占用。
- `best.pt` 和 `last.pt` 分别对应第 7,000 和第 10,000 步；后期训练 loss 下降而验证 loss 小幅回升，生成时优先用 `best.pt`。

从 `best.pt` 重新加载，空提示、`t=0.8`、seed 42 的实际输出：

```text
秋風送客怨楚絲，白蓮浮動天邊月。
玉輪金磧金鳳高，金城玉關心獨死。
```

该样本在 35 个新 token 后遇到 EOS；`春` 和 `明月` 两个提示的示例也正常生成并遇到 EOS，详见 `runs/tang-poetry/summary.json`。模型已学到一些诗句形式和常见意象，语义连贯性、平仄和押韵仍不稳定，示例不能代表可靠的诗歌创作能力。

临时检查覆盖字符词表只读取训练集、BF16 更新、checkpoint 保存/加载、优化器断点恢复、采样 seed 复现、词表外提示报错，以及原字节模型的加载和 greedy 兼容性；没有保留额外检查脚本。
