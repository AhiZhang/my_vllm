"""独立训练入口：导入同目录的 Decoder-only 参考实现。

在 my_vllm 目录下运行：
训练：python reference/train.py train --steps 5000
生成：python reference/train.py generate --checkpoint runs/tinyshakespeare/best.pt --temperature 0.8
续训：python reference/train.py train --resume runs/tinyshakespeare/last.pt --steps 10000
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import random
import time
import urllib.request

import torch
import torch.nn.functional as F

if __package__:
    from .decoder_only import ByteTokenizer, DecoderOnlyTransformer, ModelConfig
else:
    from decoder_only import ByteTokenizer, DecoderOnlyTransformer, ModelConfig


ROOT = Path(__file__).resolve().parent.parent  # my_vllm 公共语料与 checkpoint 根目录
DEFAULT_DATA = ROOT / "data" / "tinyshakespeare.txt"
DATA_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
TOKENIZER_SPEC = {"type": "utf8_bytes", "vocab_size": 258, "bos_token_id": 257, "eos_token_id": 256}


class CharacterTokenizer:
    """从训练集建立字符表；UNK=0、EOS=1、BOS=2，其余每个字符一个 ID。"""

    unk_token_id, eos_token_id, bos_token_id = 0, 1, 2

    def __init__(self, characters: list[str]):
        if len(set(characters)) != len(characters) or any(len(c) != 1 for c in characters):
            raise ValueError("字符词表必须由不重复的单个字符组成。")
        self.characters = characters
        self.char_to_id = {char: index + 3 for index, char in enumerate(characters)}
        self.vocab_size = len(characters) + 3

    def encode(self, text: str) -> list[int]:
        return [self.bos_token_id, *(self.char_to_id.get(c, self.unk_token_id) for c in text)]

    def decode(self, token_ids: list[int]) -> str:
        return "".join("�" if token_id == self.unk_token_id else self.characters[token_id - 3]
                       for token_id in token_ids if token_id not in (self.bos_token_id, self.eos_token_id))

    def spec(self) -> dict:
        return {"type": "characters", "characters": self.characters, "vocab_size": self.vocab_size,
                "unk_token_id": self.unk_token_id, "bos_token_id": self.bos_token_id,
                "eos_token_id": self.eos_token_id}


def tokenizer_from_spec(spec: dict):
    if spec == TOKENIZER_SPEC:
        return ByteTokenizer()
    if spec.get("type") == "characters":
        tokenizer = CharacterTokenizer(spec["characters"])
        if tokenizer.spec() == spec:
            return tokenizer
    raise ValueError("checkpoint 的 tokenizer 格式不受支持。")


def tokenizer_spec(tokenizer) -> dict:
    return tokenizer.spec() if isinstance(tokenizer, CharacterTokenizer) else TOKENIZER_SPEC


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def prepare_data(path: Path, seq_len: int, tokenizer_type: str = "bytes",
                 saved_tokenizer: dict | None = None, split_seed: int | None = None):
    if not path.is_file():
        if path.resolve() != DEFAULT_DATA.resolve():
            raise FileNotFoundError(f"自定义语料不存在：{path}")
        print(f"Downloading TinyShakespeare from {DATA_URL}", flush=True)
        path.parent.mkdir(parents=True, exist_ok=True)
        request = urllib.request.Request(DATA_URL, headers={"User-Agent": "my-vllm-training"})
        with urllib.request.urlopen(request, timeout=60) as response:
            raw = response.read()
        raw.decode("utf-8")  # 确认完整文本可以解码，再保存。
        temporary = path.with_suffix(".download")
        temporary.write_bytes(raw)
        temporary.replace(path)

    text = path.read_text(encoding="utf-8").replace("\r\n", "\n")
    # 先按完整段落划分集合，精确去重，之后才 tokenization / 切窗。
    documents = list(dict.fromkeys(part.strip() for part in text.split("\n\n") if part.strip()))
    if len(documents) < 2:
        raise ValueError("语料至少需要两个用空行分隔的不同段落，才能划分训练和验证集。")
    if split_seed is not None:
        random.Random(split_seed).shuffle(documents)
    split = max(1, min(len(documents) - 1, int(len(documents) * 0.9)))
    if saved_tokenizer:
        tokenizer = tokenizer_from_spec(saved_tokenizer)
    elif tokenizer_type == "characters":
        # 验证集中的独有字符使用 UNK；词表不能提前读取验证文本。
        tokenizer = CharacterTokenizer(sorted(set("\n".join(documents[:split]))))
    else:
        tokenizer = ByteTokenizer()

    def encode(parts: list[str]) -> torch.Tensor:
        token_ids = []
        for part in parts:
            token_ids.extend(tokenizer.encode(part + "\n"))
            token_ids.append(tokenizer.eos_token_id)
        return torch.tensor(token_ids, dtype=torch.long)

    train_tokens = encode(documents[:split])
    val_tokens = encode(documents[split:])
    if train_tokens.numel() < seq_len + 1 or val_tokens.numel() < 2:
        raise ValueError("语料太短：训练集需要至少 seq_len+1 个 token，验证集至少需要 2 个。")
    metadata = {
        "path": str(path.resolve()), "source_url": DATA_URL if path.resolve() == DEFAULT_DATA.resolve() else None,
        "sha256": file_hash(path), "bytes": path.stat().st_size,
        "split": "seeded_shuffle_then_90_percent" if split_seed is not None else "first_90_percent_of_exact_deduplicated_paragraphs",
        "split_seed": split_seed,
        "documents": len(documents), "train_documents": split,
        "validation_documents": len(documents) - split,
        "train_tokens": train_tokens.numel(), "validation_tokens": val_tokens.numel(),
    }
    if isinstance(tokenizer, CharacterTokenizer):
        metadata["validation_unknown_tokens"] = int((val_tokens == tokenizer.unk_token_id).sum())
    return train_tokens, val_tokens, metadata, tokenizer


def next_token_loss(model, block: torch.Tensor, reduction: str = "mean", precision: str = "float32") -> torch.Tensor:
    # T+1 个 token -> T 个输入和 T 个右移标签；每个位置都参与训练。
    inputs, targets = block[:, :-1], block[:, 1:]
    with torch.autocast(device_type=block.device.type, dtype=torch.bfloat16, enabled=precision == "bf16"):
        logits = model(inputs)
        return F.cross_entropy(logits.reshape(-1, model.config.vocab_size), targets.reshape(-1), reduction=reduction)


@torch.inference_mode()
def evaluate(model, tokens: torch.Tensor, seq_len: int, batch_size: int, device: torch.device,
             precision: str = "float32") -> float:
    """顺序遍历整个验证 token 流，按目标 token 数加权，包括最后的短窗口。"""
    was_training = model.training
    model.eval()
    total_loss, count = 0.0, 0
    full_blocks = (tokens.numel() - 1) // seq_len
    if full_blocks:
        windows = tokens.unfold(0, seq_len + 1, seq_len)
        for start in range(0, full_blocks, batch_size):
            block = windows[start:start + batch_size].to(device)
            total_loss += next_token_loss(model, block, reduction="sum", precision=precision).item()
            count += block.size(0) * seq_len
    tail = tokens[full_blocks * seq_len:]
    if tail.numel() > 1:
        total_loss += next_token_loss(model, tail.unsqueeze(0).to(device), reduction="sum", precision=precision).item()
        count += tail.numel() - 1
    model.train(was_training)
    return total_loss / count


def learning_rate(step: int, total_steps: int, base_lr: float, warmup: int) -> float:
    if warmup and step <= warmup:
        return base_lr * step / warmup
    progress = (step - warmup) / max(1, total_steps - warmup)
    return base_lr * (0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress)))


def save_checkpoint(path: Path, model, optimizer, generator, step: int, best_step: int,
                    best_loss: float, initial_loss: float, validation_loss: float,
                    train_config: dict, data: dict) -> None:
    checkpoint = {
        "format_version": 1, "model_config": asdict(model.config), "tokenizer": tokenizer_spec(model.tokenizer),
        "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
        "step": step, "best_step": best_step, "best_validation_loss": best_loss,
        "initial_validation_loss": initial_loss, "validation_loss": validation_loss,
        "train_config": train_config, "data": data,
        "batch_rng_state": generator.get_state(), "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_states": torch.cuda.get_rng_state_all() if next(model.parameters()).device.type == "cuda" else [],
    }
    temporary = path.with_suffix(".pt.tmp")
    torch.save(checkpoint, temporary)
    temporary.replace(path)


def load_model(path: Path, device: torch.device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    tokenizer = tokenizer_from_spec(checkpoint["tokenizer"])
    model = DecoderOnlyTransformer(ModelConfig(**checkpoint["model_config"])).to(device)
    if tokenizer.vocab_size != model.config.vocab_size:
        raise ValueError("checkpoint 的词表大小与模型配置不同。")
    model.tokenizer = tokenizer
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    return model, checkpoint


@torch.inference_mode()
def sample_next_token(logits: torch.Tensor, temperature: float,
                      generator: torch.Generator | None = None) -> torch.Tensor:
    """输入 [B,V]；t=0 取 argmax，t>0 按 softmax(logits/t) 抽样。"""
    if not math.isfinite(temperature) or temperature < 0:
        raise ValueError("temperature 必须是有限的非负数；0 表示 greedy。")
    if temperature == 0:
        return logits.argmax(dim=-1, keepdim=True)
    scores = logits.float()
    # 超出 float32 正常数值范围的温度改用 double，避免缩放时下溢或溢出。
    limits = torch.finfo(torch.float32)
    if not limits.tiny <= temperature <= limits.max:
        scores = scores.double()
    scores = scores - scores.amax(dim=-1, keepdim=True)
    probabilities = torch.softmax(scores / temperature, dim=-1)
    # 对缩放后的 logits 继续 argmax 不会改变选择，必须按概率分布抽样。
    return torch.multinomial(probabilities, num_samples=1, generator=generator)


@torch.inference_mode()
def sample(model, prompt: str, device: torch.device, max_new_tokens: int,
           temperature: float = 0.8, seed: int | None = None,
           use_cache: bool = True, trace: bool = False) -> dict:
    if not math.isfinite(temperature) or temperature < 0:
        raise ValueError("temperature 必须是有限的非负数；0 表示 greedy。")
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens 不能为负数。")
    tokenizer = getattr(model, "tokenizer", ByteTokenizer())
    if isinstance(tokenizer, CharacterTokenizer):
        missing = sorted(set(prompt) - tokenizer.char_to_id.keys())
        if missing:
            raise ValueError(f"提示含训练词表外的字符：{''.join(missing)!r}；语料为繁体时请使用繁体提示。")
    inputs = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)
    if inputs.size(1) > model.config.max_seq_len:
        raise ValueError("输入长度超出模型支持的上下文范围。")
    rng = torch.Generator(device=device).manual_seed(seed) if seed is not None and temperature > 0 else None
    model.eval()
    output = inputs
    past_key_values = None
    for step in range(max_new_tokens):
        if output.size(1) >= model.config.max_seq_len:
            break
        current_ids = output[:, -1:] if past_key_values is not None else output
        if use_cache:
            logits, past_key_values = model(current_ids, trace=trace and step < 2,
                                            past_key_values=past_key_values, use_cache=True)
        else:
            logits = model(current_ids, trace=trace and step == 0)
        last_logits = logits[:, -1, :]
        if isinstance(tokenizer, CharacterTokenizer):
            # BOS 仅用于输入；UNK 不应作为可读的生成结果。
            last_logits[:, [tokenizer.bos_token_id, tokenizer.unk_token_id]] = -torch.inf
        next_token = sample_next_token(last_logits, temperature, generator=rng)
        output = torch.cat((output, next_token), dim=1)
        if trace:
            cached = past_key_values[0][0].size(2) if past_key_values is not None else 0
            print(f"Step {step + 1}: input_tokens={current_ids.size(1)}, "
                  f"cached_tokens={cached}, next_token_id={next_token.item()}")
        if next_token.item() == tokenizer.eos_token_id:
            break
    return {"prompt": prompt, "text": tokenizer.decode(output[0].tolist()),
            "new_tokens": output.size(1) - inputs.size(1),
            "ended_with_eos": output[0, -1].item() == tokenizer.eos_token_id,
            "temperature": temperature, "seed": seed, "kv_cache": use_cache}


def train(args, device: torch.device) -> None:
    if not math.isfinite(args.sample_temperature) or args.sample_temperature < 0:
        raise ValueError("sample_temperature 必须是有限的非负数。")
    restored = torch.load(args.resume, map_location="cpu", weights_only=True) if args.resume else None
    if restored:
        tokenizer_from_spec(restored["tokenizer"])
        # 续训恢复原来的训练配方；--steps 表示新的总步数。
        for name in ("batch_size", "seq_len", "lr", "warmup_steps", "grad_clip", "seed"):
            setattr(args, name, restored["train_config"][name])
        args.split_seed = restored["data"].get("split_seed")
        args.precision = restored["train_config"].get("precision", "float32")
    config = ModelConfig(**restored["model_config"]) if restored else ModelConfig(
        hidden_size=args.hidden_size, num_heads=args.num_heads, num_layers=args.num_layers,
        intermediate_size=args.intermediate_size, max_seq_len=args.max_seq_len)
    if args.precision == "bf16" and (device.type != "cuda" or not torch.cuda.is_bf16_supported()):
        raise ValueError("bf16 训练需要支持 BF16 的 CUDA GPU；CPU 请使用 float32。")
    if not 1 <= args.seq_len <= config.max_seq_len:
        raise ValueError(f"seq_len 必须在 1..{config.max_seq_len} 之间。")
    if min(args.batch_size, args.steps, args.eval_every, args.log_every) < 1 or args.lr <= 0 or args.grad_clip <= 0 or args.warmup_steps < 0:
        raise ValueError("步数、batch、日志间隔、学习率和裁剪阈值需有效。")
    args.warmup_steps = min(args.warmup_steps, max(0, args.steps - 1))
    start_step = restored["step"] if restored else 0
    if args.steps <= start_step:
        raise ValueError("--steps 必须大于 checkpoint 中已完成的步数。")
    output_dir = (args.output_dir or (args.resume.parent if args.resume else ROOT / "runs" / "tinyshakespeare")).resolve()
    if restored and output_dir != args.resume.resolve().parent:
        raise ValueError("续训应继续写入原 checkpoint 目录。")
    if not restored and any((output_dir / name).exists() for name in ("best.pt", "last.pt", "metrics.jsonl")):
        raise FileExistsError("输出目录已有训练结果，请使用 --resume 或新的 --output-dir。")
    if restored and not (output_dir / "best.pt").is_file():
        raise FileNotFoundError("续训目录需要保留 best.pt。")
    data_path = args.data or (Path(restored["data"]["path"]) if restored else DEFAULT_DATA)
    train_tokens, val_tokens, data, tokenizer = prepare_data(
        data_path, args.seq_len, args.tokenizer, restored["tokenizer"] if restored else None, args.split_seed)
    if restored and data["sha256"] != restored["data"]["sha256"]:
        raise ValueError("续训语料 SHA256 与 checkpoint 不一致。")

    torch.manual_seed(args.seed)
    config.vocab_size = tokenizer.vocab_size
    model = DecoderOnlyTransformer(config).to(device).train()
    model.tokenizer = tokenizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    generator = torch.Generator().manual_seed(args.seed + 1)
    if restored:
        model.load_state_dict(restored["model_state_dict"], strict=True)
        optimizer.load_state_dict(restored["optimizer_state_dict"])
        generator.set_state(restored["batch_rng_state"])
        torch.set_rng_state(restored["torch_rng_state"])
        if device.type == "cuda" and restored["cuda_rng_states"]:
            torch.cuda.set_rng_state_all(restored["cuda_rng_states"])
    train_config = {name: getattr(args, name) for name in
                    ("batch_size", "seq_len", "lr", "warmup_steps", "grad_clip", "seed", "steps", "precision")}
    output_dir.mkdir(parents=True, exist_ok=True)
    source_files = (Path(__file__).resolve(), Path(__file__).with_name("decoder_only.py").resolve())
    source_hashes = {path.relative_to(ROOT).as_posix(): file_hash(path) for path in source_files}
    write_json(output_dir / "run_config.json", {
        "model_config": asdict(config), "tokenizer": tokenizer_spec(tokenizer), "training": train_config,
        "generation": {"temperature": args.sample_temperature, "seed": args.seed},
        "data": data, "source_sha256": source_hashes,
        "torch_version": str(torch.__version__), "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
    })
    write_json(output_dir / "tokenizer.json", tokenizer_spec(tokenizer))
    metrics_path = output_dir / "metrics.jsonl"

    def record(values):
        with metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(values, ensure_ascii=False) + "\n")

    print(json.dumps({"parameters": sum(p.numel() for p in model.parameters()), "device": str(device),
                      "data": data, "output_dir": str(output_dir), "training": train_config}, indent=2), flush=True)
    initial_loss = restored["initial_validation_loss"] if restored else evaluate(model, val_tokens, args.seq_len, args.batch_size, device, args.precision)
    best_loss = restored["best_validation_loss"] if restored else initial_loss
    best_step = restored["best_step"] if restored else 0
    current_val_loss = restored["validation_loss"] if restored else initial_loss

    def checkpoint(name, step):
        save_checkpoint(output_dir / name, model, optimizer, generator, step,
                        best_step, best_loss, initial_loss, current_val_loss, train_config, data)

    if not restored:
        record({"kind": "validation", "step": 0, "validation_loss": initial_loss})
        checkpoint("best.pt", 0)
        checkpoint("last.pt", 0)
    print(f"Initial validation loss: {initial_loss:.4f}; resuming at step {start_step}", flush=True)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    offsets = torch.arange(args.seq_len + 1)
    started = time.perf_counter()
    recent_loss, recent_count = 0.0, 0
    for step in range(start_step + 1, args.steps + 1):
        model.train()
        starts = torch.randint(0, train_tokens.numel() - args.seq_len, (args.batch_size,), generator=generator)
        block = train_tokens[starts[:, None] + offsets].to(device)
        lr = learning_rate(step, args.steps, args.lr, args.warmup_steps)
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.zero_grad(set_to_none=True)
        loss = next_token_loss(model, block, precision=args.precision)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip, error_if_nonfinite=True)
        optimizer.step()
        recent_loss += loss.item()
        recent_count += 1
        if step % args.log_every == 0 or step == args.steps:
            values = {"kind": "train", "step": step, "train_loss_recent": recent_loss / recent_count,
                      "lr": lr, "elapsed_seconds": time.perf_counter() - started}
            record(values)
            print(f"step {step:5d}/{args.steps}: train_loss={values['train_loss_recent']:.4f}, lr={lr:.2e}", flush=True)
            recent_loss, recent_count = 0.0, 0
        if step % args.eval_every == 0 or step == args.steps:
            val_loss = evaluate(model, val_tokens, args.seq_len, args.batch_size, device, args.precision)
            if not math.isfinite(val_loss):
                raise RuntimeError("验证 loss 出现非有限值。")
            improved = val_loss < best_loss
            current_val_loss = val_loss
            if improved:
                best_loss, best_step = val_loss, step
                checkpoint("best.pt", step)
            checkpoint("last.pt", step)
            record({"kind": "validation", "step": step, "validation_loss": val_loss,
                    "best_validation_loss": best_loss, "best_step": best_step})
            print(f"  validation_loss={val_loss:.4f}, best={best_loss:.4f} at step {best_step}", flush=True)

    elapsed = time.perf_counter() - started
    peak_mib = torch.cuda.max_memory_allocated(device) / 1024**2 if device.type == "cuda" else None
    # 从磁盘独立加载 best，验证保存的权重确实可以用于温度采样。
    best_model, best_checkpoint = load_model(output_dir / "best.pt", device)
    prompts = args.sample_prompt or (["", "春", "明月"] if isinstance(tokenizer, CharacterTokenizer)
                                    else ["ROMEO:\n", "First Citizen:\n", "The capital of France is"])
    examples = [sample(best_model, prompt, device, 96, temperature=args.sample_temperature, seed=args.seed)
                for prompt in prompts]
    summary = {
        "completed_steps": args.steps, "steps_this_invocation": args.steps - start_step,
        "trained_target_tokens": args.steps * args.batch_size * args.seq_len,
        "initial_validation_loss": initial_loss, "best_validation_loss": best_loss,
        "last_validation_loss": current_val_loss,
        "best_step": best_step, "best_perplexity": math.exp(best_loss),
        "training_seconds_this_invocation": elapsed, "peak_allocated_mib": peak_mib,
        "best_checkpoint": str(output_dir / "best.pt"), "last_checkpoint": str(output_dir / "last.pt"),
        "loaded_best_step": best_checkpoint["step"], "samples": examples,
        "sample_note": "Temperature samples are illustrative (t=0 is greedy); validation loss is measured on held-out paragraphs.",
    }
    write_json(output_dir / "summary.json", summary)
    print("TRAINING_COMPLETE:", json.dumps(summary, ensure_ascii=False), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    training = commands.add_parser("train", help="训练；默认自动下载 TinyShakespeare")
    training.add_argument("--data", type=Path, help="可选：自定义 UTF-8 文本，段落用空行分隔")
    training.add_argument("--output-dir", type=Path)
    training.add_argument("--resume", type=Path, help="恢复 last.pt；继续写入原目录并恢复原训练配方")
    training.add_argument("--steps", type=int, default=5000, help="训练目标总步数，包括续训前已完成步数")
    training.add_argument("--batch-size", type=int, default=32)
    training.add_argument("--seq-len", type=int, default=256)
    training.add_argument("--tokenizer", choices=("bytes", "characters"), default="bytes")
    training.add_argument("--split-seed", type=int, help="可选：按给定种子打乱整篇文档后划分集合")
    training.add_argument("--hidden-size", type=int, default=128)
    training.add_argument("--num-heads", type=int, default=4)
    training.add_argument("--num-layers", type=int, default=2)
    training.add_argument("--intermediate-size", type=int, default=512)
    training.add_argument("--max-seq-len", type=int, default=256)
    training.add_argument("--precision", choices=("float32", "bf16"), default="float32")
    training.add_argument("--lr", type=float, default=3e-4)
    training.add_argument("--warmup-steps", type=int, default=100)
    training.add_argument("--grad-clip", type=float, default=1.0)
    training.add_argument("--seed", type=int, default=42)
    training.add_argument("--eval-every", type=int, default=250)
    training.add_argument("--log-every", type=int, default=100)
    training.add_argument("--sample-temperature", type=float, default=0.8, help="训练结束后示例生成的温度，0 表示 greedy")
    training.add_argument("--sample-prompt", action="append", help="可重复指定训练完成后展示的提示文本")
    inference = commands.add_parser("generate", help="加载已保存的训练权重并进行温度采样")
    inference.add_argument("--checkpoint", type=Path, required=True)
    inference.add_argument("--prompt", help="默认：字节模型为 ROMEO: 换行，字符模型为空提示")
    inference.add_argument("--max-new-tokens", type=int, default=96)
    inference.add_argument("--temperature", "--t", "-t", type=float, default=0.8, help="采样温度，默认0.8；0=greedy，越大越随机")
    inference.add_argument("--seed", type=int, default=None, help="可选：固定采样随机种子，便于在同一设备复现")
    inference.add_argument("--no-kv-cache", dest="use_cache", action="store_false",
                           help="关闭 KV cache，每轮重算完整上下文以供对照")
    inference.add_argument("--trace", action="store_true", help="打印 prefill、首次 decode 的形状和缓存长度")
    for command in (training, inference):
        command.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    torch.set_num_threads(2)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else "cpu" if args.device == "auto" else args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA 不可用。")
    if args.command == "train":
        train(args, device)
    else:
        if args.max_new_tokens < 0:
            parser.error("max-new-tokens 不能为负数。")
        if not math.isfinite(args.temperature) or args.temperature < 0:
            parser.error("temperature 必须是有限的非负数；0 表示 greedy。")
        model, checkpoint = load_model(args.checkpoint, device)
        prompt = args.prompt if args.prompt is not None else "" if isinstance(model.tokenizer, CharacterTokenizer) else "ROMEO:\n"
        result = sample(model, prompt, device, args.max_new_tokens, temperature=args.temperature,
                        seed=args.seed, use_cache=args.use_cache, trace=args.trace)
        print(f"Checkpoint step: {checkpoint['step']}; validation loss: {checkpoint['validation_loss']:.4f}")
        print(f"Temperature: {args.temperature}; seed: {args.seed}")
        print(f"KV cache: {args.use_cache}")
        print(f"Output text: {result['text']!r}")


if __name__ == "__main__":
    main()
