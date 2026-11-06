"""把 chinese-poetry 的唐诗 JSON 转为训练语料；原仓库只读。

在 my_vllm 目录下运行：python reference/prepare_poetry.py
输出 data/tang_poetry/poems.txt、manifest.json 及原库许可证。
每首正文作为一个文档，以空行分隔，保留原文繁体、标点及逐联换行。
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess
import unicodedata


ROOT = Path(__file__).resolve().parent.parent


def prepare(source: Path, output: Path) -> dict:
    files = sorted((source / "全唐诗").glob("poet.tang.*.json"),
                   key=lambda path: int(path.stem.rsplit(".", 1)[1]))
    if not files:
        raise FileNotFoundError(f"没有找到 {source / '全唐诗'}/poet.tang.*.json")
    counts = Counter()
    documents, seen, source_hashes = [], set(), []
    for path in files:
        raw = path.read_bytes()
        source_hashes.append({"path": path.relative_to(source).as_posix(),
                              "sha256": hashlib.sha256(raw).hexdigest()})
        for poem in json.loads(raw):
            counts["raw_poems"] += 1
            paragraphs = poem.get("paragraphs") if isinstance(poem, dict) else None
            if not isinstance(paragraphs, list) or not paragraphs or not all(isinstance(p, str) for p in paragraphs):
                counts["invalid_structure"] += 1
                continue
            lines = [unicodedata.normalize("NFC", "".join(line.split())) for line in paragraphs]
            body = "\n".join(line for line in lines if line)
            if not body or any(c in "�□" or unicodedata.category(c).startswith("C")
                               for c in body.replace("\n", "")):
                counts["missing_or_invalid_characters"] += 1
                continue
            if any(c.isascii() and c.isalnum() for c in body):
                counts["ascii_letters_or_digits"] += 1
                continue
            # 忽略标点和排版后相同的正文只保留一首，之后才会划分训练/验证集。
            key = "".join(c for c in body if not c.isspace() and not unicodedata.category(c).startswith("P"))
            if not key:
                counts["empty_body"] += 1
                continue
            if key in seen:
                counts["duplicate_bodies"] += 1
                continue
            seen.add(key)
            documents.append(body)
    if len(documents) < 2:
        raise ValueError("清洗后可用诗歌不足两首。")
    corpus = ("\n\n".join(documents) + "\n").encode("utf-8")
    output.mkdir(parents=True, exist_ok=True)
    corpus_path = output / "poems.txt"
    if corpus_path.exists() and corpus_path.read_bytes() != corpus:
        raise FileExistsError("语料输出已有不同内容，请指定新的 --output-dir。")
    result = subprocess.run(["git", "-C", str(source), "rev-parse", "HEAD"],
                            capture_output=True, text=True, check=False)
    manifest = {
        "source": str(source.resolve()), "source_url": "https://github.com/chinese-poetry/chinese-poetry",
        "source_commit": result.stdout.strip() if result.returncode == 0 else None,
        "selection": "全唐诗/poet.tang.*.json (Tang poems only)",
        "source_files": len(files), "source_sha256": source_hashes,
        "preparation": "NFC; remove whitespace inside lines; reject missing/corrupt text and ASCII alphanumerics; deduplicate bodies ignoring punctuation/whitespace; body only",
        "counts": {**dict(counts), "retained_poems": len(documents)},
        "body_characters": sum(map(len, documents)),
        "max_body_characters": max(map(len, documents)),
        "all_corpus_unique_characters": len(set("\n".join(documents))),
        "corpus": str(corpus_path.resolve()), "corpus_bytes": len(corpus),
        "corpus_sha256": hashlib.sha256(corpus).hexdigest(),
        "preparation_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "format": "one poem body per document; blank line separates documents; original traditional Chinese retained",
        "split": "performed by train.py before building tokenizer and token windows",
    }
    corpus_path.write_bytes(corpus)
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output / "SOURCE_LICENSE.txt").write_bytes((source / "LICENSE").read_bytes())
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "data" / "chinese-poetry")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data" / "tang_poetry")
    args = parser.parse_args()
    manifest = prepare(args.source, args.output_dir)
    print(json.dumps({key: value for key, value in manifest.items() if key != "source_sha256"},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
