#!/usr/bin/env python3
"""Deterministically split a text corpus into independent train/validation files.

The splitter operates on document blocks separated by blank lines. Each document
is assigned to train or validation using a stable hash seeded by --seed.
"""

import argparse
import hashlib
import json
from pathlib import Path


def stable_hash_bucket(text: str, seed: int) -> float:
    payload = f"{seed}:{text}".encode("utf-8", errors="ignore")
    digest = hashlib.sha1(payload).hexdigest()
    value = int(digest[:15], 16)
    return value / float(16 ** 15)


def split_corpus(input_path: Path, train_path: Path, val_path: Path, val_ratio: float, seed: int):
    stats = {
        "input": str(input_path),
        "train_output": str(train_path),
        "val_output": str(val_path),
        "val_ratio": val_ratio,
        "seed": seed,
        "documents_total": 0,
        "documents_train": 0,
        "documents_val": 0,
        "chars_train": 0,
        "chars_val": 0,
    }

    def write_doc(doc_text: str, train_f, val_f):
        if not doc_text:
            return
        stats["documents_total"] += 1
        bucket = stable_hash_bucket(doc_text, seed)
        if bucket < val_ratio:
            val_f.write(doc_text)
            val_f.write("\n\n")
            stats["documents_val"] += 1
            stats["chars_val"] += len(doc_text)
        else:
            train_f.write(doc_text)
            train_f.write("\n\n")
            stats["documents_train"] += 1
            stats["chars_train"] += len(doc_text)

    with input_path.open("r", encoding="utf-8", errors="ignore") as src, \
         train_path.open("w", encoding="utf-8") as train_f, \
         val_path.open("w", encoding="utf-8") as val_f:
        doc_lines = []
        for line in src:
            if line.strip() == "":
                if doc_lines:
                    doc_text = "".join(doc_lines).strip("\n")
                    write_doc(doc_text, train_f, val_f)
                    doc_lines = []
                continue
            doc_lines.append(line)

        if doc_lines:
            doc_text = "".join(doc_lines).strip("\n")
            write_doc(doc_text, train_f, val_f)

    stats["bytes_train"] = train_path.stat().st_size if train_path.exists() else 0
    stats["bytes_val"] = val_path.stat().st_size if val_path.exists() else 0

    if stats["documents_train"] == 0 or stats["documents_val"] == 0:
        raise RuntimeError(
            "Split produced an empty train or validation set. "
            "Adjust --val-ratio or use a larger corpus."
        )

    return stats


def main():
    parser = argparse.ArgumentParser(description="Split corpus into independent train/validation files")
    parser.add_argument("--input", required=True, help="Path to source text corpus")
    parser.add_argument("--train-output", required=True, help="Output path for train corpus")
    parser.add_argument("--val-output", required=True, help="Output path for validation corpus")
    parser.add_argument("--val-ratio", type=float, default=0.02, help="Validation document ratio (0, 1)")
    parser.add_argument("--seed", type=int, default=42, help="Seed for deterministic hashing")
    parser.add_argument("--stats-output", default=None, help="Optional JSON path for split stats")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output files")
    args = parser.parse_args()

    if not 0.0 < args.val_ratio < 1.0:
        raise ValueError("--val-ratio must be in (0, 1)")

    input_path = Path(args.input)
    train_path = Path(args.train_output)
    val_path = Path(args.val_output)

    if not input_path.is_file():
        raise FileNotFoundError(f"Input corpus not found: {input_path}")

    for out_path in (train_path, val_path):
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if out_path.exists() and not args.overwrite:
            raise FileExistsError(
                f"Output file already exists: {out_path} (pass --overwrite to replace)"
            )

    stats = split_corpus(
        input_path=input_path,
        train_path=train_path,
        val_path=val_path,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    if args.stats_output:
        stats_path = Path(args.stats_output)
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")

    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
