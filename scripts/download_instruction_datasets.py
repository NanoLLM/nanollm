#!/usr/bin/env python3
"""Download and prepare instruction-following datasets for fine-tuning.

Sources:
  - Tulu V2 instruction mix (alpaca, openorca, wizardlm) via HuggingFace datasets.
  - Self-instruct / InstructWild style rows from the ShareGPT Vicuna corpus.

Output:
  data/instruct/instruct_raw.txt          -- raw normalized turns
  data/instruct/instruct_summary.json     -- per-source / per-category stats
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

from datasets import load_dataset  # noqa: E402
from tqdm import tqdm  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "python"))

from chat_template import RECORD_SEPARATOR, format_turn, normalize_chat_corpus  # noqa: E402

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_CATEGORY_MAP: Dict[str, str] = {
    "alpaca": "general",
    "openorca": "reasoning",
    "wizardlm": "general",
    "sharegpt": "general",
}

# Minimal English instruction categories used for balancing downstream.
_CATEGORY_LABELS = sorted([
    "general",
    "reasoning",
    "creative",
    "math",
    "coding",
    "summarization",
    "translation",
    "explanation",
    "list",
    "qa_factual",
])


def _write_turn(
    handle,
    user: str,
    assistant: str,
    category: str = "general",
    source: str = "unknown",
) -> Tuple[int, Dict[str, int]]:
    """Write one canonical turn; return (chars, per-category stats)."""
    turn = format_turn(user, assistant)
    if not turn:
        return 0, {"skipped": 1}
    handle.write(turn)
    handle.write(RECORD_SEPARATOR)
    chars = len(turn)
    return chars, {category: 1}


# ---------------------------------------------------------------------------
# Dataset loaders
# ---------------------------------------------------------------------------


def download_alpaca(output_dir: Path, sample_size: int | None = None) -> Path:
    """tatsu-lab/alpaca (instruction / input / output)."""
    print("[alpaca] Downloading tatsu-lab/alpaca ...")
    output_file = output_dir / "instruct_alpaca.txt"
    ds = load_dataset("tatsu-lab/alpaca", split="train")

    total_chars = 0
    cat_stats: Dict[str, int] = defaultdict(int)
    with open(output_file, "w", encoding="utf-8") as f:
        for sample in tqdm(ds, desc="alpaca", unit="row"):
            user_parts: List[str] = []
            if sample.get("instruction"):
                user_parts.append(sample["instruction"])
            if sample.get("input"):
                user_parts.append(sample["input"])
            user = " ".join(user_parts).strip()
            assistant = (sample.get("output") or "").strip()
            if user and assistant:
                chars, cs = _write_turn(f, user, assistant, category="general", source="alpaca")
                total_chars += chars
                cat_stats.update(cs)
            if sample_size and total_chars >= sample_size:
                break

    print(f"  -> {total_chars:,} chars, {cat_stats}")
    return output_file


def download_openorca(output_dir: Path, sample_size: int | None = None) -> Path:
    """Open-Orca/OpenOrca (question / response)."""
    print("[openorca] Downloading Open-Orca/OpenOrca ...")
    output_file = output_dir / "instruct_openorca.txt"
    ds = load_dataset("Open-Orca/OpenOrca", split="train", streaming=True)

    total_chars = 0
    cat_stats: Dict[str, int] = defaultdict(int)
    count = 0
    with open(output_file, "w", encoding="utf-8") as f:
        for sample in tqdm(ds, desc="openorca", unit="row"):
            user = (sample.get("question") or "").strip()
            assistant = (sample.get("response") or "").strip()
            if user and assistant:
                chars, cs = _write_turn(f, user, assistant, category="reasoning", source="openorca")
                total_chars += chars
                cat_stats.update(cs)
                count += 1
            # sample_size is a character budget (same as alpaca), not a row count.
            if sample_size and total_chars >= sample_size:
                break

    print(f"  -> {total_chars:,} chars ({count} turns), {cat_stats}")
    return output_file


def download_wizardlm(output_dir: Path, sample_size: int | None = None) -> Path:
    """WizardLM/WizardLM_evol_instruct_V2_196k."""
    print("[wizardlm] Downloading WizardLM_evol_instruct_V2_196k ...")
    output_file = output_dir / "instruct_wizardlm.txt"
    ds = load_dataset("WizardLM/WizardLM_evol_instruct_V2_196k", split="train", streaming=True)

    total_chars = 0
    cat_stats: Dict[str, int] = defaultdict(int)
    count = 0
    with open(output_file, "w", encoding="utf-8") as f:
        for sample in tqdm(ds, desc="wizardlm", unit="row"):
            user = (sample.get("instruction") or "").strip()
            assistant = (sample.get("output") or "").strip()
            if user and assistant:
                chars, cs = _write_turn(f, user, assistant, category="general", source="wizardlm")
                total_chars += chars
                cat_stats.update(cs)
                count += 1
            if sample_size and total_chars >= sample_size:
                break

    print(f"  -> {total_chars:,} chars ({count} turns), {cat_stats}")
    return output_file


def download_instruct_wiki(output_dir: Path, sample_size: int | None = None) -> Path:
    """facebook/instruct_wiki_4 (instruct / response)."""
    print("[instruct_wiki] Downloading facebook/instruct_wiki_4 ...")
    output_file = output_dir / "instruct_instructwiki.txt"
    ds = load_dataset("facebook/instruct_wiki_4", split="train", streaming=True)

    total_chars = 0
    cat_stats: Dict[str, int] = defaultdict(int)
    count = 0
    with open(output_file, "w", encoding="utf-8") as f:
        for sample in tqdm(ds, desc="instructwiki", unit="row"):
            user = (sample.get("instruct") or sample.get("instruction") or "").strip()
            assistant = (sample.get("response") or sample.get("answer") or "").strip()
            if user and assistant:
                chars, cs = _write_turn(f, user, assistant, category="explanation", source="instructwiki")
                total_chars += chars
                cat_stats.update(cs)
                count += 1
            if sample_size and total_chars >= sample_size:
                break

    print(f"  -> {total_chars:,} chars ({count} turns), {cat_stats}")
    return output_file


def download_ultrachat_200k(output_dir: Path, sample_size: int | None = None) -> Path:
    """HuggingFaceH4/ultrachat_200k (conversation -> user/assistant pairs)."""
    print("[ultrachat] Downloading HuggingFaceH4/ultrachat_200k ...")
    output_file = output_dir / "instruct_ultrachat.txt"
    ds = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft", streaming=True)

    total_chars = 0
    cat_stats: Dict[str, int] = defaultdict(int)
    count = 0
    with open(output_file, "w", encoding="utf-8") as f:
        for sample in tqdm(ds, desc="ultrachat", unit="row"):
            messages = sample.get("messages", [])
            # Extract last user/assistant pair per sample.
            pairs: List[Tuple[str, str]] = []
            pending_user: str | None = None
            for msg in messages:
                role = msg.get("role", "")
                content = (msg.get("content") or "").strip()
                if not content:
                    continue
                if role == "user":
                    pending_user = content
                elif role == "assistant" and pending_user:
                    pairs.append((pending_user, content))
                    pending_user = None

            for user, assistant in pairs:
                chars, cs = _write_turn(f, user, assistant, category="general", source="ultrachat")
                total_chars += chars
                cat_stats.update(cs)
                count += 1
            if sample_size and total_chars >= sample_size:
                break

    print(f"  -> {total_chars:,} chars ({count} turns), {cat_stats}")
    return output_file


# ---------------------------------------------------------------------------
# Combine
# ---------------------------------------------------------------------------


def combine_datasets(
    output_dir: Path,
    source_files: List[Path],
    output_file: Path,
) -> Path:
    """Merge all source files into one corpus; write summary stats."""
    print(f"\nCombining {len(source_files)} datasets -> {output_file}")

    total_chars = 0
    total_turns = 0
    source_stats: Dict[str, Dict[str, int]] = {}

    with open(output_file, "w", encoding="utf-8") as out:
        for src in source_files:
            if not src.is_file():
                print(f"  skipping (missing): {src.name}")
                continue
            print(f"  adding {src.name} ...")
            text = src.read_text(encoding="utf-8")
            # Normalize the corpus.
            formatted, norm_stats = normalize_chat_corpus(text)
            if not formatted:
                print(f"    -> no usable turns after normalization")
                continue

            out.write(formatted)
            chars = len(formatted)
            total_chars += chars
            total_turns += norm_stats.get("turns_out", 0)

            source_stats[src.stem] = {
                "chars": chars,
                "turns": norm_stats.get("turns_out", 0),
                "blocks_in": norm_stats.get("blocks_in", 0),
                "blocks_out": norm_stats.get("blocks_out", 0),
                "skipped": norm_stats.get("skipped", 0),
            }

    summary = {
        "total_chars": total_chars,
        "total_turns": total_turns,
        "source_count": len(source_files),
        "sources_added": len(source_stats),
        "sources": source_stats,
    }
    summary_path = output_dir / "instruct_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(
        f"  total: {total_turns:,} turns, {total_chars:,} chars "
        f"({total_chars / (1024 * 1024):.2f} MB)"
    )
    print(f"  summary: {summary_path}")
    return output_file


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download and prepare instruction-following datasets"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "instruct",
        help="Output directory (default: data/instruct)",
    )
    parser.add_argument(
        "--sources",
        nargs="+",
        default=["alpaca", "openorca", "wizardlm", "ultrachat"],
        choices=["alpaca", "openorca", "wizardlm", "instructwiki", "ultrachat"],
        help="Which sources to download",
    )
    parser.add_argument(
        "--sample-size-chars",
        type=int,
        default=5_000_000,
        help="Per-source character cap for device-sized runs",
    )
    parser.add_argument(
        "--combine-only",
        action="store_true",
        help="Skip download; only combine existing source files",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    source_files: List[Path] = []
    sample = args.sample_size_chars if not args.combine_only else None

    if not args.combine_only:
        downloaders = {
            "alpaca": download_alpaca,
            "openorca": download_openorca,
            "wizardlm": download_wizardlm,
            "instructwiki": download_instruct_wiki,
            "ultrachat": download_ultrachat_200k,
        }
        for src in args.sources:
            try:
                path = downloaders[src](args.output_dir, sample_size=sample)
                if path and path.is_file():
                    source_files.append(path)
            except Exception as exc:
                print(f"  Warning: {src} failed: {exc}")

    combined = combine_datasets(args.output_dir, source_files, args.output_dir / "instruct_raw.txt")
    print(f"\nDone. Raw corpus: {combined}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
