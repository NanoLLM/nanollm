#!/usr/bin/env python3
"""Build 3x FineWeb pretrain + broader chat mix with a held-out prompt suite."""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
sys.path.insert(0, str(PROJECT_ROOT / "python"))

from download_fineweb import download_fineweb  # noqa: E402
from prepare_chat_data import prepare_chat_data, _split_turns, _join_turns  # noqa: E402
from chat_template import RECORD_SEPARATOR, normalize_chat_corpus  # noqa: E402

# Prompts reserved for held-out eval; excluded from train/val chat text.
HELD_OUT_PROMPTS = [
    "What is 9 plus 6?",
    "Name a primary color.",
    "Explain evaporation in one sentence.",
    "Write a haiku about snow.",
    "Who invented the telephone?",
]


def _user_prompt(turn: str) -> str:
    first = turn.split("\n", 1)[0].strip()
    if first.lower().startswith("user:"):
        first = first[5:].strip()
    return " ".join(first.lower().split())


def _filter_held_out(turns: list[str], held_out: list[str]) -> list[str]:
    blocked = {" ".join(p.lower().split()) for p in held_out}
    return [t for t in turns if _user_prompt(t) not in blocked]


def build_chat_mix(
    distilled_path: Path,
    curated_path: Path,
    seed_path: Path,
    out_dir: Path,
    curated_turns: int,
    seed: int,
    seed_repeats: int,
) -> dict:
    distilled = distilled_path.read_text(encoding="utf-8")
    curated = curated_path.read_text(encoding="utf-8") if curated_path.is_file() else ""
    seed_text = seed_path.read_text(encoding="utf-8") if seed_path.is_file() else ""

    curated_norm, _ = normalize_chat_corpus(curated) if curated.strip() else ("", {})
    curated_turns_list = _split_turns(curated_norm)
    rng = random.Random(seed)
    rng.shuffle(curated_turns_list)
    curated_sample = curated_turns_list[: max(0, curated_turns)]

    distilled_norm, _ = normalize_chat_corpus(distilled)
    distilled_turns = _split_turns(distilled_norm)
    mixed = distilled_turns + curated_sample
    mixed = _filter_held_out(mixed, HELD_OUT_PROMPTS)
    rng.shuffle(mixed)
    raw_mix = _join_turns(mixed)

    out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = out_dir / "chat_scaled_raw.txt"
    raw_path.write_text(raw_mix, encoding="utf-8")

    train_text, val_text, stats = prepare_chat_data(
        raw_mix,
        seed_text=seed_text,
        val_ratio=0.05,
        seed=seed,
        seed_repeats=seed_repeats,
    )
    train_path = out_dir / "chat_scaled_train.txt"
    val_path = out_dir / "chat_scaled_val.txt"
    train_path.write_text(train_text, encoding="utf-8")
    val_path.write_text(val_text, encoding="utf-8")

    held_out_path = out_dir / "held_out_chat_prompts.json"
    held_out_path.write_text(json.dumps({"prompts": HELD_OUT_PROMPTS}, indent=2) + "\n", encoding="utf-8")

    meta = {
        "distilled_path": str(distilled_path),
        "curated_path": str(curated_path),
        "curated_turns_requested": curated_turns,
        "curated_turns_used": len(curated_sample),
        "distilled_turns": len(distilled_turns),
        "mixed_turns_before_prepare": len(mixed),
        "seed_repeats": seed_repeats,
        "train_path": str(train_path),
        "val_path": str(val_path),
        "held_out_path": str(held_out_path),
        "prepare_stats": stats,
    }
    (out_dir / "chat_scaled_metadata.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return meta


def ensure_fineweb_3x(
    output_dir: Path,
    baseline_samples: int,
    scale: int,
    force: bool,
) -> Path:
    target_samples = baseline_samples * scale
    out_file = output_dir / f"fineweb_{scale}x.txt"
    meta_file = output_dir / f"fineweb_{scale}x_metadata.json"
    if out_file.is_file() and not force:
        meta = json.loads(meta_file.read_text()) if meta_file.is_file() else {}
        if int(meta.get("samples", 0)) >= target_samples:
            print(f"Reusing existing {out_file} ({meta.get('samples')} samples)")
            return out_file

    # Prefer concatenating the existing baseline + newly streamed unique docs.
    baseline = output_dir / "fineweb.txt"
    if baseline.is_file() and baseline_samples > 0:
        extra = output_dir / f"fineweb_{scale}x_extra.txt"
        download_fineweb(
            str(output_dir),
            sample_size=target_samples - baseline_samples,
            output_file=str(extra),
            metadata_file=str(output_dir / f"fineweb_{scale}x_extra_metadata.json"),
            skip_samples=baseline_samples,
        )
        with open(out_file, "w", encoding="utf-8") as dest:
            shutil.copyfileobj(open(baseline, "r", encoding="utf-8"), dest)
            dest.write("\n")
            shutil.copyfileobj(open(extra, "r", encoding="utf-8"), dest)
        total_chars = out_file.stat().st_size
        meta = {
            "dataset": "FineWeb",
            "split": "sample-10BT",
            "samples": target_samples,
            "approx_bytes": total_chars,
            "file": str(out_file),
            "scale": scale,
            "baseline_samples": baseline_samples,
            "construction": "baseline_concat_extra_unique",
        }
        meta_file.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
        print(f"Wrote {out_file} (~{total_chars / 1e6:.1f} MB)")
        return out_file

    download_fineweb(
        str(output_dir),
        sample_size=target_samples,
        output_file=str(out_file),
        metadata_file=str(meta_file),
    )
    return out_file


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fineweb-dir", type=Path, default=PROJECT_ROOT / "data/fineweb")
    parser.add_argument("--chat-dir", type=Path, default=PROJECT_ROOT / "data/chat")
    parser.add_argument("--out-dir", type=Path, default=PROJECT_ROOT / "data/scaled")
    parser.add_argument("--baseline-samples", type=int, default=74400)
    parser.add_argument("--scale", type=int, default=3)
    parser.add_argument("--curated-turns", type=int, default=8000)
    parser.add_argument("--seed-repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force-fineweb", action="store_true")
    parser.add_argument("--skip-fineweb", action="store_true")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    if not args.skip_fineweb:
        fineweb = ensure_fineweb_3x(
            args.fineweb_dir,
            baseline_samples=args.baseline_samples,
            scale=args.scale,
            force=args.force_fineweb,
        )
    else:
        fineweb = args.fineweb_dir / f"fineweb_{args.scale}x.txt"

    chat_meta = build_chat_mix(
        distilled_path=args.chat_dir / "teacher_distilled_qwen3_5_3k.txt",
        curated_path=args.chat_dir / "chat_curated.txt",
        seed_path=PROJECT_ROOT / "scripts/chat_capability_seed.txt",
        out_dir=args.out_dir,
        curated_turns=args.curated_turns,
        seed=args.seed,
        seed_repeats=args.seed_repeats,
    )

    summary = {
        "pretrain_data": str(fineweb),
        "chat_train": chat_meta["train_path"],
        "chat_val": chat_meta["val_path"],
        "held_out_prompts": chat_meta["held_out_path"],
        "scale": args.scale,
        "seed_repeats": args.seed_repeats,
        "chat": chat_meta,
    }
    summary_path = args.out_dir / "scaled_data_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Wrote {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
