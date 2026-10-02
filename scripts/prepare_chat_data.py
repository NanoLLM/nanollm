#!/usr/bin/env python3
"""Prepare chat train/validation files with filtering, dedup, and seed examples."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "python"))

from chat_template import RECORD_SEPARATOR, normalize_chat_corpus  # noqa: E402


def _load_filter_module():
    filter_path = PROJECT_ROOT / "scripts" / "filter_chat_data.py"
    spec = importlib.util.spec_from_file_location("filter_chat_data", filter_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load filter module from {filter_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_filter_module = None


def filter_chat_corpus(text: str):
    global _filter_module
    if _filter_module is None:
        _filter_module = _load_filter_module()
    return _filter_module.filter_chat_corpus(text)


def _split_turns(text: str) -> list[str]:
    return [turn.strip() for turn in text.split(RECORD_SEPARATOR) if turn.strip()]


def _join_turns(turns: list[str]) -> str:
    if not turns:
        return ""
    return RECORD_SEPARATOR.join(turns) + RECORD_SEPARATOR


def _user_group_key(turn: str) -> str:
    first_line = turn.split("\n", 1)[0].strip()
    if first_line.lower().startswith("user:"):
        first_line = first_line[5:].strip()
    return " ".join(first_line.lower().split())


def _excluded_prompt_keys(prompts: list[str]) -> set[str]:
    return {" ".join(prompt.lower().split()) for prompt in prompts if prompt.strip()}


def _prompt_tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def _prompt_similarity(left: str, right: str) -> float:
    a, b = _prompt_tokens(left), _prompt_tokens(right)
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _filter_excluded_prompts(
    turns: list[str],
    prompts: list[str],
    similarity_threshold: float = 1.0,
) -> tuple[list[str], int]:
    blocked = _excluded_prompt_keys(prompts)
    kept = []
    for turn in turns:
        prompt = _user_group_key(turn)
        exact = prompt in blocked
        fuzzy = any(
            _prompt_similarity(prompt, blocked_prompt) >= similarity_threshold
            for blocked_prompt in blocked
        )
        if not exact and not fuzzy:
            kept.append(turn)
    return kept, len(turns) - len(kept)


def _grouped_train_val_split(
    turns: list[str],
    val_ratio: float,
    seed: int,
) -> tuple[list[str], list[str]]:
    """Split by normalized user prompt clusters to reduce train/val leakage."""
    groups: dict[str, list[str]] = defaultdict(list)
    for turn in turns:
        groups[_user_group_key(turn)].append(turn)

    group_keys = sorted(groups.keys())
    rng = random.Random(seed)
    rng.shuffle(group_keys)

    val_group_count = max(1, int(len(group_keys) * val_ratio)) if len(group_keys) > 1 else 0
    val_keys = set(group_keys[:val_group_count])
    train_turns: list[str] = []
    val_turns: list[str] = []
    for key, items in groups.items():
        if key in val_keys:
            val_turns.extend(items)
        else:
            train_turns.extend(items)
    rng.shuffle(train_turns)
    rng.shuffle(val_turns)
    return train_turns, val_turns


def prepare_chat_data(
    raw_text: str,
    seed_text: str = "",
    val_ratio: float = 0.05,
    seed: int = 42,
    seed_repeats: int = 40,
    excluded_prompts: list[str] | None = None,
    exclude_similarity_threshold: float = 1.0,
) -> tuple[str, str, dict]:
    formatted, norm_stats = normalize_chat_corpus(raw_text)
    filtered, filter_stats = filter_chat_corpus(formatted)

    corpus_turns = _split_turns(filtered)
    corpus_turns, excluded_corpus_turns = _filter_excluded_prompts(
        corpus_turns, excluded_prompts or [], exclude_similarity_threshold
    )
    train_turns, val_turns = _grouped_train_val_split(corpus_turns, val_ratio=val_ratio, seed=seed)

    seed_turns: list[str] = []
    if seed_text.strip():
        normalized_seed, _ = normalize_chat_corpus(seed_text)
        seed_turns = _split_turns(normalized_seed)
        seed_turns, excluded_seed_turns = _filter_excluded_prompts(
            seed_turns, excluded_prompts or [], exclude_similarity_threshold
        )
        if seed_turns:
            seed_pool = seed_turns * seed_repeats
            rng = random.Random(seed + 1)
            rng.shuffle(seed_pool)
            spacing = max(1, math.ceil(len(train_turns) / max(len(seed_pool), 1)))
            mixed_turns: list[str] = []
            seed_idx = 0
            for corpus_idx, turn in enumerate(train_turns, start=1):
                mixed_turns.append(turn)
                if corpus_idx % spacing == 0 and seed_idx < len(seed_pool):
                    mixed_turns.append(seed_pool[seed_idx])
                    seed_idx += 1
            mixed_turns.extend(seed_pool[seed_idx:])
            train_turns = mixed_turns
    else:
        excluded_seed_turns = 0

    stats = {
        **{f"normalize_{k}": v for k, v in norm_stats.items()},
        **{f"filter_{k}": v for k, v in filter_stats.items()},
        "train_turns": len(train_turns),
        "val_turns": len(val_turns),
        "seed_turns": len(seed_turns),
        "seed_repeats": seed_repeats if seed_turns else 0,
        "seed_examples": len(seed_turns) * seed_repeats,
        "group_count": len({_user_group_key(t) for t in corpus_turns}),
        "excluded_prompt_count": len(_excluded_prompt_keys(excluded_prompts or [])),
        "excluded_corpus_turns": excluded_corpus_turns,
        "excluded_seed_turns": excluded_seed_turns,
        "exclude_similarity_threshold": exclude_similarity_threshold,
    }
    return _join_turns(train_turns), _join_turns(val_turns), stats


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare curated chat train/validation files")
    parser.add_argument(
        "--input",
        type=Path,
        default=PROJECT_ROOT / "data/chat/chat_combined.txt",
        help="Raw chat corpus",
    )
    parser.add_argument(
        "--seed-data",
        type=Path,
        default=PROJECT_ROOT / "scripts/chat_capability_seed.txt",
        help="Seed examples prepended before split",
    )
    parser.add_argument(
        "--train-output",
        type=Path,
        default=PROJECT_ROOT / "data/chat/chat_train.txt",
        help="Training output path",
    )
    parser.add_argument(
        "--val-output",
        type=Path,
        default=PROJECT_ROOT / "data/chat/chat_val.txt",
        help="Validation output path",
    )
    parser.add_argument("--val-ratio", type=float, default=0.05, help="Validation turn fraction")
    parser.add_argument("--seed", type=int, default=42, help="Shuffle seed")
    parser.add_argument(
        "--seed-repeats",
        type=int,
        default=40,
        help="Repeat seed turns in train only (oversample verified capability examples)",
    )
    parser.add_argument(
        "--exclude-prompts-file",
        type=Path,
        default=None,
        help="JSON suite containing cases/prompts to exclude from corpus and seed data",
    )
    parser.add_argument(
        "--exclude-similarity-threshold",
        type=float,
        default=1.0,
        help="Also exclude prompts at or above this token-Jaccard similarity (default: exact only)",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print stats only")
    args = parser.parse_args()

    if not args.input.is_file():
        raise SystemExit(f"Input not found: {args.input}")

    raw_text = args.input.read_text(encoding="utf-8")
    seed_text = args.seed_data.read_text(encoding="utf-8") if args.seed_data.is_file() else ""
    excluded_prompts: list[str] = []
    if args.exclude_prompts_file is not None:
        payload = json.loads(args.exclude_prompts_file.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            excluded_prompts = [str(prompt) for prompt in payload]
        elif payload.get("cases"):
            excluded_prompts = [str(case["prompt"]) for case in payload["cases"]]
        else:
            excluded_prompts = [str(prompt) for prompt in payload.get("prompts", [])]
    train_text, val_text, stats = prepare_chat_data(
        raw_text,
        seed_text=seed_text,
        val_ratio=args.val_ratio,
        seed=args.seed,
        seed_repeats=max(1, args.seed_repeats),
        excluded_prompts=excluded_prompts,
        exclude_similarity_threshold=args.exclude_similarity_threshold,
    )

    for key, value in stats.items():
        print(f"{key}={value}")
    print(f"train_chars={len(train_text):,} val_chars={len(val_text):,}")

    if args.dry_run:
        return 0

    args.train_output.parent.mkdir(parents=True, exist_ok=True)
    args.val_output.parent.mkdir(parents=True, exist_ok=True)
    args.train_output.write_text(train_text, encoding="utf-8")
    args.val_output.write_text(val_text, encoding="utf-8")
    print(f"Wrote {args.train_output}")
    print(f"Wrote {args.val_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
