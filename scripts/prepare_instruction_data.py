#!/usr/bin/env python3
"""Prepare instruction train/validation files with quality filtering, dedup, and category balancing.

Pipeline:
  1. Normalize raw corpus to User/Assistant turns.
  2. Filter by length, noise, and style heuristics (reuses filter_chat_data logic).
  3. Category-balance by inserting seed examples from each instruction category.
  4. Group-aware train/val split to reduce leakage.

Output:
  data/instruct/instruct_train.txt
  data/instruct/instruct_val.txt
  data/instruct/instruct_prep_stats.json
"""
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


# ---------------------------------------------------------------------------
# Category heuristics: map a turn to one of the 10 instruction categories.
# These are lightweight keyword/pattern rules — not LLM-labeled.
# ---------------------------------------------------------------------------

_CATEGORY_RULES = [
    (
        "math",
        [
            r"(?i)(?:what is|calculate|solve|compute|evaluate)\s",
            r"(?i)(?:plus|minus|times|divided by)\s",
            r"\b\d+\s*[+\-*/]\s*\d+\b",
        ],
    ),
    (
        "coding",
        [
            r"(?i)(?:write a function|implement|code|debug|fix)\s",
            r"(?i)(?:python|javascript|cpp|java)\s",
            r"(?i)def\s+\w+",
        ],
    ),
    (
        "creative",
        [
            r"(?i)(?:write a .*poem|haiku|story|joke|song|lyric|verse)",
            r"(?i)(?:compose|create|imagine|invent)\s",
            r"(?i)(?:in the style of|like a)",
        ],
    ),
    (
        "translation",
        [
            r"(?i)(?:translate|in (?:spanish|french|german|japanese|chinese|korean|arabic))",
            r"(?i)(?:how do you say .+ in)",
        ],
    ),
    (
        "summarization",
        [
            r"(?i)(?:summarize|summary|brief|overview)\s",
            r"(?i)(?:in short|key points|bullet points)",
        ],
    ),
    (
        "explanation",
        [
            r"(?i)(?:why|how|explain|what is|what are|describe)\s",
            r"(?i)(?:because|therefore|however|moreover)",
        ],
    ),
    (
        "qa_factual",
        [
            r"(?i)(?:who is|when did|where is|which|capital of)",
            r"(?i)(?:(?:list|name|tell me) .+ about)",
        ],
    ),
    (
        "list",
        [
            r"(?i)(?:list|give me|name \d+|enumerate)\s",
            r"(?i)(?:top \d+|at least \d+|five .+ examples)",
        ],
    ),
]

_REASONING_KEYWORDS = [
    r"(?i)(?:reason|logic|deduce|infer|conclude|if then|therefore)",
]


def _categorize_turn(user: str, assistant: str) -> str:
    """Heuristically assign one of the 10 instruction categories."""
    combined = f"{user} {assistant}".lower()

    for cat, patterns in _CATEGORY_RULES:
        for pat in patterns:
            if re.search(pat, combined):
                return cat

    for pat in _REASONING_KEYWORDS:
        if re.search(pat, combined):
            return "reasoning"

    return "general"


# ---------------------------------------------------------------------------
# Turn helpers (shared with prepare_chat_data.py)
# ---------------------------------------------------------------------------


def _split_turns(text: str) -> list[str]:
    return [turn.strip() for turn in text.split(RECORD_SEPARATOR) if turn.strip()]


def _join_turns(turns: list[str]) -> str:
    if not turns:
        return ""
    return RECORD_SEPARATOR.join(turns) + RECORD_SEPARATOR


def _user_group_key(turn: str) -> str:
    first_line = turn.split("\n", 1)[0].strip()
    prefix = "User: "
    if first_line.lower().startswith("user:"):
        first_line = first_line[5:].strip()
    return " ".join(first_line.lower().split())


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


# ---------------------------------------------------------------------------
# Main preparation
# ---------------------------------------------------------------------------


def prepare_instruction_data(
    raw_text: str,
    seed_text: str = "",
    val_ratio: float = 0.05,
    seed: int = 42,
    seed_repeats: int = 40,
    max_turns: int | None = None,
) -> tuple[str, str, dict]:
    """Normalize, filter, categorize, balance, and split instruction data."""

    # 1) Normalize
    formatted, norm_stats = normalize_chat_corpus(raw_text)

    # 2) Filter
    filtered, filter_stats = filter_chat_corpus(formatted)

    # 3) Categorize
    turns = _split_turns(filtered)
    categorized: dict[str, list[str]] = defaultdict(list)
    cat_stats: dict[str, int] = defaultdict(int)
    for turn in turns:
        # Extract user text from the turn for categorization.
        lines = turn.split("\n")
        user_part = ""
        if lines and lines[0].startswith("User: "):
            user_part = lines[0][6:]
        assistant_part = "\n".join(l for l in lines[1:] if l.strip())
        if user_part and assistant_part:
            cat = _categorize_turn(user_part, assistant_part)
        else:
            cat = "general"
        categorized[cat].append(turn)
        cat_stats[cat] += 1

    # Flatten back while preserving category labels for seed mixing.
    all_turns: list[str] = []
    for cat in sorted(categorized):
        all_turns.extend(categorized[cat])

    # 4) Apply seed repeats (seed contains category-balanced examples).
    seed_turns: list[str] = []
    if seed_text.strip():
        normalized_seed, _ = normalize_chat_corpus(seed_text)
        seed_turns = _split_turns(normalized_seed)
        if seed_turns:
            seed_pool = seed_turns * seed_repeats
            rng = random.Random(seed + 1)
            rng.shuffle(seed_pool)
            spacing = max(1, math.ceil(len(all_turns) / max(len(seed_pool), 1)))
            mixed: list[str] = []
            seed_idx = 0
            for idx, turn in enumerate(all_turns, start=1):
                mixed.append(turn)
                if idx % spacing == 0 and seed_idx < len(seed_pool):
                    mixed.append(seed_pool[seed_idx])
                    seed_idx += 1
            mixed.extend(seed_pool[seed_idx:])
            all_turns = mixed

    # 5) Cap total turns (optional, for small datasets).
    if max_turns and len(all_turns) > max_turns:
        rng = random.Random(seed)
        rng.shuffle(all_turns)
        all_turns = all_turns[:max_turns]

    # 6) Train/val split.
    train_turns, val_turns = _grouped_train_val_split(all_turns, val_ratio=val_ratio, seed=seed)

    stats = {
        **{f"normalize_{k}": v for k, v in norm_stats.items()},
        **{f"filter_{k}": v for k, v in filter_stats.items()},
        "train_turns": len(train_turns),
        "val_turns": len(val_turns),
        "seed_turns": len(seed_turns),
        "seed_repeats": seed_repeats if seed_turns else 0,
        "seed_examples": len(seed_turns) * seed_repeats,
        "category_counts": dict(cat_stats),
        "total_turns": len(turns),
    }

    return _join_turns(train_turns), _join_turns(val_turns), stats


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Prepare instruction train/validation files with filtering and balancing"
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=PROJECT_ROOT / "data" / "instruct" / "instruct_raw.txt",
        help="Raw instruction corpus",
    )
    parser.add_argument(
        "--seed-data",
        type=Path,
        default=PROJECT_ROOT / "scripts" / "instruction_chat_seed.txt",
        help="Category-balanced seed examples",
    )
    parser.add_argument(
        "--train-output",
        type=Path,
        default=PROJECT_ROOT / "data" / "instruct" / "instruct_train.txt",
        help="Training output path",
    )
    parser.add_argument(
        "--val-output",
        type=Path,
        default=PROJECT_ROOT / "data" / "instruct" / "instruct_val.txt",
        help="Validation output path",
    )
    parser.add_argument(
        "--stats-output",
        type=Path,
        default=PROJECT_ROOT / "data" / "instruct" / "instruct_prep_stats.json",
        help="Preparation stats JSON output",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.05,
        help="Validation split ratio",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--seed-repeats",
        type=int,
        default=40,
        help="How many times to repeat seed examples",
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=None,
        help="Cap total turns (for small-device datasets)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print stats only",
    )
    args = parser.parse_args()

    if not args.input.is_file():
        raise SystemExit(f"Input not found: {args.input}")

    text = args.input.read_text(encoding="utf-8")

    seed_text = ""
    if args.seed_data.is_file():
        seed_text = args.seed_data.read_text(encoding="utf-8")

    train, val, stats = prepare_instruction_data(
        raw_text=text,
        seed_text=seed_text,
        val_ratio=args.val_ratio,
        seed=args.seed,
        seed_repeats=args.seed_repeats,
        max_turns=args.max_turns,
    )

    print(
        f"train_turns={stats['train_turns']} val_turns={stats['val_turns']} "
        f"seed_examples={stats['seed_examples']}"
    )
    print(f"categories: {dict(stats['category_counts'])}")

    if args.dry_run:
        return 0

    args.train_output.parent.mkdir(parents=True, exist_ok=True)
    args.val_output.parent.mkdir(parents=True, exist_ok=True)
    args.stats_output.parent.mkdir(parents=True, exist_ok=True)

    args.train_output.write_text(train, encoding="utf-8")
    args.val_output.write_text(val, encoding="utf-8")
    args.stats_output.write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")

    print(f"Wrote {args.train_output}")
    print(f"Wrote {args.val_output}")
    print(f"Wrote {args.stats_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
