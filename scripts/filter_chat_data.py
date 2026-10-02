#!/usr/bin/env python3
"""Filter formatted chat data to a smaller, higher-quality curated subset."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "python"))

from chat_template import (  # noqa: E402
    ASSISTANT_PREFIX,
    RECORD_SEPARATOR,
    USER_PREFIX,
    format_turn,
)

_NUMBERED_LIST_RE = re.compile(r"^\d+\.\s", re.MULTILINE)
_INSTRUCTION_NOISE = (
    "relevantate",
    "given the sentence",
    "given sentence",
    "identify the",
    "rewrite the",
    "classify the",
    "extract the",
    "hypothesis",
    "premise",
    "entailment",
    "contradiction",
    "neutral label",
    "natural language inference",
    "nli task",
    "sentence pair",
    "formulate a hypothesis",
    "rdf triplet",
    "choose your answer",
    "output 1 for",
    "respond with the label",
)
_GENERIC_USER_PREFIXES = (
    "you are an ai assistant",
    "you are a helpful assistant",
    "you are an ai assistant that",
    "instructions:",
    "instruction:",
)
_MCQ_OPTION_RE = re.compile(r"\b[A-D]\.\s")


def _parse_turn(block: str) -> tuple[str, str] | None:
    block = block.strip()
    if not block.startswith(USER_PREFIX):
        return None
    lines = block.split("\n")
    user = lines[0][len(USER_PREFIX) :].strip()
    assistant_lines: list[str] = []
    for line in lines[1:]:
        if line.startswith(ASSISTANT_PREFIX):
            assistant_lines.append(line[len(ASSISTANT_PREFIX) :].strip())
        elif assistant_lines:
            assistant_lines.append(line.strip())
    assistant = "\n".join(line for line in assistant_lines if line).strip()
    if not user or not assistant:
        return None
    return user, assistant


def _normalize_key(user: str, assistant: str) -> str:
    user_key = " ".join(user.lower().split())
    assistant_key = " ".join(assistant.lower().split())[:240]
    return f"{user_key}\n{assistant_key}"


def is_good_turn(user: str, assistant: str) -> bool:
    if len(user) < 4 or len(assistant) < 8:
        return False
    # Tiny device models learn short question/answer behavior more reliably than
    # long-form document completion.
    if len(user) > 240 or len(assistant) > 320:
        return False

    user_lower = user.lower().strip()
    combined = f"{user} {assistant}".lower()
    if any(phrase in combined for phrase in _INSTRUCTION_NOISE):
        return False
    if any(user_lower.startswith(prefix) for prefix in _GENERIC_USER_PREFIXES):
        return False

    lower = assistant.lower()
    numbered = _NUMBERED_LIST_RE.findall(assistant)
    if len(numbered) >= 3:
        return False

    if assistant.count("?") >= 4:
        return False

    if lower.count("user:") > 0 or lower.count("assistant:") > 0:
        return False

    mcq_hits = _MCQ_OPTION_RE.findall(assistant)
    if len(mcq_hits) >= 2 and len(assistant.split()) <= 24:
        return False

    if user.endswith(".") and "?" not in user and len(user.split()) <= 6 and len(assistant.split()) >= 12:
        return False

    return True


def filter_chat_corpus(text: str) -> tuple[str, dict]:
    blocks = [b.strip() for b in text.split(RECORD_SEPARATOR) if b.strip()]
    kept_turns: list[str] = []
    seen_keys: set[str] = set()
    stats = {
        "turns_in": 0,
        "turns_out": 0,
        "skipped": 0,
        "deduped": 0,
    }

    for block in blocks:
        stats["turns_in"] += 1
        parsed = _parse_turn(block)
        if not parsed:
            stats["skipped"] += 1
            continue
        user, assistant = parsed
        if not is_good_turn(user, assistant):
            stats["skipped"] += 1
            continue
        dedupe_key = _normalize_key(user, assistant)
        if dedupe_key in seen_keys:
            stats["deduped"] += 1
            continue
        seen_keys.add(dedupe_key)
        turn = format_turn(user, assistant)
        if turn:
            kept_turns.append(turn)
            stats["turns_out"] += 1
        else:
            stats["skipped"] += 1

    output = RECORD_SEPARATOR.join(kept_turns)
    if output:
        output += RECORD_SEPARATOR
    return output, stats


def main() -> int:
    parser = argparse.ArgumentParser(description="Filter chat data to a curated subset")
    parser.add_argument(
        "--input",
        type=Path,
        default=PROJECT_ROOT / "data/chat/chat_formatted.txt",
        help="Formatted chat corpus (User/Assistant turns)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "data/chat/chat_curated.txt",
        help="Curated output path",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print stats only")
    args = parser.parse_args()

    if not args.input.is_file():
        raise SystemExit(f"Input not found: {args.input}")

    text = args.input.read_text(encoding="utf-8")
    filtered, stats = filter_chat_corpus(text)
    print(
        f"turns_in={stats['turns_in']} turns_out={stats['turns_out']} "
        f"skipped={stats['skipped']} deduped={stats['deduped']}"
    )
    print(f"output_chars={len(filtered):,}")

    if args.dry_run:
        return 0

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(filtered, encoding="utf-8")
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
