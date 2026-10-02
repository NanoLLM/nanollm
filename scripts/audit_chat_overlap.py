#!/usr/bin/env python3
"""Fail when frozen transfer prompts leak into chat training artifacts."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "python"))

from chat_template import normalize_chat_corpus  # noqa: E402
from prepare_chat_data import _split_turns, _user_group_key  # noqa: E402


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def _jaccard(left: str, right: str) -> float:
    a, b = _tokens(left), _tokens(right)
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _prompts_from_chat(path: Path) -> list[str]:
    normalized, _ = normalize_chat_corpus(path.read_text(encoding="utf-8"))
    return [_user_group_key(turn) for turn in _split_turns(normalized)]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--chat-file", type=Path, action="append", required=True)
    parser.add_argument("--threshold", type=float, default=0.8)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    payload = json.loads(args.suite.read_text(encoding="utf-8"))
    frozen = [" ".join(str(case["prompt"]).lower().split()) for case in payload["cases"]]
    exact: list[dict] = []
    high_overlap: list[dict] = []
    audited_prompts = 0
    for path in args.chat_file:
        prompts = _prompts_from_chat(path)
        audited_prompts += len(prompts)
        for prompt in prompts:
            for test_prompt in frozen:
                similarity = _jaccard(prompt, test_prompt)
                row = {
                    "file": str(path),
                    "training_prompt": prompt,
                    "transfer_prompt": test_prompt,
                    "jaccard": round(similarity, 4),
                }
                if prompt == test_prompt:
                    exact.append(row)
                elif similarity >= args.threshold:
                    high_overlap.append(row)

    report = {
        "schema_version": 1,
        "suite": str(args.suite),
        "threshold": args.threshold,
        "audited_files": [str(path) for path in args.chat_file],
        "audited_prompts": audited_prompts,
        "frozen_prompts": len(frozen),
        "exact_matches": exact,
        "high_overlap_matches": high_overlap,
        "passed": not exact and not high_overlap,
    }
    text = json.dumps(report, indent=2) + "\n"
    print(text, end="")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
