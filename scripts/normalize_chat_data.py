#!/usr/bin/env python3
"""Normalize heterogeneous chat corpora to User/Assistant training format."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "python"))

from chat_template import normalize_chat_corpus  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Normalize chat data to User/Assistant format")
    parser.add_argument(
        "--input",
        type=Path,
        default=PROJECT_ROOT / "data/chat/chat_combined.txt",
        help="Raw chat corpus",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "data/chat/chat_formatted.txt",
        help="Normalized output path",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print stats only")
    args = parser.parse_args()

    if not args.input.is_file():
        raise SystemExit(f"Input not found: {args.input}")

    text = args.input.read_text(encoding="utf-8")
    formatted, stats = normalize_chat_corpus(text)

    print(
        f"blocks_in={stats['blocks_in']} blocks_out={stats['blocks_out']} "
        f"turns_out={stats['turns_out']} skipped={stats['skipped']}"
    )
    print(f"output_chars={len(formatted):,}")

    if args.dry_run:
        return 0

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(formatted, encoding="utf-8")
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
