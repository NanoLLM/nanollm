#!/usr/bin/env python3
"""Bootstrap small training inputs for a fresh checkout.

Fresh checkouts have no data/ (gitignored). This script regenerates the
inputs that can be rebuilt without a local GPU/teacher model:

  --tiny    data/fineweb/fineweb_tiny.txt (first 1000 FineWeb docs;
            sliced from fineweb.txt if present, else streamed from HF)
  --3x      data/fineweb/fineweb_3x.txt (baseline + 2x extra FineWeb docs;
            reuses prepare_scaled_training_data.ensure_fineweb_3x)
  --report  Print status of all pipeline data inputs without downloading
            (including GPU-only ones like teacher-distilled chat)

Usage:
  python scripts/ensure_finetune_data.py --report
  python scripts/ensure_finetune_data.py --tiny
  python scripts/ensure_finetune_data.py --tiny --3x
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FINETUNE_DIR = PROJECT_ROOT / "data" / "fineweb"
CHAT_DIR = PROJECT_ROOT / "data" / "chat"

TINY = FINETUNE_DIR / "fineweb_tiny.txt"
BASELINE = FINETUNE_DIR / "fineweb.txt"
THREEX = FINETUNE_DIR / "fineweb_3x.txt"
TEACHER = CHAT_DIR / "teacher_distilled_qwen3_5_3k.txt"
TINY_LINES = 1000


def make_tiny(force: bool = False) -> Path:
    if TINY.is_file() and not force:
        print(f"Reusing existing {TINY}")
        return TINY
    FINETUNE_DIR.mkdir(parents=True, exist_ok=True)
    if BASELINE.is_file():
        with open(BASELINE, "r", encoding="utf-8") as src, \
             open(TINY, "w", encoding="utf-8") as dst:
            for i, line in enumerate(src):
                if i >= TINY_LINES:
                    break
                dst.write(line)
        print(f"Wrote {TINY} (first {TINY_LINES} lines of {BASELINE})")
        return TINY
    print(f"{BASELINE} missing — streaming {TINY_LINES} FineWeb samples from HuggingFace")
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from download_fineweb import download_fineweb
    return Path(download_fineweb(
        str(FINETUNE_DIR),
        sample_size=TINY_LINES,
        output_file=str(TINY),
    ))


def make_3x() -> Path:
    if THREEX.is_file():
        print(f"Reusing existing {THREEX}")
        return THREEX
    print(f"Building {THREEX} (streams ~692 MB from HuggingFace on first run)")
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from prepare_scaled_training_data import ensure_fineweb_3x
    return ensure_fineweb_3x(FINETUNE_DIR, baseline_samples=74400, scale=3, force=False)


ALL_INPUTS = (
    (TINY, "tiny", "python scripts/ensure_finetune_data.py --tiny"),
    (BASELINE, "baseline", "python scripts/download_fineweb.py --output_dir data/fineweb --sample_size 74400"),
    (THREEX, "3x", "python scripts/ensure_finetune_data.py --3x"),
    (TEACHER, "teacher", "python scripts/distill_short_chat.py (needs a local Qwen/Qwen3.5-4B + GPU)"),
)


def report(soft: bool = False, needs: list[str] | None = None) -> int:
    """Report on data inputs.

    needs: optional list of tags (tiny|baseline|3x|teacher) to limit the
           report to; None = all.
    soft:  exit 0 when every *required* (needs) input is present, even if
           other optional inputs are missing.
    """
    missing = []
    for path, tag, how in ALL_INPUTS:
        if needs is not None and tag not in needs:
            continue
        state = "OK      " if path.is_file() else "MISSING"
        print(f"{state} {path.relative_to(PROJECT_ROOT)}")
        if not path.is_file():
            missing.append((path, how))
    if missing:
        print("\nRegenerate with:")
        for path, how in missing:
            print(f"  {path.relative_to(PROJECT_ROOT)}: {how}")
        return 0 if soft else 1
    print("\nAll pipeline data inputs present.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--tiny", action="store_true", help="Ensure data/fineweb/fineweb_tiny.txt")
    group.add_argument("--3x", action="store_true", help="Ensure data/fineweb/fineweb_3x.txt")
    group.add_argument("--report", action="store_true", help="Report data input status only")
    parser.add_argument("--force", action="store_true", help="Regenerate even if present (tiny only)")
    parser.add_argument("--needs", default="",
                        help="Comma list of inputs to check/report (tiny,baseline,3x,teacher). "
                             "Implies --report when --tiny/--3x are not set.")
    parser.add_argument("--soft", action="store_true",
                        help="Exit 0 when the --needs inputs are present, even if others are missing")
    flags = vars(parser.parse_args())
    want_tiny = flags.get("tiny", False)
    want_3x = flags.get("3x", False)
    needs = [x for x in flags["needs"].split(",") if x] or None

    if want_tiny:
        make_tiny(force=flags.get("force", False))
    if want_3x:
        make_3x()
    if flags.get("report", False) or needs is not None or not (want_tiny or want_3x):
        return report(soft=flags.get("soft", False), needs=needs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
