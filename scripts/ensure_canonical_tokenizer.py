#!/usr/bin/env python3
"""Ensure the pinned Cardputer vocab=2048 BPE tokenizer exists under data/tokenizers/."""
from __future__ import annotations

import hashlib
import json
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CANONICAL = PROJECT_ROOT / "data" / "tokenizers" / "cardputer_vocab2048_v1"
DEFAULT_SOURCE = (
    PROJECT_ROOT / "checkpoints" / "moe_run_20260718_045345" / "tokenizer"
)
# Committed fallback (identical tokenizer, MD5 a46f7490099ae81e07f48d6185e8c4ed):
# the checkpoint dir above is gitignored, so a fresh checkout must use this.
FALLBACK_SOURCES = (
    PROJECT_ROOT / "releases" / "cardputer_mqa_ctx224_v1" / "tokenizer",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_canonical(
    canonical_dir: Path = DEFAULT_CANONICAL,
    source_dir: Path = DEFAULT_SOURCE,
) -> dict:
    canonical_dir = canonical_dir.resolve()
    source_dir = source_dir.resolve()
    tokenizer_dst = canonical_dir / "tokenizer.json"
    vocab_info_dst = canonical_dir / "vocab_info.json"
    manifest_path = canonical_dir / "manifest.json"

    if not (source_dir / "tokenizer.json").is_file():
        raise FileNotFoundError(f"Tokenizer source missing: {source_dir / 'tokenizer.json'}")

    canonical_dir.mkdir(parents=True, exist_ok=True)
    for name in ("tokenizer.json", "vocab_info.json"):
        src = source_dir / name
        dst = canonical_dir / name
        if src.is_file():
            shutil.copy2(src, dst)

    manifest = {
        "schema_version": 1,
        "name": "cardputer_vocab2048_v1",
        "vocab_size": 2048,
        "source_dir": str(source_dir),
        "tokenizer_sha256": sha256_file(tokenizer_dst),
        "provenance": "Promoted science-boost lineage (moe_run_20260718_045345)",
    }
    if vocab_info_dst.is_file():
        manifest["vocab_info_sha256"] = sha256_file(vocab_info_dst)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def resolve_source(source_dir: Path) -> Path:
    """Prefer the explicit/local source; fall back to the committed release copy."""
    if (source_dir / "tokenizer.json").is_file():
        return source_dir
    for fallback in FALLBACK_SOURCES:
        if (fallback / "tokenizer.json").is_file():
            print(
                f"Tokenizer source missing: {source_dir} — using committed fallback {fallback}",
                file=sys.stderr,
            )
            return fallback
    raise FileNotFoundError(f"Tokenizer source missing: {source_dir / 'tokenizer.json'}")


def main() -> int:
    canonical = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_CANONICAL
    source = Path(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_SOURCE
    manifest = ensure_canonical(canonical, resolve_source(source))
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
