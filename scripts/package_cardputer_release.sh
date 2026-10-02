#!/usr/bin/env bash
# Package a public Cardputer MQA ctx224 release:
#   - Foundation pretrain checkpoint (fine-tune starting point)
#   - Optional chat SFT checkpoint + embedded firmware
#   - Canonical tokenizer + int8 exports + PROGMEM header snapshots
#
# Usage:
#   ./scripts/package_cardputer_release.sh
#     # foundation release (tag: cardputer_mqa_ctx224_v1)
#   VARIANT=chat ./scripts/package_cardputer_release.sh
#     # chat SFT release (tag: cardputer_mqa_ctx224_chat_v1)
#   ./scripts/package_cardputer_release.sh --both
#     # build foundation + chat releases
#   SKIP_FIRMWARE=1 ./scripts/package_cardputer_release.sh
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python3}"
PIO_BIN="${PIO_BIN:-pio}"
PIO_ENV="${PIO_ENV:-m5stack_cardputer_nopsram}"

VARIANT="${VARIANT:-foundation}"
BUILD_BOTH=0
if [[ "${1:-}" == "--both" ]]; then
  BUILD_BOTH=1
  shift
fi

FOUNDATION_CKPT="${FOUNDATION_CKPT:-checkpoints/moe_run_20260718_045345/model_pretrain.pt}"
CHAT_CKPT="${CHAT_CKPT:-checkpoints/science_boost_l18_20260721/final4/model_best.pt}"
CANONICAL_TOKENIZER="${CANONICAL_TOKENIZER:-data/tokenizers/cardputer_vocab2048_v1}"
SKIP_FIRMWARE="${SKIP_FIRMWARE:-0}"
UPDATE_SRC_HEADERS="${UPDATE_SRC_HEADERS:-1}"

log() { echo "[$(date -Is)] $*"; }

require_file() {
  if [[ ! -f "$1" ]]; then
    echo "Missing required file: $1" >&2
    exit 1
  fi
}

package_one_release() {
  local variant="$1"
  local release_tag deploy_ckpt sft_recipe clean_transfer_json description
  case "$variant" in
    foundation)
      release_tag="${RELEASE_TAG:-cardputer_mqa_ctx224_v1}"
      deploy_ckpt="${DEPLOY_CHECKPOINT:-$FOUNDATION_CKPT}"
      sft_recipe="pretrain_only"
      clean_transfer_json=""
      description="Foundation pretrain release for fine-tuning; firmware embeds the pretrain checkpoint."
      ;;
    chat)
      release_tag="${RELEASE_TAG:-cardputer_mqa_ctx224_chat_v1}"
      deploy_ckpt="${DEPLOY_CHECKPOINT:-$CHAT_CKPT}"
      sft_recipe="science_boost_v2_60x"
      clean_transfer_json="releases/cardputer_mqa_ctx224_chat_v1/chat/clean_transfer_summary.json"
      description="Chat SFT release (science-boost); firmware embeds the promoted SFT checkpoint."
      ;;
    *)
      echo "Unknown VARIANT=$variant (use foundation or chat)" >&2
      return 1
      ;;
  esac

  local release_dir="${RELEASE_DIR:-$PROJECT_ROOT/releases/$release_tag}"
  _package_release_impl "$variant" "$release_tag" "$release_dir" "$deploy_ckpt" \
    "$sft_recipe" "$clean_transfer_json" "$description"
}

_package_release_impl() {
  local variant="$1" release_tag="$2" release_dir="$3" deploy_ckpt="$4"
  local sft_recipe="$5" clean_transfer_json="$6" description="$7"

  local model_header_src="$PROJECT_ROOT/esp32_m5stack/src/model_weights.h"
  local vocab_header_src="$PROJECT_ROOT/esp32_m5stack/src/vocab_weights.h"
  local firmware_build_dir="$PROJECT_ROOT/esp32_m5stack/.pio/build/$PIO_ENV"

  log "=== Cardputer release ($variant): $release_tag ==="
  require_file "$FOUNDATION_CKPT"
  require_file "$deploy_ckpt"
  require_file "$CANONICAL_TOKENIZER/tokenizer.json"

  "$PYTHON_BIN" scripts/ensure_canonical_tokenizer.py "$CANONICAL_TOKENIZER" \
    checkpoints/moe_run_20260718_045345/tokenizer

  local foundation_dir="$release_dir/foundation"
  local chat_dir="$release_dir/chat"
  local deploy_dir="$release_dir/deploy"
  local tokenizer_dir="$release_dir/tokenizer"
  mkdir -p "$foundation_dir" "$chat_dir" "$deploy_dir" "$tokenizer_dir"

  log "Copying canonical tokenizer"
  cp -a "$CANONICAL_TOKENIZER/." "$tokenizer_dir/"

  log "Copying foundation checkpoint"
  cp "$FOUNDATION_CKPT" "$foundation_dir/model_pretrain.pt"
  if [[ -f "$(dirname "$FOUNDATION_CKPT")/tokenizer/tokenizer.json" ]]; then
    cp -a "$(dirname "$FOUNDATION_CKPT")/tokenizer" "$foundation_dir/"
  fi

  if [[ -f "$CHAT_CKPT" ]]; then
    log "Copying chat SFT checkpoint"
    cp "$CHAT_CKPT" "$chat_dir/model_best.pt"
    printf '%s\n' "$CHAT_CKPT" > "$chat_dir/checkpoint_path.txt"
    cat > "$chat_dir/sft_recipe.txt" <<EOF
recipe=$sft_recipe
seed_data=scripts/chat_capability_seed_science_boost_v2.txt
stages=chat20:20x1e-4,capability12:12x5e-5,eos8:8x3e-5,final4:4x1e-5
seed_repeats=60
init_pretrain=$FOUNDATION_CKPT
EOF
    if [[ -n "$clean_transfer_json" && -f "$clean_transfer_json" ]]; then
      cp "$clean_transfer_json" "$chat_dir/clean_transfer_summary.json"
    fi
  fi

  log "Exporting deploy int8 weights: $deploy_ckpt"
  "$PYTHON_BIN" python/export_weights.py \
    --checkpoint "$deploy_ckpt" \
    --output "$deploy_dir/model.bin" \
    --allow-moe-export

  log "Exporting deploy PROGMEM headers"
  "$PYTHON_BIN" python/export_weights_header.py \
    --checkpoint "$deploy_ckpt" \
    --output "$deploy_dir/model_weights.h"

  local deploy_tokenizer="$(dirname "$deploy_ckpt")/tokenizer/tokenizer.json"
  if [[ ! -f "$deploy_tokenizer" ]]; then
    deploy_tokenizer="$tokenizer_dir/tokenizer.json"
  fi
  require_file "$deploy_tokenizer"

  "$PYTHON_BIN" python/export_vocab_header.py \
    --tokenizer "$deploy_tokenizer" \
    --output "$deploy_dir/vocab_weights.h"

  if [[ "$UPDATE_SRC_HEADERS" == "1" && "$variant" == "chat" ]]; then
    log "Updating esp32_m5stack/src embedded headers (chat deploy)"
    cp "$deploy_dir/model_weights.h" "$model_header_src"
    cp "$deploy_dir/vocab_weights.h" "$vocab_header_src"
  elif [[ "$UPDATE_SRC_HEADERS" == "1" && "$variant" == "foundation" && "$BUILD_BOTH" != "1" ]]; then
    log "Updating esp32_m5stack/src embedded headers (foundation deploy)"
    cp "$deploy_dir/model_weights.h" "$model_header_src"
    cp "$deploy_dir/vocab_weights.h" "$vocab_header_src"
  fi

  if [[ "$SKIP_FIRMWARE" != "1" ]]; then
    if ! command -v "$PIO_BIN" >/dev/null 2>&1; then
      echo "PlatformIO not found; set SKIP_FIRMWARE=1 or install pio." >&2
      exit 1
    fi
    log "Building firmware ($PIO_ENV) for $variant"
    (cd esp32_m5stack && "$PIO_BIN" run -e "$PIO_ENV")
    local firmware_bin="$firmware_build_dir/firmware.bin"
    require_file "$firmware_bin"
    cp "$firmware_bin" "$deploy_dir/firmware.bin"
  fi

  local manifest="$release_dir/manifest.json"
  log "Writing manifest"
  "$PYTHON_BIN" - "$manifest" "$release_tag" "$release_dir" "$variant" "$description" \
    "$FOUNDATION_CKPT" "$deploy_ckpt" "$CHAT_CKPT" "$sft_recipe" "$clean_transfer_json" <<'PY'
import hashlib
import json
import subprocess
import sys
from pathlib import Path

(manifest_path, tag, release_dir, variant, description,
 foundation, deploy, chat, sft_recipe, clean_json) = sys.argv[1:11]
release = Path(release_dir)
root = release.parent.parent

def sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def file_entry(rel: str) -> dict:
    p = release / rel
    return {
        "path": rel,
        "exists": p.is_file(),
        "size_bytes": p.stat().st_size if p.is_file() else None,
        "sha256": sha256(p) if p.is_file() else None,
    }

git = {"commit_short": None, "commit_full": None, "dirty": False}
try:
    git["commit_short"] = subprocess.check_output(
        ["git", "rev-parse", "--short", "HEAD"], cwd=root, text=True
    ).strip()
    git["commit_full"] = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip()
    git["dirty"] = bool(subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=root, text=True
    ).strip())
except Exception:
    pass

metrics = {}
if clean_json and Path(clean_json).is_file():
    summary = json.loads(Path(clean_json).read_text())
    metrics["clean_transfer"] = summary.get("clean_transfer")
    metrics["category_scores"] = summary.get("category_scores")

payload = {
    "schema_version": 2,
    "release_tag": tag,
    "variant": variant,
    "profile": "cardputer-mqa-ctx224",
    "description": description,
    "git": git,
    "checkpoints": {
        "foundation_pretrain": foundation,
        "deploy_checkpoint": deploy,
        "chat_sft": chat if Path(chat).is_file() else None,
    },
    "sft": {
        "recipe": sft_recipe,
        "metrics": metrics or None,
    },
    "tokenizer": {
        "canonical_dir": "data/tokenizers/cardputer_vocab2048_v1",
        "release_copy": "tokenizer/",
    },
    "cardputer_profile": {
        "vocab_size": 2048,
        "block_size": 224,
        "d_model": 80,
        "n_layers": 18,
        "n_heads": 4,
        "n_kv_heads": 1,
        "moe_n_experts": 4,
        "moe_top_k": 1,
        "moe_shared_d_ff": 160,
        "d_ff": 320,
    },
    "artifacts": {
        "foundation_pretrain": file_entry("foundation/model_pretrain.pt"),
        "chat_model_best": file_entry("chat/model_best.pt"),
        "deploy_model_bin": file_entry("deploy/model.bin"),
        "deploy_firmware": file_entry("deploy/firmware.bin"),
        "deploy_model_weights_h": file_entry("deploy/model_weights.h"),
        "deploy_vocab_weights_h": file_entry("deploy/vocab_weights.h"),
        "tokenizer_json": file_entry("tokenizer/tokenizer.json"),
    },
    "commands": {
        "fine_tune_from_foundation": (
            f"INIT_PRETRAIN=releases/{tag}/foundation/model_pretrain.pt "
            "./scripts/train_science_boost_sft.sh"
        ),
        "flash_firmware": (
            "cd esp32_m5stack && pio run -e m5stack_cardputer_nopsram -t upload"
        ),
        "serial_parity": (
            f"python scripts/verify_cardputer_serial.py --reset --checkpoint {deploy}"
        ),
    },
}
Path(manifest_path).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
print(json.dumps(payload, indent=2))
PY

  ln -sfn "$(cd "$(dirname "$FOUNDATION_CKPT")" && pwd)" "$PROJECT_ROOT/checkpoints/cardputer_foundation_v1"
  if [[ "$variant" == "chat" ]]; then
    ln -sfn "$(cd "$(dirname "$CHAT_CKPT")" && pwd)" "$PROJECT_ROOT/checkpoints/cardputer_chat_v1"
  fi

  log "Release ready: $release_dir"
  log "Manifest: $manifest"
}

if [[ "$BUILD_BOTH" == "1" ]]; then
  UPDATE_SRC_HEADERS=0 package_one_release foundation
  UPDATE_SRC_HEADERS=1 package_one_release chat
else
  package_one_release "$VARIANT"
fi
