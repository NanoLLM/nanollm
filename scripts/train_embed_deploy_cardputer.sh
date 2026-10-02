#!/usr/bin/env bash
# Train NanoLLM on FineWeb, embed weights + vocab, and deploy to the M5Stack Cardputer.
#
# Usage:
#   ./scripts/train_embed_deploy_cardputer.sh [DATA_FILE]
#
# Environment overrides:
#   EPOCHS (default: 3)
#   BATCH_SIZE (default: 32)
#   BLOCK_SIZE (default: 128)
#   VOCAB_SIZE (default: 256)
#   LEARNING_RATE (default: 3e-4)
#   OUTPUT_DIR (default: checkpoints/cardputer_run_<timestamp>)
#   SAVE_EVERY (default: 1)
#   D_MODEL (default: 48)
#   N_HEADS (default: 4)
#   D_FF (default: 128)
#   DROPOUT (default: 0.1)
#   CHAT_DATA_FILE (default: data/chat/chat_formatted.txt)
#   CHAT_EPOCHS (default: 3)
#   CHAT_BATCH_SIZE (default: 64)
#   CHAT_LEARNING_RATE (default: 1e-4)
#   CHAT_SAVE_EVERY (default: 1)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON_BIN=${PYTHON_BIN:-python3}
PIO_BIN=${PIO_BIN:-pio}
PIO_ENV=${PIO_ENV:-m5stack_cardputer_nopsram}

DATA_FILE="${1:-$PROJECT_ROOT/data/fineweb/fineweb.txt}"
EPOCHS="${EPOCHS:-1}"
BATCH_SIZE="${BATCH_SIZE:-128}"
BLOCK_SIZE="${BLOCK_SIZE:-256}"
VOCAB_SIZE="${VOCAB_SIZE:-10000}"
N_LAYERS="${N_LAYERS:-7}"
LEARNING_RATE="${LEARNING_RATE:-3e-4}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/checkpoints/cardputer_run_$(date +%Y%m%d_%H%M%S)}"
CHECKPOINT_SYMLINK_DIR="$PROJECT_ROOT/checkpoints"
MODEL_HEADER="$PROJECT_ROOT/esp32_m5stack/src/model_weights.h"
VOCAB_HEADER="$PROJECT_ROOT/esp32_m5stack/src/vocab_weights.h"
PLATFORMIO_INI="$PROJECT_ROOT/esp32_m5stack/platformio.ini"
ESP32_DIR="$PROJECT_ROOT/esp32_m5stack"
SAVE_EVERY="${SAVE_EVERY:-1}"
D_MODEL="${D_MODEL:-512}"
N_HEADS="${N_HEADS:-8}"
D_FF="${D_FF:-256}"
DROPOUT="${DROPOUT:-0.1}"
CHAT_DATA_FILE="${CHAT_DATA_FILE:-$PROJECT_ROOT/data/chat/chat_formatted.txt}"
CHAT_EPOCHS="${CHAT_EPOCHS:-3}"
CHAT_BATCH_SIZE="${CHAT_BATCH_SIZE:-64}"
CHAT_LEARNING_RATE="${CHAT_LEARNING_RATE:-1e-4}"
CHAT_SAVE_EVERY="${CHAT_SAVE_EVERY:-1}"

function require_file() {
    local path="$1"
    if [[ ! -f "$path" ]]; then
        echo "Missing file: $path" >&2
        exit 1
    fi
}

function ensure_flag_enabled() {
    local flag="$1"
    local file="$2"
    if grep -q ";[[:space:]]*$flag" "$file"; then
        sed -i "s/;[[:space:]]*$flag/$flag/" "$file"
    fi
    if ! grep -q "^[[:space:]]*$flag" "$file"; then
        sed -i "/^build_flags =/a\    $flag" "$file"
    fi
}

if [[ ! -f "$DATA_FILE" ]]; then
    echo "Training data not found: $DATA_FILE" >&2
    exit 1
fi

if [[ ! -f "$CHAT_DATA_FILE" ]]; then
    RAW_CHAT="${CHAT_RAW:-$PROJECT_ROOT/data/chat/chat_combined.txt}"
    if [[ -f "$RAW_CHAT" ]]; then
        echo "=== Normalizing chat data to User/Assistant format ==="
        "$PYTHON_BIN" "$PROJECT_ROOT/scripts/normalize_chat_data.py" \
            --input "$RAW_CHAT" \
            --output "$CHAT_DATA_FILE"
    fi
fi

if [[ ! -f "$CHAT_DATA_FILE" ]]; then
    echo "Chat fine-tuning data not found: $CHAT_DATA_FILE" >&2
    echo "Run scripts/download_chat_datasets.py or override CHAT_DATA_FILE." >&2
    exit 1
fi


if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    echo "python3 not found (override with PYTHON_BIN)." >&2
    exit 1
fi

if ! command -v "$PIO_BIN" >/dev/null 2>&1; then
    echo "PlatformIO CLI not found (override with PIO_BIN)." >&2
    exit 1
fi

echo "=== Training NanoLLM on $DATA_FILE ==="
mkdir -p "$OUTPUT_DIR"

"$PYTHON_BIN" "$PROJECT_ROOT/python/train.py" \
    --data "$DATA_FILE" \
    --output_dir "$OUTPUT_DIR" \
    --epochs "$EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --learning_rate "$LEARNING_RATE" \
    --block_size "$BLOCK_SIZE" \
    --vocab_size "$VOCAB_SIZE" \
    --d_model "$D_MODEL" \
    --n_layers "$N_LAYERS" \
    --n_heads "$N_HEADS" \
    --d_ff "$D_FF" \
    --dropout "$DROPOUT" \
    --save_every "$SAVE_EVERY"

BEST_CHECKPOINT="$OUTPUT_DIR/model_best.pt"
TOKENIZER_JSON="$OUTPUT_DIR/tokenizer/tokenizer.json"
require_file "$BEST_CHECKPOINT"
require_file "$TOKENIZER_JSON"

BASELINE_CHECKPOINT="$OUTPUT_DIR/model_fineweb.pt"
cp "$BEST_CHECKPOINT" "$BASELINE_CHECKPOINT"
echo "Saved FineWeb baseline checkpoint to $BASELINE_CHECKPOINT"

echo "=== Fine-tuning NanoLLM on chat dataset ($CHAT_DATA_FILE) ==="
"$PYTHON_BIN" "$PROJECT_ROOT/python/train.py" \
    --data "$CHAT_DATA_FILE" \
    --output_dir "$OUTPUT_DIR" \
    --epochs "$CHAT_EPOCHS" \
    --batch_size "$CHAT_BATCH_SIZE" \
    --learning_rate "$CHAT_LEARNING_RATE" \
    --block_size "$BLOCK_SIZE" \
    --vocab_size "$VOCAB_SIZE" \
    --d_model "$D_MODEL" \
    --n_layers "$N_LAYERS" \
    --n_heads "$N_HEADS" \
    --d_ff "$D_FF" \
    --dropout "$DROPOUT" \
    --save_every "$CHAT_SAVE_EVERY" \
    --init_checkpoint "$BASELINE_CHECKPOINT"

BEST_CHECKPOINT="$OUTPUT_DIR/model_best.pt"
require_file "$BEST_CHECKPOINT"

# Quick sanity check so we can spot regressions immediately after training.
echo "=== Running short generation test (debug) ==="
"$PYTHON_BIN" "$PROJECT_ROOT/example_inference.py" \
    --checkpoint "$BEST_CHECKPOINT" \
    --prompt "Hello Cardputer" \
    --max_tokens 20 \
    --temperature 0.8 \
    || echo "Warning: debug generation failed (continuing)"

# Keep legacy paths in sync for other tooling.
mkdir -p "$CHECKPOINT_SYMLINK_DIR"
cp "$BEST_CHECKPOINT" "$CHECKPOINT_SYMLINK_DIR/model_best.pt"
rm -rf "$CHECKPOINT_SYMLINK_DIR/tokenizer"
cp -a "$OUTPUT_DIR/tokenizer" "$CHECKPOINT_SYMLINK_DIR/"

echo "=== Exporting embedded model weights ==="
"$PYTHON_BIN" "$PROJECT_ROOT/python/export_weights_header.py" \
    --checkpoint "$BEST_CHECKPOINT" \
    --output "$MODEL_HEADER" \
    --namespace nanollm

echo "=== Exporting embedded tokenizer vocab ==="
"$PYTHON_BIN" "$PROJECT_ROOT/python/export_vocab_header.py" \
    --tokenizer "$TOKENIZER_JSON" \
    --output "$VOCAB_HEADER" \
    --namespace nanollm

ensure_flag_enabled "-DNANOLLM_USE_EMBEDDED_WEIGHTS" "$PLATFORMIO_INI"
ensure_flag_enabled "-DNANOLLM_USE_EMBEDDED_VOCAB" "$PLATFORMIO_INI"

echo "=== Building firmware (embedded weights) ==="
cd "$ESP32_DIR"
"$PIO_BIN" run -e "$PIO_ENV"

echo "=== Ready to flash ==="
echo "Place the Cardputer in download mode (OFF -> hold G0 -> plug USB -> release)."
read -rp "Press Enter to upload firmware..." _
"$PIO_BIN" run -e "$PIO_ENV" --target upload

echo "=== Deployment complete ==="
echo "You can now open a serial monitor via: $PIO_BIN device monitor"
