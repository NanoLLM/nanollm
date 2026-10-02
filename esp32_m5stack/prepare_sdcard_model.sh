#!/usr/bin/env bash
# Export and stage NanoLLM artifacts onto a mounted SD card.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

MODE="dense"
CHECKPOINT_PATH="$PROJECT_ROOT/checkpoints/model_best.pt"
MOUNT_PATH=""
TARGET_DIR="nanollm"
BASENAME="model"
NO_EXPORT=0

print_usage() {
    cat <<EOF
Usage: $(basename "$0") --mount <mounted_sd_path> [options]

Required:
  --mount <path>            Mounted SD card path on host (e.g. /media/
                            $USER/CARDPUTER)

Options:
  --mode <dense|moe>        Export mode (default: dense)
  --checkpoint <path>       Checkpoint path (default: checkpoints/model_best.pt)
  --target-dir <name>       Subdirectory on SD card (default: nanollm)
  --basename <name>         Output base name (default: model)
  --python <binary>         Python executable (default: python)
  --no-export               Skip export and only copy from staging dir
  -h, --help                Show this help message

Examples:
  $(basename "$0") --mount /media/$USER/CARDPUTER --mode dense
  $(basename "$0") --mount /media/$USER/CARDPUTER --mode moe --basename model_moe
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --mount)
            MOUNT_PATH="$2"
            shift 2
            ;;
        --mode)
            MODE="$2"
            shift 2
            ;;
        --checkpoint)
            CHECKPOINT_PATH="$2"
            shift 2
            ;;
        --target-dir)
            TARGET_DIR="$2"
            shift 2
            ;;
        --basename)
            BASENAME="$2"
            shift 2
            ;;
        --python)
            PYTHON_BIN="$2"
            shift 2
            ;;
        --no-export)
            NO_EXPORT=1
            shift
            ;;
        -h|--help)
            print_usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1"
            print_usage
            exit 1
            ;;
    esac
done

if [[ -z "$MOUNT_PATH" ]]; then
    echo "Error: --mount is required"
    print_usage
    exit 1
fi

if [[ "$MODE" != "dense" && "$MODE" != "moe" ]]; then
    echo "Error: --mode must be 'dense' or 'moe'"
    exit 1
fi

if [[ ! -d "$MOUNT_PATH" ]]; then
    echo "Error: mount path does not exist or is not a directory: $MOUNT_PATH"
    exit 1
fi

STAGING_DIR="$PROJECT_ROOT/weights"
OUTPUT_BIN="$STAGING_DIR/${BASENAME}.bin"
OUTPUT_CONFIG="$STAGING_DIR/${BASENAME}_config.json"
OUTPUT_FORMAT="$STAGING_DIR/${BASENAME}_format.json"

if [[ "$NO_EXPORT" -eq 0 ]]; then
    if [[ ! -f "$CHECKPOINT_PATH" ]]; then
        echo "Error: checkpoint not found: $CHECKPOINT_PATH"
        exit 1
    fi

    mkdir -p "$STAGING_DIR"

    echo "Exporting $MODE model artifacts from checkpoint..."
    if [[ "$MODE" == "dense" ]]; then
        "$PYTHON_BIN" "$PROJECT_ROOT/python/export_weights.py" \
            --checkpoint "$CHECKPOINT_PATH" \
            --output "$OUTPUT_BIN"
    else
        "$PYTHON_BIN" "$PROJECT_ROOT/python/export_weights.py" \
            --checkpoint "$CHECKPOINT_PATH" \
            --output "$OUTPUT_BIN" \
            --allow-moe-export
    fi
fi

if [[ ! -f "$OUTPUT_BIN" ]]; then
    echo "Error: missing output binary: $OUTPUT_BIN"
    exit 1
fi
if [[ ! -f "$OUTPUT_CONFIG" ]]; then
    echo "Error: missing output config: $OUTPUT_CONFIG"
    exit 1
fi

DEST_DIR="$MOUNT_PATH/$TARGET_DIR"
mkdir -p "$DEST_DIR"

echo "Copying artifacts to SD card: $DEST_DIR"
cp -f "$OUTPUT_BIN" "$DEST_DIR/"
cp -f "$OUTPUT_CONFIG" "$DEST_DIR/"

# Tokenizer and vocab artifacts are emitted into weights/ by export_weights.py when available.
for optional_file in vocab.json tokenizer_info.json "${BASENAME}_format.json"; do
    if [[ -f "$STAGING_DIR/$optional_file" ]]; then
        cp -f "$STAGING_DIR/$optional_file" "$DEST_DIR/"
    fi
done

MANIFEST="$DEST_DIR/${BASENAME}_sd_manifest.txt"
{
    echo "mode=$MODE"
    echo "checkpoint=$CHECKPOINT_PATH"
    echo "exported_bin=$(basename "$OUTPUT_BIN")"
    echo "exported_config=$(basename "$OUTPUT_CONFIG")"
    if [[ -f "$OUTPUT_FORMAT" ]]; then
        echo "exported_format=$(basename "$OUTPUT_FORMAT")"
    fi
} > "$MANIFEST"

echo ""
echo "Done. Files on SD card:"
ls -lh "$DEST_DIR"
echo ""
echo "Tip: run 'sync' before physically removing the SD card."
