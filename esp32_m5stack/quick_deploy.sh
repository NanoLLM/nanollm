#!/bin/bash
# Quick deploy script - non-interactive version
# Usage: ./quick_deploy.sh [embedded|spiffs]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
CHECKPOINT_PATH="$PROJECT_ROOT/checkpoints/model_best.pt"
OUTPUT_HEADER="$SCRIPT_DIR/src/model_weights.h"

METHOD=${1:-embedded}

echo "NanoLLM Quick Deploy to M5Stack Cardputer"
echo "Method: $METHOD"
echo ""

# Check checkpoint
if [ ! -f "$CHECKPOINT_PATH" ]; then
    echo "Error: Model checkpoint not found: $CHECKPOINT_PATH"
    exit 1
fi

if [ "$METHOD" == "embedded" ]; then
    # Generate embedded weights
    echo "Generating embedded weights..."
    cd "$PROJECT_ROOT/python"
    python export_weights_header.py \
        --checkpoint "$CHECKPOINT_PATH" \
        --output "$OUTPUT_HEADER" \
        --namespace nanollm
    
    # Enable embedded weights
    sed -i 's/; -DNANOLLM_USE_EMBEDDED_WEIGHTS/-DNANOLLM_USE_EMBEDDED_WEIGHTS/' "$SCRIPT_DIR/platformio.ini" 2>/dev/null || true
    
    cd "$SCRIPT_DIR"
    echo "Building and uploading firmware..."
    pio run --target upload
    
elif [ "$METHOD" == "spiffs" ]; then
    # Export weights
    echo "Exporting weights..."
    cd "$PROJECT_ROOT/python"
    python export_weights.py \
        --checkpoint "$CHECKPOINT_PATH" \
        --output "$PROJECT_ROOT/weights/model.bin"
    
    # Prepare SPIFFS
    mkdir -p "$SCRIPT_DIR/data"
    cp "$PROJECT_ROOT/weights/model.bin" "$SCRIPT_DIR/data/"
    cp "$PROJECT_ROOT/weights/model_config.json" "$SCRIPT_DIR/data/"
    
    # Copy vocab file if it exists
    if [ -f "$PROJECT_ROOT/weights/vocab.json" ]; then
        cp "$PROJECT_ROOT/weights/vocab.json" "$SCRIPT_DIR/data/"
    fi
    
    # Disable embedded weights
    sed -i 's/-DNANOLLM_USE_EMBEDDED_WEIGHTS/; -DNANOLLM_USE_EMBEDDED_WEIGHTS/' "$SCRIPT_DIR/platformio.ini" 2>/dev/null || true
    
    cd "$SCRIPT_DIR"
    echo "Building, uploading filesystem and firmware..."
    pio run --target uploadfs
    pio run --target upload
else
    echo "Error: Invalid method. Use 'embedded' or 'spiffs'"
    exit 1
fi

echo ""
echo "✓ Deployment complete!"
echo "The Cardputer should now be running NanoLLM."

