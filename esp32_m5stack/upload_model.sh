#!/bin/bash
# Script to prepare and upload model files to ESP32 SPIFFS

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
WEIGHTS_DIR="$PROJECT_ROOT/weights"
DATA_DIR="$SCRIPT_DIR/data"

echo "NanoLLM Model Upload Script for M5Stack Cardputer"
echo "=================================================="

# Check if model files exist
if [ ! -f "$WEIGHTS_DIR/model.bin" ]; then
    echo "Error: model.bin not found in $WEIGHTS_DIR"
    echo "Please train and export the model first:"
    echo "  cd python"
    echo "  python train.py --data your_data.txt"
    echo "  python export_weights.py --checkpoint ../checkpoints/model_best.pt"
    exit 1
fi

if [ ! -f "$WEIGHTS_DIR/model_config.json" ]; then
    echo "Error: model_config.json not found in $WEIGHTS_DIR"
    exit 1
fi

# Create data directory
mkdir -p "$DATA_DIR"

# Copy model files
echo "Copying model files to data directory..."
cp "$WEIGHTS_DIR/model.bin" "$DATA_DIR/"
cp "$WEIGHTS_DIR/model_config.json" "$DATA_DIR/"

# Copy vocab file if it exists
if [ -f "$WEIGHTS_DIR/vocab.json" ]; then
    cp "$WEIGHTS_DIR/vocab.json" "$DATA_DIR/"
    echo "  - $DATA_DIR/vocab.json"
fi

echo "Files copied:"
echo "  - $DATA_DIR/model.bin ($(du -h "$DATA_DIR/model.bin" | cut -f1))"
echo "  - $DATA_DIR/model_config.json"

echo ""
echo "To upload to ESP32 SPIFFS:"
echo "  cd $SCRIPT_DIR"
echo "  pio run --target uploadfs"
echo ""
echo "Or using Arduino IDE:"
echo "  Tools -> ESP32 Sketch Data Upload"

