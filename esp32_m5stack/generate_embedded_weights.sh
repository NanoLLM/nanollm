#!/bin/bash
# Script to generate embedded weights header file from trained model

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
CHECKPOINT_PATH="$PROJECT_ROOT/checkpoints/model_best.pt"
OUTPUT_HEADER="$SCRIPT_DIR/src/model_weights.h"

echo "NanoLLM Embedded Weights Generator"
echo "==================================="

# Check if checkpoint exists
if [ ! -f "$CHECKPOINT_PATH" ]; then
    echo "Error: Model checkpoint not found: $CHECKPOINT_PATH"
    echo "Please train a model first:"
    echo "  cd python"
    echo "  python train.py --data your_data.txt"
    exit 1
fi

# Generate header files
echo "Generating embedded weights header..."
cd "$PROJECT_ROOT/python"
python export_weights_header.py \
    --checkpoint "$CHECKPOINT_PATH" \
    --output "$OUTPUT_HEADER" \
    --namespace nanollm

if [ $? -ne 0 ]; then
    echo "Error: Failed to generate weights header file"
    exit 1
fi

# Generate vocab header
VOCAB_HEADER="$SCRIPT_DIR/src/vocab_weights.h"
TOKENIZER_PATH="$PROJECT_ROOT/checkpoints/tokenizer/tokenizer.json"

if [ -f "$TOKENIZER_PATH" ]; then
    echo "Generating embedded vocab header..."
    python export_vocab_header.py \
        --tokenizer "$TOKENIZER_PATH" \
        --output "$VOCAB_HEADER" \
        --namespace nanollm
    
    if [ $? -eq 0 ]; then
        echo "✓ Vocab header generated: $VOCAB_HEADER"
    else
        echo "Warning: Failed to generate vocab header (vocab will use SPIFFS)"
    fi
else
    echo "Warning: Tokenizer not found at $TOKENIZER_PATH"
    echo "  Vocab will need to be loaded from SPIFFS"
fi

if [ $? -eq 0 ]; then
    echo ""
    echo "✓ Header files generated:"
    echo "  - $OUTPUT_HEADER"
    if [ -f "$VOCAB_HEADER" ]; then
        echo "  - $VOCAB_HEADER"
    fi
    echo ""
    echo "To use embedded weights and vocab:"
    echo "1. Uncomment NANOLLM_USE_EMBEDDED_WEIGHTS in platformio.ini"
    echo "2. Uncomment NANOLLM_USE_EMBEDDED_VOCAB in platformio.ini"
    echo "3. Rebuild the project: pio run"
    echo ""
    echo "Weights size: $(du -h "$OUTPUT_HEADER" | cut -f1)"
    if [ -f "$VOCAB_HEADER" ]; then
        echo "Vocab size: $(du -h "$VOCAB_HEADER" | cut -f1)"
    fi
else
    echo "Error: Failed to generate header files"
    exit 1
fi

