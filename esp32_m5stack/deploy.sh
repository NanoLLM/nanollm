#!/bin/bash
# Deploy NanoLLM to M5Stack Cardputer
# This script packages weights and flashes firmware to the device

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
CHECKPOINT_PATH="$PROJECT_ROOT/checkpoints/model_best.pt"
OUTPUT_HEADER="$SCRIPT_DIR/src/model_weights.h"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}NanoLLM M5Stack Cardputer Deployment${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Check if checkpoint exists
if [ ! -f "$CHECKPOINT_PATH" ]; then
    echo -e "${RED}Error: Model checkpoint not found: $CHECKPOINT_PATH${NC}"
    echo ""
    echo "Please train a model first:"
    echo "  cd python"
    echo "  python train.py --data your_data.txt"
    exit 1
fi

# Ask user for deployment method
echo -e "${YELLOW}Deployment Method:${NC}"
echo "1) Embedded weights (weights in firmware - recommended)"
echo "2) SPIFFS (weights in file system)"
echo ""
read -p "Select method [1-2] (default: 1): " method
method=${method:-1}

if [ "$method" == "1" ]; then
    echo ""
    echo -e "${BLUE}Step 1: Generating embedded weights header...${NC}"
    
    # Generate header file
    cd "$PROJECT_ROOT/python"
    python export_weights_header.py \
        --checkpoint "$CHECKPOINT_PATH" \
        --output "$OUTPUT_HEADER" \
        --namespace nanollm
    
    if [ $? -ne 0 ]; then
        echo -e "${RED}Failed to generate weights header!${NC}"
        exit 1
    fi
    
    echo -e "${GREEN}✓ Header file generated: $OUTPUT_HEADER${NC}"
    echo ""
    
    # Check if NANOLLM_USE_EMBEDDED_WEIGHTS is enabled
    if ! grep -q "^-DNANOLLM_USE_EMBEDDED_WEIGHTS" "$SCRIPT_DIR/platformio.ini"; then
        echo -e "${YELLOW}Enabling embedded weights in platformio.ini...${NC}"
        # Uncomment the flag
        sed -i 's/; -DNANOLLM_USE_EMBEDDED_WEIGHTS/-DNANOLLM_USE_EMBEDDED_WEIGHTS/' "$SCRIPT_DIR/platformio.ini"
        echo -e "${GREEN}✓ Enabled embedded weights${NC}"
    fi
    
    # Check if vocab header exists and enable embedded vocab
    VOCAB_HEADER="$SCRIPT_DIR/src/vocab_weights.h"
    if [ -f "$VOCAB_HEADER" ]; then
        if ! grep -q "^-DNANOLLM_USE_EMBEDDED_VOCAB" "$SCRIPT_DIR/platformio.ini"; then
            echo -e "${YELLOW}Enabling embedded vocab in platformio.ini...${NC}"
            sed -i 's/; -DNANOLLM_USE_EMBEDDED_VOCAB/-DNANOLLM_USE_EMBEDDED_VOCAB/' "$SCRIPT_DIR/platformio.ini"
            echo -e "${GREEN}✓ Enabled embedded vocab${NC}"
        fi
    else
        echo -e "${YELLOW}Vocab header not found, vocab will use SPIFFS${NC}"
    fi
    
elif [ "$method" == "2" ]; then
    echo ""
    echo -e "${BLUE}Step 1: Preparing SPIFFS files...${NC}"
    
    # Export weights to binary format
    cd "$PROJECT_ROOT/python"
    python export_weights.py \
        --checkpoint "$CHECKPOINT_PATH" \
        --output "$PROJECT_ROOT/weights/model.bin"
    
    if [ $? -ne 0 ]; then
        echo -e "${RED}Failed to export weights!${NC}"
        exit 1
    fi
    
    # Create data directory
    mkdir -p "$SCRIPT_DIR/data"
    
    # Copy model files
    cp "$PROJECT_ROOT/weights/model.bin" "$SCRIPT_DIR/data/"
    cp "$PROJECT_ROOT/weights/model_config.json" "$SCRIPT_DIR/data/"
    
    # Copy vocab file if it exists
    if [ -f "$PROJECT_ROOT/weights/vocab.json" ]; then
        cp "$PROJECT_ROOT/weights/vocab.json" "$SCRIPT_DIR/data/"
        echo -e "${GREEN}  - Copied vocab.json for BPE tokenizer${NC}"
    fi
    
    echo -e "${GREEN}✓ Model files prepared for SPIFFS${NC}"
    echo ""
    
    # Ensure embedded weights are disabled
    if grep -q "^-DNANOLLM_USE_EMBEDDED_WEIGHTS" "$SCRIPT_DIR/platformio.ini"; then
        echo -e "${YELLOW}Disabling embedded weights in platformio.ini...${NC}"
        sed -i 's/-DNANOLLM_USE_EMBEDDED_WEIGHTS/; -DNANOLLM_USE_EMBEDDED_WEIGHTS/' "$SCRIPT_DIR/platformio.ini"
        echo -e "${GREEN}✓ Disabled embedded weights${NC}"
    fi
else
    echo -e "${RED}Invalid selection!${NC}"
    exit 1
fi

# Step 2: Build firmware
echo ""
echo -e "${BLUE}Step 2: Building firmware...${NC}"
cd "$SCRIPT_DIR"

if ! command -v pio &> /dev/null; then
    echo -e "${RED}PlatformIO not found!${NC}"
    echo "Please install PlatformIO:"
    echo "  pip install platformio"
    exit 1
fi

pio run

if [ $? -ne 0 ]; then
    echo -e "${RED}Build failed!${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Firmware built successfully${NC}"
echo ""

# Step 3: Upload filesystem (if using SPIFFS)
if [ "$method" == "2" ]; then
    echo -e "${BLUE}Step 3: Uploading filesystem (SPIFFS)...${NC}"
    echo -e "${YELLOW}Make sure the Cardputer is in download mode:${NC}"
    echo "  1. Set switch to OFF"
    echo "  2. Hold G0 button"
    echo "  3. Connect USB-C"
    echo "  4. Release G0 button"
    echo ""
    read -p "Press Enter when ready..."
    
    pio run --target uploadfs
    
    if [ $? -ne 0 ]; then
        echo -e "${RED}Filesystem upload failed!${NC}"
        exit 1
    fi
    
    echo -e "${GREEN}✓ Filesystem uploaded${NC}"
    echo ""
fi

# Step 4: Upload firmware
echo -e "${BLUE}Step 4: Uploading firmware...${NC}"
if [ "$method" == "2" ]; then
    echo -e "${YELLOW}Cardputer should still be in download mode${NC}"
else
    echo -e "${YELLOW}Make sure the Cardputer is in download mode:${NC}"
    echo "  1. Set switch to OFF"
    echo "  2. Hold G0 button"
    echo "  3. Connect USB-C"
    echo "  4. Release G0 button"
    echo ""
    read -p "Press Enter when ready..."
fi

pio run --target upload

if [ $? -ne 0 ]; then
    echo -e "${RED}Firmware upload failed!${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Firmware uploaded${NC}"
echo ""

# Step 5: Monitor (optional)
echo -e "${BLUE}Step 5: Opening serial monitor...${NC}"
echo -e "${YELLOW}Press Ctrl+C to exit monitor${NC}"
echo ""
sleep 2

pio device monitor

