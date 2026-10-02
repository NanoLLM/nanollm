#!/bin/bash
# Convenience script to download all datasets

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "NanoLLM Dataset Downloader"
echo "=========================="
echo ""

# Check if datasets library is installed
if ! python -c "import datasets" 2>/dev/null; then
    echo "Installing required packages..."
    pip install datasets huggingface-hub
fi

# Download FineWeb (base training corpus)
echo "1. Downloading FineWeb dataset..."
echo "   (Large web corpus for base training)"
read -p "   Download FineWeb? [y/N]: " download_fineweb
if [[ $download_fineweb =~ ^[Yy]$ ]]; then
    read -p "   Sample size (press Enter for all): " sample_size
    sample_size=${sample_size:-""}
    
    if [ -z "$sample_size" ]; then
        python "$SCRIPT_DIR/download_fineweb.py" --output_dir "$PROJECT_ROOT/data/fineweb"
    else
        python "$SCRIPT_DIR/download_fineweb.py" --output_dir "$PROJECT_ROOT/data/fineweb" --sample_size "$sample_size"
    fi
fi

echo ""

# Download chat datasets (for fine-tuning)
echo "2. Downloading chat datasets..."
echo "   (For fine-tuning on chat/conversation tasks)"
read -p "   Download chat datasets? [y/N]: " download_chat
if [[ $download_chat =~ ^[Yy]$ ]]; then
    read -p "   Which datasets? [all/sharegpt/alpaca/wizardlm/openorca]: " datasets
    datasets=${datasets:-all}
    
    read -p "   Sample size for streaming datasets (press Enter for all): " sample_size
    sample_size=${sample_size:-""}
    
    if [ -z "$sample_size" ]; then
        python "$SCRIPT_DIR/download_chat_datasets.py" --output_dir "$PROJECT_ROOT/data/chat" --datasets "$datasets" --combine
    else
        python "$SCRIPT_DIR/download_chat_datasets.py" --output_dir "$PROJECT_ROOT/data/chat" --datasets "$datasets" --sample_size "$sample_size" --combine
    fi
fi

echo ""
echo "✓ Dataset download complete!"
echo ""
echo "Next steps:"
echo "1. Train base model: python python/train.py --data data/fineweb/fineweb.txt"
echo "2. Fine-tune for chat: python python/train.py --data data/chat/chat_combined.txt"

