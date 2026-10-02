#!/bin/bash
# Example training script

# Create sample data if it doesn't exist
if [ ! -f "sample_data.txt" ]; then
    echo "Creating sample data..."
    cat > sample_data.txt << 'EOF'
The quick brown fox jumps over the lazy dog.
Machine learning is a subset of artificial intelligence.
Natural language processing enables computers to understand human language.
Deep learning uses neural networks with multiple layers.
Transformers have revolutionized the field of NLP.
EOF
fi

# Train the model
echo "Starting training..."
cd python
python train.py \
    --data ../sample_data.txt \
    --output_dir ../checkpoints \
    --epochs 10 \
    --batch_size 32 \
    --vocab_size 500 \
    --d_model 64 \
    --n_layers 1 \
    --n_heads 2 \
    --d_ff 128 \
    --block_size 128 \
    --mixed_precision

echo "Training complete!"
echo "Exporting weights..."
python export_weights.py \
    --checkpoint ../checkpoints/model_best.pt \
    --output ../weights/model.bin

echo "Done! Model weights exported to ../weights/model.bin"

