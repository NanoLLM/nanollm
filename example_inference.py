#!/usr/bin/env python3
"""
Example inference script using the trained model.
"""
import sys
import os

# Add python directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'python'))

import torch
import argparse
import json
from model import NanoLLM
from tokenizer import BPETokenizer


# Removed old vocab loading - now using BPE tokenizer


def main():
    parser = argparse.ArgumentParser(description='Run inference with NanoLLM')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    # Tokenizer is loaded from checkpoint, no need for separate vocab file
    parser.add_argument('--prompt', type=str, default='Hello', help='Input prompt')
    parser.add_argument('--max_tokens', type=int, default=50, help='Maximum tokens to generate')
    parser.add_argument('--temperature', type=float, default=1.0, help='Sampling temperature')
    parser.add_argument('--top_k', type=int, help='Top-k sampling (optional)')
    
    args = parser.parse_args()
    
    # Load checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    config = checkpoint['config']
    
    # Load tokenizer
    tokenizer_path = checkpoint.get('tokenizer_path')
    if tokenizer_path and os.path.exists(tokenizer_path):
        print(f"Loading tokenizer from {tokenizer_path}...")
        tokenizer = BPETokenizer.from_file(tokenizer_path)
    else:
        # Fallback: try to find tokenizer in checkpoint directory
        checkpoint_dir = os.path.dirname(args.checkpoint)
        tokenizer_path = os.path.join(checkpoint_dir, "tokenizer", "tokenizer.json")
        if os.path.exists(tokenizer_path):
            print(f"Loading tokenizer from {tokenizer_path}...")
            tokenizer = BPETokenizer.from_file(tokenizer_path)
        else:
            print("Error: Tokenizer not found!")
            print(f"  Expected at: {checkpoint.get('tokenizer_path', 'N/A')}")
            print(f"  Or at: {tokenizer_path}")
            return
    
    # Create model
    model = NanoLLM(**config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Model config: {config}")
    size_mb, n_params = model.get_model_size_mb(quantized=True)
    print(f"Model size (int8): {size_mb:.2f} MB")
    print(f"Parameters: {n_params:,}")
    print(f"Tokenizer vocab_size: {tokenizer.get_vocab_size()}")
    
    # Encode prompt
    prompt_tokens = tokenizer.encode(args.prompt)
    prompt_tensor = torch.tensor([prompt_tokens], dtype=torch.long)
    
    print(f"\nPrompt: '{args.prompt}'")
    print(f"Prompt tokens: {prompt_tokens}")
    print(f"\nGenerating {args.max_tokens} tokens...\n")
    
    # Generate
    with torch.no_grad():
        generated = model.generate(
            prompt_tensor,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k
        )
    
    generated_tokens = generated[0].tolist()
    
    # Decode
    output_text = tokenizer.decode(generated_tokens)
    
    print("Generated text:")
    print(output_text)
    print(f"\nGenerated tokens: {generated_tokens}")


if __name__ == '__main__':
    main()

