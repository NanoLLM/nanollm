#!/usr/bin/env python3
"""
Export BPE tokenizer to a format usable by C++ inference.
Exports vocabulary and merge rules for C++ implementation.
"""
import argparse
import json
import os
from tokenizer import BPETokenizer


def export_tokenizer_for_cpp(tokenizer_path, output_dir):
    """
    Export tokenizer to C++ compatible format.
    Creates a simple vocab mapping file.
    """
    print(f"Loading tokenizer from {tokenizer_path}...")
    tokenizer = BPETokenizer.from_file(tokenizer_path)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Get vocabulary
    vocab = tokenizer.tokenizer.get_vocab()
    vocab_size = len(vocab)
    
    # Create reverse mapping (id -> token)
    id_to_token = {v: k for k, v in vocab.items()}
    
    # Export as JSON for C++ to read
    vocab_export = {
        "vocab_size": vocab_size,
        "id_to_token": id_to_token,
        "token_to_id": vocab
    }
    
    vocab_file = os.path.join(output_dir, "vocab.json")
    with open(vocab_file, 'w', encoding='utf-8') as f:
        json.dump(vocab_export, f, indent=2, ensure_ascii=False)
    
    print(f"✓ Vocabulary exported to {vocab_file}")
    print(f"  Vocab size: {vocab_size}")
    
    # Also save tokenizer path reference
    tokenizer_info = {
        "tokenizer_path": tokenizer_path,
        "vocab_file": vocab_file,
        "vocab_size": vocab_size
    }
    
    info_file = os.path.join(output_dir, "tokenizer_info.json")
    with open(info_file, 'w') as f:
        json.dump(tokenizer_info, f, indent=2)
    
    print(f"  Info file: {info_file}")
    
    return vocab_file


def main():
    parser = argparse.ArgumentParser(description='Export BPE tokenizer for C++')
    parser.add_argument('--tokenizer', type=str, required=True,
                       help='Path to tokenizer.json file')
    parser.add_argument('--output', type=str, default='./weights',
                       help='Output directory for vocab files')
    
    args = parser.parse_args()
    
    export_tokenizer_for_cpp(args.tokenizer, args.output)


if __name__ == '__main__':
    main()

