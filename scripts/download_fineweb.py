#!/usr/bin/env python3
"""
Download and prepare FineWeb dataset for training.
FineWeb is a large, open web corpus for language model training.

Dataset: https://huggingface.co/datasets/HuggingFaceFW/fineweb
"""
import argparse
import os
import json
from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm


def download_fineweb(
    output_dir,
    sample_size=None,
    split="sample-10BT",
    text_column="text",
    output_file=None,
    metadata_file=None,
    skip_samples=0,
):
    """
    Download FineWeb dataset.
    
    Args:
        output_dir: Directory to save the dataset
        sample_size: Number of samples to download (None for all)
        split: Dataset split to use (sample-10BT, sample-100BT, etc.)
        text_column: Column name containing text
        output_file: Explicit output path (default: output_dir/fineweb.txt)
        metadata_file: Explicit metadata path (default: beside output_file)
        skip_samples: Skip this many leading stream samples before writing
    """
    print(f"Downloading FineWeb dataset (split: {split})...")
    print(f"Output directory: {output_dir}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Load dataset from HuggingFace
        print("Loading dataset from HuggingFace...")
        dataset = load_dataset(
            "HuggingFaceFW/fineweb",
            name=split,
            split="train",
            streaming=True  # Use streaming for large datasets
        )
        
        if output_file is None:
            output_file = os.path.join(output_dir, "fineweb.txt")
        total_samples = 0
        total_chars = 0
        seen = 0
        
        print(f"Writing to {output_file}...")
        if skip_samples:
            print(f"Skipping first {skip_samples:,} stream samples...")
        with open(output_file, 'w', encoding='utf-8') as f:
            for sample in tqdm(dataset, desc="Processing samples"):
                if text_column in sample:
                    text = sample[text_column]
                    if text and len(text.strip()) > 0:
                        seen += 1
                        if seen <= skip_samples:
                            continue
                        # Write text with newline separator
                        f.write(text.strip())
                        f.write('\n\n')  # Double newline for paragraph separation
                        total_samples += 1
                        total_chars += len(text)
                        
                        if sample_size and total_samples >= sample_size:
                            break
        
        print(f"\n✓ Dataset downloaded successfully!")
        print(f"  Samples: {total_samples:,}")
        print(f"  Characters: {total_chars:,}")
        print(f"  File: {output_file}")
        print(f"  Size: {os.path.getsize(output_file) / (1024*1024):.2f} MB")
        
        # Save metadata
        metadata = {
            "dataset": "FineWeb",
            "split": split,
            "samples": total_samples,
            "total_chars": total_chars,
            "file": output_file,
            "skip_samples": skip_samples,
        }
        
        if metadata_file is None:
            metadata_file = str(Path(output_file).with_name(Path(output_file).stem + "_metadata.json"))
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"  Metadata: {metadata_file}")
        
        return output_file
        
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        print("\nTroubleshooting:")
        print("1. Install datasets library: pip install datasets")
        print("2. Check internet connection")
        print("3. Try a smaller sample_size first")
        raise


def main():
    parser = argparse.ArgumentParser(description='Download FineWeb dataset')
    parser.add_argument('--output_dir', type=str, default='./data/fineweb',
                       help='Output directory for dataset')
    parser.add_argument('--output_file', type=str, default=None,
                       help='Explicit output text path (default: output_dir/fineweb.txt)')
    parser.add_argument('--metadata_file', type=str, default=None,
                       help='Explicit metadata JSON path')
    parser.add_argument('--sample_size', type=int, default=None,
                       help='Number of samples to download (None for all)')
    parser.add_argument('--skip_samples', type=int, default=0,
                       help='Skip this many leading stream samples before writing')
    parser.add_argument('--split', type=str, default='sample-10BT',
                       choices=['sample-10BT', 'sample-100BT', 'CC-MAIN-2024-10'],
                       help='Dataset split to use')
    parser.add_argument('--text_column', type=str, default='text',
                       help='Column name containing text')
    
    args = parser.parse_args()
    
    download_fineweb(
        args.output_dir,
        sample_size=args.sample_size,
        split=args.split,
        text_column=args.text_column,
        output_file=args.output_file,
        metadata_file=args.metadata_file,
        skip_samples=args.skip_samples,
    )


if __name__ == '__main__':
    main()

