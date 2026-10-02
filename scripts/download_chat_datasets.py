#!/usr/bin/env python3
"""
Download and prepare chat/conversation datasets for fine-tuning.
Downloads multiple open datasets suitable for chat fine-tuning.
"""
import argparse
import os
import json
import sys
from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "python"))

from chat_template import RECORD_SEPARATOR, format_turn  # noqa: E402


def _write_turn(handle, user: str, assistant: str) -> int:
    turn = format_turn(user, assistant)
    if not turn:
        return 0
    handle.write(turn)
    handle.write(RECORD_SEPARATOR)
    return len(turn)


def download_sharegpt(output_dir, sample_size=None):
    """Download ShareGPT dataset (conversations)."""
    print("Downloading ShareGPT dataset...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        dataset = load_dataset("anon8231489123/ShareGPT_Vicuna_unfiltered", split="train", streaming=True)
        
        output_file = os.path.join(output_dir, "sharegpt.txt")
        total_conversations = 0
        total_chars = 0
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for sample in tqdm(dataset, desc="Processing ShareGPT"):
                if 'conversations' in sample:
                    conversations = sample['conversations']
                    if conversations:
                        pending_user = None
                        for conv in conversations:
                            value = conv.get('value', '').strip()
                            if not value:
                                continue
                            role = conv.get('from', '').lower()
                            if role in ('human', 'user'):
                                pending_user = value
                            elif role in ('gpt', 'assistant', 'chatgpt') and pending_user:
                                total_chars += _write_turn(f, pending_user, value)
                                total_conversations += 1
                                pending_user = None
                        
                        if sample_size and total_conversations >= sample_size:
                            break
        
        print(f"✓ ShareGPT: {total_conversations:,} conversations, {total_chars:,} chars")
        return output_file
        
    except Exception as e:
        print(f"Warning: Could not download ShareGPT: {e}")
        return None


def download_alpaca(output_dir):
    """Download Alpaca dataset (instruction-following)."""
    print("Downloading Alpaca dataset...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        dataset = load_dataset("tatsu-lab/alpaca", split="train")
        
        output_file = os.path.join(output_dir, "alpaca.txt")
        total_samples = 0
        total_chars = 0
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for sample in tqdm(dataset, desc="Processing Alpaca"):
                user_parts = []
                if sample.get('instruction'):
                    user_parts.append(sample['instruction'])
                if sample.get('input'):
                    user_parts.append(sample['input'])
                user = ' '.join(user_parts).strip()
                assistant = (sample.get('output') or '').strip()
                if user and assistant:
                    total_chars += _write_turn(f, user, assistant)
                    total_samples += 1
        
        print(f"✓ Alpaca: {total_samples:,} samples, {total_chars:,} chars")
        return output_file
        
    except Exception as e:
        print(f"Warning: Could not download Alpaca: {e}")
        return None


def download_wizardlm(output_dir, sample_size=None):
    """Download WizardLM dataset."""
    print("Downloading WizardLM dataset...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        dataset = load_dataset("WizardLM/WizardLM_evol_instruct_V2_196k", split="train", streaming=True)
        
        output_file = os.path.join(output_dir, "wizardlm.txt")
        total_samples = 0
        total_chars = 0
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for sample in tqdm(dataset, desc="Processing WizardLM"):
                user = (sample.get('instruction') or '').strip()
                assistant = (sample.get('output') or '').strip()
                if user and assistant:
                    total_chars += _write_turn(f, user, assistant)
                    total_samples += 1

                    if sample_size and total_samples >= sample_size:
                        break
        
        print(f"✓ WizardLM: {total_samples:,} samples, {total_chars:,} chars")
        return output_file
        
    except Exception as e:
        print(f"Warning: Could not download WizardLM: {e}")
        return None


def download_openorca(output_dir, sample_size=None):
    """Download OpenOrca dataset."""
    print("Downloading OpenOrca dataset...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        dataset = load_dataset("Open-Orca/OpenOrca", split="train", streaming=True)
        
        output_file = os.path.join(output_dir, "openorca.txt")
        total_samples = 0
        total_chars = 0
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for sample in tqdm(dataset, desc="Processing OpenOrca"):
                user = (sample.get('question') or '').strip()
                assistant = (sample.get('response') or '').strip()
                if user and assistant:
                    total_chars += _write_turn(f, user, assistant)
                    total_samples += 1

                    if sample_size and total_samples >= sample_size:
                        break
        
        print(f"✓ OpenOrca: {total_samples:,} samples, {total_chars:,} chars")
        return output_file
        
    except Exception as e:
        print(f"Warning: Could not download OpenOrca: {e}")
        return None


def combine_datasets(output_dir, output_file="chat_combined.txt"):
    """Combine all downloaded chat datasets into one file."""
    print(f"\nCombining datasets into {output_file}...")
    
    combined_file = os.path.join(output_dir, output_file)
    dataset_files = [
        os.path.join(output_dir, "sharegpt.txt"),
        os.path.join(output_dir, "alpaca.txt"),
        os.path.join(output_dir, "wizardlm.txt"),
        os.path.join(output_dir, "openorca.txt"),
    ]
    
    total_chars = 0
    files_combined = 0
    
    with open(combined_file, 'w', encoding='utf-8') as outfile:
        for dataset_file in dataset_files:
            if os.path.exists(dataset_file):
                print(f"  Adding {os.path.basename(dataset_file)}...")
                with open(dataset_file, 'r', encoding='utf-8') as infile:
                    content = infile.read()
                    outfile.write(content)
                    total_chars += len(content)
                    files_combined += 1
    
    if files_combined > 0:
        print(f"\n✓ Combined {files_combined} datasets")
        print(f"  Total size: {total_chars:,} characters")
        print(f"  File: {combined_file}")
        print(f"  Size: {os.path.getsize(combined_file) / (1024*1024):.2f} MB")
        return combined_file
    else:
        print("No datasets to combine!")
        return None


def main():
    parser = argparse.ArgumentParser(description='Download chat datasets for fine-tuning')
    parser.add_argument('--output_dir', type=str, default='./data/chat',
                       help='Output directory for datasets')
    parser.add_argument('--datasets', type=str, nargs='+',
                       choices=['sharegpt', 'alpaca', 'wizardlm', 'openorca', 'all'],
                       default=['all'],
                       help='Datasets to download')
    parser.add_argument('--sample_size', type=int, default=None,
                       help='Sample size for streaming datasets (None for all)')
    parser.add_argument('--combine', action='store_true',
                       help='Combine all datasets into one file')
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    datasets_to_download = args.datasets
    if 'all' in datasets_to_download:
        datasets_to_download = ['sharegpt', 'alpaca', 'wizardlm', 'openorca']
    
    downloaded_files = []
    
    if 'sharegpt' in datasets_to_download:
        file = download_sharegpt(args.output_dir, args.sample_size)
        if file:
            downloaded_files.append(file)
    
    if 'alpaca' in datasets_to_download:
        file = download_alpaca(args.output_dir)
        if file:
            downloaded_files.append(file)
    
    if 'wizardlm' in datasets_to_download:
        file = download_wizardlm(args.output_dir, args.sample_size)
        if file:
            downloaded_files.append(file)
    
    if 'openorca' in datasets_to_download:
        file = download_openorca(args.output_dir, args.sample_size)
        if file:
            downloaded_files.append(file)
    
    if args.combine and downloaded_files:
        combine_datasets(args.output_dir)
    
    print(f"\n✓ Download complete!")
    print(f"  Downloaded {len(downloaded_files)} datasets to {args.output_dir}")


if __name__ == '__main__':
    main()

