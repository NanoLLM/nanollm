#!/usr/bin/env python3
"""
Export BPE tokenizer vocabulary to C++ header file for firmware embedding.
Creates a header file with vocab as const arrays that can be compiled into firmware.
"""
import argparse
import json
import os
from tokenizer import BPETokenizer


def export_vocab_to_header(tokenizer_path, output_header_path, namespace="nanollm"):
    """Export tokenizer vocabulary to C++ header file."""
    print(f"Loading tokenizer from: {tokenizer_path}")
    
    tokenizer = BPETokenizer.from_file(tokenizer_path)
    vocab = tokenizer.tokenizer.get_vocab()
    vocab_size = len(vocab)

    with open(tokenizer_path, "r", encoding="utf-8") as f:
        tokenizer_json = json.load(f)

    merges = tokenizer_json.get("model", {}).get("merges", [])
    merge_pairs = []
    missing_merges = 0
    for merge in merges:
        if isinstance(merge, str):
            parts = merge.split()
        elif isinstance(merge, (list, tuple)) and len(merge) == 2:
            parts = list(merge)
        else:
            missing_merges += 1
            continue

        if len(parts) != 2:
            continue

        left, right = parts
        if left in vocab and right in vocab:
            merge_pairs.append((vocab[left], vocab[right]))
        else:
            missing_merges += 1

    if missing_merges:
        print(f"Warning: {missing_merges} merges missing from vocab; they were skipped")
    
    print(f"Vocab size: {vocab_size}")
    
    # Create reverse mapping (id -> token)
    id_to_token = {v: k for k, v in vocab.items()}
    
    # Create output directory
    os.makedirs(os.path.dirname(output_header_path) if os.path.dirname(output_header_path) else '.', exist_ok=True)
    
    # Open header file
    header_name = os.path.basename(output_header_path).upper().replace('.', '_')
    
    with open(output_header_path, 'w') as f:
        # Write header guard and includes
        f.write(f"""#ifndef {header_name}
#define {header_name}

#include <cstdint>
#include <cstddef>
#include <Arduino.h>

namespace {namespace} {{

// Vocabulary size
constexpr size_t VOCAB_SIZE = {vocab_size};

// Token strings stored as PROGMEM
// Format: length byte followed by characters
""")
        
        # Store tokens as length-prefixed strings in PROGMEM
        # We'll create arrays for token data and a lookup table
        f.write("// Token data (length-prefixed strings)\n")
        f.write("struct TokenData {\n")
        f.write("    uint8_t length;\n")
        f.write("    const char* data;\n")
        f.write("};\n\n")
        
        # Create token strings as PROGMEM arrays
        token_strings = []
        token_data_arrays = []
        
        for token_id in range(vocab_size):
            token = id_to_token.get(token_id, "")
            if not token:
                token = "<UNK>"
            
            # Escape the token for C++ and track UTF-8 byte length (not Unicode char count).
            escaped_token = token.replace('\\', '\\\\').replace('"', '\\"').replace('\n', '\\n').replace('\r', '\\r').replace('\t', '\\t')
            token_byte_len = len(token.encode('utf-8'))
            
            # Create PROGMEM string
            array_name = f"TOKEN_{token_id}_DATA"
            token_data_arrays.append((token_id, array_name, escaped_token, token_byte_len))
            
            # Only write comment for first few and last few tokens to reduce file size
            if token_id < 5 or token_id >= vocab_size - 5:
                f.write(f"// Token {token_id}\n")
            f.write(f"const char {array_name}[] PROGMEM = \"{escaped_token}\";\n")
        
        # Create lookup table
        f.write("// Token lookup table\n")
        f.write("const TokenData TOKEN_LOOKUP[VOCAB_SIZE] PROGMEM = {\n")
        for token_id, array_name, token, length in token_data_arrays:
            f.write(f"    {{{length}, {array_name}}},  // {token_id}\n")
        f.write("};\n\n")

        # Merge table (ordered by rank)
        f.write("// BPE merge table (ordered by rank)\n")
        f.write("struct MergeEntry {\n")
        f.write("    uint16_t left;\n")
        f.write("    uint16_t right;\n")
        f.write("};\n\n")

        merge_count = len(merge_pairs)
        f.write(f"constexpr size_t MERGE_COUNT = {merge_count};\n")
        if merge_count:
            f.write("const MergeEntry MERGE_TABLE[MERGE_COUNT] PROGMEM = {\n")
            for idx, (left_id, right_id) in enumerate(merge_pairs):
                line_end = ",\n" if idx < merge_count - 1 else "\n"
                f.write(f"    {{{left_id}, {right_id}}},  // rank {idx}{line_end}")
            f.write("};\n\n")
        else:
            f.write("const MergeEntry* MERGE_TABLE = nullptr;\n\n")
        
        # Helper functions
        f.write(f"""
// Get token string by ID
inline String GetTokenById(int id) {{
    if (id < 0 || id >= VOCAB_SIZE) return "";
    
    TokenData token_data;
    memcpy_P(&token_data, &TOKEN_LOOKUP[id], sizeof(TokenData));
    
    String result = "";
    result.reserve(token_data.length);
    for (int i = 0; i < token_data.length; i++) {{
        char c;
        memcpy_P(&c, &token_data.data[i], 1);
        result += c;
    }}
    return result;
}}

// Find token ID by string (linear search - could be optimized)
inline int FindTokenId(const String& token) {{
    for (int i = 0; i < VOCAB_SIZE; i++) {{
        TokenData token_data;
        memcpy_P(&token_data, &TOKEN_LOOKUP[i], sizeof(TokenData));
        
        String current_token = "";
        current_token.reserve(token_data.length);
        for (int j = 0; j < token_data.length; j++) {{
            char c;
            memcpy_P(&c, &token_data.data[j], 1);
            current_token += c;
        }}
        
        if (current_token == token) {{
            return i;
        }}
    }}
    
    // Try UNK token
    for (int i = 0; i < VOCAB_SIZE; i++) {{
        String unk_token = GetTokenById(i);
        if (unk_token == "<UNK>") {{
            return i;
        }}
    }}
    
    return 0; // Default
}}

}} // namespace {namespace}

#endif // {header_name}
""")
    
    print(f"\nHeader file created: {output_header_path}")
    print(f"Total size: {os.path.getsize(output_header_path) / 1024:.2f} KB")
    
    return output_header_path


def main():
    parser = argparse.ArgumentParser(description='Export BPE tokenizer vocab to C++ header')
    parser.add_argument('--tokenizer', type=str, required=True,
                       help='Path to tokenizer.json file')
    parser.add_argument('--output', type=str, default='vocab_weights.h',
                       help='Output header file path')
    parser.add_argument('--namespace', type=str, default='nanollm',
                       help='C++ namespace')
    
    args = parser.parse_args()
    
    export_vocab_to_header(args.tokenizer, args.output, args.namespace)


if __name__ == '__main__':
    main()

