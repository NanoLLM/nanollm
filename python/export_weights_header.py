#!/usr/bin/env python3
"""
Export PyTorch model weights to C/C++ header file for firmware embedding.
Creates a header file with weights as const arrays that can be compiled into firmware.
"""
import argparse
import torch
import numpy as np
import struct
import json
import os
from model import NanoLLM, MoEFeedForward
from quantize import quantize_weights


TYPES_HEADER = """#include "model_weights_types.h"
"""


def export_quantized_linear(f, prefix, layer, has_bias=True):
    """Write one QuantizedLayer's PROGMEM tensor(s) and return QuantizedLayer init fields."""
    weight = layer.weight.data
    weight_q, scale = quantize_weights(weight)
    rows, cols = weight_q.shape

    f.write(f"// {prefix}: {rows}x{cols}\n")
    f.write(f"constexpr float {prefix}_SCALE = {scale:.8f}f;\n")
    f.write(f"constexpr size_t {prefix}_SIZE = {weight_q.size};\n")
    f.write(f"constexpr size_t {prefix}_ROWS = {rows};\n")
    f.write(f"constexpr size_t {prefix}_COLS = {cols};\n")
    f.write(f"const int8_t {prefix}_DATA[] PROGMEM = {{\n")
    write_array_data(f, weight_q.flatten(), 16)
    f.write("};\n\n")

    bias_init = f"{{ nullptr, 1.0f, 0, 0, 0 }}"
    if has_bias and layer.bias is not None:
        bias_q, bias_scale = quantize_weights(layer.bias)
        f.write(f"constexpr float {prefix}_BIAS_SCALE = {bias_scale:.8f}f;\n")
        f.write(f"constexpr size_t {prefix}_BIAS_SIZE = {bias_q.size};\n")
        f.write(f"const int8_t {prefix}_BIAS_DATA[] PROGMEM = {{\n")
        write_array_data(f, bias_q.flatten(), 16)
        f.write("};\n\n")
        bias_init = (
            f"{{ {prefix}_BIAS_DATA, {prefix}_BIAS_SCALE, "
            f"{prefix}_BIAS_SIZE, 0, 0 }}"
        )
    elif has_bias:
        f.write(f"constexpr float {prefix}_BIAS_SCALE = 1.0f;\n")
        f.write(f"constexpr size_t {prefix}_BIAS_SIZE = 0;\n")
        f.write(f"const int8_t* {prefix}_BIAS_DATA = nullptr;\n\n")

    weight_init = (
        f"{{ {prefix}_DATA, {prefix}_SCALE, {prefix}_SIZE, "
        f"{prefix}_ROWS, {prefix}_COLS }}"
    )
    return weight_init, bias_init


def export_layernorm(f, prefix, norm_layer):
    weight = norm_layer.weight.data
    bias = norm_layer.bias.data if norm_layer.bias is not None else None

    weight_q, scale = quantize_weights(weight)
    f.write(f"constexpr float {prefix}_WEIGHT_SCALE = {scale:.8f}f;\n")
    f.write(f"constexpr size_t {prefix}_WEIGHT_SIZE = {weight_q.size};\n")
    f.write(f"const int8_t {prefix}_WEIGHT_DATA[] PROGMEM = {{\n")
    write_array_data(f, weight_q.flatten(), 16)
    f.write("};\n\n")

    if bias is not None:
        bias_q, bias_scale = quantize_weights(bias)
        f.write(f"constexpr float {prefix}_BIAS_SCALE = {bias_scale:.8f}f;\n")
        f.write(f"constexpr size_t {prefix}_BIAS_SIZE = {bias_q.size};\n")
        f.write(f"const int8_t {prefix}_BIAS_DATA[] PROGMEM = {{\n")
        write_array_data(f, bias_q.flatten(), 16)
        f.write("};\n\n")
    else:
        f.write(f"constexpr float {prefix}_BIAS_SCALE = 1.0f;\n")
        f.write(f"constexpr size_t {prefix}_BIAS_SIZE = 0;\n")
        f.write(f"const int8_t* {prefix}_BIAS_DATA = nullptr;\n\n")

    return (
        f"{{ {prefix}_WEIGHT_DATA, {prefix}_WEIGHT_SCALE, {prefix}_WEIGHT_SIZE, 0, 0 }}",
        f"{{ {prefix}_BIAS_DATA, {prefix}_BIAS_SCALE, {prefix}_BIAS_SIZE, 0, 0 }}",
    )


def export_weights_to_header(checkpoint_path, output_header_path, namespace="nanollm"):
    """Export model weights to C++ header file."""
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    config = checkpoint.get('config', {})
    use_rope = bool(config.get('use_rope', False))
    config['n_kv_heads'] = int(config.get('n_kv_heads') or config['n_heads'])
    model = NanoLLM(**config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Model config: {config}")
    size_mb, n_params = model.get_model_size_mb(quantized=True)
    print(f"Model size (int8): {size_mb:.2f} MB")
    
    # Create output directory
    os.makedirs(os.path.dirname(output_header_path) if os.path.dirname(output_header_path) else '.', exist_ok=True)
    
    # Open header file
    header_name = os.path.basename(output_header_path).upper().replace('.', '_')
    
    with open(output_header_path, 'w') as f:
        # Write header guard and includes
        f.write(f"""#ifndef {header_name}
#define {header_name}

#include <pgmspace.h>
{TYPES_HEADER}
namespace {namespace} {{

// Quantization scales and PROGMEM weight blobs follow.
""")
        
        scale_idx = 0
        scales = []
        
        # Export token embedding
        weight = model.token_embedding.weight.data
        weight_q, scale = quantize_weights(weight)
        scales.append(scale)
        rows, cols = weight_q.shape
        
        f.write(f"// Token embedding: {rows}x{cols}\n")
        f.write(f"constexpr float TOKEN_EMBEDDING_SCALE = {scale:.8f}f;\n")
        f.write(f"constexpr size_t TOKEN_EMBEDDING_SIZE = {weight_q.size};\n")
        f.write(f"constexpr size_t TOKEN_EMBEDDING_ROWS = {rows};\n")
        f.write(f"constexpr size_t TOKEN_EMBEDDING_COLS = {cols};\n")
        f.write(f"const int8_t TOKEN_EMBEDDING_DATA[] PROGMEM = {{\n")
        write_array_data(f, weight_q.flatten(), 16)
        f.write("};\n\n")
        
        if use_rope:
            f.write("// RoPE enabled: no learned position embedding table\n")
            f.write("constexpr float POS_EMBEDDING_SCALE = 1.0f;\n")
            f.write("constexpr size_t POS_EMBEDDING_SIZE = 0;\n")
            f.write("constexpr size_t POS_EMBEDDING_ROWS = 0;\n")
            f.write("constexpr size_t POS_EMBEDDING_COLS = 0;\n")
            f.write("inline constexpr const int8_t* POS_EMBEDDING_DATA = nullptr;\n\n")
        else:
            # Export position embedding
            weight = model.pos_embedding.weight.data
            weight_q, scale = quantize_weights(weight)
            scales.append(scale)
            rows, cols = weight_q.shape
            
            f.write(f"// Position embedding: {rows}x{cols}\n")
            f.write(f"constexpr float POS_EMBEDDING_SCALE = {scale:.8f}f;\n")
            f.write(f"constexpr size_t POS_EMBEDDING_SIZE = {weight_q.size};\n")
            f.write(f"constexpr size_t POS_EMBEDDING_ROWS = {rows};\n")
            f.write(f"constexpr size_t POS_EMBEDDING_COLS = {cols};\n")
            f.write(f"const int8_t POS_EMBEDDING_DATA[] PROGMEM = {{\n")
            write_array_data(f, weight_q.flatten(), 16)
            f.write("};\n\n")
        
        use_moe = bool(config.get('use_moe', False))
        block_inits = []

        for block_idx, block in enumerate(model.blocks):
            f.write(f"// Block {block_idx}\n")
            attn_inits = []
            for attn_name, attn_layer in [
                ('Q', block.attention.w_q),
                ('K', block.attention.w_k),
                ('V', block.attention.w_v),
                ('O', block.attention.w_o),
            ]:
                w_init, _ = export_quantized_linear(
                    f, f"BLOCK_{block_idx}_ATTN_{attn_name}", attn_layer, has_bias=False
                )
                attn_inits.append(w_init)

            norm1_w, norm1_b = export_layernorm(f, f"BLOCK_{block_idx}_NORM1", block.norm1)
            norm2_w, norm2_b = export_layernorm(f, f"BLOCK_{block_idx}_NORM2", block.norm2)

            ff = block.feed_forward
            if isinstance(ff, MoEFeedForward):
                n_experts = ff.n_experts
                top_k = ff.top_k
                expert_d_ff = ff.experts[0].linear1.out_features
                has_shared = ff.shared_expert is not None
                shared_d_ff = ff.shared_expert.linear1.out_features if has_shared else 0

                expert_layer_inits = []
                for expert_idx, expert in enumerate(ff.experts):
                    ff1_w, ff1_b = export_quantized_linear(
                        f, f"BLOCK_{block_idx}_E{expert_idx}_FF1", expert.linear1
                    )
                    ff2_w, ff2_b = export_quantized_linear(
                        f, f"BLOCK_{block_idx}_E{expert_idx}_FF2", expert.linear2
                    )
                    expert_layer_inits.append(
                        f"    {{ {ff1_w}, {ff1_b}, {ff2_w}, {ff2_b} }}"
                    )

                f.write(f"static const MoEExpertLayers BLOCK_{block_idx}_EXPERTS[{n_experts}] PROGMEM = {{\n")
                f.write(",\n".join(expert_layer_inits))
                f.write("\n};\n\n")

                router_w, _ = export_quantized_linear(
                    f, f"BLOCK_{block_idx}_MOE_ROUTER", ff.router, has_bias=False
                )

                shared_inits = ["{ nullptr, 1.0f, 0, 0, 0 }"] * 4
                if has_shared:
                    s_ff1_w, s_ff1_b = export_quantized_linear(
                        f, f"BLOCK_{block_idx}_SHARED_FF1", ff.shared_expert.linear1
                    )
                    s_ff2_w, s_ff2_b = export_quantized_linear(
                        f, f"BLOCK_{block_idx}_SHARED_FF2", ff.shared_expert.linear2
                    )
                    shared_inits = [s_ff1_w, s_ff1_b, s_ff2_w, s_ff2_b]

                block_inits.append(
                    f"    {{ // Block {block_idx}\n"
                    f"        {attn_inits[0]}, {attn_inits[1]}, {attn_inits[2]}, {attn_inits[3]},\n"
                    f"        {norm1_w}, {norm1_b}, {norm2_w}, {norm2_b},\n"
                    f"        true,\n"
                    f"        {{ nullptr, 1.0f, 0, 0, 0 }}, {{ nullptr, 1.0f, 0, 0, 0 }},\n"
                    f"        {{ nullptr, 1.0f, 0, 0, 0 }}, {{ nullptr, 1.0f, 0, 0, 0 }},\n"
                    f"        {n_experts}, {top_k}, {expert_d_ff}, {'true' if has_shared else 'false'}, {shared_d_ff},\n"
                    f"        {router_w},\n"
                    f"        BLOCK_{block_idx}_EXPERTS,\n"
                    f"        {shared_inits[0]}, {shared_inits[1]}, {shared_inits[2]}, {shared_inits[3]}\n"
                    f"    }}"
                )
            else:
                ff1_w, ff1_b = export_quantized_linear(f, f"BLOCK_{block_idx}_FF1", ff.linear1)
                ff2_w, ff2_b = export_quantized_linear(f, f"BLOCK_{block_idx}_FF2", ff.linear2)
                block_inits.append(
                    f"    {{ // Block {block_idx}\n"
                    f"        {attn_inits[0]}, {attn_inits[1]}, {attn_inits[2]}, {attn_inits[3]},\n"
                    f"        {norm1_w}, {norm1_b}, {norm2_w}, {norm2_b},\n"
                    f"        false,\n"
                    f"        {ff1_w}, {ff1_b}, {ff2_w}, {ff2_b},\n"
                    f"        0, 1, 0, false, 0,\n"
                    f"        {{ nullptr, 1.0f, 0, 0, 0 }},\n"
                    f"        nullptr,\n"
                    f"        {{ nullptr, 1.0f, 0, 0, 0 }}, {{ nullptr, 1.0f, 0, 0, 0 }},\n"
                    f"        {{ nullptr, 1.0f, 0, 0, 0 }}, {{ nullptr, 1.0f, 0, 0, 0 }}\n"
                    f"    }}"
                )
        
        # Final norm
        weight = model.norm.weight.data
        bias = model.norm.bias.data if model.norm.bias is not None else None
        
        weight_q, scale = quantize_weights(weight)
        scales.append(scale)
        
        f.write("// Final layer norm\n")
        f.write(f"constexpr float FINAL_NORM_WEIGHT_SCALE = {scale:.8f}f;\n")
        f.write(f"constexpr size_t FINAL_NORM_WEIGHT_SIZE = {weight_q.size};\n")
        f.write(f"const int8_t FINAL_NORM_WEIGHT_DATA[] PROGMEM = {{\n")
        write_array_data(f, weight_q.flatten(), 16)
        f.write("};\n\n")
        
        if bias is not None:
            bias_q, bias_scale = quantize_weights(bias)
            scales.append(bias_scale)
            f.write(f"constexpr float FINAL_NORM_BIAS_SCALE = {bias_scale:.8f}f;\n")
            f.write(f"constexpr size_t FINAL_NORM_BIAS_SIZE = {bias_q.size};\n")
            f.write(f"const int8_t FINAL_NORM_BIAS_DATA[] PROGMEM = {{\n")
            write_array_data(f, bias_q.flatten(), 16)
            f.write("};\n\n")
        else:
            f.write(f"constexpr float FINAL_NORM_BIAS_SCALE = 1.0f;\n")
            f.write(f"constexpr size_t FINAL_NORM_BIAS_SIZE = 0;\n")
            f.write(f"const int8_t* FINAL_NORM_BIAS_DATA = nullptr;\n\n")
        
        # LM head (tied to token embedding at train time — reuse PROGMEM blob)
        f.write("// LM head (tied to token embedding)\n")
        f.write(
            "static const QuantizedLayer LM_HEAD_LAYER PROGMEM = "
            "{ TOKEN_EMBEDDING_DATA, TOKEN_EMBEDDING_SCALE, TOKEN_EMBEDDING_SIZE, "
            "TOKEN_EMBEDDING_ROWS, TOKEN_EMBEDDING_COLS };\n\n"
        )
        
        n_layers = config['n_layers']

        f.write("// Transformer blocks array\n")
        f.write(f"static const TransformerBlock BLOCKS_ARRAY[{n_layers}] PROGMEM = {{\n")
        f.write(",\n".join(block_inits))
        f.write("\n};\n\n")

        f.write("// Complete embedded weights structure (weights live in flash, not RAM)\n")
        f.write("static const EmbeddedWeights EMBEDDED_WEIGHTS PROGMEM = {\n")
        f.write(f"    {'true' if use_moe else 'false'},\n")
        f.write("    { TOKEN_EMBEDDING_DATA, TOKEN_EMBEDDING_SCALE, TOKEN_EMBEDDING_SIZE, 0, 0 },\n")
        if use_rope:
            f.write("    { nullptr, 1.0f, 0, 0, 0 },\n")
        else:
            f.write("    { POS_EMBEDDING_DATA, POS_EMBEDDING_SCALE, POS_EMBEDDING_SIZE, 0, 0 },\n")
        f.write("    BLOCKS_ARRAY,\n")
        f.write(f"    {n_layers},\n")
        f.write("    { FINAL_NORM_WEIGHT_DATA, FINAL_NORM_WEIGHT_SCALE, FINAL_NORM_WEIGHT_SIZE, 0, 0 },\n")
        f.write("    { FINAL_NORM_BIAS_DATA, FINAL_NORM_BIAS_SCALE, FINAL_NORM_BIAS_SIZE, 0, 0 },\n")
        f.write("    LM_HEAD_LAYER\n")
        f.write("};\n\n")

        f.write(f"""
inline ModelConfig GetModelConfig() {{
    ModelConfig cfg;
    cfg.vocab_size = {config['vocab_size']};
    cfg.d_model = {config['d_model']};
    cfg.n_layers = {config['n_layers']};
    cfg.n_heads = {config['n_heads']};
    cfg.n_kv_heads = {config['n_kv_heads']};
    cfg.d_ff = {config['d_ff']};
    cfg.max_seq_len = {config['max_seq_len']};
    cfg.quantized = true;
    cfg.use_moe = {'true' if use_moe else 'false'};
    cfg.moe_n_experts = {config.get('moe_n_experts', 0)};
    cfg.moe_top_k = {config.get('moe_top_k', 1)};
    cfg.moe_shared_d_ff = {config.get('moe_shared_d_ff', 0)};
    cfg.use_rope = {'true' if use_rope else 'false'};
    return cfg;
}}

inline const EmbeddedWeights* GetEmbeddedWeights() {{
    return &EMBEDDED_WEIGHTS;
}}

}} // namespace {namespace}

#endif // {header_name}
""")
    
    print(f"\nHeader file created: {output_header_path}")
    print(f"Total size: {os.path.getsize(output_header_path) / 1024:.2f} KB")
    
    # Also save config as JSON for reference
    config_path = output_header_path.replace('.h', '_config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"Config saved: {config_path}")


def write_array_data(f, data, items_per_line=16):
    """Write array data with proper formatting."""
    for i in range(0, len(data), items_per_line):
        line = data[i:i+items_per_line]
        f.write("    " + ", ".join(f"{int(x):4d}" for x in line))
        if i + items_per_line < len(data):
            f.write(",\n")
        else:
            f.write("\n")


def main():
    parser = argparse.ArgumentParser(description='Export PyTorch model to C++ header file')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to PyTorch checkpoint')
    parser.add_argument('--output', type=str, default='model_weights.h', help='Output header file path')
    parser.add_argument('--namespace', type=str, default='nanollm', help='C++ namespace')
    
    args = parser.parse_args()
    
    export_weights_to_header(args.checkpoint, args.output, args.namespace)


if __name__ == '__main__':
    main()

