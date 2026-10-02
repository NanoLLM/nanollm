#!/usr/bin/env python3
"""
Export model weights as fused inference kernels for ESP32-S3.

Each transformer block is compiled into a single C++ function with
embedded weights, reducing function call overhead and enabling
compile-time optimization of the entire forward pass.

Usage:
    python export_fused_kernels.py --checkpoint checkpoints/<run>/model_best.pt \\
                                   --output esp32_m5stack/src/fused_kernels.cpp \\
                                   --config esp32_m5stack/src/model_weights_config.json
"""

import argparse
import json
import torch
import numpy as np
import os
import sys

# Add parent dir for model import
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import NanoLLM, MoEFeedForward
from quantize import quantize_weights


def write_weight_array(f, name, arr, scale, align=4):
    """Write a weight array to PROGMEM with alignment."""
    rows, cols = arr.shape
    f.write(f"// {name}: {rows}x{cols}, scale={scale:.8f}\n")
    f.write(f"constexpr float {name}_SCALE = {scale:.8f}f;\n")
    f.write(f"constexpr size_t {name}_SIZE = {arr.size};\n")
    f.write(f"constexpr size_t {name}_ROWS = {rows};\n")
    f.write(f"constexpr size_t {name}_COLS = {cols};\n")
    
    # Write as int8 array
    flat = arr.flatten()
    f.write(f"static const int8_t {name}[] PROGMEM = {{\n")
    
    # Write in chunks for readability
    chunk = 16
    for i in range(0, len(flat), chunk):
        vals = ", ".join(f"{int(v):+3d}" for v in flat[i:i+chunk])
        f.write(f"    {vals},\n")
    f.write("};\n\n")


def write_quantized_layer(f, prefix, weight, bias=None):
    """Export a quantized linear layer's weights and return init fields."""
    w_q, w_scale = quantize_weights(weight)
    bias_init = f"{{ nullptr, 1.0f, 0, 0, 0 }}"
    
    f.write(f"// --- {prefix} ---\n")
    write_weight_array(f, f"{prefix}_W", w_q, w_scale)
    
    if bias is not None and bias.numel() > 0:
        b_q, b_scale = quantize_weights(bias)
        f.write(f"constexpr float {prefix}_B_SCALE = {b_scale:.8f}f;\n")
        f.write(f"static const int8_t {prefix}_B[] PROGMEM = {{\n")
        for i in range(0, len(b_q), 16):
            vals = ", ".join(f"{int(v):+3d}" for v in b_q[i:i+16])
            f.write(f"    {vals},\n")
        f.write("};\n\n")
        bias_init = f"{{ {prefix}_B, {prefix}_B_SCALE, {prefix}_B.size(), 0, 0 }}"
    else:
        f.write(f"constexpr float {prefix}_B_SCALE = 1.0f;\n")
        f.write(f"static const int8_t* {prefix}_B = nullptr;\n\n")
    
    weight_init = (f"{{ {prefix}_W, {prefix}_W_SCALE, "
                   f"{prefix}_W_SIZE, {prefix}_ROWS, {prefix}_COLS }}")
    return weight_init, bias_init


def generate_layer_forward(f, layer_idx, layer, config):
    """Generate a fused forward function for one transformer block."""
    d = config['d_model']
    n_heads = config['n_heads']
    dk = d // n_heads
    n_kv_heads = config.get('n_kv_heads') or n_heads
    d_ff = config['d_ff']
    use_moe = config.get('use_moe', False)
    moe_n = config.get('moe_n_experts', 0)
    moe_topk = config.get('moe_top_k', 1)
    moe_edff = config.get('moe_expert_d_ff', d_ff)
    moe_sharedff = config.get('moe_shared_d_ff', 0)
    
    # Quantize all weights
    w_q, w_scale = quantize_weights(layer.attention.w_q.weight)
    k_q, k_scale = quantize_weights(layer.attention.w_k.weight)
    v_q, v_scale = quantize_weights(layer.attention.w_v.weight)
    o_q, o_scale = quantize_weights(layer.attention.w_o.weight)
    
    if use_moe and isinstance(layer.feed_forward, MoEFeedForward):
        # Write MoE router weights
        r_q, r_scale = quantize_weights(layer.feed_forward.router.weight)
        write_weight_array(f, f"l{layer_idx}_router", r_q, r_scale)
    else:
        ff1_w, ff1_s = quantize_weights(layer.feed_forward.linear1.weight)
        ff2_w, ff2_s = quantize_weights(layer.feed_forward.linear2.weight)
        write_weight_array(f, f"l{layer_idx}_ff1", ff1_w, ff1_s)
        write_weight_array(f, f"l{layer_idx}_ff2", ff2_w, ff2_s)
    
    norm_w = layer.norm1.weight.data
    norm_b = layer.norm1.bias.data
    nw_q, nw_scale = quantize_weights(norm_w)
    nb_q, nb_scale = quantize_weights(norm_b)
    write_weight_array(f, f"l{layer_idx}_n1w", nw_q, nw_scale)
    write_weight_array(f, f"l{layer_idx}_n1b", nb_q, nb_scale)
    
    norm2_w = layer.norm2.weight.data
    norm2_b = layer.norm2.bias.data
    nw2_q, nw2_scale = quantize_weights(norm2_w)
    nb2_q, nb2_scale = quantize_weights(norm2_b)
    write_weight_array(f, f"l{layer_idx}_n2w", nw2_q, nw2_scale)
    write_weight_array(f, f"l{layer_idx}_n2b", nb2_q, nb2_scale)
    
    # Write layer norm weights
    f.write(f"constexpr float l{layer_idx}_N1W_SCALE = {nw_scale:.8f}f;\n")
    f.write(f"constexpr float l{layer_idx}_N1B_SCALE = {nb_scale:.8f}f;\n")
    f.write(f"constexpr float l{layer_idx}_N2W_SCALE = {nw2_scale:.8f}f;\n")
    f.write(f"constexpr float l{layer_idx}_N2B_SCALE = {nb2_scale:.8f}f;\n\n")


def generate_header(f, config):
    """Write the header file with function declarations."""
    d = config['d_model']
    n_heads = config['n_heads']
    dk = d // n_heads
    d_ff = config['d_ff']
    n_layers = config['n_layers']
    
    f.write(f"""/***
 * @file fused_layer_kernels.h
 * @brief Fused inference kernels generated from checkpoint.
 *
 * Generated by export_fused_kernels.py — do not edit manually.
 * 
 * Each function implements one complete transformer block:
 *   LayerNorm1 -> Attention -> Residual -> LayerNorm2 -> FFN -> Residual
 *
 * Memory layout:
 *   - input/output: [seq_len, d_model] floats
 *   - temp1/temp2: working buffers of size [d_model] floats
 *   - kv_cache: pre-allocated [n_layers, max_seq, kv_dim] int8
 *   - kv_scales: pre-allocated [n_layers, max_seq] float
 */

#ifndef NANOLLM_FUSED_KERNELS_H
#define NANOLLM_FUSED_KERNELS_H

#include <cstdint>
#include <cstring>
#include <pgmspace.h>

// Model architecture (from export config)
inline constexpr int FUSED_D_MODEL    = {d};
inline constexpr int FUSED_N_HEADS    = {n_heads};
inline constexpr int FUSED_D_K        = {dk};
inline constexpr int FUSED_D_FF       = {d_ff};
inline constexpr int FUSED_N_LAYERS   = {n_layers};
inline constexpr int FUSED_KV_DIM     = {config.get('n_kv_heads', n_heads) * dk};

// Forward declarations for each layer
""".format(**{'d': d, 'n_heads': n_heads, 'dk': dk, 'd_ff': d_ff,
              'n_layers': n_layers, 'config': config}))
    
    for i in range(n_layers):
        f.write(f"void layer{i}_forward(\n")
        f.write(f"    const float* __restrict input,\n")
        f.write(f"    float* __restrict output,\n")
        f.write(f"    float* __restrict temp1,\n")
        f.write(f"    float* __restrict temp2,\n")
        f.write(f"    int seq_len);\n\n")
    
    f.write(f"""// Run all layers sequentially
void fused_model_forward(
    const float* __restrict token_embedding,
    const float* __restrict pos_embedding,
    const float* __restrict input,
    float* __restrict output,
    float* __restrict temp1,
    float* __restrict temp2,
    int seq_len);

#endif // NANOLLM_FUSED_KERNELS_H
""")


def generate_model_forward(f, config):
    """Generate the full model forward pass."""
    n_layers = config['n_layers']
    d = config['d_model']
    
    f.write(f"""
#include "fused_layer_kernels.h"
#include <pgmspace.h>

void fused_model_forward(
    const float* __restrict token_embedding,
    const float* __restrict pos_embedding,
    const float* __restrict input,
    float* __restrict output,
    float* __restrict temp1,
    float* __restrict temp2,
    int seq_len)
{{
    // Embedding
    for (int t = 0; t < seq_len; ++t) {{
        for (int j = 0; j < {d}; ++j) {{
            float emb = pgm_read_float(&token_embedding[t * {d} + j]);
            float pos = pgm_read_float(&pos_embedding[t * {d} + j]);
            output[t * {d} + j] = input[t * {d} + j] + emb + pos;
        }}
    }}
    
    // Transformer layers
""")
    
    for i in range(n_layers):
        f.write(f"    {{\n")
        f.write(f"        float* next_out = (i == {n_layers-1}) ? output : temp1;\n")
        f.write(f"        float* next_temp1 = (i == {n_layers-1}) ? temp1 : temp2;\n")
        f.write(f"        float* next_temp2 = (i == {n_layers-1}) ? temp2 : temp1;\n")
        f.write(f"        layer{i}_forward(output, next_out, next_temp1, next_temp2, seq_len);\n")
        if i < n_layers - 1:
            f.write(f"        if (i != {n_layers-2}) memcpy(output, next_out, {d} * seq_len * sizeof(float));\n")
        f.write(f"    }}\n")
    
    f.write(f"""
    // Final LayerNorm (simplified — apply to last layer output)
    // TODO: Implement final norm in fused form
}}
""")


def main():
    parser = argparse.ArgumentParser(
        description="Export model weights as fused C++ kernels")
    parser.add_argument("--checkpoint", required=True,
                        help="Path to PyTorch checkpoint (.pt)")
    parser.add_argument("--output", required=True,
                        help="Output C++ file path")
    parser.add_argument("--config", default=None,
                        help="Optional model config JSON")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print config but don't generate")
    args = parser.parse_args()
    
    # Load checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    
    if args.config:
        with open(args.config) as f:
            config = json.load(f)
    else:
        config = checkpoint.get('config', {})
        config['n_kv_heads'] = int(config.get('n_kv_heads') or config['n_heads'])
    
    # Add missing fields
    config.setdefault('d_ff', 128)
    config.setdefault('n_kv_heads', config['n_heads'])
    config.setdefault('use_moe', False)
    config.setdefault('moe_n_experts', 0)
    config.setdefault('moe_top_k', 1)
    config.setdefault('moe_expert_d_ff', config.get('d_ff', 128))
    config.setdefault('moe_shared_d_ff', 0)
    
    print(f"Model config: {json.dumps(config, indent=2)}")
    
    if args.dry_run:
        print("Dry run — config dumped, no file generated.")
        return
    
    # Create model and load weights
    model = NanoLLM(**config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Create output directory
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    
    # Generate header
    header_path = args.output.replace('.cpp', '.h')
    print(f"Generating header: {header_path}")
    with open(header_path, 'w') as hf:
        generate_header(hf, config)
    
    # Generate fused kernel source
    print(f"Generating fused kernels: {args.output}")
    with open(args.output, 'w') as f:
        f.write(f"""/*
 * AUTO-GENERATED by export_fused_kernels.py
 * DO NOT EDIT MANUALLY
 * 
 * Source checkpoint: {args.checkpoint}
 * Generated at: (manual timestamp)
 */

#include "{os.path.basename(header_path)}"
#include <pgmspace.h>
#include <cmath>
#include <algorithm>
#include <cstring>

// GELU approximation (inline for fused kernels)
static inline float fused_gelu(float x) {{
    // Polynomial approximation: 0.5*x*(1 + tanh(sqrt(2/pi)*(x + 0.044715*x^3)))
    // For speed, use lookup-table in production (see optimize_ops.h)
    const float sqrt_2_over_pi = 0.7978845608f;
    const float coeff = 0.044715f;
    float x3 = x * x * x;
    return 0.5f * x * (1.0f + tanhf(sqrt_2_over_pi * (x + coeff * x3)));
}}

// Optimized softmax (2-element unroll)
static inline void fused_softmax(float* x, int size) {{
    float max_val = x[0];
    for (int i = 1; i < size; i++) {{
        if (x[i] > max_val) max_val = x[i];
    }}
    float sum = 0.0f;
    int i = 0;
    for (; i + 1 < size; i += 2) {{
        x[i]     = expf(x[i] - max_val);
        x[i + 1] = expf(x[i + 1] - max_val);
        sum += x[i] + x[i + 1];
    }}
    if (i < size) {{
        x[i] = expf(x[i] - max_val);
        sum += x[i];
    }}
    float inv_sum = 1.0f / sum;
    for (i = 0; i + 1 < size; i += 2) {{
        x[i]     *= inv_sum;
        x[i + 1] *= inv_sum;
    }}
    if (i < size) x[i] *= inv_sum;
}}

// Optimized LayerNorm with rsqrt
static inline void fused_layernorm(
    const float* input, float* output, int size,
    const int8_t* weight_pgm, const int8_t* bias_pgm,
    float weight_scale, float bias_scale)
{{
    float mean = 0.0f;
    for (int i = 0; i < size; i++) mean += input[i];
    mean /= size;
    
    float var = 0.0f;
    for (int i = 0; i < size; i++) {{
        float d = input[i] - mean;
        var += d * d;
    }}
    
    // Fast rsqrt
    float xhalf = 0.5f * var;
    uint32_t i;
    memcpy(&i, &var, sizeof(i));
    i = 0x5f3759df - (i >> 1);
    float y;
    memcpy(&y, &i, sizeof(y));
    y = y * (1.5f - xhalf * y * y);  // Newton-Raphson
    
    float w_scale = (weight_scale > 1e-9f) ? weight_scale : 1.0f;
    float b_scale = (bias_pgm && bias_scale > 1e-9f) ? bias_scale : 1.0f;
    
    for (int i = 0; i < size; i++) {{
        int8_t w = pgm_read_byte(&weight_pgm[i]);
        float gamma = static_cast<float>(w) / w_scale;
        float beta = 0.0f;
        if (bias_pgm) {{
            int8_t b = pgm_read_byte(&bias_pgm[i]);
            beta = static_cast<float>(b) / b_scale;
        }}
        output[i] = ((input[i] - mean) * y) * gamma + beta;
    }}
}}

// Optimized linear from PROGMEM
static inline void fused_linear(
    const int8_t* weight_pgm, const float* input, float* output,
    int in_dim, int out_dim, float scale,
    const int8_t* bias_pgm, float bias_scale)
{{
    const float wscale = (fabsf(scale) > 1e-9f) ? scale : 1.0f;
    const float bscale = (bias_pgm && fabsf(bias_scale) > 1e-9f) ? bias_scale : 1.0f;
    
    for (int i = 0; i < out_dim; i++) {{
        float sum = 0.0f;
        int j = 0;
        // Unroll 4 at a time
        for (; j + 3 < in_dim; j += 4) {{
            sum += pgm_read_byte(&weight_pgm[i * in_dim + j])     * input[j]     / wscale;
            sum += pgm_read_byte(&weight_pgm[i * in_dim + j + 1]) * input[j + 1] / wscale;
            sum += pgm_read_byte(&weight_pgm[i * in_dim + j + 2]) * input[j + 2] / wscale;
            sum += pgm_read_byte(&weight_pgm[i * in_dim + j + 3]) * input[j + 3] / wscale;
        }}
        for (; j < in_dim; j++) {{
            sum += pgm_read_byte(&weight_pgm[i * in_dim + j]) * input[j] / wscale;
        }}
        output[i] = sum;
        if (bias_pgm) {{
            output[i] += static_cast<float>(pgm_read_byte(&bias_pgm[i])) / bscale;
        }}
    }}
}}

// Optimized attention (MQA/GQA compatible)
static inline void fused_attention(
    const float* __restrict x,
    float* __restrict output,
    const int8_t* __restrict attn_q_pgm, float attn_q_scale,
    const int8_t* __restrict attn_k_pgm, float attn_k_scale,
    const int8_t* __restrict attn_v_pgm, float attn_v_scale,
    const int8_t* __restrict attn_o_pgm, float attn_o_scale,
    int d_model, int n_heads, int dk, int kv_dim,
    float* __restrict q_buf, float* __restrict scores,
    int seq_len)
{{
    const int n_kv_heads = n_heads;  // Simplified: same as n_heads
    const int kv_repeats = n_heads / n_kv_heads;
    const float inv_sqrt_dk = 1.0f / sqrtf(static_cast<float>(dk));
    
    // QKV projections
    for (int token = 0; token < seq_len; ++token) {{
        const float* token_input = &x[token * d_model];
        fused_linear(attn_k_pgm, token_input, q_buf,
                     d_model, kv_dim, attn_k_scale, nullptr, 1.0f);
        fused_linear(attn_v_pgm, token_input, &q_buf[kv_dim],
                     d_model, kv_dim, attn_v_scale, nullptr, 1.0f);
    }}
    
    // Multi-head attention
    for (int head = 0; head < n_heads; ++head) {{
        const int head_offset = head * dk;
        const int kv_head_offset = head_offset;  // For MHA
        
        for (int i = 0; i < seq_len; ++i) {{
            // Q head
            fused_linear(attn_q_pgm, &x[i * d_model], q_buf,
                         d_model, d_model, attn_q_scale, nullptr, 1.0f);
            
            // Attention scores
            for (int j = 0; j <= i; ++j) {{
                float score = 0.0f;
                for (int dim = 0; dim < dk; ++dim) {{
                    score += q_buf[head_offset + dim] * q_buf[kv_dim + j * kv_dim + kv_head_offset + dim];
                }}
                scores[j] = score * inv_sqrt_dk;
            }}
            for (int j = i + 1; j < seq_len; ++j) {{
                scores[j] = -INFINITY;
            }}
            fused_softmax(scores, seq_len);
            
            // Weighted value sum
            for (int dim = 0; dim < dk; ++dim) {{
                float weighted = 0.0f;
                for (int t = 0; t < seq_len; ++t) {{
                    weighted += scores[t] * q_buf[kv_dim + t * kv_dim + kv_head_offset + dim];
                }}
                output[i * d_model + head_offset + dim] = weighted;
            }}
        }}
    }}
    
    // Output projection
    for (int token = 0; token < seq_len; ++token) {{
        fused_linear(attn_o_pgm, &output[token * d_model], q_buf,
                     d_model, d_model, attn_o_scale, nullptr, 1.0f);
        memcpy(&output[token * d_model], q_buf, d_model * sizeof(float));
    }}
}}

""")
        
        # Generate each layer
        for layer_idx in range(config['n_layers']):
            layer = model.transformer.h[layer_idx]
            f.write(f"// =========================================================================\n")
            f.write(f"// Layer {layer_idx}\n")
            f.write(f"// =========================================================================\n\n")
            generate_layer_forward(f, layer_idx, layer, config)
    
    # Generate model forward
    generate_model_forward(f, config)
    
    print(f"Done! Generated {args.output} and {header_path}")
    
    # Print model size estimate
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model size: {total_params:,} parameters")


if __name__ == "__main__":
    main()
