#ifndef NANOLLM_MODEL_WEIGHTS_TYPES_H
#define NANOLLM_MODEL_WEIGHTS_TYPES_H

#include <cstddef>
#include <cstdint>

namespace nanollm {

struct ModelConfig {
    int vocab_size = 0;
    int d_model = 0;
    int n_layers = 0;
    int n_heads = 0;
    int n_kv_heads = 0;
    int d_ff = 0;
    int max_seq_len = 0;
    bool quantized = true;
    bool use_moe = false;
    int moe_n_experts = 0;
    int moe_top_k = 1;
    int moe_shared_d_ff = 0;
    bool use_rope = false;
};

struct QuantizedLayer {
    const int8_t* data;
    float scale;
    size_t size;
    size_t rows;
    size_t cols;
};

struct MoEExpertLayers {
    QuantizedLayer ff1_weight;
    QuantizedLayer ff1_bias;
    QuantizedLayer ff2_weight;
    QuantizedLayer ff2_bias;
};

// Unified block: dense FFN or MoE FFN (weights live in flash / PROGMEM).
struct TransformerBlock {
    QuantizedLayer attn_q;
    QuantizedLayer attn_k;
    QuantizedLayer attn_v;
    QuantizedLayer attn_o;
    QuantizedLayer norm1_weight;
    QuantizedLayer norm1_bias;
    QuantizedLayer norm2_weight;
    QuantizedLayer norm2_bias;

    bool use_moe_block = false;

    // Dense FFN (use_moe_block == false)
    QuantizedLayer ff1_weight;
    QuantizedLayer ff1_bias;
    QuantizedLayer ff2_weight;
    QuantizedLayer ff2_bias;

    // MoE FFN (use_moe_block == true)
    int moe_n_experts = 0;
    int moe_top_k = 1;
    int moe_expert_d_ff = 0;
    bool moe_has_shared = false;
    int moe_shared_d_ff = 0;
    QuantizedLayer moe_router;
    const MoEExpertLayers* moe_experts = nullptr;
    QuantizedLayer shared_ff1_weight;
    QuantizedLayer shared_ff1_bias;
    QuantizedLayer shared_ff2_weight;
    QuantizedLayer shared_ff2_bias;
};

struct EmbeddedWeights {
    bool use_moe = false;
    QuantizedLayer token_embedding;
    QuantizedLayer pos_embedding;
    const TransformerBlock* blocks = nullptr;
    size_t n_blocks = 0;
    QuantizedLayer final_norm_weight;
    QuantizedLayer final_norm_bias;
    QuantizedLayer lm_head;
};

}  // namespace nanollm

#endif  // NANOLLM_MODEL_WEIGHTS_TYPES_H
