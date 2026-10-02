#ifndef NANOLLM_MODEL_H
#define NANOLLM_MODEL_H

#include <cstdint>
#include <vector>
#include <string>

// Fixed-point arithmetic for ESP32 compatibility
typedef int8_t qint8_t;
typedef int16_t qint16_t;

struct ModelConfig {
    int vocab_size;
    int d_model;
    int n_layers;
    int n_heads;
    int n_kv_heads;
    int d_ff;
    int max_seq_len;
    bool quantized;
    bool use_moe;
    int moe_n_experts;
    int moe_top_k;
    int moe_shared_d_ff;
    bool use_rope;
};

class NanoLLM {
public:
    NanoLLM();
    ~NanoLLM();
    
    // Load model from binary file
    bool load(const std::string& weights_path, const std::string& config_path);
    
    // Generate tokens
    std::vector<int> generate(const std::vector<int>& prompt, int max_new_tokens = 50, float temperature = 1.0f);
    void resetCache();
    int decodeStep(int token, float temperature = 1.0f);
    
    // Get model info
    ModelConfig getConfig() const { return config; }
    size_t getMemoryUsage() const;

    // Inference timing (populated by generate()); ms precision.
    // prefill_ms: full forward over the prompt (first-token path).
    // decode_ms: sum over all per-token decode steps (each re-runs a full
    //   forward over the growing history in the desktop runtime).
    // tokens_decoded: number of per-token steps executed.
    double prefill_ms = 0.0;
    double decode_ms = 0.0;
    int tokens_decoded = 0;

private:
    ModelConfig config;
    
    // Quantization scales
    std::vector<float> weight_scales;
    std::vector<float> bias_scales;
    
    // Model weights (quantized int8)
    std::vector<qint8_t> token_embedding;
    std::vector<qint8_t> pos_embedding;
    
    // Transformer block weights
    struct BlockWeights {
        std::vector<qint8_t> attn_q, attn_k, attn_v, attn_o;
        std::vector<qint8_t> norm1_weight, norm1_bias;
        std::vector<qint8_t> norm2_weight, norm2_bias;
        std::vector<qint8_t> ff1_weight, ff1_bias;
        std::vector<qint8_t> ff2_weight, ff2_bias;
        std::vector<float> scales;  // Per-layer scales

        bool use_moe = false;
        int moe_n_experts = 0;
        int moe_top_k = 0;
        int moe_expert_d_ff = 0;
        bool moe_has_shared = false;

        std::vector<qint8_t> moe_router_weight, moe_router_bias;
        float moe_router_weight_scale = 1.0f;
        float moe_router_bias_scale = 1.0f;

        std::vector<std::vector<qint8_t>> moe_ff1_weight;
        std::vector<std::vector<qint8_t>> moe_ff1_bias;
        std::vector<std::vector<qint8_t>> moe_ff2_weight;
        std::vector<std::vector<qint8_t>> moe_ff2_bias;
        std::vector<float> moe_ff1_weight_scales;
        std::vector<float> moe_ff1_bias_scales;
        std::vector<float> moe_ff2_weight_scales;
        std::vector<float> moe_ff2_bias_scales;

        std::vector<qint8_t> moe_shared_ff1_weight, moe_shared_ff1_bias;
        std::vector<qint8_t> moe_shared_ff2_weight, moe_shared_ff2_bias;
        float moe_shared_ff1_weight_scale = 1.0f;
        float moe_shared_ff1_bias_scale = 1.0f;
        float moe_shared_ff2_weight_scale = 1.0f;
        float moe_shared_ff2_bias_scale = 1.0f;
    };
    std::vector<BlockWeights> blocks;
    
    std::vector<qint8_t> norm_weight, norm_bias;
    std::vector<qint8_t> lm_head;
    
    // Working memory (reused for inference)
    std::vector<float> hidden_state;
    std::vector<float> temp_buffer1;
    std::vector<float> temp_buffer2;
    std::vector<qint8_t> kv_key_cache, kv_value_cache;
    std::vector<float> kv_key_scales, kv_value_scales;
    std::vector<int> decode_history;
    int cache_len = 0;
    int gen_depth_ = 0;
    std::vector<float> rope_inv_freq;
    
    // Helper functions
    void dequantize(const qint8_t* quantized, float* output, size_t size, float scale);
    void linear(const qint8_t* weight, const float* input, float* output, 
                int in_dim, int out_dim, float scale, const qint8_t* bias = nullptr, float bias_scale = 1.0f);
    void layer_norm(const float* input, float* output, int size,
                    const qint8_t* weight, float weight_scale,
                    const qint8_t* bias, float bias_scale);
    void attention(const float* x, float* output, int block_idx, int seq_len);
    void feed_forward(const float* x, float* output, int block_idx, int seq_len);
    void gelu(float* x, int size);
    void softmax(float* x, int size);
    
    // Memory management
    void allocateBuffers();
};

#endif // NANOLLM_MODEL_H

