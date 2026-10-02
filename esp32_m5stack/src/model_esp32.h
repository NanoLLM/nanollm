#ifndef NANOLLM_MODEL_ESP32_H
#define NANOLLM_MODEL_ESP32_H

#include <cstdint>
#include <vector>
#include <FS.h>
#include <Arduino.h>

// Fixed-point arithmetic for ESP32 compatibility
typedef int8_t qint8_t;
typedef int16_t qint16_t;

// Forward declaration for embedded weights structure
#ifdef NANOLLM_USE_EMBEDDED_WEIGHTS
namespace nanollm {
    struct QuantizedLayer;
    struct MoEExpertLayers;
    struct TransformerBlock;
    struct EmbeddedWeights;
}
#endif

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
    typedef void (*TokenCallback)(int token, int step, void* user_data);

    NanoLLM();
    ~NanoLLM();
    
    static constexpr size_t kSramWarningThresholdBytes = 200 * 1024;  // 200KB budget for no-PSRAM builds
    static constexpr int kMaxInferenceSeqLenNoPsram = 256;  // Post-P1.2 streaming + trimmed activations

    // Load model from SPIFFS file
    bool load(const char* weights_path, const char* config_path);
    
    // Load model from embedded header file (firmware partition)
    bool loadFromEmbedded();
    
    // Generate tokens
    std::vector<int> generate(const std::vector<int>& prompt, int max_new_tokens = 50, float temperature = 1.0f,
                              TokenCallback callback = nullptr, void* callback_user_data = nullptr);
    void resetCache();
    int decodeStep(int token, float temperature = 1.0f);
    
    // Get model info
    ModelConfig getConfig() const { return config; }
    size_t getMemoryUsage() const;
    
private:
    ModelConfig config;
    
    // Pointer to embedded weights structure (when using NANOLLM_USE_EMBEDDED_WEIGHTS)
    // This allows zero-copy access to weights stored in PROGMEM
#ifdef NANOLLM_USE_EMBEDDED_WEIGHTS
    const nanollm::EmbeddedWeights* embedded_weights_ptr = nullptr;
    const nanollm::TransformerBlock* getEmbeddedBlocksBase() const;
    nanollm::TransformerBlock loadEmbeddedBlock(int block_idx) const;
    nanollm::MoEExpertLayers loadEmbeddedMoEExpert(const nanollm::TransformerBlock& block, int expert_id) const;
    nanollm::QuantizedLayer loadEmbeddedLayer(const nanollm::QuantizedLayer* layer_ptr) const;
    void feed_forward_embedded_moe(const float* x, float* output, const nanollm::TransformerBlock& block, int seq_len);
#endif
    
    // Quantization scales (used for SPIFFS-loaded weights)
    std::vector<float> weight_scales;
    std::vector<float> bias_scales;
    
    // Model weights (quantized int8) - used for SPIFFS-loaded weights
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
    std::vector<float> attn_q;
    std::vector<float> attn_k;
    std::vector<float> attn_v;
    std::vector<qint8_t> kv_key_cache;
    std::vector<qint8_t> kv_value_cache;
    std::vector<float> kv_key_scales;
    std::vector<float> kv_value_scales;
    std::vector<float> attn_scores;
    std::vector<float> logits_buffer;
    std::vector<float> rope_inv_freq;
    std::vector<float> moe_router_logits;
    std::vector<int> moe_topk_indices;
    std::vector<float> moe_topk_logits;
    std::vector<float> moe_topk_gates;
    std::vector<int> moe_expert_order;
    std::vector<float> moe_expert_hidden;
    std::vector<float> moe_expert_output;
    std::vector<float> moe_shared_hidden;
    
    // Helper functions
    void dequantize(const qint8_t* quantized, float* output, size_t size, float scale);
    void dequantizeFromProgmem(const int8_t* quantized_pgm, float* output, size_t size, float scale);
    void linear(const qint8_t* weight, const float* input, float* output, 
                int in_dim, int out_dim, float scale, const qint8_t* bias = nullptr, float bias_scale = 1.0f);
    void linearFromProgmem(const int8_t* weight_pgm, const float* input, float* output,
                           int in_dim, int out_dim, float scale, const int8_t* bias_pgm = nullptr, float bias_scale = 1.0f);
    void layer_norm(const float* input, float* output, int size, 
                    const qint8_t* weight, const qint8_t* bias,
                    float weight_scale, float bias_scale);
    void layer_normFromProgmem(const float* input, float* output, int size,
                               const int8_t* weight_pgm, const int8_t* bias_pgm,
                               float weight_scale, float bias_scale);
    void attention(const float* x, float* output, int block_idx, int seq_len);
    void feed_forward(const float* x, float* output, int block_idx, int seq_len);
    void gelu(float* x, int size);
    void softmax(float* x, int size);
    
    // Helper to check if using embedded weights
    inline bool isUsingEmbeddedWeights() const {
#ifdef NANOLLM_USE_EMBEDDED_WEIGHTS
        return embedded_weights_ptr != nullptr;
#else
        return false;
#endif
    }
    
    // Memory management
    bool allocateBuffers();
    int inference_seq_len = 0;
    std::vector<int> decode_history;
    
    // ESP32-specific file reading
    bool readConfigFromFile(const char* config_path);
    bool readWeightsFromFile(const char* weights_path);
};

#endif // NANOLLM_MODEL_ESP32_H

