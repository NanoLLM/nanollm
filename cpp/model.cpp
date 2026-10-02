#include "model.h"
#include "rope_ops.h"
#include <fstream>
#include <cmath>
#include <cstring>
#include <algorithm>
#include <random>
#include <limits>
#include <chrono>

// For JSON parsing (simple implementation)
#include <sstream>
#include <iostream>

namespace {

int linear_out_dim(size_t weight_size, int in_dim) {
    if (in_dim <= 0) {
        return 0;
    }
    return static_cast<int>(weight_size / static_cast<size_t>(in_dim));
}

} // namespace

// Simple JSON parser for config
void parseConfig(const std::string& json_str, ModelConfig& config) {
    // Simple extraction - assumes JSON format
    std::istringstream iss(json_str);
    std::string line;
    
    while (std::getline(iss, line)) {
        if (line.find("\"vocab_size\"") != std::string::npos) {
            size_t pos = line.find(':');
            if (pos != std::string::npos) {
                config.vocab_size = std::stoi(line.substr(pos + 1));
            }
        } else if (line.find("\"d_model\"") != std::string::npos) {
            size_t pos = line.find(':');
            if (pos != std::string::npos) {
                config.d_model = std::stoi(line.substr(pos + 1));
            }
        } else if (line.find("\"n_layers\"") != std::string::npos) {
            size_t pos = line.find(':');
            if (pos != std::string::npos) {
                config.n_layers = std::stoi(line.substr(pos + 1));
            }
        } else if (line.find("\"n_heads\"") != std::string::npos) {
            size_t pos = line.find(':');
            if (pos != std::string::npos) {
                config.n_heads = std::stoi(line.substr(pos + 1));
            }
        } else if (line.find("\"n_kv_heads\"") != std::string::npos) {
            size_t pos = line.find(':');
            if (pos != std::string::npos) {
                config.n_kv_heads = std::stoi(line.substr(pos + 1));
            }
        } else if (line.find("\"d_ff\"") != std::string::npos) {
            size_t pos = line.find(':');
            if (pos != std::string::npos) {
                config.d_ff = std::stoi(line.substr(pos + 1));
            }
        } else if (line.find("\"max_seq_len\"") != std::string::npos) {
            size_t pos = line.find(':');
            if (pos != std::string::npos) {
                config.max_seq_len = std::stoi(line.substr(pos + 1));
            }
        } else if (line.find("\"quantized\"") != std::string::npos) {
            size_t pos = line.find(':');
            if (pos != std::string::npos) {
                std::string val = line.substr(pos + 1);
                config.quantized = (val.find("true") != std::string::npos);
            }
        } else if (line.find("\"use_moe\"") != std::string::npos) {
            size_t pos = line.find(':');
            if (pos != std::string::npos) {
                std::string val = line.substr(pos + 1);
                config.use_moe = (val.find("true") != std::string::npos);
            }
        } else if (line.find("\"moe_n_experts\"") != std::string::npos) {
            size_t pos = line.find(':');
            if (pos != std::string::npos) {
                config.moe_n_experts = std::stoi(line.substr(pos + 1));
            }
        } else if (line.find("\"moe_top_k\"") != std::string::npos) {
            size_t pos = line.find(':');
            if (pos != std::string::npos) {
                config.moe_top_k = std::stoi(line.substr(pos + 1));
            }
        } else if (line.find("\"moe_shared_d_ff\"") != std::string::npos) {
            size_t pos = line.find(':');
            if (pos != std::string::npos) {
                config.moe_shared_d_ff = std::stoi(line.substr(pos + 1));
            }
        } else if (line.find("\"use_rope\"") != std::string::npos) {
            size_t pos = line.find(':');
            if (pos != std::string::npos) {
                std::string val = line.substr(pos + 1);
                config.use_rope = (val.find("true") != std::string::npos);
            }
        }
    }
}

NanoLLM::NanoLLM() {
    config = {0};
}

NanoLLM::~NanoLLM() {
    // Cleanup handled by vectors
}

bool NanoLLM::load(const std::string& weights_path, const std::string& config_path) {
    // Load config
    std::ifstream config_file(config_path);
    if (!config_file.is_open()) {
        std::cerr << "Failed to open config file: " << config_path << std::endl;
        return false;
    }
    
    std::string config_json((std::istreambuf_iterator<char>(config_file)),
                            std::istreambuf_iterator<char>());
    parseConfig(config_json, config);
    config_file.close();
    
    // Reset previously loaded weights/scales
    weight_scales.clear();
    bias_scales.clear();
    blocks.clear();

    if (config.moe_top_k <= 0) {
        config.moe_top_k = 1;
    }
    if (config.n_kv_heads <= 0) {
        config.n_kv_heads = config.n_heads;
    }
    if (config.n_heads <= 0 || config.d_model % config.n_heads != 0 ||
        config.n_kv_heads <= 0 || config.n_heads % config.n_kv_heads != 0) {
        std::cerr << "Invalid n_heads/n_kv_heads configuration" << std::endl;
        return false;
    }

    // Load weights
    std::ifstream weights_file(weights_path, std::ios::binary);
    if (!weights_file.is_open()) {
        std::cerr << "Failed to open weights file: " << weights_path << std::endl;
        return false;
    }
    
    // Read header (supports dense legacy and experimental MoE format)
    bool is_moe_format = false;
    bool quantized = false;
    uint32_t n_layers = 0;

    char magic[4] = {0};
    weights_file.read(magic, 4);
    if (weights_file.gcount() != 4) {
        std::cerr << "Failed to read model header" << std::endl;
        return false;
    }

    const bool is_dense_versioned = std::memcmp(magic, "NLMD", 4) == 0;
    if (std::memcmp(magic, "NLMO", 4) == 0 || is_dense_versioned) {
        is_moe_format = true;
        uint32_t version = 0;
        uint8_t quantized_flag = 0;
        weights_file.read(reinterpret_cast<char*>(&version), sizeof(uint32_t));
        weights_file.read(reinterpret_cast<char*>(&quantized_flag), sizeof(uint8_t));
        weights_file.ignore(3); // struct.pack native alignment for I?I
        weights_file.read(reinterpret_cast<char*>(&n_layers), sizeof(uint32_t));
        if (version == 2) {
            uint32_t n_kv_heads = 0;
            weights_file.read(reinterpret_cast<char*>(&n_kv_heads), sizeof(uint32_t));
            config.n_kv_heads = static_cast<int>(n_kv_heads);
        } else if (version != 1 || is_dense_versioned) {
            std::cerr << "Unsupported model format version: " << version << std::endl;
            return false;
        }
        is_moe_format = !is_dense_versioned;
        quantized = (quantized_flag != 0);
    } else {
        // Legacy dense format starts with bool+padding+uint32. Rewind and parse old header.
        weights_file.seekg(0, std::ios::beg);
        uint8_t quantized_flag = 0;
        weights_file.read(reinterpret_cast<char*>(&quantized_flag), sizeof(uint8_t));
        weights_file.ignore(3);
        weights_file.read(reinterpret_cast<char*>(&n_layers), sizeof(uint32_t));
        quantized = (quantized_flag != 0);
    }
    
    if (quantized != config.quantized) {
        std::cerr << "Quantization mismatch!" << std::endl;
        return false;
    }
    
    // Helper to read quantized layer
    auto read_quantized_layer = [&](std::vector<qint8_t>& weights, float& scale) {
        weights_file.read(reinterpret_cast<char*>(&scale), sizeof(float));
        uint32_t rows, cols;
        weights_file.read(reinterpret_cast<char*>(&rows), sizeof(uint32_t));
        weights_file.read(reinterpret_cast<char*>(&cols), sizeof(uint32_t));
        weights.resize(rows * cols);
        weights_file.read(reinterpret_cast<char*>(weights.data()), rows * cols);
    };
    
    auto read_quantized_bias = [&](std::vector<qint8_t>& bias, float& scale) {
        weights_file.read(reinterpret_cast<char*>(&scale), sizeof(float));
        uint32_t size;
        weights_file.read(reinterpret_cast<char*>(&size), sizeof(uint32_t));
        if (size > 0) {
            bias.resize(size);
            weights_file.read(reinterpret_cast<char*>(bias.data()), size);
        } else {
            bias.clear();
        }
    };

    auto skip_bias_stub = [&]() {
        uint32_t size = 0;
        weights_file.read(reinterpret_cast<char*>(&size), sizeof(uint32_t));
        if (size > 0) {
            weights_file.ignore(size);
        }
    };
    
    // Read token embedding
    float scale;
    read_quantized_layer(token_embedding, scale);
    weight_scales.push_back(scale);
    
    if (!config.use_rope) {
        read_quantized_layer(pos_embedding, scale);
        weight_scales.push_back(scale);
    } else {
        pos_embedding.clear();
        weight_scales.push_back(1.0f);
    }
    
    // Read transformer blocks
    blocks.resize(config.n_layers);
    for (int i = 0; i < config.n_layers; i++) {
        read_quantized_layer(blocks[i].attn_q, scale);
        blocks[i].scales.push_back(scale);
        skip_bias_stub();
        read_quantized_layer(blocks[i].attn_k, scale);
        blocks[i].scales.push_back(scale);
        skip_bias_stub();
        read_quantized_layer(blocks[i].attn_v, scale);
        blocks[i].scales.push_back(scale);
        skip_bias_stub();
        read_quantized_layer(blocks[i].attn_o, scale);
        blocks[i].scales.push_back(scale);
        skip_bias_stub();
        
        read_quantized_bias(blocks[i].norm1_weight, scale);
        blocks[i].scales.push_back(scale);
        read_quantized_bias(blocks[i].norm1_bias, scale);
        blocks[i].scales.push_back(scale);
        
        read_quantized_bias(blocks[i].norm2_weight, scale);
        blocks[i].scales.push_back(scale);
        read_quantized_bias(blocks[i].norm2_bias, scale);
        blocks[i].scales.push_back(scale);
        
        if (!is_moe_format) {
            read_quantized_layer(blocks[i].ff1_weight, scale);
            blocks[i].scales.push_back(scale);
            read_quantized_bias(blocks[i].ff1_bias, scale);
            blocks[i].scales.push_back(scale);

            read_quantized_layer(blocks[i].ff2_weight, scale);
            blocks[i].scales.push_back(scale);
            read_quantized_bias(blocks[i].ff2_bias, scale);
            blocks[i].scales.push_back(scale);
        } else {
            blocks[i].use_moe = true;
            uint32_t n_experts = 0;
            uint32_t top_k = 0;
            uint32_t expert_d_ff = 0;
            uint8_t has_shared = 0;
            weights_file.read(reinterpret_cast<char*>(&n_experts), sizeof(uint32_t));
            weights_file.read(reinterpret_cast<char*>(&top_k), sizeof(uint32_t));
            weights_file.read(reinterpret_cast<char*>(&expert_d_ff), sizeof(uint32_t));
            weights_file.read(reinterpret_cast<char*>(&has_shared), sizeof(uint8_t));

            blocks[i].moe_n_experts = static_cast<int>(n_experts);
            blocks[i].moe_top_k = static_cast<int>(top_k);
            blocks[i].moe_expert_d_ff = static_cast<int>(expert_d_ff);
            blocks[i].moe_has_shared = (has_shared != 0);

            // Router
            read_quantized_layer(blocks[i].moe_router_weight, blocks[i].moe_router_weight_scale);
            read_quantized_bias(blocks[i].moe_router_bias, blocks[i].moe_router_bias_scale);

            // Experts
            blocks[i].moe_ff1_weight.resize(blocks[i].moe_n_experts);
            blocks[i].moe_ff1_bias.resize(blocks[i].moe_n_experts);
            blocks[i].moe_ff2_weight.resize(blocks[i].moe_n_experts);
            blocks[i].moe_ff2_bias.resize(blocks[i].moe_n_experts);
            blocks[i].moe_ff1_weight_scales.resize(blocks[i].moe_n_experts, 1.0f);
            blocks[i].moe_ff1_bias_scales.resize(blocks[i].moe_n_experts, 1.0f);
            blocks[i].moe_ff2_weight_scales.resize(blocks[i].moe_n_experts, 1.0f);
            blocks[i].moe_ff2_bias_scales.resize(blocks[i].moe_n_experts, 1.0f);

            for (int expert = 0; expert < blocks[i].moe_n_experts; ++expert) {
                read_quantized_layer(blocks[i].moe_ff1_weight[expert], blocks[i].moe_ff1_weight_scales[expert]);
                read_quantized_bias(blocks[i].moe_ff1_bias[expert], blocks[i].moe_ff1_bias_scales[expert]);
                read_quantized_layer(blocks[i].moe_ff2_weight[expert], blocks[i].moe_ff2_weight_scales[expert]);
                read_quantized_bias(blocks[i].moe_ff2_bias[expert], blocks[i].moe_ff2_bias_scales[expert]);
            }

            if (blocks[i].moe_has_shared) {
                read_quantized_layer(blocks[i].moe_shared_ff1_weight, blocks[i].moe_shared_ff1_weight_scale);
                read_quantized_bias(blocks[i].moe_shared_ff1_bias, blocks[i].moe_shared_ff1_bias_scale);
                read_quantized_layer(blocks[i].moe_shared_ff2_weight, blocks[i].moe_shared_ff2_weight_scale);
                read_quantized_bias(blocks[i].moe_shared_ff2_bias, blocks[i].moe_shared_ff2_bias_scale);
            }
        }
    }
    
    // Read final norm
    read_quantized_bias(norm_weight, scale);
    weight_scales.push_back(scale);
    read_quantized_bias(norm_bias, scale);
    weight_scales.push_back(scale);
    
    // Read lm_head
    read_quantized_layer(lm_head, scale);
    weight_scales.push_back(scale);
    skip_bias_stub();
    if (lm_head.empty()) {
        lm_head = token_embedding;
        if (!weight_scales.empty()) {
            weight_scales.back() = weight_scales.front();
        }
    }
    
    weights_file.close();
    
    // Allocate buffers
    allocateBuffers();
    
    return true;
}

void NanoLLM::allocateBuffers() {
    // Allocate working memory for inference
    // Max sequence length * d_model
    hidden_state.resize(config.max_seq_len * config.d_model);
    temp_buffer1.resize(config.max_seq_len * config.d_model);
    temp_buffer2.resize(config.max_seq_len * config.d_model);
    const int kv_dim = (config.d_model / config.n_heads) * config.n_kv_heads;
    const size_t kv_elems = static_cast<size_t>(config.n_layers) * config.max_seq_len * kv_dim;
    const size_t scale_elems = static_cast<size_t>(config.n_layers) * config.max_seq_len;
    kv_key_cache.resize(kv_elems);
    kv_value_cache.resize(kv_elems);
    kv_key_scales.resize(scale_elems, 1.0f);
    kv_value_scales.resize(scale_elems, 1.0f);
    cache_len = 0;
    rope_inv_freq.clear();
    if (config.use_rope && config.n_heads > 0) {
        const int d_k = config.d_model / config.n_heads;
        rope_inv_freq.resize(d_k / 2);
        nanollm_rope::fill_inv_freq(d_k, rope_inv_freq.data());
    }
}

void NanoLLM::dequantize(const qint8_t* quantized, float* output, size_t size, float scale) {
    for (size_t i = 0; i < size; i++) {
        output[i] = static_cast<float>(quantized[i]) / scale;
    }
}

void NanoLLM::linear(const qint8_t* weight, const float* input, float* output,
                     int in_dim, int out_dim, float scale,
                     const qint8_t* bias, float bias_scale) {
    // Matrix multiplication: output = input @ weight^T
    for (int i = 0; i < out_dim; i++) {
        float sum = 0.0f;
        for (int j = 0; j < in_dim; j++) {
            sum += input[j] * (static_cast<float>(weight[i * in_dim + j]) / scale);
        }
        output[i] = sum;
        if (bias) {
            output[i] += static_cast<float>(bias[i]) / bias_scale;
        }
    }
}

void NanoLLM::layer_norm(const float* input, float* output, int size,
                         const qint8_t* weight, float weight_scale,
                         const qint8_t* bias, float bias_scale) {
    // Compute mean
    float mean = 0.0f;
    for (int i = 0; i < size; i++) {
        mean += input[i];
    }
    mean /= size;
    
    // Compute variance
    float variance = 0.0f;
    for (int i = 0; i < size; i++) {
        float diff = input[i] - mean;
        variance += diff * diff;
    }
    variance /= size;
    float std_dev = std::sqrt(variance + 1e-5f);
    float inv_std = 1.0f / std_dev;
    float w_scale = (std::abs(weight_scale) > 1e-9f) ? weight_scale : 1.0f;
    float b_scale = (bias && std::abs(bias_scale) > 1e-9f) ? bias_scale : 1.0f;
    
    // Normalize and scale
    for (int i = 0; i < size; i++) {
        float gamma = weight ? (static_cast<float>(weight[i]) / w_scale) : 1.0f;
        float beta = (bias) ? (static_cast<float>(bias[i]) / b_scale) : 0.0f;
        output[i] = ((input[i] - mean) * inv_std) * gamma + beta;
    }
}

void NanoLLM::gelu(float* x, int size) {
    // Match PyTorch nn.GELU() default (exact): 0.5 * x * (1 + erf(x / sqrt(2))).
    constexpr float inv_sqrt_2 = 0.7071067811865475f;

    for (int i = 0; i < size; i++) {
        float x_val = x[i];
        x[i] = 0.5f * x_val * (1.0f + std::erf(x_val * inv_sqrt_2));
    }
}

void NanoLLM::softmax(float* x, int size) {
    // Find max for numerical stability
    float max_val = x[0];
    for (int i = 1; i < size; i++) {
        if (x[i] > max_val) max_val = x[i];
    }
    
    // Compute exp and sum
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        x[i] = std::exp(x[i] - max_val);
        sum += x[i];
    }
    
    // Normalize
    for (int i = 0; i < size; i++) {
        x[i] /= sum;
    }
}

void NanoLLM::attention(const float* x, float* output, int block_idx, int seq_len) {
    if (config.n_heads <= 0) {
        std::cerr << "[WARN] Invalid head count (" << config.n_heads << "); forcing to 1" << std::endl;
        config.n_heads = 1;
    } else if (config.d_model % config.n_heads != 0) {
        std::cerr << "[WARN] d_model=" << config.d_model
                  << " not divisible by n_heads=" << config.n_heads
                  << "; forcing head count to 1" << std::endl;
        config.n_heads = 1;
    }

    int d_k = config.d_model / config.n_heads;
    const int n_kv_heads = config.n_kv_heads > 0 ? config.n_kv_heads : config.n_heads;
    const int kv_dim = n_kv_heads * d_k;
    const int kv_repeats = config.n_heads / n_kv_heads;
    
    // Ensure we don't exceed buffer sizes
    if (seq_len > config.max_seq_len) seq_len = config.max_seq_len;
    if (seq_len <= 0) seq_len = 1;
    
    const int n_heads = config.n_heads;
    const float inv_sqrt_dk = 1.0f / std::sqrt(static_cast<float>(d_k));

    std::vector<float> Q(seq_len * config.d_model);
    std::vector<float> K(seq_len * kv_dim);
    std::vector<float> V(seq_len * kv_dim);

    for (int token = 0; token < seq_len; ++token) {
        const float* token_input = &x[token * config.d_model];
        linear(blocks[block_idx].attn_q.data(), token_input,
               &Q[token * config.d_model], config.d_model, config.d_model,
               blocks[block_idx].scales[0]);
        linear(blocks[block_idx].attn_k.data(), token_input,
               &K[token * kv_dim], config.d_model, kv_dim,
               blocks[block_idx].scales[1]);
        if (config.use_rope && !rope_inv_freq.empty()) {
            nanollm_rope::apply_heads(&K[token * kv_dim], n_kv_heads, d_k, token, rope_inv_freq.data());
        }
        linear(blocks[block_idx].attn_v.data(), token_input,
               &V[token * kv_dim], config.d_model, kv_dim,
               blocks[block_idx].scales[2]);
    }

    std::vector<float> attn_values(seq_len * config.d_model, 0.0f);
    std::vector<float> row_scores(seq_len);

    for (int head = 0; head < n_heads; ++head) {
        const int head_offset = head * d_k;
        const int kv_head_offset = (head / kv_repeats) * d_k;
        for (int i = 0; i < seq_len; ++i) {
            const int q_base = i * config.d_model + head_offset;
            if (config.use_rope && !rope_inv_freq.empty()) {
                nanollm_rope::apply_head(&Q[q_base], d_k, i, rope_inv_freq.data());
            }
            for (int j = 0; j < seq_len; ++j) {
                if (j > i) {
                    row_scores[j] = -std::numeric_limits<float>::infinity();
                    continue;
                }
                const int k_base = j * kv_dim + kv_head_offset;
                float score = 0.0f;
                for (int dim = 0; dim < d_k; ++dim) {
                    score += Q[q_base + dim] * K[k_base + dim];
                }
                row_scores[j] = score * inv_sqrt_dk;
            }
            softmax(row_scores.data(), seq_len);

            const int out_base = i * config.d_model + head_offset;
            for (int dim = 0; dim < d_k; ++dim) {
                float weighted = 0.0f;
                for (int t = 0; t < seq_len; ++t) {
                    weighted += row_scores[t] * V[t * kv_dim + kv_head_offset + dim];
                }
                attn_values[out_base + dim] = weighted;
            }
        }
    }
    
    // Output projection per token
    for (int token = 0; token < seq_len; ++token) {
     linear(blocks[block_idx].attn_o.data(),
         &attn_values[token * config.d_model],
         &output[token * config.d_model],
         config.d_model, config.d_model, blocks[block_idx].scales[3]);
    }
}

void NanoLLM::feed_forward(const float* x, float* output, int block_idx, int seq_len) {
    if (!blocks[block_idx].use_moe) {
        std::vector<float> ff_hidden(config.d_ff);
        const qint8_t* ff1_bias_ptr = blocks[block_idx].ff1_bias.empty() ? nullptr : blocks[block_idx].ff1_bias.data();
        const qint8_t* ff2_bias_ptr = blocks[block_idx].ff2_bias.empty() ? nullptr : blocks[block_idx].ff2_bias.data();
        float ff1_bias_scale = blocks[block_idx].scales[9];
        float ff2_bias_scale = blocks[block_idx].scales[11];

        for (int token = 0; token < seq_len; ++token) {
            const float* token_input = &x[token * config.d_model];
            float* token_output = &output[token * config.d_model];

            linear(blocks[block_idx].ff1_weight.data(), token_input, ff_hidden.data(),
                config.d_model, config.d_ff, blocks[block_idx].scales[8],
                ff1_bias_ptr, ff1_bias_scale);
            gelu(ff_hidden.data(), config.d_ff);
            linear(blocks[block_idx].ff2_weight.data(), ff_hidden.data(), token_output,
                config.d_ff, config.d_model, blocks[block_idx].scales[10],
                ff2_bias_ptr, ff2_bias_scale);
        }
        return;
    }

    const BlockWeights& b = blocks[block_idx];
    const int n_experts = std::max(1, b.moe_n_experts);
    const int top_k = std::min(std::max(1, b.moe_top_k), n_experts);
    const int expert_d_ff = std::max(1, b.moe_expert_d_ff);

    std::vector<float> router_logits(n_experts);
    std::vector<int> topk_indices(top_k, 0);
    std::vector<float> topk_logits(top_k, -std::numeric_limits<float>::infinity());
    std::vector<float> topk_gates(top_k, 0.0f);
    std::vector<float> expert_hidden(expert_d_ff);
    std::vector<float> expert_output(config.d_model);
    const int shared_d_ff = std::max(
        1,
        linear_out_dim(b.moe_shared_ff1_weight.size(), config.d_model));
    std::vector<float> shared_hidden(shared_d_ff);

    const qint8_t* router_bias_ptr = b.moe_router_bias.empty() ? nullptr : b.moe_router_bias.data();

    for (int token = 0; token < seq_len; ++token) {
        const float* token_input = &x[token * config.d_model];
        float* token_output = &output[token * config.d_model];
        std::fill(token_output, token_output + config.d_model, 0.0f);

        // Router: d_model -> n_experts
        linear(b.moe_router_weight.data(), token_input, router_logits.data(),
               config.d_model, n_experts, b.moe_router_weight_scale,
               router_bias_ptr, b.moe_router_bias_scale);

        // Select top-k experts.
        std::vector<int> order(n_experts);
        for (int i = 0; i < n_experts; ++i) {
            order[i] = i;
        }
        std::partial_sort(order.begin(), order.begin() + top_k, order.end(),
            [&](int a, int c) {
                if (router_logits[a] == router_logits[c]) {
                    return a < c;
                }
                return router_logits[a] > router_logits[c];
            });

        for (int k = 0; k < top_k; ++k) {
            topk_indices[k] = order[k];
            topk_logits[k] = router_logits[order[k]];
        }

        if (top_k == 1) {
            float denom = 0.0f;
            for (int i = 0; i < n_experts; ++i) {
                denom += std::exp(router_logits[i] - topk_logits[0]);
            }
            topk_gates[0] = denom > 0.0f ? 1.0f / denom : 1.0f;
        } else {
            // Softmax over top-k gates.
            float max_logit = topk_logits[0];
            for (int k = 1; k < top_k; ++k) {
                if (topk_logits[k] > max_logit) {
                    max_logit = topk_logits[k];
                }
            }
            float denom = 0.0f;
            for (int k = 0; k < top_k; ++k) {
                topk_gates[k] = std::exp(topk_logits[k] - max_logit);
                denom += topk_gates[k];
            }
            if (denom > 0.0f) {
                for (int k = 0; k < top_k; ++k) {
                    topk_gates[k] /= denom;
                }
            }
        }

        // Routed experts.
        for (int k = 0; k < top_k; ++k) {
            int expert = topk_indices[k];
            float gate = topk_gates[k];

            const qint8_t* ff1_bias_ptr = b.moe_ff1_bias[expert].empty() ? nullptr : b.moe_ff1_bias[expert].data();
            const qint8_t* ff2_bias_ptr = b.moe_ff2_bias[expert].empty() ? nullptr : b.moe_ff2_bias[expert].data();

            linear(b.moe_ff1_weight[expert].data(), token_input, expert_hidden.data(),
                   config.d_model, expert_d_ff, b.moe_ff1_weight_scales[expert],
                   ff1_bias_ptr, b.moe_ff1_bias_scales[expert]);
            gelu(expert_hidden.data(), expert_d_ff);
            linear(b.moe_ff2_weight[expert].data(), expert_hidden.data(), expert_output.data(),
                   expert_d_ff, config.d_model, b.moe_ff2_weight_scales[expert],
                   ff2_bias_ptr, b.moe_ff2_bias_scales[expert]);

            for (int j = 0; j < config.d_model; ++j) {
                token_output[j] += gate * expert_output[j];
            }
        }

        // Optional shared expert.
        if (b.moe_has_shared && !b.moe_shared_ff1_weight.empty() && !b.moe_shared_ff2_weight.empty()) {
            const qint8_t* shared_ff1_bias_ptr = b.moe_shared_ff1_bias.empty() ? nullptr : b.moe_shared_ff1_bias.data();
            const qint8_t* shared_ff2_bias_ptr = b.moe_shared_ff2_bias.empty() ? nullptr : b.moe_shared_ff2_bias.data();

            linear(b.moe_shared_ff1_weight.data(), token_input, shared_hidden.data(),
                   config.d_model, shared_d_ff, b.moe_shared_ff1_weight_scale,
                   shared_ff1_bias_ptr, b.moe_shared_ff1_bias_scale);
            gelu(shared_hidden.data(), shared_d_ff);
            linear(b.moe_shared_ff2_weight.data(), shared_hidden.data(), expert_output.data(),
                   shared_d_ff, config.d_model, b.moe_shared_ff2_weight_scale,
                   shared_ff2_bias_ptr, b.moe_shared_ff2_bias_scale);

            for (int j = 0; j < config.d_model; ++j) {
                token_output[j] += expert_output[j];
            }
        }
    }
}

std::vector<int> NanoLLM::generate(const std::vector<int>& prompt, int max_new_tokens, float temperature) {
    // Timing: the outermost call is the prompt prefill; each recursive call
    // (max_new_tokens==1) is one decode step that re-runs a full forward over
    // the growing history in this desktop runtime.
    const auto _gen_t0 = std::chrono::steady_clock::now();
    const bool _gen_outer = (gen_depth_ == 0);
    if (_gen_outer) {
        prefill_ms = 0.0;
        decode_ms = 0.0;
        tokens_decoded = 0;
    }
    ++gen_depth_;
    int seq_len = std::min(static_cast<int>(prompt.size()), config.max_seq_len);
    if (seq_len <= 0) seq_len = 1;
    if (seq_len > config.max_seq_len) seq_len = config.max_seq_len;
    const size_t context_start = prompt.size() > static_cast<size_t>(seq_len)
        ? prompt.size() - static_cast<size_t>(seq_len)
        : 0;
    
    // Ensure buffers are allocated
    if (hidden_state.size() < static_cast<size_t>(seq_len * config.d_model)) {
        allocateBuffers();
    }
    
    // Initialize hidden state with embeddings
    for (int i = 0; i < seq_len; i++) {
        if (i >= static_cast<int>(prompt.size())) break;
        int token = prompt[context_start + static_cast<size_t>(i)];
        if (token < 0 || token >= config.vocab_size) token = 0;
        int pos = i;
        
        // Token embedding
        if (weight_scales.size() > 0 && token_embedding.size() > static_cast<size_t>(token * config.d_model + config.d_model - 1)) {
            float tok_scale = weight_scales[0];
            for (int j = 0; j < config.d_model; j++) {
                hidden_state[i * config.d_model + j] = 
                    static_cast<float>(token_embedding[token * config.d_model + j]) / tok_scale;
            }
        }
        
        // Position embedding (learned absolute positions; skipped when RoPE is enabled).
        if (!config.use_rope && weight_scales.size() > 1 &&
            pos_embedding.size() > static_cast<size_t>(pos * config.d_model + config.d_model - 1)) {
            float pos_scale = weight_scales[1];
            for (int j = 0; j < config.d_model; j++) {
                hidden_state[i * config.d_model + j] += 
                    static_cast<float>(pos_embedding[pos * config.d_model + j]) / pos_scale;
            }
        }
    }
    
    // Transformer blocks
    for (int block_idx = 0; block_idx < config.n_layers; block_idx++) {
        // Layer norm 1
        for (int i = 0; i < seq_len; i++) {
            layer_norm(&hidden_state[i * config.d_model], &temp_buffer1[i * config.d_model],
                      config.d_model, blocks[block_idx].norm1_weight.data(),
                      blocks[block_idx].scales[4],
                      blocks[block_idx].norm1_bias.data(), blocks[block_idx].scales[5]);
        }
        
        // Attention
    attention(temp_buffer1.data(), temp_buffer2.data(), block_idx, seq_len);
        
        // Residual
        for (int i = 0; i < seq_len * config.d_model; i++) {
            hidden_state[i] += temp_buffer2[i];
        }
        
        // Layer norm 2
        for (int i = 0; i < seq_len; i++) {
            layer_norm(&hidden_state[i * config.d_model], &temp_buffer1[i * config.d_model],
                      config.d_model, blocks[block_idx].norm2_weight.data(),
                      blocks[block_idx].scales[6],
                      blocks[block_idx].norm2_bias.data(), blocks[block_idx].scales[7]);
        }
        
        // Feed forward
    feed_forward(temp_buffer1.data(), temp_buffer2.data(), block_idx, seq_len);
        
        // Residual
        for (int i = 0; i < seq_len * config.d_model; i++) {
            hidden_state[i] += temp_buffer2[i];
        }
    }
    
    // Final norm
    // Scale indices: 0=token_emb, 1=pos_emb, then per block: attn(4) + norm1(2) + norm2(2) + ff(4) = 12
    // Final: norm_weight, norm_bias, lm_head = 3 more
    const size_t norm_weight_idx = 2;  // token + pos scales occupy the first two entries
    float norm_scale = 1.0f;
    if (weight_scales.size() > norm_weight_idx && std::abs(weight_scales[norm_weight_idx]) > 1e-9f) {
        norm_scale = weight_scales[norm_weight_idx];
    }
    float norm_bias_scale = 1.0f;
    if (weight_scales.size() > norm_weight_idx + 1 && std::abs(weight_scales[norm_weight_idx + 1]) > 1e-9f) {
        norm_bias_scale = weight_scales[norm_weight_idx + 1];
    }

    for (int i = 0; i < seq_len; i++) {
    layer_norm(&hidden_state[i * config.d_model], &temp_buffer1[i * config.d_model],
        config.d_model, norm_weight.data(), norm_scale,
        norm_bias.data(), norm_bias_scale);
    }
    
    // LM head
    std::vector<float> logits(config.vocab_size);
    float lm_head_scale = 1.0f;
    if (!weight_scales.empty() && std::abs(weight_scales.back()) > 1e-9f) {
        lm_head_scale = weight_scales.back();
    }
    linear(lm_head.data(), &temp_buffer1[(seq_len - 1) * config.d_model], logits.data(),
           config.d_model, config.vocab_size, lm_head_scale);

    // Record the forward-pass wall time (embeddings -> all blocks -> LM head).
    {
        const double _ms =
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - _gen_t0).count();
        if (gen_depth_ == 1) {
            prefill_ms += _ms;          // outermost call: full prompt forward
        } else {
            decode_ms += _ms;           // recursive single-token forward
            tokens_decoded += 1;
        }
    }

    // Sample
    std::vector<int> generated = prompt;
    
    for (int t = 0; t < max_new_tokens; t++) {
        if (t > 0) {
            std::vector<int> step = generate(generated, 1, temperature);
            if (step.size() <= generated.size()) {
                break;
            }
            int next_token = step.back();
            generated.push_back(next_token);
            if (next_token == 3) {
                break;
            }
            continue;
        }

        // Apply temperature
        if (temperature > 0.0f) {
            for (int i = 0; i < config.vocab_size; i++) {
                logits[i] /= temperature;
            }
        }
        
        softmax(logits.data(), config.vocab_size);
        
        // Sample (use argmax for deterministic output in testing)
        int next_token = 0;
        float max_prob = logits[0];
        for (int i = 1; i < config.vocab_size; i++) {
            if (logits[i] > max_prob) {
                max_prob = logits[i];
                next_token = i;
            }
        }
        
        generated.push_back(next_token);
        if (next_token == 3) {  // <EOS>, fixed by the tokenizer artifact contract
            break;
        }
    }
    
    --gen_depth_;
    return generated;
}

void NanoLLM::resetCache() {
    cache_len = 0;
    decode_history.clear();
    std::fill(kv_key_cache.begin(), kv_key_cache.end(), 0);
    std::fill(kv_value_cache.begin(), kv_value_cache.end(), 0);
    std::fill(kv_key_scales.begin(), kv_key_scales.end(), 1.0f);
    std::fill(kv_value_scales.begin(), kv_value_scales.end(), 1.0f);
}

int NanoLLM::decodeStep(int token, float temperature) {
    const int d = config.d_model;
    const int d_k = d / config.n_heads;
    const int kv_dim = d_k * config.n_kv_heads;
    const int repeats = config.n_heads / config.n_kv_heads;
    if (cache_len >= config.max_seq_len) {
        std::vector<int> replay(decode_history.begin() + 1, decode_history.end());
        replay.push_back(token);
        resetCache();
        int last = -1;
        for (int replay_token : replay) {
            last = decodeStep(replay_token, temperature);
        }
        return last;
    }
    if (token < 0 || token >= config.vocab_size) token = 0;
    decode_history.push_back(token);
    std::vector<float> x(d), normed(d), q(d), k(kv_dim), v(kv_dim), values(d), projected(d);
    for (int i = 0; i < d; ++i) {
        x[i] = static_cast<float>(token_embedding[token * d + i]) / weight_scales[0];
        if (!config.use_rope) {
            x[i] += static_cast<float>(pos_embedding[cache_len * d + i]) / weight_scales[1];
        }
    }
    auto quantize_cache = [](const std::vector<float>& src, qint8_t* dst) {
        float max_abs = 0.0f;
        for (float value : src) max_abs = std::max(max_abs, std::abs(value));
        float scale = max_abs > 0.0f ? 127.0f / max_abs : 1.0f;
        for (size_t i = 0; i < src.size(); ++i) {
            dst[i] = static_cast<qint8_t>(std::max(-127.0f, std::min(127.0f, std::round(src[i] * scale))));
        }
        return scale;
    };
    std::vector<float> scores(cache_len + 1);
    for (int layer = 0; layer < config.n_layers; ++layer) {
        BlockWeights& b = blocks[layer];
        layer_norm(x.data(), normed.data(), d, b.norm1_weight.data(), b.scales[4],
                   b.norm1_bias.data(), b.scales[5]);
        linear(b.attn_q.data(), normed.data(), q.data(), d, d, b.scales[0]);
        linear(b.attn_k.data(), normed.data(), k.data(), d, kv_dim, b.scales[1]);
        linear(b.attn_v.data(), normed.data(), v.data(), d, kv_dim, b.scales[2]);
        if (config.use_rope && !rope_inv_freq.empty()) {
            nanollm_rope::apply_heads(q.data(), config.n_heads, d_k, cache_len, rope_inv_freq.data());
            nanollm_rope::apply_heads(k.data(), config.n_kv_heads, d_k, cache_len, rope_inv_freq.data());
        }
        const size_t cache_base = static_cast<size_t>(layer) * config.max_seq_len * kv_dim;
        const size_t scale_base = static_cast<size_t>(layer) * config.max_seq_len;
        qint8_t* key = &kv_key_cache[cache_base + static_cast<size_t>(cache_len) * kv_dim];
        qint8_t* value = &kv_value_cache[cache_base + static_cast<size_t>(cache_len) * kv_dim];
        kv_key_scales[scale_base + cache_len] = quantize_cache(k, key);
        kv_value_scales[scale_base + cache_len] = quantize_cache(v, value);
        std::fill(values.begin(), values.end(), 0.0f);
        for (int head = 0; head < config.n_heads; ++head) {
            int kv_head = head / repeats;
            for (int t = 0; t <= cache_len; ++t) {
                float score = 0.0f;
                size_t base = cache_base + static_cast<size_t>(t) * kv_dim + kv_head * d_k;
                for (int dim = 0; dim < d_k; ++dim) {
                    score += q[head * d_k + dim] *
                        static_cast<float>(kv_key_cache[base + dim]) / kv_key_scales[scale_base + t];
                }
                scores[t] = score / std::sqrt(static_cast<float>(d_k));
            }
            softmax(scores.data(), cache_len + 1);
            for (int dim = 0; dim < d_k; ++dim) {
                for (int t = 0; t <= cache_len; ++t) {
                    size_t idx = cache_base + static_cast<size_t>(t) * kv_dim + kv_head * d_k + dim;
                    values[head * d_k + dim] += scores[t] *
                        static_cast<float>(kv_value_cache[idx]) / kv_value_scales[scale_base + t];
                }
            }
        }
        linear(b.attn_o.data(), values.data(), projected.data(), d, d, b.scales[3]);
        for (int i = 0; i < d; ++i) x[i] += projected[i];
        layer_norm(x.data(), normed.data(), d, b.norm2_weight.data(), b.scales[6],
                   b.norm2_bias.data(), b.scales[7]);
        feed_forward(normed.data(), projected.data(), layer, 1);
        for (int i = 0; i < d; ++i) x[i] += projected[i];
    }
    layer_norm(x.data(), normed.data(), d, norm_weight.data(), weight_scales[2],
               norm_bias.data(), weight_scales[3]);
    std::vector<float> logits(config.vocab_size);
    linear(lm_head.data(), normed.data(), logits.data(), d, config.vocab_size, weight_scales.back());
    if (temperature > 0.0f) {
        for (float& value : logits) value /= temperature;
    }
    ++cache_len;
    return static_cast<int>(std::max_element(logits.begin(), logits.end()) - logits.begin());
}

size_t NanoLLM::getMemoryUsage() const {
    size_t total = 0;
    total += hidden_state.size() * sizeof(float);
    total += temp_buffer1.size() * sizeof(float);
    total += temp_buffer2.size() * sizeof(float);
    total += kv_key_cache.size() * sizeof(qint8_t);
    total += kv_value_cache.size() * sizeof(qint8_t);
    total += kv_key_scales.size() * sizeof(float);
    total += kv_value_scales.size() * sizeof(float);
    return total;
}

