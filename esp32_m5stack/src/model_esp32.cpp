#include "model_esp32.h"
#include "optimize_ops.h"
#include "rope_ops.h"
#include "system_utils.h"
#include <ArduinoJson.h>
#include <SPIFFS.h>
#include <esp32-hal-psram.h>
#include <esp_heap_caps.h>
#include <cmath>
#include <algorithm>
#include <random>
#include <cstring>
#include <limits>
#include <new>
#include <pgmspace.h>

#ifdef NANOLLM_USE_EMBEDDED_WEIGHTS
#include "model_weights.h"
#endif

NanoLLM::NanoLLM() {
    config = {0};
}

NanoLLM::~NanoLLM() {
    // Cleanup handled by vectors
}

#ifdef NANOLLM_USE_EMBEDDED_WEIGHTS
const nanollm::TransformerBlock* NanoLLM::getEmbeddedBlocksBase() const {
    if (!embedded_weights_ptr) {
        return nullptr;
    }
    return reinterpret_cast<const nanollm::TransformerBlock*>(pgm_read_ptr(&embedded_weights_ptr->blocks));
}

nanollm::TransformerBlock NanoLLM::loadEmbeddedBlock(int block_idx) const {
    nanollm::TransformerBlock block{};
    const nanollm::TransformerBlock* blocks_base = getEmbeddedBlocksBase();
    if (blocks_base) {
        memcpy_P(&block, &blocks_base[block_idx], sizeof(block));
    }
    return block;
}

nanollm::QuantizedLayer NanoLLM::loadEmbeddedLayer(const nanollm::QuantizedLayer* layer_ptr) const {
    nanollm::QuantizedLayer layer{};
    if (layer_ptr) {
        memcpy_P(&layer, layer_ptr, sizeof(layer));
    }
    return layer;
}

nanollm::MoEExpertLayers NanoLLM::loadEmbeddedMoEExpert(const nanollm::TransformerBlock& block, int expert_id) const {
    nanollm::MoEExpertLayers expert{};
    const nanollm::MoEExpertLayers* experts_base =
        reinterpret_cast<const nanollm::MoEExpertLayers*>(pgm_read_ptr(&block.moe_experts));
    if (experts_base && expert_id >= 0) {
        memcpy_P(&expert, &experts_base[expert_id], sizeof(expert));
    }
    return expert;
}

void NanoLLM::feed_forward_embedded_moe(const float* x, float* output,
                                        const nanollm::TransformerBlock& block, int seq_len) {
    const int n_experts = std::max(1, block.moe_n_experts);
    const int top_k = std::min(std::max(1, block.moe_top_k), n_experts);
    const int expert_d_ff = std::max(1, block.moe_expert_d_ff);
    const int shared_d_ff = std::max(1, block.moe_shared_d_ff);

    float* router_logits = moe_router_logits.data();
    int* topk_indices = moe_topk_indices.data();
    float* topk_logits = moe_topk_logits.data();
    float* topk_gates = moe_topk_gates.data();
    int* order = moe_expert_order.data();
    float* expert_hidden = moe_expert_hidden.data();
    float* expert_output = moe_expert_output.data();
    float* shared_hidden = moe_shared_hidden.data();

    for (int token = 0; token < seq_len; ++token) {
        const float* token_input = &x[token * config.d_model];
        float* token_output = &output[token * config.d_model];
        std::fill(token_output, token_output + config.d_model, 0.0f);

        linearFromProgmem(block.moe_router.data, token_input, router_logits,
                          config.d_model, n_experts, block.moe_router.scale,
                          nullptr, 1.0f);

        if (top_k == 1) {
            int best_expert = 0;
            float best_logit = router_logits[0];
            for (int i = 1; i < n_experts; ++i) {
                if (router_logits[i] > best_logit) {
                    best_logit = router_logits[i];
                    best_expert = i;
                }
            }
            float denom = 0.0f;
            for (int i = 0; i < n_experts; ++i) {
                denom += expf(router_logits[i] - best_logit);
            }
            topk_indices[0] = best_expert;
            topk_gates[0] = denom > 0.0f ? 1.0f / denom : 1.0f;
        } else {
            for (int i = 0; i < n_experts; ++i) {
                order[i] = i;
            }
            std::partial_sort(order, order + top_k, order + n_experts,
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

            float max_logit = topk_logits[0];
            for (int k = 1; k < top_k; ++k) {
                if (topk_logits[k] > max_logit) {
                    max_logit = topk_logits[k];
                }
            }
            float denom = 0.0f;
            for (int k = 0; k < top_k; ++k) {
                topk_gates[k] = expf(topk_logits[k] - max_logit);
                denom += topk_gates[k];
            }
            if (denom > 0.0f) {
                for (int k = 0; k < top_k; ++k) {
                    topk_gates[k] /= denom;
                }
            }
        }

        for (int k = 0; k < top_k; ++k) {
            int expert_id = topk_indices[k];
            float gate = topk_gates[k];
            nanollm::MoEExpertLayers expert = loadEmbeddedMoEExpert(block, expert_id);

            linearFromProgmem(expert.ff1_weight.data, token_input, expert_hidden,
                              config.d_model, expert_d_ff, expert.ff1_weight.scale,
                              expert.ff1_bias.data, expert.ff1_bias.scale);
            gelu(expert_hidden, expert_d_ff);
            linearFromProgmem(expert.ff2_weight.data, expert_hidden, expert_output,
                              expert_d_ff, config.d_model, expert.ff2_weight.scale,
                              expert.ff2_bias.data, expert.ff2_bias.scale);

            for (int j = 0; j < config.d_model; ++j) {
                token_output[j] += gate * expert_output[j];
            }
        }

        if (block.moe_has_shared && block.moe_shared_d_ff > 0) {
            linearFromProgmem(block.shared_ff1_weight.data, token_input, shared_hidden,
                              config.d_model, shared_d_ff, block.shared_ff1_weight.scale,
                              block.shared_ff1_bias.data, block.shared_ff1_bias.scale);
            gelu(shared_hidden, shared_d_ff);
            linearFromProgmem(block.shared_ff2_weight.data, shared_hidden, expert_output,
                              shared_d_ff, config.d_model, block.shared_ff2_weight.scale,
                              block.shared_ff2_bias.data, block.shared_ff2_bias.scale);

            for (int j = 0; j < config.d_model; ++j) {
                token_output[j] += expert_output[j];
            }
        }
    }
}
#endif

bool NanoLLM::readConfigFromFile(const char* config_path) {
    File config_file = SPIFFS.open(config_path, "r");
    if (!config_file) {
        Serial.printf("Failed to open config file: %s\n", config_path);
        return false;
    }
    
    // Parse JSON
    DynamicJsonDocument doc(1024);
    DeserializationError error = deserializeJson(doc, config_file);
    config_file.close();
    
    if (error) {
        Serial.printf("JSON parse error: %s\n", error.c_str());
        return false;
    }
    
    config.vocab_size = doc["vocab_size"] | 256;
    config.d_model = doc["d_model"] | 64;
    config.n_layers = doc["n_layers"] | 1;
    config.n_heads = doc["n_heads"] | 2;
    config.n_kv_heads = doc["n_kv_heads"] | config.n_heads;
        if (config.n_heads <= 0 || config.n_kv_heads <= 0 ||
            config.n_heads % config.n_kv_heads != 0) {
            Serial.printf("[ERR] Invalid head count (%d); forcing to 1\n", config.n_heads);
            config.n_heads = 1;
            config.n_kv_heads = 1;
        } else if (config.d_model % config.n_heads != 0) {
            Serial.printf(
                "[ERR] d_model=%d not divisible by n_heads=%d; forcing head count to 1\n",
                config.d_model,
                config.n_heads);
            config.n_heads = 1;
            config.n_kv_heads = 1;
        }
    config.d_ff = doc["d_ff"] | 128;
    config.max_seq_len = doc["max_seq_len"] | NanoLLM::kMaxInferenceSeqLenNoPsram;
    config.quantized = doc["quantized"] | true;
    config.use_moe = doc["use_moe"] | false;
    config.moe_n_experts = doc["moe_n_experts"] | 0;
    config.moe_top_k = doc["moe_top_k"] | 0;
    config.moe_shared_d_ff = doc["moe_shared_d_ff"] | 0;
    config.use_rope = doc["use_rope"] | false;
    
    return true;
}

bool NanoLLM::readWeightsFromFile(const char* weights_path) {
    File weights_file = SPIFFS.open(weights_path, "r");
    if (!weights_file) {
        Serial.printf("Failed to open weights file: %s\n", weights_path);
        return false;
    }

    weight_scales.clear();
    bias_scales.clear();
    blocks.clear();

    bool is_moe_format = false;
    bool quantized = false;
    uint32_t n_layers = 0;

    // Header supports legacy dense (?I) and experimental MoE (NLMO + I?I)
    char header_magic[4] = {0};
    if (weights_file.readBytes(header_magic, 4) != 4) {
        Serial.println("Failed to read model header");
        weights_file.close();
        return false;
    }

    const bool is_dense_versioned = memcmp(header_magic, "NLMD", 4) == 0;
    if (memcmp(header_magic, "NLMO", 4) == 0 || is_dense_versioned) {
        is_moe_format = true;
        uint32_t version = 0;
        uint8_t quantized_flag = 0;
        if (weights_file.readBytes((char*)&version, sizeof(uint32_t)) != sizeof(uint32_t)) {
            weights_file.close();
            return false;
        }
        if (weights_file.readBytes((char*)&quantized_flag, sizeof(uint8_t)) != sizeof(uint8_t)) {
            weights_file.close();
            return false;
        }
        // Native alignment from struct.pack('I?I')
        weights_file.seek(weights_file.position() + 3);
        if (weights_file.readBytes((char*)&n_layers, sizeof(uint32_t)) != sizeof(uint32_t)) {
            weights_file.close();
            return false;
        }
        if (version == 2) {
            uint32_t n_kv_heads = 0;
            if (weights_file.readBytes((char*)&n_kv_heads, sizeof(uint32_t)) != sizeof(uint32_t)) {
                weights_file.close();
                return false;
            }
            config.n_kv_heads = static_cast<int>(n_kv_heads);
        } else if (version != 1 || is_dense_versioned) {
            Serial.printf("Unsupported model format version: %lu\n", static_cast<unsigned long>(version));
            weights_file.close();
            return false;
        }
        is_moe_format = !is_dense_versioned;
        quantized = (quantized_flag != 0);
    } else {
        // Legacy dense format starts at byte 0 with struct.pack('?I')
        weights_file.seek(0);
        uint8_t quantized_flag = 0;
        if (weights_file.readBytes((char*)&quantized_flag, sizeof(uint8_t)) != sizeof(uint8_t)) {
            weights_file.close();
            return false;
        }
        weights_file.seek(weights_file.position() + 3);
        if (weights_file.readBytes((char*)&n_layers, sizeof(uint32_t)) != sizeof(uint32_t)) {
            weights_file.close();
            return false;
        }
        quantized = (quantized_flag != 0);
    }

    if (quantized != config.quantized) {
        Serial.println("Quantization mismatch!");
        weights_file.close();
        return false;
    }
    
    // Helper to read quantized layer
    auto read_quantized_layer = [&](std::vector<qint8_t>& weights, float& scale) -> bool {
        if (weights_file.readBytes((char*)&scale, sizeof(float)) != sizeof(float)) return false;
        uint32_t rows, cols;
        if (weights_file.readBytes((char*)&rows, sizeof(uint32_t)) != sizeof(uint32_t)) return false;
        if (weights_file.readBytes((char*)&cols, sizeof(uint32_t)) != sizeof(uint32_t)) return false;
        weights.resize(rows * cols);
        if (weights_file.readBytes((char*)weights.data(), rows * cols) != rows * cols) return false;
        return true;
    };
    
    auto read_quantized_bias = [&](std::vector<qint8_t>& bias, float& scale) -> bool {
        if (weights_file.readBytes((char*)&scale, sizeof(float)) != sizeof(float)) return false;
        uint32_t size;
        if (weights_file.readBytes((char*)&size, sizeof(uint32_t)) != sizeof(uint32_t)) return false;
        if (size > 0) {
            bias.resize(size);
            if (weights_file.readBytes((char*)bias.data(), size) != size) return false;
        } else {
            bias.clear();
        }
        return true;
    };

    auto skip_bias_stub = [&]() -> bool {
        uint32_t size = 0;
        if (weights_file.readBytes((char*)&size, sizeof(uint32_t)) != sizeof(uint32_t)) return false;
        if (size > 0) {
            weights_file.seek(weights_file.position() + size);
        }
        return true;
    };
    
    // Read token embedding
    float scale;
    if (!read_quantized_layer(token_embedding, scale)) {
        weights_file.close();
        return false;
    }
    weight_scales.push_back(scale);
    
    // Read position embedding (skipped for RoPE models — positions applied in attention).
    if (!config.use_rope) {
        if (!read_quantized_layer(pos_embedding, scale)) {
            weights_file.close();
            return false;
        }
        weight_scales.push_back(scale);
    } else {
        pos_embedding.clear();
        weight_scales.push_back(1.0f);
    }
    
    // Read transformer blocks
    blocks.resize(n_layers);
    for (uint32_t i = 0; i < n_layers; i++) {
        if (!read_quantized_layer(blocks[i].attn_q, scale)) { weights_file.close(); return false; }
        blocks[i].scales.push_back(scale);
        if (!skip_bias_stub()) { weights_file.close(); return false; }
        if (!read_quantized_layer(blocks[i].attn_k, scale)) { weights_file.close(); return false; }
        blocks[i].scales.push_back(scale);
        if (!skip_bias_stub()) { weights_file.close(); return false; }
        if (!read_quantized_layer(blocks[i].attn_v, scale)) { weights_file.close(); return false; }
        blocks[i].scales.push_back(scale);
        if (!skip_bias_stub()) { weights_file.close(); return false; }
        if (!read_quantized_layer(blocks[i].attn_o, scale)) { weights_file.close(); return false; }
        blocks[i].scales.push_back(scale);
        if (!skip_bias_stub()) { weights_file.close(); return false; }
        
        if (!read_quantized_bias(blocks[i].norm1_weight, scale)) { weights_file.close(); return false; }
        blocks[i].scales.push_back(scale);
        if (!read_quantized_bias(blocks[i].norm1_bias, scale)) { weights_file.close(); return false; }
        blocks[i].scales.push_back(scale);
        
        if (!read_quantized_bias(blocks[i].norm2_weight, scale)) { weights_file.close(); return false; }
        blocks[i].scales.push_back(scale);
        if (!read_quantized_bias(blocks[i].norm2_bias, scale)) { weights_file.close(); return false; }
        blocks[i].scales.push_back(scale);
        
        if (!is_moe_format) {
            if (!read_quantized_layer(blocks[i].ff1_weight, scale)) { weights_file.close(); return false; }
            blocks[i].scales.push_back(scale);
            if (!read_quantized_bias(blocks[i].ff1_bias, scale)) { weights_file.close(); return false; }
            blocks[i].scales.push_back(scale);

            if (!read_quantized_layer(blocks[i].ff2_weight, scale)) { weights_file.close(); return false; }
            blocks[i].scales.push_back(scale);
            if (!read_quantized_bias(blocks[i].ff2_bias, scale)) { weights_file.close(); return false; }
            blocks[i].scales.push_back(scale);
        } else {
            blocks[i].use_moe = true;
            uint32_t n_experts = 0;
            uint32_t top_k = 0;
            uint32_t expert_d_ff = 0;
            uint8_t has_shared = 0;

            if (weights_file.readBytes((char*)&n_experts, sizeof(uint32_t)) != sizeof(uint32_t)) { weights_file.close(); return false; }
            if (weights_file.readBytes((char*)&top_k, sizeof(uint32_t)) != sizeof(uint32_t)) { weights_file.close(); return false; }
            if (weights_file.readBytes((char*)&expert_d_ff, sizeof(uint32_t)) != sizeof(uint32_t)) { weights_file.close(); return false; }
            if (weights_file.readBytes((char*)&has_shared, sizeof(uint8_t)) != sizeof(uint8_t)) { weights_file.close(); return false; }

            blocks[i].moe_n_experts = static_cast<int>(n_experts);
            blocks[i].moe_top_k = static_cast<int>(top_k);
            blocks[i].moe_expert_d_ff = static_cast<int>(expert_d_ff);
            blocks[i].moe_has_shared = (has_shared != 0);

            if (!read_quantized_layer(blocks[i].moe_router_weight, blocks[i].moe_router_weight_scale)) { weights_file.close(); return false; }
            if (!read_quantized_bias(blocks[i].moe_router_bias, blocks[i].moe_router_bias_scale)) { weights_file.close(); return false; }

            blocks[i].moe_ff1_weight.resize(blocks[i].moe_n_experts);
            blocks[i].moe_ff1_bias.resize(blocks[i].moe_n_experts);
            blocks[i].moe_ff2_weight.resize(blocks[i].moe_n_experts);
            blocks[i].moe_ff2_bias.resize(blocks[i].moe_n_experts);
            blocks[i].moe_ff1_weight_scales.resize(blocks[i].moe_n_experts, 1.0f);
            blocks[i].moe_ff1_bias_scales.resize(blocks[i].moe_n_experts, 1.0f);
            blocks[i].moe_ff2_weight_scales.resize(blocks[i].moe_n_experts, 1.0f);
            blocks[i].moe_ff2_bias_scales.resize(blocks[i].moe_n_experts, 1.0f);

            for (int expert = 0; expert < blocks[i].moe_n_experts; ++expert) {
                if (!read_quantized_layer(blocks[i].moe_ff1_weight[expert], blocks[i].moe_ff1_weight_scales[expert])) { weights_file.close(); return false; }
                if (!read_quantized_bias(blocks[i].moe_ff1_bias[expert], blocks[i].moe_ff1_bias_scales[expert])) { weights_file.close(); return false; }
                if (!read_quantized_layer(blocks[i].moe_ff2_weight[expert], blocks[i].moe_ff2_weight_scales[expert])) { weights_file.close(); return false; }
                if (!read_quantized_bias(blocks[i].moe_ff2_bias[expert], blocks[i].moe_ff2_bias_scales[expert])) { weights_file.close(); return false; }
            }

            if (blocks[i].moe_has_shared) {
                if (!read_quantized_layer(blocks[i].moe_shared_ff1_weight, blocks[i].moe_shared_ff1_weight_scale)) { weights_file.close(); return false; }
                if (!read_quantized_bias(blocks[i].moe_shared_ff1_bias, blocks[i].moe_shared_ff1_bias_scale)) { weights_file.close(); return false; }
                if (!read_quantized_layer(blocks[i].moe_shared_ff2_weight, blocks[i].moe_shared_ff2_weight_scale)) { weights_file.close(); return false; }
                if (!read_quantized_bias(blocks[i].moe_shared_ff2_bias, blocks[i].moe_shared_ff2_bias_scale)) { weights_file.close(); return false; }
            }
        }
    }
    
    // Read final norm
    if (!read_quantized_bias(norm_weight, scale)) { weights_file.close(); return false; }
    weight_scales.push_back(scale);
    if (!read_quantized_bias(norm_bias, scale)) { weights_file.close(); return false; }
    weight_scales.push_back(scale);
    
    // Read lm_head
    if (!read_quantized_layer(lm_head, scale)) { weights_file.close(); return false; }
    weight_scales.push_back(scale);
    if (!skip_bias_stub()) { weights_file.close(); return false; }
    if (lm_head.empty()) {
        lm_head = token_embedding;
        if (!weight_scales.empty()) {
            weight_scales.back() = weight_scales.front();
        }
    }
    
    weights_file.close();
    return true;
}

bool NanoLLM::load(const char* weights_path, const char* config_path) {
    // Load config first
    if (!readConfigFromFile(config_path)) {
        return false;
    }

    // Load weights
    if (!readWeightsFromFile(weights_path)) {
        return false;
    }
    
    Serial.printf("Free heap after weights: %u bytes (internal), %u bytes (PSRAM)\n",
                  static_cast<unsigned>(ESP.getFreeHeap()),
                  static_cast<unsigned>(heap_caps_get_free_size(MALLOC_CAP_SPIRAM)));

    // Allocate buffers
    if (!allocateBuffers()) {
        Serial.println("Failed to allocate inference buffers (out of memory).");
        return false;
    }
    
    return true;
}

bool NanoLLM::allocateBuffers() {
    inference_seq_len = config.max_seq_len;
#ifdef NANOLLM_NO_PSRAM
    inference_seq_len = std::min(inference_seq_len, kMaxInferenceSeqLenNoPsram);
#endif

    const size_t model_elems = static_cast<size_t>(inference_seq_len) * config.d_model;
    const size_t kv_dim = static_cast<size_t>(config.d_model / config.n_heads) * config.n_kv_heads;
    const bool use_int8_kv_cache = config.n_kv_heads != config.n_heads;
    const size_t kv_elems = use_int8_kv_cache
        ? static_cast<size_t>(config.n_layers) * inference_seq_len * kv_dim : 0;
    const size_t kv_scale_elems = use_int8_kv_cache
        ? static_cast<size_t>(config.n_layers) * inference_seq_len : 0;
    const size_t score_elems = static_cast<size_t>(inference_seq_len);

    Serial.printf("Allocating inference buffers for seq_len=%d (model max=%d)\n",
                  inference_seq_len, config.max_seq_len);
    Serial.printf("Free heap before buffers: %u bytes (internal), %u bytes (PSRAM)\n",
                  static_cast<unsigned>(ESP.getFreeHeap()),
                  static_cast<unsigned>(heap_caps_get_free_size(MALLOC_CAP_SPIRAM)));

    try {
        const size_t single_token = config.d_model;
        if (use_int8_kv_cache) {
            // Match moe_tradeoff_estimator cached-MQA working set: 4*d + S floats
            // plus int8 K/V caches. Full-seq float activations are not needed for decodeStep.
            hidden_state.resize(single_token);
            temp_buffer1.resize(single_token);
            temp_buffer2.resize(single_token);
            attn_q.resize(single_token);
            attn_k.clear();
            attn_v.clear();
        } else {
            const size_t seq_elems = model_elems;
            hidden_state.resize(model_elems);
            temp_buffer1.resize(seq_elems);
            temp_buffer2.resize(seq_elems);
            attn_q.resize(single_token);
            attn_k.resize(model_elems);
            attn_v.resize(model_elems);
        }
        kv_key_cache.resize(kv_elems);
        kv_value_cache.resize(kv_elems);
        kv_key_scales.resize(kv_scale_elems);
        kv_value_scales.resize(kv_scale_elems);
        attn_scores.resize(score_elems);
        logits_buffer.clear();                   // Greedy decode streams LM head rows; no full-vocab logits buffer.
        rope_inv_freq.clear();
        if (config.use_rope && config.n_heads > 0) {
            const int d_k = config.d_model / config.n_heads;
            rope_inv_freq.resize(d_k / 2);
            nanollm_rope::fill_inv_freq(d_k, rope_inv_freq.data());
        }
        if (config.use_moe) {
            const int n_experts = std::max(1, config.moe_n_experts);
            const int top_k = std::min(std::max(1, config.moe_top_k), n_experts);
            const int expert_d_ff = std::max(1, config.d_ff);
            const int shared_d_ff = std::max(1, config.moe_shared_d_ff);
            moe_router_logits.resize(n_experts);
            moe_topk_indices.resize(top_k);
            moe_topk_logits.resize(top_k);
            moe_topk_gates.resize(top_k);
            moe_expert_order.resize(n_experts);
            moe_expert_hidden.resize(expert_d_ff);
            moe_expert_output.resize(config.d_model);
            moe_shared_hidden.resize(shared_d_ff);
        }
    } catch (const std::bad_alloc&) {
        hidden_state.clear();
        temp_buffer1.clear();
        temp_buffer2.clear();
        attn_q.clear();
        attn_k.clear();
        attn_v.clear();
        kv_key_cache.clear();
        kv_value_cache.clear();
        kv_key_scales.clear();
        kv_value_scales.clear();
        attn_scores.clear();
        logits_buffer.clear();
        moe_router_logits.clear();
        moe_topk_indices.clear();
        moe_topk_logits.clear();
        moe_topk_gates.clear();
        moe_expert_order.clear();
        moe_expert_hidden.clear();
        moe_expert_output.clear();
        moe_shared_hidden.clear();
        return false;
    }

    size_t working_bytes = getMemoryUsage();
    float working_kb = static_cast<float>(working_bytes) / 1024.0f;
    Serial.printf("NanoLLM working memory: %.1f KB (%lu bytes)\n", working_kb, static_cast<unsigned long>(working_bytes));
    if (working_bytes > kSramWarningThresholdBytes) {
        Serial.printf("Warning: working memory exceeds SRAM budget (%.1f KB > %.1f KB). Consider reducing d_model or max_seq_len.\n",
                      working_kb, static_cast<float>(kSramWarningThresholdBytes) / 1024.0f);
    }
    return true;
}

void NanoLLM::dequantize(const qint8_t* quantized, float* output, size_t size, float scale) {
    for (size_t i = 0; i < size; i++) {
        output[i] = static_cast<float>(quantized[i]) / scale;
    }
}

void NanoLLM::dequantizeFromProgmem(const int8_t* quantized_pgm, float* output, size_t size, float scale) {
    for (size_t i = 0; i < size; i++) {
        int8_t val = pgm_read_byte(&quantized_pgm[i]);
        output[i] = static_cast<float>(val) / scale;
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

void NanoLLM::linearFromProgmem(const int8_t* weight_pgm, const float* input, float* output,
                                int in_dim, int out_dim, float scale,
                                const int8_t* bias_pgm, float bias_scale) {
    // Matrix multiplication with PROGMEM weights: output = input @ weight^T
    for (int i = 0; i < out_dim; i++) {
        float sum = 0.0f;
        for (int j = 0; j < in_dim; j++) {
            int8_t w = pgm_read_byte(&weight_pgm[i * in_dim + j]);
            sum += input[j] * (static_cast<float>(w) / scale);
        }
        output[i] = sum;
        if (bias_pgm) {
            int8_t b = pgm_read_byte(&bias_pgm[i]);
            output[i] += static_cast<float>(b) / bias_scale;
        }
    }
}

void NanoLLM::layer_norm(const float* input, float* output, int size,
                         const qint8_t* weight, const qint8_t* bias,
                         float weight_scale, float bias_scale) {
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
    float std_dev = sqrtf(variance + 1e-5f);
    float w_scale = (fabsf(weight_scale) > 1e-9f) ? weight_scale : 1.0f;
    float b_scale = (bias && fabsf(bias_scale) > 1e-9f) ? bias_scale : 1.0f;
    
    // Normalize and scale
    for (int i = 0; i < size; i++) {
        float gamma = weight ? (static_cast<float>(weight[i]) / w_scale) : 1.0f;
        float beta = bias ? (static_cast<float>(bias[i]) / b_scale) : 0.0f;
        output[i] = ((input[i] - mean) / std_dev) * gamma + beta;
    }
}

void NanoLLM::layer_normFromProgmem(const float* input, float* output, int size,
                                    const int8_t* weight_pgm, const int8_t* bias_pgm,
                                    float weight_scale, float bias_scale) {
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
    float std_dev = sqrtf(variance + 1e-5f);
    float w_scale = (fabsf(weight_scale) > 1e-9f) ? weight_scale : 1.0f;
    float b_scale = (bias_pgm && fabsf(bias_scale) > 1e-9f) ? bias_scale : 1.0f;
    
    // Normalize and scale with PROGMEM weights
    for (int i = 0; i < size; i++) {
        int8_t w = pgm_read_byte(&weight_pgm[i]);
        float gamma = static_cast<float>(w) / w_scale;
        float beta = 0.0f;
        if (bias_pgm) {
            int8_t b = pgm_read_byte(&bias_pgm[i]);
            beta = static_cast<float>(b) / b_scale;
        }
        output[i] = ((input[i] - mean) / std_dev) * gamma + beta;
    }
}

void NanoLLM::gelu(float* x, int size) {
    const float sqrt_2_over_pi = 0.7978845608f;
    const float coeff = 0.044715f;
    
    for (int i = 0; i < size; i++) {
        float x_val = x[i];
        float x3 = x_val * x_val * x_val;
        x[i] = 0.5f * x_val * (1.0f + tanhf(sqrt_2_over_pi * (x_val + coeff * x3)));
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
        x[i] = expf(x[i] - max_val);
        sum += x[i];
    }
    
    // Normalize
    for (int i = 0; i < size; i++) {
        x[i] /= sum;
    }
}

namespace {

int linear_out_dim(size_t weight_size, int in_dim) {
    if (in_dim <= 0) {
        return 0;
    }
    return static_cast<int>(weight_size / static_cast<size_t>(in_dim));
}

int argmaxLinear(const qint8_t* weight, const float* input, int in_dim, int out_dim, float scale) {
    int best_idx = 0;
    float best = -INFINITY;
    const float weight_scale = (fabsf(scale) > 1e-9f) ? scale : 1.0f;
    for (int i = 0; i < out_dim; ++i) {
        float sum = 0.0f;
        const qint8_t* row = &weight[i * in_dim];
        for (int j = 0; j < in_dim; ++j) {
            sum += input[j] * (static_cast<float>(row[j]) / weight_scale);
        }
        if (sum > best) {
            best = sum;
            best_idx = i;
        }
    }
    return best_idx;
}

int argmaxLinearFromProgmem(const int8_t* weight_pgm, const float* input, int in_dim, int out_dim, float scale) {
    int best_idx = 0;
    float best = -INFINITY;
    const float weight_scale = (fabsf(scale) > 1e-9f) ? scale : 1.0f;
    for (int i = 0; i < out_dim; ++i) {
        float sum = 0.0f;
        const int row_offset = i * in_dim;
        for (int j = 0; j < in_dim; ++j) {
            int8_t w = pgm_read_byte(&weight_pgm[row_offset + j]);
            sum += input[j] * (static_cast<float>(w) / weight_scale);
        }
        if (sum > best) {
            best = sum;
            best_idx = i;
        }
    }
    return best_idx;
}

} // namespace

void NanoLLM::attention(const float* x, float* output, int block_idx, int seq_len) {
    if (config.n_heads <= 0) {
        Serial.printf("[ERR] Invalid head count (%d); forcing to 1\n", config.n_heads);
        config.n_heads = 1;
    } else if (config.d_model % config.n_heads != 0) {
        Serial.printf(
            "[ERR] d_model=%d not divisible by n_heads=%d; forcing head count to 1\n",
            config.d_model,
            config.n_heads);
        config.n_heads = 1;
    }

    int d_k = config.d_model / config.n_heads;
    const int n_heads = config.n_heads;
    const int n_kv_heads = config.n_kv_heads > 0 ? config.n_kv_heads : n_heads;
    const int kv_dim = n_kv_heads * d_k;
    const int kv_repeats = n_heads / n_kv_heads;
    const float inv_sqrt_dk = 1.0f / sqrtf(static_cast<float>(d_k));
    if (seq_len > config.max_seq_len) {
        seq_len = config.max_seq_len;
    }
    if (seq_len <= 0) {
        seq_len = 1;
    }

    float* Q_single = attn_q.data();
    const size_t cache_base = static_cast<size_t>(block_idx) * inference_seq_len * kv_dim;
    const size_t scale_base = static_cast<size_t>(block_idx) * inference_seq_len;
    qint8_t* K = kv_key_cache.empty() ? nullptr : kv_key_cache.data() + cache_base;
    qint8_t* V = kv_value_cache.empty() ? nullptr : kv_value_cache.data() + cache_base;
    float* K_float = attn_k.empty() ? nullptr : attn_k.data();
    float* V_float = attn_v.empty() ? nullptr : attn_v.data();
    const bool preserve_mha_float = n_kv_heads == n_heads && K_float && V_float;
    float* K_scales = kv_key_scales.empty() ? nullptr : kv_key_scales.data() + scale_base;
    float* V_scales = kv_value_scales.empty() ? nullptr : kv_value_scales.data() + scale_base;
    float* scores = attn_scores.data();
    auto quantize_cache = [](const float* src, qint8_t* dst, int size) {
        float max_abs = 0.0f;
        for (int i = 0; i < size; ++i) max_abs = std::max(max_abs, fabsf(src[i]));
        const float scale = max_abs > 0.0f ? 127.0f / max_abs : 1.0f;
        for (int i = 0; i < size; ++i) {
            dst[i] = static_cast<qint8_t>(std::max(-127.0f, std::min(127.0f, roundf(src[i] * scale))));
        }
        return scale;
    };
    const int score_elems = seq_len;
    if (score_elems > 0) {
        std::fill(scores, scores + score_elems, 0.0f);
    }
    if (seq_len * config.d_model > 0) {
        std::fill(output, output + seq_len * config.d_model, 0.0f);
    }

#ifdef NANOLLM_USE_EMBEDDED_WEIGHTS
    if (isUsingEmbeddedWeights()) {
        nanollm::TransformerBlock block = loadEmbeddedBlock(block_idx);

        for (int token = 0; token < seq_len; ++token) {
            const float* token_input = &x[token * config.d_model];
            float* key_out = preserve_mha_float ? &K_float[token * kv_dim] : Q_single;
            linearFromProgmem(block.attn_k.data, token_input, key_out,
                              config.d_model, kv_dim, block.attn_k.scale);
            if (config.use_rope && !rope_inv_freq.empty()) {
                nanollm_rope::apply_heads(key_out, n_kv_heads, d_k, token, rope_inv_freq.data());
            }
            if (!preserve_mha_float) K_scales[token] = quantize_cache(key_out, &K[token * kv_dim], kv_dim);
            float* value_out = preserve_mha_float ? &V_float[token * kv_dim] : Q_single;
            linearFromProgmem(block.attn_v.data, token_input, value_out,
                              config.d_model, kv_dim, block.attn_v.scale);
            if (!preserve_mha_float) V_scales[token] = quantize_cache(value_out, &V[token * kv_dim], kv_dim);
        }

        for (int head = 0; head < n_heads; ++head) {
            const int head_offset = head * d_k;
            const int kv_head_offset = (head / kv_repeats) * d_k;
            for (int i = 0; i < seq_len; ++i) {
                linearFromProgmem(block.attn_q.data, &x[i * config.d_model], Q_single,
                                  config.d_model, config.d_model, block.attn_q.scale);
                if (config.use_rope && !rope_inv_freq.empty()) {
                    nanollm_rope::apply_heads(Q_single, n_heads, d_k, i, rope_inv_freq.data());
                }
                for (int j = 0; j < seq_len; ++j) {
                    if (j > i) {
                        scores[j] = -INFINITY;
                        continue;
                    }
                    float score = 0.0f;
                    const int k_base = j * kv_dim + kv_head_offset;
                    for (int dim = 0; dim < d_k; ++dim) {
                        const float key = preserve_mha_float ? K_float[k_base + dim]
                            : static_cast<float>(K[k_base + dim]) / K_scales[j];
                        score += Q_single[head_offset + dim] * key;
                    }
                    scores[j] = score * inv_sqrt_dk;
                }
                softmax(scores, seq_len);

                float* head_out = &output[i * config.d_model + head_offset];
                for (int dim = 0; dim < d_k; ++dim) {
                    float weighted = 0.0f;
                    for (int t = 0; t < seq_len; ++t) {
                        const int idx = t * kv_dim + kv_head_offset + dim;
                        const float value = preserve_mha_float ? V_float[idx]
                            : static_cast<float>(V[idx]) / V_scales[t];
                        weighted += scores[t] * value;
                    }
                    head_out[dim] = weighted;
                }
            }
        }

        for (int token = 0; token < seq_len; ++token) {
            float* token_out = &output[token * config.d_model];
            linearFromProgmem(block.attn_o.data, token_out, Q_single,
                              config.d_model, config.d_model, block.attn_o.scale);
            memcpy(token_out, Q_single, static_cast<size_t>(config.d_model) * sizeof(float));
        }
    } else
#endif
    {
        for (int token = 0; token < seq_len; ++token) {
            const float* token_input = &x[token * config.d_model];
            float* key_out = preserve_mha_float ? &K_float[token * kv_dim] : Q_single;
            linear(blocks[block_idx].attn_k.data(), token_input, key_out,
                   config.d_model, kv_dim, blocks[block_idx].scales[1]);
            if (config.use_rope && !rope_inv_freq.empty()) {
                nanollm_rope::apply_heads(key_out, n_kv_heads, d_k, token, rope_inv_freq.data());
            }
            if (!preserve_mha_float) K_scales[token] = quantize_cache(key_out, &K[token * kv_dim], kv_dim);
            float* value_out = preserve_mha_float ? &V_float[token * kv_dim] : Q_single;
            linear(blocks[block_idx].attn_v.data(), token_input, value_out,
                   config.d_model, kv_dim, blocks[block_idx].scales[2]);
            if (!preserve_mha_float) V_scales[token] = quantize_cache(value_out, &V[token * kv_dim], kv_dim);
        }

        for (int head = 0; head < n_heads; ++head) {
            const int head_offset = head * d_k;
            const int kv_head_offset = (head / kv_repeats) * d_k;
            for (int i = 0; i < seq_len; ++i) {
                linear(blocks[block_idx].attn_q.data(), &x[i * config.d_model], Q_single,
                       config.d_model, config.d_model, blocks[block_idx].scales[0]);
                if (config.use_rope && !rope_inv_freq.empty()) {
                    nanollm_rope::apply_heads(Q_single, n_heads, d_k, i, rope_inv_freq.data());
                }
                for (int j = 0; j < seq_len; ++j) {
                    if (j > i) {
                        scores[j] = -INFINITY;
                        continue;
                    }
                    float score = 0.0f;
                    const int k_base = j * kv_dim + kv_head_offset;
                    for (int dim = 0; dim < d_k; ++dim) {
                        const float key = preserve_mha_float ? K_float[k_base + dim]
                            : static_cast<float>(K[k_base + dim]) / K_scales[j];
                        score += Q_single[head_offset + dim] * key;
                    }
                    scores[j] = score * inv_sqrt_dk;
                }
                softmax(scores, seq_len);

                float* head_out = &output[i * config.d_model + head_offset];
                for (int dim = 0; dim < d_k; ++dim) {
                    float weighted = 0.0f;
                    for (int t = 0; t < seq_len; ++t) {
                        const int idx = t * kv_dim + kv_head_offset + dim;
                        const float value = preserve_mha_float ? V_float[idx]
                            : static_cast<float>(V[idx]) / V_scales[t];
                        weighted += scores[t] * value;
                    }
                    head_out[dim] = weighted;
                }
            }
        }

        for (int token = 0; token < seq_len; ++token) {
            float* token_out = &output[token * config.d_model];
            linear(blocks[block_idx].attn_o.data(), token_out, Q_single,
                   config.d_model, config.d_model, blocks[block_idx].scales[3]);
            memcpy(token_out, Q_single, static_cast<size_t>(config.d_model) * sizeof(float));
        }
    }
}

void NanoLLM::feed_forward(const float* x, float* output, int block_idx, int seq_len) {
    if (seq_len > config.max_seq_len) {
        seq_len = config.max_seq_len;
    }
    if (seq_len <= 0) {
        seq_len = 1;
    }

#ifdef NANOLLM_USE_EMBEDDED_WEIGHTS
    if (isUsingEmbeddedWeights()) {
        nanollm::TransformerBlock block = loadEmbeddedBlock(block_idx);
        if (block.use_moe_block) {
            feed_forward_embedded_moe(x, output, block, seq_len);
            return;
        }

        for (int token = 0; token < seq_len; ++token) {
            const float* token_input = &x[token * config.d_model];
            float* token_output = &output[token * config.d_model];
            linearFromProgmem(block.ff1_weight.data, token_input, temp_buffer1.data(),
                             config.d_model, config.d_ff, block.ff1_weight.scale,
                             block.ff1_bias.data, block.ff1_bias.scale);
            gelu(temp_buffer1.data(), config.d_ff);
            linearFromProgmem(block.ff2_weight.data, temp_buffer1.data(), token_output,
                             config.d_ff, config.d_model, block.ff2_weight.scale,
                             block.ff2_bias.data, block.ff2_bias.scale);
        }
        return;
    } else
#endif
    {
        if (blocks[block_idx].use_moe) {
            const auto& b = blocks[block_idx];
            const int n_experts = std::max(1, b.moe_n_experts);
            const int top_k = std::min(std::max(1, b.moe_top_k), n_experts);
            const int expert_d_ff = std::max(1, b.moe_expert_d_ff);
            const int shared_d_ff = std::max(
                1,
                linear_out_dim(b.moe_shared_ff1_weight.size(), config.d_model));

            float* router_logits = moe_router_logits.data();
            int* topk_indices = moe_topk_indices.data();
            float* topk_logits = moe_topk_logits.data();
            float* topk_gates = moe_topk_gates.data();
            int* order = moe_expert_order.data();
            float* expert_hidden = moe_expert_hidden.data();
            float* expert_output = moe_expert_output.data();
            float* shared_hidden = moe_shared_hidden.data();

            const qint8_t* router_bias_ptr = b.moe_router_bias.empty() ? nullptr : b.moe_router_bias.data();

            for (int token = 0; token < seq_len; ++token) {
                const float* token_input = &x[token * config.d_model];
                float* token_output = &output[token * config.d_model];
                std::fill(token_output, token_output + config.d_model, 0.0f);

                linear(b.moe_router_weight.data(), token_input, router_logits,
                       config.d_model, n_experts, b.moe_router_weight_scale,
                       router_bias_ptr, b.moe_router_bias_scale);

                if (top_k == 1) {
                    int best_expert = 0;
                    float best_logit = router_logits[0];
                    for (int i = 1; i < n_experts; ++i) {
                        if (router_logits[i] > best_logit) {
                            best_logit = router_logits[i];
                            best_expert = i;
                        }
                    }
                    float denom = 0.0f;
                    for (int i = 0; i < n_experts; ++i) {
                        denom += expf(router_logits[i] - best_logit);
                    }
                    topk_indices[0] = best_expert;
                    topk_gates[0] = denom > 0.0f ? 1.0f / denom : 1.0f;
                } else {
                    for (int i = 0; i < n_experts; ++i) {
                        order[i] = i;
                    }
                    std::partial_sort(order, order + top_k, order + n_experts,
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

                    float max_logit = topk_logits[0];
                    for (int k = 1; k < top_k; ++k) {
                        if (topk_logits[k] > max_logit) {
                            max_logit = topk_logits[k];
                        }
                    }
                    float denom = 0.0f;
                    for (int k = 0; k < top_k; ++k) {
                        topk_gates[k] = expf(topk_logits[k] - max_logit);
                        denom += topk_gates[k];
                    }
                    if (denom > 0.0f) {
                        for (int k = 0; k < top_k; ++k) {
                            topk_gates[k] /= denom;
                        }
                    }
                }

                for (int k = 0; k < top_k; ++k) {
                    int expert = topk_indices[k];
                    float gate = topk_gates[k];

                    const qint8_t* ff1_bias_ptr = b.moe_ff1_bias[expert].empty() ? nullptr : b.moe_ff1_bias[expert].data();
                    const qint8_t* ff2_bias_ptr = b.moe_ff2_bias[expert].empty() ? nullptr : b.moe_ff2_bias[expert].data();

                          linear(b.moe_ff1_weight[expert].data(), token_input, expert_hidden,
                           config.d_model, expert_d_ff, b.moe_ff1_weight_scales[expert],
                           ff1_bias_ptr, b.moe_ff1_bias_scales[expert]);
                          gelu(expert_hidden, expert_d_ff);
                          linear(b.moe_ff2_weight[expert].data(), expert_hidden, expert_output,
                           expert_d_ff, config.d_model, b.moe_ff2_weight_scales[expert],
                           ff2_bias_ptr, b.moe_ff2_bias_scales[expert]);

                    for (int j = 0; j < config.d_model; ++j) {
                        token_output[j] += gate * expert_output[j];
                    }
                }

                if (b.moe_has_shared && !b.moe_shared_ff1_weight.empty() && !b.moe_shared_ff2_weight.empty()) {
                    const qint8_t* shared_ff1_bias_ptr = b.moe_shared_ff1_bias.empty() ? nullptr : b.moe_shared_ff1_bias.data();
                    const qint8_t* shared_ff2_bias_ptr = b.moe_shared_ff2_bias.empty() ? nullptr : b.moe_shared_ff2_bias.data();

                          linear(b.moe_shared_ff1_weight.data(), token_input, shared_hidden,
                           config.d_model, shared_d_ff, b.moe_shared_ff1_weight_scale,
                           shared_ff1_bias_ptr, b.moe_shared_ff1_bias_scale);
                          gelu(shared_hidden, shared_d_ff);
                          linear(b.moe_shared_ff2_weight.data(), shared_hidden, expert_output,
                           shared_d_ff, config.d_model, b.moe_shared_ff2_weight_scale,
                           shared_ff2_bias_ptr, b.moe_shared_ff2_bias_scale);

                    for (int j = 0; j < config.d_model; ++j) {
                        token_output[j] += expert_output[j];
                    }
                }
            }
            return;
        }

        // Dense feed-forward (SPIFFS)
        for (int token = 0; token < seq_len; ++token) {
            const float* token_input = &x[token * config.d_model];
            float* token_output = &output[token * config.d_model];
            linear(blocks[block_idx].ff1_weight.data(), token_input, temp_buffer1.data(),
                   config.d_model, config.d_ff, blocks[block_idx].scales[8],
                   blocks[block_idx].ff1_bias.data(), blocks[block_idx].scales[9]);
            gelu(temp_buffer1.data(), config.d_ff);
            linear(blocks[block_idx].ff2_weight.data(), temp_buffer1.data(), token_output,
                   config.d_ff, config.d_model, blocks[block_idx].scales[10],
                   blocks[block_idx].ff2_bias.data(), blocks[block_idx].scales[11]);
        }
    }
}

std::vector<int> NanoLLM::generate(const std::vector<int>& prompt, int max_new_tokens, float temperature,
                                   TokenCallback callback, void* callback_user_data) {
    std::vector<int> generated = prompt;
    if (generated.empty()) {
        generated.push_back(0);
    }

    // MQA / GQA: use int8 KV-cache decodeStep (fits Cardputer SRAM at long context).
    if (config.n_kv_heads != config.n_heads) {
        resetCache();
        int next_token = -1;
        for (size_t i = 0; i < generated.size(); ++i) {
            nano_feed_watchdog();
            next_token = decodeStep(generated[i], temperature);
        }
        for (int step = 0; step < max_new_tokens; ++step) {
            nano_feed_watchdog();
            if (next_token < 0) {
                break;
            }
            generated.push_back(next_token);
            if (callback) {
                callback(next_token, step, callback_user_data);
            }
            if (next_token == 3) {  // <EOS>
                break;
            }
            next_token = decodeStep(next_token, temperature);
        }
        return generated;
    }

    const int max_context = std::max(1, std::min(config.max_seq_len, inference_seq_len > 0 ? inference_seq_len : config.max_seq_len));
    const size_t single_token_elems = static_cast<size_t>(config.d_model);

    for (int step = 0; step < max_new_tokens; ++step) {
        nano_feed_watchdog();
        const int seq_len = std::min(static_cast<int>(generated.size()), max_context);
        const size_t context_start = generated.size() - static_cast<size_t>(seq_len);
        const size_t seq_elems = static_cast<size_t>(seq_len) * single_token_elems;

        if (hidden_state.size() < seq_elems) {
            Serial.println("generate(): context exceeds hidden_state buffer.");
            return generated;
        }

#ifdef NANOLLM_USE_EMBEDDED_WEIGHTS
        if (isUsingEmbeddedWeights()) {
            // Read embedding structures from PROGMEM
            nanollm::QuantizedLayer tok_emb = loadEmbeddedLayer(&embedded_weights_ptr->token_embedding);

            // Initialize hidden state with the cropped autoregressive context.
            for (int i = 0; i < seq_len; i++) {
                int token = generated[context_start + static_cast<size_t>(i)];

                // Token embedding
                for (int j = 0; j < config.d_model; j++) {
                    int8_t tok_val = pgm_read_byte(&tok_emb.data[token * config.d_model + j]);
                    hidden_state[i * config.d_model + j] = static_cast<float>(tok_val) / tok_emb.scale;
                }

                if (!config.use_rope) {
                    nanollm::QuantizedLayer pos_emb = loadEmbeddedLayer(&embedded_weights_ptr->pos_embedding);
                    int pos = i;
                    for (int j = 0; j < config.d_model; j++) {
                        int8_t pos_val = pgm_read_byte(&pos_emb.data[pos * config.d_model + j]);
                        hidden_state[i * config.d_model + j] += static_cast<float>(pos_val) / pos_emb.scale;
                    }
                }
            }
        } else
#endif
        {
            // Initialize hidden state with embeddings from SPIFFS
            for (int i = 0; i < seq_len; i++) {
                int token = generated[context_start + static_cast<size_t>(i)];

                // Token embedding
                float tok_scale = weight_scales[0];
                for (int j = 0; j < config.d_model; j++) {
                    hidden_state[i * config.d_model + j] =
                        static_cast<float>(token_embedding[token * config.d_model + j]) / tok_scale;
                }

                if (!config.use_rope) {
                    int pos = i;
                    float pos_scale = weight_scales[1];
                    for (int j = 0; j < config.d_model; j++) {
                        hidden_state[i * config.d_model + j] +=
                            static_cast<float>(pos_embedding[pos * config.d_model + j]) / pos_scale;
                    }
                }
            }
        }

        // Transformer blocks
        for (int block_idx = 0; block_idx < config.n_layers; block_idx++) {
            if ((block_idx & 3) == 0) {
                nano_feed_watchdog();
            }
#ifdef NANOLLM_USE_EMBEDDED_WEIGHTS
            if (isUsingEmbeddedWeights()) {
                // Read block structure from PROGMEM
                nanollm::TransformerBlock block = loadEmbeddedBlock(block_idx);

                // Layer norm 1
                for (int i = 0; i < seq_len; i++) {
                    layer_normFromProgmem(&hidden_state[i * config.d_model], &temp_buffer1[i * config.d_model],
                                         config.d_model, block.norm1_weight.data,
                                         block.norm1_bias.data, block.norm1_weight.scale,
                                         block.norm1_bias.scale);
                }

                // Attention
                attention(temp_buffer1.data(), temp_buffer2.data(), block_idx, seq_len);

                // Residual
                for (int i = 0; i < seq_len * config.d_model; i++) {
                    hidden_state[i] += temp_buffer2[i];
                }

                // Layer norm 2
                for (int i = 0; i < seq_len; i++) {
                    layer_normFromProgmem(&hidden_state[i * config.d_model], &temp_buffer1[i * config.d_model],
                                         config.d_model, block.norm2_weight.data,
                                         block.norm2_bias.data, block.norm2_weight.scale,
                                         block.norm2_bias.scale);
                }

                // Feed forward
                feed_forward(temp_buffer1.data(), temp_buffer2.data(), block_idx, seq_len);

                // Residual
                for (int i = 0; i < seq_len * config.d_model; i++) {
                    hidden_state[i] += temp_buffer2[i];
                }
            } else
#endif
            {
                // Use SPIFFS-loaded weights
                // Layer norm 1
                for (int i = 0; i < seq_len; i++) {
                    layer_norm(&hidden_state[i * config.d_model], &temp_buffer1[i * config.d_model],
                              config.d_model, blocks[block_idx].norm1_weight.data(),
                              blocks[block_idx].norm1_bias.data(), blocks[block_idx].scales[4],
                              blocks[block_idx].scales[5]);
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
                              blocks[block_idx].norm2_bias.data(), blocks[block_idx].scales[6],
                              blocks[block_idx].scales[7]);
                }

                // Feed forward
                feed_forward(temp_buffer1.data(), temp_buffer2.data(), block_idx, seq_len);

                // Residual
                for (int i = 0; i < seq_len * config.d_model; i++) {
                    hidden_state[i] += temp_buffer2[i];
                }
            }
        }

#ifdef NANOLLM_USE_EMBEDDED_WEIGHTS
        if (isUsingEmbeddedWeights()) {
            // Read final norm and lm_head from PROGMEM
            nanollm::QuantizedLayer final_norm_w = loadEmbeddedLayer(&embedded_weights_ptr->final_norm_weight);
            nanollm::QuantizedLayer final_norm_b = loadEmbeddedLayer(&embedded_weights_ptr->final_norm_bias);
            nanollm::QuantizedLayer lm_head_layer = loadEmbeddedLayer(&embedded_weights_ptr->lm_head);
            if (lm_head_layer.size == 0) {
                lm_head_layer = loadEmbeddedLayer(&embedded_weights_ptr->token_embedding);
            }

            // Final norm
            for (int i = 0; i < seq_len; i++) {
                layer_normFromProgmem(&hidden_state[i * config.d_model], &temp_buffer1[i * config.d_model],
                                     config.d_model, final_norm_w.data, final_norm_b.data,
                                     final_norm_w.scale, final_norm_b.scale);
            }

            int next_token = argmaxLinearFromProgmem(lm_head_layer.data,
                                                     &temp_buffer1[(seq_len - 1) * config.d_model],
                                                     config.d_model,
                                                     config.vocab_size,
                                                     lm_head_layer.scale);

            generated.push_back(next_token);
            if (next_token == 3) {  // <EOS>, fixed by the tokenizer artifact contract
                break;
            }
            if (callback) {
                callback(next_token, step, callback_user_data);
            }
        } else
#endif
        {
            // Use SPIFFS-loaded weights
            // Final norm
            for (int i = 0; i < seq_len; i++) {
                layer_norm(&hidden_state[i * config.d_model], &temp_buffer1[i * config.d_model],
                          config.d_model, norm_weight.data(), norm_bias.data(),
                          weight_scales[2], weight_scales[3]);
            }

            int next_token = argmaxLinear(lm_head.data(),
                                          &temp_buffer1[(seq_len - 1) * config.d_model],
                                          config.d_model,
                                          config.vocab_size,
                                          weight_scales.back());

            generated.push_back(next_token);
            if (next_token == 3) {  // <EOS>, fixed by the tokenizer artifact contract
                break;
            }
            if (callback) {
                callback(next_token, step, callback_user_data);
            }
        }
    }

    return generated;
}

void NanoLLM::resetCache() {
    decode_history.clear();
    std::fill(kv_key_cache.begin(), kv_key_cache.end(), 0);
    std::fill(kv_value_cache.begin(), kv_value_cache.end(), 0);
    std::fill(kv_key_scales.begin(), kv_key_scales.end(), 1.0f);
    std::fill(kv_value_scales.begin(), kv_value_scales.end(), 1.0f);
}

int NanoLLM::decodeStep(int token, float temperature) {
    if (config.n_kv_heads == config.n_heads) {
        // Preserve the legacy float-MHA path exactly.
        decode_history.push_back(token);
        const size_t max_context = static_cast<size_t>(std::max(1, inference_seq_len));
        if (decode_history.size() > max_context) decode_history.erase(decode_history.begin());
        std::vector<int> result = generate(decode_history, 1, temperature);
        return result.size() > decode_history.size() ? result.back() : -1;
    }
    const int d = config.d_model;
    const int d_k = d / config.n_heads;
    const int kv_dim = d_k * config.n_kv_heads;
    const int repeats = config.n_heads / config.n_kv_heads;
    int pos = static_cast<int>(decode_history.size());
    if (pos >= inference_seq_len) {
        std::vector<int> replay(decode_history.begin() + 1, decode_history.end());
        replay.push_back(token);
        resetCache();
        int last = -1;
        for (int replay_token : replay) {
            last = decodeStep(replay_token, temperature);
        }
        return last;
    }
    decode_history.push_back(token);
    if (token < 0 || token >= config.vocab_size) token = 0;
    std::vector<float> x(d), normed(d), q(d), k(kv_dim), v(kv_dim), values(d), projected(d);
    std::vector<float> scores(pos + 1);
#ifdef NANOLLM_USE_EMBEDDED_WEIGHTS
    if (isUsingEmbeddedWeights()) {
        nanollm::QuantizedLayer tok = loadEmbeddedLayer(&embedded_weights_ptr->token_embedding);
        for (int i = 0; i < d; ++i) {
            int8_t tok_val = pgm_read_byte(&tok.data[token * d + i]);
            x[i] = static_cast<float>(tok_val) / tok.scale;
        }
        if (!config.use_rope) {
            nanollm::QuantizedLayer position = loadEmbeddedLayer(&embedded_weights_ptr->pos_embedding);
            for (int i = 0; i < d; ++i) {
                int8_t pos_val = pgm_read_byte(&position.data[pos * d + i]);
                x[i] += static_cast<float>(pos_val) / position.scale;
            }
        }
    } else
#endif
    {
        for (int i = 0; i < d; ++i) {
            x[i] = static_cast<float>(token_embedding[token * d + i]) / weight_scales[0];
            if (!config.use_rope) {
                x[i] += static_cast<float>(pos_embedding[pos * d + i]) / weight_scales[1];
            }
        }
    }
    auto quantize_cache = [](const float* src, qint8_t* dst, int size) {
        float max_abs = 0.0f;
        for (int i = 0; i < size; ++i) max_abs = std::max(max_abs, fabsf(src[i]));
        float scale = max_abs > 0.0f ? 127.0f / max_abs : 1.0f;
        for (int i = 0; i < size; ++i) {
            dst[i] = static_cast<qint8_t>(std::max(-127.0f, std::min(127.0f, roundf(src[i] * scale))));
        }
        return scale;
    };
    for (int layer = 0; layer < config.n_layers; ++layer) {
#ifdef NANOLLM_USE_EMBEDDED_WEIGHTS
        nanollm::TransformerBlock embedded_block{};
        if (isUsingEmbeddedWeights()) {
            embedded_block = loadEmbeddedBlock(layer);
            layer_normFromProgmem(x.data(), normed.data(), d,
                                  embedded_block.norm1_weight.data, embedded_block.norm1_bias.data,
                                  embedded_block.norm1_weight.scale, embedded_block.norm1_bias.scale);
            linearFromProgmem(embedded_block.attn_q.data, normed.data(), q.data(),
                              d, d, embedded_block.attn_q.scale);
            linearFromProgmem(embedded_block.attn_k.data, normed.data(), k.data(),
                              d, kv_dim, embedded_block.attn_k.scale);
            linearFromProgmem(embedded_block.attn_v.data, normed.data(), v.data(),
                              d, kv_dim, embedded_block.attn_v.scale);
        } else
#endif
        {
            BlockWeights& block = blocks[layer];
            layer_norm(x.data(), normed.data(), d, block.norm1_weight.data(),
                       block.norm1_bias.data(), block.scales[4], block.scales[5]);
            linear(block.attn_q.data(), normed.data(), q.data(), d, d, block.scales[0]);
            linear(block.attn_k.data(), normed.data(), k.data(), d, kv_dim, block.scales[1]);
            linear(block.attn_v.data(), normed.data(), v.data(), d, kv_dim, block.scales[2]);
        }
        if (config.use_rope && !rope_inv_freq.empty()) {
            nanollm_rope::apply_heads(q.data(), config.n_heads, d_k, pos, rope_inv_freq.data());
            nanollm_rope::apply_heads(k.data(), config.n_kv_heads, d_k, pos, rope_inv_freq.data());
        }
        const size_t cache_base = static_cast<size_t>(layer) * inference_seq_len * kv_dim;
        const size_t scale_base = static_cast<size_t>(layer) * inference_seq_len;
        kv_key_scales[scale_base + pos] =
            quantize_cache(k.data(), &kv_key_cache[cache_base + static_cast<size_t>(pos) * kv_dim], kv_dim);
        kv_value_scales[scale_base + pos] =
            quantize_cache(v.data(), &kv_value_cache[cache_base + static_cast<size_t>(pos) * kv_dim], kv_dim);
        std::fill(values.begin(), values.end(), 0.0f);
        for (int head = 0; head < config.n_heads; ++head) {
            int kv_head = head / repeats;
            for (int t = 0; t <= pos; ++t) {
                float score = 0.0f;
                size_t base = cache_base + static_cast<size_t>(t) * kv_dim + kv_head * d_k;
                for (int dim = 0; dim < d_k; ++dim) {
                    score += q[head * d_k + dim] *
                        static_cast<float>(kv_key_cache[base + dim]) / kv_key_scales[scale_base + t];
                }
                scores[t] = score / sqrtf(static_cast<float>(d_k));
            }
            softmax(scores.data(), pos + 1);
            for (int dim = 0; dim < d_k; ++dim) {
                for (int t = 0; t <= pos; ++t) {
                    size_t idx = cache_base + static_cast<size_t>(t) * kv_dim + kv_head * d_k + dim;
                    values[head * d_k + dim] += scores[t] *
                        static_cast<float>(kv_value_cache[idx]) / kv_value_scales[scale_base + t];
                }
            }
        }
#ifdef NANOLLM_USE_EMBEDDED_WEIGHTS
        if (isUsingEmbeddedWeights()) {
            linearFromProgmem(embedded_block.attn_o.data, values.data(), projected.data(),
                              d, d, embedded_block.attn_o.scale);
            for (int i = 0; i < d; ++i) x[i] += projected[i];
            layer_normFromProgmem(x.data(), normed.data(), d,
                                  embedded_block.norm2_weight.data, embedded_block.norm2_bias.data,
                                  embedded_block.norm2_weight.scale, embedded_block.norm2_bias.scale);
        } else
#endif
        {
            BlockWeights& block = blocks[layer];
            linear(block.attn_o.data(), values.data(), projected.data(), d, d, block.scales[3]);
            for (int i = 0; i < d; ++i) x[i] += projected[i];
            layer_norm(x.data(), normed.data(), d, block.norm2_weight.data(),
                       block.norm2_bias.data(), block.scales[6], block.scales[7]);
        }
        feed_forward(normed.data(), projected.data(), layer, 1);
        for (int i = 0; i < d; ++i) x[i] += projected[i];
    }
#ifdef NANOLLM_USE_EMBEDDED_WEIGHTS
    if (isUsingEmbeddedWeights()) {
        nanollm::QuantizedLayer final_w = loadEmbeddedLayer(&embedded_weights_ptr->final_norm_weight);
        nanollm::QuantizedLayer final_b = loadEmbeddedLayer(&embedded_weights_ptr->final_norm_bias);
        nanollm::QuantizedLayer head = loadEmbeddedLayer(&embedded_weights_ptr->lm_head);
        if (head.size == 0) {
            head = loadEmbeddedLayer(&embedded_weights_ptr->token_embedding);
        }
        layer_normFromProgmem(x.data(), normed.data(), d, final_w.data, final_b.data,
                              final_w.scale, final_b.scale);
        return argmaxLinearFromProgmem(head.data, normed.data(), d, config.vocab_size, head.scale);
    }
#endif
    layer_norm(x.data(), normed.data(), d, norm_weight.data(), norm_bias.data(),
               weight_scales[2], weight_scales[3]);
    return argmaxLinear(lm_head.data(), normed.data(), d, config.vocab_size, weight_scales.back());
}

size_t NanoLLM::getMemoryUsage() const {
    size_t total = 0;
    total += hidden_state.size() * sizeof(float);
    total += temp_buffer1.size() * sizeof(float);
    total += temp_buffer2.size() * sizeof(float);
    total += attn_q.size() * sizeof(float);
    total += attn_k.size() * sizeof(float);
    total += attn_v.size() * sizeof(float);
    total += kv_key_cache.size() * sizeof(qint8_t);
    total += kv_value_cache.size() * sizeof(qint8_t);
    total += kv_key_scales.size() * sizeof(float);
    total += kv_value_scales.size() * sizeof(float);
    total += attn_scores.size() * sizeof(float);
    total += logits_buffer.size() * sizeof(float);
    total += moe_router_logits.size() * sizeof(float);
    total += moe_topk_indices.size() * sizeof(int);
    total += moe_topk_logits.size() * sizeof(float);
    total += moe_topk_gates.size() * sizeof(float);
    total += moe_expert_order.size() * sizeof(int);
    total += moe_expert_hidden.size() * sizeof(float);
    total += moe_expert_output.size() * sizeof(float);
    total += moe_shared_hidden.size() * sizeof(float);
    return total;
}

