#include "model_esp32.h"
#include <pgmspace.h>

#ifdef NANOLLM_USE_EMBEDDED_WEIGHTS
#include "model_weights.h"
#endif

bool NanoLLM::loadFromEmbedded() {
#ifdef NANOLLM_USE_EMBEDDED_WEIGHTS
    const nanollm::ModelConfig embedded_config = nanollm::GetModelConfig();
    config.vocab_size = embedded_config.vocab_size;
    config.d_model = embedded_config.d_model;
    config.n_layers = embedded_config.n_layers;
    config.n_heads = embedded_config.n_heads;
    config.n_kv_heads = embedded_config.n_kv_heads > 0
        ? embedded_config.n_kv_heads : embedded_config.n_heads;
    if (config.n_heads <= 0) {
        Serial.printf("[ERR] Embedded config has invalid head count (%d); forcing to 1\n", config.n_heads);
        config.n_heads = 1;
        config.n_kv_heads = 1;
    } else if (config.d_model % config.n_heads != 0) {
        Serial.printf(
            "[ERR] Embedded config d_model=%d not divisible by n_heads=%d; forcing head count to 1\n",
            config.d_model,
            config.n_heads);
        config.n_heads = 1;
        config.n_kv_heads = 1;
    }
    if (config.n_kv_heads <= 0 || config.n_heads % config.n_kv_heads != 0) {
        Serial.println("[ERR] Embedded config has invalid n_kv_heads");
        return false;
    }
    config.d_ff = embedded_config.d_ff;
    config.max_seq_len = embedded_config.max_seq_len;
    config.quantized = embedded_config.quantized;
    config.use_moe = embedded_config.use_moe;
    config.moe_n_experts = embedded_config.moe_n_experts;
    config.moe_top_k = embedded_config.moe_top_k;
    config.moe_shared_d_ff = embedded_config.moe_shared_d_ff;
    config.use_rope = embedded_config.use_rope;

    embedded_weights_ptr = nanollm::GetEmbeddedWeights();
    if (!embedded_weights_ptr) {
        Serial.println("Failed to get embedded weights pointer");
        return false;
    }

    Serial.printf("Loaded embedded weights: %d layers, d_model=%d, vocab=%d, moe=%s, rope=%s\n",
                  config.n_layers, config.d_model, config.vocab_size,
                  config.use_moe ? "yes" : "no",
                  config.use_rope ? "yes" : "no");

    Serial.printf("Free heap after embedded metadata: %u bytes (weights stay in flash)\n",
                  static_cast<unsigned>(ESP.getFreeHeap()));

    if (!allocateBuffers()) {
        return false;
    }
    return true;
#else
    Serial.println("Embedded weights not enabled. Define NANOLLM_USE_EMBEDDED_WEIGHTS and include model_weights.h");
    return false;
#endif
}

