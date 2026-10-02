# Embedded Weights Zero-Copy Design

## Overview

The embedded weights system has been restructured to enable **zero-copy** access to model weights stored in PROGMEM on ESP32. The inference procedure now simply sets a pointer to a pre-structured weights array and reads directly from flash memory, eliminating the need to copy all weights into RAM.

## Key Design Principles

1. **Pointer-Based Access**: The `NanoLLM` class maintains a pointer to `EmbeddedWeights` structure in PROGMEM
2. **Structured Layout**: All weights are organized in a hierarchical structure that mirrors the computation graph
3. **Minimal RAM Usage**: Only working buffers are allocated in RAM; weights remain in flash
4. **Dual-Mode Support**: Code supports both embedded (PROGMEM) and SPIFFS-loaded weights seamlessly

## Architecture

### Data Structures (in `model_weights.h`)

```cpp
// Basic quantized layer with metadata
struct QuantizedLayer {
    const int8_t* data;      // Pointer to PROGMEM array
    float scale;              // Dequantization scale
    size_t size;              // Total elements
    size_t rows, cols;        // Dimensions (for matrices)
};

// Complete transformer block
struct TransformerBlock {
    QuantizedLayer attn_q, attn_k, attn_v, attn_o;
    QuantizedLayer norm1_weight, norm1_bias;
    QuantizedLayer norm2_weight, norm2_bias;
    QuantizedLayer ff1_weight, ff1_bias;
    QuantizedLayer ff2_weight, ff2_bias;
};

// Top-level weights structure
struct EmbeddedWeights {
    QuantizedLayer token_embedding;
    QuantizedLayer pos_embedding;
    TransformerBlock* blocks;
    size_t n_blocks;
    QuantizedLayer final_norm_weight;
    QuantizedLayer final_norm_bias;
    QuantizedLayer lm_head;
};
```

### Implementation Details

#### Export Process (`python/export_weights_header.py`)

1. **Quantize weights** to int8 with per-layer scales
2. **Generate PROGMEM arrays** for each weight tensor
3. **Create structure initializers** that populate `TransformerBlock` and `EmbeddedWeights`
4. **Export single header** with complete model ready for compilation

The exporter generates:
- Individual weight arrays marked `PROGMEM`
- Block array `BLOCKS_ARRAY[n]` with all transformer layers
- Complete structure `EMBEDDED_WEIGHTS` ready for pointer access

#### Loading (`esp32_m5stack/src/model_embedded.cpp`)

```cpp
bool NanoLLM::loadFromEmbedded() {
    // Get config
    config = nanollm::GetModelConfig();
    
    // Set pointer to pre-structured weights
    embedded_weights_ptr = nanollm::GetEmbeddedWeights();
    
    // Allocate working buffers only
    allocateBuffers();
    
    return true;
}
```

**Zero copies**: The entire model loading is a single pointer assignment!

#### Inference (`esp32_m5stack/src/model_esp32.cpp`)

The inference code uses specialized PROGMEM-aware functions:

- `dequantizeFromProgmem()`: Read int8 from flash and dequantize
- `linearFromProgmem()`: Matrix multiply with weights in flash
- `layer_normFromProgmem()`: Layer normalization with flash weights

Example usage:
```cpp
void NanoLLM::attention(const float* x, float* output, int block_idx) {
    if (isUsingEmbeddedWeights()) {
        // Read block metadata from PROGMEM
        nanollm::TransformerBlock block;
        memcpy_P(&block, &embedded_weights_ptr->blocks[block_idx], 
                 sizeof(nanollm::TransformerBlock));
        
        // Use weights directly from flash via pointers
        linearFromProgmem(block.attn_q.data, x, Q.data(),
                         config.d_model, config.d_model, block.attn_q.scale);
        // ... rest of attention
    } else {
        // Use SPIFFS-loaded weights from RAM
        linear(blocks[block_idx].attn_q.data(), x, Q.data(), ...);
    }
}
```

## Memory Benefits

### Before (Copying Approach)
- Token embedding: 256 × 32 = 8,192 bytes copied to RAM
- Position embedding: 128 × 32 = 4,096 bytes copied to RAM
- Per block: ~16KB copied to RAM
- **Total RAM**: ~28KB for single-layer model

### After (Pointer Approach)
- Embedded weights pointer: 4 bytes
- Working buffers (temp_buffer1, temp_buffer2): ~12KB
- **Total RAM**: ~12KB for same model

**RAM savings**: ~16KB (57% reduction)

## Compilation Flags

Enable embedded weights mode in `platformio.ini`:

```ini
build_flags = 
    -DNANOLLM_USE_EMBEDDED_WEIGHTS
```

## Workflow

1. **Train model**: `python python/train.py --config ...`
2. **Export header**: `python python/export_weights_header.py --checkpoint model.pt --output esp32_m5stack/src/model_weights.h`
3. **Build firmware**: `cd esp32_m5stack && pio run`
4. **Flash**: `pio run --target upload`

The firmware binary contains the model weights in its flash partition, ready for zero-copy access.

## Backward Compatibility

The SPIFFS-based loading still works when `NANOLLM_USE_EMBEDDED_WEIGHTS` is not defined. All inference functions check `isUsingEmbeddedWeights()` and route to the appropriate implementation.

## Future Enhancements

- **XIP (Execute In Place)**: ESP32-S3 supports XIP from external flash, could eliminate `memcpy_P` for structure metadata
- **DMA transfers**: Use ESP32 DMA for copying weight chunks during computation
- **Compressed weights**: Apply run-length encoding or other compression to PROGMEM arrays
