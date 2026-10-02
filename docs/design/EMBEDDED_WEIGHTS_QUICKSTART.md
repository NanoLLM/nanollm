# Quick Start: Using Embedded Weights

## Generate Embedded Weights Header

After training a model, export it to a C++ header file:

```bash
python python/export_weights_header.py \
    --checkpoint checkpoints/cardputer_run_20251113_124604/model_best.pt \
    --output esp32_m5stack/src/model_weights.h \
    --namespace nanollm
```

This creates header files with:
- All model weights as `const int8_t PROGMEM` arrays (dense or MoE)
- Pre-initialized `EmbeddedWeights` structure (`model_weights_types.h` for MoE)
- Embedded BPE vocab in `vocab_weights.h` (required when `NANOLLM_USE_EMBEDDED_VOCAB` is set)

Or use the full pipeline (exports both headers automatically):

```bash
./scripts/train_moe_pipeline.sh --export
```

## Build and Flash

### Option 1: Quick Deploy Script

```bash
cd esp32_m5stack
./quick_deploy.sh embedded
```

This script:
1. Generates the embedded weights header
2. Enables `NANOLLM_USE_EMBEDDED_WEIGHTS` flag
3. Builds firmware with PlatformIO
4. Uploads to ESP32

### Option 2: Manual Build

```bash
cd esp32_m5stack
pio run -e m5stack_cardputer_nopsram -t upload
```

Default env is `m5stack_cardputer_nopsram` (7 MiB app partition, embedded weights + vocab).

## Code Usage

In your ESP32 application:

```cpp
#include "model_esp32.h"

void setup() {
    Serial.begin(115200);
    
    NanoLLM model;
    
    // Load embedded weights (zero-copy!)
    if (!model.loadFromEmbedded()) {
        Serial.println("Failed to load embedded weights");
        return;
    }
    
    Serial.printf("Model loaded: %d params, %d layers\n",
                  model.getConfig().d_model,
                  model.getConfig().n_layers);
    
    // Generate tokens
    std::vector<int> prompt = {72, 101, 108, 108, 111};  // "Hello"
    auto output = model.generate(prompt, 20);
    
    for (int token : output) {
        Serial.printf("%d ", token);
    }
}
```

## Switching Between Embedded and SPIFFS

The code automatically detects which mode to use:

### Embedded Mode
- Define `NANOLLM_USE_EMBEDDED_WEIGHTS` 
- Include generated `model_weights.h`
- Call `model.loadFromEmbedded()`
- Weights stored in firmware flash partition
- Zero RAM overhead for weight storage

### SPIFFS Mode
- Don't define `NANOLLM_USE_EMBEDDED_WEIGHTS`
- Upload `model.bin` and `model_config.json` to SPIFFS
- Call `model.load("/model.bin", "/model_config.json")`
- Weights loaded into RAM from filesystem
- Allows runtime model updates

## Memory Comparison

For a typical small model (d_model=32, n_layers=1):

| Mode     | RAM Usage | Flash Usage | Loading Time |
|----------|-----------|-------------|--------------|
| Embedded | ~12 KB    | ~28 KB      | <1 ms        |
| SPIFFS   | ~28 KB    | ~28 KB      | ~100 ms      |

## Troubleshooting

### Build Error: "model_weights.h not found"
- Run `export_weights_header.py` first to generate the header
- Ensure the output path matches the include path

### Runtime Error: "Failed to get embedded weights pointer"
- Verify `NANOLLM_USE_EMBEDDED_WEIGHTS` is defined
- Check that `model_weights.h` was included in the build
- Ensure the header file is valid (not empty/truncated)

### Flash Overflow
- Your model is too large for the flash partition
- Options:
  1. Use a smaller model (reduce d_model, n_layers)
  2. Switch to SPIFFS mode
  3. Enable flash compression in PlatformIO

### Inference Results Differ from Python
- Verify the checkpoint file matches the exported header
- Check that quantization is enabled consistently
- Temperature and sampling settings may differ

## Best Practices

1. **Version Control**: Don't commit `model_weights.h` (add to `.gitignore`)
2. **CI/CD**: Generate headers as part of your build pipeline
3. **Testing**: Validate inference on desktop before flashing to ESP32
4. **Size Monitoring**: Track firmware size to avoid flash overflow
