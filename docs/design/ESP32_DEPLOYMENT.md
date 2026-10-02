# ESP32 Deployment Guide

This guide explains how to deploy the NanoLLM model on ESP32 devices, specifically the M5Stack Cardputer.

## Hardware Constraints

- **Flash Storage**: 4MB (or external storage)
- **SRAM**: 520KB
- **Model Size**: ~6MB (firmware + weights)
- **Inference Memory**: ~200KB RAM

## Memory Management Strategy

### 1. Weight Storage
- Store quantized weights (int8) in flash memory
- Use SPIFFS or LittleFS file system
- Or embed weights directly in program memory (if small enough)

### 2. Inference Memory
- Reuse buffers for activations
- Process in chunks if sequence length is too long
- Use fixed-size arrays instead of dynamic allocation

### 3. Optimizations
- Fixed-point arithmetic where possible
- Reduce precision for intermediate calculations
- Cache frequently used values

## ESP32-Specific Modifications

### 1. File System Setup

```cpp
#include <SPIFFS.h>

void setup() {
    if (!SPIFFS.begin(true)) {
        Serial.println("SPIFFS Mount Failed");
        return;
    }
}
```

### 2. Loading Weights from Flash

Modify `model.cpp` to load from SPIFFS:

```cpp
bool NanoLLM::load(const std::string& weights_path, const std::string& config_path) {
    // Open file from SPIFFS
    File weights_file = SPIFFS.open(weights_path.c_str(), "r");
    if (!weights_file) {
        Serial.println("Failed to open weights file");
        return false;
    }
    
    // Read weights...
    // (similar to existing implementation)
}
```

### 3. Memory-Efficient Matrix Operations

For ESP32, consider using:
- Fixed-size buffers
- In-place operations where possible
- Reduced precision (int16 instead of float32 for some operations)

### 4. Power Management

```cpp
// Reduce CPU frequency during inference
setCpuFrequencyMhz(80);  // Instead of 240MHz

// After inference
setCpuFrequencyMhz(240);
```

## PlatformIO Configuration

Create `platformio.ini`:

```ini
[env:esp32dev]
platform = espressif32
board = esp32dev
framework = arduino

monitor_speed = 115200

build_flags = 
    -O3
    -DARDUINO_ESP32_DEV
    -DBOARD_HAS_PSRAM
    -mfix-esp32-psram-cache-issue

lib_deps = 
    SPIFFS
```

## Arduino Framework Integration

1. Copy `model.h` and `model.cpp` to your Arduino project
2. Modify file I/O to use Arduino's File API
3. Adjust memory allocation for ESP32 constraints
4. Use `PROGMEM` for constant data

## Example ESP32 Sketch

```cpp
#include <Arduino.h>
#include <SPIFFS.h>
#include "model.h"

NanoLLM model;

void setup() {
    Serial.begin(115200);
    
    // Initialize SPIFFS
    if (!SPIFFS.begin(true)) {
        Serial.println("SPIFFS failed");
        return;
    }
    
    // Load model
    if (!model.load("/model.bin", "/model_config.json")) {
        Serial.println("Model load failed");
        return;
    }
    
    Serial.println("Model loaded!");
}

void loop() {
    // Generate text
    std::vector<int> prompt = {72, 101, 108, 108, 111};  // "Hello"
    std::vector<int> generated = model.generate(prompt, 20, 1.0f);
    
    // Print result
    for (int token : generated) {
        Serial.print((char)token);
    }
    Serial.println();
    
    delay(5000);
}
```

## Memory Profiling

Monitor memory usage:

```cpp
void printMemoryInfo() {
    Serial.printf("Free heap: %d bytes\n", ESP.getFreeHeap());
    Serial.printf("Largest free block: %d bytes\n", ESP.getMaxAllocHeap());
    Serial.printf("PSRAM: %d bytes\n", ESP.getFreePsram());
}
```

## Troubleshooting

### Out of Memory Errors
- Reduce `max_seq_len` in model config
- Reduce batch size to 1
- Use smaller model dimensions
- Process sequences in smaller chunks

### Slow Inference
- Reduce model size (fewer layers)
- Use lower precision
- Optimize matrix operations
- Consider using ESP32-S3 (more RAM)

### Flash Storage Issues
- Use external SD card for weights
- Compress weights further
- Use external flash chip

## Performance Targets

- **Inference Time**: < 1 second per token (on ESP32)
- **Memory Usage**: < 200KB during inference
- **Model Size**: < 6MB total

## Next Steps

1. Test model loading on ESP32
2. Profile memory usage
3. Optimize bottlenecks
4. Add streaming inference for longer sequences
5. Implement power-saving modes

