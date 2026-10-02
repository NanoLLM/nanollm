# Embedded Weights and Vocab

The M5Stack Cardputer project supports embedding both model weights and vocabulary directly into the firmware, eliminating the need for SPIFFS.

## Benefits

- **No SPIFFS required**: Everything is in firmware
- **Faster startup**: No file system loading
- **More reliable**: No file system corruption issues
- **Single firmware file**: Easier deployment

## Trade-offs

- **Larger firmware**: ~6MB+ for weights + ~30KB for vocab
- **Slower compilation**: Large header files take time to compile
- **Requires recompilation**: Can't update weights without rebuilding

## Usage

### 1. Generate Embedded Headers

Run the generation script:

```bash
cd esp32_m5stack
./generate_embedded_weights.sh
```

This creates:
- `src/model_weights.h` - Model weights (int8 quantized)
- `src/vocab_weights.h` - BPE vocabulary

### 2. Enable in platformio.ini

Uncomment the flags:

```ini
build_flags = 
    -DNANOLLM_USE_EMBEDDED_WEIGHTS
    -DNANOLLM_USE_EMBEDDED_VOCAB
```

### 3. Build and Upload

```bash
pio run --target upload
```

No SPIFFS upload needed!

## Automatic Deployment

The `deploy.sh` script automatically:
1. Generates both headers
2. Enables the flags in platformio.ini
3. Builds and uploads firmware

```bash
./deploy.sh
# Select option 1 (Embedded weights)
```

## File Sizes

Typical sizes:
- **Weights header**: ~4-6MB (int8 quantized)
- **Vocab header**: ~20-30KB (500 tokens)
- **Total firmware**: ~8-10MB

## Memory Layout

With embedded weights:
- **Flash**: ~10MB (firmware + weights + vocab)
- **RAM**: ~200KB (inference only)
- **No SPIFFS needed**: Can disable SPIFFS partition

## Fallback Behavior

If embedded vocab fails to load:
- Automatically falls back to SPIFFS (`/vocab.json`)
- If SPIFFS also fails, uses byte-level encoding

## Troubleshooting

### "Out of memory" during compilation
- Reduce vocab_size (e.g., 300 instead of 500)
- Use SPIFFS for vocab, embedded for weights only

### "Firmware too large"
- Check partition table in platformio.ini
- May need custom partition table with larger app partition

### "Vocab not loading"
- Check that `vocab_weights.h` exists
- Verify `NANOLLM_USE_EMBEDDED_VOCAB` is uncommented
- Check serial monitor for error messages

## Hybrid Approach

You can mix approaches:
- **Embedded weights** + **SPIFFS vocab**: Best of both worlds
- **SPIFFS weights** + **Embedded vocab**: Less common

Just enable the flags you want in platformio.ini.
