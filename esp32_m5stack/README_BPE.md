# BPE Tokenizer on ESP32

The M5Stack Cardputer project now supports BPE (Byte Pair Encoding) tokenization for better text encoding.

## How It Works

1. **Training**: Python trains a BPE tokenizer on your data (vocab_size=500)
2. **Export**: Tokenizer vocabulary is exported to `vocab.json`
3. **Deployment**: `vocab.json` is uploaded to SPIFFS along with model files
4. **Runtime**: ESP32 loads the tokenizer and uses it for encoding/decoding

## Files Required

For BPE tokenization, you need:
- `model.bin` - Model weights
- `model_config.json` - Model configuration
- `vocab.json` - BPE vocabulary (exported automatically)

## Automatic Export

When you export weights using `export_weights.py`, the vocab.json is automatically created:

```bash
python python/export_weights.py --checkpoint checkpoints/model_best.pt --output weights/model.bin
```

This creates:
- `weights/model.bin`
- `weights/model_config.json`
- `weights/vocab.json` ← BPE vocabulary

## Deployment

The deployment scripts automatically handle vocab.json:

```bash
cd esp32_m5stack
./deploy.sh
# or
./quick_deploy.sh spiffs
```

The vocab.json file will be:
- Copied to `data/` directory
- Uploaded to SPIFFS
- Loaded automatically at runtime

## Memory Usage

BPE tokenizer on ESP32:
- **Vocab storage**: ~20-30KB (for vocab_size=500)
- **Runtime memory**: Minimal (just lookup tables)
- **Total overhead**: ~30KB

## Fallback Behavior

If `vocab.json` is not found:
- System falls back to byte-level encoding
- Model still works, but with less efficient encoding
- Display shows "BPE: No" status

## Verification

After deployment, check the serial monitor:
- "Loading tokenizer..." message
- "Tokenizer: 500 tokens" (if loaded)
- "BPE: Yes" on model info screen

## Troubleshooting

### "Tokenizer not found"
- Ensure `vocab.json` exists in `weights/` directory
- Check that deployment script copied it to `data/`
- Verify SPIFFS upload completed successfully

### "JSON parse error"
- Vocab file might be corrupted
- Check file size (should be ~20-30KB)
- Try re-exporting: `python python/export_tokenizer.py --tokenizer ...`

### Out of memory
- Reduce vocab_size (e.g., 300 instead of 500)
- Increase ArduinoJson buffer size in `bpe_tokenizer_esp32.cpp`
- Use embedded weights instead of SPIFFS

## Performance

BPE tokenization provides:
- **Better compression**: Fewer tokens per text
- **Better quality**: Handles OOV words better
- **Slight overhead**: ~30KB memory + parsing time

For a 6MB model, the 30KB overhead is negligible and the benefits are significant.

