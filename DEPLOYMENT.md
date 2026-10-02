# Deployment Guide

This guide covers deploying NanoLLM to the M5Stack Cardputer.

## Prerequisites

1. **Trained Model**: You need a trained model checkpoint
   ```bash
   cd python
   python train.py --data your_data.txt --output_dir ../checkpoints
   ```

2. **PlatformIO**: Install if not already installed
   ```bash
   pip install platformio
   ```

3. **M5Stack Cardputer**: Hardware ready with USB-C cable

## Quick Deployment

### Automated Script (Recommended)

The deployment script handles everything automatically:

```bash
cd esp32_m5stack
./deploy.sh
```

The script will:
- Ask you to choose between embedded weights or SPIFFS
- Generate the necessary files
- Build the firmware
- Guide you through uploading
- Open the serial monitor

### Quick Deploy (Non-Interactive)

For automated/CI deployments:

```bash
cd esp32_m5stack

# Embedded weights (recommended - faster, simpler)
./quick_deploy.sh embedded

# Or SPIFFS (for larger models or easier updates)
./quick_deploy.sh spiffs
```

## Deployment Methods

### Method 1: Embedded Weights (Recommended)

**Pros:**
- Faster loading (~0.1-0.5s vs 1-2s)
- Simpler deployment (single firmware file)
- More reliable (no file system)

**Cons:**
- Larger firmware size
- Requires rebuild to update model

**Steps:**
1. Generate embedded weights header:
   ```bash
   ./generate_embedded_weights.sh
   ```

2. Enable in `platformio.ini`:
   ```ini
   build_flags = 
       ...
       -DNANOLLM_USE_EMBEDDED_WEIGHTS
   ```

3. Build and upload:
   ```bash
   pio run --target upload
   ```

### Method 2: SPIFFS (File System)

**Pros:**
- Smaller firmware
- Easy to update model without rebuilding
- Can store multiple models

**Cons:**
- Slower loading
- Requires file system setup
- More complex deployment

**Steps:**
1. Prepare model files:
   ```bash
   ./upload_model.sh
   ```

2. Upload filesystem:
   ```bash
   pio run --target uploadfs
   ```

3. Upload firmware:
   ```bash
   pio run --target upload
   ```

## Entering Download Mode

Before uploading, the Cardputer must be in download mode:

1. **Set the switch on the top to OFF**
2. **Hold down the G0 button**
3. **Connect USB-C cable to your computer**
4. **Release the G0 button**

The device is now in download mode and ready for flashing.

## Verification

After deployment:

1. **Serial Monitor**: The deployment script opens the monitor automatically
   ```bash
   pio device monitor
   ```

2. **Expected Output**:
   - "Initializing SPIFFS..." (if using SPIFFS)
   - "Loading model..."
   - "Model loaded successfully!"
   - Model information (vocab size, dimensions, etc.)
   - Generated text

3. **Display**: The Cardputer screen should show:
   - Model loading status
   - Model information
   - Generated text

## Troubleshooting

### "Model files not found"
- **SPIFFS**: Ensure filesystem was uploaded (`pio run --target uploadfs`)
- **Embedded**: Check that `model_weights.h` exists and `NANOLLM_USE_EMBEDDED_WEIGHTS` is enabled

### "Failed to load model"
- Check model file sizes match expected values
- Verify config file matches model architecture
- Check serial output for specific error messages

### Upload fails
- Ensure device is in download mode (see above)
- Check USB cable connection
- Try different USB port
- Check PlatformIO detects the device: `pio device list`

### Firmware too large
- Reduce model size (d_model, n_layers)
- Use SPIFFS instead of embedded weights
- Adjust partition table in `platformio.ini`

### Build errors
- Ensure all dependencies are installed
- Check PlatformIO version compatibility
- Verify board selection is correct

## Updating the Model

### Embedded Weights:
1. Retrain model (if needed)
2. Regenerate header: `./generate_embedded_weights.sh`
3. Rebuild and upload: `pio run --target upload`

### SPIFFS:
1. Retrain model (if needed)
2. Export weights: `python export_weights.py ...`
3. Upload filesystem: `pio run --target uploadfs`
4. No firmware rebuild needed!

## Performance Tips

1. **Use embedded weights** for faster startup
2. **Optimize model size** to fit in available flash
3. **Monitor memory usage** during inference
4. **Adjust generation parameters** (max tokens, temperature)

## Advanced: Custom Partition Tables

For very large models, you may need a custom partition table:

1. Create `partitions.csv`:
   ```
   # Name,   Type, SubType, Offset,  Size, Flags
   nvs,      data, nvs,     0x9000,  0x6000,
   phy_init, data, phy,     0xf000,  0x1000,
   factory,  app,  factory, 0x10000, 0x200000,
   spiffs,   data, spiffs,  ,        0x100000,
   ```

2. Update `platformio.ini`:
   ```ini
   board_build.partitions = partitions.csv
   ```

## CI/CD Integration

For automated deployments:

```yaml
# Example GitHub Actions
- name: Deploy to Cardputer
  run: |
    cd esp32_m5stack
    ./quick_deploy.sh embedded
```

## Next Steps

- Customize the display output
- Add keyboard input for prompts
- Implement model selection menu
- Add configuration options

