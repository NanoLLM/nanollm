# M5Stack Cardputer NanoLLM Project

This project deploys the NanoLLM model on the M5Stack Cardputer, displaying generated text on the device's screen.

## Hardware

- **M5Stack Cardputer**
  - ESP32-S3 processor
  - 1.14" TFT Display (240x135 pixels)
  - QWERTY keyboard
  - USB-C connector

## Prerequisites

1. **PlatformIO** (recommended) or Arduino IDE
2. **M5Stack Cardputer** hardware
3. **Trained model files** (`model.bin` and `model_config.json`)

## Setup

### 1. Install PlatformIO

```bash
# Install PlatformIO Core
pip install platformio

# Or use PlatformIO IDE
```

### 2. Prepare Model Files

First, train and export your model:

```bash
# From project root
cd python
python train.py --data your_data.txt --output_dir ../checkpoints
python export_weights.py --checkpoint ../checkpoints/model_best.pt --output ../weights/model.bin
```

### 3. Upload Model to SPIFFS

The model files need to be uploaded to the ESP32's SPIFFS file system.

#### Using PlatformIO:

```bash
cd esp32_m5stack

# Create data directory
mkdir -p data

# Copy model files
cp ../weights/model.bin data/
cp ../weights/model_config.json data/

# Upload filesystem
pio run --target uploadfs
```

#### Using Arduino IDE:

1. Install [ESP32FS plugin](https://github.com/me-no-dev/arduino-esp32fs-plugin)
2. Create a `data` folder in your sketch directory
3. Copy `model.bin` and `model_config.json` to the `data` folder
4. Tools → ESP32 Sketch Data Upload

### 4. Stage Artifacts to SD Card (Easy)

If your Cardputer SD card is mounted on your development machine, you can export
and copy artifacts in one step:

```bash
cd esp32_m5stack
./prepare_sdcard_model.sh --mount /media/$USER/CARDPUTER --mode dense
```

For MoE artifacts:

```bash
cd esp32_m5stack
./prepare_sdcard_model.sh --mount /media/$USER/CARDPUTER --mode moe --basename model_moe
```

By default, files are copied into `/nanollm` on the SD card.

Note:
- Current firmware loads model files from SPIFFS or embedded headers.
- SD card staging is provided so artifacts are easy to move/manage now and is ready
   for upcoming SD-backed loading flows.

## Quick Deployment

The easiest way to deploy is using the deployment script:

```bash
cd esp32_m5stack
./deploy.sh
```

This interactive script will:
1. Generate embedded weights (or prepare SPIFFS files)
2. Build the firmware
3. Upload to the Cardputer
4. Open serial monitor

For non-interactive deployment:

```bash
# Embedded weights (recommended)
./quick_deploy.sh embedded

# Or SPIFFS
./quick_deploy.sh spiffs
```

## Serial verification (PyTorch parity)

On boot the firmware prints `@READY` and accepts newline-terminated commands over USB serial (115200 baud):

| Command | Response |
|---------|----------|
| `PING` | `@PONG` |
| `INFO` | `@INFO\|<json>` model config + heap |
| `ENCODE\|<text>` | `@ENCODE\|<json>` prompt token IDs |
| `GENERATE\|<prompt>\|<max_new_tokens>` | `@GENERATE\|<json>` generated tokens + next token |

Compare against local PyTorch:

```bash
python scripts/verify_cardputer_serial.py \
  --checkpoint checkpoints/moe_small_chat_20260630/model_best.pt \
  --port /dev/ttyACM0 \
  --reset \
  --max-new-tokens 1
```

Use `--encode-only` to verify tokenizer parity only. After retraining, re-export the embedded vocab before rebuilding firmware:

```bash
python python/export_vocab_header.py \
  --tokenizer checkpoints/<run>/tokenizer/tokenizer.json \
  --output esp32_m5stack/src/vocab_weights.h
```

The device runs **int8-quantized** weights from SPIFFS `model.bin`. Compare against the same
quantized runtime in Python (not the FP32 checkpoint):

```bash
python scripts/verify_cardputer_serial.py --reset --max-new-tokens 1
python scripts/compare_quantized_runtime.py
```

`python/quantized_runtime.py` loads `model.bin` and mirrors C++/ESP32 int8 math.
Shared `python/quantize.py` defines the export quantization formula.

## Manual Building and Uploading

### Using PlatformIO:

```bash
cd esp32_m5stack
pio run --target upload
```

### Using Arduino IDE:

1. Install M5Stack board support:
   - File → Preferences → Additional Board Manager URLs
   - Add: `https://m5stack.oss-cn-shenzhen.aliyuncs.com/resource/arduino/package_m5stack_index.json`
   - Tools → Board → Boards Manager → Search "M5Stack" → Install

2. Install libraries:
   - Sketch → Include Library → Manage Libraries
   - Search and install:
     - M5Cardputer
     - ArduinoJson
     - SPIFFS

3. Select board:
   - Tools → Board → M5Stack Arduino → M5Stack Cardputer

4. Upload:
   - Connect Cardputer via USB-C
   - Set switch to OFF, hold G0, connect USB, release G0
   - Tools → Port → Select your port
   - Click Upload

## Usage

1. Power on the Cardputer
2. The device will automatically:
   - Initialize SPIFFS
   - Load the model
   - Display model information
   - Generate text from prompt "Hello"

3. Press any key on the keyboard to regenerate text

## Display

The generated text is displayed on the Cardputer's 1.14" TFT screen with:
- Word wrapping
- Scrollable text
- Status messages

## Memory Management

Two deployment modes:

| Mode | Env | Weights | Partition |
|------|-----|---------|-----------|
| **PROGMEM (default)** | `m5stack_cardputer_nopsram` | Flash-backed, zero RAM copy | `partitions_embedded.csv` (~7 MiB app) |
| SPIFFS | `m5stack_cardputer` | Loaded into heap | `default.csv` |

**128-token MoE** (current default, `scripts/moe_tradeoff_estimator.py --cardputer-optimal`):

- 44 layers, d_model=44, 4 experts, **vocab=1024**
- ~4.2 MiB int8 weights + ~115 KiB vocab PROGMEM (~67% of app partition)
- ~200 KiB working RAM (logits buffer limits vocab above 1024)

Context length is capped at **128 tokens** in firmware (`NANOLLM_NO_PSRAM`). The `S²`
attention buffer dominates RAM; wider/deeper models require shorter context.

Working RAM (not flash) is the main limiter for `d_model` and `vocab_size`. Size models with:

```bash
python scripts/moe_tradeoff_estimator.py --cardputer-optimal
```

If you encounter memory issues:
- Reduce `d_model` or `n_layers` (use `--profile cardputer-safe`)
- Reduce `max_seq_len` (`BLOCK_SIZE`)
- Reduce vocabulary size

## Troubleshooting

### Model files not found
- Ensure SPIFFS is properly formatted
- Verify files are uploaded to SPIFFS
- Check file names match exactly

### Out of memory
- Reduce model dimensions
- Check available heap: `ESP.getFreeHeap()`
- Use smaller sequence length

### Display issues
- Check display initialization
- Verify rotation settings
- Check text size and wrapping

### Build errors
- Ensure all libraries are installed
- Check PlatformIO/Arduino version compatibility
- Verify board selection is correct

## Customization

### Change prompt

Edit `main.cpp`:

```cpp
String prompt = "Your custom prompt here";
```

### Adjust generation parameters

```cpp
std::vector<int> generated = model.generate(prompt_tokens, 50, 1.0f);
//                                                          ^^^  ^^^
//                                                    max_tokens  temperature
```

### Modify display

The display code is in the `printToDisplay()` and `generateAndDisplay()` functions.

## Performance

Expected performance on ESP32-S3:
- Model loading: 1-2 seconds
- Text generation: ~0.5-1 second per token
- Memory usage: ~200KB during inference

## Embedded Weights (Firmware Partition)

PROGMEM embedding is the **recommended** path for no-PSRAM Cardputer builds.

Environment `m5stack_cardputer_nopsram` enables:
- `-DNANOLLM_USE_EMBEDDED_WEIGHTS`
- `-DNANOLLM_USE_EMBEDDED_VOCAB`
- `partitions_embedded.csv` (7 MiB factory app)

After training, export both headers (done automatically by `train_moe_pipeline.sh`):

```bash
python python/export_weights_header.py \
  --checkpoint checkpoints/<run>/model_best.pt \
  --output esp32_m5stack/src/model_weights.h

python python/export_vocab_header.py \
  --tokenizer checkpoints/<run>/tokenizer/tokenizer.json \
  --output esp32_m5stack/src/vocab_weights.h

cd esp32_m5stack && pio run -e m5stack_cardputer_nopsram -t upload
```

MoE weights use shared structs in `src/model_weights_types.h`. See [README_EMBEDDED.md](README_EMBEDDED.md) for details.

Quick start (legacy dense path):

```bash
./generate_embedded_weights.sh
pio run -e m5stack_cardputer_nopsram -t upload
```

## Next Steps

- Add keyboard input for custom prompts
- Implement text scrolling for long outputs
- Add model selection menu
- Implement temperature and top-k sampling controls

