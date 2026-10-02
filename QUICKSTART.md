# Quick Start Guide

## Prerequisites

- Python 3.8+ with PyTorch (`pip install -r requirements.txt`)
- CUDA-capable GPU for real training (CPU works for the `quick` smoke profile)
- CMake 3.10+ and a C++17 compiler for desktop C++ inference
- PlatformIO (`pio`) for ESP32 Cardputer firmware
- `huggingface_hub` for dataset downloads (in `requirements.txt`)

## The fast path: interactive launcher

```bash
pip install -r requirements.txt

python scripts/nanollm_tui.py
```

The TUI (curses, stdlib-only — no extra dependencies) walks the full lifecycle:
**datasets → pretrain → SFT → post-train → firmware → verify**, with a target
(`pc_gpu` or `cardputer`), a profile, and per-field model/training overrides.
It sizes the model against the target's flash/RAM budget before you commit to
a run. Useful flags:

```bash
python scripts/nanollm_tui.py --dry-run                        # print the exact commands
python scripts/nanollm_tui.py --target cardputer --profile cardputer-mqa-ctx224
python scripts/nanollm_tui.py --run datasets --yes             # headless, one stage at a time
```

Everything below can also be done by hand.

## Training a model

1. **Prepare your training data** (plain text), or let the pipeline download
   FineWeb + chat data automatically on a fresh checkout:

   ```bash
   python scripts/ensure_finetune_data.py --report   # what's present + regen commands
   ```

2. **Train the model** (MoE pipeline — the real training entry point):

   ```bash
   ./scripts/train_moe_pipeline.sh --profile quick       # smoke model, minutes
   ./scripts/train_moe_pipeline.sh --export              # full: pretrain + chat SFT + PROGMEM embed
   ```

   Or train a minimal dense model directly:

   ```bash
   cd python
   python train.py \
       --data ../sample_data.txt \
       --output_dir ../checkpoints \
       --epochs 2 \
       --batch_size 16 \
       --vocab_size 500 \
       --d_model 64 \
       --n_layers 1 \
       --n_heads 2 \
       --d_ff 128 \
       --block_size 128
   ```

3. **Export weights for C++** (the pipeline does this with `--export`):

   ```bash
   python python/export_weights.py \
       --checkpoint checkpoints/model_best.pt \
       --output weights/model.bin \
       --allow-moe-export
   ```

## Running Python inference

```bash
# Interactive REPL chat against a checkpoint (chat or raw mode)
python scripts/interactive_chat.py --checkpoint checkpoints/model_best.pt

# Int8-quantized parity runtime (used by the C++/ESP32 parity tests)
python scripts/compare_quantized_runtime.py
```

## Building C++ inference

```bash
cmake -S cpp -B cpp/build
cmake --build cpp/build
```

## Running C++ inference

```bash
./cpp/build/inference weights/model.bin weights/model_config.json "Hello" 50
```

## Cardputer firmware

```bash
cd esp32_m5stack
pio run -e m5stack_cardputer_nopsram -t upload
python scripts/verify_cardputer_serial.py --reset \
    --checkpoint checkpoints/model_best.pt
```

## Model size guidelines

Size any candidate before training — RAM (~200 KiB working) is the binding
constraint, not flash:

```bash
python scripts/moe_tradeoff_estimator.py --cardputer-optimal --json
python scripts/moe_tradeoff_estimator.py --cardputer-max-flash
```

Validated flash-backed Cardputer config (the default `cardputer-mqa-ctx224`
profile): vocab 2048 · d_model 80 · 18 layers · 4 heads (MQA) · 4 experts
top-1 + shared(160) · ctx 224 → 4,661,760 params, ~4.45 MiB int8 PROGMEM,
~191 KiB working RAM. Dense fallback: `--dense-model`.

## Memory usage (ESP32 inference)

- **Weights**: int8, read from flash via `pgm_read_byte` — never copied to heap.
- **Activations**: float32, ~`4 × (3·S·D + 3·D + S + V)` bytes; the streamed
  LM-head argmax removes the full-vocab logits buffer.
- Total working RAM for the validated config: ~191 KiB of a ~200 KiB budget.

## Troubleshooting

### Out of Memory during training
- Reduce `batch_size` or `block_size`; scale LR linearly when you change batch.
- GPU tips for tiny models: `PRETRAIN_BATCH_SIZE=256`, `CHAT_BATCH_SIZE=128`,
  `NUM_WORKERS=8`, `AMP=bf16`, `TF32=1`. `torch.compile` is not worth it here.

### SFT fails with "smaller than the configured block size"
- Your val split is empty/too small. Use a chat file with many conversation
  groups (the committed seeds), or let `python/train.py` fall back to an
  internal train split (it now warns instead of crashing).

### Model too large for the Cardputer
- Reduce `d_model`, `n_layers`, or `max_seq_len`; re-check with the estimator.
- Wider vocab requires a smaller `d_model` (RAM, not flash, is the cap).

### C++ build errors
- Ensure C++17 support; verify `CMakeLists.txt` paths in `cpp/`.

## Next steps

- See [DEPLOYMENT.md](DEPLOYMENT.md) for ESP32-specific instructions.
- See [README.md](README.md) for architecture, recipes, and parity tests.
- See [scripts/README.md](scripts/README.md) for the full script catalog.
