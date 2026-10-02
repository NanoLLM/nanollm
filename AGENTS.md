# AGENTS.md

This file defines how coding agents should work in this repository.

## 1) Mission

Maintain and evolve NanoLLM as a compact end-to-end pipeline:

- Train in Python.
- Export weights and tokenizer artifacts.
- Run inference in desktop C++.
- Deploy on ESP32 M5Stack Cardputer.

Primary quality bar:

- Functional parity across Python, C++, and ESP32 inference paths.
- Stable serialization and tokenizer compatibility across all runtimes.
- Keep ESP32 memory and firmware constraints in mind for every change.

## 2) Repository Map

- `python/`: model, tokenizer, training, exports, `quantize.py`, `quantized_runtime.py`.
- `cpp/`: desktop inference and C++ tests.
- `esp32_m5stack/`: embedded inference, chat UI, PlatformIO deployment.
- `scripts/`: dataset helpers, MoE pipeline, sizing estimator, serial verification.
- `autoresearch/`: Cardputer-constrained overnight agentic training loop (see §13).
- `test_end_to_end.py`: pipeline-level verification.

Key scripts:

| Script | Purpose |
|--------|---------|
| `scripts/train_moe_pipeline.sh` | FineWeb pretrain → chat fine-tune → PROGMEM embed (default; pinned vocab=2048 tokenizer) |
| `scripts/package_cardputer_release.sh` | Public Cardputer foundation release (export + firmware + manifest) |
| `scripts/moe_tradeoff_estimator.py` | Flash/RAM sizing; `--cardputer-optimal`, `--cardputer-max-flash` |
| `scripts/verify_cardputer_serial.py` | Compare device serial output vs Python quantized runtime |
| `scripts/interactive_chat.py` | Qualitative REPL testing against checkpoints |
| `scripts/compare_quantized_runtime.py` | Python vs C++ int8 parity |

## 3) Critical Invariants (Do Not Break)

1. Weight tying:
   - In `python/model.py`, `lm_head.weight` is tied to `token_embedding.weight`.
   - Any embedding shape or logic change must preserve this relationship or intentionally migrate all consumers.

2. Tokenizer compatibility:
   - Python tokenizer behavior must remain compatible with:
     - `cpp/bpe_tokenizer.cpp`
     - `esp32_m5stack/src/bpe_tokenizer_esp32.cpp`
   - If normalization/token ID logic changes, update Python and both C++ implementations together.

3. Export artifact contract:
   - Exported outputs are consumed by C++ and ESP32 loaders.
   - Keep artifact names and schema stable unless intentionally versioning:
     - `model.bin`
     - `model_config.json`
     - `vocab.json`
     - `tokenizer_info.json`

4. Quantized binary order:
   - Loader and exporter must agree on exact serialization order.
   - MoE blocks use experimental NLMO v1 layout — see `docs/design/MOE_EXPORT_EXPERIMENTAL.md`.
   - If changing export format, update all readers and sanity checks together.

5. Embedded constraints:
   - **Working RAM** (~200 KiB budget) limits `d_model`, `max_seq_len`, and `vocab_size` during inference.
   - **Flash** (7 MiB app partition in `partitions_embedded.csv`) limits PROGMEM weight size.
   - Weights in `m5stack_cardputer_nopsram` are read from flash via `pgm_read_byte` — not copied to heap.
   - Re-export **both** `model_weights.h` and `vocab_weights.h` after training when vocab changes.

## 4) Cardputer MoE Deployment (Current Default)

Environment: `m5stack_cardputer_nopsram` in `esp32_m5stack/platformio.ini`

- `-DNANOLLM_USE_EMBEDDED_WEIGHTS` + `-DNANOLLM_USE_EMBEDDED_VOCAB`
- Partition: `partitions_embedded.csv` (~7 MiB factory app)

**Validated flash-backed middle-ground config** (512-token context, no-PSRAM):

```
VOCAB_SIZE=8192  D_MODEL=32  N_LAYERS=124  N_HEADS=4
D_FF=128  MOE_N_EXPERTS=4  MOE_TOP_K=1  MOE_SHARED_D_FF=64  BLOCK_SIZE=512
~5.6M params, ~5.5 MiB int8 weights, ~194 KiB working RAM (~96% flash partition)
```

Vocab is stored in flash (`vocab_weights.h` PROGMEM). At 512-token context, the
streamed LM-head argmax removes the full-vocab logits buffer, but
`3 * max_seq_len * d_model` working buffers still force a narrow model;
increasing context usually requires reducing `d_model`.

Size new models with:

```bash
python scripts/moe_tradeoff_estimator.py --cardputer-optimal --json
python scripts/moe_tradeoff_estimator.py --cardputer-max-flash   # max weight bytes
```

## 5) Preferred Workflow

1. Understand scope and affected runtime(s): Python only, desktop C++, ESP32, or cross-stack.
2. Make minimal, focused edits.
3. Validate as close to the changed layer as possible.
4. For format/schema changes, validate all downstream consumers.

## 6) Common Commands

From repository root:

Python setup:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Train (dense):

```bash
python python/train.py --data sample_data.txt --epochs 2 --batch_size 16
```

Train + deploy MoE (Cardputer):

```bash
./scripts/train_moe_pipeline.sh --export          # embeds weights + vocab by default
./scripts/train_moe_pipeline.sh --profile quick   # fast smoke (fineweb_tiny + sample_data)
cd esp32_m5stack && pio run -e m5stack_cardputer_nopsram -t upload
python scripts/verify_cardputer_serial.py --reset --checkpoint checkpoints/<run>/model_best.pt
```

Export:

```bash
python python/export_weights.py --checkpoint checkpoints/model_best.pt --output weights/model.bin --allow-moe-export
python python/export_weights_header.py --checkpoint checkpoints/model_best.pt --output esp32_m5stack/src/model_weights.h
python python/export_vocab_header.py --tokenizer checkpoints/<run>/tokenizer/tokenizer.json --output esp32_m5stack/src/vocab_weights.h
```

Desktop C++ build:

```bash
cmake -S cpp -B cpp/build
cmake --build cpp/build
./cpp/build/inference weights/model.bin "hello"
```

End-to-end verification:

```bash
python test_end_to_end.py
python scripts/compare_quantized_runtime.py
```

## 7) Edit Guidelines

- Preserve public interfaces unless migration work is explicit.
- Keep serialization changes atomic across producer and consumers.
- Favor deterministic behavior over hidden magic.
- Add concise comments only where logic is non-obvious.
- Avoid broad refactors when solving a scoped issue.

## 8) Testing Expectations

After relevant edits, run an appropriate subset:

- Python-only logic: run the affected script and a quick smoke check.
- C++ runtime changes: rebuild in `cpp/build` and run `test_inference` or `inference`.
- Export/tokenizer changes: run Python export, then desktop inference load.
- Cross-stack changes: run `python test_end_to_end.py` when feasible.
- ESP32 MoE changes: rebuild `m5stack_cardputer_nopsram`, flash, run `verify_cardputer_serial.py`.

Document what was run and what was not run if time or environment limits full validation.

## 9) Agent Behavior

- Be explicit about assumptions.
- Prefer evidence from repository files over guesses.
- If unexpected unrelated file changes appear, stop and ask before proceeding.
- Never use destructive git operations unless explicitly requested.

## 10) When Adding New Parameters or Artifacts

Propagate changes in lockstep through:

1. Python producer/export.
2. Desktop C++ loader/inference.
3. ESP32 loader/inference (`model_esp32.cpp`, `model_embedded.cpp`, `model_weights_types.h`).
4. Tests and docs.

If full propagation is not done in one change, gate incomplete paths with clear errors.

## 11) Cardputer sizing constraints

- **Working RAM (~200 KiB)** caps `max_seq_len` and `vocab_size` before flash does.
- At **128 context, d_model=44**: **vocab max = 1024** (logits buffer uses remaining headroom).
- Larger vocab requires smaller `d_model` (e.g. vocab=2048 needs d≤40). See `docs/design/MOE_CARDPUTER_RESEARCH.md` § "Vocabulary vs context tradeoff".
- Run `python scripts/moe_tradeoff_estimator.py --cardputer-optimal` before changing defaults.

## 12) Training performance (workstation)

This repo targets tiny models (~4M params); GPU is usually under-utilized with default batch sizes.

Recommended on RTX 3090/40xx (CUDA + bf16):

- `PRETRAIN_BATCH_SIZE=256`, `CHAT_BATCH_SIZE=128`, `NUM_WORKERS=8`, `AMP=bf16`, `TF32=1`
- Set `CUDA_VISIBLE_DEVICES` if one GPU is busy with other jobs.
- Scale LR linearly when increasing batch size (not done automatically).
- `torch.compile` is **not** recommended for these small models (overhead dominates).

Pipeline env vars: `PRETRAIN_BATCH_SIZE`, `CHAT_BATCH_SIZE`, `NUM_WORKERS`, `AMP`, `TF32`, `CUDA_DEVICE`.

## 12.1) Instruction-following fine-tuning

NanoLLM supports a distinct **instruction fine-tuning** workflow (separate from chat fine-tuning) that uses curated instruction datasets with category-balanced seed injection.

### Data paths

| Variable | Default | Description |
|----------|---------|-------------|
| `INSTRUCT_RAW` | `data/instruct/instruct_raw.txt` | Raw downloaded instruction corpus |
| `INSTRUCT_TRAIN` | `data/instruct/instruct_train.txt` | Processed training split |
| `INSTRUCT_VAL` | `data/instruct/instruct_val.txt` | Processed validation split |
| `INSTRUCT_SEED` | `scripts/instruction_chat_seed.txt` | Seed examples (40 examples, 10 categories) |
| `INSTRUCT_SEED_REPEATS` | `3` | How many times to repeat each seed in training data |

### Pipeline workflow

1. **Download datasets:**
   ```bash
   python scripts/download_instruction_datasets.py --output-dir data/instruct
   ```
   Sources: alpaca, openorca, wizardlm, instruct_wiki_4, ultrachat_200k.

2. **Prepare (filter + category-balance + split):**
   ```bash
   python scripts/prepare_instruction_data.py \
       --input data/instruct/instruct_raw.txt \
       --seed-data scripts/instruction_chat_seed.txt \
       --train-output data/instruct/instruct_train.txt \
       --val-output data/instruct/instruct_val.txt \
       --seed-repeats 3
   ```

3. **Fine-tune via pipeline:**
   ```bash
   ./scripts/train_moe_pipeline.sh --instruction-finetune --export
   ```
   The `--instruction-finetune` flag switches the fine-tune stage to use `data/instruct/` instead of `data/chat/` and adds `--instruct_benchmark` to the training arguments.

4. **Benchmark instruction-following (standalone):**
   ```bash
   python scripts/benchmark_instruction_following.py \
       --checkpoint checkpoints/<run>/model_best.pt \
       --output instruct_benchmark_best.json
   ```

### Category system

The 10 instruction categories tracked by `prepare_instruction_data.py` and `benchmark_instruction_following.py`:

| Category | Patterns | Examples |
|----------|----------|----------|
| `math` | what is, calculate, solve, compute, evaluate | "What is 7 times 8?" |
| `reasoning` | why, explain, logic, therefore | "Why is the sky blue?" |
| `creative` | poem, haiku, story, joke, song | "Write a haiku about snow." |
| `coding` | function, implement, code, debug | "Write a function to reverse a string." |
| `translation` | translate, french, spanish, german | "Translate 'hello' to French." |
| `explanation` | how does, what is the difference | "How does a car engine work?" |
| `qa_factual` | who, when, where, capital | "What is the capital of France?" |
| `list` | list, enumerate, name 5 | "List 5 programming languages." |
| `summarization` | summarize, summary, short version | "Summarize the plot of Hamlet." |
| `general` | (fallback) | anything not matching other categories |

### Chat vs instruction fine-tuning

| Feature | Chat fine-tuning | Instruction fine-tuning |
|---------|-----------------|------------------------|
| Data source | `data/chat/*.txt` | `data/instruct/*.txt` |
| Preparation | `prepare_chat_data.py` | `prepare_instruction_data.py` |
| Seed file | `scripts/chat_capability_seed.txt` | `scripts/instruction_chat_seed.txt` |
| Benchmark | `chat_eval.py` (fixed prompts) | `benchmark_instruction_following.py` (category-level) |
| Pipeline flag | (default) | `--instruction-finetune` |

## 13) Autoresearch (agentic overnight experiments)

For autonomous training experiments under Cardputer flash/RAM constraints, point the agent at:

- [`autoresearch/program.md`](autoresearch/program.md) — agent instructions (human-edited)
- [`autoresearch/train.py`](autoresearch/train.py) — **only** file the agent may modify
- [`autoresearch/prepare.py`](autoresearch/prepare.py) — fixed data/eval/time budget/Cardputer gate

Production training and deploy remain `python/train.py` and `scripts/train_moe_pipeline.sh`. Autoresearch winners are Python research candidates; porting to export/ESP32 is a separate step.
