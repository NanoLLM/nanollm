# Scripts Reference

Scripts to download and prepare open datasets, train, size, evaluate, and deploy
NanoLLM models. This page covers **all** script categories; the dataset sections at
the bottom go into detail on the download helpers.

## Catalog

### Training & pipelines

| Script | Purpose |
|---|---|
| `nanollm_tui.py` | **Interactive launcher (curses, stdlib-only)**: datasets → pretrain → SFT → post-train → firmware → verify, for `pc_gpu` or `cardputer` targets. `--dry-run` / `--run <stage> --yes` for headless use |
| `train_moe_pipeline.sh` | **Default full pipeline**: FineWeb pretrain → chat (or instruction) fine-tune → export → PROGMEM embed. Flags: `--export`, `--profile <name>` (see list at the bottom), `--instruction-finetune`, `--pretrain-only` |
| `train_clean_transfer_sft.sh` | Clean-transfer SFT recipe (frozen held-out suite) |
| `train_science_boost_sft.sh` | Science-boost SFT recipe (pinned release fine-tune) |
| `train_embed_deploy_cardputer.sh` | Train + export + PROGMEM embed for the Cardputer |

### Data preparation

| Script | Purpose |
|---|---|
| `download_datasets.sh` | Interactive download of all datasets |
| `download_fineweb.py` | FineWeb pretrain corpus (HF) |
| `download_chat_datasets.py` | ShareGPT / Alpaca / WizardLM / OpenOrca |
| `download_instruction_datasets.py` | alpaca / openorca / wizardlm / instruct_wiki_4 / ultrachat |
| `split_corpus_train_val.py` | Deterministic doc-hash train/val split |
| `prepare_chat_data.py` | Normalize + seed-inject chat data |
| `prepare_instruction_data.py` | Filter, category-balance (10 categories), split instruction data |
| `prepare_scaled_training_data.py` | Build scaled pretraining corpus (train/val + tokenizer fit) |
| `filter_chat_data.py`, `normalize_chat_data.py`, `audit_chat_overlap.py` | Chat data hygiene |
| `distill_short_chat.py` | Create concise teacher-distilled chat data |
| `ensure_finetune_data.py` | Fresh-checkout data bootstrap: `--report` (status + regen commands), `--tiny`, `--3x` |
| `ensure_canonical_tokenizer.py` | Enforce the pinned Cardputer vocab=2048 tokenizer (falls back to the committed release copy) |

### Sizing & deployment

| Script | Purpose |
|---|---|
| `moe_tradeoff_estimator.py` | Flash/RAM sizing; `--cardputer-optimal`, `--cardputer-max-flash` |
| `package_cardputer_release.sh` | Public Cardputer release bundle (export + firmware + manifest) |
| `mqa_deploy_chain.sh` | QA-model deploy chain |

### Evaluation & verification

| Script | Purpose |
|---|---|
| `interactive_chat.py` | REPL chat against a checkpoint (qualitative) |
| `compare_quantized_runtime.py` | Python int8 vs C++ int8 parity |
| `verify_cardputer_serial.py` | Device serial output vs Python quantized runtime |
| `eval_chat_checkpoint.py` | Chat eval on a checkpoint |
| `benchmark_instruction_following.py` | 10-category instruction benchmark |
| `benchmark_public_llm.py` | Public MCQ benchmarks (PIQA / ARC / HellaSwag / WinoGrande) |
| `benchmark_gelu_lut.py` | GELU LUT accuracy/perf benchmark (C++ runtime) |
| `router_diagnostics.py` | MoE router load / collapse diagnostics |

### Seed files

- `chat_capability_seed.txt`, `chat_capability_seed_science_boost_v2.txt`, `chat_capability_seed_transfer_v1.txt` — chat SFT seeds
- `instruction_chat_seed.txt` — instruction SFT seeds (40 examples, 10 categories)

## Available Datasets

### Base Training Corpus

**FineWeb** - Large web corpus for base language model training
- Source: HuggingFace (HuggingFaceFW/fineweb)
- Size: Multiple splits available (10BT, 100BT samples)
- Format: Plain text
- Use case: Pre-training base models

### Chat/Conversation Datasets (Fine-tuning)

**ShareGPT** - Human conversations
- Source: anon8231489123/ShareGPT_Vicuna_unfiltered
- Format: Conversations
- Use case: Chat fine-tuning

**Alpaca** - Instruction-following dataset
- Source: tatsu-lab/alpaca
- Format: Instruction-input-output triplets
- Use case: Instruction following

**WizardLM** - Evolved instructions
- Source: WizardLM/WizardLM_evol_instruct_V2_196k
- Format: Instruction-output pairs
- Use case: Complex instruction following

**OpenOrca** - Open-source reasoning dataset
- Source: Open-Orca/OpenOrca
- Format: System prompt-question-response
- Use case: Reasoning and chat

## Quick Start

### Download All Datasets (Interactive)

```bash
cd scripts
./download_datasets.sh
```

### Download FineWeb Only

```bash
python scripts/download_fineweb.py --output_dir data/fineweb --sample_size 10000
```

### Download Chat Datasets

```bash
# All chat datasets
python scripts/download_chat_datasets.py --output_dir data/chat --datasets all --combine

# Specific datasets
python scripts/download_chat_datasets.py --output_dir data/chat --datasets alpaca wizardlm --combine
```

## Usage Examples

### 1. Base Model Training

```bash
# Download FineWeb (sample for testing)
python scripts/download_fineweb.py --output_dir data/fineweb --sample_size 50000

# Create independent train/validation files (deterministic split by document hash)
python scripts/split_corpus_train_val.py \
    --input data/fineweb/fineweb.txt \
    --train-output data/fineweb/fineweb_train.txt \
    --val-output data/fineweb/fineweb_val.txt \
    --val-ratio 0.02 \
    --overwrite

# Train base model
python python/train.py \
    --data data/fineweb/fineweb_train.txt \
    --val_data data/fineweb/fineweb_val.txt \
    --output_dir checkpoints/base
```

### 2. Chat Fine-tuning

```bash
# Download chat datasets
python scripts/download_chat_datasets.py --output_dir data/chat --datasets all --combine

# Fine-tune on chat data
python python/train.py \
    --data data/chat/chat_combined.txt \
    --output_dir checkpoints/chat \
    --epochs 5
```

### 3. Combined Training

```bash
# Download both
python scripts/download_fineweb.py --output_dir data/fineweb --sample_size 100000
python scripts/download_chat_datasets.py --output_dir data/chat --datasets alpaca --combine

# Combine for training
cat data/fineweb/fineweb.txt data/chat/chat_combined.txt > data/combined.txt

# Train
python python/train.py --data data/combined.txt
```

## Dataset Details

### FineWeb

**Splits:**
- `sample-10BT`: 10 billion token sample
- `sample-100BT`: 100 billion token sample
- `CC-MAIN-2024-10`: Full Common Crawl 2024-10

**Usage:**
```bash
python scripts/download_fineweb.py \
    --output_dir data/fineweb \
    --split sample-10BT \
    --sample_size 100000
```

### Chat Datasets

**ShareGPT:**
- Large collection of human conversations
- Good for conversational AI
- Streaming dataset (can be large)

**Alpaca:**
- 52K instruction-following examples
- Good for instruction following
- Small, fast to download

**WizardLM:**
- 196K evolved instructions
- Complex, multi-step tasks
- Streaming dataset

**OpenOrca:**
- Reasoning-focused conversations
- System prompts included
- Streaming dataset

## Memory and Storage

**FineWeb:**
- Sample-10BT: ~40GB (full)
- Sample-100BT: ~400GB (full)
- Use `--sample_size` to limit

**Chat Datasets:**
- Alpaca: ~50MB
- ShareGPT: Several GB (streaming)
- WizardLM: Several GB (streaming)
- OpenOrca: Several GB (streaming)

## Requirements

```bash
pip install datasets huggingface-hub
```

## Troubleshooting

### "Dataset not found"
- Check internet connection
- Verify dataset names on HuggingFace
- Try smaller sample sizes first

### Out of memory
- Use `--sample_size` to limit downloads
- Download datasets one at a time
- Use streaming mode (default)

### Slow downloads
- FineWeb is very large - use sample splits
- Chat datasets are smaller but still substantial
- Consider downloading overnight

## Data Format

All datasets are converted to plain text format:
- One document per line (or paragraph)
- Double newlines separate documents
- UTF-8 encoding
- Ready for training pipeline

## Next Steps

After downloading:
1. Check data quality: `head data/fineweb/fineweb.txt`
2. Train with the MoE pipeline: `./scripts/train_moe_pipeline.sh --export`
3. Flash Cardputer: `cd esp32_m5stack && pio run -e m5stack_cardputer_nopsram -t upload`
4. Verify: `python scripts/verify_cardputer_serial.py --reset`

## MoE Training Pipeline

Full Cardputer workflow (pretrain → chat fine-tune → PROGMEM embed):

```bash
# Full training (13L flash-optimal config by default)
./scripts/train_moe_pipeline.sh --export

# Fast smoke test (~minutes)
./scripts/train_moe_pipeline.sh --profile quick --export

# Size a custom config
python scripts/moe_tradeoff_estimator.py --cardputer-optimal --json
```

Profiles (see the `case "$PROFILE"` block in `train_moe_pipeline.sh`): `cardputer-mqa-ctx224` (released MQA 2048-vocab model — the default), `quick` (smoke), `cardputer`, `cardputer-safe`, `cardputer-smoke`, `cardputer-chat`, `cardputer-chat-wide`, `cardputer-mqa-wide96`, `cardputer-mqa-bal192-88`, `cardputer-mqa-ctx208-l20`, `cardputer-dense-capacity`, `cardputer-dense-mac`, `cardputer-balanced`, `cardputer-legacy`, `gpu8g-mqa-instruct`, `desktop`, plus dated research profiles.

See also: `MOE_CARDPUTER_RESEARCH.md`, `scripts/interactive_chat.py`

