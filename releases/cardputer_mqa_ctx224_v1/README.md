# Cardputer MQA ctx224 v1 — public release

Standard **ESP32 M5Stack Cardputer** deploy profile for NanoLLM: foundation pretrain for fine-tuning, plus embedded firmware artifacts.

## Profile

| Field | Value |
|-------|------|
| Context | 224 |
| Vocab | 2048 (pinned BPE) |
| Width / depth | d=80, L=18 |
| Attention | MQA, n_kv=1 |
| MoE | 4 experts, top-1, shared d_ff=160 |
| Params | ~4.66M (after full SFT) / ~4.8M trainable at pretrain |

## Contents

| Path | Purpose |
|------|---------|
| `foundation/model_pretrain.pt` | **Fine-tune starting point** (July lineage, canonical tokenizer) |
| `tokenizer/` | Pinned BPE (`cardputer_vocab2048_v1`) — always use for new runs |
| `deploy/` | int8 `model.bin`, PROGMEM headers, `firmware.bin` (embedded deploy checkpoint) |
| `manifest.json` | SHA256 hashes, git commit, commands |

**Chat SFT release:** [`../cardputer_mqa_ctx224_chat_v1/`](../cardputer_mqa_ctx224_chat_v1/) — science-boost firmware for on-device chat (clean-transfer 0.72).

## Quick start

### 1. Flash firmware (from repo root, after packaging)

```bash
cd esp32_m5stack
pio run -e m5stack_cardputer_nopsram -t upload
pio device monitor
```

Default package embeds the **foundation pretrain** weights. Expect weak chat until you fine-tune.

For ready-to-use chat firmware, use the **chat release** (`VARIANT=chat` or `releases/cardputer_mqa_ctx224_chat_v1/`).

### 2. Fine-tune from foundation

```bash
INIT_PRETRAIN=releases/cardputer_mqa_ctx224_v1/foundation/model_pretrain.pt \
  ./scripts/train_science_boost_sft.sh
```

Or baseline clean-transfer SFT:

```bash
INIT_PRETRAIN=releases/cardputer_mqa_ctx224_v1/foundation/model_pretrain.pt \
  ./scripts/train_clean_transfer_sft.sh
```

Tokenizer is pinned automatically via `train_moe_pipeline.sh` → `data/tokenizers/cardputer_vocab2048_v1`.

### 3. Re-package after your fine-tune

```bash
DEPLOY_CHECKPOINT=checkpoints/my_run/final4/model_best.pt \
  ./scripts/package_cardputer_release.sh
```

### 4. Verify device parity

```bash
python scripts/verify_cardputer_serial.py --reset \
  --checkpoint checkpoints/science_boost_l18_20260721/final4/model_best.pt \
  --max-new-tokens 4
```

## Rebuild this release

```bash
./scripts/package_cardputer_release.sh              # foundation
VARIANT=chat ./scripts/package_cardputer_release.sh  # chat SFT
./scripts/package_cardputer_release.sh --both       # both releases
```

Options:

- `VARIANT=foundation|chat` — release variant
- `DEPLOY_CHECKPOINT=...` — override embedded checkpoint
- `CHAT_CKPT=...` — chat SFT checkpoint copied to `chat/`
- `SKIP_FIRMWARE=1` — export only, no `pio run`
- `RELEASE_TAG=cardputer_mqa_ctx224_v1` — output directory name

## License

Same as the NanoLLM repository root.
