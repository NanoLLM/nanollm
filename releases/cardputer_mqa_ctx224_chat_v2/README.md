# Cardputer MQA ctx224 chat v2 — public release

**Science-boost chat SFT (v3 seed)** on the standard Cardputer profile, built
on the new **e20 3x-corpus foundation** (20-epoch pretrain, val loss 3.188).
Firmware embeds the promoted conversational checkpoint
(clean-transfer **0.76**, up from v1's 0.72).

Predecessor: [`../cardputer_mqa_ctx224_chat_v1/README.md`](../cardputer_mqa_ctx224_chat_v1/README.md).

## What changed vs v1

| | v1 | v2 |
|---|----|----|
| Foundation pretrain | `moe_run_20260718_045345` (10ep, val 3.30) | `moe_run_20261002_133235` (20ep, 3x corpus, val 3.188) |
| SFT seed | `chat_capability_seed_science_boost_v2.txt` | `chat_capability_seed_science_boost_v3.txt` |
| Seed changes | — | removed the "assistant is responding" template line; de-duplicated arithmetic sums; +12 held-out-safe science paraphrases |
| clean-transfer | 0.72 (arithmetic 0.6, science 0.2, greeting 0.8) | **0.76** (arithmetic 1.0, science 0.6, greeting 1.0) |
| SFT run | `science_boost_l18_20260721` (seed 44) | `science_boost_l18_20261003v3` (seed 42) |

## Profile

Identical to v1: `S=224`, `V=2048`, `d=80`, `L=18`, MQA `n_kv=1`, 4-expert
MoE top-1, `shared_d_ff=160`. Only the weights differ, so the esp32 firmware
source is unchanged.

## Contents

| Path | Purpose |
|------|---------|
| `foundation/model_pretrain.pt` | e20 fine-tune starting point |
| `chat/model_best.pt` | **Science-boost v3 SFT checkpoint** (deployed in firmware) |
| `chat/sft_recipe.txt` | SFT stages and seed configuration |
| `chat/clean_transfer_summary.json` | Held-out transfer metrics (0.76) |
| `deploy/` | int8 `model.bin`, PROGMEM headers, `firmware.bin` |
| `tokenizer/` | Pinned BPE v1 (vocab 2048) |
| `manifest.json` | SHA256 hashes + commands + pinned_artifacts |

## Quick start

### Flash chat firmware

```bash
# Headers already copied to esp32_m5stack/src/ during packaging.
cd esp32_m5stack
pio run -e m5stack_cardputer_nopsram -t upload
pio device monitor
```

### Verify parity

```bash
python scripts/verify_cardputer_serial.py --reset \
  --checkpoint releases/cardputer_mqa_ctx224_chat_v2/chat/model_best.pt \
  --max-new-tokens 4
```

### Fine-tune from the new foundation

```bash
INIT_PRETRAIN=releases/cardputer_mqa_ctx224_chat_v2/foundation/model_pretrain.pt \
  ./scripts/train_science_boost_sft.sh
```

### Rebuild this release

```bash
VARIANT=chat RELEASE_TAG=cardputer_mqa_ctx224_chat_v2 \
FOUNDATION_CKPT=checkpoints/moe_run_20261002_133235/model_pretrain.pt \
CHAT_CKPT=checkpoints/science_boost_l18_20261003v3/final4/model_best.pt \
SFT_RECIPE=science_boost_v3_60x \
SEED_DATA_FILE=scripts/chat_capability_seed_science_boost_v3.txt \
CLEAN_TRANSFER_JSON=paper/results/science_boost_l18_20261003v3_seed42_summary.json \
./scripts/package_cardputer_release.sh
```

## SFT recipe

- Init: `foundation/model_pretrain.pt` (e20 3x-corpus lineage)
- Seed: `scripts/chat_capability_seed_science_boost_v3.txt`, 60× repeats
- Stages: `20→12→8→4` epochs at `1e-4 / 5e-5 / 3e-5 / 1e-5`
- Selection: validation loss (held-out-safe; audit: 0 exact / 0 high-overlap vs the frozen `clean_transfer_v1` suite at 0.8)

## License

Same as the NanoLLM repository root.
