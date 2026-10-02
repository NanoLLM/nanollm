# Cardputer MQA ctx224 chat v1 — public release

**Science-boost chat SFT** on the standard Cardputer profile. Firmware embeds the promoted conversational checkpoint (clean-transfer **0.72**).

Companion foundation-only release: [`../cardputer_mqa_ctx224_v1/README.md`](../cardputer_mqa_ctx224_v1/README.md).

## Profile

Same as foundation release: `S=224`, `V=2048`, `d=80`, `L=18`, MQA `n_kv=1`, 4-expert MoE top-1.

## Contents

| Path | Purpose |
|------|---------|
| `foundation/model_pretrain.pt` | Fine-tune starting point (included for continuity) |
| `chat/model_best.pt` | **Science-boost SFT checkpoint** (deployed in firmware) |
| `chat/sft_recipe.txt` | SFT stages and seed configuration |
| `chat/clean_transfer_summary.json` | Held-out transfer metrics |
| `deploy/` | int8 `model.bin`, PROGMEM headers, `firmware.bin` |
| `tokenizer/` | Pinned BPE v1 |
| `manifest.json` | SHA256 hashes + commands |

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
  --checkpoint releases/cardputer_mqa_ctx224_chat_v1/chat/model_best.pt \
  --max-new-tokens 4
```

### Rebuild this release

```bash
VARIANT=chat ./scripts/package_cardputer_release.sh
```

Build **both** foundation + chat:

```bash
./scripts/package_cardputer_release.sh --both
```

## SFT recipe

- Init: `foundation/model_pretrain.pt` (July lineage)
- Seed: `scripts/chat_capability_seed_science_boost_v2.txt`, 60× repeats
- Stages: `20→12→8→4` epochs at `1e-4 / 5e-5 / 3e-5 / 1e-5`
- Selection: validation loss (held-out-safe)

## License

Same as the NanoLLM repository root.
