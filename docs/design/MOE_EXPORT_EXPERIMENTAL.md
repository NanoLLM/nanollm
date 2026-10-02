# Experimental MoE Export

This repository now supports an **experimental** MoE export path for Python-side artifacts.

Important:

- Desktop C++ runtime supports this format for parity testing.
- ESP32 runtime supports this format via SPIFFS-loaded weights.
- Embedded-header (`NANOLLM_USE_EMBEDDED_WEIGHTS`) MoE is still unsupported.
- Dense export format and workflow remain unchanged.

## Dense (unchanged)

```bash
python python/export_weights.py \
  --checkpoint checkpoints/model_best.pt \
  --output weights/model.bin
```

## MoE (experimental)

```bash
python python/export_weights.py \
  --checkpoint checkpoints/model_best.pt \
  --output weights/model_moe.bin \
  --allow-moe-export
```

To stage MoE artifacts directly onto a mounted Cardputer SD card:

```bash
cd esp32_m5stack
./prepare_sdcard_model.sh --mount /media/$USER/CARDPUTER --mode moe --basename model_moe
```

Generated files:

- `*_config.json` (contains `format: nanollm_moe_v1`)
- `*_format.json` (record layout summary)
- `*.bin` (experimental `NLMO` magic format)

## Why gated behind `--allow-moe-export`

MoE artifacts are still marked experimental while runtime support evolves (especially for embedded-header and SD-paged experts). The explicit flag prevents accidental format drift in regular dense workflows.

## Next required runtime steps

1. Add MoE config fields to embedded loader.
2. Add embedded-header MoE runtime path.
3. Add SD-backed expert paging + cache for large expert sets.
