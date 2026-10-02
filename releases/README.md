# Releases

Each release directory contains a **manifest** (config, hashes, tokenizer) that is
tracked in git. Heavy artifacts (`.pt` checkpoints, `model.bin`, `firmware.bin`,
PROGMEM headers) are **not** in this repository.

## Where to download

| Artifact | Location |
|----------|----------|
| Firmware (`firmware.bin`), int8 weights (`model.bin`), PROGMEM headers | **GitHub Releases** — tag per release (e.g. `cardputer-mqa-ctx224-chat-v1`) |
| PyTorch checkpoints (`.pt`), training configs | **Hugging Face** — [`josedayo/nanollm`](https://huggingface.co/josedayo/nanollm) (or your HF org) |
| Tokenizer (BPE vocab, `tokenizer.json`) | In git under each release's `tokenizer/` dir (small, ~100 KB) |

## Release layout (in git)

```
releases/<release_name>/
  manifest.json          # SHA256 hashes, build commit, profile config
  README.md              # Quick start, verification commands
  deploy/                # model_config.json, model_format.json, tokenizer_info.json
  tokenizer/             # tokenizer.json, vocab.json, manifest.json
  chat/                  # (chat variant) sft_recipe.txt, clean_transfer_summary.json
```

## Publishing a new release

1. Run `./scripts/package_cardputer_release.sh` to build artifacts.
2. Tag the repo: `git tag cardputer-mqa-ctx224-chat-v1`.
3. On GitHub: create a release for that tag; upload `firmware.bin`, `model.bin`,
   `model_weights.h`, `vocab_weights.h` as release assets.
4. Push `.pt` checkpoints to Hugging Face:
   ```bash
   huggingface-cli upload josedayo/nanollm \
     releases/<name>/chat/model_best.pt \
     chat/model_best.pt
   ```
5. Update the release `manifest.json` with the HF path for checkpoints.

## Existing releases

| Tag | Variant | Description |
|-----|---------|-------------|
| `cardputer-mqa-ctx224-v1` | foundation | Pretrained base (no chat SFT) |
| `cardputer-mqa-ctx224-chat-v1` | chat | Science-boost SFT (clean-transfer 0.72) |
