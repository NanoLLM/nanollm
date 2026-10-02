#!/usr/bin/env bash
# Held-out-safe full chat training cycle for the mqa_ctx224 architecture.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

RUN_TAG="${RUN_TAG:-20260718}"
RUN_ROOT="${RUN_ROOT:-checkpoints/clean_transfer_l18_${RUN_TAG}}"
DATA_DIR="${DATA_DIR:-data/clean_transfer_v1}"
RESULTS_DIR="${RESULTS_DIR:-logs/results}"
RAW_CHAT="${RAW_CHAT:-data/chat/teacher_distilled_qwen3_5_3k.txt}"
SEED_DATA="${SEED_DATA:-scripts/chat_capability_seed_transfer_v1.txt}"
TRANSFER_SUITE="${TRANSFER_SUITE:-data/eval/clean_transfer_v1.json}"
INIT_PRETRAIN="${INIT_PRETRAIN:-checkpoints/moe_run_20260718_045345/model_pretrain.pt}"
PROFILE="${PROFILE:-cardputer-mqa-ctx224}"
CUDA_DEVICE="${CUDA_DEVICE:-1}"
TRAIN_SEED="${TRAIN_SEED:-42}"
CHAT_SEED_REPEATS="${CHAT_SEED_REPEATS:-40}"

TRAIN_DATA="$DATA_DIR/chat_train.txt"
VAL_DATA="$DATA_DIR/chat_val.txt"
AUDIT_JSON="$RESULTS_DIR/clean_transfer_v1_overlap_audit.json"

for path in "$RAW_CHAT" "$SEED_DATA" "$TRANSFER_SUITE" "$INIT_PRETRAIN"; do
  if [[ ! -f "$path" ]]; then
    echo "Missing required file: $path" >&2
    exit 1
  fi
done

mkdir -p "$DATA_DIR" "$RESULTS_DIR" "$RUN_ROOT"

python scripts/prepare_chat_data.py \
  --input "$RAW_CHAT" \
  --seed-data "$SEED_DATA" \
  --train-output "$TRAIN_DATA" \
  --val-output "$VAL_DATA" \
  --seed "$TRAIN_SEED" \
  --seed-repeats "$CHAT_SEED_REPEATS" \
  --exclude-prompts-file "$TRANSFER_SUITE" \
  --exclude-similarity-threshold 0.8

python scripts/audit_chat_overlap.py \
  --suite "$TRANSFER_SUITE" \
  --chat-file "$TRAIN_DATA" \
  --chat-file "$VAL_DATA" \
  --chat-file "$SEED_DATA" \
  --output "$AUDIT_JSON"

run_stage() {
  local name="$1"
  local init_checkpoint="$2"
  local epochs="$3"
  local lr="$4"
  local output_dir="$RUN_ROOT/$name"
  local stage_marker="$output_dir/.clean_stage_complete"

  if [[ -f "$output_dir/model_best.pt" && -f "$stage_marker" ]]; then
    echo "Skipping completed stage: $output_dir"
    return
  fi

  CUDA_DEVICE="$CUDA_DEVICE" \
  OUTPUT_DIR="$output_dir" \
  PROFILE="$PROFILE" \
  N_KV_HEADS=1 \
  CHAT_DATA="$RAW_CHAT" \
  CHAT_SEED="$SEED_DATA" \
  CHAT_TRAIN="$TRAIN_DATA" \
  CHAT_VAL="$VAL_DATA" \
  CHAT_EXCLUDE_PROMPTS_FILE="$TRANSFER_SUITE" \
  CHAT_EXCLUDE_SIMILARITY_THRESHOLD=0.8 \
  CHAT_SEED_REPEATS="$CHAT_SEED_REPEATS" \
  CHAT_BATCH_SIZE=128 \
  CHAT_EPOCHS="$epochs" \
  CHAT_LR="$lr" \
  CHAT_EVAL=1 \
  CHAT_EVAL_SELECT=0 \
  CHAT_EVAL_REQUIRE_HARD=0 \
  EARLY_STOP_PATIENCE=0 \
  TRAIN_SEED="$TRAIN_SEED" \
  AMP=bf16 TF32=1 NUM_WORKERS=8 \
    ./scripts/train_moe_pipeline.sh --finetune-only \
      --init-checkpoint "$init_checkpoint" \
      --no-embed --no-latest-link
  touch "$stage_marker"
}

run_stage "chat20" "$INIT_PRETRAIN" 20 1e-4
run_stage "capability12" "$RUN_ROOT/chat20/model_best.pt" 12 5e-5
run_stage "eos8" "$RUN_ROOT/capability12/model_best.pt" 8 3e-5
run_stage "final4" "$RUN_ROOT/eos8/model_best.pt" 4 1e-5

printf '%s\n' "$RUN_ROOT/final4/model_best.pt" > "$RUN_ROOT/final_checkpoint.txt"
python - "$RUN_ROOT" "$INIT_PRETRAIN" "$TRANSFER_SUITE" "$SEED_DATA" "$TRAIN_DATA" "$VAL_DATA" <<'PY'
import json
import sys
from pathlib import Path

run_root, init, suite, seed, train, val = sys.argv[1:]
payload = {
    "schema_version": 1,
    "run_root": run_root,
    "init_checkpoint": init,
    "transfer_suite": suite,
    "seed_data": seed,
    "train_data": train,
    "val_data": val,
    "stages": [
        {"name": "chat20", "epochs": 20, "learning_rate": 1e-4},
        {"name": "capability12", "epochs": 12, "learning_rate": 5e-5},
        {"name": "eos8", "epochs": 8, "learning_rate": 3e-5},
        {"name": "final4", "epochs": 4, "learning_rate": 1e-5},
    ],
    "checkpoint_selection": "validation_loss",
}
Path(run_root, "run_metadata.json").write_text(json.dumps(payload, indent=2) + "\n")
PY

echo "Clean transfer SFT complete: $RUN_ROOT/final4/model_best.pt"
