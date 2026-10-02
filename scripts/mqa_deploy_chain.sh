#!/usr/bin/env bash
# Chain capability/EOS/final/export/flash for MQA Cardputer deploy after base train completes.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

BASE_DIR="${BASE_DIR:-checkpoints/mqa_cardputer_wide_seed42_20260713}"
INIT="${INIT:-$BASE_DIR/model_best.pt}"

common_env() {
  CUDA_DEVICE="${CUDA_DEVICE:-0}"
  TRAIN_SEED="${TRAIN_SEED:-42}"
  N_KV_HEADS="${N_KV_HEADS:-1}"
  PROFILE="${PROFILE:-cardputer-chat-wide}"
  CHAT_DATA="${CHAT_DATA:-data/chat/teacher_distilled_qwen3_5_3k.txt}"
  CHAT_BATCH_SIZE="${CHAT_BATCH_SIZE:-128}"
  CHAT_SEED_REPEATS="${CHAT_SEED_REPEATS:-40}"
  WINDOW_STRIDE_TOKENS="${WINDOW_STRIDE_TOKENS:-56}"
  CHAT_EVAL_SELECT="${CHAT_EVAL_SELECT:-1}"
  CHAT_EVAL_REQUIRE_HARD="${CHAT_EVAL_REQUIRE_HARD:-0}"
  EARLY_STOP_PATIENCE="${EARLY_STOP_PATIENCE:-0}"
  AMP="${AMP:-bf16}"
  TF32="${TF32:-1}"
  NUM_WORKERS="${NUM_WORKERS:-8}"
}

wait_for_checkpoint() {
  local marker="$1"
  echo "Waiting for training marker: $marker"
  while [[ ! -f "$marker" ]]; do
    sleep 120
  done
  echo "Training stage complete: $marker"
}

run_stage() {
  local out_dir="$1"
  local init_ckpt="$2"
  local epochs="$3"
  local lr="$4"
  common_env
  OUTPUT_DIR="$out_dir" \
  CUDA_DEVICE="$CUDA_DEVICE" TRAIN_SEED="$TRAIN_SEED" N_KV_HEADS="$N_KV_HEADS" PROFILE="$PROFILE" \
  CHAT_DATA="$CHAT_DATA" CHAT_BATCH_SIZE="$CHAT_BATCH_SIZE" CHAT_EPOCHS="$epochs" CHAT_LR="$lr" \
  CHAT_SEED_REPEATS="$CHAT_SEED_REPEATS" WINDOW_STRIDE_TOKENS="$WINDOW_STRIDE_TOKENS" \
  CHAT_EVAL_SELECT="$CHAT_EVAL_SELECT" CHAT_EVAL_REQUIRE_HARD="$CHAT_EVAL_REQUIRE_HARD" \
  EARLY_STOP_PATIENCE="$EARLY_STOP_PATIENCE" AMP="$AMP" TF32="$TF32" NUM_WORKERS="$NUM_WORKERS" \
  ./scripts/train_moe_pipeline.sh --finetune-only --init-checkpoint "$init_ckpt" --no-embed --no-latest-link
}

wait_for_checkpoint "$BASE_DIR/training_complete"
INIT="$BASE_DIR/model_best.pt"
[[ -f "$INIT" ]] || { echo "Missing $INIT after training_complete"; exit 1; }

CAP_DIR="${BASE_DIR%/}_capability_20260713"
EOS_DIR="${BASE_DIR%/}_eos_20260713"
FINAL_DIR="${BASE_DIR%/}_final_20260713"

run_stage "$CAP_DIR" "$INIT" 20 5e-5
run_stage "$EOS_DIR" "$CAP_DIR/model_best.pt" 12 3e-5
run_stage "$FINAL_DIR" "$EOS_DIR/model_best.pt" 6 1e-5

DEPLOY_CKPT="$FINAL_DIR/model_best.pt"
echo "Exporting and embedding MQA checkpoint: $DEPLOY_CKPT"
CUDA_DEVICE="$CUDA_DEVICE" TRAIN_SEED="$TRAIN_SEED" N_KV_HEADS="$N_KV_HEADS" \
  OUTPUT_DIR="$FINAL_DIR" PROFILE="$PROFILE" \
  ./scripts/train_moe_pipeline.sh --export --init-checkpoint "$DEPLOY_CKPT" --no-latest-link

echo "Building and flashing Cardputer firmware..."
cd esp32_m5stack
pio run -e m5stack_cardputer_nopsram -t upload
cd "$PROJECT_ROOT"

echo "Running serial parity + latency capture..."
python scripts/verify_cardputer_serial.py --reset --checkpoint "$DEPLOY_CKPT" --max-new-tokens 4

echo "MQA deploy chain complete."
