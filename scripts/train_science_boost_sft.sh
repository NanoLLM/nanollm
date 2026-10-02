#!/usr/bin/env bash
# Held-out-safe SFT with boosted science/factual/writing seed examples.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

RUN_TAG="${RUN_TAG:-20260721}"
RUN_ROOT="${RUN_ROOT:-checkpoints/science_boost_l18_${RUN_TAG}}"
DATA_DIR="${DATA_DIR:-data/science_boost_v1}"
RESULTS_DIR="${RESULTS_DIR:-logs/results}"
RAW_CHAT="${RAW_CHAT:-data/chat/teacher_distilled_qwen3_5_3k.txt}"
SEED_DATA="${SEED_DATA:-scripts/chat_capability_seed_science_boost_v2.txt}"
TRANSFER_SUITE="${TRANSFER_SUITE:-data/eval/clean_transfer_v1.json}"
INIT_PRETRAIN="${INIT_PRETRAIN:-checkpoints/moe_run_20260718_045345/model_pretrain.pt}"
PROFILE="${PROFILE:-cardputer-mqa-ctx224}"
CUDA_DEVICE="${CUDA_DEVICE:-1}"
TRAIN_SEED="${TRAIN_SEED:-44}"
CHAT_SEED_REPEATS="${CHAT_SEED_REPEATS:-60}"

export RUN_TAG RUN_ROOT DATA_DIR SEED_DATA CHAT_SEED_REPEATS INIT_PRETRAIN PROFILE CUDA_DEVICE TRAIN_SEED

echo "=== Science-boost SFT (seed=${TRAIN_SEED}, repeats=${CHAT_SEED_REPEATS}) ==="
echo "run_root=$RUN_ROOT init=$INIT_PRETRAIN"

RUN_ROOT="$RUN_ROOT" \
DATA_DIR="$DATA_DIR" \
SEED_DATA="$SEED_DATA" \
CHAT_SEED_REPEATS="$CHAT_SEED_REPEATS" \
INIT_PRETRAIN="$INIT_PRETRAIN" \
TRAIN_SEED="$TRAIN_SEED" \
CUDA_DEVICE="$CUDA_DEVICE" \
  ./scripts/train_clean_transfer_sft.sh

FINAL_CKPT="$RUN_ROOT/final4/model_best.pt"
SUITE_OUT="$RESULTS_DIR/science_boost_l18_${RUN_TAG}_seed${TRAIN_SEED}_suite.json"
SUMMARY_OUT="$RESULTS_DIR/science_boost_l18_${RUN_TAG}_seed${TRAIN_SEED}_summary.json"

python scripts/eval_chat_checkpoint.py \
  --checkpoint "$FINAL_CKPT" \
  --cases-file "$TRANSFER_SUITE" \
  --greedy \
  --output "$SUITE_OUT"

python - "$SUITE_OUT" "$FINAL_CKPT" "$SUMMARY_OUT" "$TRAIN_SEED" "$CHAT_SEED_REPEATS" <<'PY'
import json, sys
from pathlib import Path
suite_path, ckpt, out, seed, repeats = sys.argv[1:]
suite = json.loads(Path(suite_path).read_text())
summary = {
    "experiment": "science_boost_sft_l18",
    "seed": int(seed),
    "chat_seed_repeats": int(repeats),
    "seed_data": "scripts/chat_capability_seed_science_boost_v2.txt",
    "final_checkpoint": ckpt,
    "clean_transfer": suite.get("aggregate_score") or suite.get("score"),
    "category_scores": suite.get("category_scores"),
    "suite_artifact": suite_path,
    "baseline_promoted_l18_seed44": 0.52,
}
Path(out).write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))
PY

# Optional research-tracker bookkeeping (only if the helper is present).
if [[ -f scripts/update_performance_research_tracker.py ]]; then
  python scripts/update_performance_research_tracker.py \
    --experiment exp2_science_boost \
    --status completed \
    --artifact "$SUMMARY_OUT" || true
fi

echo "DONE science-boost: $FINAL_CKPT"
