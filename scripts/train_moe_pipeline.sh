#!/usr/bin/env bash
# Train a compact MoE NanoLLM: FineWeb pretrain -> chat fine-tune -> PROGMEM embed (default).
#
# Defaults target ESP32 M5Stack Cardputer with flash-backed weights (partitions_embedded.csv):
# weights live in firmware flash, not RAM — see scripts/moe_tradeoff_estimator.py --cardputer-optimal.
#
# Default architecture is the Cardputer deploy profile cardputer-mqa-ctx224:
#   V=2048, S=224, D=80, L=18, H=4, n_kv=1, d_ff=320, 4 experts top-1, shared=160
# (clean-transfer recipe: val_loss selection, held-out suite excluded from SFT).
#
# Usage:
#   ./scripts/train_moe_pipeline.sh
#   ./scripts/train_moe_pipeline.sh --no-embed          # skip PROGMEM header export
#   ./scripts/train_moe_pipeline.sh --export            # also write weights/model.bin (SPIFFS/desktop)
#   ./scripts/train_moe_pipeline.sh --profile cardputer-chat-wide
#   ./scripts/train_moe_pipeline.sh --instruction-finetune  # use instruction data instead of chat
#   PRETRAIN_DATA=data/fineweb/fineweb_tiny.txt ./scripts/train_moe_pipeline.sh
#
# Tokenizer (Cardputer vocab=2048 profiles):
#   Default: reuse pinned data/tokenizers/cardputer_vocab2048_v1 (promoted lineage).
#   TRAIN_TOKENIZER=1 or --train-tokenizer: fit a fresh BPE on PRETRAIN_DATA+CHAT_DATA.
#   REUSE_TOKENIZER=path: explicit override (dir or tokenizer.json).
#
# Environment overrides (examples):
#   OUTPUT_DIR=checkpoints/my_moe_run EPOCHS=3 ./scripts/train_moe_pipeline.sh
#   MOE_N_EXPERTS=4 BLOCK_SIZE=128 ./scripts/train_moe_pipeline.sh --export
#
# Resume an interrupted run (re-use the same OUTPUT_DIR):
#   OUTPUT_DIR=checkpoints/moe_run_20260701_091051 ./scripts/train_moe_pipeline.sh --export
#   AUTO_RESUME=0 ./scripts/train_moe_pipeline.sh   # force a new moe_run_* directory
#
# Symlink: checkpoints/latest -> active run (see LATEST_LINK)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python3}"
TRAIN_PY="$PROJECT_ROOT/python/train.py"
EXPORT_PY="$PROJECT_ROOT/python/export_weights.py"
EMBED_HEADER_PY="$PROJECT_ROOT/python/export_weights_header.py"
EMBED_VOCAB_PY="$PROJECT_ROOT/python/export_vocab_header.py"
NORMALIZE_CHAT_PY="$PROJECT_ROOT/scripts/normalize_chat_data.py"
FILTER_CHAT_PY="$PROJECT_ROOT/scripts/filter_chat_data.py"
EVAL_CHAT_PY="$PROJECT_ROOT/scripts/eval_chat_checkpoint.py"
CHAT_FORMATTED="${CHAT_FORMATTED:-$PROJECT_ROOT/data/chat/chat_formatted.txt}"
CHAT_CURATED="${CHAT_CURATED:-$PROJECT_ROOT/data/chat/chat_curated.txt}"
CHAT_TRAIN="${CHAT_TRAIN:-$PROJECT_ROOT/data/chat/chat_train.txt}"
CHAT_VAL="${CHAT_VAL:-$PROJECT_ROOT/data/chat/chat_val.txt}"
CHAT_SEED="${CHAT_SEED:-$PROJECT_ROOT/scripts/chat_capability_seed_transfer_v1.txt}"
CHAT_EXCLUDE_PROMPTS_FILE="${CHAT_EXCLUDE_PROMPTS_FILE:-$PROJECT_ROOT/data/eval/clean_transfer_v1.json}"
CHAT_EXCLUDE_SIMILARITY_THRESHOLD="${CHAT_EXCLUDE_SIMILARITY_THRESHOLD:-0.8}"
PREPARE_CHAT_PY="$PROJECT_ROOT/scripts/prepare_chat_data.py"
PREPARE_INSTRUCT_PY="$PROJECT_ROOT/scripts/prepare_instruction_data.py"
DOWNLOAD_INSTRUCT_PY="$PROJECT_ROOT/scripts/download_instruction_datasets.py"
EVAL_CHAT_PY="$PROJECT_ROOT/scripts/eval_chat_checkpoint.py"
BENCHMARK_INSTRUCT_PY="$PROJECT_ROOT/scripts/benchmark_instruction_following.py"
CHAT_FORMATTED="${CHAT_FORMATTED:-$PROJECT_ROOT/data/chat/chat_formatted.txt}"
CHAT_CURATED="${CHAT_CURATED:-$PROJECT_ROOT/data/chat/chat_curated.txt}"
CHAT_TRAIN="${CHAT_TRAIN:-$PROJECT_ROOT/data/chat/chat_train.txt}"
CHAT_VAL="${CHAT_VAL:-$PROJECT_ROOT/data/chat/chat_val.txt}"
INSTRUCT_TRAIN="${INSTRUCT_TRAIN:-$PROJECT_ROOT/data/instruct/instruct_train.txt}"
INSTRUCT_VAL="${INSTRUCT_VAL:-$PROJECT_ROOT/data/instruct/instruct_val.txt}"
INSTRUCT_SEED="${INSTRUCT_SEED:-$PROJECT_ROOT/scripts/instruction_chat_seed.txt}"
INSTRUCT_RAW="${INSTRUCT_RAW:-$PROJECT_ROOT/data/instruct/instruct_raw.txt}"
INSTRUCT_PREP_STATS="${INSTRUCT_PREP_STATS:-$PROJECT_ROOT/data/instruct/instruct_prep_stats.json}"
INSTRUCT_SEED_REPEATS="${INSTRUCT_SEED_REPEATS:-3}"
MODEL_HEADER="$PROJECT_ROOT/esp32_m5stack/src/model_weights.h"
VOCAB_HEADER="$PROJECT_ROOT/esp32_m5stack/src/vocab_weights.h"
ENSURE_TOKENIZER_PY="$PROJECT_ROOT/scripts/ensure_canonical_tokenizer.py"
CANONICAL_TOKENIZER_DIR="${CANONICAL_TOKENIZER_DIR:-$PROJECT_ROOT/data/tokenizers/cardputer_vocab2048_v1}"
CANONICAL_TOKENIZER_SOURCE="${CANONICAL_TOKENIZER_SOURCE:-$PROJECT_ROOT/checkpoints/moe_run_20260718_045345/tokenizer}"
TRAIN_TOKENIZER="${TRAIN_TOKENIZER:-0}"

# Defaults — Cardputer deploy profile (mqa_ctx224, L=18).
# ~4.8M params, MQA cached decode, S=224 context; flash ~5.1 MiB firmware est.
PROFILE="${PROFILE:-cardputer-mqa-ctx224}"

PRETRAIN_DATA="${PRETRAIN_DATA:-$PROJECT_ROOT/data/fineweb/fineweb_3x.txt}"
# Teacher-distilled chat + transfer seed (held-out-safe); avoid alpaca+contaminated seed.
CHAT_DATA="${CHAT_DATA:-$PROJECT_ROOT/data/chat/teacher_distilled_qwen3_5_3k.txt}"
LATEST_LINK="${LATEST_LINK:-$PROJECT_ROOT/checkpoints/latest}"
ACTIVE_RUN_FILE="${ACTIVE_RUN_FILE:-$PROJECT_ROOT/checkpoints/.active_run}"

# Instruction-following fine-tuning flag.
# When enabled, uses data/instruct/ instead of data/chat/ for the fine-tune stage.
INSTRUCT_FINE_TUNING="${INSTRUCT_FINE_TUNING:-0}"

# Set explicitly by the user (env or future flag). When unset, resolve_output_dir picks
# an in-progress run instead of minting a new moe_run_YYYYMMDD_* folder on each invoke.
USER_OUTPUT_DIR=0
if [[ -v OUTPUT_DIR ]]; then
    USER_OUTPUT_DIR=1
fi
OUTPUT_DIR="${OUTPUT_DIR:-}"

# Model shape (cardputer profile overrides these; env vars apply when PROFILE is unchanged)
VOCAB_SIZE="${VOCAB_SIZE:-2048}"
D_MODEL="${D_MODEL:-80}"
N_LAYERS="${N_LAYERS:-18}"
N_HEADS="${N_HEADS:-4}"
N_KV_HEADS="${N_KV_HEADS:-1}"
D_FF="${D_FF:-320}"                # routed expert hidden size
BLOCK_SIZE="${BLOCK_SIZE:-224}"
DROPOUT="${DROPOUT:-0.02}"

# MoE (FFN-only, top-1 + shared expert — see MOE_CARDPUTER_RESEARCH.md)
USE_MOE="${USE_MOE:-1}"
USE_ROPE="${USE_ROPE:-0}"
MOE_N_EXPERTS="${MOE_N_EXPERTS:-4}"
MOE_TOP_K="${MOE_TOP_K:-1}"
MOE_SHARED_D_FF="${MOE_SHARED_D_FF:-160}"

# Track env overrides so profiles do not clobber explicit PRETRAIN_EPOCHS / CHAT_EPOCHS / CHAT_LR.
USER_PRETRAIN_EPOCHS=0
USER_CHAT_EPOCHS=0
USER_CHAT_LR=0
USER_WINDOW_STRIDE=0
if [[ -v PRETRAIN_EPOCHS ]]; then
    USER_PRETRAIN_EPOCHS=1
fi
if [[ -v CHAT_EPOCHS ]]; then
    USER_CHAT_EPOCHS=1
fi
if [[ -v CHAT_LR ]]; then
    USER_CHAT_LR=1
fi
if [[ -v WINDOW_STRIDE_TOKENS ]]; then
    USER_WINDOW_STRIDE=1
fi

# Pretrain (domain LM on FineWeb) — val loss still drops past 5 epochs on fineweb_3x.
PRETRAIN_EPOCHS="${PRETRAIN_EPOCHS:-10}"
PRETRAIN_BATCH_SIZE="${PRETRAIN_BATCH_SIZE:-256}"
PRETRAIN_LR="${PRETRAIN_LR:-2e-4}"
PRETRAIN_SAVE_EVERY="${PRETRAIN_SAVE_EVERY:-1}"

# Chat fine-tune (capability-dense SFT needs enough steps to overwrite web-completion priors)
CHAT_EPOCHS="${CHAT_EPOCHS:-10}"
CHAT_BATCH_SIZE="${CHAT_BATCH_SIZE:-256}"
CHAT_LR="${CHAT_LR:-5e-5}"
CHAT_LR_CONTINUE="${CHAT_LR_CONTINUE:-5e-5}"
CHAT_SAVE_EVERY="${CHAT_SAVE_EVERY:-1}"
CHAT_FILTER="${CHAT_FILTER:-1}"
CHAT_EVAL="${CHAT_EVAL:-1}"
CHAT_ASSISTANT_ONLY="${CHAT_ASSISTANT_ONLY:-1}"
# Select by validation loss (default). Contaminated fixed-prompt selection overstates transfer.
CHAT_EVAL_SELECT="${CHAT_EVAL_SELECT:-0}"
CHAT_EVAL_REQUIRE_HARD="${CHAT_EVAL_REQUIRE_HARD:-0}"
CHAT_SEED_REPEATS="${CHAT_SEED_REPEATS:-40}"

NUM_WORKERS="${NUM_WORKERS:-8}"
TOKEN_CACHE_CHUNK_CHARS="${TOKEN_CACHE_CHUNK_CHARS:-1000000}"
WINDOW_STRIDE_TOKENS="${WINDOW_STRIDE_TOKENS:-32}"

# GPU training tuning (RTX 3090/40xx: large batch + bf16 + tf32 — see AGENTS.md §12)
AMP="${AMP:-auto}"          # auto | bf16 | fp16 | none
TF32="${TF32:-1}"           # 1 enables TF32 matmul on Ampere+
CUDA_DEVICE="${CUDA_DEVICE:-}"  # optional GPU index, e.g. 0 or 1

# Training monitoring (passed through to python/train.py)
VAL_SPLIT="${VAL_SPLIT:-0.05}"
VAL_DATA="${VAL_DATA:-}"
GRAD_CLIP="${GRAD_CLIP:-2.0}"
WARMUP_RATIO="${WARMUP_RATIO:-0.05}"
# The greedy capability score is intentionally discrete; run the configured SFT budget.
EARLY_STOP_PATIENCE="${EARLY_STOP_PATIENCE:-0}"
TRAIN_SEED="${TRAIN_SEED:-42}"
TENSORBOARD="${TENSORBOARD:-1}"
SAVE_EVERY_STEPS="${SAVE_EVERY_STEPS:-10000}"

MONITORING_FLAGS=(
    --grad_clip "$GRAD_CLIP"
    --warmup_ratio "$WARMUP_RATIO"
    --early_stop_patience "$EARLY_STOP_PATIENCE"
    --seed "$TRAIN_SEED"
    --save_every_steps "$SAVE_EVERY_STEPS"
)
if [[ "$TENSORBOARD" == "1" ]]; then
    MONITORING_FLAGS+=(--tensorboard)
else
    MONITORING_FLAGS+=(--no-tensorboard)
fi

if [[ -n "$VAL_DATA" ]]; then
    MONITORING_FLAGS+=(--val_data "$VAL_DATA")
else
    MONITORING_FLAGS+=(--val_split "$VAL_SPLIT")
fi

DO_EXPORT=0
DO_EMBED=1
DO_SANITY=1
PRETRAIN_ONLY=0
FINETUNE_ONLY=0
DRY_RUN=0
AUTO_RESUME="${AUTO_RESUME:-0}"
INIT_CHECKPOINT="${INIT_CHECKPOINT:-}"
USER_REUSE_TOKENIZER=0
if [[ -v REUSE_TOKENIZER && -n "$REUSE_TOKENIZER" ]]; then
    USER_REUSE_TOKENIZER=1
fi
REUSE_TOKENIZER="${REUSE_TOKENIZER:-}"
USER_TRAIN_TOKENIZER=0
if [[ "$TRAIN_TOKENIZER" == "1" ]]; then
    USER_TRAIN_TOKENIZER=1
fi
DO_LATEST_LINK=1

# Set by detect_pipeline_resume: fresh | pretrain_partial | pretrain_done | finetune_partial | complete
PIPELINE_STATE="fresh"
RESUME_CHECKPOINT=""

usage() {
    cat <<'EOF'
NanoLLM MoE training pipeline

Stages:
  1) Pretrain BPE tokenizer + MoE model on FineWeb (or PRETRAIN_DATA)
  2) Fine-tune on chat data (CHAT_DATA) with frozen architecture/tokenizer
  2b) Or: fine-tune on instruction data with INSTRUCT_FINE_TUNING=1
  3) Export PROGMEM header to esp32_m5stack/src/model_weights.h (default, flash-backed)
  4) Optional: export int8 model.bin for SPIFFS/desktop (--export)

Options:
  --help              Show this message
  --profile NAME      Preset: cardputer-mqa-ctx224 (default), cardputer-mqa-ctx208-l20,
                      cardputer-chat, cardputer-chat-wide, cardputer-mqa-wide96,
                      cardputer-mqa-bal192-88, cardputer-autoresearch-d256,
                      cardputer-autoresearch-ctx224, cardputer-dense-capacity,
                      cardputer-dense-mac, cardputer, cardputer-balanced,
                      cardputer-smoke, cardputer-safe, gpu8g-mqa-instruct,
                      quick, desktop
  --pretrain-only     Run stage 1 only
  --finetune-only     Run stage 2 only (requires --init-checkpoint or INIT_CHECKPOINT)
  --init-checkpoint PATH
                      Checkpoint for fine-tune-only mode
  --instruction-finetune
                      Use instruction data (data/instruct/) instead of chat data
                      (data/chat/). Prepares category-balanced training data.
  --no-resume         Start a new moe_run_* directory; do not continue checkpoints/latest
  --no-latest-link    Do not update checkpoints/latest symlink
  --export            Also export weights/model.bin after training (SPIFFS/desktop)
  --no-embed          Skip PROGMEM header export (embed is on by default)
  --no-sanity         Skip post-training generation smoke test
  --train-tokenizer   Fit a fresh BPE instead of the pinned canonical tokenizer
  --dry-run           Print commands without executing

Profiles:
  cardputer-mqa-ctx224  224 ctx, vocab=2048: 18L, d=80, MQA kv=1 (default)
  cardputer-mqa-ctx208-l20  208 ctx, vocab=2048: 20L, d=80, MQA kv=1 (depth probe)
  cardputer-chat        192 ctx, vocab=2048: 44L, d=48 (legacy width-first chat)
  cardputer-chat-wide   112 ctx, vocab=2048: 18L, d=80
    cardputer          384 ctx, vocab=8192: 78L, d=40 (~5.8M params, width/depth compromise)
    cardputer-balanced 464 ctx, vocab=8192: 78L, d=36 (~5.6M params, width/ctx compromise)
    cardputer-legacy   512 ctx, vocab=8192: 124L, d=32 (previous deep/narrow default)
    cardputer-smoke    512 ctx, vocab=8192: 4L d=32 smoke test for on-device boot/OOM check
  cardputer-safe     128 ctx, vocab=512: 12L, d=40 (~463K params)
  gpu8g-mqa-instruct 32K ctx, vocab=4096, RoPE: 24L, d=128, MQA kv=1 (~15.8M params, RTX 3090)
  quick           Tiny 128-context smoke: 4L, d=32, fineweb_tiny + sample_data.txt
  desktop         Wider model for GPU experimentation (not for Cardputer)

Sizing reference:
  python scripts/moe_tradeoff_estimator.py --cardputer-optimal
  python scripts/moe_tradeoff_estimator.py --spiffs-optimal   # SPIFFS/RAM-loaded weights

Environment:
  PRETRAIN_DATA, CHAT_DATA, OUTPUT_DIR, EPOCHS-style vars — see script header.
  AUTO_RESUME=0     Always create a new timestamped moe_run_* directory (default)
  AUTO_RESUME=1     Reuse checkpoints/latest or .active_run when OUTPUT_DIR is unset
  OUTPUT_DIR=...    Pin an explicit run directory (required to resume a specific older run)
  LATEST_LINK=...   Symlink path for active run (default: checkpoints/latest)
  CANONICAL_TOKENIZER_DIR  Pinned vocab=2048 tokenizer (default: data/tokenizers/cardputer_vocab2048_v1)
  TRAIN_TOKENIZER=1        Fit fresh BPE on PRETRAIN_DATA+CHAT_DATA instead of canonical reuse
  REUSE_TOKENIZER=path     Explicit tokenizer dir or tokenizer.json override
EOF
}

canonical_tokenizer_for_profile() {
    case "$PROFILE" in
        cardputer-mqa-*|cardputer-chat*|cardputer-autoresearch-*|cardputer-dense-*|gpu8g-*)
            printf '%s' "$CANONICAL_TOKENIZER_DIR"
            ;;
        *)
            printf '%s' ""
            ;;
    esac
}

ensure_canonical_tokenizer() {
    local canon_dir="$1"
    local source_dir="$2"
    if [[ "$DRY_RUN" == "1" ]]; then
        echo "[dry-run] would ensure canonical tokenizer at $canon_dir"
        return 0
    fi
    if [[ ! -f "$canon_dir/tokenizer.json" ]]; then
        echo "Bootstrapping canonical tokenizer: $canon_dir"
        "$PYTHON_BIN" "$ENSURE_TOKENIZER_PY" "$canon_dir" "$source_dir"
    fi
    if [[ ! -f "$canon_dir/tokenizer.json" ]]; then
        echo "Canonical tokenizer missing after bootstrap: $canon_dir/tokenizer.json" >&2
        exit 1
    fi
}

resolve_pretrain_tokenizer() {
    # Fine-tune stages always inherit tokenizer from --init-checkpoint in train.py.
    if [[ "$FINETUNE_ONLY" == "1" ]]; then
        return 0
    fi

    if [[ "$USER_REUSE_TOKENIZER" == "1" ]]; then
        echo "Tokenizer: explicit REUSE_TOKENIZER=$REUSE_TOKENIZER"
        return 0
    fi

    if [[ "$USER_TRAIN_TOKENIZER" == "1" ]]; then
        REUSE_TOKENIZER=""
        echo "Tokenizer: training fresh BPE (TRAIN_TOKENIZER=1)"
        return 0
    fi

    local canon_dir
    canon_dir="$(canonical_tokenizer_for_profile)"
    if [[ -n "$canon_dir" && "$VOCAB_SIZE" == "2048" ]]; then
        ensure_canonical_tokenizer "$canon_dir" "$CANONICAL_TOKENIZER_SOURCE"
        REUSE_TOKENIZER="$canon_dir"
        echo "Tokenizer: pinned canonical $canon_dir"
        return 0
    fi

    REUSE_TOKENIZER=""
    echo "Tokenizer: training fresh BPE (no canonical for profile=$PROFILE vocab=$VOCAB_SIZE)"
}

normalize_output_dir() {
    if [[ -d "$OUTPUT_DIR" ]]; then
        OUTPUT_DIR="$(cd "$OUTPUT_DIR" && pwd)"
    elif [[ -n "$OUTPUT_DIR" && "$OUTPUT_DIR" != /* ]]; then
        OUTPUT_DIR="$PROJECT_ROOT/$OUTPUT_DIR"
    fi
}

resolve_symlink_path() {
    local path="$1"
    if [[ -z "$path" ]]; then
        return 1
    fi
    if [[ -L "$path" || -d "$path" ]]; then
        local resolved
        resolved="$(readlink -f "$path" 2>/dev/null || true)"
        if [[ -n "$resolved" && -d "$resolved" ]]; then
            printf '%s' "$resolved"
            return 0
        fi
    fi
    return 1
}

new_output_dir() {
    printf '%s/checkpoints/moe_run_%s' "$PROJECT_ROOT" "$(date +%Y%m%d_%H%M%S)"
}

run_has_training_checkpoints() {
    local dir="$1"
    shopt -s nullglob
    local ckpts=("$dir"/model_epoch_*.pt "$dir"/model_step_*.pt "$dir"/model_best.pt "$dir"/model_pretrain.pt)
    shopt -u nullglob
    [[ ${#ckpts[@]} -gt 0 ]]
}

compute_pipeline_state_for_dir() {
    local scan_dir="$1"
    local state="fresh"
    local resume_ckpt=""
    local resume_pretrain_epochs="$PRETRAIN_EPOCHS"
    local resume_chat_epochs="$CHAT_EPOCHS"

    if [[ ! -d "$scan_dir" ]]; then
        printf '%s|||' "$state"
        return 0
    fi

    if [[ -f "$scan_dir/training_complete" ]]; then
        state="complete"
        printf '%s|||' "$state"
        return 0
    fi

    local latest_info latest_epoch latest_ckpt
    latest_info="$(find_latest_epoch_checkpoint "$scan_dir")"
    latest_epoch="${latest_info%%|*}"
    latest_ckpt="${latest_info#*|}"

    local total_target=$((PRETRAIN_EPOCHS + CHAT_EPOCHS))

    if [[ -f "$scan_dir/model_pretrain.pt" ]]; then
        local pretrain_epoch
        pretrain_epoch="$(read_checkpoint_epoch "$scan_dir/model_pretrain.pt")"
        if [[ "$PRETRAIN_ONLY" == "1" ]]; then
            if [[ "$latest_epoch" -ge "$PRETRAIN_EPOCHS" && "$latest_epoch" -gt 0 ]]; then
                state="complete"
            elif [[ "$latest_epoch" -gt 0 && "$latest_epoch" -lt "$PRETRAIN_EPOCHS" ]]; then
                resume_ckpt="$latest_ckpt"
                resume_pretrain_epochs=$((PRETRAIN_EPOCHS - latest_epoch))
                state="pretrain_partial"
            fi
            printf '%s|%s|%s|%s' "$state" "$resume_ckpt" "$resume_pretrain_epochs" "$resume_chat_epochs"
            return 0
        fi

        if [[ "$latest_epoch" -ge "$total_target" && "$latest_epoch" -gt "$pretrain_epoch" ]]; then
            state="complete"
            printf '%s|||' "$state"
            return 0
        fi

        if [[ "$latest_epoch" -gt "$pretrain_epoch" ]]; then
            if [[ "$latest_epoch" -lt "$total_target" && -n "$latest_ckpt" ]]; then
                resume_ckpt="$latest_ckpt"
                resume_chat_epochs=$((total_target - latest_epoch))
                state="finetune_partial"
            else
                state="pretrain_done"
            fi
            printf '%s|%s|%s|%s' "$state" "$resume_ckpt" "$resume_pretrain_epochs" "$resume_chat_epochs"
            return 0
        fi

        state="pretrain_done"
        printf '%s|||' "$state"
        return 0
    fi

    if [[ "$latest_epoch" -ge "$PRETRAIN_EPOCHS" && "$latest_epoch" -gt 0 ]]; then
        if [[ "$PRETRAIN_ONLY" == "1" ]]; then
            state="complete"
        else
            state="pretrain_done"
        fi
        printf '%s|||' "$state"
        return 0
    fi

    if [[ "$latest_epoch" -gt 0 && "$latest_epoch" -lt "$PRETRAIN_EPOCHS" && -n "$latest_ckpt" ]]; then
        resume_ckpt="$latest_ckpt"
        resume_pretrain_epochs=$((PRETRAIN_EPOCHS - latest_epoch))
        state="pretrain_partial"
        printf '%s|%s|%s|%s' "$state" "$resume_ckpt" "$resume_pretrain_epochs" "$resume_chat_epochs"
        return 0
    fi

    if [[ -f "$scan_dir/model_best.pt" ]]; then
        local best_epoch
        best_epoch="$(read_checkpoint_epoch "$scan_dir/model_best.pt")"
        if [[ "$best_epoch" -ge "$PRETRAIN_EPOCHS" ]]; then
            state="pretrain_done"
        elif [[ "$best_epoch" -gt 0 && "$best_epoch" -lt "$PRETRAIN_EPOCHS" ]]; then
            resume_ckpt="$scan_dir/model_best.pt"
            resume_pretrain_epochs=$((PRETRAIN_EPOCHS - best_epoch))
            state="pretrain_partial"
        fi
    fi

    printf '%s|%s|%s|%s' "$state" "$resume_ckpt" "$resume_pretrain_epochs" "$resume_chat_epochs"
}

pipeline_run_incomplete() {
    local dir="$1"
    local state_info state

    [[ -d "$dir" ]] || return 1
    [[ -f "$dir/training_complete" ]] && return 1
    run_has_training_checkpoints "$dir" || return 1

    state_info="$(compute_pipeline_state_for_dir "$dir")"
    state="${state_info%%|*}"
    [[ "$state" != "complete" ]]
}

find_newest_incomplete_run() {
    local dir best_dir="" best_mtime=0 candidate_mtime

    shopt -s nullglob
    for dir in "$PROJECT_ROOT"/checkpoints/moe_run_*/; do
        [[ -d "$dir" ]] || continue
        if ! pipeline_run_incomplete "$dir"; then
            continue
        fi
        candidate_mtime="$(stat -c %Y "$dir" 2>/dev/null || echo 0)"
        if [[ -z "$best_dir" || "$candidate_mtime" -gt "$best_mtime" ]]; then
            best_dir="$(cd "$dir" && pwd)"
            best_mtime="$candidate_mtime"
        fi
    done
    shopt -u nullglob

    if [[ -n "$best_dir" ]]; then
        printf '%s' "$best_dir"
        return 0
    fi
    return 1
}

NEW_OUTPUT_RUN=0

resolve_default_output_dir() {
    local candidate=""

    if [[ "$AUTO_RESUME" != "1" ]]; then
        OUTPUT_DIR="$(new_output_dir)"
        NEW_OUTPUT_RUN=1
        return 0
    fi

    if [[ -f "$ACTIVE_RUN_FILE" ]]; then
        candidate="$(resolve_symlink_path "$(tr -d '[:space:]' < "$ACTIVE_RUN_FILE")" || true)"
        if [[ -n "$candidate" ]] && pipeline_run_incomplete "$candidate"; then
            OUTPUT_DIR="$candidate"
            echo "AUTO_RESUME: continuing in-progress run at $OUTPUT_DIR (via $ACTIVE_RUN_FILE)"
            return 0
        fi
    fi

    candidate="$(resolve_symlink_path "$LATEST_LINK" || true)"
    if [[ -n "$candidate" ]] && pipeline_run_incomplete "$candidate"; then
        OUTPUT_DIR="$candidate"
        echo "AUTO_RESUME: continuing in-progress run at $OUTPUT_DIR (via $LATEST_LINK)"
        return 0
    fi

    candidate="$(find_newest_incomplete_run || true)"
    if [[ -n "$candidate" ]]; then
        OUTPUT_DIR="$candidate"
        echo "AUTO_RESUME: continuing newest in-progress run at $OUTPUT_DIR"
        return 0
    fi

    OUTPUT_DIR="$(new_output_dir)"
    NEW_OUTPUT_RUN=1
}

record_active_run() {
    if [[ "$DRY_RUN" == "1" || -z "$OUTPUT_DIR" || "$NEW_OUTPUT_RUN" != "1" ]]; then
        return 0
    fi
    mkdir -p "$(dirname "$ACTIVE_RUN_FILE")"
    printf '%s\n' "$OUTPUT_DIR" > "$ACTIVE_RUN_FILE"
}

mark_training_complete() {
    if [[ "$DRY_RUN" == "1" || -z "$OUTPUT_DIR" ]]; then
        return 0
    fi
    touch "$OUTPUT_DIR/training_complete"
    if [[ -f "$ACTIVE_RUN_FILE" ]]; then
        local active
        active="$(tr -d '[:space:]' < "$ACTIVE_RUN_FILE")"
        if [[ "$active" == "$OUTPUT_DIR" ]]; then
            rm -f "$ACTIVE_RUN_FILE"
        fi
    fi
}

update_latest_link() {
    if [[ "$DO_LATEST_LINK" != "1" || -z "$LATEST_LINK" ]]; then
        return 0
    fi

    normalize_output_dir
    local link_parent
    link_parent="$(dirname "$LATEST_LINK")"

    if [[ "$DRY_RUN" == "1" ]]; then
        echo "[dry-run] would update latest link: $LATEST_LINK -> $OUTPUT_DIR"
        return 0
    fi

    mkdir -p "$OUTPUT_DIR" "$link_parent"
    OUTPUT_DIR="$(cd "$OUTPUT_DIR" && pwd)"
    ln -sfn "$OUTPUT_DIR" "$LATEST_LINK"
    echo "Latest checkpoint link: $LATEST_LINK -> $OUTPUT_DIR"
}

read_checkpoint_epoch() {
    local ckpt="$1"
    local name="${ckpt##*/}"
    if [[ "$name" =~ ^model_epoch_([0-9]+)\.pt$ ]]; then
        echo "${BASH_REMATCH[1]}"
        return 0
    fi
    "$PYTHON_BIN" - "$ckpt" <<'PY'
import sys
import torch

data = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
print(int(data.get("epoch", 0) or 0))
PY
}

find_latest_epoch_checkpoint() {
    local dir="$1"
    local best_epoch=0
    local best_path=""
    local f epoch from_name
    shopt -s nullglob
    for f in "$dir"/model_epoch_*.pt; do
        from_name=0
        if [[ "${f##*/}" =~ ^model_epoch_([0-9]+)\.pt$ ]]; then
            epoch="${BASH_REMATCH[1]}"
            from_name=1
        else
            epoch="$(read_checkpoint_epoch "$f")"
        fi
        if [[ "$epoch" -ge "$best_epoch" ]]; then
            best_epoch=$epoch
            best_path=$f
        fi
    done
    shopt -u nullglob
    printf '%s|%s' "$best_epoch" "$best_path"
}

ensure_pretrain_baseline() {
    local baseline="$OUTPUT_DIR/model_pretrain.pt"
    local source="${1:-$OUTPUT_DIR/model_best.pt}"
    if [[ "$DRY_RUN" == "1" ]]; then
        echo "[dry-run] would save pretrain baseline: $baseline"
        return 0
    fi
    if [[ -f "$baseline" ]]; then
        return 0
    fi
    require_file "$source" "Need a checkpoint to create model_pretrain.pt"
    cp "$source" "$baseline"
    echo "Saved pretrain baseline: $baseline"
}

detect_pipeline_resume() {
    PIPELINE_STATE="fresh"
    RESUME_CHECKPOINT=""
    RESUME_PRETRAIN_EPOCHS="$PRETRAIN_EPOCHS"
    RESUME_CHAT_EPOCHS="$CHAT_EPOCHS"

    if [[ "$AUTO_RESUME" != "1" || "$FINETUNE_ONLY" == "1" ]]; then
        return 0
    fi

    if [[ ! -d "$OUTPUT_DIR" ]]; then
        return 0
    fi

    local state_info
    state_info="$(compute_pipeline_state_for_dir "$OUTPUT_DIR")"
    PIPELINE_STATE="${state_info%%|*}"
    state_info="${state_info#*|}"
    RESUME_CHECKPOINT="${state_info%%|*}"
    state_info="${state_info#*|}"
    RESUME_PRETRAIN_EPOCHS="${state_info%%|*}"
    RESUME_CHAT_EPOCHS="${state_info#*|}"

    if [[ "$PIPELINE_STATE" == "pretrain_done" && ! -f "$OUTPUT_DIR/model_pretrain.pt" ]]; then
        local latest_info latest_ckpt
        latest_info="$(find_latest_epoch_checkpoint "$OUTPUT_DIR")"
        latest_ckpt="${latest_info#*|}"
        ensure_pretrain_baseline "${latest_ckpt:-$OUTPUT_DIR/model_best.pt}"
    fi
}

apply_profile() {
    # RoPE is gpu8g-only; learned positions are required for ESP32 PROGMEM export.
    USE_ROPE=0
    case "$1" in
        cardputer-chat)
            # Width-first profile for short, useful chat under the no-PSRAM RAM budget.
            VOCAB_SIZE=2048
            D_MODEL=48
            N_HEADS=4
            N_LAYERS=44
            D_FF=192
            MOE_N_EXPERTS=4
            MOE_SHARED_D_FF=96
            BLOCK_SIZE=192
            if [[ "$USER_WINDOW_STRIDE" == "0" ]]; then
                WINDOW_STRIDE_TOKENS=192
            fi
            ;;
        cardputer-chat-wide)
            # Short-chat profile: trade context for a much wider residual stream
            # and fewer sequential layers at similar flash and active MACs.
            VOCAB_SIZE=2048
            D_MODEL=80
            N_HEADS=4
            N_LAYERS=18
            D_FF=320
            MOE_N_EXPERTS=4
            MOE_SHARED_D_FF=160
            BLOCK_SIZE=112
            if [[ "$USER_WINDOW_STRIDE" == "0" ]]; then
                WINDOW_STRIDE_TOKENS=112
            fi
            ;;
        cardputer-mqa-ctx224)
            # Default deploy shape (mqa_ctx224 L=18):
            # longer context + MQA; better PIQA/ARC and near-zero HellaSwag truncation
            # vs S=112 width-first baselines at similar flash. (Promoted 2026-07-15
            # clean-transfer flash build used L=16; training default is now L=18.)
            VOCAB_SIZE=2048
            D_MODEL=80
            N_HEADS=4
            N_LAYERS=18
            D_FF=320
            MOE_N_EXPERTS=4
            MOE_TOP_K=1
            MOE_SHARED_D_FF=160
            BLOCK_SIZE=224
            N_KV_HEADS=1
            if [[ "$USER_WINDOW_STRIDE" == "0" ]]; then
                WINDOW_STRIDE_TOKENS=112
            fi
            ;;
        cardputer-mqa-ctx208-l20)
            # Depth probe: L=20 at D=80 needs S<=208 for ~200 KiB working RAM
            # (S=224 overflows; estimator ~197 KiB at S=208). Same width/MoE as
            # promoted L=18; compare clean-transfer after held-out-safe SFT.
            VOCAB_SIZE=2048
            D_MODEL=80
            N_HEADS=4
            N_LAYERS=20
            D_FF=320
            MOE_N_EXPERTS=4
            MOE_TOP_K=1
            MOE_SHARED_D_FF=160
            BLOCK_SIZE=208
            N_KV_HEADS=1
            if [[ "$USER_WINDOW_STRIDE" == "0" ]]; then
                WINDOW_STRIDE_TOKENS=104
            fi
            ;;
        cardputer-mqa-wide96)
            # MQA width-heavy: spend flash on d_model at short context.
            VOCAB_SIZE=2048
            D_MODEL=96
            N_HEADS=4
            N_LAYERS=18
            D_FF=384
            MOE_N_EXPERTS=4
            MOE_SHARED_D_FF=192
            BLOCK_SIZE=112
            N_KV_HEADS=1
            if [[ "$USER_WINDOW_STRIDE" == "0" ]]; then
                WINDOW_STRIDE_TOKENS=112
            fi
            ;;
        cardputer-mqa-bal192-88)
            # MQA balanced: modest context + modest width under Cardputer budgets.
            VOCAB_SIZE=2048
            D_MODEL=88
            N_HEADS=4
            N_LAYERS=18
            D_FF=352
            MOE_N_EXPERTS=4
            MOE_SHARED_D_FF=176
            BLOCK_SIZE=192
            N_KV_HEADS=1
            if [[ "$USER_WINDOW_STRIDE" == "0" ]]; then
                WINDOW_STRIDE_TOKENS=96
            fi
            ;;
        cardputer-autoresearch-d256)
            # Autoresearch overnight champion (val_bpb 1.6136 @ 5-min budget):
            # wide-shallow MoE, MQA 16→4, shared expert 256. Fits ~94.7% flash.
            VOCAB_SIZE=2048
            D_MODEL=256
            N_HEADS=18
            N_LAYERS=3
            D_FF=768
            MOE_N_EXPERTS=4
            MOE_TOP_K=1
            MOE_SHARED_D_FF=256
            BLOCK_SIZE=112
            N_KV_HEADS=4
            if [[ "$USER_WINDOW_STRIDE" == "0" ]]; then
                WINDOW_STRIDE_TOKENS=112
            fi
            ;;
        cardputer-autoresearch-ctx224)
            # Trade champion width for capacity-matrix context (S=224).
            # Keep L=3 wide-shallow MoE; true MQA (kv=1) for working-RAM headroom.
            VOCAB_SIZE=2048
            D_MODEL=192
            N_HEADS=8
            N_LAYERS=3
            D_FF=640
            MOE_N_EXPERTS=4
            MOE_TOP_K=1
            MOE_SHARED_D_FF=256
            BLOCK_SIZE=224
            N_KV_HEADS=1
            if [[ "$USER_WINDOW_STRIDE" == "0" ]]; then
                WINDOW_STRIDE_TOKENS=112
            fi
            ;;
        cardputer-dense-capacity)
            # Dense control matched to the deployed MoE's total parameter/flash capacity.
            USE_MOE=0
            VOCAB_SIZE=2048
            D_MODEL=80
            N_HEADS=4
            N_LAYERS=16
            D_FF=1444
            BLOCK_SIZE=112
            if [[ "$USER_WINDOW_STRIDE" == "0" ]]; then
                WINDOW_STRIDE_TOKENS=112
            fi
            ;;
        cardputer-dense-mac)
            # Dense control matched to the deployed MoE's active FFN MACs.
            USE_MOE=0
            VOCAB_SIZE=2048
            D_MODEL=80
            N_HEADS=4
            N_LAYERS=16
            D_FF=480
            BLOCK_SIZE=112
            if [[ "$USER_WINDOW_STRIDE" == "0" ]]; then
                WINDOW_STRIDE_TOKENS=112
            fi
            ;;
        cardputer)
            VOCAB_SIZE=8192
            D_MODEL=40
            N_HEADS=4
            N_LAYERS=78
            D_FF=160
            MOE_N_EXPERTS=4
            MOE_SHARED_D_FF=80
            BLOCK_SIZE=384
            ;;
        cardputer-legacy)
            VOCAB_SIZE=8192
            D_MODEL=32
            N_HEADS=4
            N_LAYERS=124
            D_FF=128
            MOE_N_EXPERTS=4
            MOE_SHARED_D_FF=64
            BLOCK_SIZE=512
            ;;
        cardputer-balanced)
            # Wider than cardputer (d=36) while keeping vocab=8192 and near-full context (464).
            # See scripts/moe_tradeoff_estimator.py — max S at d=36/V=8192 is ~464 KiB RAM budget.
            VOCAB_SIZE=8192
            D_MODEL=36
            N_HEADS=4
            N_LAYERS=78
            D_FF=184
            MOE_N_EXPERTS=4
            MOE_SHARED_D_FF=92
            BLOCK_SIZE=464
            ;;
        cardputer-smoke)
            # Fast on-device smoke: full Cardputer vocab/ctx, tiny depth for minutes-not-hours train
            VOCAB_SIZE=8192
            D_MODEL=32
            N_HEADS=4
            N_LAYERS=4
            D_FF=128
            MOE_N_EXPERTS=4
            MOE_SHARED_D_FF=64
            BLOCK_SIZE=512
            PRETRAIN_DATA="$PROJECT_ROOT/data/fineweb/fineweb_tiny.txt"
            CHAT_DATA="$PROJECT_ROOT/data/fineweb/fineweb_tiny.txt"
            if [[ "$USER_PRETRAIN_EPOCHS" == "0" ]]; then
                PRETRAIN_EPOCHS=1
            fi
            if [[ "$USER_CHAT_EPOCHS" == "0" ]]; then
                CHAT_EPOCHS=1
            fi
            PRETRAIN_BATCH_SIZE=32
            CHAT_BATCH_SIZE=16
            ;;
        cardputer-safe)
            VOCAB_SIZE=512
            D_MODEL=40
            N_HEADS=4
            N_LAYERS=12
            D_FF=80
            MOE_N_EXPERTS=4
            MOE_SHARED_D_FF=40
            BLOCK_SIZE=128
            if [[ "$USER_PRETRAIN_EPOCHS" == "0" ]]; then
                PRETRAIN_EPOCHS=3
            fi
            if [[ "$USER_CHAT_EPOCHS" == "0" ]]; then
                CHAT_EPOCHS=2
            fi
            ;;
        gpu8g-mqa-instruct)
            # Scaled-up cardputer-mqa-ctx224 for GPU instruction training (not ESP32):
            # 1.6x width, 1.33x depth, 32K context, vocab=4096, RoPE (no pos table).
            # ~15.8M params (RoPE avoids a 32K learned pos table); batch 2 @ S=32768 ~12 GiB on 3090.
            VOCAB_SIZE=4096
            USE_ROPE=1
            D_MODEL=128
            N_HEADS=4
            N_LAYERS=24
            D_FF=512
            MOE_N_EXPERTS=4
            MOE_TOP_K=1
            MOE_SHARED_D_FF=256
            BLOCK_SIZE=32768
            N_KV_HEADS=1
            if [[ "$USER_WINDOW_STRIDE" == "0" ]]; then
                WINDOW_STRIDE_TOKENS=16384
            fi
            if [[ "$USER_PRETRAIN_EPOCHS" == "0" ]]; then
                PRETRAIN_EPOCHS=8
            fi
            if [[ "$USER_CHAT_EPOCHS" == "0" ]]; then
                CHAT_EPOCHS=12
            fi
            DO_EMBED=0
            ;;
        quick)
            D_MODEL=32
            N_HEADS=4
            N_LAYERS=4
            D_FF=64
            MOE_N_EXPERTS=4
            MOE_SHARED_D_FF=16
            VOCAB_SIZE=512
            BLOCK_SIZE=128
            PRETRAIN_DATA="$PROJECT_ROOT/data/fineweb/fineweb_tiny.txt"
            # Real committed chat corpus (many conversation groups) so the
            # 5% validation split is non-empty. sample_data.txt (1 group)
            # produced an empty val file and crashed TextDataset.
            CHAT_DATA="${QUICK_CHAT_DATA:-$PROJECT_ROOT/data/chat/teacher_distilled_qwen3_5_3k.txt}"
            if [[ "$USER_PRETRAIN_EPOCHS" == "0" ]]; then
                PRETRAIN_EPOCHS=1
            fi
            if [[ "$USER_CHAT_EPOCHS" == "0" ]]; then
                CHAT_EPOCHS=1
            fi
            PRETRAIN_BATCH_SIZE=32
            CHAT_BATCH_SIZE=16
            ;;
        desktop)
            VOCAB_SIZE=512
            D_MODEL=64
            N_HEADS=4
            D_FF=128
            N_LAYERS=8
            MOE_N_EXPERTS=4
            MOE_SHARED_D_FF=32
            BLOCK_SIZE=128
            if [[ "$USER_PRETRAIN_EPOCHS" == "0" ]]; then
                PRETRAIN_EPOCHS=3
            fi
            if [[ "$USER_CHAT_EPOCHS" == "0" ]]; then
                CHAT_EPOCHS=2
            fi
            DO_EMBED=0
            ;;
        *)
            echo "Unknown profile: $1 (use cardputer-chat, cardputer-chat-wide, cardputer-mqa-*, cardputer-mqa-ctx208-l20, cardputer-autoresearch-d256, cardputer-autoresearch-ctx224, cardputer-dense-*, cardputer, cardputer-legacy, cardputer-balanced, cardputer-smoke, cardputer-safe, gpu8g-mqa-instruct, quick, or desktop)" >&2
            exit 1
            ;;
    esac
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --help|-h)
            usage
            exit 0
            ;;
        --profile)
            PROFILE="$2"
            shift 2
            ;;
        --pretrain-only)
            PRETRAIN_ONLY=1
            shift
            ;;
        --finetune-only)
            FINETUNE_ONLY=1
            shift
            ;;
        --init-checkpoint)
            INIT_CHECKPOINT="$2"
            shift 2
            ;;
        --no-resume)
            AUTO_RESUME=0
            shift
            ;;
        --no-latest-link)
            DO_LATEST_LINK=0
            shift
            ;;
        --export)
            DO_EXPORT=1
            shift
            ;;
        --no-embed)
            DO_EMBED=0
            shift
            ;;
        --embed)
            DO_EMBED=1
            shift
            ;;
        --no-sanity)
            DO_SANITY=0
            shift
            ;;
        --train-tokenizer)
            TRAIN_TOKENIZER=1
            USER_TRAIN_TOKENIZER=1
            shift
            ;;
        --instruction-finetune)
            INSTRUCT_FINE_TUNING=1
            shift
            ;;
        --dry-run)
            DRY_RUN=1
            shift
            ;;
        *)
            echo "Unknown option: $1" >&2
            usage >&2
            exit 1
            ;;
    esac
done

apply_profile "$PROFILE"
RESUME_PRETRAIN_EPOCHS="$PRETRAIN_EPOCHS"
RESUME_CHAT_EPOCHS="$CHAT_EPOCHS"

require_file() {
    local path="$1"
    local hint="$2"
    if [[ ! -f "$path" ]]; then
        echo "Missing file: $path" >&2
        [[ -n "$hint" ]] && echo "$hint" >&2
        exit 1
    fi
}

require_checkpoint() {
    if [[ "$DRY_RUN" == "1" ]]; then
        return 0
    fi
    require_file "$1" "${2:-}"
}

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    echo "Python not found (set PYTHON_BIN)." >&2
    exit 1
fi

if [[ "$USE_MOE" == "1" ]]; then
    MOE_FLAGS=(--use_moe --moe_n_experts "$MOE_N_EXPERTS" --moe_top_k "$MOE_TOP_K" --moe_shared_d_ff "$MOE_SHARED_D_FF")
    ESTIMATOR_MODEL_FLAGS=()
else
    MOE_FLAGS=()
    ESTIMATOR_MODEL_FLAGS=(--dense-model)
fi

ROPE_FLAGS=()
if [[ "$USE_ROPE" == "1" ]]; then
    ROPE_FLAGS=(--use_rope)
fi

AMP_FLAGS=(--amp "$AMP")
if [[ "$TF32" == "1" ]]; then
    AMP_FLAGS+=(--tf32)
fi
if [[ -n "$CUDA_DEVICE" ]]; then
    export CUDA_VISIBLE_DEVICES="$CUDA_DEVICE"
fi

common_model_flags=(
    --vocab_size "$VOCAB_SIZE"
    --d_model "$D_MODEL"
    --n_layers "$N_LAYERS"
    --n_heads "$N_HEADS"
    --n_kv_heads "$N_KV_HEADS"
    --d_ff "$D_FF"
    --block_size "$BLOCK_SIZE"
    --dropout "$DROPOUT"
    --num_workers "$NUM_WORKERS"
    --token_cache_chunk_chars "$TOKEN_CACHE_CHUNK_CHARS"
    --window_stride_tokens "$WINDOW_STRIDE_TOKENS"
    "${MOE_FLAGS[@]}"
    "${ROPE_FLAGS[@]}"
    "${MONITORING_FLAGS[@]}"
    "${AMP_FLAGS[@]}"
)

run_cmd() {
    if [[ "$DRY_RUN" == "1" ]]; then
        printf '+'
        printf ' %q' "$@"
        printf '\n'
    else
        "$@"
    fi
}

print_config() {
    cat <<EOF
=== NanoLLM MoE Training Pipeline ===
Profile          : $PROFILE
Output directory : $OUTPUT_DIR
Latest link      : ${LATEST_LINK:-disabled} ($([[ "$DO_LATEST_LINK" == "1" ]] && echo "update on" || echo "off"))
Resume           : auto=$AUTO_RESUME state=$PIPELINE_STATE
Device           : $("$PYTHON_BIN" -c "import torch; print('cuda' if torch.cuda.is_available() else 'cpu')" 2>/dev/null || echo "unknown")
AMP              : $AMP (TF32=$TF32, workers=$NUM_WORKERS, window_stride=$WINDOW_STRIDE_TOKENS)
Pretrain batch   : $PRETRAIN_BATCH_SIZE | Chat batch: $CHAT_BATCH_SIZE
Validation       : $([[ -n "$VAL_DATA" ]] && echo "independent file ($VAL_DATA)" || echo "split=$VAL_SPLIT")
TensorBoard      : $([[ "$TENSORBOARD" == "1" ]] && echo "enabled (output_dir/tensorboard)" || echo "disabled")

Model:
  vocab=$VOCAB_SIZE d_model=$D_MODEL layers=$N_LAYERS heads=$N_HEADS kv_heads=$N_KV_HEADS
  expert_d_ff=$D_FF block_size=$BLOCK_SIZE dropout=$DROPOUT
  moe_experts=$MOE_N_EXPERTS top_k=$MOE_TOP_K shared_d_ff=$MOE_SHARED_D_FF
  position_encoding=$([[ "$USE_ROPE" == "1" ]] && echo rope || echo learned)

Deploy:
  embed=$([[ "$DO_EMBED" == "1" ]] && echo "yes (PROGMEM flash)" || echo "no")
  export_bin=$([[ "$DO_EXPORT" == "1" ]] && echo "yes (SPIFFS/desktop)" || echo "no")

Pretrain:
  data=$PRETRAIN_DATA
  epochs=$RESUME_PRETRAIN_EPOCHS (target=$PRETRAIN_EPOCHS) batch=$PRETRAIN_BATCH_SIZE lr=$PRETRAIN_LR
  resume_checkpoint=${RESUME_CHECKPOINT:-none}

Chat fine-tune:
  data=$CHAT_DATA
  epochs=$RESUME_CHAT_EPOCHS (target=$CHAT_EPOCHS) batch=$CHAT_BATCH_SIZE lr=$CHAT_LR
Tokenizer          : ${REUSE_TOKENIZER:-train fresh on PRETRAIN_DATA+CHAT_DATA}
EOF
    if [[ "$DRY_RUN" == "0" ]]; then
        "$PYTHON_BIN" "$PROJECT_ROOT/scripts/moe_tradeoff_estimator.py" \
            "${ESTIMATOR_MODEL_FLAGS[@]}" \
            --vocab-size "$VOCAB_SIZE" \
            --d-model "$D_MODEL" \
            --d-ff "$D_FF" \
            --n-layers "$N_LAYERS" \
            --n-heads "$N_HEADS" \
            --n-kv-heads "$N_KV_HEADS" \
            --max-seq-len "$BLOCK_SIZE" \
            --moe-experts "$MOE_N_EXPERTS" \
            --moe-shared-d-ff "$MOE_SHARED_D_FF" 2>/dev/null \
            | grep -E 'Cardputer feasible|Estimated int8|working activation|flash' || true
    fi
}

if [[ "$USER_OUTPUT_DIR" == "0" && -z "$OUTPUT_DIR" ]]; then
    resolve_default_output_dir
elif [[ -z "$OUTPUT_DIR" ]]; then
    OUTPUT_DIR="$(new_output_dir)"
fi
normalize_output_dir

detect_pipeline_resume
run_cmd mkdir -p "$OUTPUT_DIR"
record_active_run
update_latest_link
resolve_pretrain_tokenizer
print_config

PRETRAIN_CHECKPOINT=""
BEST_CHECKPOINT=""
SKIP_PRETRAIN=0
SKIP_FINETUNE=0
PRETRAIN_INIT_ARGS=()
if [[ -n "$REUSE_TOKENIZER" && "$FINETUNE_ONLY" == "0" ]]; then
    PRETRAIN_INIT_ARGS+=(--reuse_tokenizer "$REUSE_TOKENIZER")
fi
CHAT_INIT_CHECKPOINT=""

# Fresh checkouts have no data/ — auto-bootstrap the tiny FineWeb slice when a
# quick/smoke profile wants it (cheap: 1000 docs, sliced from fineweb.txt if
# present, otherwise streamed from HuggingFace).
if [[ "$DRY_RUN" != "1" && "$SKIP_PRETRAIN" != "1" ]]; then
    tiny_path="$PROJECT_ROOT/data/fineweb/fineweb_tiny.txt"
    if [[ ! -f "$tiny_path" && ( "$PRETRAIN_DATA" == *fineweb_tiny* || "$CHAT_DATA" == *fineweb_tiny* ) ]]; then
        echo "=== Bootstrapping fineweb_tiny.txt (fresh checkout) ==="
        run_cmd "$PYTHON_BIN" "$SCRIPT_DIR/ensure_finetune_data.py" --tiny
    fi
fi

if [[ "$FINETUNE_ONLY" == "1" ]]; then
    if [[ -z "$INIT_CHECKPOINT" ]]; then
        echo "--finetune-only requires --init-checkpoint or INIT_CHECKPOINT" >&2
        exit 1
    fi
    require_file "$INIT_CHECKPOINT" ""
    PRETRAIN_CHECKPOINT="$INIT_CHECKPOINT"
    SKIP_PRETRAIN=1
else
    require_file "$PRETRAIN_DATA" "Regenerate with scripts/ensure_finetune_data.py (--tiny/--3x) or scripts/download_fineweb.py, or set PRETRAIN_DATA."
    # Allow continuing pretrain from an existing checkpoint into a new OUTPUT_DIR
    # (e.g. extend epochs 5→8 without mutating the original run).
    if [[ -n "$INIT_CHECKPOINT" && "$SKIP_PRETRAIN" == "0" ]]; then
        require_file "$INIT_CHECKPOINT" ""
        PRETRAIN_INIT_ARGS+=(--init_checkpoint "$INIT_CHECKPOINT")
    fi
fi

if [[ "$PRETRAIN_ONLY" == "0" || "$FINETUNE_ONLY" == "1" ]]; then
    if [[ "$INSTRUCT_FINE_TUNING" == "1" ]]; then
        require_file "$INSTRUCT_RAW" "Set INSTRUCT_RAW or run scripts/download_instruction_datasets.py + prepare_instruction_data.py."
    else
        require_file "$CHAT_DATA" "Regenerate: python scripts/distill_short_chat.py (teacher model + GPU) or scripts/download_chat_datasets.py, or set CHAT_DATA. Status: python scripts/ensure_finetune_data.py --report."
    fi
fi

case "$PIPELINE_STATE" in
    complete)
        echo
        echo "=== Resume: training already complete in $OUTPUT_DIR — skipping train stages ==="
        SKIP_PRETRAIN=1
        SKIP_FINETUNE=1
        BEST_CHECKPOINT="$OUTPUT_DIR/model_best.pt"
        require_checkpoint "$BEST_CHECKPOINT" ""
        ;;
    pretrain_done)
        echo
        echo "=== Resume: pretrain complete — skipping stage 1 ==="
        SKIP_PRETRAIN=1
        PRETRAIN_CHECKPOINT="$OUTPUT_DIR/model_pretrain.pt"
        require_checkpoint "$PRETRAIN_CHECKPOINT" ""
        ;;
    pretrain_partial)
        echo
        echo "=== Resume: pretrain from epoch checkpoint ==="
        echo "  checkpoint: $RESUME_CHECKPOINT"
        echo "  remaining pretrain epochs: $RESUME_PRETRAIN_EPOCHS / $PRETRAIN_EPOCHS"
        if [[ -z "$RESUME_CHECKPOINT" ]]; then
            echo "No epoch checkpoint found; falling back to model_best.pt" >&2
            RESUME_CHECKPOINT="$OUTPUT_DIR/model_best.pt"
        fi
        require_checkpoint "$RESUME_CHECKPOINT" ""
        PRETRAIN_INIT_ARGS=(--init_checkpoint "$RESUME_CHECKPOINT")
        ;;
    finetune_partial)
        echo
        echo "=== Resume: chat fine-tune from epoch checkpoint ==="
        echo "  checkpoint: $RESUME_CHECKPOINT"
        echo "  remaining chat epochs: $RESUME_CHAT_EPOCHS / $CHAT_EPOCHS"
        require_checkpoint "$RESUME_CHECKPOINT" ""
        SKIP_PRETRAIN=1
        PRETRAIN_CHECKPOINT="$OUTPUT_DIR/model_pretrain.pt"
        require_checkpoint "$PRETRAIN_CHECKPOINT" ""
        CHAT_INIT_CHECKPOINT="$RESUME_CHECKPOINT"
        ;;
esac

if [[ "$FINETUNE_ONLY" == "0" && "$SKIP_PRETRAIN" == "0" ]]; then
    if [[ "$RESUME_PRETRAIN_EPOCHS" -le 0 ]]; then
        echo "Pretrain already reached target ($PRETRAIN_EPOCHS epochs)." >&2
        SKIP_PRETRAIN=1
        ensure_pretrain_baseline "$OUTPUT_DIR/model_best.pt"
        PRETRAIN_CHECKPOINT="$OUTPUT_DIR/model_pretrain.pt"
    else
        echo
        if [[ " ${PRETRAIN_INIT_ARGS[*]} " == *" --init_checkpoint "* ]]; then
            echo "=== Stage 1/2: MoE pretrain on FineWeb (resuming) ==="
        else
            echo "=== Stage 1/2: MoE pretrain on FineWeb ==="
        fi
        run_cmd mkdir -p "$OUTPUT_DIR"

        run_cmd "$PYTHON_BIN" "$TRAIN_PY" \
            --data "$PRETRAIN_DATA" \
            --tokenizer_data "$PRETRAIN_DATA" "$CHAT_DATA" \
            --output_dir "$OUTPUT_DIR" \
            --epochs "$RESUME_PRETRAIN_EPOCHS" \
            --batch_size "$PRETRAIN_BATCH_SIZE" \
            --learning_rate "$PRETRAIN_LR" \
            --save_every "$PRETRAIN_SAVE_EVERY" \
            "${PRETRAIN_INIT_ARGS[@]}" \
            "${common_model_flags[@]}"

        PRETRAIN_CHECKPOINT="$OUTPUT_DIR/model_best.pt"
        require_checkpoint "$PRETRAIN_CHECKPOINT" ""

        ensure_pretrain_baseline "$PRETRAIN_CHECKPOINT"
    fi
fi

if [[ "$PRETRAIN_ONLY" == "0" && "$SKIP_FINETUNE" == "0" ]]; then
    if [[ -z "$CHAT_INIT_CHECKPOINT" ]]; then
        CHAT_INIT_CHECKPOINT="${PRETRAIN_CHECKPOINT:-$INIT_CHECKPOINT}"
    fi
    if [[ -z "$CHAT_INIT_CHECKPOINT" ]]; then
        echo "Chat fine-tune requires a pretrain checkpoint (model_pretrain.pt)." >&2
        exit 1
    fi
    if [[ "$RESUME_CHAT_EPOCHS" -le 0 && "$PIPELINE_STATE" != "pretrain_done" && "$PIPELINE_STATE" != "fresh" ]]; then
        if [[ "$INSTRUCT_FINE_TUNING" == "1" ]]; then
            echo "Instruction fine-tune already reached target ($CHAT_EPOCHS epochs)." >&2
        else
            echo "Chat fine-tune already reached target ($CHAT_EPOCHS epochs)." >&2
        fi
    else
        echo
        if [[ "$PIPELINE_STATE" == "finetune_partial" ]]; then
            if [[ "$INSTRUCT_FINE_TUNING" == "1" ]]; then
                echo "=== Stage 2/2: Instruction fine-tune (resuming) ==="
            else
                echo "=== Stage 2/2: Chat fine-tune (resuming) ==="
            fi
        else
            if [[ "$INSTRUCT_FINE_TUNING" == "1" ]]; then
                echo "=== Stage 2/2: Instruction fine-tune ==="
            else
                echo "=== Stage 2/2: Chat fine-tune ==="
            fi
        fi

        # Select data source and prepare script based on INSTRUCT_FINE_TUNING flag.
        if [[ "$INSTRUCT_FINE_TUNING" == "1" ]]; then
            echo "=== Preparing instruction train/validation data ==="
            run_cmd "$PYTHON_BIN" "$PREPARE_INSTRUCT_PY" \
                --input "$INSTRUCT_RAW" \
                --seed-data "$INSTRUCT_SEED" \
                --train-output "$INSTRUCT_TRAIN" \
                --val-output "$INSTRUCT_VAL" \
                --seed-repeats "$INSTRUCT_SEED_REPEATS"
            FINETUNE_DATA="$INSTRUCT_TRAIN"
            VAL_DATA="$INSTRUCT_VAL"
        else
            echo "=== Preparing chat train/validation data ==="
            CHAT_PREPARE_FLAGS=()
            if [[ -n "$CHAT_EXCLUDE_PROMPTS_FILE" ]]; then
                CHAT_PREPARE_FLAGS+=(
                    --exclude-prompts-file "$CHAT_EXCLUDE_PROMPTS_FILE"
                    --exclude-similarity-threshold "$CHAT_EXCLUDE_SIMILARITY_THRESHOLD"
                )
            fi
            run_cmd "$PYTHON_BIN" "$PREPARE_CHAT_PY" \
                --input "$CHAT_DATA" \
                --seed-data "$CHAT_SEED" \
                --train-output "$CHAT_TRAIN" \
                --val-output "$CHAT_VAL" \
                --seed-repeats "$CHAT_SEED_REPEATS" \
                "${CHAT_PREPARE_FLAGS[@]}"
            FINETUNE_DATA="$CHAT_TRAIN"
            VAL_DATA="$CHAT_VAL"
        fi
        EFFECTIVE_CHAT_LR="$CHAT_LR"
        if [[ "$FINETUNE_ONLY" == "1" && "$USER_CHAT_LR" == "0" ]]; then
            EFFECTIVE_CHAT_LR="$CHAT_LR_CONTINUE"
            echo "Using continued fine-tune LR: $EFFECTIVE_CHAT_LR"
        fi
        CHAT_TRAIN_FLAGS=(--chat_eval --chat_eval_greedy)
        if [[ "$CHAT_EVAL" == "0" ]]; then
            CHAT_TRAIN_FLAGS=()
        fi
        if [[ "$CHAT_EVAL_SELECT" == "1" ]]; then
            CHAT_TRAIN_FLAGS+=(--chat_eval_select)
        fi
        if [[ "$CHAT_EVAL_REQUIRE_HARD" == "1" ]]; then
            CHAT_TRAIN_FLAGS+=(--chat_eval_require_hard)
        fi
        if [[ "$CHAT_ASSISTANT_ONLY" == "1" ]]; then
            CHAT_TRAIN_FLAGS+=(--assistant_only_loss)
        fi
        # Add instruction benchmark when the flag is set.
        if [[ "$INSTRUCT_FINE_TUNING" == "1" ]]; then
            CHAT_TRAIN_FLAGS+=(--instruct_benchmark)
        fi
        CHAT_MONITORING_FLAGS=("${MONITORING_FLAGS[@]}")
        CHAT_MONITORING_FLAGS+=(--val_data "$VAL_DATA")
        chat_model_flags=(
            --vocab_size "$VOCAB_SIZE"
            --d_model "$D_MODEL"
            --n_layers "$N_LAYERS"
            --n_heads "$N_HEADS"
            --n_kv_heads "$N_KV_HEADS"
            --d_ff "$D_FF"
            --block_size "$BLOCK_SIZE"
            --dropout "$DROPOUT"
            --num_workers "$NUM_WORKERS"
            --token_cache_chunk_chars "$TOKEN_CACHE_CHUNK_CHARS"
            --window_stride_tokens "$WINDOW_STRIDE_TOKENS"
            "${MOE_FLAGS[@]}"
            "${CHAT_MONITORING_FLAGS[@]}"
            "${AMP_FLAGS[@]}"
        )
        run_cmd "$PYTHON_BIN" "$TRAIN_PY" \
            --data "$FINETUNE_DATA" \
            --output_dir "$OUTPUT_DIR" \
            --epochs "$RESUME_CHAT_EPOCHS" \
            --batch_size "$CHAT_BATCH_SIZE" \
            --learning_rate "$EFFECTIVE_CHAT_LR" \
            --save_every "$CHAT_SAVE_EVERY" \
            --init_checkpoint "$CHAT_INIT_CHECKPOINT" \
            "${CHAT_TRAIN_FLAGS[@]}" \
            "${chat_model_flags[@]}"

        BEST_CHECKPOINT="$OUTPUT_DIR/model_best.pt"
        require_checkpoint "$BEST_CHECKPOINT" ""
    fi
fi

if [[ "$PRETRAIN_ONLY" == "1" ]]; then
    BEST_CHECKPOINT="$OUTPUT_DIR/model_best.pt"
fi

if [[ -z "$BEST_CHECKPOINT" && -f "$OUTPUT_DIR/model_best.pt" ]]; then
    BEST_CHECKPOINT="$OUTPUT_DIR/model_best.pt"
fi

if [[ "$DO_EMBED" == "1" && -n "$BEST_CHECKPOINT" ]]; then
    echo
    echo "=== Export PROGMEM weights (firmware flash, zero RAM copy) ==="
    run_cmd "$PYTHON_BIN" "$EMBED_HEADER_PY" \
        --checkpoint "$BEST_CHECKPOINT" \
        --output "$MODEL_HEADER"
    TOKENIZER_JSON="$OUTPUT_DIR/tokenizer/tokenizer.json"
    if [[ -f "$TOKENIZER_JSON" ]]; then
        echo "=== Export PROGMEM vocab ==="
        run_cmd "$PYTHON_BIN" "$EMBED_VOCAB_PY" \
            --tokenizer "$TOKENIZER_JSON" \
            --output "$VOCAB_HEADER"
    else
        echo "Warning: tokenizer not found at $TOKENIZER_JSON — skipping vocab embed" >&2
    fi
    echo "Embedded header: $MODEL_HEADER"
    echo "Embedded vocab:  $VOCAB_HEADER"
    echo "Flash partition: esp32_m5stack/partitions_embedded.csv (~7 MiB app)"
    echo "Build/upload: cd esp32_m5stack && pio run -e m5stack_cardputer_nopsram -t upload"
fi

if [[ "$DO_EXPORT" == "1" && -n "$BEST_CHECKPOINT" ]]; then
    echo
    echo "=== Export int8 weights ==="
    WEIGHTS_DIR="$PROJECT_ROOT/weights"
    run_cmd mkdir -p "$WEIGHTS_DIR"
    run_cmd "$PYTHON_BIN" "$EXPORT_PY" \
        --checkpoint "$BEST_CHECKPOINT" \
        --output "$WEIGHTS_DIR/model.bin" \
        --allow-moe-export
    echo "Exported: $WEIGHTS_DIR/model.bin"
    echo "Config:   $WEIGHTS_DIR/model_config.json (if written by exporter)"
fi

if [[ "$DO_SANITY" == "1" && -n "$BEST_CHECKPOINT" && "$DRY_RUN" == "0" ]]; then
    if [[ "$CHAT_EVAL" != "0" && -f "$EVAL_CHAT_PY" ]]; then
        echo
        echo "=== Sanity check: greedy chat prompt suite ==="
        "$PYTHON_BIN" "$EVAL_CHAT_PY" \
            --checkpoint "$BEST_CHECKPOINT" \
            --output "$OUTPUT_DIR/chat_eval_latest.json" \
            --greedy \
            || echo "Warning: chat eval failed (checkpoint still saved)."
    fi
fi

if [[ "$DRY_RUN" == "0" && -n "$BEST_CHECKPOINT" ]]; then
    final_state="$(compute_pipeline_state_for_dir "$OUTPUT_DIR")"
    if [[ "${final_state%%|*}" == "complete" ]]; then
        mark_training_complete
    fi
    update_latest_link
    echo
    echo "=== Training complete ==="
    echo "Best checkpoint : $BEST_CHECKPOINT"
    echo "Latest link     : $LATEST_LINK -> $OUTPUT_DIR"
    echo "Tokenizer       : $OUTPUT_DIR/tokenizer/tokenizer.json"
    echo
    echo "Try qualitative testing:"
    echo "  $PYTHON_BIN scripts/interactive_chat.py --checkpoint $LATEST_LINK/model_best.pt"
    if [[ "$DO_EMBED" == "1" ]]; then
        echo
        echo "Flash firmware (embedded weights already exported):"
        echo "  cd esp32_m5stack && pio run -e m5stack_cardputer_nopsram -t upload"
    fi
    if [[ "$DO_EXPORT" == "0" ]]; then
        echo
        echo "Optional SPIFFS/desktop export:"
        echo "  $PYTHON_BIN $EXPORT_PY --checkpoint $BEST_CHECKPOINT --output weights/model.bin --allow-moe-export"
    fi
    if [[ "$DO_EMBED" == "0" ]]; then
        echo
        echo "Embed weights in firmware flash:"
        echo "  $PYTHON_BIN $EMBED_HEADER_PY --checkpoint $BEST_CHECKPOINT --output $MODEL_HEADER"
    fi
fi
