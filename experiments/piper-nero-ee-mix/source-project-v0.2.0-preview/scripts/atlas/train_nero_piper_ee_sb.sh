#!/usr/bin/env bash
# Safe launcher for NERO-target Piper+NERO EE mixed training.
# Default mode is check-only. Real training requires XR1_ALLOW_TRAINING=1.

set -Eeuo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd -- "$SCRIPT_DIR/../.." && pwd)

XR1_ROOT=${XR1_ROOT:-/mnt/panhe/xr1}
PYTHON_BIN=${XR1_PYTHON:-$XR1_ROOT/.venv-piper-ef/bin/python}
CANN_ENV=${CANN_ENV:-/usr/local/Ascend/cann-9.0.0/set_env.sh}

DATA_CONFIG=nero3_piper_ee_sb_20261125
MODEL_CONFIG=posttrain_nero_ee_sb_full
TRAINER_CONFIG=deepspeed_npu_bf16_zero2

BATCH_SIZE=${XR1_BATCH_SIZE:-10}
ACCUMULATION=${XR1_ACCUMULATE_GRAD_BATCHES:-1}
EPOCHS=${XR1_EPOCHS:-3}
SAVE_INTERVAL=${XR1_SAVE_INTERVAL:-1000}
EXP_NAME=${XR1_EXP_NAME:-nero-piper-ee-sb-v0.1}
PROJECT_NAME=${XR1_PROJECT_NAME:-nero-piper-ee-mix}
OUTPUT_ROOT=${XR1_OUTPUT_ROOT:-/mnt/panhe/output/nero-piper-ee-sb-v0.1}
VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
NODES=${XR1_NODES:-1}
MAX_LENGTH=${MAX_LENGTH:-20000}
CHECK_ONLY=${XR1_CHECK_ONLY:-1}
MAX_STEPS_OVERRIDE=${XR1_MAX_STEPS_OVERRIDE:-}

die() { echo "[ERROR] $*" >&2; exit 2; }

[[ -d "$XR1_ROOT" ]] || die "XR1_ROOT does not exist: $XR1_ROOT"
[[ -x "$PYTHON_BIN" ]] || die "Python environment not found: $PYTHON_BIN"
[[ -f "$CANN_ENV" ]] || die "CANN env script not found: $CANN_ENV"
[[ -f "$XR1_ROOT/scripts/train.sh" ]] || die "Missing XR1 scripts/train.sh"
[[ -f "$XR1_ROOT/tools/train.py" ]] || die "Missing XR1 tools/train.py"

for value in "$BATCH_SIZE" "$ACCUMULATION" "$EPOCHS" "$SAVE_INTERVAL" "$NODES"; do
    [[ "$value" =~ ^[1-9][0-9]*$ ]] || die "Expected positive integer, got: $value"
done

source "$CANN_ENV"

export XR1_DEVICE_TYPE=npu
export ASCEND_RT_VISIBLE_DEVICES="$VISIBLE_DEVICES"
IFS=, read -r -a visible_cards <<< "$VISIBLE_DEVICES"
RESOURCE_GPU=${#visible_cards[@]}
[[ "$RESOURCE_GPU" -gt 0 ]] || die "No visible NPU device listed"
export RESOURCE_GPU
export MASTER_PORT=${XR1_MASTER_PORT:-29732}
unset WORLD_SIZE RANK LOCAL_RANK MASTER_ADDR
[[ "$NODES" == "1" ]] || die "v0.2 launcher supports one Atlas node only; do not set XR1_NODES>1"
export WORLD_SIZE=1 RANK=0 MASTER_ADDR=127.0.0.1
export XR1_PYTHON="$PYTHON_BIN"
export XR1_BASE_VLM=${XR1_BASE_VLM:-/mnt/wzh/checkpoints/Qwen3-VL-4B-Instruct}
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export XR1_NUM_WORKERS=${XR1_NUM_WORKERS:-8}
export MAX_LENGTH
export WANDB_MODE=${WANDB_MODE:-offline}
export WANDB_DIR=${WANDB_DIR:-$OUTPUT_ROOT/wandb}
export XR1_WANDB_SAVE_DIR=${XR1_WANDB_SAVE_DIR:-$WANDB_DIR}
export XR1_SAVE_TOP_K=${XR1_SAVE_TOP_K:-1}
export XR1_DISABLE_CHECKPOINTS=${XR1_DISABLE_CHECKPOINTS:-0}

sync_config() {
    local src="$1"
    local dst="$2"
    [[ -f "$src" ]] || die "Project config missing: $src"
    mkdir -p "$(dirname "$dst")"
    if [[ -f "$dst" ]]; then
        if cmp -s "$src" "$dst"; then
            echo "[OK] XR1 config matches project snapshot: $dst"
        else
            die "XR1 config differs from project snapshot: $dst. Compare manually; launcher will not overwrite it."
        fi
    else
        cp "$src" "$dst"
        echo "[COPIED] $dst"
    fi
}

sync_config "$PROJECT_ROOT/configs/data/nero3_piper_ee_sb_20261125.yaml" \
    "$XR1_ROOT/configs/data/nero3_piper_ee_sb_20261125.yaml"
sync_config "$PROJECT_ROOT/configs/model/posttrain_nero_ee_sb_full.yaml" \
    "$XR1_ROOT/configs/model/posttrain_nero_ee_sb_full.yaml"
sync_config "$PROJECT_ROOT/configs/trainer/deepspeed_npu_bf16_zero2.yaml" \
    "$XR1_ROOT/configs/trainer/deepspeed_npu_bf16_zero2.yaml"

if [[ -n "${XR1_CKPT_PATH:-}" ]]; then
    [[ -e "$XR1_CKPT_PATH" ]] || die "XR1_CKPT_PATH does not exist: $XR1_CKPT_PATH"
fi

WORLD_SIZE_TOTAL=$((RESOURCE_GPU * NODES))

if [[ -n "$MAX_STEPS_OVERRIDE" ]]; then
    [[ "$MAX_STEPS_OVERRIDE" =~ ^[1-9][0-9]*$ ]] || die "XR1_MAX_STEPS_OVERRIDE must be positive"
    MAX_STEPS="$MAX_STEPS_OVERRIDE"
    EPOCH_MODE=0
    unset XR1_SAVE_EPOCHS
    export XR1_SAVE_TOP_K=${XR1_SAVE_TOP_K:-1}
    echo "[PLAN] fixed-step mode: max_steps=$MAX_STEPS (epoch-selected checkpointing disabled)"
else
    PLAN=$("$PYTHON_BIN" "$PROJECT_ROOT/scripts/validation/plan_training_steps.py" \
        --config "$PROJECT_ROOT/configs/data/nero3_piper_ee_sb_20261125.yaml" \
        --epochs "$EPOCHS" \
        --batch-size "$BATCH_SIZE" \
        --world-size "$WORLD_SIZE_TOTAL" \
        --accumulation "$ACCUMULATION" \
        --counts)
    read -r EFFECTIVE_FRAMES STEPS_PER_EPOCH MAX_STEPS DATA_EPISODES <<< "$PLAN"
    EPOCH_MODE=1
    export XR1_SAVE_EPOCHS=${XR1_SAVE_EPOCHS:-$EPOCHS}
    export XR1_SAVE_TOP_K=${XR1_SAVE_TOP_K:--1}
    echo "[PLAN] episodes=$DATA_EPISODES effective_frames=$EFFECTIVE_FRAMES"
    echo "[PLAN] steps_per_epoch=$STEPS_PER_EPOCH epochs=$EPOCHS max_steps=$MAX_STEPS"
fi

echo "[PLAN] XR1_ROOT=$XR1_ROOT"
echo "[PLAN] Python=$PYTHON_BIN"
echo "[PLAN] NPU=$ASCEND_RT_VISIBLE_DEVICES (world_size=$WORLD_SIZE_TOTAL)"
echo "[PLAN] batch_per_device=$BATCH_SIZE accumulation=$ACCUMULATION"
echo "[PLAN] model=$MODEL_CONFIG data=$DATA_CONFIG"
echo "[PLAN] output=$OUTPUT_ROOT"
echo "[PLAN] selected_checkpoint_epochs=${XR1_SAVE_EPOCHS:-periodic} checkpoint_top_k=${XR1_SAVE_TOP_K} resume=${XR1_CKPT_PATH:-none}"

if [[ "$CHECK_ONLY" == "1" ]]; then
    echo "[CHECK ONLY] No torchrun started and no training output directory created."
    exit 0
fi

[[ "${XR1_ALLOW_TRAINING:-0}" == "1" ]] || die \
    "Training is blocked. After checking NPU availability, set XR1_ALLOW_TRAINING=1 explicitly."
[[ "$XR1_DISABLE_CHECKPOINTS" == "0" ]] || die "Formal training must keep checkpoints enabled."

mkdir -p "$OUTPUT_ROOT" "$WANDB_DIR"

extra_overrides=()
if [[ "$EPOCH_MODE" == "1" ]]; then
    extra_overrides+=("+data.params.finite_epoch=true" "+trainer.max_epochs=$EPOCHS")
fi
if [[ -n "${XR1_CKPT_PATH:-}" ]]; then
    extra_overrides+=("trainer.ckpt_path=$XR1_CKPT_PATH")
fi

exec bash "$XR1_ROOT/scripts/train.sh" \
    "data=$DATA_CONFIG" \
    "model=$MODEL_CONFIG" \
    "trainer=$TRAINER_CONFIG" \
    "trainer.max_steps=$MAX_STEPS" \
    "trainer.save_interval=$SAVE_INTERVAL" \
    "trainer.accumulate_grad_batches=$ACCUMULATION" \
    "data.params.train_datasets.batch_size=$BATCH_SIZE" \
    "trainer.project=$PROJECT_NAME" \
    "trainer.exp_name=$EXP_NAME" \
    "trainer.default_root_dir=$OUTPUT_ROOT" \
    "${extra_overrides[@]}" \
    "${@:1}"