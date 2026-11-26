#!/usr/bin/env bash
# 8-card Ascend 910C BF16 full-parameter Piper-joint training.

set -eo pipefail

XR1_DIR=${XR1_DIR:-/mnt/wzh/XPolicyLab/policy/Xiaomi_Robotics_1/xiaomi_robotics_1/xr1}
MAX_STEPS=${1:-auto}
DATA_CONFIG=${XR1_DATA_CONFIG:-real4_evo_v0_master_20261021}
MODEL_CONFIG=${XR1_MODEL_CONFIG:-posttrain_piper_joint_full}
EXP_NAME=${2:-real4-evo-v0-master-2epochs}
OUTPUT_ROOT=${XR1_OUTPUT_ROOT:-/mnt/wzh/output/xr1-910c-real4-evo-v0-master-2epochs-20261021}
SAVE_INTERVAL=${XR1_SAVE_INTERVAL:-1000}
BATCH_SIZE=${XR1_BATCH_SIZE:-8}

if [[ "$MAX_STEPS" != "auto" && ! "$MAX_STEPS" =~ ^[1-9][0-9]*$ ]]; then
  echo "MAX_STEPS must be auto or a positive integer, got: $MAX_STEPS" >&2
  exit 2
fi
if [[ ! "$SAVE_INTERVAL" =~ ^[1-9][0-9]*$ ]]; then
  echo "XR1_SAVE_INTERVAL must be a positive integer, got: $SAVE_INTERVAL" >&2
  exit 2
fi
if [[ ! "$BATCH_SIZE" =~ ^[1-9][0-9]*$ ]]; then
  echo "XR1_BATCH_SIZE must be a positive integer, got: $BATCH_SIZE" >&2
  exit 2
fi

cd "$XR1_DIR"
source /usr/local/Ascend/cann-9.0.0/set_env.sh

export XR1_DEVICE_TYPE=npu
export ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
IFS=, read -ra visible_cards <<< "$ASCEND_RT_VISIBLE_DEVICES"
export RESOURCE_GPU=${#visible_cards[@]}
ACCUMULATION=${XR1_ACCUMULATE_GRAD_BATCHES:-1}
if [[ ! "$ACCUMULATION" =~ ^[1-9][0-9]*$ ]]; then
  echo "XR1_ACCUMULATE_GRAD_BATCHES must be positive" >&2; exit 2
fi
export XR1_PYTHON=.venv/bin/python
export XR1_BASE_VLM=/mnt/wzh/checkpoints/Qwen3-VL-4B-Instruct
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export XR1_NUM_WORKERS=${XR1_NUM_WORKERS:-1}
export MAX_LENGTH=${MAX_LENGTH:-20000}
export WANDB_MODE=${WANDB_MODE:-offline}
export WANDB_DIR=${WANDB_DIR:-$OUTPUT_ROOT/wandb}
export XR1_SAVE_TOP_K=${XR1_SAVE_TOP_K:-1}
export XR1_DISABLE_CHECKPOINTS=${XR1_DISABLE_CHECKPOINTS:-0}

# Default to finite epochs; retain explicit numeric step mode for legacy commands.
if [[ "$MAX_STEPS" == "auto" ]]; then
  export XR1_EPOCHS=${XR1_EPOCHS:-2}
  if [[ ! "$XR1_EPOCHS" =~ ^[1-9][0-9]*$ ]]; then
    echo "XR1_EPOCHS must be a positive integer" >&2; exit 2
  fi
  NODES=${WORLD_SIZE:-1}
  if [[ ! "$NODES" =~ ^[1-9][0-9]*$ ]]; then
    echo "WORLD_SIZE (node count for train.sh) must be positive" >&2; exit 2
  fi
  plan_counts=$(.venv/bin/python scripts/plan_json_epochs.py \
    --data-config "$DATA_CONFIG" --epochs "$XR1_EPOCHS" \
    --batch-size "$BATCH_SIZE" --world-size "$((RESOURCE_GPU * NODES))" \
    --accumulation "$ACCUMULATION" --counts)
  read -r DATA_FRAMES STEPS_PER_EPOCH MAX_STEPS DATA_EPISODES <<< "$plan_counts"
  printf 'epoch plan: episodes=%s frames=%s epochs=%s steps_per_epoch=%s max_steps=%s accumulation=%s\n' \
    "$DATA_EPISODES" "$DATA_FRAMES" "$XR1_EPOCHS" "$STEPS_PER_EPOCH" "$MAX_STEPS" "$ACCUMULATION"
elif [[ -n "${XR1_EPOCHS:-}" ]]; then
  echo "Use auto (or omit steps) with XR1_EPOCHS; do not combine epochs with fixed steps." >&2; exit 2
fi

# Plan inspection must not start torchrun or create a training output directory.
if [[ "${XR1_CHECK_ONLY:-0}" == "1" ]]; then
  printf 'plan: data=%s model=%s cards=%s batch=%s accumulation=%s epochs=%s steps=%s output=%s\n' \
    "$DATA_CONFIG" "$MODEL_CONFIG" "$ASCEND_RT_VISIBLE_DEVICES" "$BATCH_SIZE" "$ACCUMULATION" "${XR1_EPOCHS:-step-bounded}" "$MAX_STEPS" "$OUTPUT_ROOT"
  exit 0
fi
mkdir -p "$OUTPUT_ROOT" "$WANDB_DIR"

extra_overrides=()
if [[ -n "${XR1_EPOCHS:-}" ]]; then
  if [[ ! "$XR1_EPOCHS" =~ ^[1-9][0-9]*$ ]]; then
    echo "XR1_EPOCHS must be a positive integer" >&2; exit 2
  fi
  extra_overrides+=("+data.params.finite_epoch=true" "+trainer.max_epochs=$XR1_EPOCHS")
fi
if [[ -n "${XR1_CKPT_PATH:-}" ]]; then
  extra_overrides+=("trainer.ckpt_path=$XR1_CKPT_PATH")
fi

printf 'XR-1 910C training: steps=%s batch_per_device=%s exp=%s output=%s save_interval=%s checkpoints_disabled=%s resume=%s\n' \
  "$MAX_STEPS" "$BATCH_SIZE" "$EXP_NAME" "$OUTPUT_ROOT" "$SAVE_INTERVAL" \
  "$XR1_DISABLE_CHECKPOINTS" "${XR1_CKPT_PATH:-none}"

exec bash scripts/train.sh \
  "data=$DATA_CONFIG" \
  "model=$MODEL_CONFIG" \
  trainer=deepspeed_npu_bf16_zero2 \
  "trainer.max_steps=$MAX_STEPS" \
  "trainer.save_interval=$SAVE_INTERVAL" \
  "trainer.accumulate_grad_batches=$ACCUMULATION" \
  "data.params.train_datasets.batch_size=$BATCH_SIZE" \
  "trainer.exp_name=$EXP_NAME" \
  "trainer.default_root_dir=$OUTPUT_ROOT" \
  "${extra_overrides[@]}" \
  "${@:3}"
