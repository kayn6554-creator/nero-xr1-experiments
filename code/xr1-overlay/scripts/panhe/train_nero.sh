#!/usr/bin/env bash
# Launch one NERO training run from /mnt/panhe/xr1 with the baseline hyperparameters.
#   bash scripts/panhe/train_nero.sh <exp_name> [extra hydra overrides...]
# Examples:
#   bash scripts/panhe/train_nero.sh m1-step-prompt +data.params.train_datasets.step_prompt=true
#   XR1_CHECK_ONLY=1 bash scripts/panhe/train_nero.sh m1-step-prompt   # plan only, no NPU
# 4 chips only: ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 XR1_ACCUMULATE_GRAD_BATCHES=2 (effective batch stays 80)
set -eo pipefail
EXP=${1:?usage: train_nero.sh <exp_name> [hydra overrides...]}; shift

export XR1_DIR=/mnt/panhe/xr1
cd "$XR1_DIR"
unset XR1_CKPT_PATH WORLD_SIZE RANK LOCAL_RANK MASTER_ADDR
export TASK_QUEUE_ENABLE=0 TORCH_CPP_LOG_LEVEL=ERROR
export TE_PARALLEL_COMPILER=16 MAX_COMPILE_CORE_NUMBER=8 CPU_AFFINITY_CONF=2
export MASTER_PORT=${XR1_MASTER_PORT:-29731}          # baseline used 29631
export ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export XR1_DATA_CONFIG=${XR1_DATA_CONFIG:-nero3_single_20261031}
export XR1_MODEL_CONFIG=${XR1_MODEL_CONFIG:-posttrain_nero_joint_full}
export XR1_EPOCHS=${XR1_EPOCHS:-3} XR1_BATCH_SIZE=10 XR1_NUM_WORKERS=8
export XR1_ACCUMULATE_GRAD_BATCHES=${XR1_ACCUMULATE_GRAD_BATCHES:-1}
export XR1_DISABLE_CHECKPOINTS=0 XR1_SAVE_EPOCHS=${XR1_SAVE_EPOCHS:-3} XR1_SAVE_TOP_K=-1
export XR1_OUTPUT_ROOT=${XR1_OUTPUT_ROOT:-/mnt/panhe/output/$EXP}

# Smoke test: a few steps, no checkpoint, throwaway output dir.
if [[ -n ${XR1_SMOKE_STEPS:-} ]]; then
  unset XR1_EPOCHS
  export XR1_DISABLE_CHECKPOINTS=1 XR1_OUTPUT_ROOT=/mnt/panhe/output/smoke-$EXP-$(date +%m%d-%H%M%S)
  mkdir -p /mnt/panhe/output
  bash scripts/train_910c_full_joint.sh "$XR1_SMOKE_STEPS" "smoke-$EXP" "$@" 2>&1 | tee "$XR1_OUTPUT_ROOT.log"
  exit
fi
# Fixed step budget (M2: dwell thinning makes the JSON frame count != samples):
#   XR1_MAX_STEPS=<n> keeps finite epochs but uses n for max_steps and the LR schedule.
STEPS=auto; EXTRA=()
if [[ -n ${XR1_MAX_STEPS:-} ]]; then
  STEPS=$XR1_MAX_STEPS; EXTRA=("+data.params.finite_epoch=true" "+trainer.max_epochs=$XR1_EPOCHS"); unset XR1_EPOCHS
fi
if [[ ${XR1_CHECK_ONLY:-0} == 1 ]]; then
  exec bash scripts/train_910c_full_joint.sh "$STEPS" "$EXP" "${EXTRA[@]}" "$@"
fi
if [[ -e "$XR1_OUTPUT_ROOT" || -e "$XR1_OUTPUT_ROOT.log" ]]; then
  echo "Output already exists: $XR1_OUTPUT_ROOT" >&2; exit 2
fi
mkdir -p "$(dirname "$XR1_OUTPUT_ROOT")"
git -C "$XR1_DIR" rev-parse --short HEAD > "$XR1_OUTPUT_ROOT.commit"
git -C "$XR1_DIR" status --short >> "$XR1_OUTPUT_ROOT.commit"
bash scripts/train_910c_full_joint.sh "$STEPS" "$EXP" "${EXTRA[@]}" "$@" 2>&1 | tee "$XR1_OUTPUT_ROOT.log"
