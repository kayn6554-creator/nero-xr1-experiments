#!/usr/bin/env bash
# Nero only: 3 finite epochs, no RTC, save completed epochs 2 and 3.
set -eo pipefail
cd /mnt/wzh/XPolicyLab/policy/Xiaomi_Robotics_1/xiaomi_robotics_1/xr1
unset XR1_CKPT_PATH XR1_EPOCHS
unset WORLD_SIZE RANK LOCAL_RANK MASTER_ADDR MASTER_PORT
export TASK_QUEUE_ENABLE=0 TORCH_CPP_LOG_LEVEL=ERROR
export TE_PARALLEL_COMPILER=16 MAX_COMPILE_CORE_NUMBER=8 CPU_AFFINITY_CONF=2
export MASTER_PORT=29631
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export XR1_DATA_CONFIG=nero3_single_20261031
export XR1_MODEL_CONFIG=posttrain_nero_joint_full
export XR1_EPOCHS=3 XR1_BATCH_SIZE=10 XR1_NUM_WORKERS=8
export XR1_ACCUMULATE_GRAD_BATCHES=1
export XR1_DISABLE_CHECKPOINTS=0 XR1_SAVE_EPOCHS=2,3 XR1_SAVE_TOP_K=-1
export XR1_OUTPUT_ROOT=/mnt/wzh/output/xr1-910c-nero3-single-no-rtc-3epochs-bs10-20261031
DATA_ROOT=/mnt/wzh/dataset/xr1_nero3_single_20261031
if [[ ! -f "$DATA_ROOT/SHA256SUMS" || ! -f "$DATA_ROOT/conversion_report.json" ]]; then
  echo "Validated Nero dataset has not finished transferring: $DATA_ROOT" >&2
  exit 2
fi
(cd "$DATA_ROOT" && sha256sum -c --quiet SHA256SUMS)
if [[ ${XR1_CHECK_ONLY:-0} != 1 ]]; then
  if [[ -e "$XR1_OUTPUT_ROOT" || -e "$XR1_OUTPUT_ROOT.log" ]]; then
    echo "Output already exists: $XR1_OUTPUT_ROOT. Choose a new output directory." >&2
    exit 2
  fi
  mkdir -p /mnt/wzh/output
  bash scripts/train_910c_full_joint.sh auto nero3-single-no-rtc-3epochs-bs10 \
    2>&1 | tee "$XR1_OUTPUT_ROOT.log"
else
  bash scripts/train_910c_full_joint.sh auto nero3-single-no-rtc-3epochs-bs10
fi