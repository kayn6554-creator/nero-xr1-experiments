#!/usr/bin/env bash
set -eo pipefail
cd /mnt/wzh/XPolicyLab/policy/Xiaomi_Robotics_1/xiaomi_robotics_1/xr1
unset XR1_CKPT_PATH XR1_EPOCHS XR1_CHECK_ONLY WORLD_SIZE RANK LOCAL_RANK MASTER_ADDR WANDB_RUN_ID WANDB_RESUME
source /usr/local/Ascend/cann-9.0.0/set_env.sh

export TASK_QUEUE_ENABLE=0
export TORCH_CPP_LOG_LEVEL=ERROR
export TE_PARALLEL_COMPILER=16
export MAX_COMPILE_CORE_NUMBER=8
export CPU_AFFINITY_CONF=2
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export MASTER_PORT=29643
export XR1_DATA_CONFIG=single_piper_left_all382
export XR1_MODEL_CONFIG=posttrain_piper_joint_full
export XR1_EPOCHS=3
export XR1_BATCH_SIZE=10
export XR1_NUM_WORKERS=8
export XR1_ACCUMULATE_GRAD_BATCHES=1
export XR1_DISABLE_CHECKPOINTS=0
export XR1_SAVE_INTERVAL=1000
export XR1_SAVE_TOP_K=1
export XR1_SAVE_EPOCHS=1,2,3  # explicit epoch saves take precedence over periodic saves
export WANDB_MODE=offline
export XR1_OUTPUT_ROOT=/mnt/wzh/output/xr1-910c-piper-left-from-mixed-3ep-$(date +%Y%m%d-%H%M%S)-$$
export XR1_WANDB_SAVE_DIR="$XR1_OUTPUT_ROOT"
export WANDB_DIR="$XR1_OUTPUT_ROOT"

DATA_ROOT=/mnt/wzh/dataset/xr1_single8_replay_right_20261102
PRETRAINED=/mnt/wzh/output/xr1-910c-real3-dual-evo-single4-all98-bs10-20261027/project_xr1-910c-joint-full/real3-dual-evo-single4-all98-bs10/epoch0-step22000.ckpt/checkpoint/mp_rank_00_model_states.pt
WARMUP_LR=2e-7
PEAK_LR=2e-6
MIN_LR=2e-7
EXP=piper-left-from-mixed-3ep

case "${1:-train}" in
  --prepare)
    exec .venv/bin/python -u scripts/process_data.py --stats-only \
      --dst "$DATA_ROOT" --extra-stats-path "$DATA_ROOT/data_val" \
      --action-layout piper_left_joint --action-length 30 \
      --batch-size "$XR1_BATCH_SIZE" --workers 4 --config-name "$XR1_DATA_CONFIG"
    ;;
  --check) export XR1_CHECK_ONLY=1 ;;
  train) ;;
  *) echo "usage: bash $0 [--prepare|--check|train]" >&2; exit 2 ;;
esac

test -r "$PRETRAINED" || { echo "Missing initial weights: $PRETRAINED" >&2; exit 1; }
IFS=, read -ra devices <<< "$ASCEND_RT_VISIBLE_DEVICES"
counts=$(.venv/bin/python scripts/plan_json_epochs.py --counts \
  --data-config "$XR1_DATA_CONFIG" --epochs "$XR1_EPOCHS" \
  --batch-size "$XR1_BATCH_SIZE" --world-size "${#devices[@]}" --accumulation "$XR1_ACCUMULATE_GRAD_BATCHES")
read -r frames warmup_steps total_steps episodes <<< "$counts"
printf 'frames=%s episodes=%s warmup_steps=%s total_steps=%s lr=%s -> %s -> %s\n' \
  "$frames" "$episodes" "$warmup_steps" "$total_steps" "$WARMUP_LR" "$PEAK_LR" "$MIN_LR"
printf 'initial_weights=%s\noutput=%s\n' "$PRETRAINED" "$XR1_OUTPUT_ROOT"

args=(auto "$EXP"
  "model.params.pretrained=$PRETRAINED"
  model.params.model.action_layout=piper_left_joint
  "trainer.scheduler.params.num_warmup_steps=$warmup_steps"
  "trainer.scheduler.params.warmup_lr_start=$WARMUP_LR"
  "trainer.scheduler.params.max_lr=$PEAK_LR"
  "trainer.scheduler.params.min_lr=$MIN_LR")
if [[ "${XR1_CHECK_ONLY:-0}" == 1 ]]; then
  exec bash scripts/train_910c_full_joint.sh "${args[@]}"
fi
bash scripts/train_910c_full_joint.sh "${args[@]}" 2>&1 | tee "$XR1_OUTPUT_ROOT.log"