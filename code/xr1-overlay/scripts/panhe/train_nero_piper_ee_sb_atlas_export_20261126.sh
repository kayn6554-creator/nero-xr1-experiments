#!/usr/bin/env bash
set -eo pipefail

export XR1_DIR=${XR1_DIR:-/mnt/panhe/xr1}
cd "$XR1_DIR"

source /usr/local/Ascend/cann-9.0.0/set_env.sh

unset XR1_CKPT_PATH WORLD_SIZE RANK LOCAL_RANK MASTER_ADDR
export XR1_DEVICE_TYPE=npu
export XR1_PYTHON=/mnt/panhe/xr1/.venv-piper-ef/bin/python
export XR1_BASE_VLM=/mnt/wzh/checkpoints/Qwen3-VL-4B-Instruct
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export MAX_LENGTH=${MAX_LENGTH:-20000}

export TASK_QUEUE_ENABLE=0
export TORCH_CPP_LOG_LEVEL=ERROR
export TE_PARALLEL_COMPILER=16
export MAX_COMPILE_CORE_NUMBER=8
export CPU_AFFINITY_CONF=2
export MASTER_PORT=${XR1_MASTER_PORT:-29731}

export ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
IFS=, read -ra visible_cards <<< "$ASCEND_RT_VISIBLE_DEVICES"
export RESOURCE_GPU=${#visible_cards[@]}

DATA_CONFIG=${XR1_DATA_CONFIG:-nero3_piper_ee_sb_20261125}
MODEL_CONFIG=${XR1_MODEL_CONFIG:-posttrain_nero_ee_sb_full}
EXP_NAME=${XR1_EXP_NAME:-nero-piper-ee-sb-v0.1}
BATCH_SIZE=${XR1_BATCH_SIZE:-10}
ACCUMULATION=${XR1_ACCUMULATE_GRAD_BATCHES:-1}
export XR1_NUM_WORKERS=${XR1_NUM_WORKERS:-8}
export XR1_DISABLE_CHECKPOINTS=${XR1_DISABLE_CHECKPOINTS:-0}
SAVE_INTERVAL=${XR1_SAVE_INTERVAL:-1000}

# 前置文件检查
for f in \
  "$XR1_PYTHON" \
  "$XR1_DIR/configs/data/${DATA_CONFIG}.yaml" \
  "$XR1_DIR/configs/model/${MODEL_CONFIG}.yaml" \
  "$XR1_DIR/configs/trainer/deepspeed_npu_bf16_zero2.yaml" \
  "/mnt/wzh/checkpoints/Xiaomi-Robotics-1-5B/model_states.pt" \
  "$XR1_BASE_VLM/config.json" \
  "/mnt/panhe/dataset/xr1_ee_sb_mix_20261125/nero_r1" \
  "/mnt/panhe/dataset/xr1_ee_sb_mix_20261125/nero_r2" \
  "/mnt/panhe/dataset/xr1_ee_sb_mix_20261125/piper_v2"
do
  [[ -e "$f" ]] || { echo "Missing required path: $f" >&2; exit 2; }
done

extra_overrides=("++data.params.train_datasets.step_prompt=false")

if [[ -n "${XR1_MAX_STEPS_OVERRIDE:-}" ]]; then
  # Smoke Test 模式：固定步数 + 周期保存，检查 checkpoint 写入
  MAX_STEPS="$XR1_MAX_STEPS_OVERRIDE"
  unset XR1_SAVE_EPOCHS
  export XR1_SAVE_TOP_K=${XR1_SAVE_TOP_K:-1}
  OUTPUT_ROOT=${XR1_OUTPUT_ROOT:-/mnt/panhe/output/smoke-${EXP_NAME}-$(date +%m%d-%H%M%S)}
  MODE_DESC="smoke(steps=$MAX_STEPS, save_interval=$SAVE_INTERVAL)"
else
  # 正式 3-Epoch 模式：按 Dwell Filter 有效样本数 486500 计算的精确步数
  EPOCHS=${XR1_EPOCHS:-3}
  MAX_STEPS=${XR1_MAX_STEPS:-18246}
  export XR1_SAVE_EPOCHS=${XR1_SAVE_EPOCHS:-3}
  export XR1_SAVE_TOP_K=${XR1_SAVE_TOP_K:--1}
  OUTPUT_ROOT=${XR1_OUTPUT_ROOT:-/mnt/panhe/output/${EXP_NAME}}
  extra_overrides+=("+data.params.finite_epoch=true" "+trainer.max_epochs=$EPOCHS")
  MODE_DESC="epoch3(epochs=$EPOCHS, max_steps=$MAX_STEPS, save_epochs=$XR1_SAVE_EPOCHS)"
fi

if [[ -n "${XR1_RESUME_CKPT:-}" ]]; then
  extra_overrides+=("trainer.ckpt_path=$XR1_RESUME_CKPT")
fi

export WANDB_MODE=${WANDB_MODE:-offline}
export WANDB_DIR="${OUTPUT_ROOT}/wandb"
export XR1_WANDB_SAVE_DIR="${OUTPUT_ROOT}"

echo "=== Plan ==="
echo "mode=$MODE_DESC"
echo "python=$XR1_PYTHON"
echo "data=$DATA_CONFIG model=$MODEL_CONFIG"
echo "cards=$ASCEND_RT_VISIBLE_DEVICES (RESOURCE_GPU=$RESOURCE_GPU) batch=$BATCH_SIZE accum=$ACCUMULATION workers=$XR1_NUM_WORKERS"
echo "output=$OUTPUT_ROOT"
echo "overrides=${extra_overrides[*]}"

if [[ "${XR1_CHECK_ONLY:-1}" == "1" || "${XR1_ALLOW_TRAINING:-0}" != "1" ]]; then
  echo "[CHECK_ONLY] 未启动训练。实际运行需显式设置 XR1_CHECK_ONLY=0 XR1_ALLOW_TRAINING=1"
  exit 0
fi

if [[ -z "${XR1_MAX_STEPS_OVERRIDE:-}" && -z "${XR1_RESUME_CKPT:-}" ]] && [[ -e "$OUTPUT_ROOT" || -e "${OUTPUT_ROOT}.log" ]]; then
  echo "Formal output already exists: $OUTPUT_ROOT (avoid overwrite)" >&2
  exit 2
fi

mkdir -p "$OUTPUT_ROOT" "$WANDB_DIR"

bash scripts/train.sh \
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
  "$@" 2>&1 | tee "${OUTPUT_ROOT}.log"
