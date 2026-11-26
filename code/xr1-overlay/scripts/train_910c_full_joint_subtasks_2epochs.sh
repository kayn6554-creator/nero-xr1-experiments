#!/usr/bin/env bash
# Same full-parameter 910C baseline as train_910c_full_joint.sh; filtered data, 2 epochs.
set -eo pipefail
XR1_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${XR1_DIR}"
export XR1_DATA_CONFIG="${XR1_DATA_CONFIG:-robodojo_real4_piper_joint_subtasks_2epochs}"
export XR1_EPOCHS=2
export XR1_BATCH_SIZE="${XR1_BATCH_SIZE:-8}"
export XR1_NUM_WORKERS="${XR1_NUM_WORKERS:-8}"
export XR1_SAVE_INTERVAL="${XR1_SAVE_INTERVAL:-1000}"
export XR1_SAVE_TOP_K="${XR1_SAVE_TOP_K:-1}"
export XR1_DISABLE_CHECKPOINTS="${XR1_DISABLE_CHECKPOINTS:-0}"
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export XR1_OUTPUT_ROOT="${XR1_OUTPUT_ROOT:-/mnt/wzh/output/xr1-910c-joint-subtasks-bs8-2epochs}"
export TASK_QUEUE_ENABLE="${TASK_QUEUE_ENABLE:-0}"
export TORCH_CPP_LOG_LEVEL="${TORCH_CPP_LOG_LEVEL:-ERROR}"
export TE_PARALLEL_COMPILER="${TE_PARALLEL_COMPILER:-16}"
export MAX_COMPILE_CORE_NUMBER="${MAX_COMPILE_CORE_NUMBER:-8}"
export CPU_AFFINITY_CONF="${CPU_AFFINITY_CONF:-2}"
unset XR1_CKPT_PATH STAGE2_CKPT_PATH
STEPS="$(.venv/bin/python - "${XR1_DATA_CONFIG}" "${XR1_BATCH_SIZE}" <<'PY'
import glob,json,math,sys,yaml,os
name,batch=sys.argv[1],int(sys.argv[2])
cards=os.environ['ASCEND_RT_VISIBLE_DEVICES'].split(',')
if len(cards)!=8 or len(set(cards))!=8: raise ValueError('This baseline requires eight distinct NPU ids')
if batch<=0: raise ValueError('XR1_BATCH_SIZE must be positive')
cfg=yaml.safe_load(open('configs/data/'+name+'.yaml'))
files=sorted(set(p for root in cfg['data']['params']['train_datasets']['paths'] for p in glob.glob(root+'/**/*.json',recursive=True)))
if not files: raise RuntimeError('empty training set')
frames=0
for p in files:
    d=json.load(open(p))
    if 'source' in d:
        if not d['source']['collection'].startswith('task_'): raise ValueError('Unexpected full episode: '+p)
    frames+=int(d['num_frames'])
steps=2*math.ceil(math.ceil(frames/8)/batch)
print(steps)
print(f'data={name} episodes={len(files)} frames={frames} ranks=8 batch/rank={batch} accumulation=1 epochs=2 optimizer_steps={steps}',file=sys.stderr)
PY
)"
if [[ "${1:-}" == "--check" ]]; then
  export XR1_CHECK_ONLY=1
elif [[ "$#" -gt 0 ]]; then
  echo "Usage: bash scripts/train_910c_full_joint_subtasks_2epochs.sh [--check]" >&2
  exit 2
fi
exec bash scripts/train_910c_full_joint.sh auto "${XR1_EXP_NAME:-robodojo-real4-subtasks-full-joint-bs8-2epochs}"