#!/usr/bin/env bash
# Wait for the training queue to finish, then run the offline check for M0, M1, M2, M2b on one chip.
#   nohup bash scripts/panhe/eval_after_queue.sh > /dev/null 2>&1 &
# Results: /mnt/panhe/output/eval_<name>.json and /mnt/panhe/output/eval.log
cd /mnt/panhe/xr1
source /usr/local/Ascend/cann-9.0.0/set_env.sh
export XR1_DEVICE_TYPE=npu HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 ASCEND_RT_VISIBLE_DEVICES=0
L=/mnt/panhe/output/eval.log
log() { echo "[$(date '+%F %T')] $*" | tee -a "$L"; }
log "waiting for queue to finish"
until grep -q -E "queue finished|queue stopped|smoke failed" /mnt/panhe/output/queue.log 2>/dev/null; do sleep 120; done
log "queue done, starting offline eval"
M0=/mnt/wzh/output/xr1-910c-nero3-single-no-rtc-3epochs-bs10-20261031/project_xr1-910c-joint-full/nero3-single-no-rtc-3epochs-bs10
for pair in "m0:$M0" \
            "m1:$(dirname $(ls -d /mnt/panhe/output/m1-step-prompt/project_*/*/*.ckpt 2>/dev/null | tail -1) 2>/dev/null)" \
            "m2:$(dirname $(ls -d /mnt/panhe/output/m2-full/project_*/*/*.ckpt 2>/dev/null | tail -1) 2>/dev/null)" \
            "m2b:$(dirname $(ls -d /mnt/panhe/output/m2b-no-step/project_*/*/*.ckpt 2>/dev/null | tail -1) 2>/dev/null)"; do
  name=${pair%%:*}; dir=${pair#*:}
  if [[ -z "$dir" || ! -f "$dir/config.py" ]]; then log "skip $name: no checkpoint"; continue; fi
  log "START eval $name ($dir)"
  if .venv/bin/python scripts/panhe/offline_eval.py --model-dir "$dir" --device npu:0 --out /mnt/panhe/output/eval_$name.json >> "$L" 2>&1; then
    log "DONE  eval $name"; else log "FAIL  eval $name"; fi
done
log "offline eval finished"
