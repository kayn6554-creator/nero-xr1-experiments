#!/usr/bin/env bash
# Friday-night queue: smoke test (M2 config, covers all new code) -> M1 -> M2 -> M2b.
# Start it with:  nohup bash scripts/panhe/queue_weekend.sh > /dev/null 2>&1 &
# 4 chips only:   ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 XR1_ACCUMULATE_GRAD_BATCHES=2 nohup bash ... &
# Progress:       tail -f /mnt/panhe/output/queue.log
set -o pipefail
cd /mnt/panhe/xr1
mkdir -p /mnt/panhe/output
L=/mnt/panhe/output/queue.log
log() { echo "[$(date '+%F %T')] $*" | tee -a "$L"; }
step() {
  local name=$1; shift
  log "START $name"
  if env "$@"; then log "DONE  $name"; else log "FAIL  $name (exit $?)"; return 1; fi
}
M2=(XR1_DATA_CONFIG=nero3_m2_20261106 XR1_MAX_STEPS=2793)
log "git $(git rev-parse --short HEAD) $(git status --short | wc -l) uncommitted; idle cards: $(npu-smi info | grep -c 'No running processes')/4"
step smoke XR1_SMOKE_STEPS=3 "${M2[@]}" bash scripts/panhe/train_nero.sh m2 +data.params.train_datasets.step_prompt=true || { log "smoke failed, queue stopped"; exit 1; }
step m1  bash scripts/panhe/train_nero.sh m1-step-prompt +data.params.train_datasets.step_prompt=true
step m2  "${M2[@]}" bash scripts/panhe/train_nero.sh m2-full +data.params.train_datasets.step_prompt=true
step m2b "${M2[@]}" bash scripts/panhe/train_nero.sh m2b-no-step
log "queue finished"
