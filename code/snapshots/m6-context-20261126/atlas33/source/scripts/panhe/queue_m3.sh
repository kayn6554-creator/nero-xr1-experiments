#!/usr/bin/env bash
# 11-10 second queue: wait for queue_1110.sh (M2-A) to exit -> stop the QwenKI occupy job
#   -> M3 smoke (3 steps) -> M3 full -> restart the occupy job.
# M3 = M2-A settings (step text, async_train=true, M2 stats/dwell/absolute gripper)
#      + the 11-10 datasets (nero3_m3_20261110: 11 paths, 107337 samples, 4026 steps).
# Start:    cd /mnt/panhe/xr1 && setsid nohup bash scripts/panhe/queue_m3.sh > /dev/null 2>&1 < /dev/null &
# Progress: tail -f /mnt/panhe/output/queue_m3.log
set -o pipefail
cd /mnt/panhe/xr1
mkdir -p /mnt/panhe/output
L=/mnt/panhe/output/queue_m3.log
OCC_DIR=/mnt/wzh/XPolicyLab
OCC_SH=policy/QwenKI/train_joint_910c_bs1_ep10.sh
OCC_PAT=QwenKI/scripts/train_joint.py
M3=(XR1_DATA_CONFIG=nero3_m3_20261110 XR1_MAX_STEPS=4026)
EXP=m3-all8-async
ARGS=(+data.params.train_datasets.step_prompt=true model.params.model.async_train=true)

log() { echo "[$(date '+%F %T')] $*" | tee -a "$L"; }
idle() { npu-smi info | grep -c 'No running processes'; }
wait_idle() {  # $1 = max seconds; returns 1 on timeout
  local t=0
  while [ "$(idle)" -lt 4 ]; do
    [ "$t" -ge "$1" ] && return 1
    sleep 10; t=$((t + 10))
  done
}
occupy_running() { pgrep -f "$OCC_SH" > /dev/null || pgrep -f "$OCC_PAT" > /dev/null; }

restart_occupy() {  # always runs on exit (success, failure, Ctrl-C, kill)
  trap - EXIT
  if occupy_running; then log "occupy job already running, not restarting"; log "queue finished"; return; fi
  wait_idle 600 || log "WARN NPU not idle after 10 min ($(idle)/4), restarting occupy job anyway"
  (cd "$OCC_DIR" && setsid nohup bash "$OCC_SH" > "/mnt/panhe/output/occupy_restart_$(date +%m%d_%H%M%S).out" 2>&1 < /dev/null &)
  sleep 120
  if occupy_running; then
    log "occupy job restarted (its log: /mnt/wzh/output/qwenki_stage2_joint_bs1_ep10_*.log, newest)"
  else
    log "FAIL occupy job did not come back; restart by hand: cd $OCC_DIR && bash $OCC_SH"
  fi
  log "queue finished"
}
trap 'log "queue interrupted"; exit 1' INT TERM HUP

# 0. wait for the first queue (M2-A) to finish; it restarts the occupy job when it exits
if pgrep -f "queue_1110.sh" > /dev/null; then
  log "waiting for queue_1110.sh (M2-A) to finish..."
  for i in $(seq 1 480); do pgrep -f "queue_1110.sh" > /dev/null || break; sleep 60; done
  pgrep -f "queue_1110.sh" > /dev/null && { log "queue_1110.sh still running after 8 h, queue_m3 stopped"; exit 1; }
  log "queue_1110.sh finished"
fi
[ -f configs/data/nero3_m3_20261110.yaml ] || { log "missing configs/data/nero3_m3_20261110.yaml, queue_m3 stopped"; exit 1; }
trap restart_occupy EXIT

log "git $(git rev-parse --short HEAD), $(git status --short | wc -l) uncommitted; idle NPUs $(idle)/4"

# 1. stop the occupy job (only its own process groups + its workers)
if occupy_running; then
  for p in $(pgrep -f "$OCC_SH"); do
    g=$(ps -o pgid= -p "$p" | tr -d ' ')
    [ -n "$g" ] && { log "stop occupy job: pid $p pgid $g"; kill -TERM -- "-$g" 2>/dev/null; }
  done
  sleep 30
  pkill -TERM -f "$OCC_PAT" 2>/dev/null
  if ! wait_idle 300; then
    log "occupy job still on NPU after 5 min, sending KILL"
    pkill -KILL -f "$OCC_PAT" 2>/dev/null
    wait_idle 120 || { log "NPU still busy ($(idle)/4 idle), queue stopped"; exit 1; }
  fi
fi
log "NPU free ($(idle)/4 idle)"

# 2. smoke test: 3 steps, no checkpoint, catches config/override errors
log "START smoke $EXP"
if env XR1_SMOKE_STEPS=3 "${M3[@]}" bash scripts/panhe/train_nero.sh $EXP "${ARGS[@]}" > /dev/null 2>&1; then
  log "DONE  smoke"
else
  log "FAIL  smoke (exit $?), M3 skipped; see /mnt/panhe/output/smoke-$EXP-*.log"; exit 1
fi
grep -h -m3 "async_train" /mnt/panhe/output/smoke-$EXP-*.log 2>/dev/null | cut -c1-200 | sed 's/^/    smoke log: /' | tee -a "$L"
wait_idle 300 || log "WARN NPU not idle 5 min after smoke ($(idle)/4)"

# 3. M3 full run
log "START $EXP"
if env "${M3[@]}" bash scripts/panhe/train_nero.sh $EXP "${ARGS[@]}" > /dev/null 2>&1; then
  log "DONE  $EXP"
else
  log "FAIL  $EXP (exit $?), see /mnt/panhe/output/$EXP.log"
fi
find /mnt/panhe/output/$EXP -name "*.ckpt" -o -name "model_states*.pt" 2>/dev/null | head | sed 's/^/    ckpt: /' | tee -a "$L"
# 4. restart_occupy runs from the EXIT trap