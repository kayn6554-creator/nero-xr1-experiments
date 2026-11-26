#!/usr/bin/env bash
# 11-12 queue: M4 and EF4 = M3 and EF with steps 2+3 merged into one step text (step_merge23).
#   for each run: stop the QwenKI occupy job (if running) -> wait until all NPUs are idle
#   (up to 3 h, someone else may be using them) -> smoke (3 steps) -> full run.
#   At the end (success, failure or kill) restart the occupy job if it is not running.
# Everything else exactly as M3 / EF (same data, stats, async, 4026 steps).
# Start:    cd /mnt/panhe/xr1 && setsid nohup bash scripts/panhe/queue_1112.sh > /dev/null 2>&1 < /dev/null &
# Progress: tail -f /mnt/panhe/output/queue_1112.log
set -o pipefail
cd /mnt/panhe/xr1
mkdir -p /mnt/panhe/output
L=/mnt/panhe/output/queue_1112.log
OCC_DIR=/mnt/wzh/XPolicyLab
OCC_SH=policy/QwenKI/train_joint_910c_bs1_ep10.sh
OCC_PAT=QwenKI/scripts/train_joint.py
COMMON=(+data.params.train_datasets.step_prompt=true +data.params.train_datasets.step_merge23=true model.params.model.async_train=true)

log() { echo "[$(date '+%F %T')] $*" | tee -a "$L"; }
idle() { npu-smi info | grep -c 'No running processes'; }
wait_idle() {  # $1 = max seconds; returns 1 on timeout
  local t=0
  while [ "$(idle)" -lt 4 ]; do
    [ "$t" -ge "$1" ] && return 1
    sleep 30; t=$((t + 30))
  done
}
occupy_running() { pgrep -f "$OCC_SH" > /dev/null || pgrep -f "$OCC_PAT" > /dev/null; }

stop_occupy() {
  occupy_running || return 0
  for p in $(pgrep -f "$OCC_SH"); do
    g=$(ps -o pgid= -p "$p" | tr -d ' ')
    [ -n "$g" ] && { log "stop occupy job: pid $p pgid $g"; kill -TERM -- "-$g" 2>/dev/null; }
  done
  sleep 30
  pkill -TERM -f "$OCC_PAT" 2>/dev/null
  sleep 60
  occupy_running && { log "occupy job still alive, sending KILL"; pkill -KILL -f "$OCC_PAT" 2>/dev/null; pkill -KILL -f "$OCC_SH" 2>/dev/null; }
  return 0
}

restart_occupy() {  # runs on exit
  trap - EXIT
  if occupy_running; then log "occupy job already running, not restarting"; log "queue finished"; return; fi
  wait_idle 1800 || { log "NPU busy by someone else ($(idle)/4 idle), NOT restarting occupy job"; log "queue finished"; return; }
  (cd "$OCC_DIR" && exec setsid nohup bash "$OCC_SH" > "/mnt/panhe/output/occupy_restart_$(date +%m%d_%H%M%S).out" 2>&1 < /dev/null &)
  sleep 180
  if occupy_running; then
    log "occupy job restarted (its log: /mnt/wzh/output/qwenki_stage2_joint_bs1_ep10_*.log, newest)"
  else
    log "FAIL occupy job did not come back; restart by hand: cd $OCC_DIR && bash $OCC_SH"
  fi
  log "queue finished"
}
trap restart_occupy EXIT
trap 'log "queue interrupted"; exit 1' INT TERM HUP

run_one() {  # $1 = exp name, rest = env assignments
  local exp=$1; shift
  stop_occupy
  if ! wait_idle 10800; then log "NPU not idle after 3 h ($(idle)/4), $exp skipped"; return 1; fi
  log "NPU free ($(idle)/4 idle)"
  log "START smoke $exp"
  if env XR1_SMOKE_STEPS=3 "$@" bash scripts/panhe/train_nero.sh "$exp" "${COMMON[@]}" > /dev/null 2>&1; then
    log "DONE  smoke $exp"
  else
    log "FAIL  smoke $exp (exit $?), skipped; see /mnt/panhe/output/smoke-$exp-*.log"; return 1
  fi
  wait_idle 600 || log "WARN NPU not idle 10 min after smoke ($(idle)/4)"
  log "START $exp"
  if env "$@" bash scripts/panhe/train_nero.sh "$exp" "${COMMON[@]}" > /dev/null 2>&1; then
    log "DONE  $exp"
  else
    log "FAIL  $exp (exit $?), see /mnt/panhe/output/$exp.log"
  fi
  ls -d /mnt/panhe/output/"$exp"/project_*/*/epoch*.ckpt 2>/dev/null | sed 's/^/    ckpt: /' | tee -a "$L"
  wait_idle 600 || log "WARN NPU not idle 10 min after $exp ($(idle)/4)"
}

log "git $(git rev-parse --short HEAD), $(git status --short | wc -l) uncommitted; idle NPUs $(idle)/4"
grep -q step_merge23 mibot/data/datasets/json_dataset.py || { log "apply_m4.py not applied, queue stopped"; exit 1; }
for f in configs/data/nero3_m3_20261110.yaml configs/data/nero3_ee_20261110.yaml configs/model/posttrain_nero_ee_full.yaml mibot/utils/nero_kin.py; do
  [ -f "$f" ] || { log "missing $f, queue stopped"; exit 1; }
done
for e in m4-merge23 ef4-merge23; do
  [ -e "/mnt/panhe/output/$e" ] && { log "/mnt/panhe/output/$e already exists, queue stopped"; exit 1; }
done
run_one m4-merge23 XR1_DATA_CONFIG=nero3_m3_20261110 XR1_MAX_STEPS=4026
run_one ef4-merge23 XR1_DATA_CONFIG=nero3_ee_20261110 XR1_MODEL_CONFIG=posttrain_nero_ee_full XR1_MAX_STEPS=4026