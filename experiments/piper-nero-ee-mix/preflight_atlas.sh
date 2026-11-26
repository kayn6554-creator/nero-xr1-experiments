#!/usr/bin/env bash
# Read-only preflight for the NERO-target Piper+NERO EE-SB training run.
# This script never starts training, creates output directories, installs packages,
# copies configs, stops processes, or changes shared services.
set -Eeuo pipefail

REPO_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
PROJECT_ROOT=${PIPER_PROJECT_ROOT:-$REPO_ROOT/experiments/piper-nero-ee-mix/source-project-v0.2.0-preview}
XR1_ROOT=${XR1_ROOT:-/mnt/panhe/xr1}
PYTHON_BIN=${XR1_PYTHON:-$XR1_ROOT/.venv-piper-ef/bin/python}
CANN_ENV=${CANN_ENV:-/usr/local/Ascend/cann-9.0.0/set_env.sh}
ATLAS_LAUNCHER=${ATLAS_LAUNCHER:-/mnt/panhe/train_nero_piper_ee_sb.sh}
DATA_CONFIG=${DATA_CONFIG:-$XR1_ROOT/configs/data/nero3_piper_ee_sb_20261125.yaml}
EXPECTED_LAUNCHER_SHA256=265da9feb4f9204952a0076ba5d14b33d66933cb89bf01bcdc8195fdcc104d81
EXPECTED_EFFECTIVE=486500
EXPECTED_STEPS_PER_EPOCH=6082
EXPECTED_MAX_STEPS=18246
EXPECTED_EPISODES=3611
VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}

fail() { echo "[FAIL] $*" >&2; exit 2; }
section() { printf '\n========== %s ==========\n' "$1"; }

[[ -d "$XR1_ROOT" ]] || fail "XR1 root missing: $XR1_ROOT"
[[ -x "$PYTHON_BIN" ]] || fail "Python environment missing: $PYTHON_BIN"
[[ -f "$CANN_ENV" ]] || fail "CANN environment script missing: $CANN_ENV"
[[ -f "$XR1_ROOT/scripts/train.sh" ]] || fail "XR1 scripts/train.sh missing"
[[ -f "$XR1_ROOT/tools/train.py" ]] || fail "XR1 tools/train.py missing"
[[ -f "$DATA_CONFIG" ]] || fail "Atlas mixed data config missing: $DATA_CONFIG"
[[ -f "$REPO_ROOT/experiments/piper-nero-ee-mix/atlas-source-sha256.txt" ]] || fail "Atlas source hash manifest missing"
[[ -f "$PROJECT_ROOT/scripts/validation/validate_mix_dataset.py" ]] || fail "Companion project validation scripts missing"
[[ -f "$REPO_ROOT/experiments/piper-nero-ee-mix/validation/validate_runtime_dataset.py" ]] || fail "Runtime JsonDataset validator missing from repository"

section "Host / paths"
printf 'host=%s\nuser=%s\nXR1_ROOT=%s\nPython=%s\nCANN_ENV=%s\n' \
  "$(hostname)" "$(id -un)" "$XR1_ROOT" "$PYTHON_BIN" "$CANN_ENV"
printf 'DATA_CONFIG=%s\nVISIBLE_DEVICES=%s\n' "$DATA_CONFIG" "$VISIBLE_DEVICES"

section "Atlas source/config provenance"
(cd "$XR1_ROOT" && sha256sum -c "$REPO_ROOT/experiments/piper-nero-ee-mix/atlas-source-sha256.txt") || fail "Atlas XR1 source differs from the archived 2026-11-26 snapshot; inspect before training"

IFS=, read -r -a device_array <<< "$VISIBLE_DEVICES"
DEVICE_COUNT=${#device_array[@]}
[[ "$DEVICE_COUNT" -gt 0 ]] || fail "No visible NPU devices specified"

section "Atlas Python / CANN / NPU imports"
# Environment changes are confined to this script process. CANN set_env is not run under nounset.
set +u
source "$CANN_ENV"
set -u
export ASCEND_RT_VISIBLE_DEVICES="$VISIBLE_DEVICES"
export EXPECTED_NPU_COUNT="$DEVICE_COUNT"
export PYTHONDONTWRITEBYTECODE=1
"$PYTHON_BIN" - <<'PY'
import importlib, os, platform, sys
print("Python:", sys.version.replace("\n", " "))
print("Platform:", platform.platform(), platform.machine())
required = ["numpy", "yaml", "torch", "torch_npu", "deepspeed", "lightning", "transformers", "mmengine", "hydra", "cv2", "imageio", "psutil", "wandb"]
errors = []
for name in required:
    try:
        mod = importlib.import_module(name)
        print(f"[OK] {name}: {getattr(mod, '__version__', 'version n/a')}")
    except Exception as exc:
        errors.append((name, repr(exc)))
        print(f"[FAIL] {name}: {exc!r}")
if errors:
    raise SystemExit(f"Required imports failed: {errors}")
import torch
try:
    available = torch.npu.is_available()
    count = torch.npu.device_count()
    print("torch.npu.is_available:", available)
    print("torch.npu.device_count:", count)
    expected = int(os.environ["EXPECTED_NPU_COUNT"])
    if count != expected:
        raise SystemExit(f"visible NPU count {count} != listed devices {expected}")
except Exception as exc:
    raise SystemExit(f"NPU runtime query failed: {exc!r}")
PY

section "Mixed dataset manifest / Dwell Filter mirror"
XR1_ROOT="$XR1_ROOT" "$PYTHON_BIN" \
  "$PROJECT_ROOT/scripts/validation/validate_mix_dataset.py" --config "$DATA_CONFIG"

section "Real JsonDataset epoch scan and planned steps"
XR1_ROOT="$XR1_ROOT" PYTHONPATH="$XR1_ROOT:$PROJECT_ROOT/src" \
  "$PYTHON_BIN" "$REPO_ROOT/experiments/piper-nero-ee-mix/validation/validate_runtime_dataset.py" \
  --config "$DATA_CONFIG" --mode epoch --world-size "$DEVICE_COUNT" \
  --batch-size 10 --accumulation 1 --epochs 3 \
  --expected-effective-frames "$EXPECTED_EFFECTIVE" --expected-episodes "$EXPECTED_EPISODES" \
  --expected-steps-per-epoch "$EXPECTED_STEPS_PER_EPOCH" --expected-max-steps "$EXPECTED_MAX_STEPS"

section "Actual 3-step smoke sample coverage (index only; no model forward)"
XR1_ROOT="$XR1_ROOT" PYTHONPATH="$XR1_ROOT:$PROJECT_ROOT/src" \
  "$PYTHON_BIN" "$REPO_ROOT/experiments/piper-nero-ee-mix/validation/validate_runtime_dataset.py" \
  --config "$DATA_CONFIG" --mode smoke --world-size "$DEVICE_COUNT" \
  --batch-size 10 --accumulation 1 --smoke-steps 3

section "One NERO and one Piper real sample: State / Action / Mask / image decode"
XR1_ROOT="$XR1_ROOT" PYTHONPATH="$XR1_ROOT:$PROJECT_ROOT/src" \
  "$PYTHON_BIN" "$PROJECT_ROOT/scripts/validation/validate_batch.py" --config "$DATA_CONFIG"

section "Exact Atlas launcher provenance and check-only plan"
if [[ -f "$ATLAS_LAUNCHER" ]]; then
  actual=$(sha256sum "$ATLAS_LAUNCHER" | awk '{print $1}')
  printf 'Atlas launcher SHA256=%s\n' "$actual"
  [[ "$actual" == "$EXPECTED_LAUNCHER_SHA256" ]] || fail "Atlas launcher hash differs from the 2026-11-26 exported script; inspect before use"
  XR1_CHECK_ONLY=1 bash "$ATLAS_LAUNCHER"
else
  echo "[WARN] Atlas launcher not found at $ATLAS_LAUNCHER; verify/copy the exact export before training."
fi

section "Read-only resource and process snapshot"
if command -v npu-smi >/dev/null 2>&1; then npu-smi info || true; else echo "[WARN] npu-smi not found"; fi
if command -v tmux >/dev/null 2>&1; then tmux ls 2>/dev/null || true; fi
ps -eo pid,ppid,user,etime,args | grep -E '[V]LLMWorker|[v]llm|[t]orch\.distributed\.run|[t]ools/train\.py' || true
if command -v ss >/dev/null 2>&1; then ss -lntp 2>/dev/null | grep -E ':11223([[:space:]]|$)' || true; fi
if command -v docker >/dev/null 2>&1 && [[ -S /run/docker-minimax.sock ]]; then
  echo "Docker read-only listing via the recorded Minimax socket (no container action):"
  docker -H unix:///run/docker-minimax.sock ps --format 'table {{.ID}}\t{{.Names}}\t{{.Ports}}' || true
fi

section "Output path collision check"
OUTPUT_ROOT=${XR1_OUTPUT_ROOT:-/mnt/panhe/output/nero-piper-ee-sb-v0.1}
if [[ -e "$OUTPUT_ROOT" || -e "${OUTPUT_ROOT}.log" ]]; then
  echo "[WARN] Formal output path already exists: $OUTPUT_ROOT"
  echo "Do not overwrite or delete it. Select a new XR1_OUTPUT_ROOT after inspecting the existing run."
else
  echo "[OK] Default formal output path does not currently exist."
fi

echo
echo "[PREFLIGHT COMPLETE] No training was started and no process/container was changed."
echo "A human must still verify resource ownership and authorization before running Smoke or formal training."
