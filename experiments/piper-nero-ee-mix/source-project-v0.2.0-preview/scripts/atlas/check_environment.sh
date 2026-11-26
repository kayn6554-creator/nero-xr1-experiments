#!/usr/bin/env bash
set -Eeuo pipefail

XR1_ROOT=${XR1_ROOT:-/mnt/panhe/xr1}
PYTHON_BIN=${XR1_PYTHON:-$XR1_ROOT/.venv-piper-ef/bin/python}
CANN_ENV=${CANN_ENV:-/usr/local/Ascend/cann-9.0.0/set_env.sh}

echo "XR1_ROOT=$XR1_ROOT"
echo "PYTHON_BIN=$PYTHON_BIN"
echo "CANN_ENV=$CANN_ENV"

[[ -d "$XR1_ROOT" ]] || { echo "[FAIL] XR1 repo missing"; exit 1; }
[[ -x "$PYTHON_BIN" ]] || { echo "[FAIL] project Python missing"; exit 1; }
[[ -f "$CANN_ENV" ]] || { echo "[FAIL] CANN environment script missing"; exit 1; }

# Read-only environment setup in this shell.
source "$CANN_ENV"

"$PYTHON_BIN" - <<'PY'
import importlib
import platform
import sys

print("Python:", sys.version.replace("\n", " "))
print("Platform:", platform.platform(), platform.machine())

required = ["numpy", "yaml", "torch", "torch_npu", "lightning", "mmengine", "hydra"]
failed = []
for name in required:
    try:
        mod = importlib.import_module(name)
        print(f"[OK] {name}: {getattr(mod, '__version__', 'version n/a')}")
    except Exception as exc:
        failed.append((name, repr(exc)))
        print(f"[FAIL] {name}: {exc!r}")

if failed:
    raise SystemExit(f"Missing/unimportable modules: {failed}")

import torch
print("torch.cuda device count:", torch.cuda.device_count())
try:
    print("torch.npu available:", torch.npu.is_available())
    print("torch.npu device count:", torch.npu.device_count())
except Exception as exc:
    print("NPU query failed:", repr(exc))
PY

if command -v npu-smi >/dev/null 2>&1; then
    echo
    echo "========== npu-smi info (read-only) =========="
    npu-smi info || true
fi

echo
echo "========== tmux sessions (read-only) =========="
tmux ls 2>/dev/null || true

echo
echo "Environment inspection finished. No process was stopped and no package was installed."
