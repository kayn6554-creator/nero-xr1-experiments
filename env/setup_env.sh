#!/usr/bin/env bash
# 在训练服务器上重建与基线一致的 XR-1 环境。用法：bash setup_env.sh /mnt/panhe/xr1
set -eo pipefail
conda deactivate 2>/dev/null || true
source /usr/local/Ascend/cann-9.0.0/set_env.sh

XR1_DIR=${1:-/mnt/panhe/xr1}
PY=${XR1_PYTHON_BIN:-/root/.local/share/uv/python/cpython-3.11.16-linux-aarch64-gnu/bin/python3.11}
PEFT_WHEEL=${PEFT_WHEEL:-/mnt/wzh/dataset/xr1_stage2_dual4tasks_0905_subtasks_20261012/peft-0.17.1-py3-none-any.whl}
export UV_CACHE_DIR=${UV_CACHE_DIR:-/mnt/panhe/.uv-cache}
export UV_PYTHON_DOWNLOADS=never

cd "$XR1_DIR"
uv sync --frozen --python "$PY"
uv pip install --python .venv/bin/python --no-deps "$PEFT_WHEEL"
.venv/bin/python -c "import torch, torch_npu; print(torch.__version__, torch_npu.__version__, torch.npu.is_available(), torch.npu.device_count())"
