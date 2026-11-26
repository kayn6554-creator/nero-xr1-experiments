#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd -- "$SCRIPT_DIR/../.." && pwd)

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 /absolute/path/to/last.ckpt" >&2
    exit 2
fi

CKPT=$1
shift
[[ -e "$CKPT" ]] || { echo "Checkpoint path does not exist: $CKPT" >&2; exit 2; }

export XR1_CKPT_PATH="$CKPT"
export XR1_CHECK_ONLY=${XR1_CHECK_ONLY:-0}
export XR1_ALLOW_TRAINING=${XR1_ALLOW_TRAINING:-1}

exec bash "$PROJECT_ROOT/scripts/atlas/train_nero_piper_ee_sb.sh" "$@"