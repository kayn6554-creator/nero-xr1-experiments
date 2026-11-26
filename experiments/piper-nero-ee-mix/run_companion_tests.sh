#!/usr/bin/env bash
# Run the preserved companion-project tests without modifying the original package.
set -Eeuo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT="$SCRIPT_DIR/source-project-v0.2.0-preview"
[[ -d "$PROJECT_ROOT/tests" ]] || { echo "Missing companion project tests: $PROJECT_ROOT/tests" >&2; exit 2; }
cd "$PROJECT_ROOT"
PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}" python -m pytest -q
