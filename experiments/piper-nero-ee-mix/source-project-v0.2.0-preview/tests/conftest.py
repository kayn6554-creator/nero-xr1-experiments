from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
REFERENCE_OVERLAY = ROOT / "references/nero-xr1-experiments/code/xr1-overlay"
for path in (ROOT / "src", REFERENCE_OVERLAY):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
