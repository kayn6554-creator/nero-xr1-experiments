# Piper EE-SB working overlay addendum (2026-11-26)

The following exact Atlas-export files were added to this overlay because the earlier canonical overlay did not contain all helpers required by the `nero_ee_sb` path:

- `mibot/utils/io.py`
- `mibot/utils/ef_align.py`
- `mibot/utils/piper_kin.py`
- `mibot/server/runtime/server.py`
- `scripts/panhe/apply_ee_sb.py`
- `scripts/panhe/make_ee_sb_config.py`
- `scripts/panhe/train_nero_piper_ee_sb_atlas_export_20261126.sh`

The last script is the exact export of `/mnt/panhe/train_nero_piper_ee_sb.sh` supplied on 2026-11-26. It is **not** the same file as the companion project's dynamic launcher at `experiments/piper-nero-ee-mix/source-project-v0.2.0-preview/scripts/atlas/train_nero_piper_ee_sb.sh`.

Do not overwrite existing `json_dataset.py`, `tools/train.py`, or data YAML from the working overlay based only on file size. The full Atlas export is preserved under `experiments/piper-nero-ee-mix/atlas-export-20261126/`; compare content and hashes before any future replacement. The exported `plan_json_epochs.py` is a raw-frame planner and is not the current Dwell-aware planner.