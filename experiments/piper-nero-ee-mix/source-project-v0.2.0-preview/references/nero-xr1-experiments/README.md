# NERO XR1 Experiments — inspected upstream reference

This project was reviewed against the public repository snapshot:

- Repository: https://github.com/kayn6554-creator/nero-xr1-experiments
- Branch: `main`
- Commit: `d4826f4f119a292d0ca061f637194000016360de`
- Commit date: 2026-11-23 UTC
- Commit message: `将新增任务限制为采集模式使用`

This directory is a **small, pinned reference overlay**, not a fork or replacement for the NERO experiment repository.

## Included exact helper sources

`code/xr1-overlay/mibot/utils/` contains the source files used by the training-side EE contract:

- `nero_kin.py`: NERO FK / Jacobian / DLS IK.
- `piper_kin.py`: Piper left-arm FK and rotation helpers.
- `ef_align.py`: `compose_ee_state`, `ee_relative_action`, base/flange alignment constants.
- `nero_steps.py`: shared step-text definitions.

The NERO kinematics source was checked against the pinned GitHub blob. The local `nero_kin.py` snapshot has Git blob SHA `ada3bee3b954c2f22e02c5d47487057de8c24a2f`; `nero_steps.py` has Git blob SHA `4c4d0ab4f7d8b78f71d024dd6e6a869bdb3eb44c`.

## Important: do not treat every upstream file as current deployment code

The repository's `code/collector/nero_vla_client.py` is an earlier snapshot. The 2026-11-16 deployed M5 client/server patch is represented in the repository by a patch manifest and tests, but the complete clean ZIP contents are not committed as source files.

For the currently deployed M5 Step Flag client, use the verified deployed copy / clean patch matching the recorded SHA256, not the stale collector snapshot. See `docs/09-nero-xr1-integration.md`.

## Why we do not vendor the whole repository

The NERO robot plugin, ROS camera plugin, collection UI, report server, and runtime scripts already have their own upstream repository and operational history. This project documents and integrates with those components rather than creating a second copy that can silently diverge.