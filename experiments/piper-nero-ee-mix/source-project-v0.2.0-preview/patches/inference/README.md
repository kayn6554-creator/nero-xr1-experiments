# Inference integration: `nero_ee_sb`

The existing NERO repository has two separate precedents:
- EF `nero_ee`: EE action → IK, but model State remains joint7+gripper.
- M5 `nero_joint_sb`: Step Flag boundary detection/hold, but actions remain joint-space deltas.

The new mixed model requires both features plus unified EE State. Reuse:
- `src/nero_piper_ee/deployment/nero_ee_sb_server_adapter.py`
- `src/nero_piper_ee/deployment/ee_to_nero.py`

Do not run `apply_ef_serve.py` unchanged on a `nero_ee_sb` model. The exact current 113 `serve_openpi_policy.py` source has not yet been collected into this project; therefore this folder intentionally contains no blind text-replacement script.

Before generating a patch, collect the running 113 server source, its SHA256, current `nero_ee`/`nero_joint_sb` branches and the exact environment. Then add only the new `nero_ee_sb` branch and test it against the reference adapter.