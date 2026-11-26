# Pinned upstream source map

All links below are pinned to commit `d4826f4f119a292d0ca061f637194000016360de`.

| Source | Role in this project | Integration decision |
|---|---|---|
| [nero_kin.py](https://github.com/kayn6554-creator/nero-xr1-experiments/blob/d4826f4f119a292d0ca061f637194000016360de/code/xr1-overlay/mibot/utils/nero_kin.py) | NERO FK / Jacobian / DLS IK / limits | Reuse; do not implement a second NERO kinematics model |
| [nero_steps.py](https://github.com/kayn6554-creator/nero-xr1-experiments/blob/d4826f4f119a292d0ca061f637194000016360de/code/xr1-overlay/mibot/utils/nero_steps.py) | Training/client step language | Reuse only when the model was trained with `step_prompt` |
| [apply_ef_serve.py](https://github.com/kayn6554-creator/nero-xr1-experiments/blob/d4826f4f119a292d0ca061f637194000016360de/code/inference-overlay/scripts/panhe/apply_ef_serve.py) | Legacy `nero_ee` server adapter | Reuse its FK/IK conversion logic as a reference; not directly compatible with unified EE State + Step Flag |
| [apply_ef_fast.py](https://github.com/kayn6554-creator/nero-xr1-experiments/blob/d4826f4f119a292d0ca061f637194000016360de/code/inference-overlay/scripts/panhe/apply_ef_fast.py) | EF IK tolerances / diagnostics | Reuse the established 0.1 mm / 0.001 rad defaults and 15° jump guard |
| [nero_vla_client.py](https://github.com/kayn6554-creator/nero-xr1-experiments/blob/d4826f4f119a292d0ca061f637194000016360de/code/collector/nero_vla_client.py) | NERO WebSocket client, RGB request, chunk execution | Reference request/output contract; this committed snapshot predates the full 11-16 M5 client patch |
| [nero_vla_presets.json](https://github.com/kayn6554-creator/nero-xr1-experiments/blob/d4826f4f119a292d0ca061f637194000016360de/code/collector/nero_vla_presets.json) | Existing model preset structure | Add new model preset only after a real server port is assigned |
| [nero_follower.py](https://github.com/kayn6554-creator/nero-xr1-experiments/blob/d4826f4f119a292d0ca061f637194000016360de/code/plugins/lerobot_robot_nero/lerobot_robot_nero/nero_follower.py) | Real NERO follower and safety gate | Keep as the robot execution layer |
| [config_nero_follower.py](https://github.com/kayn6554-creator/nero-xr1-experiments/blob/d4826f4f119a292d0ca061f637194000016360de/code/plugins/lerobot_robot_nero/lerobot_robot_nero/config_nero_follower.py) | Joint limits, speed/step limits, state freshness | Preserve; do not bypass |
| [ros_image_camera.py](https://github.com/kayn6554-creator/nero-xr1-experiments/blob/d4826f4f119a292d0ca061f637194000016360de/code/plugins/lerobot_camera_ros/lerobot_camera_ros/ros_image_camera.py) | ROS image/depth camera plugin | Keep as optional camera implementation; current mixed model still uses front/wrist RGB |
| [deployment-20261116.md](https://github.com/kayn6554-creator/nero-xr1-experiments/blob/d4826f4f119a292d0ca061f637194000016360de/docs/deployment-20261116.md) | Epoch3 deployment record and M5 Step Flag contract | Source of deployment lineage and client/server expectations |
| [model-matrix.md](https://github.com/kayn6554-creator/nero-xr1-experiments/blob/d4826f4f119a292d0ca061f637194000016360de/docs/model-matrix.md) | M0–M5 / EF / EF4 semantics | Source for legacy model distinctions |
| [operations-handbook.md](https://github.com/kayn6554-creator/nero-xr1-experiments/blob/d4826f4f119a292d0ca061f637194000016360de/docs/operations-handbook.md) | Atlas, inference machine, collector and checkpoint workflows | Reuse operational constraints; do not copy secrets |
