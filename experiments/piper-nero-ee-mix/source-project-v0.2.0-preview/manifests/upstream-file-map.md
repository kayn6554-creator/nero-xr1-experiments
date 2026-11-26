# XR1 上游文件映射

本项目不复制整个 XR1 工程。以下文件必须存在于 `XR1_ROOT`，并由本项目启动器调用或由部署集成使用。

| XR1 文件 | 作用 | 本项目关系 |
|---|---|---|
| `scripts/train.sh` | torchrun 启动器 | 被本项目训练脚本调用 |
| `tools/train.py` | Hydra + Lightning 训练入口 | 上游训练核心 |
| `scripts/plan_json_epochs.py` | 原始 raw-frame 规划 | 本项目替代为 dwell-aware planner |
| `mibot/data/datasets/json_dataset.py` | JSON 数据加载、EE action/state、mask | 需应用 `nero_ee_sb` 补丁 |
| `mibot/utils/io.py` | action layout、mask、normalization | 需应用 `nero_ee_sb` 补丁 |
| `mibot/utils/ef_align.py` | 两机器人 FK / EE 对齐 | 由补丁包新增 |
| `mibot/utils/piper_kin.py` | Piper FK | 由补丁包新增 |
| `mibot/utils/nero_kin.py` | NERO FK/IK | 上游或当前实验版本 |
| `mibot/models/runner/base_runner.py` | pretrained、optimizer、scheduler、training_step | 上游训练核心 |
| `mibot/models/VLA/xr1.py` | XR1 forward/generate/loss | 上游模型核心 |
| `mibot/server/deploy.py` | checkpoint 模型加载、统计量加载 | 推理基础 |
| `scripts/serve_openpi_policy.py` | OpenPI WebSocket adapter | 当前不支持 `nero_ee_sb`，待扩展 |
| NERO client repository | 发送观测、接收 actions、执行 chunk | 本轮未提供，部署集成前必须检查 |

## 训练保存相关
- `tools/train.py`
- `mibot/utils/epoch_checkpoint.py`
- `configs/trainer/deepspeed_npu_bf16_zero2.yaml`

## 数据配置
- `configs/data/nero3_piper_ee_sb_20261125.yaml`
- `configs/model/posttrain_nero_ee_sb_full.yaml`


## NERO XR1 experiment repository (pinned main commit)

Repository: `https://github.com/kayn6554-creator/nero-xr1-experiments`  
Commit: `d4826f4f119a292d0ca061f637194000016360de`

| Upstream file | Role |
|---|---|
| `code/xr1-overlay/mibot/utils/nero_kin.py` | NERO FK/Jacobian/DLS IK |
| `code/xr1-overlay/mibot/utils/nero_steps.py` | Training/client step text |
| `code/inference-overlay/scripts/panhe/apply_ef_serve.py` | Legacy EF server-side IK conversion |
| `code/inference-overlay/scripts/panhe/apply_ef_fast.py` | EF IK tolerances and diagnostics |
| `code/collector/nero_vla_client.py` | NERO client request/absolute-joint output contract (older source snapshot) |
| `code/plugins/lerobot_robot_nero/` | Existing NERO follower/control/safety plugin |
| `code/plugins/lerobot_camera_ros/` | ROS RGB/depth camera plugin |
| `docs/deployment-20261116.md` | Epoch3 deployment and M5 Step Flag integration record |
| `patches/20261116-m4-ef4-m5-stepflag/` | M5 patch manifest/tests; full deployed clean ZIP is not committed |

The new `nero_ee_sb` server must combine legacy EF conversion with M5 Step Flag behavior and the new unified EE State. Neither old patch can be applied unchanged.


## Supplied Atlas server snapshot (2026-11-25)

The exact source files supplied in `nero_piper_train_deploy_chain_20261125.tar.gz` are preserved under `references/upstream/`:
- `scripts/serve_openpi_policy.py`
- `mibot/server/deploy.py`
- `mibot/server/runtime/server.py`
- `mibot/utils/ef_align.py`
- `mibot/utils/io.py`

Their SHA256 values are recorded in `manifests/atlas-server-source-snapshot.json`. This snapshot is useful for preparing a patch, but it must not be confused with the live 113 service after the 2026-11-16 M5 patch.
