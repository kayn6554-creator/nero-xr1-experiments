# 04 — 混合训练

## 1. 配置

- Data：`nero3_piper_ee_sb_20261125`
- Model：`posttrain_nero_ee_sb_full`
- Trainer：`deepspeed_npu_bf16_zero2`
- Action layout：`nero_ee_sb`
- Batch size：当前混训 YAML 为 10。
- Action horizon：30。
- 默认训练 3 个完整 Epoch（与 NERO 专用训练入口一致）；默认只保存 Epoch 3。
- 默认 checkpoint interval：1000 optimizer steps。

原始 `train_910c_full_joint.sh` 的默认值是 Piper Joint 配置、`.venv/bin/python`、旧输出目录，因此不要直接把它当作本项目启动器。

## 2. 先检查数据

```bash
XR1_ROOT=/mnt/panhe/xr1 \
/mnt/panhe/xr1/.venv-piper-ef/bin/python \
scripts/validation/validate_mix_dataset.py \
--config configs/data/nero3_piper_ee_sb_20261125.yaml
```

该检查会按 JsonDataset 的 Dwell Filter 规则统计有效 anchor 数。

## 3. 实例化 NERO / Piper 样本

```bash
XR1_ROOT=/mnt/panhe/xr1 PYTHONPATH=/mnt/panhe/xr1:$PWD/src \
/mnt/panhe/xr1/.venv-piper-ef/bin/python \
scripts/validation/validate_batch.py \
--config configs/data/nero3_piper_ee_sb_20261125.yaml
```

它会分别从 NERO 与 Piper 根目录实例化 `JsonDataset`，检查 state/action/mask 形状、有限值，以及 Piper 的 elbow/Step Flag mask。

## 4. 先做训练计划，不启动训练

```bash
XR1_CHECK_ONLY=1 \
bash scripts/atlas/train_nero_piper_ee_sb.sh
```

默认 `XR1_CHECK_ONLY=1`，只检查路径、同步项目配置快照并打印计划，不会启动 torchrun。

## 5. 三步 Smoke Test

先检查 NPU 空闲状态。不要自动结束其他 tmux 或训练进程。

推荐在单独的 tmux 会话中运行，不复用现有 `train` 会话。若单卡 DeepSpeed 配置在 Atlas 不适用，则由实验负责人确认后改用 8 卡 Smoke Test。

```bash
tmux new -s nero-piper-ee-smoke
```

在新会话内：
```bash
ASCEND_RT_VISIBLE_DEVICES=0 \
XR1_BATCH_SIZE=1 \
XR1_MAX_STEPS_OVERRIDE=3 \
XR1_SAVE_INTERVAL=1 \
XR1_CHECK_ONLY=0 \
XR1_ALLOW_TRAINING=1 \
bash scripts/atlas/train_nero_piper_ee_sb.sh
```

Smoke Test 通过条件：
- 模型与预训练权重加载成功。
- 至少一个包含 NERO 样本、一个包含 Piper 样本的 Batch 能进入训练。
- Forward / Backward 成功。
- Loss 有限。
- Smoke Test 使用固定小步数与周期保存，检查 checkpoint 写入；正式训练则使用 Epoch 3 选择性保存。
- 日志记录配置、数据路径和 global step。

## 6. 正式训练

Smoke Test、数据集读取、NPU 可用性和训练配置确认后：

```bash
XR1_CHECK_ONLY=0 \
XR1_ALLOW_TRAINING=1 \
bash scripts/atlas/train_nero_piper_ee_sb.sh
```

默认输出：
`/mnt/panhe/output/nero-piper-ee-sb-v0.1/`

默认使用 8 个可见 NPU、batch size 10、3 epochs；默认通过 `SelectedEpochCheckpoint` 保存 `epoch3-step<global_step>.ckpt`。若需要更改，使用 `XR1_BATCH_SIZE`、`XR1_EPOCHS`、`ASCEND_RT_VISIBLE_DEVICES` 等变量显式指定。

## 7. 重要

- 不要在已有任务占卡时直接开训。
- 不要关闭 checkpoint。
- 不要把训练启动脚本直接放进旧 `tmux train` 会话覆盖原任务。
- 训练前保存配置快照和数据 manifest。


正式训练默认与 NERO `train_nero.sh` 对齐：`XR1_EPOCHS=3`、`XR1_SAVE_EPOCHS=3`、`XR1_SAVE_TOP_K=-1`、`XR1_NUM_WORKERS=8`。完整 checkpoint 血缘见 `docs/10-epoch3-checkpoint-lineage.md`。
