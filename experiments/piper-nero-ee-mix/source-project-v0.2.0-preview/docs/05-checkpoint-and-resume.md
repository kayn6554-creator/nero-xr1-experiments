# 05 — Checkpoint 与断点续训

## 1. 代码中的保存行为

`tools/train.py` 上游默认逻辑：
- 默认 `save_interval` 从 Trainer 配置读取，当前为 1000。
- 未设置 `XR1_SAVE_EPOCHS` 时，默认 `XR1_SAVE_TOP_K=1`、`save_last="link"`，按 optimizer step 保存。
- 设置 `XR1_SAVE_EPOCHS` 后，使用 `SelectedEpochCheckpoint`，按指定 Epoch 结束时保存，`save_top_k=-1`、`save_last=False`；此时不会同时启用周期 `last.ckpt`。
- `XR1_DISABLE_CHECKPOINTS=1` 会关闭 checkpoint，不适用于正式训练。
- `trainer.fit(..., ckpt_path=ckpt_path)` 接收完整续训 checkpoint。

## 2. 完整续训 vs 推理加载

两者不能混淆：

**续训**应使用 Lightning/DeepSpeed checkpoint，通过 `trainer.ckpt_path` 恢复训练状态。理论上包括模型、优化器、scheduler、global step 等训练状态；但 Atlas 上是否完整恢复，还必须做一次中断/恢复实验验证。

**推理**由 `mibot/server/deploy.py` 加载 `mp_rank_00_model_states.pt` 的模型参数，严格加载到模型结构中。它不是完整续训 checkpoint 的替代品。

## 3. 检查 checkpoint

```bash
python scripts/training/inspect_checkpoint.py \
  --path /mnt/panhe/output/nero-piper-ee-sb-v0.1
```

工具会查找 `mp_rank_00_model_states.pt` 并显示 `last.ckpt` 的实际目标。

## 4. 恢复训练

默认正式训练只保存 Epoch 3，因此如果训练在 Epoch 3 完成前异常中断，默认策略没有周期 `last.ckpt` 可恢复。

若希望保留 Epoch 1/2/3 的完整 epoch-end checkpoint，可在正式训练前显式设置：
```bash
XR1_SAVE_EPOCHS=1,2,3
```

先检查上次输出目录与完整 checkpoint：
```bash
find /mnt/panhe/output/nero-piper-ee-sb-v0.1 -maxdepth 5 \
  -name mp_rank_00_model_states.pt -o -name '*.ckpt'
```

然后将实际存在的完整 checkpoint 路径传入：
```bash
bash scripts/training/resume_training.sh \
  /mnt/panhe/output/nero-piper-ee-sb-v0.1/epoch2-step<global_step>.ckpt
```

不要默认假设 `last.ckpt` 存在。

恢复前必须确认：
- checkpoint 路径存在且结构完整；
- 使用相同模型配置和兼容的数据/训练配置；
- `max_steps` 不小于已完成 global step；
- optimizer/scheduler 状态能在 Atlas 上恢复；
- 没有同时运行的旧训练进程。

## 5. 待实测项

必须做一次短训练 → detach → 重新 attach → 停止/恢复验证，核对：
- `global_step`
- optimizer state
- scheduler step / learning rate
- model weights
- dataloader/sampler 的恢复行为
- DeepSpeed ZeRO-2 checkpoint shard 完整性

不要把 tmux detach 描述成“保存 checkpoint”。它只让进程继续运行。


## 6. Epoch 3 正式部署产物

正式混训默认保存 Epoch 3，使用 XR1 `SelectedEpochCheckpoint`：
- `XR1_SAVE_EPOCHS=3`
- `XR1_SAVE_TOP_K=-1`
- 文件名 `epoch3-step<global_step>.ckpt`

新混训的 global_step 由数据量决定，不沿用旧 NERO-only 的 4026。部署前运行 `scripts/deployment/inspect_deployment_artifact.py` 并记录 config/weights SHA256。

注意：当前混训模型仍从基础 `Xiaomi-Robotics-1-5B/model_states.pt` 初始化，不是从旧 NERO Epoch 3 checkpoint 续训。