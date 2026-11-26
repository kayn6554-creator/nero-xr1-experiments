# 08 — 常见问题

## Hydra 找不到配置
确认项目配置已同步到 XR1：
- `configs/data/nero3_piper_ee_sb_20261125.yaml`
- `configs/model/posttrain_nero_ee_sb_full.yaml`
- `configs/trainer/deepspeed_npu_bf16_zero2.yaml`

如果目标文件已存在但内容不同，启动器会拒绝覆盖；人工比较后再处理。

## 找不到 Python / torch_npu
检查：
- CANN 环境是否 source；
- `.venv-piper-ef/bin/python` 是否存在；
- `torch`、`torch_npu`、`lightning`、`mmengine` 是否能 import。

## 数据路径不存在
确认 `/mnt/panhe/dataset/xr1_ee_sb_mix_20261125/` 下三个入口存在，软链接目标可读。

## 训练步数不一致
本项目 planner 会按 JsonDataset 的 dwell_keep 规则统计有效样本。仍需与实际 DataLoader / DistributedSampler 日志对照，尤其检查 world size、batch size、drop_last 和 accumulation。

## Checkpoint 恢复失败
检查 checkpoint 目录、DeepSpeed shards、训练配置兼容性和 `XR1_CKPT_PATH`。不要只拿 `mp_rank_00_model_states.pt` 当作完整续训 checkpoint。

## NERO EE 服务拒绝 action_layout
当前上游 OpenPI 服务没有 `nero_ee_sb` 分支。这是已知缺口，需要完成 EE State 与 IK adapter 集成，不要通过删除 layout 检查来绕过。

## IK 失败
不要发送失败或 NaN 关节目标。先检查：
- 当前 q0 是否在限位内；
- 目标 EE pose 是否可达；
- flange/TCP 变换是否正确；
- elbow target 是否超出 joint 3 限位；
- 局部增量坐标定义是否与训练一致。