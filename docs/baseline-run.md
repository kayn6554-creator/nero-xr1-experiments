# 基线训练记录（学长，2026-10-31）

输出：`/mnt/wzh/output/xr1-910c-nero3-single-no-rtc-3epochs-bs10-20261031`
启动：`scripts/train_910c_nero3_3epochs.sh` → `scripts/train_910c_full_joint.sh` → `scripts/train.sh`

## 数据

- 转换脚本 `scripts/prepare_nero3_xr1.py`：LeRobot v3 → XR-1 JSON
- 3 个数据集 `nero_weigh_v1`、`nero_temp_v1`、`nero_wind_v1`，共 261 条、94125 帧，10 Hz
- 每条 JSON 的 `source` 里保留了 `step_index`、`item_index`、`camera_stamp_ms`，但训练时没用
- 非 success 的轨迹标为 `ongoing`；损失里 success 和 ongoing 没有区别
- 每条轨迹一句固定指令（`instruction.general`），训练时末尾加 ` /no_cot`

## 动作与状态

- 布局 `nero_joint`：60 维容器里，关节占 0–6 维，夹爪占第 7 维
- 关节：未来 30 帧动作相对当前状态的变化量（不累加）
- 夹爪：同样是变化量 = 未来主臂指令 − 当前从臂开度；归一化均值 0、std 0.1
- 状态：q01/q99 映射到 [-1, 1]；夹爪 q99 固定为 0.1

## 模型与训练

| 项目 | 值 |
|---|---|
| 初始化 | `Xiaomi-Robotics-1-5B/model_states.pt` |
| 方式 | 全参数微调（LoRA 关闭），DeepSpeed ZeRO-2，bf16 |
| async_train（RTC） | false |
| training_repeat | 4 |
| 芯片 | 8 × 910C，每芯片 batch 10，有效 batch 80 |
| 步数 | 3 epoch × 1177 = 3531 |
| 学习率 | cosine，warmup 500 步，5e-7 → 5e-6 → 5e-7 |
| 优化器 | FusedAdam，betas (0.9, 0.95)，weight decay 0.1，梯度裁剪 1.0 |
| seed | 42 |
| 耗时 | 约 2 小时（15:41 → 17:47） |

## Checkpoint 与部署

- 每个 checkpoint 72 GB，其中优化器状态 8 × 8.3 GB，模型权重 `mp_rank_00_model_states.pt` 11 GB
- 推理只需 `config.py` + `<ckpt>/checkpoint/mp_rank_00_model_states.pt`
- 推理服务 `scripts/serve_openpi_policy.py`：OpenPI websocket 协议；每次请求读取 `prompt`；Nero 夹爪还原为 `delta + state`；RTC 前缀已禁用
