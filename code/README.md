# 代码改动（叠加在上游 xr1 上）

`xr1-overlay/` 里的文件按原路径复制到上游 xr1 目录（服务器上是 `/mnt/panhe/xr1`），然后按顺序运行两个修改脚本：

```bash
cp -r code/xr1-overlay/* /mnt/panhe/xr1/
cd /mnt/panhe/xr1
.venv/bin/python scripts/panhe/apply_m1.py   # 步骤文本 + 启动脚本路径 + 推理服务回传指令
.venv/bin/python scripts/panhe/apply_m2.py   # 夹爪绝对目标 + 停留段采样 + 推理服务夹爪还原 + 固定步数
.venv/bin/python scripts/panhe/make_m2_config.py
```

修改脚本对每处改动先确认原文恰好出现一次，不符合就整体停止、不写入。所有新功能都是开关，默认关闭，关闭时行为和上游一致。

## 文件

| 文件 | 作用 |
|---|---|
| `mibot/utils/nero_steps.py` | 步骤文本的唯一定义，训练和客户端共用 |
| `scripts/panhe/apply_m1.py` | M1：`JsonDataset` 加 `step_prompt`；`train_910c_full_joint.sh` 的代码路径可覆盖；推理服务响应带 `prompt_used` |
| `scripts/panhe/apply_m2.py` | M2：`gripper_absolute`、`dwell_keep/dwell_tail` 停留段采样；推理服务按配置还原夹爪；`train_nero.sh` 支持 `XR1_MAX_STEPS` |
| `scripts/panhe/make_m2_config.py` | 生成 `configs/data/nero3_m2_20261106.yaml` |
| `scripts/panhe/prepare_nero_fail_prefix.py` | 失败数据 → XR-1 JSON，只保留失败步之前的步骤 |
| `scripts/panhe/train_nero.sh` | 训练启动：照抄基线超参数；支持 `XR1_CHECK_ONLY`、`XR1_SMOKE_STEPS`、`XR1_MAX_STEPS`、4 芯片 + 梯度累积 |
| `scripts/panhe/queue_weekend.sh` | 周五夜间队列：冒烟 → M1 → M2 → M2b |
| `scripts/panhe/check_steps.py` | 检查数据里的 `step_index` |
| `scripts/panhe/check_prompts.py` | 打印每步切换处的训练指令 |
| `scripts/panhe/dwell_stats.py` | 示范数据停留段统计（A3） |
| `scripts/panhe/check_m2.py` | M2 数据检查：样本数、删除统计、步数、夹爪范围、读一个完整样本 |

## 训练命令

```bash
# M1
bash scripts/panhe/train_nero.sh m1-step-prompt +data.params.train_datasets.step_prompt=true
# M2
XR1_DATA_CONFIG=nero3_m2_20261106 XR1_MAX_STEPS=2793 bash scripts/panhe/train_nero.sh m2-full +data.params.train_datasets.step_prompt=true
# M2b
XR1_DATA_CONFIG=nero3_m2_20261106 XR1_MAX_STEPS=2793 bash scripts/panhe/train_nero.sh m2b-no-step
```
