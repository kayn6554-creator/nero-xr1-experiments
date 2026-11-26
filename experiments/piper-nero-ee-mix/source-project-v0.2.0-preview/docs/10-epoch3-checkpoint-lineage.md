# 10 — Epoch 3 部署版本与训练产物血缘

## 1. 当前部署基准

用户本轮确认：**当前 NERO 部署使用 Epoch 3。**

NERO XR1 仓库 `docs/deployment-20261116.md` 记录：
- 2026-11-16 M4 / EF4 / M5 部署使用 checkpoint 名称 `epoch3-step4026.ckpt`。
- M4、EF4 权重文件大小均记录为 `11,004,891,197 bytes`。
- 文档明确给出了 M4、EF4 的权重路径；M5 的 checkpoint 文件大小没有在该段单独列出。
- 同一仓库模型矩阵记录 M3、M4、EF4、M5 的训练步数为 4026。

不要把旧模型 `epoch3-step4026.ckpt` 的步数直接套到新混训模型。

## 2. 新混训默认训练 3 个完整 Epoch

项目训练入口现与 NERO `scripts/panhe/train_nero.sh` 的默认策略对齐：
- 默认 `XR1_EPOCHS=3`。
- `finite_epoch=true`，每个 Epoch 遍历完整的 Dwell Filter 后数据集。
- 使用混合数据实际有效样本数、world size、per-device batch 和 gradient accumulation 计算 max_steps。
- 默认 `XR1_SAVE_EPOCHS=3`。
- 默认 `XR1_SAVE_TOP_K=-1`，由 `SelectedEpochCheckpoint` 保存选定 Epoch。
- 预期 checkpoint 命名为 `epoch3-step<实际global_step>.ckpt`。

新混训数据量与旧 NERO-only 数据量不同，所以新 checkpoint 的 step 数不会预设为 4026。以实际训练日志和生成目录为准。

## 3. 新混训的初始化权重与 Epoch 3 部署权重不是一回事

当前 `configs/model/posttrain_nero_ee_sb_full.yaml` 明确使用：

`/mnt/wzh/checkpoints/Xiaomi-Robotics-1-5B/model_states.pt`

这是 Xiaomi-Robotics-1 的基础预训练权重。

因此当前方案是：
- 从 XR1 基础预训练权重初始化；
- 用 NERO + Piper V2 EE-SB 混合数据训练；
- 训练 3 个 Epoch；
- 选取新训练产物 Epoch 3；
- 经过部署检查后再部署到 NERO。

它**不是**从旧 NERO M5 / M3 的 `epoch3-step4026.ckpt` 继续训练。除非明确决定更改初始化方案，否则不要擅自把旧 NERO checkpoint 填入 `model.params.pretrained`。

## 4. 训练启动前检查

```bash
XR1_CHECK_ONLY=1 bash scripts/atlas/train_nero_piper_ee_sb.sh
```

默认计划应显示：
- epochs = 3
- selected checkpoint epoch = 3
- `finite_epoch=true`
- 训练预算由有效样本数计算

Smoke Test 使用固定小步数时，不走 Epoch 3 checkpoint callback；它与正式 Epoch 3 保存策略分开。

## 5. 训练完成后定位 Epoch 3

不要使用未核对的 `epoch*` 通配符自动选择模型。先列出实际 checkpoint：

```bash
find /mnt/panhe/output/nero-piper-ee-sb-v0.1 -maxdepth 4 \
  -type f -name 'mp_rank_00_model_states.pt' -print
find /mnt/panhe/output/nero-piper-ee-sb-v0.1 -maxdepth 2 \
  -type d -name 'epoch*.ckpt' -print
```

确认：
1. checkpoint 目录名称包含 `epoch3`；
2. 对应 `checkpoint/mp_rank_00_model_states.pt` 存在且非空；
3. 同一训练输出目录的 `config.py` 中 `action_layout=nero_ee_sb`；
4. 日志显示 Epoch 3 正常结束，而不是异常中断；
5. 记录 global_step、权重 SHA256、config SHA256。

## 6. 部署文件核验

```bash
python scripts/deployment/inspect_deployment_artifact.py \
  --model-dir /data/hep/panhe/models/nero-piper-ee-sb \
  --checkpoint epoch3-step<实际global_step>.ckpt \
  --expected-layout nero_ee_sb \
  --output /data/hep/panhe/models/nero-piper-ee-sb/deployment-manifest.json
```

核验结果写入部署 manifest。权重哈希只能证明文件身份，不能证明模型成功加载或真机安全。

## 7. 禁止混用

- 不要把旧 M5 `config.py` 与新混训 Epoch 3 权重混用。
- 不要把旧 `epoch3-step4026.ckpt` 目录名硬编码为新混训模型 checkpoint。
- 不要把基础预训练权重误称为 NERO Epoch 3。
- 不要覆盖已有 M3/M4/EF4/M5 模型目录。
- 不要在没有记录 SHA256 的情况下把模型从 Atlas 复制到 113。
