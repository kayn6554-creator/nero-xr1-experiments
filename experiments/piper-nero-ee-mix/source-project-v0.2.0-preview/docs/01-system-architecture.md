# 01 — 系统架构

## 1. 数据到模型

```text
NERO 7-DoF recordings ── FK ──┐
                              ├─ unified EE State / EE Action ── JsonDataset
Piper 6-DoF recordings ─ FK ──┘                                  │
                                                                  ▼
                                                      per-sample Action Mask
                                                                  │
                                                                  ▼
                                                        XR1 / BaseRunner
                                                                  │
                                                                  ▼
                                                      Lightning + DeepSpeed
                                                                  │
                                                                  ▼
                                                         NERO-target checkpoint
```

## 2. 训练入口

已从所提供源码确认：

```text
project train_nero_piper_ee_sb.sh
  └─ XR1/scripts/train.sh
       └─ python -m torch.distributed.run
            └─ XR1/tools/train.py
                 ├─ Hydra config
                 ├─ BaseDataModule / JsonDataset
                 ├─ BaseRunner
                 ├─ XR1 model
                 └─ Lightning Trainer + DeepSpeed ZeRO-2
```

本项目只管理专用配置、数据预检、启动参数和验证工具；XR1 核心模型与训练器仍由上游工程提供。

## 3. NERO 推理

```text
NERO observation:
  front image + wrist image + joint7 + gripper
                    │
                    ▼
             NERO forward kinematics
                    │
                    ▼
        60-D EE State (10 valid dimensions)
                    │
                    ▼
            q01/q99 state normalization
                    │
                    ▼
               XR1 generate()
                    │
                    ▼
       denormalize 30 × 60 EE action chunk
                    │
                    ▼
  current flange pose + local EE delta + elbow delta
                    │
                    ▼
               NERO inverse kinematics
                    │
                    ▼
            absolute NERO joint targets
                    │
                    ▼
         existing NERO client / controller
```

## 4. 坐标约定

训练代码使用：
- State：NERO/Piper 各自 FK 的法兰位姿，经各自 base alignment 变换后编码。
- Action：以当前帧法兰坐标系表达的局部相对位移与相对旋转。

当前 `ef_align.py` 中：
- `PIPER_TO_NERO_FLANGE_R = I`
- `PIPER_BASE_ALIGN = I`
- `NERO_BASE_ALIGN = I`

这是当前配置，不等于已完成完整的物理坐标标定。法兰姿态、base frame、TCP/gripper offset 必须在真机部署前再次验证。
