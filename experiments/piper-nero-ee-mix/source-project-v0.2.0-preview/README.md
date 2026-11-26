# NERO × Piper EE Mixed Training

**Version:** `0.2.0-preview`  
**Date:** 2026-11-25

本项目用于把 Piper 左臂 V2 数据转换到统一的末端执行器（EE）状态/动作语义，与 NERO 数据混合训练，最终得到**部署到真实 NERO 七关节机械臂**的策略。

> 这是一个独立的训练与部署伴随工程，不复制整套 XR1/NERO 仓库。训练配置、数据检查、Epoch 3 checkpoint 规范和 `nero_ee_sb` 推理适配器已整理；113 WebSocket 服务集成、Atlas 实际训练和真机 Live 验证仍待完成。

## 目标

当前 NERO 部署基准为 Epoch 3。新混训正式训练默认 3 Epoch，并保存 Epoch 3；新模型的 global_step 以实际混训数据计划为准，不沿用旧模型的 `4026`。

- 训练：2× NERO + 1× Piper V2，统一使用 `nero_ee_sb`。
- State：统一为 60 维容器中的 10 个有效 EE 状态维度。
- Action：统一为 60 维容器中的 9 个有效动作维度。
- NERO 专属监督：保留 elbow（slot 7）与 Step Flag（slot 8）。
- Piper 缺少的 elbow / Step Flag 监督按样本 mask，不伪造标签。
- 部署：真实 NERO 观测 → EE State → 模型预测 → IK → NERO 关节目标。
- 验收：以 NERO 真机任务表现为准，不以混训 Loss 下降作为最终成功标准。

## 目录

- `configs/`：本项目使用的混训、模型和 Trainer 配置。
- `scripts/atlas/`：Atlas 环境检查与安全训练启动器。
- `scripts/validation/`：混合数据检查和考虑 Dwell Filter 的训练步数规划。
- `scripts/training/`：Checkpoint 检查和恢复启动入口。
- `scripts/deployment/`：Epoch 3 部署文件与 SHA256 核验。
- `src/nero_piper_ee/deployment/`：统一 EE State、EE Action→IK、Step Flag 适配模块。
- `patches/`：XR1 训练补丁和推理/采集端集成说明。
- `references/nero-xr1-experiments/`：固定 commit 的 NERO 参考代码 overlay。
- `references/upstream/`：本轮 Atlas 训练包中的 server/deploy 源码快照（不等同于 113 实际运行版本）。
- `tests/`：FK/IK 回环、EE State parity、Step Flag 与 metadata 测试。

## 运行前提

- Atlas 上已有 XR1 原始工程，默认路径 `/mnt/panhe/xr1`。
- Python 环境：`/mnt/panhe/xr1/.venv-piper-ef`。
- CANN 环境由系统提供，默认 `/usr/local/Ascend/cann-9.0.0/set_env.sh`。
- 本地预训练权重和 Qwen3-VL 权重已在配置指定路径就绪。
- 混训数据软链接目录已准备：`/mnt/panhe/dataset/xr1_ee_sb_mix_20261125/`。
- 不修改系统 CANN，不复用旧 `.venv`；本项目新增产物默认放在 `/mnt/panhe/`。

## 首次使用

1. 将本项目放在 Atlas 的 `/mnt/panhe/nero-piper-ee-mix/`。
2. 检查 `docs/03-atlas-environment.md`。
3. 先运行数据结构与 Dwell Filter 检查：

   ```bash
   XR1_ROOT=/mnt/panhe/xr1 \
   /mnt/panhe/xr1/.venv-piper-ef/bin/python \
   scripts/validation/validate_mix_dataset.py \
   --config configs/data/nero3_piper_ee_sb_20261125.yaml
   ```

4. 分别实例化一条 NERO 和 Piper 样本，检查 State / Action / Mask：

   ```bash
   XR1_ROOT=/mnt/panhe/xr1 PYTHONPATH=/mnt/panhe/xr1:$PWD/src \
   /mnt/panhe/xr1/.venv-piper-ef/bin/python \
   scripts/validation/validate_batch.py \
   --config configs/data/nero3_piper_ee_sb_20261125.yaml
   ```

5. 先做训练计划检查（默认不启动训练）：

   ```bash
   XR1_CHECK_ONLY=1 bash scripts/atlas/train_nero_piper_ee_sb.sh
   ```

6. 在检查 Atlas NPU 空闲状态、确认环境与 Batch 后，再显式允许启动训练。不要直接覆盖现有 `tmux train` 会话。

运行本地 EE-SB 转换、Step Flag 和 metadata 测试（使用固定的 NERO kinematics reference overlay）：

```bash
PYTHONPATH=$PWD/src:$PWD/references/nero-xr1-experiments/code/xr1-overlay \
python -m pytest -q tests
```

详细流程见 `docs/`，尤其是 `docs/09-nero-xr1-integration.md` 与 `docs/10-epoch3-checkpoint-lineage.md`。

## 重要安全边界

- 训练启动器默认 `XR1_CHECK_ONLY=1`，不会启动训练。
- 真正启动训练还需要显式设置 `XR1_ALLOW_TRAINING=1`。
- 脚本不会杀进程、关闭 tmux 或修改系统 CANN。
- 部署转换模块只计算目标关节，不直接连接或驱动机器人。
- 在完成坐标系、TCP、夹爪单位、IK 与客户端协议验证之前，不得将新模型直接用于无人值守真机执行。
