# Project Status — v0.2.0-preview

**更新日期：** 2026-11-25（v0.2.0）

## 已从所提供代码确认

- 训练入口为 `scripts/train.sh` → `torch.distributed.run` → `tools/train.py`。
- `tools/train.py` 使用 Lightning `Trainer.fit()`，并从 Hydra 配置构建 DataModule、BaseRunner 和 XR1 模型。
- 混训模型布局为 `nero_ee_sb`。
- 混训数据配置为 `configs/data/nero3_piper_ee_sb_20261125.yaml`。
- 模型配置为 `configs/model/posttrain_nero_ee_sb_full.yaml`。
- Trainer 配置使用 DeepSpeed ZeRO-2 与 BF16 mixed。
- 上游 `tools/train.py` 默认每 1000 optimizer steps 保存并保留 `last.ckpt`；本项目正式训练通过 `XR1_SAVE_EPOCHS=3` 改用 Epoch 3 选择性保存，Smoke Test 才使用周期保存。
- `trainer.fit(..., ckpt_path=...)` 是代码中的续训入口。
- `mibot/server/deploy.py` 的推理权重加载路径与训练续训路径不同。
- 当前 Atlas 训练包的 `scripts/serve_openpi_policy.py` 不接受 `nero_ee_sb`；旧 NERO `apply_ef_serve.py` 仅支持旧 `nero_ee` State contract。
- `mibot/utils/nero_kin.py` 提供 FK、带 elbow 目标的 IK 和关节限位。
- 已收录 2026-11-25 Atlas `serve_openpi_policy.py` 与 deploy/runtime 源码快照；已新增训练对齐的 `nero_ee_sb` State/Action/Step Flag adapter，但尚未集成到 113 当前运行的 WebSocket server。

## 数据配置记录

- NERO：504 episodes、141,330 raw frames；在索引中重复两份。
- Piper V2：2,603 episodes、293,693 raw frames。
- 2× NERO + 1× Piper 的 raw frame 总量约为 576,353；两侧 raw frame 占比接近 1:1。
- 配置启用 `dwell_keep`，因此有效训练 anchor 数小于或等于 raw frame 数。
- 当前混训 YAML 设置 `batch_size: 10`、`action_length: 30`、`gripper_absolute: true`。
- 当前混训 YAML 未开启 `step_prompt`；Step Flag 监督仍由 NERO 的 `source.step_index` 生成。

## v0.2.0 新增

- NERO XR1 GitHub main 固定 commit 的源代码映射与 reference overlay。
- Atlas 安全启动器（默认只检查，不训练）。
- 逐样本 Dwell Filter 计数器与训练步数规划器。
- Checkpoint 目录检查工具与续训启动包装。
- 训练同款 EE State 打包、EE Action→NERO FK/IK、Step Flag 边界 hold 与 OpenPI metadata contract。
- 固定 NERO 上游参考 commit、Epoch 3 checkpoint lineage 和 113/Client 集成说明。

## 本地验证结果

- `pytest`: 10 passed（NERO FK/IK、EE State parity、Step Flag 边界、metadata contract）。
- Python compile: PASS。
- Shell syntax: PASS。
- 尚未在 Atlas NPU、113 CUDA 服务或 NERO 真机上验证。

## 仍未完成 / 不可宣称已验证

1. 93/94 个依赖包在 Atlas 新环境中的完整离线安装与导入测试。
2. Atlas 上 NERO + Piper 混合 JsonDataset 的真实 Batch 测试。
3. 真实 `BaseDataModule` 实际 DataLoader batch 数与本项目 planner 的逐项对照。
4. DeepSpeed/Lightning 在 Atlas 上保存、断点恢复时 optimizer、scheduler、global step 与 sampler 状态的完整运行验证。
5. `PIPER_BASE_ALIGN`、`NERO_BASE_ALIGN`、`PIPER_TO_NERO_FLANGE_R` 的真机坐标/TCP 校验。
6. 新 EE 策略的 OpenPI WebSocket 服务集成。
7. NERO 真机客户端的请求字段、输出动作调度、Step Flag 接口核对。
8. IK 连续性、关节限位、夹爪范围、异常停止策略的真机验证。
9. NERO-only EF 基线与 NERO+Piper 混训模型的同条件对照实验。

## 当前阶段判断

**可进入数据预检和训练环境验证；不应直接跳过 Smoke Test 开始长时间训练；尚不能将模型直接部署到 NERO 真机。**


## NERO 仓库审查新增结论

- 已固定检查 `nero-xr1-experiments` commit `d4826f4f119a292d0ca061f637194000016360de`。
- 旧 `nero_ee` EF 服务输入仍是 joint7+gripper；新 `nero_ee_sb` 模型 State 是 10 维 EE State，不能直接套用旧 EF patch。
- NERO Client 仍可维持 state8 请求和 absolute joint8 回包；服务端承担 EE State packing 与 IK。
- 新混训配置 `step_prompt=false`，Step Flag 仍为 true；客户端必须把 step-file gate 与 Step Prompt 解耦。
- NERO `NeroFollower` 插件已经具备控制锁、急停保持、限位、单步限幅和状态过期检查，应直接复用。
- 旧 NERO Epoch3 `epoch3-step4026.ckpt` 与新混训 Epoch3 是不同模型血缘；新混训 global_step 不预设为 4026。
- 本地测试现覆盖 NERO FK/IK、EE State parity、Step Flag 边界和 metadata contract；未在 Atlas、113 或真机验证。