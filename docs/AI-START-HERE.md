# 新 AI 对话接手指南

**用途：** 让没有任何历史聊天记录的新 AI，仅凭本仓库就能理解项目、区分实验、找到代码并安全地继续工作。

## 1. 先理解项目是什么

我们不是从零预训练 VLA。项目主体是将 Xiaomi-Robotics-1（XR-1）下游适配到 NERO 七关节单臂 + 夹爪，围绕真实机器人多阶段操作，研究动作表示、步骤边界、数据质量、感知输入和部署可靠性。

实验室使用三类节点：

| 节点 | 角色 | 主要用途 |
|---|---|---|
| 102 | 采集 / 真机 | ROS、NERO 主从臂、相机、LeRobot 采集、测试网页 |
| 113 | 推理 | RTX 4090、OpenPI WebSocket、真机策略服务 |
| Atlas `.15` / `.33` | 训练 / 数据 | Ascend NPU、XR-1 训练代码、数据转换和训练 |

`.15` 与 `.33` 是不同主机，不能把两者路径、进程或资源状态视为同一台机器。登录入口和历史隧道见 `docs/infrastructure-access-20261126.md`。

## 2. 当前最重要的两个实验方向

<details>
<summary>实验线 A：M6 — NERO RGB-D（M5 的感知扩展）</summary>

- 目标：验证 front/wrist 深度信息能否改善 NERO 操作中的三维空间理解和动作决策。
- 保持：M5 的 NERO joint-space `nero_joint_sb`、30 步动作块、Step Flag 和人工确认闭环。
- 改变：模型输入从 front/wrist RGB 扩展为 RGB-D。
- 已有：102 ROS RGB-D 读取和深度视频存储基础。
- 未有：Client 请求、Atlas Dataset/Collate、模型深度分支、113 深度解码和完整训练/部署链。
- 计划初始化：M5 Epoch-3 模型权重；加载前必须做结构兼容检查。
- M5 线上服务保持不动。

详见 `docs/research-lines/M6-RGBD.md`。
</details>

<details>
<summary>实验线 B：Piper + NERO EE-SB 混训（独立平行实验）</summary>

- 目标：让 Piper V2 左臂数据增强 NERO 策略，最终在 NERO 真机上评估。
- 训练表示：统一 EE State/Action，布局 `nero_ee_sb`。
- 当前环境：Atlas 独立 `.venv-piper-ef` 已搭建，核心依赖导入通过。
- 数据：2 份 NERO 索引 + 1 份 Piper V2 索引；真实 JsonDataset 扫描得到 486,500 个 Dwell Filter 后有效 anchor。
- 训练计划：8 个 NPU 逻辑设备、每卡 batch 10、accumulation 1、3 Epoch，18,246 optimizer steps。
- 样本验证：NERO 与 Piper 各一条真实 JSON、State/Action/Mask 和两路图像解码通过。
- 阻塞：两次 Smoke 尝试 OOM；用户补充其中有单卡尝试。当时 `.15` / `.33` 的共享 NPU 被双机 vLLM 任务占用，但原始 OOM 日志未归档，不能把两次尝试说成完全相同的卡数/参数；正式训练尚未开始。
- 不要把此线误认为 M6，也不要将 M5 checkpoint 误作本线的初始化权重。本线当前模型配置从 XR1 基础预训练权重初始化。

详见 `docs/research-lines/PIPER-NERO-EE-MIX.md`。
</details>

## 3. 建议的新对话工作顺序

1. 阅读 `docs/AI-HANDOFF-2026-11-26.md`，先确认最新状态和阻塞。
2. 如果处理 M6，阅读 `docs/research-lines/M6-RGBD.md`。
3. 如果处理 Piper 混训，阅读 `docs/research-lines/PIPER-NERO-EE-MIX.md`，再查看 `experiments/piper-nero-ee-mix/`。
4. 根据具体任务检查 `code/` 下的正式 overlay；若版本不明，回到 `code/snapshots/` 或实验线的原始归档。
5. 先做只读检查、代码审查和本地单元测试。Atlas、113 和真机操作必须按实时资源与权限重新确认。

## 4. 代码目录如何理解

| 目录 | 作用 | 使用规则 |
|---|---|---|
| `code/xr1-overlay/` | Atlas XR-1 自有改动、配置、脚本 | 不是完整上游 XR-1 仓库 |
| `code/inference-overlay/` | 113 推理服务补丁和工具 | 不能直接覆盖正在运行的 113 文件 |
| `code/collector/` | 102 Client/Web/采集辅助代码 | 部分为历史快照，先核对部署版本 |
| `code/plugins/` | NERO follower/teleoperator/相机插件 | 保留现有控制保护 |
| `code/snapshots/` | 机器来源快照 | 只读归档，不要覆盖 |
| `experiments/piper-nero-ee-mix/` | Piper-NERO 伴随工程和 Atlas 导出 | 区分原始包、项目启动器、实际 Atlas 脚本 |

## 5. 这个仓库能做什么、不能单独做什么

仓库可以让 AI 继续进行：源码阅读、数据/配置契约检查、补丁编写、静态测试、实验计划、部署适配设计、文档和报告维护。

仓库**不是完整运行环境**。真正训练和真机测试还依赖：
- 实验室授权的完整 Xiaomi-Robotics-1 / XPolicyLab 上游代码；
- Atlas 真实数据集、预训练权重、Qwen3-VL 权重和当前环境；
- 113 当前正在运行的 server/client 版本；
- 102 真机、ROS 驱动、相机、机械臂及现场安全人员。

这些二进制和私有数据不提交 GitHub。若工作进入上述环节，AI 应先核对机器上的实时版本，不能从仓库快照猜测。

## 6. 不要重复询问用户已经提供过的事实

本仓库已经记录：实验线定义、机器分工、路径、训练配置、数据统计、Step Flag 语义、当前阻塞、已完成验证和未完成部署链。新 AI 应先搜索文档与源文件；只有实时状态、权限或研究选择确实缺失时才向用户提问。

## 7. 新对话首条提示词

可以直接复制下面这段给一个全新 AI 对话：

```text
你现在接手 NERO × XR-1 实验仓库。请先完整阅读 AGENTS.md、README.md、docs/AI-START-HERE.md、docs/AI-HANDOFF-2026-11-26.md，再阅读本轮任务对应的研究线文档。

不要要求我重复项目背景。请严格区分 M6 RGB-D 与 Piper + NERO EE-SB 两条平行实验线；区分已在仓库验证、用户报告、历史快照、研究建议和未知事项。先检查仓库现有代码和文档，不要凭印象重写。

除非我明确要求，不要启动 Atlas 训练、停止任何共享进程/容器、修改 102/113 在线服务或执行 NERO 真机动作。所有远端资源状态必须实时核对。

开始工作前先告诉我：你理解的当前状态、当前任务最直接的文件入口、可以独立完成的工作，以及真正需要外部数据/权限的阻塞。然后直接开展可独立完成的工作，不要重复询问已有信息。
```

## 8. 本轮审计新增入口

- 仓库可接手性审计：`docs/REPO-AUDIT-2026-11-26.md`。
- Piper Atlas 只读前检：`experiments/piper-nero-ee-mix/preflight_atlas.sh`。
- 真实 JsonDataset epoch / Smoke index 复核：`experiments/piper-nero-ee-mix/validation/validate_runtime_dataset.py`。
- M6 深度录制帧对齐修复：`code/collector/nero_depth.py`，对应测试 `tests/test_nero_depth_alignment.py`。该修复尚未部署到 102。

两份 Piper 压缩包的上传顺序和权威关系见 `experiments/piper-nero-ee-mix/README.md`；不要混用伴随工程 launcher 与 Atlas 实际 launcher。

## 9. 本地运行 Piper 伴随工程测试

从仓库根目录执行：

```bash
bash experiments/piper-nero-ee-mix/run_companion_tests.sh
```

不要为了修复 pytest 导入路径而修改原始 `source-project-v0.2.0-preview/` 归档。
