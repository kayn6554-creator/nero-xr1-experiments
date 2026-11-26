# 2026-11-26 仓库可接手性审计

## 结论

仓库现在可以让一个没有历史聊天记录的新 AI **理解项目、定位代码、审查两条实验线、运行本地测试、准备 Atlas 前检**。但仓库不是完整运行环境；不能仅凭 GitHub 自动完成 NPU 训练、读取真实数据、部署 113 服务或控制 NERO 真机。

本次审计的目标不是让文档看起来完整，而是明确：哪些事情 AI 可以直接开始，哪些事情必须等待真实机器、数据或权限。

## 1. 新 AI 可以不问用户就开始的工作

- 阅读 `AGENTS.md`、`docs/AI-START-HERE.md`、`docs/AI-HANDOFF-2026-11-26.md`，理解 M0–M5 历史和当前两条平行线。
- M6：沿 `docs/research-lines/M6-RGBD.md` 的源码地图继续审查 RGB-D 采集、Dataset、Collate、XR1、Client/Server 接口；维护 `code/collector/nero_depth.py` 的单元测试。
- Piper：沿 `docs/research-lines/PIPER-NERO-EE-MIX.md` 和 `experiments/piper-nero-ee-mix/` 审查真实 Atlas launcher、数据契约、训练步数和部署 adapter；运行本地 companion project 测试；准备 `preflight_atlas.sh`。
- 在仓库范围内补代码、测试、配置校验和文档；所有结果必须注明是本地 CPU 验证、用户报告还是 Atlas/真机实测。

## 2. 本轮发现并处理的真实缺口

### M6：深度录制可能出现开头帧错位

原 `DepthRecorder.add()` 只有在第一张有效深度到来时才创建视频 writer。如果 episode 开头 RGB 已经保存但 depth 为 None，之后生成的深度视频会缺少开头帧，按 RGB frame_index 读取时可能静默错位。

已在工作 overlay `code/collector/nero_depth.py` 修复：首次有效深度到来时回填前面缺失的零帧；中间缺帧写零；全程无深度明确标记失败；检查帧数与尺寸。新增 `tests/test_nero_depth_alignment.py`，本地 4 passed。原始三端快照未修改。

**仍需现场工作：** 该修复尚未部署到 102，尚未在真实 ROS / PyAV 环境做短 episode 录制与回读验证。

### Piper：缺少能复核真实 JsonDataset 的一键前检

此前只有 Dwell Filter mirror planner 和单条 NERO/Piper sample validator；没有仓库脚本能在开训前直接实例化真实 `JsonDataset(finite_epoch=True)`，重新核对 486,500 anchors / 6,082 steps / 18,246 max steps，也没有确认 3-step Smoke 实际截取的样本 index 同时包含两种机器人。

已新增：
- `experiments/piper-nero-ee-mix/validation/validate_runtime_dataset.py`：真实 JsonDataset epoch 与 Smoke index 检查。
- `experiments/piper-nero-ee-mix/atlas-source-sha256.txt`：Atlas 关键源码和配置哈希。
- `experiments/piper-nero-ee-mix/preflight_atlas.sh`：串联环境导入、源码哈希、数据扫描、样本检查、Atlas launcher check-only 和资源快照。

这些工具只在本地完成语法/编译检查，尚未在 Atlas 运行；它们不能证明 NPU 空闲，也不会启动训练。

## 3. 两条实验线的最新准确状态

### M6 RGB-D

- RGB-D ROS 读取与独立深度视频保存代码存在。
- Client 当前推理观测仍只送两路 RGB；113 M5 服务仍只处理 RGB；Atlas Dataset / Collate / XR1 模型尚未接入 depth。
- M6 计划从 M5 Epoch-3 warm-start，保留 `nero_joint_sb`、30-step 和 Step Flag，只增加观察模态。
- M6 后续建议按“数据完整性 → 独立深度 encoder → 显式 3D 几何 → 闭环视觉伺服 / 不确定性”逐级推进。详见 M6 专项文档第 13 节。

### Piper + NERO EE-SB

- Atlas `.venv-piper-ef` 与 94 个离线依赖安装、核心导入和真实样本检查均为用户报告的已完成实测。
- 数据：3,611 JSON，576,353 raw frames，Dwell Filter 后 486,500 anchors；8 NPU / batch10 / accumulation1 下 6,082 steps/epoch，3 epochs 18,246 steps。
- 正式训练尚未开始。两次 Smoke 尝试 OOM；用户补充其中有单卡 Smoke 尝试，当时 `.15` / `.33` 的 NPU 被双机 vLLM 任务占用。
- 当前 Atlas 实际 launcher SHA256：`265da9feb4f9204952a0076ba5d14b33d66933cb89bf01bcdc8195fdcc104d81`。不应与 companion project launcher 混用。
- 资源释放并获得授权后，先运行 `preflight_atlas.sh`，再按 8-NPU 正式拓扑做 3-step Smoke；Smoke 通过后才正式 3-Epoch。

## 4. 新对话的来源优先级

1. 当前用户在本轮给出的最新事实与权限决定。
2. `docs/AI-HANDOFF-2026-11-26.md` 和对应专项研究文档。
3. 第一份上传包 `nero_piper_ee_atlas_export_20261126.tar.gz`：Atlas 实际状态导出。
4. 第二份上传包 `nero-piper-ee-mix-v0.2.0-preview(1).zip`：伴随工程代码和整理报告。
5. `code/snapshots/`：按来源保存的只读历史快照。
6. 历史 README、旧对话和旧脚本；必须检查日期和适用范围。

不要把两个 Piper launcher、两份 YAML 或 Atlas `.15` / `.33` 当成同一对象。

## 5. 仓库无法单独提供的内容

这些不是文档缺失，而是有意不提交的运行材料或尚未采集的证据：

- NERO / Piper 原始数据集、RGB-D episode、原始视频和完整 `depth_info.json` 样本。
- Xiaomi-Robotics-1 / Qwen3-VL 模型权重、M5 checkpoint、Piper 混训 checkpoint。
- Atlas `.venv-piper-ef` 二进制环境、完整 wheels 缓存、实时 NPU 分配和当前进程所有权。
- 113 当前运行服务的完整文件版本、GPU 分配和当前 PID；仓库服务快照不代表当前在线。
- 102 完整 ROS 驱动/工作空间构建信息、真实相机标定文件和现场 RGB-D 样本。
- M6 depth 请求/模型融合的真实端到端测试；Piper `nero_ee_sb` 的 113 WebSocket server 与 Client 集成。

需要上述内容时，新 AI 应先检查仓库是否已有更新快照，再使用已授权的连接或请用户提供精确文件；不得根据历史路径猜测实时状态。

## 6. 新 AI 的最短启动顺序

1. 读 `AGENTS.md`。
2. 读 `docs/AI-START-HERE.md`。
3. 读 `docs/AI-HANDOFF-2026-11-26.md`。
4. 根据本轮任务只选 M6 或 Piper 专项文档，不把两条线混成一条。
5. 改代码前先对照 `code/snapshots/` 与当前 overlay；完成后运行相关测试并更新 handoff。

## 7. 可直接复制给新 AI 的首条提示词

```text
你现在接手 NERO × XR-1 实验仓库。请先完整阅读 AGENTS.md、README.md、docs/AI-START-HERE.md、docs/AI-HANDOFF-2026-11-26.md，再阅读本轮任务对应的研究线文档。

不要要求我重复项目背景。请严格区分 M6 RGB-D 与 Piper + NERO EE-SB 两条平行实验线；区分已在仓库验证、用户报告、历史快照、研究建议和未知事项。先检查仓库现有代码和文档，不要凭印象重写。

除非我明确要求，不要启动 Atlas 训练、停止任何共享进程/容器、修改 102/113 在线服务或执行 NERO 真机动作。所有远端资源状态必须实时核对。

开始工作前先告诉我：你理解的当前状态、当前任务最直接的文件入口、可以独立完成的工作，以及真正需要外部数据/权限的阻塞。然后直接开展可独立完成的工作，不要重复询问已有信息。
```

## 8. 审计后的总判断

现在一个新 AI 可以从仓库开始继续“代码和研究工作”，不再需要先补一大段历史聊天。它仍然不能绕过真实数据、模型权重、Atlas 资源和现场安全要求；这类依赖已被明确标出，而不是用猜测填平。
