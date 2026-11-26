# NERO × XR-1 实验仓库

NERO 七关节单臂上的 Xiaomi-Robotics-1（XR-1）下游适配、训练、部署与真实机器人实验档案。仓库保存项目自有改动、配置、补丁、代码快照、实验协议和 AI 交接材料；不包含完整实验室上游仓库、数据集、模型权重或运行环境。

> **新 AI / 新同学请先读：** [`AGENTS.md`](AGENTS.md) → [`docs/AI-START-HERE.md`](docs/AI-START-HERE.md) → [`docs/AI-HANDOFF-2026-11-26.md`](docs/AI-HANDOFF-2026-11-26.md)。不要只根据旧 README 或历史实验记录判断当前状态。

## 当前项目状态（2026-11-26）

既有主线已经迭代到 M5 Step Flag。现在有两条彼此独立的后续实验线：

| 实验线 | 研究问题 | 当前状态 | 核心约束 |
|---|---|---|---|
| **M6 RGB-D** | 深度信息是否改善 NERO 的空间理解、接近和接触动作 | 设计与数据接入准备 | 以 M5 为基线，保持 `nero_joint_sb` / 30-step / Step Flag；只扩展 RGB-D |
| **Piper + NERO EE-SB 混训** | Piper V2 数据能否增强 NERO 策略 | Atlas 环境、样本和步数预检完成；Smoke 因共享 NPU 占用受阻，尚未正式训练 | 独立 `nero_ee_sb` 动作/状态空间，NERO 是最终部署目标 |

**不要把两条线合并。** M6 第一阶段不混入 Piper；Piper 混训不等于 M6，也不从 M5 checkpoint 初始化。

## 最新交接与研究文档

- [新 AI 接手指南](docs/AI-START-HERE.md)
- [当前状态 / 已完成 / 阻塞 / 下一步](docs/AI-HANDOFF-2026-11-26.md)
- [M6：NERO RGB-D 研究与工程路线](docs/research-lines/M6-RGBD.md)
- [Piper + NERO EE-SB 混训方案与训练前清单](docs/research-lines/PIPER-NERO-EE-MIX.md)
- [三机 / Atlas 登录与资源安全](docs/infrastructure-access-20261126.md)
- [三端源码与 RGB-D 工程状态](docs/m6-context-20261126.md)
- [Piper 混训原始工程归档](experiments/piper-nero-ee-mix/README.md)

## 仓库结构

| 路径 | 内容 |
|---|---|
| `code/xr1-overlay/` | Atlas XR-1 训练配置、脚本与项目改动；不是完整上游 |
| `code/inference-overlay/` | 113 推理服务补丁、包装器和辅助工具 |
| `code/collector/` | 102 Client/Web/采集辅助代码 |
| `code/plugins/` | NERO follower/teleoperator 与 ROS 相机插件 |
| `code/snapshots/` | 按机器/来源保留的历史源码快照，不覆盖 |
| `experiments/piper-nero-ee-mix/` | Piper 混训伴随工程和 2026-11-26 Atlas 导出 |
| `docs/` | 操作、实验、研究路线、证据边界和 AI 交接 |
| `patches/` | 对明确上游版本的增量补丁 |
| `env/` | 历史环境文件；不能视为当前三台机器的完整锁定环境 |

## 机器角色

- **102：** 真机、ROS、RGB/RGB-D 采集、Client/Web。
- **113：** RTX 4090 推理服务；历史 M5 服务端口 6713。
- **Atlas `.15` / `.33`：** 不同主机；训练和数据工作目录主要记录在 `/mnt/panhe/xr1`。资源状态必须实时检查。

登录路径和端口隧道见 [基础设施访问手册](docs/infrastructure-access-20261126.md)。仓库不保存密码、私钥或 Token。

## 重要安全边界

- 不要将本仓库直接覆盖到正在运行的 Atlas、113 或 102。
- 不要擅自终止其他人的进程、容器、训练、推理或 tmux 会话。
- 不要在未确认资源归属时启动训练。
- 不要替换 M5 6713 或其他已部署服务来测试 M6/Piper 模型。
- 不要绕过 NERO IK、Client、Follower 的关节限位、动作限幅、控制锁或急停保护。
- 不要把 Step Flag 当成任务成功信号。
- 不要上传数据集、模型权重、视频、凭据或完整 Python 环境。
- 公开仓库只保留无凭据的操作命令；实验室若要求隐藏内网拓扑，应将访问文档迁移到私有资料库。

## 历史记录

M0–M5 的模型差异、训练布局、部署细节和实验协议保留在 `docs/model-evolution-and-mechanisms.md`、`docs/model-matrix.md`、`docs/deployment-20261116.md`、`docs/experiment-protocol-20261116.md` 等历史文件中。旧文件中的日期和状态不要自动更新成“当前”。
