# NERO × XR-1 实验仓库

在 NERO 单臂（7 关节 + 夹爪）上微调 Xiaomi-Robotics-1。此仓库保存实验改动、配置、采集/测试工具和操作记录，不包含实验室上游代码本体、数据集或模型权重。

**当前快照：2026-11-13。M5 训练端为 `m5-stepflag`；M5 推理截断、回包字段和网页展示尚未实现。** 本次是代码归档，不是服务器部署，也没有重新训练或真机验证。

## 阅读入口

- [完整操作与交接手册](docs/operations-handbook.md)：三机分工、代码收集、环境、数据、训练、权重传输、推理与网页、验证及排错。
- [模型与实验矩阵](docs/model-matrix.md)：区分历史方案和实际实现。
- [代码恢复说明](code/README.md)：overlay、补丁的适用前提及不能重复执行的步骤。
- [来源与待补材料](docs/provenance-and-gaps.md)：依赖版本、证据边界、未收集配置。
- [源文件清单与校验值](docs/source-manifest.json)。

## 目录

| 路径 | 内容 |
|---|---|
| `code/xr1-overlay/` | Atlas 自有训练脚本、步骤/运动学工具和 NERO 配置 |
| `code/inference-overlay/` | 113 的 EF 服务、日志、提速补丁及启动包装器 |
| `code/collector/` | 采集与 VLA 测试网页、客户端、报告与辅助脚本 |
| `code/plugins/` | NERO follower、leader 和 ROS 相机插件 |
| `patches/` | 针对明确版本的差异补丁 |
| `docs/`、`notes/`、`analysis/` | 操作说明、历史实验记录和分析代码 |
| `env/` | 早期 Atlas/推理环境快照，非三台机器最新依赖锁定保证 |

## 当前状态

- M0～M3、EF：已有训练/评测记录；M4、EF4、M5：训练代码与输出目录已核对，最终日志及 checkpoint 完整性待单独收集。
- 8 个实际训练任务：称重、水温、风速、噪声、红外容器、TDS、pH、土壤温度。步骤表中的 soil、irobj 是预留定义，不表示已有训练数据。
- Atlas 和 113 服务代码不同；不能把 Atlas 服务文件直接覆盖推理机。
- 集成网页依赖 ROS 驱动与本地插件，不是仅启动一个 Python 文件即可独立复现的应用。

仓库保持原先的上游约定：XPolicyLab 内部基线 `76613fe`，Atlas 原始副本基线 `54915a9`。获取上游需使用实验室授权渠道。历史文档保留原日期；当前状态以本页及模型矩阵为准。
