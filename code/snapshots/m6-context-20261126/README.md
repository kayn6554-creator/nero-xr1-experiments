# M6 三端代码快照（2026-11-26）

本目录按来源保留 102、113、Atlas 33 的去重源码/配置文件，路径结构与采集包内的源码路径对应。四个原始包的哈希与盘点说明见 [docs/m6-source-manifest-20261126.json](../../docs/m6-source-manifest-20261126.json)。

- `102/`：NERO Client/Web、模型预设、深度记录器和活动 ROS RGB-D 插件。
- `113/`：实际部署的 OpenPI 服务、服务入口、推理侧 Dataset / IO / 运动学代码。
- `atlas33/`：XR-1 训练代码、模型与运行时、训练脚本、M2/M3/M5/EE/Piper 数据配置及补充训练配置。

这是一份源码快照，不是完整可运行环境；不包含数据集、模型权重、运行日志、PID、Python 环境或凭据。Atlas 33 是 `10.20.10.15` Atlas 主机之外的独立节点快照。RGB-D 采集/存储已有基础，但模型输入链路仍未接通。第三方源码版权见 [docs/third-party-source-notes-20261126.md](../../docs/third-party-source-notes-20261126.md)。
