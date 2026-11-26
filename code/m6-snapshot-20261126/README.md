# M6 三端代码快照（2026-11-26）

本目录对应 2026-11-26 从 102、113、Atlas 33 收集的代码和配置。详细主机信息、登录、深度数据结构、M5 checkpoint、资源状态和 RGB-D 接入缺口见：

- [`docs/infrastructure-access-20261126.md`](../../docs/infrastructure-access-20261126.md)
- [`docs/m6-context-20261126.md`](../../docs/m6-context-20261126.md)
- [`docs/third-party-source-notes-20261126.md`](../../docs/third-party-source-notes-20261126.md)

原始采集包未提交：其中包含重复源码、运行时元数据、PID、缓存和临时文件。代码文件以仓库 `code/collector`、`code/inference-overlay`、`code/xr1-overlay` 的可读源码为主；来源与哈希记录在 `docs/m6-source-manifest-20261126.json`。

重要边界：

- 不包含数据集、视频、模型权重、完整 Python 环境或密码。
- Atlas 33 是 `.15` Atlas 主机之外的独立节点快照。
- RGB-D 目前完成采集/存储侧基础，不代表已经进入 M5/M6 模型训练或推理。
- `nero_ee_sb` 是 Piper→NERO 混合训练的独立动作布局实验线，不等同于 M5 的 `nero_joint_sb`。
