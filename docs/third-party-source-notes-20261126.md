# 第三方源码来源说明（2026-11-26）

本仓库包含 NERO 本地适配代码，也可能引用或保存少量上游源码快照。请区分本地改动与第三方代码：

- Atlas 收集包中的 `setup.py` 声明项目名 `Xiaomi-Robotics-1`，来源链接为 `https://github.com/XiaomiRobotics/Xiaomi-Robotics-1`，并声明 Apache License 2.0；相关文件头包含 `Copyright (C) 2026 Xiaomi Corporation.`。
- `mibot/models/VLM/qwen3vl.py` 保留 Qwen / Hugging Face 的 Apache 2.0 版权与许可声明。
- 本仓库自有 NERO 适配、实验配置和工具不应覆盖或删除上游版权头。
- 本仓库不声称包含完整的 Xiaomi-Robotics-1 上游仓库，也不声称本项目所有内容均自动适用同一许可证。
- 在继续公开分发第三方源码前，应以各文件现存版权头、上游 LICENSE 和具体文件来源为准；不要删除 LICENSE/NOTICE，不要把第三方贡献改写为本项目原创。


## 2026-11-26 Piper-NERO 工程归档补充

experiments/piper-nero-ee-mix/ 中保存了完整 companion project v0.2.0-preview 与 Atlas 2026-11-26 export。它们包含第三方 XR-1 / Qwen 相关参考文件、上游服务快照和本地 NERO 适配代码。新增归档时保留了原文件内容与版权头；不要将 references/upstream、references/baseline-xr1 或 Atlas overlay 中的 Xiaomi/Qwen 代码重新标注为本项目原创。

Atlas export 中的 uv.lock、tools/train.py、mibot/* 和配置是特定日期的源码快照，不代表仓库包含完整上游。公开分发或复制这些文件前，仍应核对各文件 LICENSE/NOTICE 与上游许可证要求。
