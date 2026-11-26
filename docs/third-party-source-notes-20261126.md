# 第三方源码来源说明（2026-11-26）

本仓库包含 NERO 本地适配代码，也可能引用或保存少量上游源码快照。请区分本地改动与第三方代码：

- Atlas 收集包中的 `setup.py` 声明项目名 `Xiaomi-Robotics-1`，来源链接为 `https://github.com/XiaomiRobotics/Xiaomi-Robotics-1`，并声明 Apache License 2.0；相关文件头包含 `Copyright (C) 2026 Xiaomi Corporation.`。
- `mibot/models/VLM/qwen3vl.py` 保留 Qwen / Hugging Face 的 Apache 2.0 版权与许可声明。
- 本仓库自有 NERO 适配、实验配置和工具不应覆盖或删除上游版权头。
- 本仓库不声称包含完整的 Xiaomi-Robotics-1 上游仓库，也不声称本项目所有内容均自动适用同一许可证。
- 在继续公开分发第三方源码前，应以各文件现存版权头、上游 LICENSE 和具体文件来源为准；不要删除 LICENSE/NOTICE，不要把第三方贡献改写为本项目原创。
