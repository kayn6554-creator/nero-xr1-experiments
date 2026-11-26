# XR1 补丁说明

本目录保存 `nero_ee_sb` 对 XR1 的修改来源。

## 内容
- `apply_ee_sb.py`：基于已知 XR1 源码文本锚点，修改 `mibot/utils/io.py` 与 `mibot/data/datasets/json_dataset.py`，并派生模型配置。
- `nero_ee_sb.patch`：由交接包中的 `.before_ee_sb` 备份和已修改文件生成的 unified diff。
- `overlay/mibot/utils/ef_align.py`：Piper/NERO 法兰对齐、相对 EE action、统一 EE state。
- `overlay/mibot/utils/piper_kin.py`：Piper V2 左臂 FK。

## 应用原则
1. 先备份 XR1 目标文件。
2. 确认目标源码版本与本补丁锚点匹配。
3. `apply_ee_sb.py` 与 unified diff 二选一，不要重复应用。
4. 应用后检查 `git diff`（若仓库有 Git）或对照备份文件。
5. 执行数据集读取测试。

`apply_ee_sb.py` 使用严格文本匹配，若源码版本不同会断言失败；不要通过删除断言强行应用。