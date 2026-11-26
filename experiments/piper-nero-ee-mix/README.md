# Piper + NERO EE-SB 混训工程归档

这里同时保留两份用户提供的原始工程材料，方便 AI/开发者追溯版本：

- `source-project-v0.2.0-preview/`：第一个 ZIP 的完整解压文件，77 个文件，保留原目录结构。
- `atlas-export-20261126/`：第二个 TAR.GZ 的完整解压文件，30 个文件，保留原目录结构；其中 `atlas_new/train_nero_piper_ee_sb.sh` 是 `/mnt/panhe/train_nero_piper_ee_sb.sh` 的本轮导出版本。

**不要把两份脚本、配置和规划器混成同一版本。** 具体区别、数据统计、环境和当前阻塞见：
- `docs/research-lines/PIPER-NERO-EE-MIX.md`
- `docs/AI-HANDOFF-2026-11-26.md`

## 快速定位

| 任务 | 文件 |
|---|---|
| companion project 总入口 | `source-project-v0.2.0-preview/README.md` |
| 当前工程状态（原包版本） | `source-project-v0.2.0-preview/PROJECT_STATUS.md` |
| 详细训练指南 | `source-project-v0.2.0-preview/docs/04-training-guide.md` |
| Dataset / Action contract | `source-project-v0.2.0-preview/docs/02-data-and-action-contract.md` |
| Atlas 当前启动器导出 | `atlas-export-20261126/atlas_new/train_nero_piper_ee_sb.sh` |
| Atlas 环境 freeze | `atlas-export-20261126/atlas_new/venv_piper_ef_freeze.txt` |
| Atlas 当前 `tools/train.py` | `atlas-export-20261126/xr1_overlay/tools/train.py` |
| Atlas 当前 JsonDataset | `atlas-export-20261126/xr1_overlay/mibot/data/datasets/json_dataset.py` |
| Atlas EE-SB patch script | `atlas-export-20261126/xr1_overlay/scripts/panhe/apply_ee_sb.py` |
| 混训 YAML（Atlas 导出） | `atlas-export-20261126/xr1_overlay/configs/data/nero3_piper_ee_sb_20261125.yaml` |
| 部署适配器 | `source-project-v0.2.0-preview/src/nero_piper_ee/deployment/` |

## 原始材料说明

- 两份材料来自用户上传，不代表仓库作者在当前容器中访问过 Atlas。
- `source-project-v0.2.0-preview` 的 `PROJECT_STATUS.md` 是 2026-11-25 v0.2.0-preview 状态；它在本次用户补充后已过时。当前状态以 `docs/research-lines/PIPER-NERO-EE-MIX.md` 和 AI handoff 为准。
- `atlas-export-20261126/xr1_overlay/scripts/plan_json_epochs.py` 只按 raw JSON frame 计数，未应用 Dwell Filter；不要拿它作为当前 18,246-step 计划的依据。
- Atlas 导出 YAML 未显式写 `step_prompt: false`，但实际 Atlas launcher 使用 Hydra override `++data.params.train_datasets.step_prompt=false`。伴随工程 YAML 显式写了 false。
- 本归档不包含数据集、模型权重、完整 `.venv` 或 Smoke 日志目录。
