# Piper + NERO EE-SB 混训工程归档

这里保留两份本轮用户提供的原始工程材料。它们的来源角色不同，不能相互覆盖：

- **第一个上传包 / 当前 Atlas 状态导出：** `atlas-export-20261126/`，来自 `nero_piper_ee_atlas_export_20261126.tar.gz`，30 个文件。包含今天上午实际 Atlas 的训练启动器、XR1 源码/配置快照和 `.venv-piper-ef` freeze。`atlas_new/train_nero_piper_ee_sb.sh` 是 `/mnt/panhe/train_nero_piper_ee_sb.sh` 的本轮导出版本。
- **第二个上传包 / 伴随工程与整理报告：** `source-project-v0.2.0-preview/`，来自 `nero-piper-ee-mix-v0.2.0-preview(1).zip`，77 个文件。包含项目说明、验证脚本、配置、部署 adapter、测试和参考补丁。

**权威关系：** 当前 Atlas 实际运行脚本、训练代码和配置以第一份 Atlas export 为版本追溯依据；伴随工程提供详细解释和工具，但不能把它的 launcher 当成 Atlas 线上脚本。

## 快速定位

| 任务 | 文件 |
|---|---|
| 当前 Atlas 实际启动器 | `atlas-export-20261126/atlas_new/train_nero_piper_ee_sb.sh` |
| Atlas 环境 freeze | `atlas-export-20261126/atlas_new/venv_piper_ef_freeze.txt` |
| Atlas 当前 `tools/train.py` | `atlas-export-20261126/xr1_overlay/tools/train.py` |
| Atlas 当前 JsonDataset | `atlas-export-20261126/xr1_overlay/mibot/data/datasets/json_dataset.py` |
| Atlas EE-SB patch script | `atlas-export-20261126/xr1_overlay/scripts/panhe/apply_ee_sb.py` |
| Atlas 实际混训 YAML | `atlas-export-20261126/xr1_overlay/configs/data/nero3_piper_ee_sb_20261125.yaml` |
| 伴随工程总入口 | `source-project-v0.2.0-preview/README.md` |
| 伴随工程状态（原包版本） | `source-project-v0.2.0-preview/PROJECT_STATUS.md` |
| 详细训练指南 | `source-project-v0.2.0-preview/docs/04-training-guide.md` |
| Dataset / Action contract | `source-project-v0.2.0-preview/docs/02-data-and-action-contract.md` |
| 部署 adapter | `source-project-v0.2.0-preview/src/nero_piper_ee/deployment/` |
| Atlas 只读前检 | `preflight_atlas.sh` |
| 真实 JsonDataset epoch / Smoke index 检查 | `validation/validate_runtime_dataset.py` |
| Atlas 关键源码 SHA256 | `atlas-source-sha256.txt` |
| 本地伴随工程测试 | `run_companion_tests.sh` |

## 使用规则

- 两份原始归档保持原样；不要修改或覆盖它们。
- `source-project-v0.2.0-preview/PROJECT_STATUS.md` 是 2026-11-25 v0.2.0-preview 状态；本轮之后的当前状态以 `docs/research-lines/PIPER-NERO-EE-MIX.md` 和 `docs/AI-HANDOFF-2026-11-26.md` 为准。
- Atlas 导出中的 `scripts/plan_json_epochs.py` 只按 raw JSON frames 计数，不应用 Dwell Filter；不要用它作为当前 18,246-step 计划依据。
- Atlas 导出 YAML 未显式写 `step_prompt: false`，但实际 Atlas launcher 显式传 `++data.params.train_datasets.step_prompt=false`。伴随工程 YAML 显式写 false。
- 伴随工程 launcher 的 `sync_config()` 在 check-only 判断之前执行，可能创建目录并复制缺失配置；严格只读检查使用新增 `preflight_atlas.sh`。
- 本归档不包含数据集、模型权重、完整 `.venv` 或 Smoke 原始日志。前检脚本尚未在 Atlas 实测。

## 运行前

在 Atlas 上将仓库/伴随工程传到可访问路径后，从仓库根目录运行：

```bash
bash experiments/piper-nero-ee-mix/preflight_atlas.sh
```

确认资源归属和授权后，才按专项文档执行 8-NPU Smoke 和正式训练。

## 本地单元测试

从仓库根目录运行：

```bash
bash experiments/piper-nero-ee-mix/run_companion_tests.sh
```

该 wrapper 不修改伴随工程原始归档，只设置测试导入路径。
