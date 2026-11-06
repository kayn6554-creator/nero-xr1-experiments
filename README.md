# NERO × XR-1 实验仓库

在 NERO 单臂（7 关节 + 夹爪）上微调 Xiaomi-Robotics-1（XR-1），排查真机失败原因并尝试提升成功率。三个任务：称重、测水温、测风速，每个任务 4 步。

## 仓库里有什么

| 目录 | 内容 |
|---|---|
| `docs/` | 服务器环境说明、基线训练记录、实验方案摘要 |
| `env/` | 复现训练环境需要的文件：`pyproject.toml`、`uv.lock`、包清单、建环境脚本 |
| `patches/` | 我们对学长代码的改动，以补丁形式保存（相对上游基线） |
| `scripts/` | 我们自己写的脚本（数据处理、评测、面板等） |
| `notes/` | 每次实验的记录 |

**不放进仓库的**：上游代码本体、数据集、checkpoint、训练日志、任何密码。

## 上游代码

- 仓库：XPolicyLab（实验室内部），路径 `policy/Xiaomi_Robotics_1/xiaomi_robotics_1/xr1`
- 基线版本：commit `76613fe`（fix(xr1): serve native Nero eight-dimensional single-arm checkpoints）
- 服务器上的工作副本：`/mnt/panhe/xr1`，本地 git 基线提交 `54915a9`（未修改的原样副本）

拿到上游代码后，按 `patches/` 里的顺序 `git apply` 即可得到我们的版本。

## 进度

- [x] 2026-11-06 前期工作：核实基线训练配置，建好与学长完全一致的 uv 环境（包清单逐行一致，8 个 NPU 可用）
- [ ] M1：按帧加入步骤文本
- [ ] M2：夹爪绝对目标 + 裁短停留段 + 失败数据前缀
- [ ] 步骤分类器 S
- [ ] 测试面板与评测脚本
