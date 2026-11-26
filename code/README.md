# 代码恢复与快照使用

本目录是自有改动集合，不是完整可独立运行的 XR-1 仓库。恢复前必须拥有匹配的实验室上游，并在独立副本中操作。不要将整理后的仓库直接覆盖三台正在工作的机器。

## 当前源码来源

- `xr1-overlay/`：Atlas 上传包里的自有脚本和配置；其中 `train_nero.sh` 已含 M2 的固定步数修改。
- `inference-overlay/`：113 的实际补丁/启动包装器，包含 EF 接入、逐块日志和 IK 提速。
- `collector/`：采集机家目录当前脚本；初版仍使用家目录和机器路径，安装前按手册核对。
- `plugins/`：三个可编辑安装插件。旧备份和 egg-info 不属于源代码安装材料。

## 从原始 XR-1 基线恢复的顺序

以下是基于脚本依赖分析的恢复方案，尚未在完整匹配上游和数据上重跑。先在新副本备份/记录 git diff，禁止在已打补丁目录盲目重复执行。

1. 复制 `xr1-overlay/` 到独立的原始 XR-1 副本。
2. 运行 `apply_m1.py`：它修改数据提示、训练路径和服务回包。
3. **旧 apply_m2.py 与当前 train_nero.sh 有已知冲突**：后者已经包含 M2 的新文本，前者仍要求旧文本恰好出现一次，因此原样运行会停止且不写任何修改。历史脚本原样保留。本次不伪造“全部可一键恢复”：需要在新副本调整 apply_m2.py，使训练脚本已应用的两处被识别为已完成，再应用数据加载器/服务的修改；或使用经过核对的旧训练脚本作为补丁起点。不要对线上脚本回退。
4. M2 后再应用 `apply_ef.py`。注意它要求目标 `configs/model/posttrain_nero_ee_full.yaml` 和 `mibot/utils/nero_kin.py` 不存在；若第 1 步复制了当前快照，两文件已存在，脚本会拒绝。恢复时应先将这两个目标移到副本外备份，运行补丁后与快照核对。
5. 使用当前 `nero_steps.py`，应用 `apply_m4.py`，再应用 `apply_m5sb.py`。这些补丁在已应用时会停止，不是通用幂等安装器。
6. 已生成的 NERO YAML 随仓库提供；不要无条件重跑 make 配置脚本。M3、EF、M5 生成器拒绝覆盖已有文件，且需要真实数据和训练依赖。需重算时应另建输出路径或备份已有配置。
7. 模型配置中的绝对路径、数据目录、处理器和预训练文件必须在目标机器可用；不能只改启动脚本而忽略 YAML。

## 从上传的 Atlas 服务恢复 113 服务

`patches/113-serving-from-atlas-20261113.patch` 是用户提供的 Atlas `serve_openpi_policy.py` 到 113 实际文件的精确差异。只用于这个已含 M1/M2 的 Atlas 服务版本：

```bash
# 在独立 XR-1 副本中，PATCH 指向本仓库补丁绝对路径
PATCH=/path/to/nero-xr1-experiments/patches/113-serving-from-atlas-20261113.patch
git apply --check "$PATCH"
git apply "$PATCH"
```

该补丁已包含 EF 接入、日志和提速效果。不要再重复运行 `apply_ef_serve.py`、`apply_ef_log.py`、`apply_ef_fast.py`。也可在匹配的旧服务副本按这三个脚本的顺序恢复，但不要混用两种方式。

服务依赖 Atlas 的 EF 布局/运动学支持，仍需复制 `serve_fast.py`。该补丁不增加 M5，也不增加 M4/EF4 的文本转换。

所有脚本都应先读说明，再在隔离目录检查；此仓库保留现场快照，明确记录旧脚本的恢复障碍，没有将未经测试的修复混入归档提交。


## 2026-11-16：113 服务与采集端增量补丁

本节更新 11-13 代码说明中的历史限制。11-16 clean patch 已在真实 113 与采集机应用，不是仅停留在补丁草案：

- 113 `mibot/utils/io.py`：增加 `nero_joint_sb` action layout mask。
- 113 `scripts/serve_openpi_policy.py`：支持 M4/EF4 Merge23 prompt 转换；支持 M5 第 9 个训练通道反归一化、连续帧边界判断、边界动作保持、step_flag / step_boundary 回包。
- 采集机 `nero_vla_client.py`：解析 Flag；M5 Live 在边界处截断本次执行并等待 step 文件变化；保存 Flag 与边界诊断。
- 采集机 `nero_vla_web.py`：保留原网页，新增 M4/EF4/M5 预设及 M5 状态展示。
- `nero_vla_presets.json`：包含 M4 6711、EF4 6712、M5 6713，均 exec_steps=30、slow=2.0、step_prompt=true、gripper_mode=absolute。

补丁包 SHA256、各源码 SHA256、apply 顺序和验证结果见 [2026-11-16 部署记录](../docs/deployment-20261116.md) 及 [补丁说明](../patches/README.md)。

**不要再把 11-13 旧快照中的“M5 尚不支持 nero_joint_sb”当作当前线上状态。** 也不要在已经应用本补丁的服务器上重复运行 apply 脚本；脚本会因原始 SHA256 不匹配而拒绝覆盖。恢复/回滚应使用现场备份和记录的版本。

M5 代码已部署不等于真实机械臂边界检测通过；首次 Live 前仍须 Check、Dry、受监督 Pilot。


## 2026-11-16：模型机理与网页手册

- [模型演进与工作机理](../docs/model-evolution-and-mechanisms.md)：M0/M1/M2/M2b/M2A/M3/EF/M4/EF4/M5 的训练改动和推理机制。
- [VLA 采集与测试台使用手册](../docs/vla-web-usage-manual.md)：恢复现场、网页参数、Check/Dry/Live、M5 人工确认。

当前 `code/collector/nero_vla_web.py` 与 `nero_vla_client.py` 仍是仓库 11-13 源码快照；11-16 clean patch 的完整最终文件没有在仓库内，因此手册已明确版本同步限制。不要将旧源码快照误当成线上最终版本。


## 2026-11-26 Piper EE-SB working overlay

为使新 AI 不必在原始归档中手动寻找缺失依赖，以下 Atlas 导出文件已补入 code/xr1-overlay/：
- mibot/utils/io.py
- mibot/utils/ef_align.py
- mibot/utils/piper_kin.py
- mibot/server/runtime/server.py
- scripts/panhe/apply_ee_sb.py
- scripts/panhe/make_ee_sb_config.py
- scripts/panhe/train_nero_piper_ee_sb_atlas_export_20261126.sh

该 launcher 是 /mnt/panhe/train_nero_piper_ee_sb.sh 的 2026-11-26 导出版本。它与 companion project 的动态规划版 launcher 不同。完整原始目录、hash 和版本差异见 experiments/piper-nero-ee-mix/。
