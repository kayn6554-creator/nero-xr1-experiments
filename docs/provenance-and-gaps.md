# 来源、依赖与待补材料

## 已接收快照

| 上传包 | 来源 |
|---|---|
| `nero-xr1-experiments-main.zip` | 原仓库 11-06 快照 |
| `xr1-code-20261113.tar.gz` | Atlas `/mnt/panhe/xr1` |
| `xr1-inference-20261113.tar.gz` | 113 的 `scripts/`、`mibot/`、`configs/` |
| `nero-vla-client-20261113.tar.gz` | 网页、客户端、任务表和服务预设 |
| `nero-vla-extras-20261113.tar.gz` | 报告服务、场景、姿态/回放工具 |
| `nero-collection-20261113.tar.gz` | 录制、报告库、修正/验证、环境/定位脚本 |
| `nero-plugins-20261113.tar.gz` | 三个插件及深度/相机检查/报告入口 |

源码复制的 SHA256 见 source-manifest.json。排除 `.venv`、`.git`、缓存、编译产物、权重、数据、egg-info 和旧备份；当前选定源码原样保存，不以“整理”为名改变硬件行为。

## 上游依赖

- Evo-RL：https://github.com/MINT-SJTU/Evo-RL.git，提交 `ec9842d8a8dd0371b4b50021547552e4608ec5ee`。采集机路径 `/data/panhe/Evo-RL`。Git 显示仅 `src/lerobot/__init__.py` 被清空。修改动机尚未确认，不能直接恢复，也不能把清空当作所有环境都必需的安装步骤。
- pyAgxArm：https://github.com/agilexrobotics/pyAgxArm.git，提交 `cc498c00af0bcb9e297943e94f4792c0e3ee5b2c`。路径 `/data/kaiz/projects/NERO-arm/pyAgxArm`；用户提供的 `git status --short` 为空。
- 三个本地插件无独立 Git 仓库，源码及 pyproject.toml 存于 `code/plugins/`。
- ROS Humble；工作空间 `/data/kaiz/projects/NERO-arm/agx_arm_ws`，驱动源码/构建版本及启动文件尚未收集。
- XR-1 使用实验室内部 XPolicyLab 基线。仓库不重新发布完整内部上游树；原有版权声明不移除，不擅自给第三方代码重新授权。

## 已知缺口

1. 各次训练输出中的最终 config.py、完整日志、checkpoint 文件列表与权重哈希未收集，不能仅从目录名证明训练成功。
2. 113 实际进程启动命令、所用 checkpoint 和 GPU 映射需要现场核对；预设端口是配置快照，不保证当前服务存活。
3. 集成采集流程中的定位网页（8765）、`ik_click` 配置及 ROS 驱动属于外部依赖，未完整归档；完整重建机器前还需补齐。
4. `env_evo.sh` 加载时提示缺少 `/home/kaiz/data/projects/NERO-arm/agx_arm_ws/install/local_setup.bash`。模块定位成功不代表 ROS 加载完整；需管理员/工作空间维护者检查旧构建前缀，不能直接删改共享安装。
5. 当前客户端不会读取、保存或显示 M5 边界回包。113 当前服务不接受 `nero_joint_sb`，也没有 M4/EF4 的步骤合并服务补丁。
6. Atlas 队列含停启常驻 QwenKI 任务的环境特定逻辑。只能用于已约定的任务；不能把按名称停止进程的脚本直接推广到其他共享服务器。

本次发布验证仅涵盖静态语法、配置解析、复制哈希和补丁适用性，不包括 NPU/CUDA 训练、权重加载或机械臂实测。


## 2026-11-16 增量补丁与现场部署来源

新增来源为用户在 2026-11-16 提供并在服务器应用的干净补丁包：

- 文件名：`nero_m4_ef4_m5_inference_patch_20261116_clean.zip`
- SHA256：`5fc4a4b2866e9f233f5b438c1998c994fa8b91a32e70c48a04a519b876d2eb67`
- 113 原文件 SHA256 校验后应用；备份后缀为 `.before_m5_20261116_134821`。
- 采集机原文件 SHA256 校验后应用；网页在确认后单独重启。
- clean patch 中主要文件 SHA256：
  - collector/nero_vla_client.py: `fbde6d04072013dcca24a2ba7bd2e2b8365e2ccc1d6236ec0f3eb21558323d67`
  - collector/nero_vla_web.py: `0afd1eff6a72be939138c8225e497521548fde52812387c3b81f7d2034dfc117`
  - collector/nero_vla_presets.json: `0f7a82e95b5f06a6d0dcd0cb45fe8949ac6a65c1bca2e9c46a51364f74ba6dc1`
  - 113/scripts/serve_openpi_policy.py: `e5128dfcc22bc94651fd8993c9089be3c82c2ea67efef058943c6ee2e34e8d54`
  - 113/mibot/utils/io.py: `b2a70d5b0eacf480fbaf035cfe365291cbc94738c6eadae93d7b5a3b424d8c94`

本日现场部署结果、metadata、端口/GPU、EF 基线 OOM 和验证边界见 [deployment-20261116.md](deployment-20261116.md)。

此前本文件第 5 项“当前客户端/服务不支持 M5”的旧缺口已经关闭：M5 Step Flag 服务、客户端记录、网页展示及人工步骤文件门控均已部署。剩余缺口是远端专项测试环境依赖、全模型图像 Dry、M5 Live 物理闭环和 EF 原始基线服务恢复。

## 2026-11-16 实验材料

用户确认第一轮任务为 temp、ph、ircup，temp 水位沿用训练采集条件；场景定义沿用采集机 nero_tasks.json / nero_scenes.json。详细任务、成功判据、控制变量、Pilot、正式对照和数据结构见 [experiment-protocol-20261116.md](experiment-protocol-20261116.md)。


## 2026-11-16 后续现场反馈

- 主臂 CAN：can1 序列号映射正确，接口 UP，1 Mbps，ERROR-ACTIVE；10 秒 candump 无帧，RX packets 无增长。用户重新插拔 USB-CAN 后恢复。根因未确认。
- M5：用户报告约 1–2 次初步测试中，每步结束后稳定暂停，按下一步按钮后继续。尚未取得对应 run_id、原始 Flag 序列、边界索引、视频或评测数据。
- 文档：新增 `docs/model-evolution-and-mechanisms.md` 与 `docs/vla-web-usage-manual.md`。

## 当前仍需补齐

1. 从采集机重新收集 2026-11-16 最终部署版 `nero_vla_web.py`、`nero_vla_client.py`，替换或并列归档 11-13 历史源码快照；核对文件哈希。
2. 取回 M5 Pilot 的 run_id、summary.json、chunks.npz、live_status.json 和外接视频，标注真实步骤边界。
3. 对四模型完成完整图像请求 Dry；M5 Live 多次验证早触发、迟触发、漏检和人工确认恢复。
4. EF 原始基线 6710 仍未恢复，EF/EF4 严格对照仍被阻塞。
