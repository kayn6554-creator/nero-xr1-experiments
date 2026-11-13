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
