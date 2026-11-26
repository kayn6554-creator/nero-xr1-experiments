# NERO × XR-1 当前 AI 交接状态

**状态截止：2026-11-26**  
**本文件用途：** 新 AI 对话的项目状态单一入口。历史细节仍以各专项文档和源文件为准。

## 1. 一句话总览

项目已完成 NERO 上 XR-1 的多轮动作/步骤训练与 M5 Step Flag 部署；当前并行推进两条新研究线：**M6 在 NERO 上增加 RGB-D 感知，Piper + NERO 则用统一 EE-SB 表示混合训练并在 NERO 上验收。** 两条线目标、数据、动作布局和模型初始化不同，必须独立管理。

## 2. 既有主线：M0–M5

- 基座：Xiaomi-Robotics-1-5B / Qwen3-VL-4B-Instruct 下游适配，不是从零预训练。
- 机器人：NERO 七关节单臂 + 夹爪。
- M0–M3：围绕任务训练、步骤组织和失败前缀数据进行迭代；M3 出现过跨步骤动作/串步问题。
- EF/EF4：独立探索末端动作表示和 IK，不应把它们描述成 M5 的直接上游或同一动作空间消融。
- M5：从 M3 的 joint-space 路线增加 Step Flag 辅助监督。模型训练时保留自然跨步骤动作，额外预测步骤边界；推理端检测边界后截断动作块并保持，Client 等待人工确认后继续。
- M5 Step Flag 不是任务成功检测器，也不能证明当前子目标已经完成。边界预测与物理完成验证是两件事。
- 线上 M5 历史服务：113 `172.16.11.113:6713`，action horizon 30，机器人动作 `[30,8]`，`nero_joint_sb`；端口/GPU/PID 必须实时核对，不把历史快照当成当前在线保证。
- M5 checkpoint 记录路径：
  `/mnt/panhe/output/m5-stepflag/project_xr1-910c-joint-full/m5-stepflag/epoch3-step4026.ckpt/checkpoint/mp_rank_00_model_states.pt`
  这是计划 M6 warm-start 的候选模型权重；使用前需核对文件、配置、SHA256 和严格加载结果。

历史实验计数、证据边界和具体定义见：
- `docs/model-evolution-and-mechanisms.md`
- `docs/model-matrix.md`
- `docs/deployment-20261116.md`
- `docs/experiment-protocol-20261116.md`

## 3. 实验线 A：M6 RGB-D

**当前阶段：研究/工程设计与数据接入准备；未完成训练。**

已确认：
- 102 ROS 相机插件可读取对齐 RGB-D；depth 为 uint16 毫米，0 为无效值。
- ROS RGB-D 配对默认容差 3 ms，缓存 45 帧。
- `nero_depth.py` 已有 FFV1 / `gray16le` 深度视频保存和读取，按 front/wrist、episode 保存。
- 113 当前 M5 服务仍只使用 front/wrist RGB；没有 depth request/decode 分支。
- Atlas 当前 Dataset/Collate 没有完整 depth episode 读取和模型融合链路。

M6 的设计约束：
- 保持 M5 `nero_joint_sb`、30-step horizon、Step Flag 语义和 NERO client 输出接口。
- 仅改变观察输入：从 RGB 扩展为 RGB-D。
- 计划从 M5 Epoch-3 模型参数 warm-start；新深度编码模块单独初始化。
- M5 线上服务与模型目录不覆盖。
- Piper 数据不进入 M6 第一阶段。

完整问题、架构候选、数据契约、里程碑与验收见 `docs/research-lines/M6-RGBD.md`。

## 4. 实验线 B：Piper + NERO EE-SB 混训

**当前阶段：训练前置工作基本完成；正式训练尚未启动。**

### 4.1 目标和表示

目标是用 Piper V2 左臂数据增强 NERO 策略，最终模型部署到真实 NERO，不是训练一个 Piper/NERO 通用策略。

训练使用 `nero_ee_sb`：
- State：60 维容器，10 个有效维度（EE flange position 3 + Rotation 6D 6 + gripper 1）。
- Action：60 维容器，9 个有效维度（局部 flange 平移 3 + 旋转向量 3 + 夹爪 1 + NERO j3/elbow 1 + Step Flag 1）。
- NERO 样本监督 slot 7 elbow 与 slot 8 Step Flag。
- Piper 样本对 slot 7、8 做 mask；不伪造 NERO elbow/Step Flag 标签。
- 当前配置 `gripper_absolute=true`，Piper 夹爪值按 ×0.1 转成 NERO 米制尺度。
- 当前混训不使用 Step Prompt；实际 Atlas launcher 显式传入 `++data.params.train_datasets.step_prompt=false`。
- 本线模型配置从基础 XR1 `model_states.pt` 初始化，不是从 M5 checkpoint 初始化。

### 4.2 已完成且有实测记录的工作（用户提供的 2026-11-26 进展）

**Atlas 环境**
- 独立环境：`/mnt/panhe/xr1/.venv-piper-ef`，Python 3.11.18 ARM64；不复用旧 `.venv`。
- CANN 9.0.0 保持系统原样。
- 离线清理 wheel 目录 `/mnt/panhe/wheels/pkgs_clean`。
- `uv sync --frozen --offline --no-dev --no-install-project --find-links /mnt/panhe/wheels/pkgs_clean` 完成 94 个依赖包安装。
- 将 NumPy 固定回 2.2.6；以 `opencv-python-headless 4.12.0.88` 覆写 `cv2`，避免 `libGL.so.1` 误链接。
- 核心导入通过：torch 2.9.0+cpu、torch_npu 2.9.0.post2、DeepSpeed 0.18.9、Lightning 2.5.3、Transformers 4.57.1、mmengine、Hydra、cv2 4.12.0、imageio、psutil、wandb；NPU device_count=8。
- 精确 freeze 清单在 `experiments/piper-nero-ee-mix/atlas-export-20261126/atlas_new/venv_piper_ef_freeze.txt`。

**数据 / 样本**
- 混合索引：`/mnt/panhe/dataset/xr1_ee_sb_mix_20261125/`，包含 `nero_r1`、`nero_r2`、`piper_v2`。
- 2×NERO + 1×Piper V2；共扫描 3,611 个 JSON。
- Raw frames：576,353；Dwell Filter 移除 89,853；每 Epoch 有效 anchor：486,500。
- 8 个逻辑 NPU × per-device batch 10 × accumulation 1：每 Epoch 6,082 steps；3 Epoch 总计 18,246 steps。
- NERO `episode_000041.json` 实例化：action `(30,60)`、state `(1,60)`、两路图像 `(448,320)`，mask slot 7/8 为 true。
- Piper `episode_0001624.json` 实例化：action `(30,60)`、state `(1,60)`、AV1/MP4 图像解码成功，两路图像 `(448,320)`，mask slot 7/8 为 false。
- 这些是用户报告的真实 Atlas 样本检查结果；新 AI 不应声称自己重新访问了数据集。

**训练启动器**
- 唯一新增 Atlas 启动脚本：`/mnt/panhe/train_nero_piper_ee_sb.sh`。
- 对应原始导出在 `experiments/piper-nero-ee-mix/atlas-export-20261126/atlas_new/train_nero_piper_ee_sb.sh`。
- SHA256：`265da9feb4f9204952a0076ba5d14b33d66933cb89bf01bcdc8195fdcc104d81`。
- 默认 check-only；真实启动需 `XR1_CHECK_ONLY=0` 且 `XR1_ALLOW_TRAINING=1`。
- Smoke 使用 `XR1_MAX_STEPS_OVERRIDE`；正式模式默认 3 Epoch、18,246 steps、`XR1_SAVE_EPOCHS=3`。
- 显式关闭 Step Prompt：`++data.params.train_datasets.step_prompt=false`。
- 该脚本固定默认值对应本次数据、8 NPU、batch 10、accumulation 1；更改数据或并行拓扑后必须重新计算 steps。
- 伴随工程中的动态规划版 launcher 是另一份脚本，不能混称为 Atlas 实际导出版本。具体差异见专项文档。

**当前阻塞**
- 两次 3-step Smoke Test 均 OOM。
- 用户通过进程链和 `/run/docker-minimax.sock` 排查，报告 `.15` 与 `.33` 的 16 张卡正被同事的双机 `minimax-m3.1-flash` vLLM 容器占用：容器 ID `cb0d5733d75f`，TP=8、PP=2、端口 11223。
- 这不是已经证明的训练代码错误，也不能仅凭 OOM 归因于模型显存需求。
- 不能擅自停止该容器或其他共享服务。等待资源释放并经授权后再重跑 Smoke。
- 尚无正式训练、Epoch 3 混训 checkpoint 或 NERO 部署结果。

专项文档与完整源代码：
- `docs/research-lines/PIPER-NERO-EE-MIX.md`
- `experiments/piper-nero-ee-mix/README.md`
- `experiments/piper-nero-ee-mix/source-project-v0.2.0-preview/`
- `experiments/piper-nero-ee-mix/atlas-export-20261126/`

## 5. Atlas 访问与资源注意事项

- `.15`：Atlas 主训练路径已记录为 `/mnt/panhe/xr1`。
- `.33`：另一台 Atlas 节点，2026-11-26 代码快照来自该节点；曾作为双机 vLLM 服务的第二节点。
- 已有 VPN/跳板机/隧道步骤见 `docs/infrastructure-access-20261126.md`。文档不保存密码或私钥。
- 2026-11-26 的 NPU 占用是历史用户报告，不是当前实时状态。开训前重新执行只读资源检查，并确认任务归属。

## 6. 新 AI 可以立即做什么

不需要用户重复项目背景即可：
1. 审查 Piper 项目的启动器、配置差异、Dataset/Mask、Checkpoint 逻辑和部署 adapter。
2. 在本地重跑伴随工程单元测试（本次审阅环境：Python 3.13.5 / CPU；10 tests passed），并做 Python compile、YAML parse、`bash -n`。
3. 补齐 M6 数据读取/模型融合设计和接口测试方案。
4. 生成训练前检查清单、记录模板和 NERO 评测脚本设计。
5. 通过 GitHub 维护文档和代码。

不能仅凭仓库完成：
- 实际访问 Atlas 当前资源状态；
- 读取未上传的 102/113/Atlas 文件、真实数据集或模型权重；
- 运行 NPU Smoke/正式训练；
- 确认 113 当前服务代码与进程；
- 真机 Live。

## 7. 本轮审阅的本地验证边界

本次对用户上传的两个归档执行：
- Piper companion project：10 个 pytest 通过。
- 两个归档内 46 个 Python 文件 compile 检查通过。
- 12 个 YAML 文件解析通过。
- 7 个 shell 文件 `bash -n` 通过。
- 未发现常见密钥/私钥模式。
- 这些均为当前容器内的静态/单元测试；不代表 Atlas、113 或 NERO 真机验证。