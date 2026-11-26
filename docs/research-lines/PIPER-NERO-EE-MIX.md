# 平行实验线：Piper V2 + NERO EE-SB 混合训练，并在 NERO 真机验收

**工程包：** `nero-piper-ee-mix-v0.2.0-preview`  
**Atlas 导出：** `2026-11-26`  
**当前状态：** 训练前置工作已基本完成；真实 NERO/Piper 样本与环境已验证；Smoke Test 因共享 NPU 占用失败；正式训练尚未开始。

## 1. 目标不是训练 Piper，而是让 Piper 帮助 NERO

最终目标：

**利用 Piper 左臂 V2 的真实操作数据增强 NERO 策略，训练完成后把模型部署到真实 NERO 七关节机械臂上，观察是否比 NERO-only 策略更好。**

Piper 是数据来源，不是最终部署目标。不能把本项目描述成 Piper/NERO 双机器人通用策略，也不能把训练 Loss 下降当作迁移成功。

## 2. 为什么采用 EE-SB

Piper 左臂 6 轴、NERO 7 轴，关节数和运动学结构不同，不能把两者关节向量直接拼接成一个共同动作空间。

当前方案将两台机器人的动作映射为统一的末端法兰相对运动语义：
- 位置增量 3 维；
- 旋转向量增量 3 维；
- 夹爪目标 1 维；
- NERO elbow / j3 增量 1 维；
- Step Flag 1 维。

Piper 没有 NERO elbow 和 Step Flag 的对应监督，因此这两维必须按样本屏蔽。不能给 Piper 填假标签，也不能让 Piper 的零 padding 被误认为真实监督。

## 3. 与 M6 的关系：两条平行、正交的实验线

| 维度 | M6 RGB-D | Piper + NERO EE-SB |
|---|---|---|
| 科学问题 | 深度是否提升 NERO 三维空间理解 | 跨机器人数据能否迁移/增强 NERO 策略 |
| 训练数据 | 第一阶段仅 NERO | NERO + Piper V2 |
| 观察输入 | RGB → RGB-D | 当前仍按 RGB 训练 |
| State / Action | 保持 M5 `nero_joint_sb` | 统一 `nero_ee_sb` |
| 初始化 | 计划 M5 Epoch-3 warm-start | 当前配置为 XR1 base pretrained |
| 最终评估 | NERO | NERO |
| 当前进度 | 设计/数据接入准备 | 环境和样本预检完成，等待可用 NPU 后 Smoke |

不要同时把 RGB-D、Piper 数据、EE 动作表示、Step Flag 和新的模型结构全部合在一次训练里，否则即使结果变化，也无法判断是哪一项造成的。先分别验证，未来是否组合由独立实验决定。

## 4. 数据集组成与采样

混合索引根目录：

```text
/mnt/panhe/dataset/xr1_ee_sb_mix_20261125/
├── nero_r1/
├── nero_r2/
└── piper_v2/
```

用户提供的实际统计：
- `nero_r1`：504 episodes；
- `nero_r2`：同一份 NERO 数据的第二个索引入口，504 episodes；
- `piper_v2`：2,603 episodes；
- 合计 3,611 个 JSON。
- NERO 通过两个不同软链接路径重复进入 `_json_files`。当前 JsonDataset 用路径字符串 set 去重，而不是 resolve 软链接目标，因此 `nero_r1` 与 `nero_r2` 会形成两倍 NERO 采样权重。这是有意的数据配比，不是两份独立采集。
- Raw frames：576,353。
- Dwell Filter 删除：89,853。
- 每 Epoch 有效 anchor：486,500。
- raw frame 组成大致接近 NERO:Piper = 1:1；Dwell Filter 后的逐机器人有效样本分布需要在正式训练前再次导出，不能用 raw frame 比例替代。

### 真实样本检查（用户报告）

NERO `episode_000041.json`：
- action `(30, 60)`；
- state `(1, 60)`；
- front/wrist 图像均解码为 `(448, 320)`；
- action mask 的 slot 7（NERO elbow）和 slot 8（Step Flag）均启用。

Piper `episode_0001624.json`：
- action `(30, 60)`；
- state `(1, 60)`；
- AV1/MP4 两路图像均解码为 `(448, 320)`；
- mask slot 7 / 8 均关闭。

这是 Atlas 上真实 JsonDataset 样本检查的用户报告结果；本次本地审阅没有直接读取 Atlas 数据目录。

## 5. `nero_ee_sb` 数据与动作契约

### 5.1 State：60 维容器，10 个有效维度

| Slot | 语义 |
|---|---|
| 0–2 | EE flange position，base-aligned 表达 |
| 3–8 | Rotation 6D |
| 9 | Gripper opening（m） |
| 10–59 | padding，零 |

State 用混合数据的 `q01/q99` 做 quantile normalization。训练与部署必须使用同一组统计量。

### 5.2 Action：60 维容器，9 个有效维度

| Slot | 语义 | NERO | Piper |
|---|---|---|---|
| 0–2 | 当前 flange frame 下局部位置增量（m） | supervise | supervise |
| 3–5 | 当前 flange frame 下相对旋转向量（rad） | supervise | supervise |
| 6 | 夹爪目标（m，当前配置 absolute） | supervise | supervise |
| 7 | NERO j3/elbow 增量（rad） | supervise | **mask** |
| 8 | Step Flag | 有可靠 `source.step_index` 时 supervise | **mask** |
| 9–59 | padding | mask | mask |

Step Flag 当前目标由 NERO `source.step_index` 的未来步骤变化生成。它不是 Step Prompt。混训当前不使用 Step Prompt，Atlas 实际 launcher 显式传 `++data.params.train_datasets.step_prompt=false`。

### 5.3 Piper 与 NERO 坐标 / 夹爪

已记录的实现：
- Piper 通过自身 FK 计算 flange pose，不能把 Piper 关节角当作 NERO 关节角。
- NERO 使用其 MDH FK/IK。
- 当前 `PIPER_TO_NERO_FLANGE_R=I`，`PIPER_BASE_ALIGN=I`，`NERO_BASE_ALIGN=I`。
- 用户报告的工作姿态观察：Piper link6 与 NERO MDH frame 7 的 flange 轴方向关系相似，当前局部增量方向按单位旋转矩阵对齐。
- 这只说明当前实现和已观察工作姿态的约定；不等于完成全工作空间、TCP/gripper offset、base frame 和标定精度验证。
- 当前配置 `gripper_absolute=true`；Piper 夹爪归一化值按 ×0.1 转为 NERO 对应米制尺度。部署前必须再次核实 NERO 控制接口的实际夹爪单位、范围和开合方向。

## 6. 模型、数据与训练配置

当前配置：
- Data：`nero3_piper_ee_sb_20261125`
- Model：`posttrain_nero_ee_sb_full`
- Trainer：`deepspeed_npu_bf16_zero2`
- `action_layout: nero_ee_sb`
- `action_length: 30`
- per-device `batch_size: 10`
- `gripper_absolute: true`
- `dwell_keep.default=10`，step 3 保留 20；`dwell_tail=10`；速度阈值 2 deg/s；夹爪变化阈值 0.001 m。
- model config 当前 `async_train=false`、BF16、SDPA、full fine-tuning（LoRA disabled）。
- 当前模型初始化路径：`/mnt/wzh/checkpoints/Xiaomi-Robotics-1-5B/model_states.pt`。
- Qwen3-VL 路径：`/mnt/wzh/checkpoints/Qwen3-VL-4B-Instruct`。
- 不要将 M5 `/mnt/panhe/output/m5-stepflag/.../epoch3-step4026.ckpt` 擅自替换为本线初始化权重。本线当前 YAML 明确指向 XR1 base pretrained。

## 7. Atlas 环境：已完成的离线环境准备

环境路径：

```text
/mnt/panhe/xr1/.venv-piper-ef
```

用户提供的实测环境事实：
- Python 3.11.18，Linux ARM64；
- 系统 CANN 9.0.0；
- uv 0.13.9；
- 独立于旧 `.venv` 和系统 Python；
- wheel 清理目录 `/mnt/panhe/wheels/pkgs_clean`；
- uv cache `/mnt/panhe/.uv-cache`。

离线安装过程解决的问题：
1. 剔除错误平台包：`psutil-7.2.2-cp313t`、`ninja-musllinux`、`wandb-musllinux`。
2. 使用正确 Python 3.11 / aarch64 wheel；补齐 `hf-xet==1.6.0`、兼容版 psutil/ninja/wandb。
3. 通过 `UV_PROJECT_ENVIRONMENT=/mnt/panhe/xr1/.venv-piper-ef`、`UV_CACHE_DIR=/mnt/panhe/.uv-cache` 和 `uv sync --frozen --offline --no-dev --no-install-project --find-links /mnt/panhe/wheels/pkgs_clean` 完成 94 个依赖包安装。
4. 将 NumPy 修正为锁定版本 2.2.6。
5. `opencv-python` 与 `opencv-python-headless` 并存时，`cv2` 曾误链接 `libGL.so.1`；用 aarch64 的 `opencv-python-headless==4.12.0.88` wheel 以 `--reinstall --no-deps` 覆写，解决导入问题。

核心导入测试通过（用户报告）：
- torch 2.9.0+cpu；
- torch_npu 2.9.0.post2；
- DeepSpeed 0.18.9；
- Lightning 2.5.3；
- Transformers 4.57.1；
- mmengine、Hydra；
- cv2 4.12.0、imageio、psutil、wandb；
- `torch.npu.device_count() == 8`。

注意：`torch` 显示 `+cpu` 不能单独用来判断 NPU 不可用；本环境通过 `torch_npu` 导入和 NPU device_count 检查。

freeze 清单：`atlas-export-20261126/atlas_new/venv_piper_ef_freeze.txt`。完整 requirements / wheel URL 在伴随工程 `source-project-v0.2.0-preview/manifests/dependencies/` 和 Atlas export 中。不要重新下载或在线安装，除非环境缺失并获得明确授权。

## 8. 训练步数：为什么是 18,246，而不是旧脚本的 raw-frame 估计

用户报告已直接实例化真实 `JsonDataset(finite_epoch=True)` 并全量扫描 3,611 个 JSON：
- Raw frames = 576,353；
- Dwell Filter removed = 89,853；
- effective anchors / epoch = 486,500。

按 8 个 NPU 逻辑设备、per-device batch 10、gradient accumulation 1：
- 每 rank 样本约 `ceil(486500 / 8) = 60,813`；
- 每 rank batch 数 `ceil(60,813 / 10) = 6,082`；
- 每 Epoch optimizer steps = 6,082；
- 3 Epoch = 18,246。

旧 `atlas-export-20261126/atlas_new/xr1_overlay/scripts/plan_json_epochs.py` 只累加 JSON 的 `num_frames`，不执行 Dwell Filter，曾产生约 21,615 步的错误估计。**不要使用它作为当前训练步数依据。**

伴随工程中的 `scripts/validation/plan_training_steps.py` 会调用 `mix_dataset_utils.py` 中的 Dwell Filter 计数逻辑；用户报告的真实 JsonDataset 扫描结果优先级更高。正式开训前若数据、world size、batch 或 accumulation 变化，重新扫描并与实际 DataLoader / DistributedSampler 日志核对。

## 9. 两份启动器的区别（非常重要）

### A. 伴随工程动态规划版

路径：

`source-project-v0.2.0-preview/scripts/atlas/train_nero_piper_ee_sb.sh`

特征：
- 默认 `XR1_CHECK_ONLY=1`；
- 真训练还需 `XR1_ALLOW_TRAINING=1`；
- 通过 `scripts/validation/plan_training_steps.py` 动态计算 Dwell Filter 后的步数；
- 校验/同步伴随工程配置到 XR1，配置不一致时拒绝覆盖；
- 适合伴随工程目录的受控启动流程。

### B. Atlas 实际导出脚本

原始位置：`/mnt/panhe/train_nero_piper_ee_sb.sh`  
归档位置：`atlas-export-20261126/atlas_new/train_nero_piper_ee_sb.sh`  
SHA256：`265da9feb4f9204952a0076ba5d14b33d66933cb89bf01bcdc8195fdcc104d81`

特征：
- 从 `/mnt/panhe/xr1` 执行 `scripts/train.sh`；
- 使用 `.venv-piper-ef/bin/python`；
- 显式设置 `++data.params.train_datasets.step_prompt=false`；
- 默认 `XR1_CHECK_ONLY=1`；
- 只有 `XR1_CHECK_ONLY=0` 且 `XR1_ALLOW_TRAINING=1` 才进入训练；
- `XR1_MAX_STEPS_OVERRIDE` 用于固定步数 Smoke；
- 正式模式默认 3 Epoch、`MAX_STEPS=18246`、`XR1_SAVE_EPOCHS=3`；
- 固定步数对应本次 3,611 JSON、486,500 effective anchors、8 NPU、batch10、accum1。数据/资源配置变化后不能继续盲用硬编码 18,246。

两份脚本不是同一版本，不要把一个覆盖另一个。原始包的 SHA256 和目录结构见 `SOURCE-ARCHIVE-MANIFEST.json`。

## 10. 训练前置工作与当前阻塞

### 已完成
- [x] `.venv-piper-ef` 独立环境搭建。
- [x] 94 个依赖离线安装与核心库导入。
- [x] NERO/Piper 数据软链接索引准备。
- [x] `nero_ee_sb` Dataset / IO / FK / IK / mask 代码与配置合入 `/mnt/panhe/xr1`（用户报告）。
- [x] NERO 与 Piper 真实样本实例化、动作/状态/mask 检查。
- [x] 两路 RGB 图像解码。
- [x] Dwell Filter 有效样本数和 3-Epoch 步数计算。
- [x] 隔离训练启动器编写。
- [x] Smoke OOM 的共享资源原因已排查到 vLLM 容器占用（用户报告）。

### 未完成
- [ ] 资源释放后重新确认 `.15` 与 `.33` NPU 分配。
- [ ] 重新运行 3-step Smoke，确认模型加载、NERO/Piper batch、forward/backward、有限 loss。
- [ ] 检查 Smoke checkpoint 是否成功写入并能读取。
- [ ] 正式 3-Epoch 训练。
- [ ] Epoch 3 checkpoint / config / SHA256 核验。
- [ ] NERO-only EF 与 NERO+Piper EE-SB 的固定测试集对照。
- [ ] `nero_ee_sb` WebSocket server 集成。
- [ ] Client Step Flag gate 与 Step Prompt 解耦。
- [ ] 新服务独立端口 Dry / Client Dry / 受监督 Live Pilot。

## 11. 两次 Smoke OOM 的正确解释

用户提供的排查结果：
- 两次 Smoke 日志目录：`/mnt/panhe/output/smoke-nero-piper-ee-sb-v0.1-1126-*`，时间后缀对应约 10:34:28、10:35:26。
- 进程链与 `/run/docker-minimax.sock` 排查发现 `.15` 和 `.33` 的 16 张卡被同事的双机 `minimax-m3.1-flash` vLLM 容器占满。
- 容器 ID：`cb0d5733d75f`；TP=8、PP=2；服务端口 11223。
- 用户报告 tmux `train` 本身空闲，避免误判为旧训练脚本占卡。

解释边界：
- 这两次 OOM 不构成模型质量或训练逻辑失败证据；
- 也不能保证代码一定没有其他问题，资源释放后仍需重新 Smoke；
- 不允许停止该容器、kill 相关进程或改动其他人的服务。需要资源所有者释放或明确授权。

## 12. 下一步执行顺序：拿到卡后可以直接照此推进

### A. 资源未释放期间（现在可以做）

1. 复核仓库源文件与 `/mnt/panhe/xr1` 实际版本差异，尤其 `json_dataset.py`、`io.py`、`ef_align.py`、`piper_kin.py`、`nero_kin.py`、`apply_ee_sb.py`、YAML 和 `tools/train.py`。
2. 在 Atlas 只读检查 `.venv-piper-ef`、CANN、模型权重、Qwen processor 和数据软链接是否仍在位。
3. 用当前数据重新跑 `validate_mix_dataset.py` 和 `validate_batch.py`；保存 JSON manifest。
4. 对照 `486500 / 6082 / 18246`，如不一致，先找出数据变动、Dwell Filter、world size、batch、accumulation、sampler 或 drop_last 的原因。
5. 对比实际 `/mnt/panhe/train_nero_piper_ee_sb.sh` 与归档 SHA256；若变化，收集新版本而不是直接覆盖。
6. 检查 `tools/train.py` 的 Epoch checkpoint callback、`XR1_SAVE_EPOCHS`、`XR1_DISABLE_CHECKPOINTS` 和实际配置。
7. 准备唯一的 Smoke 输出目录和正式输出目录；确认不覆盖旧产物。

### B. 资源可用且获准后

1. 先执行 `npu-smi info`、进程/容器归属检查；确认不是其他同事的服务仍在使用。
2. 用独立 tmux 会话，不复用/覆盖旧 `train`。
3. 运行 3-step Smoke。正式配置预期 8 个 NPU worker；此前单卡 Smoke OOM 发生在共享 NPU 被 vLLM 占用时，不能据此判断代码失败。资源释放并经所有者确认后，优先按正式 8 卡拓扑执行 Smoke。
4. Smoke 必须证明：预训练权重加载、NERO/Piper batch 都实际进入训练、forward/backward、loss 有限、checkpoint 写入。
5. Smoke 通过后记录完整命令、环境、配置 SHA256、数据 manifest、GPU/NPU 分配，再启动 3-Epoch 正式训练。
6. 正式训练输出目录默认 `/mnt/panhe/output/nero-piper-ee-sb-v0.1`；启动前必须确认该目录不存在或明确使用新实验名。
7. 训练结束核验 Epoch 3、实际 global_step、checkpoint 权重、config.py、SHA256、日志和进程退出。

### C. 训练完成后（不要提前部署）

1. 固定 NERO held-out 测试集，与 NERO-only EF 基线做相同条件对照。
2. 记录整体/逐步成功、首个失败阶段、动作平滑、EE 误差、IK 成功率/残差/耗时、Step Flag 边界检测、推理延迟和人工干预。
3. 以训练新 checkpoint 为基础单独做 `nero_ee_sb` 部署服务；当前 113 server 不支持该 layout，不能直接沿用旧 `apply_ef_serve.py`。
4. 新服务使用独立模型目录和端口，绝不覆盖 M5 6713。
5. 先 Unit → WebSocket Dry → Client Dry → 受监督 Live Pilot，保留 NERO Client/Follower 原有安全保护。

## 13. 部署链当前缺口

现有 companion project 已提供：
- `src/nero_piper_ee/deployment/ee_to_nero.py`
- `src/nero_piper_ee/deployment/nero_ee_sb_server_adapter.py`
- metadata contract 示例
- FK/IK、Step Flag、adapter 测试。

但这不等于 113 已可部署：
- 113 当前服务需要从真实机器重新收集运行版本和 SHA256；
- 旧 `apply_ef_serve.py` 只支持 `nero_ee`；
- `nero_ee_sb` 输入需先将 NERO `state[8]` 经 FK 和训练侧 `compose_ee_state()` 转成模型 State；
- 模型输出需反归一化、先做 Step Flag 边界检测，再只对有效前缀做 IK；
- 客户端仍收 `[30,8]` absolute joint，但需要允许 `step_flag.enabled=true` 且 `step_prompt.enabled=false`：边界等待依赖 step file，Step Prompt 是独立开关；
- M5 Client 的当前线上版本不能被仓库旧快照覆盖。

完整接口见 companion project `docs/09-nero-xr1-integration.md`。

## 14. 本地审阅验证结果

本次对用户提供的两个压缩包在隔离容器执行：
- companion project pytest：10 passed；
- 46 个 Python 文件 compile：通过；
- 12 个 YAML parse：通过；
- 7 个 shell `bash -n`：通过；
- 常见 token/private-key 模式扫描：未发现匹配。

本地运行环境为 Python 3.13.5 / CPU，与 Atlas Python 3.11.18 ARM64 / CANN / NPU 不同。上述仅是静态/单元测试，不替代 Atlas、113 或真机验证。

## 15. 2026-11-26 审阅补充：两份压缩包的权威关系

用户本轮说明的上传顺序是：
1. **第一个压缩包** `nero_piper_ee_atlas_export_20261126.tar.gz`：今天上午 Atlas 实际运行状态的导出。它是当前 Atlas launcher、XR1 源码/配置快照和环境 freeze 的优先追溯依据。
2. **第二个压缩包** `nero-piper-ee-mix-v0.2.0-preview(1).zip`：伴随工程代码、文档、验证工具和部署适配器。它是详细工程说明和工具来源，不等同于 Atlas 上实际运行的脚本。

仓库按文件名分别保存两份原包，内容不覆盖。当前 Atlas 的真实配置以 `/mnt/panhe/xr1/configs/data/nero3_piper_ee_sb_20261125.yaml` 为准；伴随工程 YAML 与 Atlas 导出 YAML 不是字节级相同文件，不要为了“统一”而覆盖。

## 16. 新增只读训练前检查工具

新增：
- `experiments/piper-nero-ee-mix/preflight_atlas.sh`：面向 Atlas 的一键只读前检。
- `experiments/piper-nero-ee-mix/validation/validate_runtime_dataset.py`：直接实例化 XR1 真实 `JsonDataset`，复核完整 Epoch 有效样本数/steps，并检查 3-step Smoke 实际截取的样本索引是否同时包含 NERO 与 Piper。
- `experiments/piper-nero-ee-mix/atlas-source-sha256.txt`：记录 Atlas 导出中训练入口、Dataset、IO、FK/IK helper、配置等关键文件 SHA256。

在已将仓库或该目录完整传到 Atlas 后，从仓库根目录运行：

```bash
bash experiments/piper-nero-ee-mix/preflight_atlas.sh
```

该脚本使用 Atlas 实际配置，而不是伴随工程副本；依次检查 Atlas 源码 SHA256、Python/CANN/NPU 导入、混合数据扫描、真实 `JsonDataset(finite_epoch=True)`、Smoke 样本源覆盖及 `DistributedSampler` 各 rank 的 NERO/Piper 分布、NERO/Piper 单样本解码、实际 Atlas launcher SHA256 与 check-only 计划，最后打印 NPU / 进程 / 容器只读快照和输出目录冲突检查。

**边界：** 该脚本在本地只做过 `bash -n` / Python compile 检查，尚未在 Atlas 上运行；它不会启动训练、安装包、复制配置、创建训练输出目录、停止进程或容器。输出中的 NPU 空闲与否仍需由资源所有者/管理员确认。

伴随工程动态 launcher 的 `sync_config()` 在进入 check-only 判断之前执行：可能创建配置目录并复制缺失配置。因此它不是严格只读检查工具。当前 Atlas 现场前检优先使用新增 `preflight_atlas.sh` 和实际导出脚本的 `XR1_CHECK_ONLY=1`。

## 17. 拿到资源后的推荐执行顺序

当前训练配置预期 8 个 NPU worker、per-device batch 10、accumulation 1。此前单卡 Smoke OOM 发生在共享 NPU 被其他 vLLM 服务占用的时段，不能单独据此推断代码错误或单卡模型显存需求。资源释放后，优先按正式训练拓扑的 8 个可用 NPU 做 3-step Smoke，而不是在资源仍被占用时重复试卡。

### 17.1 先确认资源归属

- 先运行 `preflight_atlas.sh`；检查 `npu-smi info`、进程链、tmux、端口 11223 和 Minimax Docker socket 的只读信息。
- `.15` 与 `.33` 曾被同一个双机 vLLM 任务共同占用；只看到目标节点没有训练进程不等于 NPU 空闲。
- 不得停止容器、kill 进程或回收显存。必须等资源所有者释放或获得明确授权。
- 若可见 NPU 数量不是 8，前检会因预期 steps 不匹配而失败；先重新核对训练拓扑并重新计算计划，不要继续使用硬编码 18,246。

### 17.2 3-step Smoke（仅在资源确认可用并获准后）

先建立独立 tmux 会话：

```bash
tmux new -s nero-piper-ee-smoke
```

在会话中运行实际 Atlas 导出脚本：

```bash
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
XR1_CHECK_ONLY=0 \
XR1_ALLOW_TRAINING=1 \
XR1_MAX_STEPS_OVERRIDE=3 \
XR1_SAVE_INTERVAL=1 \
XR1_OUTPUT_ROOT=/mnt/panhe/output/smoke-nero-piper-ee-sb-v0.1-YYYYMMDD-HHMMSS \
bash /mnt/panhe/train_nero_piper_ee_sb.sh
```

Smoke 必须确认：基础模型与 Qwen processor 加载成功；实际训练样本同时包含 NERO/Piper；forward/backward 正常；loss 有限；至少一个 Smoke checkpoint 成功写入并可读取。3-step Smoke 的 Dataset index 预期为 3 × 10 × 8 = 240 个样本；新增 runtime validator 会提前检查这些索引和 DistributedSampler rank 分配中是否两种机器人都有样本，但这仍不替代真正的 forward/backward。

### 17.3 正式 3-Epoch 训练

Smoke 通过、配置与输出目录确认无冲突后，再在独立 tmux 会话运行：

```bash
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
XR1_CHECK_ONLY=0 \
XR1_ALLOW_TRAINING=1 \
XR1_OUTPUT_ROOT=/mnt/panhe/output/nero-piper-ee-sb-v0.1 \
bash /mnt/panhe/train_nero_piper_ee_sb.sh
```

正式模式使用当前 Atlas 配置默认 3 Epoch / 18,246 steps / `XR1_SAVE_EPOCHS=3`。启动前必须确认输出目录不存在；若已存在，选用新实验名和新路径，绝不删除或覆盖旧实验。

## 18. 当前进度的精确表述

- 已完成：离线环境、核心依赖导入、NERO/Piper 真实样本 Dataset 检查、Dwell Filter 统计、训练步数计算、Atlas 实际启动器和混合配置准备。
- 尚未完成：Atlas 新增前检脚本的远端执行、资源释放后的成功 Smoke、正式 3-Epoch 训练、checkpoint 验证、NERO held-out 对照、113 `nero_ee_sb` 服务和 Client 接入。
- 两个 OOM 日志目录没有随压缩包上传到 GitHub；本仓库记录的是用户提供的排查结论，不是原始日志复核结果。当前最准确口径是“两次 Smoke 尝试 OOM；用户补充其中有单卡 Smoke 尝试；当时共享 NPU 被双机 vLLM 占用”。不要把两次尝试描述成已由仓库日志独立确认的完全相同 3-step / 8-card 实验。

## 19. 本地伴随工程测试入口

从仓库根目录运行：

```bash
bash experiments/piper-nero-ee-mix/run_companion_tests.sh
```

该 wrapper 只为原始伴随工程设置 `PYTHONPATH` 并调用 `python -m pytest`，不修改 `source-project-v0.2.0-preview/` 原包。直接在某些 pytest 安装方式下运行裸 `pytest -q` 可能因 `scripts` namespace package 不在导入路径而报 `ModuleNotFoundError`；wrapper 已处理这个路径问题。
