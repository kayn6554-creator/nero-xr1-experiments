# NERO × XR-1 操作与交接手册

快照日期：2026-11-13。面向接手实验的同学，说明从收集代码到训练、部署和评测的顺序。除明确注明用户已执行的步骤外，示例命令属于恢复/部署模板，未在新环境重新执行。复制命令时不要复制终端提示符、历史输出或 `$`。

## 1. 三台机器与文件位置

| 机器 | 职责 | 已确认路径 |
|---|---|---|
| Atlas | NPU 训练、转换数据 | `/mnt/panhe/xr1`、`/mnt/panhe/output`、`/mnt/panhe/dataset` |
| 113 推理机 | GPU 加载权重、OpenPI WebSocket 服务 | `/data/hep/panhe/xr1`、`/data/hep/panhe/models` |
| 采集机 | ROS、主从臂、相机、录制和网页测试 | `/home/panhe`、`/data/panhe/lerobot_data`、`/data/panhe/vla_runs` |

公网仓库的操作模板用主机变量表示连接信息。已有代码/历史文档中的局域网地址是现场快照；迁移时集中核对。密码、SSH 私钥和访问令牌不能写进仓库。

在本地终端设置自己的真实地址：

```bash
JUMP_HOST=<跳板机地址>
ATLAS_HOST=<Atlas地址>
INFER_HOST=<推理机地址>
COLLECT_HOST=<采集机地址>
```

Windows PowerShell 使用 `$env:` 或显式替换地址；不要照抄 Bash 变量定义。

## 2. 登录与只读确认

Atlas 两跳登录（用户已验证这种方式）：

```bash
ssh -p 2222 root@<跳板机地址>
# 下面一条在跳板机执行
ssh root@<Atlas地址>
ls -ld /mnt/panhe/xr1 /mnt/panhe/output
cd /mnt/panhe/xr1
git status --short
ls -lh scripts/panhe/
```

或者从本地使用 `ssh -J root@<跳板机地址>:2222 root@<Atlas地址>`。密码由操作者在自己的终端输入。

推理机：`ssh hep@<推理机地址>`。采集机：`ssh panhe@<采集机地址>`，也可直接使用该机本地终端。

预期：目录可读；当前 Git 差异和脚本列表可以对应到本仓库快照。目录存在不代表训练完成，日志时间戳也不代表 checkpoint 可加载。

## 3. 本次代码收集过程

### 3.1 Atlas

先检查大小，避免把 2.4 GB 环境、权重和数据打包：

```bash
cd /mnt/panhe/xr1
du -h --max-depth=1 . 2>/dev/null | sort -h
ls -lh /mnt/panhe/output/
tar -czf /mnt/panhe/xr1-code-20261113.tar.gz \
  --exclude='.venv' --exclude='.git' \
  --exclude='kernel_meta' --exclude='outputs' --exclude='wandb' \
  --exclude='__pycache__' --exclude='.pytest_cache' \
  --exclude='*.pt' --exclude='*.pth' --exclude='*.ckpt' \
  --exclude='*.safetensors' --exclude='*.mp4' \
  -C /mnt/panhe xr1
ls -lh /mnt/panhe/xr1-code-20261113.tar.gz
```

本次生成约 14 MB。原包仍包含部分额外调试资料，整理上传时再次按文件用途筛选，压缩包本体不提交 GitHub。今后先用 `tar -tzf` 检查文件列表；文件大小只用于粗查，完整性以哈希为准。

本地 PowerShell 下载：

```powershell
scp -J root@<跳板机地址>:2222 root@<Atlas地址>:/mnt/panhe/xr1-code-20261113.tar.gz "$HOME\Downloads\"
Get-FileHash "$HOME\Downloads\xr1-code-20261113.tar.gz" -Algorithm SHA256
```

源端运行 `sha256sum /mnt/panhe/xr1-code-20261113.tar.gz`，与本地比较。本次历史下载已收到可解压文件，但未提供源端传输哈希，不能追认已完成传输哈希验证。

### 3.2 113 推理机

```bash
tar -czf /data/hep/panhe/xr1-inference-20261113.tar.gz \
  --exclude='__pycache__' --exclude='*.pyc' \
  -C /data/hep/panhe/xr1 scripts mibot configs
ls -lh /data/hep/panhe/xr1-inference-20261113.tar.gz
```

本次约 240 KB。从本地下载：

```powershell
scp hep@<推理机地址>:/data/hep/panhe/xr1-inference-20261113.tar.gz "$HOME\Downloads\"
```

### 3.3 采集机

用户分批收集网页/客户端、辅助脚本、录制/报告、插件。完整文件清单见 `source-manifest.json`；安装材料分别存于 `code/collector` 和 `code/plugins`。小包便于发现依赖遗漏；不要将整个家目录或整个 Conda 环境上传。

下面是以后重新收集自有脚本的统一模板，执行前确认每个文件存在：

```bash
tar -czf /home/panhe/nero-collector-source.tar.gz -C /home/panhe \
  nero_vla_web.py nero_vla_client.py nero_vla_ping.py nero_vla_presets.json \
  nero_tasks.json nero_steps.py nero_scenes.json nero_report_server.py \
  nero_replay_chunk.py nero_goto_leader.py nero_init.py check_limits.py teleop_tune.py \
  nero_record.py nero_record.sh nero_report_lib.py nero_steps_apply.py nero_verify.py \
  env_evo.sh pos_on.sh pos_off.sh nero_depth.py nero_cam_check.py nero_report.py
```

```bash
tar -czf /home/panhe/nero-plugins-source.tar.gz \
  --exclude='__pycache__' --exclude='*.pyc' --exclude='*.egg-info' \
  -C /data/panhe lerobot_robot_nero lerobot_teleoperator_nero lerobot_camera_ros
```

源包可能带旧备份，发布时只选当前源码。`scp panhe@<采集机地址>:/home/panhe/<包名> <本地目标目录>` 下载。

## 4. 依赖与环境恢复

### Atlas

历史环境使用 CANN 9.0.0、Python 3.11、uv 锁文件、torch 2.9.0 和 torch_npu 2.9.0.post2。硬件记录为 4 张卡、每卡 2 个芯片，共 8 个逻辑设备。不要修改共享 CANN。

```bash
source /usr/local/Ascend/cann-9.0.0/set_env.sh
cd /mnt/panhe/xr1
.venv/bin/python -c 'import torch, torch_npu; print(torch.__version__, torch_npu.__version__, torch.npu.is_available(), torch.npu.device_count())'
npu-smi info
```

历史结果为 NPU 可用且设备数 8。`torch` 显示 `+cpu` 不等于没有 NPU 支持。新建环境参考 `env/setup_env.sh`：必须把配套 pyproject.toml/uv.lock 放在正确根目录并核对 Python 与本地 peft wheel 路径；不能认为运行脚本会自动复制这些文件。

### 采集机

已确认激活顺序：先 Conda，再 ROS，再工作空间。

```bash
source ~/miniconda3/etc/profile.d/conda.sh
source ~/env_evo.sh
python -c 'import sys; print(sys.executable)'
```

`env_evo.sh` 激活 `evo-rl`。目前有旧构建前缀缺失提示，见来源文档。不要用随意建立软链接掩盖工作空间问题。三个插件在自己的源码目录执行 `python -m pip install -e <插件目录>`，且解释器必须是 evo-rl；驱动和相机启动方式另行核对，不能通过 import 成功代替设备检查。

### 推理机

参考 `env/requirements-serve.txt`，使用 CUDA 环境，不安装 Atlas 的 torch-npu 作为 GPU 替代。模型处理器目录需要匹配 Qwen3-VL。启动前 `nvidia-smi` 检查 GPU 使用者与可用显存，选实验室允许的卡。旧记录中的 GPU 号不是永久分配。

## 5. 数据采集、标注、检查和转换

1. 启动并确认 ROS 驱动、主从臂、两路相机；驱动启动命令尚未收集，使用现场已验证流程。
2. 编辑/核对 `nero_tasks.json` 的任务键、dataset、英文任务和步骤定义。相机与采集参数以 `nero_record.sh` 为准：头部/腕部图像及深度话题，front 为同步参考。
3. 使用已有报告服务和测试台。报告服务默认 8766，测试台默认 8767；采集控制通过 tmux 向录制终端发按键。tmux 目标以 `nero_report_server.py` 配置为准，不能随意 attach 到其他训练会话后发键。
4. 录制脚本先检查从臂状态、相机与主从对齐，再开始录制。以下命令会进入真实硬件流程，应在现场准备完成后执行：

```bash
source ~/env_evo.sh
~/nero_record.sh weigh
```

`weigh` 可换为任务表中存在的键。已有数据集会续录；空元数据目录可能被脚本改名备份。不要同时启动两个录制进程写同一数据集。

5. 采集时按实际步骤切换标签，区分成功与失败。失败前缀转换依赖失败步标签与末帧 step_index 一致，错误标签会影响训练数据保留范围。
6. 场景定义在 `nero_scenes.json`；当前场景和轨迹标签位于数据根 `_labels/`。数据本体、标签和深度应一起备份，不能只复制 RGB 视频。
7. 录制结束生成报告；报告界面的步骤修正先写入 report/step_edits.json，不立即修改原始数据。

```bash
python ~/nero_report.py /data/panhe/lerobot_data/<数据集名>
python ~/nero_verify.py /data/panhe/lerobot_data/<数据集名>
```

需要将修正写回时，先停止录制并备份，然后执行：

```bash
python ~/nero_steps_apply.py /data/panhe/lerobot_data/<数据集名>
python ~/nero_verify.py /data/panhe/lerobot_data/<数据集名>
```

该操作会改写字段/统计量，不是只读。检查报告中的帧数、步骤、图像和状态一致性，失败时保留备份，勿直接重采覆盖。

### Atlas 转换

成功基线转换依赖上游 `scripts/prepare_nero3_xr1.py`；新增任务使用 `prepare_nero_new_1110.py`，失败数据使用 `prepare_nero_fail_prefix.py`。先查看参数：

```bash
.venv/bin/python scripts/panhe/prepare_nero_fail_prefix.py --help
.venv/bin/python scripts/panhe/prepare_nero_new_1110.py --help
```

失败数据的已记录用法：

```bash
.venv/bin/python scripts/panhe/prepare_nero_fail_prefix.py \
  --source /mnt/panhe/lerobot \
  --output /mnt/panhe/dataset/xr1_nero_fail_prefix_20261106 --dry-run
```

先检查计数和标签匹配，再在新的输出目录正式转换（移除 `--dry-run`）。不要覆盖已被训练引用的数据。视频转码使用 imageio-ffmpeg；成功前缀 JSON 虽截断帧数，媒体可能仍是整段视频，不能用视频文件时长当训练帧数。

## 6. Atlas 训练与资源使用

用户约定：使用训练资源前可停下约定的常驻 QwenKI 训练任务，完成后恢复。`tmux attach -t train` 只是进入会话，不代表已停任务。先确认当前 PID、进程组和任务归属；不要停止其他人的训练。

已有 queue 脚本实现停任务→等待空闲→3 步冒烟→训练→恢复常驻任务。脚本包含共享路径、进程模式和超时逻辑，是现场历史快照，不应未经核对就启动。退出 trap 也可能恢复任务，读脚本后再决定使用。

训练入口链：`queue_*.sh` → `train_nero.sh` → `train_910c_full_joint.sh` → `train.sh` → `tools/train.py`。

先解决代码恢复说明中的补丁前提，然后在已准备的 Atlas 目录检查计划：

```bash
source /usr/local/Ascend/cann-9.0.0/set_env.sh
cd /mnt/panhe/xr1
XR1_CHECK_ONLY=1 XR1_DATA_CONFIG=nero3_m5sb_20261112 XR1_MAX_STEPS=4026 \
  bash scripts/panhe/train_nero.sh m5-stepflag \
  +data.params.train_datasets.step_prompt=true model.params.model.async_train=true
```

计划检查不应启动 NPU 训练。真实运行前检查输出目录是否存在，现有 train_nero.sh 会拒绝复用已有输出，保护历史实验。默认 8 芯片 × 每设备 batch 10，有效 batch 80；4 芯片时使用梯度累积 2，需同时核对实际 epoch/优化器步数。

为新实验名做冒烟（会使用 NPU，只有获分配资源后执行）：

```bash
XR1_SMOKE_STEPS=3 XR1_DATA_CONFIG=nero3_m5sb_20261112 \
  bash scripts/panhe/train_nero.sh m5-stepflag-check \
  +data.params.train_datasets.step_prompt=true model.params.model.async_train=true
```

观察 loss 是否有限、数据是否可解码、维度是否正确、进程是否正常退出。冒烟不保存 checkpoint，也不能证明模型质量。

正式训练使用单独新实验名和相应数据/模型配置；保持超参数、命令及 git 状态记录。M3/M5 4026 步，EF 另设 `XR1_MODEL_CONFIG=posttrain_nero_ee_full`。M4/EF4 的合并开关由队列额外传入 `+data.params.train_datasets.step_merge23=true`。

训练结束依次检查：日志停止原因、实际 global_step/epoch、最终 config.py、checkpoint 权重存在且非零、进程退出、资源释放、约定常驻任务恢复。目录和日志文件存在不等于成功。

## 7. 权重与最终配置收集/传输

每次推理至少保留输出中的 `config.py` 和选定 checkpoint 的 `checkpoint/mp_rank_00_model_states.pt`，不用复制优化器分片。不要混用不同实验的 config.py 与权重。

用户给出的远程 `tail` + `ls` 是检查命令，不是传输。先在 Atlas 定位：

```bash
EXP=m5-stepflag
find /mnt/panhe/output/"$EXP" -maxdepth 5 -type f \
  \( -name config.py -o -name mp_rank_00_model_states.pt \) -print
```

选定唯一 checkpoint，记录路径、步数和 SHA256。不要用未核对的 `epoch*` 通配符自动挑权重。随后在推理机执行模板：

```bash
EXP=m5-stepflag
CKPT=epoch3-step4026.ckpt  # 替换为选定的真实 checkpoint 名
mkdir -p /data/hep/panhe/models/"$EXP"/"$CKPT"/checkpoint
# 把下面源路径替换为上一步确认的真实文件路径
scp -J root@<跳板机地址>:2222 \
  root@<Atlas地址>:/<确认过的config.py路径> \
  /data/hep/panhe/models/"$EXP"/config.py
scp -J root@<跳板机地址>:2222 \
  root@<Atlas地址>:/<确认过的模型权重路径> \
  /data/hep/panhe/models/"$EXP"/"$CKPT"/checkpoint/mp_rank_00_model_states.pt
sha256sum /data/hep/panhe/models/"$EXP"/"$CKPT"/checkpoint/mp_rank_00_model_states.pt
```

源端对同一权重执行 sha256sum，比较两端值。保留源 config.py 的副本，部署副本中的处理器路径改成推理机实际路径；不要修改动作布局、归一化统计或步骤开关来“绕过”加载错误。

## 8. 推理服务

当前 113 已支持关节版与 EF；EF 解码为 30×8 绝对关节+夹爪，客户端不需要 IK。IK 采用逐帧种子及 j3 目标，默认容差 0.1 mm / 0.001 rad，超过关节跳变门限或不收敛会拒绝整个请求。`serve_fast.py` 用经过误差检查的线性运算替换视觉 patch embedding 的 Conv3d。

先使用实际 Python 环境查看入口参数：

```bash
cd /data/hep/panhe/xr1
.venv/bin/python scripts/panhe/serve_fast.py --help
```

运行模板（先确认 GPU 空闲、端口未占用、模型布局已支持）：

```bash
EXP=ef-all8-async
CKPT=epoch3-step4026.ckpt  # 必须与实际部署目录一致
CUDA_VISIBLE_DEVICES=<已分配GPU号> .venv/bin/python -u scripts/panhe/serve_fast.py \
  --model-dir /data/hep/panhe/models/"$EXP" \
  --checkpoint "$CKPT" \
  --processor /data/hep/panhe/Qwen3-VL-4B-Instruct \
  --device cuda:0 --host 0.0.0.0 --port <空闲端口> \
  2>&1 | tee -a /data/hep/panhe/models/"$EXP"/serve.log
```

服务加载器会拼接 `<model>/<checkpoint>/checkpoint/mp_rank_00_model_states.pt`，因此 `--checkpoint` 必须是 checkpoint 目录名，不能传单个权重文件路径。模板中 checkpoint/processor 必须按真实文件确认，当前包未包含实际服务启动命令，因此此模板不是已验证的现场启动记录。`CUDA_VISIBLE_DEVICES` 选物理卡后，进程内部通常使用 cuda:0。

**不得用该模板直接启动当前 M5：服务尚不支持 nero_joint_sb。** M4/EF4 同样需要核对步骤文本转换支持，不能仅在预设里新增名字就部署完成。

## 9. 客户端、网页与评测

保持 `code/collector` 中各文件在同一安装位置，当前代码默认家目录。网页通过 env_evo.sh 启动子进程，ROS 命令另用系统 Python。先核对预设 host/port、任务表和模型对应关系。

```bash
/usr/bin/python3 ~/nero_vla_web.py
```

默认本机 8767；远程访问可在本地建立隧道：

```bash
ssh -L 8767:127.0.0.1:8767 panhe@<采集机地址>
```

浏览器访问 `http://127.0.0.1:8767`。若已存在网页进程，不要重复启动占同一端口，也不要贸然停止当前采集。

先检查服务连接，然后只做 check / dry，检查相机新鲜度、同步差、状态、动作布局和 prompt_used。确认动作/夹爪检查通过后，现场操作者按已有真机流程运行。dry 读取设备但不执行预测动作；不要将 check/dry 与 live 混淆。

当前客户端每块记录 state/actions、步骤、指令、延迟及相机信息；保存 summary.json、chunks.npz、feedback.npz 和图像。网页评测保存任务 jsonl，并支持逐步成功/失败和备注。M5 新边界字段目前不会被这些脚本保存或展示。

评测记录应包含：模型及 checkpoint、任务/物品/场景、水位/摆放、执行步数、slow、步骤提示、失败步与失败类型。采用同一时段交替顺序，条件成功率的分母是实际走到该步的次数。原始 runs、eval、服务日志与训练配置保持关联，不要只保留截图或成功率。

## 10. GitHub 更新与回滚

1. 拉取仓库最新 main，检查已有修改；将采集/训练/推理快照按来源分开。
2. 仅提交自有代码、必要配置、补丁和文档。数据、权重、环境、运行日志、编译缓存、密码不提交。
3. 保留第三方来源与版本，不把内部上游完整拷贝误作自有开源项目。
4. 解析 Python/Shell/JSON/YAML、检查补丁与文件哈希。静态检查结果不能表述为真机验证。
5. 提交信息写清快照时间和未完成功能；部署修改另作提交，便于分开回滚。
6. 回滚优先 git revert 已发布提交；不要 force push 抹掉历史。服务器恢复使用部署前备份和记录的版本，不在机械臂执行过程中覆盖运行脚本。

## 11. 常见问题

| 现象 | 判断与下一步 |
|---|---|
| 把提示符、`-rw-r--r--` 粘贴后报 command not found | 只复制代码框命令；不需要改文件权限 |
| Git diff 停在 `:` / `(END)` | 按 q 退出分页；用 git --no-pager 读取 |
| pyAgxArm dubious ownership | 确认目录是已知共享驱动，用单次 `git -c safe.directory=<精确路径>` 读取；不配置全局通配信任 |
| apply 脚本 expected 1 match / already applied | 停止；检查版本和是否已应用，不靠全局替换强行通过 |
| M5 unsupported action_layout | 是推理支持尚未实现，不要改成 nero_joint 隐藏错误 |
| 只看到 `.commit` / `.log`，没有可加载权重 | 检查训练终止原因及 checkpoint 完整性 |
| ROS 旧路径 not found | 检查工作空间构建前缀和 source 链；Python 包能 import 不足以证明环境完整 |
| 修改了 JSON 但网页没有变化 | 有些任务/预设在启动时加载，按代码确认后在无运行任务时重启 |
| 新机器缺少 8765 定位服务或 TCP 配置 | 这些属于尚未收齐的外部组件，不要编造默认值进行真机控制 |

## 12. 尚需补齐

最终训练 config.py/日志/权重校验值、三机完整环境版本、驱动/定位服务启动配置、实际 GPU/端口启动命令、M5 推理与网页功能。后续补充须标日期和验证机器，避免把设计方案误记为已运行结果。


---

# 2026-11-16 现场部署补充（优先于本手册中 11-13 的旧状态描述）

本节是后续真实部署结果，覆盖本手册早期写下的“M5 尚不支持”“M4/EF4 仍需核对服务转换”“M5 字段不会保存”等旧快照描述。旧段落保留作为历史快照，不再代表 11-16 的当前状态。

详细逐项记录见 [2026-11-16 部署记录](deployment-20261116.md)，正式实验步骤见 [第一轮实验协议](experiment-protocol-20261116.md)。

## A. 2026-11-16 已部署的模型服务

| 模型 | 端口 | GPU | 目录 | 状态 |
|---|---:|---:|---|---|
| M3 | 6709 | 7 | `m3-all8-async` | 原服务保留 |
| M4 | 6711 | 1 | `m4-merge23` | 已启动、metadata 验证 |
| EF4 | 6712 | 2 | `ef4-merge23` | 已启动，用户确认 metadata 正常 |
| M5 | 6713 | 6 | `m5-stepflag` | 已启动、metadata 验证 |

113 根目录仍为 `/data/hep/panhe/xr1`；模型位于 `/data/hep/panhe/models`；处理器位于 `/data/hep/panhe/Qwen3-VL-4B-Instruct`。M4、EF4、M5 config.py 中的旧处理器路径已改为该实际路径，并保留修改前备份。

M4 metadata 的关键值：`nero_joint`、action_dim 8、horizon 30、step_merge23 true、step_flag disabled、absolute gripper。

M5 metadata 的关键值：`nero_joint_sb`、action_dim 8、horizon 30、step_flag enabled、channel 8、threshold 0.5、consecutive_frames 2、hold_after_boundary true、manual_step_confirmation true。

EF4 服务启动并监听 6712；本次对话未保留完整 metadata JSON，因此只记录用户确认“元数据没有问题”，不推测未展示的具体字段。

## B. 采集网页部署

采集机 `panhe@192.168.1.102`：

- 网页 `/home/panhe/nero_vla_web.py`，端口 8767，tmux `nero:10.0`。
- 报告服务 PID 227912，端口 8766。
- 仅重启了网页进程；报告服务和其他 tmux 窗口保持运行。
- 用户确认 M5、M4、EF4 均能从网页连接。

## C. 2026-11-16 补丁

干净补丁包：`nero_m4_ef4_m5_inference_patch_20261116_clean.zip`，SHA256：

`5fc4a4b2866e9f233f5b438c1998c994fa8b91a32e70c48a04a519b876d2eb67`

113 的 apply_113.sh 和采集机 apply_collector.sh 均成功；应用脚本先校验原文件 SHA256、备份旧文件，不自动启动服务或重启网页。用户在 113 上对补丁包 SHA256SUMS 执行校验，所有文件均 OK。

## D. M5 实际推理与客户端能力（已实现）

- 服务端支持 `nero_joint_sb`，训练辅助通道 8 不进入机器人动作接口；机器人仍收到 `actions [30,8]`。
- 服务端回传 `step_flag` 和 `step_boundary`。
- 默认阈值 0.5、连续 2 帧；边界帧及之后保持在边界前最后有效动作；若首帧即触发则保持请求当前状态。
- Live 模式只执行边界前动作，然后等待网页“本步完成”按钮更新 step 文件，再继续。
- Dry 模式只记录 Flag，不等待、不驱动机械臂。
- Flag 序列、边界索引、暂停/等待/确认信息进入 summary.json、chunks.npz、live_status.json。
- 网页显示边界预测状态，并明确提示边界预测不等于任务成功。

M5 的推理/客户端/网页实现已经部署。仍待验证的是模型真实 Flag 预测质量和机械臂上的暂停—人工确认—继续闭环，不是代码尚未开发。

## E. EF 基线与共享 GPU 限制

EF 原始服务 6710 当前未运行。旧 efsrv 尝试在 GPU 0 启动时发生 CUDA OOM。EF4 正常启动不代表 EF 基线可用。不得用 M3 代替 EF 做 EF/EF4 消融。

GPU 是共享资源；部署前 GPU 0–5 均有其他进程。M4、EF4、M3、M5 分别使用 GPU 1、2、7、6。不得按旧快照 kill 其他用户进程；任何重启前重新检查 PID、端口和 GPU 显存。

## F. 验证边界

- 用户确认 113 补丁应用、模型服务启动、网页连接。
- 用户报告编译检查完成。
- 本次隔离环境对 clean patch 执行 Python 编译检查通过，Client Step Flag 专项测试 3 cases PASS，服务端 Merge23 / action mask / Step Flag 边界逻辑专项测试 PASS。
- 113 虚拟环境中的专项测试曾因所用解释器缺少 numpy 而未成功运行；不要把这次环境依赖错误写成逻辑测试失败，也不要写成远端测试通过。
- 尚未完成真实图像请求 Dry 全覆盖、M5 Live 边界闭环验收和本轮正式真机实验。

**本手册早期第 8、9、11、12 节中关于 M5 不支持、M5 不记录边界字段、M4/EF4 服务转换尚未实现的表述，均由本补充覆盖。**


---

## 2026-11-16：采集与 VLA 测试台操作手册入口

新增 [VLA 采集与测试台使用手册](vla-web-usage-manual.md)。手册按“恢复现场 → 打开网页 → 选择任务/模型 → Check → Dry → Live → 评测”组织，包含：

- 采集机 8767、报告服务 8766、113 模型 WebSocket 服务端口的区分；
- SSH 隧道、进程/端口/tmux/CAN 只读检查；
- can0 从臂、can1 主臂 USB 序列号映射；
- Check、Dry、动作块回放、Live、模型预设各输入项解释；
- M5 Step Flag 边界暂停和人工“本步完成”确认流程；
- 数据文件、日志、常见问题和安全门槛。

注意：手册中逐项 UI 字段主要来自仓库 11-13 网页源码快照；M5 新流程依据 11-16 补丁说明与现场记录。由于最终部署版网页/客户端源码尚未完整归档，手册已明确标出这一版本同步限制，不能把它误称为逐行核对过的线上源码手册。


> **2026-11-26 状态提示：** 本手册主体是 2026-11-13 的历史操作快照，不应作为当前登录入口、训练脚本或 M6/Piper 状态的单一依据。当前 Atlas VPN/SSH/隧道和资源安全以 infrastructure-access-20261126.md 为准；新 AI 应先读 AI-HANDOFF-2026-11-26.md。Piper + NERO EE-SB 训练不要直接套用本手册中的旧 train_nero.sh、旧 .venv 或 raw-frame step planner。
