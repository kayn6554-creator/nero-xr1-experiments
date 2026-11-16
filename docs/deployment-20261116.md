# NERO × XR-1：2026-11-16 推理部署与采集端集成记录

> 状态快照：2026-11-16。本文记录本日用户终端实际执行并确认的部署结果。服务启动/元数据/网页连接不等同于真机 Live 实验通过。

## 1. 本日完成范围

本次工作是在 2026-11-13 仓库快照上补齐 M4、EF4、M5 的推理端和采集端支持，并把四个模型接入现有测试网页。没有重写测试网页，没有替换报告服务，也没有重新训练模型。

完成事项：

1. 在 113 推理机应用 M4 / EF4 / M5 服务补丁。
2. 在采集机应用对应的 Client、Web、模型预设补丁。
3. 修正 M4、EF4、M5 配置中的 Qwen3-VL 本地处理器路径。
4. 启动 M5、M4、EF4 服务，保留既有 M3 服务。
5. 验证 M4、M5 元数据；用户确认 EF4 元数据无异常。
6. 重启原有网页进程，保留 8766 报告服务。
7. 用户确认 M5、M4、EF4 均可从测试网页连接。

**本日尚未完成：** 真实机械臂 Live Pilot、EF 原始基线 6710 服务恢复、正式对照实验。

## 2. 补丁来源与应用记录

使用的干净补丁包：**nero_m4_ef4_m5_inference_patch_20261116_clean.zip**

- ZIP SHA256：`5fc4a4b2866e9f233f5b438c1998c994fa8b91a32e70c48a04a519b876d2eb67`
- 补丁包内 SHA256SUMS 检查：用户在 113 上逐项执行 `sha256sum -c SHA256SUMS`，所有列出的文件均为 OK。
- 113 执行 `bash apply_113.sh` 成功，修改：
  - `mibot/utils/io.py`
  - `scripts/serve_openpi_policy.py`
- 原文件备份后缀：`.before_m5_20261116_134821`。
- 采集机执行 `bash apply_collector.sh` 成功，修改：
  - `~/nero_vla_client.py`
  - `~/nero_vla_web.py`
  - `~/nero_vla_presets.json`
- 采集端原文件备份后缀由 apply 脚本按执行时间生成 `.before_m5_<timestamp>`。
- 两个 apply 脚本均不自动重启服务；113 上已有服务需在确认后单独重启，采集网页需手动重启。

补丁源码和差异文件归档在 [2026-11-16 M4/EF4/M5 补丁目录](../patches/20261116-m4-ef4-m5-stepflag/README.md)。推理服务的实验室上游文件没有整体复制进仓库，只保存必要差异，避免把内部上游完整源码误当作本项目自有代码。

## 3. 机器与服务状态

### 3.1 113 推理机

工作目录：`/data/hep/panhe/xr1`

模型根目录：`/data/hep/panhe/models`

Qwen3-VL 处理器目录：`/data/hep/panhe/Qwen3-VL-4B-Instruct`

| 模型 | 模型目录 | WebSocket | GPU | tmux 会话 | 本日状态 |
|---|---|---:|---:|---|---|
| M3 | `m3-all8-async` | 6709 | 7 | `m3srv` | 原有服务保留 |
| M4 | `m4-merge23` | 6711 | 1 | `m4-merge23-6711` | 新服务启动成功 |
| EF4 | `ef4-merge23` | 6712 | 2 | `ef4-merge23-6712` | 新服务启动成功 |
| M5 | `m5-stepflag` | 6713 | 6 | `m5-stepflag-6713` | 新服务启动成功 |

本日新模型使用的 checkpoint 名称为 `epoch3-step4026.ckpt`。M4、EF4 的权重文件检查结果均为 11,004,891,197 bytes：

- `/data/hep/panhe/models/m4-merge23/epoch3-step4026.ckpt/checkpoint/mp_rank_00_model_states.pt`
- `/data/hep/panhe/models/ef4-merge23/epoch3-step4026.ckpt/checkpoint/mp_rank_00_model_states.pt`

M4、EF4、M5 的 config.py 原先仍引用 `/mnt/wzh/checkpoints/Qwen3-VL-4B-Instruct`，已改为推理机实际存在的路径 `/data/hep/panhe/Qwen3-VL-4B-Instruct`。三份配置均保留了 `.before_qwen_path_20261116_135944` 备份。

M4 启动时使用物理 GPU 1，通过 `CUDA_VISIBLE_DEVICES=1` 映射到进程内 `cuda:0`；EF4 使用物理 GPU 2；M5 在物理 GPU 6；M3 在物理 GPU 7。启动命令的 `--device cuda:0` 是可见设备内编号，不是物理 GPU 编号。

### 3.2 元数据验证

**M4（6711）实际收到的 metadata：**

| 字段 | 实际值 |
|---|---|
| protocol | `openpi-websocket-v1` |
| action_horizon | 30 |
| action_dim | 8 |
| action_layout | `nero_joint` |
| camera_keys | front, wrist |
| action_semantics | `absolute_joint` |
| state_layout | `joint7,gripper` |
| step_merge23 | true |
| step_flag.enabled | false |
| gripper_range | 0.0–0.1 |
| gripper_mode | absolute |
| RTC | disabled |

**M5（6713）实际收到的 metadata：**

| 字段 | 实际值 |
|---|---|
| protocol | `openpi-websocket-v1` |
| action_horizon | 30 |
| action_dim | 8（机器人动作回包维度） |
| action_layout | `nero_joint_sb` |
| camera_keys | front, wrist |
| action_semantics | `absolute_joint` |
| state_layout | `joint7,gripper` |
| step_merge23 | false |
| step_flag.enabled | true |
| step_flag.channel | 8（训练输出第 9 通道，0 起始索引） |
| step_flag.threshold | 0.5 |
| step_flag.consecutive_frames | 2 |
| step_flag.hold_after_boundary | true |
| step_flag.manual_step_confirmation | true |

EF4 服务启动成功，用户确认其元数据没有问题；本次对话没有粘贴 EF4 的完整 metadata JSON，因此仓库不伪造逐字段原始回包。

M5 日志显示权重键全部匹配，Qwen3VL VisionPatchEmbed 的 Conv3d → F.linear 替换相对误差为 5.21e-03，6713 开始监听。M5 日志还记录了本机及采集机 IP 的 WebSocket 连接建立/关闭事件。

### 3.3 GPU 资源与共享约束

本日部署前的 nvidia-smi 显示 8 张 RTX 4090 均有其他进程占用。M4 加载后 GPU 1 显存使用为 19,825 MiB、剩余 4,259 MiB。EF4 启动成功，但没有保存启动后的完整 nvidia-smi，因此不记录其最终剩余显存。

GPU 0–5 上存在其他 Python / JAX / N2M 进程。本轮没有终止这些进程。后续任何重启、换卡或显存清理前，都必须重新检查 PID、进程归属和 GPU 使用情况，不得按旧快照直接 kill。

### 3.4 EF 原始基线未部署

6710 当前没有监听。旧 `efsrv` 会话中曾尝试启动 `ef-all8-async`，加载阶段发生 CUDA OOM，进程退出。该失败不等于 EF4 部署失败，也不等于 EF 模型权重损坏。

因此，**EF vs EF4 的严格对照目前被阻塞**。M3/M4 与 M3/M5 两组对照不受此项影响。

## 4. M4 / EF4 推理端实现

服务端从模型数据配置读取 `step_merge23`。当其为 true 时，把采集端传来的四步 Step k/4 文本映射成训练时的三段文本：

| 客户端当前步骤 | 服务端送入模型的步骤文本 |
|---|---|
| Step 1/4 | Step 1/3，保留原 Step 1 名称 |
| Step 2/4 | Step 2/3: move to the target and measure. |
| Step 3/4 | Step 2/3: move to the target and measure. |
| Step 4/4 | Step 3/3，保留原 Step 4 名称 |

因此 M4/EF4 仍使用网页的四步任务流程，但服务端把第 2、3 步统一成训练使用的中间阶段提示。Client 会检查服务 metadata：服务端声明 `step_merge23=true` 时，必须启用 `step_prompt`。

## 5. M5 Step Flag 端到端实现

M5 的训练布局是 `nero_joint_sb`：

- 机器人动作仍为 8 维：7 个关节 + 夹爪。
- 训练模型额外输出索引 8 的 Step Flag。
- Flag 是反归一化后的 0/1 目标回归值，不是经过校准的概率。
- 服务 metadata 的 `action_dim=8` 指机器人动作回包，不代表模型内部没有第 9 个辅助通道。

服务端默认阈值 0.5、连续 2 帧触发，可通过环境变量 `NERO_STEP_FLAG_THRESHOLD` 和 `NERO_STEP_FLAG_CONSECUTIVE` 调整。当前首轮实验应固定默认值，不要在不同模型/任务间临时调整。

边界索引 k 的定义：预测块中 [0, k) 属于当前步骤，k 及之后属于下一步骤。触发后：

- k > 0：动作 k 及之后保持在动作 k-1 的目标。
- k = 0：整个动作块保持在请求时的当前关节/夹爪状态。
- 回包仍保持原有 `actions [30,8]`，另带 `step_flag` 和 `step_boundary` 诊断字段。

采集端 Client：

- 检查 M5 必须启用步骤文本，并提供 step 文件。
- Live 模式若边界落在本次执行区间内，只执行边界前有效动作，然后等待网页现有“本步完成”按钮更新 `~/.nero_vla_step`。
- 人工确认后才继续下一次推理；不会自动把 Flag 当作任务成功，也不会自动跳到下一步。
- Dry 模式记录 Flag 与边界预测，不等待、不驱动机械臂。
- 将 Flag 序列、边界索引、等待状态、暂停状态、确认步骤写入 `summary.json`、`chunks.npz`、`live_status.json`。

这意味着 M5 的推理端、客户端和网页交互代码已经完成部署；剩余工作是通过 Dry / Live Pilot 验证真实模型边界预测是否可靠，而不是重新开发这些功能。

## 6. 采集机网页与报告服务

采集机：`panhe@192.168.1.102`

- 测试网页：8767，脚本 `/home/panhe/nero_vla_web.py`
- 报告服务：8766，脚本 `/home/panhe/nero_report_server.py`
- 原网页进程位于 tmux `nero:10.0`。
- 已定位网页 pane 后，只停止并重启该 pane 的网页进程。
- 报告服务 PID 227912、端口 8766 在重启前后均保持运行，没有重启。
- 网页重新启动后，用户确认 M5 连接正常；M4、EF4 部署后也确认可以从网页连接。

网页连接成功证明 WebSocket 握手/metadata 路径可用，不等于模型已完成真实图像推理，更不等于机械臂 Live 动作已通过。

## 7. 测试与验证边界

已确认：

- 补丁包 SHA256SUMS 检查通过。
- 113 补丁应用成功，M5 服务权重加载成功并监听 6713。
- M4 服务权重加载成功并监听 6711，metadata 与预期一致。
- EF4 服务成功监听 6712，用户确认 metadata 正常。
- M3 原有 6709 服务保留。
- 采集端补丁应用成功，网页重启，M3/M4/EF4/M5 均可连接。
- 用户报告编译检查完成。
- 本仓库对应 clean patch 的两个专项单元测试在本次隔离容器中通过：Client Step Flag 3 cases PASS；M4/EF4 merge23 + M5 action-mask / boundary logic PASS。

未确认：

- 113 的 .venv 中专项单元测试尚未成功运行。此前使用的解释器缺少 numpy，系统也没有 `python` 命令；不要把该次失败记为逻辑测试失败，也不要记为远端测试通过。
- 尚未执行 M3/M4/EF4/M5 的完整图像请求 Dry 测试。
- 尚未进行任何本轮真机 Live Pilot。
- 尚未恢复 EF 原始基线服务。

## 8. 开始实验前的要求

1. 保持 M3/M4/EF4/M5 服务不变，开始前重新检查四个端口和 GPU 进程。
2. 每个模型先做网页 Check，再做 Dry；确认 metadata、两路相机、动作维度、延迟、首帧偏差和动作跳变。
3. M5 Dry 应能记录 Flag 序列；Dry 不会等待人工切步。
4. 只有现场急停可用、机械臂无其他控制任务、人员离开运动范围时，才允许单次 Live。
5. M5 Live 必须确认边界暂停后，网页“本步完成”按钮能更新 step 文件，且下一步骤能继续。
6. EF4 必须检查 IK 日志和拒绝原因；EF 原始基线未恢复前，不对 EF/EF4 作改进结论。
7. 所有运行使用独立 run_id，原始数据和历史评测不覆盖。

实验任务与正式协议见 [第一轮实验协议](experiment-protocol-20261116.md)。


## 9. 2026-11-16 后续真机初步反馈（部署后）

### 9.1 主臂 CAN 数据恢复

用户报告开始测试时主臂没有数据，但 can1 显示 UP / LOWER_UP、bitrate 1 Mbps、ERROR-ACTIVE，且 USB 序列号映射正确。被动监听 10 秒无帧，RX packets 从 251,228,743 到 251,228,743 不变。用户随后重新插拔后恢复。

记录边界：这是“重新插拔后恢复”的现场观察；没有进一步确认根因是 USB 接触、适配器固件状态、主臂供电还是其他物理链路因素。今后若复现，先只读核对 USB 序列号、CAN 状态、RX 计数和 candump；不要把拔插作为未诊断情况下的唯一修复方案。

### 9.2 M5 初步 Live 观察

用户使用 M5 进行了约 1–2 次测试，观察到每个步骤结束后机器人稳定停住，直到操作者按下下一步骤按钮才继续。

这与 M5 预期的端到端闭环一致：
1. 模型输出动作和 Step Flag；
2. 服务端检测边界并保持边界后的动作；
3. Client 只执行边界前动作并等待；
4. 人工确认当前物理步骤完成，网页更新 step 文件；
5. Client 读取步骤变化后继续推理。

**证据等级：初步现场观察。** 尚未收集这些运行的 run_id、视频、Flag 原始序列、边界索引和逐步成功评测，不能据此报告边界 precision/recall/F1、提前/延迟误差或任务成功率。下一步应保留原始日志和视频，正式做 M5 Pilot。
