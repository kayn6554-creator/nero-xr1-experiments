# 09 — 与 NERO XR1 真机工程集成

## 1. 上游仓库与版本

已检查：
`https://github.com/kayn6554-creator/nero-xr1-experiments`

本次固定参考版本：
- branch: `main`
- commit: `d4826f4f119a292d0ca061f637194000016360de`
- commit date: 2026-11-23 UTC

主要复用：
- `code/xr1-overlay/mibot/utils/nero_kin.py`
- `code/xr1-overlay/mibot/utils/nero_steps.py`
- `code/inference-overlay/scripts/panhe/apply_ef_serve.py`
- `code/inference-overlay/scripts/panhe/apply_ef_fast.py`
- `code/collector/nero_vla_client.py`
- `code/plugins/lerobot_robot_nero/`
- `code/plugins/lerobot_camera_ros/`

本项目不另写 NERO FK/IK，也不另写机械臂 ROS 控制器。

## 2. 现有 NERO 客户端协议

已提交的 `nero_vla_client.py` 请求格式：

- `state`: float32 `[8]`，joint1..joint7 (rad) + gripper (m)。
- `images`: `front`、`wrist`，RGB uint8。
- `prompt`: task instruction string。

客户端预期机器人动作回包：
- `actions`: float `[30,8]`，7 个绝对关节目标 + 绝对夹爪目标。
- `action_semantics`: `absolute_joint`。

因此，新的 EE-SB 模型不需要改变机械臂执行接口。**服务端内部负责 EE State 打包和 EE Action → IK；对客户端仍返回绝对关节动作。**

## 3. 为什么不能直接套用旧 `apply_ef_serve.py`

旧 EF 服务的 `nero_ee`：
- 客户端传入 joint7 + gripper；
- 模型 State 仍然是 joint7 + gripper；
- 模型 Action 是 EE 增量 + gripper + j3 增量；
- 服务端做 IK，回传 `[T,8]` 绝对关节动作。

新混训 `nero_ee_sb`：
- 客户端请求仍是 joint7 + gripper；
- 服务端必须先 FK，再调用训练时的 `compose_ee_state()` 构造 EE State；
- 模型 State 的有效维度为 10：位置 3 + Rotation 6D 6 + gripper 1；
- 模型 Action 的有效维度为 9：EE 位姿增量 6 + gripper 1 + elbow 1 + Step Flag 1；
- 机器人回包仍是 `[T,8]`。

因此新服务需要组合三部分能力：
1. `nero_ee` 的 FK/IK 转换；
2. `nero_joint_sb` 的 Step Flag 边界检测与 hold 规则；
3. `nero_ee_sb` 训练时的 EE State 归一化/布局。

不能把旧 EF 服务的 State packing 原样保留。

## 4. 服务端正确的数据流

### 4.1 输入 State

客户端发送：
`state8 = [q1, q2, q3, q4, q5, q6, q7, gripper_m]`

服务端：
1. `T0 = nero_kin.fk(state8[:7])`
2. `packed_state = compose_ee_state(T0, state8[7], is_piper=False)`
3. 得到 `[1,60]`，只有 slot 0–9 有效。
4. 使用混训配置中的 `q01/q99` 归一化，送入模型。

必须调用训练侧同一个 `compose_ee_state()`，不能自行重新定义 Rotation 6D 顺序或 base alignment。该函数使用 `NERO_BASE_ALIGN` 和 `piper_kin.rot6d()`。

### 4.2 输出 Action

模型生成并反归一化后：
- slot 0–2：局部法兰平移增量，m。
- slot 3–5：局部法兰旋转向量增量，rad。
- slot 6：夹爪目标，当前配置 `gripper_absolute: true`。
- slot 7：NERO j3/elbow 相对 chunk-start 的增量，rad。
- slot 8：Step Flag，反归一化后的 0/1 目标。

每个动作的 EE target：
`T_target = T_start @ T_delta`

其中 `T_delta` 的旋转由 `rot_exp(action[3:6])` 构造，平移由 `action[0:3]` 构造。

j3 目标：
`q3_target = q_start[2] + action[7]`

然后用现有 `nero_kin.ik()` 逐帧求解，seed 使用前一帧求出的关节状态。

### 4.3 Step Flag 与 IK 顺序

当前 M5 规则：
- 阈值：0.5。
- 连续帧：2。
- 第一次满足连续阈值的起点为 boundary index `k`。
- `[0,k)` 才属于当前步骤。
- `k` 及之后保持在 `k-1` 的最后有效目标。
- `k=0` 时整个块保持请求时当前关节状态。
- 回包仍为 `[30,8]`，另带 `step_flag` 和 `step_boundary`。

新服务建议**先检测 Step Flag，再对 `[0,k)` 做 IK**。这样不会因为边界之后那些本来就不应该执行的动作不可达，而拒绝整块请求。

IK 默认沿用现有 EF 经验：
- `NERO_EE_TOL_MM=0.1`
- `NERO_EE_TOL_RAD=0.001`
- `NERO_EE_MAX_JUMP_DEG=15`

任何有效前缀中的 IK 不收敛或关节跳变超限，都必须拒绝该请求；不能发送 NaN、未求解目标或部分错误动作。

## 5. Metadata 契约

`configs/deployment/nero_ee_sb_metadata.example.json` 是新服务应提供的 metadata 示例。

最重要的区分：
- `action_dim=8`：客户端收到的机器人动作维度。
- `model_action_dim=9`：模型内部有效预测通道数。
- `state_layout=joint7,gripper`：客户端请求格式。
- `model_state_layout=ee_flange_pos3_rot6d6_gripper1`：模型输入格式。
- `step_flag.enabled=true`。
- `step_prompt.enabled=false`：当前混训 YAML 没有开启 Step Prompt。

## 6. Step Flag 与 Step Prompt 必须独立

当前混训配置没有 `step_prompt: true`。因此：
- NERO `source.step_index` 仍用于生成 slot 8 的 Step Flag 监督。
- 训练 prompt 不追加 `Step k/4`。
- 推理端不应自行添加步骤文本。
- 客户端仍可读取 `step_file`，在边界处等待人工确认。

现有 2026-11-16 M5 Client 补丁文档要求 M5 同时启用 `step_prompt` 和 `step_file`，这是针对当时启用了 Step Prompt 的 `nero_joint_sb` 模型。新模型不能照搬这一前置条件。

需要在部署版 M5 Client 上调整校验逻辑：
- 若 metadata `step_flag.enabled=true`，要求 `step_file`，启用边界暂停/人工确认。
- 只有 metadata `step_prompt.enabled=true` 时，才要求 `--step-prompt`。
- `step_prompt.enabled=false` 时，不得把步骤文本追加到 prompt。

当前 GitHub main 中的 `code/collector/nero_vla_client.py` 是早期快照；2026-11-16 完整部署 Client 补丁只在仓库中保留了 hash/测试清单，完整 clean ZIP 未提交。不要用早期快照覆盖采集机已部署的 M5 Client。

## 7. 机器人执行层：复用原 NERO 插件

原仓库 `NeroFollower` 已包含：
- `/feedback/joint_states` 状态订阅。
- `/control/move_js` 关节命令发布。
- `/control/joint_states` 夹爪命令发布。
- `/control_enable` 控制锁。
- `/emergency_stop` 原地保持。
- 定位模式下不发送主臂指令。

原 `NeroFollowerConfig` 的关键安全参数：
- `read_only=True` 默认只读。
- `manage_control_gate=True`。
- gripper max 0.100 m，force 1 N。
- 单步 joint 限幅 0.035 rad。
- 单步 gripper 限幅 0.007 m。
- 主从重新对齐容差 5°。
- 跟踪误差上限 15°。
- 状态最大年龄 0.5 s。
- j7 使用实测收紧后的 ±80°，关节限位两端另留 1° margin。

这些参数属于已验证执行层，不要在本项目中绕过或复制出另一套控制插件。

## 8. 相机与输入边界

现有客户端使用 front + wrist RGB，`LatestCam` 采用队列深度 1，按时间戳配对两路图像；历史实测对齐目标为 18 ms 左右。`cam_impl=plugin` 可使用 ROS 相机插件并支持额外深度图像。

但当前 `nero_ee_sb` 训练配置与 OpenPI metadata 仍只定义 front/wrist RGB：
- 不把 depth 图像自动塞进模型。
- 不增加第三路图像。
- 不改变 prompt 图像顺序。
- 深度相机如需进入模型，需要另行训练/验证，不属于本轮接口兼容工作。

## 9. 113 服务端代码集成的证据要求

GitHub main 记录了 2026-11-16 的部署补丁哈希，但没有提交完整部署后的 `serve_openpi_policy.py` 文件。本项目 `references/upstream/scripts/serve_openpi_policy.py` 已保存 2026-11-25 训练包中的 Atlas 原始服务源码快照；但它不是 113 上叠加 2026-11-16 M5 补丁后的最终运行版本。

因此本项目**暂不提供会盲目覆盖 113 服务文件的 apply 脚本**。集成前必须：
1. 从 113 复制当前运行的 `scripts/serve_openpi_policy.py`。
2. 保存 `sha256sum` 和原文件备份。
3. 对照其中现有 `nero_ee`、`nero_joint_sb` 逻辑。
4. 将本项目 `nero_ee_sb_server_adapter.py` 集成到该版本。
5. 做单元测试、metadata 测试和 WebSocket Dry。
6. 确认输出为 `[30,8]` 后再更新客户端 preset。
7. 只在空闲端口启动新服务，不覆盖 M5 6713。

这一步需要拿到 113 的当前 server 源码才能生成可验证的精确补丁。
