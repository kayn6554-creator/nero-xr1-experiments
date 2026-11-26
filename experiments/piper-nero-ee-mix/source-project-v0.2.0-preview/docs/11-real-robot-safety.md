# 11 — 真机安全与验证分层

## 1. 三层保护必须同时保留

### 模型服务层
- EE Action → IK。
- IK 不收敛即拒绝整块。
- 默认最大相邻 IK 关节跳变 15°。
- Step Flag 边界之后不继续求解/执行动作。

### NERO Client 层
现有 NERO client 的检查包括：
- metadata `action_dim=8`。
- `action_semantics=absolute_joint`。
- 动作有限值。
- 首帧偏离当前状态上限 5°。
- chunk 内相邻动作跳变上限 30°。
- 夹爪单位/范围检查。
- 只执行配置指定的前若干步，再重新观测和推理。
- Live 异常时断开机器人插件。

### NERO Follower 插件层
- read-only 默认打开。
- 非只读时通过 `/control_enable` 管理控制锁。
- `/emergency_stop` 原地保持。
- 关节限位、单步限幅、夹爪限幅。
- 主从重新对齐检查。
- 跟踪误差和状态过期检查。
- 网页定位模式下不执行模型命令。

不得因为服务端已经做过 IK，就删除 Client 或 Follower 的保护。

## 2. 推荐测试顺序

1. **Unit**：FK/IK 回环、Step Flag 阈值/连续帧、边界 hold、关节跳变。
2. **Server metadata**：`action_dim=8`、`model_action_dim=9`、`state_layout` 与 `model_state_layout` 明确区分。
3. **WebSocket Dry**：真实请求 front/wrist RGB + joint state，确认输出 `[30,8]` 和 Step Flag 字段。
4. **Client Dry**：只读连接从臂和相机，检查首帧偏差、动作跳变、夹爪范围、IK 报告和 Step Flag。
5. **受监督 Live Pilot**：现场急停可用、执行区域清空、单一客户端、低速/短执行段。
6. **多次重复**：检查 Step Flag 提前触发、迟触发、漏检、边界 k=0、人工确认后恢复。
7. **正式实验**：记录 run_id、模型 checkpoint、metadata、原始图像、动作块、feedback 和人工判定。

## 3. 不允许的操作

- 不允许将 EE 增量直接发给关节控制器。
- 不允许把 slot 8 Step Flag 当成第 8 个关节。
- 不允许绕过 IK 失败或关节跳变错误。
- 不允许通过 `step_prompt=true` 伪造训练中不存在的条件输入。
- 不允许在已有 M5 6713 服务上直接替换文件/权重。
- 不允许将 WebSocket 握手或 metadata 正常当作真机 Live 成功。