# Changelog

## 0.2.0-preview — 2026-11-25

- 检查并固定 NERO XR1 experiments GitHub main commit。
- 纳入 NERO kinematics、Piper kinematics、EE alignment 和 step text reference overlay。
- 修正 EE State：现有客户端 state8 在服务端通过 FK + 训练同款 `compose_ee_state()` 转成模型 60-D State 容器。
- 将 EE Action → NERO IK 与 Step Flag 边界检测/hold 合并到一个 adapter。
- 明确服务端输出维度：模型 9 个有效 Action 通道，客户端仍接收 8 维 absolute joint action。
- 明确当前混训 `step_prompt=false`、`step_flag=true`，客户端步骤文件门控与 prompt 文本解耦。
- 训练默认 Epoch 从 2 修正为 3，默认通过 SelectedEpochCheckpoint 保存 Epoch 3。
- 新增 Epoch3 训练/部署血缘、NERO 客户端/插件/相机集成、安全和部署 artifact manifest 文档。
- 新增部署文件 SHA256 检查脚本和 6 项 adapter/metadata 测试。
- 不对 113 当前服务文件做未经校验的自动覆盖；等待获取运行中 server 源码后再生成精确补丁。
