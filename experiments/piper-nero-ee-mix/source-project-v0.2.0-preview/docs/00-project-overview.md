# 00 — 项目总览

## 1. 最终目标

本项目的最终目标不是训练一个面向 Piper 和 NERO 的通用双机器人策略，而是：

**利用 Piper 左臂数据增强 NERO 策略，训练完成后部署到真实 NERO 七关节机械臂。**

因此，训练阶段可以共享统一的 EE 语义，但部署和验收以 NERO 为唯一目标机器人。

## 2. 为什么统一 EE

Piper 左臂是 6 轴，NERO 是 7 轴。直接拼接关节动作没有自然的一一对应关系。

当前方案将两者的动作都转换为相对当前法兰坐标系的末端位姿增量：

- 局部位置增量 3 维。
- 局部旋转轴角增量 3 维。
- 夹爪 1 维。
- NERO elbow 增量 1 维。
- Step Flag 1 维。

Piper 没有 NERO elbow 与 Step Flag 的对应监督，因此按样本屏蔽这两维。

## 3. 阶段边界

### Training
Piper + NERO 数据进入统一 `nero_ee_sb` 数据布局，XR1 模型仍保持原有 60 维 action/state 容器。

### Evaluation
先验证数据和模型，再用固定的 NERO 测试集比较 NERO-only EF 与混训模型。

### Deployment
NERO 原始关节状态经 FK 转为 EE State；模型预测经反归一化、目标位姿重建和 IK，转换为 NERO 七关节目标与夹爪目标。

## 4. 不属于本项目的事情

- 不修改系统 CANN。
- 不重写 XR1 的 VLM/DiT/Trainer。
- 不训练 Piper 部署模型。
- 不把 Piper 的关节状态直接输入 NERO 真机策略。
- 不以训练 loss 下降代替真机成功率验证。
