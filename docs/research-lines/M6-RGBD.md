# M6：在 NERO 上验证 RGB-D 是否真正改善 VLA 的三维空间理解

**实验线定位：** NERO-only、M5-derived、RGB-D 感知扩展  
**状态：** 设计与前期工程准备；尚未完成 Dataset → Model → Inference 的端到端深度接入，也尚未训练 M6。

## 1. M6 要回答什么问题

当前 NERO 策略主要依赖 front/wrist RGB。RGB 可以提供颜色、纹理和物体类别线索，但距离、深度、表面几何和接触位置需要模型从二维图像间接推断。对机械臂而言，模型不仅要知道“是什么”，还要判断：

- 目标离夹爪有多远；
- 物体/容器的三维位置和高度；
- 探针、夹爪与目标表面之间的相对距离；
- 遮挡后物体是否仍在原位置；
- 机械臂接近、插入、对准和接触时，空间关系是否发生变化。

M6 的核心问题不是“把深度图也传进去就会变好”，而是：

**在保持原有 NERO 动作学习问题不变的条件下，显式加入真实深度信息，能否提高模型的空间定位、接近和接触动作质量？在哪些任务/场景下有帮助，在哪些情况下没有帮助甚至会带来负面影响？**

这是一条可验证的感知研究线，而不是单纯的传感器接入任务。

## 2. 与 M5 的关系：只改变观察输入，尽量不改变控制问题

M6 以 M5 为基线。为避免同时改变太多因素，第一阶段应保留：

| 项目 | M5 基线 | M6 第一阶段约束 |
|---|---|---|
| 机器人 | NERO 7-DoF + gripper | 不变 |
| 任务数据 | NERO 数据 | 先不引入 Piper |
| Action layout | `nero_joint_sb` | 保持 |
| Action horizon | 30 | 保持 |
| 机器人动作输出 | `[30,8]`，7 关节 + gripper | 保持 |
| Step Flag | 训练辅助通道 index 8 | 保持 |
| 步骤门控 | M5 边界检测、动作截断/保持、人工确认 | 保持 |
| 视觉输入 | front/wrist RGB | 扩展为 front/wrist RGB-D |
| 初始化 | M5 已训练模型 | 计划从 M5 Epoch-3 权重 warm-start，先验证 key/shape |

M5 的 Step Flag 只表达模型预测的步骤边界，不代表当前子目标已物理完成。M6 不应把它改造成成功检测器，也不应借 RGB-D 试验之名修改 Step Flag 语义。

候选 M5 权重路径（Atlas 历史清单）：

```text
/mnt/panhe/output/m5-stepflag/
  project_xr1-910c-joint-full/m5-stepflag/
    epoch3-step4026.ckpt/
      checkpoint/mp_rank_00_model_states.pt
```

该路径是已记录的候选初始化来源，不是本次重新读取或重新加载的证明。M6 开始前必须在 Atlas 检查文件存在、大小、SHA256、对应 `config.py` 和实际模型 key。优先使用模型参数 warm-start；不要把完整 Lightning/DeepSpeed optimizer/scheduler 状态误当成新 M6 的默认续训状态。

## 3. 现有 RGB-D 工程基础

### 3.1 102 采集端已有能力

`code/snapshots/m6-context-20261126/102/` 中保留了本轮采集端源码。

已确认：
- `lerobot_camera_ros/ros_image_camera.py` 的 `CamFrame` 含 `color`、`depth`、`stamp_ms`、`recv`、`depth_stamp_ms`。
- 深度为对齐到彩色图的 `uint16` 毫米值，0 表示无效。
- ROS 时间戳配对默认容差 3 ms，缓存 45 帧。
- `nero_depth.py` 已实现深度视频读写；使用 FFV1 / `gray16le`，每个 episode、每个 camera 一段视频。
- 目录结构：

```text
<dataset>/
└── depth/
    ├── front/episode_000123.mkv
    ├── wrist/episode_000123.mkv
    └── depth_info.json
```

- 每条深度视频按 episode 帧号保存；缺帧会用全零无效帧补齐，以保持索引对齐。

### 3.2 当前仍然断开的环节

| 环节 | 状态 |
|---|---|
| ROS 读取 aligned RGB-D | 已有 |
| depth episode 保存/读取工具 | 已有 |
| Client 使用 plugin camera 读取 depth | 部分已有 |
| Client 默认 fast camera 路径读取 depth | 未有 |
| Client → WebSocket 请求包含 depth | 未接通 |
| Atlas JsonDataset 按 episode/frame 读取 depth | 未接通 |
| Collate 组织 depth 张量/有效性 mask | 未实现 |
| XR-1 模型深度编码/融合 | 未实现 |
| 113 服务接收、解码和预处理 depth | 未实现 |
| M5 → M6 权重 warm-start | 未验证 |
| RGB-only vs RGB-D 对照 | 未进行 |

**重要：** 采集插件能读取深度，不等于 Client、训练 Dataset 或模型已经使用深度。不能把 M6 描述成“已经支持 RGB-D 推理”。

## 4. 推荐研究假设

以下是 M6 可检验的假设，不是已经得到的实验结论。

- **H1：** 深度信息对目标接近、插入、对准、测量等依赖距离/高度的动作有帮助，尤其在 RGB 外观相似但空间位置不同的场景中。
- **H2：** front depth 更可能帮助全局场景布局和目标位置；wrist depth 更可能帮助局部接触、夹持和末端相对几何。两者作用可能不同。
- **H3：** 深度质量会决定收益。无效值、反光/透明表面、边缘空洞、时间错位或尺度不一致可能使 RGB-D 模型比 RGB-only 更差。
- **H4：** 深度带来的提升不一定表现为总成功率统一上涨，可能首先体现在接近距离、插入高度、碰撞/擦碰、末端定位误差和失败阶段分布上。

## 5. 模型输入设计：先做最小可验证版本，再决定复杂融合

不要在没有数据质量审计的情况下先定最终网络结构。建议分层推进。

### 方案 A：深度图像化输入（最小工程基线）

把每张深度图做一致的米制转换、裁剪和有效性编码，再作为额外视觉输入送入模型。

优点：
- 改动相对少；
- 可以快速验证模型是否能利用深度线索；
- 适合作为早期 sanity baseline。

风险：
- 如果把 uint16 深度简单伪彩色化，模型可能只学习颜色映射，而不是度量几何；
- Qwen3-VL 的 RGB 图像预处理不等于原生理解 uint16 depth；
- front/wrist RGB 与 depth 的图像顺序、token 数量和 prompt 标记必须明确。

### 方案 B：独立 Depth Encoder + 视觉 Token 融合（推荐作为主研究候选）

保持现有 RGB / Qwen3-VL 路径和预训练权重，新增轻量深度编码器：
1. 输入米制深度与 validity mask；
2. 将 depth patch 编码到与 RGB visual token 兼容的隐藏维度；
3. 加入 camera/view embedding 和空间位置编码；
4. 在选定层与 RGB token 做拼接、交叉注意力或门控融合；
5. 训练新增深度模块，同时对原 RGB/VLA 参数做受控微调。

优点：
- RGB 和 depth 各自保留合适的输入处理；
- 能显式研究融合位置和融合强度；
- 便于做 depth-off、front-only、wrist-only 等消融。

风险：
- 需要改模型、collate 和 checkpoint loading；
- 新增 token 会增加显存、序列长度和训练成本；
- 需要检查 XR-1 / Qwen3-VL 的视觉 token 接口，不可凭想象修改。

### 方案 C：几何显式表示（后续探索，不作为第一版前置条件）

在有可靠内参、外参和对齐深度后，可从 depth 生成 XYZ、点云、局部高度图、TSDF 或 voxel 特征，再由几何 encoder 与视觉 token 融合。

这可能更直接地表达空间结构，但工程成本、标定要求和算力开销更高。建议在方案 A/B 证明深度确实有增益后再做。

## 6. 深度数据契约：必须先于模型训练确定

每个样本应能稳定取回：

```text
front_rgb[t]
wrist_rgb[t]
front_depth_mm[t]       # uint16, aligned to front RGB
wrist_depth_mm[t]       # uint16, aligned to wrist RGB
front_depth_valid[t]
wrist_depth_valid[t]
rgb_stamp_ms[t]
depth_stamp_ms[t]
camera_intrinsics
camera_extrinsics / alignment metadata
depth_scale
```

必须检查：
- RGB 与 depth episode 名称完全对应；
- RGB / depth 帧数一致；
- 时间戳差分布符合采集设置（插件默认配对容差 3 ms，但保存后仍需审计）；
- 深度单位确实是毫米，0 是无效值；
- 图像尺寸、对齐方式和相机内参一致；
- 深度有效像素比例、距离分布、离群值和缺帧率；
- front/wrist 两路各自独立统计，不要把不同相机的深度范围混成一套假设。

预处理建议：
- `uint16 mm → float32 meter`；
- 保留 validity mask；
- 采用固定、可记录的物理范围裁剪与归一化参数；
- 不建议对每帧单独 min-max 归一化，因为那会抹掉帧与帧之间的绝对距离尺度；
- 缺失深度不得静默当作“真实的 0 米”。可以选择剔除样本，或使用显式 validity mask / missing-view 标记。

具体 clip range 应根据真实数据分布和任务工作距离确定，不能在未检查数据前写死一个看似合理的数值。

## 7. M6 实施路线与验收门槛

### Phase 0 — 数据资产审计

- [ ] 统计当前 NERO 训练集每个 episode 的 RGB 与 depth 覆盖率。
- [ ] 检查 `depth_info.json` schema、帧数、视频编码、单位和时间戳。
- [ ] 可视化若干 front/wrist RGB-depth 配对帧，确认深度确实对齐到彩色图。
- [ ] 统计无效像素、距离分布、异常值和缺帧。
- [ ] 形成不可变的数据 manifest 和 train/validation/test episode 划分。

**通过标准：** 明确有多少训练/评估 episode 可用 RGB-D；缺失样本如何处理；不会将测试 episode 混入训练。

### Phase 1 — Dataset / Collate 接入

- [ ] `JsonDataset` 根据 episode 和 frame index 读取 front/wrist depth。
- [ ] 让 RGB、depth、mask 使用同一个 sample anchor 和同一组时间索引。
- [ ] `CustomCollate` 输出固定形状张量，明确 dtype、单位、view 顺序和 mask。
- [ ] 为无效深度、缺失视频、帧数不匹配写单元测试。
- [ ] 检查 MAX_LENGTH / 视觉 token 预算、batch 显存和 dataloader worker。

**通过标准：** 同一个 sample 能稳定返回 RGB、depth、mask、state、action；抽样可重复；不会错位或静默填错数据。

### Phase 2 — 模型与 M5 Warm-start

- [ ] 先检查 M5 checkpoint 与 M6 model class 的 key/shape。
- [ ] 保留原 RGB backbone 和 action head 的参数。
- [ ] 新增 depth encoder/fusion 层并记录初始化方式。
- [ ] 对 M5 权重加载结果输出 missing/unexpected keys；不允许用 `strict=False` 后不审查地继续训练。
- [ ] 做 RGB-only forward、RGB-D forward、backward、梯度有限性和显存测试。
- [ ] 验证 Step Flag 输出仍为原语义，机器人动作仍是 8 维。

**通过标准：** M5 原有路径正确加载；新增深度分支确实获得梯度；输入输出维度和动作统计未被意外改变。

### Phase 3 — 训练前 Smoke 与正式训练

- [ ] 使用独立 M6 config、experiment name、output root。
- [ ] 先单 batch 前向，再短步数 forward/backward smoke。
- [ ] 检查 loss、梯度、显存、吞吐和 checkpoint 保存。
- [ ] 正式训练时固定数据版本、split、seed、batch、训练步数和 checkpoint 选择。
- [ ] 保留 M5 baseline，不覆盖原模型目录。

**通过标准：** 训练确实使用 RGB-D，而不是只把 depth 放进 batch 却没有进入模型计算图。

### Phase 4 — 113 推理接入

- [ ] Client 在新 M6 preset 下传 front/wrist RGB-D 和时间/有效性元数据。
- [ ] Server 使用与训练一致的深度解码、单位转换、裁剪、归一化和 view 顺序。
- [ ] metadata 明确声明 RGB-D 输入，不伪装成普通 RGB。
- [ ] 新服务使用独立端口、模型目录和日志。
- [ ] M5 6713 保持不变。

### Phase 5 — Dry / 受监督 Live / 对照实验

- [ ] WebSocket Dry：确认四路图像/深度字段、时间戳、模型输出和延迟。
- [ ] Client Dry：确认不下发动作时可完整记录输入输出。
- [ ] 受监督 Live Pilot：保留 M5 的 Step Flag、边界保持、人工确认和机器人执行层保护。
- [ ] M5 vs M6 使用相同 NERO 任务、场景、物体位置、prompt、执行参数和评估规则。
- [ ] 记录每次失败属于感知、模型动作、Step Flag、IK/控制还是环境问题。

## 8. 评价指标

主要指标：
- 整体任务成功率、逐步骤成功率、首个失败步骤；
- 接近/插入/测量的空间误差（能可靠测量时）；
- 末端轨迹平滑性、接触/碰撞、物体掉落；
- 完成时间、推理延迟、人工干预；
- 深度有效像素比例、时间差、深度缺失/异常与失败的关系。

必须做的消融：
- M5 RGB-only；
- M6 RGB-D；
- M6 depth-off（保持 M6 网络/训练流程但屏蔽深度，区分参数量变化与深度信息贡献）；
- 条件允许时 front-only depth / wrist-only depth。

先报告每个任务的原始计数、失败阶段和置信区间，不把少量 Pilot 运行包装成稳定成功率。

## 9. M6 的边界：哪些事现在不要做

- 不与 Piper EE-SB 混训合并。
- 不改变 M5 的动作空间、Step Flag 语义和 Client 执行安全机制。
- 不因“深度相机已经能出图”就跳过 Dataset/Collate/Model/Server 的输入契约。
- 不在深度时间戳和坐标标定未审计前做点云/TSDF 的复杂路线。
- 不覆盖 M5 6713、M5 checkpoint 或原始 NERO 数据。
- 不声称 M6 已训练或已部署。

## 10. 当前需要新 AI 先完成的工作

1. 阅读 `nero_depth.py` 与 `ros_image_camera.py`，把真实 depth_info schema 和调用接口写成单元测试。
2. 对照 Atlas `JsonDataset`、`CustomCollate`、Qwen3-VL visual token path，画出最小 depth tensor 的数据流。
3. 设计 Dataset 输出契约和模型 fusion interface，先写 shape/unit tests，不急着选最终融合网络。
4. 形成 M5→M6 权重加载报告，确认新增模块的 key mapping。
5. 等用户提供/连接实际 RGB-D episode 后再做真实数据审计；在没有数据时不要假称数据覆盖率已知。
