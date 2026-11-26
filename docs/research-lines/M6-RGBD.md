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

## 11. 按当前源码定位的接入地图

这一节把 M6 设计落到仓库中真实存在的函数。新 AI 不必重新猜测从哪里接入。

### 11.1 采集端：深度已经进入相机对象，但 Client 仍丢弃它

- `code/plugins/lerobot_camera_ros/lerobot_camera_ros/ros_image_camera.py`：`CamFrame` 同时保存 `color`、`depth`、`stamp_ms`、`recv`、`depth_stamp_ms`；`_on_image()` / `_on_depth()` 收到两路消息后由 `_try_pair()` 按时间戳配对；`latest()`、`recent()`、`read_nearest()` 可取得配对帧。
- `code/collector/nero_vla_client.py`：`NeroIO.__init__()` 在 `--cam-impl plugin` 时创建支持深度话题的 `RosImageCamera`；但 `NeroIO._observe_once()` 目前只取 `h.color`、`w.color`，返回 state + 两路 RGB + 时间差/画面年龄，没有把 `h.depth` / `w.depth` 放进观测。
- 同文件 `build_request()` 目前只发 `state`、`images`、`prompt`。`_pack_array()` / `_unpack_array()` 已支持 NumPy ndarray 的 dtype、shape、bytes 序列化，理论上可承载 uint16 深度数组，但必须补协议测试，不能只因序列化函数通用就认定端到端已支持。
- `code/collector/nero_record.py` 的 `_add_with_phase()` 已把 `CamFrame.depth` 交给 `DepthRecorder.add()`；`_save_inner()` 在 episode 保存前调用 `finish(episode_index, episode_size)`。`camera.depth_rgb_ms` 记录深度与彩色帧时间戳差，可用于离线筛查。

### 11.2 数据保存与 Dataset：深度不在现有 JSON 的 RGB 视频字段里

- `code/collector/nero_depth.py`：深度单独存为 `depth/front/episode_XXXXXX.mkv` 和 `depth/wrist/episode_XXXXXX.mkv`，FFV1 / `gray16le`，uint16 毫米，0 表示无效；`depth/depth_info.json` 记录每个 episode 的帧数和状态。
- `code/xr1-overlay/mibot/data/datasets/json_dataset.py`：`JsonDataset.__getitem__()` 通过 `_images()` 读取 prompt 中列出的 RGB 视频；当前没有读取 `depth/` 的分支。新增深度读取时，必须先核对真实 JSON 中 episode index 与 `depth/episode_XXXXXX.mkv` 的对应字段，不要默认 JSON 文件名就等于 episode index。
- `JsonDataset.__getitem__()` 返回的 payload 目前包含 `messages`、`action`、`action_mask`、`state`、`vlm_action_target` 等字段。深度应作为独立 tensor / validity mask 返回，不要伪装成普通 RGB 图片。

### 11.3 Collate / 模型：不能只在 Dataset 里加一个 depth 字段

- `code/xr1-overlay/mibot/data/collate/custom_collate.py`：`CustomCollate.__call__()` 对 RGB 消息调用 Qwen Processor；随后把非 `messages` payload 用 `default_collate(payloads)` 合并。因此 Dataset 返回固定形状的深度 tensor 后，Collate 有机会直接堆叠它，但还需要明确 dtype、分辨率、无效像素 mask 和 padding 约定。
- `code/xr1-overlay/mibot/models/runner/base_runner.py`：`training_step()` 将整个 batch 传给 `self.model(batch)`。
- `code/xr1-overlay/mibot/models/VLA/xr1.py`：`xr1.forward()` 当前会把剩余 `batch` 直接传入 `self.vlm(**batch, ...)`。如果只新增 `depth` key 而不在 XR1 模型中显式 pop / encode / fuse，它会继续传给 Qwen3-VL；这不是一个已经定义好的深度输入接口。
- `code/xr1-overlay/mibot/models/VLM/qwen3vl.py`：当前视觉输入是 `pixel_values`、`image_grid_thw` 等 Qwen RGB/视频字段，没有原生 uint16 depth encoder。M6 必须明确在哪一层产生深度 token、如何融合到 VLM/action context，以及如何控制 token 长度。

### 11.4 推理端：采集、协议、服务三处必须同时改

- `code/collector/nero_vla_client.py`：保持 M5 默认请求不变；建议为 M6 新增明确的观测结构或独立 preset，在 `build_request()` 中添加独立 `depths` / `depth_valid` / 时间戳字段，不要把深度覆盖到 `images`。
- `code/inference-overlay/scripts/serve_openpi_policy.py`：`XR1OpenPIPolicy._prepare()` 当前只按 `front`、`wrist` 调 `image_to_pil()`，然后送入 Qwen Processor；需要单独验证、解包和归一化深度，不能让 uint16 深度走 RGB `image_to_pil()` 路径。
- Client / Server metadata 应声明 depth 支持、单位、尺寸、相机 key、对齐方式和无效值语义；M5 metadata 不变，M6 使用独立服务端口和模型目录。

## 12. 本轮源码审阅发现并修复的 RGB-D 帧对齐问题

原 `DepthRecorder.add()` 在某路相机 episode 开头连续收到 `None` 时，不创建 writer；直到首张有效深度到来才开始写视频。这样即使 `finish()` 发现视频帧数少于 RGB episode 帧数，已经写出的深度视频仍会从较晚的 RGB frame_index 开始，后续按相同 frame_index 读取会发生静默错位。

仓库工作 overlay `code/collector/nero_depth.py` 已修复：
- 每次 `add()` 都累计 episode 帧号；某相机首次收到有效深度时，先按其分辨率回填之前缺失的全零帧。
- 已创建 writer 后遇到缺失深度，继续写全零帧。
- 某相机整个 episode 都没有有效深度时，`depth_info.json` 明确记录 `ok=false` 和缺失原因，不伪造视频。
- 深度尺寸变化或 dtype 不符合 uint16 时显式报错；`finish()` 同时检查 recorder 调用次数和视频帧数是否等于 RGB episode 帧数。
- 新增 `tests/test_nero_depth_alignment.py`，覆盖开头缺帧、episode 中间缺帧、整段无深度、帧数不一致和尺寸变化。

本地 CPU 单元测试结果：4 passed。**此修复目前只提交到 GitHub 工作 overlay，尚未部署到采集机 102，也没有用真实 ROS / PyAV 录制回放验证。** 在 102 部署前，应先备份当前文件，再做一条短 episode 的 RGB/depth 帧数、时间戳和 `load_depth()` 回读检查。原始快照 `code/snapshots/m6-context-20261126/102/home/nero_depth.py` 保持不变。

## 13. M6 后续研究路线：先验证深度有用，再研究几何表示

建议把 M6 拆成连续但可独立验收的阶段。这里是研究规划，不代表这些模块已经实现。

### M6-A：可靠 RGB-D 数据闭环（先做，不训练）

研究前提是每一帧 RGB 都能找到对应深度，而且知道深度是否可信。先完成：
- front / wrist 两路彩色与深度帧的 episode 对齐；统计每条 episode 的 RGB 帧数、depth 帧数、时间戳差、有效像素比例、全零帧比例。
- 保存并核对相机内参、彩色-深度对齐关系、相机外参；记录深度单位、无效值、裁剪范围和分辨率。
- 用 3–5 条短 episode 做录制、退出、重启、回读测试，确认 FFV1 解码帧数与 LeRobot episode 完全一致。
- 先输出可视化检查：RGB、深度灰度图、有效像素 mask、深度伪彩色仅用于人工检查，不作为训练输入替代物。

### M6-B：最小深度感知基线（第一轮模型实验）

核心问题：在动作空间、数据、训练预算都不变时，增加真实度量深度是否能改善 NERO 的空间定位和接触动作？

建议第一轮保持 M5 的 `nero_joint_sb`、30-step、Step Flag、NERO 数据和相同训练/评测任务，只增加 front/wrist depth 输入。先比较：
1. M5 RGB-only 原始基线；
2. RGB + 深度的简单输入基线（仅作为对照，需清楚说明深度如何编码）；
3. RGB 主干不变 + 独立轻量 depth encoder + validity mask + gated fusion。

推荐优先实现第 3 种：深度分支独立处理米制深度和无效像素，输出有限数量的空间 tokens，再通过小型 cross-attention / gated residual 注入视觉-动作上下文。这样 RGB 预训练路径尽量不动，也能让模型学习何时相信深度。front 与 wrist 保留 camera identity，避免两路视角混淆。

深度预处理必须保留度量意义：uint16 毫米转 float32 米；0 值转 validity mask；根据传感器有效范围裁剪；不要对每张图独立 min-max 归一化，否则相同像素值会失去跨帧距离含义。

### M6-C：从深度图走向显式 3D 几何

如果 M6-B 证明深度有收益，再研究模型是否需要更明确的几何表示：
- 用相机内参将有效深度像素反投影为 XYZ point cloud；必要时结合外参变换到机器人 base / tool 坐标系。
- 对比点云 encoder、稀疏 voxel / 3D CNN、局部 TSDF / occupancy 表示与 2D depth encoder。
- 研究 RGB 语义 token 与 3D 几何 token 的融合：RGB 更擅长识别“是什么”，几何分支补充“在哪里、离末端多远、是否有空间通过”。
- 控制点云数量、体素分辨率和 token 预算；不能因为引入 3D 表示就忽略 Qwen3-VL 原有图像 token 和 `MAX_LENGTH` / 显存成本。

### M6-D：深度辅助的闭环视觉伺服（后续扩展，不作为 M6 首轮验收条件）

可进一步研究：执行一个动作 chunk 后，利用新 RGB-D 观测估计末端与目标之间的相对位姿误差，再修正下一 chunk；把 VLA 的长程语义决策与局部几何闭环控制结合。需要明确它是“VLA + depth-aware visual servo”系统，而不是简单声称 VLA 自带视觉伺服。首轮应先完成 M6-B 的开放环对照，再独立引入闭环控制变量。

### M6-E：深度不可靠时的鲁棒性与不确定性

深度相机可能遇到反光、透明物体、遮挡、量程边缘、飞点、时间同步异常。可研究 validity-aware gating、深度 dropout 训练、噪声扰动、RGB-only fallback，以及模型是否能识别“当前深度不值得信任”。这比单纯把深度图作为第三张图片更有研究价值。

### 评测设计建议

优先选择对空间信息敏感的任务和场景变化：目标平移/高度变化、工具尖端对准、插入深度、接触位置、容器边缘避碰、局部遮挡。主要报告整体与逐步骤成功率、首次失败阶段、末端位置/姿态误差、接触/插入误差、碰撞与人工干预、推理延迟；同时报告深度有效像素比例和 RGB-depth 时间差。不要只比较训练 loss。

## 14. M5 checkpoint warm-start 的工程陷阱

`code/xr1-overlay/mibot/models/runner/base_runner.py` 的 `BaseRunner.configure_model()` 会加载 checkpoint 后检查 `missing_keys` 和 `unexpected_keys`；当前任何 mismatch 都会抛错。M6 新增 depth encoder / fusion 模块后，M5 checkpoint 中缺少这些新模块参数是预期现象，但其他既有模块必须严格匹配。

因此不能简单地把 `strict=False` 当作解决方案，也不能忽略所有 missing keys。实现时应建立明确 allowlist（仅限新增的 depth encoder / fusion 参数），记录每个 missing / unexpected key；任何 M5 原有 VLM、state/action projector、DiT、Action head 或 Step Flag 相关参数不匹配，都必须停止加载并排查。为此新增专门测试：M5 权重加载后，除 allowlist 中的新深度模块外，旧模型参数全部命中。

## 15. M6 当前可直接继续的任务与外部阻塞

新 AI 无需用户重复背景即可继续：
1. 审查并测试 `nero_depth.py` 帧对齐修复；准备 102 的备份和最小回归命令，但不要声称已部署。
2. 设计一份真实 RGB-D episode manifest，至少含 episode index、RGB/depth frame count、相机 key、时间戳差统计、有效像素比例、内外参来源。
3. 在没有真实 RGB-D 样本的情况下先写 Dataset / protocol 单元测试和接口草案，不要猜 JSON episode index 的映射。
4. 依据第 11 节文件地图完成 Client → MessagePack → 113 server 的 depth 字段设计，再开始模型融合。

当前仓库仍没有真实 RGB-D episode、相机标定文件、完整 ROS 驱动工作空间或 113 当前运行服务的完整源码。因此，Dataset episode join、实际深度质量和端到端推理必须等真实文件/机器访问后确认。
