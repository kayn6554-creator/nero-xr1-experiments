# 02 — 数据与动作契约

## 1. 当前数据混合方式

| 来源 | Episode | Raw frames | 索引权重 |
|---|---:|---:|---:|
| NERO 原始数据 | 504 | 141,330 | 2 份 |
| Piper V2 左臂 | 2,603 | 293,693 | 1 份 |

索引路径：
- `nero_r1`
- `nero_r2`
- `piper_v2`

NERO 两份通过独立软链接目录进入数据集索引。当前 JsonDataset 对路径字符串去重，而不会解析软链接目标，因此两个 NERO 入口会形成两倍采样权重。

Raw frame 量：
- NERO：282,660（重复后）
- Piper：293,693
- 合计：576,353

启用 Dwell Filter 后，实际 anchor 数会进一步减少；训练步数应使用过滤后的有效样本数计算。

## 2. State：60 维容器中的 10 个有效维度

| Slot | 内容 |
|---|---|
| 0–2 | EE flange position，base frame 表达 |
| 3–8 | Rotation 6D |
| 9 | Gripper opening，米 |
| 10–59 | padding，零 |

State 使用混合数据集统计得到的 `q01/q99` 做 quantile normalization。

## 3. Action：60 维容器中的 9 个有效维度

| Slot | 内容 | NERO | Piper |
|---|---|---|---|
| 0–2 | 当前 flange frame 下的局部位置增量，m | 监督 | 监督 |
| 3–5 | 当前 flange frame 下的相对旋转轴角，rad | 监督 | 监督 |
| 6 | 夹爪目标，m（当前配置为 absolute） | 监督 | 监督 |
| 7 | NERO joint index 2 / j3 的增量，rad | 监督 | mask |
| 8 | Step Flag | 有 `source.step_index` 时监督 | mask |
| 9–59 | padding | mask | mask |

注意：slot 7 是 NERO 的 elbow 增量；它不是通用 EE 维度。Piper 样本在该通道必须 mask。

Step Flag 当前定义：当未来帧的 `step_index` 与当前帧不同，目标为 1，否则为 0。当前混训 YAML 没有开启 `step_prompt`，但 NERO 数据的 Step Flag 动作监督仍会生成。

## 4. 夹爪

混训配置：
```yaml
gripper_absolute: true
```

Piper 归一化夹爪值当前乘以 `0.1` 转为 NERO 对应的米制尺度。真机部署前需要核实 NERO 控制接口的实际夹爪单位和有效范围。

## 5. 图像与提示词

- NERO：保留 front / wrist 两个视角。
- Piper：使用 ego / wrist_left；`wrist_right` 及对应 prompt marker 被剥离。
- Step Prompt 当前关闭；Step Flag 是独立 Action 通道，不要混为一谈。

## 6. 统计量

混训 YAML 保存：
- `mean`: `[30, 60]`
- `std`: `[30, 60]`
- `q01`: `[1, 60]`
- `q99`: `[1, 60]`

slot 10–59 的 state quantiles 使用零 padding。训练与推理必须使用同一份统计量。