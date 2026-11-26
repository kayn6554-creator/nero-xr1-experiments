# 06 — NERO 真机部署

## 1. 当前部署基准

当前 NERO 已部署模型以 Epoch 3 为基准。现有 NERO 仓库 2026-11-16 记录的 checkpoint 名称为 `epoch3-step4026.ckpt`；M4、EF4 权重路径有明确记录。新 Piper+NERO 混训模型也默认训练 3 Epoch，但其 global_step 与旧模型不同，必须使用新训练实际生成的 `epoch3-step<global_step>.ckpt`。

## 2. 客户端输入不变

沿用现有 NERO 客户端请求：
- `state [8]` = 7 关节角 + gripper(m)。
- `images.front`、`images.wrist` RGB。
- `prompt`。

客户端不需要自己计算 FK，也不需要知道模型的 EE State 维度。

## 3. 服务端必须新增的逻辑

新模型 `nero_ee_sb` 与旧 `nero_ee` 的差别是：模型 State 已统一为 EE State。因此服务端必须：
1. 从请求 `state [8]` 计算 NERO flange FK。
2. 使用训练侧 `compose_ee_state()` 生成 `[1,60]`。
3. 使用新混训配置 `q01/q99` 做 State normalization。
4. 模型生成 30×60 action，使用新配置 mean/std 反归一化。
5. 提取 Step Flag，先做边界检测。
6. 仅对边界前动作执行 NERO IK。
7. 输出 `[30,8]` 绝对关节目标，并独立返回 `step_flag`、`step_boundary` 和 `ik` 报告。

具体字段和实现见 `docs/09-nero-xr1-integration.md`。可复用模块：
- `src/nero_piper_ee/deployment/ee_to_nero.py`
- `src/nero_piper_ee/deployment/nero_ee_sb_server_adapter.py`

## 4. 现有服务文件的状态

当前 NERO 仓库的旧 `apply_ef_serve.py` 只支持 `nero_ee`，不能直接支持新的 EE State + Step Flag 布局。当前训练源码包中的 `serve_openpi_policy.py` 也不支持 `nero_ee_sb`。

GitHub 仓库没有提交 2026-11-16 最终部署版 `serve_openpi_policy.py` 完整文件。因此先从 113 取回正在运行的 server 源码和 SHA256，再做精确集成。不要盲目应用旧 apply 脚本，也不要覆盖 M5 6713。

## 5. 客户端与 Step Prompt

当前混训配置明确 `step_prompt: false`，但 `step_flag.enabled: true`：
- Client 需要 `step_file`，用于边界处等待人工确认。
- Client 不得启用 `--step-prompt`。
- 需要调整已部署 M5 Client 的校验逻辑，使 Step Flag 门控不再强制要求 Step Prompt。
- Client 仍执行 30×8 absolute joint actions，保留原有执行保护。

## 6. 部署前后

部署前：
```bash
python scripts/deployment/inspect_deployment_artifact.py \
  --model-dir /data/hep/panhe/models/nero-piper-ee-sb \
  --checkpoint epoch3-step<实际global_step>.ckpt \
  --expected-layout nero_ee_sb \
  --output /data/hep/panhe/models/nero-piper-ee-sb/deployment-manifest.json
```

随后按 `docs/09-nero-xr1-integration.md` 完成服务端集成、metadata、WebSocket Dry、Client Dry 与受监督 Live Pilot。

部署时使用新端口，不替换已有 M3/M4/EF4/M5 服务。新端口应先由推理机现场检查确认，项目不预设端口。