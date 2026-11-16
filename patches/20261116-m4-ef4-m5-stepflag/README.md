# 2026-11-16 M4 / EF4 / M5 推理端与采集端补丁

来源：`nero_m4_ef4_m5_inference_patch_20261116_clean.zip`

ZIP SHA256：`5fc4a4b2866e9f233f5b438c1998c994fa8b91a32e70c48a04a519b876d2eb67`

本目录记录本次补丁的可追溯清单与测试源码。补丁应用到真实服务器的完整文件来自上述 clean ZIP；ZIP 本体不提交为二进制仓库文件。若需要重放原补丁，应使用保留的原始 clean ZIP，并先核对哈希。

## 文件哈希

| 文件 | SHA256 |
|---|---|
| `collector/nero_vla_client.py` | `fbde6d04072013dcca24a2ba7bd2e2b8365e2ccc1d6236ec0f3eb21558323d67` |
| `collector/nero_vla_web.py` | `0afd1eff6a72be939138c8225e497521548fde52812387c3b81f7d2034dfc117` |
| `collector/nero_vla_presets.json` | `0f7a82e95b5f06a6d0dcd0cb45fe8949ac6a65c1bca2e9c46a51364f74ba6dc1` |
| `113/scripts/serve_openpi_policy.py` | `e5128dfcc22bc94651fd8993c9089be3c82c2ea67efef058943c6ee2e34e8d54` |
| `113/mibot/utils/io.py` | `b2a70d5b0eacf480fbaf035cfe365291cbc94738c6eadae93d7b5a3b424d8c94` |

## 113 改动

- action layout 增加 `nero_joint_sb`，action mask 开启 0–8。
- 支持 `step_merge23` prompt 映射，M4/EF4 服务端文本与训练配置一致。
- M5 将反归一化后的索引 8 作为 Step Flag；默认阈值 0.5、连续 2 帧。
- 边界索引 k：只保留 [0,k) 当前步骤动作，k 及之后保持在 k-1 目标；k=0 保持请求当前状态。
- 机器人回包仍为 `actions [30,8]`，另回传 `step_flag`、`step_boundary`。
- 环境变量：`NERO_STEP_FLAG_THRESHOLD`、`NERO_STEP_FLAG_CONSECUTIVE`。

## 采集端改动

- Client 要求 M5 启用 step_prompt 和 step_file。
- Live 模式边界处截断动作，等待网页“本步完成”按钮更新 step 文件；Dry 模式只记录不等待。
- `summary.json`、`chunks.npz`、`live_status.json` 记录 Flag、边界和等待/确认状态。
- Web 保留原 UI，增加 M4/EF4/M5 预设及 M5 状态显示。

## 应用记录

- 113 执行 `apply_113.sh` 成功；备份后缀 `.before_m5_20261116_134821`。
- 采集机执行 `apply_collector.sh` 成功；网页手动重启，报告服务 8766 保持运行。
- 两个 apply 脚本会校验原文件哈希，不匹配则拒绝覆盖；应用后不要重复运行。
- 端口：M3 6709、M4 6711、EF4 6712、M5 6713。

## 测试

- `tests/test_step_flag_logic.py`：Merge23 文本映射、action mask、无边界、单帧尖峰忽略、边界截断、首帧边界保持。
- `tests/test_client_stepflag.py`：Flag 响应解析、无 Flag 兼容、人工 step 文件更新。
- 隔离环境 Python 编译检查通过；Client 专项测试 3 cases PASS；服务端逻辑测试 PASS。
- 113 虚拟环境的测试曾因所选 Python 缺少 numpy 而未成功运行，不能记为远端测试 PASS。
- 这些测试不加载 checkpoint、不连接真实机械臂；真实边界预测质量仍需 Pilot。

详细部署状态和实验准备见 [deployment-20261116](../../docs/deployment-20261116.md)。
