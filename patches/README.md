# 补丁

`113-serving-from-atlas-20261113.patch`：用户上传的 Atlas 服务文件到 113 实际服务文件的差异，含 EF IK、逐请求日志和提速。适用起点不是原始上游，而是已经包含 M1/M2 的 2026-11-13 Atlas 快照。

先 `git apply --check` 再应用。不得再叠加同效果的 apply_ef_serve / apply_ef_log / apply_ef_fast。依赖与完整限制见 [代码说明](../code/README.md)。此补丁不包含 M5 或 M4/EF4 服务支持。


## 2026-11-16：M4 / EF4 / M5 推理端与采集端增量补丁

本次补丁包基于 2026-11-16 收集的 113 推理端和采集机文件，不重写原有网页或客户端。

补丁包文件：`nero_m4_ef4_m5_inference_patch_20261116_clean.zip`

SHA256：`5fc4a4b2866e9f233f5b438c1998c994fa8b91a32e70c48a04a519b876d2eb67`

关键文件 SHA256：

| 文件 | SHA256 |
|---|---|
| collector/nero_vla_client.py | `fbde6d04072013dcca24a2ba7bd2e2b8365e2ccc1d6236ec0f3eb21558323d67` |
| collector/nero_vla_web.py | `0afd1eff6a72be939138c8225e497521548fde52812387c3b81f7d2034dfc117` |
| collector/nero_vla_presets.json | `0f7a82e95b5f06a6d0dcd0cb45fe8949ac6a65c1bca2e9c46a51364f74ba6dc1` |
| 113/scripts/serve_openpi_policy.py | `e5128dfcc22bc94651fd8993c9089be3c82c2ea67efef058943c6ee2e34e8d54` |
| 113/mibot/utils/io.py | `b2a70d5b0eacf480fbaf035cfe365291cbc94738c6eadae93d7b5a3b424d8c94` |

补丁内容：

1. `nero_joint_sb` action mask 支持训练辅助通道 8。
2. M4/EF4 服务端将四步提示映射为训练使用的 Merge23 三段文本。
3. M5 服务端读取第 9 个模型输出通道，默认阈值 0.5、连续 2 帧触发边界；保持边界前最后有效动作，机器人回包仍为 [30,8]。
4. M5 服务端额外返回 `step_flag` 与 `step_boundary`。
5. Client 检查 M5 的 step_prompt 和 step_file，Live 在边界处截断动作并等待人工步骤确认；Dry 只记录不等待。
6. Client 将 Flag、边界索引、暂停/等待/确认状态写入 summary.json、chunks.npz、live_status.json。
7. Web 增加 M4/EF4/M5 预设与 M5 状态展示，保留现有 UI 和流程。

应用顺序：

1. 113：`bash apply_113.sh`，默认根目录 `/data/hep/panhe/xr1`。
2. 采集机：`bash apply_collector.sh`，默认目标当前用户家目录。
3. 按本日现场做法，在确认后单独重启网页；apply 脚本不重启网页或模型服务。
4. 服务端应用补丁后启动新模型服务；已运行旧进程不会自动热更新。

两个 apply 脚本都会校验原始文件 SHA256，不匹配就拒绝覆盖，并在写入前备份。**已应用后不得重复执行。**

补丁包内 tests：

- `tests/test_step_flag_logic.py`：Merge23 映射、M5 action mask、无边界、单帧尖峰忽略、边界截断、首帧边界保持。
- `tests/test_client_stepflag.py`：Flag 响应解析、无 Flag 兼容、step file 人工确认。

本次隔离环境执行 Python 编译检查通过；Client Step Flag 3 cases PASS；服务端逻辑专项测试 PASS。113 虚拟环境中的测试曾因所选解释器缺少 numpy 未完成，不将其记录为远端 PASS。详细现场状态见 [deployment-20261116.md](../docs/deployment-20261116.md)。


## 模型机理与操作文档

- [M0–M5 / EF–EF4 模型演进与工作机理](../docs/model-evolution-and-mechanisms.md)
- [NERO VLA 采集与测试台使用手册](../docs/vla-web-usage-manual.md)

2026-11-16 用户初步报告 M5 每步边界后暂停、按下一步后继续（约 1–2 次观察）；完整证据等级和待采集材料见 [部署记录](../docs/deployment-20261116.md)。
