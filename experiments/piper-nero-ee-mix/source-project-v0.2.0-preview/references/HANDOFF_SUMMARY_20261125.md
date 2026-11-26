# Piper -> NERO EF 混合训练（nero_ee_sb）阶段交接与代码说明

日期：2026-11-25
工程目录：/mnt/panhe/xr1
新环境：/mnt/panhe/xr1/.venv-piper-ef (Python 3.11.18 ARM64, UV 0.13.9)

## 一、核心技术路线与约束
1. 目标：使用 Piper 左臂 V2 数据增强 NERO 策略泛化（路线 A：统一 EF 动作语义 + M5 Step Flag + 统一 EE State，新布局 nero_ee_sb）。
2. 隔离红线：所有产物/脚本/包/输出仅限 /mnt/panhe/；不改系统 CANN 9.0.0；不复用/不修复旧 .venv；不用 Hugging Face 或代理。
3. 权重状态：本地模型 /mnt/wzh/checkpoints/Xiaomi-Robotics-1-5B/model_states.pt 与 /mnt/wzh/checkpoints/Qwen3-VL-4B-Instruct 已在位。

## 二、本轮已验证的事实与关键发现
1. 坐标系对齐：实测工作姿态下 Piper (link6) 与 NERO (MDH frame 7) 均以 Z_flange 为夹爪朝向/末轴旋转轴、X_flange 朝下、Y_flange 横向水平，ee_relative_action 使用 PIPER_TO_NERO_FLANGE_R=eye(3) 局部增量方向一致；两端工作空间基座范围不同，由统一 q01/q99 各自归一化。
2. 归一化契约：io.py 要求 q01/q99 形状 (1, 60)，有效维 0..9 满足 q99 > q01，10..59 填 0.0。
3. step_prompt 与图像：11 个 NERO 目录均满足 dataset='nero_*' 且有 step_index；Piper 自动剥离 observations.wrist_right 及其 prompt 标记。
4. Piper 数据筛选：v2_manifest 共 2633 条，过滤 source.split=='val' (30 条) 后，2603 条 (293,693 帧) 全部通过合法性校验。
5. 采样配比：_json_files 使用 set() 存路径字符串，通过建立 nero_r1 (504) + nero_r2 (504) + piper_v2 (2603) 软链接目录实现 2x NERO + 1x Piper 混合 (~1:1 帧比)。
6. 训练脚本坑点：train_nero.sh 含 git -C 且 set -eo pipefail（Atlas 无 git 会失败）；train_910c_full_joint.sh 硬编码 .venv/bin/python，开训需复制修改为 .venv-piper-ef/bin/python。
7. 占卡说明：Atlas 空闲时 tmux train 会话挂有 QwenKI 占卡脚本，中断不丢进度。

## 三、文件清单与作用
- mibot/utils/piper_kin.py: Piper 6轴 URDF A 正运动学
- mibot/utils/nero_kin.py: NERO 7轴 MDH 正/逆运动学
- mibot/utils/ef_align.py: 双机器人法兰对齐、相对 EF 动作与统一 10 维 EE State
- scripts/panhe/apply_ee_sb.py: 备份并向 io.py / json_dataset.py 注入 nero_ee_sb 分支与模型配置
- mibot/utils/io.py (+ .before_ee_sb): 新增 NERO_EE_SB_ACTION_PARTS 与 nero_ee_sb
- mibot/data/datasets/json_dataset.py (+ .before_ee_sb): 实现 nero_ee_sb 动作/State/Mask/dwell/去右腕
- configs/model/posttrain_nero_ee_sb_full.yaml: 模型配置 (action_layout: nero_ee_sb)
- scripts/panhe/make_ee_sb_config.py: 混合统计量计算与配置生成脚本（纯 NumPy）
- configs/data/nero3_piper_ee_sb_20261125.yaml: 混合数据配置与 mean/std/q01/q99
- /mnt/panhe/dataset/xr1_ee_sb_mix_20261125/: 软链接数据集目录 (nero_r1=504, nero_r2=504, piper_v2=2603)
- scripts/panhe/requirements_locked.txt: uv.lock 导出的锁定依赖
- scripts/panhe/wheel_urls_aarch64_cp311.txt: linux_aarch64 + cp311 的 93 个精确包 URL

## 四、当前状态与后续 3 步
- [已完成] ef_align 数值验证、nero_ee_sb 代码补丁、混合软链接索引、混合 stats 与 YAML 生成、93 个依赖 URL 提取。
- [进行中] 本机下载 93 个包 -> 跳板机中转 -> Atlas 离线装入 .venv-piper-ef。
- [待做] 1) JsonDataset 实测 1 条 NERO + 1 条 Piper；2) 生成隔离训练脚本 (去 git、用 .venv-piper-ef)；3) 确认空闲 NPU 后跑 3-step smoke test 并挂后台开训。