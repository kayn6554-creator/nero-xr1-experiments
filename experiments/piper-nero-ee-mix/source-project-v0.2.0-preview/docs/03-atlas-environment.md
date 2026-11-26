# 03 — Atlas 环境

## 1. 登录路径

从 Windows 本机：
```bash
ssh -p 10022 root@172.16.36.16
```

从跳板机：
```bash
ssh root@10.20.10.15
```

## 2. 已约定的环境

| 项目 | 路径 / 版本 |
|---|---|
| XR1 repo | `/mnt/panhe/xr1` |
| 新 Python 环境 | `/mnt/panhe/xr1/.venv-piper-ef` |
| Python | 3.11.18 |
| CANN | 9.0.0 |
| 本地模型权重 | `/mnt/wzh/checkpoints/Xiaomi-Robotics-1-5B/model_states.pt` |
| Qwen VLM | `/mnt/wzh/checkpoints/Qwen3-VL-4B-Instruct` |
| 混合数据索引 | `/mnt/panhe/dataset/xr1_ee_sb_mix_20261125/` |

所有新增脚本、wheel、临时包、日志和训练输出应放在 `/mnt/panhe/` 下。不要修改系统 CANN，也不要复用旧 `.venv`。

## 3. 检查环境

```bash
bash scripts/atlas/check_environment.sh
```

检查脚本只读取环境信息，不安装、不卸载、不修改系统。

## 4. 依赖

`manifests/dependencies/` 收录本轮锁定依赖和 wheel URL 清单。离线安装前先校验文件名、Python ABI、平台 tag 和 SHA256。不要盲目对原始 requirements 执行整包安装；源码包的构建依赖需要单独处理。

## 5. tmux 操作约定

当前 Atlas 已知会话：
- `download`
- `qwenki-convert`
- `star`
- `train`
- `xr1-rtc6-20261030`

查看：
```bash
tmux ls
```

连接：
```bash
tmux attach -t train
```

detach：
```text
Ctrl-b，然后按 d
```

detach 只让 tmux 内进程继续运行，不会立即触发 checkpoint。Checkpoint 由训练程序按配置周期保存。不要 kill/替换既有 `train` 会话；训练前先检查实际 NPU 占用并确认是否有其他任务。