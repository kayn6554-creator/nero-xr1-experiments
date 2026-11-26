# 实验室三机与 Atlas 登录手册

**信息快照：2026-11-26**  
**适用范围：NERO × XR-1 采集、推理与训练环境**

> 本文记录的是本项目实际使用过的主机、路径和连接方式。内网地址、端口和权限可能变化；连接前应以实验室管理员的最新通知为准。本文不保存任何 VPN 密码、Linux 密码、SSH 私钥、访问令牌或主机密钥。

## 1. 三机分工

| 节点 | 地址 / 主机名 | 职责 | 已记录的关键路径 |
|---|---|---|---|
| 采集机 102 | `192.168.1.102`；`gzz2-MS-7D90` | NERO 真机、ROS 2、相机、LeRobot 采集、测试网页与报告 | `/home/panhe`；`/data/panhe/lerobot_data`；`/data/panhe/lerobot_camera_ros` |
| 推理机 113 | `172.16.11.113`；`ubuntugml98`；用户 `hep` | RTX 4090 推理、OpenPI WebSocket 服务 | `/data/hep/panhe/xr1`；`/data/hep/panhe/models`；`/data/hep/panhe/Qwen3-VL-4B-Instruct` |
| Atlas 主训练节点 | `10.20.10.15` | Atlas 训练/数据工作目录（此前用户明确确认此地址为 Atlas 主机） | `/mnt/panhe/xr1`；`/mnt/panhe/output`；`/mnt/panhe/dataset` |
| Atlas 33 代码快照节点 | `10.20.10.33`；`bms-13201202-001` | 2026-11-26 本次 M6 训练代码与配置快照来源；4 张 Ascend 910 物理卡、8 个逻辑 NPU 设备 | `/mnt/panhe/xr1`；`/mnt/panhe/output` |

**不要把 `10.20.10.15` 与 `10.20.10.33` 当成同一台机器。** `.15` 是用户后来明确确认的 Atlas 主机；`.33` 是本次采集包实际来自的另一节点，主机名为 `bms-13201202-001`。两台机器的环境、资源状态和 SSH 路由必须分别核对。

## 2. Atlas 主机登录

### 2.1 前置条件

1. 先连接实验室 VPN。
2. VPN 账号、密码由实验室管理员或用户本人保管；在本地 VPN 客户端中输入，不写入命令、脚本或 GitHub。
3. 首次 SSH 出现 host-key / fingerprint 提示时，先通过实验室可信渠道核对指纹，再决定是否接受。

### 2.2 已确认的 Atlas `.15` 登录顺序

在 Windows PowerShell、Linux 或 macOS 终端中，VPN 连接成功后：

```bash
ssh -p 10022 root@172.16.36.16
```

登录入口机后，在远端 shell 中继续：

```bash
ssh root@10.20.10.15
```

登录 Atlas 后先只读确认：

```bash
hostname
whoami
ls -ld /mnt/panhe/xr1 /mnt/panhe/output /mnt/panhe/dataset
cd /mnt/panhe/xr1
```

凭据均由 SSH 交互式提示输入。不要把密码写成 `sshpass` 参数，也不要把终端历史、密码文件或私钥提交到仓库。

### 2.3 通过入口机分段传文件

此前的操作约定优先采用分段 SCP，而不是假设 ProxyJump 一定可用。示例：

```bash
# 本地终端：先把文件传到入口机临时目录
scp -P 10022 ./package.tar.gz root@172.16.36.16:/tmp/
```

然后在入口机上：

```bash
scp /tmp/package.tar.gz root@10.20.10.15:/mnt/panhe/packages/
```

目标目录不存在时，先在 Atlas 上创建；传输后两端分别运行 `sha256sum` 核对。若入口机策略或路径变化，以管理员给出的最新流程为准。

## 3. Atlas 33 的独立端口隧道

2026-11-26 的 Atlas 33 快照来自 `10.20.10.33`，项目根目录为 `/mnt/panhe/xr1`。此前记录过以下独立隧道命令：

```bash
ssh -N -o ExitOnForwardFailure=yes \
  -J root@10.42.206.98 \
  -L 0.0.0.0:11223:127.0.0.1:11223 \
  root@10.20.10.33
```

该命令用于把本地 `11223` 转发到 `.33` 节点上的 `127.0.0.1:11223` 服务。它是**端口转发命令，不是 `.15` Atlas 的通用登录方式**。运行后终端会保持前台占用；需要结束隧道时使用 `Ctrl+C`。

曾记录过经 `172.16.36.16:10022` 再 ProxyJump 到 `.33` 的尝试，但该多跳路径曾遇到 `administratively prohibited`，不要把它写成已验证的通用登录路线。`.33` 的交互式 SSH 登录路线如有变化，先向管理员确认。

## 4. 采集机 102

- 地址：`192.168.1.102`；主机名：`gzz2-MS-7D90`。
- Python：`3.10.22`；采集环境解释器记录为 `/home/panhe/miniconda3/envs/evo-rl/bin/python`。
- 当前 ROS 相机插件实际导入路径：`/data/panhe/lerobot_camera_ros/lerobot_camera_ros/ros_image_camera.py`。
- 测试网页：`8767`；报告服务：`8766`。
- 数据目录：`/data/panhe/lerobot_data`。

### 4.1 RGB-D ROS 话题

头部相机：

```text
/cam/head/color/image_raw
/cam/head/aligned_depth_to_color/image_raw
/cam/head/aligned_depth_to_color/camera_info
```

腕部相机：

```text
/cam/wrist/color/image_raw
/cam/wrist/aligned_depth_to_color/image_raw
/cam/wrist/aligned_depth_to_color/camera_info
```

另有 `/depth/image_rect_raw`、`/depth/camera_info`、`/extrinsics/depth_to_color` 等 RealSense 原始话题。实际使用深度时优先确认 `aligned_depth_to_color` 与彩色图分辨率、时间戳和单位一致。

当前插件支持对齐到彩色图的 `16UC1` / `mono16` 深度帧；深度以 `uint16` 毫米表示，0 表示无效值。彩色与深度按 ROS 时间戳配对，容差默认 `3 ms`，缓存长度默认 `45` 帧。

## 5. 推理机 113

- 地址：`172.16.11.113`；主机名：`ubuntugml98`；用户：`hep`。
- Python：`3.11.18`。
- GPU：8 × RTX 4090。
- 项目：`/data/hep/panhe/xr1`；模型目录：`/data/hep/panhe/models`；Processor：`/data/hep/panhe/Qwen3-VL-4B-Instruct`。

2026-11-26 10:30 的进程快照记录：

| 服务 | 端口 | 模型目录 | GPU 快照 |
|---|---:|---|---:|
| M3 | 6709 | `m3-all8-async` | 7 |
| M4 | 6711 | `m4-merge23` | 1 |
| EF4 | 6712 | `ef4-merge23` | 2 |
| M5 | 6713 | `m5-stepflag` | 6 |

这些是当时的进程记录，不是永久 GPU 分配。重新启动、换卡或停止进程前必须重新检查 `nvidia-smi`、PID 归属和端口监听；不得按旧 PID 直接 `kill`。

## 6. 资源与共享机器安全

Atlas 33 在 2026-11-26 11:01 的快照中未发现训练进程或 tmux 会话，但 8 个 NPU 逻辑设备上有 `VLLMWorker_PP` 进程，HBM 使用约 57 GB / 64 GB，部分 AICore 利用率较高。**不要因为未发现训练进程就认为 NPU 空闲，也不要擅自终止这些进程。**

运行队列脚本前，先阅读脚本中停止/恢复常驻任务的逻辑，确认 PID、父进程、任务归属和实验室授权。`tmux attach` 只表示进入会话，不代表任务已经停止。

## 7. 不应提交到公开仓库的内容

- VPN / Linux 密码、SSH 私钥、访问令牌、`.env` 文件。
- 私有模型权重、checkpoint、数据集、个人采集视频。
- 原始运行日志、PID 列表、临时文件、完整 Conda/venv 环境。
- 未经确认的个人账号或其他实验室成员的进程信息。

公开文档只保留复现实验所需的主机角色、必要的内网地址、路径、端口和无凭据命令模板。若网络拓扑属于实验室限制公开的信息，应将该文件移至私有文档库，不应继续放在公开 GitHub。


## 8. Client 默认服务地址的注意事项

2026-11-26 的 102 Client 源码中，命令行默认值仍是 --server-host 172.16.11.115 --server-port 6699，这是历史默认配置；它不代表当前 M5 服务地址。当前 M5 预设是 172.16.11.113:6713。实际运行前以 nero_vla_presets.json 和推理机当前端口监听为准，必要时显式传入 --server-host / --server-port。不要因为脚本默认值就误连旧服务。
