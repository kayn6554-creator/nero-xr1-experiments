# 训练服务器与环境

## 登录

先连跳板机，再进宿主机（账号密码问学长，不写进仓库）：

```bash
ssh -J root@<跳板机>:2222 root@<宿主机>
```

文件传输同样走跳板机：

```bash
scp -J root@<跳板机>:2222 root@<宿主机>:/mnt/panhe/<文件> .
```

## 硬件

- 4 张 Ascend 910（每张 2 个芯片），共 8 个芯片，每个 64 GB HBM
- 内存 1 TB；`/mnt` 64 TB、`/mnt1` 256 TB（共享 NFS）
- CANN：`/usr/local/Ascend/cann-9.0.0`

## 使用约定

- 所有修改只在 `/mnt/panhe` 下进行；学长目录和其他人的目录只读
- 不动系统安装的 CANN；所有依赖装在自己的 uv 环境里
- 不配代理；只用国内源（阿里云 PyPI）和 ModelScope，HuggingFace 不可用
- 尽量少用管理员权限：不装系统级软件、不 kill 别人的进程、不用 `rm -rf`
- 占用 NPU 前先看 `npu-smi info`，并和学长确认

## 建环境

每个命令块开头都先加载 CANN：

```bash
conda deactivate 2>/dev/null; true
source /usr/local/Ascend/cann-9.0.0/set_env.sh
```

然后执行 `env/setup_env.sh`。要点：

- `uv sync --frozen` 按 `uv.lock` 原样安装，镜像为阿里云
- uv 不能自己下载 Python（连不上 GitHub），所以用 `--python` 指定已有的 3.11 解释器，并设 `UV_PYTHON_DOWNLOADS=never`
- 缓存放在 `/mnt/panhe/.uv-cache`，不写进 root 家目录
- peft 0.17.1 是本地 wheel，单独 `--no-deps` 安装

## 验证结果（2026-11-06）

- 与学长环境的 `uv pip freeze` 逐行对比：完全一致
- `torch 2.9.0+cpu`、`torch_npu 2.9.0.post2`，`torch.npu.is_available() == True`，识别 8 个设备
- torch 显示 `+cpu` 是正常的：NPU 支持来自 torch_npu
