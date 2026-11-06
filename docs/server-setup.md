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

# 推理机（hep@172.16.11.113）

- 8 × RTX 4090 24 GB，与他人共用；驱动 595.112.07。用前看 `nvidia-smi`，选剩余显存 ≥ 15 GB 的卡（11-06 是 6、7 号）
- 根目录只剩 37 GB；我们的东西都放 `/data/hep/panhe`（`/data` 下不能直接建目录，`/data/hep` 是 hep 账号自己的）
- 系统 Python 3.8，没有 conda；用 `pip3 install --user uv` 装 uv，再 `uv python install 3.11`
- 能访问 GitHub、阿里云和清华 PyPI 镜像；阿里云下载慢（约 250 kB/s），装 torch 用清华镜像
- 能直接连 Atlas 跳板机：代码、处理器文件、模型权重都用 `ssh -J ... "tar czf - ..." | tar xzf -` 直接拉

```bash
cd /data/hep/panhe
export UV_CACHE_DIR=/data/hep/panhe/.uv-cache UV_PYTHON_INSTALL_DIR=/data/hep/panhe/.uv-python UV_PYTHON_DOWNLOADS=never
export UV_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
# 代码（不含 .venv 和缓存）和 Qwen3-VL 处理器文件（不含权重）
ssh -J root@<跳板机>:2222 root@<Atlas> "cd /mnt/panhe && tar czf - --exclude=xr1/.venv --exclude=xr1/outputs --exclude=xr1/kernel_meta --exclude=xr1/robocasa20 --exclude='__pycache__' xr1" | tar xzf -
ssh -J root@<跳板机>:2222 root@<Atlas> "cd /mnt/wzh/checkpoints && tar czf - --exclude='*.safetensors' --exclude='*.bin' --exclude='*.pt' Qwen3-VL-4B-Instruct" | tar xzf -
cd xr1 && ~/.local/bin/uv venv --python 3.11 .venv
~/.local/bin/uv pip install --python .venv/bin/python -r /data/hep/panhe/requirements-serve.txt   # 内容同本仓库 env/requirements-serve.txt
```

推理只需要模型目录下的 `config.py` 和 `<epoch>.ckpt/checkpoint/mp_rank_00_model_states.pt`（11 GB）；`config.py` 里的 `vlm_config_path` 要改成推理机上的处理器目录。

# 采集机（panhe@192.168.1.102）

- 客户端和测试台在家目录：`~/nero_vla_client.py`、`~/nero_vla_web.py`、`~/nero_steps.py`（与训练用的 `mibot/utils/nero_steps.py` 同一份）
- 测试台：`/usr/bin/python3 ~/nero_vla_web.py`，本机 `ssh -L 8767:127.0.0.1:8767 panhe@192.168.1.102` 后打开 http://127.0.0.1:8767
- 运行记录 `/data/panhe/vla_runs/<时间>/<模式>/`，评测记录 `/data/panhe/vla_eval/<任务>.jsonl`
