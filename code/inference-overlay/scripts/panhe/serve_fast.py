"""启动 serve_openpi_policy，但把 Qwen3-VL 的 patch embed（Conv3d，核=步长）换成等价的 F.linear。
原因：torch 2.9 + 4090 上这个 bf16 conv3d 一次要 ~8.6 s，换成矩阵乘法后是毫秒级。数学上完全等价。
用法和 serve_openpi_policy.py 一样，参数原样透传。"""
import logging, math, os, sys
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))                    # scripts/
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))   # 仓库根目录
import serve_openpi_policy as S  # noqa: E402


def fast_patch_embed(model) -> int:
    n = 0
    for m in model.modules():
        conv = getattr(m, "proj", None)
        if not (type(m).__name__.endswith("VisionPatchEmbed") and isinstance(conv, torch.nn.Conv3d)):
            continue
        if tuple(conv.kernel_size) != tuple(conv.stride) or any(conv.padding) or conv.groups != 1:
            raise RuntimeError(f"patch embed 不是“核=步长、无填充”，不能替换：{conv}")
        k = conv.in_channels * math.prod(conv.kernel_size)
        w = conv.weight.view(conv.out_channels, k)

        def fwd(hidden_states, _c=conv, _w=w, _k=k):
            x = hidden_states.reshape(-1, _k).to(_w.dtype)
            return F.linear(x, _w, _c.bias).view(-1, _c.out_channels)

        x = torch.randn(32, k, device=w.device, dtype=w.dtype)
        with torch.no_grad():
            ref, new = m.forward(x).float(), fwd(x).float()
        err = (ref - new).abs().max().item() / (ref.abs().max().item() + 1e-6)
        if err > 2e-2:
            raise RuntimeError(f"替换后输出对不上（相对误差 {err:.3g}），不替换")
        m.forward = fwd
        n += 1
        print(f"[serve_fast] {type(m).__name__}: Conv3d → F.linear，相对误差 {err:.2e}", flush=True)
    if n == 0:
        print("[serve_fast] 没找到 VisionPatchEmbed，未替换", flush=True)
    return n


_orig_init = S.XR1OpenPIPolicy.__init__


def _init(self, *a, **kw):
    _orig_init(self, *a, **kw)
    fast_patch_embed(self.model)


S.XR1OpenPIPolicy.__init__ = _init

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    S.main()