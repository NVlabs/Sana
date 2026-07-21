"""Rigorous per-frame video quality vs a baseline: SSIM (windowed Gaussian),
PSNR, and LPIPS(alex). Replaces the coarse global-luma SSIM scalar.

Usage: video_quality_metrics.py <baseline_frames_dir> <opt_dir1> [opt_dir2 ...]
Each dir holds aligned per-frame PNGs (same count/order). Prints mean+/-std over
frames for each opt dir. Run on a GPU node (LPIPS) with the sparse_attn venv.
"""
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

dev = "cuda" if torch.cuda.is_available() else "cpu"


def load_frames(d):
    ps = sorted(Path(d).glob("*.png"))
    return ps


def to_tensor(p):
    a = np.asarray(Image.open(p).convert("RGB")).astype(np.float32)  # HWC 0..255
    return torch.from_numpy(a).permute(2, 0, 1).unsqueeze(0).to(dev)  # 1,3,H,W


def _gauss(ws=11, sigma=1.5):
    c = torch.arange(ws).float() - ws // 2
    g = torch.exp(-(c ** 2) / (2 * sigma ** 2))
    g = (g / g.sum()).to(dev)
    return (g[:, None] * g[None, :])[None, None]  # 1,1,ws,ws


_W = _gauss()


def ssim(x, y):  # x,y: 1,3,H,W in 0..255 -> luma windowed SSIM
    def luma(t):
        return (0.299 * t[:, 0] + 0.587 * t[:, 1] + 0.114 * t[:, 2]).unsqueeze(1)
    x, y = luma(x), luma(y)
    C1, C2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    mux = F.conv2d(x, _W); muy = F.conv2d(y, _W)
    mux2, muy2, muxy = mux * mux, muy * muy, mux * muy
    sx = F.conv2d(x * x, _W) - mux2
    sy = F.conv2d(y * y, _W) - muy2
    sxy = F.conv2d(x * y, _W) - muxy
    s = ((2 * muxy + C1) * (2 * sxy + C2)) / ((mux2 + muy2 + C1) * (sx + sy + C2))
    return s.mean().item()


def psnr(x, y):
    mse = ((x - y) ** 2).mean().item()
    return 99.0 if mse < 1e-9 else 10 * np.log10(255.0 ** 2 / mse)


def main():
    base = sys.argv[1]
    opts = sys.argv[2:]
    bframes = load_frames(base)
    lp = None
    try:
        import lpips
        lp = lpips.LPIPS(net="alex", verbose=False).to(dev).eval()
    except Exception as e:
        print(f"[warn] LPIPS unavailable: {e}")
    print(f"baseline: {base} ({len(bframes)} frames)")
    for od in opts:
        oframes = load_frames(od)
        n = min(len(bframes), len(oframes))
        ss, pp, ll = [], [], []
        for i in range(n):
            xb, xo = to_tensor(bframes[i]), to_tensor(oframes[i])
            ss.append(ssim(xb, xo)); pp.append(psnr(xb, xo))
            if lp is not None:
                with torch.no_grad():
                    ll.append(lp(xb / 127.5 - 1, xo / 127.5 - 1).item())
        # label by the run dir (…/<run>/outputs/frames), not the literal "frames"
        po = Path(od)
        name = po.parents[1].name if po.name == "frames" and len(po.parents) >= 2 else po.name
        lpm = f"LPIPS={np.mean(ll):.4f}+/-{np.std(ll):.4f}" if ll else "LPIPS=n/a"
        print(f"  {name:42s} SSIM={np.mean(ss):.4f}+/-{np.std(ss):.4f}  "
              f"PSNR={np.mean(pp):.2f}dB  {lpm}  (n={n})")


if __name__ == "__main__":
    main()
