"""Small helpers: CUDA-event timers and MP4 export (Sol-H3 h3_runtime.encoding, the encoder the Sol-H3 engine ships)."""
from __future__ import annotations

import importlib


def event(torch):
    e = torch.cuda.Event(enable_timing=True)
    e.record()
    return e


def save_mp4(video, audio, sample_rate, path, fps=24):
    """video: [T, C, H, W] float in [0, 1] on the GPU; audio: [channels, samples] or None."""
    try:
        enc = importlib.import_module("h3_runtime.encoding")
    except ImportError:                        # h3_runtime not loaded yet: import it from the Sol-H3 directory
        from . import solh3

        enc = solh3.load(None, modules=("encoding",)).encoding
    job = enc.start_fast_video_encode(video, fps=fps, output_path=str(path), audio=audio, audio_sample_rate=sample_rate,
                                      chunk_frames=8, fragmented=False)
    job.wait()
    return str(path)


def psnr_u8(a8, b8, torch, chunk=8):
    """Mean / min per-frame PSNR (dB) of two uint8 [T, C, H, W] videos."""
    mse = torch.cat([(a8[j:j + chunk].float() - b8[j:j + chunk].float()).pow(2).mean(dim=(1, 2, 3))
                     for j in range(0, a8.shape[0], chunk)])
    p = 10 * torch.log10(255.0 ** 2 / mse.clamp_min(1e-10))
    return float(p.mean()), float(p.min())


def to_u8(video, torch):
    return (video.float().clamp(0, 1) * 255).round().to(torch.uint8)


def ssim_frames(x8, y8, torch, step=4, chunk=8):
    """Per-frame SSIM (Gaussian 11x11, sigma 1.5, C1 = .01^2, C2 = .03^2, RGB channels averaged) on every `step`-th frame of uint8
    [T, C, H, W] videos (valid convolution, as skimage / the original SSIM); the usual definition."""
    import torch.nn.functional as F

    g = torch.exp(-((torch.arange(11, device=x8.device, dtype=torch.float32) - 5) ** 2) / (2 * 1.5 ** 2))
    g = g / g.sum()
    w = (g[:, None] * g[None, :]).expand(3, 1, 11, 11).contiguous()
    idx = list(range(0, x8.shape[0], step))
    out = []
    for j in range(0, len(idx), chunk):
        sel = idx[j:j + chunk]
        x, y = x8[sel].float() / 255.0, y8[sel].float() / 255.0
        mx, my = F.conv2d(x, w, groups=3), F.conv2d(y, w, groups=3)
        sxx = F.conv2d(x * x, w, groups=3) - mx * mx
        syy = F.conv2d(y * y, w, groups=3) - my * my
        sxy = F.conv2d(x * y, w, groups=3) - mx * my
        c1, c2 = 0.01 ** 2, 0.03 ** 2
        m = ((2 * mx * my + c1) * (2 * sxy + c2)) / ((mx * mx + my * my + c1) * (sxx + syy + c2))
        out.append(m.mean(dim=(1, 2, 3)))
    return torch.cat(out)


def load_lpips(pkgs_dir, device):
    """LPIPS (AlexNet) from a directory holding the `lpips` package (AlexNet weights via TORCH_HOME). lpips/__init__ imports its
    trainer, which imports scipy.ndimage.zoom (unused for distances); an import-time stub is used when scipy is absent."""
    import importlib.util
    import sys
    import types

    if pkgs_dir not in sys.path:
        sys.path.append(pkgs_dir)
    if importlib.util.find_spec("scipy") is None:
        sp, nd = types.ModuleType("scipy"), types.ModuleType("scipy.ndimage")
        nd.zoom = None
        sp.ndimage = nd
        sys.modules["scipy"], sys.modules["scipy.ndimage"] = sp, nd
    import lpips

    return lpips.LPIPS(net="alex", verbose=False).to(device).eval()


def lpips_frames(lp, x8, y8, torch, step=8):
    """LPIPS on every `step`-th frame of two uint8 [T, C, H, W] videos."""
    vals = []
    with torch.no_grad():
        for j in range(0, x8.shape[0], step):
            vals.append(float(lp(x8[j:j + 1].float() / 127.5 - 1, y8[j:j + 1].float() / 127.5 - 1)))
    return vals


def rel_mse(a, b):
    """||a - b||^2 / ||b||^2 (b = reference)."""
    return float((a.float() - b.float()).pow(2).sum() / b.float().pow(2).sum().clamp_min(1e-30))
