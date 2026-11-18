"""Draft frames in, refined video out.

read_draft takes 124 frames picked at 24 fps (frame i of the source is int(i * fps / 24)),
at least 113 real frames (shorter clips are refused), the tail repeats the last real frame; each frame is
cover-resized (bilinear; antialiased when shrinking) and center-cropped to the LQ canvas.
"""
import shutil
import subprocess

import numpy as np
import torch
import torch.nn.functional as F

FRAMES = 124
MIN_REAL = 113


def _cover(x, height, width):
    h, w = x.shape[-2:]
    if (h, w) == (height, width):
        return x
    scale = max(height / h, width / w)
    sh, sw = max(height, round(h * scale)), max(width, round(w * scale))
    x = F.interpolate(x, size=(sh, sw), mode='bilinear', align_corners=False, antialias=scale < 1)
    i, j = (sh - height) // 2, (sw - width) // 2
    return x[..., i:i + height, j:j + width]


def read_draft(path, height, width):
    """-> float [124, 3, height, width] in [0, 1] (CPU), number of real frames, source fps."""
    import av
    out = torch.empty((FRAMES, 3, height, width), dtype=torch.float32)
    valid = 0
    with av.open(str(path)) as video:
        stream = video.streams.video[0]
        stream.thread_type = 'AUTO'
        fps = float(stream.average_rate or 24.)
        wanted = [int(i * fps / 24.) for i in range(FRAMES)]
        for index, frame in enumerate(video.decode(stream)):
            if index < wanted[valid]:
                continue
            x = torch.from_numpy(frame.to_ndarray(format='rgb24')).permute(2, 0, 1)[None].float() / 255.
            x = _cover(x, height, width)
            while valid < FRAMES and wanted[valid] == index:
                out[valid] = x[0]
                valid += 1
            if valid == FRAMES:
                break
    if valid < MIN_REAL:
        raise ValueError(f'{path}: {valid} frames at 24 fps; the model needs at least {MIN_REAL} (about 4.7 s)')
    out[valid:] = out[valid - 1]
    return out.clamp(0, 1), valid, fps


def write_video(frames, path, audio_from=None, crf=12):
    """frames: uint8 [T, H, W, 3]; H.264 at 24 fps, the first audio stream of `audio_from` muxed in (cut to the
    video), when it has one."""
    ffmpeg = shutil.which('ffmpeg')
    if ffmpeg is None:
        import imageio_ffmpeg
        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    t, h, w, _ = frames.shape
    cmd = [ffmpeg, '-nostdin', '-y', '-v', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{w}x{h}', '-r', '24',
           '-i', '-']
    if audio_from is not None:
        cmd += ['-i', str(audio_from), '-map', '0:v:0', '-map', '1:a:0?', '-c:a', 'aac', '-b:a', '192k', '-shortest']
    cmd += ['-c:v', 'libx264', '-preset', 'slow', '-crf', str(crf), '-pix_fmt', 'yuv420p', '-movflags', '+faststart',
            str(path)]
    subprocess.run(cmd, input=np.ascontiguousarray(frames).tobytes(), check=True)
