# Copyright (c) 2026 Bytedance Ltd. and/or its affiliate
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Write generated frames to disk as H.264 mp4 (or png for a single frame)."""

import imageio
import numpy as np
from PIL import Image


def _imageio_mimwrite_h264(frames_uint8, save_path, fps=16, quality=10, crf=8):
    """Write uint8 RGB frames as an H.264 mp4.

    libx264 + yuv420p keeps the output broadly playable; ``-crf`` controls the
    quality-rate trade-off (lower is higher quality). This avoids the tiny,
    blurry output of cv2's legacy ``mp4v`` writer.
    """
    imageio.mimwrite(
        save_path,
        frames_uint8,
        fps=fps,
        codec="libx264",
        quality=quality,
        macro_block_size=1,
        output_params=["-pix_fmt", "yuv420p", "-crf", str(crf)],
    )


def export_to_video(video_frames, output_video_path, fps=16, quality=10, crf=8):
    """Save a list of PIL images / ndarrays as an mp4."""
    frames = []
    for f in video_frames:
        if isinstance(f, Image.Image):
            frames.append(np.asarray(f))
        elif isinstance(f, np.ndarray):
            if f.dtype != np.uint8:
                f = (np.clip(f, 0.0, 1.0) * 255).astype(np.uint8)
            frames.append(f)
        else:
            raise TypeError(f"export_to_video: unsupported frame type {type(f)}")
    _imageio_mimwrite_h264(frames, output_video_path, fps=fps, quality=quality, crf=crf)
    return output_video_path


def save_output(output: np.ndarray, save_path: str, fps: int = 16):
    """Save a decoded clip `[T, H, W, C]` in [0, 1] as mp4, or png if T == 1."""
    if output.shape[0] == 1:
        imageio.imwrite(save_path, (np.clip(output[0], 0.0, 1.0) * 255).astype(np.uint8))
    else:
        export_to_video(output, save_path, fps=fps)
    return save_path
