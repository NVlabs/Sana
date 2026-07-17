"""Minimal `decord` drop-in backed by imageio-ffmpeg.

`decord` has no aarch64 (Grace/ARM) wheel, so this shim provides just the API
surface LingBot-Video's refiner uses (utils.load_refiner_video_tensor):
    from decord import VideoReader, cpu
    vr = VideoReader(path, ctx=cpu(0))
    len(vr); vr.get_avg_fps(); vr.get_batch(indices).asnumpy()
Frames are returned as HxWxC uint8 numpy arrays, identical layout to decord.
"""
from __future__ import annotations

import numpy as np
import imageio.v2 as imageio


class _CPUContext:
    def __init__(self, device_id: int = 0):
        self.device_id = device_id


def cpu(device_id: int = 0) -> _CPUContext:
    return _CPUContext(device_id)


def gpu(device_id: int = 0) -> _CPUContext:  # pragma: no cover - not used
    return _CPUContext(device_id)


class _Batch:
    def __init__(self, arr: np.ndarray):
        self._arr = arr

    def asnumpy(self) -> np.ndarray:
        return self._arr


class VideoReader:
    def __init__(self, uri, ctx=None, num_threads: int = 0, **_):
        self._reader = imageio.get_reader(str(uri), "ffmpeg")
        self._meta = self._reader.get_meta_data()
        n = self._meta.get("nframes", None)
        if n is None or n == float("inf") or n <= 0:
            # nframes can be unreliable; count by iterating (videos here are short).
            try:
                n = self._reader.count_frames()
            except Exception:
                n = sum(1 for _ in self._reader)
                self._reader = imageio.get_reader(str(uri), "ffmpeg")
        self._n = int(n)
        # Cache decoded frames; refiner reads the whole clip once.
        self._cache: dict[int, np.ndarray] = {}

    def __len__(self) -> int:
        return self._n

    def get_avg_fps(self) -> float:
        return float(self._meta.get("fps", 24.0))

    def _frame(self, i: int) -> np.ndarray:
        i = int(i)
        if i not in self._cache:
            self._cache[i] = np.asarray(self._reader.get_data(i))
        return self._cache[i]

    def get_batch(self, indices) -> _Batch:
        arr = np.stack([self._frame(int(i)) for i in indices], axis=0)
        return _Batch(arr)
