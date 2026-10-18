"""Read uploaded semantic video in native replay order, one batch at a time."""

import io

import av
import numpy as np
from PIL import Image

MAX_VIDEO_BYTES = 256 * 1024 * 1024
MAX_FRAMES = 14400


class SemanticVideo:
    """Decode bounded, ordered batches using the native replay's RGB conversion.

    PyAV presentation order and Pillow nearest-neighbor resizing match the
    upstream replay client. Audio is ignored. Frames are never interpolated.
    """

    def __init__(self, data: bytes, max_frames: int = 0):
        self.container = None
        if not data or len(data) > MAX_VIDEO_BYTES:
            raise ValueError("Semantic video must contain data and be at most 256 MiB")
        if not 0 <= max_frames <= MAX_FRAMES:
            raise ValueError("Frame limit must be between 0 and 14400")
        self.limit = max_frames or MAX_FRAMES
        self.decoded = 0
        self.tail_frames = 0
        try:
            self.container = av.open(io.BytesIO(data))
            if self.container.format.name not in (
                "mov,mp4,m4a,3gp,3g2,mj2",
                "matroska,webm",
            ):
                raise ValueError("Use a self-contained MP4, MOV, MKV or WebM video")
            if not self.container.streams.video:
                raise ValueError("The upload has no video track")
            stream = self.container.streams.video[0]
            if not 0 < stream.width * stream.height <= 8_294_400:
                raise ValueError("Semantic video exceeds the 4K pixel budget")
            if (
                stream.average_rate is None
                or abs(float(stream.average_rate) - 24) > 0.01
            ):
                raise ValueError(
                    "Provide a 24 FPS semantic video; automatic retiming is disabled"
                )
            if not max_frames and stream.frames > MAX_FRAMES:
                raise ValueError(
                    "Video exceeds 10 minutes; select an explicit frame limit"
                )
            self.total_frames = (
                min(stream.frames, self.limit) if stream.frames else None
            )
            self.frames = iter(self.container.decode(video=0))
            self.first = self._decode_one()
            if self.first is None:
                raise ValueError("The video has no decodable frames")
        except (ValueError, av.FFmpegError, OSError) as error:
            self.close()
            raise ValueError(f"Invalid semantic video: {error}") from error

    def _decode_one(self) -> np.ndarray | None:
        if self.decoded >= self.limit:
            return None
        frame = next(self.frames, None)
        if frame is None:
            return None
        if frame.width * frame.height > 8_294_400:
            raise ValueError("Decoded frame exceeds the 4K pixel budget")
        image = frame.to_image().resize((1344, 768), Image.Resampling.NEAREST)
        self.decoded += 1
        return np.asarray(image, dtype=np.uint8)

    def read(self, count: int) -> np.ndarray | None:
        """Return a full native batch, or discard an incomplete scheduling tail."""
        if not 1 <= count <= 48:
            raise ValueError("Unexpected native semantic demand")
        batch = []
        for _ in range(count):
            frame = self.first
            self.first = None
            if frame is None:
                frame = self._decode_one()
            if frame is None:
                self.tail_frames = len(batch)
                self.close()
                return None
            batch.append(frame)
        return np.stack(batch)

    def close(self) -> None:
        """Release the decoder and compressed file when the run ends."""
        if self.container is not None:
            self.container.close()
            self.container = None
