"""Fast CUDA-to-fragmented-MP4 handoff for resident H3 inference.

The regular diffusers export path first materializes every decoded frame as a
PIL image.  At 768x1344x124 that performs a 1.5 GiB pageable D2H copy and then
millions of Python/PIL object operations.  This module keeps the pipeline output
as a CUDA tensor, quantizes it to RGB8 on the GPU, copies into pinned host memory
in small chunks, and lets PyAV/libx264 consume chunks as soon as they are ready.
"""

from __future__ import annotations

import queue
import threading
import time
from pathlib import Path

import torch


class VideoEncodeError(RuntimeError):
    """An output-only failure which does not poison distributed model state."""


class FastVideoEncodeJob:
    """Background fMP4 encoding job with an explicit GPU-staging boundary."""

    def __init__(
        self,
        video: torch.Tensor,
        *,
        fps: int,
        output_path: str,
        audio: torch.Tensor | None,
        audio_sample_rate: int | None,
        chunk_frames: int,
        fragmented: bool,
        encoder_threads: int,
        pyav_zero_copy: bool,
        overlap_audio: bool,
    ) -> None:
        if video.ndim != 4 or video.shape[1] not in (1, 3, 4):
            raise VideoEncodeError(
                "fast encoder expects a [frames, channels, height, width] tensor; "
                f"got {tuple(video.shape)}"
            )
        if not video.is_cuda:
            raise VideoEncodeError("fast encoder expects the video tensor on CUDA")
        if chunk_frames <= 0:
            raise VideoEncodeError("chunk_frames must be positive")
        if encoder_threads < 0:
            raise VideoEncodeError("encoder_threads must be non-negative")
        if video.shape[1] != 3:
            raise VideoEncodeError("fast encoder currently requires RGB video")
        if audio is not None and audio_sample_rate is None:
            raise VideoEncodeError("audio_sample_rate is required when audio is provided")

        self.output_path = str(output_path)
        self.fragmented = fragmented
        self.fps = int(fps)
        self.chunk_frames = int(chunk_frames)
        self.encoder_threads = int(encoder_threads)
        self.pyav_zero_copy = bool(pyav_zero_copy)
        self.overlap_audio = bool(overlap_audio)
        self.audio_sample_rate = audio_sample_rate
        self.started_at = time.perf_counter()
        self.staged_at: float | None = None
        self.finished_at: float | None = None
        self.error: BaseException | None = None
        self.timings: dict[str, float] = {}
        self._done = threading.Event()
        self._chunks: queue.Queue[tuple[int, int, torch.cuda.Event] | None] = queue.Queue()
        self._video = video

        frames, _, height, width = video.shape
        self.frames = int(frames)
        self.height = int(height)
        self.width = int(width)

        try:
            # One full RGB8 result is 4x smaller than the float32 pipeline output.  Keeping
            # disjoint pinned storage means libx264 can read earlier chunks while CUDA fills
            # later ones, without a buffer-reuse race.
            self._host_video = torch.empty(
                (self.frames, self.height, self.width, 3),
                dtype=torch.uint8,
                pin_memory=True,
            )
            self._audio = None if audio is None else audio.detach().to("cpu")
            self._copy_stream = torch.cuda.Stream(device=video.device)
            self._thread = threading.Thread(
                target=self._encode_worker,
                name=f"h3-fmp4-{Path(self.output_path).stem}",
                daemon=True,
            )
            self._thread.start()

            last_event = None
            with torch.cuda.stream(self._copy_stream):
                for start in range(0, self.frames, self.chunk_frames):
                    end = min(start + self.chunk_frames, self.frames)
                    # This is pixel-equivalent to diffusers' numpy/PIL path:
                    # THWC = round(clamp01(TCHW) * 255), cast to uint8.
                    rgb8 = (
                        video[start:end]
                        .permute(0, 2, 3, 1)
                        .mul(255.0)
                        .round()
                        .to(torch.uint8)
                        .contiguous()
                    )
                    self._host_video[start:end].copy_(rgb8, non_blocking=True)
                    event = torch.cuda.Event()
                    event.record(self._copy_stream)
                    self._chunks.put((start, end, event))
                    last_event = event
            self._chunks.put(None)
            self._staged_event = last_event
        except BaseException as error:
            self._chunks.put(None)
            self.error = error
            self._done.set()
            raise VideoEncodeError(f"failed to start fast video encoder: {error}") from error

    @property
    def staging_s(self) -> float | None:
        return None if self.staged_at is None else self.staged_at - self.started_at

    @property
    def elapsed_s(self) -> float | None:
        return None if self.finished_at is None else self.finished_at - self.started_at

    def wait_staged(self) -> None:
        """Wait only until CUDA no longer owns output work, allowing the next request."""

        if self.staged_at is None:
            if self._staged_event is not None:
                self._staged_event.synchronize()
            self.staged_at = time.perf_counter()
            # All D2H copies are complete. Dropping the source releases the 1.5 GiB fp32
            # decoded tensor before another generation enters the model.
            self._video = None

    def wait(self) -> None:
        self.wait_staged()
        self._done.wait()
        self._thread.join()
        if self.error is not None:
            try:
                Path(self.output_path).unlink(missing_ok=True)
            except OSError:
                pass
            raise VideoEncodeError(f"fast video encoder failed: {self.error}") from self.error

    def _encode_worker(self) -> None:
        container = None
        phase = {
            "container_setup_s": 0.0,
            "cuda_wait_s": 0.0,
            "frame_wrap_s": 0.0,
            "video_submit_s": 0.0,
            "mux_s": 0.0,
            "video_flush_s": 0.0,
            "audio_s": 0.0,
            "audio_wait_s": 0.0,
            "audio_mux_s": 0.0,
            "close_s": 0.0,
        }
        try:
            from diffusers.utils.export_utils import _import_av, _prepare_audio_stream

            av = _import_av()
            options = None
            if self.fragmented:
                options = {
                    "movflags": "empty_moov+default_base_moof+frag_keyframe",
                    # Fragment at the same cadence at which RGB chunks become available.
                    # This changes only container boundaries, not H.264 pixels or quality.
                    "frag_duration": str(round(1_000_000 * self.chunk_frames / self.fps)),
                }
            begin = time.perf_counter()
            container = av.open(self.output_path, mode="w", format="mp4", options=options)
            stream = container.add_stream("libx264", rate=self.fps)
            stream.width = self.width
            stream.height = self.height
            stream.pix_fmt = "yuv420p"
            if self.encoder_threads > 0:
                stream.codec_context.thread_count = self.encoder_threads

            audio_stream = None
            if self._audio is not None:
                audio_stream = _prepare_audio_stream(container, int(self.audio_sample_rate))
            phase["container_setup_s"] = time.perf_counter() - begin

            audio_result: list[tuple[list, float]] = []
            audio_error: list[BaseException] = []

            def encode_audio() -> None:
                try:
                    audio_result.append(
                        self._encode_audio_packets(
                            audio_stream,
                            self._audio,
                            int(self.audio_sample_rate),
                            av,
                        )
                    )
                except BaseException as error:
                    audio_error.append(error)

            audio_thread = None
            if self._audio is not None and self.overlap_audio:
                # Audio and video have independent codec contexts. Encode AAC concurrently,
                # but keep every container.mux call on this worker thread because PyAV's
                # output container is not a concurrent writer.
                audio_thread = threading.Thread(
                    target=encode_audio,
                    name=f"h3-aac-{Path(self.output_path).stem}",
                    daemon=True,
                )
                audio_thread.start()

            while True:
                item = self._chunks.get()
                if item is None:
                    break
                start, end, event = item
                begin = time.perf_counter()
                event.synchronize()
                phase["cuda_wait_s"] += time.perf_counter() - begin
                for frame_array in self._host_video[start:end].numpy():
                    begin = time.perf_counter()
                    if self.pyav_zero_copy:
                        # The AVFrame retains the numpy owner and directly points at the pinned
                        # RGB8 allocation.  `_host_video` lives until this job finishes, so x264
                        # may safely buffer frames without an intermediate per-frame memcpy.
                        frame = av.VideoFrame.from_numpy_buffer(frame_array, format="rgb24")
                    else:
                        frame = av.VideoFrame.from_ndarray(frame_array, format="rgb24")
                    phase["frame_wrap_s"] += time.perf_counter() - begin
                    begin = time.perf_counter()
                    packets = stream.encode(frame)
                    phase["video_submit_s"] += time.perf_counter() - begin
                    for packet in packets:
                        begin = time.perf_counter()
                        container.mux(packet)
                        phase["mux_s"] += time.perf_counter() - begin

            begin = time.perf_counter()
            packets = stream.encode()
            phase["video_flush_s"] = time.perf_counter() - begin
            for packet in packets:
                begin = time.perf_counter()
                container.mux(packet)
                phase["mux_s"] += time.perf_counter() - begin
            if self._audio is not None:
                if audio_thread is None:
                    encode_audio()
                else:
                    begin = time.perf_counter()
                    audio_thread.join()
                    phase["audio_wait_s"] = time.perf_counter() - begin
                if audio_error:
                    raise audio_error[0]
                audio_packets, phase["audio_s"] = audio_result[0]
                begin = time.perf_counter()
                for packet in audio_packets:
                    container.mux(packet)
                phase["audio_mux_s"] = time.perf_counter() - begin
            begin = time.perf_counter()
            container.close()
            phase["close_s"] = time.perf_counter() - begin
            container = None
        except BaseException as error:
            self.error = error
        finally:
            if container is not None:
                try:
                    container.close()
                except BaseException as close_error:
                    if self.error is None:
                        self.error = close_error
            self.finished_at = time.perf_counter()
            self.timings = {
                name: round(seconds, 6) for name, seconds in phase.items()
            }
            self._done.set()

    @staticmethod
    def _encode_audio_packets(audio_stream, samples, sample_rate: int, av_module):
        """Encode AAC without touching the shared output container."""

        started = time.perf_counter()
        if samples.ndim == 1:
            samples = samples[:, None]
        if samples.shape[1] != 2 and samples.shape[0] == 2:
            samples = samples.T
        if samples.shape[1] != 2:
            raise ValueError(f"Expected samples with 2 channels; got shape {samples.shape}.")
        if samples.dtype != torch.int16:
            samples = torch.clip(samples, -1.0, 1.0)
            samples = (samples * 32767.0).to(torch.int16)
        frame = av_module.AudioFrame.from_ndarray(
            samples.contiguous().reshape(1, -1).cpu().numpy(),
            format="s16",
            layout="stereo",
        )
        frame.sample_rate = sample_rate
        codec = audio_stream.codec_context
        resampler = av_module.audio.resampler.AudioResampler(
            format=codec.format or "fltp",
            layout=codec.layout or "stereo",
            rate=codec.sample_rate or sample_rate,
        )
        packets = []
        next_pts = 0
        for converted in resampler.resample(frame):
            if converted.pts is None:
                converted.pts = next_pts
            next_pts += converted.samples
            converted.sample_rate = sample_rate
            packets.extend(audio_stream.encode(converted))
        packets.extend(audio_stream.encode())
        return packets, time.perf_counter() - started

def start_fast_video_encode(
    video: torch.Tensor,
    *,
    fps: int,
    output_path: str,
    audio: torch.Tensor | None = None,
    audio_sample_rate: int | None = None,
    chunk_frames: int = 8,
    fragmented: bool = True,
    encoder_threads: int = 0,
    pyav_zero_copy: bool = False,
    overlap_audio: bool = False,
) -> FastVideoEncodeJob:
    """Start a pinned-memory, chunk-pipelined H.264/fMP4 encode."""

    try:
        return FastVideoEncodeJob(
            video,
            fps=fps,
            output_path=output_path,
            audio=audio,
            audio_sample_rate=audio_sample_rate,
            chunk_frames=chunk_frames,
            fragmented=fragmented,
            encoder_threads=encoder_threads,
            pyav_zero_copy=pyav_zero_copy,
            overlap_audio=overlap_audio,
        )
    except VideoEncodeError:
        raise
    except BaseException as error:
        raise VideoEncodeError(f"failed to initialize fast video encoder: {error}") from error
