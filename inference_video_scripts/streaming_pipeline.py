"""Chunk-pipelined orchestrator for streaming Sana-WM inference.

Drives three CUDA streams (stage-1 DiT, LTX-2 refiner, causal LTX-2 VAE) so
one chunk is in flight per stage. Every AR chunk produces one decoded video
chunk, which is written progressively into an MP4 via :class:`StreamingMp4Writer`.

Cadence (canonical recipe ``distilled-4step + source-sink-1``):

* Stage-1 chunks of ``base_chunk_frames=3`` latents (chunk 0 absorbs the
  ``8k+1`` stride remainder and covers ``[0, 4)``).
* Refiner blocks of ``block_size=3`` latents; the sink at frame 0 is captured
  as the attention anchor but never refined.
* Decode chunks of ``block_size`` latents, plus the sink on chunk 0. The sink
  pixel frame is dropped on the way to ffmpeg.

The pipeline is 1:1 between stages: refiner block ``k`` depends on stage-1
chunk ``k``; decode chunk ``k`` depends on refiner block ``k``.
"""

from __future__ import annotations

from collections import deque
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator
import os
import time

import numpy as np
import torch

from diffusion.model.ltx2 import CausalVaeStreamingDecoder
from diffusion.refiner.diffusers_ltx2_refiner import RefinerChunkRunner
from inference_video_scripts.streaming_mp4_writer import StreamingMp4Writer


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _pixel_chunk_to_cpu_uint8(pixel_chunk: torch.Tensor) -> torch.Tensor:
    """Convert a decoded pixel chunk ``[-1, 1]`` to a CPU uint8 ``(B,T,H,W,3)``.

    The float->uint8 conversion and the channel-last permute/contiguous run on
    the GPU (the decode stream); the device->host copy is the LAST op and is
    issued ``non_blocking`` so it overlaps with the next chunk. The returned CPU
    tensor must therefore only be read after the chunk's decode event has been
    waited on (``_emit_ready`` does this via the ``pending`` queue).

    The earlier "copy bf16 to CPU first, convert on CPU" variant was a
    correctness bug: chaining a CPU ``.contiguous()``/``.float()`` straight onto
    the ``non_blocking`` D->H copy reads the host buffer before the transfer has
    landed, so partially-transferred garbage leaks into early chunks as
    grey/colored horizontal bands. Converting on-device keeps every host-side
    read behind the decode event. The on-GPU uint8 temporary is tiny
    (~one chunk: a few tens of MB) so it is safe even on the 32GB 5090.
    """
    pixel_uint8 = ((pixel_chunk + 1.0) * 127.5).clamp(0, 255).to(torch.uint8)
    return pixel_uint8.permute(0, 2, 3, 4, 1).contiguous().to("cpu", non_blocking=True)


@dataclass
class StreamingPipelineConfig:
    """Static settings for one streaming-inference call."""

    sink_size: int = 1
    block_size: int = 3
    fps: int = 16
    output_path: str | Path = "streaming_output.mp4"
    mp4_crf: int = 18
    mp4_preset: str = "medium"
    mp4_encoder: str = "libx264"
    drop_first_pixel: bool = True
    output_mode: str = "mp4"
    profile_cuda: bool = False
    sample_frames_path: str | Path | None = None
    sample_frame_stride: int = 0
    lazy_vae_decoder: bool = False
    sequential_offload: bool = False
    stage1_done_callback: Callable[[], None] | None = None
    stage1_chunk_ends: tuple[int, ...] | None = None
    decoded_chunk_callback: Callable[[np.ndarray, int, int], None] | None = None
    progress_callback: Callable[[dict[str, object]], None] | None = None


@dataclass
class StreamingPipelineResult:
    output_path: Path | None
    n_pixel_frames: int
    n_refiner_blocks: int
    n_decode_chunks: int
    output_mode: str
    wall_seconds: float
    first_chunk_seconds: float | None
    first_chunk_frames: int | None
    steady_state_seconds: float | None
    steady_state_frames_per_second: float | None
    steady_state_realtime_factor: float | None
    frames_per_second: float
    realtime_factor: float
    stage1_cuda_seconds: float | None = None
    refiner_cuda_seconds: float | None = None
    decode_cuda_seconds: float | None = None
    sample_frames_path: Path | None = None
    sampled_frame_count: int = 0
    sampled_frame_indices: list[int] | None = None


@torch.inference_mode()
def run_streaming_inference(
    *,
    stage1_chunk_iter: Iterator[tuple[int, torch.Tensor, int, int]],
    n_stage1_chunks: int,
    z_init: torch.Tensor,
    refiner_runner: RefinerChunkRunner,
    vae_streaming_decoder: CausalVaeStreamingDecoder,
    pixel_h: int,
    pixel_w: int,
    config: StreamingPipelineConfig,
    logger=None,
) -> StreamingPipelineResult:
    """Drive the three-stage chunked pipeline end-to-end.

    Args:
        stage1_chunk_iter: Iterator (typically from
            ``SelfForcingFlowEulerCamCtrl.sample_chunks``) yielding
            ``(chunk_idx, latent_view, start_f, end_f)`` after each AR chunk.
            ``latent_view`` is a view into the sampler's in-place latents
            tensor; the orchestrator defensively mirror-copies it so the
            refiner never races with subsequent stage-1 writes.
        n_stage1_chunks: Number of chunks the iterator will yield.
        z_init: ``(B, C, T_latent, H_lat, W_lat)`` initial latent tensor with
            sink populated at ``[:, :, :sink_size]``.
        refiner_runner: A :class:`RefinerChunkRunner` already configured with
            the prompt, sigmas, fps and seed.
        vae_streaming_decoder: A :class:`CausalVaeStreamingDecoder` wrapping
            ``AutoencoderKLCausalLTX2Video``; reset before the first call.
        pixel_h, pixel_w: Decoded frame dimensions (``vae_stride * H_lat``).
        config: Static configuration.
        logger: Optional logger; falls back to ``print``.

    Returns:
        :class:`StreamingPipelineResult` describing the produced MP4.
    """
    log = (logger.info if logger is not None else print)

    sink_size = int(config.sink_size)
    block_size = int(config.block_size)
    T_latent = int(z_init.shape[2])

    n_active = max(T_latent - sink_size, 0)
    n_refiner = n_active // block_size
    if n_refiner * block_size != n_active:
        raise ValueError(
            f"Active latent frames ({n_active}) must be divisible by "
            f"block_size ({block_size}); got remainder {n_active % block_size}."
        )
    if n_stage1_chunks <= 0:
        raise ValueError("n_stage1_chunks must be > 0.")
    n_decode = n_refiner
    stage1_chunk_ends = tuple(int(v) for v in config.stage1_chunk_ends or ())
    if stage1_chunk_ends:
        if len(stage1_chunk_ends) != n_stage1_chunks:
            raise ValueError(
                f"stage1_chunk_ends has {len(stage1_chunk_ends)} entries, "
                f"but n_stage1_chunks={n_stage1_chunks}."
            )
    elif n_stage1_chunks == n_refiner:
        stage1_chunk_ends = tuple(sink_size + (i + 1) * block_size for i in range(n_stage1_chunks))
    else:
        raise ValueError("stage1_chunk_ends is required when refiner block size differs from stage-1 chunking.")

    refiner_ready_stage_idx: list[int] = []
    next_stage_idx = 0
    for k_ref in range(n_refiner):
        block_end = sink_size + (k_ref + 1) * block_size
        while next_stage_idx < len(stage1_chunk_ends) and stage1_chunk_ends[next_stage_idx] < block_end:
            next_stage_idx += 1
        if next_stage_idx >= len(stage1_chunk_ends):
            raise ValueError(
                f"Refiner block {k_ref} ends at latent frame {block_end}, "
                "but no Stage-1 chunk produces that frame."
            )
        refiner_ready_stage_idx.append(next_stage_idx)

    output_mode = str(config.output_mode)
    if output_mode not in {"mp4", "cpu", "discard"}:
        raise ValueError(f"output_mode must be one of mp4/cpu/discard; got {output_mode!r}.")

    log(
        f"[stream] T_latent={T_latent} sink={sink_size} block={block_size} "
        f"stage1_chunks={n_stage1_chunks} refiner_blocks={n_refiner} "
        f"decode_chunks={n_decode} output_mode={output_mode}"
    )
    if config.progress_callback is not None:
        config.progress_callback(
            {
                "phase": "stream_start",
                "message": "streaming pipeline started",
                "stage1_done": 0,
                "stage1_total": n_stage1_chunks,
                "refiner_done": 0,
                "refiner_total": n_refiner,
                "decode_done": 0,
                "decode_total": n_decode,
                "frames": 0,
                "output_mode": output_mode,
            }
        )
    if bool(config.sequential_offload):
        return _run_streaming_inference_sequential(
            stage1_chunk_iter=stage1_chunk_iter,
            n_stage1_chunks=n_stage1_chunks,
            z_init=z_init,
            refiner_runner=refiner_runner,
            vae_streaming_decoder=vae_streaming_decoder,
            pixel_h=pixel_h,
            pixel_w=pixel_w,
            config=config,
            logger=logger,
        )

    # Pre-allocated mirror buffers. Slices into these are handed across
    # streams; the storage is long-lived so no record_stream is needed.
    latents_full = torch.empty_like(z_init)
    latents_full[:, :, :sink_size] = z_init[:, :, :sink_size]
    refined_full = torch.empty_like(z_init)
    refined_full[:, :, :sink_size] = z_init[:, :, :sink_size]
    predecode_sink = _env_flag("SANA_WM_STREAMING_PREDECODE_SINK") and sink_size > 0
    direct_refined_blocks = _env_flag("SANA_WM_STREAMING_DIRECT_REFINED_BLOCKS") and predecode_sink
    refined_blocks: list[torch.Tensor | None] | None = [None] * n_refiner if direct_refined_blocks else None

    device = z_init.device
    on_cuda = device.type == "cuda"
    stage1_stream = torch.cuda.Stream(device=device) if on_cuda else None
    refiner_stream = torch.cuda.Stream(device=device) if on_cuda else None
    decode_stream = torch.cuda.Stream(device=device) if on_cuda else None

    # The mirror buffers and their sink frames above were populated on the
    # current (default) stream. The worker streams read those sink frames
    # cold-start before any inter-stream event is recorded (refiner reads
    # ``latents_full[:sink]`` as the block-0 seed; decode reads
    # ``refined_full[:sink]`` for chunk 0 / the predecode), so each worker
    # stream must wait for the setup writes or it races the populate and
    # decodes garbage into the first chunk (flaky green/teal banding).
    if on_cuda:
        _setup_stream = torch.cuda.current_stream(device)
        for _s in (stage1_stream, refiner_stream, decode_stream):
            _s.wait_stream(_setup_stream)

    def _on(stream):
        return torch.cuda.stream(stream) if stream is not None else nullcontext()

    def _new_event():
        return torch.cuda.Event() if device.type == "cuda" else None

    def _new_timing_event():
        if device.type != "cuda" or not bool(config.profile_cuda):
            return None
        return torch.cuda.Event(enable_timing=True)

    def _record(event, stream):
        if event is not None and stream is not None:
            event.record(stream)

    def _wait(stream, event):
        if stream is not None and event is not None:
            stream.wait_event(event)

    mark_cudagraph_step = _env_flag("SANA_WM_CUDAGRAPH_MARK_STEP")

    def _mark_step_begin():
        if not mark_cudagraph_step:
            return
        mark_fn = getattr(getattr(torch, "compiler", None), "cudagraph_mark_step_begin", None)
        if mark_fn is not None:
            mark_fn()

    stage1_events = [_new_event() for _ in range(n_stage1_chunks)]
    refiner_events = [_new_event() for _ in range(n_refiner)]
    decode_events = [_new_event() for _ in range(n_decode)]
    stage1_timing_events: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []
    refiner_timing_events: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []
    decode_timing_events: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []

    vae_streaming_decoder.reset()
    decoder_on_device = True
    if bool(config.lazy_vae_decoder) and device.type == "cuda":
        decoder = getattr(vae_streaming_decoder.vae, "decoder", None)
        if decoder is not None:
            decoder.to("cpu")
            torch.cuda.empty_cache()
            decoder_on_device = False
    predecoded_sink = False

    pending: deque[tuple[torch.cuda.Event | None, torch.Tensor | int, int]] = deque()
    writer = None

    def _get_writer() -> StreamingMp4Writer:
        nonlocal writer
        if writer is None:
            writer = StreamingMp4Writer(
                config.output_path,
                height=int(pixel_h),
                width=int(pixel_w),
                fps=int(config.fps),
                crf=int(config.mp4_crf),
                preset=str(config.mp4_preset),
                encoder=str(config.mp4_encoder),
            )
        return writer

    n_pixel_frames = 0
    sample_frames: list[np.ndarray] = []
    sample_frame_indices: list[int] = []
    sample_frames_path = Path(config.sample_frames_path) if config.sample_frames_path is not None else None
    sample_frame_stride = int(config.sample_frame_stride)
    if sample_frames_path is not None and sample_frame_stride <= 0:
        raise ValueError("sample_frame_stride must be > 0 when sample_frames_path is set.")

    def _collect_sample_frames(pixel_np: np.ndarray, frame_base: int) -> None:
        if sample_frames_path is None:
            return
        for local_idx in range(0, int(pixel_np.shape[0])):
            frame_idx = int(frame_base + local_idx)
            if frame_idx % sample_frame_stride == 0:
                sample_frames.append(pixel_np[local_idx].copy())
                sample_frame_indices.append(frame_idx)

    def _sample_discard_frames(pixel_chunk: torch.Tensor, frame_base: int, drop_first: bool) -> None:
        if sample_frames_path is None:
            return
        drop = 1 if drop_first else 0
        n_out = max(0, int(pixel_chunk.shape[2]) - drop)
        local_indices = [i for i in range(n_out) if (frame_base + i) % sample_frame_stride == 0]
        if not local_indices:
            return
        src_indices = torch.as_tensor(
            [drop + i for i in local_indices],
            device=pixel_chunk.device,
            dtype=torch.long,
        )
        selected = pixel_chunk.index_select(2, src_indices)
        pixel_uint8 = (selected.float() * 127.5 + 127.5).clamp(0, 255).to(torch.uint8)
        pixel_np = pixel_uint8.permute(0, 2, 3, 4, 1).contiguous().to("cpu").numpy()[0]  # blocking: .numpy() reads immediately
        for frame, local_idx in zip(pixel_np, local_indices, strict=True):
            sample_frames.append(frame.copy())
            sample_frame_indices.append(int(frame_base + local_idx))

    t_start: float | None = None
    first_chunk_seconds: float | None = None
    first_chunk_frames: int | None = None
    try:
        if predecode_sink:
            # Seed the causal VAE decoder cache with the non-output sink latent.
            # This is equivalent to decoding [sink + block0] and dropping the
            # sink pixel, but lets the timed stream decode only output frames.
            decoder = getattr(vae_streaming_decoder.vae, "decoder", None)
            if decoder is not None and not decoder_on_device:
                decoder.to(device)
                decoder_on_device = True
            with _on(decode_stream):
                _mark_step_begin()
                _ = vae_streaming_decoder.decode_chunk(refined_full[:, :, :sink_size])
            if decode_stream is not None:
                decode_stream.synchronize()
            predecoded_sink = True
            if bool(config.lazy_vae_decoder) and device.type == "cuda" and decoder is not None:
                decoder.to("cpu")
                torch.cuda.empty_cache()
                decoder_on_device = False

        t_start = time.perf_counter()
        if _env_flag("SANA_WM_REFINER_PRECAPTURE_SINK") and sink_size > 0:
            with _on(refiner_stream):
                timing_start = _new_timing_event()
                timing_end = _new_timing_event()
                _record(timing_start, refiner_stream)
                _mark_step_begin()
                refiner_runner.pre_capture_sink(latents_full[:, :, :sink_size])
                _record(timing_end, refiner_stream)
                if timing_start is not None and timing_end is not None:
                    refiner_timing_events.append((timing_start, timing_end))

        # Schedule per timestep:
        #   t=0:  stage1[0]
        #   t=1:  stage1[1], refiner[0]
        #   t=2:  stage1[2], refiner[1], decode[0]
        #   ...
        t = 0
        next_ref = 0
        next_dec = 0
        refiner_first = _env_flag("SANA_WM_STREAMING_REFINER_FIRST")
        decode_current = _env_flag("SANA_WM_STREAMING_DECODE_CURRENT")

        def _try_launch_refiner(max_ready_stage_idx: int) -> bool:
            nonlocal next_ref
            if next_ref >= n_refiner or refiner_ready_stage_idx[next_ref] > max_ready_stage_idx:
                return False

            k_ref = next_ref
            _wait(refiner_stream, stage1_events[refiner_ready_stage_idx[k_ref]])
            block_start = sink_size + k_ref * block_size
            block_end = block_start + block_size
            with _on(refiner_stream):
                timing_start = _new_timing_event()
                timing_end = _new_timing_event()
                _record(timing_start, refiner_stream)
                clean_block = latents_full[:, :, block_start:block_end]
                sink_seed = latents_full[:, :, :sink_size] if k_ref == 0 else None
                _mark_step_begin()
                refined_block = refiner_runner.refine_block(
                    block_idx=k_ref,
                    clean_block=clean_block,
                    block_start=block_start,
                    block_end=block_end,
                    sink_seed_frames=sink_seed,
                )
                if refined_blocks is not None:
                    refined_blocks[k_ref] = refined_block
                else:
                    refined_full[:, :, block_start:block_end].copy_(refined_block, non_blocking=True)
                _record(timing_end, refiner_stream)
                if timing_start is not None and timing_end is not None:
                    refiner_timing_events.append((timing_start, timing_end))
            _record(refiner_events[k_ref], refiner_stream)
            next_ref += 1
            if config.progress_callback is not None:
                config.progress_callback(
                    {
                        "phase": "refiner",
                        "message": f"refiner block {next_ref}/{n_refiner}",
                        "refiner_done": next_ref,
                        "refiner_total": n_refiner,
                        "decode_done": next_dec,
                        "decode_total": n_decode,
                        "frames": n_pixel_frames,
                        "output_mode": output_mode,
                    }
                )
            return True

        def _try_launch_decode(max_refiner_idx_exclusive: int) -> bool:
            nonlocal decoder_on_device, next_dec
            if next_dec >= n_decode or next_dec >= max_refiner_idx_exclusive:
                return False

            k_dec = next_dec
            _wait(decode_stream, refiner_events[k_dec])
            if refined_blocks is not None:
                z_slice = refined_blocks[k_dec]
                if z_slice is None:
                    raise RuntimeError(f"Refined block {k_dec} was not produced before decode launch.")
            elif k_dec == 0 and predecoded_sink:
                z_slice = refined_full[:, :, sink_size : sink_size + block_size]
            elif k_dec == 0:
                z_slice = refined_full[:, :, : sink_size + block_size]
            else:
                z_slice = refined_full[
                    :, :, sink_size + k_dec * block_size : sink_size + (k_dec + 1) * block_size
                ]
            with _on(decode_stream):
                timing_start = _new_timing_event()
                timing_end = _new_timing_event()
                _record(timing_start, decode_stream)
                if not decoder_on_device:
                    decoder = getattr(vae_streaming_decoder.vae, "decoder", None)
                    if decoder is not None:
                        decoder.to(device)
                        decoder_on_device = True
                _mark_step_begin()
                pixel_chunk = vae_streaming_decoder.decode_chunk(z_slice)
                if output_mode == "discard":
                    n_frames = int(pixel_chunk.shape[2])
                    drop_first = bool(k_dec == 0 and config.drop_first_pixel and not predecoded_sink)
                    if drop_first:
                        n_frames -= 1
                    _sample_discard_frames(pixel_chunk, n_pixel_frames, drop_first)
                    pixel_out = max(0, n_frames)
                else:
                    pixel_out = _pixel_chunk_to_cpu_uint8(pixel_chunk)
                _record(timing_end, decode_stream)
                if timing_start is not None and timing_end is not None:
                    decode_timing_events.append((timing_start, timing_end))
            _record(decode_events[k_dec], decode_stream)
            pending.append((decode_events[k_dec], pixel_out, k_dec))
            next_dec += 1
            return True

        # --- Output invariant: a streamed video must never contain an all-uniform
        # (blank/gray) frame. Such a frame can only come from a transient decode
        # glitch under sustained multi-stream load; carry the previous good frame
        # forward so the defect never reaches the MP4 or the live callback.
        _last_good_frame: list[np.ndarray | None] = [None]

        def _guard_blank_frames(frames_np: np.ndarray) -> np.ndarray:
            if frames_np.shape[0] == 0:
                return frames_np
            flat = frames_np.reshape(frames_np.shape[0], -1)
            uniform = flat.max(axis=1) == flat.min(axis=1)  # per-frame all-pixels-equal
            if not uniform.any():
                _last_good_frame[0] = frames_np[-1]
                return frames_np
            out = frames_np.copy()
            replaced = 0
            for i in range(out.shape[0]):
                if uniform[i] and _last_good_frame[0] is not None:
                    out[i] = _last_good_frame[0]
                    replaced += 1
                elif not uniform[i]:
                    _last_good_frame[0] = out[i]
            if replaced:
                log(f"[stream] guarded {replaced} blank decoded frame(s) (carried previous frame forward)")
            return out

        def _emit_ready(_pixel_out, _k: int) -> None:
            """Emit one decoded chunk to the MP4 writer / callback (single source
            of truth for the in-loop flush and the final drain)."""
            nonlocal n_pixel_frames, first_chunk_frames, first_chunk_seconds
            if first_chunk_seconds is None:
                assert t_start is not None
                first_chunk_seconds = time.perf_counter() - t_start
            if output_mode == "discard":
                n_frames = int(_pixel_out)
            else:
                pixel_np = _pixel_out.numpy()[0]
                if _k == 0 and config.drop_first_pixel and not predecoded_sink:
                    pixel_np = pixel_np[1:]
                pixel_np = _guard_blank_frames(pixel_np)
                n_frames = int(pixel_np.shape[0])
                _collect_sample_frames(pixel_np, n_pixel_frames)
                if config.decoded_chunk_callback is not None and n_frames > 0:
                    config.decoded_chunk_callback(pixel_np, n_pixel_frames, int(_k))
                if output_mode == "mp4":
                    _get_writer().write_chunk(pixel_np)
            n_pixel_frames += n_frames
            if first_chunk_frames is None:
                first_chunk_frames = n_frames
            if config.progress_callback is not None:
                config.progress_callback(
                    {
                        "phase": "decode",
                        "message": f"decoded chunk {int(_k) + 1}/{n_decode}",
                        "decode_done": int(_k) + 1,
                        "decode_total": n_decode,
                        "frames": n_pixel_frames,
                        "output_mode": output_mode,
                    }
                )

        while t < n_stage1_chunks or next_ref < n_refiner or next_dec < n_decode:
            refiner_launched_before = next_ref
            if refiner_first:
                _try_launch_refiner(t - 1)

            # --- stage-1 chunk t ---
            if t < n_stage1_chunks:
                if config.progress_callback is not None:
                    config.progress_callback(
                        {
                            "phase": "stage1_running",
                            "message": f"stage-1 chunk {t + 1}/{n_stage1_chunks} running",
                            "stage1_done": t,
                            "stage1_total": n_stage1_chunks,
                            "refiner_done": next_ref,
                            "refiner_total": n_refiner,
                            "decode_done": next_dec,
                            "decode_total": n_decode,
                            "frames": n_pixel_frames,
                            "output_mode": output_mode,
                        }
                    )
                with _on(stage1_stream):
                    timing_start = _new_timing_event()
                    timing_end = _new_timing_event()
                    _record(timing_start, stage1_stream)
                    _, latent_view, start_f, end_f = next(stage1_chunk_iter)
                    latents_full[:, :, start_f:end_f].copy_(latent_view, non_blocking=True)
                    _record(timing_end, stage1_stream)
                    if timing_start is not None and timing_end is not None:
                        stage1_timing_events.append((timing_start, timing_end))
                _record(stage1_events[t], stage1_stream)
                if config.progress_callback is not None:
                    config.progress_callback(
                        {
                            "phase": "stage1",
                            "message": f"stage-1 chunk {t + 1}/{n_stage1_chunks}",
                            "stage1_done": t + 1,
                            "stage1_total": n_stage1_chunks,
                            "refiner_done": next_ref,
                            "refiner_total": n_refiner,
                            "decode_done": next_dec,
                            "decode_total": n_decode,
                            "frames": n_pixel_frames,
                            "output_mode": output_mode,
                        }
                    )

            # --- refiner block t - 1 ---
            if not refiner_first or refiner_launched_before == next_ref:
                _try_launch_refiner(min(t, n_stage1_chunks - 1))

            # --- decode chunk t - 2 ---
            decode_ready_ref = next_ref if decode_current else refiner_launched_before
            _try_launch_decode(decode_ready_ref)

            # --- Flush any ready decoded chunks. In discard mode this only
            # retires the CUDA work; no CPU copy or encoder path is exercised.
            while pending and (pending[0][0] is None or pending[0][0].query()):
                _event, _pixel_out, _k = pending.popleft()
                _emit_ready(_pixel_out, _k)
            t += 1

        # Drain.
        while pending:
            _event, _pixel_out, _k = pending.popleft()
            if _event is not None:
                _event.synchronize()
            _emit_ready(_pixel_out, _k)

        if config.progress_callback is not None:
            config.progress_callback(
                {
                    "phase": "finalize",
                    "message": "finalizing MP4" if output_mode == "mp4" else "finalizing preview stream",
                    "decode_done": n_decode,
                    "decode_total": n_decode,
                    "frames": n_pixel_frames,
                    "output_mode": output_mode,
                }
            )
        out_path = writer.close() if writer is not None else None
        if sample_frames_path is not None:
            sample_frames_path.parent.mkdir(parents=True, exist_ok=True)
            frames = (
                np.stack(sample_frames, axis=0)
                if sample_frames
                else np.empty((0, int(pixel_h), int(pixel_w), 3), dtype=np.uint8)
            )
            np.savez_compressed(
                sample_frames_path,
                frames=frames,
                frame_indices=np.asarray(sample_frame_indices, dtype=np.int64),
            )
    except Exception:
        if writer is not None:
            writer.close()
        raise

    assert t_start is not None
    wall_seconds = time.perf_counter() - t_start
    frames_per_second = float(n_pixel_frames) / wall_seconds if wall_seconds > 0.0 else 0.0
    realtime_factor = frames_per_second / float(config.fps) if config.fps else 0.0
    steady_state_seconds: float | None = None
    steady_state_frames_per_second: float | None = None
    steady_state_realtime_factor: float | None = None
    if first_chunk_seconds is not None and first_chunk_frames is not None:
        steady_frames = max(0, int(n_pixel_frames) - int(first_chunk_frames))
        steady_seconds = wall_seconds - float(first_chunk_seconds)
        if steady_frames > 0 and steady_seconds > 0.0:
            steady_state_seconds = steady_seconds
            steady_state_frames_per_second = float(steady_frames) / steady_seconds
            steady_state_realtime_factor = steady_state_frames_per_second / float(config.fps) if config.fps else 0.0

    def _sum_cuda_seconds(events: list[tuple[torch.cuda.Event, torch.cuda.Event]]) -> float | None:
        if device.type != "cuda" or not events:
            return None
        total_ms = 0.0
        for start, end in events:
            total_ms += float(start.elapsed_time(end))
        return total_ms / 1000.0

    stage1_cuda_seconds = _sum_cuda_seconds(stage1_timing_events)
    refiner_cuda_seconds = _sum_cuda_seconds(refiner_timing_events)
    decode_cuda_seconds = _sum_cuda_seconds(decode_timing_events)
    log(
        f"[stream] output_mode={output_mode} frames={n_pixel_frames} "
        f"wall={wall_seconds:.3f}s fps={frames_per_second:.3f} "
        f"realtime={realtime_factor:.3f}x first_chunk={first_chunk_seconds} "
        f"steady_fps={steady_state_frames_per_second} steady_realtime={steady_state_realtime_factor} "
        f"cuda_stage1={stage1_cuda_seconds} cuda_refiner={refiner_cuda_seconds} "
        f"cuda_decode={decode_cuda_seconds} "
        f"sample_frames={sample_frames_path} sample_count={len(sample_frame_indices)} "
        f"path={out_path} (refiner_blocks={n_refiner}, decode_chunks={n_decode})"
    )
    return StreamingPipelineResult(
        output_path=out_path,
        n_pixel_frames=n_pixel_frames,
        n_refiner_blocks=n_refiner,
        n_decode_chunks=n_decode,
        output_mode=output_mode,
        wall_seconds=wall_seconds,
        first_chunk_seconds=first_chunk_seconds,
        first_chunk_frames=first_chunk_frames,
        steady_state_seconds=steady_state_seconds,
        steady_state_frames_per_second=steady_state_frames_per_second,
        steady_state_realtime_factor=steady_state_realtime_factor,
        frames_per_second=frames_per_second,
        realtime_factor=realtime_factor,
        stage1_cuda_seconds=stage1_cuda_seconds,
        refiner_cuda_seconds=refiner_cuda_seconds,
        decode_cuda_seconds=decode_cuda_seconds,
        sample_frames_path=sample_frames_path,
        sampled_frame_count=len(sample_frame_indices),
        sampled_frame_indices=sample_frame_indices,
    )


@torch.inference_mode()
def _run_streaming_inference_sequential(
    *,
    stage1_chunk_iter: Iterator[tuple[int, torch.Tensor, int, int]],
    n_stage1_chunks: int,
    z_init: torch.Tensor,
    refiner_runner: RefinerChunkRunner,
    vae_streaming_decoder: CausalVaeStreamingDecoder,
    pixel_h: int,
    pixel_w: int,
    config: StreamingPipelineConfig,
    logger=None,
) -> StreamingPipelineResult:
    """Memory-first path: finish stage-1, offload it, then refine/decode."""
    log = (logger.info if logger is not None else print)

    sink_size = int(config.sink_size)
    block_size = int(config.block_size)
    T_latent = int(z_init.shape[2])
    n_refiner = max(T_latent - sink_size, 0) // block_size
    n_decode = n_refiner
    output_mode = str(config.output_mode)
    device = z_init.device

    log("[stream] sequential_offload=1: stage1 -> offload -> refiner -> decode")
    if config.progress_callback is not None:
        config.progress_callback(
            {
                "phase": "stream_start",
                "message": "sequential streaming pipeline started",
                "stage1_done": 0,
                "stage1_total": n_stage1_chunks,
                "refiner_done": 0,
                "refiner_total": n_refiner,
                "decode_done": 0,
                "decode_total": n_decode,
                "frames": 0,
                "output_mode": output_mode,
            }
        )

    latents_full = torch.empty_like(z_init)
    latents_full[:, :, :sink_size] = z_init[:, :, :sink_size]
    refined_full = torch.empty_like(z_init)
    refined_full[:, :, :sink_size] = z_init[:, :, :sink_size]

    writer = None
    if output_mode == "mp4":
        writer = StreamingMp4Writer(
            config.output_path,
            height=int(pixel_h),
            width=int(pixel_w),
            fps=int(config.fps),
            crf=int(config.mp4_crf),
            preset=str(config.mp4_preset),
            encoder=str(config.mp4_encoder),
        )

    sample_frames: list[np.ndarray] = []
    sample_frame_indices: list[int] = []
    sample_frames_path = Path(config.sample_frames_path) if config.sample_frames_path is not None else None
    sample_frame_stride = int(config.sample_frame_stride)

    def _collect_sample_frames(pixel_np: np.ndarray, frame_base: int) -> None:
        if sample_frames_path is None:
            return
        for local_idx in range(0, int(pixel_np.shape[0])):
            frame_idx = int(frame_base + local_idx)
            if frame_idx % sample_frame_stride == 0:
                sample_frames.append(pixel_np[local_idx].copy())
                sample_frame_indices.append(frame_idx)

    def _sample_discard_frames(pixel_chunk: torch.Tensor, frame_base: int, drop_first: bool) -> None:
        if sample_frames_path is None:
            return
        drop = 1 if drop_first else 0
        n_out = max(0, int(pixel_chunk.shape[2]) - drop)
        local_indices = [i for i in range(n_out) if (frame_base + i) % sample_frame_stride == 0]
        if not local_indices:
            return
        src_indices = torch.as_tensor(
            [drop + i for i in local_indices],
            device=pixel_chunk.device,
            dtype=torch.long,
        )
        selected = pixel_chunk.index_select(2, src_indices)
        pixel_uint8 = (selected.float() * 127.5 + 127.5).clamp(0, 255).to(torch.uint8)
        pixel_np = pixel_uint8.permute(0, 2, 3, 4, 1).contiguous().to("cpu").numpy()[0]  # blocking: .numpy() reads immediately
        for frame, local_idx in zip(pixel_np, local_indices, strict=True):
            sample_frames.append(frame.copy())
            sample_frame_indices.append(int(frame_base + local_idx))

    t_start = time.perf_counter()
    stage1_t0 = time.perf_counter()
    first_chunk_seconds: float | None = None
    first_chunk_frames: int | None = None
    n_pixel_frames = 0
    out_path: Path | None = None
    try:
        for _ in range(n_stage1_chunks):
            if config.progress_callback is not None:
                config.progress_callback(
                    {
                        "phase": "stage1_running",
                        "message": f"stage-1 chunk {_ + 1}/{n_stage1_chunks} running",
                        "stage1_done": _,
                        "stage1_total": n_stage1_chunks,
                        "frames": n_pixel_frames,
                        "output_mode": output_mode,
                    }
                )
            chunk_idx, latent_view, start_f, end_f = next(stage1_chunk_iter)
            latents_full[:, :, start_f:end_f].copy_(latent_view, non_blocking=True)
            if config.progress_callback is not None:
                config.progress_callback(
                    {
                        "phase": "stage1",
                        "message": f"stage-1 chunk {int(chunk_idx) + 1}/{n_stage1_chunks}",
                        "stage1_done": int(chunk_idx) + 1,
                        "stage1_total": n_stage1_chunks,
                        "frames": n_pixel_frames,
                        "output_mode": output_mode,
                    }
                )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        stage1_seconds = time.perf_counter() - stage1_t0

        if config.stage1_done_callback is not None:
            config.stage1_done_callback()

        refiner_t0 = time.perf_counter()
        for k_ref in range(n_refiner):
            block_start = sink_size + k_ref * block_size
            block_end = block_start + block_size
            clean_block = latents_full[:, :, block_start:block_end]
            sink_seed = latents_full[:, :, :sink_size] if k_ref == 0 else None
            if _env_flag("SANA_WM_CUDAGRAPH_MARK_STEP"):
                mark_fn = getattr(getattr(torch, "compiler", None), "cudagraph_mark_step_begin", None)
                if mark_fn is not None:
                    mark_fn()
            refined_block = refiner_runner.refine_block(
                block_idx=k_ref,
                clean_block=clean_block,
                block_start=block_start,
                block_end=block_end,
                sink_seed_frames=sink_seed,
            )
            refined_full[:, :, block_start:block_end].copy_(refined_block, non_blocking=True)
            if config.progress_callback is not None:
                config.progress_callback(
                    {
                        "phase": "refiner",
                        "message": f"refiner block {k_ref + 1}/{n_refiner}",
                        "refiner_done": k_ref + 1,
                        "refiner_total": n_refiner,
                        "frames": n_pixel_frames,
                        "output_mode": output_mode,
                    }
                )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        refiner_seconds = time.perf_counter() - refiner_t0

        decoder = getattr(vae_streaming_decoder.vae, "decoder", None)
        if decoder is not None:
            decoder.to(device)
        vae_streaming_decoder.reset()

        decode_t0 = time.perf_counter()
        for k_dec in range(n_decode):
            if k_dec == 0:
                z_slice = refined_full[:, :, : sink_size + block_size]
            else:
                z_slice = refined_full[
                    :, :, sink_size + k_dec * block_size : sink_size + (k_dec + 1) * block_size
                ]
            if _env_flag("SANA_WM_CUDAGRAPH_MARK_STEP"):
                mark_fn = getattr(getattr(torch, "compiler", None), "cudagraph_mark_step_begin", None)
                if mark_fn is not None:
                    mark_fn()
            pixel_chunk = vae_streaming_decoder.decode_chunk(z_slice)
            if output_mode == "discard":
                n_frames = int(pixel_chunk.shape[2])
                drop_first = bool(k_dec == 0 and config.drop_first_pixel)
                if drop_first:
                    n_frames -= 1
                _sample_discard_frames(pixel_chunk, n_pixel_frames, drop_first)
            else:
                pixel_uint8 = (pixel_chunk.float() * 127.5 + 127.5).clamp(0, 255).to(torch.uint8)
                pixel_np = pixel_uint8.permute(0, 2, 3, 4, 1).contiguous().to("cpu").numpy()[0]
                if k_dec == 0 and config.drop_first_pixel:
                    pixel_np = pixel_np[1:]
                n_frames = int(pixel_np.shape[0])
                _collect_sample_frames(pixel_np, n_pixel_frames)
                if config.decoded_chunk_callback is not None and n_frames > 0:
                    config.decoded_chunk_callback(pixel_np, n_pixel_frames, int(k_dec))
                if output_mode == "mp4":
                    assert writer is not None
                    writer.write_chunk(pixel_np)
            if first_chunk_seconds is None:
                first_chunk_seconds = time.perf_counter() - t_start
                first_chunk_frames = n_frames
            n_pixel_frames += max(0, n_frames)
            if config.progress_callback is not None:
                config.progress_callback(
                    {
                        "phase": "decode",
                        "message": f"decoded chunk {k_dec + 1}/{n_decode}",
                        "decode_done": k_dec + 1,
                        "decode_total": n_decode,
                        "frames": n_pixel_frames,
                        "output_mode": output_mode,
                    }
                )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        decode_seconds = time.perf_counter() - decode_t0

        if config.progress_callback is not None:
            config.progress_callback(
                {
                    "phase": "finalize",
                    "message": "finalizing MP4" if output_mode == "mp4" else "finalizing preview stream",
                    "decode_done": n_decode,
                    "decode_total": n_decode,
                    "frames": n_pixel_frames,
                    "output_mode": output_mode,
                }
            )
        out_path = writer.close() if writer is not None else None
        if sample_frames_path is not None:
            sample_frames_path.parent.mkdir(parents=True, exist_ok=True)
            frames = (
                np.stack(sample_frames, axis=0)
                if sample_frames
                else np.empty((0, int(pixel_h), int(pixel_w), 3), dtype=np.uint8)
            )
            np.savez_compressed(
                sample_frames_path,
                frames=frames,
                frame_indices=np.asarray(sample_frame_indices, dtype=np.int64),
            )
    except Exception:
        if writer is not None:
            writer.close()
        raise

    wall_seconds = time.perf_counter() - t_start
    frames_per_second = float(n_pixel_frames) / wall_seconds if wall_seconds > 0.0 else 0.0
    realtime_factor = frames_per_second / float(config.fps) if config.fps else 0.0
    steady_state_seconds: float | None = None
    steady_state_frames_per_second: float | None = None
    steady_state_realtime_factor: float | None = None
    if first_chunk_seconds is not None and first_chunk_frames is not None:
        steady_frames = max(0, int(n_pixel_frames) - int(first_chunk_frames))
        steady_seconds = wall_seconds - float(first_chunk_seconds)
        if steady_frames > 0 and steady_seconds > 0.0:
            steady_state_seconds = steady_seconds
            steady_state_frames_per_second = float(steady_frames) / steady_seconds
            steady_state_realtime_factor = steady_state_frames_per_second / float(config.fps) if config.fps else 0.0

    log(
        f"[stream] sequential output_mode={output_mode} frames={n_pixel_frames} "
        f"wall={wall_seconds:.3f}s fps={frames_per_second:.3f} realtime={realtime_factor:.3f}x "
        f"stage1={stage1_seconds:.3f}s refiner={refiner_seconds:.3f}s decode={decode_seconds:.3f}s "
        f"path={out_path} (refiner_blocks={n_refiner}, decode_chunks={n_decode})"
    )
    return StreamingPipelineResult(
        output_path=out_path,
        n_pixel_frames=n_pixel_frames,
        n_refiner_blocks=n_refiner,
        n_decode_chunks=n_decode,
        output_mode=output_mode,
        wall_seconds=wall_seconds,
        first_chunk_seconds=first_chunk_seconds,
        first_chunk_frames=first_chunk_frames,
        steady_state_seconds=steady_state_seconds,
        steady_state_frames_per_second=steady_state_frames_per_second,
        steady_state_realtime_factor=steady_state_realtime_factor,
        frames_per_second=frames_per_second,
        realtime_factor=realtime_factor,
        stage1_cuda_seconds=stage1_seconds,
        refiner_cuda_seconds=refiner_seconds,
        decode_cuda_seconds=decode_seconds,
        sample_frames_path=sample_frames_path,
        sampled_frame_count=len(sample_frame_indices),
        sampled_frame_indices=sample_frame_indices,
    )
