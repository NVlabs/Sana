#!/usr/bin/env python3
"""Resident four-GB200 LTX-2.5 Stage-2 refiner with head/context parallelism."""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import math
import os
import platform
import time
from datetime import timedelta
from pathlib import Path, PurePosixPath
from typing import Any, Callable, TypeVar

import torch
import torch.distributed as dist
from safetensors import safe_open

from ltx_core.allocator_trim_strategy import AllocatorTrimStrategy
from ltx_core.batch_split import BatchSplitAdapter
from ltx_core.components.diffusion_steps import EulerDiffusionStep
from ltx_core.components.noisers import GaussianNoiser
from ltx_core.components.patchifiers import VideoLatentPatchifier
from ltx_core.loader import LTXV_LORA_COMFY_RENAMING_MAP, LoraPathStrengthAndSDOps
from ltx_core.loader.registry import ModelRegistry
from ltx_core.loader.sft_loader import SafetensorsModelStateDictLoader
from ltx_core.multigpu.transformer.attention import AttentionManager
from ltx_core.tools import VideoLatentTools
from ltx_core.types import Audio, VIDEO_SCALE_FACTORS, VideoLatentShape, VideoPixelShape
from ltx_pipelines.multigpu.sp_builder import SequenceParallelBuilder
from ltx_pipelines.multigpu.weight_tracker import TransformerWeightTracker
from ltx_pipelines.utils.blocks import DiffusionStage, PromptEncoder, VideoUpsampler
from ltx_pipelines.utils.denoisers import SimpleDenoiser
from ltx_pipelines.utils.helpers import create_noised_state
from ltx_pipelines.utils.media_io import (
    decode_audio_from_file,
    decode_video_by_frame,
    encode_video,
    get_videostream_metadata,
    video_preprocess,
)
from ltx_pipelines.utils.model_paths import ModelPaths
from ltx_pipelines.utils.samplers import euler_denoising_loop
from ltx_pipelines.utils.types import OffloadMode

from sol_attention import STAGE2_TAUS, Stage2SolAttention


WIDTH = 1920
HEIGHT = 1088
FRAME_COUNT = 241
FPS = 24.0
WORLD_SIZE = 4
STAGE2_SIGMAS = (0.909375, 0.725, 0.421875, 0.0)
LORA_STRENGTH = 0.8
MAX_VIDEO_TOKENS = 65536
EXPECTED_VIDEO_TOKENS = 63240  # 31 latent frames x 34 rows x 60 columns
TAEHV_PARALLEL_ELEMENT_LIMIT = 100_000_000
TAEHV_WEIGHT_SHA256 = "007788e6b9cb7f77e8589ae30ba7456b119d38b0d017e1d349c1c1d11e3d6339"
T = TypeVar("T")


def _cuda_sync() -> None:
    torch.cuda.synchronize()


def _timed_cuda(fn: Callable[[], T]) -> tuple[T, float]:
    _cuda_sync()
    started = time.perf_counter()
    value = fn()
    _cuda_sync()
    return value, time.perf_counter() - started


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _broadcast_rank0_error(error: str | None) -> None:
    payload = [error]
    dist.broadcast_object_list(payload, src=0)
    if payload[0] is not None:
        raise RuntimeError(str(payload[0]))


def _load_taehv_class(source_path: Path) -> type[torch.nn.Module]:
    spec = importlib.util.spec_from_file_location("ltx25_refiner_taehv", source_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import TAEHV source at {source_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.TAEHV


def _load_latent_statistics(
    video_vae_path: Path, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    with safe_open(str(video_vae_path), framework="pt", device="cpu") as checkpoint:
        keys = set(checkpoint.keys())
        prefix = next(
            (
                candidate
                for candidate in ("per_channel_statistics.", "vae.per_channel_statistics.")
                if f"{candidate}mean-of-means" in keys
                and f"{candidate}std-of-means" in keys
            ),
            None,
        )
        if prefix is None:
            raise RuntimeError(
                "video VAE does not contain the required per-channel latent statistics"
            )
        mean = checkpoint.get_tensor(f"{prefix}mean-of-means").float()
        std = checkpoint.get_tensor(f"{prefix}std-of-means").float()
    if mean.shape != (128,) or std.shape != (128,) or not bool(torch.all(std > 0)):
        raise RuntimeError(f"invalid LTX latent statistics: {mean.shape=} {std.shape=}")
    shape = (1, 128, 1, 1, 1)
    return mean.view(shape).to(device), std.view(shape).to(device)


def _module_residency(name: str, module: torch.nn.Module) -> dict[str, Any]:
    tensors = [*module.named_parameters(), *module.named_buffers()]
    if not tensors:
        raise RuntimeError(f"resident module {name!r} exposes no tensors")
    bad = [f"{tensor_name}={tensor.device}" for tensor_name, tensor in tensors if tensor.is_meta or tensor.device.type != "cuda"]
    if bad:
        raise RuntimeError(f"resident module {name!r} has non-CUDA tensors: {', '.join(bad[:8])}")
    storages: set[tuple[int, int]] = set()
    storage_bytes = 0
    for _, tensor in tensors:
        storage = tensor.untyped_storage()
        key = (storage.data_ptr(), storage.nbytes())
        if key not in storages:
            storages.add(key)
            storage_bytes += storage.nbytes()
    return {
        "tensor_count": len(tensors),
        "unique_storage_count": len(storages),
        "unique_storage_bytes": storage_bytes,
        "devices": sorted({str(tensor.device) for _, tensor in tensors}),
        "dtypes": sorted({str(tensor.dtype) for _, tensor in tensors}),
    }


def _cuda_memory() -> dict[str, int]:
    return {
        "allocated_bytes": torch.cuda.memory_allocated(),
        "reserved_bytes": torch.cuda.memory_reserved(),
        "max_allocated_bytes": torch.cuda.max_memory_allocated(),
        "max_reserved_bytes": torch.cuda.max_memory_reserved(),
    }


def _safe_source_path(input_root: Path, relative_path: str) -> Path:
    pure = PurePosixPath(relative_path)
    if pure.is_absolute() or ".." in pure.parts or pure.suffix.lower() != ".mp4":
        raise ValueError(f"unsafe or unsupported manifest file: {relative_path!r}")
    root = input_root.resolve()
    path = (root / Path(*pure.parts)).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"manifest file escapes INPUT_ROOT: {relative_path!r}") from exc
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(path)
    return path


def _load_single_record(manifest: Path, input_root: Path) -> dict[str, Any]:
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    if not isinstance(payload, list) or len(payload) != 1 or not isinstance(payload[0], dict):
        raise ValueError("the refiner manifest must be a JSON array containing exactly one row")
    record = dict(payload[0])
    if record.get("index") != 0:
        raise ValueError("the only manifest row must have index 0")
    if not isinstance(record.get("prompt"), str) or not record["prompt"].strip():
        raise ValueError("manifest row 0 must contain a non-empty prompt")
    if not isinstance(record.get("seed"), int):
        raise ValueError("manifest row 0 must contain an integer seed")
    if not isinstance(record.get("file"), str):
        raise ValueError("manifest row 0 must contain a relative MP4 file")
    record["_source_path"] = str(_safe_source_path(input_root, record["file"]))
    return record


def _tae_pixels_from_ltx(pixels: torch.Tensor) -> torch.Tensor:
    if pixels.ndim != 5 or pixels.shape[1] != 3:
        raise ValueError(f"expected NCTHW RGB pixels, got {tuple(pixels.shape)}")
    return pixels.add(1).mul(0.5).clamp_(0, 1).permute(0, 2, 1, 3, 4).contiguous()


def _normalize_audio(audio: Audio | None) -> Audio | None:
    if audio is None:
        return None
    waveform = audio.waveform
    if waveform.ndim == 3 and waveform.shape[0] == 1:
        waveform = waveform.squeeze(0)
    if waveform.ndim != 2:
        raise ValueError(f"unexpected audio shape: {tuple(waveform.shape)}")
    if waveform.shape[0] == 1:
        waveform = waveform.repeat(2, 1)
    elif waveform.shape[0] != 2 and waveform.shape[1] == 2:
        waveform = waveform.transpose(0, 1)
    if waveform.shape[0] != 2:
        raise ValueError(f"audio cannot be converted to stereo: {tuple(waveform.shape)}")
    return Audio(waveform=waveform.contiguous(), sampling_rate=audio.sampling_rate)


def _verify_video(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(path)
    metadata = get_videostream_metadata(str(path))
    actual = (metadata.width, metadata.height, metadata.frames, float(metadata.fps))
    expected = (WIDTH, HEIGHT, FRAME_COUNT, FPS)
    if actual[:3] != expected[:3] or not math.isclose(actual[3], expected[3], rel_tol=0, abs_tol=1e-6):
        raise RuntimeError(f"invalid output video metadata: expected {expected}, got {actual}")
    return {
        "width": metadata.width,
        "height": metadata.height,
        "frames": metadata.frames,
        "fps": float(metadata.fps),
        "duration_s": metadata.frames / metadata.fps,
        "bytes": path.stat().st_size,
    }


def _actual_sol_backend() -> str:
    from techniques.sparse_backends.sol_attn_backend import _load_sol_attn

    _load_sol_attn()
    from sol_attn import get_sol_attn_backend

    return str(get_sol_attn_backend(torch.cuda.current_device()))


class ResidentRefiner:
    """All model modules remain resident on every rank for warm-up and measurement."""

    @torch.inference_mode()
    def __init__(self, args: argparse.Namespace) -> None:
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
        self.dtype = torch.bfloat16
        if self.world_size != WORLD_SIZE:
            raise RuntimeError(f"expected world size {WORLD_SIZE}, got {self.world_size}")
        capability = tuple(torch.cuda.get_device_capability(self.device))
        if capability != (10, 0):
            raise RuntimeError(f"rank {self.rank} requires GB200/SM100, got capability {capability}")

        pixel_shape = VideoPixelShape(
            batch=1,
            frames=FRAME_COUNT,
            height=HEIGHT,
            width=WIDTH,
            fps=FPS,
        )
        latent_shape = VideoLatentShape.from_pixel_shape(pixel_shape, scale_factors=VIDEO_SCALE_FACTORS)
        self.video_tools = VideoLatentTools(
            VideoLatentPatchifier(patch_size=1),
            latent_shape,
            FPS,
            scale_factors=VIDEO_SCALE_FACTORS,
        )
        token_count = latent_shape.frames * latent_shape.height * latent_shape.width
        if token_count != EXPECTED_VIDEO_TOKENS:
            raise RuntimeError(f"unexpected Stage-2 video token count: {token_count}")

        model_paths = ModelPaths.from_split(
            transformer_path=str(args.transformer),
            text_encoder_path=str(args.text_encoder),
            video_vae_path=str(args.video_vae),
        )
        registry = ModelRegistry()
        lora = LoraPathStrengthAndSDOps(
            str(args.refiner_lora),
            LORA_STRENGTH,
            LTXV_LORA_COMFY_RENAMING_MAP,
        )
        stage = DiffusionStage.from_checkpoint(
            model_paths.transformer(),
            self.dtype,
            self.device,
            loras=(lora,),
            quantization=None,
            registry=registry,
            compilation_config=None,
            alloc_trim_strategy=AllocatorTrimStrategy.DEFER,
            offload_mode=OffloadMode.NONE,
        )
        model_config = stage._transformer_builder.model_config()["transformer"]
        if int(model_config["num_layers"]) != 48:
            raise RuntimeError(f"expected 48 transformer layers, got {model_config['num_layers']}")
        num_heads = int(model_config["num_attention_heads"])
        head_dim = int(model_config["attention_head_dim"])
        if num_heads % WORLD_SIZE != 0 or head_dim != 128:
            raise RuntimeError(f"invalid head-CP shape: {num_heads=} {head_dim=}")
        self.local_heads = num_heads // WORLD_SIZE
        self.total_heads = num_heads

        attention_manager = AttentionManager(
            max_tokens=MAX_VIDEO_TOKENS,
            num_heads=num_heads,
            head_dim=head_dim,
            tensor_dtype=self.dtype,
            group=dist.group.WORLD,
        )
        stage._transformer_builder = SequenceParallelBuilder(
            inner=stage._transformer_builder,
            attn_mgr=attention_manager,
            registry=registry,
            tracker=TransformerWeightTracker(group=dist.group.WORLD, no_lora_swap=True),
        )

        prompt_block = PromptEncoder(
            model_paths,
            self.dtype,
            self.device,
            registry=registry,
            offload_mode=OffloadMode.NONE,
            alloc_trim_strategy=AllocatorTrimStrategy.DEFER,
        )
        upsampler_block = VideoUpsampler(
            model_paths.video_vae(),
            str(args.upsampler),
            self.dtype,
            self.device,
            registry=registry,
            alloc_trim_strategy=AllocatorTrimStrategy.DEFER,
        )

        self.transformer = stage._build_transformer(video_tools=self.video_tools).requires_grad_(False)
        self.sol_attention = Stage2SolAttention(self.transformer)
        self.sol_backend = _actual_sol_backend()
        if self.sol_backend != "cute_sm100":
            raise RuntimeError(f"rank {self.rank} expected cute_sm100, got {self.sol_backend!r}")
        self.wrapped_transformer = BatchSplitAdapter(self.transformer, max_batch_size=1)
        self.gemma = prompt_block._build_text_encoder().eval().requires_grad_(False)
        self.embeddings_processor = prompt_block._build_embeddings_processor().eval().requires_grad_(False)
        self.spatial_upsampler = (
            upsampler_block._upsampler_builder.build(device=self.device, dtype=self.dtype)
            .eval()
            .requires_grad_(False)
        )
        taehv_class = _load_taehv_class(args.taehv_source)
        self.taehv = (
            taehv_class(str(args.taehv_checkpoint))
            .to(device=self.device, dtype=self.dtype)
            .eval()
            .requires_grad_(False)
        )
        self.latent_mean, self.latent_std = _load_latent_statistics(args.video_vae, self.device)
        self.sigmas = torch.tensor(STAGE2_SIGMAS, dtype=torch.float32, device=self.device)
        tae_input_elements = FRAME_COUNT * 3 * (HEIGHT // 2) * (WIDTH // 2)
        self.taehv_parallel = tae_input_elements < TAEHV_PARALLEL_ELEMENT_LIMIT
        if self.taehv_parallel:
            raise RuntimeError("the fixed 1080p10s workload must use sequential TAEHV execution")

        self.residency = {
            "transformer": _module_residency("transformer", self.transformer),
            "gemma": _module_residency("gemma", self.gemma),
            "embeddings_processor": _module_residency("embeddings_processor", self.embeddings_processor),
            "spatial_upsampler": _module_residency("spatial_upsampler", self.spatial_upsampler),
            "taehv_wide": _module_residency("taehv_wide", self.taehv),
        }
        self.guard_attempts = 0
        self._install_weight_load_guard()
        gc.collect()
        torch.cuda.empty_cache()
        _cuda_sync()

    def _install_weight_load_guard(self) -> None:
        def forbidden_safetensors_load(*_args: Any, **_kwargs: Any) -> Any:
            self.guard_attempts += 1
            raise RuntimeError("safetensors weight load attempted after resident preload")

        def forbidden_torch_load(*_args: Any, **_kwargs: Any) -> Any:
            self.guard_attempts += 1
            raise RuntimeError("torch.load attempted after resident preload")

        SafetensorsModelStateDictLoader.load = forbidden_safetensors_load
        torch.load = forbidden_torch_load

    def prepare_input(self, source_path: Path) -> tuple[torch.Tensor, dict[str, Any]]:
        metadata = get_videostream_metadata(str(source_path))
        actual = (metadata.width, metadata.height, metadata.frames, float(metadata.fps))
        expected = (WIDTH, HEIGHT, FRAME_COUNT, FPS)
        if actual[:3] != expected[:3] or not math.isclose(actual[3], expected[3], rel_tol=0, abs_tol=1e-6):
            raise ValueError(f"input must be exactly {expected}, got {actual}")
        frames = decode_video_by_frame(str(source_path), device=self.device, frame_cap=FRAME_COUNT)
        pixels = video_preprocess(frames, HEIGHT // 2, WIDTH // 2, self.dtype, self.device)
        tae_pixels = _tae_pixels_from_ltx(pixels)
        return tae_pixels, {
            "width": metadata.width,
            "height": metadata.height,
            "frames": metadata.frames,
            "fps": float(metadata.fps),
            "taehv_pixels_shape": list(tae_pixels.shape),
        }

    def encode_prompt(self, prompt: str) -> torch.Tensor:
        raw_outputs = self.gemma.encode([prompt])
        if len(raw_outputs) != 1:
            raise RuntimeError(f"Gemma returned {len(raw_outputs)} outputs")
        hidden_states, attention_mask = raw_outputs[0]
        processed = self.embeddings_processor.process_hidden_states(hidden_states, attention_mask)
        return processed.video_encoding.detach()

    def tae_encode(self, pixels: torch.Tensor) -> torch.Tensor:
        latent = self.taehv.encode_video(pixels, parallel=False, show_progress_bar=False)
        expected = (1, 31, 128, 17, 30)
        if tuple(latent.shape) != expected:
            raise RuntimeError(f"unexpected TAEHV latent shape {tuple(latent.shape)}, expected {expected}")
        return latent.permute(0, 2, 1, 3, 4).contiguous()

    def upsample(self, normalized: torch.Tensor) -> torch.Tensor:
        raw = normalized.float() * self.latent_std + self.latent_mean
        upsampled_raw = self.spatial_upsampler(raw.to(self.dtype)).float()
        expected = (1, 128, 31, 34, 60)
        if tuple(upsampled_raw.shape) != expected:
            raise RuntimeError(f"unexpected upsampled latent shape {tuple(upsampled_raw.shape)}, expected {expected}")
        return ((upsampled_raw - self.latent_mean) / self.latent_std).to(self.dtype)

    def prepare_denoise(self, latent: torch.Tensor, seed: int) -> tuple[Any, torch.Generator]:
        generator = torch.Generator(device=self.device).manual_seed(seed)
        state = create_noised_state(
            tools=self.video_tools,
            conditionings=[],
            noiser=GaussianNoiser(generator=generator),
            dtype=self.dtype,
            device=self.device,
            noise_scale=STAGE2_SIGMAS[0],
            initial_latent=latent,
        )
        return state, generator

    def denoise(self, state: Any, context: torch.Tensor) -> Any:
        self.sol_attention.begin_denoise()
        state, _ = euler_denoising_loop(
            sigmas=self.sigmas,
            video_state=state,
            audio_state=None,
            stepper=EulerDiffusionStep(),
            transformer=self.wrapped_transformer,
            denoiser=SimpleDenoiser(v_context=context, a_context=None),
        )
        if state is None:
            raise RuntimeError("Stage-2 denoising returned no video state")
        return state

    def finish_denoise(self, state: Any) -> torch.Tensor:
        state = self.video_tools.clear_conditioning(state)
        state = self.video_tools.unpatchify(state)
        return state.latent

    def tae_decode(self, latent: torch.Tensor, *, validate: bool) -> torch.Tensor:
        decoded = self.taehv.decode_video(
            latent.permute(0, 2, 1, 3, 4).to(self.dtype),
            parallel=False,
            show_progress_bar=False,
        )
        expected = (1, FRAME_COUNT, 3, HEIGHT, WIDTH)
        if tuple(decoded.shape) != expected:
            raise RuntimeError(f"unexpected TAEHV decoded shape {tuple(decoded.shape)}, expected {expected}")
        if validate and not bool(torch.isfinite(decoded).all().item()):
            raise RuntimeError("TAEHV warm-up output contains NaN or Inf")
        return decoded

    def write_video(self, decoded: torch.Tensor, source_path: Path, output_path: Path) -> None:
        audio = _normalize_audio(
            decode_audio_from_file(str(source_path), device=torch.device("cpu"), max_duration=FRAME_COUNT / FPS)
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = output_path.with_name(f".{output_path.stem}.partial-{os.getpid()}.mp4")
        temporary.unlink(missing_ok=True)

        def chunks():
            for start in range(0, FRAME_COUNT, 16):
                chunk = decoded[0, start : start + 16]
                yield chunk.permute(0, 2, 3, 1).float().contiguous()

        try:
            encode_video(
                video=chunks(),
                fps=int(FPS),
                audio=audio,
                output_path=str(temporary),
                video_chunks_number=math.ceil(FRAME_COUNT / 16),
                crf=19,
                preset="veryfast",
                thread_count=8,
            )
            if not temporary.is_file() or temporary.stat().st_size == 0:
                raise RuntimeError(f"video encoder produced no output: {temporary}")
            os.replace(temporary, output_path)
        finally:
            temporary.unlink(missing_ok=True)

    def _checked_attention_stats(self) -> dict[str, Any]:
        stats = self.sol_attention.stats()
        stats["selected_backend"] = self.sol_backend
        kernel_calls = int(stats.get("kernel", {}).get("kernel_calls", -1))
        expected = {
            "completed_steps": 3,
            "dense_calls": 3,
            "sol_calls": 141,
        }
        for key, value in expected.items():
            if int(stats.get(key, -1)) != value:
                raise RuntimeError(f"rank {self.rank} invalid attention {key}: {stats.get(key)}")
        if kernel_calls != 141 or self.sol_backend != "cute_sm100":
            raise RuntimeError(
                f"rank {self.rank} did not execute 141 SM100 Sol kernels: "
                f"backend={self.sol_backend} kernel_calls={kernel_calls}"
            )
        return stats

    @torch.inference_mode()
    def run_sample(
        self,
        record: dict[str, Any],
        *,
        warmup: bool,
        output_path: Path | None,
    ) -> dict[str, Any] | None:
        dist.barrier()
        torch.cuda.reset_peak_memory_stats()
        _cuda_sync()
        wall_started = time.perf_counter()
        source_path = Path(record["_source_path"])

        (pixels, input_info), input_decode_resize_s = _timed_cuda(lambda: self.prepare_input(source_path))
        context, gemma_embedding_s = _timed_cuda(lambda: self.encode_prompt(record["prompt"]))
        normalized, taehv_encode_s = _timed_cuda(lambda: self.tae_encode(pixels))
        del pixels
        upscaled, latent_upsample_s = _timed_cuda(lambda: self.upsample(normalized))
        del normalized
        _, replica_sync_s = _timed_cuda(
            lambda: (
                dist.broadcast(context, src=0),
                dist.broadcast(upscaled, src=0),
            )
        )
        (state, generator), denoise_prepare_s = _timed_cuda(
            lambda: self.prepare_denoise(upscaled, int(record["seed"]))
        )
        del upscaled

        dist.barrier()
        _cuda_sync()
        denoise_started = time.perf_counter()
        state = self.denoise(state, context)
        _cuda_sync()
        dist.barrier()
        transformer_denoise_s = time.perf_counter() - denoise_started
        del context
        latent, denoise_finish_s = _timed_cuda(lambda: self.finish_denoise(state))
        del state, generator
        decoded, taehv_decode_s = _timed_cuda(lambda: self.tae_decode(latent, validate=warmup))
        del latent

        attention = self._checked_attention_stats()
        h264_encode_mux_s = 0.0
        output_info: dict[str, Any] = {}
        output_error: str | None = None
        if self.rank == 0 and output_path is not None:
            try:
                _, h264_encode_mux_s = _timed_cuda(lambda: self.write_video(decoded, source_path, output_path))
                output_info = _verify_video(output_path)
            except Exception as exc:  # synchronize rank-0 I/O failure across all ranks
                output_error = f"rank-0 output failed: {type(exc).__name__}: {exc}"
        _broadcast_rank0_error(output_error if self.rank == 0 else None)
        _cuda_sync()
        dist.barrier()
        local_wall_s = time.perf_counter() - wall_started
        del decoded
        gc.collect()
        _cuda_sync()
        dist.barrier()

        local = {
            "rank": self.rank,
            "device": torch.cuda.get_device_name(self.device),
            "device_capability": list(torch.cuda.get_device_capability(self.device)),
            "total_heads": self.total_heads,
            "local_heads": self.local_heads,
            "full_parameter_replica": True,
            "full_sequence_tokens_per_local_head": EXPECTED_VIDEO_TOKENS,
            "timings": {
                "input_decode_resize_s": input_decode_resize_s,
                "gemma_embedding_s": gemma_embedding_s,
                "taehv_encode_s": taehv_encode_s,
                "latent_upsample_s": latent_upsample_s,
                "replica_sync_s": replica_sync_s,
                "denoise_prepare_s": denoise_prepare_s,
                "transformer_denoise_s": transformer_denoise_s,
                "denoise_finish_s": denoise_finish_s,
                "taehv_decode_s": taehv_decode_s,
                "h264_encode_mux_s": h264_encode_mux_s,
                "sample_wall_s": local_wall_s,
            },
            "attention": attention,
            "weight_load_guard_attempts": self.guard_attempts,
            "cuda_memory": _cuda_memory(),
        }
        gathered: list[dict[str, Any] | None] = [None] * WORLD_SIZE
        dist.all_gather_object(gathered, local)
        if self.rank != 0:
            return None

        per_rank = [item for item in gathered if item is not None]
        if len(per_rank) != WORLD_SIZE or any(item["weight_load_guard_attempts"] != 0 for item in per_rank):
            raise RuntimeError("resident inference recorded an invalid rank or post-preload weight load")
        phase_names = tuple(local["timings"])
        phase_max = {
            name: round(max(float(item["timings"][name]) for item in per_rank), 6)
            for name in phase_names
        }
        selected_model_total = sum(
            phase_max[name]
            for name in ("gemma_embedding_s", "taehv_encode_s", "transformer_denoise_s", "taehv_decode_s")
        )
        aggregate_attention = {
            "dense_calls": sum(int(item["attention"]["dense_calls"]) for item in per_rank),
            "sol_calls": sum(int(item["attention"]["sol_calls"]) for item in per_rank),
            "kernel_calls": sum(int(item["attention"]["kernel"]["kernel_calls"]) for item in per_rank),
        }
        return {
            "status": "succeeded",
            "warmup": warmup,
            "index": record["index"],
            "prompt_id": record.get("prompt_id"),
            "seed": record["seed"],
            "input": str(source_path),
            "output": str(output_path) if output_path is not None else None,
            "output_info": output_info,
            **phase_max,
            "selected_model_total_s": round(selected_model_total, 6),
            "parallelism": {
                "kind": "head_context_parallel",
                "degree": WORLD_SIZE,
                "parameter_replication": "full_per_rank",
                "activation_layout": "sequence_sharded_outside_attention",
                "attention_layout": "full_sequence_local_heads",
                "total_heads": self.total_heads,
                "local_heads_per_rank": self.local_heads,
                "video_tokens": EXPECTED_VIDEO_TOKENS,
            },
            "attention": {
                "per_rank": [item["attention"] for item in per_rank],
                "aggregate": aggregate_attention,
                "stage2_taus": list(STAGE2_TAUS),
            },
            "per_rank": per_rank,
            "weight_load_guard_attempts": 0,
            "all_inference_modules_simultaneously_gpu_resident": True,
            "timestamp_epoch_s": time.time(),
            "input_info": input_info,
        }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--metadata-dir", type=Path, required=True)
    parser.add_argument("--transformer", type=Path, required=True)
    parser.add_argument("--text-encoder", type=Path, required=True)
    parser.add_argument("--video-vae", type=Path, required=True)
    parser.add_argument("--upsampler", type=Path, required=True)
    parser.add_argument("--refiner-lora", type=Path, required=True)
    parser.add_argument("--taehv-source", type=Path, required=True)
    parser.add_argument("--taehv-checkpoint", type=Path, required=True)
    return parser


def _validate_paths(args: argparse.Namespace) -> None:
    required = (
        args.input_root,
        args.manifest,
        args.transformer,
        args.text_encoder,
        args.video_vae,
        args.upsampler,
        args.refiner_lora,
        args.taehv_source,
        args.taehv_checkpoint,
    )
    for path in required:
        if not path.exists() or (path.is_file() and path.stat().st_size == 0):
            raise FileNotFoundError(path)


def main() -> int:
    args = _build_parser().parse_args()
    local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
    if local_rank < 0:
        raise RuntimeError("launch with torch.distributed.run/torchrun")
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", timeout=timedelta(minutes=30))
    rank = dist.get_rank()
    try:
        if dist.get_world_size() != WORLD_SIZE or torch.cuda.device_count() != WORLD_SIZE:
            raise RuntimeError(
                f"expected exactly {WORLD_SIZE} visible GPUs/ranks, got "
                f"devices={torch.cuda.device_count()} world={dist.get_world_size()}"
            )
        _validate_paths(args)
        record = _load_single_record(args.manifest, args.input_root)

        verification_error: str | None = None
        if rank == 0:
            try:
                actual_sha = _sha256(args.taehv_checkpoint)
                if actual_sha != TAEHV_WEIGHT_SHA256:
                    raise RuntimeError(
                        f"TAEHV weight SHA mismatch: expected {TAEHV_WEIGHT_SHA256}, got {actual_sha}"
                    )
                args.output_dir.mkdir(parents=True, exist_ok=True)
                args.metadata_dir.mkdir(parents=True, exist_ok=True)
            except Exception as exc:
                verification_error = f"rank-0 setup validation failed: {type(exc).__name__}: {exc}"
        _broadcast_rank0_error(verification_error if rank == 0 else None)

        load_started = time.perf_counter()
        models = ResidentRefiner(args)
        _cuda_sync()
        dist.barrier()
        local_load_s = time.perf_counter() - load_started
        load_times: list[float | None] = [None] * WORLD_SIZE
        dist.all_gather_object(load_times, local_load_s)

        residency_local = {
            "rank": rank,
            "device": torch.cuda.get_device_name(local_rank),
            "device_capability": list(torch.cuda.get_device_capability(local_rank)),
            "modules": models.residency,
            "cuda_memory": _cuda_memory(),
        }
        residency_ranks: list[dict[str, Any] | None] = [None] * WORLD_SIZE
        dist.all_gather_object(residency_ranks, residency_local)
        if rank == 0:
            _write_json(
                args.metadata_dir / "residency.json",
                {
                    "status": "succeeded",
                    "hardware": "4xGB200",
                    "world_size": WORLD_SIZE,
                    "parallelism": "4-way head/context parallel",
                    "parameter_replication": "full_per_rank",
                    "dtype": "torch.bfloat16",
                    "offload": False,
                    "quantization": None,
                    "compile": False,
                    "cache": False,
                    "all_inference_modules_simultaneously_gpu_resident": True,
                    "taehv_temporal_execution": "sequential",
                    "preload_wall_s_max_excluded": round(max(float(value) for value in load_times if value is not None), 6),
                    "per_rank": [item for item in residency_ranks if item is not None],
                },
            )

        warmup = models.run_sample(record, warmup=True, output_path=None)
        if rank == 0:
            assert warmup is not None
            _write_json(args.metadata_dir / "warmup.json", warmup)

        output_path = args.output_dir / "ltx25_refiner_1920x1088_241f.mp4"
        measured = models.run_sample(record, warmup=False, output_path=output_path)
        if rank == 0:
            assert measured is not None
            _write_json(args.metadata_dir / "samples" / "00.json", measured)
            _write_json(
                args.metadata_dir / "benchmark.json",
                {
                    "schema_version": 1,
                    "hardware": "4xGB200",
                    "host": platform.node(),
                    "torch": torch.__version__,
                    "cuda": torch.version.cuda,
                    "workload": {
                        "width": WIDTH,
                        "height": HEIGHT,
                        "frames": FRAME_COUNT,
                        "fps": FPS,
                        "duration_s": FRAME_COUNT / FPS,
                        "stage2_sigmas": list(STAGE2_SIGMAS),
                        "stage2_updates": 3,
                        "taus": list(STAGE2_TAUS),
                        "lora_strength": LORA_STRENGTH,
                    },
                    "timing_scope": (
                        "resident post-warmup single-video E2E; model loading, Sol first-use "
                        "compilation/autotuning, and the complete warm-up request are excluded"
                    ),
                    "sample_wall_s": measured["sample_wall_s"],
                    "phases_s": {
                        key: measured[key]
                        for key in (
                            "input_decode_resize_s",
                            "gemma_embedding_s",
                            "taehv_encode_s",
                            "latent_upsample_s",
                            "replica_sync_s",
                            "denoise_prepare_s",
                            "transformer_denoise_s",
                            "denoise_finish_s",
                            "taehv_decode_s",
                            "h264_encode_mux_s",
                        )
                    },
                    "attention": measured["attention"],
                    "output": measured["output"],
                },
            )
            _write_json(
                args.metadata_dir / "success.json",
                {
                    "status": "succeeded",
                    "rendered_outputs": 1,
                    "measured_hot_samples": 1,
                    "sample_wall_s": measured["sample_wall_s"],
                    "weight_load_guard_attempts": measured["weight_load_guard_attempts"],
                    "all_inference_modules_simultaneously_gpu_resident": True,
                    "attention_aggregate": measured["attention"]["aggregate"],
                    "output": measured["output"],
                    "timestamp_epoch_s": time.time(),
                },
            )
            print(json.dumps({"sample_wall_s": measured["sample_wall_s"], "output": measured["output"]}, indent=2))
        dist.barrier()
        return 0
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    raise SystemExit(main())
