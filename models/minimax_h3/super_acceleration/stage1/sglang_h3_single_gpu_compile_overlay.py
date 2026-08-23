"""Single-GPU local VAE-core compile overlay for pinned MiniMax-H3 SGLang.

The first request reaches denoising only after its first-frame encode.  At that
point this overlay wraps the video VAE's local encoder and decoder modules with
``torch.compile``.  The first decode compiles the decoder; subsequent requests
use hot compiled encoder and decoder graphs.  No tiling mode or collective is
changed or captured.
"""

from __future__ import annotations

import os
import types
from collections.abc import Callable
from typing import Any

import torch


PINNED_SGLANG_COMMIT = "12eadf86f12aec2e6f81a6e38b61b964a4c6b529"
DECODER_TILE_BATCH_SIZE_ENV = "SGLANG_H3_VAE_DECODER_TILE_BATCH_SIZE"
DECODER_TILE_BATCH_SIZE_CHOICES = (1, 2, 4, 15)
PINNED_TILE_SIZE = 256
PINNED_TILE_OVERLAP = 64
_INSTALL_STATE: dict[str, Any] | None = None


def _decoder_tile_batch_size_from_env() -> int:
    raw = os.environ.get(DECODER_TILE_BATCH_SIZE_ENV, "1")
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(
            f"{DECODER_TILE_BATCH_SIZE_ENV} must be one of "
            f"{DECODER_TILE_BATCH_SIZE_CHOICES}, got {raw!r}"
        ) from exc
    if str(value) != raw.strip() or value not in DECODER_TILE_BATCH_SIZE_CHOICES:
        raise ValueError(
            f"{DECODER_TILE_BATCH_SIZE_ENV} must be one of "
            f"{DECODER_TILE_BATCH_SIZE_CHOICES}, got {raw!r}"
        )
    return value


def _configure_single_gpu_decoder_tile_batching(
    vae: Any,
    *,
    batch_size: int,
) -> dict[str, Any]:
    """Batch native decoder tiles without changing their geometry or order."""

    if (
        isinstance(batch_size, bool)
        or batch_size not in DECODER_TILE_BATCH_SIZE_CHOICES
    ):
        raise ValueError(
            "decoder tile batch size must be one of "
            f"{DECODER_TILE_BATCH_SIZE_CHOICES}, got {batch_size!r}"
        )
    required = ("_run_tile_tasks", "tile_size", "tile_overlap_min")
    missing = [name for name in required if not hasattr(vae, name)]
    if missing:
        raise RuntimeError(f"MiniMax-H3 video VAE is missing tile fields: {missing}")
    geometry = {
        "tile_size": int(vae.tile_size),
        "tile_overlap_min": int(vae.tile_overlap_min),
        "decoder_tile_size": int(getattr(vae, "decoder_tile_size", vae.tile_size)),
        "decoder_tile_overlap_min": int(
            getattr(vae, "decoder_tile_overlap_min", vae.tile_overlap_min)
        ),
    }
    expected_geometry = {
        "tile_size": PINNED_TILE_SIZE,
        "tile_overlap_min": PINNED_TILE_OVERLAP,
        "decoder_tile_size": PINNED_TILE_SIZE,
        "decoder_tile_overlap_min": PINNED_TILE_OVERLAP,
    }
    if geometry != expected_geometry:
        raise RuntimeError(
            f"decoder tile batching requires pinned geometry: {geometry} != "
            f"{expected_geometry}"
        )

    existing = getattr(vae, "_h3_single_gpu_decoder_tile_batch", None)
    if existing is not None:
        if int(existing["batch_size"]) != batch_size:
            raise RuntimeError(
                "decoder tile batching is already configured with a different "
                f"batch size: {existing}"
            )
        return dict(existing)

    marker = {
        "installed": batch_size > 1,
        "batch_size": batch_size,
        "choices": list(DECODER_TILE_BATCH_SIZE_CHOICES),
        "tile_size": PINNED_TILE_SIZE,
        "tile_overlap_min": PINNED_TILE_OVERLAP,
        "native_tile_order_preserved": True,
        "native_stitch_preserved": True,
    }
    if batch_size == 1:
        # This is the compatibility/default path: do not replace any method.
        vae._h3_single_gpu_decoder_tile_batch = marker
        return dict(marker)

    original_run_tile_tasks = vae._run_tile_tasks

    def batched_run_tile_tasks(
        self: Any,
        tiles: list[torch.Tensor],
        tile_indices: list[int],
        forward_fn: Callable[[torch.Tensor], torch.Tensor],
        stack_tiling: bool,
        cls_agg: Any = None,
    ) -> list[torch.Tensor]:
        is_decoder = (
            getattr(forward_fn, "__self__", None) is self
            and getattr(forward_fn, "__name__", "") in {"decode", "decode_base"}
        )
        if not is_decoder:
            return original_run_tile_tasks(
                tiles,
                tile_indices,
                forward_fn,
                stack_tiling,
                cls_agg,
            )
        if stack_tiling:
            raise RuntimeError(
                "decoder tile microbatch owns stacking; native stack_tiling must be false"
            )
        if cls_agg is not None:
            raise RuntimeError(
                "decoder tile microbatch does not accept an encoder cls_agg"
            )
        if not tiles:
            raise RuntimeError("decoder tile microbatch received no tiles")
        expected_indices = list(range(len(tiles)))
        if tile_indices != expected_indices:
            raise RuntimeError(
                "decoder tile microbatch requires one-GPU complete native tile order "
                f"{expected_indices}, got {tile_indices}"
            )

        reference = tiles[0]
        if not isinstance(reference, torch.Tensor) or reference.ndim < 1:
            raise RuntimeError("decoder tiles must be tensors with a batch dimension")
        reference_shape = tuple(int(value) for value in reference.shape)
        sample_batch_size = int(reference.shape[0])
        if sample_batch_size < 1:
            raise RuntimeError("decoder tile sample batch must be non-empty")
        for index, tile in enumerate(tiles):
            if not isinstance(tile, torch.Tensor):
                raise RuntimeError(f"decoder tile {index} is not a tensor")
            if tuple(int(value) for value in tile.shape) != reference_shape:
                raise RuntimeError(
                    f"decoder tile {index} shape {tuple(tile.shape)} != {reference_shape}"
                )
            if tile.dtype != reference.dtype or tile.device != reference.device:
                raise RuntimeError(
                    f"decoder tile {index} dtype/device differs from the first tile"
                )

        outputs: list[torch.Tensor] = []
        for start in range(0, len(tile_indices), batch_size):
            chunk_indices = tile_indices[start : start + batch_size]
            tile_batch = torch.cat([tiles[index] for index in chunk_indices], dim=0)
            output_batch = forward_fn(tile_batch)
            if not isinstance(output_batch, torch.Tensor) or output_batch.ndim < 1:
                raise RuntimeError("decoder tile batch output must be a tensor")
            expected_leading = len(chunk_indices) * sample_batch_size
            if int(output_batch.shape[0]) != expected_leading:
                raise RuntimeError(
                    "decoder tile batch output leading dimension "
                    f"{int(output_batch.shape[0])} != {expected_leading}"
                )
            outputs.extend(
                output_batch.unflatten(
                    0,
                    (len(chunk_indices), sample_batch_size),
                ).unbind(dim=0)
            )
        if len(outputs) != len(tiles):
            raise RuntimeError(
                f"decoder tile batch returned {len(outputs)} tiles, expected {len(tiles)}"
            )
        return outputs

    vae._run_tile_tasks = types.MethodType(batched_run_tile_tasks, vae)
    vae._h3_single_gpu_decoder_tile_batch = marker
    return dict(marker)


def _compile_single_gpu_vae_cores(
    vae: Any,
    *,
    mode: str,
    decoder_tile_batch_size: int = 1,
) -> dict[str, Any]:
    marker = getattr(vae, "_h3_single_gpu_compile", None)
    requested = {
        "mode": str(mode),
        "fullgraph": False,
        "dynamic": False,
        "decoder_tile_batch_size": int(decoder_tile_batch_size),
    }
    if marker is not None:
        if any(marker.get(key) != value for key, value in requested.items()):
            raise RuntimeError(f"VAE was compiled with different options: {marker}")
        return dict(marker)
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("single-GPU VAE compile requires exactly one visible CUDA GPU")
    if bool(getattr(vae, "training", True)):
        vae.eval()
    if not callable(getattr(vae, "encoder", None)) or not callable(
        getattr(vae, "decoder", None)
    ):
        raise RuntimeError("MiniMax-H3 video VAE has no callable encoder/decoder")

    tile_batching = _configure_single_gpu_decoder_tile_batching(
        vae,
        batch_size=decoder_tile_batch_size,
    )

    # Keep all VAE orchestration and whatever native one-rank tiling profile the
    # pinned runtime selected unchanged.  Only these two local neural cores are
    # compiled; there is no NCCL graph at world size one.
    tiling_before = {
        name: getattr(vae, name, None)
        for name in (
            "stack_tiling",
            "encoder_tiling",
            "decoder_tiling",
            "parallel_tiling",
            "parallel_decode_mode",
        )
    }
    if hasattr(vae, "prepare_decoder_autocast_weights"):
        vae.prepare_decoder_autocast_weights(torch.float16)
    compile_kwargs = {
        "mode": str(mode),
        "fullgraph": False,
        "dynamic": False,
    }
    vae.decoder = torch.compile(vae.decoder, **compile_kwargs)
    vae.encoder = torch.compile(vae.encoder, **compile_kwargs)
    tiling_after = {name: getattr(vae, name, None) for name in tiling_before}
    if tiling_after != tiling_before:
        raise RuntimeError(
            f"local VAE compile unexpectedly changed tiling: {tiling_before} -> {tiling_after}"
        )
    marker = {
        **requested,
        "installed": True,
        "scope": "single_gpu_local_video_vae_encoder_decoder_only",
        "cudagraphs": False,
        "collectives_captured": False,
        "tiling_changed": False,
        "tiling": tiling_before,
        "decoder_tile_batching": tile_batching,
        "warmup_complete": False,
    }
    vae._h3_single_gpu_compile = marker
    return dict(marker)


def install_single_gpu_vae_compile_overlay(
    *,
    arm: str,
    enabled: bool,
    mode: str = "max-autotune-no-cudagraphs",
    decoder_tile_batch_size: int | None = None,
) -> dict[str, Any]:
    """Install a denoising-stage hook that compiles local VAE cores once."""

    global _INSTALL_STATE
    if arm not in {"teacher", "student"}:
        raise ValueError(f"compile overlay arm must be teacher/student, got {arm!r}")
    if enabled and mode != "max-autotune-no-cudagraphs":
        raise ValueError(f"unsupported formal compile mode {mode!r}")
    if decoder_tile_batch_size is None:
        decoder_tile_batch_size = _decoder_tile_batch_size_from_env()
    if (
        isinstance(decoder_tile_batch_size, bool)
        or decoder_tile_batch_size not in DECODER_TILE_BATCH_SIZE_CHOICES
    ):
        raise ValueError(
            "decoder tile batch size must be one of "
            f"{DECODER_TILE_BATCH_SIZE_CHOICES}, got {decoder_tile_batch_size!r}"
        )
    if not enabled and decoder_tile_batch_size != 1:
        raise ValueError("decoder tile batching requires the VAE compile overlay")
    requested = {
        "arm": arm,
        "enabled": bool(enabled),
        "mode": str(mode),
        "decoder_tile_batch_size": decoder_tile_batch_size,
    }
    if _INSTALL_STATE is not None:
        if _INSTALL_STATE["config"] != requested:
            raise RuntimeError(f"a different single-GPU compile overlay is active: {_INSTALL_STATE}")
        return dict(_INSTALL_STATE)

    if not enabled:
        _INSTALL_STATE = {
            "installed": False,
            "name": "sglang_minimax_h3_single_gpu_vae_compile_v1",
            "pinned_sglang_commit": PINNED_SGLANG_COMMIT,
            "config": requested,
            "scope": "eager",
        }
        return dict(_INSTALL_STATE)

    from sglang.multimodal_gen.runtime.pipelines import minimax_h3_pipeline

    current_stage = minimax_h3_pipeline.MiniMaxH3DenoisingStage
    if getattr(current_stage, "_h3_single_gpu_vae_compile_overlay", False):
        raise RuntimeError("unexpected pre-installed single-GPU VAE compile stage")

    class MiniMaxH3SingleGPUVAECompileStage(current_stage):
        def __init__(self, transformer: Any, pipeline: Any = None) -> None:
            super().__init__(transformer=transformer, pipeline=pipeline)
            if pipeline is None:
                raise RuntimeError("VAE compile stage requires its owning pipeline")
            self._ff_compile_pipeline = pipeline

        def _run_full_loop(self, batch: Any, server_args: Any) -> None:
            runtime_compile = bool(getattr(server_args, "enable_torch_compile", False))
            if not runtime_compile:
                raise RuntimeError("VAE compile overlay requires enable_torch_compile")
            vae = self._ff_compile_pipeline.get_module("video_vae")
            cache_hit = getattr(vae, "_h3_single_gpu_compile", None) is not None
            compile_marker = _compile_single_gpu_vae_cores(
                vae,
                mode=mode,
                decoder_tile_batch_size=decoder_tile_batch_size,
            )
            request_marker = {
                **compile_marker,
                "new_install": not cache_hit,
                "cache_hit": cache_hit,
            }
            batch.extra["minimax_h3_single_gpu_vae_compile"] = request_marker
            print(
                "[H3SingleGPUVAECompile] "
                f"new_install={int(not cache_hit)} cache_hit={int(cache_hit)} "
                f"scope={compile_marker['scope']} "
                f"decoder_tile_batch_size={decoder_tile_batch_size}",
                flush=True,
            )
            return super()._run_full_loop(batch, server_args)

    MiniMaxH3SingleGPUVAECompileStage.__name__ = "MiniMaxH3SingleGPUVAECompileStage"
    MiniMaxH3SingleGPUVAECompileStage.__qualname__ = "MiniMaxH3SingleGPUVAECompileStage"
    MiniMaxH3SingleGPUVAECompileStage._h3_single_gpu_vae_compile_overlay = True
    MiniMaxH3SingleGPUVAECompileStage._h3_single_gpu_vae_compile_stock_stage = current_stage
    minimax_h3_pipeline.MiniMaxH3DenoisingStage = MiniMaxH3SingleGPUVAECompileStage

    _INSTALL_STATE = {
        "installed": True,
        "name": "sglang_minimax_h3_single_gpu_vae_compile_v1",
        "pinned_sglang_commit": PINNED_SGLANG_COMMIT,
        "config": requested,
        "scope": "single_gpu_local_video_vae_encoder_decoder_only",
        "cudagraphs": False,
        "collectives_captured": False,
        "tiling_changed": False,
        "decoder_tile_batching": {
            "batch_size": decoder_tile_batch_size,
            "choices": list(DECODER_TILE_BATCH_SIZE_CHOICES),
            "tile_size": PINNED_TILE_SIZE,
            "tile_overlap_min": PINNED_TILE_OVERLAP,
            "native_tile_order_preserved": True,
            "native_stitch_preserved": True,
        },
        "warmup_contract": (
            "first request compiles decoder after an eager first-frame encode; "
            "all subsequent requests use hot compiled encoder and decoder"
        ),
    }
    return dict(_INSTALL_STATE)


__all__ = [
    "DECODER_TILE_BATCH_SIZE_CHOICES",
    "DECODER_TILE_BATCH_SIZE_ENV",
    "install_single_gpu_vae_compile_overlay",
]
