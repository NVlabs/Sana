"""RTX 4090 pipeline stages that differ from pinned SGLang MiniMax-H3."""

from __future__ import annotations

import contextlib
import os
import time

import torch

from sglang.multimodal_gen.runtime.managers.memory_managers.layerwise_offload import (
    is_layerwise_offloaded_module,
)
from sglang.multimodal_gen.runtime.pipelines.minimax_h3_pipeline import (
    MiniMaxH3Pipeline,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages import InputValidationStage
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3 import (
    MiniMaxH3AudioEncodingStage,
    MiniMaxH3DecodingStage,
    MiniMaxH3DenoisingStage,
    MiniMaxH3LatentPreparationStage,
    MiniMaxH3TextEncodingStage,
    MiniMaxH3TimestepPreparationStage,
    MiniMaxH3VisualEncodingStage,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.release_metadata import (
    MiniMaxH3PartitionAdmissionStage,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger


logger = init_logger(__name__)


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _resident_vae_dtype() -> torch.dtype | None:
    raw_dtype = os.environ.get("H3_FULL_VAE_DTYPE", "").strip().lower()
    if not raw_dtype:
        return None
    aliases = {
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
        "fp32": torch.float32,
        "float32": torch.float32,
    }
    try:
        return aliases[raw_dtype]
    except KeyError as exc:
        raise ValueError(
            "H3_FULL_VAE_DTYPE must be bf16, fp16, or fp32; "
            f"got {raw_dtype!r}"
        ) from exc


@contextlib.contextmanager
def _full_video_vae_residency(video_vae):
    """Materialize the layerwise-offloaded video VAE for one decode."""

    enabled = _env_flag("H3_FULL_VAE_AFTER_DENOISE")
    was_layerwise = (
        enabled
        and video_vae is not None
        and is_layerwise_offloaded_module(video_vae)
    )
    if not was_layerwise:
        yield
        return

    device = torch.get_device_module()
    resident_dtype = _resident_vae_dtype()
    device.synchronize()
    device.empty_cache()
    if resident_dtype is not None:
        video_vae.to(dtype=resident_dtype)

    started = time.perf_counter()
    video_vae.disable_offload()
    if resident_dtype is not None:
        video_vae.to(dtype=resident_dtype)
    device.synchronize()
    logger.info(
        "MiniMax-H3 video VAE became resident in %.3f s (dtype=%s)",
        time.perf_counter() - started,
        resident_dtype,
    )
    try:
        yield
    finally:
        video_vae.enable_offload()
        device.synchronize()
        device.empty_cache()


@contextlib.contextmanager
def _bounded_video_vae_tile_batch(video_vae):
    """Bound stacked spatial-tile batches without changing tile geometry."""

    raw_batch_size = os.environ.get("H3_FULL_VAE_TILE_BATCH_SIZE", "0")
    try:
        tile_batch_size = int(raw_batch_size)
    except ValueError as exc:
        raise ValueError(
            "H3_FULL_VAE_TILE_BATCH_SIZE must be an integer, got "
            f"{raw_batch_size!r}"
        ) from exc

    original_run_tile_tasks = getattr(video_vae, "_run_tile_tasks", None)
    if tile_batch_size <= 0 or not callable(original_run_tile_tasks):
        yield
        return

    def run_tile_tasks(tiles, tile_indices, forward_fn, stack_tiling, cls_agg=None):
        if not stack_tiling or len(tile_indices) <= tile_batch_size:
            return original_run_tile_tasks(
                tiles, tile_indices, forward_fn, stack_tiling, cls_agg
            )

        outputs = []
        for start in range(0, len(tile_indices), tile_batch_size):
            chunk_indices = tile_indices[start : start + tile_batch_size]
            outputs.extend(
                original_run_tile_tasks(
                    tiles, chunk_indices, forward_fn, True, cls_agg
                )
            )
        return outputs

    video_vae._run_tile_tasks = run_tile_tasks
    try:
        yield
    finally:
        video_vae._run_tile_tasks = original_run_tile_tasks


class RTX4090MiniMaxH3DenoisingStage(MiniMaxH3DenoisingStage):
    """Keep explicit DiT layerwise offload after compile warmup."""

    def _maybe_offload_during_compile(self, module: object) -> None:
        requested = set(self.server_args.layerwise_offload_components or ())
        if requested.intersection({"dit", "transformer"}):
            return
        super()._maybe_offload_during_compile(module)


class RTX4090MiniMaxH3DecodingStage(MiniMaxH3DecodingStage):
    """Use one post-denoise H2D transfer for the complete video VAE."""

    @contextlib.contextmanager
    def use_declared_component(self, *, component_name: str, module):
        with super().use_declared_component(
            component_name=component_name,
            module=module,
        ) as selected:
            if component_name != "video_vae" or selected is None:
                yield selected
                return
            with (
                _full_video_vae_residency(selected),
                _bounded_video_vae_tile_batch(selected),
            ):
                yield selected

    def _get_vae_decode_fn(
        self,
        vae,
        server_args: ServerArgs,
        *,
        decode_fn=None,
        compiled_callable=None,
    ):
        if (
            vae is self.video_vae
            and _env_flag("H3_FULL_VAE_AFTER_DENOISE")
            and not _env_flag("H3_FULL_VAE_TORCH_COMPILE")
        ):
            return decode_fn or vae.decode
        return super()._get_vae_decode_fn(
            vae,
            server_args,
            decode_fn=decode_fn,
            compiled_callable=compiled_callable,
        )


class MiniMaxH3RTX4090Pipeline(MiniMaxH3Pipeline):
    """Stock MiniMax-H3 wiring with two RTX-4090 memory-stage overrides."""

    def create_pipeline_stages(self, server_args: ServerArgs) -> None:
        release_metadata = getattr(self, "release_metadata", None)
        sigma_shift_scales = (
            release_metadata.sigma_shift_scales
            if release_metadata is not None
            else None
        )
        self.add_stage(InputValidationStage())
        if release_metadata is not None:
            self.add_stage(MiniMaxH3PartitionAdmissionStage(release_metadata))
        self.add_stage(
            MiniMaxH3TextEncodingStage(
                text_encoder=self.get_module("text_encoder"),
                tokenizer=self.get_module("tokenizer"),
                processor=self.get_module("processor"),
            )
        )
        self.add_stage(
            MiniMaxH3VisualEncodingStage(
                video_vae=self.get_module("video_vae"),
                vae_arch_config=server_args.pipeline_config.vae_config.arch_config,
            )
        )
        self.add_stage(
            MiniMaxH3AudioEncodingStage(
                audio_vae=self.get_module("audio_vae"),
                vae_arch_config=server_args.pipeline_config.audio_vae_config.arch_config,
            )
        )
        self.add_stage(MiniMaxH3LatentPreparationStage())
        self.add_stage(
            MiniMaxH3TimestepPreparationStage(
                sigma_shift_scales=sigma_shift_scales,
            )
        )
        self.add_stage(
            RTX4090MiniMaxH3DenoisingStage(
                transformer=self.get_module("transformer"),
                pipeline=self,
            )
        )
        self.add_stage(
            RTX4090MiniMaxH3DecodingStage(
                video_vae=self.get_module("video_vae"),
                audio_vae=self.get_module("audio_vae"),
            )
        )


EntryClass = MiniMaxH3RTX4090Pipeline


__all__ = [
    "MiniMaxH3RTX4090Pipeline",
    "RTX4090MiniMaxH3DecodingStage",
    "RTX4090MiniMaxH3DenoisingStage",
]
