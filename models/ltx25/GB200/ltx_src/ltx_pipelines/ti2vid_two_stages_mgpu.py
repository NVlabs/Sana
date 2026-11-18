"""Multi-GPU two-stage text/image-to-video runner.
Runs :class:`TI2VidTwoStagesPipeline` across multiple GPUs with:
- **Stage 1** -- sequence parallelism (SP)
- **Stage 2** -- tiled data parallelism (TDP) on height + width with overlap
- **Gemma** -- Accelerate-based parallelization
- **VAE** -- distributed decoding
Requires ``ltx-kernels`` to be installed (transitive via SP builder).
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable, Iterator, Sequence
from multiprocessing import SimpleQueue

import torch
import torch.distributed as dist

from ltx_core.allocator_trim_strategy import AllocatorTrimStrategy
from ltx_core.components.guiders import MultiModalGuiderParams
from ltx_core.loader import LTXV_LORA_COMFY_RENAMING_MAP, LoraPathStrengthAndSDOps
from ltx_core.loader.registry import ModelRegistry
from ltx_core.model.transformer.compiling import CompilationConfig
from ltx_core.model.video_vae import get_video_chunks_number
from ltx_core.model.video_vae.transformer import DiffVAEMode
from ltx_core.multigpu.transformer.attention import AttentionManager
from ltx_core.quantization import QuantizationPolicy
from ltx_core.quantization.fp8_cast import build_policy as _build_fp8_cast_policy
from ltx_core.tiling import DimensionTilingConfig, TileCountConfig, balanced_tile_split
from ltx_pipelines.multigpu.controller import MGPUController
from ltx_pipelines.multigpu.gemma_builders import AccelerateGemmaBuilder
from ltx_pipelines.multigpu.runner import MGPURunner
from ltx_pipelines.multigpu.sp_builder import SequenceParallelBuilder
from ltx_pipelines.multigpu.tdp_builder import TiledDataParallelBuilder
from ltx_pipelines.multigpu.vae_builders import DistributedDecoderBuilder
from ltx_pipelines.multigpu.weight_tracker import TransformerWeightTracker
from ltx_pipelines.ti2vid_two_stages import TI2VidTwoStagesPipeline
from ltx_pipelines.utils.constants import TDP_DISTILLED_SIGMAS
from ltx_pipelines.utils.media_io import HDRColorSpace, encode_video, resolve_hdr_color_space, vae_dtype_for_hdr
from ltx_pipelines.utils.model_paths import ModelPaths
from ltx_pipelines.utils.opt_stack import begin_request as begin_opt_request
from ltx_pipelines.utils.opt_timing import begin_request as begin_timed_request
from ltx_pipelines.utils.opt_timing import end_request as end_timed_request
from ltx_pipelines.utils.opt_timing import time_output_call
from ltx_pipelines.utils.types import DEFAULT_AUTO_DURATION, AutoDuration

logger = logging.getLogger(__name__)

# Stage 1 at 512x768, 121 frames = 6144 video tokens + audio tokens.
_DEFAULT_SP_MAX_TOKENS = 32768
# Rank that collects distributed-VAE tiles and encodes the assembled video.
_DRIVER_RANK = 0


class TI2VidTwoStagesRunner(MGPURunner):
    """Distributed :class:`TI2VidTwoStagesPipeline`: SP stage 1 + TDP stage 2 + Gemma + distributed VAE."""

    @torch.inference_mode()
    def setup(
        self,
        *,
        model_paths: ModelPaths,
        prompt_enhancer_gemma_root: str | None = None,
        spatial_upsampler_path: str,
        vae_queue: SimpleQueue,
        distilled_lora_path: str,
        compilation_config: CompilationConfig | None = None,
        sp_max_tokens: int = _DEFAULT_SP_MAX_TOKENS,
        quantization: Callable[[], QuantizationPolicy] | None = None,
        diffvae_optimization: DiffVAEMode = DiffVAEMode.CHUNKED_EAGER,
    ) -> None:
        # quantization is a picklable zero-arg builder (built per worker, post-spawn); default fp8-cast.
        quantization_policy = (
            quantization() if quantization is not None else _build_fp8_cast_policy(model_paths.transformer())
        )
        distilled_lora = [LoraPathStrengthAndSDOps(distilled_lora_path, 1.0, LTXV_LORA_COMFY_RENAMING_MAP)]
        registry = ModelRegistry()
        pipeline = TI2VidTwoStagesPipeline(
            model_paths=model_paths,
            distilled_lora=distilled_lora,
            spatial_upsampler_path=spatial_upsampler_path,
            prompt_enhancer_gemma_root=prompt_enhancer_gemma_root,
            loras=[],
            registry=registry,
            quantization=quantization_policy,
            compilation_config=compilation_config,
            alloc_trim_strategy=AllocatorTrimStrategy.DEFER,
            diffvae_optimization=diffvae_optimization,
        )
        tracker = TransformerWeightTracker(group=self.groups.transformer_group)

        # Stage 1: sequence parallelism.
        model_cfg = pipeline.stage_1._transformer_builder.model_config().get("transformer", {})
        attn_mgr = AttentionManager(
            max_tokens=sp_max_tokens,
            num_heads=model_cfg["num_attention_heads"],
            head_dim=model_cfg["attention_head_dim"],
            tensor_dtype=pipeline.dtype,
            group=self.groups.transformer_group,
        )
        # Stage-1 parallelism: sequence (vendor default) or guidance-batch (ours).
        #
        # SP splits the sequence and pays an All2All at every attention layer. CFG
        # parallelism splits the guidance batch that _guided_denoise already assembles,
        # leaving one all_gather per step as the only communication. Env-selected so one
        # build serves both arms of a comparison.
        #
        # REQUIRES the guidance batch to arrive intact -- see the max_batch_size override
        # in utils/blocks.py under this same switch. Without it the batch is already
        # split to 1 before the builder sees it, every rank redundantly computes every
        # guidance pass, and the arm silently measures "SP turned off" instead.
        if os.environ.get("LTX_S1_PARALLEL") == "cfg":
            from ltx_pipelines.multigpu.cfgp_builder import CFGParallelBuilder

            pipeline.stage_1._transformer_builder = CFGParallelBuilder(
                inner=pipeline.stage_1._transformer_builder,
                group=self.groups.transformer_group,
                # the same registry and tracker the SP branch receives: without them the
                # builder cannot report GPU-resident weights, and the vendor's capture /
                # reduce-overhead paths abort
                registry=registry,
                tracker=tracker,
            )
        else:
            pipeline.stage_1._transformer_builder = SequenceParallelBuilder(
                inner=pipeline.stage_1._transformer_builder,
                attn_mgr=attn_mgr,
                registry=registry,
                tracker=tracker,
            )

        # Stage 2: tiled data parallelism -- balanced 2D spatial grid over the group (one tile/rank).
        # height takes the smaller factor of world_size, width the larger; size-aware split is a follow-up.
        tdp_height_tiles, tdp_width_tiles = balanced_tile_split(dist.get_world_size(self.groups.transformer_group))
        tdp_tiling = TileCountConfig(
            height=DimensionTilingConfig(num_tiles=tdp_height_tiles, overlap=5),
            width=DimensionTilingConfig(num_tiles=tdp_width_tiles, overlap=5),
        )
        pipeline.stage_2._transformer_builder = TiledDataParallelBuilder(
            inner=pipeline.stage_2._transformer_builder,
            group=self.groups.transformer_group,
            tiling=tdp_tiling,
            registry=registry,
            tracker=tracker,
        )

        # Accelerate Gemma parallelization. Capture shared-vs-separate before replacing
        # the encode builder so a shared alias is re-bound to the new instance.
        pe = pipeline.prompt_encoder
        separate_enhancer = pe._enhancer_text_encoder_builder is not pe._text_encoder_builder
        pe._text_encoder_builder = AccelerateGemmaBuilder(
            gemma_root_path=model_paths.text_encoder(),
            gemma_group=self.groups.gemma_group,
            broadcast_group=self.groups.transformer_group,
            registry=registry,
            src_rank=_DRIVER_RANK,
            dtype=pipeline.dtype,
        )
        if separate_enhancer:
            assert prompt_enhancer_gemma_root is not None
            pe._enhancer_text_encoder_builder = AccelerateGemmaBuilder(
                gemma_root_path=prompt_enhancer_gemma_root,
                gemma_group=self.groups.gemma_group,
                broadcast_group=self.groups.transformer_group,
                registry=registry,
                src_rank=_DRIVER_RANK,
                dtype=pipeline.dtype,
            )
        else:
            pe._enhancer_text_encoder_builder = pe._text_encoder_builder

        # Distributed VAE decoding: balanced 2D spatial grid over the group (one tile/rank).
        # height takes the smaller factor of world_size, width the larger; size-aware split is a follow-up.
        vae_height_tiles, vae_width_tiles = balanced_tile_split(dist.get_world_size(self.groups.vae_group))
        vae_tiling = TileCountConfig(
            height=DimensionTilingConfig(num_tiles=vae_height_tiles, overlap=4),
            width=DimensionTilingConfig(num_tiles=vae_width_tiles, overlap=4),
        )
        pipeline.video_decoder._decoder_builder = DistributedDecoderBuilder(
            inner=pipeline.video_decoder._decoder_builder,
            queue=vae_queue,
            vae_group=self.groups.vae_group,
            vae_tiling=vae_tiling,
            driver_rank=_DRIVER_RANK,
            registry=registry,
        )

        self._pipeline = pipeline

    @torch.inference_mode()
    def __call__(  # noqa: PLR0913
        self,
        *,
        output_path: str,
        prompt: str,
        negative_prompt: str,
        seed: int,
        height: int,
        width: int,
        frame_rate: float,
        num_inference_steps: int,
        video_guider_params: MultiModalGuiderParams,
        audio_guider_params: MultiModalGuiderParams,
        num_frames: int | AutoDuration = DEFAULT_AUTO_DURATION,
        images: list | None = None,
        enhance_prompt: bool = False,
        enhance_static_cache: bool = False,
        hdr: HDRColorSpace | None = None,
        generated_keyframes: int | Sequence[int] = 0,
    ) -> Iterator[str | None]:
        # The pipeline raises ValueError on invalid input (symmetric across ranks); the controller
        # catches that and turns it into a recoverable RunnerError. Anything else is fatal.
        begin_opt_request()
        begin_timed_request(num_inference_steps)
        hdr = resolve_hdr_color_space(images=images or [], hdr=hdr)
        vae_dtype = vae_dtype_for_hdr(hdr, torch.bfloat16)
        video, audio, num_frames, tiling_config = self._pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt,
            seed=seed,
            height=height,
            width=width,
            num_frames=num_frames,
            frame_rate=frame_rate,
            num_inference_steps=num_inference_steps,
            video_guider_params=video_guider_params,
            audio_guider_params=audio_guider_params,
            images=images or [],
            vae_dtype=vae_dtype,
            color_space=hdr,
            tiling_config=None,
            enhance_prompt=enhance_prompt,
            enhance_static_cache=enhance_static_cache,
            generated_keyframes=generated_keyframes,
            stage_2_sigmas=TDP_DISTILLED_SIGMAS,
        )
        if dist.get_rank() != _DRIVER_RANK:
            end_timed_request()
            yield None  # workers: nothing to encode
            return
        time_output_call(
            encode_video,
            video=video,
            fps=frame_rate,
            audio=audio,
            output_path=output_path,
            video_chunks_number=get_video_chunks_number(num_frames, tiling_config),
            color_space=hdr,
        )
        end_timed_request()
        yield output_path


if __name__ == "__main__":
    from ltx_pipelines.utils.args import (
        add_generated_keyframes_arg,
        default_2_stage_arg_parser,
        resolve_cli_params,
    )

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    params = resolve_cli_params()
    parser = add_generated_keyframes_arg(default_2_stage_arg_parser(params=params, supports_auto_duration=True))
    args = parser.parse_args()

    vae_queue = torch.multiprocessing.get_context("spawn").SimpleQueue()
    controller = MGPUController(TI2VidTwoStagesRunner)
    controller.start(
        model_paths=args.model_paths,
        prompt_enhancer_gemma_root=args.prompt_enhancer_gemma_root,
        spatial_upsampler_path=args.spatial_upsampler_path,
        vae_queue=vae_queue,
        distilled_lora_path=args.distilled_lora[0].path,
        compilation_config=args.compile,
        diffvae_optimization=args.diffvae_optimization,
    )
    try:
        for _ in controller.stream(
            output_path=args.output_path,
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            seed=args.seed,
            height=args.height,
            width=args.width,
            num_frames=args.num_frames,
            frame_rate=args.frame_rate,
            num_inference_steps=args.num_inference_steps,
            video_guider_params=MultiModalGuiderParams(
                cfg_scale=args.video_cfg_guidance_scale,
                stg_scale=args.video_stg_guidance_scale,
                rescale_scale=args.video_rescale_scale,
                modality_scale=args.a2v_guidance_scale,
                skip_step=args.video_skip_step,
                stg_blocks=args.video_stg_blocks,
            ),
            audio_guider_params=MultiModalGuiderParams(
                cfg_scale=args.audio_cfg_guidance_scale,
                stg_scale=args.audio_stg_guidance_scale,
                rescale_scale=args.audio_rescale_scale,
                modality_scale=args.v2a_guidance_scale,
                skip_step=args.audio_skip_step,
                stg_blocks=args.audio_stg_blocks,
            ),
            images=args.images,
            enhance_prompt=args.enhance_prompt,
            enhance_static_cache=args.enhance_static_cache,
            hdr=args.hdr,
            generated_keyframes=args.num_generated_keyframes,
        ):
            pass  # drive the job to completion; the runner writes the file as a side effect
    finally:
        controller.shutdown()
