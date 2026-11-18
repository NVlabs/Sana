"""Pipeline blocks — each block owns its model lifecycle.
Blocks build a model on each ``__call__``, use it, then free GPU memory.
This eliminates manual ``del model; cleanup_memory()`` in pipelines: each
block is self-contained, so no central model-coordinator object is needed.
"""

from __future__ import annotations

import copy
import logging
import os
import sys
from collections.abc import Iterator, Sequence
from contextlib import AbstractContextManager, contextmanager
from dataclasses import replace
from typing import Callable, TypeVar

import torch

from ltx_core.allocator_trim_strategy import AllocatorTrimStrategy
from ltx_core.batch_split import BatchSplitAdapter
from ltx_core.block_streaming import DISK_CPU_SLOTS, StreamingModelBuilder
from ltx_core.components.diffusion_steps import EulerDiffusionStep
from ltx_core.components.noisers import Noiser
from ltx_core.components.patchifiers import AudioPatchifier, VideoLatentPatchifier
from ltx_core.components.protocols import DiffusionStepProtocol
from ltx_core.conditioning import VideoGeneratedKeyframeSlots
from ltx_core.duration_head import (
    DURATION_HEAD_KEY_OPS,
    DurationHead,
    DurationHeadConfigurator,
)
from ltx_core.loader import SDOps
from ltx_core.loader.attention_ops import set_attention_module_op
from ltx_core.loader.fuse_loras import bf16_fuse_rule
from ltx_core.loader.module_ops import ModuleOps
from ltx_core.loader.primitives import BuilderProtocol, LoraPathStrengthAndSDOps, ModelBuilderProtocol
from ltx_core.loader.registry import ModelRegistry, Registry
from ltx_core.loader.single_gpu_model_builder import SingleGPUModelBuilder as Builder
from ltx_core.model.audio_vae import (
    AUDIO_VAE_DECODER_COMFY_KEYS_FILTER,
    AUDIO_VAE_ENCODER_COMFY_KEYS_FILTER,
    VOCODER_COMFY_KEYS_FILTER,
    AudioDecoderConfigurator,
    AudioEncoderConfigurator,
    VocoderConfigurator,
)
from ltx_core.model.audio_vae import (
    decode_audio as vae_decode_audio,
)
from ltx_core.model.model_protocol import LTXModelProtocol, ModelConfigurator
from ltx_core.model.transformer import (
    LTXV_MODEL_COMFY_RENAMING_MAP,
    LTXModelConfigurator,
    X0Model,
)
from ltx_core.model.transformer.attention import (
    AttentionCallable,
    AttentionFunction,
)
from ltx_core.model.transformer.compiling import (
    CompilationConfig,
    build_compile_transformer_op,
    modify_sd_ops_for_compilation,
)
from ltx_core.model.upsampler import LatentUpsamplerConfigurator, upsample_video
from ltx_core.model.video_vae import (
    CHANNELS_LAST_3D_WEIGHTS,
    MEMORY_EFFICIENT_DECODE,
    VAE_DECODER_COMFY_KEYS_FILTER,
    VAE_ENCODER_COMFY_KEYS_FILTER,
    TilingConfig,
    VideoDecoderConfigurator,
    VideoEncoder,
    VideoEncoderConfigurator,
    is_diffusion_video_vae,
    video_decoder_sd_ops_for_checkpoint,
)
from ltx_core.model.video_vae.transformer import (
    DiffVAEMode,
    NAttentionKind,
    build_diffvae_mode_op,
    na_dsl_available,
)
from ltx_core.quantization import QuantizationPolicy, fp8_cast_fuse_rule
from ltx_core.text_encoders.gemma import (
    EMBEDDINGS_PROCESSOR_KEY_OPS,
    EmbeddingsProcessorConfigurator,
    GemmaTextEncoderConfigurator,
    gemma_model_type,
    get_gemma_ops,
    resolve_gemma_weight_paths,
)
from ltx_core.text_encoders.gemma.embeddings_processor import EmbeddingsProcessor, EmbeddingsProcessorOutput
from ltx_core.tools import AudioLatentTools, LatentTools, VideoLatentTools
from ltx_core.types import (
    VIDEO_SCALE_FACTORS,
    Audio,
    AudioLatentShape,
    LatentState,
    SpatioTemporalScaleFactors,
    VideoLatentShape,
    VideoPixelShape,
)
from ltx_pipelines.utils.args import ImageConditioningInput
from ltx_pipelines.utils.constants import detect_params
from ltx_pipelines.utils.gpu_model import gpu_model
from ltx_pipelines.utils.helpers import (
    cleanup_memory,
    create_noised_state,
    generate_enhanced_prompt,
    seconds_to_clamped_num_frames,
)
from ltx_pipelines.utils.model_paths import ModelPaths
from ltx_pipelines.utils.samplers import euler_denoising_loop
from ltx_pipelines.utils.types import AutoDuration, Denoiser, ModalitySpec, OffloadMode

_ENCODE_MODEL_TYPES = frozenset({"gemma3", "gemma4", "gemma4_unified"})

logger = logging.getLogger(__name__)

T = TypeVar("T")
_M = TypeVar("_M", bound=torch.nn.Module)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _chain_quantization(
    sd_ops: SDOps,
    module_ops: tuple[ModuleOps, ...],
    quantization: QuantizationPolicy,
) -> tuple[SDOps, tuple[ModuleOps, ...]]:
    chained_sd_ops = sd_ops
    if quantization.sd_ops is not None:
        chained_sd_ops = SDOps(
            name=f"sd_ops_chain_{sd_ops.name}+{quantization.sd_ops.name}",
            mapping=(*sd_ops.mapping, *quantization.sd_ops.mapping),
        )
    return chained_sd_ops, (*module_ops, *quantization.module_ops)


def _apply_compile_ops(
    sd_ops: SDOps,
    module_ops: tuple[ModuleOps, ...],
    loras: tuple[LoraPathStrengthAndSDOps, ...],
    number_of_layers: int,
    compilation_config: CompilationConfig,
) -> tuple[SDOps, tuple[ModuleOps, ...], tuple[LoraPathStrengthAndSDOps, ...]]:
    """Rewrite sd_ops/module_ops/LoRAs for compiled blocks (params land under ``_orig_mod``)."""
    sd_ops = modify_sd_ops_for_compilation(sd_ops, number_of_layers)
    compile_op = build_compile_transformer_op(compilation_config)
    module_ops = (*module_ops, compile_op)
    loras = tuple(
        LoraPathStrengthAndSDOps(
            lora.path,
            lora.strength,
            modify_sd_ops_for_compilation(lora.sd_ops, number_of_layers),
        )
        for lora in loras
    )
    return sd_ops, module_ops, loras


def _ensure_cudagraph_compatible_builder(
    builder: ModelBuilderProtocol[LTXModelProtocol], compilation_config: CompilationConfig | None
) -> None:
    """CUDA-graph compile paths need GPU-resident weights across dispose/rebuild.
    Covers ``capture=True`` and inductor cudagraph modes (``reduce-overhead`` /
    ``max-autotune``). Checked lazily at prepare/build time so MGPU runners can
    wrap the stage's builder with SP/TDP (+ weight tracker) after construction.
    """
    if compilation_config is None:
        return
    cudagraph_modes = frozenset({"reduce-overhead", "max-autotune"})
    needs_resident = compilation_config.capture or compilation_config.mode in cudagraph_modes
    if not needs_resident or builder.keeps_gpu_resident_weights:
        return
    raise ValueError(
        "CompilationConfig capture / mode=reduce-overhead|max-autotune requires a builder "
        "with keeps_gpu_resident_weights=True (SequenceParallelBuilder / "
        "TiledDataParallelBuilder with a weight tracker, or StreamingModelBuilder). "
        "Plain SingleGPUModelBuilder reloads weights onto fresh GPU storages each stage, "
        "which invalidates CUDA graphs (or repeatedly recaptures until Dynamo gives up)."
    )


@contextmanager
def _streaming_model(
    builder: StreamingModelBuilder,
    target_device: torch.device,
    dtype: torch.dtype,
    alloc_trim_strategy: AllocatorTrimStrategy = AllocatorTrimStrategy.TRIM,
) -> Iterator:
    """Build a streaming wrapper, yield it, then tear down and free memory.
    The builder's own ``cpu_slots_count`` selects RAM vs disk streaming.
    ``teardown()`` always runs -- it releases non-memory resources (forward
    hooks, the disk I/O worker thread, open file handles) that GC would not
    reclaim promptly. ``dispose`` always metas parameter storage; ``TRIM``
    also runs ``cleanup_memory()``, while ``DEFER`` leaves the CUDA cache warm.
    """
    wrapped = builder.build(device=target_device, dtype=dtype)
    try:
        yield wrapped
    finally:
        wrapped.teardown()
        wrapped.dispose()
        if alloc_trim_strategy == AllocatorTrimStrategy.TRIM:
            cleanup_memory()


def _build_state(
    spec: ModalitySpec,
    tools: LatentTools,
    noiser: Noiser,
    dtype: torch.dtype,
    device: torch.device,
) -> LatentState:
    """Create a noised latent state from a modality spec and tools."""
    state = create_noised_state(
        tools=tools,
        conditionings=spec.conditionings,
        noiser=noiser,
        dtype=dtype,
        device=device,
        noise_scale=spec.noise_scale,
        initial_latent=spec.initial_latent,
    )
    if spec.frozen:
        state = replace(
            state,
            denoise_mask=torch.zeros_like(state.denoise_mask),
            frozen=True,
        )
    return state


def _cleanup_iter(
    it: Iterator[torch.Tensor],
    model: torch.nn.Module,
    alloc_trim_strategy: AllocatorTrimStrategy = AllocatorTrimStrategy.TRIM,
) -> Iterator[torch.Tensor]:
    """Wrap an iterator to release *model* memory (per ``alloc_trim_strategy``) once exhausted or abandoned."""
    with gpu_model(model, alloc_trim_strategy=alloc_trim_strategy):
        yield from it


# ---------------------------------------------------------------------------
# DiffusionStage
# ---------------------------------------------------------------------------


class DiffusionStage:
    """Owns transformer lifecycle. Builds on each call, frees on exit.
    Replaces the manual build-transformer / ``del transformer`` pattern that
    every pipeline previously repeated.
    """

    def __init__(
        self,
        transformer_builder: ModelBuilderProtocol[LTXModelProtocol],
        dtype: torch.dtype,
        device: torch.device,
        *,
        quantization: QuantizationPolicy | None = None,
        compilation_config: CompilationConfig | None = None,
        alloc_trim_strategy: AllocatorTrimStrategy = AllocatorTrimStrategy.TRIM,
        scale_factors: SpatioTemporalScaleFactors = VIDEO_SCALE_FACTORS,
    ) -> None:
        """Construct a stage from a single pre-built transformer ``builder``.
        Holds only that builder plus build-time configuration (dtype, device,
        quantization, compilation). Turning a checkpoint path + LoRA set into a
        builder -- and choosing a :class:`StreamingModelBuilder` when offloading --
        lives in :meth:`from_checkpoint`, which is how pipelines normally create a
        stage. A :class:`StreamingModelBuilder` selects the block-streaming build
        path; any other builder uses the standard (all-on-GPU) path.
        ``quantization`` and ``compilation_config`` are applied lazily by
        :meth:`_prepared_builder` on both the standard and streaming paths.
        ``scale_factors`` are the video VAE's spatiotemporal downscaling factors,
        defaulting to the 32x32x8 layout.
        """
        self._transformer_builder = transformer_builder
        self._dtype = dtype
        self._device = device
        self._quantization = quantization
        self._compilation_config = compilation_config
        self._alloc_trim_strategy = alloc_trim_strategy
        self.video_scale_factors = scale_factors

    @classmethod
    def from_checkpoint(  # noqa: PLR0913
        cls,
        checkpoint_path: str,
        dtype: torch.dtype,
        device: torch.device,
        loras: tuple[LoraPathStrengthAndSDOps, ...] = (),
        quantization: QuantizationPolicy | None = None,
        registry: Registry | None = None,
        compilation_config: CompilationConfig | None = None,
        alloc_trim_strategy: AllocatorTrimStrategy = AllocatorTrimStrategy.TRIM,
        offload_mode: OffloadMode = OffloadMode.NONE,
        model_configurator: type[ModelConfigurator] = LTXModelConfigurator,
        model_sd_ops: SDOps = LTXV_MODEL_COMFY_RENAMING_MAP,
        scale_factors: SpatioTemporalScaleFactors = VIDEO_SCALE_FACTORS,
    ) -> "DiffusionStage":
        """Build a stage from a checkpoint path and LoRA set.
        Constructs a single transformer builder from ``checkpoint_path`` +
        ``loras`` and delegates to ``__init__``. When ``offload_mode != OffloadMode.NONE``
        that builder is a :class:`StreamingModelBuilder` (fuse rule +
        ``cpu_slots_count`` for the mode); otherwise it is the standard single-GPU
        builder. Quantization and compilation stay on the stage and are applied
        lazily at build time. This is the high-level entry point used by pipelines;
        ``__init__`` itself takes an already-built builder.
        ``model_configurator`` / ``model_sd_ops`` let callers (e.g. the audio-only
        T2A pipeline) override the model class configurator and the state-dict key
        mapping. A quantization policy that pins its own configurator takes
        precedence over ``model_configurator``.
        """
        # A quantization policy may pin its own configurator; otherwise use the one
        # provided by the caller (defaults to the audio-video LTXModelConfigurator).
        configurator = (
            quantization.model_configurator
            if quantization is not None and quantization.model_configurator is not None
            else model_configurator
        )

        transformer_builder: ModelBuilderProtocol[LTXModelProtocol]
        if offload_mode == OffloadMode.NONE:
            transformer_builder = Builder(
                model_path=checkpoint_path,
                model_class_configurator=configurator,
                model_sd_ops=model_sd_ops,
                loras=tuple(loras),
                registry=registry or ModelRegistry(cache_models=True, cache_weights=False),
            )
        else:
            transformer_builder = cls._build_streaming_builder(
                checkpoint_path=checkpoint_path,
                configurator=configurator,
                model_sd_ops=model_sd_ops,
                loras=tuple(loras),
                quantization=quantization,
                registry=registry or ModelRegistry(cache_models=True, cache_weights=False),
                offload_mode=offload_mode,
            )

        return cls(
            transformer_builder,
            dtype,
            device,
            quantization=quantization,
            compilation_config=compilation_config,
            alloc_trim_strategy=alloc_trim_strategy,
            scale_factors=scale_factors,
        )

    @staticmethod
    def _build_streaming_builder(
        *,
        checkpoint_path: str,
        configurator: type[ModelConfigurator],
        model_sd_ops: SDOps,
        loras: tuple[LoraPathStrengthAndSDOps, ...],
        quantization: QuantizationPolicy | None,
        registry: Registry,
        offload_mode: OffloadMode,
    ) -> StreamingModelBuilder:
        """Construct the streaming transformer builder for an offloading stage.
        Returns a :class:`StreamingModelBuilder` with the checkpoint path, LoRAs,
        fuse rule (bf16 or fp8_cast), and ``cpu_slots_count`` for ``offload_mode``.
        Compilation / quantization sd_ops and module_ops are not applied here --
        the stage applies them lazily via :meth:`_prepared_builder`.
        """
        # WeightsProvider currently only supports plain bf16 + fp8_cast LoRA fusion
        # (no companion-key emission). Quantization policies that emit
        # companion keys (e.g. ``.weight_scale``) cannot be streamed yet.
        if quantization is not None and quantization.fuse_rule is not fp8_cast_fuse_rule:
            raise ValueError(
                "Block streaming is not supported with this quantization policy "
                "(only bf16 and fp8_cast are currently supported)."
            )
        return StreamingModelBuilder(
            model_class_configurator=configurator,
            model_path=checkpoint_path,
            model_sd_ops=model_sd_ops,
            loras=loras,
            registry=registry,
            fuse_rule=quantization.fuse_rule if quantization is not None else bf16_fuse_rule,
            blocks_attr="transformer_blocks",
            blocks_prefix="transformer_blocks",
            cpu_slots_count=DISK_CPU_SLOTS if offload_mode == OffloadMode.DISK else None,
        )

    @property
    def supports_generated_keyframes(self) -> bool:
        """Whether this stage's checkpoint was trained with the keyframe absolute-position embedding.
        Read from the checkpoint's transformer config, so it is answered before any weights are
        built. Generated keyframe slots on a checkpoint without it would be denoised as unmarked
        tokens while still costing a full latent frame of tokens each, so pipelines refuse rather
        than silently degrade.
        """
        transformer_config = self._transformer_builder.model_config().get("transformer", {})
        return bool(transformer_config.get("use_keyframes_abs_pos_embedding", False))

    def assert_generated_keyframes_supported(self) -> None:
        """Raise unless this stage's checkpoint declares the keyframe embedding.
        Call this from a pipeline's ``__call__`` preamble, before any model is built: the answer
        comes from the checkpoint config alone, so an unsupported request should fail immediately
        rather than after the text encoder has been loaded. :meth:`__call__` re-checks as a backstop
        for callers that build conditioning items directly.
        """
        if self.supports_generated_keyframes:
            return
        raise ValueError(
            f"Generated keyframe slots were requested, but the checkpoint at "
            f"{self._transformer_builder.checkpoint} does not set 'use_keyframes_abs_pos_embedding' "
            f"in its transformer config, so it has no keyframe absolute-position embedding. Use a "
            f"generated-keyframe checkpoint or drop the keyframe request."
        )

    def _assert_supports_conditionings(self, video: ModalitySpec | None) -> None:
        if video is None or self.supports_generated_keyframes:
            return
        if any(isinstance(item, VideoGeneratedKeyframeSlots) for item in video.conditionings):
            self.assert_generated_keyframes_supported()

    def with_attention(self, attention: AttentionFunction | AttentionCallable | None) -> "DiffusionStage":
        """Return a new ``DiffusionStage`` that pins the transformer build to ``attention``.
        Functional: never mutates ``self``. The returned stage shares all other
        configuration with the original; only the underlying builders' ``module_ops``
        gain a ``set_attention_module_op(attention)`` entry so subsequent transformer
        builds use that kernel. ``attention=None`` is a no-op (returns ``self``).
        """
        if attention is None:
            return self
        op = set_attention_module_op(attention)
        new = copy.copy(self)
        new._transformer_builder = self._transformer_builder.with_module_ops(
            (*self._transformer_builder.module_ops, op),
        )
        return new

    def with_builder(self, builder: ModelBuilderProtocol[LTXModelProtocol]) -> "DiffusionStage":
        """Return a new ``DiffusionStage`` that builds its transformer from ``builder``.
        Functional: never mutates ``self``; shares all other configuration (dtype, device,
        quantization, compilation). Affects the standard (non-offload) build path.
        """
        new = copy.copy(self)
        new._transformer_builder = builder
        return new

    def with_loras(self, loras: tuple[LoraPathStrengthAndSDOps, ...]) -> "DiffusionStage":
        """Return a new ``DiffusionStage`` built with exactly ``loras`` (replacing the current set)."""
        return self.with_builder(self._transformer_builder.with_loras(loras))

    def _prepared_builder(self) -> ModelBuilderProtocol[LTXModelProtocol]:
        """Return the configured builder with the stage's build-time ops applied.
        Compilation and quantization live on the stage (not on the builder) and are
        applied here, lazily, for both the standard and streaming paths. This keeps
        the builder holding only raw sd_ops/module_ops/LoRAs, so ``with_loras`` /
        ``with_builder`` swap them consistently regardless of the build path. The
        returned copy preserves the builder's concrete type (e.g. a
        ``StreamingModelBuilder`` stays one).
        """
        builder = self._transformer_builder
        _ensure_cudagraph_compatible_builder(builder, self._compilation_config)
        sd_ops = builder.model_sd_ops
        module_ops = builder.module_ops
        loras = builder.loras
        if self._quantization is not None:
            sd_ops, module_ops = _chain_quantization(sd_ops, module_ops, self._quantization)
            builder = builder.with_fuse_rule(self._quantization.fuse_rule)
        if self._compilation_config is not None:
            number_of_layers = builder.model_config()["transformer"]["num_layers"]
            sd_ops, module_ops, loras = _apply_compile_ops(
                sd_ops, module_ops, loras, number_of_layers, self._compilation_config
            )
        return builder.with_module_ops(module_ops).with_sd_ops(sd_ops).with_loras(loras)

    def _build_transformer(self, *, device: torch.device | None = None, **kwargs: object) -> X0Model:
        target = device or self._device
        return X0Model(self._prepared_builder().build(device=target, **kwargs)).to(target).eval()

    @property
    def _is_streaming(self) -> bool:
        """Whether the configured builder uses the block-streaming build path."""
        return isinstance(self._transformer_builder, StreamingModelBuilder)

    @contextmanager
    def _streaming_transformer_ctx(self) -> Iterator[X0Model]:
        builder = self._prepared_builder()
        assert isinstance(builder, StreamingModelBuilder)
        with _streaming_model(builder, self._device, self._dtype, self._alloc_trim_strategy) as streaming_wrapper:
            yield X0Model(streaming_wrapper).eval()

    def _transformer_ctx(self, **kwargs: object) -> AbstractContextManager:
        if self._is_streaming:
            return self._streaming_transformer_ctx()
        return gpu_model(self._build_transformer(**kwargs), alloc_trim_strategy=self._alloc_trim_strategy)

    def __call__(  # noqa: PLR0913
        self,
        denoiser: Denoiser,
        sigmas: torch.Tensor,
        noiser: Noiser,
        width: int,
        height: int,
        frames: int,
        fps: float,
        video: ModalitySpec | None = None,
        audio: ModalitySpec | None = None,
        stepper: DiffusionStepProtocol | None = None,
        loop: Callable[..., tuple[LatentState | None, LatentState | None]] | None = None,
        max_batch_size: int = 1,
    ) -> tuple[LatentState | None, LatentState | None]:
        """Build transformer -> run denoising loop -> free transformer.
        Returns ``(video_state | None, audio_state | None)`` with cleared
        conditionings and unpatchified latents for present modalities.
        """
        if video is None and audio is None:
            raise ValueError("At least one of `video` or `audio` must be provided")
        self._assert_supports_conditionings(video)

        if loop is None:
            loop = euler_denoising_loop
        if stepper is None:
            stepper = EulerDiffusionStep()

        pixel_shape = VideoPixelShape(batch=1, frames=frames, height=height, width=width, fps=fps)

        # Build video_tools up front so it can be forwarded to the transformer
        # context (required by TiledDataParallelBuilder in multi-GPU mode).
        video_tools: LatentTools | None = None
        if video is not None:
            v_shape = VideoLatentShape.from_pixel_shape(
                pixel_shape,
                scale_factors=self.video_scale_factors,
            )
            video_tools = VideoLatentTools(
                VideoLatentPatchifier(patch_size=1), v_shape, fps, scale_factors=self.video_scale_factors
            )

        mode = "streaming" if self._is_streaming else "standard"
        logger.info("Building transformer (%s) from %s", mode, self._transformer_builder.checkpoint)
        with self._transformer_ctx(video_tools=video_tools) as transformer:
            logger.info(
                "Running denoising loop (%d steps, %dx%d %d frames @ %.1f fps)",
                len(sigmas) - 1,
                width,
                height,
                frames,
                fps,
            )
            video_state: LatentState | None = None
            if video is not None and video_tools is not None:
                video_state = _build_state(video, video_tools, noiser, self._dtype, self._device)

            audio_tools: LatentTools | None = None
            audio_state: LatentState | None = None
            if audio is not None:
                a_shape = AudioLatentShape.from_video_pixel_shape(pixel_shape)
                audio_tools = AudioLatentTools(AudioPatchifier(patch_size=1), a_shape)
                audio_state = _build_state(audio, audio_tools, noiser, self._dtype, self._device)

            # DiffusionStage.__call__ defaults max_batch_size=1 and the mgpu entry point
            # never threads --max-batch-size down here, so the four guidance passes are
            # split into four sequential forwards BEFORE the transformer wrapper sees
            # them. Under CFG parallelism the batch must arrive intact: the ranks ARE the
            # split. Measured without this, every rank saw batch=1 and redundantly
            # computed a whole pass, so the arm looked like a slower baseline.
            #
            # Not derived from the state's batch, which is 1 (one video) -- the guidance
            # batch is assembled inside _guided_denoise per step. Just disable splitting.
            _mbs = max_batch_size
            if os.environ.get("LTX_S1_PARALLEL") == "cfg":
                _mbs = 1 << 20
            wrapped = BatchSplitAdapter(transformer, max_batch_size=_mbs)  # type: ignore[arg-type]
            video_state, audio_state = loop(
                sigmas=sigmas,
                video_state=video_state,
                audio_state=audio_state,
                stepper=stepper,
                transformer=wrapped,
                denoiser=denoiser,
            )

            if video_state is not None and video_tools is not None:
                video_state = video_tools.clear_conditioning(video_state)
                video_state = video_tools.unpatchify(video_state)
            if audio_state is not None and audio_tools is not None:
                audio_state = audio_tools.clear_conditioning(audio_state)
                audio_state = audio_tools.unpatchify(audio_state)

            return video_state, audio_state


class RecordingDiffusionStage:
    """Wraps a :class:`DiffusionStage` and records the states each call produced.
    The supported way to reach a stage's non-video outputs -- currently the generated keyframe
    latents on ``LatentState.generated_keyframes`` -- from *outside* a pipeline, without the
    pipeline having to grow a diagnostics return value or a lifecycle callback. Pipelines hold
    their stage in a plain attribute, so a caller substitutes this in::
        pipeline.stage = RecordingDiffusionStage(pipeline.stage)
        video, audio = pipeline(...)
        keyframes = pipeline.stage.video_states[0].generated_keyframes
    Delegates every other attribute to the wrapped stage, so a substituted stage stays usable
    wherever the original was. Recording holds references to the returned states, which keeps
    their latents alive -- fine for diagnostics on one run, not something to leave enabled in a
    long-lived serving process.
    """

    def __init__(self, stage: DiffusionStage) -> None:
        self._stage = stage
        self.video_states: list[LatentState] = []
        self.audio_states: list[LatentState | None] = []

    def __call__(self, *args: object, **kwargs: object) -> tuple[LatentState | None, LatentState | None]:
        video_state, audio_state = self._stage(*args, **kwargs)  # type: ignore[arg-type]
        if video_state is not None:
            self.video_states.append(video_state)
            self.audio_states.append(audio_state)
        return video_state, audio_state

    @property
    def generated_keyframes(self) -> torch.Tensor | None:
        """Generated keyframe latents from the first recorded call, if it produced any."""
        for state in self.video_states:
            if state.generated_keyframes is not None:
                return state.generated_keyframes
        return None

    def __getattr__(self, name: str) -> object:
        return getattr(self._stage, name)


# ---------------------------------------------------------------------------
# PromptEncoder
# ---------------------------------------------------------------------------


class PromptEncoder:
    """Owns text encoder + embeddings processor lifecycle.
    Loads Gemma, optionally enhances then encodes prompts, frees Gemma, then loads the
    embeddings processor. ``_enhancer_text_encoder_builder`` is always set: when enhance
    and encode share a checkpoint it aliases ``_text_encoder_builder`` (same object /
    residency); a distinct builder means sequential enhance-then-encode contexts.
    """

    def __init__(
        self,
        model_paths: ModelPaths,
        dtype: torch.dtype,
        device: torch.device,
        registry: Registry | None = None,
        offload_mode: OffloadMode = OffloadMode.NONE,
        text_encoder_builder: BuilderProtocol | None = None,
        alloc_trim_strategy: AllocatorTrimStrategy = AllocatorTrimStrategy.TRIM,
        prompt_enhancer_gemma_root: str | None = None,
        enhancer_text_encoder_builder: BuilderProtocol | None = None,
    ) -> None:
        text_encoder_path = model_paths.text_encoder()
        embeddings_paths = model_paths.embeddings_weight_paths
        if not embeddings_paths:
            raise ValueError(
                "PromptEncoder requires ModelPaths.embeddings_weight_paths "
                "(transformer checkpoint and, in split mode, the text-encoder file)."
            )
        self._text_encoder_path = text_encoder_path
        self._embeddings_paths = embeddings_paths
        self._dtype = dtype
        self._device = device
        self._offload_mode = offload_mode
        self._alloc_trim_strategy = alloc_trim_strategy
        self._prompt_enhancer_gemma_root = prompt_enhancer_gemma_root

        registry = registry or ModelRegistry(cache_models=True, cache_weights=False)
        self._registry = registry

        if text_encoder_builder is not None:
            encode_type = text_encoder_builder.model_config().get("model_type") or gemma_model_type(text_encoder_path)
        else:
            encode_type = gemma_model_type(text_encoder_path)
        if encode_type not in _ENCODE_MODEL_TYPES:
            raise ValueError(
                f"Encode root model_type={encode_type!r} is not supported for encoding; "
                f"expected one of {sorted(_ENCODE_MODEL_TYPES)}."
            )
        self._encode_model_type = encode_type

        if text_encoder_builder is not None:
            if offload_mode != OffloadMode.NONE:
                raise ValueError(
                    "text_encoder_builder cannot be used with offload_mode != OffloadMode.NONE "
                    "because no streaming text encoder builder is available."
                )
            self._text_encoder_builder = text_encoder_builder
            self._streaming_text_encoder_builder = None
        else:
            gemma_sd_ops, gemma_module_ops = get_gemma_ops(text_encoder_path)
            weight_paths = resolve_gemma_weight_paths(text_encoder_path)
            self._text_encoder_builder = Builder(
                model_path=weight_paths,
                model_class_configurator=GemmaTextEncoderConfigurator.with_gemma_model_path(text_encoder_path),
                model_sd_ops=gemma_sd_ops,
                module_ops=gemma_module_ops,
                registry=registry,
            )
            self._streaming_text_encoder_builder = StreamingModelBuilder(
                model_path=weight_paths,
                model_class_configurator=GemmaTextEncoderConfigurator.with_gemma_model_path(text_encoder_path),
                model_sd_ops=gemma_sd_ops,
                module_ops=gemma_module_ops,
                registry=registry,
                blocks_attr="model.model.language_model.layers",
                blocks_prefix="model.model.language_model.layers",
                cpu_slots_count=DISK_CPU_SLOTS if offload_mode == OffloadMode.DISK else None,
            )

        # Always set: shared enhance aliases the encode builder (same object / residency).
        if enhancer_text_encoder_builder is not None:
            self._enhancer_text_encoder_builder = enhancer_text_encoder_builder
        elif prompt_enhancer_gemma_root is not None and prompt_enhancer_gemma_root != text_encoder_path:
            enhancer_sd_ops, enhancer_module_ops = get_gemma_ops(prompt_enhancer_gemma_root)
            enhancer_paths = resolve_gemma_weight_paths(prompt_enhancer_gemma_root)
            self._enhancer_text_encoder_builder = Builder(
                model_path=enhancer_paths,
                model_class_configurator=GemmaTextEncoderConfigurator.with_gemma_model_path(prompt_enhancer_gemma_root),
                model_sd_ops=enhancer_sd_ops,
                module_ops=enhancer_module_ops,
                registry=registry,
            )
        else:
            self._enhancer_text_encoder_builder = self._text_encoder_builder

        self._embeddings_processor_builder = Builder(
            model_path=embeddings_paths,
            model_class_configurator=EmbeddingsProcessorConfigurator.with_gemma_model_path(text_encoder_path),
            model_sd_ops=EMBEDDINGS_PROCESSOR_KEY_OPS,
            registry=registry,
        )

    def _build_text_encoder(self) -> torch.nn.Module:
        """Build the Gemma text encoder (non-streaming path)."""
        return self._text_encoder_builder.build(device=self._device, dtype=self._dtype).eval()

    def _build_enhancer_text_encoder(self) -> torch.nn.Module:
        return self._enhancer_text_encoder_builder.build(device=self._device, dtype=self._dtype).eval()

    def _build_embeddings_processor(self) -> EmbeddingsProcessor:
        """Build the embeddings processor on the target device."""
        return self._embeddings_processor_builder.build(device=self._device, dtype=self._dtype).eval()

    def _text_encoder_ctx(self) -> AbstractContextManager:
        if self._offload_mode != OffloadMode.NONE:
            return _streaming_model(
                self._streaming_text_encoder_builder, self._device, self._dtype, self._alloc_trim_strategy
            )
        return gpu_model(self._build_text_encoder(), alloc_trim_strategy=self._alloc_trim_strategy)

    def __call__(
        self,
        prompts: list[str],
        *,
        enhance_first_prompt: bool = False,
        enhance_prompt_image: str | None = None,
        enhance_prompt_seed: int = 42,
        enhance_static_cache: bool = False,
    ) -> list[EmbeddingsProcessorOutput]:
        """Encode *prompts* through Gemma -> embeddings processor, freeing each model after use."""
        prompts = list(prompts)
        enhance_kwargs = {
            "image_path": enhance_prompt_image,
            "seed": enhance_prompt_seed,
            "static_cache": enhance_static_cache,
        }
        separate_enhancer = self._enhancer_text_encoder_builder is not self._text_encoder_builder

        if enhance_first_prompt and separate_enhancer:
            logger.info(
                "Enhancing with separate Gemma%s",
                f" from {self._prompt_enhancer_gemma_root}" if self._prompt_enhancer_gemma_root else "",
            )
            with gpu_model(
                self._build_enhancer_text_encoder(),
                alloc_trim_strategy=self._alloc_trim_strategy,
            ) as enhancer:
                prompts[0] = generate_enhanced_prompt(enhancer, prompts[0], **enhance_kwargs)
        elif enhance_first_prompt and self._encode_model_type != "gemma3":
            raise ValueError(
                f"Prompt enhancement with encode root model_type={self._encode_model_type!r} "
                "requires --prompt-enhancer-gemma-root pointing at a generative instruct checkpoint "
                "(e.g. gemma3 or gemma4 E2B-it)."
            )

        logger.info("Building text encoder from %s", self._text_encoder_path)
        with self._text_encoder_ctx() as text_encoder:
            if enhance_first_prompt and not separate_enhancer:
                prompts[0] = generate_enhanced_prompt(text_encoder, prompts[0], **enhance_kwargs)
            raw_outputs = text_encoder.encode(prompts)
        logger.info("Text encoder done, building embeddings processor from %s", self._embeddings_paths)

        with gpu_model(
            self._build_embeddings_processor(), alloc_trim_strategy=self._alloc_trim_strategy
        ) as embeddings_processor:
            result = [embeddings_processor.process_hidden_states(hs, mask) for hs, mask in raw_outputs]
        logger.info("Prompt encoding complete")
        return result


# ---------------------------------------------------------------------------
# DurationPredictor
# ---------------------------------------------------------------------------


class DurationPredictor:
    """Predicts shot duration (in frames) from ``PromptEncoder`` output.
    Unlike most blocks, the model is held directly rather than rebuilt on every call:
    DurationHead is a few MB, so there's no memory pressure motivating the
    build-on-call / free-on-exit pattern used for the large transformer/VAE blocks.
    Construct via :meth:`from_checkpoint`.
    """

    def __init__(self, head: DurationHead) -> None:
        """Construct from an already-built, already-loaded head."""
        self._head = head

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | None,
        dtype: torch.dtype,
        device: torch.device,
        model_configurator: type[ModelConfigurator[DurationHead]] = DurationHeadConfigurator,
        model_sd_ops: SDOps = DURATION_HEAD_KEY_OPS,
    ) -> "DurationPredictor | None":
        """Build a predictor from a checkpoint path, or ``None`` if unavailable.
        Returns ``None`` when ``checkpoint_path`` is ``None`` (split pack without a
        duration-head file), or when the file has no DurationHead weights (pre-2.5 /
        gemma3 monoliths). The underlying loader uses ``strict=False``, so a missing
        head does not raise -- it silently leaves parameters on the meta device;
        checked here so callers get ``None`` instead of a predictor that would crash
        later.
        No registry: the head is a few MB, so there's no benefit to caching its weights or
        shell across pipeline instances the way the large transformer/VAE builders do.
        """
        if checkpoint_path is None:
            return None
        builder = Builder(
            model_path=checkpoint_path,
            model_class_configurator=model_configurator,
            model_sd_ops=model_sd_ops,
        )
        head = builder.build(device=device, dtype=dtype).eval()
        if any(param.is_meta for param in head.parameters()):
            logger.info(
                "No DurationHead weights found in %s; auto-duration prediction unavailable.",
                checkpoint_path,
            )
            return None
        return cls(head)

    def __call__(
        self,
        video_encoding: torch.Tensor | None,
        audio_encoding: torch.Tensor | None,
        *,
        frame_rate: float,
        min_seconds: float = 1.0,
        max_seconds: float = 20.0,
    ) -> int:
        """Predict a frame count from caption connector tokens, snapped to the VAE's grid.
        ``min_seconds``/``max_seconds`` clamp the prediction so a misbehaving prediction can't
        request a degenerate or OOM-sized generation; the defaults are 1s and 20s. The result is
        a frame count snapped to the VAE's ``8k + 1`` causal temporal grid. Callers doing offline
        analysis of raw predictions (rather than feeding a real generation) can pass a much wider
        range to see the unclamped duration.
        """
        if video_encoding is None and audio_encoding is None:
            raise ValueError("DurationPredictor requires at least one of video_encoding / audio_encoding")
        seconds_pred = self._head(video_encoding, audio_encoding)
        if seconds_pred.shape != (1,):
            raise ValueError(
                f"DurationPredictor only supports a single-item batch, got prediction shape {tuple(seconds_pred.shape)}"
            )
        seconds = seconds_pred.item()
        min_frames = round(min_seconds * frame_rate)
        max_frames = round(max_seconds * frame_rate)
        num_frames = seconds_to_clamped_num_frames(
            seconds, frame_rate=frame_rate, min_frames=min_frames, max_frames=max_frames
        )
        if seconds > max_seconds or seconds < min_seconds:
            logger.warning(
                "DurationHead prediction clamped: raw %.2fs outside [%.2fs, %.2fs], using %.2fs (%d frames) @ %.2f fps",
                seconds,
                min_seconds,
                max_seconds,
                num_frames / frame_rate,
                num_frames,
                frame_rate,
            )
        else:
            logger.info("DurationHead predicted %.2fs (%d frames @ %.2f fps)", seconds, num_frames, frame_rate)
        return num_frames


def require_num_frames_source(num_frames: int | AutoDuration, duration_predictor: DurationPredictor | None) -> None:
    """Guard against an unsatisfiable auto-duration request.
    Call at the very top of a pipeline's ``__call__`` -- before prompt encoding or any other
    work -- so a checkpoint without DurationHead weights (anything predating 2.5) fails fast
    with a clear message instead of after paying for work whose result would be discarded.
    """
    if isinstance(num_frames, AutoDuration) and duration_predictor is None:
        raise ValueError(
            "num_frames was AutoDuration but this checkpoint has no DurationHead weights to "
            "auto-predict duration from (DurationHead ships from LTX 2.5 / gemma4 onward). "
            "Pass num_frames explicitly."
        )


def resolve_num_frames(
    num_frames: int | AutoDuration,
    duration_predictor: DurationPredictor | None,
    *,
    video_encoding: torch.Tensor | None,
    audio_encoding: torch.Tensor | None,
    frame_rate: float,
) -> int:
    """Resolve ``num_frames`` to a concrete frame count, predicting it if ``AutoDuration``.
    Call after prompt encoding (once ``video_encoding``/``audio_encoding`` exist) and after
    ``require_num_frames_source`` has already validated a predictor is available when needed.
    """
    if not isinstance(num_frames, AutoDuration):
        return num_frames
    return duration_predictor(
        video_encoding,
        audio_encoding,
        frame_rate=frame_rate,
        min_seconds=num_frames.min_seconds,
        max_seconds=num_frames.max_seconds,
    )


# ---------------------------------------------------------------------------
# ImageConditioner
# ---------------------------------------------------------------------------


class ImageConditioner:
    """Owns video encoder lifecycle, and the CRF image conditionings are re-compressed at.
    Builds the encoder, passes it to the user-supplied callable, then frees it.
    Also the single place that knows what H.264 CRF an image conditioning should be
    re-compressed at: that value is a property of the model generation, and this block
    already holds the checkpoint it belongs to. Pipelines call ``resolve_crf`` so callers
    may omit the CRF and still condition the way the model was trained.
    """

    def __init__(
        self,
        checkpoint_path: str,
        dtype: torch.dtype,
        device: torch.device,
        registry: Registry | None = None,
        alloc_trim_strategy: AllocatorTrimStrategy = AllocatorTrimStrategy.TRIM,
    ) -> None:
        self._checkpoint_path = checkpoint_path
        self._default_image_crf: int | None = None
        self._dtype = dtype
        self._device = device
        self._encoder_builder = Builder(
            model_path=checkpoint_path,
            model_class_configurator=VideoEncoderConfigurator,
            model_sd_ops=VAE_ENCODER_COMFY_KEYS_FILTER,
            registry=registry or ModelRegistry(cache_models=True, cache_weights=False),
        )
        self._alloc_trim_strategy = alloc_trim_strategy

    @property
    def default_image_crf(self) -> int:
        """The H.264 CRF image conditionings should be re-compressed at for this checkpoint.
        Read from the checkpoint's ``model_version`` metadata (see ``detect_params``) rather
        than hardcoded, because the value the model was trained with changed between
        generations. Resolved on first use and cached, so a run without image conditionings
        never reads the header.
        """
        if self._default_image_crf is None:
            self._default_image_crf = detect_params(self._checkpoint_path).default_image_crf
        return self._default_image_crf

    def resolve_crf(self, images: Sequence[ImageConditioningInput]) -> list[ImageConditioningInput]:
        """Fill in ``default_image_crf`` for every conditioning that left its CRF unset (``crf=None``).
        Call near the top of a pipeline's ``__call__``: omitting the CRF then means "use what
        matches this model", while an explicitly passed CRF (including ``0`` for no
        re-compression) is always honoured.
        """
        return [image if image.crf is not None else image._replace(crf=self.default_image_crf) for image in images]

    def _build_encoder(self) -> VideoEncoder:
        return self._encoder_builder.build(device=self._device, dtype=self._dtype).eval()

    def __call__(self, fn: Callable[[VideoEncoder], T]) -> T:
        """Build video encoder → call *fn(encoder)* → free encoder."""
        with gpu_model(self._build_encoder(), alloc_trim_strategy=self._alloc_trim_strategy) as encoder:
            return fn(encoder)


# ---------------------------------------------------------------------------
# VideoUpsampler
# ---------------------------------------------------------------------------


class VideoUpsampler:
    """Owns video encoder + spatial upsampler lifecycle."""

    def __init__(
        self,
        checkpoint_path: str,
        upsampler_path: str,
        dtype: torch.dtype,
        device: torch.device,
        registry: Registry | None = None,
        alloc_trim_strategy: AllocatorTrimStrategy = AllocatorTrimStrategy.TRIM,
    ) -> None:
        self._upsampler_path = upsampler_path
        self._dtype = dtype
        self._device = device
        self._encoder_builder = Builder(
            model_path=checkpoint_path,
            model_class_configurator=VideoEncoderConfigurator,
            model_sd_ops=VAE_ENCODER_COMFY_KEYS_FILTER,
            registry=registry or ModelRegistry(cache_models=True, cache_weights=False),
        )
        self._upsampler_builder = Builder(
            model_path=upsampler_path,
            model_class_configurator=LatentUpsamplerConfigurator,
            registry=registry or ModelRegistry(cache_models=True, cache_weights=False),
        )
        self._alloc_trim_strategy = alloc_trim_strategy

    def __call__(self, latent: torch.Tensor) -> torch.Tensor:
        """Upsample *latent* using video encoder + spatial upsampler, then free both."""
        logger.info("Building video encoder + spatial upsampler from %s", self._upsampler_path)
        with (
            gpu_model(
                self._encoder_builder.build(device=self._device, dtype=self._dtype).eval(),
                alloc_trim_strategy=self._alloc_trim_strategy,
            ) as encoder,
            gpu_model(
                self._upsampler_builder.build(device=self._device, dtype=self._dtype).eval(),
                alloc_trim_strategy=self._alloc_trim_strategy,
            ) as upsampler,
        ):
            return upsample_video(latent=latent, video_encoder=encoder, upsampler=upsampler)


# ---------------------------------------------------------------------------
# VideoDecoder
# ---------------------------------------------------------------------------


class VideoDecoder:
    """Owns video decoder lifecycle.
    Returns an iterator that cleans up the decoder after all chunks are consumed.
    """

    def __init__(
        self,
        checkpoint_path: str,
        dtype: torch.dtype,
        device: torch.device,
        registry: Registry | None = None,
        memory_efficient: bool = True,
        decoder_builder: BuilderProtocol | None = None,
        alloc_trim_strategy: AllocatorTrimStrategy = AllocatorTrimStrategy.TRIM,
        diffvae_optimization: DiffVAEMode = DiffVAEMode.CHUNKED_EAGER,
    ) -> None:
        self._checkpoint_path = checkpoint_path
        self._dtype = dtype
        self._device = device
        self._diffvae_optimization = diffvae_optimization
        diffusion_vae = is_diffusion_video_vae(self._checkpoint_path)
        cfg = diffvae_optimization.resolve() if diffusion_vae else None
        use_dsl = cfg is not None and cfg.attention is NAttentionKind.BLACKWELL_DSL
        if decoder_builder is not None:
            self._decoder_builder = decoder_builder
        else:
            if diffusion_vae and use_dsl and not na_dsl_available():
                raise RuntimeError(
                    "diffvae_optimization=blackwell_dsl needs a datacenter Blackwell GPU and "
                    "nvidia-cutlass-dsl (uv sync --group kernels)"
                )
            sd_ops = (
                video_decoder_sd_ops_for_checkpoint(
                    self._checkpoint_path,
                    diffusion_vae=True,
                    na_dsl_kernel=use_dsl,
                )
                if diffusion_vae
                else VAE_DECODER_COMFY_KEYS_FILTER
            )
            module_ops: tuple[ModuleOps, ...] = ()
            # channels-last + memory-efficient decode apply to ConvVideoDecoder only
            if memory_efficient and not diffusion_vae:
                sd_ops = SDOps(
                    name=f"sd_ops_chain_{sd_ops.name}+{CHANNELS_LAST_3D_WEIGHTS.name}",
                    mapping=(*sd_ops.mapping, *CHANNELS_LAST_3D_WEIGHTS.mapping),
                )
                module_ops = (MEMORY_EFFICIENT_DECODE,)
            self._decoder_builder = Builder(
                model_path=self._checkpoint_path,
                model_class_configurator=VideoDecoderConfigurator,
                model_sd_ops=sd_ops,
                registry=registry or ModelRegistry(cache_models=True, cache_weights=False),
                module_ops=module_ops,
            )
        # DiffVAE ModuleOps from ``diffvae_optimization`` policy (default ChunkedEager).
        # MGPU wraps this builder in ``DistributedDecoderBuilder`` — keep the op on the
        # inner builder so it still runs before the distributed wrapper is constructed.
        if diffusion_vae:
            op = build_diffvae_mode_op(
                diffvae_optimization,
                CompilationConfig() if diffvae_optimization.resolve().compile_blocks else None,
            )
            self._decoder_builder = self._decoder_builder.with_module_ops(
                (*self._decoder_builder.module_ops, op),
            )
        self._alloc_trim_strategy = alloc_trim_strategy

    @property
    def checkpoint_path(self) -> str:
        return self._checkpoint_path

    @property
    def diffvae_optimization(self) -> DiffVAEMode:
        return self._diffvae_optimization

    def __call__(
        self,
        latent: torch.Tensor,
        tiling_config: TilingConfig | None = None,
        generator: torch.Generator | None = None,
        *,
        dtype: torch.dtype | None = None,
    ) -> Iterator[torch.Tensor]:
        """Decode *latent* to pixel-space video chunks. Decoder freed after exhaustion.
        ``dtype`` overrides the constructor dtype for this call only (e.g. float32 for
        HDR raw-in/raw-out). Omitted → same dtype as on construction (SDR default).
        """
        build_dtype = self._dtype if dtype is None else dtype
        latent = latent.to(dtype=build_dtype)
        logger.info("Building video decoder from %s", self._checkpoint_path)
        decoder = self._decoder_builder.build(device=self._device, dtype=build_dtype).eval()
        return _cleanup_iter(
            decoder.decode_video(latent, tiling_config, generator),
            decoder,
            alloc_trim_strategy=self._alloc_trim_strategy,
        )


# ---------------------------------------------------------------------------
# AudioDecoder
# ---------------------------------------------------------------------------


class AudioDecoder:
    """Owns audio decoder + vocoder lifecycle."""

    def __init__(
        self,
        checkpoint_path: str,
        dtype: torch.dtype,
        device: torch.device,
        registry: Registry | None = None,
        alloc_trim_strategy: AllocatorTrimStrategy = AllocatorTrimStrategy.TRIM,
    ) -> None:
        self._checkpoint_path = checkpoint_path
        self._dtype = dtype
        self._device = device
        self._decoder_builder = Builder(
            model_path=checkpoint_path,
            model_class_configurator=AudioDecoderConfigurator,
            model_sd_ops=AUDIO_VAE_DECODER_COMFY_KEYS_FILTER,
            registry=registry or ModelRegistry(cache_models=True, cache_weights=False),
        )
        self._vocoder_builder = Builder(
            model_path=checkpoint_path,
            model_class_configurator=VocoderConfigurator,
            model_sd_ops=VOCODER_COMFY_KEYS_FILTER,
            registry=registry or ModelRegistry(cache_models=True, cache_weights=False),
        )
        self._alloc_trim_strategy = alloc_trim_strategy

    def __call__(self, latent: torch.Tensor) -> Audio:
        """Decode audio *latent* through VAE decoder + vocoder, then free both."""
        logger.info("Building audio decoder + vocoder from %s", self._checkpoint_path)
        # The vocoder always runs in fp32 (bf16 accumulation degrades spectral
        # metrics). On CUDA/CPU it is stored in bf16 and autocast upcasts per-op to
        # save memory; MPS has no fp32 autocast, so store it in fp32 directly and
        # avoid the per-call cast. Negligible footprint for this small model.
        vocoder_dtype = torch.float32 if self._device.type == "mps" else self._dtype
        with (
            gpu_model(
                self._decoder_builder.build(device=self._device, dtype=self._dtype).eval(),
                alloc_trim_strategy=self._alloc_trim_strategy,
            ) as decoder,
            gpu_model(
                self._vocoder_builder.build(device=self._device, dtype=vocoder_dtype).eval(),
                alloc_trim_strategy=self._alloc_trim_strategy,
            ) as vocoder,
        ):
            return vae_decode_audio(latent, decoder, vocoder)


# ---------------------------------------------------------------------------
# AudioEncoder
# ---------------------------------------------------------------------------


class AudioConditioner:
    """Owns audio encoder lifecycle.
    Builds the encoder, passes it to the user-supplied callable, then frees it.
    Mirrors :class:`ImageConditioner` for the audio modality.
    """

    def __init__(
        self,
        checkpoint_path: str,
        dtype: torch.dtype,
        device: torch.device,
        registry: Registry | None = None,
        alloc_trim_strategy: AllocatorTrimStrategy = AllocatorTrimStrategy.TRIM,
    ) -> None:
        self._dtype = dtype
        self._device = device
        self._alloc_trim_strategy = alloc_trim_strategy
        self._encoder_builder = Builder(
            model_path=checkpoint_path,
            model_class_configurator=AudioEncoderConfigurator,
            model_sd_ops=AUDIO_VAE_ENCODER_COMFY_KEYS_FILTER,
            registry=registry or ModelRegistry(cache_models=True, cache_weights=False),
        )

    def __call__(self, fn: Callable[[torch.nn.Module], T]) -> T:
        """Build audio encoder → call *fn(encoder)* → free encoder."""
        with gpu_model(
            self._encoder_builder.build(device=self._device, dtype=self._dtype).eval(),
            alloc_trim_strategy=self._alloc_trim_strategy,
        ) as encoder:
            return fn(encoder)

# --- our optimizations, applied at import time -------------------------------
# Here rather than in a sitecustomize: the mgpu pipeline is started through
# runpy.run_module(run_name="__main__") and spawns workers that re-import
# everything, and an external injector lost to both of those.
from ltx_pipelines.utils.opt_stack import install_cache as _install_cache
from ltx_pipelines.utils.opt_timing import install as _install_timing

_install_cache()
_install_timing(sys.modules[__name__])
