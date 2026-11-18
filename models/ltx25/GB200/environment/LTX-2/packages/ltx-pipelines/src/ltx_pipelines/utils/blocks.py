"""Pipeline blocks — each block owns its model lifecycle.
Blocks build a model on each ``__call__``, use it, then free GPU memory.
This eliminates manual ``del model; cleanup_memory()`` in pipelines: each
block is self-contained, so no central model-coordinator object is needed.
"""

from __future__ import annotations

import copy
import logging
from collections.abc import Iterator
from contextlib import AbstractContextManager, contextmanager
from dataclasses import replace
from typing import Callable, TypeVar

import torch

from ltx_core.allocator_trim_strategy import AllocatorTrimStrategy
import os as _os_bs
from ltx_core.batch_split import BatchSplitAdapter
from ltx_core.block_streaming import DISK_CPU_SLOTS, StreamingModelBuilder
from ltx_core.components.diffusion_steps import EulerDiffusionStep
from ltx_core.components.noisers import Noiser
from ltx_core.components.patchifiers import AudioPatchifier, VideoLatentPatchifier
from ltx_core.components.protocols import DiffusionStepProtocol
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
    build_compile_diffusion_decoder_op,
    build_cutlass_fna_diffusion_decoder_op,
    natten_available,
)
from ltx_core.quantization import QuantizationPolicy, fp8_cast_fuse_rule
from ltx_core.text_encoders.gemma import (
    EMBEDDINGS_PROCESSOR_KEY_OPS,
    EmbeddingsProcessorConfigurator,
    GemmaTextEncoderConfigurator,
    gemma_model_type,
    get_gemma_ops,
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
from ltx_core.utils import find_matching_file
from ltx_pipelines.utils.gpu_model import gpu_model
from ltx_pipelines.utils.helpers import (
    cleanup_memory,
    create_noised_state,
    generate_enhanced_prompt,
    seconds_to_clamped_num_frames,
)
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
        state = replace(state, denoise_mask=torch.zeros_like(state.denoise_mask))
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
            # never threads --max-batch-size down here, so the guidance batch is split
            # into 4 sequential chunks BEFORE the transformer wrapper sees it. That is
            # why CFG parallelism measured batch=1 at 160 calls (40 steps x 4) and every
            # rank redundantly computed a whole pass. Under CFG parallelism the batch
            # must arrive intact -- the ranks are the split.
            _mbs = max_batch_size
            if _os_bs.environ.get("LTX_S1_PARALLEL") == "cfg":
                # NOT the state's batch -- that is 1 (one video). The guidance batch is
                # assembled inside _guided_denoise per step, so reading it here always
                # gave 1 and the previous override was a no-op. Just disable splitting.
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
        checkpoint_path: str,
        gemma_root: str,
        dtype: torch.dtype,
        device: torch.device,
        registry: Registry | None = None,
        offload_mode: OffloadMode = OffloadMode.NONE,
        text_encoder_builder: BuilderProtocol | None = None,
        alloc_trim_strategy: AllocatorTrimStrategy = AllocatorTrimStrategy.TRIM,
        prompt_enhancer_gemma_root: str | None = None,
        enhancer_text_encoder_builder: BuilderProtocol | None = None,
    ) -> None:
        self._gemma_root = gemma_root
        self._checkpoint_path = checkpoint_path
        self._dtype = dtype
        self._device = device
        self._offload_mode = offload_mode
        self._alloc_trim_strategy = alloc_trim_strategy
        self._prompt_enhancer_gemma_root = prompt_enhancer_gemma_root

        registry = registry or ModelRegistry(cache_models=True, cache_weights=False)
        self._registry = registry

        if text_encoder_builder is not None:
            encode_type = text_encoder_builder.model_config().get("model_type") or gemma_model_type(gemma_root)
        else:
            encode_type = gemma_model_type(gemma_root)
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
            gemma_sd_ops, gemma_module_ops = get_gemma_ops(gemma_root)
            model_folder = find_matching_file(gemma_root, "model*.safetensors").parent
            weight_paths = tuple(str(p) for p in model_folder.rglob("*.safetensors"))
            self._text_encoder_builder = Builder(
                model_path=weight_paths,
                model_class_configurator=GemmaTextEncoderConfigurator.with_gemma_model_path(gemma_root),
                model_sd_ops=gemma_sd_ops,
                module_ops=gemma_module_ops,
                registry=registry,
            )
            self._streaming_text_encoder_builder = StreamingModelBuilder(
                model_path=weight_paths,
                model_class_configurator=GemmaTextEncoderConfigurator.with_gemma_model_path(gemma_root),
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
        elif prompt_enhancer_gemma_root is not None and prompt_enhancer_gemma_root != gemma_root:
            enhancer_sd_ops, enhancer_module_ops = get_gemma_ops(prompt_enhancer_gemma_root)
            enhancer_folder = find_matching_file(prompt_enhancer_gemma_root, "model*.safetensors").parent
            enhancer_paths = tuple(str(p) for p in enhancer_folder.rglob("*.safetensors"))
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
            model_path=checkpoint_path,
            model_class_configurator=EmbeddingsProcessorConfigurator.with_gemma_model_path(gemma_root),
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

        logger.info("Building text encoder from %s", self._gemma_root)
        with self._text_encoder_ctx() as text_encoder:
            if enhance_first_prompt and not separate_enhancer:
                prompts[0] = generate_enhanced_prompt(text_encoder, prompts[0], **enhance_kwargs)
            raw_outputs = text_encoder.encode(prompts)
        logger.info("Text encoder done, building embeddings processor from %s", self._checkpoint_path)

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
        checkpoint_path: str,
        dtype: torch.dtype,
        device: torch.device,
        model_configurator: type[ModelConfigurator[DurationHead]] = DurationHeadConfigurator,
        model_sd_ops: SDOps = DURATION_HEAD_KEY_OPS,
    ) -> "DurationPredictor | None":
        """Build a predictor from a checkpoint path, or ``None`` if it has no DurationHead weights.
        DurationHead ships only from 2.4 checkpoints onward. Older checkpoints have no
        ``duration_head.*`` weights, and the underlying loader loads state dicts with
        ``strict=False``, so a missing head does not raise -- it silently leaves the head's
        parameters on the meta device. Checked here so callers get ``None`` instead of a
        predictor that would crash later, deep inside a forward pass.
        No registry: the head is a few MB, so there's no benefit to caching its weights or
        shell across pipeline instances the way the large transformer/VAE builders do.
        """
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
    work -- so a checkpoint without DurationHead weights (anything predating 2.4) fails fast
    with a clear message instead of after paying for work whose result would be discarded.
    """
    if isinstance(num_frames, AutoDuration) and duration_predictor is None:
        raise ValueError(
            "num_frames was AutoDuration but this checkpoint has no DurationHead weights to "
            "auto-predict duration from (DurationHead ships from 2.4 checkpoints onward). "
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
    """Owns video encoder lifecycle.
    Builds the encoder, passes it to the user-supplied callable, then frees it.
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
        self._encoder_builder = Builder(
            model_path=checkpoint_path,
            model_class_configurator=VideoEncoderConfigurator,
            model_sd_ops=VAE_ENCODER_COMFY_KEYS_FILTER,
            registry=registry or ModelRegistry(cache_models=True, cache_weights=False),
        )
        self._alloc_trim_strategy = alloc_trim_strategy

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
        vae_compile: bool = False,
    ) -> None:
        self._checkpoint_path = checkpoint_path
        self._dtype = dtype
        self._device = device
        self._vae_compile = vae_compile
        diffusion_vae = is_diffusion_video_vae(self._checkpoint_path)
        if decoder_builder is not None:
            self._decoder_builder = decoder_builder
        else:
            if diffusion_vae and not natten_available():
                raise RuntimeError(
                    f"Diffusion VAE checkpoint {self._checkpoint_path!r} requires natten: "
                    "uv sync --package ltx-core --extra natten"
                )
            sd_ops = (
                video_decoder_sd_ops_for_checkpoint(self._checkpoint_path, diffusion_vae=True)
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
        # DiffVAE ModuleOps: compile when ``vae_compile``, else pin NA to cutlass-fna
        # (lower peak VRAM). MGPU wraps this builder in ``DistributedDecoderBuilder`` —
        # keep the op on the inner builder so it still runs before the distributed
        # wrapper is constructed.
        if diffusion_vae:
            op = (
                build_compile_diffusion_decoder_op(CompilationConfig())
                if vae_compile
                else build_cutlass_fna_diffusion_decoder_op()
            )
            self._decoder_builder = self._decoder_builder.with_module_ops(
                (*self._decoder_builder.module_ops, op),
            )
        self._alloc_trim_strategy = alloc_trim_strategy

    def __call__(
        self,
        latent: torch.Tensor,
        tiling_config: TilingConfig | None = None,
        generator: torch.Generator | None = None,
    ) -> Iterator[torch.Tensor]:
        """Decode *latent* to pixel-space video chunks. Decoder freed after exhaustion."""
        logger.info("Building video decoder from %s", self._checkpoint_path)
        decoder = self._decoder_builder.build(device=self._device, dtype=self._dtype).eval()
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


# Stage-level timing, appended here so it is active in SPAWNED multi-GPU workers too
# (patching from the parent never reaches them -- that is why the vendor SP run
# produced no timing at all). Enabled only when STAGE_TIME_FILE is set.
import os as _os  # noqa: E402
if _os.environ.get("STAGE_TIME_FILE"):
    try:
        from ltx_core.opt.stage_timer import install as _install_stage_timer
        _install_stage_timer()
    except Exception as _e:  # never let instrumentation break a run
        print(f"[STAGE] install failed: {_e}", flush=True)

# Kernel fusion / FB-cache for the SPAWNED mgpu workers (parent-side patches never
# reach them). Kernels need the built model, so they go in on the first forward.
if _os.environ.get("LTX_STACK_FP4") == "1":
    # Low precision arms on the first forward for the same reason as the kernels:
    # the spawned workers never see a parent-side patch.
    import ltx_core.model.transformer.model as _mm_fp4
    from ltx_core.opt.nvfp4 import swap_linears as _fp4swap
    _of4 = _mm_fp4.LTXModel.forward

    def _f4(self, video, audio, perturbations):
        # Per MODEL, not once per process: stage 2 disposes and rebuilds the
        # transformer (and merges the distilled LoRA), so a process-wide latch
        # would leave stage 2 running bf16 while claiming to be fp4.
        if not getattr(self, "_ltx_fp4", False):
            self._ltx_fp4 = True
            n = _fp4swap(self)
            print(f"[stack] nvfp4 linears swapped: {n}", flush=True)
            assert n > 0, "nvfp4 swap was a no-op"
        return _of4(self, video, audio, perturbations)

    _mm_fp4.LTXModel.forward = _f4

if _os.environ.get("LTX_STACK_KERNELS") == "1" or _os.environ.get("LTX_STACK_CACHE") == "1":
    import ltx_core.model.transformer.model as _mm_stack
    from ltx_core.opt.mgpu_stack import install_cache as _ic, install_kernels as _ik
    if _os.environ.get("LTX_STACK_CACHE") == "1":
        _ic()
    _of = _mm_stack.LTXModel.forward
    def _f(self, video, audio, perturbations):
        if _os.environ.get("LTX_STACK_KERNELS") == "1":
            _ik(self)
        return _of(self, video, audio, perturbations)
    _mm_stack.LTXModel.forward = _f


# O7 CUDA graph. Installed LAST so it wraps the lazy kernel / fp4 installers --
# they then run inside the capture warmup, before the capture itself.
#
# Mutually exclusive with the step cache: the cache decides per step whether to
# skip blocks, and a graph would bake in whatever the capture step happened to
# decide. Assert rather than silently pick a winner.
if _os.environ.get("LTX_STACK_GRAPH") == "1":
    assert _os.environ.get("LTX_STACK_CACHE") != "1", (
        "LTX_STACK_GRAPH and LTX_STACK_CACHE are mutually exclusive: the cache "
        "skip decision is data dependent and cannot be captured")
    import ltx_core.model.transformer.model as _mm_graph
    from ltx_core.opt.cuda_graph import install as _install_graph
    print(f"[stack] cuda graph installed: {_install_graph(_mm_graph)}", flush=True)


# FP4 on the VAE decoder. Separate switch from the transformer's: the decoder is
# a different model with different shapes, and it is the largest remaining cost
# after stage 1 (2.82 s) now that the DiT is down to ~6.9 s.
if _os.environ.get("LTX_STACK_VAE_FP4") == "1":
    from ltx_core.opt.nvfp4 import swap_linears as _vfp4

    def _vae_predicate(name: str) -> bool:
        # every Linear in the decoder; the swap already skips anything whose
        # in/out features do not tile 128
        return True

    def _arm_vae(dec):
        if getattr(dec, "_ltx_vae_fp4", False):
            return
        dec._ltx_vae_fp4 = True
        n = _vfp4(dec, _vae_predicate)
        print(f"[stack] vae decoder nvfp4 linears swapped: {n}", flush=True)

    try:
        from ltx_core.model.video_vae.diffusion_video_decoder import (
            DiffusionVideoDecoder as _Dec)
    except Exception as _e:
        print(f"[stack] vae fp4: import failed {_e}", flush=True)
        _Dec = None
    if _Dec is not None:
        _odec = _Dec.forward_diff_step

        def _dec_fwd(self, *a, **k):
            _arm_vae(self)
            return _odec(self, *a, **k)

        _Dec.forward_diff_step = _dec_fwd
        print("[stack] vae fp4 hook installed", flush=True)
    else:
        print("[stack] vae fp4: decoder class not found, hook NOT installed", flush=True)


# Op-level breakdown of the video decode, printed once. Kept out of the way of
# the timing path: profiling perturbs it, so never quote a number from a run
# with this on.
if _os.environ.get("LTX_VAE_PROFILE") == "1":
    import torch as _t_prof
    from ltx_core.model.video_vae.diffusion_video_decoder import (
        DiffusionVideoDecoder as _PDec)
    _pdone = {"n": False}
    _pfwd = _PDec.forward_diff_step

    def _prof_fwd(self, *a, **k):
        if _pdone["n"]:
            return _pfwd(self, *a, **k)
        _pdone["n"] = True
        from torch.profiler import ProfilerActivity, profile
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as pr:
            out = _pfwd(self, *a, **k)
            _t_prof.cuda.synchronize()
        ev = pr.key_averages()
        tot = sum(e.self_device_time_total for e in ev)
        print(f"[vaeprof] total self CUDA time {tot/1e3:.1f} ms", flush=True)
        rows = sorted(ev, key=lambda e: -e.self_device_time_total)[:18]
        for e in rows:
            if e.self_device_time_total <= 0:
                continue
            print(f"[vaeprof] {e.self_device_time_total/1e3:9.2f} ms "
                  f"{100*e.self_device_time_total/max(tot,1):5.1f}%  "
                  f"n={e.count:<6d} {e.key[:60]}", flush=True)
        return out

    _PDec.forward_diff_step = _prof_fwd
    print("[stack] vae profiler installed", flush=True)


# Record what na3d is actually called with. Written to a file, not printed: the
# mgpu ranks are spawned workers whose stdout never reaches the parent.
if _os.environ.get("LTX_NA3D_PROBE") == "1":
    try:
        import natten as _nat

        _orig_na3d = _nat.na3d
        _seen = set()

        def _na3d_probe(q, k, v, **kw):
            key = (tuple(q.shape), str(q.dtype),
                   str(kw.get("kernel_size")), str(kw.get("dilation")),
                   str(kw.get("is_causal")))
            if key not in _seen:
                _seen.add(key)
                line = (f"[na3d] q={tuple(q.shape)} {q.dtype} "
                        f"kernel_size={kw.get('kernel_size')} "
                        f"dilation={kw.get('dilation')} "
                        f"is_causal={kw.get('is_causal')} "
                        f"other={ {a: b for a, b in kw.items()
                                   if a not in ('kernel_size', 'dilation', 'is_causal')} }")
                f = _os.environ.get("STAGE_TIME_FILE")
                if f:
                    with open(f, "a") as fh:
                        fh.write(line + "\n")
                print(line, flush=True)
            return _orig_na3d(q, k, v, **kw)

        _nat.na3d = _na3d_probe
        print("[stack] na3d probe installed", flush=True)
    except Exception as _e:
        print(f"[stack] na3d probe failed: {_e}", flush=True)


# Per-layer na3d backend selection. On by default: it is both faster (na3d
# 547 -> 294 ms per decode pass, 1.86x) and the only thing that keeps the
# 1920x1088 decode from dying inside cutlass-fna. LTX_NA3D_BACKEND=off restores
# whatever the model asked for.
if _os.environ.get("LTX_NA3D_BACKEND", "auto") != "off":
    try:
        import natten as _nat_sel

        _CUTLASS_KERNELS = {(3, 5, 5)}
        _na3d_orig = _nat_sel.na3d
        _na3d_stats = {}

        def _na3d_routed(q, k, v, **kw):
            ks = kw.get("kernel_size")
            kst = tuple(ks) if isinstance(ks, (tuple, list)) else (ks,)
            want = "cutlass-fna" if kst in _CUTLASS_KERNELS else None
            asked = kw.get("backend")
            if want is None:
                kw.pop("backend", None)      # let natten choose (Blackwell FNA)
            else:
                kw["backend"] = want
            key = (kst, asked, want or "default")
            if key not in _na3d_stats:
                _na3d_stats[key] = 0
                line = (f"[na3d-sel] kernel={kst} asked={asked} -> "
                        f"{want or 'default'}")
                f = _os.environ.get("STAGE_TIME_FILE")
                if f:
                    with open(f, "a") as fh:
                        fh.write(line + "\n")
                print(line, flush=True)
            _na3d_stats[key] += 1
            return _na3d_orig(q, k, v, **kw)

        _nat_sel.na3d = _na3d_routed
        print("[stack] na3d backend router installed", flush=True)
    except Exception as _e:
        print(f"[stack] na3d router failed: {_e}", flush=True)


# Save the decoder's input latent once, for the TAEHV comparison.
if _os.environ.get("LTX_DUMP_LATENT"):
    import torch as _t_dump
    from ltx_core.model.video_vae.diffusion_video_decoder import (
        DiffusionVideoDecoder as _DDec)
    _dumped = {"n": False}
    _dpre = _DDec.forward_pre_diffusion

    def _dump_pre(self, *a, **k):
        if not _dumped["n"]:
            _dumped["n"] = True
            for i, x in enumerate(a):
                if _t_dump.is_tensor(x) and x.ndim >= 4:
                    path = _os.environ["LTX_DUMP_LATENT"]
                    _t_dump.save({"latent": x.detach().float().cpu(),
                                  "shape": tuple(x.shape), "argpos": i}, path)
                    print(f"[dump] saved decoder input {tuple(x.shape)} {x.dtype} -> {path}",
                          flush=True)
                    break
            else:
                print(f"[dump] no >=4D tensor in args: "
                      f"{[type(x).__name__ for x in a]}", flush=True)
        return _dpre(self, *a, **k)

    _DDec.forward_pre_diffusion = _dump_pre
    print("[stack] latent dump installed", flush=True)


# Full pre-tiling latent, for the TAEHV comparison.
if _os.environ.get("LTX_DUMP_FULL"):
    import torch as _t_full
    _fdone = {"n": False}
    _fo = VideoDecoder.__call__

    def _full_call(self, *a, **k):
        if not _fdone["n"]:
            _fdone["n"] = True
            cands = [x for x in list(a) + list(k.values())
                     if _t_full.is_tensor(x) and x.ndim >= 4]
            if cands:
                x = max(cands, key=lambda t: t.numel())
                path = _os.environ["LTX_DUMP_FULL"]
                _t_full.save({"latent": x.detach().float().cpu(),
                              "shape": tuple(x.shape)}, path)
                print(f"[dumpfull] {tuple(x.shape)} {x.dtype} -> {path}", flush=True)
            else:
                print(f"[dumpfull] no >=4D tensor; args={[type(z).__name__ for z in a]} "
                      f"kwargs={list(k)}", flush=True)
        return _fo(self, *a, **k)

    VideoDecoder.__call__ = _full_call
    print("[stack] full latent dump installed", flush=True)


# Wall-clock vs GPU-busy for the decoder and for one stage-1 forward.
if _os.environ.get("LTX_UTIL_PROBE") == "1":
    import time as _t_u
    import torch as _t_up

    def _profile_once(fn, label, *a, **k):
        from torch.profiler import ProfilerActivity, profile
        _t_up.cuda.synchronize()
        w0 = _t_u.perf_counter()
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as pr:
            out = fn(*a, **k)
            _t_up.cuda.synchronize()
        wall = _t_u.perf_counter() - w0
        ev = pr.key_averages()
        busy = sum(e.self_device_time_total for e in ev) / 1e6      # us -> s
        print(f"[util] {label}: wall={wall * 1e3:9.1f} ms  gpu_busy={busy * 1e3:9.1f} ms  "
              f"util={100 * busy / max(wall, 1e-9):5.1f}%", flush=True)
        top = sorted(ev, key=lambda e: -e.self_device_time_total)[:25]
        _shown = 0.0
        for e in top:
            if e.self_device_time_total <= 0:
                continue
            _shown += e.self_device_time_total
            print(f"[util]    {e.self_device_time_total / 1e3:8.2f} ms "
                  f"{100 * e.self_device_time_total / max(busy * 1e6, 1):5.1f}%  "
                  f"n={e.count:<5d} {e.key[:52]}", flush=True)
        print(f"[util]    ---- shown {100 * _shown / max(busy * 1e6, 1):.1f}%, "
              f"tail {100 * (busy * 1e6 - _shown) / max(busy * 1e6, 1):.1f}% in ops below",
              flush=True)
        return out

    # --- video decoder (whole call, so tiling and scheduling are included)
    _dec_done = {"n": False}
    _dec_o = VideoDecoder.__call__

    def _dec_u(self, *a, **k):
        if _dec_done["n"]:
            return _dec_o(self, *a, **k)
        _dec_done["n"] = True

        def _drain(*x, **y):
            r = _dec_o(self, *x, **y)
            # the decoder hands back an iterator; the decode happens on consumption
            if hasattr(r, "__iter__") and not _t_up.is_tensor(r):
                return iter(list(r))   # drain to force the decode, hand back an iterator
            return r

        return _profile_once(_drain, "VideoDecoder(drained)", *a, **k)

    VideoDecoder.__call__ = _dec_u

    # --- one stage-1 transformer forward, after warmup so it is steady state
    import ltx_core.model.transformer.model as _mm_u
    _f_o = _mm_u.LTXModel.forward
    _f_n = {"n": 0}

    if _os.environ.get("LTX_STACK_CACHE") == "1":
        print("[util] WARNING: LTX_STACK_CACHE=1 -- call 12 may be a SKIPPED step, "
              "which runs only block 0. DiT utilization needs cache off.", flush=True)
    if _os.environ.get("LTX_BGRAPH") == "1":
        print("[util] WARNING: LTX_BGRAPH=1 -- a profiler cannot see inside graph "
              "replays, so DiT utilization would be meaningless. Run with "
              "LTX_BGRAPH=0.", flush=True)

    def _f_u(self, video, audio, perturbations):
        _f_n["n"] += 1
        if _f_n["n"] != 12:            # 12th call: past warmup, still stage 1
            return _f_o(self, video, audio, perturbations)
        return _profile_once(_f_o, "DiT forward (call 12)", self, video, audio,
                             perturbations)

    _mm_u.LTXModel.forward = _f_u
    print("[stack] utilization probe installed", flush=True)


# torch.compile on the transformer blocks. Never tested on this project -- it was
# switched off early and the config has changed twice since. Worth reopening now
# because the lever changed: at 4 GPU / M=6144 the block graph was worth 1.4x
# (launch bound), at 2 GPU / M=16320 it is worth 1.8% (GPU busy). Removing launch
# overhead is spent; fusing away intermediate tensors is not, and that is what
# compile does and graphs do not.
#
# Regional compilation (per block) rather than whole-model: far shorter compile
# time and, per the diffusers work, nearly the same result.
if _os.environ.get("LTX_COMPILE", "0") != "0":
    import torch as _t_c
    import ltx_core.model.transformer.model as _mm_c
    _mode = _os.environ.get("LTX_COMPILE")           # e.g. default / max-autotune
    _cdone = {"n": False}
    _co = _mm_c.LTXModel.forward

    def _cf(self, video, audio, perturbations):
        if not _cdone["n"]:
            _cdone["n"] = True
            n = 0
            for blk in self.transformer_blocks:
                blk.forward = _t_c.compile(blk.forward, mode=_mode, dynamic=False)
                n += 1
            print(f"[stack] torch.compile mode={_mode} on {n} blocks", flush=True)
        return _co(self, video, audio, perturbations)

    _mm_c.LTXModel.forward = _cf


# Rank the glue ops by the source line that emits them.
if _os.environ.get("LTX_GLUE_ATTRIB") == "1":
    import torch as _t_g
    from collections import defaultdict as _dd
    import ltx_core.model.transformer.model as _mm_g

    for _bad, _why in (("LTX_COMPILE_UNUSED", "n/a"),
                       ("LTX_BGRAPH", "graph replays are opaque to the profiler"),
                       ("LTX_STACK_CACHE", "a skipped step runs only block 0")):
        if _os.environ.get(_bad, "0") not in ("0", ""):
            print(f"[glue] WARNING: {_bad} is on -- {_why}; counts will be wrong",
                  flush=True)

    _g_o = _mm_g.LTXModel.forward
    _g_n = {"n": 0}
    _WATCH = {"aten::copy_", "aten::add", "aten::add_", "aten::mul", "aten::mul_",
              "aten::cat", "aten::to", "aten::contiguous", "aten::clone",
              "aten::slice", "aten::_to_copy", "aten::div", "aten::sub",
              "aten::permute", "aten::reshape", "aten::view", "aten::expand"}

    def _g_f(self, video, audio, perturbations):
        _g_n["n"] += 1
        if _g_n["n"] != 10:
            return _g_o(self, video, audio, perturbations)
        from torch.profiler import ProfilerActivity, profile
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                     with_stack=True, record_shapes=True) as pr:
            out = _g_o(self, video, audio, perturbations)
            _t_g.cuda.synchronize()

        agg = _dd(lambda: [0, 0.0, None])
        total_dev = 0.0
        for e in pr.events():
            dev = getattr(e, "self_device_time_total", None)
            if dev is None:
                dev = getattr(e, "self_cuda_time_total", 0.0)
            total_dev += max(dev, 0.0)
            if e.name not in _WATCH:
                continue
            frame = "?"
            for f in (getattr(e, "stack", None) or []):
                if "ltx_core" in f or "ltx_pipelines" in f:
                    frame = f.split("site-packages/")[-1].split("baseline-snapshot/")[-1]
                    break
            k = (e.name, frame)
            agg[k][0] += 1
            agg[k][1] += max(dev, 0.0)
            if agg[k][2] is None:
                agg[k][2] = str(getattr(e, "input_shapes", ""))[:56]

        rows = sorted(agg.items(), key=lambda kv: -kv[1][1])
        print(f"[glue] total device time in forward: {total_dev / 1e3:.1f} ms", flush=True)
        print(f"[glue] {'op':<18} {'ms':>8} {'%':>5} {'n':>6}  site / shapes", flush=True)
        for (op, frame), (n, us, shp) in rows[:28]:
            print(f"[glue] {op:<18} {us / 1e3:8.2f} "
                  f"{100 * us / max(total_dev, 1):5.1f} {n:6d}  {frame[:74]}", flush=True)
            print(f"[glue] {'':<18} {'':>8} {'':>5} {'':>6}    {shp}", flush=True)
        return out

    _mm_g.LTXModel.forward = _g_f
    print("[stack] glue attribution installed", flush=True)


# Per-call-site attribution using TorchFunctionMode: every torch op that passes
# through gets charged to the first frame inside ltx_core / ltx_pipelines.
if _os.environ.get("LTX_GLUE_SITE") == "1":
    import sys as _sys_gs
    import torch as _t_gs
    from collections import defaultdict as _dd_gs
    from torch.overrides import TorchFunctionMode as _TFM
    import ltx_core.model.transformer.model as _mm_gs

    _GS = _dd_gs(lambda: [0, None])

    class _SiteMode(_TFM):
        def __torch_function__(self, func, types, args=(), kwargs=None):
            kwargs = kwargs or {}
            name = getattr(func, "__name__", str(func))
            f = _sys_gs._getframe(1)
            site = "?"
            for _ in range(14):
                if f is None:
                    break
                fn = f.f_code.co_filename
                if "ltx_core" in fn or "ltx_pipelines" in fn:
                    short = fn.split("/ltx_core/")[-1].split("/ltx_pipelines/")[-1]
                    site = f"{short}:{f.f_lineno}"
                    break
                f = f.f_back
            k = (name, site)
            _GS[k][0] += 1
            if _GS[k][1] is None:
                a0 = args[0] if args else None
                _GS[k][1] = (tuple(a0.shape) if _t_gs.is_tensor(a0) else "")
            return func(*args, **kwargs)

    _o_gs = _mm_gs.LTXModel.forward
    _n_gs = {"n": 0}

    def _f_gs(self, video, audio, perturbations):
        _n_gs["n"] += 1
        if _n_gs["n"] != 10:
            return _o_gs(self, video, audio, perturbations)
        _GS.clear()
        with _SiteMode():
            out = _o_gs(self, video, audio, perturbations)
        _t_gs.cuda.synchronize()
        rows = sorted(_GS.items(), key=lambda kv: -kv[1][0])
        tot = sum(v[0] for v in _GS.values())
        print(f"[site] {tot} torch ops in one forward; top call sites by count",
              flush=True)
        print(f"[site] {'op':<16} {'n':>6}  {'shape':<22} site", flush=True)
        for (op, site), (n, shp) in rows[:40]:
            print(f"[site] {op:<16} {n:6d}  {str(shp):<22} {site[:66]}", flush=True)
        return out

    _mm_gs.LTXModel.forward = _f_gs
    print("[stack] glue site attribution installed", flush=True)


# Device time per (op, call site): TorchFunctionMode supplies the label,
# the profiler supplies the timing.
if _os.environ.get("LTX_DEVTIME") == "1":
    import sys as _sys_dt
    import torch as _t_dt
    from torch.overrides import TorchFunctionMode as _TFM_dt
    from torch.profiler import record_function as _rf
    import ltx_core.model.transformer.model as _mm_dt

    _COST = {"copy_", "add", "add_", "mul", "mul_", "cat", "clone", "to",
             "_to_copy", "contiguous", "div", "sub", "rsqrt", "pow", "mean",
             "silu", "gelu", "sigmoid", "softmax", "empty_like", "zeros_like",
             "index_select", "unbind", "stack", "chunk", "linear", "matmul",
             "bmm", "scaled_dot_product_attention"}

    class _LabelMode(_TFM_dt):
        def __torch_function__(self, func, types, args=(), kwargs=None):
            kwargs = kwargs or {}
            name = getattr(func, "__name__", "?")
            if name not in _COST:
                return func(*args, **kwargs)
            f = _sys_dt._getframe(1)
            site = "?"
            for _ in range(14):
                if f is None:
                    break
                fn = f.f_code.co_filename
                if "ltx_core" in fn or "ltx_pipelines" in fn:
                    site = (fn.split("/ltx_core/")[-1].split("/ltx_pipelines/")[-1]
                            + ":" + str(f.f_lineno))
                    break
                f = f.f_back
            with _rf(f"S|{name}|{site}"):
                return func(*args, **kwargs)

    _o_dt = _mm_dt.LTXModel.forward
    _n_dt = {"n": 0}

    def _f_dt(self, video, audio, perturbations):
        _n_dt["n"] += 1
        if _n_dt["n"] != 10:
            return _o_dt(self, video, audio, perturbations)
        from torch.profiler import ProfilerActivity, profile
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as pr:
            with _LabelMode():
                out = _o_dt(self, video, audio, perturbations)
            _t_dt.cuda.synchronize()
        ev = pr.key_averages()
        tot = sum(e.self_device_time_total for e in ev)
        rows = [e for e in ev if e.key.startswith("S|")]
        rows.sort(key=lambda e: -e.device_time_total)
        print(f"[dev] forward total self-device {tot / 1e3:.1f} ms; "
              f"top sites by DEVICE time", flush=True)
        print(f"[dev] {'ms':>9} {'%':>5} {'n':>6}  op / site", flush=True)
        for e in rows[:26]:
            _, op, site = e.key.split("|", 2)
            print(f"[dev] {e.device_time_total / 1e3:9.2f} "
                  f"{100 * e.device_time_total / max(tot, 1):5.1f} {e.count:6d}  "
                  f"{op} @ {site[:62]}", flush=True)
        return out

    _mm_dt.LTXModel.forward = _f_dt
    print("[stack] device-time attribution installed", flush=True)


# Dump all Python stacks on SIGUSR1. ptrace_scope=1 on these nodes makes py-spy
# useless, and a hang after the last log line is otherwise unattributable.
if _os.environ.get("LTX_FAULT", "1") == "1":
    import faulthandler as _fh
    import signal as _sig
    try:
        _fh.enable()
        _fh.register(_sig.SIGUSR1, all_threads=True, chain=False)
        print(f"[fault] SIGUSR1 stack dump armed pid={_os.getpid()}", flush=True)
    except Exception as _e:
        print(f"[fault] arm failed: {_e}", flush=True)
