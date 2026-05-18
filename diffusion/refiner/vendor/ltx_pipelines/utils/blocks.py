"""Pipeline blocks — each block owns its model lifecycle.
Blocks build a model on each ``__call__``, use it, then free GPU memory.
This eliminates manual ``del model; cleanup_memory()`` in pipelines and
removes the need for :class:`ModelLedger`.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import AbstractContextManager, contextmanager
from dataclasses import replace
from typing import Callable, TypeVar

import torch

from diffusion.refiner.vendor.ltx_core.batch_split import BatchSplitAdapter
from diffusion.refiner.vendor.ltx_core.components.diffusion_steps import EulerDiffusionStep
from diffusion.refiner.vendor.ltx_core.components.noisers import Noiser
from diffusion.refiner.vendor.ltx_core.components.patchifiers import AudioPatchifier, VideoLatentPatchifier
from diffusion.refiner.vendor.ltx_core.components.protocols import DiffusionStepProtocol
from diffusion.refiner.vendor.ltx_core.layer_streaming import LayerStreamingWrapper
from diffusion.refiner.vendor.ltx_core.loader import SDOps
from diffusion.refiner.vendor.ltx_core.loader.primitives import LoraPathStrengthAndSDOps
from diffusion.refiner.vendor.ltx_core.loader.registry import DummyRegistry, Registry
from diffusion.refiner.vendor.ltx_core.loader.single_gpu_model_builder import SingleGPUModelBuilder as Builder
from diffusion.refiner.vendor.ltx_core.model.audio_vae import (
    AUDIO_VAE_DECODER_COMFY_KEYS_FILTER,
    AUDIO_VAE_ENCODER_COMFY_KEYS_FILTER,
    VOCODER_COMFY_KEYS_FILTER,
    AudioDecoderConfigurator,
    AudioEncoderConfigurator,
    VocoderConfigurator,
)
from diffusion.refiner.vendor.ltx_core.model.audio_vae import decode_audio as vae_decode_audio
from diffusion.refiner.vendor.ltx_core.model.transformer import (
    LTXV_MODEL_COMFY_RENAMING_MAP,
    LTXModelConfigurator,
    X0Model,
)
from diffusion.refiner.vendor.ltx_core.model.transformer.compiling import (
    COMPILE_TRANSFORMER,
    modify_sd_ops_for_compilation,
)
from diffusion.refiner.vendor.ltx_core.model.upsampler import LatentUpsamplerConfigurator, upsample_video
from diffusion.refiner.vendor.ltx_core.model.video_vae import (
    VAE_DECODER_COMFY_KEYS_FILTER,
    VAE_ENCODER_COMFY_KEYS_FILTER,
    TilingConfig,
    VideoDecoderConfigurator,
    VideoEncoder,
    VideoEncoderConfigurator,
)
from diffusion.refiner.vendor.ltx_core.quantization import QuantizationPolicy
from diffusion.refiner.vendor.ltx_core.text_encoders.gemma import (
    EMBEDDINGS_PROCESSOR_KEY_OPS,
    GEMMA_LLM_KEY_OPS,
    GEMMA_MODEL_OPS,
    EmbeddingsProcessorConfigurator,
    GemmaTextEncoderConfigurator,
    module_ops_from_gemma_root,
)
from diffusion.refiner.vendor.ltx_core.text_encoders.gemma.embeddings_processor import EmbeddingsProcessorOutput
from diffusion.refiner.vendor.ltx_core.tools import AudioLatentTools, LatentTools, VideoLatentTools
from diffusion.refiner.vendor.ltx_core.types import (
    Audio,
    AudioLatentShape,
    LatentState,
    VideoLatentShape,
    VideoPixelShape,
)
from diffusion.refiner.vendor.ltx_core.utils import find_matching_file
from diffusion.refiner.vendor.ltx_pipelines.utils.gpu_model import gpu_model
from diffusion.refiner.vendor.ltx_pipelines.utils.helpers import (
    cleanup_memory,
    create_noised_state,
    generate_enhanced_prompt,
)
from diffusion.refiner.vendor.ltx_pipelines.utils.samplers import euler_denoising_loop
from diffusion.refiner.vendor.ltx_pipelines.utils.types import Denoiser, ModalitySpec

logger = logging.getLogger(__name__)

T = TypeVar("T")
_M = TypeVar("_M", bound=torch.nn.Module)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


@contextmanager
def _streaming_model(
    model: _M,
    layers_attr: str,
    target_device: torch.device,
    prefetch_count: int,
) -> Iterator[_M]:
    """Wrap *model* with :class:`LayerStreamingWrapper`, yield it, then tear down."""
    wrapped = LayerStreamingWrapper(
        model,
        layers_attr=layers_attr,
        target_device=target_device,
        prefetch_count=prefetch_count,
    )
    try:
        yield wrapped  # type: ignore[misc]
    finally:
        wrapped.teardown()
        wrapped.to("meta")
        cleanup_memory()
        # Flush the host (pinned) memory cache so that freed pinned pages are
        # returned to the OS.  Without this, sequential streaming models
        # (e.g. text encoder then transformer) exhaust host memory because the
        # CachingHostAllocator keeps freed blocks cached indefinitely.
        torch.cuda.synchronize(device=target_device)
        try:
            if hasattr(torch._C, "_host_emptyCache"):
                torch._C._host_emptyCache()
        except Exception:
            logger.warning("Host empty cache cleanup failed; ignoring.", exc_info=True)


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


def _cleanup_iter(it: Iterator[torch.Tensor], model: torch.nn.Module) -> Iterator[torch.Tensor]:
    """Wrap an iterator to clean up *model* memory once it is exhausted or abandoned."""
    with gpu_model(model):
        yield from it


# ---------------------------------------------------------------------------
# DiffusionStage
# ---------------------------------------------------------------------------


class DiffusionStage:
    """Owns transformer lifecycle. Builds on each call, frees on exit.
    Replaces the manual ``model_ledger.transformer()`` / ``del transformer``
    pattern in every pipeline.
    """

    def __init__(
        self,
        checkpoint_path: str,
        dtype: torch.dtype,
        device: torch.device,
        loras: tuple[LoraPathStrengthAndSDOps, ...] = (),
        quantization: QuantizationPolicy | None = None,
        registry: Registry | None = None,
        torch_compile: bool = False,
    ) -> None:
        self._dtype = dtype
        self._device = device
        self._quantization = quantization
        self._torch_compile = torch_compile
        self._transformer_builder = Builder(
            model_path=checkpoint_path,
            model_class_configurator=LTXModelConfigurator,
            model_sd_ops=LTXV_MODEL_COMFY_RENAMING_MAP,
            loras=tuple(loras),
            registry=registry or DummyRegistry(),
        )

    def _build_transformer(self, *, device: torch.device | None = None, **kwargs: object) -> X0Model:
        target = device or self._device
        sd_ops = self._transformer_builder.model_sd_ops
        module_ops = self._transformer_builder.module_ops
        loras = self._transformer_builder.loras
        if self._torch_compile:
            module_ops = (*module_ops, COMPILE_TRANSFORMER)
            number_of_layers = self._transformer_builder.model_config()["transformer"]["num_layers"]
            sd_ops = modify_sd_ops_for_compilation(sd_ops, number_of_layers)
            loras = tuple(
                LoraPathStrengthAndSDOps(
                    lora.path,
                    lora.strength,
                    modify_sd_ops_for_compilation(
                        lora.sd_ops if lora.sd_ops is not None else SDOps(name="identity"), number_of_layers
                    ),
                )
                for lora in loras
            )
        if self._quantization is not None:
            module_ops = (*module_ops, *self._quantization.module_ops)
            sd_ops = SDOps(
                name=f"sd_ops_chain_{sd_ops.name}+{self._quantization.sd_ops.name}",
                mapping=(*sd_ops.mapping, *self._quantization.sd_ops.mapping),
            )

        builder = self._transformer_builder.with_module_ops(module_ops).with_sd_ops(sd_ops).with_loras(loras)
        return X0Model(builder.build(device=target, **kwargs)).to(target).eval()

    def _transformer_ctx(
        self,
        streaming_prefetch_count: int | None,
        **kwargs: object,
    ) -> AbstractContextManager:
        if streaming_prefetch_count is not None:
            return _streaming_model(
                self._build_transformer(device=torch.device("cpu"), **kwargs),
                layers_attr="velocity_model.transformer_blocks",
                target_device=self._device,
                prefetch_count=streaming_prefetch_count,
            )
        return gpu_model(self._build_transformer(**kwargs))

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
        streaming_prefetch_count: int | None = None,
        max_batch_size: int = 1,
    ) -> tuple[LatentState | None, LatentState | None]:
        """Build transformer → run denoising loop → free transformer.
        Args:
            width: Output width in pixels.
            height: Output height in pixels.
            frames: Number of output frames.
            fps: Frame rate.
            loop: Denoising loop function. Must accept
                ``(sigmas, video_state, audio_state, stepper, transformer, denoiser)``
                as the first six positional arguments. When ``None``, resolves to
                :func:`euler_denoising_loop` at call time.
            streaming_prefetch_count: When set, build the transformer on CPU and
                wrap with :class:`LayerStreamingWrapper` for memory-efficient
                inference, prefetching this many layers ahead.
            max_batch_size: Maximum batch size per transformer forward pass.
                Guided denoisers make up to 4 transformer calls per step.
                When set to a value > 1, the transformer batches multiple
                calls together, reducing layer-streaming PCIe transfers.
                Default ``1`` preserves sequential behavior.
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

        video_state: LatentState | None = None
        video_tools: LatentTools | None = None
        if video is not None:
            v_shape = VideoLatentShape.from_pixel_shape(pixel_shape)
            video_tools = VideoLatentTools(VideoLatentPatchifier(patch_size=1), v_shape, fps)
            video_state = _build_state(video, video_tools, noiser, self._dtype, self._device)

        audio_state: LatentState | None = None
        audio_tools: LatentTools | None = None
        if audio is not None:
            a_shape = AudioLatentShape.from_video_pixel_shape(pixel_shape)
            audio_tools = AudioLatentTools(AudioPatchifier(patch_size=1), a_shape)
            audio_state = _build_state(audio, audio_tools, noiser, self._dtype, self._device)

        with self._transformer_ctx(streaming_prefetch_count, video_tools=video_tools) as base_transformer:
            transformer = BatchSplitAdapter(base_transformer, max_batch_size=max_batch_size)
            video_state, audio_state = loop(
                sigmas=sigmas,
                video_state=video_state,
                audio_state=audio_state,
                stepper=stepper,
                transformer=transformer,
                denoiser=denoiser,
            )

        # Post-process: clear conditionings and unpatchify
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
    Loads Gemma, encodes prompts, frees Gemma, then loads the embeddings
    processor to produce final outputs.
    """

    def __init__(
        self,
        checkpoint_path: str,
        gemma_root: str,
        dtype: torch.dtype,
        device: torch.device,
        registry: Registry | None = None,
    ) -> None:
        self._dtype = dtype
        self._device = device

        module_ops = module_ops_from_gemma_root(gemma_root)
        model_folder = find_matching_file(gemma_root, "model*.safetensors").parent
        weight_paths = [str(p) for p in model_folder.rglob("*.safetensors")]

        self._text_encoder_builder = Builder(
            model_path=tuple(weight_paths),
            model_class_configurator=GemmaTextEncoderConfigurator,
            model_sd_ops=GEMMA_LLM_KEY_OPS,
            module_ops=(GEMMA_MODEL_OPS, *module_ops),
            registry=registry or DummyRegistry(),
        )
        self._embeddings_processor_builder = Builder(
            model_path=checkpoint_path,
            model_class_configurator=EmbeddingsProcessorConfigurator,
            model_sd_ops=EMBEDDINGS_PROCESSOR_KEY_OPS,
            registry=registry or DummyRegistry(),
        )

    def _text_encoder_ctx(
        self,
        streaming_prefetch_count: int | None,
    ) -> AbstractContextManager:
        if streaming_prefetch_count is not None:
            return _streaming_model(
                self._text_encoder_builder.build(device=torch.device("cpu"), dtype=self._dtype).eval(),
                layers_attr="model.model.language_model.layers",
                target_device=self._device,
                prefetch_count=streaming_prefetch_count,
            )
        return gpu_model(self._text_encoder_builder.build(device=self._device, dtype=self._dtype).eval())

    def __call__(
        self,
        prompts: list[str],
        *,
        enhance_first_prompt: bool = False,
        enhance_prompt_image: str | None = None,
        enhance_prompt_seed: int = 42,
        streaming_prefetch_count: int | None = None,
    ) -> list[EmbeddingsProcessorOutput]:
        """Encode *prompts* through Gemma → embeddings processor, freeing each model after use."""
        with self._text_encoder_ctx(streaming_prefetch_count) as text_encoder:
            if enhance_first_prompt:
                prompts = list(prompts)
                prompts[0] = generate_enhanced_prompt(
                    text_encoder, prompts[0], enhance_prompt_image, seed=enhance_prompt_seed
                )
            raw_outputs = [text_encoder.encode(p) for p in prompts]

        with gpu_model(
            self._embeddings_processor_builder.build(device=self._device, dtype=self._dtype).to(self._device).eval()
        ) as embeddings_processor:
            return [embeddings_processor.process_hidden_states(hs, mask) for hs, mask in raw_outputs]


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
    ) -> None:
        self._dtype = dtype
        self._device = device
        self._encoder_builder = Builder(
            model_path=checkpoint_path,
            model_class_configurator=VideoEncoderConfigurator,
            model_sd_ops=VAE_ENCODER_COMFY_KEYS_FILTER,
            registry=registry or DummyRegistry(),
        )

    def _build_encoder(self) -> VideoEncoder:
        return self._encoder_builder.build(device=self._device, dtype=self._dtype).to(self._device).eval()

    def __call__(self, fn: Callable[[VideoEncoder], T]) -> T:
        """Build video encoder → call *fn(encoder)* → free encoder."""
        with gpu_model(self._build_encoder()) as encoder:
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
    ) -> None:
        self._dtype = dtype
        self._device = device
        self._encoder_builder = Builder(
            model_path=checkpoint_path,
            model_class_configurator=VideoEncoderConfigurator,
            model_sd_ops=VAE_ENCODER_COMFY_KEYS_FILTER,
            registry=registry or DummyRegistry(),
        )
        self._upsampler_builder = Builder(
            model_path=upsampler_path,
            model_class_configurator=LatentUpsamplerConfigurator,
            registry=registry or DummyRegistry(),
        )

    def __call__(self, latent: torch.Tensor) -> torch.Tensor:
        """Upsample *latent* using video encoder + spatial upsampler, then free both."""
        with (
            gpu_model(
                self._encoder_builder.build(device=self._device, dtype=self._dtype).to(self._device).eval()
            ) as encoder,
            gpu_model(
                self._upsampler_builder.build(device=self._device, dtype=self._dtype).to(self._device).eval()
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
    ) -> None:
        self._dtype = dtype
        self._device = device
        self._decoder_builder = Builder(
            model_path=checkpoint_path,
            model_class_configurator=VideoDecoderConfigurator,
            model_sd_ops=VAE_DECODER_COMFY_KEYS_FILTER,
            registry=registry or DummyRegistry(),
        )

    def __call__(
        self,
        latent: torch.Tensor,
        tiling_config: TilingConfig | None = None,
        generator: torch.Generator | None = None,
    ) -> Iterator[torch.Tensor]:
        """Decode *latent* to pixel-space video chunks. Decoder freed after exhaustion."""
        decoder = self._decoder_builder.build(device=self._device, dtype=self._dtype).to(self._device).eval()
        return _cleanup_iter(decoder.decode_video(latent, tiling_config, generator), decoder)


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
    ) -> None:
        self._dtype = dtype
        self._device = device
        self._decoder_builder = Builder(
            model_path=checkpoint_path,
            model_class_configurator=AudioDecoderConfigurator,
            model_sd_ops=AUDIO_VAE_DECODER_COMFY_KEYS_FILTER,
            registry=registry or DummyRegistry(),
        )
        self._vocoder_builder = Builder(
            model_path=checkpoint_path,
            model_class_configurator=VocoderConfigurator,
            model_sd_ops=VOCODER_COMFY_KEYS_FILTER,
            registry=registry or DummyRegistry(),
        )

    def __call__(self, latent: torch.Tensor) -> Audio:
        """Decode audio *latent* through VAE decoder + vocoder, then free both."""
        with (
            gpu_model(
                self._decoder_builder.build(device=self._device, dtype=self._dtype).to(self._device).eval()
            ) as decoder,
            gpu_model(
                self._vocoder_builder.build(device=self._device, dtype=self._dtype).to(self._device).eval()
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
    ) -> None:
        self._dtype = dtype
        self._device = device
        self._encoder_builder = Builder(
            model_path=checkpoint_path,
            model_class_configurator=AudioEncoderConfigurator,
            model_sd_ops=AUDIO_VAE_ENCODER_COMFY_KEYS_FILTER,
            registry=registry or DummyRegistry(),
        )

    def __call__(self, fn: Callable[[torch.nn.Module], T]) -> T:
        """Build audio encoder → call *fn(encoder)* → free encoder."""
        with gpu_model(
            self._encoder_builder.build(device=self._device, dtype=self._dtype).to(self._device).eval()
        ) as encoder:
            return fn(encoder)
