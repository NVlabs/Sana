"""One-step H3 refinement using standard Diffusers LTX-2 components."""

from __future__ import annotations

import math

import torch
from diffusers import (
    AutoencoderKLLTX2Video,
    DiffusionPipeline,
    FlowMatchEulerDiscreteScheduler,
    LTX2ConditionPipeline,
    LTX2LatentUpsamplePipeline,
    LTX2VideoDiffusionDecoderModel,
    LTX2VideoTransformer3DModel,
)
from diffusers.pipelines.ltx.pipeline_output import LTXPipelineOutput
from diffusers.pipelines.ltx2.connectors import LTX2TextConnectors
from diffusers.pipelines.ltx2.latent_upsampler import LTX2LatentUpsamplerModel
from diffusers.pipelines.ltx2.pipeline_ltx2_diffusion_decode import (
    LTX2VideoDiffusionDecodePipeline,
)
from diffusers.utils.torch_utils import randn_tensor
from transformers import PreTrainedModel, PreTrainedTokenizerBase


DEFAULT_SIGMA = 0.9093750119


def output_geometry(width: int, height: int, frames: int) -> tuple[int, int, int]:
    """Return the x2 conditioning canvas and valid LTX temporal length."""
    if width <= 0 or height <= 0 or frames <= 0:
        raise ValueError("width, height and frame count must be positive")
    return (
        math.ceil(width / 64) * 64,
        math.ceil(height / 64) * 64,
        1 + (frames - 1) // 8 * 8,
    )


class SoLRefinerH3Pipeline(DiffusionPipeline):
    """Refine H3 video with a pretrained, offline-merged transformer.

    The model directory contains ordinary Diffusers components. The runtime has
    no adapter path or LoRA merge operation. Audio generation is disabled.
    """

    model_cpu_offload_seq = "text_encoder->connectors->vae->latent_upsampler->transformer->diffusion_decoder"
    _optional_components = ["text_encoder", "tokenizer", "connectors"]

    def __init__(
        self,
        vae: AutoencoderKLLTX2Video,
        transformer: LTX2VideoTransformer3DModel,
        latent_upsampler: LTX2LatentUpsamplerModel,
        diffusion_decoder: LTX2VideoDiffusionDecoderModel,
        scheduler: FlowMatchEulerDiscreteScheduler,
        text_encoder: PreTrainedModel = None,
        tokenizer: PreTrainedTokenizerBase = None,
        connectors: LTX2TextConnectors = None,
    ):
        super().__init__()
        self.register_modules(
            vae=vae,
            transformer=transformer,
            latent_upsampler=latent_upsampler,
            diffusion_decoder=diffusion_decoder,
            scheduler=scheduler,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            connectors=connectors,
        )

    def _encode_text(self, prompt: str, device: torch.device):
        if (
            self.text_encoder is None
            or self.tokenizer is None
            or self.connectors is None
        ):
            raise ValueError(
                "Provide prompt embeddings or load all text-conditioning components"
            )
        text_pipeline = LTX2ConditionPipeline(
            scheduler=self.scheduler,
            vae=self.vae,
            audio_vae=None,
            text_encoder=self.text_encoder,
            tokenizer=self.tokenizer,
            connectors=self.connectors,
            transformer=self.transformer,
            vocoder=None,
        )
        embeddings, mask, _, _ = text_pipeline.encode_prompt(
            prompt=prompt,
            do_classifier_free_guidance=False,
            max_sequence_length=1024,
            device=device,
        )
        embeddings, _, mask = self.connectors(
            embeddings, mask, padding_side=self.tokenizer.padding_side
        )
        return embeddings, mask

    @torch.no_grad()
    def denoise_latents(
        self,
        noisy_latents: torch.Tensor,
        prompt_embeds: torch.Tensor,
        *,
        sigma: float = DEFAULT_SIGMA,
        frame_rate: float = 24.0,
        prompt_attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Perform exactly one conditional-only velocity prediction and Euler step."""
        if not 0 < sigma <= 1 or frame_rate <= 0:
            raise ValueError("sigma must be in (0, 1] and frame_rate must be positive")
        if (
            noisy_latents.ndim != 5
            or noisy_latents.shape[1] != self.transformer.config.in_channels
        ):
            raise ValueError(
                "Expected [batch, latent_channels, frames, height, width] latents"
            )
        if (
            self.scheduler.config.use_dynamic_shifting
            or self.scheduler.config.shift != 1
        ):
            raise ValueError(
                "The one-step checkpoint requires an unshifted FlowMatch Euler schedule"
            )
        device = self._execution_device
        dtype = self.transformer.dtype
        x = noisy_latents.to(device=device, dtype=dtype)
        batch, channels, frames, height, width = x.shape
        packed = LTX2ConditionPipeline._pack_latents(x, 1, 1)
        self.scheduler.set_timesteps(sigmas=[sigma], device=device)
        if len(self.scheduler.timesteps) != 1 or not torch.isclose(
            self.scheduler.sigmas[0], torch.tensor(sigma, device=device), atol=1e-6
        ):
            raise ValueError("Scheduler altered the checkpoint's one-step sigma")
        timestep = self.scheduler.timesteps[0].expand(batch)
        # Diffusers prompt AdaLN accepts the scaled timestep, not raw sigma.
        velocity, _ = self.transformer(
            hidden_states=packed,
            audio_hidden_states=torch.zeros(
                batch,
                1,
                self.transformer.config.audio_in_channels,
                device=device,
                dtype=dtype,
            ),
            encoder_hidden_states=prompt_embeds.to(device=device, dtype=dtype),
            audio_encoder_hidden_states=torch.zeros(
                batch,
                1,
                self.transformer.config.audio_cross_attention_dim,
                device=device,
                dtype=dtype,
            ),
            timestep=timestep[:, None].expand(batch, packed.shape[1]),
            audio_timestep=timestep,
            sigma=timestep,
            audio_sigma=timestep,
            encoder_attention_mask=(
                prompt_attention_mask.to(device)
                if prompt_attention_mask is not None
                else None
            ),
            num_frames=frames,
            height=height,
            width=width,
            fps=frame_rate,
            audio_num_frames=1,
            isolate_modalities=True,
            return_dict=False,
        )
        denoised = self.scheduler.step(
            velocity.float(),
            self.scheduler.timesteps[0],
            packed.float(),
            return_dict=False,
        )[0]
        return (
            denoised.reshape(batch, frames, height, width, channels)
            .permute(0, 4, 1, 2, 3)
            .contiguous()
        )

    @torch.no_grad()
    def __call__(
        self,
        video: list,
        prompt: str | None = None,
        *,
        width: int = 1920,
        height: int = 1080,
        frame_rate: float = 24.0,
        sigma: float = DEFAULT_SIGMA,
        generator: torch.Generator | None = None,
        decoder_generator: torch.Generator | None = None,
        prompt_embeds: torch.Tensor | None = None,
        prompt_attention_mask: torch.Tensor | None = None,
        output_type: str = "np",
        return_dict: bool = True,
    ):
        if (prompt is None) == (prompt_embeds is None):
            raise ValueError("Provide exactly one of prompt or prompt_embeds")
        if output_type not in {"np", "pt", "latent"}:
            raise ValueError("output_type must be np, pt, or latent")
        canvas_width, canvas_height, frames = output_geometry(width, height, len(video))
        device = self._execution_device
        if prompt_embeds is None:
            prompt_embeds, prompt_attention_mask = self._encode_text(prompt, device)
        upsample = LTX2LatentUpsamplePipeline(
            vae=self.vae, latent_upsampler=self.latent_upsampler
        )
        pixels = upsample.video_processor.preprocess_video(
            video[:frames], height=canvas_height // 2, width=canvas_width // 2
        ).to(device=device, dtype=self.vae.dtype)
        # Use the posterior mode, matching the deterministic reference encoder.
        encoded = self.vae.encode(pixels).latent_dist.mode()
        latents = upsample(
            latents=encoded,
            width=canvas_width // 2,
            height=canvas_height // 2,
            num_frames=frames,
            latents_normalized=False,
            output_type="latent",
        ).frames
        latents = LTX2ConditionPipeline._normalize_latents(
            latents,
            self.vae.latents_mean,
            self.vae.latents_std,
            self.vae.config.scaling_factor,
        ).to(device=device, dtype=self.transformer.dtype)
        noise = randn_tensor(
            latents.shape, generator=generator, device=device, dtype=latents.dtype
        )
        noisy = ((1 - sigma) * latents.float() + sigma * noise.float()).to(
            latents.dtype
        )
        denoised = self.denoise_latents(
            noisy,
            prompt_embeds,
            sigma=sigma,
            frame_rate=frame_rate,
            prompt_attention_mask=prompt_attention_mask,
        )
        if output_type == "latent":
            result = LTX2ConditionPipeline._denormalize_latents(
                denoised,
                self.vae.latents_mean,
                self.vae.latents_std,
                self.vae.config.scaling_factor,
            )
        else:
            decoder = LTX2VideoDiffusionDecodePipeline(
                diffusion_decoder=self.diffusion_decoder,
                scheduler=self.scheduler,
                vae=self.vae,
            )
            result = decoder(
                denoised,
                generator=decoder_generator
                if decoder_generator is not None
                else generator,
                output_type=output_type,
                denormalize=True,
            ).frames
            top, left = (canvas_height - height) // 2, (canvas_width - width) // 2
            if output_type == "np":
                result = result[:, :, top : top + height, left : left + width, :]
            else:
                result = result[:, :, :, top : top + height, left : left + width]
        self.maybe_free_model_hooks()
        return LTXPipelineOutput(frames=result) if return_dict else (result,)
