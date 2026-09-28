"""Original SoL-Refiner with Diffusers LTX-2.3 components."""

import math

import torch
from diffusers import (
    AutoencoderKLLTX2Video,
    DiffusionPipeline,
    FlowMatchEulerDiscreteScheduler,
    LTX2ConditionPipeline,
    LTX2LatentUpsamplePipeline,
    LTX2VideoTransformer3DModel,
)
from diffusers.pipelines.ltx.pipeline_output import LTXPipelineOutput
from diffusers.models.transformers.transformer_ltx2 import LTX2AudioVideoRotaryPosEmbed
from diffusers.pipelines.ltx2.connectors import LTX2TextConnectors
from diffusers.pipelines.ltx2.latent_upsampler import LTX2LatentUpsamplerModel
from diffusers.utils.torch_utils import randn_tensor
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from .sampling import DEFAULT_NEGATIVE_PROMPT, StepCache, sigma_schedule


class ReferenceRotaryEmbedding(LTX2AudioVideoRotaryPosEmbed):
    """Preserve the coordinate and RoPE precision used to train these checkpoints."""

    def forward(self, coords, device=None):
        frequencies = super().forward(coords.to(self.coordinate_dtype), device)
        return tuple(t.to(self.coordinate_dtype) for t in frequencies)


class SoLRefinerPipeline(DiffusionPipeline):
    model_cpu_offload_seq = (
        "text_encoder->connectors->vae->latent_upsampler->transformer->vae"
    )
    _optional_components = ["text_encoder", "tokenizer", "connectors"]

    def __init__(
        self,
        vae: AutoencoderKLLTX2Video,
        transformer: LTX2VideoTransformer3DModel,
        latent_upsampler: LTX2LatentUpsamplerModel,
        scheduler: FlowMatchEulerDiscreteScheduler,
        text_encoder: PreTrainedModel = None,
        tokenizer: PreTrainedTokenizerBase = None,
        connectors: LTX2TextConnectors = None,
        variant: str = "one-step",
    ):
        super().__init__()
        sigma_schedule(variant)
        self.register_modules(
            vae=vae,
            transformer=transformer,
            latent_upsampler=latent_upsampler,
            scheduler=scheduler,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            connectors=connectors,
        )
        self.register_to_config(variant=variant)
        if transformer is not None:
            from .optimization import RefinerAttention

            transformer.rope.__class__ = ReferenceRotaryEmbedding
            transformer.rope.coordinate_dtype = transformer.dtype
            # Native LTX uses torch RMSNorm, including its BF16 rounding order.
            for block in transformer.transformer_blocks:
                block.attn1.set_processor(RefinerAttention(None, 0, False))
                block.attn2.set_processor(RefinerAttention(None, 0, False))
                for name in ("norm1", "norm2", "norm3"):
                    original = getattr(block, name)
                    norm = torch.nn.RMSNorm(
                        transformer.config.num_attention_heads
                        * transformer.config.attention_head_dim,
                        eps=original.eps,
                        elementwise_affine=False,
                    )
                    setattr(block, name, norm)
                for attention in (block.attn1, block.attn2):
                    for name in ("norm_q", "norm_k"):
                        original = getattr(attention, name)
                        norm = torch.nn.RMSNorm(original.weight.shape, eps=original.eps)
                        norm.weight = original.weight
                        setattr(attention, name, norm)

        self.acceleration = None
        self.last_run = {}

    def enable_sol_engine(self, *, tau=None, density=None):
        from .optimization import SolEngine

        if self.acceleration is None:
            self.acceleration = SolEngine(self.transformer, tau=tau, density=density)

    def _encode_text(self, prompt):
        text = LTX2ConditionPipeline(
            scheduler=self.scheduler,
            vae=self.vae,
            audio_vae=None,
            text_encoder=self.text_encoder,
            tokenizer=self.tokenizer,
            connectors=self.connectors,
            transformer=self.transformer,
            vocoder=None,
        )
        embeddings, mask, _, _ = text.encode_prompt(
            prompt=prompt,
            device=self._execution_device,
            do_classifier_free_guidance=False,
            max_sequence_length=1024,
        )
        embeddings, _, mask = self.connectors(
            embeddings, mask, padding_side=self.tokenizer.padding_side
        )
        return embeddings, mask

    def _velocity(self, x, context, mask, sigma, grid, fps):
        b = x.shape[0]
        timestep = sigma.expand(b) * 1000
        dtype, device = self.transformer.dtype, self._execution_device
        return self.transformer(
            hidden_states=x.to(dtype),
            encoder_hidden_states=context.to(device=device, dtype=dtype),
            encoder_attention_mask=mask.to(device) if mask is not None else None,
            audio_hidden_states=torch.zeros(
                b,
                1,
                self.transformer.config.audio_in_channels,
                device=device,
                dtype=dtype,
            ),
            audio_encoder_hidden_states=torch.zeros(
                b,
                1,
                self.transformer.config.audio_cross_attention_dim,
                device=device,
                dtype=dtype,
            ),
            timestep=timestep[:, None].expand(b, x.shape[1]),
            audio_timestep=timestep,
            sigma=timestep,
            audio_sigma=timestep,
            num_frames=grid[0],
            height=grid[1],
            width=grid[2],
            fps=fps,
            audio_num_frames=1,
            isolate_modalities=True,
            return_dict=False,
        )[0]

    @torch.no_grad()
    def denoise_latents(
        self,
        noisy_latents,
        prompt_embeds,
        *,
        negative_prompt_embeds=None,
        prompt_attention_mask=None,
        negative_attention_mask=None,
        frame_rate=16.0,
        teacache=False,
    ):
        variant = self.config.variant
        if teacache and variant == "one-step":
            raise ValueError("TeaCache requires the multi-step model")
        if variant == "multi-step" and negative_prompt_embeds is None:
            raise ValueError(
                "The multi-step model requires negative prompt embeddings for CFG=3"
            )
        if noisy_latents.ndim != 5 or frame_rate <= 0:
            raise ValueError("Expected BCFHW latents and a positive frame rate")
        sigmas = sigma_schedule(variant).to(self._execution_device)
        if (
            self.scheduler.config.shift != 1
            or self.scheduler.config.use_dynamic_shifting
        ):
            raise ValueError(
                "The saved scheduler must use shift=1 without dynamic shifting"
            )
        self.scheduler.set_timesteps(
            sigmas=sigmas[:-1].cpu().tolist(), device=self._execution_device
        )
        batch, channels, *grid = noisy_latents.shape
        x = LTX2ConditionPipeline._pack_latents(
            noisy_latents.to(self._execution_device), 1, 1
        )
        cache = StepCache() if teacache else None
        calls = 0
        if self.acceleration:
            self.acceleration.begin(grid, self._execution_device)
        for step, sigma in enumerate(sigmas[:-1]):
            if self.acceleration:
                self.acceleration.step = step
            if cache and cache.reuse(x, step):
                clean = cache.prediction
            else:
                v = self._velocity(
                    x, prompt_embeds, prompt_attention_mask, sigma, grid, frame_rate
                )
                calls += 1
                clean = x.float() - sigma * v.float()
                if variant == "multi-step":
                    vu = self._velocity(
                        x,
                        negative_prompt_embeds,
                        negative_attention_mask,
                        sigma,
                        grid,
                        frame_rate,
                    )
                    uncond = x.float() - sigma * vu.float()
                    clean = uncond + 3.0 * (clean - uncond)
                    calls += 1
                if cache:
                    cache.prediction = clean.detach()
            velocity = (x.float() - clean) / sigma
            x = self.scheduler.step(
                velocity, self.scheduler.timesteps[step], x.float(), return_dict=False
            )[0]
        self.last_run = {
            "variant": variant,
            "steps": len(sigmas) - 1,
            "transformer_calls": calls,
            "cached_steps": cache.skipped if cache else 0,
            "engine": "sol" if self.acceleration else "baseline",
        }
        return x.reshape(batch, *grid, channels).permute(0, 4, 1, 2, 3).contiguous()

    @torch.no_grad()
    def __call__(
        self,
        video,
        prompt,
        *,
        negative_prompt=DEFAULT_NEGATIVE_PROMPT,
        width=2048,
        height=1152,
        frame_rate=16.0,
        generator=None,
        noise=None,
        teacache=False,
        output_type="np",
    ):
        if min(width, height, len(video)) <= 0:
            raise ValueError("Video and output dimensions must be nonempty")
        if output_type not in {"np", "latent"}:
            raise ValueError("output_type must be np or latent")
        cw, ch = math.ceil(width / 64) * 64, math.ceil(height / 64) * 64
        frames = 1 + (len(video) - 1) // 8 * 8
        pos, pos_mask = self._encode_text(prompt)
        neg, neg_mask = (
            self._encode_text(negative_prompt)
            if self.config.variant == "multi-step"
            else (None, None)
        )
        upsample = LTX2LatentUpsamplePipeline(
            vae=self.vae, latent_upsampler=self.latent_upsampler
        )
        pixels = upsample.video_processor.preprocess_video(
            video[:frames], height=ch // 2, width=cw // 2
        )
        encoded = self.vae.encode(
            pixels.to(self._execution_device, self.vae.dtype)
        ).latent_dist.mode()
        raw = upsample(
            latents=encoded,
            width=cw // 2,
            height=ch // 2,
            num_frames=frames,
            latents_normalized=False,
            output_type="latent",
            adain_factor=1.0 if self.config.variant == "one-step" else 0.0,
        ).frames
        latent = LTX2ConditionPipeline._normalize_latents(
            raw,
            self.vae.latents_mean,
            self.vae.latents_std,
            self.vae.config.scaling_factor,
        ).to(self._execution_device, self.transformer.dtype)
        if noise is None:
            noise = randn_tensor(
                latent.shape,
                generator=generator,
                device=latent.device,
                dtype=latent.dtype,
            )
        elif noise.shape != latent.shape:
            raise ValueError("Noise must match the conditioning latent shape")
        sigma = float(sigma_schedule(self.config.variant)[0])
        noisy = (1 - sigma) * latent + sigma * noise.to(latent)
        denoised = self.denoise_latents(
            noisy,
            pos,
            negative_prompt_embeds=neg,
            prompt_attention_mask=pos_mask,
            negative_attention_mask=neg_mask,
            frame_rate=frame_rate,
            teacache=teacache,
        )
        raw = LTX2ConditionPipeline._denormalize_latents(
            denoised,
            self.vae.latents_mean,
            self.vae.latents_std,
            self.vae.config.scaling_factor,
        )
        if output_type == "latent":
            result = raw
        else:
            decoded = self.vae.decode(raw.to(self.vae.dtype), return_dict=False)[0]
            result = upsample.video_processor.postprocess_video(
                decoded, output_type="np"
            )
            top, left = (ch - height) // 2, (cw - width) // 2
            result = result[:, :, top : top + height, left : left + width]
        self.maybe_free_model_hooks()
        return LTXPipelineOutput(frames=result)
