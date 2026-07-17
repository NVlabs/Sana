from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.distributed as dist
from torchvision.transforms.functional import normalize as normalize_image_tensor

from diffusers import DiffusionPipeline
from diffusers.utils import BaseOutput
from diffusers.utils.torch_utils import randn_tensor

from .utils import (
    LOW_NOISE_TAIL_V1_DEFAULT_STEPS,
    batch_cfg_prompt_inputs,
    compute_refiner_sigmas,
)
from .scheduling_flow_unipc import FlowUniPCMultistepScheduler


TOKEN_LENGTH = 37698
HIDDEN_STATE_SKIP_LAYER = 0

PROMPT_TEMPLATE = (
    "<|im_start|>system\nGiven a user input that may include a text prompt alone, "
    "a text prompt with an image reference, or a text prompt with a video reference "
    "or a video reference alone, generate an \"Enhanced prompt\" that provides detailed "
    "visual descriptions suitable for video generation. Evaluate the level of detail "
    "in the user's input: if it is simple, enrich it by adding specifics about colors, "
    "shapes, sizes, textures, lighting, motion dynamics, camera movement, temporal "
    "progression, and spatial relationships to create vivid, concrete, and temporally "
    "coherent scenes to create vivid and concrete scenes. Please generate only the "
    "enhanced description for the prompt below and avoid including any additional "
    "commentary or evaluations:<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n"
    "<|im_start|>assistant\n"
)
IMG_PROMPT_TEMPLATE = "<|vision_start|><|image_pad|><|vision_end|>"
VIDEO_PROMPT_TEMPLATE = "<|vision_start|><|video_pad|><|vision_end|>"

DEFAULT_NEGATIVE_PROMPT = (
    '{"universal_negative": {"visual_quality": ["low quality", "worst quality", "blurry", "pixelated", "jpeg artifacts", "low resolution", "unstable color", "color flicker", "underexposed", "overexposed", "invisible subject", "subject hidden in darkness"], "artistic_style": ["painting", "illustration", "drawing", "cartoon", "3d render", "cgi", "sketch", "digital art"], "composition_and_content": ["text", "watermark", "signature", "logo", "subtitles", "pillarboxed", "side bars", "portrait image in landscape frame"], "temporal_and_motion_stability": ["flickering", "jittery", "motion blur", "temporal inconsistency", "warping", "morphing", "incoherent motion", "unnatural movement", "static object with sudden jump", "frame-to-frame inconsistency"], "material_and_structure": ["plastic-like glass", "unrealistic texture", "deformed bottle", "liquid freezing improperly", "distorted reflections"]}}'
)

# Still-image default (t2i): drops the whole temporal/motion block and the video-only
# codec/temporal terms that cannot apply to a single frame.
DEFAULT_NEGATIVE_PROMPT_IMAGE = (
    '{"universal_negative": {"visual_quality": ["low quality", "worst quality", "blurry", "pixelated", "jpeg artifacts", "low resolution", "underexposed", "overexposed", "invisible subject", "subject hidden in darkness"], "artistic_style": ["painting", "illustration", "drawing", "cartoon", "3d render", "cgi", "sketch", "digital art"], "composition_and_content": ["text", "watermark", "signature", "logo", "pillarboxed", "side bars", "portrait image in landscape frame"], "material_and_structure": ["plastic-like glass", "unrealistic texture", "deformed bottle", "distorted reflections"]}}'
)


@dataclass
class LingBotVideoPipelineOutput(BaseOutput):
    frames: Union[List[np.ndarray], torch.Tensor]


def _module_dtype(module: torch.nn.Module) -> torch.dtype:
    try:
        return next(module.parameters()).dtype
    except StopIteration:
        return torch.float32


def _transformer_timestep(timestep: torch.Tensor, transformer_dtype: torch.dtype) -> torch.Tensor:
    sigma = timestep.float() / 1000.0
    if transformer_dtype in {torch.bfloat16, torch.float16}:
        sigma = sigma.to(transformer_dtype)
    return (sigma * 1000.0).float()


def _transformer_autocast(device: torch.device, transformer_dtype: torch.dtype):
    if device.type != "cuda" or transformer_dtype not in {torch.bfloat16, torch.float16}:
        return nullcontext()
    return torch.autocast(device_type="cuda", dtype=transformer_dtype)


def _module_device(module: torch.nn.Module) -> torch.device:
    try:
        return next(module.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def _group_global_rank(group: Optional[Any], group_rank: int) -> int:
    if group is None:
        return group_rank
    get_global_rank = getattr(dist, "get_global_rank", None)
    if get_global_rank is None:
        return group_rank
    return int(get_global_rank(group, group_rank))


class LingBotVideoPipeline(DiffusionPipeline):
    """Minimal LingBotVideo t2v/t2i pipeline.

    Standard CFG runs as two independent transformer forwards unless batched CFG
    or CFG parallelism is explicitly requested.
    """

    model_cpu_offload_seq = "text_encoder->transformer->vae"

    def __init__(self, transformer, vae, text_encoder, processor, scheduler):
        super().__init__()
        if (
            scheduler is not None
            and scheduler.__class__.__name__ != FlowUniPCMultistepScheduler.__name__
        ):
            raise TypeError(
                "LingBotVideoPipeline requires vendored FlowUniPCMultistepScheduler; "
                f"got {scheduler.__class__.__name__}."
            )
        self.register_modules(
            transformer=transformer,
            vae=vae,
            text_encoder=text_encoder,
            processor=processor,
            scheduler=scheduler,
        )
        self.vae_scale_factor_temporal = 4
        self.vae_scale_factor_spatial = 8
        self.token_length = TOKEN_LENGTH
        self.hidden_state_skip_layer = HIDDEN_STATE_SKIP_LAYER
        self.prompt_template = PROMPT_TEMPLATE
        self.img_prompt_template = IMG_PROMPT_TEMPLATE
        self.video_prompt_template = VIDEO_PROMPT_TEMPLATE
        self._crop_start: Optional[int] = None

    @staticmethod
    def check_inputs(height: int, width: int, num_frames: int) -> None:
        if num_frames != 1 and (num_frames - 1) % 4 != 0:
            raise ValueError(f"`num_frames` must be 1 or 4n+1, got {num_frames}.")
        if height % 16 != 0 or width % 16 != 0:
            raise ValueError(f"`height` and `width` must be multiples of 16, got {height}x{width}.")

    @staticmethod
    def _apply_inpainting(latents: torch.Tensor, cond_latent: torch.Tensor) -> torch.Tensor:
        cond_t = cond_latent.shape[2]
        latents[:, :, :cond_t] = cond_latent.float()
        return latents

    @staticmethod
    def apply_text_to_template(text: str, template: str = PROMPT_TEMPLATE) -> str:
        return template.format(text)

    def _compute_crop_start(self) -> int:
        if self._crop_start is None:
            marker = "<|USER_INPUT_MARKER|>"
            marked = self.prompt_template.format(marker)
            marker_pos = marked.find(marker)
            if marker_pos < 0:
                self._crop_start = 0
            else:
                prefix = self.processor(
                    text=marked[:marker_pos],
                    images=None,
                    videos=None,
                    return_tensors="pt",
                )
                self._crop_start = int(prefix["input_ids"].shape[1])
        return self._crop_start

    def _build_prompt_inputs(
        self,
        prompt: Union[str, List[str]],
        images: Optional[Any] = None,
        videos: Optional[Any] = None,
        video_metadata: Optional[Any] = None,
        video_kwargs: Optional[Dict[str, Any]] = None,
    ):
        if isinstance(prompt, str):
            prompts = [prompt]
        else:
            prompts = list(prompt)

        visual_template = ""
        if images is not None:
            visual_template = self.img_prompt_template
        elif videos is not None:
            visual_template = self.video_prompt_template

        texts = [
            self.apply_text_to_template(visual_template + text, self.prompt_template)
            for text in prompts
        ]
        kwargs = dict(video_kwargs or {})
        return self.processor(
            text=texts,
            images=images,
            videos=videos,
            video_metadata=video_metadata,
            do_resize=False,
            truncation=True,
            max_length=self.token_length,
            padding="longest",
            return_tensors="pt",
            **kwargs,
        )

    @torch.no_grad()
    def encode_prompt(
        self,
        prompt: Union[str, List[str]],
        *,
        images: Optional[Any] = None,
        videos: Optional[Any] = None,
        video_metadata: Optional[Any] = None,
        video_kwargs: Optional[Dict[str, Any]] = None,
        device: Optional[Union[str, torch.device]] = None,
        return_inputs: bool = False,
    ):
        if self.text_encoder is None or self.processor is None:
            raise ValueError("`text_encoder` and `processor` are required for encode_prompt().")

        device = torch.device(device) if device is not None else self._execution_device
        inputs = self._build_prompt_inputs(
            prompt,
            images=images,
            videos=videos,
            video_metadata=video_metadata,
            video_kwargs=video_kwargs,
        )
        inputs = inputs.to(device)
        outputs = self.text_encoder(
            **inputs,
            output_hidden_states=self.hidden_state_skip_layer is not None,
        )
        if self.hidden_state_skip_layer is not None:
            prompt_embeds = outputs.hidden_states[-(self.hidden_state_skip_layer + 1)]
        else:
            prompt_embeds = outputs.last_hidden_state

        prompt_mask = inputs["attention_mask"]
        crop_start = self._compute_crop_start()
        if crop_start > 0:
            prompt_embeds = prompt_embeds[:, crop_start:]
            prompt_mask = prompt_mask[:, crop_start:]

        # Batch=1 can drop right padding before DiT inference.
        if prompt_embeds.shape[0] == 1:
            true_len = int(prompt_mask[0].sum().item())
            prompt_embeds = prompt_embeds[:, :true_len]
            prompt_mask = prompt_mask[:, :true_len]

        if return_inputs:
            return prompt_embeds, prompt_mask, inputs
        return prompt_embeds, prompt_mask

    def prepare_latents(
        self,
        num_frames: int,
        height: int,
        width: int,
        generator: Optional[torch.Generator],
        latents: Optional[torch.Tensor],
        device: torch.device,
    ) -> torch.Tensor:
        latent_frames = (num_frames - 1) // self.vae_scale_factor_temporal + 1
        latent_height = height // self.vae_scale_factor_spatial
        latent_width = width // self.vae_scale_factor_spatial
        shape = (
            1,
            self.transformer.config.in_channels,
            latent_frames,
            latent_height,
            latent_width,
        )
        if latents is None:
            return randn_tensor(shape, generator=generator, device=device, dtype=torch.float32)
        if tuple(latents.shape) != shape:
            raise ValueError(f"`latents` shape must be {shape}, got {tuple(latents.shape)}.")
        return latents.to(device=device, dtype=torch.float32)

    def _dit_latent_to_vae(self, latents: torch.Tensor) -> torch.Tensor:
        mean = torch.tensor(self.vae.config.latents_mean, device=latents.device, dtype=torch.float32)
        std_inv = 1.0 / torch.tensor(
            self.vae.config.latents_std, device=latents.device, dtype=torch.float32
        )
        mean = mean.view(1, -1, 1, 1, 1)
        std_inv = std_inv.view(1, -1, 1, 1, 1)
        return latents.float() / std_inv + mean

    def _vae_latent_to_dit(self, latents: torch.Tensor) -> torch.Tensor:
        mean = torch.tensor(self.vae.config.latents_mean, device=latents.device, dtype=torch.float32)
        std_inv = 1.0 / torch.tensor(
            self.vae.config.latents_std, device=latents.device, dtype=torch.float32
        )
        mean = mean.view(1, -1, 1, 1, 1)
        std_inv = std_inv.view(1, -1, 1, 1, 1)
        return (latents.float() - mean) * std_inv

    @torch.no_grad()
    def encode_video_latent(
        self,
        video: torch.Tensor,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        if self.vae is None:
            raise ValueError("`vae` is required to encode video latents.")
        vae_device = _module_device(self.vae)
        video = video.to(device=vae_device, dtype=torch.float32)
        bsz, channels, frames, height, width = video.shape
        flat_video = video.permute(0, 2, 1, 3, 4).reshape(bsz * frames, channels, height, width)
        norm_flat_video = normalize_image_tensor(
            flat_video,
            [0.5, 0.5, 0.5],
            [0.5, 0.5, 0.5],
            inplace=False,
        )
        norm_video = (
            norm_flat_video.reshape(bsz, frames, channels, height, width)
            .permute(0, 2, 1, 3, 4)
            .contiguous()
        )
        with torch.autocast(
            "cuda",
            dtype=torch.bfloat16,
            enabled=vae_device.type == "cuda",
        ):
            encoded = self.vae.encode(norm_video)
        if hasattr(encoded, "latent_dist"):
            latents = encoded.latent_dist.sample(generator)
        else:
            latents = encoded[0] if isinstance(encoded, tuple) else encoded
        return self._vae_latent_to_dit(latents).to(latents)

    @torch.no_grad()
    def _decode_latents(
        self,
        latents: torch.Tensor,
    ) -> List[np.ndarray]:
        vae_device = _module_device(self.vae)
        vae_dtype = _module_dtype(self.vae)
        vae_latents = self._dit_latent_to_vae(latents).to(device=vae_device, dtype=torch.float32)
        if vae_latents.ndim == 5:
            vae_latents = vae_latents.contiguous(memory_format=torch.channels_last_3d)
        autocast_dtype = (
            vae_dtype
            if vae_device.type == "cuda" and vae_dtype in {torch.bfloat16, torch.float16}
            else None
        )
        with torch.autocast(
            "cuda",
            dtype=autocast_dtype or torch.bfloat16,
            enabled=autocast_dtype is not None,
        ):
            decoded = self.vae.decode(vae_latents)
        frames = decoded[0] if isinstance(decoded, tuple) else decoded.sample
        frames = frames.float().clamp_(-1, 1)
        frames = (frames + 1.0) / 2.0
        frames = frames.permute(0, 2, 3, 4, 1).cpu().numpy()
        return [video for video in frames]

    @torch.no_grad()
    def __call__(
        self,
        prompt: str,
        negative_prompt: str = DEFAULT_NEGATIVE_PROMPT,
        height: int = 480,
        width: int = 480,
        num_frames: int = 81,
        num_inference_steps: int = 40,
        guidance_scale: float = 6.0,
        shift: float = 3.0,
        generator: Optional[torch.Generator] = None,
        latents: Optional[torch.Tensor] = None,
        cond_latent: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        prompt_mask: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_mask: Optional[torch.Tensor] = None,
        output_type: str = "np",
        cfg_parallel_group: Optional[Any] = None,
        batch_cfg: bool = False,
        null_cond_clone_zero: bool = False,
        t_thresh: Optional[float] = None,
        refiner_sigma_tail_steps: int = LOW_NOISE_TAIL_V1_DEFAULT_STEPS,
        offload_vae_during_denoise: bool = False,
        return_dict: bool = True,
    ) -> Union[LingBotVideoPipelineOutput, Tuple[Union[List[np.ndarray], torch.Tensor]]]:
        self.check_inputs(height, width, num_frames)
        if self.transformer is None or self.scheduler is None:
            raise ValueError("`transformer` and `scheduler` are required for generation.")

        device = self._execution_device
        do_cfg = guidance_scale > 1.0
        requested_batch_cfg = bool(batch_cfg)
        effective_batch_cfg = requested_batch_cfg
        batch_cfg_fallback_reason = None
        self._last_batch_cfg_requested = requested_batch_cfg
        self._last_effective_batch_cfg = bool(effective_batch_cfg and do_cfg)
        self._last_batch_cfg_fallback_reason = batch_cfg_fallback_reason

        cfg_parallel = cfg_parallel_group is not None
        if cfg_parallel and effective_batch_cfg:
            raise ValueError("`cfg_parallel_group` and `batch_cfg` are mutually exclusive.")
        cfg_parallel_rank = 0
        cfg_parallel_world_size = 1
        if cfg_parallel:
            if not dist.is_available() or not dist.is_initialized():
                raise ValueError("`cfg_parallel_group` requires an initialized process group.")
            if not do_cfg:
                raise ValueError("CFG parallel requires `guidance_scale > 1.0`.")
            cfg_parallel_rank = dist.get_rank(cfg_parallel_group)
            cfg_parallel_world_size = dist.get_world_size(cfg_parallel_group)
            if cfg_parallel_world_size != 2:
                raise ValueError(
                    f"CFG parallel currently requires exactly 2 ranks, got {cfg_parallel_world_size}."
                )

        if prompt_embeds is not None:
            if prompt_mask is None:
                raise ValueError("`prompt_mask` is required when `prompt_embeds` is provided.")
            prompt_embeds = prompt_embeds.to(device=device)
            prompt_mask = prompt_mask.to(device=device)
        if negative_prompt_embeds is not None:
            if negative_prompt_mask is None:
                raise ValueError(
                    "`negative_prompt_mask` is required when `negative_prompt_embeds` is provided."
                )
            negative_prompt_embeds = negative_prompt_embeds.to(device=device)
            negative_prompt_mask = negative_prompt_mask.to(device=device)

        if cfg_parallel and cfg_parallel_rank == 1:
            if negative_prompt_embeds is not None:
                negative_embeds, negative_mask = negative_prompt_embeds, negative_prompt_mask
            else:
                negative_embeds, negative_mask = self.encode_prompt(negative_prompt, device=device)
            prompt_embeds = prompt_mask = None
        else:
            if prompt_embeds is None:
                prompt_embeds, prompt_mask = self.encode_prompt(prompt, device=device)
            if do_cfg and not cfg_parallel:
                if null_cond_clone_zero:
                    negative_embeds = torch.zeros_like(prompt_embeds)
                    negative_mask = prompt_mask.clone()
                elif negative_prompt_embeds is not None:
                    negative_embeds, negative_mask = negative_prompt_embeds, negative_prompt_mask
                else:
                    negative_embeds, negative_mask = self.encode_prompt(negative_prompt, device=device)

        latents = self.prepare_latents(num_frames, height, width, generator, latents, device)
        # Clean temporal-prefix condition (e.g. the ti2v refiner's first-frame
        # latent): written into the latent before sampling and after every
        # scheduler step, so the fixed frames stay clean while the rest denoise
        # against them through attention.
        if cond_latent is not None:
            cond_latent = cond_latent.to(device=device, dtype=torch.float32)
            latents = self._apply_inpainting(latents, cond_latent)
        sigmas = compute_refiner_sigmas(
            sigma_max=float(self.scheduler.sigma_max),
            sigma_min=float(self.scheduler.sigma_min),
            num_inference_steps=num_inference_steps,
            shift=shift,
            t_thresh=t_thresh,
            tail_steps=refiner_sigma_tail_steps,
        )
        if sigmas is None:
            self.scheduler.set_timesteps(num_inference_steps, device=device, shift=shift)
        else:
            self.scheduler.set_timesteps(
                int(sigmas.shape[0]),
                device=device,
                sigmas=sigmas,
                shift=1.0,
            )
        transformer_dtype = _module_dtype(self.transformer)
        vae_restore_device: Optional[torch.device] = None
        vae_offloaded = False
        if offload_vae_during_denoise and output_type == "np" and self.vae is not None:
            vae_device = _module_device(self.vae)
            if vae_device.type == "cuda":
                self.vae.to("cpu")
                torch.cuda.empty_cache()
                vae_restore_device = vae_device
                vae_offloaded = True
        cfg_latent_src = _group_global_rank(cfg_parallel_group, 0)
        cfg_uncond_src = _group_global_rank(cfg_parallel_group, 1)
        for i, timestep in enumerate(self.progress_bar(self.scheduler.timesteps)):
            if cfg_parallel:
                dist.broadcast(latents, src=cfg_latent_src, group=cfg_parallel_group)
            timestep_batch = _transformer_timestep(timestep, transformer_dtype).expand(1).to(device)
            latent_model_input = latents
            if cfg_parallel:
                if cfg_parallel_rank == 0:
                    branch_embeds = prompt_embeds
                    branch_mask = prompt_mask
                    branch_name = "transformer.cond"
                else:
                    branch_embeds = negative_embeds
                    branch_mask = negative_mask
                    branch_name = "transformer.uncond"
                branch_model_input = branch_embeds.to(transformer_dtype)
                with _transformer_autocast(device, transformer_dtype):
                    branch_noise_pred = self.transformer(
                        latent_model_input,
                        timestep_batch,
                        branch_model_input,
                        encoder_attention_mask=branch_mask,
                        return_dict=False,
                    )[0].float()
                if cfg_parallel_rank == 0:
                    noise_pred = branch_noise_pred
                    noise_pred_uncond = torch.empty_like(noise_pred)
                else:
                    noise_pred_uncond = branch_noise_pred
                dist.broadcast(noise_pred_uncond, src=cfg_uncond_src, group=cfg_parallel_group)
                if cfg_parallel_rank != 0:
                    continue
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred - noise_pred_uncond)
            else:
                prompt_model_input = prompt_embeds.to(transformer_dtype)
                if do_cfg and effective_batch_cfg:
                    negative_model_input = negative_embeds.to(transformer_dtype)
                    cfg_embeds, cfg_mask = batch_cfg_prompt_inputs(
                        prompt_model_input,
                        prompt_mask,
                        negative_model_input,
                        negative_mask,
                        null_cond_clone_zero=False,
                    )
                    cfg_latents = torch.cat([latent_model_input, latent_model_input], dim=0)
                    cfg_timesteps = torch.cat([timestep_batch, timestep_batch], dim=0)
                    with _transformer_autocast(device, transformer_dtype):
                        noise_batched = self.transformer(
                            cfg_latents,
                            cfg_timesteps,
                            cfg_embeds,
                            encoder_attention_mask=cfg_mask,
                            return_dict=False,
                        )[0].float()
                    noise_pred, noise_pred_uncond = noise_batched.chunk(2, dim=0)
                    noise_pred = noise_pred_uncond + guidance_scale * (
                        noise_pred - noise_pred_uncond
                    )
                else:
                    with _transformer_autocast(device, transformer_dtype):
                        noise_pred = self.transformer(
                            latent_model_input,
                            timestep_batch,
                            prompt_model_input,
                            encoder_attention_mask=prompt_mask,
                            return_dict=False,
                        )[0].float()

                if do_cfg and not effective_batch_cfg:
                    negative_model_input = negative_embeds.to(transformer_dtype)
                    with _transformer_autocast(device, transformer_dtype):
                        noise_pred_uncond = self.transformer(
                            latent_model_input,
                            timestep_batch,
                            negative_model_input,
                            encoder_attention_mask=negative_mask,
                            return_dict=False,
                        )[0].float()
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred - noise_pred_uncond)

            latents = self.scheduler.step(
                noise_pred,
                timestep,
                latents,
                return_dict=False,
                generator=generator,
            )[0]
            if cond_latent is not None:
                latents = self._apply_inpainting(latents, cond_latent)

        if cfg_parallel:
            dist.barrier(group=cfg_parallel_group)
            if cfg_parallel_rank != 0:
                frames = latents if output_type == "latent" else []
                self.maybe_free_model_hooks()
                if not return_dict:
                    return (frames,)
                return LingBotVideoPipelineOutput(frames=frames)

        if output_type == "latent":
            frames = latents
        elif output_type == "np":
            if vae_offloaded and vae_restore_device is not None:
                self.vae.to(device=vae_restore_device)
                torch.cuda.empty_cache()
            frames = self._decode_latents(latents)
        else:
            raise ValueError(f"Unsupported output_type: {output_type}")

        self.maybe_free_model_hooks()
        if not return_dict:
            return (frames,)
        return LingBotVideoPipelineOutput(frames=frames)
