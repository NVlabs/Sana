from __future__ import annotations

import math
from typing import Any, Optional, Tuple, Union

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from PIL import Image
from diffusers.utils.torch_utils import randn_tensor

from .pipeline_lingbot_video import (
    DEFAULT_NEGATIVE_PROMPT,
    LingBotVideoPipeline,
    LingBotVideoPipelineOutput,
    _group_global_rank,
    _module_device,
    _module_dtype,
    _transformer_autocast,
    _transformer_timestep,
)
from .utils import batch_cfg_prompt_inputs


IMAGE_MIN_TOKEN_NUM = 4
IMAGE_MAX_TOKEN_NUM = 16384
MAX_RATIO = 200
SPATIAL_MERGE_SIZE = 2


def _round_by_factor(number: float, factor: int) -> int:
    return round(number / factor) * factor


def _ceil_by_factor(number: float, factor: int) -> int:
    return math.ceil(number / factor) * factor


def _floor_by_factor(number: float, factor: int) -> int:
    return math.floor(number / factor) * factor


def smart_resize(
    height: int,
    width: int,
    factor: int,
    min_pixels: Optional[int] = None,
    max_pixels: Optional[int] = None,
) -> Tuple[int, int]:
    max_pixels = max_pixels if max_pixels is not None else IMAGE_MAX_TOKEN_NUM * factor**2
    min_pixels = min_pixels if min_pixels is not None else IMAGE_MIN_TOKEN_NUM * factor**2
    if max_pixels < min_pixels:
        raise ValueError("max_pixels must be greater than or equal to min_pixels.")
    if max(height, width) / min(height, width) > MAX_RATIO:
        raise ValueError(f"absolute aspect ratio must be smaller than {MAX_RATIO}.")

    resized_height = max(factor, _round_by_factor(height, factor))
    resized_width = max(factor, _round_by_factor(width, factor))
    if resized_height * resized_width > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        resized_height = _floor_by_factor(height / beta, factor)
        resized_width = _floor_by_factor(width / beta, factor)
    elif resized_height * resized_width < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        resized_height = _ceil_by_factor(height * beta, factor)
        resized_width = _ceil_by_factor(width * beta, factor)
    return resized_height, resized_width


def _pixel_tensor_to_pil(pixel: torch.Tensor) -> Image.Image:
    """Match torchvision.transforms.ToPILImage for a float CHW image in [0, 1]."""
    frame = pixel[0, :, 0].detach().cpu().clamp(0, 1)
    array = frame.permute(1, 2, 0).mul(255).byte().numpy()
    return Image.fromarray(array, mode="RGB")


class LingBotVideoImageToVideoPipeline(LingBotVideoPipeline):
    """LingBotVideo ti2v pipeline.

    The condition frame is used twice: as visual input for Qwen3-VL and as a
    clean latent that is written into the beginning of the diffusion latent
    before sampling and after every scheduler step.
    """

    def preprocess_image(self, image: Image.Image, height: int, width: int) -> torch.Tensor:
        if image is None:
            raise ValueError("`image` is required when `image_tensor` is not provided.")
        raw = torch.from_numpy(np.array(image.convert("RGB"))).permute(2, 0, 1).unsqueeze(0).contiguous()
        old_h, old_w = raw.shape[-2:]
        scale = max(height / old_h, width / old_w)
        new_h = max(math.ceil(old_h * scale), height)
        new_w = max(math.ceil(old_w * scale), width)
        resized = F.interpolate(raw, size=(new_h, new_w), mode="bilinear", align_corners=False)
        top = int(round((new_h - height) / 2.0))
        left = int(round((new_w - width) / 2.0))
        cropped = resized[:, :, top : top + height, left : left + width].float() / 255.0
        return cropped.unsqueeze(2)

    def _vision_patch_size(self) -> int:
        for obj in (
            getattr(getattr(self.text_encoder, "config", None), "vision_config", None),
            getattr(getattr(self.processor, "image_processor", None), "config", None),
            getattr(self.processor, "image_processor", None),
        ):
            patch = getattr(obj, "patch_size", None)
            if patch is not None:
                return int(patch)
        return 16

    def _vlm_image(self, pixel: torch.Tensor) -> Image.Image:
        image = _pixel_tensor_to_pil(pixel)
        patch_factor = self._vision_patch_size() * SPATIAL_MERGE_SIZE
        width, height = image.size
        resized_height, resized_width = smart_resize(height, width, factor=patch_factor)
        return image.resize((resized_width, resized_height))

    @torch.no_grad()
    def encode_image_latent(
        self,
        pixel: torch.Tensor,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        if self.vae is None:
            raise ValueError("`vae` is required to encode image latents.")
        device = _module_device(self.vae)
        pixel = pixel.to(device=device, dtype=torch.float32)
        norm_pixel = (pixel - 0.5) / 0.5
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            latents = self.vae.encode(norm_pixel).latent_dist.sample(generator)

        mean = torch.tensor(self.vae.config.latents_mean, device=latents.device, dtype=torch.float32)
        std_inv = 1.0 / torch.tensor(
            self.vae.config.latents_std, device=latents.device, dtype=torch.float32
        )
        mean = mean.view(1, -1, 1, 1, 1)
        std_inv = std_inv.view(1, -1, 1, 1, 1)
        return (latents.float() - mean) * std_inv

    @torch.no_grad()
    def __call__(
        self,
        prompt: str,
        image: Optional[Image.Image] = None,
        image_tensor: Optional[torch.Tensor] = None,
        cond_latent: Optional[torch.Tensor] = None,
        negative_prompt: str = DEFAULT_NEGATIVE_PROMPT,
        height: int = 480,
        width: int = 480,
        num_frames: int = 81,
        num_inference_steps: int = 40,
        guidance_scale: float = 6.0,
        shift: float = 3.0,
        generator: Optional[torch.Generator] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        prompt_mask: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_mask: Optional[torch.Tensor] = None,
        output_type: str = "np",
        cfg_parallel_group: Optional[Any] = None,
        batch_cfg: bool = False,
        null_cond_clone_zero: bool = False,
        return_dict: bool = True,
    ) -> Union[LingBotVideoPipelineOutput, Tuple[Union[list, torch.Tensor]]]:
        self.check_inputs(height, width, num_frames)
        if self.transformer is None or self.scheduler is None:
            raise ValueError("`transformer` and `scheduler` are required for generation.")

        device = self._execution_device
        do_cfg = guidance_scale > 1.0
        requested_batch_cfg = bool(batch_cfg)
        effective_batch_cfg = requested_batch_cfg and do_cfg
        self._last_batch_cfg_requested = requested_batch_cfg
        self._last_effective_batch_cfg = effective_batch_cfg
        self._last_batch_cfg_fallback_reason = None

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

        pixel = image_tensor if image_tensor is not None else self.preprocess_image(image, height, width)
        pixel = pixel.to(device=device, dtype=torch.float32)
        vlm_image = self._vlm_image(pixel)

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
                negative_embeds, negative_mask = self.encode_prompt(
                    negative_prompt, images=[vlm_image], device=device
                )
            prompt_embeds = prompt_mask = None
        else:
            if prompt_embeds is None:
                prompt_embeds, prompt_mask = self.encode_prompt(prompt, images=[vlm_image], device=device)
            if do_cfg and not cfg_parallel:
                if null_cond_clone_zero:
                    negative_embeds = torch.zeros_like(prompt_embeds)
                    negative_mask = prompt_mask.clone()
                elif negative_prompt_embeds is not None:
                    negative_embeds, negative_mask = negative_prompt_embeds, negative_prompt_mask
                else:
                    negative_embeds, negative_mask = self.encode_prompt(
                        negative_prompt, images=[vlm_image], device=device
                    )

        if cond_latent is None:
            cond_latent = self.encode_image_latent(pixel, generator=generator)
        cond_latent = cond_latent.to(device=device, dtype=torch.float32)

        latents = self.prepare_latents(num_frames, height, width, generator, latents, device)
        latents = self._apply_inpainting(latents, cond_latent)
        self.scheduler.set_timesteps(num_inference_steps, device=device, shift=shift)
        transformer_dtype = _module_dtype(self.transformer)
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
                else:
                    branch_embeds = negative_embeds
                    branch_mask = negative_mask
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
                        null_cond_clone_zero=null_cond_clone_zero,
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

                    if do_cfg:
                        negative_model_input = negative_embeds.to(transformer_dtype)
                        with _transformer_autocast(device, transformer_dtype):
                            noise_pred_uncond = self.transformer(
                                latent_model_input,
                                timestep_batch,
                                negative_model_input,
                                encoder_attention_mask=negative_mask,
                                return_dict=False,
                            )[0].float()
                        noise_pred = noise_pred_uncond + guidance_scale * (
                            noise_pred - noise_pred_uncond
                        )

            latents = self.scheduler.step(
                noise_pred,
                timestep,
                latents,
                return_dict=False,
                generator=generator,
            )[0]
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
            frames = self._decode_latents(latents)
        else:
            raise ValueError(f"Unsupported output_type: {output_type}")

        self.maybe_free_model_hooks()
        if not return_dict:
            return (frames,)
        return LingBotVideoPipelineOutput(frames=frames)
