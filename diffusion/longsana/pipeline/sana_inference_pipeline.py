from typing import List, Optional

import imageio
import torch
import torch.distributed as dist
from einops import rearrange
from termcolor import colored

from diffusion.model.nets.basic_modules import CachedGLUMBConvTemp
from diffusion.model.nets.sana_blocks import CachedCausalAttention


class SanaInferencePipeline:
    def __init__(self, args, device, generator, text_encoder, vae, **kwargs):
        """
        SANA inference pipeline: generate a full video without gradients.

        The initialization signature is consistent with the use in Trainer:
            SanaInferencePipeline(args, device, generator, text_encoder, vae)
        """
        self.args = args
        self.device = device
        self.generator = generator
        self.text_encoder = text_encoder
        self.vae = vae

        self.scheduler = generator.get_scheduler()
        # hyperparams
        self.num_frame_per_block = int(getattr(args, "num_frame_per_block", kwargs.get("num_frame_per_block", 10)))
        self.denoising_step_list = list(
            getattr(args, "denoising_step_list", kwargs.get("denoising_step_list", [1000, 750, 500, 250]))
        )
        if len(self.denoising_step_list) > 0 and self.denoising_step_list[-1] == 0:
            self.denoising_step_list = self.denoising_step_list[:-1]
        self.flow_shift = float(getattr(args, "timestep_shift", kwargs.get("timestep_shift", 3.0)))
        print(f"[SanaInferencePipeline] denoising_step_list={self.denoising_step_list}")

        inner = generator.model if hasattr(generator, "model") else generator
        try:
            p = next(inner.parameters())
            self.model_device = p.device
            self.model_dtype = p.dtype
        except Exception:
            self.model_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

        # cache helpers
        self.cached_modules = None
        self.num_model_blocks = 0
        self.num_cached_blocks = int(getattr(args, "num_cached_blocks", -1))
        print(f"[SanaInferencePipeline] num_cached_blocks={self.num_cached_blocks}")

        self._initialize_cached_modules()

    def _initialize_cached_modules(self):
        if self.cached_modules is not None:
            return self.cached_modules
        model = self.generator.model if hasattr(self.generator, "model") else self.generator
        model = model.module if hasattr(model, "module") else model

        cached_modules = []

        def collect_from_block(block, block_idx):
            attention_modules = []
            conv_modules = []

            def collect_recursive(module):
                if isinstance(module, CachedCausalAttention):
                    attention_modules.append(module)
                elif isinstance(module, CachedGLUMBConvTemp):
                    conv_modules.append(module)
                for child in module.children():
                    collect_recursive(child)

            collect_recursive(block)
            return attention_modules + conv_modules

        if hasattr(model, "blocks"):
            blocks = model.blocks
        elif hasattr(model, "transformer_blocks"):
            blocks = model.transformer_blocks
        elif hasattr(model, "layers"):
            blocks = model.layers
        else:
            raise ValueError("Sana model does not have any blocks")

        self.num_model_blocks = len(blocks)
        for block_idx, block in enumerate(blocks):
            block_modules = collect_from_block(block, block_idx)
            cached_modules.append(block_modules)

        self.cached_modules = cached_modules
        return cached_modules

    def _create_autoregressive_segments(self, total_frames: int, base_chunk_frames: int) -> List[int]:
        remained_frames = total_frames % base_chunk_frames
        num_chunks = total_frames // base_chunk_frames
        chunk_indices = [0]
        for i in range(num_chunks):
            cur_idx = chunk_indices[-1] + base_chunk_frames
            if i == 0:
                cur_idx += remained_frames
            chunk_indices.append(cur_idx)
        if chunk_indices[-1] < total_frames:
            chunk_indices.append(total_frames)
        return chunk_indices

    def _initialize_kv_cache(self, num_chunks: int):
        kv_cache: list = []
        for _ in range(num_chunks):
            kv_cache.append([[None, None, None] for _ in range(self.num_model_blocks)])
        return kv_cache

    def _accumulate_kv_cache(self, kv_cache, chunk_idx):
        if chunk_idx == 0:
            return kv_cache[0]
        cur_kv_cache = kv_cache[chunk_idx]
        for block_id in range(self.num_model_blocks):
            cur_kv_cache[block_id][2] = kv_cache[chunk_idx - 1][block_id][2]
            cum_vk, cum_k_sum = None, None
            start_chunk_idx = chunk_idx - self.num_cached_blocks if self.num_cached_blocks > 0 else 0
            for i in range(start_chunk_idx, chunk_idx):
                prev = kv_cache[i][block_id]
                if prev[0] is not None and prev[1] is not None:
                    if cum_vk is None:
                        cum_vk = prev[0].clone()
                        cum_k_sum = prev[1].clone()
                    else:
                        cum_vk += prev[0]
                        cum_k_sum += prev[1]
            if chunk_idx > 0:
                assert cum_vk is not None and cum_k_sum is not None
            cur_kv_cache[block_id][0] = cum_vk
            cur_kv_cache[block_id][1] = cum_k_sum
        return cur_kv_cache

    @torch.no_grad()
    def inference(
        self,
        noise: torch.Tensor,
        text_prompts: List[str] = None,
        return_latents: bool = True,
        initial_latent: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        """
        Generate a full video.

        Args:
            noise: Gaussian noise latent of shape [B, T, C, H, W] or [B, C, T, H, W].
            text_prompts: Text prompts (length=B).
            return_latents: If True, return latent (B,T,C,H,W); otherwise, return pixel (B,T,C,H,W, normalized to 0..1 by upstream).
            initial_latent: Optional initial latent of shape [B, T0, C, H, W] (commonly T0=1).
        Returns:
            video: If return_latents=True, return [B, T, C, H, W]; otherwise, return pixel [B, T, C, H, W]
            info: dict
        """
        # normalize the latent shape to B,C,T,H,W
        if noise.dim() != 5:
            raise ValueError("noise should be a 5D tensor")

        latents_bcthw = noise
        if initial_latent is not None:
            if initial_latent.dim() != 5:
                raise ValueError("initial_latent must be 5D [B, T0, C, H, W]")
            # initial: BTCHW -> BCTHW
            init_bcthw = initial_latent.permute(0, 2, 1, 3, 4).contiguous()
            latents_bcthw = torch.cat([init_bcthw, latents_bcthw], dim=2)

        b, c, total_t, h, w = latents_bcthw.shape

        condition = None
        mask = None
        if text_prompts is not None:
            motion_score = getattr(self.args, "motion_score", 0)
            if motion_score > 0:
                text_prompts = [f"{prompt} motion score: {motion_score}." for prompt in text_prompts]
            text_embeddings = self.text_encoder.forward_chi(text_prompts, use_chi_prompt=True)
            condition = text_embeddings.get("prompt_embeds", None)
            mask = text_embeddings.get("mask", None)

        chunk_indices = self._create_autoregressive_segments(total_t, self.num_frame_per_block)
        num_chunks = len(chunk_indices) - 1
        kv_cache = self._initialize_kv_cache(num_chunks)

        if condition is not None and condition.shape[0] == b:
            condition = condition.repeat_interleave(num_chunks, dim=0)
            mask = mask[None].repeat_interleave(num_chunks, dim=0) if mask is not None else None

        output = torch.zeros_like(latents_bcthw)

        steps = max(1, len(self.denoising_step_list))
        print(colored(f"[SanaInferencePipeline] num_chunks={num_chunks}, steps={steps}", "red"))
        for chunk_idx in range(num_chunks):
            start_f = chunk_indices[chunk_idx]
            end_f = chunk_indices[chunk_idx + 1]
            local_latent = latents_bcthw[:, :, start_f:end_f]

            chunk_condition = condition[chunk_idx].unsqueeze(0) if condition is not None else None
            chunk_mask = mask[chunk_idx] if mask is not None else None

            chunk_kv_cache = self._accumulate_kv_cache(kv_cache, chunk_idx)
            batch_size = local_latent.shape[0]
            current_num_frames = local_latent.shape[2]
            for index, current_timestep in enumerate(self.denoising_step_list):
                timestep = (
                    torch.ones(local_latent.shape[0], device=self.model_device, dtype=self.model_dtype)
                    * current_timestep
                )
                if index < len(self.denoising_step_list) - 1:
                    flow_pred, pred_x0, _ = self.generator(
                        noisy_image_or_video=local_latent,
                        condition=chunk_condition,
                        timestep=timestep,
                        start_f=start_f,
                        end_f=end_f,
                        save_kv_cache=False,
                        mask=chunk_mask,
                        kv_cache=chunk_kv_cache,
                    )  # (B, C, F, H, W)
                    flow_pred = rearrange(flow_pred, "b c f h w -> b f c h w")
                    pred_x0 = rearrange(pred_x0, "b c f h w -> b f c h w")
                    next_timestep = self.denoising_step_list[index + 1]
                    local_latent = self.scheduler.add_noise(
                        pred_x0.flatten(0, 1),
                        torch.randn_like(pred_x0.flatten(0, 1)),
                        next_timestep
                        * torch.ones([batch_size * current_num_frames], device=noise.device, dtype=torch.long),
                    ).unflatten(0, pred_x0.shape[:2])
                    local_latent = rearrange(local_latent, "b f c h w -> b c f h w")

                else:
                    flow_pred, pred_x0, _ = self.generator(
                        noisy_image_or_video=local_latent,
                        condition=chunk_condition,
                        timestep=timestep,
                        start_f=start_f,
                        end_f=end_f,
                        save_kv_cache=False,
                        mask=chunk_mask,
                        kv_cache=chunk_kv_cache,
                    )
                    output[:, :, start_f:end_f] = pred_x0.to(output.device)

            # update kv cache
            latent_for_cache = output[:, :, start_f:end_f]
            timestep_zero = torch.zeros(latent_for_cache.shape[0], device=self.model_device, dtype=self.model_dtype)
            _, _, updated_kv_cache = self.generator(
                noisy_image_or_video=latent_for_cache,
                condition=chunk_condition,
                timestep=timestep_zero,
                start_f=start_f,
                end_f=end_f,
                save_kv_cache=True,
                mask=chunk_mask,
                kv_cache=chunk_kv_cache,
            )
            kv_cache[chunk_idx] = updated_kv_cache

        # output
        video_btchw = output.permute(0, 2, 1, 3, 4).contiguous()  # B,T,C,H,W
        info = {
            "total_frames": total_t,
            "num_chunks": num_chunks,
            "chunk_indices": chunk_indices,
        }

        if return_latents:
            return video_btchw, info
        else:
            pixel_bcthw = self.vae.decode_to_pixel(output)
            if isinstance(pixel_bcthw, list):
                pixel_bcthw = torch.stack(pixel_bcthw, dim=0)
            pixel_btchw = pixel_bcthw.permute(0, 2, 1, 3, 4).contiguous()
            return pixel_btchw, info
