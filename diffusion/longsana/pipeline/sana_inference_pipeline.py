from typing import List, Optional

import imageio
import torch
import torch.distributed as dist
from einops import rearrange
from termcolor import colored

from diffusion.model.nets.basic_modules import CachedGLUMBConvTemp
from diffusion.model.nets.sana_blocks import CachedCausalAttention
from diffusion.scheduler.sana_streaming_cache import (
    accumulate_fixed_rope_kv_cache,
    promote_fixed_rope_full_history_cache,
)


class SanaInferencePipeline:
    _V2V_CACHE_SLOTS = 6

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
        self.v2v = bool(getattr(args, "v2v", kwargs.get("v2v", False)))
        self.sink_token = bool(getattr(args, "sink_token", kwargs.get("sink_token", False)))
        self._full_history_softmax_cache = self.num_cached_blocks < 0
        self.block_is_state_cached: List[bool] = []
        print(f"[SanaInferencePipeline] num_cached_blocks={self.num_cached_blocks}")

        self._initialize_cached_modules()
        if self.v2v:
            self._detect_v2v_cache_blocks()

    def _model_blocks(self):
        model = self.generator
        for _ in range(4):
            if any(hasattr(model, name) for name in ("blocks", "transformer_blocks", "layers")):
                break
            if hasattr(model, "module"):
                model = model.module
                continue
            if hasattr(model, "model"):
                model = model.model
                continue
            break

        if hasattr(model, "blocks"):
            return model.blocks
        if hasattr(model, "transformer_blocks"):
            return model.transformer_blocks
        if hasattr(model, "layers"):
            return model.layers
        raise ValueError("Sana model does not have any blocks")

    def _detect_v2v_cache_blocks(self):
        """Record the fixed-RoPE cache representation used by each block."""
        from diffusion.model.nets.sana_v2v_attn_blocks import (
            V2VAfterRoPEGatedSoftmaxAttention,
            V2VStateCachedBiGDNAttention,
        )

        block_is_state_cached = []
        for block in self._model_blocks():
            attn = getattr(block, "attn", None)
            cache_type = getattr(attn, "fixed_rope_cache_type", None)
            if cache_type == "state":
                block_is_state_cached.append(True)
            elif cache_type == "softmax":
                block_is_state_cached.append(False)
            elif cache_type is not None:
                raise ValueError(f"Unsupported fixed-RoPE cache type: {cache_type!r}")
            elif isinstance(attn, V2VStateCachedBiGDNAttention):
                block_is_state_cached.append(True)
            elif isinstance(attn, V2VAfterRoPEGatedSoftmaxAttention):
                block_is_state_cached.append(False)
            else:
                raise ValueError(f"Unsupported V2V attention block: {type(attn).__name__}")
        self.block_is_state_cached = block_is_state_cached

    def _initialize_cached_modules(self):
        if self.cached_modules is not None:
            return self.cached_modules
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

        blocks = self._model_blocks()

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
        slots = self._V2V_CACHE_SLOTS if self.v2v else 3
        kv_cache: list = []
        for _ in range(num_chunks):
            kv_cache.append([[None] * slots for _ in range(self.num_model_blocks)])
        return kv_cache

    @staticmethod
    def _copy_kv_cache_slots(kv_cache):
        """Copy mutable cache containers while sharing their history tensors."""
        return [list(block_cache) for block_cache in kv_cache]

    @staticmethod
    def _validate_v2v_conditioning(image_vae_embeds, latents):
        if not torch.is_tensor(image_vae_embeds) or image_vae_embeds.dim() != 5:
            raise ValueError("V2V image_vae_embeds must be a 5D tensor [B, C, T, H, W]")

        batch_size, channels, total_frames, height, width = latents.shape
        if image_vae_embeds.shape[0] != batch_size:
            raise ValueError(f"V2V conditioning batch size is {image_vae_embeds.shape[0]}, expected {batch_size}")
        if image_vae_embeds.shape[1] != channels:
            raise ValueError(f"V2V conditioning has {image_vae_embeds.shape[1]} channels, expected {channels}")
        if image_vae_embeds.shape[2] < total_frames:
            raise ValueError(
                f"V2V conditioning has {image_vae_embeds.shape[2]} frames, expected at least {total_frames}"
            )
        if image_vae_embeds.shape[-2:] != (height, width):
            raise ValueError(
                "V2V conditioning spatial shape is " f"{tuple(image_vae_embeds.shape[-2:])}, expected {(height, width)}"
            )
        if image_vae_embeds.device != latents.device:
            raise ValueError(
                f"V2V conditioning is on {image_vae_embeds.device}, expected the latent device {latents.device}"
            )

    @staticmethod
    def _slice_v2v_data_info(data_info, start_f, end_f):
        local_data_info = dict(data_info)
        image_vae_embeds = data_info["image_vae_embeds"]
        local_data_info["image_vae_embeds"] = image_vae_embeds[:, :, start_f:end_f]
        expected_frames = end_f - start_f
        if local_data_info["image_vae_embeds"].shape[2] != expected_frames:
            raise ValueError("V2V conditioning does not cover the requested chunk: " f"requested=[{start_f}, {end_f})")
        return local_data_info

    def _prepare_v2v_kv_cache(self, kv_cache, chunk_idx):
        return accumulate_fixed_rope_kv_cache(
            kv_cache,
            chunk_idx,
            block_is_state_cached=self.block_is_state_cached,
            num_cached_blocks=self.num_cached_blocks,
            sink_token=self.sink_token,
            full_history_softmax_cache=self._full_history_softmax_cache,
            chunk_indices=self._chunk_indices,
            spatial_hw=self._spatial_hw,
            cache_slots=self._V2V_CACHE_SLOTS,
        )

    def _v2v_cache_positions(self, chunk_idx, start_f, end_f, sink_num, num_cached_frames, device):
        current_num_frames = end_f - start_f
        if sink_num > 0:
            non_sink_count = num_cached_frames - sink_num + current_num_frames
            if non_sink_count < current_num_frames:
                raise ValueError("Invalid V2V cache frame accounting")
            sink_fi = torch.arange(sink_num, device=device)
            window_start_f = end_f - non_sink_count
            remaining_fi = torch.arange(window_start_f, end_f, device=device)
            return 0, end_f, torch.cat([sink_fi, remaining_fi], dim=0)

        cache_start_chunk = max(chunk_idx - self.num_cached_blocks, 0) if self.num_cached_blocks > 0 else 0
        return self._chunk_indices[cache_start_chunk], end_f, None

    def _update_v2v_kv_cache(self, kv_cache, chunk_idx, updated_kv_cache):
        if updated_kv_cache is None or len(updated_kv_cache) != self.num_model_blocks:
            raise ValueError("V2V generator must return one cache entry per model block")
        if any(len(block_cache) != self._V2V_CACHE_SLOTS for block_cache in updated_kv_cache):
            raise ValueError("V2V cache entries must contain six slots")

        kv_cache[chunk_idx] = updated_kv_cache
        if self._full_history_softmax_cache and chunk_idx > 0:
            promote_fixed_rope_full_history_cache(
                kv_cache,
                chunk_idx,
                block_is_state_cached=self.block_is_state_cached,
            )

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
        generator: Optional[torch.Generator] = None,
        **kwargs,
    ):
        """
        Generate a full video.

        Args:
            noise: Gaussian noise latent of shape [B, T, C, H, W] or [B, C, T, H, W].
            text_prompts: Text prompts (length=B).
            return_latents: If True, return latent (B,T,C,H,W); otherwise, return pixel (B,T,C,H,W, normalized to 0..1 by upstream).
            initial_latent: Optional initial latent of shape [B, T0, C, H, W] (commonly T0=1).
            generator: Optional random generator used for denoising re-noise.
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

        if self.v2v:
            data_info = kwargs.get("data_info", {})
            if data_info is None:
                data_info = {}
            if not isinstance(data_info, dict):
                raise ValueError("data_info must be a dictionary")
            if "image_vae_embeds" not in data_info:
                raise ValueError("V2V inference requires data_info['image_vae_embeds']")
            self._validate_v2v_conditioning(data_info["image_vae_embeds"], latents_bcthw)
            self._spatial_hw = h * w

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
        self._chunk_indices = chunk_indices
        num_chunks = len(chunk_indices) - 1
        kv_cache = self._initialize_kv_cache(num_chunks)

        if self.v2v:
            if condition is None or condition.shape[0] != b:
                raise ValueError("V2V inference requires one text prompt per input video")
            condition = condition.unsqueeze(1).expand(b, num_chunks, *condition.shape[1:])
            mask = mask.unsqueeze(1).expand(b, num_chunks, *mask.shape[1:]) if mask is not None else None
        elif condition is not None and condition.shape[0] == b:
            condition = condition.repeat_interleave(num_chunks, dim=0)
            mask = mask[None].repeat_interleave(num_chunks, dim=0) if mask is not None else None

        output = torch.zeros_like(latents_bcthw)

        steps = max(1, len(self.denoising_step_list))
        print(colored(f"[SanaInferencePipeline] num_chunks={num_chunks}, steps={steps}", "red"))
        for chunk_idx in range(num_chunks):
            start_f = chunk_indices[chunk_idx]
            end_f = chunk_indices[chunk_idx + 1]
            local_latent = latents_bcthw[:, :, start_f:end_f]

            if self.v2v:
                chunk_condition = condition[:, chunk_idx]
                chunk_mask = mask[:, chunk_idx] if mask is not None else None
                local_data_info = self._slice_v2v_data_info(data_info, start_f, end_f)
                chunk_kv_cache, _, sink_num, num_cached_frames = self._prepare_v2v_kv_cache(kv_cache, chunk_idx)
                rope_start_f, rope_end_f, frame_index = self._v2v_cache_positions(
                    chunk_idx,
                    start_f,
                    end_f,
                    sink_num,
                    num_cached_frames,
                    local_latent.device,
                )
                v2v_model_kwargs = {
                    "data_info": local_data_info,
                    "frame_index": frame_index,
                }
            else:
                chunk_condition = condition[chunk_idx].unsqueeze(0) if condition is not None else None
                chunk_mask = mask[chunk_idx] if mask is not None else None
                chunk_kv_cache = self._accumulate_kv_cache(kv_cache, chunk_idx)
                rope_start_f, rope_end_f = start_f, end_f
                v2v_model_kwargs = {}

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
                        start_f=rope_start_f,
                        end_f=rope_end_f,
                        save_kv_cache=False,
                        mask=chunk_mask,
                        kv_cache=chunk_kv_cache,
                        **v2v_model_kwargs,
                    )  # (B, C, F, H, W)
                    flow_pred = rearrange(flow_pred, "b c f h w -> b f c h w")
                    pred_x0 = rearrange(pred_x0, "b c f h w -> b f c h w")
                    next_timestep = self.denoising_step_list[index + 1]
                    flattened_pred_x0 = pred_x0.flatten(0, 1)
                    denoising_noise = torch.randn(
                        flattened_pred_x0.shape,
                        device=flattened_pred_x0.device,
                        dtype=flattened_pred_x0.dtype,
                        generator=generator,
                    )
                    local_latent = self.scheduler.add_noise(
                        flattened_pred_x0,
                        denoising_noise,
                        next_timestep
                        * torch.ones([batch_size * current_num_frames], device=noise.device, dtype=torch.long),
                    ).unflatten(0, pred_x0.shape[:2])
                    local_latent = rearrange(local_latent, "b f c h w -> b c f h w")

                else:
                    flow_pred, pred_x0, _ = self.generator(
                        noisy_image_or_video=local_latent,
                        condition=chunk_condition,
                        timestep=timestep,
                        start_f=rope_start_f,
                        end_f=rope_end_f,
                        save_kv_cache=False,
                        mask=chunk_mask,
                        kv_cache=chunk_kv_cache,
                        **v2v_model_kwargs,
                    )
                    output[:, :, start_f:end_f] = pred_x0.to(output.device)

            # update kv cache
            latent_for_cache = output[:, :, start_f:end_f]
            timestep_zero = torch.zeros(latent_for_cache.shape[0], device=self.model_device, dtype=self.model_dtype)
            cache_for_update = self._copy_kv_cache_slots(chunk_kv_cache) if self.v2v else chunk_kv_cache
            _, _, updated_kv_cache = self.generator(
                noisy_image_or_video=latent_for_cache,
                condition=chunk_condition,
                timestep=timestep_zero,
                start_f=rope_start_f,
                end_f=rope_end_f,
                save_kv_cache=True,
                mask=chunk_mask,
                kv_cache=cache_for_update,
                **v2v_model_kwargs,
            )
            if self.v2v:
                self._update_v2v_kv_cache(kv_cache, chunk_idx, updated_kv_cache)
            else:
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
