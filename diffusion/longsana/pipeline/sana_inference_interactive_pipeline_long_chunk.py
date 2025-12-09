from typing import List, Optional

import torch
from einops import rearrange

from diffusion.model.nets.basic_modules import CachedGLUMBConvTemp
from diffusion.model.nets.sana_blocks import CachedCausalAttention


class SanaInferenceInteractivePipelineLongChunk:
    def __init__(self, args, device, generator, text_encoder, vae, **kwargs):
        self.args = args
        self.device = device
        self.generator = generator
        self.text_encoder = text_encoder
        self.vae = vae

        self.denoising_step_list = list(
            getattr(args, "denoising_step_list", kwargs.get("denoising_step_list", [1000, 750, 500, 250]))
        )
        if len(self.denoising_step_list) > 0 and self.denoising_step_list[-1] == 0:
            self.denoising_step_list = self.denoising_step_list[:-1]

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
        self.recache_on_prompt_switch = bool(
            getattr(args, "recache_on_prompt_switch", kwargs.get("recache_on_prompt_switch", True))
        )

        self._initialize_cached_modules()

    def _normalize_prompts(self, prompts_in):
        """normalize the prompts that may be nested (list/tuple) or contain single element tuples to List[str].
        rules:
        - if it is a string, return [str]
        - if it is a list/tuple, then for each element:
            - if it is a string, keep it
            - if it is a list/tuple, if the length is 1, take the first element; otherwise, concatenate the stringified elements with spaces
        - for other types, convert to string.
        """
        # convert the top level to an iterable list
        if isinstance(prompts_in, str):
            seq = [prompts_in]
        elif isinstance(prompts_in, (list, tuple)):
            seq = list(prompts_in)
        else:
            seq = [str(prompts_in)]

        normalized = []
        for elem in seq:
            if isinstance(elem, str):
                normalized.append(elem)
            elif isinstance(elem, (list, tuple)):
                if len(elem) == 1:
                    normalized.append(str(elem[0]))
                else:
                    normalized.append(" ".join(str(x) for x in elem))
            else:
                normalized.append(str(elem))
        return normalized

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
    def inference_interactive(
        self,
        noise: torch.Tensor,
        prompts: List[str],
        return_latents: bool = True,
        initial_latent: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        if noise.dim() != 5:
            raise ValueError("noise should be a 5D tensor")

        latents_bcthw = noise
        if initial_latent is not None:
            if initial_latent.dim() != 5:
                raise ValueError("initial_latent must be 5D [B, T0, C, H, W]")
            init_bcthw = initial_latent.permute(0, 2, 1, 3, 4).contiguous()
            latents_bcthw = torch.cat([init_bcthw, latents_bcthw], dim=2)

        b, c, total_t, h, w = latents_bcthw.shape

        # normalize the prompts, ensure that the subsequent concatenation and encoding are both List[str]
        prompts = self._normalize_prompts(prompts)

        motion_score = getattr(self.args, "motion_score", 0)
        if motion_score is not None and motion_score > 0:
            prompts = [f"{p} motion score: {motion_score}." for p in prompts]

        num_segments = len(prompts)
        if num_segments <= 0:
            raise ValueError("prompts must be a non-empty list")

        try:
            print(f"[SanaInferenceInteractivePipeline] num_segments={num_segments}, total_t={total_t}")
        except Exception:
            pass

        # divide the total frames by the number of prompts, and the remainder is distributed upfront
        base = total_t // num_segments
        rem = total_t % num_segments
        lengths = [(base + 1) if i < rem else base for i in range(num_segments)]
        chunk_indices = [0]
        for L in lengths:
            chunk_indices.append(chunk_indices[-1] + L)
        num_chunks = len(chunk_indices) - 1

        text_embeddings = self.text_encoder.forward_chi(prompts, use_chi_prompt=True)
        full_condition = text_embeddings.get("prompt_embeds", None)
        full_mask = text_embeddings.get("mask", None)

        output = torch.zeros_like(latents_bcthw)
        kv_cache = self._initialize_kv_cache(num_chunks)

        try:
            mapping = [f"[{chunk_indices[i]}:{chunk_indices[i+1]}) -> prompt[{i}]" for i in range(num_chunks)]
            print("[SanaInferenceInteractivePipeline] segment mapping:", "; ".join(mapping))
        except Exception:
            pass

        for chunk_idx in range(num_chunks):
            start_f = chunk_indices[chunk_idx]
            end_f = chunk_indices[chunk_idx + 1]
            local_latent = latents_bcthw[:, :, start_f:end_f]

            try:
                cur_prompt = prompts[chunk_idx] if isinstance(prompts, (list, tuple)) else str(prompts)
                print(
                    f"[SanaInferenceInteractivePipeline] switch to prompt[{chunk_idx}] frames [{start_f}:{end_f}): {cur_prompt}"
                )
            except Exception:
                pass

            chunk_condition = None
            chunk_mask = None
            if full_condition is not None:
                chunk_condition = full_condition[chunk_idx : chunk_idx + 1]
                if full_mask is not None:
                    # maintain batch dimension (1, L) to avoid downstream attention mask dimension mismatch
                    chunk_mask = full_mask[chunk_idx : chunk_idx + 1]

            # before switching to a new prompt, recompute the KV cache of the previous segment using the "new prompt embeddings + previous segment video" at t=0
            if self.recache_on_prompt_switch and chunk_idx > 0:
                prev_start_f = chunk_indices[chunk_idx - 1]
                prev_end_f = chunk_indices[chunk_idx]
                prev_latent_for_cache = output[:, :, prev_start_f:prev_end_f]
                # first accumulate the cache up to the previous segment, ensuring the third item (history handle) inherits
                prev_accumulated_kv = self._accumulate_kv_cache(kv_cache, chunk_idx - 1)
                timestep_zero_prev = torch.zeros(
                    prev_latent_for_cache.shape[0], device=self.model_device, dtype=self.model_dtype
                )
                print(
                    f"[SanaInferenceInteractivePipelineLongChunk] recache: prev_latent_for_cache.shape={prev_latent_for_cache.shape}"
                )
                try:
                    _, _, updated_prev_kv = self.generator(
                        noisy_image_or_video=prev_latent_for_cache,
                        condition=chunk_condition,
                        timestep=timestep_zero_prev,
                        start_f=prev_start_f,
                        end_f=prev_end_f,
                        save_kv_cache=True,
                        mask=chunk_mask,
                        kv_cache=prev_accumulated_kv,
                    )
                    kv_cache[chunk_idx - 1] = updated_prev_kv
                except Exception as e:
                    try:
                        print(f"[SanaInferenceInteractivePipeline] recache failed at switch to chunk {chunk_idx}: {e}")
                    except Exception:
                        pass

            chunk_kv_cache = self._accumulate_kv_cache(kv_cache, chunk_idx)
            batch_size = local_latent.shape[0]
            current_num_frames = local_latent.shape[2]

            print(f"[SanaInferenceInteractivePipelineLongChunk] start_f={start_f}, end_f={end_f}")
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
                    )
                    flow_pred = rearrange(flow_pred, "b c f h w -> b f c h w")
                    pred_x0 = rearrange(pred_x0, "b c f h w -> b f c h w")
                    next_timestep = self.denoising_step_list[index + 1]
                    local_latent = (
                        self.generator.get_scheduler()
                        .add_noise(
                            pred_x0.flatten(0, 1),
                            torch.randn_like(pred_x0.flatten(0, 1)),
                            next_timestep
                            * torch.ones([batch_size * current_num_frames], device=noise.device, dtype=torch.long),
                        )
                        .unflatten(0, pred_x0.shape[:2])
                    )
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

        video_btchw = output.permute(0, 2, 1, 3, 4).contiguous()
        info = {
            "total_frames": total_t,
            "num_chunks": num_chunks,
            "chunk_indices": chunk_indices,
        }
        if return_latents:
            return video_btchw, info
        pixel_bcthw = self.vae.decode_to_pixel(output)
        if isinstance(pixel_bcthw, list):
            pixel_bcthw = torch.stack(pixel_bcthw, dim=0)
        pixel_btchw = pixel_bcthw.permute(0, 2, 1, 3, 4).contiguous()
        return pixel_btchw, info
