from typing import List, Optional, Tuple

import imageio
import torch
import torch.distributed as dist
from einops import rearrange
from termcolor import colored

from diffusion.model.nets.basic_modules import CachedGLUMBConvTemp
from diffusion.model.nets.sana_blocks import CachedCausalAttention


# 功能：生成整个长视频
class SanaTrainingPipeline:
    def __init__(
        self,
        denoising_step_list: List[int],
        scheduler,
        generator,
        same_step_across_blocks: bool = False,
        last_step_only: bool = False,
        num_max_frames: int = 21,
        context_noise: int = 0,
        batch_size: int = 1,
        **kwargs,
    ):
        """
        Sana training pipeline, refer to SelfForcingTrainingPipeline's interface

        Args:
            denoising_step_list: denoising step list
            scheduler: scheduler
            generator: Sana video generation model
            num_frame_per_block: number of frames per block
            same_step_across_blocks: whether to use the same step across all blocks
            context_noise: context noise
        """
        self.scheduler = scheduler
        # the generator here is expected to be SanaModelWrapper (FSDP can wrap)
        self.generator = generator
        self.denoising_step_list = denoising_step_list
        if self.denoising_step_list[-1] == 0:
            self.denoising_step_list = self.denoising_step_list[:-1]  # remove the zero timestep for inference

        # Sana specific hyperparameters
        # if the wrapper is passed, get the underlying SANA model for cache scanning
        self.sana_model = generator.model if hasattr(generator, "model") else generator
        # compatible: if the constructor is not explicitly provided, allow reading from kwargs
        self.num_frame_per_block = int(kwargs.get("num_frame_per_block", 10))
        self.num_max_frames = num_max_frames
        # number of chunks to generate per 'clip' call; default to 1
        self.num_chunks_per_clip = int(kwargs.get("num_chunks_per_clip", 2))
        self.context_noise = context_noise

        self.flow_shift = float(kwargs.get("timestep_shift", 3.0))
        # KV cache相关
        self.chunk_indices = None
        self.kv_cache = None
        self.cached_modules = None
        self.num_model_blocks = 0
        self.batch_size = batch_size
        # derive device/dtype from the model parameters
        try:
            p = next(self.sana_model.parameters())
            self.device = p.device
            self.dtype = p.dtype
        except Exception:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

        self.same_step_across_blocks = same_step_across_blocks
        print(f"[SanaTrainingPipeline] same_step_across_blocks={self.same_step_across_blocks}")
        self.last_step_only = last_step_only

        # initialize cached modules
        self._initialize_cached_modules()

        self.reset_state()
        self.num_cached_blocks = int(kwargs.get("num_cached_blocks", -1))
        self.update_kv_cache_by_end = kwargs.get("update_kv_cache_by_end", False)
        print(
            colored(
                f"Additional parameters: num_cached_blocks {self.num_cached_blocks}, update_kv_cache_by_end : {self.update_kv_cache_by_end}, last_step_only {last_step_only}"
            )
        )

    def reset_state(self):
        """reset training state"""
        chunk_indices = self._create_autoregressive_segments(self.num_max_frames, self.num_frame_per_block)
        self.state = {
            "current_chunk_index": 0,
            "conditional_info": None,
            "unconditional_info": None,
            "conditional_info_real": None,
            "unconditional_info_real": None,
            "chunk_indices": chunk_indices,
            # "has_switched": False,  # 跟踪是否已经切换过prompt
            # "previous_frames": None,  # 存储上一次生成的帧，用于重叠（最多21帧）
            # "temp_max_length": None,  # 当前序列的临时最大长度
        }

    def reach_max_frames(self) -> bool:
        """check if can generate more"""
        return self.state["current_chunk_index"] >= len(self.state["chunk_indices"]) - 1

    def setup_sequence(
        self,
        conditional_dict: dict,
        unconditional_dict: dict,
        conditional_dict_real: dict,
        unconditional_dict_real: dict,
    ):
        """setup new sequence"""

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        batch_size = self.batch_size
        self.reset_state()
        self.state["conditional_info"] = conditional_dict
        self.state["unconditional_info"] = unconditional_dict
        self.state["conditional_info_real"] = conditional_dict_real
        self.state["unconditional_info_real"] = unconditional_dict_real

        # initialize per-chunk KV cache containers
        self._initialize_kv_cache(batch_size=batch_size, dtype=self.dtype, device=self.device)

    def generate_and_sync_list(self, num_blocks, num_denoising_steps, device):
        rank = dist.get_rank() if dist.is_initialized() else 0
        if rank == 0:
            indices = torch.randint(
                low=0,
                high=num_denoising_steps,
                size=(num_blocks,),
                device=device,
            )
            if self.last_step_only:
                indices = torch.ones_like(indices) * (num_denoising_steps - 1)
        else:
            indices = torch.empty(num_blocks, dtype=torch.long, device=device)
        if dist.is_initialized():
            dist.broadcast(indices, src=0)
        return indices.tolist()

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

    def reach_max_frames(self) -> bool:
        """check if can generate more"""
        return self.state["current_chunk_index"] >= len(self.state["chunk_indices"]) - 1

    def generate_chunk_with_cache(
        self,
        noise: torch.Tensor,
        prompt_embeds: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        current_start_frame: int = 0,
        requires_grad: bool = True,
        return_sim_step: bool = False,
    ) -> Tuple[torch.Tensor, Optional[int], Optional[int]]:
        """
        denoise in chunk-wise manner, input/output is consistent with sample in SelfForcingFlowEuler.sample:
        - input/output: latents (B, C, T, H, W)
        - update latents chunk by chunk, timestep by timestep, and save/sync KV cache at the end of each chunk
        current_start_frame is used for RoPE in streaming training
        """
        # adapt input (B, C, T, H, W))
        if noise.dim() != 5:
            raise ValueError("noise should be 5D tensor")

        # print(f"[SanaTrainingPipeline] noise.shape={noise.shape}")

        latents = noise.clone()

        batch_size, num_latent_channels, video_frames, height, width = latents.shape
        device = latents.device

        condition = prompt_embeds.clone()
        if mask is not None:
            mask = mask.clone()

        # chunk split
        chunk_indices = self.create_autoregressive_segments(video_frames)
        num_chunks = len(chunk_indices) - 1

        # Determine gradient-enabled range — if requires_grad=False, disable everywhere
        if not requires_grad:
            start_gradient_frame_index = video_frames  # Out of range: no gradients anywhere
        else:
            pass

        if condition.shape[0] == batch_size:
            condition = condition.repeat_interleave(num_chunks, dim=0)
            mask = mask[None].repeat_interleave(num_chunks, dim=0) if mask is not None else None

        # each chunk internally will rebuild the scheduler and get timesteps (align with SelfForcingFlowEuler.sample)
        steps = max(1, len(self.denoising_step_list))

        # generate and sync exit_flags for each chunk (decide which denoise step to exit and as final result)
        exit_flags = self.generate_and_sync_list(num_chunks, steps, device)

        output = torch.zeros_like(latents)
        previous_chunk_index = 0
        for _kv_cache in self.kv_cache:
            if _kv_cache[0][-1] is not None:
                previous_chunk_index += 1

        # pred_x0 style
        for chunk_idx in range(num_chunks):
            # setup internal cache
            chunk_kv_cache = self._accumulate_kv_cache(self.kv_cache, chunk_idx + previous_chunk_index)

            chunk_condition = condition[chunk_idx].unsqueeze(0)
            chunk_mask = mask[chunk_idx][None] if mask is not None else None

            start_f = chunk_indices[chunk_idx]
            end_f = chunk_indices[chunk_idx + 1]
            latent_model_input = latents[:, :, start_f:end_f]

            # select exit step for current chunk
            exit_step_idx = exit_flags[0] if self.same_step_across_blocks else exit_flags[chunk_idx]
            batch_size = latent_model_input.shape[0]
            current_num_frames = latent_model_input.shape[2]
            for index, current_timestep in enumerate(self.denoising_step_list):
                timestep = torch.ones(latent_model_input.shape[0], device=device, dtype=torch.int64) * current_timestep
                is_exit_step = index == exit_step_idx
                if not is_exit_step:
                    with torch.no_grad():
                        flow_pred, pred_x0, _ = self.generator(
                            noisy_image_or_video=latent_model_input,
                            condition=chunk_condition,
                            timestep=timestep,
                            start_f=(start_f + current_start_frame),
                            end_f=(end_f + current_start_frame),
                            save_kv_cache=False,
                            mask=chunk_mask,
                            kv_cache=chunk_kv_cache,
                        )
                    if index < len(self.denoising_step_list) - 1:
                        flow_pred = rearrange(flow_pred, "b c f h w -> b f c h w")
                        pred_x0 = rearrange(pred_x0, "b c f h w -> b f c h w")
                        next_timestep = self.denoising_step_list[index + 1]
                        latent_model_input = self.scheduler.add_noise(
                            pred_x0.flatten(0, 1),
                            torch.randn_like(pred_x0.flatten(0, 1)),
                            next_timestep
                            * torch.ones([batch_size * current_num_frames], device=noise.device, dtype=torch.long),
                        ).unflatten(0, pred_x0.shape[:2])
                        latent_model_input = rearrange(latent_model_input, "b f c h w -> b c f h w")

                else:
                    flow_pred_grad, pred_x0_grad, _ = self.generator(
                        noisy_image_or_video=latent_model_input,
                        condition=chunk_condition,
                        timestep=timestep,
                        start_f=(start_f + current_start_frame),
                        end_f=(end_f + current_start_frame),
                        save_kv_cache=False,
                        mask=chunk_mask,
                        kv_cache=chunk_kv_cache,
                    )
                    if self.update_kv_cache_by_end and index < len(self.denoising_step_list) - 1:
                        flow_pred = rearrange(flow_pred_grad.detach(), "b c f h w -> b f c h w")
                        pred_x0 = rearrange(pred_x0_grad.detach(), "b c f h w -> b f c h w")
                        next_timestep = self.denoising_step_list[index + 1]
                        latent_model_input = self.scheduler.add_noise(
                            pred_x0.flatten(0, 1),
                            torch.randn_like(pred_x0.flatten(0, 1)),
                            next_timestep
                            * torch.ones([batch_size * current_num_frames], device=noise.device, dtype=torch.long),
                        ).unflatten(0, pred_x0.shape[:2])
                        latent_model_input = rearrange(latent_model_input, "b f c h w -> b c f h w")

                    else:
                        pred_x0 = pred_x0_grad
                        # exit current chunk timestep loop
                        break

            # immediately write to external kv cache: explicitly pass kv_cache, underlying returns updated kv_cache
            output[:, :, start_f:end_f] = pred_x0_grad.to(output.device)
            latent_model_input_for_cache = pred_x0
            timestep_zero = torch.zeros(latent_model_input_for_cache.shape[0], device=device, dtype=self.dtype)

            # add context noise
            pred_x0 = rearrange(pred_x0, "b c f h w -> b f c h w")
            denoised_pred = self.scheduler.add_noise(
                pred_x0.flatten(0, 1),
                torch.randn_like(pred_x0.flatten(0, 1)),
                timestep_zero * torch.ones([batch_size * current_num_frames], device=noise.device, dtype=torch.long),
            ).unflatten(0, pred_x0.shape[:2])
            latent_model_input_for_cache = rearrange(denoised_pred, "b f c h w -> b c f h w")
            with torch.no_grad():
                _, _, updated_kv_cache = self.generator(
                    noisy_image_or_video=latent_model_input_for_cache,
                    condition=chunk_condition,
                    timestep=timestep_zero,
                    start_f=(start_f + current_start_frame),
                    end_f=(end_f + current_start_frame),
                    save_kv_cache=True,
                    mask=chunk_mask,
                    kv_cache=chunk_kv_cache,
                )
            self.kv_cache[chunk_idx + previous_chunk_index] = updated_kv_cache

        # denoised timestep range (refer to last chunk timesteps and exit_flags)

        if not self.same_step_across_blocks:
            denoised_timestep_from, denoised_timestep_to = None, None
        else:
            if len(self.denoising_step_list) > 0:
                exit_idx = exit_flags[0]
                denoised_timestep_from = int(self.denoising_step_list[0])
                denoised_timestep_to = int(self.denoising_step_list[exit_idx])
            else:
                denoised_timestep_from, denoised_timestep_to = None, None

        return output, denoised_timestep_from, denoised_timestep_to

    def create_autoregressive_segments(self, total_frames):
        remained_frames = total_frames % self.num_frame_per_block
        num_chunks = total_frames // self.num_frame_per_block
        chunk_indices = [0]
        for i in range(num_chunks):
            cur_idx = chunk_indices[-1] + self.num_frame_per_block
            if i == 0:  # the first chunk is larger if there are remained frames
                cur_idx += remained_frames
            chunk_indices.append(cur_idx)
        return chunk_indices

    def inference_with_trajectory(
        self,
        noise: torch.Tensor,
        prompt_embeds: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        current_start_frame: int = 0,
        requires_grad: bool = True,
        return_sim_step: bool = False,
    ) -> Tuple[torch.Tensor, Optional[int], Optional[int]]:
        """
        denoise in chunk-wise manner, input/output is consistent with sample in SelfForcingFlowEuler.sample:
        - input/output: latents (B, C, T, H, W)
        - update latents chunk by chunk, timestep by timestep, and save/sync KV cache at the end of each chunk
        """
        # adapt input (B, C, T, H, W))
        if noise.dim() != 5:
            raise ValueError("noise should be 5D tensor")

        latents = noise

        batch_size, num_latent_channels, video_frames, height, width = latents.shape
        device = latents.device
        self._initialize_kv_cache(batch_size, self.dtype, device)

        # NOTE: noise is the entire long video latents, here only return the current clip frames
        # so do not allocate output until the current clip frame range is determined

        # prompt/cross-attn information is handled by the wrapper internally
        condition = prompt_embeds.clone()
        if mask is not None:
            mask = mask.clone()

        # chunk split
        chunk_indices = self.create_autoregressive_segments(video_frames)
        num_chunks = len(chunk_indices) - 1

        if condition.shape[0] == batch_size:
            condition = condition.repeat_interleave(num_chunks, dim=0)
            mask = mask[None].repeat_interleave(num_chunks, dim=0) if mask is not None else None

        # each chunk internally will rebuild the scheduler and get timesteps (align with SelfForcingFlowEuler.sample)
        steps = max(1, len(self.denoising_step_list))

        # generate and sync exit_flags for each chunk (decide which denoise step to exit and as final result)
        exit_flags = self.generate_and_sync_list(num_chunks, steps, device)

        output = torch.zeros_like(latents)
        # pred_x0 style
        for chunk_idx in range(num_chunks):
            # setup internal cache
            chunk_kv_cache = self._accumulate_kv_cache(self.kv_cache, chunk_idx)

            chunk_condition = condition[chunk_idx].unsqueeze(0)
            chunk_mask = mask[chunk_idx][None] if mask is not None else None

            start_f = chunk_indices[chunk_idx]
            end_f = chunk_indices[chunk_idx + 1]
            latent_model_input = latents[:, :, start_f:end_f]

            # select exit step for current chunk
            exit_step_idx = exit_flags[0] if self.same_step_across_blocks else exit_flags[chunk_idx]
            batch_size = latent_model_input.shape[0]
            current_num_frames = latent_model_input.shape[2]
            for index, current_timestep in enumerate(self.denoising_step_list):
                timestep = torch.ones(latent_model_input.shape[0], device=device, dtype=torch.int64) * current_timestep
                is_exit_step = index == exit_step_idx
                if not is_exit_step:
                    with torch.no_grad():
                        flow_pred, pred_x0, _ = self.generator(
                            noisy_image_or_video=latent_model_input,
                            condition=chunk_condition,
                            timestep=timestep,
                            start_f=start_f,
                            end_f=end_f,
                            save_kv_cache=False,
                            mask=chunk_mask,
                            kv_cache=chunk_kv_cache,
                        )
                    if index < len(self.denoising_step_list) - 1:
                        flow_pred = rearrange(flow_pred, "b c f h w -> b f c h w")
                        pred_x0 = rearrange(pred_x0, "b c f h w -> b f c h w")
                        next_timestep = self.denoising_step_list[index + 1]
                        latent_model_input = self.scheduler.add_noise(
                            pred_x0.flatten(0, 1),
                            torch.randn_like(pred_x0.flatten(0, 1)),
                            next_timestep
                            * torch.ones([batch_size * current_num_frames], device=noise.device, dtype=torch.long),
                        ).unflatten(0, pred_x0.shape[:2])
                        latent_model_input = rearrange(latent_model_input, "b f c h w -> b c f h w")

                else:
                    # 直接使用 (B, C, T, H, W) 作为 SANA 输入 (B, C, F, H, W)
                    flow_pred_grad, pred_x0_grad, _ = self.generator(
                        noisy_image_or_video=latent_model_input,
                        condition=chunk_condition,
                        timestep=timestep,
                        start_f=start_f,
                        end_f=end_f,
                        save_kv_cache=False,
                        mask=chunk_mask,
                        kv_cache=chunk_kv_cache,
                    )
                    if self.update_kv_cache_by_end and index < len(self.denoising_step_list) - 1:
                        flow_pred = rearrange(flow_pred_grad.detach(), "b c f h w -> b f c h w")
                        pred_x0 = rearrange(pred_x0_grad.detach(), "b c f h w -> b f c h w")
                        next_timestep = self.denoising_step_list[index + 1]
                        latent_model_input = self.scheduler.add_noise(
                            pred_x0.flatten(0, 1),
                            torch.randn_like(pred_x0.flatten(0, 1)),
                            next_timestep
                            * torch.ones([batch_size * current_num_frames], device=noise.device, dtype=torch.long),
                        ).unflatten(0, pred_x0.shape[:2])
                        latent_model_input = rearrange(latent_model_input, "b f c h w -> b c f h w")

                    else:
                        pred_x0 = pred_x0_grad
                        # 退出当前 chunk 的时间步循环
                        break

            # immediately write to external KV cache: explicitly pass kv_cache, underlying returns updated kv_cache
            output[:, :, start_f:end_f] = pred_x0_grad.to(output.device)
            latent_model_input_for_cache = pred_x0
            timestep_zero = torch.zeros(latent_model_input_for_cache.shape[0], device=device, dtype=self.dtype)

            # add context noise
            pred_x0 = rearrange(pred_x0, "b c f h w -> b f c h w")
            denoised_pred = self.scheduler.add_noise(
                pred_x0.flatten(0, 1),
                torch.randn_like(pred_x0.flatten(0, 1)),
                timestep_zero * torch.ones([batch_size * current_num_frames], device=noise.device, dtype=torch.long),
            ).unflatten(0, pred_x0.shape[:2])
            latent_model_input_for_cache = rearrange(denoised_pred, "b f c h w -> b c f h w")
            with torch.no_grad():
                _, _, updated_kv_cache = self.generator(
                    noisy_image_or_video=latent_model_input_for_cache,
                    condition=chunk_condition,
                    timestep=timestep_zero,
                    start_f=start_f,
                    end_f=end_f,
                    save_kv_cache=True,
                    mask=chunk_mask,
                    kv_cache=chunk_kv_cache,
                )
            self.kv_cache[chunk_idx] = updated_kv_cache

        # denoised timestep range (refer to the last chunk timesteps and exit_flags)

        if not self.same_step_across_blocks:
            denoised_timestep_from, denoised_timestep_to = None, None
        else:
            if len(self.denoising_step_list) > 0:
                exit_idx = exit_flags[0]
                denoised_timestep_from = int(self.denoising_step_list[0])
                denoised_timestep_to = int(self.denoising_step_list[exit_idx])
            else:
                denoised_timestep_from, denoised_timestep_to = None, None

        return output, denoised_timestep_from, denoised_timestep_to

    def _accumulate_kv_cache(self, kv_cache, chunk_idx):
        """recalculate and accumulate KV cache, align with ar_flow_euler_sampler.accumulate_kv_cache
        - cur_kv_cache[block_id] structure is [cum_vk, k_sum, tconv]
        - tconv is directly inherited from the previous block; cum_vk and k_sum are the sum of each block from 0 to chunk_idx-1
        """
        if chunk_idx == 0:
            return kv_cache[0]
        cur_kv_cache = kv_cache[chunk_idx]
        for block_id in range(self.num_model_blocks):
            # inherit the tconv cache from the previous block
            cur_kv_cache[block_id][2] = kv_cache[chunk_idx - 1][block_id][2]
            cum_vk, cum_k_sum = None, None
            #
            #  accumulate the incremental of all historical blocks
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
                # historical should produce non-empty cumulative
                assert cum_vk is not None and cum_k_sum is not None, "Cumulative vk and k_sum should not be None"

            cur_kv_cache[block_id][0] = cum_vk
            cur_kv_cache[block_id][1] = cum_k_sum

        return cur_kv_cache

    def _initialize_cached_modules(self):
        """initialize cached modules, refer to SelfForcingFlowEuler's implementation"""
        if self.cached_modules is not None:
            return self.cached_modules

        # Handle both DDP wrapped and unwrapped models
        model = self.sana_model.module if hasattr(self.sana_model, "module") else self.sana_model

        # Organize modules by block index
        cached_modules = []

        def collect_from_block(block, block_idx):
            """Collect cached modules from a single transformer block"""
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

        # get the blocks of the model
        if hasattr(model, "blocks"):  # Common pattern
            blocks = model.blocks
        elif hasattr(model, "transformer_blocks"):
            blocks = model.transformer_blocks
        elif hasattr(model, "layers"):
            blocks = model.layers
        else:
            raise ValueError("Sana model does not have any blocks")

        # Collect modules from each block
        self.num_model_blocks = len(blocks)
        for block_idx, block in enumerate(blocks):
            block_modules = collect_from_block(block, block_idx)
            cached_modules.append(block_modules)

        self.cached_modules = cached_modules

    def _initialize_kv_cache(self, batch_size, dtype, device):
        """Initialize per-chunk KV cache containers for SANA cached modules."""
        chunk_indices = self.state["chunk_indices"]
        num_chunks = max(0, len(chunk_indices) - 1)
        kv_cache: list = []
        for _ in range(num_chunks):
            # For each chunk, per-block entries: [cum_vk, k_sum, tconv]
            kv_cache.append([[None, None, None] for _ in range(self.num_model_blocks)])
        self.kv_cache = kv_cache

    def _initialize_crossattn_cache(self, batch_size, dtype, device):
        pass

    def clear_kv_cache(self):
        chunk_indices = self.state["chunk_indices"]
        num_chunks = max(0, len(chunk_indices) - 1)
        self.kv_cache = []
        for _ in range(num_chunks):
            self.kv_cache.append([[None, None, None] for _ in range(self.num_model_blocks)])

        print(f"[SanaTrainingPipeline] clear_kv_cache for {num_chunks} chunks")

    def _clear_cache_gradients(self):
        """Detach gradients from all KV cache tensors (external and module-internal).
        This prevents autograd from tracking historical caches across chunks/clips.
        """
        # 1) External kv_cache list: shape [num_chunks][num_blocks][3]
        kv_cache = getattr(self, "kv_cache", None)
        if kv_cache is not None:
            for chunk_idx in range(len(kv_cache)):
                block_list = kv_cache[chunk_idx]
                for block_id in range(len(block_list)):
                    cache_triplet = block_list[block_id]
                    for i in range(len(cache_triplet)):
                        t = cache_triplet[i]
                        if isinstance(t, torch.Tensor):
                            if t.grad is not None:
                                t.grad = None
                            try:
                                t.detach_()
                                t.requires_grad_(False)
                            except Exception:
                                cache_triplet[i] = t.detach()
                                cache_triplet[i].requires_grad_(False)

        # 2) Module-internal caches
        cached_modules = getattr(self, "cached_modules", None)
        if cached_modules is not None:
            for block_modules in cached_modules:
                for module in block_modules:
                    module_cache = getattr(module, "kv_cache", None)
                    if isinstance(module_cache, list):
                        for i in range(len(module_cache)):
                            t = module_cache[i]
                            if isinstance(t, torch.Tensor):
                                if t.grad is not None:
                                    t.grad = None
                                try:
                                    t.detach_()
                                    t.requires_grad_(False)
                                except Exception:
                                    module_cache[i] = t.detach()
                                    module_cache[i].requires_grad_(False)
