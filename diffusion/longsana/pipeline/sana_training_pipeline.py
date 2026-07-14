from typing import List, Optional, Tuple

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


# Generate the full long video.
class SanaTrainingPipeline:
    """Chunk-wise SANA training pipeline.

    V2V models use six cache slots per block. Existing LongSANA models keep
    their original three-slot cache behavior.
    """

    _V2V_CACHE_SLOTS = 6

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
        # KV-cache state.
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
        self.v2v = bool(kwargs.get("v2v", False))
        self.sink_token = bool(kwargs.get("sink_token", False))
        self._full_history_softmax_cache = self.num_cached_blocks < 0
        self.block_is_state_cached: List[bool] = []
        if self.v2v:
            self._detect_v2v_cache_blocks()
        print(
            colored(
                "Additional parameters: "
                f"num_cached_blocks {self.num_cached_blocks}, "
                f"update_kv_cache_by_end : {self.update_kv_cache_by_end}, "
                f"last_step_only {last_step_only}"
            )
        )

    def reset_state(self):
        """reset training state"""
        chunk_indices = self._create_autoregressive_segments(self.num_max_frames, self.num_frame_per_block)
        self._chunk_indices = chunk_indices
        self.state = {
            "current_chunk_index": 0,
            "conditional_info": None,
            "unconditional_info": None,
            "conditional_info_real": None,
            "unconditional_info_real": None,
            "chunk_indices": chunk_indices,
            # "has_switched": False,  # Track whether the prompt has already switched.
            # "previous_frames": None,  # Store previous frames for overlap, up to 21 frames.
            # "temp_max_length": None,  # Temporary maximum length for the current sequence.
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

    def _model_blocks(self):
        model = self.sana_model.module if hasattr(self.sana_model, "module") else self.sana_model
        if hasattr(model, "blocks"):
            return model.blocks
        if hasattr(model, "transformer_blocks"):
            return model.transformer_blocks
        if hasattr(model, "layers"):
            return model.layers
        raise ValueError("Sana model does not have blocks/transformer_blocks/layers")

    def _detect_v2v_cache_blocks(self) -> None:
        """Record whether each V2V attention block uses state or token cache."""
        from diffusion.model.nets.sana_v2v_attn_blocks import (
            V2VAfterRoPEGatedSoftmaxAttention,
            V2VStateCachedBiGDNAttention,
        )

        block_is_state_cached = []
        for block in self._model_blocks():
            attn = getattr(block, "attn", None)
            if isinstance(attn, V2VStateCachedBiGDNAttention):
                block_is_state_cached.append(True)
            elif isinstance(attn, V2VAfterRoPEGatedSoftmaxAttention):
                block_is_state_cached.append(False)
            else:
                raise ValueError(f"Unsupported V2V attention block: {type(attn).__name__}")

        self.block_is_state_cached = block_is_state_cached

    @staticmethod
    def _slice_chunk_data_info(
        image_vae_embeds: Optional[torch.Tensor],
        start_f: int,
        end_f: int,
    ) -> dict:
        """Build data_info whose V2V conditioning exactly matches one chunk."""
        if image_vae_embeds is None:
            return {}
        if image_vae_embeds.dim() != 5:
            raise ValueError("image_vae_embeds must be [B, C, T, H, W]")

        chunk_frames = end_f - start_f
        if end_f > image_vae_embeds.shape[2]:
            raise ValueError(
                "image_vae_embeds does not cover the requested chunk: "
                f"frames={image_vae_embeds.shape[2]}, requested=[{start_f}, {end_f})"
            )
        chunk_embeds = image_vae_embeds[:, :, start_f:end_f]
        if chunk_embeds.shape[2] != chunk_frames:
            raise ValueError(f"V2V conditioning has {chunk_embeds.shape[2]} frames, expected {chunk_frames}")
        return {"image_vae_embeds": chunk_embeds}

    def cache_positions(
        self,
        chunk_idx: int,
        start_f: int,
        end_f: int,
        sink_num: int,
        num_cached_frames: int,
        device: torch.device,
    ) -> Tuple[int, int, Optional[torch.Tensor]]:
        """Return position arguments for the current cache window."""
        if not self.v2v:
            return start_f, end_f, None

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
        if cache_start_chunk < len(self._chunk_indices):
            rope_start_f = self._chunk_indices[cache_start_chunk]
        else:
            rope_start_f = max(0, start_f - num_cached_frames)
        return rope_start_f, end_f, None

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
        image_vae_embeds: Optional[torch.Tensor] = None,
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
        self._spatial_hw = height * width

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
        if self.v2v:
            previous_chunk_index = current_start_frame // self.num_frame_per_block
        else:
            # Preserve the legacy path's cache-progress detection.  Its first
            # autoregressive chunk can be larger than num_frame_per_block.
            previous_chunk_index = sum(chunk_cache[0][-1] is not None for chunk_cache in (self.kv_cache or []))
        self._ensure_kv_cache_capacity(previous_chunk_index + num_chunks)

        # pred_x0 style
        for chunk_idx in range(num_chunks):
            # setup internal cache
            global_chunk_idx = chunk_idx + previous_chunk_index
            chunk_kv_cache, _, sink_num, num_cached_frames = self.prepare_kv_cache(global_chunk_idx)

            chunk_condition = condition[chunk_idx].unsqueeze(0)
            chunk_mask = mask[chunk_idx][None] if mask is not None else None

            start_f = chunk_indices[chunk_idx]
            end_f = chunk_indices[chunk_idx + 1]
            latent_model_input = latents[:, :, start_f:end_f]
            local_data_info = self._slice_chunk_data_info(image_vae_embeds, start_f, end_f)

            absolute_start_f = start_f + current_start_frame
            absolute_end_f = end_f + current_start_frame
            rope_start_f, rope_end_f, frame_index = self.cache_positions(
                global_chunk_idx,
                absolute_start_f,
                absolute_end_f,
                sink_num,
                num_cached_frames,
                device,
            )
            data_info_kwargs = {"data_info": local_data_info} if local_data_info else {}
            position_kwargs = {"frame_index": frame_index} if self.v2v else {}

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
                            start_f=rope_start_f,
                            end_f=rope_end_f,
                            save_kv_cache=False,
                            mask=chunk_mask,
                            kv_cache=chunk_kv_cache,
                            **position_kwargs,
                            **data_info_kwargs,
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
                        start_f=rope_start_f,
                        end_f=rope_end_f,
                        save_kv_cache=False,
                        mask=chunk_mask,
                        kv_cache=chunk_kv_cache,
                        **position_kwargs,
                        **data_info_kwargs,
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
            cache_for_update = self.copy_kv_cache_slots(chunk_kv_cache) if self.v2v else chunk_kv_cache
            with torch.no_grad():
                _, _, updated_kv_cache = self.generator(
                    noisy_image_or_video=latent_model_input_for_cache,
                    condition=chunk_condition,
                    timestep=timestep_zero,
                    start_f=rope_start_f,
                    end_f=rope_end_f,
                    save_kv_cache=True,
                    mask=chunk_mask,
                    kv_cache=cache_for_update,
                    **position_kwargs,
                    **data_info_kwargs,
                )
            if self.v2v:
                self.update_kv_cache(global_chunk_idx, updated_kv_cache)
            else:
                self.kv_cache[global_chunk_idx] = updated_kv_cache

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
                    # Use (B, C, T, H, W) directly as SANA input (B, C, F, H, W).
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
                        # Exit the current chunk timestep loop.
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

    def _empty_block_cache(self) -> list:
        slots = self._V2V_CACHE_SLOTS if self.v2v else 3
        return [None] * slots

    def _empty_chunk_cache(self) -> list:
        return [self._empty_block_cache() for _ in range(self.num_model_blocks)]

    @staticmethod
    def copy_kv_cache_slots(kv_cache: list) -> list:
        """Copy mutable slot containers while sharing their history tensors."""
        return [list(block_cache) for block_cache in kv_cache]

    def _ensure_kv_cache_capacity(self, num_chunks: int) -> None:
        if self.kv_cache is None:
            self.kv_cache = []
        while len(self.kv_cache) < num_chunks:
            self.kv_cache.append(self._empty_chunk_cache())

    def initialize_kv_cache(
        self,
        num_chunks: Optional[int] = None,
        batch_size: Optional[int] = None,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ) -> list:
        """Initialize an empty cache and return it.

        ``batch_size``, ``dtype`` and ``device`` remain accepted for parity
        with the existing private API; cache tensors are allocated lazily by
        the attention modules.
        """
        del batch_size, dtype, device
        if num_chunks is None:
            num_chunks = max(0, len(self.state["chunk_indices"]) - 1)
        self.kv_cache = [self._empty_chunk_cache() for _ in range(num_chunks)]
        return self.kv_cache

    def prepare_kv_cache(self, chunk_idx: int):
        """Return ``(cache, cached_chunks, sink_frames, cached_frames)``."""
        self._ensure_kv_cache_capacity(chunk_idx + 1)
        if self.v2v:
            return accumulate_fixed_rope_kv_cache(
                self.kv_cache,
                chunk_idx,
                block_is_state_cached=self.block_is_state_cached,
                num_cached_blocks=self.num_cached_blocks,
                sink_token=self.sink_token,
                full_history_softmax_cache=self._full_history_softmax_cache,
                chunk_indices=self._chunk_indices,
                spatial_hw=self._spatial_hw,
                cache_slots=self._V2V_CACHE_SLOTS,
            )
        return self._accumulate_kv_cache(self.kv_cache, chunk_idx), 0, 0, 0

    def promote_kv_cache(self, chunk_idx: int) -> None:
        """Promote a full-history softmax cache after saving the current chunk."""
        if not self.v2v or not self._full_history_softmax_cache or chunk_idx == 0:
            return
        promote_fixed_rope_full_history_cache(
            self.kv_cache,
            chunk_idx,
            block_is_state_cached=self.block_is_state_cached,
        )

    def update_kv_cache(self, chunk_idx: int, updated_kv_cache: list) -> list:
        """Store a model-produced cache and apply full-history promotion."""
        self._ensure_kv_cache_capacity(chunk_idx + 1)
        if self.v2v:
            if updated_kv_cache is None or len(updated_kv_cache) != self.num_model_blocks:
                raise ValueError("V2V generator must return one cache entry per model block")
            if any(len(block_cache) != self._V2V_CACHE_SLOTS for block_cache in updated_kv_cache):
                raise ValueError("V2V cache entries must contain six slots")
        self.kv_cache[chunk_idx] = updated_kv_cache
        self.promote_kv_cache(chunk_idx)
        return self.kv_cache[chunk_idx]

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

        blocks = self._model_blocks()

        # Collect modules from each block
        self.num_model_blocks = len(blocks)
        for block_idx, block in enumerate(blocks):
            block_modules = collect_from_block(block, block_idx)
            cached_modules.append(block_modules)

        self.cached_modules = cached_modules

    def _initialize_kv_cache(self, batch_size, dtype, device):
        """Initialize per-chunk KV cache containers for SANA cached modules."""
        return self.initialize_kv_cache(batch_size=batch_size, dtype=dtype, device=device)

    def _initialize_crossattn_cache(self, batch_size, dtype, device):
        pass

    def clear_kv_cache(self):
        num_chunks = max(0, len(self.state["chunk_indices"]) - 1)
        self.initialize_kv_cache(num_chunks=num_chunks)

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
