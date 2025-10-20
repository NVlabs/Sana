import os

import torch
from diffusers import FlowMatchEulerDiscreteScheduler
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3 import retrieve_timesteps
from diffusers.utils.torch_utils import randn_tensor
from tqdm import tqdm

from diffusion.model.nets.basic_modules import CachedGLUMBConvTemp
from diffusion.model.nets.sana_blocks import CachedCausalAttention


class AutoregressiveChunkFlowEuler:
    """
    Autoregressive extension of ChunkFlowEuler that can generate videos longer than the training length
    by using overlapping segments while maintaining the 2-chunk structure.
    """

    def __init__(
        self,
        model_fn,
        condition,
        uncondition,
        cfg_scale,
        flow_shift=3.0,
        model_kwargs=None,
        base_model_frames=21,
        base_chunk_index=[0, 11],
    ):
        self.model = model_fn
        self.condition = condition
        self.uncondition = uncondition
        self.cfg_scale = cfg_scale
        self.model_kwargs = model_kwargs or {}
        self.mask = model_kwargs.pop("mask", None)
        self.scheduler = FlowMatchEulerDiscreteScheduler(shift=flow_shift)

        # Model training configuration
        self.model_chunk_size = (
            base_model_frames - base_chunk_index[-1]
        )  # all the new chunks should be the same size as last chunk
        self.model_num_chunks = len(base_chunk_index)  # Number of chunks during training
        self.base_model_frames = base_model_frames  # Total frames during training (with overlap)
        self.base_chunk_index = base_chunk_index

    def create_autoregressive_segments(self, total_frames):
        """
        Create autoregressive segments for long video generation.
        Frame structure: first chunk (11 frames) + subsequent chunks (10 frames each)
        Total frames per segment = 11 + 10*(num_chunks-1)

        For autoregressive generation:
        - Segment 1: chunks with sizes [11, 10, 10, ...] → chunk_indices [0, 11, 21, ...]
        - Segment 2: chunks with sizes [10, 10, 10, ...] → chunk_indices [0, 10, 20, ...]
        - Segment 3: chunks with sizes [10, 10, 10, ...] → chunk_indices [0, 10, 20, ...]

        Args:
            total_frames: Total number of frames to generate

        Returns:
            List of (start_idx, end_idx, chunk_indices) tuples for each segment
        """
        current_start = 0

        # Calculate frames per segment based on model configuration
        first_segment_frames = self.base_chunk_index[1] - self.base_chunk_index[0]
        subsequent_segment_frames = self.model_chunk_size  # 10*N

        segment_idx = 0
        chunk_indices = [0]  # Start with first chunk at 0

        while current_start < total_frames:

            # First segment: 11 + 10*(N-1) frames with chunk_indices [0, 11, 21, ...]
            segment_frames = first_segment_frames if segment_idx == 0 else subsequent_segment_frames
            current_start += segment_frames
            chunk_indices.append(current_start)
            segment_idx += 1

        return chunk_indices

    def sample(self, latents, steps=50, generator=None, interval_k=0.5, **kwargs):
        """
        Main autoregressive sampling method.

        Args:
            latents: Input latents of shape (1, C, total_frames, H, W)
            steps: Number of denoising steps per segment
            generator: Random generator
            interval_k: Interval ratio for chunk staggering (second chunk starts after interval_k * steps)

        Returns:
            Denoised latents for the full video
        """
        device = self.condition.device
        timesteps, num_inference_steps = retrieve_timesteps(self.scheduler, steps, device, None)
        do_classifier_free_guidance = self.cfg_scale > 1
        batch_size, num_latent_channels, total_frames, height, width = latents.shape

        # Handle short videos with single segment
        if total_frames <= self.base_model_frames:
            raise "Please use ChunkFlowEuler for short videos"

        # For long videos, use autoregressive generation
        chunk_indices = self.create_autoregressive_segments(
            total_frames
        )  # the last value of chunk_indices is the total frames, each chunk is chunk_indices[i] to chunk_indices[i+1]
        num_chunks = len(chunk_indices) - 1
        # Build the timestep matrix
        # Calculate when each chunk starts denoising
        chunk_start_steps = []
        for i in range(num_chunks):
            start_step = int(i * interval_k * steps)
            chunk_start_steps.append(start_step)

        # Total steps needed
        total_steps_needed = chunk_start_steps[-1] + steps if num_chunks > 1 else steps

        # Initialize timestep tracking matrix
        timestep_matrix = torch.full((num_chunks, total_steps_needed), -1, dtype=torch.float32, device=latents.device)

        # Fill the timestep matrix
        for chunk_idx in range(num_chunks):
            chunk_start = chunk_start_steps[chunk_idx]
            for step_idx, t in enumerate(timesteps):
                global_step = chunk_start + step_idx
                timestep_matrix[chunk_idx, global_step] = t.item()
            # Set remaining steps to 0 (fully denoised)
            for global_step in range(chunk_start + steps, total_steps_needed):
                timestep_matrix[chunk_idx, global_step] = 0

        assert (
            self.condition.shape[0] == batch_size or self.condition.shape[0] == num_chunks
        ), f"condition shape: {self.condition.shape}, batch_size: {batch_size}, num_chunks: {num_chunks}"
        if self.condition.shape[0] == batch_size:
            self.condition = self.condition.repeat_interleave(num_chunks, dim=0)
            self.mask = self.mask[None].repeat_interleave(num_chunks, dim=0) if self.mask is not None else None

        # Main denoising loop
        for global_step in tqdm(range(total_steps_needed), disable=os.getenv("DPM_TQDM", "False") == "True"):
            # Split latents by chunks, every step, use the latents in the past step
            chunk_latents = []
            for i in range(num_chunks):
                start, end = chunk_indices[i], chunk_indices[i + 1]
                chunk_latents.append(latents[:, :, start:end].clone())  #

            # Determine which chunks are active (started denoising)
            active_chunk_indices = []
            active_timesteps = []

            for chunk_idx in range(num_chunks):
                current_timestep = timestep_matrix[chunk_idx, global_step]
                if current_timestep >= 0:  # Chunk has started denoising
                    active_chunk_indices.append(chunk_idx)
                    active_timesteps.append(current_timestep)

            # Skip if no chunks are active
            if not active_chunk_indices:
                continue

            # sequentially get model_num_chunks chunks from active_chunk_indices
            for chunk_idx in active_chunk_indices:
                if active_timesteps[chunk_idx] == 0:
                    # already denoised
                    continue

                # use the prompt of the chunk with highest timestep
                prompt_index = active_chunk_indices[-1]
                prompt_embeds = self.condition[prompt_index].unsqueeze(0)
                mask = self.mask[prompt_index] if self.mask is not None else None
                if do_classifier_free_guidance:
                    prompt_embeds = torch.cat([self.uncondition, prompt_embeds], dim=0)

                # retrieve chunk_latents and timesteps and previously model_num_chunks chunks
                active_latents_list = []
                active_timestep_list = []
                active_chunk_index_list = []
                for i in range(max(chunk_idx - self.model_num_chunks + 1, 0), chunk_idx + 1):
                    _chunk_latents = chunk_latents[i]
                    active_latents_list.append(_chunk_latents)
                    active_timestep_list.extend([active_timesteps[i]] * _chunk_latents.shape[2])
                    active_chunk_index_list.append(chunk_indices[i])
                concatenated_latents = torch.cat(active_latents_list, dim=2)
                timestep_tensor = torch.tensor(active_timestep_list, device=device, dtype=torch.float32)  # f
                timestep_tensor = timestep_tensor.view(1, 1, -1, 1, 1).expand(
                    batch_size, num_latent_channels, -1, height, width
                )  # b,c,f,h,w

                # Prepare model inputs
                latent_model_input = (
                    torch.cat([concatenated_latents] * 2) if do_classifier_free_guidance else concatenated_latents
                )

                if do_classifier_free_guidance:
                    timestep_tensor = torch.cat([timestep_tensor, timestep_tensor], dim=0)

                # Model forward pass
                # chunk index should be the start index of the active chunks
                self.model_kwargs["chunk_index"] = [
                    active_chunk_index_list[i] - active_chunk_index_list[0] for i in range(len(active_chunk_index_list))
                ]  # after split, the chunk index should be the start index of the active chunks
                noise_pred = self.model(
                    latent_model_input,
                    timestep_tensor[:, :1, :, 0, 0],  # b,1,f
                    prompt_embeds,
                    mask=mask,
                    **self.model_kwargs,
                )

                if isinstance(noise_pred, Transformer2DModelOutput):
                    noise_pred = noise_pred[0]

                # Perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + self.cfg_scale * (noise_pred_text - noise_pred_uncond)
                    timestep_tensor = timestep_tensor.chunk(2)[0]

                # Compute the previous noisy sample x_t -> x_t-1
                latents_dtype = concatenated_latents.dtype
                concat_shape = concatenated_latents.shape

                # Get the actual timestep value for scheduler
                t = timesteps[min(global_step, len(timesteps) - 1)]

                # Reshape for scheduler
                denoised_latents = self.scheduler.step(
                    -noise_pred.reshape(batch_size, num_latent_channels, -1).transpose(1, 2),
                    t,
                    concatenated_latents.reshape(batch_size, num_latent_channels, -1).transpose(1, 2),
                    per_token_timesteps=timestep_tensor.reshape(batch_size, num_latent_channels, -1)[:, 0],
                    return_dict=False,
                )[0]
                denoised_latents = denoised_latents.transpose(1, 2).reshape(concat_shape)

                if denoised_latents.dtype != latents_dtype:
                    denoised_latents = denoised_latents.to(latents_dtype)

                # update latents of this chunk
                start, end = chunk_indices[chunk_idx], chunk_indices[chunk_idx + 1]
                latents[:, :, start:end] = denoised_latents[
                    :, :, self.model_kwargs["chunk_index"][-1] :
                ]  # replace the latents of this chunk with the last chunk in denoised latents

        return latents


class SelfForcingFlowEuler:
    """
    Modified version that uses internal cache but keeps your existing pipeline structure
    """

    def __init__(
        self,
        model_fn,
        condition,
        uncondition,
        cfg_scale,
        flow_shift=3.0,
        model_kwargs=None,
        base_chunk_frames=10,
    ):
        self.model = model_fn
        self.condition = condition
        self.uncondition = uncondition
        self.cfg_scale = cfg_scale
        self.model_kwargs = model_kwargs or {}
        self.mask = model_kwargs.pop("mask", None)
        self.flow_shift = flow_shift
        self.base_chunk_frames = base_chunk_frames  # Total frames during training (with overlap)
        self.rank = os.environ.get("RANK", 0)
        self.cached_modules = None  # Will be populated on first use
        # init cache
        self.get_cached_modules_by_block()

    def create_autoregressive_segments(self, total_frames):
        remained_frames = total_frames % self.base_chunk_frames
        num_chunks = total_frames // self.base_chunk_frames
        chunk_indices = [0]
        for i in range(num_chunks):
            cur_idx = chunk_indices[-1] + self.base_chunk_frames
            if i == 0:  # the first chunk is larger if there are remained frames
                cur_idx += remained_frames
            chunk_indices.append(cur_idx)
        return chunk_indices

    def get_cached_modules_by_block(self):
        """Get cached modules organized by transformer blocks"""
        if self.cached_modules is not None:
            return self.cached_modules

        # Handle both DDP wrapped and unwrapped models
        model = self.model.module if hasattr(self.model, "module") else self.model

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

        # Assuming your model has blocks organized in a list/sequential structure
        # Adjust this based on your actual model architecture
        if hasattr(model, "blocks"):  # Common pattern
            blocks = model.blocks
        elif hasattr(model, "transformer_blocks"):
            blocks = model.transformer_blocks
        elif hasattr(model, "layers"):
            blocks = model.layers
        else:
            raise ValueError("Model does not have any blocks")

        # Collect modules from each block
        self.num_model_blocks = len(blocks)
        for block_idx, block in enumerate(blocks):
            block_modules = collect_from_block(block, block_idx)
            cached_modules.append(block_modules)

        self.cached_modules = cached_modules
        print(f"Found {len(cached_modules)} blocks with cached modules")
        return cached_modules

    def setup_internal_caches_for_chunk(self, chunk_kv_cache):
        """Set up internal cache in modules using the accumulated cache"""
        cached_modules = self.get_cached_modules_by_block()

        for block_id in range(self.num_model_blocks):
            block_cache = chunk_kv_cache[block_id]  # [cusum_vk, cumsum_k_sum, tconv_cache]
            block_modules = cached_modules[block_id]

            for module in block_modules:
                # Each module gets its own reference to the cache for this block
                module.kv_cache = block_cache

    def sync_cache_from_modules_to_external(self, kv_cache, chunk_idx):
        """Sync the cache from modules back to external kv_cache after saving"""
        cached_modules = self.get_cached_modules_by_block()
        for block_id in range(self.num_model_blocks):
            block_modules = cached_modules[block_id]

            # Get the cache from the first module in the block (they should all reference the same cache)
            if block_modules:
                module_cache = block_modules[0].kv_cache
                if module_cache is not None:
                    # Update the external cache with what the modules saved
                    kv_cache[chunk_idx][block_id][0] = module_cache[0]  # cusum_vk
                    kv_cache[chunk_idx][block_id][1] = module_cache[1]  # cumsum_k_sum
                    kv_cache[chunk_idx][block_id][2] = module_cache[2]  # tconv_cache

    def accumulate_kv_cache(self, kv_cache, chunk_idx):
        """Keep your existing accumulation logic"""
        if chunk_idx == 0:
            return kv_cache[0]

        cur_kv_cache = kv_cache[
            chunk_idx
        ]  # num_model_blocks items, each item contains 4 values for cum_vk, k_sum, tconv and save_or_not
        for block_id in range(self.num_model_blocks):
            cur_kv_cache[block_id][2] = kv_cache[chunk_idx - 1][block_id][
                2
            ]  # use the tconv cache of the previous chunk
            cum_vk, cum_k_sum = None, None

            # Accumulate from all previous chunks
            for i in range(chunk_idx):
                if kv_cache[i][block_id][0] is not None and kv_cache[i][block_id][1] is not None:
                    if cum_vk is None:
                        cum_vk = kv_cache[i][block_id][0].clone()
                        cum_k_sum = kv_cache[i][block_id][1].clone()
                    else:
                        cum_vk += kv_cache[i][block_id][0]
                        cum_k_sum += kv_cache[i][block_id][1]
            if chunk_idx > 0:
                assert cum_vk is not None and cum_k_sum is not None, "Cumulative vk and k_sum should not be None"

            cur_kv_cache[block_id][0] = cum_vk
            cur_kv_cache[block_id][1] = cum_k_sum
        return cur_kv_cache

    def sample(self, latents, steps=50, generator=None, **kwargs):
        """
        Keep your existing sample method, just add internal cache setup
        """
        device = self.condition.device

        do_classifier_free_guidance = self.cfg_scale > 1
        batch_size, num_latent_channels, total_frames, height, width = latents.shape

        # Handle short videos with single segment
        if total_frames <= self.base_chunk_frames:
            raise ValueError("Please use FlowEuler for short videos")

        # For long videos, use autoregressive generation
        chunk_indices = self.create_autoregressive_segments(
            total_frames
        )  # the last value of chunk_indices is the total frames, each chunk is chunk_indices[i] to chunk_indices[i+1]
        num_chunks = len(chunk_indices) - 1
        kv_cache = []
        for i in range(num_chunks):
            kv_cache.append(
                [[None, None, None] for j in range(self.num_model_blocks)]
            )  # (cusum_vk, cumsum_k_sum, tconv_cache)

        assert (
            self.condition.shape[0] == batch_size or self.condition.shape[0] == num_chunks
        ), f"condition shape: {self.condition.shape}, batch_size: {batch_size}, num_chunks: {num_chunks}"
        if self.condition.shape[0] == batch_size:
            self.condition = self.condition.repeat_interleave(num_chunks, dim=0)
            self.mask = self.mask[None].repeat_interleave(num_chunks, dim=0) if self.mask is not None else None

        # Main denoising loop
        for chunk_idx in range(num_chunks):
            chunk_kv_cache = self.accumulate_kv_cache(kv_cache, chunk_idx)
            # NEW: Set up internal caches in modules
            self.setup_internal_caches_for_chunk(chunk_kv_cache)

            # denoise a chunk
            prompt_embeds = self.condition[chunk_idx].unsqueeze(0)
            if do_classifier_free_guidance:
                prompt_embeds = torch.cat([self.uncondition, prompt_embeds], dim=0)

            mask = self.mask[chunk_idx] if self.mask is not None else None

            self.scheduler = FlowMatchEulerDiscreteScheduler(shift=self.flow_shift)
            timesteps, num_inference_steps = retrieve_timesteps(self.scheduler, steps, device, None)

            for i, t in tqdm(
                list(enumerate(timesteps)),
                disable=os.getenv("DPM_TQDM", "False") == "True",
                desc=f"Processing chunk {chunk_idx}",
            ):

                # expand the latents if we are doing classifier free guidance
                latent_model_input = (
                    torch.cat([latents[:, :, chunk_indices[chunk_idx] : chunk_indices[chunk_idx + 1]]] * 2)
                    if do_classifier_free_guidance
                    else latents[:, :, chunk_indices[chunk_idx] : chunk_indices[chunk_idx + 1]]
                )
                # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                timestep = t.expand(latent_model_input.shape[0])

                noise_pred = self.model(
                    latent_model_input,
                    timestep,
                    prompt_embeds,
                    start_f=chunk_indices[chunk_idx],
                    end_f=chunk_indices[chunk_idx + 1],
                    save_kv_cache=False,  # Same parameter as your original
                    mask=mask,
                    **self.model_kwargs,
                )

                if isinstance(noise_pred, Transformer2DModelOutput):
                    noise_pred = noise_pred[0]

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + self.cfg_scale * (noise_pred_text - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                latents_dtype = latents.dtype
                latents[:, :, chunk_indices[chunk_idx] : chunk_indices[chunk_idx + 1]] = self.scheduler.step(
                    noise_pred,
                    t,
                    latents[:, :, chunk_indices[chunk_idx] : chunk_indices[chunk_idx + 1]],
                    return_dict=False,
                )[0]

                if latents.dtype != latents_dtype:
                    latents = latents.to(latents_dtype)

            # forward once to save the kv cache
            latent_model_input = (
                torch.cat([latents[:, :, chunk_indices[chunk_idx] : chunk_indices[chunk_idx + 1]]] * 2)
                if do_classifier_free_guidance
                else latents[:, :, chunk_indices[chunk_idx] : chunk_indices[chunk_idx + 1]]
            )
            # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
            timestep = torch.zeros(latent_model_input.shape[0], device=device)

            noise_pred = self.model(
                latent_model_input,
                timestep,
                prompt_embeds,
                start_f=chunk_indices[chunk_idx],
                end_f=chunk_indices[chunk_idx + 1],
                save_kv_cache=True,  # Same parameter as your original
                mask=mask,
                **self.model_kwargs,
            )

            # CRITICAL: Sync the saved cache back to external kv_cache
            self.sync_cache_from_modules_to_external(kv_cache, chunk_idx)

        # Clean up internal caches
        if self.cached_modules:
            for block_modules in self.cached_modules:
                for module in block_modules:
                    module.kv_cache = None

        return latents
