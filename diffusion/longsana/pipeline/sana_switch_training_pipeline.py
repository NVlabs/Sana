from typing import Optional, Tuple

import torch
import torch.distributed as dist
from einops import rearrange

from diffusion.longsana.pipeline.sana_training_pipeline import SanaTrainingPipeline


class SanaSwitchTrainingPipeline(SanaTrainingPipeline):
    """
    This pipeline is used to train the SANA model with switch prompt.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prompt1_grad_on_switch = bool(kwargs.get("prompt1_grad_on_switch", True))
        self.recache_on_prompt_switch = bool(
            getattr(args, "recache_on_prompt_switch", kwargs.get("recache_on_prompt_switch", True))
        )

    def generate_chunk_with_cache(
        self,
        noise: torch.Tensor,
        prompt_embeds: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        current_start_frame: int = 0,
        requires_grad: bool = True,
        switch_frame_index: Optional[int] = None,
        switch_prompt_embeds: Optional[torch.Tensor] = None,
        return_sim_step: bool = False,
    ) -> Tuple[torch.Tensor, Optional[int], Optional[int]]:
        if switch_frame_index is None or switch_prompt_embeds is None:
            return super().generate_chunk_with_cache(
                noise=noise,
                prompt_embeds=prompt_embeds,
                mask=mask,
                current_start_frame=current_start_frame,
                requires_grad=requires_grad,
                return_sim_step=return_sim_step,
            )

        if noise.dim() != 5:
            raise ValueError("noise should be 5D tensor")

        latents = noise.clone()
        batch_size, num_latent_channels, video_frames, height, width = latents.shape
        device = latents.device

        # chunk
        chunk_indices = self.create_autoregressive_segments(video_frames)
        num_chunks = len(chunk_indices) - 1

        # gradient start range
        if not requires_grad:
            start_gradient_frame_index = video_frames
        else:
            start_gradient_frame_index = 0 if self.prompt1_grad_on_switch else int(switch_frame_index)

        # prepare condition and mask
        cond1 = prompt_embeds.clone()
        cond2 = switch_prompt_embeds.clone()
        if cond1.shape[0] == batch_size:
            cond1 = cond1.repeat_interleave(num_chunks, dim=0)
            cond2 = cond2.repeat_interleave(num_chunks, dim=0)
            mask = mask[None].repeat_interleave(num_chunks, dim=0) if mask is not None else None

        steps = max(1, len(self.denoising_step_list))
        exit_flags = self.generate_and_sync_list(num_chunks, steps, device)

        output = torch.zeros_like(latents)

        # already written chunk count
        previous_chunk_index = 0
        for _kv_cache in self.kv_cache:
            if _kv_cache[0][-1] is not None:
                previous_chunk_index += 1

        using_second_prompt = False

        for chunk_idx in range(num_chunks):
            # absolute frame range
            start_f = chunk_indices[chunk_idx]
            end_f = chunk_indices[chunk_idx + 1]

            # check if switch is triggered before entering current chunk
            if (not using_second_prompt) and ((start_f + current_start_frame) >= switch_frame_index):
                using_second_prompt = True
                if self.recache_on_prompt_switch and chunk_idx > 0:
                    prev_start_f = chunk_indices[chunk_idx - 1]
                    prev_end_f = chunk_indices[chunk_idx]
                    prev_latent_for_cache = output[:, :, prev_start_f:prev_end_f]
                    # use new prompt condition/mask
                    recache_condition = cond2[chunk_idx].unsqueeze(0)
                    recache_mask = mask[chunk_idx][None] if mask is not None else None
                    # accumulate previous chunk cache, ensure third item (history handle) is inherited
                    prev_accumulated_kv = self._accumulate_kv_cache(self.kv_cache, chunk_idx - 1 + previous_chunk_index)
                    timestep_zero_prev = torch.zeros(prev_latent_for_cache.shape[0], device=device, dtype=torch.int64)
                    try:
                        _, _, updated_prev_kv = self.generator(
                            noisy_image_or_video=prev_latent_for_cache,
                            condition=recache_condition,
                            timestep=timestep_zero_prev,
                            start_f=prev_start_f + current_start_frame,
                            end_f=prev_end_f + current_start_frame,
                            save_kv_cache=True,
                            mask=recache_mask,
                            kv_cache=prev_accumulated_kv,
                        )
                        self.kv_cache[chunk_idx - 1 + previous_chunk_index] = updated_prev_kv
                    except Exception as e:
                        try:
                            print(f"[SanaSwitchTrainingPipeline] recache failed at switch to chunk {chunk_idx}: {e}")
                        except Exception:
                            pass

            # accumulate history kv cache if needed
            chunk_kv_cache = self._accumulate_kv_cache(self.kv_cache, chunk_idx + previous_chunk_index)

            # select condition for current chunk
            chunk_condition = (cond2 if using_second_prompt else cond1)[chunk_idx].unsqueeze(0)
            chunk_mask = mask[chunk_idx][None] if mask is not None else None

            latent_model_input = latents[:, :, start_f:end_f]

            # select exit step
            exit_step_idx = exit_flags[0] if self.same_step_across_blocks else exit_flags[chunk_idx]
            bsz = latent_model_input.shape[0]
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
                            * torch.ones(
                                [bsz * current_num_frames],
                                device=noise.device,
                                dtype=torch.long,
                            ),
                        ).unflatten(0, pred_x0.shape[:2])
                        latent_model_input = rearrange(latent_model_input, "b f c h w -> b c f h w")
                else:
                    # last step: backprop if needed
                    if start_f + current_start_frame < start_gradient_frame_index:
                        with torch.no_grad():
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
                            * torch.ones(
                                [bsz * current_num_frames],
                                device=noise.device,
                                dtype=torch.long,
                            ),
                        ).unflatten(0, pred_x0.shape[:2])
                        latent_model_input = rearrange(latent_model_input, "b f c h w -> b c f h w")
                    else:
                        pred_x0 = pred_x0_grad
                        break

            # write output, and run with context noise again to save kv cache
            output[:, :, start_f:end_f] = pred_x0_grad.to(output.device)
            latent_model_input_for_cache = pred_x0
            timestep_zero = torch.zeros(latent_model_input_for_cache.shape[0], device=device, dtype=self.dtype)

            pred_x0 = rearrange(pred_x0, "b c f h w -> b f c h w")
            denoised_pred = self.scheduler.add_noise(
                pred_x0.flatten(0, 1),
                torch.randn_like(pred_x0.flatten(0, 1)),
                timestep_zero
                * torch.ones(
                    [bsz * current_num_frames],
                    device=noise.device,
                    dtype=torch.long,
                ),
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

        # denoised timestep range (same as parent class)
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

    def inference_with_trajectory(
        self,
        noise: torch.Tensor,
        prompt_embeds: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        current_start_frame: int = 0,
        requires_grad: bool = True,
        switch_frame_index: Optional[int] = None,
        switch_prompt_embeds: Optional[torch.Tensor] = None,
        return_sim_step: bool = False,
    ) -> Tuple[torch.Tensor, Optional[int], Optional[int]]:
        if switch_frame_index is None or switch_prompt_embeds is None:
            return super().inference_with_trajectory(
                noise,
                prompt_embeds,
                mask=mask,
                current_start_frame=current_start_frame,
                requires_grad=requires_grad,
                return_sim_step=return_sim_step,
            )

        if noise.dim() != 5:
            raise ValueError("noise should be 5D tensor")

        latents = noise
        batch_size, num_latent_channels, video_frames, height, width = latents.shape
        device = latents.device
        self._initialize_kv_cache(batch_size, self.dtype, device)

        # chunk split
        chunk_indices = self.create_autoregressive_segments(video_frames)
        num_chunks = len(chunk_indices) - 1

        # condition/mask copy
        cond1 = prompt_embeds.clone()
        cond2 = switch_prompt_embeds.clone()
        if cond1.shape[0] == batch_size:
            cond1 = cond1.repeat_interleave(num_chunks, dim=0)
            cond2 = cond2.repeat_interleave(num_chunks, dim=0)
            mask = mask[None].repeat_interleave(num_chunks, dim=0) if mask is not None else None

        steps = max(1, len(self.denoising_step_list))
        exit_flags = self.generate_and_sync_list(num_chunks, steps, device)

        output = torch.zeros_like(latents)

        using_second_prompt = False
        for chunk_idx in range(num_chunks):
            start_f = chunk_indices[chunk_idx]
            end_f = chunk_indices[chunk_idx + 1]

            # switch before the chunk starts
            if (not using_second_prompt) and ((start_f + current_start_frame) >= switch_frame_index):
                using_second_prompt = True

            # accumulate history if needed
            chunk_kv_cache = self._accumulate_kv_cache(self.kv_cache, chunk_idx)

            chunk_condition = (cond2 if using_second_prompt else cond1)[chunk_idx].unsqueeze(0)
            chunk_mask = mask[chunk_idx][None] if mask is not None else None
            latent_model_input = latents[:, :, start_f:end_f]

            exit_step_idx = exit_flags[0] if self.same_step_across_blocks else exit_flags[chunk_idx]
            bsz = latent_model_input.shape[0]
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
                            * torch.ones(
                                [bsz * current_num_frames],
                                device=noise.device,
                                dtype=torch.long,
                            ),
                        ).unflatten(0, pred_x0.shape[:2])
                        latent_model_input = rearrange(latent_model_input, "b f c h w -> b c f h w")
                else:
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
                            * torch.ones(
                                [bsz * current_num_frames],
                                device=noise.device,
                                dtype=torch.long,
                            ),
                        ).unflatten(0, pred_x0.shape[:2])
                        latent_model_input = rearrange(latent_model_input, "b f c h w -> b c f h w")
                    else:
                        pred_x0 = pred_x0_grad
                        break

            output[:, :, start_f:end_f] = pred_x0_grad.to(output.device)

            pred_x0 = rearrange(pred_x0, "b c f h w -> b f c h w")
            timestep_zero = torch.zeros(latent_model_input.shape[0], device=device, dtype=self.dtype)
            denoised_pred = self.scheduler.add_noise(
                pred_x0.flatten(0, 1),
                torch.randn_like(pred_x0.flatten(0, 1)),
                timestep_zero
                * torch.ones(
                    [bsz * current_num_frames],
                    device=noise.device,
                    dtype=torch.long,
                ),
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
