from typing import Dict, Optional

import torch
from einops import rearrange

from diffusion.longsana.pipeline.sana_training_pipeline import SanaTrainingPipeline


class SanaReverseRegPipeline(SanaTrainingPipeline):
    """Fixed-RoPE edit-to-source cycle pipeline used for V2V regularization."""

    def denoise_chunk(
        self,
        source_chunk_bcfhw: torch.Tensor,
        edited_chunk_bcfhw: torch.Tensor,
        prompt_embeds: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        current_start_frame: int = 0,
    ) -> Dict[str, torch.Tensor]:
        """Denoise source chunks while conditioning each chunk on the edited video.

        The source is independently noised at one sampled self-forcing timestep
        per sub-chunk.  Cache updates use the model prediction, never the clean
        source, so the reverse cycle does not introduce teacher forcing.
        """
        if not self.v2v:
            raise ValueError("SanaReverseRegPipeline requires a V2V model")
        if source_chunk_bcfhw.dim() != 5 or edited_chunk_bcfhw.dim() != 5:
            raise ValueError("source and edited chunks must be [B, C, F, H, W]")
        if source_chunk_bcfhw.shape != edited_chunk_bcfhw.shape:
            raise ValueError(
                "source and edited chunks must have identical shapes, got "
                f"{tuple(source_chunk_bcfhw.shape)} and {tuple(edited_chunk_bcfhw.shape)}"
            )
        if source_chunk_bcfhw.device != edited_chunk_bcfhw.device:
            raise ValueError("source and edited chunks must be on the same device")
        if len(self.denoising_step_list) == 0:
            raise ValueError("denoising_step_list must contain at least one non-zero timestep")

        device = source_chunk_bcfhw.device
        batch_size, _, total_frames, height, width = source_chunk_bcfhw.shape
        if total_frames == 0:
            raise ValueError("source and edited chunks must contain at least one frame")

        self._spatial_hw = height * width
        chunk_indices = self.create_autoregressive_segments(total_frames)
        num_sub_chunks = len(chunk_indices) - 1
        if num_sub_chunks == 0:
            raise ValueError(f"reverse chunks must contain at least {self.num_frame_per_block} frames")
        previous_chunk_index = current_start_frame // self.num_frame_per_block
        self._ensure_kv_cache_capacity(previous_chunk_index + num_sub_chunks)

        condition = prompt_embeds.clone()
        if mask is not None:
            mask = mask.clone()
        if condition.shape[0] == batch_size:
            condition = condition.repeat_interleave(num_sub_chunks, dim=0)
            mask = mask[None].repeat_interleave(num_sub_chunks, dim=0) if mask is not None else None

        exit_flags = self.generate_and_sync_list(
            num_sub_chunks,
            len(self.denoising_step_list),
            device,
        )

        flow_predictions = []
        x0_predictions = []
        sampled_noises = []
        sampled_timesteps = []

        for sub_idx, (start_f, end_f) in enumerate(zip(chunk_indices[:-1], chunk_indices[1:])):
            global_chunk_idx = previous_chunk_index + sub_idx
            chunk_kv_cache, _, sink_num, num_cached_frames = self.prepare_kv_cache(global_chunk_idx)

            sub_source = source_chunk_bcfhw[:, :, start_f:end_f]
            sub_edited = edited_chunk_bcfhw[:, :, start_f:end_f]
            sub_num_frames = end_f - start_f
            sub_condition = condition[sub_idx].unsqueeze(0)
            sub_mask = mask[sub_idx][None] if mask is not None else None

            absolute_start_f = current_start_frame + start_f
            absolute_end_f = current_start_frame + end_f
            rope_start_f, rope_end_f, frame_index = self.cache_positions(
                global_chunk_idx,
                absolute_start_f,
                absolute_end_f,
                sink_num,
                num_cached_frames,
                device,
            )

            exit_step_idx = exit_flags[0] if self.same_step_across_blocks else exit_flags[sub_idx]
            exit_timestep = self.denoising_step_list[exit_step_idx]
            timestep = torch.full(
                (batch_size,),
                exit_timestep,
                device=device,
                dtype=torch.int64,
            )

            sampled_noise = torch.randn_like(sub_source)
            source_bfchw = rearrange(sub_source, "b c f h w -> b f c h w")
            noise_bfchw = rearrange(sampled_noise, "b c f h w -> b f c h w")
            noisy_source = self.scheduler.add_noise(
                source_bfchw.flatten(0, 1),
                noise_bfchw.flatten(0, 1),
                exit_timestep
                * torch.ones(
                    batch_size * sub_num_frames,
                    device=device,
                    dtype=torch.long,
                ),
            ).unflatten(0, (batch_size, sub_num_frames))
            latent_model_input = rearrange(noisy_source, "b f c h w -> b c f h w")
            local_data_info = {"image_vae_embeds": sub_edited}

            flow_pred, pred_x0, _ = self.generator(
                noisy_image_or_video=latent_model_input,
                condition=sub_condition,
                timestep=timestep,
                start_f=rope_start_f,
                end_f=rope_end_f,
                frame_index=frame_index,
                save_kv_cache=False,
                mask=sub_mask,
                kv_cache=chunk_kv_cache,
                data_info=local_data_info,
            )

            flow_predictions.append(flow_pred)
            x0_predictions.append(pred_x0)
            sampled_noises.append(sampled_noise)
            sampled_timesteps.append(timestep)

            timestep_zero = torch.zeros(batch_size, device=device, dtype=self.dtype)
            cache_input = rearrange(pred_x0.detach(), "b c f h w -> b f c h w")
            cache_noisy = self.scheduler.add_noise(
                cache_input.flatten(0, 1),
                torch.randn_like(cache_input.flatten(0, 1)),
                timestep_zero
                * torch.ones(
                    batch_size * sub_num_frames,
                    device=device,
                    dtype=torch.long,
                ),
            ).unflatten(0, (batch_size, sub_num_frames))
            latent_for_cache = rearrange(cache_noisy, "b f c h w -> b c f h w")
            cache_for_update = self.copy_kv_cache_slots(chunk_kv_cache)

            with torch.no_grad():
                _, _, updated_kv_cache = self.generator(
                    noisy_image_or_video=latent_for_cache,
                    condition=sub_condition,
                    timestep=timestep_zero,
                    start_f=rope_start_f,
                    end_f=rope_end_f,
                    frame_index=frame_index,
                    save_kv_cache=True,
                    mask=sub_mask,
                    kv_cache=cache_for_update,
                    data_info=local_data_info,
                )
            self.update_kv_cache(global_chunk_idx, updated_kv_cache)

        return {
            "flow_pred": torch.cat(flow_predictions, dim=2),
            "pred_x0": torch.cat(x0_predictions, dim=2),
            "timestep": torch.stack(sampled_timesteps, dim=1),
            "noise": torch.cat(sampled_noises, dim=2),
        }
