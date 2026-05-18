from __future__ import annotations

import copy
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, Sequence

import torch

from .ltx_refiner import LTXRefiner, LTXRefinerConfig, _first_stage2_sigma
from .ltx_streaming_patch import BlockTailCache, LTX2BlockStateReusePatch


def _calculate_shift(
    image_seq_len: int,
    *,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
) -> float:
    m = (max_shift - base_shift) / float(max_seq_len - base_seq_len)
    b = float(base_shift) - m * float(base_seq_len)
    return float(image_seq_len) * m + b


@dataclass
class StreamingWindowEntry:
    window_index: int
    global_start_frame: int
    global_end_frame_exclusive: int
    valid_num_frames: int
    pad_count: int
    latent: torch.Tensor


class LTXRefinerStreaming:
    """Isolated streaming helper for future low-level stage-2 denoising work.

    This class deliberately avoids adding more experimental behavior to the
    existing `LTXRefiner` user-facing entrypoints. It exposes the pieces needed
    to build a custom streaming denoise loop later:

    - stage-2 window slicing
    - prompt conditioning preparation
    - packed latent preparation that matches the high-level pipeline semantics
    - packed noisy latent preparation using the pipeline's own helper
    """

    def __init__(self, config_or_refiner: LTXRefinerConfig | LTXRefiner) -> None:
        if isinstance(config_or_refiner, LTXRefiner):
            self.refiner = config_or_refiner
            self._owns_refiner = False
        else:
            self.refiner = LTXRefiner(config_or_refiner)
            self._owns_refiner = True

    @property
    def device(self) -> torch.device:
        return self.refiner.device

    @property
    def dtype(self) -> torch.dtype:
        return self.refiner.dtype

    @property
    def vae_dtype(self) -> torch.dtype:
        return self.refiner.vae_dtype

    @property
    def pipe(self):
        return self.refiner._pipe

    @property
    def stage2_sigmas(self) -> list[float]:
        return self.refiner.stage2_sigmas

    def resolve_stage2_sigma_schedule(
        self,
        *,
        num_inference_steps: int,
        sigma_start_index: int = 0,
    ) -> list[float]:
        all_sigmas = list(self.stage2_sigmas)
        if int(sigma_start_index) < 0:
            raise ValueError("sigma_start_index must be >= 0.")
        if int(sigma_start_index) >= len(all_sigmas):
            raise ValueError(
                f"sigma_start_index={sigma_start_index} is out of range for stage-2 sigma list of length {len(all_sigmas)}."
            )
        if int(num_inference_steps) <= 0:
            raise ValueError("num_inference_steps must be > 0.")
        selected = all_sigmas[int(sigma_start_index) : int(sigma_start_index) + int(num_inference_steps)]
        if not selected:
            raise ValueError("Resolved sigma schedule is empty.")
        return [float(x) for x in selected]

    def close(self) -> None:
        if self._owns_refiner:
            self.refiner.close()

    def encode_video(self, video: torch.Tensor) -> torch.Tensor:
        return self.refiner.encode_video(video)

    def decode_latent_preview(self, latents: torch.Tensor) -> torch.Tensor:
        return self.refiner.decode_latent_preview(latents)

    def pack_video_latents(self, latents: torch.Tensor) -> torch.Tensor:
        return self.refiner._pack_video_latents(latents)

    def unpack_video_latents(
        self,
        latents: torch.Tensor,
        *,
        latent_num_frames: int,
        latent_height: int,
        latent_width: int,
    ) -> torch.Tensor:
        return self.refiner._unpack_video_latents(
            latents,
            latent_num_frames=latent_num_frames,
            latent_height=latent_height,
            latent_width=latent_width,
        )

    def build_window_entries(
        self,
        stage2_latent: torch.Tensor,
        *,
        window_num_frames: int,
        overlap_num_frames: int,
        pad_last_window: bool,
    ) -> list[StreamingWindowEntry]:
        entries = self.refiner._build_streaming_window_entries(
            stage2_latent,
            window_num_frames=int(window_num_frames),
            context_num_frames=int(overlap_num_frames),
            pad_last_window=bool(pad_last_window),
        )
        return [
            StreamingWindowEntry(
                window_index=int(idx),
                global_start_frame=int(entry["global_start_frame"]),
                global_end_frame_exclusive=int(entry["global_end_frame_exclusive"]),
                valid_num_frames=int(entry["valid_num_frames"]),
                pad_count=int(entry["pad_count"]),
                latent=entry["latent"],
            )
            for idx, entry in enumerate(entries)
        ]

    def prepare_prompt_conditioning(
        self,
        *,
        prompts: Sequence[str] | str,
        negative_prompt: Sequence[str] | str | None,
        batch_size: int,
        guidance_scale: float,
        max_sequence_length: int = 1024,
    ) -> dict[str, torch.Tensor | None]:
        prompt_batch = self.refiner._normalize_prompt_batch(prompts, batch_size, field_name="prompts")
        negative_prompt_batch = self.refiner._resolve_negative_prompt_batch(negative_prompt, batch_size)
        do_cfg = float(guidance_scale) > 1.0
        device = getattr(self.pipe, "_execution_device", None) or self.device

        (
            prompt_embeds,
            prompt_attention_mask,
            negative_prompt_embeds,
            negative_prompt_attention_mask,
        ) = self.pipe.encode_prompt(
            prompt=prompt_batch,
            negative_prompt=negative_prompt_batch,
            do_classifier_free_guidance=do_cfg,
            num_videos_per_prompt=1,
            prompt_embeds=None,
            negative_prompt_embeds=None,
            prompt_attention_mask=None,
            negative_prompt_attention_mask=None,
            max_sequence_length=int(max_sequence_length),
            device=device,
            dtype=self.dtype,
        )

        if do_cfg:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            prompt_attention_mask = torch.cat([negative_prompt_attention_mask, prompt_attention_mask], dim=0)

        conditioning: dict[str, torch.Tensor | None] = {
            "prompt_embeds": prompt_embeds,
            "prompt_attention_mask": prompt_attention_mask,
            "negative_prompt_embeds": negative_prompt_embeds,
            "negative_prompt_attention_mask": negative_prompt_attention_mask,
        }

        connectors = getattr(self.pipe, "connectors", None)
        if callable(connectors):
            additive_attention_mask = (1 - prompt_attention_mask.to(prompt_embeds.dtype)) * -1000000.0
            try:
                connector_prompt_embeds, connector_audio_prompt_embeds, connector_attention_mask = connectors(
                    prompt_embeds,
                    additive_attention_mask,
                    additive_mask=True,
                )
            except TypeError:
                connector_prompt_embeds, connector_audio_prompt_embeds, connector_attention_mask = connectors(
                    prompt_embeds,
                    additive_attention_mask,
                )
            conditioning.update(
                {
                    "connector_prompt_embeds": connector_prompt_embeds,
                    "connector_audio_prompt_embeds": connector_audio_prompt_embeds,
                    "connector_attention_mask": connector_attention_mask,
                }
            )
        else:
            conditioning.update(
                {
                    "connector_prompt_embeds": None,
                    "connector_audio_prompt_embeds": None,
                    "connector_attention_mask": None,
                }
            )

        return conditioning

    def prepare_packed_clean_video_latents(self, stage2_latent: torch.Tensor) -> torch.Tensor:
        latents = self.prepare_normalized_raw_video_latents(stage2_latent)
        return self.pack_video_latents(latents)

    def prepare_normalized_raw_video_latents(self, stage2_latent: torch.Tensor) -> torch.Tensor:
        latents = stage2_latent.to(device=self.device, dtype=self.dtype)
        return self.refiner.raw_to_normalized_latent(latents)

    def prepare_raw_noisy_video_latents(
        self,
        stage2_latent: torch.Tensor,
        *,
        noise_scale: float | None,
        seed: int | None,
    ) -> torch.Tensor:
        normalized_latents = self.prepare_normalized_raw_video_latents(stage2_latent)
        resolved_noise_scale = _first_stage2_sigma(self.stage2_sigmas) if noise_scale is None else float(noise_scale)
        generator = torch.Generator(device=self.device).manual_seed(int(seed)) if seed is not None else None
        noise = torch.randn(
            normalized_latents.shape,
            generator=generator,
            device=normalized_latents.device,
            dtype=normalized_latents.dtype,
        )
        scale = torch.tensor(
            [float(resolved_noise_scale)],
            device=normalized_latents.device,
            dtype=normalized_latents.dtype,
        )[:, None, None, None, None]
        return (1 - scale) * normalized_latents + scale * noise

    def prepare_packed_noisy_video_latents(
        self,
        stage2_latent: torch.Tensor,
        *,
        noise_scale: float | None,
        seed: int | None,
    ) -> torch.Tensor:
        resolved_noise_scale = _first_stage2_sigma(self.stage2_sigmas) if noise_scale is None else float(noise_scale)
        packed_latents = self.prepare_packed_clean_video_latents(stage2_latent)
        generator = torch.Generator(device=self.device).manual_seed(int(seed)) if seed is not None else None
        create_noised_state = getattr(self.pipe, "_create_noised_state", None)
        if not callable(create_noised_state):
            raise RuntimeError(
                "LTX2Pipeline does not expose `_create_noised_state`; cannot build packed noisy latents."
            )
        noised_latents = create_noised_state(
            packed_latents.to(device=self.device, dtype=self.dtype),
            noise_scale=float(resolved_noise_scale),
            generator=generator,
        )
        return noised_latents.to(device=self.device, dtype=self.dtype)

    def prepare_scheduler_state(
        self,
        *,
        num_inference_steps: int,
        video_sequence_length: int | None = None,
        sigmas: Sequence[float] | None = None,
    ) -> dict[str, Any]:
        scheduler = copy.deepcopy(self.pipe.scheduler)
        resolved_sigmas = list(self.stage2_sigmas if sigmas is None else sigmas)
        kwargs: dict[str, Any] = {}
        if video_sequence_length is not None:
            kwargs["mu"] = _calculate_shift(
                int(video_sequence_length),
                base_seq_len=int(self.pipe.scheduler.config.get("base_image_seq_len", 1024)),
                max_seq_len=int(self.pipe.scheduler.config.get("max_image_seq_len", 4096)),
                base_shift=float(self.pipe.scheduler.config.get("base_shift", 0.95)),
                max_shift=float(self.pipe.scheduler.config.get("max_shift", 2.05)),
            )
        try:
            scheduler.set_timesteps(
                num_inference_steps=int(num_inference_steps),
                device=self.device,
                sigmas=resolved_sigmas,
                **kwargs,
            )
        except TypeError:
            scheduler.set_timesteps(
                num_inference_steps=int(num_inference_steps),
                device=self.device,
                **kwargs,
            )
        timesteps = getattr(scheduler, "timesteps", None)
        if timesteps is None:
            raise RuntimeError("Scheduler did not expose `timesteps` after set_timesteps().")
        return {
            "scheduler": scheduler,
            "timesteps": timesteps,
            "sigmas": resolved_sigmas,
        }

    def prepare_packed_window_latents(
        self,
        window_entries: Sequence[StreamingWindowEntry],
        *,
        noisy_full_latents: torch.Tensor | None = None,
        clean_full_latents: torch.Tensor | None = None,
    ) -> list[dict[str, Any]]:
        if noisy_full_latents is None and clean_full_latents is None:
            raise ValueError("Either noisy_full_latents or clean_full_latents must be provided.")

        packed_entries: list[dict[str, Any]] = []
        for entry in window_entries:
            if noisy_full_latents is not None:
                raw_window = noisy_full_latents[
                    :,
                    :,
                    int(entry.global_start_frame) : int(entry.global_end_frame_exclusive),
                ]
            else:
                raw_window = clean_full_latents[
                    :,
                    :,
                    int(entry.global_start_frame) : int(entry.global_end_frame_exclusive),
                ]
            if int(raw_window.shape[2]) < int(entry.latent.shape[2]):
                pad_count = int(entry.latent.shape[2]) - int(raw_window.shape[2])
                raw_window = torch.cat([raw_window, raw_window[:, :, -1:].repeat(1, 1, pad_count, 1, 1)], dim=2)
            packed_entries.append(
                {
                    "window_index": int(entry.window_index),
                    "global_start_frame": int(entry.global_start_frame),
                    "global_end_frame_exclusive": int(entry.global_end_frame_exclusive),
                    "valid_num_frames": int(entry.valid_num_frames),
                    "pad_count": int(entry.pad_count),
                    "raw_latent": raw_window.to(device=self.device, dtype=self.dtype),
                    "packed_latent": self.pack_video_latents(raw_window.to(device=self.device, dtype=self.dtype)),
                }
            )
        return packed_entries

    def prepare_streaming_inputs(
        self,
        *,
        stage2_latent: torch.Tensor,
        prompts: Sequence[str] | str,
        negative_prompt: Sequence[str] | str | None,
        window_num_frames: int,
        overlap_num_frames: int,
        pad_last_window: bool,
        num_inference_steps: int,
        sigma_start_index: int,
        guidance_scale: float,
        noise_scale: float | None,
        seed: int,
    ) -> dict[str, Any]:
        if stage2_latent.ndim != 5:
            raise ValueError(
                f"Expected stage2_latent with shape [B, C, F, H, W], got shape={tuple(stage2_latent.shape)}"
            )

        batch_size = int(stage2_latent.shape[0])
        selected_sigmas = self.resolve_stage2_sigma_schedule(
            num_inference_steps=int(num_inference_steps),
            sigma_start_index=int(sigma_start_index),
        )
        window_entries = self.build_window_entries(
            stage2_latent,
            window_num_frames=int(window_num_frames),
            overlap_num_frames=int(overlap_num_frames),
            pad_last_window=bool(pad_last_window),
        )
        raw_noisy_latents = self.prepare_raw_noisy_video_latents(
            stage2_latent,
            noise_scale=float(selected_sigmas[0]) if noise_scale is None else float(noise_scale),
            seed=int(seed),
        )
        return {
            "window_entries": window_entries,
            "conditioning": self.prepare_prompt_conditioning(
                prompts=prompts,
                negative_prompt=negative_prompt,
                batch_size=batch_size,
                guidance_scale=float(guidance_scale),
            ),
            "raw_noisy_latents": raw_noisy_latents,
            "packed_clean_latents": self.prepare_packed_clean_video_latents(stage2_latent),
            "packed_noisy_latents": self.prepare_packed_noisy_video_latents(
                stage2_latent,
                noise_scale=float(selected_sigmas[0]) if noise_scale is None else float(noise_scale),
                seed=int(seed),
            ),
            "packed_window_latents": self.prepare_packed_window_latents(
                window_entries,
                noisy_full_latents=raw_noisy_latents,
            ),
            "scheduler_state": self.prepare_scheduler_state(
                num_inference_steps=int(num_inference_steps),
                video_sequence_length=int(stage2_latent.shape[2] * stage2_latent.shape[3] * stage2_latent.shape[4]),
                sigmas=selected_sigmas,
            ),
            "noise_scale": float(selected_sigmas[0]) if noise_scale is None else float(noise_scale),
            "sigmas": selected_sigmas,
            "sigma_start_index": int(sigma_start_index),
        }

    def _prepare_window_audio_latents(
        self,
        *,
        window_num_frames: int,
        frame_rate: float,
        noise_scale: float,
        seed: int,
    ) -> tuple[torch.Tensor, int]:
        duration_s = float(window_num_frames) * 8.0 / float(frame_rate)
        audio_latents_per_second = (
            float(self.pipe.audio_sampling_rate)
            / float(self.pipe.audio_hop_length)
            / float(self.pipe.audio_vae_temporal_compression_ratio)
        )
        audio_num_frames = round(duration_s * audio_latents_per_second)
        num_mel_bins = int(self.pipe.audio_vae.config.mel_bins)
        prepare_audio_latents = getattr(self.pipe, "prepare_audio_latents", None)
        if not callable(prepare_audio_latents):
            raise RuntimeError("LTX2Pipeline does not expose `prepare_audio_latents`; low-level streaming cannot run.")
        generator = torch.Generator(device=self.device).manual_seed(int(seed))
        audio_latents = prepare_audio_latents(
            1,
            num_channels_latents=int(self.pipe.audio_vae.config.latent_channels),
            audio_latent_length=int(audio_num_frames),
            num_mel_bins=num_mel_bins,
            noise_scale=float(noise_scale),
            dtype=torch.float32,
            device=self.device,
            generator=generator,
            latents=None,
        )
        return audio_latents.to(device=self.device, dtype=self.dtype), int(audio_num_frames)

    def _denoise_single_window_low_level(
        self,
        *,
        normalized_noisy_window: torch.Tensor,
        conditioning: dict[str, torch.Tensor | None],
        overlap_num_frames: int,
        previous_tail_step_cache: list[torch.Tensor] | None,
        previous_block_state_cache: list[dict[int, BlockTailCache]] | None,
        hidden_state_reuse_patch: LTX2BlockStateReusePatch | None,
        fps: float,
        seed: int,
        num_inference_steps: int,
        selected_sigmas: Sequence[float],
        guidance_scale: float,
        resolved_noise_scale: float,
    ) -> dict[str, Any]:
        window_latents = self.pack_video_latents(normalized_noisy_window.to(device=self.device, dtype=self.dtype))
        latent_num_frames = int(normalized_noisy_window.shape[2])
        latent_height = int(normalized_noisy_window.shape[3])
        latent_width = int(normalized_noisy_window.shape[4])
        video_sequence_length = int(latent_num_frames * latent_height * latent_width)

        scheduler_state = self.prepare_scheduler_state(
            num_inference_steps=int(num_inference_steps),
            video_sequence_length=video_sequence_length,
            sigmas=selected_sigmas,
        )
        scheduler = scheduler_state["scheduler"]
        window_timesteps = scheduler_state["timesteps"]
        window_sigmas = getattr(scheduler, "sigmas", None)

        audio_latents, audio_num_frames = self._prepare_window_audio_latents(
            window_num_frames=latent_num_frames,
            frame_rate=float(fps),
            noise_scale=resolved_noise_scale,
            seed=int(seed),
        )
        audio_scheduler = copy.deepcopy(scheduler)

        video_coords = self.pipe.transformer.rope.prepare_video_coords(
            window_latents.shape[0],
            latent_num_frames,
            latent_height,
            latent_width,
            window_latents.device,
            fps=float(fps),
        )
        audio_coords = self.pipe.transformer.audio_rope.prepare_audio_coords(
            audio_latents.shape[0],
            audio_num_frames,
            audio_latents.device,
        )
        do_cfg = float(guidance_scale) > 1.0
        if do_cfg:
            video_coords = video_coords.repeat((2,) + (1,) * (video_coords.ndim - 1))
            audio_coords = audio_coords.repeat((2,) + (1,) * (audio_coords.ndim - 1))

        next_tail_step_cache: list[torch.Tensor] = []
        next_block_state_cache: list[dict[int, BlockTailCache]] | None = None
        prompt_embeds = conditioning["prompt_embeds"]
        cache_context = getattr(self.pipe.transformer, "cache_context", None)

        patch_started = False
        if hidden_state_reuse_patch is not None:
            hidden_state_reuse_patch.begin_window(
                num_inference_steps=int(num_inference_steps),
                overlap_num_frames=int(overlap_num_frames),
                latent_num_frames=int(latent_num_frames),
                latent_height=int(latent_height),
                latent_width=int(latent_width),
                audio_num_frames=int(audio_num_frames),
                previous_window_cache=previous_block_state_cache,
            )
            patch_started = True

        try:
            for step_index, timestep_value in enumerate(window_timesteps):
                if hidden_state_reuse_patch is not None:
                    hidden_state_reuse_patch.set_step_index(int(step_index))
                latent_model_input = torch.cat([window_latents] * 2) if do_cfg else window_latents
                latent_model_input = latent_model_input.to(prompt_embeds.dtype)
                audio_latent_model_input = torch.cat([audio_latents] * 2) if do_cfg else audio_latents
                audio_latent_model_input = audio_latent_model_input.to(prompt_embeds.dtype)
                timestep_batch = timestep_value.expand(latent_model_input.shape[0])
                if window_sigmas is not None and int(step_index) < len(window_sigmas):
                    sigma_value = window_sigmas[step_index]
                    sigma_batch = sigma_value.expand(latent_model_input.shape[0]).to(
                        device=latent_model_input.device,
                        dtype=latent_model_input.dtype,
                    )
                else:
                    sigma_batch = timestep_batch.to(device=latent_model_input.device, dtype=latent_model_input.dtype)

                context_manager = cache_context("cond_uncond") if callable(cache_context) else nullcontext()
                with context_manager:
                    noise_pred_video, noise_pred_audio = self.pipe.transformer(
                        hidden_states=latent_model_input,
                        audio_hidden_states=audio_latent_model_input,
                        encoder_hidden_states=conditioning["connector_prompt_embeds"],
                        audio_encoder_hidden_states=conditioning["connector_audio_prompt_embeds"],
                        timestep=timestep_batch,
                        sigma=sigma_batch,
                        encoder_attention_mask=conditioning["connector_attention_mask"],
                        audio_encoder_attention_mask=conditioning["connector_attention_mask"],
                        num_frames=latent_num_frames,
                        height=latent_height,
                        width=latent_width,
                        fps=float(fps),
                        audio_num_frames=audio_num_frames,
                        video_coords=video_coords,
                        audio_coords=audio_coords,
                        attention_kwargs=None,
                        return_dict=False,
                    )

                noise_pred_video = noise_pred_video.float()
                noise_pred_audio = noise_pred_audio.float()
                if do_cfg:
                    noise_pred_video_uncond, noise_pred_video_text = noise_pred_video.chunk(2)
                    noise_pred_video = noise_pred_video_uncond + float(guidance_scale) * (
                        noise_pred_video_text - noise_pred_video_uncond
                    )
                    noise_pred_audio_uncond, noise_pred_audio_text = noise_pred_audio.chunk(2)
                    noise_pred_audio = noise_pred_audio_uncond + float(guidance_scale) * (
                        noise_pred_audio_text - noise_pred_audio_uncond
                    )

                window_latents = scheduler.step(noise_pred_video, timestep_value, window_latents, return_dict=False)[0]
                audio_latents = audio_scheduler.step(
                    noise_pred_audio, timestep_value, audio_latents, return_dict=False
                )[0]

                unpacked_window = self.unpack_video_latents(
                    window_latents,
                    latent_num_frames=latent_num_frames,
                    latent_height=latent_height,
                    latent_width=latent_width,
                )
                if previous_tail_step_cache is not None and int(overlap_num_frames) > 0:
                    unpacked_window[:, :, : int(overlap_num_frames)] = previous_tail_step_cache[step_index].to(
                        device=unpacked_window.device,
                        dtype=unpacked_window.dtype,
                    )
                    window_latents = self.pack_video_latents(unpacked_window)
                if int(overlap_num_frames) > 0:
                    next_tail_step_cache.append(unpacked_window[:, :, -int(overlap_num_frames) :].detach().cpu())
        finally:
            if hidden_state_reuse_patch is not None and patch_started:
                next_block_state_cache = hidden_state_reuse_patch.finish_window()

        final_window = self.unpack_video_latents(
            window_latents,
            latent_num_frames=latent_num_frames,
            latent_height=latent_height,
            latent_width=latent_width,
        )
        return {
            "final_window": final_window,
            "next_tail_step_cache": next_tail_step_cache,
            "next_block_state_cache": next_block_state_cache,
            "latent_num_frames": int(latent_num_frames),
            "latent_height": int(latent_height),
            "latent_width": int(latent_width),
        }

    @torch.inference_mode()
    def run_refine_only_streaming_low_level(
        self,
        latents: torch.Tensor,
        prompts: Sequence[str] | str,
        negative_prompt: Sequence[str] | str | None = None,
        *,
        latents_normalized: bool = True,
        window_num_frames: int = 11,
        overlap_num_frames: int = 4,
        pad_last_window: bool = True,
        fps: float = 24.0,
        seed: int = 0,
        num_inference_steps: int = 3,
        sigma_start_index: int = 0,
        guidance_scale: float = 1.0,
        noise_scale: float | None = None,
        enable_hidden_state_reuse: bool = False,
        guidance_rescale: float = 0.0,
        return_intermediates: bool = False,
    ) -> torch.Tensor | dict[str, Any]:
        del guidance_rescale
        stage2_latent = latents if not latents_normalized else self.refiner.normalized_to_raw_latent(latents)
        prepared = self.prepare_streaming_inputs(
            stage2_latent=stage2_latent,
            prompts=prompts,
            negative_prompt=negative_prompt,
            window_num_frames=int(window_num_frames),
            overlap_num_frames=int(overlap_num_frames),
            pad_last_window=bool(pad_last_window),
            num_inference_steps=int(num_inference_steps),
            sigma_start_index=int(sigma_start_index),
            guidance_scale=float(guidance_scale),
            noise_scale=noise_scale,
            seed=int(seed),
        )

        window_entries: list[StreamingWindowEntry] = prepared["window_entries"]
        conditioning: dict[str, torch.Tensor | None] = prepared["conditioning"]
        raw_noisy_latents: torch.Tensor = prepared["raw_noisy_latents"]
        resolved_noise_scale = float(prepared["noise_scale"])
        selected_sigmas: list[float] = [float(x) for x in prepared["sigmas"]]

        stitched_chunks: list[torch.Tensor] = []
        tail_step_cache: list[torch.Tensor] | None = None
        block_state_cache: list[dict[int, BlockTailCache]] | None = None
        window_metadata: list[dict[str, Any]] = []
        hidden_state_reuse_patch = (
            LTX2BlockStateReusePatch(self.pipe.transformer) if bool(enable_hidden_state_reuse) else None
        )

        patch_context = hidden_state_reuse_patch if hidden_state_reuse_patch is not None else nullcontext()
        with patch_context:
            for window_index, entry in enumerate(window_entries):
                normalized_noisy_window = raw_noisy_latents[
                    :,
                    :,
                    int(entry.global_start_frame) : int(entry.global_end_frame_exclusive),
                ]
                if int(normalized_noisy_window.shape[2]) < int(window_num_frames):
                    pad_count = int(window_num_frames) - int(normalized_noisy_window.shape[2])
                    normalized_noisy_window = torch.cat(
                        [normalized_noisy_window, normalized_noisy_window[:, :, -1:].repeat(1, 1, pad_count, 1, 1)],
                        dim=2,
                    )

                window_result = self._denoise_single_window_low_level(
                    normalized_noisy_window=normalized_noisy_window,
                    conditioning=conditioning,
                    overlap_num_frames=int(overlap_num_frames),
                    previous_tail_step_cache=tail_step_cache,
                    previous_block_state_cache=block_state_cache,
                    hidden_state_reuse_patch=hidden_state_reuse_patch,
                    fps=float(fps),
                    seed=int(seed),
                    num_inference_steps=int(num_inference_steps),
                    selected_sigmas=selected_sigmas,
                    guidance_scale=float(guidance_scale),
                    resolved_noise_scale=float(resolved_noise_scale),
                )
                final_window = window_result["final_window"]
                contributed = final_window
                if window_index > 0 and int(overlap_num_frames) > 0:
                    contributed = contributed[:, :, int(overlap_num_frames) :]
                stitched_chunks.append(contributed.detach().cpu())
                tail_step_cache = window_result["next_tail_step_cache"]
                block_state_cache = window_result["next_block_state_cache"]
                window_metadata.append(
                    {
                        "window_index": int(window_index),
                        "global_start_frame": int(entry.global_start_frame),
                        "global_end_frame": int(entry.global_end_frame_exclusive - 1),
                        "window_num_frames": int(window_result["latent_num_frames"]),
                        "overlap_num_frames": int(overlap_num_frames),
                        "num_inference_steps": int(num_inference_steps),
                        "sigma_start_index": int(prepared["sigma_start_index"]),
                        "sigmas": selected_sigmas,
                        "shared_noise_seed": int(seed),
                        "step0_noise_scale": float(resolved_noise_scale),
                        "context_step_cache_depth": len(window_result["next_tail_step_cache"]),
                        "hidden_state_reuse_enabled": bool(enable_hidden_state_reuse),
                        "block_state_cache_depth": (
                            0
                            if window_result["next_block_state_cache"] is None
                            else len(window_result["next_block_state_cache"])
                        ),
                    }
                )

        stitched_normalized = torch.cat(stitched_chunks, dim=2)[:, :, : int(stage2_latent.shape[2])]
        refined_latent = self.refiner.normalized_to_raw_latent(
            stitched_normalized.to(device=self.device, dtype=self.dtype)
        )
        if not return_intermediates:
            return refined_latent
        return {
            "stage2_latent": stage2_latent,
            "normalized_noisy_latent": raw_noisy_latents,
            "refined_latent": refined_latent,
            "window_metadata": window_metadata,
            "window_num_frames": int(window_num_frames),
            "overlap_num_frames": int(overlap_num_frames),
            "num_windows": len(window_metadata),
            "sigma_start_index": int(prepared["sigma_start_index"]),
            "sigmas": selected_sigmas,
            "shared_noise_seed": int(seed),
            "step0_noise_scale": float(resolved_noise_scale),
            "hidden_state_reuse_enabled": bool(enable_hidden_state_reuse),
        }

    @torch.inference_mode()
    def run_refine_only_center_window_low_level(
        self,
        latents: torch.Tensor,
        prompts: Sequence[str] | str,
        negative_prompt: Sequence[str] | str | None = None,
        *,
        latents_normalized: bool = True,
        window_num_frames: int = 15,
        pad_last_window: bool = True,
        fps: float = 24.0,
        seed: int = 0,
        num_inference_steps: int = 3,
        sigma_start_index: int = 0,
        guidance_scale: float = 1.0,
        noise_scale: float | None = None,
        enable_hidden_state_reuse: bool = False,
        guidance_rescale: float = 0.0,
        return_intermediates: bool = False,
    ) -> torch.Tensor | dict[str, Any]:
        del guidance_rescale
        if int(window_num_frames) < 3 or int(window_num_frames) % 3 != 0:
            raise ValueError("center-window mode requires window_num_frames to be divisible by 3 and >= 3.")

        stage2_latent = latents if not latents_normalized else self.refiner.normalized_to_raw_latent(latents)
        total_num_frames = int(stage2_latent.shape[2])
        keep_num_frames = int(window_num_frames) // 3
        overlap_num_frames = int(keep_num_frames * 2)

        left_pad = stage2_latent[:, :, :1].repeat(1, 1, keep_num_frames, 1, 1)
        right_pad = stage2_latent[:, :, -1:].repeat(1, 1, keep_num_frames, 1, 1)
        padded_stage2_latent = torch.cat([left_pad, stage2_latent, right_pad], dim=2)

        prepared = self.prepare_streaming_inputs(
            stage2_latent=padded_stage2_latent,
            prompts=prompts,
            negative_prompt=negative_prompt,
            window_num_frames=int(window_num_frames),
            overlap_num_frames=int(overlap_num_frames),
            pad_last_window=bool(pad_last_window),
            num_inference_steps=int(num_inference_steps),
            sigma_start_index=int(sigma_start_index),
            guidance_scale=float(guidance_scale),
            noise_scale=noise_scale,
            seed=int(seed),
        )

        window_entries: list[StreamingWindowEntry] = prepared["window_entries"]
        conditioning: dict[str, torch.Tensor | None] = prepared["conditioning"]
        raw_noisy_latents: torch.Tensor = prepared["raw_noisy_latents"]
        resolved_noise_scale = float(prepared["noise_scale"])
        selected_sigmas: list[float] = [float(x) for x in prepared["sigmas"]]

        kept_chunks: list[torch.Tensor] = []
        tail_step_cache: list[torch.Tensor] | None = None
        block_state_cache: list[dict[int, BlockTailCache]] | None = None
        window_metadata: list[dict[str, Any]] = []
        hidden_state_reuse_patch = (
            LTX2BlockStateReusePatch(self.pipe.transformer) if bool(enable_hidden_state_reuse) else None
        )

        patch_context = hidden_state_reuse_patch if hidden_state_reuse_patch is not None else nullcontext()
        with patch_context:
            for window_index, entry in enumerate(window_entries):
                normalized_noisy_window = raw_noisy_latents[
                    :,
                    :,
                    int(entry.global_start_frame) : int(entry.global_end_frame_exclusive),
                ]
                if int(normalized_noisy_window.shape[2]) < int(window_num_frames):
                    pad_count = int(window_num_frames) - int(normalized_noisy_window.shape[2])
                    normalized_noisy_window = torch.cat(
                        [normalized_noisy_window, normalized_noisy_window[:, :, -1:].repeat(1, 1, pad_count, 1, 1)],
                        dim=2,
                    )

                window_result = self._denoise_single_window_low_level(
                    normalized_noisy_window=normalized_noisy_window,
                    conditioning=conditioning,
                    overlap_num_frames=int(overlap_num_frames),
                    previous_tail_step_cache=tail_step_cache,
                    previous_block_state_cache=block_state_cache,
                    hidden_state_reuse_patch=hidden_state_reuse_patch,
                    fps=float(fps),
                    seed=int(seed),
                    num_inference_steps=int(num_inference_steps),
                    selected_sigmas=selected_sigmas,
                    guidance_scale=float(guidance_scale),
                    resolved_noise_scale=float(resolved_noise_scale),
                )
                final_window = window_result["final_window"]

                global_keep_start = int(entry.global_start_frame)
                global_keep_end_exclusive = min(total_num_frames, int(entry.global_start_frame) + int(keep_num_frames))
                keep_valid_num_frames = max(0, int(global_keep_end_exclusive - global_keep_start))
                if keep_valid_num_frames > 0:
                    kept = final_window[:, :, int(keep_num_frames) : int(keep_num_frames) + keep_valid_num_frames]
                    kept_chunks.append(kept.detach().cpu())

                tail_step_cache = window_result["next_tail_step_cache"]
                block_state_cache = window_result["next_block_state_cache"]
                window_metadata.append(
                    {
                        "window_index": int(window_index),
                        "global_window_start_frame": int(max(0, int(entry.global_start_frame) - int(keep_num_frames))),
                        "global_window_end_frame": int(
                            min(total_num_frames - 1, int(entry.global_start_frame) + int(2 * keep_num_frames) - 1)
                        ),
                        "global_keep_start_frame": int(global_keep_start),
                        "global_keep_end_frame": (
                            int(global_keep_end_exclusive - 1) if keep_valid_num_frames > 0 else None
                        ),
                        "window_num_frames": int(window_result["latent_num_frames"]),
                        "left_context_num_frames": int(keep_num_frames),
                        "middle_keep_num_frames": int(keep_num_frames),
                        "right_context_num_frames": int(keep_num_frames),
                        "overlap_num_frames": int(overlap_num_frames),
                        "keep_valid_num_frames": int(keep_valid_num_frames),
                        "num_inference_steps": int(num_inference_steps),
                        "sigma_start_index": int(prepared["sigma_start_index"]),
                        "sigmas": selected_sigmas,
                        "shared_noise_seed": int(seed),
                        "step0_noise_scale": float(resolved_noise_scale),
                        "context_step_cache_depth": len(window_result["next_tail_step_cache"]),
                        "hidden_state_reuse_enabled": bool(enable_hidden_state_reuse),
                        "block_state_cache_depth": (
                            0
                            if window_result["next_block_state_cache"] is None
                            else len(window_result["next_block_state_cache"])
                        ),
                    }
                )

        stitched_normalized = torch.cat(kept_chunks, dim=2)[:, :, :total_num_frames]
        refined_latent = self.refiner.normalized_to_raw_latent(
            stitched_normalized.to(device=self.device, dtype=self.dtype)
        )
        if not return_intermediates:
            return refined_latent
        return {
            "stage2_latent": stage2_latent,
            "normalized_noisy_latent": raw_noisy_latents,
            "refined_latent": refined_latent,
            "window_metadata": window_metadata,
            "window_num_frames": int(window_num_frames),
            "left_context_num_frames": int(keep_num_frames),
            "middle_keep_num_frames": int(keep_num_frames),
            "right_context_num_frames": int(keep_num_frames),
            "num_windows": len(window_metadata),
            "sigma_start_index": int(prepared["sigma_start_index"]),
            "sigmas": selected_sigmas,
            "shared_noise_seed": int(seed),
            "step0_noise_scale": float(resolved_noise_scale),
            "hidden_state_reuse_enabled": bool(enable_hidden_state_reuse),
        }
