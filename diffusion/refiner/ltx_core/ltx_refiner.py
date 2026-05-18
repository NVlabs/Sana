from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import torch

try:
    from diffusers import FlowMatchEulerDiscreteScheduler
    from diffusers.pipelines.ltx2 import LTX2LatentUpsamplePipeline, LTX2Pipeline
    from diffusers.pipelines.ltx2.latent_upsampler import LTX2LatentUpsamplerModel
    from diffusers.pipelines.ltx2.utils import STAGE_2_DISTILLED_SIGMA_VALUES
except Exception as exc:  # pragma: no cover - import error surfaced at runtime
    raise ImportError(
        "diffusers LTX-2 components are required. "
        "Your environment is likely using an older diffusers build. "
        "Install the GitHub main version, for example: "
        "`pip install -U git+https://github.com/huggingface/diffusers.git`"
    ) from exc

DEFAULT_DISTILLED_LORA_WEIGHT = "ltx-2-19b-distilled-lora-384.safetensors"


def _cleanup_memory() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def _resolve_lora_source(
    path_or_repo: str | None,
    fallback_repo: str,
    default_weight_name: str,
) -> tuple[str, str | None]:
    if path_or_repo:
        if os.path.isfile(path_or_repo):
            return os.path.dirname(path_or_repo), os.path.basename(path_or_repo)
        if os.path.isdir(path_or_repo):
            return path_or_repo, default_weight_name
        return path_or_repo, default_weight_name
    return fallback_repo, default_weight_name


def _upsampler_mode_flags(mode: str) -> tuple[bool, bool]:
    normalized = str(mode).strip().lower().replace("-", "_")
    if normalized == "spatial":
        return True, False
    if normalized == "temporal":
        return False, True
    if normalized in {"spatiotemporal", "spatial_temporal", "both"}:
        return True, True
    raise ValueError(f"Unsupported upsampler mode: {mode}")


def _normalize_upsampler_mode(mode: str) -> str:
    spatial_upsample, temporal_upsample = _upsampler_mode_flags(mode)
    if spatial_upsample and temporal_upsample:
        return "spatiotemporal"
    if spatial_upsample:
        return "spatial"
    return "temporal"


def _first_stage2_sigma(stage2_sigmas: Sequence[float]) -> float:
    if not stage2_sigmas:
        raise ValueError("Stage-2 sigma list is empty.")
    return float(stage2_sigmas[0])


def _build_latent_temporal_intervals(
    total_frames: int,
    chunk_size: int,
    overlap: int,
) -> list[tuple[int, int, int, int]]:
    """Split *total_frames* latent frames into overlapping chunks.

    Returns ``(start, end, left_ramp, right_ramp)`` tuples.
    """
    if total_frames <= chunk_size:
        return [(0, total_frames, 0, 0)]
    if overlap >= chunk_size:
        raise ValueError(f"overlap must be < chunk_size, got {overlap} >= {chunk_size}")

    stride = chunk_size - overlap
    amount = (total_frames + chunk_size - 2 * overlap - 1) // stride
    starts = [i * stride for i in range(amount)]
    ends = [s + chunk_size for s in starts]
    ends[-1] = total_frames
    left_ramps = [0] + [overlap] * (amount - 1)
    right_ramps = [overlap] * (amount - 1) + [0]
    return list(zip(starts, ends, left_ramps, right_ramps))


def _compute_trapezoidal_mask_1d(
    length: int,
    ramp_left: int,
    ramp_right: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """1-D trapezoidal blending mask: linear fade-in / plateau / fade-out."""
    if length <= 0:
        raise ValueError("Mask length must be positive.")
    ramp_left = max(0, min(ramp_left, length))
    ramp_right = max(0, min(ramp_right, length))
    mask = torch.ones(length, device=device, dtype=dtype)
    if ramp_left > 0:
        fade_in = torch.linspace(0.0, 1.0, ramp_left + 2, device=device, dtype=dtype)[1:-1]
        mask[:ramp_left] *= fade_in
    if ramp_right > 0:
        fade_out = torch.linspace(1.0, 0.0, ramp_right + 2, device=device, dtype=dtype)[1:-1]
        mask[-ramp_right:] *= fade_out
    return mask.clamp_(0, 1)


@dataclass
class LTXRefinerConfig:
    checkpoint_path: str
    distilled_lora_path: str
    distilled_lora_strength: float = 1.0
    enable_cpu_offload: bool = True
    dtype: torch.dtype = torch.bfloat16
    vae_dtype: torch.dtype | None = None
    upsampler_path: str | None = None
    spatial_upsampler_path: str | None = None
    spatial_upsampler_path_stage2: str | None = None
    temporal_upsampler_path: str | None = None
    distilled_lora_weight: str | None = None
    device: str | None = None
    logger: logging.Logger | None = None


class LTXRefiner:
    def __init__(self, config: LTXRefinerConfig) -> None:
        self._config = config
        self.logger = config.logger or logging.getLogger("Sana")
        self.device = torch.device(config.device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.dtype = config.dtype
        self.vae_dtype = config.vae_dtype or config.dtype
        self._sigmas = torch.tensor(STAGE_2_DISTILLED_SIGMA_VALUES, dtype=torch.float32)
        self._upsample_pipes: dict[tuple[str, str], LTX2LatentUpsamplePipeline] = {}

        self._pipe = LTX2Pipeline.from_pretrained(config.checkpoint_path, torch_dtype=self.dtype)
        self._pipe.scheduler = FlowMatchEulerDiscreteScheduler.from_config(
            self._pipe.scheduler.config,
            use_dynamic_shifting=False,
            shift_terminal=None,
        )

        if hasattr(self._pipe, "vae") and isinstance(self._pipe.vae, torch.nn.Module):
            self._pipe.vae.to(dtype=self.vae_dtype)

        self._enable_main_pipe_offload()

        lora_source, weight_name = _resolve_lora_source(
            config.distilled_lora_path,
            config.checkpoint_path,
            config.distilled_lora_weight or DEFAULT_DISTILLED_LORA_WEIGHT,
        )
        if lora_source:
            self._pipe.load_lora_weights(
                lora_source,
                adapter_name="stage_2_distilled",
                weight_name=weight_name,
            )
            self._pipe.set_adapters("stage_2_distilled", float(config.distilled_lora_strength))
            self.logger.info(
                "[LTXRefiner] Loaded distilled LoRA from %s (weight=%s, strength=%.3f)",
                lora_source,
                weight_name,
                float(config.distilled_lora_strength),
            )

    @property
    def stage2_sigmas(self) -> list[float]:
        return [float(x) for x in self._sigmas.detach().cpu().tolist()]

    def _enable_main_pipe_offload(self) -> None:
        if (
            self.device.type == "cuda"
            and self._config.enable_cpu_offload
            and hasattr(self._pipe, "enable_sequential_cpu_offload")
        ):
            self.logger.info("[LTXRefiner] enable_sequential_cpu_offload=True")
            self._pipe.enable_sequential_cpu_offload(device=self.device)
            return
        if self.device.type == "cuda":
            self.logger.info("[LTXRefiner] enable_sequential_cpu_offload=False (keeping weights on GPU)")
        self._pipe.to(self.device)

    def _release_after_direct_vae_call(self) -> None:
        if self.device.type != "cuda":
            return
        if hasattr(self._pipe, "maybe_free_model_hooks"):
            self._pipe.maybe_free_model_hooks()
        elif isinstance(getattr(self._pipe, "vae", None), torch.nn.Module):
            self._pipe.vae.to("cpu")
        _cleanup_memory()

    def _resolve_default_upsampler_path(self, requested_path: str | None, upsampler_mode: str) -> str:
        if requested_path:
            return requested_path
        if upsampler_mode == "spatial" and self._config.spatial_upsampler_path:
            return self._config.spatial_upsampler_path
        if upsampler_mode == "temporal" and self._config.temporal_upsampler_path:
            return self._config.temporal_upsampler_path
        if self._config.upsampler_path:
            return self._config.upsampler_path

        base_repo = getattr(self._pipe, "repo_id", None) or getattr(self._pipe.config, "_name_or_path", None)
        if base_repo is None:
            raise ValueError("Unable to resolve LTX-2 repo id or path for latent upsampler.")
        if os.path.isfile(base_repo):
            self.logger.warning(
                "[LTXRefiner] checkpoint_path is a file. Using its directory for latent upsampler: %s",
                os.path.dirname(base_repo),
            )
            base_repo = os.path.dirname(base_repo)
        return str(base_repo)

    def _resolve_second_spatial_upsampler_path(self, requested_path: str | None) -> str:
        if requested_path:
            return requested_path
        if self._config.spatial_upsampler_path_stage2:
            return self._config.spatial_upsampler_path_stage2
        return self._resolve_default_upsampler_path(None, "spatial")

    def _get_upsample_pipe(
        self, upsampler_path: str | None = None, upsampler_mode: str = "spatial"
    ) -> LTX2LatentUpsamplePipeline:
        normalized_mode = _normalize_upsampler_mode(upsampler_mode)
        resolved_path = self._resolve_default_upsampler_path(upsampler_path, normalized_mode)
        cache_key = (str(resolved_path), normalized_mode)
        if cache_key not in self._upsample_pipes:
            spatial_upsample, temporal_upsample = _upsampler_mode_flags(normalized_mode)
            load_errors: list[str] = []
            latent_upsampler = None
            for subfolder in ("latent_upsampler", None):
                try:
                    kwargs: dict[str, Any] = {
                        "pretrained_model_name_or_path": resolved_path,
                        "spatial_upsample": spatial_upsample,
                        "temporal_upsample": temporal_upsample,
                        "torch_dtype": self.dtype,
                    }
                    if subfolder is not None:
                        kwargs["subfolder"] = subfolder
                    latent_upsampler = LTX2LatentUpsamplerModel.from_pretrained(**kwargs)
                    break
                except Exception as exc:
                    tried = f"{resolved_path}/{subfolder}" if subfolder is not None else resolved_path
                    load_errors.append(f"{tried}: {type(exc).__name__}: {exc}")

            if latent_upsampler is None:
                raise RuntimeError(
                    "Failed to load the latent upsampler for "
                    f"mode={normalized_mode!r} from {resolved_path!r}. "
                    "Tried both the `latent_upsampler` subfolder and the repo root.\n" + "\n".join(load_errors)
                )

            upsample_pipe = LTX2LatentUpsamplePipeline(vae=self._pipe.vae, latent_upsampler=latent_upsampler)
            if self.device.type == "cuda" and self._config.enable_cpu_offload:
                if hasattr(upsample_pipe, "enable_sequential_cpu_offload"):
                    upsample_pipe.enable_sequential_cpu_offload(device=self.device)
                elif hasattr(upsample_pipe, "enable_model_cpu_offload"):
                    upsample_pipe.enable_model_cpu_offload(device=self.device)
                else:
                    upsample_pipe.to(self.device)
            else:
                upsample_pipe.to(self.device)
            self._upsample_pipes[cache_key] = upsample_pipe
        return self._upsample_pipes[cache_key]

    def _normalize_prompt_batch(
        self,
        prompts: Sequence[str] | str,
        batch_size: int,
        *,
        field_name: str,
    ) -> list[str]:
        if isinstance(prompts, str):
            values = [prompts]
        else:
            values = [str(x) for x in prompts]
        if len(values) == 1 and batch_size > 1:
            values = values * batch_size
        if len(values) != batch_size:
            raise ValueError(f"{field_name} batch size mismatch: got {len(values)} values for batch_size={batch_size}.")
        return values

    def _resolve_negative_prompt_batch(
        self,
        negative_prompt: Sequence[str] | str | None,
        batch_size: int,
    ) -> list[str]:
        if negative_prompt is None:
            return [""] * batch_size
        return self._normalize_prompt_batch(negative_prompt, batch_size, field_name="negative_prompt")

    def normalized_to_raw_latent(self, latent_norm: torch.Tensor) -> torch.Tensor:
        vae = self._pipe.vae
        latents_mean = vae.latents_mean.view(1, -1, 1, 1, 1).to(device=latent_norm.device, dtype=latent_norm.dtype)
        latents_std = vae.latents_std.view(1, -1, 1, 1, 1).to(device=latent_norm.device, dtype=latent_norm.dtype)
        scaling = float(vae.config.scaling_factor)
        return latent_norm * latents_std / scaling + latents_mean

    def raw_to_normalized_latent(self, latents_raw: torch.Tensor) -> torch.Tensor:
        vae = self._pipe.vae
        latents_mean = vae.latents_mean.view(1, -1, 1, 1, 1).to(device=latents_raw.device, dtype=latents_raw.dtype)
        latents_std = vae.latents_std.view(1, -1, 1, 1, 1).to(device=latents_raw.device, dtype=latents_raw.dtype)
        scaling = float(vae.config.scaling_factor)
        return (latents_raw - latents_mean) * scaling / latents_std

    def _build_stage2_call_kwargs(
        self,
        *,
        stage2_latent: torch.Tensor,
        prompts: Sequence[str] | str,
        negative_prompt: Sequence[str] | str | None,
        seed: int | None,
        num_inference_steps: int,
        guidance_scale: float,
        noise_scale: float | None,
        stg_scale: float,
        modality_scale: float,
        guidance_rescale: float,
        use_cross_timestep: bool,
        output_type: str,
    ) -> dict[str, Any]:
        batch_size = int(stage2_latent.shape[0])
        prompt_batch = self._normalize_prompt_batch(prompts, batch_size, field_name="prompts")
        negative_prompt_batch = self._resolve_negative_prompt_batch(negative_prompt, batch_size)
        sigmas = self.stage2_sigmas
        if noise_scale is None:
            noise_scale = _first_stage2_sigma(sigmas)

        generator = torch.Generator(device=self.device).manual_seed(int(seed)) if seed is not None else None
        return {
            "latents": stage2_latent.to(device=self.device, dtype=self.dtype),
            "audio_latents": None,
            "prompt": prompt_batch,
            "negative_prompt": negative_prompt_batch,
            "num_inference_steps": int(num_inference_steps),
            "noise_scale": float(noise_scale),
            "sigmas": sigmas,
            "guidance_scale": float(guidance_scale),
            "stg_scale": float(stg_scale),
            "modality_scale": float(modality_scale),
            "guidance_rescale": float(guidance_rescale),
            "use_cross_timestep": bool(use_cross_timestep),
            "output_type": output_type,
            "return_dict": False,
            "generator": generator,
        }

    def _build_pipeline_metadata(
        self,
        *,
        output_latent: torch.Tensor,
        fps_scale: float,
        input_num_frames: int | None,
        input_height: int | None,
        input_width: int | None,
        input_fps: float | None,
        spatial_scale: float,
        temporal_scale_num: int,
        temporal_scale_den: int,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "fps_scale": float(fps_scale),
            "output_latent_shape": list(output_latent.shape),
        }
        if input_num_frames is not None:
            payload["frames"] = int((int(input_num_frames) * temporal_scale_num) / temporal_scale_den)
        if input_height is not None:
            payload["height"] = int(round(float(input_height) * spatial_scale))
        if input_width is not None:
            payload["width"] = int(round(float(input_width) * spatial_scale))
        if input_fps is not None:
            payload["fps"] = float(input_fps) * float(fps_scale)
        return payload

    @torch.inference_mode()
    def encode_video(self, video: torch.Tensor) -> torch.Tensor:
        vae = self._pipe.vae
        posterior = vae.encode(video.to(device=self.device, dtype=self.vae_dtype)).latent_dist
        z = posterior.mode()
        if not torch.isfinite(z).all():
            raise RuntimeError("VAE encode produced non-finite latents before normalization.")
        z = self.raw_to_normalized_latent(z)
        if not torch.isfinite(z).all():
            raise RuntimeError("VAE latent normalization produced non-finite values.")
        self._release_after_direct_vae_call()
        return z.to(self.dtype)

    @torch.inference_mode()
    def upsample_latent(
        self,
        latent: torch.Tensor,
        upsampler_path: str | None = None,
        upsampler_mode: str = "spatial",
        latents_normalized: bool = True,
    ) -> torch.Tensor:
        upsample_pipe = self._get_upsample_pipe(upsampler_path=upsampler_path, upsampler_mode=upsampler_mode)
        upscaled = upsample_pipe(
            latents=latent.to(device=self.device, dtype=self.dtype),
            latents_normalized=bool(latents_normalized),
            output_type="latent",
            return_dict=False,
        )[0]
        if hasattr(upsample_pipe, "maybe_free_model_hooks"):
            upsample_pipe.maybe_free_model_hooks()
        _cleanup_memory()
        return upscaled

    @torch.inference_mode()
    def refine_latent(
        self,
        upsampled_latent: torch.Tensor,
        prompts: Sequence[str] | str,
        negative_prompt: Sequence[str] | str | None = None,
        fps: float = 24.0,
        seed: int = 0,
        num_inference_steps: int = 3,
        guidance_scale: float = 1.0,
        noise_scale: float | None = None,
        stg_scale: float = 0.0,
        modality_scale: float = 1.0,
        guidance_rescale: float = 0.0,
        use_cross_timestep: bool = True,
    ) -> torch.Tensor:
        del fps
        self.logger.info("[LTXRefiner] refine_latent start. latent=%s", tuple(upsampled_latent.shape))
        call_kwargs = self._build_stage2_call_kwargs(
            stage2_latent=upsampled_latent,
            prompts=prompts,
            negative_prompt=negative_prompt,
            seed=seed,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            noise_scale=noise_scale,
            stg_scale=stg_scale,
            modality_scale=modality_scale,
            guidance_rescale=guidance_rescale,
            use_cross_timestep=use_cross_timestep,
            output_type="latent",
        )
        latents, _ = self._pipe(**call_kwargs)
        self.logger.info("[LTXRefiner] refine_latent done. out=%s", tuple(latents.shape))
        return latents

    @torch.inference_mode()
    def refine(
        self,
        stage2_latent: torch.Tensor,
        prompts: Sequence[str] | str,
        negative_prompt: Sequence[str] | str | None = None,
        *,
        fps: float = 24.0,
        seed: int = 0,
        num_inference_steps: int = 3,
        guidance_scale: float = 1.0,
        noise_scale: float | None = None,
        stg_scale: float = 0.0,
        modality_scale: float = 1.0,
        guidance_rescale: float = 0.0,
        use_cross_timestep: bool = True,
        save_audio: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        del fps
        call_kwargs = self._build_stage2_call_kwargs(
            stage2_latent=stage2_latent,
            prompts=prompts,
            negative_prompt=negative_prompt,
            seed=seed,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            noise_scale=noise_scale,
            stg_scale=stg_scale,
            modality_scale=modality_scale,
            guidance_rescale=guidance_rescale,
            use_cross_timestep=use_cross_timestep,
            output_type="np" if save_audio else "latent",
        )
        video_out, audio_out = self._pipe(**call_kwargs)

        if save_audio:
            decoded = torch.from_numpy((video_out[0] * 255.0).round().astype("uint8"))
            audio = audio_out[0].detach().cpu() if audio_out is not None else None
            return decoded, audio

        decoded = self.decode_latent(video_out, generator=call_kwargs["generator"])[0].to(torch.uint8)
        self._release_after_direct_vae_call()
        return decoded, None

    @torch.inference_mode()
    def decode_latent(
        self,
        latents: torch.Tensor,
        *,
        decode_timestep: float = 0.0,
        decode_noise_scale: float | None = None,
        output_type: str = "np",
        generator: torch.Generator | None = None,
    ) -> list[torch.Tensor]:
        latents = latents.to(device=self.device, dtype=self.dtype)

        if not self._pipe.vae.config.timestep_conditioning:
            timestep = None
        else:
            noise = torch.randn_like(latents, generator=generator)
            if decode_noise_scale is None:
                decode_noise_scale = decode_timestep
            timestep = torch.tensor([float(decode_timestep)], device=latents.device, dtype=latents.dtype)
            decode_noise_scale = torch.tensor([float(decode_noise_scale)], device=latents.device, dtype=latents.dtype)[
                :, None, None, None, None
            ]
            latents = (1 - decode_noise_scale) * latents + decode_noise_scale * noise

        latents = latents.to(self._pipe.vae.dtype)
        video = self._pipe.vae.decode(latents, timestep, return_dict=False)[0]
        video = self._pipe.video_processor.postprocess_video(video, output_type=output_type)

        if isinstance(video, list):
            video = np.stack(video, axis=0)
        video = (video * 255.0).round().astype("uint8")
        video = torch.from_numpy(video)
        return [video[i] for i in range(video.shape[0])]

    @torch.inference_mode()
    def decode_latent_preview(self, latents: torch.Tensor) -> torch.Tensor:
        decoded = self.decode_latent(latents.to(device=self.device, dtype=self.vae_dtype))[0].to(torch.uint8)
        self._release_after_direct_vae_call()
        return decoded.cpu()

    @torch.inference_mode()
    def run_refine_only(
        self,
        latents: torch.Tensor,
        prompts: Sequence[str] | str,
        negative_prompt: Sequence[str] | str | None = None,
        *,
        latents_normalized: bool = True,
        fps: float = 24.0,
        seed: int = 0,
        num_inference_steps: int = 3,
        guidance_scale: float = 1.0,
        noise_scale: float | None = None,
        stg_scale: float = 0.0,
        modality_scale: float = 1.0,
        guidance_rescale: float = 0.0,
        use_cross_timestep: bool = True,
        return_intermediates: bool = False,
        input_num_frames: int | None = None,
        input_height: int | None = None,
        input_width: int | None = None,
        input_fps: float | None = None,
    ) -> torch.Tensor | dict[str, Any]:
        stage2_latent = latents if not latents_normalized else self.normalized_to_raw_latent(latents)
        refined_latent = self.refine_latent(
            upsampled_latent=stage2_latent,
            prompts=prompts,
            negative_prompt=negative_prompt,
            fps=fps,
            seed=seed,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            noise_scale=noise_scale,
            stg_scale=stg_scale,
            modality_scale=modality_scale,
            guidance_rescale=guidance_rescale,
            use_cross_timestep=use_cross_timestep,
        )
        if not return_intermediates:
            return refined_latent
        payload = {
            "stage2_latent": stage2_latent,
            "refined_latent": refined_latent,
        }
        payload.update(
            self._build_pipeline_metadata(
                output_latent=refined_latent,
                fps_scale=1.0,
                input_num_frames=input_num_frames,
                input_height=input_height,
                input_width=input_width,
                input_fps=input_fps,
                spatial_scale=1.0,
                temporal_scale_num=1,
                temporal_scale_den=1,
            )
        )
        return payload

    @torch.inference_mode()
    def run_refine_only_chunked(
        self,
        latents: torch.Tensor,
        prompts: Sequence[str] | str,
        negative_prompt: Sequence[str] | str | None = None,
        *,
        latents_normalized: bool = True,
        chunk_seconds: float = 50.0,
        overlap_seconds: float = 4.0,
        fps: float = 16.0,
        seed: int = 0,
        num_inference_steps: int = 3,
        guidance_scale: float = 1.0,
        noise_scale: float | None = None,
        stg_scale: float = 0.0,
        modality_scale: float = 1.0,
        guidance_rescale: float = 0.0,
        use_cross_timestep: bool = True,
        return_intermediates: bool = False,
        input_num_frames: int | None = None,
        input_height: int | None = None,
        input_width: int | None = None,
        input_fps: float | None = None,
    ) -> torch.Tensor | dict[str, Any]:
        """Refine-only with temporal chunking and trapezoidal overlap blending."""
        total_latent_frames = int(latents.shape[2])
        temporal_compress = 8
        chunk_latent = max(1, round(chunk_seconds * fps / temporal_compress))
        overlap_latent = max(0, round(overlap_seconds * fps / temporal_compress))
        overlap_latent = min(overlap_latent, chunk_latent // 2)

        if total_latent_frames <= chunk_latent:
            self.logger.info(
                "[LTXRefiner] chunked refine: video fits in one chunk (%d <= %d), using full refine.",
                total_latent_frames,
                chunk_latent,
            )
            return self.run_refine_only(
                latents,
                prompts,
                negative_prompt,
                latents_normalized=latents_normalized,
                fps=fps,
                seed=seed,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                noise_scale=noise_scale,
                stg_scale=stg_scale,
                modality_scale=modality_scale,
                guidance_rescale=guidance_rescale,
                use_cross_timestep=use_cross_timestep,
                return_intermediates=return_intermediates,
                input_num_frames=input_num_frames,
                input_height=input_height,
                input_width=input_width,
                input_fps=input_fps,
            )

        stage2_latent = latents if not latents_normalized else self.normalized_to_raw_latent(latents)
        intervals = _build_latent_temporal_intervals(total_latent_frames, chunk_latent, overlap_latent)
        self.logger.info(
            "[LTXRefiner] chunked refine: %d latent frames -> %d chunks "
            "(chunk=%d, overlap=%d, chunk_sec=%.1f, overlap_sec=%.1f)",
            total_latent_frames,
            len(intervals),
            chunk_latent,
            overlap_latent,
            chunk_seconds,
            overlap_seconds,
        )

        buffer = torch.zeros_like(stage2_latent)
        weights = torch.zeros(
            (1, 1, total_latent_frames, 1, 1),
            device=stage2_latent.device,
            dtype=stage2_latent.dtype,
        )

        for idx, (start, end, left_ramp, right_ramp) in enumerate(intervals):
            chunk = stage2_latent[:, :, start:end, :, :]
            self.logger.info(
                "[LTXRefiner] chunk %d/%d: latent frames [%d, %d), left_ramp=%d, right_ramp=%d",
                idx + 1,
                len(intervals),
                start,
                end,
                left_ramp,
                right_ramp,
            )
            refined_chunk = self.refine_latent(
                upsampled_latent=chunk,
                prompts=prompts,
                negative_prompt=negative_prompt,
                fps=fps,
                seed=seed,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                noise_scale=noise_scale,
                stg_scale=stg_scale,
                modality_scale=modality_scale,
                guidance_rescale=guidance_rescale,
                use_cross_timestep=use_cross_timestep,
            )
            chunk_len = end - start
            mask = _compute_trapezoidal_mask_1d(
                chunk_len,
                left_ramp,
                right_ramp,
                device=refined_chunk.device,
                dtype=refined_chunk.dtype,
            ).view(1, 1, chunk_len, 1, 1)

            buffer[:, :, start:end, :, :] += refined_chunk * mask
            weights[:, :, start:end, :, :] += mask

        refined_latent = buffer / weights.clamp(min=1e-8)

        if not return_intermediates:
            return refined_latent
        payload = {
            "stage2_latent": stage2_latent,
            "refined_latent": refined_latent,
            "num_chunks": len(intervals),
            "chunk_latent_frames": chunk_latent,
            "overlap_latent_frames": overlap_latent,
        }
        payload.update(
            self._build_pipeline_metadata(
                output_latent=refined_latent,
                fps_scale=1.0,
                input_num_frames=input_num_frames,
                input_height=input_height,
                input_width=input_width,
                input_fps=input_fps,
                spatial_scale=1.0,
                temporal_scale_num=1,
                temporal_scale_den=1,
            )
        )
        return payload

    @torch.inference_mode()
    def run_single_upsample_refine(
        self,
        latents: torch.Tensor,
        prompts: Sequence[str] | str,
        negative_prompt: Sequence[str] | str | None = None,
        *,
        upsampler_path: str | None = None,
        upsampler_mode: str = "spatial",
        latents_normalized: bool = True,
        fps: float = 24.0,
        seed: int = 0,
        num_inference_steps: int = 3,
        guidance_scale: float = 1.0,
        noise_scale: float | None = None,
        stg_scale: float = 0.0,
        modality_scale: float = 1.0,
        guidance_rescale: float = 0.0,
        use_cross_timestep: bool = True,
        return_intermediates: bool = False,
        input_num_frames: int | None = None,
        input_height: int | None = None,
        input_width: int | None = None,
        input_fps: float | None = None,
    ) -> torch.Tensor | dict[str, Any]:
        normalized_mode = _normalize_upsampler_mode(upsampler_mode)
        stage2_latent = self.upsample_latent(
            latents,
            upsampler_path=upsampler_path,
            upsampler_mode=normalized_mode,
            latents_normalized=latents_normalized,
        )
        refined_latent = self.refine_latent(
            upsampled_latent=stage2_latent,
            prompts=prompts,
            negative_prompt=negative_prompt,
            fps=fps,
            seed=seed,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            noise_scale=noise_scale,
            stg_scale=stg_scale,
            modality_scale=modality_scale,
            guidance_rescale=guidance_rescale,
            use_cross_timestep=use_cross_timestep,
        )
        if not return_intermediates:
            return refined_latent

        spatial_upsample, temporal_upsample = _upsampler_mode_flags(normalized_mode)
        payload = {
            "upsampled_latent": stage2_latent,
            "refined_latent": refined_latent,
        }
        payload.update(
            self._build_pipeline_metadata(
                output_latent=refined_latent,
                fps_scale=2.0 if temporal_upsample else 1.0,
                input_num_frames=input_num_frames,
                input_height=input_height,
                input_width=input_width,
                input_fps=input_fps,
                spatial_scale=2.0 if spatial_upsample else 1.0,
                temporal_scale_num=2 if temporal_upsample else 1,
                temporal_scale_den=1 if temporal_upsample else 1,
            )
        )
        if temporal_upsample and input_num_frames is not None:
            payload["frames"] = int(input_num_frames * 2 - 1)
        return payload

    @torch.inference_mode()
    def run_temporal_then_spatial(
        self,
        latents: torch.Tensor,
        prompts: Sequence[str] | str,
        negative_prompt: Sequence[str] | str | None = None,
        *,
        temporal_upsampler_path: str | None = None,
        spatial_upsampler_path: str | None = None,
        latents_normalized: bool = True,
        fps: float = 24.0,
        seed: int = 0,
        num_inference_steps: int = 3,
        guidance_scale: float = 1.0,
        noise_scale: float | None = None,
        stg_scale: float = 0.0,
        modality_scale: float = 1.0,
        guidance_rescale: float = 0.0,
        use_cross_timestep: bool = True,
        return_intermediates: bool = False,
        input_num_frames: int | None = None,
        input_height: int | None = None,
        input_width: int | None = None,
        input_fps: float | None = None,
    ) -> torch.Tensor | dict[str, Any]:
        temporal_latent = self.upsample_latent(
            latents,
            upsampler_path=temporal_upsampler_path,
            upsampler_mode="temporal",
            latents_normalized=latents_normalized,
        )
        stage2_latent = self.upsample_latent(
            temporal_latent,
            upsampler_path=spatial_upsampler_path,
            upsampler_mode="spatial",
            latents_normalized=False,
        )
        refined_latent = self.refine_latent(
            upsampled_latent=stage2_latent,
            prompts=prompts,
            negative_prompt=negative_prompt,
            fps=fps,
            seed=seed,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            noise_scale=noise_scale,
            stg_scale=stg_scale,
            modality_scale=modality_scale,
            guidance_rescale=guidance_rescale,
            use_cross_timestep=use_cross_timestep,
        )
        if not return_intermediates:
            return refined_latent
        payload = {
            "temporal_latent": temporal_latent,
            "upsampled_latent": stage2_latent,
            "refined_latent": refined_latent,
        }
        payload.update(
            self._build_pipeline_metadata(
                output_latent=refined_latent,
                fps_scale=2.0,
                input_num_frames=input_num_frames,
                input_height=input_height,
                input_width=input_width,
                input_fps=input_fps,
                spatial_scale=2.0,
                temporal_scale_num=2,
                temporal_scale_den=1,
            )
        )
        if input_num_frames is not None:
            payload["frames"] = int(input_num_frames * 2 - 1)
        return payload

    @torch.inference_mode()
    def run_spatial_then_spatial(
        self,
        latents: torch.Tensor,
        prompts: Sequence[str] | str,
        negative_prompt: Sequence[str] | str | None = None,
        *,
        spatial_upsampler_path: str | None = None,
        spatial_upsampler_path_stage2: str | None = None,
        latents_normalized: bool = True,
        fps: float = 24.0,
        seed: int = 0,
        num_inference_steps: int = 3,
        guidance_scale: float = 1.0,
        noise_scale: float | None = None,
        stg_scale: float = 0.0,
        modality_scale: float = 1.0,
        guidance_rescale: float = 0.0,
        use_cross_timestep: bool = True,
        return_intermediates: bool = False,
        input_num_frames: int | None = None,
        input_height: int | None = None,
        input_width: int | None = None,
        input_fps: float | None = None,
    ) -> torch.Tensor | dict[str, Any]:
        spatial_latent = self.upsample_latent(
            latents,
            upsampler_path=spatial_upsampler_path,
            upsampler_mode="spatial",
            latents_normalized=latents_normalized,
        )
        stage2_latent = self.upsample_latent(
            spatial_latent,
            upsampler_path=self._resolve_second_spatial_upsampler_path(spatial_upsampler_path_stage2),
            upsampler_mode="spatial",
            latents_normalized=False,
        )
        refined_latent = self.refine_latent(
            upsampled_latent=stage2_latent,
            prompts=prompts,
            negative_prompt=negative_prompt,
            fps=fps,
            seed=seed,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            noise_scale=noise_scale,
            stg_scale=stg_scale,
            modality_scale=modality_scale,
            guidance_rescale=guidance_rescale,
            use_cross_timestep=use_cross_timestep,
        )
        if not return_intermediates:
            return refined_latent
        payload = {
            "spatial_latent": spatial_latent,
            "upsampled_latent": stage2_latent,
            "refined_latent": refined_latent,
        }
        payload.update(
            self._build_pipeline_metadata(
                output_latent=refined_latent,
                fps_scale=1.0,
                input_num_frames=input_num_frames,
                input_height=input_height,
                input_width=input_width,
                input_fps=input_fps,
                spatial_scale=4.0,
                temporal_scale_num=1,
                temporal_scale_den=1,
            )
        )
        return payload

    @torch.inference_mode()
    def run_spatial_then_temporal(
        self,
        latents: torch.Tensor,
        prompts: Sequence[str] | str,
        negative_prompt: Sequence[str] | str | None = None,
        *,
        spatial_upsampler_path: str | None = None,
        temporal_upsampler_path: str | None = None,
        latents_normalized: bool = True,
        fps: float = 24.0,
        seed: int = 0,
        num_inference_steps: int = 3,
        guidance_scale: float = 1.0,
        noise_scale: float | None = None,
        stg_scale: float = 0.0,
        modality_scale: float = 1.0,
        guidance_rescale: float = 0.0,
        use_cross_timestep: bool = True,
        return_intermediates: bool = False,
        input_num_frames: int | None = None,
        input_height: int | None = None,
        input_width: int | None = None,
        input_fps: float | None = None,
    ) -> torch.Tensor | dict[str, Any]:
        spatial_latent = self.upsample_latent(
            latents,
            upsampler_path=spatial_upsampler_path,
            upsampler_mode="spatial",
            latents_normalized=latents_normalized,
        )
        stage2_latent = self.upsample_latent(
            spatial_latent,
            upsampler_path=temporal_upsampler_path,
            upsampler_mode="temporal",
            latents_normalized=False,
        )
        refined_latent = self.refine_latent(
            upsampled_latent=stage2_latent,
            prompts=prompts,
            negative_prompt=negative_prompt,
            fps=fps,
            seed=seed,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            noise_scale=noise_scale,
            stg_scale=stg_scale,
            modality_scale=modality_scale,
            guidance_rescale=guidance_rescale,
            use_cross_timestep=use_cross_timestep,
        )
        if not return_intermediates:
            return refined_latent
        payload = {
            "spatial_latent": spatial_latent,
            "upsampled_latent": stage2_latent,
            "refined_latent": refined_latent,
        }
        payload.update(
            self._build_pipeline_metadata(
                output_latent=refined_latent,
                fps_scale=2.0,
                input_num_frames=input_num_frames,
                input_height=input_height,
                input_width=input_width,
                input_fps=input_fps,
                spatial_scale=2.0,
                temporal_scale_num=2,
                temporal_scale_den=1,
            )
        )
        if input_num_frames is not None:
            payload["frames"] = int(input_num_frames * 2 - 1)
        return payload

    @torch.inference_mode()
    def run_temporal_then_temporal(
        self,
        latents: torch.Tensor,
        prompts: Sequence[str] | str,
        negative_prompt: Sequence[str] | str | None = None,
        *,
        temporal_upsampler_path: str | None = None,
        latents_normalized: bool = True,
        fps: float = 24.0,
        seed: int = 0,
        num_inference_steps: int = 3,
        guidance_scale: float = 1.0,
        noise_scale: float | None = None,
        stg_scale: float = 0.0,
        modality_scale: float = 1.0,
        guidance_rescale: float = 0.0,
        use_cross_timestep: bool = True,
        return_intermediates: bool = False,
        input_num_frames: int | None = None,
        input_height: int | None = None,
        input_width: int | None = None,
        input_fps: float | None = None,
    ) -> torch.Tensor | dict[str, Any]:
        temporal_latent = self.upsample_latent(
            latents,
            upsampler_path=temporal_upsampler_path,
            upsampler_mode="temporal",
            latents_normalized=latents_normalized,
        )
        stage2_latent = self.upsample_latent(
            temporal_latent,
            upsampler_path=temporal_upsampler_path,
            upsampler_mode="temporal",
            latents_normalized=False,
        )
        refined_latent = self.refine_latent(
            upsampled_latent=stage2_latent,
            prompts=prompts,
            negative_prompt=negative_prompt,
            fps=fps,
            seed=seed,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            noise_scale=noise_scale,
            stg_scale=stg_scale,
            modality_scale=modality_scale,
            guidance_rescale=guidance_rescale,
            use_cross_timestep=use_cross_timestep,
        )
        if not return_intermediates:
            return refined_latent
        payload = {
            "temporal_latent": temporal_latent,
            "upsampled_latent": stage2_latent,
            "refined_latent": refined_latent,
        }
        payload.update(
            self._build_pipeline_metadata(
                output_latent=refined_latent,
                fps_scale=4.0,
                input_num_frames=input_num_frames,
                input_height=input_height,
                input_width=input_width,
                input_fps=input_fps,
                spatial_scale=1.0,
                temporal_scale_num=4,
                temporal_scale_den=1,
            )
        )
        if input_num_frames is not None:
            payload["frames"] = int(input_num_frames * 4 - 3)
        return payload

    @torch.inference_mode()
    def run_spatial_refine_spatial_refine(
        self,
        latents: torch.Tensor,
        prompts: Sequence[str] | str,
        negative_prompt: Sequence[str] | str | None = None,
        *,
        spatial_upsampler_path: str | None = None,
        spatial_upsampler_path_stage2: str | None = None,
        latents_normalized: bool = True,
        fps: float = 24.0,
        seed: int = 0,
        num_inference_steps: int = 3,
        guidance_scale: float = 1.0,
        noise_scale: float | None = None,
        stg_scale: float = 0.0,
        modality_scale: float = 1.0,
        guidance_rescale: float = 0.0,
        use_cross_timestep: bool = True,
        return_intermediates: bool = False,
        input_num_frames: int | None = None,
        input_height: int | None = None,
        input_width: int | None = None,
        input_fps: float | None = None,
    ) -> torch.Tensor | dict[str, Any]:
        spatial_latent = self.upsample_latent(
            latents,
            upsampler_path=spatial_upsampler_path,
            upsampler_mode="spatial",
            latents_normalized=latents_normalized,
        )
        refined_spatial_latent = self.refine_latent(
            upsampled_latent=spatial_latent,
            prompts=prompts,
            negative_prompt=negative_prompt,
            fps=fps,
            seed=seed,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            noise_scale=noise_scale,
            stg_scale=stg_scale,
            modality_scale=modality_scale,
            guidance_rescale=guidance_rescale,
            use_cross_timestep=use_cross_timestep,
        )
        stage2_latent = self.upsample_latent(
            refined_spatial_latent,
            upsampler_path=self._resolve_second_spatial_upsampler_path(spatial_upsampler_path_stage2),
            upsampler_mode="spatial",
            latents_normalized=False,
        )
        refined_latent = self.refine_latent(
            upsampled_latent=stage2_latent,
            prompts=prompts,
            negative_prompt=negative_prompt,
            fps=fps,
            seed=seed,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            noise_scale=noise_scale,
            stg_scale=stg_scale,
            modality_scale=modality_scale,
            guidance_rescale=guidance_rescale,
            use_cross_timestep=use_cross_timestep,
        )
        if not return_intermediates:
            return refined_latent
        payload = {
            "spatial_latent": spatial_latent,
            "refined_spatial_latent": refined_spatial_latent,
            "upsampled_latent": stage2_latent,
            "refined_latent": refined_latent,
        }
        payload.update(
            self._build_pipeline_metadata(
                output_latent=refined_latent,
                fps_scale=1.0,
                input_num_frames=input_num_frames,
                input_height=input_height,
                input_width=input_width,
                input_fps=input_fps,
                spatial_scale=4.0,
                temporal_scale_num=1,
                temporal_scale_den=1,
            )
        )
        return payload

    @torch.inference_mode()
    def run_temporal_refine_spatial_refine(
        self,
        latents: torch.Tensor,
        prompts: Sequence[str] | str,
        negative_prompt: Sequence[str] | str | None = None,
        *,
        temporal_upsampler_path: str | None = None,
        spatial_upsampler_path: str | None = None,
        latents_normalized: bool = True,
        fps: float = 24.0,
        seed: int = 0,
        num_inference_steps: int = 3,
        guidance_scale: float = 1.0,
        noise_scale: float | None = None,
        stg_scale: float = 0.0,
        modality_scale: float = 1.0,
        guidance_rescale: float = 0.0,
        use_cross_timestep: bool = True,
        return_intermediates: bool = False,
        input_num_frames: int | None = None,
        input_height: int | None = None,
        input_width: int | None = None,
        input_fps: float | None = None,
    ) -> torch.Tensor | dict[str, Any]:
        temporal_latent = self.upsample_latent(
            latents,
            upsampler_path=temporal_upsampler_path,
            upsampler_mode="temporal",
            latents_normalized=latents_normalized,
        )
        refined_temporal_latent = self.refine_latent(
            upsampled_latent=temporal_latent,
            prompts=prompts,
            negative_prompt=negative_prompt,
            fps=fps * 2.0,
            seed=seed,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            noise_scale=noise_scale,
            stg_scale=stg_scale,
            modality_scale=modality_scale,
            guidance_rescale=guidance_rescale,
            use_cross_timestep=use_cross_timestep,
        )
        stage2_latent = self.upsample_latent(
            refined_temporal_latent,
            upsampler_path=spatial_upsampler_path,
            upsampler_mode="spatial",
            latents_normalized=False,
        )
        refined_latent = self.refine_latent(
            upsampled_latent=stage2_latent,
            prompts=prompts,
            negative_prompt=negative_prompt,
            fps=fps * 2.0,
            seed=seed,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            noise_scale=noise_scale,
            stg_scale=stg_scale,
            modality_scale=modality_scale,
            guidance_rescale=guidance_rescale,
            use_cross_timestep=use_cross_timestep,
        )
        if not return_intermediates:
            return refined_latent
        payload = {
            "temporal_latent": temporal_latent,
            "refined_temporal_latent": refined_temporal_latent,
            "upsampled_latent": stage2_latent,
            "refined_latent": refined_latent,
        }
        payload.update(
            self._build_pipeline_metadata(
                output_latent=refined_latent,
                fps_scale=2.0,
                input_num_frames=input_num_frames,
                input_height=input_height,
                input_width=input_width,
                input_fps=input_fps,
                spatial_scale=2.0,
                temporal_scale_num=2,
                temporal_scale_den=1,
            )
        )
        if input_num_frames is not None:
            payload["frames"] = int(input_num_frames * 2 - 1)
        return payload

    def _pack_video_latents(self, latents: torch.Tensor) -> torch.Tensor:
        spatial_patch_size = int(getattr(self._pipe.transformer.config, "patch_size", 1))
        temporal_patch_size = int(getattr(self._pipe.transformer.config, "patch_size_t", 1))
        return self._pipe._pack_latents(latents, spatial_patch_size, temporal_patch_size)

    def _unpack_video_latents(
        self,
        latents: torch.Tensor,
        *,
        latent_num_frames: int,
        latent_height: int,
        latent_width: int,
    ) -> torch.Tensor:
        spatial_patch_size = int(getattr(self._pipe.transformer.config, "patch_size", 1))
        temporal_patch_size = int(getattr(self._pipe.transformer.config, "patch_size_t", 1))
        return self._pipe._unpack_latents(
            latents,
            int(latent_num_frames),
            int(latent_height),
            int(latent_width),
            spatial_patch_size,
            temporal_patch_size,
        )

    def _validate_streaming_window_shapes(
        self, window_latents: Sequence[torch.Tensor]
    ) -> tuple[int, int, int, int, int]:
        if not window_latents:
            raise ValueError("window_latents must not be empty.")

        first = window_latents[0]
        if first.ndim != 5:
            raise ValueError(f"Expected 5D latent tensor [B, C, F, H, W], got shape={tuple(first.shape)}")
        batch_size, channels, num_frames, height, width = (int(x) for x in first.shape)
        for idx, item in enumerate(window_latents[1:], start=1):
            if item.ndim != 5:
                raise ValueError(f"Window {idx} is not 5D. Got shape={tuple(item.shape)}")
            current = tuple(int(x) for x in item.shape)
            if current != (batch_size, channels, num_frames, height, width):
                raise ValueError(
                    "All window latents must have identical shapes. "
                    f"Expected {(batch_size, channels, num_frames, height, width)}, got {current} at index {idx}."
                )
        return batch_size, channels, num_frames, height, width

    def _build_streaming_window_entries(
        self,
        stage2_latent: torch.Tensor,
        *,
        window_num_frames: int,
        context_num_frames: int,
        pad_last_window: bool,
    ) -> list[dict[str, Any]]:
        total_num_frames = int(stage2_latent.shape[2])
        if int(window_num_frames) <= 0:
            raise ValueError("window_num_frames must be > 0.")
        if int(context_num_frames) < 0:
            raise ValueError("context_num_frames must be >= 0.")
        if int(context_num_frames) >= int(window_num_frames):
            raise ValueError(
                f"context_num_frames must be smaller than window_num_frames. "
                f"Got context={context_num_frames}, window={window_num_frames}."
            )

        stride = int(window_num_frames) - int(context_num_frames)
        entries: list[dict[str, Any]] = []
        start = 0
        while start < total_num_frames:
            end = min(total_num_frames, start + int(window_num_frames))
            current = stage2_latent[:, :, start:end]
            valid_len = int(current.shape[2])
            pad_count = 0
            if valid_len < int(window_num_frames):
                if not bool(pad_last_window):
                    break
                pad_count = int(window_num_frames) - valid_len
                current = torch.cat([current, current[:, :, -1:].repeat(1, 1, pad_count, 1, 1)], dim=2)
            entries.append(
                {
                    "latent": current,
                    "global_start_frame": int(start),
                    "global_end_frame_exclusive": int(end),
                    "valid_num_frames": int(valid_len),
                    "pad_count": int(pad_count),
                }
            )
            if end >= total_num_frames:
                break
            start += stride
        return entries

    def _create_stage2_noised_latents(
        self,
        stage2_latent: torch.Tensor,
        *,
        noise_scale: float,
        seed: int | None,
    ) -> torch.Tensor:
        latents = stage2_latent.to(device=self.device, dtype=self.dtype)
        generator = torch.Generator(device=self.device).manual_seed(int(seed)) if seed is not None else None
        create_noised_state = getattr(self._pipe, "_create_noised_state", None)
        if callable(create_noised_state):
            for args, kwargs in (
                ((latents,), {"noise_scale": float(noise_scale), "generator": generator}),
                ((latents, float(noise_scale)), {"generator": generator}),
                ((latents, float(noise_scale), generator), {}),
            ):
                try:
                    noised = create_noised_state(*args, **kwargs)
                    return noised.to(device=self.device, dtype=self.dtype)
                except TypeError:
                    continue

        noise = torch.randn(latents.shape, generator=generator, device=latents.device, dtype=latents.dtype)
        scale = torch.tensor([float(noise_scale)], device=latents.device, dtype=latents.dtype)[
            :, None, None, None, None
        ]
        return (1 - scale) * latents + scale * noise

    @torch.inference_mode()
    def refine_latent_streaming_windows(
        self,
        window_latents: Sequence[torch.Tensor],
        prompts: Sequence[str] | str,
        negative_prompt: Sequence[str] | str | None = None,
        *,
        context_num_frames: int,
        fps: float = 24.0,
        seed: int = 0,
        num_inference_steps: int = 3,
        guidance_scale: float = 1.0,
        noise_scale: float | None = None,
        stg_scale: float = 0.0,
        modality_scale: float = 1.0,
        guidance_rescale: float = 0.0,
        use_cross_timestep: bool = True,
        return_intermediates: bool = False,
    ) -> torch.Tensor | dict[str, Any]:
        _, _, window_num_frames, latent_height, latent_width = self._validate_streaming_window_shapes(window_latents)
        if int(context_num_frames) < 0:
            raise ValueError("context_num_frames must be >= 0.")
        if int(context_num_frames) >= int(window_num_frames):
            raise ValueError(
                f"context_num_frames must be smaller than the window size. Got context={context_num_frames}, "
                f"window_num_frames={window_num_frames}."
            )

        tail_step_cache: list[torch.Tensor] | None = None
        tail_clean_cache: torch.Tensor | None = None
        stitched_chunks: list[torch.Tensor] = []
        refined_windows: list[torch.Tensor] = []
        per_window_metadata: list[dict[str, Any]] = []

        for window_index, raw_window_latent in enumerate(window_latents):
            current_window = raw_window_latent.detach().clone().to(device=self.device, dtype=self.dtype)
            prev_tail_clean_cache = tail_clean_cache
            context_pre_replace_l1: float | None = None
            context_post_replace_l1: float | None = None
            context_final_prefix_l1: float | None = None
            if tail_clean_cache is not None and int(context_num_frames) > 0:
                context_pre_replace_l1 = float(
                    (
                        raw_window_latent[:, :, : int(context_num_frames)].detach().float().cpu()
                        - tail_clean_cache[:, :, : int(context_num_frames)].detach().float().cpu()
                    )
                    .abs()
                    .mean()
                    .item()
                )
                current_window[:, :, : int(context_num_frames)] = tail_clean_cache.to(
                    device=current_window.device,
                    dtype=current_window.dtype,
                )
                context_post_replace_l1 = float(
                    (
                        current_window[:, :, : int(context_num_frames)].detach().float().cpu()
                        - tail_clean_cache[:, :, : int(context_num_frames)].detach().float().cpu()
                    )
                    .abs()
                    .mean()
                    .item()
                )

            next_tail_step_cache: list[torch.Tensor] = []

            def _streaming_callback(pipe, i, t, callback_kwargs):
                del pipe, t
                packed_latents = callback_kwargs["latents"]
                unpacked_latents = self._unpack_video_latents(
                    packed_latents,
                    latent_num_frames=window_num_frames,
                    latent_height=latent_height,
                    latent_width=latent_width,
                )
                if tail_step_cache is not None and int(context_num_frames) > 0:
                    unpacked_latents[:, :, : int(context_num_frames)] = tail_step_cache[i].to(
                        device=unpacked_latents.device,
                        dtype=unpacked_latents.dtype,
                    )
                if int(context_num_frames) > 0:
                    next_tail_step_cache.append(unpacked_latents[:, :, -int(context_num_frames) :].detach().cpu())
                return {"latents": self._pack_video_latents(unpacked_latents)}

            call_kwargs = self._build_stage2_call_kwargs(
                stage2_latent=current_window,
                prompts=prompts,
                negative_prompt=negative_prompt,
                seed=int(seed),
                num_inference_steps=int(num_inference_steps),
                guidance_scale=float(guidance_scale),
                noise_scale=noise_scale,
                stg_scale=float(stg_scale),
                modality_scale=float(modality_scale),
                guidance_rescale=float(guidance_rescale),
                use_cross_timestep=bool(use_cross_timestep),
                output_type="latent",
            )
            call_kwargs["callback_on_step_end"] = _streaming_callback
            call_kwargs["callback_on_step_end_tensor_inputs"] = ["latents"]

            refined_window, _ = self._pipe(**call_kwargs)
            refined_window = refined_window.to(device=self.device, dtype=self.dtype)
            if prev_tail_clean_cache is not None and int(context_num_frames) > 0:
                context_final_prefix_l1 = float(
                    (
                        refined_window[:, :, : int(context_num_frames)].detach().float().cpu()
                        - prev_tail_clean_cache[:, :, : int(context_num_frames)].detach().float().cpu()
                    )
                    .abs()
                    .mean()
                    .item()
                )
            refined_windows.append(refined_window.detach().cpu())

            if int(context_num_frames) > 0:
                tail_clean_cache = refined_window[:, :, -int(context_num_frames) :].detach().cpu()
                tail_step_cache = next_tail_step_cache
            else:
                tail_clean_cache = None
                tail_step_cache = None

            contributed = refined_window
            if window_index > 0 and int(context_num_frames) > 0:
                contributed = contributed[:, :, int(context_num_frames) :]
            stitched_chunks.append(contributed.detach().cpu())
            per_window_metadata.append(
                {
                    "window_index": int(window_index),
                    "window_num_frames": int(window_num_frames),
                    "context_num_frames": int(context_num_frames),
                    "contributed_num_frames": int(contributed.shape[2]),
                    "shared_noise_seed": int(seed),
                    "context_pre_replace_l1": context_pre_replace_l1,
                    "context_post_replace_l1": context_post_replace_l1,
                    "context_final_prefix_l1": context_final_prefix_l1,
                    "context_step_cache_depth": 0 if tail_step_cache is None else len(tail_step_cache),
                }
            )

        stitched = torch.cat(stitched_chunks, dim=2)
        if not return_intermediates:
            return stitched
        return {
            "refined_latent": stitched,
            "refined_window_latents": refined_windows,
            "window_metadata": per_window_metadata,
            "context_num_frames": int(context_num_frames),
            "window_num_frames": int(window_num_frames),
            "num_windows": len(refined_windows),
            "shared_noise_seed": int(seed),
        }

    @torch.inference_mode()
    def run_refine_only_streaming(
        self,
        latents: torch.Tensor,
        prompts: Sequence[str] | str,
        negative_prompt: Sequence[str] | str | None = None,
        *,
        latents_normalized: bool = True,
        window_num_frames: int = 81,
        context_num_frames: int = 25,
        pad_last_window: bool = True,
        fps: float = 24.0,
        seed: int = 0,
        num_inference_steps: int = 3,
        guidance_scale: float = 1.0,
        noise_scale: float | None = None,
        stg_scale: float = 0.0,
        modality_scale: float = 1.0,
        guidance_rescale: float = 0.0,
        use_cross_timestep: bool = True,
        return_intermediates: bool = False,
        input_num_frames: int | None = None,
        input_height: int | None = None,
        input_width: int | None = None,
        input_fps: float | None = None,
    ) -> torch.Tensor | dict[str, Any]:
        stage2_latent = latents if not latents_normalized else self.normalized_to_raw_latent(latents)
        if stage2_latent.ndim != 5:
            raise ValueError(
                f"run_refine_only_streaming expects raw stage-2 latents of shape [B, C, F, H, W], "
                f"got shape={tuple(stage2_latent.shape)}"
            )

        total_num_frames = int(stage2_latent.shape[2])
        if total_num_frames <= int(window_num_frames):
            return self.run_refine_only(
                stage2_latent,
                prompts=prompts,
                negative_prompt=negative_prompt,
                latents_normalized=False,
                fps=fps,
                seed=seed,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                noise_scale=noise_scale,
                stg_scale=stg_scale,
                modality_scale=modality_scale,
                guidance_rescale=guidance_rescale,
                use_cross_timestep=use_cross_timestep,
                return_intermediates=return_intermediates,
                input_num_frames=input_num_frames,
                input_height=input_height,
                input_width=input_width,
                input_fps=input_fps,
            )

        if int(window_num_frames) <= 0:
            raise ValueError("window_num_frames must be > 0.")
        if int(context_num_frames) < 0:
            raise ValueError("context_num_frames must be >= 0.")
        if int(context_num_frames) >= int(window_num_frames):
            raise ValueError(
                f"context_num_frames must be smaller than window_num_frames. "
                f"Got context={context_num_frames}, window={window_num_frames}."
            )

        stride = int(window_num_frames) - int(context_num_frames)
        window_latents: list[torch.Tensor] = []
        start = 0
        while start < total_num_frames:
            end = min(total_num_frames, start + int(window_num_frames))
            current = stage2_latent[:, :, start:end]
            valid_len = int(current.shape[2])
            if valid_len < int(window_num_frames):
                if not bool(pad_last_window):
                    break
                pad_count = int(window_num_frames) - valid_len
                current = torch.cat([current, current[:, :, -1:].repeat(1, 1, pad_count, 1, 1)], dim=2)
            window_latents.append(current)
            if end >= total_num_frames:
                break
            start += stride

        result = self.refine_latent_streaming_windows(
            window_latents,
            prompts=prompts,
            negative_prompt=negative_prompt,
            context_num_frames=int(context_num_frames),
            fps=fps,
            seed=seed,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            noise_scale=noise_scale,
            stg_scale=stg_scale,
            modality_scale=modality_scale,
            guidance_rescale=guidance_rescale,
            use_cross_timestep=use_cross_timestep,
            return_intermediates=True,
        )

        refined_latent = result["refined_latent"][:, :, :total_num_frames]
        if not return_intermediates:
            return refined_latent

        payload = {
            "stage2_latent": stage2_latent,
            "refined_latent": refined_latent,
            "window_num_frames": int(window_num_frames),
            "context_num_frames": int(context_num_frames),
            "num_windows": int(result["num_windows"]),
            "window_metadata": result["window_metadata"],
            "shared_noise_seed": int(result["shared_noise_seed"]),
        }
        payload.update(
            self._build_pipeline_metadata(
                output_latent=refined_latent,
                fps_scale=1.0,
                input_num_frames=input_num_frames,
                input_height=input_height,
                input_width=input_width,
                input_fps=input_fps,
                spatial_scale=1.0,
                temporal_scale_num=1,
                temporal_scale_den=1,
            )
        )
        return payload

    def set_decode_tiling(self, enable: bool, temporal_tiling: bool = False) -> None:
        if enable:
            if hasattr(self._pipe.vae, "enable_tiling"):
                self._pipe.vae.enable_tiling()
            if temporal_tiling:
                vae = self._pipe.vae
                # Use larger temporal tiles with more overlap for better blending.
                # Default (16 tile / 8 stride) gives only 1 latent frame overlap → bad quality.
                # 80 tile / 24 stride → 10 latent frames per tile, 3 stride, 7 overlap → smooth blending.
                vae.tile_sample_min_num_frames = 80
                vae.tile_sample_stride_num_frames = 24
                vae.use_framewise_decoding = True
                vae.use_framewise_encoding = True
                self.logger.info(
                    "[LTXRefiner] VAE tiling enabled (spatial + temporal, "
                    "tile=%d, stride=%d, ~%d latent frames overlap).",
                    vae.tile_sample_min_num_frames,
                    vae.tile_sample_stride_num_frames,
                    vae.tile_sample_min_num_frames // 8 - vae.tile_sample_stride_num_frames // 8,
                )
            else:
                self.logger.info("[LTXRefiner] VAE tiling enabled (spatial only).")
        else:
            self.logger.info("[LTXRefiner] VAE tiling disabled.")

    def release_for_decode(self) -> None:
        _cleanup_memory()

    def close(self) -> None:
        self._pipe = None
        self._upsample_pipes = {}
        _cleanup_memory()
