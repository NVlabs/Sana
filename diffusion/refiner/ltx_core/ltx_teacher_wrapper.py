"""Training-oriented frozen teacher wrapper for LTX-2 stage-2 refiner.

Built on the vendored raw LTX implementation.  Video-only for v1.

* ``forward_denoise_step``  -- single transformer forward, returns denoised prediction
* ``rollout_refine``        -- multi-step Euler rollout with per-step outputs
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, replace

import torch

from diffusion.refiner.vendor.ltx_core.components.diffusion_steps import EulerDiffusionStep
from diffusion.refiner.vendor.ltx_core.components.noisers import GaussianNoiser
from diffusion.refiner.vendor.ltx_core.components.patchifiers import VideoLatentPatchifier
from diffusion.refiner.vendor.ltx_core.loader import LoraPathStrengthAndSDOps  # re-exported for callers
from diffusion.refiner.vendor.ltx_core.loader.single_gpu_model_builder import SingleGPUModelBuilder as Builder
from diffusion.refiner.vendor.ltx_core.model.transformer import (
    LTXV_MODEL_COMFY_RENAMING_MAP,
    LTXModelConfigurator,
    X0Model,
)
from diffusion.refiner.vendor.ltx_core.tools import VideoLatentTools
from diffusion.refiner.vendor.ltx_core.types import LatentState, VideoLatentShape
from diffusion.refiner.vendor.ltx_pipelines.utils.constants import STAGE_2_DISTILLED_SIGMA_VALUES
from diffusion.refiner.vendor.ltx_pipelines.utils.helpers import (
    modality_from_latent_state,
    post_process_latent,
)

logger = logging.getLogger(__name__)


@dataclass
class TeacherStepOutput:
    """Per-step output: denoised prediction + input state + sigma."""

    video_denoised: torch.Tensor  # [B, T, D] patchified denoised prediction
    video_state: LatentState  # input state at this step (before Euler step)
    sigma: float
    step_index: int


@dataclass
class RolloutOutput:
    """Multi-step rollout result."""

    step_outputs: list[TeacherStepOutput]
    final_video_state: LatentState  # patchified
    final_latent: torch.Tensor  # [B, C, F, H, W] unpatchified


class LTXTeacherWrapper:
    """Frozen teacher for distillation.  Video-only, no CFG, no audio."""

    def __init__(
        self,
        checkpoint_path: str,
        loras: list[LoraPathStrengthAndSDOps] | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        self.device = device or (
            torch.device("cuda", torch.cuda.current_device()) if torch.cuda.is_available() else torch.device("cpu")
        )
        self.dtype = dtype

        builder = Builder(
            model_path=checkpoint_path,
            model_class_configurator=LTXModelConfigurator,
            model_sd_ops=LTXV_MODEL_COMFY_RENAMING_MAP,
            loras=tuple(loras or ()),
        )
        self._transformer: X0Model = X0Model(builder.build(device=self.device)).to(self.device).eval()
        for p in self._transformer.parameters():
            p.requires_grad_(False)

        self._sigmas = torch.tensor(
            STAGE_2_DISTILLED_SIGMA_VALUES,
            dtype=torch.float32,
            device=self.device,
        )
        self._stepper = EulerDiffusionStep()
        self._patchifier = VideoLatentPatchifier(patch_size=1)

        logger.info("[LTXTeacherWrapper] Ready on %s, %d LoRAs merged.", self.device, len(loras or ()))

    # --- properties ---

    @property
    def transformer(self) -> X0Model:
        return self._transformer

    @property
    def stage2_sigmas(self) -> torch.Tensor:
        return self._sigmas

    @property
    def context_dim(self) -> int:
        """Input dim expected by caption_projection (for building dummy contexts)."""
        vm = self._transformer.velocity_model
        proj = getattr(vm, "caption_projection", None)
        if proj is not None:
            return proj.linear_1.in_features
        return vm.cross_attention_dim

    def is_frozen(self) -> bool:
        return all(not p.requires_grad for p in self._transformer.parameters())

    # --- state construction ---

    def create_noised_video_state(
        self,
        *,
        stage2_latent: torch.Tensor,
        noise_scale: float | None = None,
        seed: int = 0,
        fps: float = 24.0,
    ) -> tuple[LatentState, VideoLatentTools]:
        """Patchified noised state from a raw stage-2 latent [B,C,F,H,W]."""
        B, C, F, H, W = stage2_latent.shape
        tools = VideoLatentTools(
            patchifier=self._patchifier,
            target_shape=VideoLatentShape(batch=B, channels=C, frames=F, height=H, width=W),
            fps=fps,
        )
        state = tools.create_initial_state(
            device=self.device,
            dtype=self.dtype,
            initial_latent=stage2_latent.to(device=self.device, dtype=self.dtype),
        )
        scale = float(self._sigmas[0].item()) if noise_scale is None else float(noise_scale)
        noiser = GaussianNoiser(torch.Generator(device=self.device).manual_seed(seed))
        return noiser(state, scale), tools

    # --- core: single step ---

    @torch.no_grad()
    def forward_denoise_step(
        self,
        *,
        video_state: LatentState,
        v_context: torch.Tensor,
        sigmas: torch.Tensor,
        step_index: int,
    ) -> torch.Tensor:
        """Single teacher forward.  Returns denoised video prediction [B, T, D].

        LoRA is bypassed so the teacher always produces the original frozen
        refiner output, even when weights are shared with the student.
        """
        from .ltx_student_wrapper import LoRALinear

        sigma = sigmas[step_index]
        video_mod = modality_from_latent_state(video_state, v_context, sigma)
        LoRALinear.enabled = False
        try:
            denoised_video, _ = self._transformer(video=video_mod, audio=None, perturbations=None)
        finally:
            LoRALinear.enabled = True
        return denoised_video

    # --- core: multi-step rollout ---

    @torch.no_grad()
    def rollout_refine(
        self,
        *,
        stage2_latent: torch.Tensor,
        v_context: torch.Tensor,
        num_inference_steps: int | None = None,
        noise_scale: float | None = None,
        seed: int = 0,
        fps: float = 24.0,
        return_step_outputs: bool = True,
    ) -> RolloutOutput:
        """Multi-step Euler rollout.  Returns per-step outputs + final latent."""
        if stage2_latent.ndim != 5:
            raise ValueError(f"Expected [B,C,F,H,W], got {tuple(stage2_latent.shape)}")

        sigmas = self._sigmas
        if num_inference_steps is not None:
            sigmas = sigmas[: num_inference_steps + 1]
        sigmas = sigmas.to(device=self.device, dtype=torch.float32)

        video_state, video_tools = self.create_noised_video_state(
            stage2_latent=stage2_latent,
            noise_scale=noise_scale,
            seed=seed,
            fps=fps,
        )

        step_outputs: list[TeacherStepOutput] = []

        for step_idx in range(len(sigmas) - 1):
            denoised = self.forward_denoise_step(
                video_state=video_state,
                v_context=v_context,
                sigmas=sigmas,
                step_index=step_idx,
            )

            if return_step_outputs:
                step_outputs.append(
                    TeacherStepOutput(
                        video_denoised=denoised,
                        video_state=video_state,
                        sigma=float(sigmas[step_idx].item()),
                        step_index=step_idx,
                    )
                )

            denoised = post_process_latent(denoised, video_state.denoise_mask, video_state.clean_latent)
            next_latent = self._stepper.step(video_state.latent, denoised, sigmas, step_idx)
            video_state = replace(video_state, latent=next_latent)

        video_state = video_tools.clear_conditioning(video_state)
        final = video_tools.unpatchify(video_state)

        return RolloutOutput(
            step_outputs=step_outputs,
            final_video_state=video_state,
            final_latent=final.latent,
        )

    # --- lifecycle ---

    def close(self) -> None:
        if hasattr(self, "_transformer"):
            self._transformer.to("meta")
            del self._transformer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
