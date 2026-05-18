"""Reference video conditioning for IC-LoRA inference."""

import torch

from diffusion.refiner.vendor.ltx_core.components.patchifiers import get_pixel_coords
from diffusion.refiner.vendor.ltx_core.conditioning.item import ConditioningItem
from diffusion.refiner.vendor.ltx_core.conditioning.mask_utils import update_attention_mask
from diffusion.refiner.vendor.ltx_core.tools import VideoLatentTools
from diffusion.refiner.vendor.ltx_core.types import LatentState, VideoLatentShape


class VideoConditionByReferenceLatent(ConditioningItem):
    """
    Conditions video generation on a reference video latent for IC-LoRA inference.
    IC-LoRAs are trained by concatenating reference (control signal) and target tokens,
    learning to attend across both. This class replicates that setup at inference by
    appending reference tokens to the latent sequence.
    IC-LoRAs can be trained with lower-resolution references than the target (e.g., 384px
    reference for 768px output) for efficiency and better generalization. The
    `downscale_factor` scales reference positions to match target coordinates, preserving
    the learned positional relationships. This must match the factor used during training
    (stored in LoRA metadata).
    To add attention masking, wrap with :class:`ConditioningItemAttentionStrengthWrapper`.
    Args:
        latent: Reference video latents [B, C, F, H, W]
        downscale_factor: Target/reference resolution ratio (e.g., 2 = half-resolution
            reference). Spatial positions are scaled by this factor.
        strength: Conditioning strength. 1.0 = full (reference kept clean),
            0.0 = none (reference denoised). Default 1.0.
    """

    def __init__(
        self,
        latent: torch.Tensor,
        downscale_factor: int = 1,
        strength: float = 1.0,
    ):
        self.latent = latent
        self.downscale_factor = downscale_factor
        self.strength = strength

    def apply_to(
        self,
        latent_state: LatentState,
        latent_tools: VideoLatentTools,
    ) -> LatentState:
        """Append reference video tokens with scaled positions."""
        tokens = latent_tools.patchifier.patchify(self.latent)

        # Compute positions for the reference video's actual dimensions
        latent_coords = latent_tools.patchifier.get_patch_grid_bounds(
            output_shape=VideoLatentShape.from_torch_shape(self.latent.shape),
            device=self.latent.device,
        )
        positions = get_pixel_coords(
            latent_coords=latent_coords,
            scale_factors=latent_tools.scale_factors,
            causal_fix=latent_tools.causal_fix,
        )
        positions = positions.to(dtype=torch.float32)
        positions[:, 0, ...] /= latent_tools.fps

        # Scale spatial positions to match target coordinate space
        if self.downscale_factor != 1:
            positions[:, 1, ...] *= self.downscale_factor  # height axis
            positions[:, 2, ...] *= self.downscale_factor  # width axis

        denoise_mask = torch.full(
            size=(*tokens.shape[:2], 1),
            fill_value=1.0 - self.strength,
            device=self.latent.device,
            dtype=self.latent.dtype,
        )

        new_attention_mask = update_attention_mask(
            latent_state=latent_state,
            attention_mask=None,
            num_noisy_tokens=latent_tools.target_shape.token_count(),
            num_new_tokens=tokens.shape[1],
            batch_size=tokens.shape[0],
            device=self.latent.device,
            dtype=self.latent.dtype,
        )

        return LatentState(
            latent=torch.cat([latent_state.latent, tokens], dim=1),
            denoise_mask=torch.cat([latent_state.denoise_mask, denoise_mask], dim=1),
            positions=torch.cat([latent_state.positions, positions], dim=2),
            clean_latent=torch.cat([latent_state.clean_latent, tokens], dim=1),
            attention_mask=new_attention_mask,
        )
