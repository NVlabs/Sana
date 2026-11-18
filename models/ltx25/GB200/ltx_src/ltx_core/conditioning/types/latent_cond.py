import torch

from ltx_core.conditioning.exceptions import ConditioningError
from ltx_core.conditioning.item import ConditioningItem
from ltx_core.tools import LatentTools
from ltx_core.types import LatentState


class VideoConditionByLatentIndex(ConditioningItem):
    """
    Conditions video generation by injecting latents at a specific latent frame index.
    Sets the clean latents at positions corresponding to latent_idx to the injected latents,
    sets denoise strength according to the strength parameter.
    """

    def __init__(self, latent: torch.Tensor, strength: float, latent_idx: int):
        self.latent = latent
        self.strength = strength
        self.latent_idx = latent_idx

    def apply_to(self, latent_state: LatentState, latent_tools: LatentTools) -> LatentState:
        cond_batch, cond_channels, _, cond_height, cond_width = self.latent.shape
        tgt_batch, tgt_channels, tgt_frames, tgt_height, tgt_width = latent_tools.target_shape.to_torch_shape()

        if (cond_batch, cond_channels, cond_height, cond_width) != (tgt_batch, tgt_channels, tgt_height, tgt_width):
            raise ConditioningError(
                f"Can't apply image conditioning item to latent with shape {latent_tools.target_shape}, expected "
                f"shape is ({tgt_batch}, {tgt_channels}, {tgt_frames}, {tgt_height}, {tgt_width}). Make sure "
                "the image and latent have the same spatial shape."
            )

        tokens = latent_tools.patchifier.patchify(self.latent)
        start_token = latent_tools.patchifier.get_token_count(
            latent_tools.target_shape._replace(frames=self.latent_idx)
        )
        stop_token = start_token + tokens.shape[1]

        latent_state = latent_state.clone()

        latent_state.clean_latent[:, start_token:stop_token] = tokens
        latent_state.denoise_mask[:, start_token:stop_token] = 1.0 - self.strength

        return latent_state
