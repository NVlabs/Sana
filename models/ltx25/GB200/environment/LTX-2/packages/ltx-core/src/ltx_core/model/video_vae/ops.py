import torch
from einops import rearrange
from torch import nn


def patchify(x: torch.Tensor, patch_size_hw: int, patch_size_t: int = 1) -> torch.Tensor:
    """
    Rearrange spatial dimensions into channels. Divides image into patch_size x patch_size blocks
    and moves pixels from each block into separate channels (space-to-depth).
    Args:
        x: Input tensor (4D or 5D)
        patch_size_hw: Spatial patch size for height and width. With patch_size_hw=4, divides HxW into 4x4 blocks.
        patch_size_t: Temporal patch size for frames. Default=1 (no temporal patching).
    For 5D: (B, C, F, H, W) -> (B, Cx(patch_size_hw^2)x(patch_size_t), F/patch_size_t, H/patch_size_hw, W/patch_size_hw)
    Example: (B, 3, 33, 512, 512) with patch_size_hw=4, patch_size_t=1 -> (B, 48, 33, 128, 128)
    """
    if patch_size_hw == 1 and patch_size_t == 1:
        return x
    if x.dim() == 4:
        x = rearrange(x, "b c (h q) (w r) -> b (c r q) h w", q=patch_size_hw, r=patch_size_hw)
    elif x.dim() == 5:
        x = rearrange(
            x,
            "b c (f p) (h q) (w r) -> b (c p r q) f h w",
            p=patch_size_t,
            q=patch_size_hw,
            r=patch_size_hw,
        )
    else:
        raise ValueError(f"Invalid input shape: {x.shape}")

    return x


def unpatchify(x: torch.Tensor, patch_size_hw: int, patch_size_t: int = 1) -> torch.Tensor:
    """
    Rearrange channels back into spatial dimensions. Inverse of patchify - moves pixels from
    channels back into patch_size x patch_size blocks (depth-to-space).
    Args:
        x: Input tensor (4D or 5D)
        patch_size_hw: Spatial patch size for height and width. With patch_size_hw=4, expands HxW by 4x.
        patch_size_t: Temporal patch size for frames. Default=1 (no temporal expansion).
    For 5D: (B, Cx(patch_size_hw^2)x(patch_size_t), F, H, W) -> (B, C, Fxpatch_size_t, Hxpatch_size_hw, Wxpatch_size_hw)
    Example: (B, 48, 33, 128, 128) with patch_size_hw=4, patch_size_t=1 -> (B, 3, 33, 512, 512)
    """
    if patch_size_hw == 1 and patch_size_t == 1:
        return x

    if x.dim() == 4:
        x = rearrange(x, "b (c r q) h w -> b c (h q) (w r)", q=patch_size_hw, r=patch_size_hw)
    elif x.dim() == 5:
        x = rearrange(
            x,
            "b (c p r q) f h w -> b c (f p) (h q) (w r)",
            p=patch_size_t,
            q=patch_size_hw,
            r=patch_size_hw,
        )

    return x


class PerChannelStatistics(nn.Module):
    """
    Per-channel statistics for normalizing and denormalizing the latent representation.
    This statics is computed over the entire dataset and stored in model's checkpoint under VAE state_dict.
    """

    def __init__(self, latent_channels: int = 128):
        super().__init__()
        self.register_buffer("std-of-means", torch.empty(latent_channels))
        self.register_buffer("mean-of-means", torch.empty(latent_channels))

    def un_normalize(self, x: torch.Tensor) -> torch.Tensor:
        return (x * self.get_buffer("std-of-means").view(1, -1, 1, 1, 1).to(x)) + self.get_buffer("mean-of-means").view(
            1, -1, 1, 1, 1
        ).to(x)

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.get_buffer("mean-of-means").view(1, -1, 1, 1, 1).to(x)) / self.get_buffer("std-of-means").view(
            1, -1, 1, 1, 1
        ).to(x)
