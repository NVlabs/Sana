from dataclasses import dataclass, replace
from typing import Protocol

import torch
from torch._prims_common import DeviceLikeType

from ltx_core.components.patchifiers import (
    AudioLatentShape,
    AudioPatchifier,
    VideoLatentPatchifier,
    VideoLatentShape,
    get_pixel_coords,
)
from ltx_core.components.protocols import Patchifier
from ltx_core.types import LatentState, SpatioTemporalScaleFactors

DEFAULT_SCALE_FACTORS = SpatioTemporalScaleFactors.default()


class LatentTools(Protocol):
    """
    Tools for building latent states.
    """

    patchifier: Patchifier
    target_shape: VideoLatentShape | AudioLatentShape

    def create_initial_state(
        self,
        device: DeviceLikeType,
        dtype: torch.dtype,
        initial_latent: torch.Tensor | None = None,
    ) -> LatentState:
        """
        Create an initial latent state. If initial_latent is provided, it will be used to create the latent state.
        """
        ...

    def patchify(self, latent_state: LatentState) -> LatentState:
        """
        Patchify the latent state.
        """
        if latent_state.latent.shape != self.target_shape.to_torch_shape():
            raise ValueError(
                f"Latent state has shape {latent_state.latent.shape}, expected shape is "
                f"{self.target_shape.to_torch_shape()}"
            )
        latent_state = latent_state.clone()
        latent = self.patchifier.patchify(latent_state.latent)
        clean_latent = self.patchifier.patchify(latent_state.clean_latent)
        denoise_mask = self.patchifier.patchify(latent_state.denoise_mask)
        return replace(latent_state, latent=latent, denoise_mask=denoise_mask, clean_latent=clean_latent)

    def unpatchify(self, latent_state: LatentState) -> LatentState:
        """
        Unpatchify the latent state.
        """
        latent_state = latent_state.clone()
        latent = self.patchifier.unpatchify(latent_state.latent, output_shape=self.target_shape)
        clean_latent = self.patchifier.unpatchify(latent_state.clean_latent, output_shape=self.target_shape)
        denoise_mask = self.patchifier.unpatchify(
            latent_state.denoise_mask, output_shape=self.target_shape.mask_shape()
        )
        return replace(latent_state, latent=latent, denoise_mask=denoise_mask, clean_latent=clean_latent)

    def clear_conditioning(self, latent_state: LatentState) -> LatentState:
        """
        Clear the conditioning from the latent state. This method removes extra tokens from the end of the latent.
        Therefore, conditioning items should add extra tokens ONLY to the end of the latent.
        """
        latent_state = latent_state.clone()

        num_tokens = self.patchifier.get_token_count(self.target_shape)
        latent = latent_state.latent[:, :num_tokens]
        clean_latent = latent_state.clean_latent[:, :num_tokens]
        denoise_mask = torch.ones_like(latent_state.denoise_mask)[:, :num_tokens]
        positions = latent_state.positions[:, :, :num_tokens]

        return LatentState(
            latent=latent,
            denoise_mask=denoise_mask,
            positions=positions,
            clean_latent=clean_latent,
            attention_mask=None,
        )


@dataclass(frozen=True)
class VideoLatentTools(LatentTools):
    """
    Tools for building video latent states.
    """

    patchifier: VideoLatentPatchifier
    target_shape: VideoLatentShape
    fps: float
    scale_factors: SpatioTemporalScaleFactors = DEFAULT_SCALE_FACTORS
    causal_fix: bool = True

    def create_initial_state(
        self,
        device: DeviceLikeType,
        dtype: torch.dtype,
        initial_latent: torch.Tensor | None = None,
    ) -> LatentState:
        if initial_latent is not None:
            assert initial_latent.shape == self.target_shape.to_torch_shape(), (
                f"Latent shape {initial_latent.shape} does not match target shape {self.target_shape.to_torch_shape()}"
            )
        else:
            initial_latent = torch.zeros(
                *self.target_shape.to_torch_shape(),
                device=device,
                dtype=dtype,
            )

        clean_latent = initial_latent.clone()

        denoise_mask = torch.ones(
            *self.target_shape.mask_shape().to_torch_shape(),
            device=device,
            dtype=torch.float32,
        )

        latent_coords = self.patchifier.get_patch_grid_bounds(
            output_shape=self.target_shape,
            device=device,
        )

        positions = get_pixel_coords(
            latent_coords=latent_coords,
            scale_factors=self.scale_factors,
            causal_fix=self.causal_fix,
        ).float()
        positions[:, 0, ...] = positions[:, 0, ...] / self.fps

        return self.patchify(
            LatentState(
                latent=initial_latent,
                denoise_mask=denoise_mask,
                positions=positions,
                clean_latent=clean_latent,
            )
        )


@dataclass(frozen=True)
class AudioLatentTools(LatentTools):
    """
    Tools for building audio latent states.
    """

    patchifier: AudioPatchifier
    target_shape: AudioLatentShape

    def create_initial_state(
        self,
        device: DeviceLikeType,
        dtype: torch.dtype,
        initial_latent: torch.Tensor | None = None,
    ) -> LatentState:
        if initial_latent is not None:
            assert initial_latent.shape == self.target_shape.to_torch_shape(), (
                f"Latent shape {initial_latent.shape} does not match target shape {self.target_shape.to_torch_shape()}"
            )
        else:
            initial_latent = torch.zeros(
                *self.target_shape.to_torch_shape(),
                device=device,
                dtype=dtype,
            )

        clean_latent = initial_latent.clone()

        denoise_mask = torch.ones(
            *self.target_shape.mask_shape().to_torch_shape(),
            device=device,
            dtype=torch.float32,
        )

        latent_coords = self.patchifier.get_patch_grid_bounds(
            output_shape=self.target_shape,
            device=device,
        )

        return self.patchify(
            LatentState(
                latent=initial_latent, denoise_mask=denoise_mask, positions=latent_coords, clean_latent=clean_latent
            )
        )
