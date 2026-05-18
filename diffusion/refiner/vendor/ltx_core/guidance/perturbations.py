from dataclasses import dataclass
from enum import Enum

import torch
from torch._prims_common import DeviceLikeType


class PerturbationType(Enum):
    """Types of attention perturbations for STG (Spatio-Temporal Guidance)."""

    SKIP_A2V_CROSS_ATTN = "skip_a2v_cross_attn"
    SKIP_V2A_CROSS_ATTN = "skip_v2a_cross_attn"
    SKIP_VIDEO_SELF_ATTN = "skip_video_self_attn"
    SKIP_AUDIO_SELF_ATTN = "skip_audio_self_attn"


@dataclass(frozen=True)
class Perturbation:
    """A single perturbation specifying which attention type to skip and in which blocks."""

    type: PerturbationType
    blocks: list[int] | None  # None means all blocks

    def is_perturbed(self, perturbation_type: PerturbationType, block: int) -> bool:
        if self.type != perturbation_type:
            return False

        if self.blocks is None:
            return True

        return block in self.blocks


@dataclass(frozen=True)
class PerturbationConfig:
    """Configuration holding a list of perturbations for a single sample."""

    perturbations: list[Perturbation] | None

    def is_perturbed(self, perturbation_type: PerturbationType, block: int) -> bool:
        if self.perturbations is None:
            return False

        return any(perturbation.is_perturbed(perturbation_type, block) for perturbation in self.perturbations)

    @staticmethod
    def empty() -> "PerturbationConfig":
        return PerturbationConfig([])


@dataclass(frozen=True)
class BatchedPerturbationConfig:
    """Perturbation configurations for a batch, with utilities for generating attention masks."""

    perturbations: list[PerturbationConfig]

    def mask(
        self, perturbation_type: PerturbationType, block: int, device: DeviceLikeType, dtype: torch.dtype
    ) -> torch.Tensor:
        mask = torch.ones((len(self.perturbations),), device=device, dtype=dtype)
        for batch_idx, perturbation in enumerate(self.perturbations):
            if perturbation.is_perturbed(perturbation_type, block):
                mask[batch_idx] = 0

        return mask

    def mask_like(self, perturbation_type: PerturbationType, block: int, values: torch.Tensor) -> torch.Tensor:
        mask = self.mask(perturbation_type, block, values.device, values.dtype)
        return mask.view(mask.numel(), *([1] * len(values.shape[1:])))

    def any_in_batch(self, perturbation_type: PerturbationType, block: int) -> bool:
        return any(perturbation.is_perturbed(perturbation_type, block) for perturbation in self.perturbations)

    def all_in_batch(self, perturbation_type: PerturbationType, block: int) -> bool:
        return all(perturbation.is_perturbed(perturbation_type, block) for perturbation in self.perturbations)

    @staticmethod
    def empty(batch_size: int) -> "BatchedPerturbationConfig":
        return BatchedPerturbationConfig([PerturbationConfig.empty() for _ in range(batch_size)])
