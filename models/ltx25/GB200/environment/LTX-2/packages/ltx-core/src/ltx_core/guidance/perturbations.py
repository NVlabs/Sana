from dataclasses import dataclass
from enum import IntEnum

import torch
from torch._prims_common import DeviceLikeType


class PerturbationType(IntEnum):
    """Types of attention perturbations for STG (Spatio-Temporal Guidance).
    The integer value is the row index into ``BatchedPerturbationConfig._block_masks`` dim 0.
    """

    SKIP_VIDEO_SELF_ATTN = 0
    SKIP_AUDIO_SELF_ATTN = 1
    SKIP_A2V_CROSS_ATTN = 2
    SKIP_V2A_CROSS_ATTN = 3


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


class BatchedPerturbationConfig:
    """Per-block attention keep-masks for a batch, built once from a list of per-sample configs.
    Construction materializes ``_block_masks`` -- a ``(len(PerturbationType), num_blocks, B)`` tensor
    (1 = keep, 0 = perturbed) whose dim-0 row index is the ``PerturbationType`` value -- from the
    perturbation structure. The per-sample config list is NOT retained: every consumer reads the
    tensor (``mask`` indexes it; ``any_in_batch`` / ``all_in_batch`` read the host mirror).
    The host build (reading the Python structure) happens here, in ``__init__``, so it MUST be run
    eagerly OUTSIDE any ``torch.compile`` / CUDA-graph-capture region. The compiled block then reads
    perturbation purely as the runtime ``_block_masks`` tensor and never recompiles per config.
    """

    _block_masks: torch.Tensor  # keep-mask on the compute device, indexed [PerturbationType, block, sample]
    # Host mirror so any_in_batch / all_in_batch stay sync-free and graph-break-free. Present for
    # configs that may hit the eager skip shortcuts; None for compiled-only configs built via
    # ``from_masks`` (the compiled processor reads only ``_block_masks``).
    _block_masks_cpu: torch.Tensor | None

    def __init__(
        self,
        perturbations: list[PerturbationConfig],
        num_blocks: int,
        device: DeviceLikeType | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        keep = [
            [
                [not pc.is_perturbed(PerturbationType(direction), block) for pc in perturbations]
                for block in range(num_blocks)
            ]
            for direction in range(len(PerturbationType))
        ]
        self._block_masks_cpu = torch.tensor(keep, dtype=dtype, device="cpu")
        self._block_masks = self._block_masks_cpu if device is None else self._block_masks_cpu.to(device)

    @classmethod
    def from_masks(
        cls, block_masks: torch.Tensor, block_masks_cpu: torch.Tensor | None = None
    ) -> "BatchedPerturbationConfig":
        """Construct from prebuilt mask tensors (e.g. a batch-dim slice), bypassing the host build.
        ``block_masks_cpu`` is only consumed by ``any_in_batch`` / ``all_in_batch`` (the eager
        processor's skip shortcuts); pass it when the result may take that path. The compiled
        processor reads only ``_block_masks``, so callers on that path may omit the mirror.
        """
        obj = cls.__new__(cls)
        obj._block_masks = block_masks
        obj._block_masks_cpu = block_masks_cpu
        return obj

    @property
    def block_masks(self) -> torch.Tensor:
        """Keep-mask on the compute device (1 = keep, 0 = perturbed), indexed [type, block, sample]."""
        return self._block_masks

    @property
    def block_masks_cpu(self) -> torch.Tensor | None:
        """Host mirror of the keep-mask, or None for compiled-only configs (see ``from_masks``)."""
        return self._block_masks_cpu

    def batch_slice(self, start: int, end: int) -> "BatchedPerturbationConfig":
        """A view over samples ``[start:end]`` of the batch, by slicing the mask tensors.
        Slicing (never rebuilding) keeps the host mask build outside any compiled / capture region.
        """
        cpu_mask = self._block_masks_cpu[:, :, start:end] if self._block_masks_cpu is not None else None
        return BatchedPerturbationConfig.from_masks(self._block_masks[:, :, start:end], cpu_mask)

    def mask(self, perturbation_type: PerturbationType, block: int) -> torch.Tensor:
        """This block's ``(B, 1, 1)`` keep-mask for one perturbation type, as an OWNED tensor.
        A ``clone`` (not a view into ``_block_masks``) so the masks attached to a block
        (e.g. self- and cross-attention) don't alias the same storage -- aliased graph inputs are
        fragile under ``torch.compile``.
        """
        return self._block_masks[perturbation_type, block].reshape(-1, 1, 1).clone()

    def any_in_batch(self, perturbation_type: PerturbationType, block: int) -> bool:
        assert self._block_masks_cpu is not None, "host mirror required by the skip-shortcut processor path"
        return bool((self._block_masks_cpu[perturbation_type, block] == 0).any())

    def all_in_batch(self, perturbation_type: PerturbationType, block: int) -> bool:
        assert self._block_masks_cpu is not None, "host mirror required by the skip-shortcut processor path"
        return bool((self._block_masks_cpu[perturbation_type, block] == 0).all())

    @staticmethod
    def empty(
        batch_size: int,
        num_blocks: int,
        device: DeviceLikeType | None = None,
        dtype: torch.dtype | None = None,
    ) -> "BatchedPerturbationConfig":
        return BatchedPerturbationConfig(
            [PerturbationConfig.empty() for _ in range(batch_size)], num_blocks, device, dtype
        )
