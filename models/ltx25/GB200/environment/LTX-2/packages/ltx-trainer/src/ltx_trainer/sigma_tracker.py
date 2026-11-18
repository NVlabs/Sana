"""Sigma-bucketed loss tracking.
Maps each training step's per-element sigmas and losses to buckets.
Smoothing is left to wandb's UI.
"""

import bisect
from collections import defaultdict


class SigmaBucketTracker:
    """Map per-element sigma values to named buckets for per-bucket loss logging.
    By default, partitions [0, 1] into four equal-width buckets.
    Custom boundaries can be provided for non-uniform bucketing.
    Each call to update() receives per-element sigmas and losses (both [B,]),
    buckets each element, and computes the mean loss per bucket. This gives
    accurate per-sigma loss tracking even for batch_size > 1.
    """

    def __init__(
        self,
        bucket_boundaries: list[float] | None = None,
    ) -> None:
        if bucket_boundaries is None:
            bucket_boundaries = [0.0, 0.25, 0.5, 0.75, 1.0]
        if len(bucket_boundaries) < 2:
            raise ValueError("bucket_boundaries must have at least 2 elements")
        if any(bucket_boundaries[i] >= bucket_boundaries[i + 1] for i in range(len(bucket_boundaries) - 1)):
            raise ValueError("bucket_boundaries must be strictly increasing")
        self._boundaries = list(bucket_boundaries)
        self._num_buckets = len(bucket_boundaries) - 1
        self._bucket_labels = [
            f"{bucket_boundaries[i]:.2f}-{bucket_boundaries[i + 1]:.2f}" for i in range(self._num_buckets)
        ]
        self._last_metrics: dict[str, float] = {}

    def _get_bucket_index(self, sigma: float) -> int:
        """Map sigma value to bucket index."""
        idx = bisect.bisect_right(self._boundaries, sigma) - 1
        return max(0, min(idx, self._num_buckets - 1))

    def update(self, sigmas: list[float], losses: list[float]) -> None:
        """Record per-element losses into their sigma buckets.
        Args:
            sigmas: Per-element sigma values, one per batch element.
            losses: Per-element losses, one per batch element.
        """
        if not sigmas:
            self._last_metrics = {}
            return
        bucket_losses: dict[int, list[float]] = defaultdict(list)
        for sigma, loss in zip(sigmas, losses, strict=True):
            bucket_losses[self._get_bucket_index(sigma)].append(loss)
        self._last_metrics = {self._bucket_labels[b]: sum(vals) / len(vals) for b, vals in bucket_losses.items()}

    def get_metrics(self, prefix: str = "train") -> dict[str, float]:
        """Return the mean loss for each bucket hit on the last update.
        Wandb handles smoothing in the UI.
        """
        return {f"{prefix}/loss_sigma_{label}": loss for label, loss in self._last_metrics.items()}
