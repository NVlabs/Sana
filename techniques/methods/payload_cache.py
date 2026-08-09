# Copyright 2025 SGLang authors
#
# PayloadCache -- model-agnostic schedule/key policy for caching intermediate
# attention or block-layer payloads across denoising steps. Model adapters decide
# which concrete tensors are valid payloads; this class keeps the durable
# algorithmic knobs out of model-specific runtime glue.

from __future__ import annotations

from techniques.registry import register_technique
from techniques.schedule import at_steps
from techniques.technique import (
    Capability,
    Phase,
    Seam,
    Technique,
)


def _threshold_pair(value, default: tuple[int, int]) -> tuple[int, int]:
    if value is None:
        return default
    if isinstance(value, str):
        parts = [part.strip() for part in value.split(",") if part.strip()]
    else:
        parts = list(value) if isinstance(value, (list, tuple)) else []
    if len(parts) < 2:
        return default
    try:
        return int(parts[0]), int(parts[1])
    except (TypeError, ValueError):
        return default


def _normalize_mlp_config(config) -> dict[int, dict[str, object]]:
    out: dict[int, dict[str, object]] = {}
    if not isinstance(config, dict):
        return out
    for raw_timestep, raw_item in config.items():
        if not isinstance(raw_item, dict):
            continue
        try:
            timestep = int(raw_timestep)
            skip_count = int(raw_item.get("skip_count", 0))
        except (TypeError, ValueError):
            continue
        raw_blocks = raw_item.get("block", [])
        if isinstance(raw_blocks, int):
            blocks = [raw_blocks]
        else:
            try:
                blocks = [int(block) for block in raw_blocks]
            except (TypeError, ValueError):
                blocks = []
        out[timestep] = {"block": blocks, "skip_count": skip_count}
    return out


class PABBroadcastController:
    """Model-agnostic VideoSys PAB decision controller.

    This mirrors the public PAB manager control surface: attention broadcast is
    gated by attention kind, timestep thresholds, and ``count % range``; MLP
    broadcast is keyed by start timestep, block index, and skip_count. The
    cached tensor payloads remain adapter-owned.
    """

    def __init__(
        self,
        *,
        steps: int | None = None,
        cross_broadcast: bool = False,
        cross_threshold=None,
        cross_range: int | None = None,
        spatial_broadcast: bool = False,
        spatial_threshold=None,
        spatial_range: int | None = None,
        temporal_broadcast: bool = False,
        temporal_threshold=None,
        temporal_range: int | None = None,
        mlp_broadcast: bool = False,
        mlp_spatial_broadcast_config=None,
        mlp_temporal_broadcast_config=None,
    ) -> None:
        self.steps = max(1, int(steps or 1))
        self.cross_broadcast = bool(cross_broadcast)
        self.cross_threshold = _threshold_pair(cross_threshold, (0, 0))
        self.cross_range = int(cross_range or 1)
        self.spatial_broadcast = bool(spatial_broadcast)
        self.spatial_threshold = _threshold_pair(spatial_threshold, (0, 0))
        self.spatial_range = int(spatial_range or 1)
        self.temporal_broadcast = bool(temporal_broadcast)
        self.temporal_threshold = _threshold_pair(temporal_threshold, (0, 0))
        self.temporal_range = int(temporal_range or 1)
        self.mlp_broadcast = bool(mlp_broadcast)
        self.mlp_spatial_broadcast_config = _normalize_mlp_config(
            mlp_spatial_broadcast_config
        )
        self.mlp_temporal_broadcast_config = _normalize_mlp_config(
            mlp_temporal_broadcast_config
        )

    def set_steps(self, steps: int | None) -> None:
        if steps is not None:
            self.steps = max(1, int(steps))

    def attention_decision(
        self, kind: str, timestep: int | None, count: int
    ) -> tuple[bool, int]:
        kind = str(kind)
        enabled = bool(getattr(self, f"{kind}_broadcast", False))
        threshold = getattr(self, f"{kind}_threshold", (0, 0))
        range_value = max(1, int(getattr(self, f"{kind}_range", 1) or 1))
        flag = False
        if timestep is not None:
            t = int(timestep)
            flag = (
                enabled
                and count % range_value != 0
                and threshold[0] < t < threshold[1]
            )
        return bool(flag), (int(count) + 1) % self.steps

    @staticmethod
    def _is_t_in_skip_config(
        all_timesteps: list[int], timestep: int, config: dict[int, dict[str, object]]
    ) -> tuple[bool, list[int] | None]:
        for key, item in config.items():
            if key not in all_timesteps:
                continue
            index = all_timesteps.index(key)
            skip_count = int(item.get("skip_count", 0))
            end_index = index + skip_count
            if end_index >= len(all_timesteps):
                continue
            skip_window = all_timesteps[index : end_index + 1]
            if timestep in skip_window:
                return True, [all_timesteps[index], all_timesteps[end_index]]
        return False, None

    def mlp_decision(
        self,
        *,
        timestep: int | None,
        count: int,
        block_idx: int,
        all_timesteps: list[int],
        is_temporal: bool = False,
    ) -> tuple[bool, int | None, bool, list[int] | None]:
        if not self.mlp_broadcast or timestep is None:
            return False, None, False, None

        config = (
            self.mlp_temporal_broadcast_config
            if is_temporal
            else self.mlp_spatial_broadcast_config
        )
        t = int(timestep)
        in_skip_config, skip_range = self._is_t_in_skip_config(
            all_timesteps, t, config
        )
        next_flag = False
        if t in config and int(block_idx) in config[t].get("block", []):
            next_flag = True
            return False, int(count) + 1, next_flag, skip_range
        if (
            in_skip_config
            and skip_range is not None
            and int(block_idx) in config[skip_range[0]].get("block", [])
        ):
            return True, 0, next_flag, skip_range
        return False, int(count), next_flag, skip_range

    def config_dict(self) -> dict[str, object]:
        return {
            "steps": self.steps,
            "cross_broadcast": self.cross_broadcast,
            "cross_threshold": list(self.cross_threshold),
            "cross_range": self.cross_range,
            "spatial_broadcast": self.spatial_broadcast,
            "spatial_threshold": list(self.spatial_threshold),
            "spatial_range": self.spatial_range,
            "temporal_broadcast": self.temporal_broadcast,
            "temporal_threshold": list(self.temporal_threshold),
            "temporal_range": self.temporal_range,
            "mlp_broadcast": self.mlp_broadcast,
            "mlp_spatial_broadcast_config": self.mlp_spatial_broadcast_config,
            "mlp_temporal_broadcast_config": self.mlp_temporal_broadcast_config,
        }


@register_technique("payload_cache")
class PayloadCache(Technique):
    """Scheduled intermediate-payload replay policy.

    `scope` names the model-adapter payload boundary. The current transfeat use
    `attention_broadcast` for attention outputs and `block_layer_feature` for a
    full block/layer hidden-state payload. The adapter must provide shape/key
    guards and dense fallback; this policy only records when replay is allowed.
    """

    name = "payload_cache"
    phase = Phase.IN_BLOCKS
    reads = frozenset({Seam.ATTENTION, Seam.HIDDEN_STATES, Seam.RESIDUAL_CACHE})
    writes = frozenset({Seam.ATTENTION, Seam.HIDDEN_STATES, Seam.RESIDUAL_CACHE})
    required_capabilities = frozenset({Capability.SUPPORTS_STEP_CACHE})

    def __init__(
        self,
        scope: str,
        skip: str = "",
        clone_on_hit: bool = False,
        mode: str = "scheduled",
        attention_kind: str = "cross",
        steps: int | None = None,
        cross_broadcast: bool = False,
        cross_threshold=None,
        cross_range: int | None = None,
        spatial_broadcast: bool = False,
        spatial_threshold=None,
        spatial_range: int | None = None,
        temporal_broadcast: bool = False,
        temporal_threshold=None,
        temporal_range: int | None = None,
        mlp_broadcast: bool = False,
        mlp_spatial_broadcast_config=None,
        mlp_temporal_broadcast_config=None,
        enabled=True,
    ):
        self.scope = str(scope)
        self.skip = str(skip)
        self.clone_on_hit = bool(clone_on_hit)
        self.mode = str(mode or "scheduled")
        self.attention_kind = str(attention_kind or "cross")
        self.pab = (
            PABBroadcastController(
                steps=steps,
                cross_broadcast=cross_broadcast,
                cross_threshold=cross_threshold,
                cross_range=cross_range,
                spatial_broadcast=spatial_broadcast,
                spatial_threshold=spatial_threshold,
                spatial_range=spatial_range,
                temporal_broadcast=temporal_broadcast,
                temporal_threshold=temporal_threshold,
                temporal_range=temporal_range,
                mlp_broadcast=mlp_broadcast,
                mlp_spatial_broadcast_config=mlp_spatial_broadcast_config,
                mlp_temporal_broadcast_config=mlp_temporal_broadcast_config,
            )
            if self.mode == "pab"
            else None
        )
        schedule = (
            enabled
            if self.mode == "pab"
            else at_steps(self.skip, True, False)
            if self.skip
            else enabled
        )
        super().__init__(enabled=schedule)

    def cache_policy(self) -> dict[str, object]:
        policy = {
            "scope": self.scope,
            "skip": self.skip,
            "clone_on_hit": self.clone_on_hit,
            "mode": self.mode,
            "attention_kind": self.attention_kind,
        }
        if self.pab is not None:
            policy["pab"] = self.pab.config_dict()
        return policy
