from __future__ import annotations

from typing import Any

import torch
from pydantic import BaseModel, ConfigDict


class ConfigFingerprint(BaseModel):
    optimizer_type: str
    scheduler_type: str
    training_mode: str
    lora_rank: int | None = None


class RngStates(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    torch_state: torch.Tensor
    cuda_state: torch.Tensor | None = None


class TrainingState(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    global_step: int
    config_fingerprint: ConfigFingerprint
    rng_states: RngStates
    lr_scheduler_state_dict: dict[str, Any] | None = None
    optimizer_state_dict: dict[str, Any] | None = None
    wandb_run_id: str | None = None

    def to_save_dict(self) -> dict[str, Any]:
        """Build dict suitable for torch.save -- recurses BaseModel sub-models, passes tensors/dicts through."""

        def _convert(value: object) -> object:
            if isinstance(value, BaseModel):
                return {k: _convert(v) for k, v in value if v is not None}
            return value

        return {k: _convert(v) for k, v in self if v is not None}

    @classmethod
    def from_save_dict(cls, data: dict[str, Any]) -> TrainingState:
        """Construct from torch.load output with Pydantic validation."""
        return cls(
            global_step=data["global_step"],
            config_fingerprint=ConfigFingerprint(**data["config_fingerprint"]),
            rng_states=RngStates(**data["rng_states"]),
            lr_scheduler_state_dict=data.get("lr_scheduler_state_dict"),
            optimizer_state_dict=data.get("optimizer_state_dict"),
            wandb_run_id=data.get("wandb_run_id"),
        )
