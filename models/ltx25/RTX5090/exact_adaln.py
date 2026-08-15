"""Exact Stage-2 AdaLN tables for the NVFP4 LTX-2.5 profile."""

from __future__ import annotations

import gc
from collections import Counter
from pathlib import Path
from typing import Any

import torch

TABLE_FORMAT = "ltx25_exact_adaln_v1"
MODULE_PREFIXES = {
    "video_base": "model.diffusion_model.adaln_single",
    "video_cross": "model.diffusion_model.av_ca_video_scale_shift_adaln_single",
}


class ExactScheduleAdaLN(torch.nn.Module):
    def __init__(
        self,
        name: str,
        projected: torch.Tensor,
        embedded: torch.Tensor,
        scaled_timesteps: torch.Tensor,
        token_count: int,
        call_counts: Counter[tuple[str, int]],
    ) -> None:
        super().__init__()
        self.name = name
        self.token_count = token_count
        self.call_counts = call_counts
        self.register_buffer("projected", projected.contiguous(), persistent=False)
        self.register_buffer("embedded", embedded.contiguous(), persistent=False)
        self.register_buffer("scaled_timesteps", scaled_timesteps.float().contiguous(), persistent=False)

    def forward(
        self,
        timestep: torch.Tensor,
        hidden_dtype: torch.dtype | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        flattened = timestep.flatten()
        if flattened.numel() != self.token_count:
            raise RuntimeError(
                f"{self.name} table expects {self.token_count} tokens, got {flattened.numel()}"
            )
        if not torch.equal(flattened, flattened[:1].expand_as(flattened)):
            raise RuntimeError(f"{self.name} exact table requires a uniform timestep")
        matches = torch.nonzero(flattened[0] == self.scaled_timesteps, as_tuple=False).flatten()
        if matches.numel() != 1:
            raise RuntimeError(f"{self.name} has no table row for timestep {flattened[0].item()}")
        step = int(matches.item())
        if hidden_dtype is not None and hidden_dtype != self.projected.dtype:
            raise RuntimeError(f"{self.name} table dtype does not match {hidden_dtype}")
        self.call_counts[(self.name, step)] += 1
        return self.projected[step], self.embedded[step]


def _velocity_model(transformer: Any) -> Any:
    current = transformer
    while not hasattr(current, "adaln_single"):
        if hasattr(current, "_model"):
            current = current._model
        elif hasattr(current, "velocity_model"):
            current = current.velocity_model
        else:
            raise RuntimeError(f"cannot locate the LTX-2.5 velocity model under {type(current)!r}")
    return current


class LTX25ExactAdaLN:
    def __init__(self, table_path: Path, checkpoint: Path) -> None:
        self.payload = torch.load(table_path, map_location="cpu", weights_only=True)
        if self.payload.get("format") != TABLE_FORMAT:
            raise ValueError("unsupported Exact AdaLN table")
        if self.payload.get("source_size") != checkpoint.stat().st_size:
            raise ValueError("Exact AdaLN table was built from a different checkpoint")
        if self.payload.get("token_count", 0) <= 0:
            raise ValueError("invalid Exact AdaLN token count")
        if set(self.payload.get("modules", {})) != set(MODULE_PREFIXES):
            raise ValueError("Exact AdaLN table is incomplete")
        self.call_counts: Counter[tuple[str, int]] = Counter()
        self.binding: dict[str, Any] | None = None

    def install(self, transformer: Any) -> None:
        model = _velocity_model(transformer)
        if self.binding is not None:
            raise RuntimeError("Exact AdaLN is already installed")
        preprocessor = model.video_args_preprocessor
        originals = {
            "video_base": model.adaln_single,
            "video_cross": model.av_ca_video_scale_shift_adaln_single,
        }
        device = originals["video_base"].linear.weight.device
        dtype = originals["video_base"].linear.weight.dtype
        replacements = {}
        for name, original in originals.items():
            row = self.payload["modules"][name]
            if tuple(original.linear.weight.shape) != tuple(row["linear_weight_shape"]):
                raise RuntimeError(f"{name} shape changed since table generation")
            replacements[name] = ExactScheduleAdaLN(
                name,
                row["projected"].to(device=device, dtype=dtype),
                row["embedded"].to(device=device, dtype=dtype),
                self.payload["scaled_timesteps"].to(device=device),
                int(self.payload["token_count"]),
                self.call_counts,
            )

        originals["video_base"].to("cpu")
        originals["video_cross"].to("cpu")
        model.adaln_single = replacements["video_base"]
        model.av_ca_video_scale_shift_adaln_single = replacements["video_cross"]
        preprocessor.simple_preprocessor.adaln = replacements["video_base"]
        preprocessor.cross_scale_shift_adaln = replacements["video_cross"]
        self.binding = {
            "model": model,
            "preprocessor": preprocessor,
            "originals": originals,
            "replacements": replacements,
        }
        gc.collect()
        torch.cuda.empty_cache()

    def uninstall(self, transformer: Any) -> None:
        model = _velocity_model(transformer)
        if self.binding is None or self.binding["model"] is not model:
            raise RuntimeError("Exact AdaLN is not installed on this transformer")
        preprocessor = self.binding["preprocessor"]
        originals = self.binding["originals"]
        model.adaln_single = originals["video_base"]
        model.av_ca_video_scale_shift_adaln_single = originals["video_cross"]
        preprocessor.simple_preprocessor.adaln = originals["video_base"]
        preprocessor.cross_scale_shift_adaln = originals["video_cross"]
        self.binding = None
        gc.collect()
        torch.cuda.empty_cache()

    def stats(self) -> dict[str, object]:
        steps = len(self.payload["sigmas"])
        return {
            "token_count": int(self.payload["token_count"]),
            "calls": {
                name: [self.call_counts[(name, step)] for step in range(steps)]
                for name in MODULE_PREFIXES
            },
        }
