# Copyright 2026 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0
"""Four-stage stochastic sampler used by the SANA-Video 2.0 DMD preview.

The schedule and transition below intentionally mirror the deployment-aligned
rollout used during DMD training. It is not a shortened DPM-Solver schedule:
each stage predicts ``x0`` and, except for the terminal stage, re-noises it at
the next fixed sigma with a fresh draw from the caller's generator.
"""

from __future__ import annotations

from typing import Any

import torch

WAN_LITERAL_GENERATOR_SIGMA_PROFILE = "wan_literal"
SANA_SHIFT6_DPM_GENERATOR_SIGMA_PROFILE = "sana_shift6_dpm"

_GENERATOR_SIGMA_PROFILES: dict[str, tuple[float, ...]] = {
    WAN_LITERAL_GENERATOR_SIGMA_PROFILE: (1.0, 0.75, 0.5, 0.25),
    SANA_SHIFT6_DPM_GENERATOR_SIGMA_PROFILE: (
        0.9998332262,
        0.9471688271,
        0.8568977118,
        0.6663702130,
    ),
}


def resolve_fastvideo_dmd_generator_sigmas(profile: str | None) -> tuple[float, ...]:
    """Resolve a named, validated four-stage DMD generator schedule."""

    if profile not in _GENERATOR_SIGMA_PROFILES:
        choices = ", ".join(sorted(_GENERATOR_SIGMA_PROFILES))
        raise ValueError(f"Unknown FastVideo DMD generator sigma profile {profile!r}; " f"choose one of: {choices}")
    return _GENERATOR_SIGMA_PROFILES[profile]


class FastVideoDMD4Step:
    """Deployment sampler for the four-step SANA-Video DMD student."""

    def __init__(
        self,
        model: torch.nn.Module,
        condition: torch.Tensor,
        *,
        cfg_scale: float = 1.0,
        generator_sigma_profile: str = SANA_SHIFT6_DPM_GENERATOR_SIGMA_PROFILE,
        model_kwargs: dict[str, Any] | None = None,
    ) -> None:
        if float(cfg_scale) != 1.0:
            raise ValueError("FastVideoDMD4Step was trained and validated with cfg_scale=1.0")
        self.model = model
        self.condition = condition
        self.cfg_scale = float(cfg_scale)
        self.generator_sigma_profile = generator_sigma_profile
        self.sigmas = resolve_fastvideo_dmd_generator_sigmas(generator_sigma_profile)
        self.model_kwargs = dict(model_kwargs or {})
        self.last_trace: dict[str, Any] | None = None

    def sample(
        self,
        latent: torch.Tensor,
        *,
        steps: int = 4,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Run exactly four model evaluations and three stochastic transitions."""

        if int(steps) != 4:
            raise ValueError(f"FastVideoDMD4Step requires exactly 4 steps, received {steps}")
        if latent.ndim < 1 or latent.shape[0] < 1:
            raise ValueError(f"latent must include a positive batch dimension, got {latent.shape}")

        current = latent
        physical_sigmas = (*self.sigmas, 0.0)
        embedding_timesteps: list[int] = []
        transition_noise_draws = 0

        for sigma, next_sigma in zip(physical_sigmas[:-1], physical_sigmas[1:], strict=True):
            model_timestep = torch.full(
                (current.shape[0],),
                1000.0 * sigma,
                device=current.device,
                dtype=torch.float32,
            )
            embedding_timesteps.append(int(1000.0 * sigma))
            velocity = self.model(current, model_timestep, self.condition, **self.model_kwargs)
            if not isinstance(velocity, torch.Tensor):
                raise TypeError(
                    "FastVideoDMD4Step expects the model to return a tensor, " f"received {type(velocity).__name__}"
                )
            if velocity.shape != current.shape:
                raise RuntimeError(
                    "SANA velocity output shape mismatch: "
                    f"latent={tuple(current.shape)} prediction={tuple(velocity.shape)}"
                )

            pred_x0 = current - sigma * velocity
            if next_sigma > 0.0:
                noise = torch.randn(
                    current.shape,
                    device=current.device,
                    dtype=current.dtype,
                    generator=generator,
                )
                current = (1.0 - next_sigma) * pred_x0 + next_sigma * noise
                transition_noise_draws += 1
            else:
                current = pred_x0

        self.last_trace = {
            "generator_sigma_profile": self.generator_sigma_profile,
            "physical_sigmas": list(physical_sigmas),
            "embedding_timesteps": embedding_timesteps,
            "transition_noise_draws": transition_noise_draws,
            "flow_shift_applied": False,
            "latent_dtype": str(latent.dtype).removeprefix("torch."),
        }
        return current
