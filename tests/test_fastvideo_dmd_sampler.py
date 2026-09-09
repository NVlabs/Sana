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
from __future__ import annotations

import pytest
import torch

from diffusion.scheduler.fastvideo_dmd_sampler import (
    SANA_SHIFT6_DPM_GENERATOR_SIGMA_PROFILE,
    WAN_LITERAL_GENERATOR_SIGMA_PROFILE,
    FastVideoDMD4Step,
    resolve_fastvideo_dmd_generator_sigmas,
)
from inference_video_scripts.inference_sana_video import _normalize_model_state_dict, _sampler_schedule_suffix


class _ZeroVelocity(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.timesteps: list[list[float]] = []

    def forward(self, latent, timestep, _context, **_kwargs):
        self.timesteps.append(timestep.detach().cpu().tolist())
        return torch.zeros_like(latent)


@pytest.mark.parametrize(
    "profile",
    [
        WAN_LITERAL_GENERATOR_SIGMA_PROFILE,
        SANA_SHIFT6_DPM_GENERATOR_SIGMA_PROFILE,
    ],
)
def test_sampler_matches_manual_rng_reference(profile: str) -> None:
    initial = torch.arange(16, dtype=torch.float32).reshape(1, 2, 2, 2, 2)
    condition = torch.zeros(1, 1, 2, 3)
    model = _ZeroVelocity()
    rng = torch.Generator(device="cpu").manual_seed(1234)
    reference_rng = torch.Generator(device="cpu").manual_seed(1234)

    solver = FastVideoDMD4Step(model, condition, generator_sigma_profile=profile)
    actual = solver.sample(initial.clone(), steps=4, generator=rng)

    expected = initial.clone()
    physical_sigmas = (*resolve_fastvideo_dmd_generator_sigmas(profile), 0.0)
    for _sigma, next_sigma in zip(physical_sigmas[:-1], physical_sigmas[1:], strict=True):
        pred_x0 = expected
        if next_sigma > 0.0:
            noise = torch.randn(expected.shape, dtype=expected.dtype, generator=reference_rng)
            expected = (1.0 - next_sigma) * pred_x0 + next_sigma * noise
        else:
            expected = pred_x0

    torch.testing.assert_close(actual, expected, rtol=0.0, atol=5e-7)
    assert torch.equal(rng.get_state(), reference_rng.get_state())
    torch.testing.assert_close(
        torch.tensor(model.timesteps, dtype=torch.float32).flatten(),
        torch.tensor([1000.0 * sigma for sigma in physical_sigmas[:-1]], dtype=torch.float32),
        rtol=0.0,
        atol=0.0,
    )
    assert solver.last_trace == {
        "generator_sigma_profile": profile,
        "physical_sigmas": list(physical_sigmas),
        "embedding_timesteps": [int(1000.0 * sigma) for sigma in physical_sigmas[:-1]],
        "transition_noise_draws": 3,
        "flow_shift_applied": False,
        "latent_dtype": "float32",
    }


def test_sampler_preserves_bfloat16_and_requires_four_steps() -> None:
    initial = torch.randn(
        (1, 2, 2, 2, 2),
        dtype=torch.bfloat16,
        generator=torch.Generator(device="cpu").manual_seed(40),
    )
    solver = FastVideoDMD4Step(
        _ZeroVelocity(),
        torch.zeros(1, 1, 2, 3, dtype=torch.bfloat16),
    )

    output = solver.sample(initial, generator=torch.Generator(device="cpu").manual_seed(41))
    assert output.dtype == torch.bfloat16
    assert solver.last_trace is not None
    assert solver.last_trace["latent_dtype"] == "bfloat16"

    with pytest.raises(ValueError, match="exactly 4 steps"):
        solver.sample(initial, steps=3)


def test_dmd_schedule_suffix_ignores_irrelevant_flow_shift() -> None:
    suffix = _sampler_schedule_suffix(
        sampling_algo="fastvideo_dmd_4step",
        skip_type="time_uniform_flow",
        flow_shift=12.0,
        generator_sigma_profile=SANA_SHIFT6_DPM_GENERATOR_SIGMA_PROFILE,
    )
    assert suffix == "_sigmaprofile-sana_shift6_dpm"
    assert "flowshift" not in suffix

    with pytest.raises(ValueError, match="only valid"):
        _sampler_schedule_suffix(
            sampling_algo="flow_dpm-solver",
            skip_type="time_uniform_flow",
            flow_shift=12.0,
            generator_sigma_profile=SANA_SHIFT6_DPM_GENERATOR_SIGMA_PROFILE,
        )


def test_ema_export_wrapper_discards_training_state() -> None:
    tensor = torch.tensor([1.0])
    normalized = _normalize_model_state_dict(
        {
            "state_dict_ema": {"model.weight": tensor},
            "optimizer": {"state": "must not be loaded"},
        }
    )

    assert normalized == {"state_dict": {"weight": tensor}}
