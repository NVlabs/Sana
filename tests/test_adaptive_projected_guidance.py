# Copyright 2026 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0
import torch

from diffusion.guiders.adaptive_projected_guidance import AdaptiveProjectedGuidance
from diffusion.scheduler.dpm_solver import DPMS


def _guider(momentum=-0.5):
    return AdaptiveProjectedGuidance(
        guidance_scale=6.0,
        adaptive_projected_guidance_momentum=momentum,
        adaptive_projected_guidance_rescale=27,
        eta=1.0,
        mode="hw",
    )


def test_reset_clears_momentum():
    torch.manual_seed(0)
    cond, uncond = torch.randn(1, 4, 2, 8, 8), torch.randn(1, 4, 2, 8, 8)
    guider = _guider()
    first = guider(cond, uncond)[0]
    guider.reset()
    torch.testing.assert_close(guider(cond, uncond)[0], first)


def test_momentum_none_disables_momentum():
    torch.manual_seed(0)
    cond, uncond = torch.randn(1, 4, 2, 8, 8), torch.randn(1, 4, 2, 8, 8)
    guider = _guider(momentum=None)
    assert guider.momentum_buffer is None
    first = guider(cond, uncond)[0]
    torch.testing.assert_close(guider(cond, uncond)[0], first)


def test_dpms_reused_guider_gives_same_sample(monkeypatch):
    monkeypatch.setenv("DPM_TQDM", "True")

    def model(x, t, y, **kwargs):
        return 0.5 * x + y.mean(dim=(1, 2)).view(-1, 1, 1, 1, 1)

    def sample(guider, z, cond, uncond):
        solver = DPMS(
            model,
            condition=cond,
            uncondition=uncond,
            cfg_scale=6.0,
            model_type="flow",
            guidance_type="adaptive_projected_guidance",
            model_kwargs={},
            schedule="FLOW",
            apg=guider,
        )
        return solver.sample(
            z.clone(), steps=10, order=2, skip_type="time_uniform_flow", method="multistep", flow_shift=3.0
        )

    torch.manual_seed(0)
    z = torch.randn(1, 4, 2, 8, 8)
    cond, uncond = torch.randn(1, 3, 8), torch.zeros(1, 3, 8)
    guider = _guider()
    first = sample(guider, z, cond, uncond)
    torch.testing.assert_close(sample(guider, z, cond, uncond), first)
