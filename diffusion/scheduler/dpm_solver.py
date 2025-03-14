# Copyright 2024 NVIDIA CORPORATION & AFFILIATES
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

from diffusion.model import gaussian_diffusion as gd
from diffusion.model.dpm_solver import DPM_Solver, NoiseScheduleFlow, NoiseScheduleVP, model_wrapper


def DPMS(
    model,
    condition,
    uncondition,
    cfg_scale,
    pag_scale=1.0,
    pag_applied_layers=None,
    model_type="noise",  # or "x_start" or "v" or "score", "flow"
    noise_schedule="linear",
    guidance_type="classifier-free",
    model_kwargs=None,
    diffusion_steps=1000,
    schedule="VP",
    interval_guidance=None,
):
    if pag_applied_layers is None:
        pag_applied_layers = []
    if model_kwargs is None:
        model_kwargs = {}
    if interval_guidance is None:
        interval_guidance = [0, 1.0]
    betas = torch.tensor(gd.get_named_beta_schedule(noise_schedule, diffusion_steps))

    ## 1. Define the noise schedule.
    if schedule == "VP":
        noise_schedule = NoiseScheduleVP(schedule="discrete", betas=betas)
    elif schedule == "FLOW":
        noise_schedule = NoiseScheduleFlow(schedule="discrete_flow")

    ## 2. Convert your discrete-time `model` to the continuous-time
    ## noise prediction model. Here is an example for a diffusion model
    ## `model` with the noise prediction type ("noise") .
    model_fn = model_wrapper(
        model,
        noise_schedule,
        model_type=model_type,
        model_kwargs=model_kwargs,
        guidance_type=guidance_type,
        pag_scale=pag_scale,
        pag_applied_layers=pag_applied_layers,
        condition=condition,
        unconditional_condition=uncondition,
        guidance_scale=cfg_scale,
        interval_guidance=interval_guidance,
    )
    ## 3. Define dpm-solver and sample by multistep DPM-Solver.
    return DPM_Solver(model_fn, noise_schedule, algorithm_type="dpmsolver++")
