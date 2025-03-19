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

# Modified from OpenAI's diffusion repos
#     GLIDE: https://github.com/openai/glide-text2im/blob/main/glide_text2im/gaussian_diffusion.py
#     ADM:   https://github.com/openai/guided-diffusion/blob/main/guided_diffusion
#     IDDPM: https://github.com/openai/improved-diffusion/blob/main/improved_diffusion/gaussian_diffusion.py

from diffusion.model import gaussian_diffusion as gd
from diffusion.model.respace import SpacedDiffusion, space_timesteps


def Scheduler(
    timestep_respacing,
    noise_schedule="linear",
    use_kl=False,
    sigma_small=False,
    predict_xstart=False,
    predict_flow_v=False,
    learn_sigma=True,
    pred_sigma=True,
    rescale_learned_sigmas=False,
    diffusion_steps=1000,
    snr=False,
    return_startx=False,
    flow_shift=1.0,
):
    betas = gd.get_named_beta_schedule(noise_schedule, diffusion_steps)
    if use_kl:
        loss_type = gd.LossType.RESCALED_KL
    elif rescale_learned_sigmas:
        loss_type = gd.LossType.RESCALED_MSE
    else:
        loss_type = gd.LossType.MSE
    if timestep_respacing is None or timestep_respacing == "":
        timestep_respacing = [diffusion_steps]
    if predict_xstart:
        model_mean_type = gd.ModelMeanType.START_X
    elif predict_flow_v:
        model_mean_type = gd.ModelMeanType.FLOW_VELOCITY
    else:
        model_mean_type = gd.ModelMeanType.EPSILON
    return SpacedDiffusion(
        use_timesteps=space_timesteps(diffusion_steps, timestep_respacing),
        betas=betas,
        model_mean_type=model_mean_type,
        model_var_type=(
            (
                (gd.ModelVarType.FIXED_LARGE if not sigma_small else gd.ModelVarType.FIXED_SMALL)
                if not learn_sigma
                else gd.ModelVarType.LEARNED_RANGE
            )
            if pred_sigma
            else None
        ),
        loss_type=loss_type,
        snr=snr,
        return_startx=return_startx,
        # rescale_timesteps=rescale_timesteps,
        flow="flow" in noise_schedule,
        flow_shift=flow_shift,
        diffusion_steps=diffusion_steps,
    )
