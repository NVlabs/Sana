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

"""SAMPLING ONLY."""

import numpy as np
import torch

from diffusion.model import gaussian_diffusion as gd
from diffusion.model.sa_solver import NoiseScheduleVP, SASolver, model_wrapper


class SASolverSampler:
    def __init__(
        self,
        model,
        noise_schedule="linear",
        diffusion_steps=1000,
        device="cpu",
    ):
        super().__init__()
        self.model = model
        self.device = device
        to_torch = lambda x: x.clone().detach().to(torch.float32).to(device)
        betas = torch.tensor(gd.get_named_beta_schedule(noise_schedule, diffusion_steps))
        alphas = 1.0 - betas
        self.register_buffer("alphas_cumprod", to_torch(np.cumprod(alphas, axis=0)))

    def register_buffer(self, name, attr):
        if type(attr) == torch.Tensor:
            if attr.device != torch.device("cuda"):
                attr = attr.to(torch.device("cuda"))
        setattr(self, name, attr)

    @torch.no_grad()
    def sample(
        self,
        S,
        batch_size,
        shape,
        conditioning=None,
        callback=None,
        normals_sequence=None,
        img_callback=None,
        quantize_x0=False,
        eta=0.0,
        mask=None,
        x0=None,
        temperature=1.0,
        noise_dropout=0.0,
        score_corrector=None,
        corrector_kwargs=None,
        verbose=True,
        x_T=None,
        log_every_t=100,
        unconditional_guidance_scale=1.0,
        unconditional_conditioning=None,
        model_kwargs={},
        # this has to come in the same format as the conditioning, # e.g. as encoded tokens, ...
        **kwargs,
    ):
        if conditioning is not None:
            if isinstance(conditioning, dict):
                cbs = conditioning[list(conditioning.keys())[0]].shape[0]
                if cbs != batch_size:
                    print(f"Warning: Got {cbs} conditionings but batch-size is {batch_size}")
            else:
                if conditioning.shape[0] != batch_size:
                    print(f"Warning: Got {conditioning.shape[0]} conditionings but batch-size is {batch_size}")

        # sampling
        C, H, W = shape
        size = (batch_size, C, H, W)

        device = self.device
        if x_T is None:
            img = torch.randn(size, device=device)
        else:
            img = x_T

        ns = NoiseScheduleVP("discrete", alphas_cumprod=self.alphas_cumprod)

        model_fn = model_wrapper(
            self.model,
            ns,
            model_type="noise",
            guidance_type="classifier-free",
            condition=conditioning,
            unconditional_condition=unconditional_conditioning,
            guidance_scale=unconditional_guidance_scale,
            model_kwargs=model_kwargs,
        )

        sasolver = SASolver(model_fn, ns, algorithm_type="data_prediction")

        tau_t = lambda t: eta if 0.2 <= t <= 0.8 else 0

        x = sasolver.sample(
            mode="few_steps",
            x=img,
            tau=tau_t,
            steps=S,
            skip_type="time",
            skip_order=1,
            predictor_order=2,
            corrector_order=2,
            pc_mode="PEC",
            return_intermediate=False,
        )

        return x.to(device), None
