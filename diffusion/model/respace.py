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

import math
import random
from typing import Optional, Tuple, Union

import numpy as np
import torch as th

from diffusion.model import gaussian_diffusion as gd
from diffusion.model.gaussian_diffusion import GaussianDiffusion


def space_timesteps(num_timesteps, section_counts):
    """
    Create a list of timesteps to use from an original diffusion process,
    given the number of timesteps we want to take from equally-sized portions
    of the original process.
    For example, if there's 300 timesteps and the section counts are [10,15,20]
    then the first 100 timesteps are strided to be 10 timesteps, the second 100
    are strided to be 15 timesteps, and the final 100 are strided to be 20.
    If the stride is a string starting with "ddim", then the fixed striding
    from the DDIM paper is used, and only one section is allowed.
    :param num_timesteps: the number of diffusion steps in the original
                          process to divide up.
    :param section_counts: either a list of numbers, or a string containing
                           comma-separated numbers, indicating the step count
                           per section. As a special case, use "ddimN" where N
                           is a number of steps to use the striding from the
                           DDIM paper.
    :return: a set of diffusion steps from the original process to use.
    """
    if isinstance(section_counts, str):
        if section_counts.startswith("ddim"):
            desired_count = int(section_counts[len("ddim") :])
            for i in range(1, num_timesteps):
                if len(range(0, num_timesteps, i)) == desired_count:
                    return set(range(0, num_timesteps, i))
            raise ValueError(f"cannot create exactly {num_timesteps} steps with an integer stride")
        section_counts = [int(x) for x in section_counts.split(",")]
    size_per = num_timesteps // len(section_counts)
    extra = num_timesteps % len(section_counts)
    start_idx = 0
    all_steps = []
    for i, section_count in enumerate(section_counts):
        size = size_per + (1 if i < extra else 0)
        if size < section_count:
            raise ValueError(f"cannot divide section of {size} steps into {section_count}")
        if section_count <= 1:
            frac_stride = 1
        else:
            frac_stride = (size - 1) / (section_count - 1)
        cur_idx = 0.0
        taken_steps = []
        for _ in range(section_count):
            taken_steps.append(start_idx + round(cur_idx))
            cur_idx += frac_stride
        all_steps += taken_steps
        start_idx += size
    return set(all_steps)


def truncated_normal_icdf_sample(n, mu, sigma, a, b, device, dtype):
    """
    Exact inverse-CDF sampling of N(mu, sigma^2) truncated to [a, b]
    a,b in z-space. No mass at boundaries.
    """
    std_normal = th.distributions.normal.Normal(0.0, 1.0)
    Phi_a = std_normal.cdf((a - mu) / sigma)
    Phi_b = std_normal.cdf((b - mu) / sigma)
    r = th.rand(n, device=device, dtype=dtype)  # ~ U(0,1)
    q = Phi_a + r * (Phi_b - Phi_a)  # ~ U(Phi_a, Phi_b)
    z = mu + sigma * std_normal.icdf(q)  # inverse-CDF back to z
    return z


def stretched_logit_normal(n, mu, sigma, p_low, p_high, device, dtype):
    # print(f"stretched_logit_normal: p_low={p_low}, p_high={p_high}, mu={mu}, sigma={sigma}")
    std_normal = th.distributions.normal.Normal(0.0, 1.0)
    # draw z from a truncated normal between the desired quantiles
    z_lo = mu + sigma * std_normal.icdf(th.tensor(p_low, device=device, dtype=dtype))
    z_hi = mu + sigma * std_normal.icdf(th.tensor(p_high, device=device, dtype=dtype))
    z = truncated_normal_icdf_sample(n, mu, sigma, z_lo, z_hi, device, dtype)

    # map to u-space and linearly stretch [u_lo, u_hi] -> [0,1]
    u_raw = th.nn.functional.sigmoid(z)
    u_lo = th.nn.functional.sigmoid(z_lo)
    u_hi = th.nn.functional.sigmoid(z_hi)
    eps = th.finfo(dtype).eps
    u = (u_raw - u_lo) / (u_hi - u_lo + eps)
    return u


def compute_density_for_timestep_sampling(
    weighting_scheme: str,
    batch_size: int,
    logit_mean: float = None,
    logit_std: float = None,
    mode_scale: float = None,
    p_low: float = None,
    p_high: float = None,
):
    """Compute the density for sampling the timesteps when doing SD3 training.

    Courtesy: This was contributed by Rafie Walker in https://github.com/huggingface/diffusers/pull/8528.

    SD3 paper reference: https://arxiv.org/abs/2403.03206v1.
    """
    if weighting_scheme == "logit_normal":
        # See 3.1 in the SD3 paper ($rf/lognorm(0.00,1.00)$).
        u = th.normal(mean=logit_mean, std=logit_std, size=(batch_size,), device="cpu")
        u = th.nn.functional.sigmoid(u)
    elif weighting_scheme == "stretched_logit_normal":
        assert p_low is not None and p_high is not None, "p_low and p_high must be provided for stretched_logit_normal"
        # See 3.1 in the SD3 paper ($rf/lognorm(0.00,1.00)$).
        u = stretched_logit_normal(
            n=batch_size,
            mu=logit_mean,
            sigma=logit_std,
            p_low=p_low if p_low is not None else 0.0,
            p_high=p_high if p_high is not None else 1.0,
            device="cpu",
            dtype=th.float32,
        )
    elif weighting_scheme == "mode":
        u = th.rand(size=(batch_size,), device="cpu")
        u = 1 - u - mode_scale * (th.cos(math.pi * u / 2) ** 2 - 1 + u)
    elif weighting_scheme == "logit_normal_trigflow":
        sigma = th.randn(batch_size, device="cpu")
        sigma = (sigma * logit_std + logit_mean).exp()
        u = th.atan(sigma / 0.5)  # TODO: 0.5 should be a hyper-parameter
    else:
        u = th.rand(size=(batch_size,), device="cpu")
    return u


class IncrementalTimesteps:
    """
    Log-space DP + batched sampling in Pyth.
    - F: number of frames
    - T: number of timesteps
    - device/dtype configurable
    """

    def __init__(self, F: int | list | None, T: int, device: Optional[th.device] = None, dtype: th.dtype = th.float64):
        if isinstance(F, list):
            F = len(F)
        elif isinstance(F, int):
            F = F
        elif F is None:
            F = 1
        else:
            raise ValueError(f"Invalid type for F: {type(F)}")

        self.F = F
        self.T = T
        self.device = device if device is not None else th.device("cpu")
        self.dtype = dtype

        # ----- build log_mat_s (forward) -----
        log_s = th.full((T, F), float("-inf"), device=self.device, dtype=self.dtype)
        log_s[:, F - 1] = 0.0  # log(1)
        for f in range(F - 2, -1, -1):
            log_s[T - 1, f] = 0.0
            # DP: log(A+B) = logsumexp(logA, logB)
            # fill upward (t from T-2 down to 0)
            for t in range(T - 2, -1, -1):
                log_s[t, f] = th.logaddexp(log_s[t + 1, f], log_s[t, f + 1])
        self.log_mat_s = log_s

        # ----- build log_mat_e (backward) -----
        log_e = th.full((T, F), float("-inf"), device=self.device, dtype=self.dtype)
        log_e[:, 0] = 0.0
        for f in range(1, F):
            log_e[0, f] = 0.0
            for t in range(1, T):
                log_e[t, f] = th.logaddexp(log_e[t - 1, f], log_e[t, f - 1])
        self.log_mat_e = log_e

    # ---------- helpers ----------
    def _masked_multinomial_from_logweights(self, logw_col: th.Tensor, starts: th.Tensor, ends: th.Tensor) -> th.Tensor:
        """
        Vectorized categorical sampling from a single column of log-weights.
        logw_col: [T]
        starts, ends: [B], slice is [start, end)
        Returns: indices [B] in range [0, T), respecting per-batch slices.
        """
        B = starts.shape[0]
        T = logw_col.shape[0]

        # Expand to [B, T]
        logits = logw_col.expand(B, T).clone()

        # Mask out everything outside [start, end) by setting -inf
        arangeT = th.arange(T, device=self.device).unsqueeze(0).expand(B, T)  # [B, T]
        mask = (arangeT >= starts.unsqueeze(1)) & (arangeT < ends.unsqueeze(1))
        logits[~mask] = float("-inf")

        # Softmax -> probs; multinomial supports batched sampling row-wise
        probs = th.softmax(logits, dim=1)
        # multinomial expects non-negative and finite probs; mask guarantees at least one valid slot
        idx = th.multinomial(probs, num_samples=1).squeeze(1)  # [B]
        return idx

    # ---------- public APIs ----------
    @th.no_grad()
    def sample_step_sequence_batch(
        self, batch_size: int, start_preT: Optional[Union[int, th.Tensor]] = None
    ) -> th.Tensor:
        """
        Forward-only monotonic sequences (non-decreasing in t across frames).
        Returns [B, F] int64 tensor.
        """
        B = batch_size
        ts = th.zeros((B, self.F), device=self.device, dtype=th.long)

        if start_preT is None:
            preT = th.zeros(B, device=self.device, dtype=th.long)
        else:
            preT = th.as_tensor(start_preT, device=self.device, dtype=th.long)
            if preT.ndim == 0:
                preT = preT.expand(B)

        for f in range(self.F):
            starts = preT
            ends = th.full((B,), self.T, device=self.device, dtype=th.long)
            # sample from column f using log_mat_s[:, f]
            idx = self._masked_multinomial_from_logweights(self.log_mat_s[:, f], starts, ends)
            ts[:, f] = idx
            preT = idx
        return ts

    @th.no_grad()
    def sample(
        self,
        batch_size: int,
        curf: Optional[Union[int, th.Tensor]] = None,
        cur_timestep: Optional[Union[int, th.Tensor]] = None,
    ) -> th.Tensor:
        """
        Middle-anchor sampler (both sides), batched.
        - curf:
            * None: random curf per sample
            * int: same anchor frame for all
            * tensor [B]: per-sample anchor frame
        - cur_timestep:
            * None: anchor timestep is sampled uniformly in [0, T)
            * int: same anchor for all
            * tensor [B]: per-sample anchor
        Returns [B, F] int64 tensor.
        """
        B = batch_size
        ts = th.zeros((B, self.F), device=self.device, dtype=th.long)

        # resolve curf
        if curf is None:
            curfs = th.randint(0, self.F, (B,), device=self.device)
        else:
            curfs = th.as_tensor(curf, device=self.device, dtype=th.long)
            if curfs.ndim == 0:
                curfs = curfs.expand(B)

        # resolve anchor timestep
        if cur_timestep is None:
            anchors = th.randint(0, self.T, (B,), device=self.device)
        else:
            anchors = th.as_tensor(cur_timestep, device=self.device, dtype=th.long)
            if anchors.ndim == 0:
                anchors = anchors.expand(B)

        # set anchors
        ts[th.arange(B, device=self.device), curfs] = anchors

        # left side (non-increasing): use log_mat_e
        # iterate frames; vectorize across batch
        for f in range(self.F - 2, -1, -1):
            # Which samples need this f on the left of their curf?
            need = curfs > f
            if need.any():
                # hi = ts[:, f+1] + 1
                hi = ts[:, f + 1] + 1
                starts = th.zeros_like(hi)
                ends = hi.clamp_(max=self.T)  # safe guard
                idx = self._masked_multinomial_from_logweights(self.log_mat_e[:, f], starts[need], ends[need])
                ts[need, f] = idx

        # right side (non-decreasing): use log_mat_s
        for f in range(1, self.F):
            # Which samples need this f on the right of their curf?
            need = curfs < f
            if need.any():
                lo = ts[:, f - 1]
                starts = lo.clamp_(min=0)
                ends = th.full_like(starts, self.T)
                idx = self._masked_multinomial_from_logweights(self.log_mat_s[:, f], starts[need], ends[need])
                ts[need, f] = idx

        return ts


def process_timesteps(
    weighting_scheme: str | None,
    train_sampling_steps: int,
    size: Tuple,
    device: th.device,
    **kwargs,
):

    same_timestep_prob = kwargs.get("same_timestep_prob", 0.0)
    timesteps = th.randint(0, train_sampling_steps, size, device=device).long()
    if weighting_scheme in ["logit_normal", "stretched_logit_normal", "mode"]:
        bs = np.cumprod(size)[-1]  # frame-aware noise
        # adapting from diffusers.training_utils
        u = compute_density_for_timestep_sampling(
            weighting_scheme=weighting_scheme,
            batch_size=bs,
            logit_mean=kwargs.get("logit_mean", 0),
            logit_std=kwargs.get("logit_std", 1),
            p_low=kwargs.get("p_low", None),
            p_high=kwargs.get("p_high", None),
            mode_scale=None,  # not used
        )
        timesteps = (u * train_sampling_steps).long().to(device)
        timesteps = timesteps.reshape(size)
    else:
        raise ValueError(f"Invalid weighting scheme: {weighting_scheme}")

    if kwargs.get("chunk_index", None) is not None:
        if random.random() < same_timestep_prob:
            timesteps = timesteps[..., None, None].repeat(1, 1, kwargs.get("num_frames", 1))
            return timesteps

        # do chunk causal sampling
        # sample bxlen(chunk_size) timesteps
        chunk_index = kwargs.get("chunk_index")[:]  # start index of each chunk, copy the list
        chunk_index.append(kwargs.get("num_frames", 1))
        chunk_sizes = th.diff(th.tensor(chunk_index)).tolist()  # [f1, f2-f1, f3-f2, ...]
        num_chunks = len(chunk_sizes)
        strategy = kwargs.get("chunk_sampling_strategy", "uniform")
        if strategy == "uniform":
            u = compute_density_for_timestep_sampling(
                weighting_scheme=weighting_scheme,
                batch_size=size[0] * num_chunks,
                logit_mean=kwargs.get("logit_mean", 0),
                logit_std=kwargs.get("logit_std", 1),
                p_low=kwargs.get("p_low", None),
                p_high=kwargs.get("p_high", None),
                mode_scale=None,  # not used
            )
            timesteps = (u * train_sampling_steps).long().to(device)
            timesteps = timesteps.reshape(size[0], 1, num_chunks)  # b,1,num_chunks
            # repeat each value in timesteps chunk_sizes times
            frame_timesteps = [
                timesteps[:, :, i : i + 1].repeat_interleave(chunk_sizes[i], dim=-1) for i in range(num_chunks)
            ]
            timesteps = th.cat(frame_timesteps, dim=-1)  # b,1,num_frames
            timesteps = timesteps.long()
        elif strategy == "incremental":
            u = compute_density_for_timestep_sampling(
                weighting_scheme=weighting_scheme,
                batch_size=size[0],
                logit_mean=kwargs.get("logit_mean", 0),
                logit_std=kwargs.get("logit_std", 1),
                p_low=kwargs.get("p_low", None),
                p_high=kwargs.get("p_high", None),
                mode_scale=None,  # not used
            )
            base_timesteps = (u * train_sampling_steps).long().to(device)  # b
            if kwargs.get("time_sampler", None) is not None:
                timesteps_list = kwargs.get("time_sampler").sample(
                    size[0], curf=None, cur_timestep=base_timesteps
                )  # [b, num_chunks]
                timesteps_list = [timesteps_list[:, i] for i in range(num_chunks)]  # [b] * num_chunks
            else:
                timesteps_list = [base_timesteps]  # b
                # incremental sample timesteps for each chunk
                for i in range(num_chunks - 1):
                    # sample B timesteps smaller than timesteps_list[-1]
                    max_timestep = timesteps_list[-1]  # b
                    # Create uniform samples and scale by max_timestep
                    uniform_samples = th.rand(size[0], device=device)  # b
                    next_timesteps = (uniform_samples * max_timestep.float()).long()
                    timesteps_list.append(next_timesteps)
                # reverse timesteps_list, so that the first chunk has the smallest timesteps
                timesteps_list = timesteps_list[::-1]  # [b] * num_chunks
            # Now construct the final timesteps tensor
            frame_timesteps = []
            for i, chunk_timesteps in enumerate(timesteps_list):
                # Repeat each chunk's timesteps for its frames
                repeated = chunk_timesteps.unsqueeze(1).unsqueeze(2).repeat(1, 1, chunk_sizes[i])  # b,1,chunk_size
                frame_timesteps.append(repeated)

            timesteps = th.cat(frame_timesteps, dim=-1)  # b,1,num_frames

    if kwargs.get("do_i2v", False):
        if len(timesteps.shape) < 3:
            timesteps = timesteps[..., None, None].repeat(1, 1, kwargs.get("num_frames", 1))  # B,1,F
        # sample a timestep for the first frame, smaller noise
        random_timestep = th.randint(0, train_sampling_steps, (size[0], 1), device=device).long() * kwargs.get(
            "noise_multiplier", 0
        )
        timesteps[:, :, 0] = random_timestep.long()

    return timesteps


class SpacedDiffusion(GaussianDiffusion):
    """
    A diffusion process which can skip steps in a base diffusion process.
    :param use_timesteps: a collection (sequence or set) of timesteps from the
                          original diffusion process to retain.
    :param kwargs: the kwargs to create the base diffusion process.
    """

    def __init__(self, use_timesteps, **kwargs):
        self.use_timesteps = set(use_timesteps)
        self.timestep_map = []
        self.original_num_steps = len(kwargs["betas"])

        flow_shift = kwargs.pop("flow_shift")
        diffusion_steps = kwargs.pop("diffusion_steps")
        base_diffusion = GaussianDiffusion(**kwargs)  # pylint: disable=missing-kwoa
        last_alpha_cumprod = 1.0
        if kwargs.get("model_mean_type", False) == gd.ModelMeanType.FLOW_VELOCITY:
            new_sigmas = flow_shift * base_diffusion.sigmas / (1 + (flow_shift - 1) * base_diffusion.sigmas)
            self.timestep_map = new_sigmas * diffusion_steps
            # self.timestep_map = list(self.use_timesteps)
            kwargs["sigmas"] = np.array(new_sigmas)
            super().__init__(**kwargs)
        else:
            new_betas = []
            for i, alpha_cumprod in enumerate(base_diffusion.alphas_cumprod):
                if i in self.use_timesteps:
                    new_betas.append(1 - alpha_cumprod / last_alpha_cumprod)
                    last_alpha_cumprod = alpha_cumprod
                    self.timestep_map.append(i)
            kwargs["betas"] = np.array(new_betas)
            super().__init__(**kwargs)

    def p_mean_variance(self, model, *args, **kwargs):  # pylint: disable=signature-differs
        return super().p_mean_variance(self._wrap_model(model), *args, **kwargs)

    def training_losses(self, model, *args, **kwargs):  # pylint: disable=signature-differs
        return super().training_losses(self._wrap_model(model), *args, **kwargs)

    def training_losses_diffusers(self, model, *args, **kwargs):  # pylint: disable=signature-differs
        return super().training_losses_diffusers(self._wrap_model(model), *args, **kwargs)

    def condition_mean(self, cond_fn, *args, **kwargs):
        return super().condition_mean(self._wrap_model(cond_fn), *args, **kwargs)

    def condition_score(self, cond_fn, *args, **kwargs):
        return super().condition_score(self._wrap_model(cond_fn), *args, **kwargs)

    def _wrap_model(self, model):
        if isinstance(model, _WrappedModel):
            return model
        return _WrappedModel(model, self.timestep_map, self.original_num_steps)

    def _scale_timesteps(self, t):
        # Scaling is done by the wrapped model.
        return t


class _WrappedModel:
    def __init__(self, model, timestep_map, original_num_steps):
        self.model = model
        self.timestep_map = timestep_map
        # self.rescale_timesteps = rescale_timesteps
        self.original_num_steps = original_num_steps

    def __call__(self, x, timestep, **kwargs):
        if self.timestep_map is None:
            return self.model(x, timestep=timestep, **kwargs)
        if callable(self.timestep_map):
            new_ts = self.timestep_map(timestep)
        else:
            map_tensor = th.tensor(self.timestep_map, device=timestep.device, dtype=timestep.dtype)
            new_ts = map_tensor[timestep]
        # if self.rescale_timesteps:
        #     new_ts = new_ts.float() * (1000.0 / self.original_num_steps)
        return self.model(x, timestep=new_ts, **kwargs)
