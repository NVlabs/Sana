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

import math

import torch
import torch.nn.functional as F
from tqdm import tqdm


class NoiseScheduleVP:
    def __init__(
        self,
        schedule="discrete",
        betas=None,
        alphas_cumprod=None,
        continuous_beta_0=0.1,
        continuous_beta_1=20.0,
        dtype=torch.float32,
    ):
        """Thanks to DPM-Solver for their code base"""
        r"""Create a wrapper class for the forward SDE (VP type).
        ***
        Update: We support discrete-time diffusion models by implementing a picewise linear interpolation for log_alpha_t.
                We recommend to use schedule='discrete' for the discrete-time diffusion models, especially for high-resolution images.
        ***
        The forward SDE ensures that the condition distribution q_{t|0}(x_t | x_0) = N ( alpha_t * x_0, sigma_t^2 * I ).
        We further define lambda_t = log(alpha_t) - log(sigma_t), which is the half-logSNR (described in the DPM-Solver paper).
        Therefore, we implement the functions for computing alpha_t, sigma_t and lambda_t. For t in [0, T], we have:
            log_alpha_t = self.marginal_log_mean_coeff(t)
            sigma_t = self.marginal_std(t)
            lambda_t = self.marginal_lambda(t)
        Moreover, as lambda(t) is an invertible function, we also support its inverse function:
            t = self.inverse_lambda(lambda_t)
        ===============================================================
        We support both discrete-time DPMs (trained on n = 0, 1, ..., N-1) and continuous-time DPMs (trained on t in [t_0, T]).
        1. For discrete-time DPMs:
            For discrete-time DPMs trained on n = 0, 1, ..., N-1, we convert the discrete steps to continuous time steps by:
                t_i = (i + 1) / N
            e.g. for N = 1000, we have t_0 = 1e-3 and T = t_{N-1} = 1.
            We solve the corresponding diffusion ODE from time T = 1 to time t_0 = 1e-3.
            Args:
                betas: A `torch.Tensor`. The beta array for the discrete-time DPM. (See the original DDPM paper for details)
                alphas_cumprod: A `torch.Tensor`. The cumprod alphas for the discrete-time DPM. (See the original DDPM paper for details)
            Note that we always have alphas_cumprod = cumprod(1 - betas). Therefore, we only need to set one of `betas` and `alphas_cumprod`.
            **Important**:  Please pay special attention for the args for `alphas_cumprod`:
                The `alphas_cumprod` is the \hat{alpha_n} arrays in the notations of DDPM. Specifically, DDPMs assume that
                    q_{t_n | 0}(x_{t_n} | x_0) = N ( \sqrt{\hat{alpha_n}} * x_0, (1 - \hat{alpha_n}) * I ).
                Therefore, the notation \hat{alpha_n} is different from the notation alpha_t in DPM-Solver. In fact, we have
                    alpha_{t_n} = \sqrt{\hat{alpha_n}},
                and
                    log(alpha_{t_n}) = 0.5 * log(\hat{alpha_n}).
        2. For continuous-time DPMs:
            We support two types of VPSDEs: linear (DDPM) and cosine (improved-DDPM). The hyperparameters for the noise
            schedule are the default settings in DDPM and improved-DDPM:
            Args:
                beta_min: A `float` number. The smallest beta for the linear schedule.
                beta_max: A `float` number. The largest beta for the linear schedule.
                cosine_s: A `float` number. The hyperparameter in the cosine schedule.
                cosine_beta_max: A `float` number. The hyperparameter in the cosine schedule.
                T: A `float` number. The ending time of the forward process.
        ===============================================================
        Args:
            schedule: A `str`. The noise schedule of the forward SDE. 'discrete' for discrete-time DPMs,
                    'linear' or 'cosine' for continuous-time DPMs.
        Returns:
            A wrapper object of the forward SDE (VP type).

        ===============================================================
        Example:
        # For discrete-time DPMs, given betas (the beta array for n = 0, 1, ..., N - 1):
        >>> ns = NoiseScheduleVP('discrete', betas=betas)
        # For discrete-time DPMs, given alphas_cumprod (the \hat{alpha_n} array for n = 0, 1, ..., N - 1):
        >>> ns = NoiseScheduleVP('discrete', alphas_cumprod=alphas_cumprod)
        # For continuous-time DPMs (VPSDE), linear schedule:
        >>> ns = NoiseScheduleVP('linear', continuous_beta_0=0.1, continuous_beta_1=20.)
        """

        if schedule not in ["discrete", "linear", "cosine"]:
            raise ValueError(
                "Unsupported noise schedule {}. The schedule needs to be 'discrete' or 'linear' or 'cosine'".format(
                    schedule
                )
            )

        self.schedule = schedule
        if schedule == "discrete":
            if betas is not None:
                log_alphas = 0.5 * torch.log(1 - betas).cumsum(dim=0)
            else:
                assert alphas_cumprod is not None
                log_alphas = 0.5 * torch.log(alphas_cumprod)
            self.total_N = len(log_alphas)
            self.T = 1.0
            self.t_array = torch.linspace(0.0, 1.0, self.total_N + 1)[1:].reshape((1, -1)).to(dtype=dtype)
            self.log_alpha_array = log_alphas.reshape(
                (
                    1,
                    -1,
                )
            ).to(dtype=dtype)
        else:
            self.total_N = 1000
            self.beta_0 = continuous_beta_0
            self.beta_1 = continuous_beta_1
            self.cosine_s = 0.008
            self.cosine_beta_max = 999.0
            self.cosine_t_max = (
                math.atan(self.cosine_beta_max * (1.0 + self.cosine_s) / math.pi)
                * 2.0
                * (1.0 + self.cosine_s)
                / math.pi
                - self.cosine_s
            )
            self.cosine_log_alpha_0 = math.log(math.cos(self.cosine_s / (1.0 + self.cosine_s) * math.pi / 2.0))
            self.schedule = schedule
            if schedule == "cosine":
                # For the cosine schedule, T = 1 will have numerical issues. So we manually set the ending time T.
                # Note that T = 0.9946 may be not the optimal setting. However, we find it works well.
                self.T = 0.9946
            else:
                self.T = 1.0

    def marginal_log_mean_coeff(self, t):
        """
        Compute log(alpha_t) of a given continuous-time label t in [0, T].
        """
        if self.schedule == "discrete":
            return interpolate_fn(
                t.reshape((-1, 1)), self.t_array.to(t.device), self.log_alpha_array.to(t.device)
            ).reshape(-1)
        elif self.schedule == "linear":
            return -0.25 * t**2 * (self.beta_1 - self.beta_0) - 0.5 * t * self.beta_0
        elif self.schedule == "cosine":
            log_alpha_fn = lambda s: torch.log(torch.cos((s + self.cosine_s) / (1.0 + self.cosine_s) * math.pi / 2.0))
            log_alpha_t = log_alpha_fn(t) - self.cosine_log_alpha_0
            return log_alpha_t

    def marginal_alpha(self, t):
        """
        Compute alpha_t of a given continuous-time label t in [0, T].
        """
        return torch.exp(self.marginal_log_mean_coeff(t))

    def marginal_std(self, t):
        """
        Compute sigma_t of a given continuous-time label t in [0, T].
        """
        return torch.sqrt(1.0 - torch.exp(2.0 * self.marginal_log_mean_coeff(t)))

    def marginal_lambda(self, t):
        """
        Compute lambda_t = log(alpha_t) - log(sigma_t) of a given continuous-time label t in [0, T].
        """
        log_mean_coeff = self.marginal_log_mean_coeff(t)
        log_std = 0.5 * torch.log(1.0 - torch.exp(2.0 * log_mean_coeff))
        return log_mean_coeff - log_std

    def inverse_lambda(self, lamb):
        """
        Compute the continuous-time label t in [0, T] of a given half-logSNR lambda_t.
        """
        if self.schedule == "linear":
            tmp = 2.0 * (self.beta_1 - self.beta_0) * torch.logaddexp(-2.0 * lamb, torch.zeros((1,)).to(lamb))
            Delta = self.beta_0**2 + tmp
            return tmp / (torch.sqrt(Delta) + self.beta_0) / (self.beta_1 - self.beta_0)
        elif self.schedule == "discrete":
            log_alpha = -0.5 * torch.logaddexp(torch.zeros((1,)).to(lamb.device), -2.0 * lamb)
            t = interpolate_fn(
                log_alpha.reshape((-1, 1)),
                torch.flip(self.log_alpha_array.to(lamb.device), [1]),
                torch.flip(self.t_array.to(lamb.device), [1]),
            )
            return t.reshape((-1,))
        else:
            log_alpha = -0.5 * torch.logaddexp(-2.0 * lamb, torch.zeros((1,)).to(lamb))
            t_fn = (
                lambda log_alpha_t: torch.arccos(torch.exp(log_alpha_t + self.cosine_log_alpha_0))
                * 2.0
                * (1.0 + self.cosine_s)
                / math.pi
                - self.cosine_s
            )
            t = t_fn(log_alpha)
            return t

    def edm_sigma(self, t):
        return self.marginal_std(t) / self.marginal_alpha(t)

    def edm_inverse_sigma(self, edmsigma):
        alpha = 1 / (edmsigma**2 + 1).sqrt()
        sigma = alpha * edmsigma
        lambda_t = torch.log(alpha / sigma)
        t = self.inverse_lambda(lambda_t)
        return t


def model_wrapper(
    model,
    noise_schedule,
    model_type="noise",
    model_kwargs={},
    guidance_type="uncond",
    condition=None,
    unconditional_condition=None,
    guidance_scale=1.0,
    classifier_fn=None,
    classifier_kwargs={},
):
    """Thanks to DPM-Solver for their code base"""
    """Create a wrapper function for the noise prediction model.
    SA-Solver needs to solve the continuous-time diffusion SDEs. For DPMs trained on discrete-time labels, we need to
    firstly wrap the model function to a noise prediction model that accepts the continuous time as the input.
    We support four types of the diffusion model by setting `model_type`:
        1. "noise": noise prediction model. (Trained by predicting noise).
        2. "x_start": data prediction model. (Trained by predicting the data x_0 at time 0).
        3. "v": velocity prediction model. (Trained by predicting the velocity).
            The "v" prediction is derivation detailed in Appendix D of [1], and is used in Imagen-Video [2].
            [1] Salimans, Tim, and Jonathan Ho. "Progressive distillation for fast sampling of diffusion models."
                arXiv preprint arXiv:2202.00512 (2022).
            [2] Ho, Jonathan, et al. "Imagen Video: High Definition Video Generation with Diffusion Models."
                arXiv preprint arXiv:2210.02303 (2022).

        4. "score": marginal score function. (Trained by denoising score matching).
            Note that the score function and the noise prediction model follows a simple relationship:
            ```
                noise(x_t, t) = -sigma_t * score(x_t, t)
            ```
    We support three types of guided sampling by DPMs by setting `guidance_type`:
        1. "uncond": unconditional sampling by DPMs.
            The input `model` has the following format:
            ``
                model(x, t_input, **model_kwargs) -> noise | x_start | v | score
            ``
        2. "classifier": classifier guidance sampling [3] by DPMs and another classifier.
            The input `model` has the following format:
            ``
                model(x, t_input, **model_kwargs) -> noise | x_start | v | score
            ``
            The input `classifier_fn` has the following format:
            ``
                classifier_fn(x, t_input, cond, **classifier_kwargs) -> logits(x, t_input, cond)
            ``
            [3] P. Dhariwal and A. Q. Nichol, "Diffusion models beat GANs on image synthesis,"
                in Advances in Neural Information Processing Systems, vol. 34, 2021, pp. 8780-8794.
        3. "classifier-free": classifier-free guidance sampling by conditional DPMs.
            The input `model` has the following format:
            ``
                model(x, t_input, cond, **model_kwargs) -> noise | x_start | v | score
            ``
            And if cond == `unconditional_condition`, the model output is the unconditional DPM output.
            [4] Ho, Jonathan, and Tim Salimans. "Classifier-free diffusion guidance."
                arXiv preprint arXiv:2207.12598 (2022).

    The `t_input` is the time label of the model, which may be discrete-time labels (i.e. 0 to 999)
    or continuous-time labels (i.e. epsilon to T).
    We wrap the model function to accept only `x` and `t_continuous` as inputs, and outputs the predicted noise:
    ``
        def model_fn(x, t_continuous) -> noise:
            t_input = get_model_input_time(t_continuous)
            return noise_pred(model, x, t_input, **model_kwargs)
    ``
    where `t_continuous` is the continuous time labels (i.e. epsilon to T). And we use `model_fn` for SA-Solver.
    ===============================================================
    Args:
        model: A diffusion model with the corresponding format described above.
        noise_schedule: A noise schedule object, such as NoiseScheduleVP.
        model_type: A `str`. The parameterization type of the diffusion model.
                    "noise" or "x_start" or "v" or "score".
        model_kwargs: A `dict`. A dict for the other inputs of the model function.
        guidance_type: A `str`. The type of the guidance for sampling.
                    "uncond" or "classifier" or "classifier-free".
        condition: A pytorch tensor. The condition for the guided sampling.
                    Only used for "classifier" or "classifier-free" guidance type.
        unconditional_condition: A pytorch tensor. The condition for the unconditional sampling.
                    Only used for "classifier-free" guidance type.
        guidance_scale: A `float`. The scale for the guided sampling.
        classifier_fn: A classifier function. Only used for the classifier guidance.
        classifier_kwargs: A `dict`. A dict for the other inputs of the classifier function.
    Returns:
        A noise prediction model that accepts the noised data and the continuous time as the inputs.
    """

    def get_model_input_time(t_continuous):
        """
        Convert the continuous-time `t_continuous` (in [epsilon, T]) to the model input time.
        For discrete-time DPMs, we convert `t_continuous` in [1 / N, 1] to `t_input` in [0, 1000 * (N - 1) / N].
        For continuous-time DPMs, we just use `t_continuous`.
        """
        if noise_schedule.schedule == "discrete":
            return (t_continuous - 1.0 / noise_schedule.total_N) * 1000.0
        else:
            return t_continuous

    def noise_pred_fn(x, t_continuous, cond=None):
        t_input = get_model_input_time(t_continuous)
        if cond is None:
            output = model(x, t_input, **model_kwargs)
        else:
            output = model(x, t_input, cond, **model_kwargs)
        if model_type == "noise":
            return output
        elif model_type == "x_start":
            alpha_t, sigma_t = noise_schedule.marginal_alpha(t_continuous), noise_schedule.marginal_std(t_continuous)
            return (x - alpha_t[0] * output) / sigma_t[0]
        elif model_type == "v":
            alpha_t, sigma_t = noise_schedule.marginal_alpha(t_continuous), noise_schedule.marginal_std(t_continuous)
            return alpha_t[0] * output + sigma_t[0] * x
        elif model_type == "score":
            sigma_t = noise_schedule.marginal_std(t_continuous)
            return -sigma_t[0] * output

    def cond_grad_fn(x, t_input):
        """
        Compute the gradient of the classifier, i.e. nabla_{x} log p_t(cond | x_t).
        """
        with torch.enable_grad():
            x_in = x.detach().requires_grad_(True)
            log_prob = classifier_fn(x_in, t_input, condition, **classifier_kwargs)
            return torch.autograd.grad(log_prob.sum(), x_in)[0]

    def model_fn(x, t_continuous):
        """
        The noise predicition model function that is used for DPM-Solver.
        """
        if guidance_type == "uncond":
            return noise_pred_fn(x, t_continuous)
        elif guidance_type == "classifier":
            assert classifier_fn is not None
            t_input = get_model_input_time(t_continuous)
            cond_grad = cond_grad_fn(x, t_input)
            sigma_t = noise_schedule.marginal_std(t_continuous)
            noise = noise_pred_fn(x, t_continuous)
            return noise - guidance_scale * sigma_t * cond_grad
        elif guidance_type == "classifier-free":
            if guidance_scale == 1.0 or unconditional_condition is None:
                return noise_pred_fn(x, t_continuous, cond=condition)
            else:
                x_in = torch.cat([x] * 2)
                t_in = torch.cat([t_continuous] * 2)
                c_in = torch.cat([unconditional_condition, condition])
                noise_uncond, noise = noise_pred_fn(x_in, t_in, cond=c_in).chunk(2)
                return noise_uncond + guidance_scale * (noise - noise_uncond)

    assert model_type in ["noise", "x_start", "v", "score"]
    assert guidance_type in ["uncond", "classifier", "classifier-free"]
    return model_fn


class SASolver:
    def __init__(
        self,
        model_fn,
        noise_schedule,
        algorithm_type="data_prediction",
        correcting_x0_fn=None,
        correcting_xt_fn=None,
        thresholding_max_val=1.0,
        dynamic_thresholding_ratio=0.995,
    ):
        """
        Construct a SA-Solver
        The default value for algorithm_type is "data_prediction" and we recommend not to change it to
        "noise_prediction". For details, please see Appendix A.2.4 in SA-Solver paper https://arxiv.org/pdf/2309.05019.pdf
        """

        self.model = lambda x, t: model_fn(x, t.expand(x.shape[0]))
        self.noise_schedule = noise_schedule
        assert algorithm_type in ["data_prediction", "noise_prediction"]

        if correcting_x0_fn == "dynamic_thresholding":
            self.correcting_x0_fn = self.dynamic_thresholding_fn
        else:
            self.correcting_x0_fn = correcting_x0_fn

        self.correcting_xt_fn = correcting_xt_fn
        self.dynamic_thresholding_ratio = dynamic_thresholding_ratio
        self.thresholding_max_val = thresholding_max_val

        self.predict_x0 = algorithm_type == "data_prediction"

        self.sigma_min = float(self.noise_schedule.edm_sigma(torch.tensor([1e-3])))
        self.sigma_max = float(self.noise_schedule.edm_sigma(torch.tensor([1])))

    def dynamic_thresholding_fn(self, x0, t=None):
        """
        The dynamic thresholding method.
        """
        dims = x0.dim()
        p = self.dynamic_thresholding_ratio
        s = torch.quantile(torch.abs(x0).reshape((x0.shape[0], -1)), p, dim=1)
        s = expand_dims(torch.maximum(s, self.thresholding_max_val * torch.ones_like(s).to(s.device)), dims)
        x0 = torch.clamp(x0, -s, s) / s
        return x0

    def noise_prediction_fn(self, x, t):
        """
        Return the noise prediction model.
        """
        return self.model(x, t)

    def data_prediction_fn(self, x, t):
        """
        Return the data prediction model (with corrector).
        """
        noise = self.noise_prediction_fn(x, t)
        alpha_t, sigma_t = self.noise_schedule.marginal_alpha(t), self.noise_schedule.marginal_std(t)
        x0 = (x - sigma_t * noise) / alpha_t
        if self.correcting_x0_fn is not None:
            x0 = self.correcting_x0_fn(x0)
        return x0

    def model_fn(self, x, t):
        """
        Convert the model to the noise prediction model or the data prediction model.
        """

        if self.predict_x0:
            return self.data_prediction_fn(x, t)
        else:
            return self.noise_prediction_fn(x, t)

    def get_time_steps(self, skip_type, t_T, t_0, N, order, device):
        """Compute the intermediate time steps for sampling."""
        if skip_type == "logSNR":
            lambda_T = self.noise_schedule.marginal_lambda(torch.tensor(t_T).to(device))
            lambda_0 = self.noise_schedule.marginal_lambda(torch.tensor(t_0).to(device))
            logSNR_steps = lambda_T + torch.linspace(
                torch.tensor(0.0).cpu().item(), (lambda_0 - lambda_T).cpu().item() ** (1.0 / order), N + 1
            ).pow(order).to(device)
            return self.noise_schedule.inverse_lambda(logSNR_steps)
        elif skip_type == "time":
            t = torch.linspace(t_T ** (1.0 / order), t_0 ** (1.0 / order), N + 1).pow(order).to(device)
            return t
        elif skip_type == "karras":
            sigma_min = max(0.002, self.sigma_min)
            sigma_max = min(80, self.sigma_max)
            sigma_steps = torch.linspace(sigma_max ** (1.0 / 7), sigma_min ** (1.0 / 7), N + 1).pow(7).to(device)
            t = self.noise_schedule.edm_inverse_sigma(sigma_steps)
            return t
        else:
            raise ValueError(f"Unsupported skip_type {skip_type}, need to be 'logSNR' or 'time' or 'karras'")

    def denoise_to_zero_fn(self, x, s):
        """
        Denoise at the final step, which is equivalent to solve the ODE from lambda_s to infty by first-order discretization.
        """
        return self.data_prediction_fn(x, s)

    def get_coefficients_exponential_negative(self, order, interval_start, interval_end):
        """
        Calculate the integral of exp(-x) * x^order dx from interval_start to interval_end
        For calculating the coefficient of gradient terms after the lagrange interpolation,
        see Eq.(15) and Eq.(18) in SA-Solver paper https://arxiv.org/pdf/2309.05019.pdf
        For noise_prediction formula.
        """
        assert order in [0, 1, 2, 3], "order is only supported for 0, 1, 2 and 3"

        if order == 0:
            return torch.exp(-interval_end) * (torch.exp(interval_end - interval_start) - 1)
        elif order == 1:
            return torch.exp(-interval_end) * (
                (interval_start + 1) * torch.exp(interval_end - interval_start) - (interval_end + 1)
            )
        elif order == 2:
            return torch.exp(-interval_end) * (
                (interval_start**2 + 2 * interval_start + 2) * torch.exp(interval_end - interval_start)
                - (interval_end**2 + 2 * interval_end + 2)
            )
        elif order == 3:
            return torch.exp(-interval_end) * (
                (interval_start**3 + 3 * interval_start**2 + 6 * interval_start + 6)
                * torch.exp(interval_end - interval_start)
                - (interval_end**3 + 3 * interval_end**2 + 6 * interval_end + 6)
            )

    def get_coefficients_exponential_positive(self, order, interval_start, interval_end, tau):
        """
        Calculate the integral of exp(x(1+tau^2)) * x^order dx from interval_start to interval_end
        For calculating the coefficient of gradient terms after the lagrange interpolation,
        see Eq.(15) and Eq.(18) in SA-Solver paper https://arxiv.org/pdf/2309.05019.pdf
        For data_prediction formula.
        """
        assert order in [0, 1, 2, 3], "order is only supported for 0, 1, 2 and 3"

        # after change of variable(cov)
        interval_end_cov = (1 + tau**2) * interval_end
        interval_start_cov = (1 + tau**2) * interval_start

        if order == 0:
            return (
                torch.exp(interval_end_cov) * (1 - torch.exp(-(interval_end_cov - interval_start_cov))) / (1 + tau**2)
            )
        elif order == 1:
            return (
                torch.exp(interval_end_cov)
                * (
                    (interval_end_cov - 1)
                    - (interval_start_cov - 1) * torch.exp(-(interval_end_cov - interval_start_cov))
                )
                / ((1 + tau**2) ** 2)
            )
        elif order == 2:
            return (
                torch.exp(interval_end_cov)
                * (
                    (interval_end_cov**2 - 2 * interval_end_cov + 2)
                    - (interval_start_cov**2 - 2 * interval_start_cov + 2)
                    * torch.exp(-(interval_end_cov - interval_start_cov))
                )
                / ((1 + tau**2) ** 3)
            )
        elif order == 3:
            return (
                torch.exp(interval_end_cov)
                * (
                    (interval_end_cov**3 - 3 * interval_end_cov**2 + 6 * interval_end_cov - 6)
                    - (interval_start_cov**3 - 3 * interval_start_cov**2 + 6 * interval_start_cov - 6)
                    * torch.exp(-(interval_end_cov - interval_start_cov))
                )
                / ((1 + tau**2) ** 4)
            )

    def lagrange_polynomial_coefficient(self, order, lambda_list):
        """
        Calculate the coefficient of lagrange polynomial
        For lagrange interpolation
        """
        assert order in [0, 1, 2, 3]
        assert order == len(lambda_list) - 1
        if order == 0:
            return [[1]]
        elif order == 1:
            return [
                [1 / (lambda_list[0] - lambda_list[1]), -lambda_list[1] / (lambda_list[0] - lambda_list[1])],
                [1 / (lambda_list[1] - lambda_list[0]), -lambda_list[0] / (lambda_list[1] - lambda_list[0])],
            ]
        elif order == 2:
            denominator1 = (lambda_list[0] - lambda_list[1]) * (lambda_list[0] - lambda_list[2])
            denominator2 = (lambda_list[1] - lambda_list[0]) * (lambda_list[1] - lambda_list[2])
            denominator3 = (lambda_list[2] - lambda_list[0]) * (lambda_list[2] - lambda_list[1])
            return [
                [
                    1 / denominator1,
                    (-lambda_list[1] - lambda_list[2]) / denominator1,
                    lambda_list[1] * lambda_list[2] / denominator1,
                ],
                [
                    1 / denominator2,
                    (-lambda_list[0] - lambda_list[2]) / denominator2,
                    lambda_list[0] * lambda_list[2] / denominator2,
                ],
                [
                    1 / denominator3,
                    (-lambda_list[0] - lambda_list[1]) / denominator3,
                    lambda_list[0] * lambda_list[1] / denominator3,
                ],
            ]
        elif order == 3:
            denominator1 = (
                (lambda_list[0] - lambda_list[1])
                * (lambda_list[0] - lambda_list[2])
                * (lambda_list[0] - lambda_list[3])
            )
            denominator2 = (
                (lambda_list[1] - lambda_list[0])
                * (lambda_list[1] - lambda_list[2])
                * (lambda_list[1] - lambda_list[3])
            )
            denominator3 = (
                (lambda_list[2] - lambda_list[0])
                * (lambda_list[2] - lambda_list[1])
                * (lambda_list[2] - lambda_list[3])
            )
            denominator4 = (
                (lambda_list[3] - lambda_list[0])
                * (lambda_list[3] - lambda_list[1])
                * (lambda_list[3] - lambda_list[2])
            )
            return [
                [
                    1 / denominator1,
                    (-lambda_list[1] - lambda_list[2] - lambda_list[3]) / denominator1,
                    (
                        lambda_list[1] * lambda_list[2]
                        + lambda_list[1] * lambda_list[3]
                        + lambda_list[2] * lambda_list[3]
                    )
                    / denominator1,
                    (-lambda_list[1] * lambda_list[2] * lambda_list[3]) / denominator1,
                ],
                [
                    1 / denominator2,
                    (-lambda_list[0] - lambda_list[2] - lambda_list[3]) / denominator2,
                    (
                        lambda_list[0] * lambda_list[2]
                        + lambda_list[0] * lambda_list[3]
                        + lambda_list[2] * lambda_list[3]
                    )
                    / denominator2,
                    (-lambda_list[0] * lambda_list[2] * lambda_list[3]) / denominator2,
                ],
                [
                    1 / denominator3,
                    (-lambda_list[0] - lambda_list[1] - lambda_list[3]) / denominator3,
                    (
                        lambda_list[0] * lambda_list[1]
                        + lambda_list[0] * lambda_list[3]
                        + lambda_list[1] * lambda_list[3]
                    )
                    / denominator3,
                    (-lambda_list[0] * lambda_list[1] * lambda_list[3]) / denominator3,
                ],
                [
                    1 / denominator4,
                    (-lambda_list[0] - lambda_list[1] - lambda_list[2]) / denominator4,
                    (
                        lambda_list[0] * lambda_list[1]
                        + lambda_list[0] * lambda_list[2]
                        + lambda_list[1] * lambda_list[2]
                    )
                    / denominator4,
                    (-lambda_list[0] * lambda_list[1] * lambda_list[2]) / denominator4,
                ],
            ]

    def get_coefficients_fn(self, order, interval_start, interval_end, lambda_list, tau):
        """
        Calculate the coefficient of gradients.
        """
        assert order in [1, 2, 3, 4]
        assert order == len(lambda_list), "the length of lambda list must be equal to the order"
        coefficients = []
        lagrange_coefficient = self.lagrange_polynomial_coefficient(order - 1, lambda_list)
        for i in range(order):
            coefficient = 0
            for j in range(order):
                if self.predict_x0:
                    coefficient += lagrange_coefficient[i][j] * self.get_coefficients_exponential_positive(
                        order - 1 - j, interval_start, interval_end, tau
                    )
                else:
                    coefficient += lagrange_coefficient[i][j] * self.get_coefficients_exponential_negative(
                        order - 1 - j, interval_start, interval_end
                    )
            coefficients.append(coefficient)
        assert len(coefficients) == order, "the length of coefficients does not match the order"
        return coefficients

    def adams_bashforth_update(self, order, x, tau, model_prev_list, t_prev_list, noise, t):
        """
        SA-Predictor, without the "rescaling" trick in Appendix D in SA-Solver paper https://arxiv.org/pdf/2309.05019.pdf
        """
        assert order in [1, 2, 3, 4], "order of stochastic adams bashforth method is only supported for 1, 2, 3 and 4"

        # get noise schedule
        ns = self.noise_schedule
        alpha_t = ns.marginal_alpha(t)
        sigma_t = ns.marginal_std(t)
        lambda_t = ns.marginal_lambda(t)
        alpha_prev = ns.marginal_alpha(t_prev_list[-1])
        sigma_prev = ns.marginal_std(t_prev_list[-1])
        gradient_part = torch.zeros_like(x)
        h = lambda_t - ns.marginal_lambda(t_prev_list[-1])
        lambda_list = []
        for i in range(order):
            lambda_list.append(ns.marginal_lambda(t_prev_list[-(i + 1)]))
        gradient_coefficients = self.get_coefficients_fn(
            order, ns.marginal_lambda(t_prev_list[-1]), lambda_t, lambda_list, tau
        )

        for i in range(order):
            if self.predict_x0:
                gradient_part += (
                    (1 + tau**2)
                    * sigma_t
                    * torch.exp(-(tau**2) * lambda_t)
                    * gradient_coefficients[i]
                    * model_prev_list[-(i + 1)]
                )
            else:
                gradient_part += -(1 + tau**2) * alpha_t * gradient_coefficients[i] * model_prev_list[-(i + 1)]

        if self.predict_x0:
            noise_part = sigma_t * torch.sqrt(1 - torch.exp(-2 * tau**2 * h)) * noise
        else:
            noise_part = tau * sigma_t * torch.sqrt(torch.exp(2 * h) - 1) * noise

        if self.predict_x0:
            x_t = torch.exp(-(tau**2) * h) * (sigma_t / sigma_prev) * x + gradient_part + noise_part
        else:
            x_t = (alpha_t / alpha_prev) * x + gradient_part + noise_part

        return x_t

    def adams_moulton_update(self, order, x, tau, model_prev_list, t_prev_list, noise, t):
        """
        SA-Corrector, without the "rescaling" trick in Appendix D in SA-Solver paper https://arxiv.org/pdf/2309.05019.pdf
        """

        assert order in [1, 2, 3, 4], "order of stochastic adams bashforth method is only supported for 1, 2, 3 and 4"

        # get noise schedule
        ns = self.noise_schedule
        alpha_t = ns.marginal_alpha(t)
        sigma_t = ns.marginal_std(t)
        lambda_t = ns.marginal_lambda(t)
        alpha_prev = ns.marginal_alpha(t_prev_list[-1])
        sigma_prev = ns.marginal_std(t_prev_list[-1])
        gradient_part = torch.zeros_like(x)
        h = lambda_t - ns.marginal_lambda(t_prev_list[-1])
        lambda_list = []
        t_list = t_prev_list + [t]
        for i in range(order):
            lambda_list.append(ns.marginal_lambda(t_list[-(i + 1)]))
        gradient_coefficients = self.get_coefficients_fn(
            order, ns.marginal_lambda(t_prev_list[-1]), lambda_t, lambda_list, tau
        )

        for i in range(order):
            if self.predict_x0:
                gradient_part += (
                    (1 + tau**2)
                    * sigma_t
                    * torch.exp(-(tau**2) * lambda_t)
                    * gradient_coefficients[i]
                    * model_prev_list[-(i + 1)]
                )
            else:
                gradient_part += -(1 + tau**2) * alpha_t * gradient_coefficients[i] * model_prev_list[-(i + 1)]

        if self.predict_x0:
            noise_part = sigma_t * torch.sqrt(1 - torch.exp(-2 * tau**2 * h)) * noise
        else:
            noise_part = tau * sigma_t * torch.sqrt(torch.exp(2 * h) - 1) * noise

        if self.predict_x0:
            x_t = torch.exp(-(tau**2) * h) * (sigma_t / sigma_prev) * x + gradient_part + noise_part
        else:
            x_t = (alpha_t / alpha_prev) * x + gradient_part + noise_part

        return x_t

    def adams_bashforth_update_few_steps(self, order, x, tau, model_prev_list, t_prev_list, noise, t):
        """
        SA-Predictor, with the "rescaling" trick in Appendix D in SA-Solver paper https://arxiv.org/pdf/2309.05019.pdf
        """

        assert order in [1, 2, 3, 4], "order of stochastic adams bashforth method is only supported for 1, 2, 3 and 4"

        # get noise schedule
        ns = self.noise_schedule
        alpha_t = ns.marginal_alpha(t)
        sigma_t = ns.marginal_std(t)
        lambda_t = ns.marginal_lambda(t)
        alpha_prev = ns.marginal_alpha(t_prev_list[-1])
        sigma_prev = ns.marginal_std(t_prev_list[-1])
        gradient_part = torch.zeros_like(x)
        h = lambda_t - ns.marginal_lambda(t_prev_list[-1])
        lambda_list = []
        for i in range(order):
            lambda_list.append(ns.marginal_lambda(t_prev_list[-(i + 1)]))
        gradient_coefficients = self.get_coefficients_fn(
            order, ns.marginal_lambda(t_prev_list[-1]), lambda_t, lambda_list, tau
        )

        if self.predict_x0:
            if (
                order == 2
            ):  ## if order = 2 we do a modification that does not influence the convergence order similar to unipc. Note: This is used only for few steps sampling.
                # The added term is O(h^3). Empirically we find it will slightly improve the image quality.
                # ODE case
                # gradient_coefficients[0] += 1.0 * torch.exp(lambda_t) * (h ** 2 / 2 - (h - 1 + torch.exp(-h))) / (ns.marginal_lambda(t_prev_list[-1]) - ns.marginal_lambda(t_prev_list[-2]))
                # gradient_coefficients[1] -= 1.0 * torch.exp(lambda_t) * (h ** 2 / 2 - (h - 1 + torch.exp(-h))) / (ns.marginal_lambda(t_prev_list[-1]) - ns.marginal_lambda(t_prev_list[-2]))
                gradient_coefficients[0] += (
                    1.0
                    * torch.exp((1 + tau**2) * lambda_t)
                    * (h**2 / 2 - (h * (1 + tau**2) - 1 + torch.exp((1 + tau**2) * (-h))) / ((1 + tau**2) ** 2))
                    / (ns.marginal_lambda(t_prev_list[-1]) - ns.marginal_lambda(t_prev_list[-2]))
                )
                gradient_coefficients[1] -= (
                    1.0
                    * torch.exp((1 + tau**2) * lambda_t)
                    * (h**2 / 2 - (h * (1 + tau**2) - 1 + torch.exp((1 + tau**2) * (-h))) / ((1 + tau**2) ** 2))
                    / (ns.marginal_lambda(t_prev_list[-1]) - ns.marginal_lambda(t_prev_list[-2]))
                )

        for i in range(order):
            if self.predict_x0:
                gradient_part += (
                    (1 + tau**2)
                    * sigma_t
                    * torch.exp(-(tau**2) * lambda_t)
                    * gradient_coefficients[i]
                    * model_prev_list[-(i + 1)]
                )
            else:
                gradient_part += -(1 + tau**2) * alpha_t * gradient_coefficients[i] * model_prev_list[-(i + 1)]

        if self.predict_x0:
            noise_part = sigma_t * torch.sqrt(1 - torch.exp(-2 * tau**2 * h)) * noise
        else:
            noise_part = tau * sigma_t * torch.sqrt(torch.exp(2 * h) - 1) * noise

        if self.predict_x0:
            x_t = torch.exp(-(tau**2) * h) * (sigma_t / sigma_prev) * x + gradient_part + noise_part
        else:
            x_t = (alpha_t / alpha_prev) * x + gradient_part + noise_part

        return x_t

    def adams_moulton_update_few_steps(self, order, x, tau, model_prev_list, t_prev_list, noise, t):
        """
        SA-Corrector, without the "rescaling" trick in Appendix D in SA-Solver paper https://arxiv.org/pdf/2309.05019.pdf
        """

        assert order in [1, 2, 3, 4], "order of stochastic adams bashforth method is only supported for 1, 2, 3 and 4"

        # get noise schedule
        ns = self.noise_schedule
        alpha_t = ns.marginal_alpha(t)
        sigma_t = ns.marginal_std(t)
        lambda_t = ns.marginal_lambda(t)
        alpha_prev = ns.marginal_alpha(t_prev_list[-1])
        sigma_prev = ns.marginal_std(t_prev_list[-1])
        gradient_part = torch.zeros_like(x)
        h = lambda_t - ns.marginal_lambda(t_prev_list[-1])
        lambda_list = []
        t_list = t_prev_list + [t]
        for i in range(order):
            lambda_list.append(ns.marginal_lambda(t_list[-(i + 1)]))
        gradient_coefficients = self.get_coefficients_fn(
            order, ns.marginal_lambda(t_prev_list[-1]), lambda_t, lambda_list, tau
        )

        if self.predict_x0:
            if (
                order == 2
            ):  ## if order = 2 we do a modification that does not influence the convergence order similar to UniPC. Note: This is used only for few steps sampling.
                # The added term is O(h^3). Empirically we find it will slightly improve the image quality.
                # ODE case
                # gradient_coefficients[0] += 1.0 * torch.exp(lambda_t) * (h / 2 - (h - 1 + torch.exp(-h)) / h)
                # gradient_coefficients[1] -= 1.0 * torch.exp(lambda_t) * (h / 2 - (h - 1 + torch.exp(-h)) / h)
                gradient_coefficients[0] += (
                    1.0
                    * torch.exp((1 + tau**2) * lambda_t)
                    * (h / 2 - (h * (1 + tau**2) - 1 + torch.exp((1 + tau**2) * (-h))) / ((1 + tau**2) ** 2 * h))
                )
                gradient_coefficients[1] -= (
                    1.0
                    * torch.exp((1 + tau**2) * lambda_t)
                    * (h / 2 - (h * (1 + tau**2) - 1 + torch.exp((1 + tau**2) * (-h))) / ((1 + tau**2) ** 2 * h))
                )

        for i in range(order):
            if self.predict_x0:
                gradient_part += (
                    (1 + tau**2)
                    * sigma_t
                    * torch.exp(-(tau**2) * lambda_t)
                    * gradient_coefficients[i]
                    * model_prev_list[-(i + 1)]
                )
            else:
                gradient_part += -(1 + tau**2) * alpha_t * gradient_coefficients[i] * model_prev_list[-(i + 1)]

        if self.predict_x0:
            noise_part = sigma_t * torch.sqrt(1 - torch.exp(-2 * tau**2 * h)) * noise
        else:
            noise_part = tau * sigma_t * torch.sqrt(torch.exp(2 * h) - 1) * noise

        if self.predict_x0:
            x_t = torch.exp(-(tau**2) * h) * (sigma_t / sigma_prev) * x + gradient_part + noise_part
        else:
            x_t = (alpha_t / alpha_prev) * x + gradient_part + noise_part

        return x_t

    def sample_few_steps(
        self,
        x,
        tau,
        steps=5,
        t_start=None,
        t_end=None,
        skip_type="time",
        skip_order=1,
        predictor_order=3,
        corrector_order=4,
        pc_mode="PEC",
        return_intermediate=False,
    ):
        """
        For the PC-mode, please refer to the wiki page
        https://en.wikipedia.org/wiki/Predictor%E2%80%93corrector_method#PEC_mode_and_PECE_mode
        'PEC' needs one model evaluation per step while 'PECE' needs two model evaluations
        We recommend use pc_mode='PEC' for NFEs is limited. 'PECE' mode is only for test with sufficient NFEs.
        """

        skip_first_step = False
        skip_final_step = True
        lower_order_final = True
        denoise_to_zero = False

        assert pc_mode in ["PEC", "PECE"], "Predictor-corrector mode only supports PEC and PECE"
        t_0 = 1.0 / self.noise_schedule.total_N if t_end is None else t_end
        t_T = self.noise_schedule.T if t_start is None else t_start
        assert (
            t_0 > 0 and t_T > 0
        ), "Time range needs to be greater than 0. For discrete-time DPMs, it needs to be in [1 / N, 1], where N is the length of betas array"

        device = x.device
        intermediates = []
        with torch.no_grad():
            assert steps >= max(predictor_order, corrector_order - 1)
            timesteps = self.get_time_steps(
                skip_type=skip_type, t_T=t_T, t_0=t_0, N=steps, order=skip_order, device=device
            )
            assert timesteps.shape[0] - 1 == steps
            # Init the initial values.
            step = 0
            t = timesteps[step]
            noise = torch.randn_like(x)
            t_prev_list = [t]
            # do not evaluate if skip_first_step
            if skip_first_step:
                if self.predict_x0:
                    alpha_t = self.noise_schedule.marginal_alpha(t)
                    sigma_t = self.noise_schedule.marginal_std(t)
                    model_prev_list = [(1 - sigma_t) / alpha_t * x]
                else:
                    model_prev_list = [x]
            else:
                model_prev_list = [self.model_fn(x, t)]

            if self.correcting_xt_fn is not None:
                x = self.correcting_xt_fn(x, t, step)
            if return_intermediate:
                intermediates.append(x)

            # determine the first several values
            for step in tqdm(range(1, max(predictor_order, corrector_order - 1))):

                t = timesteps[step]
                predictor_order_used = min(predictor_order, step)
                corrector_order_used = min(corrector_order, step + 1)
                noise = torch.randn_like(x)
                # predictor step
                x_p = self.adams_bashforth_update_few_steps(
                    order=predictor_order_used,
                    x=x,
                    tau=tau(t),
                    model_prev_list=model_prev_list,
                    t_prev_list=t_prev_list,
                    noise=noise,
                    t=t,
                )
                # evaluation step
                model_x = self.model_fn(x_p, t)

                # update model_list
                model_prev_list.append(model_x)
                # corrector step
                if corrector_order > 0:
                    x = self.adams_moulton_update_few_steps(
                        order=corrector_order_used,
                        x=x,
                        tau=tau(t),
                        model_prev_list=model_prev_list,
                        t_prev_list=t_prev_list,
                        noise=noise,
                        t=t,
                    )
                else:
                    x = x_p

                # evaluation step if correction and mode = pece
                if corrector_order > 0:
                    if pc_mode == "PECE":
                        model_x = self.model_fn(x, t)
                        del model_prev_list[-1]
                        model_prev_list.append(model_x)

                if self.correcting_xt_fn is not None:
                    x = self.correcting_xt_fn(x, t, step)
                if return_intermediate:
                    intermediates.append(x)

                t_prev_list.append(t)

            for step in tqdm(range(max(predictor_order, corrector_order - 1), steps + 1)):
                if lower_order_final:
                    predictor_order_used = min(predictor_order, steps - step + 1)
                    corrector_order_used = min(corrector_order, steps - step + 2)

                else:
                    predictor_order_used = predictor_order
                    corrector_order_used = corrector_order
                t = timesteps[step]
                noise = torch.randn_like(x)

                # predictor step
                if skip_final_step and step == steps and not denoise_to_zero:
                    x_p = self.adams_bashforth_update_few_steps(
                        order=predictor_order_used,
                        x=x,
                        tau=0,
                        model_prev_list=model_prev_list,
                        t_prev_list=t_prev_list,
                        noise=noise,
                        t=t,
                    )
                else:
                    x_p = self.adams_bashforth_update_few_steps(
                        order=predictor_order_used,
                        x=x,
                        tau=tau(t),
                        model_prev_list=model_prev_list,
                        t_prev_list=t_prev_list,
                        noise=noise,
                        t=t,
                    )

                # evaluation step
                # do not evaluate if skip_final_step and step = steps
                if not skip_final_step or step < steps:
                    model_x = self.model_fn(x_p, t)

                # update model_list
                # do not update if skip_final_step and step = steps
                if not skip_final_step or step < steps:
                    model_prev_list.append(model_x)

                # corrector step
                # do not correct if skip_final_step and step = steps
                if corrector_order > 0:
                    if not skip_final_step or step < steps:
                        x = self.adams_moulton_update_few_steps(
                            order=corrector_order_used,
                            x=x,
                            tau=tau(t),
                            model_prev_list=model_prev_list,
                            t_prev_list=t_prev_list,
                            noise=noise,
                            t=t,
                        )
                    else:
                        x = x_p
                else:
                    x = x_p

                # evaluation step if mode = pece and step != steps
                if corrector_order > 0:
                    if pc_mode == "PECE" and step < steps:
                        model_x = self.model_fn(x, t)
                        del model_prev_list[-1]
                        model_prev_list.append(model_x)

                if self.correcting_xt_fn is not None:
                    x = self.correcting_xt_fn(x, t, step)
                if return_intermediate:
                    intermediates.append(x)

                t_prev_list.append(t)
                del model_prev_list[0]

            if denoise_to_zero:
                t = torch.ones((1,)).to(device) * t_0
                x = self.denoise_to_zero_fn(x, t)
                if self.correcting_xt_fn is not None:
                    x = self.correcting_xt_fn(x, t, step + 1)
                if return_intermediate:
                    intermediates.append(x)
        if return_intermediate:
            return x, intermediates
        else:
            return x

    def sample_more_steps(
        self,
        x,
        tau,
        steps=20,
        t_start=None,
        t_end=None,
        skip_type="time",
        skip_order=1,
        predictor_order=3,
        corrector_order=4,
        pc_mode="PEC",
        return_intermediate=False,
    ):
        """
        For the PC-mode, please refer to the wiki page
        https://en.wikipedia.org/wiki/Predictor%E2%80%93corrector_method#PEC_mode_and_PECE_mode
        'PEC' needs one model evaluation per step while 'PECE' needs two model evaluations
        We recommend use pc_mode='PEC' for NFEs is limited. 'PECE' mode is only for test with sufficient NFEs.
        """

        skip_first_step = False
        skip_final_step = False
        lower_order_final = True
        denoise_to_zero = True

        assert pc_mode in ["PEC", "PECE"], "Predictor-corrector mode only supports PEC and PECE"
        t_0 = 1.0 / self.noise_schedule.total_N if t_end is None else t_end
        t_T = self.noise_schedule.T if t_start is None else t_start
        assert (
            t_0 > 0 and t_T > 0
        ), "Time range needs to be greater than 0. For discrete-time DPMs, it needs to be in [1 / N, 1], where N is the length of betas array"

        device = x.device
        intermediates = []
        with torch.no_grad():
            assert steps >= max(predictor_order, corrector_order - 1)
            timesteps = self.get_time_steps(
                skip_type=skip_type, t_T=t_T, t_0=t_0, N=steps, order=skip_order, device=device
            )
            assert timesteps.shape[0] - 1 == steps
            # Init the initial values.
            step = 0
            t = timesteps[step]
            noise = torch.randn_like(x)
            t_prev_list = [t]
            # do not evaluate if skip_first_step
            if skip_first_step:
                if self.predict_x0:
                    alpha_t = self.noise_schedule.marginal_alpha(t)
                    sigma_t = self.noise_schedule.marginal_std(t)
                    model_prev_list = [(1 - sigma_t) / alpha_t * x]
                else:
                    model_prev_list = [x]
            else:
                model_prev_list = [self.model_fn(x, t)]

            if self.correcting_xt_fn is not None:
                x = self.correcting_xt_fn(x, t, step)
            if return_intermediate:
                intermediates.append(x)

            # determine the first several values
            for step in tqdm(range(1, max(predictor_order, corrector_order - 1))):

                t = timesteps[step]
                predictor_order_used = min(predictor_order, step)
                corrector_order_used = min(corrector_order, step + 1)
                noise = torch.randn_like(x)
                # predictor step
                x_p = self.adams_bashforth_update(
                    order=predictor_order_used,
                    x=x,
                    tau=tau(t),
                    model_prev_list=model_prev_list,
                    t_prev_list=t_prev_list,
                    noise=noise,
                    t=t,
                )
                # evaluation step
                model_x = self.model_fn(x_p, t)

                # update model_list
                model_prev_list.append(model_x)
                # corrector step
                if corrector_order > 0:
                    x = self.adams_moulton_update(
                        order=corrector_order_used,
                        x=x,
                        tau=tau(t),
                        model_prev_list=model_prev_list,
                        t_prev_list=t_prev_list,
                        noise=noise,
                        t=t,
                    )
                else:
                    x = x_p

                # evaluation step if mode = pece
                if corrector_order > 0:
                    if pc_mode == "PECE":
                        model_x = self.model_fn(x, t)
                        del model_prev_list[-1]
                        model_prev_list.append(model_x)
                if self.correcting_xt_fn is not None:
                    x = self.correcting_xt_fn(x, t, step)
                if return_intermediate:
                    intermediates.append(x)

                t_prev_list.append(t)

            for step in tqdm(range(max(predictor_order, corrector_order - 1), steps + 1)):
                if lower_order_final:
                    predictor_order_used = min(predictor_order, steps - step + 1)
                    corrector_order_used = min(corrector_order, steps - step + 2)

                else:
                    predictor_order_used = predictor_order
                    corrector_order_used = corrector_order
                t = timesteps[step]
                noise = torch.randn_like(x)

                # predictor step
                if skip_final_step and step == steps and not denoise_to_zero:
                    x_p = self.adams_bashforth_update(
                        order=predictor_order_used,
                        x=x,
                        tau=0,
                        model_prev_list=model_prev_list,
                        t_prev_list=t_prev_list,
                        noise=noise,
                        t=t,
                    )
                else:
                    x_p = self.adams_bashforth_update(
                        order=predictor_order_used,
                        x=x,
                        tau=tau(t),
                        model_prev_list=model_prev_list,
                        t_prev_list=t_prev_list,
                        noise=noise,
                        t=t,
                    )

                # evaluation step
                # do not evaluate if skip_final_step and step = steps
                if not skip_final_step or step < steps:
                    model_x = self.model_fn(x_p, t)

                # update model_list
                # do not update if skip_final_step and step = steps
                if not skip_final_step or step < steps:
                    model_prev_list.append(model_x)

                # corrector step
                # do not correct if skip_final_step and step = steps
                if corrector_order > 0:
                    if not skip_final_step or step < steps:
                        x = self.adams_moulton_update(
                            order=corrector_order_used,
                            x=x,
                            tau=tau(t),
                            model_prev_list=model_prev_list,
                            t_prev_list=t_prev_list,
                            noise=noise,
                            t=t,
                        )
                    else:
                        x = x_p
                else:
                    x = x_p

                # evaluation step if mode = pece and step != steps
                if corrector_order > 0:
                    if pc_mode == "PECE" and step < steps:
                        model_x = self.model_fn(x, t)
                        del model_prev_list[-1]
                        model_prev_list.append(model_x)

                if self.correcting_xt_fn is not None:
                    x = self.correcting_xt_fn(x, t, step)
                if return_intermediate:
                    intermediates.append(x)

                t_prev_list.append(t)
                del model_prev_list[0]

            if denoise_to_zero:
                t = torch.ones((1,)).to(device) * t_0
                x = self.denoise_to_zero_fn(x, t)
                if self.correcting_xt_fn is not None:
                    x = self.correcting_xt_fn(x, t, step + 1)
                if return_intermediate:
                    intermediates.append(x)
        if return_intermediate:
            return x, intermediates
        else:
            return x

    def sample(
        self,
        mode,
        x,
        tau,
        steps,
        t_start=None,
        t_end=None,
        skip_type="time",
        skip_order=1,
        predictor_order=3,
        corrector_order=4,
        pc_mode="PEC",
        return_intermediate=False,
    ):
        """
        For the PC-mode, please refer to the wiki page
        https://en.wikipedia.org/wiki/Predictor%E2%80%93corrector_method#PEC_mode_and_PECE_mode
        'PEC' needs one model evaluation per step while 'PECE' needs two model evaluations
        We recommend use pc_mode='PEC' for NFEs is limited. 'PECE' mode is only for test with sufficient NFEs.

        'few_steps' mode is recommended. The differences between 'few_steps' and 'more_steps' are as below:
        1) 'few_steps' do not correct at final step and do not denoise to zero, while 'more_steps' do these two.
        Thus the NFEs for 'few_steps' = steps, NFEs for 'more_steps' = steps + 2
        For most of the experiments and tasks, we find these two operations do not have much help to sample quality.
        2) 'few_steps' use a rescaling trick as in Appendix D in SA-Solver paper https://arxiv.org/pdf/2309.05019.pdf
        We find it will slightly improve the sample quality especially in few steps.
        """
        assert mode in ["few_steps", "more_steps"], "mode must be either 'few_steps' or 'more_steps'"
        if mode == "few_steps":
            return self.sample_few_steps(
                x=x,
                tau=tau,
                steps=steps,
                t_start=t_start,
                t_end=t_end,
                skip_type=skip_type,
                skip_order=skip_order,
                predictor_order=predictor_order,
                corrector_order=corrector_order,
                pc_mode=pc_mode,
                return_intermediate=return_intermediate,
            )
        else:
            return self.sample_more_steps(
                x=x,
                tau=tau,
                steps=steps,
                t_start=t_start,
                t_end=t_end,
                skip_type=skip_type,
                skip_order=skip_order,
                predictor_order=predictor_order,
                corrector_order=corrector_order,
                pc_mode=pc_mode,
                return_intermediate=return_intermediate,
            )


#############################################################
# other utility functions
#############################################################


def interpolate_fn(x, xp, yp):
    """
    A piecewise linear function y = f(x), using xp and yp as keypoints.
    We implement f(x) in a differentiable way (i.e. applicable for autograd).
    The function f(x) is well-defined for all x-axis. (For x beyond the bounds of xp, we use the outmost points of xp to define the linear function.)
    Args:
        x: PyTorch tensor with shape [N, C], where N is the batch size, C is the number of channels (we use C = 1 for DPM-Solver).
        xp: PyTorch tensor with shape [C, K], where K is the number of keypoints.
        yp: PyTorch tensor with shape [C, K].
    Returns:
        The function values f(x), with shape [N, C].
    """
    N, K = x.shape[0], xp.shape[1]
    all_x = torch.cat([x.unsqueeze(2), xp.unsqueeze(0).repeat((N, 1, 1))], dim=2)
    sorted_all_x, x_indices = torch.sort(all_x, dim=2)
    x_idx = torch.argmin(x_indices, dim=2)
    cand_start_idx = x_idx - 1
    start_idx = torch.where(
        torch.eq(x_idx, 0),
        torch.tensor(1, device=x.device),
        torch.where(
            torch.eq(x_idx, K),
            torch.tensor(K - 2, device=x.device),
            cand_start_idx,
        ),
    )
    end_idx = torch.where(torch.eq(start_idx, cand_start_idx), start_idx + 2, start_idx + 1)
    start_x = torch.gather(sorted_all_x, dim=2, index=start_idx.unsqueeze(2)).squeeze(2)
    end_x = torch.gather(sorted_all_x, dim=2, index=end_idx.unsqueeze(2)).squeeze(2)
    start_idx2 = torch.where(
        torch.eq(x_idx, 0),
        torch.tensor(0, device=x.device),
        torch.where(
            torch.eq(x_idx, K),
            torch.tensor(K - 2, device=x.device),
            cand_start_idx,
        ),
    )
    y_positions_expanded = yp.unsqueeze(0).expand(N, -1, -1)
    start_y = torch.gather(y_positions_expanded, dim=2, index=start_idx2.unsqueeze(2)).squeeze(2)
    end_y = torch.gather(y_positions_expanded, dim=2, index=(start_idx2 + 1).unsqueeze(2)).squeeze(2)
    cand = start_y + (x - start_x) * (end_y - start_y) / (end_x - start_x)
    return cand


def expand_dims(v, dims):
    """
    Expand the tensor `v` to the dim `dims`.
    Args:
        `v`: a PyTorch tensor with shape [N].
        `dim`: a `int`.
    Returns:
        a PyTorch tensor with shape [N, 1, 1, ..., 1] and the total dimension is `dims`.
    """
    return v[(...,) + (None,) * (dims - 1)]
