import types
from abc import ABC, abstractmethod
from typing import List, Optional

import torch
from einops import rearrange

from diffusion.model.nets.basic_modules import CachedGLUMBConvTemp
from diffusion.model.nets.sana_blocks import CachedCausalAttention


class SchedulerInterface(ABC):
    """
    Base class for diffusion noise schedule.
    """

    alphas_cumprod: torch.Tensor  # [T], alphas for defining the noise schedule

    @abstractmethod
    def add_noise(self, clean_latent: torch.Tensor, noise: torch.Tensor, timestep: torch.Tensor):
        """
        Diffusion forward corruption process.
        Input:
            - clean_latent: the clean latent with shape [B, C, H, W]
            - noise: the noise with shape [B, C, H, W]
            - timestep: the timestep with shape [B]
        Output: the corrupted latent with shape [B, C, H, W]
        """

    def convert_x0_to_noise(self, x0: torch.Tensor, xt: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        """
        Convert the diffusion network's x0 prediction to noise predidction.
        x0: the predicted clean data with shape [B, C, H, W]
        xt: the input noisy data with shape [B, C, H, W]
        timestep: the timestep with shape [B]

        noise = (xt-sqrt(alpha_t)*x0) / sqrt(beta_t) (eq 11 in https://arxiv.org/abs/2311.18828)
        """
        # use higher precision for calculations
        original_dtype = x0.dtype
        x0, xt, alphas_cumprod = map(lambda x: x.double().to(x0.device), [x0, xt, self.alphas_cumprod])

        alpha_prod_t = alphas_cumprod[timestep].reshape(-1, 1, 1, 1)
        beta_prod_t = 1 - alpha_prod_t

        noise_pred = (xt - alpha_prod_t ** (0.5) * x0) / beta_prod_t ** (0.5)
        return noise_pred.to(original_dtype)

    def convert_noise_to_x0(self, noise: torch.Tensor, xt: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        """
        Convert the diffusion network's noise prediction to x0 predidction.
        noise: the predicted noise with shape [B, C, H, W]
        xt: the input noisy data with shape [B, C, H, W]
        timestep: the timestep with shape [B]

        x0 = (x_t - sqrt(beta_t) * noise) / sqrt(alpha_t) (eq 11 in https://arxiv.org/abs/2311.18828)
        """
        # use higher precision for calculations
        original_dtype = noise.dtype
        noise, xt, alphas_cumprod = map(lambda x: x.double().to(noise.device), [noise, xt, self.alphas_cumprod])
        alpha_prod_t = alphas_cumprod[timestep].reshape(-1, 1, 1, 1)
        beta_prod_t = 1 - alpha_prod_t

        x0_pred = (xt - beta_prod_t ** (0.5) * noise) / alpha_prod_t ** (0.5)
        return x0_pred.to(original_dtype)

    def convert_velocity_to_x0(self, velocity: torch.Tensor, xt: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        """
        Convert the diffusion network's velocity prediction to x0 predidction.
        velocity: the predicted noise with shape [B, C, H, W]
        xt: the input noisy data with shape [B, C, H, W]
        timestep: the timestep with shape [B]

        v = sqrt(alpha_t) * noise - sqrt(beta_t) x0
        noise = (xt-sqrt(alpha_t)*x0) / sqrt(beta_t)
        given v, x_t, we have
        x0 = sqrt(alpha_t) * x_t - sqrt(beta_t) * v
        see derivations https://chatgpt.com/share/679fb6c8-3a30-8008-9b0e-d1ae892dac56
        """
        # use higher precision for calculations
        original_dtype = velocity.dtype
        velocity, xt, alphas_cumprod = map(
            lambda x: x.double().to(velocity.device), [velocity, xt, self.alphas_cumprod]
        )
        alpha_prod_t = alphas_cumprod[timestep].reshape(-1, 1, 1, 1)
        beta_prod_t = 1 - alpha_prod_t

        x0_pred = (alpha_prod_t**0.5) * xt - (beta_prod_t**0.5) * velocity
        return x0_pred.to(original_dtype)


class FlowMatchScheduler:
    def __init__(
        self,
        num_inference_steps=100,
        num_train_timesteps=1000,
        shift=3.0,
        sigma_max=1.0,
        sigma_min=0.003 / 1.002,
        inverse_timesteps=False,
        extra_one_step=False,
        reverse_sigmas=False,
    ):
        self.num_train_timesteps = num_train_timesteps
        self.shift = shift
        self.sigma_max = sigma_max
        self.sigma_min = sigma_min
        self.inverse_timesteps = inverse_timesteps
        self.extra_one_step = extra_one_step
        self.reverse_sigmas = reverse_sigmas
        self.set_timesteps(num_inference_steps)

    def set_timesteps(self, num_inference_steps=100, denoising_strength=1.0, training=False):
        sigma_start = self.sigma_min + (self.sigma_max - self.sigma_min) * denoising_strength
        if self.extra_one_step:
            self.sigmas = torch.linspace(sigma_start, self.sigma_min, num_inference_steps + 1)[:-1]
        else:
            self.sigmas = torch.linspace(sigma_start, self.sigma_min, num_inference_steps)
        if self.inverse_timesteps:
            self.sigmas = torch.flip(self.sigmas, dims=[0])
        self.sigmas = self.shift * self.sigmas / (1 + (self.shift - 1) * self.sigmas)
        if self.reverse_sigmas:
            self.sigmas = 1 - self.sigmas
        self.timesteps = self.sigmas * self.num_train_timesteps
        if training:
            x = self.timesteps
            y = torch.exp(-2 * ((x - num_inference_steps / 2) / num_inference_steps) ** 2)
            y_shifted = y - y.min()
            bsmntw_weighing = y_shifted * (num_inference_steps / y_shifted.sum())
            self.linear_timesteps_weights = bsmntw_weighing

    def step(self, model_output, timestep, sample, to_final=False):
        if timestep.ndim == 2:
            timestep = timestep.flatten(0, 1)
        self.sigmas = self.sigmas.to(model_output.device)
        self.timesteps = self.timesteps.to(model_output.device)
        timestep_id = torch.argmin((self.timesteps.unsqueeze(0) - timestep.unsqueeze(1)).abs(), dim=1)
        sigma = self.sigmas[timestep_id].reshape(-1, 1, 1, 1)
        if to_final or (timestep_id + 1 >= len(self.timesteps)).any():
            sigma_ = 1 if (self.inverse_timesteps or self.reverse_sigmas) else 0
        else:
            sigma_ = self.sigmas[timestep_id + 1].reshape(-1, 1, 1, 1)
        prev_sample = sample + model_output * (sigma_ - sigma)
        return prev_sample

    def add_noise(self, original_samples, noise, timestep):
        """
        Diffusion forward corruption process.
        Input:
            - clean_latent: the clean latent with shape [B*T, C, H, W]
            - noise: the noise with shape [B*T, C, H, W]
            - timestep: the timestep with shape [B*T]
        Output: the corrupted latent with shape [B*T, C, H, W]
        """
        if timestep.ndim == 2:
            timestep = timestep.flatten(0, 1)
        self.sigmas = self.sigmas.to(noise.device)
        self.timesteps = self.timesteps.to(noise.device)
        timestep_id = torch.argmin((self.timesteps.unsqueeze(0) - timestep.unsqueeze(1)).abs(), dim=1)
        sigma = self.sigmas[timestep_id].reshape(-1, 1, 1, 1)
        sample = (1 - sigma) * original_samples + sigma * noise
        return sample.type_as(noise)

    def training_target(self, sample, noise, timestep):
        target = noise - sample
        return target

    def training_weight(self, timestep):
        """
        Input:
            - timestep: the timestep with shape [B*T]
        Output: the corresponding weighting [B*T]
        """
        if timestep.ndim == 2:
            timestep = timestep.flatten(0, 1)
        self.linear_timesteps_weights = self.linear_timesteps_weights.to(timestep.device)
        timestep_id = torch.argmin((self.timesteps.unsqueeze(1) - timestep.unsqueeze(0)).abs(), dim=0)
        weights = self.linear_timesteps_weights[timestep_id]
        return weights


class SanaModelWrapper(torch.nn.Module):
    """
    SANA-Video Wrapper
    """

    def __init__(self, sana_model, flow_shift: float = 3.0):
        super().__init__()
        self.model = sana_model
        self.flow_shift = float(flow_shift)
        self.uniform_timestep = False  # SANA-Video supports
        self.scheduler = FlowMatchScheduler(shift=self.flow_shift, sigma_min=0.0, extra_one_step=True)
        self.scheduler.set_timesteps(1000, training=True)

    def get_scheduler(self) -> SchedulerInterface:
        """
        Update the current scheduler with the interface's static method
        """
        scheduler = self.scheduler
        scheduler.convert_x0_to_noise = types.MethodType(SchedulerInterface.convert_x0_to_noise, scheduler)
        scheduler.convert_noise_to_x0 = types.MethodType(SchedulerInterface.convert_noise_to_x0, scheduler)
        scheduler.convert_velocity_to_x0 = types.MethodType(SchedulerInterface.convert_velocity_to_x0, scheduler)
        self.scheduler = scheduler
        return scheduler

    def post_init(self):
        """
        A few custom initialization steps that should be called after the object is created.
        Currently, the only one we have is to bind a few methods to scheduler.
        We can gradually add more methods here if needed.
        """
        self.get_scheduler()

    def enable_gradient_checkpointing(self):
        if hasattr(self.model, "enable_gradient_checkpointing"):
            self.model.enable_gradient_checkpointing()

    def _convert_flow_pred_to_x0(
        self, flow_pred: torch.Tensor, xt: torch.Tensor, timestep: torch.Tensor
    ) -> torch.Tensor:
        """
        Convert flow matching's prediction to x0 prediction.
        flow_pred: the prediction with shape [B, C, H, W]
        xt: the input noisy data with shape [B, C, H, W]
        timestep: the timestep with shape [B]

        pred = noise - x0
        x_t = (1-sigma_t) * x0 + sigma_t * noise
        we have x0 = x_t - sigma_t * pred
        see derivations https://chatgpt.com/share/67bf8589-3d04-8008-bc6e-4cf1a24e2d0e
        """
        # use higher precision for calculations
        original_dtype = flow_pred.dtype
        flow_pred, xt, sigmas, timesteps = map(
            lambda x: x.double().to(flow_pred.device), [flow_pred, xt, self.scheduler.sigmas, self.scheduler.timesteps]
        )
        timestep_id = torch.argmin((timesteps.unsqueeze(0) - timestep.unsqueeze(1)).abs(), dim=1)
        sigma_t = sigmas[timestep_id].reshape(-1, 1, 1, 1)
        x0_pred = xt - sigma_t * flow_pred
        return x0_pred.to(original_dtype)

    @staticmethod
    def _convert_x0_to_flow_pred(
        scheduler, x0_pred: torch.Tensor, xt: torch.Tensor, timestep: torch.Tensor
    ) -> torch.Tensor:
        """
        Convert x0 prediction to flow matching's prediction.
        x0_pred: the x0 prediction with shape [B, C, H, W]
        xt: the input noisy data with shape [B, C, H, W]
        timestep: the timestep with shape [B]

        pred = (x_t - x_0) / sigma_t
        """
        # use higher precision for calculations
        original_dtype = x0_pred.dtype
        x0_pred, xt, sigmas, timesteps = map(
            lambda x: x.double().to(x0_pred.device), [x0_pred, xt, scheduler.sigmas, scheduler.timesteps]
        )
        timestep_id = torch.argmin((timesteps.unsqueeze(0) - timestep.unsqueeze(1)).abs(), dim=1)
        sigma_t = sigmas[timestep_id].reshape(-1, 1, 1, 1)
        flow_pred = (xt - x0_pred) / sigma_t
        return flow_pred.to(original_dtype)

    def forward(
        self,
        noisy_image_or_video: torch.Tensor,
        condition: torch.Tensor,
        timestep: torch.Tensor,
        start_f: int = None,
        end_f: int = None,
        save_kv_cache: bool = False,
        mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        if condition.dim() == 3:
            condition = condition.unsqueeze(1)
        elif condition.dim() == 2:
            condition = condition.unsqueeze(0).unsqueeze(0)

        model = self.model
        if timestep.dim() == 2:
            input_t = timestep[:, 0]
        else:
            input_t = timestep

        model_out = model(
            noisy_image_or_video,
            input_t,
            condition,
            start_f=start_f,
            end_f=end_f,
            save_kv_cache=save_kv_cache,
            mask=mask,
            **kwargs,
        )

        if isinstance(model_out, tuple) and len(model_out) == 2:
            model_out, kv_cache_ret = model_out
        else:
            kv_cache_ret = None

        try:
            from diffusers.models.modeling_outputs import Transformer2DModelOutput

            if isinstance(model_out, Transformer2DModelOutput):
                model_out = model_out[0]
        except Exception:
            pass

        if isinstance(model_out, Transformer2DModelOutput):
            model_out = model_out[0]

        flow_pred_bcfhw = model_out
        flow_pred = rearrange(flow_pred_bcfhw, "b c f h w -> b f c h w")
        noisy_image_or_video = rearrange(noisy_image_or_video, "b c f h w -> b f c h w")
        pred_x0 = self._convert_flow_pred_to_x0(
            flow_pred=flow_pred.flatten(0, 1), xt=noisy_image_or_video.flatten(0, 1), timestep=input_t
        ).unflatten(0, flow_pred.shape[:2])
        pred_x0_bcfhw = rearrange(pred_x0, "b f c h w -> b c f h w")

        return flow_pred_bcfhw, pred_x0_bcfhw, kv_cache_ret


class LongLiveFlowEuler:
    def __init__(
        self,
        model_fn,
        condition,
        model_kwargs,
        flow_shift=7.0,
        base_chunk_frames=10,
        num_cached_blocks=-1,
        denoising_step_list=[1000, 960, 889, 727],
        **kwargs,
    ):
        self.generator = SanaModelWrapper(model_fn, flow_shift=flow_shift)
        self.condition = condition
        self.mask = model_kwargs.pop("mask", None)

        self.scheduler = self.generator.get_scheduler()
        self.num_frame_per_block = base_chunk_frames
        self.denoising_step_list = denoising_step_list
        if len(self.denoising_step_list) > 0 and self.denoising_step_list[-1] == 0:
            self.denoising_step_list = self.denoising_step_list[:-1]

        inner = self.generator.model if hasattr(self.generator, "model") else self.generator
        try:
            p = next(inner.parameters())
            self.model_device = p.device
            self.model_dtype = p.dtype
        except Exception:
            self.model_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

        self.cached_modules = None
        self.num_model_blocks = 0
        self.num_cached_blocks = num_cached_blocks

        self._initialize_cached_modules()

    def _initialize_cached_modules(self):
        if self.cached_modules is not None:
            return self.cached_modules
        model = self.generator.model if hasattr(self.generator, "model") else self.generator
        model = model.module if hasattr(model, "module") else model

        cached_modules = []

        def collect_from_block(block, block_idx):
            attention_modules = []
            conv_modules = []

            def collect_recursive(module):
                if isinstance(module, CachedCausalAttention):
                    attention_modules.append(module)
                elif isinstance(module, CachedGLUMBConvTemp):
                    conv_modules.append(module)
                for child in module.children():
                    collect_recursive(child)

            collect_recursive(block)
            return attention_modules + conv_modules

        if hasattr(model, "blocks"):
            blocks = model.blocks
        elif hasattr(model, "transformer_blocks"):
            blocks = model.transformer_blocks
        elif hasattr(model, "layers"):
            blocks = model.layers
        else:
            raise ValueError("Sana model does not have any blocks")

        self.num_model_blocks = len(blocks)
        for block_idx, block in enumerate(blocks):
            block_modules = collect_from_block(block, block_idx)
            cached_modules.append(block_modules)

        self.cached_modules = cached_modules
        return cached_modules

    def _create_autoregressive_segments(self, total_frames: int, base_chunk_frames: int) -> List[int]:
        remained_frames = total_frames % base_chunk_frames
        num_chunks = total_frames // base_chunk_frames
        chunk_indices = [0]
        for i in range(num_chunks):
            cur_idx = chunk_indices[-1] + base_chunk_frames
            if i == 0:
                cur_idx += remained_frames
            chunk_indices.append(cur_idx)
        if chunk_indices[-1] < total_frames:
            chunk_indices.append(total_frames)
        return chunk_indices

    def _initialize_kv_cache(self, num_chunks: int):
        kv_cache: list = []
        for _ in range(num_chunks):
            kv_cache.append([[None, None, None] for _ in range(self.num_model_blocks)])
        return kv_cache

    def _accumulate_kv_cache(self, kv_cache, chunk_idx):
        if chunk_idx == 0:
            return kv_cache[0]
        cur_kv_cache = kv_cache[chunk_idx]
        for block_id in range(self.num_model_blocks):
            cur_kv_cache[block_id][2] = kv_cache[chunk_idx - 1][block_id][2]
            cum_vk, cum_k_sum = None, None
            start_chunk_idx = chunk_idx - self.num_cached_blocks if self.num_cached_blocks > 0 else 0
            for i in range(start_chunk_idx, chunk_idx):
                prev = kv_cache[i][block_id]
                if prev[0] is not None and prev[1] is not None:
                    if cum_vk is None:
                        cum_vk = prev[0].clone()
                        cum_k_sum = prev[1].clone()
                    else:
                        cum_vk += prev[0]
                        cum_k_sum += prev[1]
            if chunk_idx > 0:
                assert cum_vk is not None and cum_k_sum is not None
            cur_kv_cache[block_id][0] = cum_vk
            cur_kv_cache[block_id][1] = cum_k_sum
        return cur_kv_cache

    @torch.no_grad()
    def sample(self, latents: torch.Tensor, **kwargs):
        if latents.dim() != 5:
            raise ValueError("noise should be a 5D tensor")

        latents_bcthw = latents

        batch_size, c, total_t, h, w = latents_bcthw.shape

        chunk_indices = self._create_autoregressive_segments(total_t, self.num_frame_per_block)
        num_chunks = len(chunk_indices) - 1
        kv_cache = self._initialize_kv_cache(num_chunks)

        assert (
            self.condition.shape[0] == batch_size or self.condition.shape[0] == num_chunks
        ), f"condition shape: {self.condition.shape}, batch_size: {batch_size}, num_chunks: {num_chunks}"
        if self.condition.shape[0] == batch_size:
            self.condition = self.condition.repeat_interleave(num_chunks, dim=0)
            self.mask = self.mask[None].repeat_interleave(num_chunks, dim=0) if self.mask is not None else None

        condition = self.condition
        mask = self.mask

        output = torch.zeros_like(latents_bcthw)

        for chunk_idx in range(num_chunks):
            start_f = chunk_indices[chunk_idx]
            end_f = chunk_indices[chunk_idx + 1]
            local_latent = latents_bcthw[:, :, start_f:end_f]

            chunk_condition = condition[chunk_idx].unsqueeze(0) if condition is not None else None
            chunk_mask = mask[chunk_idx] if mask is not None else None

            chunk_kv_cache = self._accumulate_kv_cache(kv_cache, chunk_idx)
            batch_size = local_latent.shape[0]
            current_num_frames = local_latent.shape[2]

            for index, current_timestep in enumerate(self.denoising_step_list):
                timestep = (
                    torch.ones(local_latent.shape[0], device=self.model_device, dtype=self.model_dtype)
                    * current_timestep
                )

                if index < len(self.denoising_step_list) - 1:
                    flow_pred, pred_x0, _ = self.generator(
                        noisy_image_or_video=local_latent,
                        condition=chunk_condition,
                        timestep=timestep,
                        start_f=start_f,
                        end_f=end_f,
                        save_kv_cache=False,
                        mask=chunk_mask,
                        kv_cache=chunk_kv_cache,
                    )
                    flow_pred = rearrange(flow_pred, "b c f h w -> b f c h w")
                    pred_x0 = rearrange(pred_x0, "b c f h w -> b f c h w")
                    next_timestep = self.denoising_step_list[index + 1]
                    local_latent = self.scheduler.add_noise(
                        pred_x0.flatten(0, 1),
                        torch.randn_like(pred_x0.flatten(0, 1)),
                        next_timestep
                        * torch.ones([batch_size * current_num_frames], device=latents.device, dtype=torch.long),
                    ).unflatten(0, pred_x0.shape[:2])
                    local_latent = rearrange(local_latent, "b f c h w -> b c f h w")

                else:
                    flow_pred, pred_x0, _ = self.generator(
                        noisy_image_or_video=local_latent,
                        condition=chunk_condition,
                        timestep=timestep,
                        start_f=start_f,
                        end_f=end_f,
                        save_kv_cache=False,
                        mask=chunk_mask,
                        kv_cache=chunk_kv_cache,
                    )
                    output[:, :, start_f:end_f] = pred_x0.to(output.device)

            latent_for_cache = output[:, :, start_f:end_f]
            timestep_zero = torch.zeros(latent_for_cache.shape[0], device=self.model_device, dtype=self.model_dtype)
            _, _, updated_kv_cache = self.generator(
                noisy_image_or_video=latent_for_cache,
                condition=chunk_condition,
                timestep=timestep_zero,
                start_f=start_f,
                end_f=end_f,
                save_kv_cache=True,
                mask=chunk_mask,
                kv_cache=chunk_kv_cache,
            )
            kv_cache[chunk_idx] = updated_kv_cache

        return output
