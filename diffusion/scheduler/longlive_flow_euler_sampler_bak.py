import os
import time
import types
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

import imageio
import torch
from einops import rearrange
from termcolor import colored

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
        pass

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
    Sana模型包装器，提供与WanDiffusionWrapper兼容的接口
    """

    def __init__(self, sana_model, flow_shift: float = 3.0):
        super().__init__()
        # 直接持有底层 SANA 模型，避免初始化完整 pipeline 造成显存占用
        self.model = sana_model
        self.flow_shift = float(flow_shift)
        self.uniform_timestep = False  # Sana支持per-frame timestep
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
        """启用梯度检查点"""
        if hasattr(self.model, "enable_gradient_checkpointing"):
            self.model.enable_gradient_checkpointing()

    # TODO 使用正确的scheduler
    def get_scheduler(self):
        """获取调度器"""
        # 参考 SANA 内部：使用 diffusers 的 FlowMatchEulerDiscreteScheduler
        return self.scheduler

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
        """
        前向传播，兼容WanDiffusionWrapper的接口
        """
        # 这里需要将Sana的接口适配到Wan的接口
        # noisy_image_or_video: (B, C, F, H, W)
        # 处理 prompt_embeds 形状：期望 (B, 1, L, C)
        if condition.dim() == 3:
            condition = condition.unsqueeze(1)
        elif condition.dim() == 2:
            condition = condition.unsqueeze(0).unsqueeze(0)

        # SANA 模型前向（支持保存/使用 KV cache）
        # SANA 原始实现用 flow matching：返回的是 flow_pred，需转成 x0 以对齐 WAN 接口
        model = self.model
        # 统一将 timestep 压缩到 [B]
        if timestep.dim() == 2:
            input_t = timestep[:, 0]
        else:
            input_t = timestep

        # if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
        #     print(f"[SanaModelWrapper] noisy_image_or_video.shape={noisy_image_or_video.shape}")
        #     print(f"[SanaModelWrapper] input_t.shape={input_t.shape}")
        #     print(f"[SanaModelWrapper] condition.shape={condition.shape}")

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

        # 兼容 diffusers 输出
        try:
            from diffusers.models.modeling_outputs import Transformer2DModelOutput

            if isinstance(model_out, Transformer2DModelOutput):
                model_out = model_out[0]
        except Exception:
            pass

        if isinstance(model_out, Transformer2DModelOutput):
            model_out = model_out[0]

        # SANA 返回的是 flow_pred，形状 (B, C, F, H, W)
        flow_pred_bcfhw = model_out
        flow_pred = rearrange(flow_pred_bcfhw, "b c f h w -> b f c h w")  # (B, F, C, H, W)
        noisy_image_or_video = rearrange(noisy_image_or_video, "b c f h w -> b f c h w")  # (B, F, C, H, W)
        pred_x0 = self._convert_flow_pred_to_x0(
            flow_pred=flow_pred.flatten(0, 1), xt=noisy_image_or_video.flatten(0, 1), timestep=input_t
        ).unflatten(
            0, flow_pred.shape[:2]
        )  # (B, F, C, H, W)
        pred_x0_bcfhw = rearrange(pred_x0, "b f c h w -> b c f h w")  # (B, C, F, H, W)

        # 对齐 WAN 接口：
        # - 常规：返回 (flow_pred, pred_x0)
        # - 当底层返回了 kv_cache（如 save_kv_cache=True）：返回 (flow_pred, pred_x0, kv_cache)
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
        """
        仅推理用的 SANA 管线：无梯度地生成整段视频。

        初始化签名与 Trainer 中的使用保持一致：
            SanaInferencePipeline(args, device, generator, text_encoder, vae)
        """
        self.generator = SanaModelWrapper(model_fn, flow_shift=flow_shift)
        self.condition = condition
        self.mask = model_kwargs.pop("mask", None)
        self.num_recache = kwargs.get("num_recache", 1)

        # timestep_shift = float(getattr(args, "timestep_shift", 3.0))
        self.scheduler = self.generator.get_scheduler()
        # hyperparams
        self.num_frame_per_block = base_chunk_frames
        self.denoising_step_list = denoising_step_list
        if len(self.denoising_step_list) > 0 and self.denoising_step_list[-1] == 0:
            self.denoising_step_list = self.denoising_step_list[:-1]
        # print(f"[SanaInferencePipeline] denoising_step_list={self.denoising_step_list}")
        # model meta
        inner = self.generator.model if hasattr(self.generator, "model") else self.generator
        try:
            p = next(inner.parameters())
            self.model_device = p.device
            self.model_dtype = p.dtype
        except Exception:
            self.model_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

        # cache helpers
        self.cached_modules = None
        self.num_model_blocks = 0
        self.num_cached_blocks = num_cached_blocks
        # print(f"[SanaInferencePipeline] num_cached_blocks={self.num_cached_blocks}")

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
    def sample(self, latents: torch.Tensor, steps: int = 4, generator=None, **kwargs):
        """
        生成完整视频。

        Args:
            noise: [B, T, C, H, W] 或 [B, C, T, H, W] 的高斯噪声 latent。
            text_prompts: 文本提示（长度=B）。
            return_latents: 为 True 返回 latent（B,T,C,H,W）；否则返回像素（B,T,C,H,W，范围0..1的-1..1归一化由上游处理）。
            initial_latent: 可选的首帧 latent，形状 [B, T0, C, H, W]（常用 T0=1）。
        Returns:
            video: 若 return_latents=True，返回 [B, T, C, H, W]；否则返回像素 [B, T, C, H, W]
            info: dict
        """
        # 标准化 latent 形状到 B,C,T,H,W
        start_time = time.time()
        if latents.dim() != 5:
            raise ValueError("noise should be a 5D tensor")

        latents_bcthw = latents
        device = self.condition.device

        batch_size, c, total_t, h, w = latents_bcthw.shape

        # 文本编码：对齐 trainer 中的构造方式（支持 chi_prompt、select_index、dtype/device 对齐）

        # autoregressive 切分
        chunk_indices = self._create_autoregressive_segments(total_t, self.num_frame_per_block)
        # print(f"[SanaInferencePipeline] chunk_indices={chunk_indices}")
        num_chunks = len(chunk_indices) - 1
        kv_cache = self._initialize_kv_cache(num_chunks)

        if self.num_recache > 0 and self.condition.shape[0] >= batch_size and self.condition.shape[0] <= num_chunks:
            # import ipdb; ipdb.set_trace()
            assert num_chunks % self.condition.shape[0] == 0
            num_segments = self.condition.shape[0]
            per_chunk_conditions = []
            per_chunk_masks = []
            for i in range(num_segments):
                for j in range(num_chunks // num_segments):
                    per_chunk_conditions.append(self.condition[i])
                    per_chunk_masks.append(self.mask[i : i + 1])  # 1,L
            self.condition = torch.stack(per_chunk_conditions, dim=0)  # (num_chunks, C, L)
            self.mask = torch.stack(per_chunk_masks, dim=0)  # (num_chunks, 1, L)
            should_switch_indices = (torch.arange(num_chunks) % num_segments == 0).tolist()  # bool list, (num_chunks,)
            should_switch_indices[0] = False

        assert (
            self.condition.shape[0] == batch_size or self.condition.shape[0] == num_chunks
        ), f"condition shape: {self.condition.shape}, batch_size: {batch_size}, num_chunks: {num_chunks}"
        if self.condition.shape[0] == batch_size:
            self.condition = self.condition.repeat_interleave(num_chunks, dim=0)
            self.mask = self.mask[None].repeat_interleave(num_chunks, dim=0) if self.mask is not None else None

        condition = self.condition
        mask = self.mask

        # 输出 latent 容器（与输入相同 dtype/device）
        output = torch.zeros_like(latents_bcthw)

        # 去噪步数
        steps = max(1, len(self.denoising_step_list))
        # print(colored(f"[SanaInferencePipeline] num_chunks={num_chunks}, steps={steps}", "red"))
        # 逐 chunk 生成
        for chunk_idx in range(num_chunks):
            start_f = chunk_indices[chunk_idx]
            end_f = chunk_indices[chunk_idx + 1]
            local_latent = latents_bcthw[:, :, start_f:end_f]

            chunk_condition = condition[chunk_idx].unsqueeze(0) if condition is not None else None
            chunk_mask = mask[chunk_idx] if mask is not None else None

            if self.num_recache > 0 and should_switch_indices[chunk_idx]:
                # import ipdb; ipdb.set_trace()
                num_recache = min(self.num_recache, chunk_idx)
                prev_start_f = chunk_indices[chunk_idx - num_recache]
                prev_end_f = chunk_indices[chunk_idx]
                prev_latent_for_cache = output[:, :, prev_start_f:prev_end_f]
                prev_accumulated_kv = self._accumulate_kv_cache(kv_cache, chunk_idx - num_recache)
                timestep_zero_prev = torch.zeros(
                    prev_latent_for_cache.shape[0], device=self.model_device, dtype=self.model_dtype
                )
                _, _, updated_prev_kv = self.generator(
                    noisy_image_or_video=prev_latent_for_cache,
                    condition=chunk_condition,
                    timestep=timestep_zero_prev,
                    start_f=prev_start_f,
                    end_f=prev_end_f,
                    save_kv_cache=True,
                    mask=chunk_mask,
                    kv_cache=prev_accumulated_kv,
                )
                for recache_idx in range(num_recache):
                    for block_id in range(self.num_model_blocks):
                        kv_cache[chunk_idx - recache_idx][block_id][0] = updated_prev_kv[block_id][0] / num_recache
                        kv_cache[chunk_idx - recache_idx][block_id][1] = updated_prev_kv[block_id][1] / num_recache
                        kv_cache[chunk_idx - recache_idx][block_id][2] = updated_prev_kv[block_id][2]

            # 取累计后的 KV cache（重算式累计）
            chunk_kv_cache = self._accumulate_kv_cache(kv_cache, chunk_idx)
            # chunk_kv_cache = kv_cache[chunk_idx] # no accumulate, no update
            batch_size = local_latent.shape[0]
            current_num_frames = local_latent.shape[2]
            # 全步 xt-style 推进：xt_{i+1} = x0_i + sigma_{i+1} * flow_pred_i
            for index, current_timestep in enumerate(self.denoising_step_list):
                # print(f"[SanaInferencePipeline] step_idx={step_idx}, t={t}")
                timestep = (
                    torch.ones(local_latent.shape[0], device=self.model_device, dtype=self.model_dtype)
                    * current_timestep
                )
                # import ipdb; ipdb.set_trace()
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
                    )  # (B, C, F, H, W)
                    # import ipdb; ipdb.set_trace()
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
                    # 最后一步写回输出
                    output[:, :, start_f:end_f] = pred_x0.to(output.device)

            # 保存并更新 KV 缓存，供后续 chunk 使用
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

        # 输出
        info = {
            "total_frames": total_t,
            "num_chunks": num_chunks,
            "chunk_indices": chunk_indices,
        }
        dit_end_time = time.time()
        # print(f"[SanaInferencePipeline] dit_time={dit_end_time - start_time}")

        return output  # B,C,T,H,W
