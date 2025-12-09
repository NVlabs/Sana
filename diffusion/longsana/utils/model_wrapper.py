import types
from typing import List, Optional

import imageio
import torch
import torch.nn.functional as F
from einops import rearrange
from termcolor import colored

from diffusion.model.builder import build_model, get_tokenizer_and_text_encoder, get_vae, vae_decode, vae_encode
from diffusion.model.utils import get_weight_dtype

from .scheduler import FlowMatchScheduler, SchedulerInterface


class SanaModelWrapper(torch.nn.Module):
    def __init__(self, sana_model, flow_shift: float = 3.0):
        super().__init__()
        self.model = sana_model
        self.flow_shift = float(flow_shift)
        self.uniform_timestep = False
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

    def get_scheduler(self):
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

        # noisy_image_or_video: (B, C, F, H, W)
        # Process prompt_embeds shape: expected (B, 1, L, C)
        if condition.dim() == 3:
            condition = condition.unsqueeze(1)
        elif condition.dim() == 2:
            condition = condition.unsqueeze(0).unsqueeze(0)

        # SANA model forward (supports saving/using KV cache)
        # SANA original implementation uses flow matching: returns flow_pred, need to convert to x0 to align with WAN interface
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

        # Compatible with diffusers output
        try:
            from diffusers.models.modeling_outputs import Transformer2DModelOutput

            if isinstance(model_out, Transformer2DModelOutput):
                model_out = model_out[0]
        except Exception:
            pass

        if isinstance(model_out, Transformer2DModelOutput):
            model_out = model_out[0]

        # B, C, F, H, W
        flow_pred_bcfhw = model_out
        flow_pred = rearrange(flow_pred_bcfhw, "b c f h w -> b f c h w")  # (B, F, C, H, W)
        noisy_image_or_video = rearrange(noisy_image_or_video, "b c f h w -> b f c h w")  # (B, F, C, H, W)
        pred_x0 = self._convert_flow_pred_to_x0(
            flow_pred=flow_pred.flatten(0, 1), xt=noisy_image_or_video.flatten(0, 1), timestep=input_t
        ).unflatten(
            0, flow_pred.shape[:2]
        )  # (B, F, C, H, W)
        pred_x0_bcfhw = rearrange(pred_x0, "b f c h w -> b c f h w")  # (B, C, F, H, W)

        return flow_pred_bcfhw, pred_x0_bcfhw, kv_cache_ret


class SanaTextEncoder(torch.nn.Module):
    def __init__(self, sana_cfg, device: torch.device, dtype: torch.dtype = torch.float32):
        super().__init__()
        self.device = device
        self.cfg = sana_cfg
        self.out_dtype = dtype
        name = sana_cfg.text_encoder.text_encoder_name
        self.tokenizer, self.text_encoder = get_tokenizer_and_text_encoder(name=name, device=device)
        self.text_encoder.eval().requires_grad_(False)

    def forward_chi(self, text_prompts: List[str], use_chi_prompt: bool = True) -> dict:
        if not isinstance(text_prompts, list):
            text_prompts = [text_prompts]
        chi_list = getattr(self.cfg.text_encoder, "chi_prompt", None) if use_chi_prompt else None
        if chi_list and len(chi_list) > 0:
            chi_prompt = "\n".join(chi_list)
            prompts_all = [chi_prompt + t for t in text_prompts]
            num_chi_tokens = len(self.tokenizer.encode(chi_prompt))
            max_length_all = num_chi_tokens + self.cfg.text_encoder.model_max_length - 2
        else:
            prompts_all = text_prompts
            max_length_all = self.cfg.text_encoder.model_max_length

        tokens = self.tokenizer(
            prompts_all,
            max_length=max_length_all,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).to(device=self.device)
        select_index = [0] + list(range(-self.cfg.text_encoder.model_max_length + 1, 0))
        embs_full = self.text_encoder(tokens.input_ids, tokens.attention_mask)[0]
        embs = embs_full[:, None][:, :, select_index].squeeze(1)
        embs = embs.to(device=self.device, dtype=self.out_dtype)
        emb_masks = tokens.attention_mask[:, select_index]
        return {"prompt_embeds": embs, "mask": emb_masks}

    def forward(self, text_prompts: List[str]) -> dict:
        max_len = self.cfg.text_encoder.model_max_length
        tokens = self.tokenizer(
            text_prompts,
            max_length=max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).to(self.device)
        with torch.no_grad():
            embs_full = self.text_encoder(tokens.input_ids, tokens.attention_mask)[0]

        select_index = [0] + list(range(-max_len + 1, 0))
        embs = embs_full[:, None][:, :, select_index].squeeze(1)
        embs = embs.to(device=self.device, dtype=self.out_dtype)
        emb_masks = tokens.attention_mask[:, select_index]
        return {"prompt_embeds": embs, "mask": emb_masks}


class SanaVAEWrapper(torch.nn.Module):
    def __init__(self, sana_cfg, device: torch.device, dtype: torch.dtype):
        super().__init__()
        self.device = device
        self.dtype = dtype
        self.cfg = sana_cfg
        self.vae_name = sana_cfg.vae.vae_type
        try:
            self.vae_dtype = get_weight_dtype(sana_cfg.vae.weight_dtype)
        except Exception:
            self.vae_dtype = dtype
        self.vae = get_vae(
            self.vae_name, sana_cfg.vae.vae_pretrained, device=device, dtype=self.vae_dtype, config=sana_cfg.vae
        )

    def encode_to_latent(self, pixel: torch.Tensor) -> torch.Tensor:
        pixel_bcthw = pixel
        latent_bcthw = vae_encode(self.vae_name, self.vae, pixel_bcthw, device=self.device)
        return latent_bcthw

    def decode_to_pixel(self, latent: torch.Tensor, use_cache: bool = False) -> torch.Tensor:
        latent_bcthw = latent
        if latent_bcthw.dim() != 5:
            raise ValueError("latent must be a 5D tensor [B, C, T, H, W]")

        latent_bcthw = latent_bcthw.to(device=self.device, dtype=self.vae_dtype)
        pixel_bcthw = vae_decode(self.vae_name, self.vae, latent_bcthw)
        if isinstance(pixel_bcthw, (list, tuple)):
            if len(pixel_bcthw) == 0:
                raise RuntimeError("vae_decode returned empty list/tuple")
            if torch.is_tensor(pixel_bcthw[0]):
                pixel_bcthw = torch.stack(pixel_bcthw, dim=0)
            else:
                pixel_bcthw = torch.tensor(pixel_bcthw)
        return pixel_bcthw.to(device=self.device, dtype=torch.float32)
