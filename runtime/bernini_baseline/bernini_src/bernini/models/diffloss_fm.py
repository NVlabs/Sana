# Copyright (c) 2026 Bytedance Ltd. and/or its affiliate
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
import functools
import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.checkpoint import checkpoint
from tqdm import tqdm
from transformers.utils import logging

try:
    from torch.utils.checkpoint import (
        _pt2_selective_checkpoint_context_fn_gen as create_selective_checkpoint_contexts,
    )
except ImportError:
    from torch.utils.checkpoint import create_selective_checkpoint_contexts


def policy_fn(ctx, op, *args, **kwargs):
    return False


recompute_all_context_fn = functools.partial(create_selective_checkpoint_contexts, policy_fn)


class DiffLoss_FM(nn.Module):
    """Diffusion Loss"""

    def __init__(
        self,
        target_channels,
        z_channels,
        depth=16,
        width=1536,
        diff_net="SimpleMLPAdaLN",
        scheduler_type="FlowMatchScheduler",
        # params for diffusion
        num_inference_steps=100,
        num_train_timesteps=1000,
        shift=2.0,
        sigma_max=1.0,
        sigma_min=0.003 / 1.002,
        extra_one_step=False,
        # params for train
        grad_checkpointing=False,
        # params for sample
        diffusion_batch_mul=1,
    ):
        super().__init__()
        self.diffusion_batch_mul = diffusion_batch_mul
        self.in_channels = target_channels
        out_channels = target_channels
        if diff_net == "SimpleMLPAdaLN":
            self.net = SimpleMLPAdaLN(
                in_channels=target_channels,
                model_channels=width,
                out_channels=out_channels,  # for vlb loss
                z_channels=z_channels,
                num_res_blocks=depth,
                grad_checkpointing=grad_checkpointing,
            )
        else:
            raise NotImplementedError

        self.num_inference_steps = num_inference_steps
        if scheduler_type == "FlowMatchScheduler":
            from .scheduler import FlowMatchScheduler

            self.scheduler = FlowMatchScheduler(
                num_inference_steps=num_inference_steps,
                num_train_timesteps=num_train_timesteps,
                shift=shift,
                sigma_max=sigma_max,
                sigma_min=sigma_min,
                extra_one_step=extra_one_step,
            )
        else:
            raise NotImplementedError

        # default set to train mode; alter to infer mode in infer_edit func
        try:
            self.scheduler.set_timesteps(num_train_timesteps, training=True)
        except Exception:
            self.scheduler.set_timesteps(num_train_timesteps)

    def forward(self, target, z, mask=None):
        # refer to: https://github.com/ByteDance-Seed/VeOmni/blob/
        # c93f4471a75d7478e41c31b2648441a3f339a1d7/tasks/omni/train_wan.py#L335

        # multi noise trick
        seq_len, _ = target.shape
        
        z = z.reshape(seq_len, -1).repeat(self.diffusion_batch_mul, 1)
        target = target.reshape(seq_len, -1).repeat(self.diffusion_batch_mul, 1)

        # formal calculate loss
        x = target  # seq_len, dim
        timestep_id = torch.randint(0, self.scheduler.num_train_timesteps, (x.shape[0],))
        timestep = self.scheduler.timesteps[timestep_id].to(dtype=z.dtype, device=z.device)
        timestep = timestep.to(x.dtype)

        # sample noise
        noise = torch.randn_like(x)
        # add noise to latents
        x_t = self.scheduler.add_noise(x, noise, timestep).to(z.dtype)
        # Predict noise
        self.net = self.net.to(z.dtype)
        model_pred = self.net(x_t, timestep, c=z)
        # Compute loss
        model_target = self.scheduler.training_target(x, noise, timestep)
        weights = self.scheduler.training_weight(timestep).to(x.device)
        loss = F.mse_loss(model_pred.float(), model_target.float(), reduction="none")
        loss = loss.view(x.shape[0], -1).mean(dim=1) * weights
        loss = loss.view(self.diffusion_batch_mul, seq_len).mean(dim=0)

        if mask is not None:
            loss = loss * mask
        return loss

    def sample(self, z, cfg, num_inference_steps, img_cfg=None, verbose=True):
        # diffusion loss sampling
        # refer to: https://github.com/mi804/DiffSynth-Studio/blob/
        # c8e9a9619638736453f6bba29072e54e292d9fe3/diffsynth/pipelines/flux_image_new.py#L395
        device = z.device

        if img_cfg is not None and cfg > 1.0:
            noise = torch.randn(z.shape[0] // 3, self.in_channels).to(device)
            noise = torch.cat([noise, noise, noise], dim=0)
            model_kwargs = dict(c=z, txt_cfg_scale=cfg, img_cfg_scale=img_cfg)
            sample_fn = self.net.forward_with_txt_img_cfg

        elif cfg > 1.0:
            noise = torch.randn(z.shape[0] // 2, self.in_channels).to(device)
            noise = torch.cat([noise, noise], dim=0)
            model_kwargs = dict(c=z, cfg_scale=cfg)
            sample_fn = self.net.forward_with_cfg
        
        else:
            noise = torch.randn(z.shape[0], self.in_channels).to(device)
            model_kwargs = dict(c=z)
            sample_fn = self.net.forward

        # Prepare timesteps
        try:
            self.scheduler.set_timesteps(num_inference_steps, training=False)
        except Exception:
            self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps.to(device)

        # Denoising loop
        samples = noise.to(z.dtype)
        progress_bar = (
            tqdm(timesteps, desc=f"Vit diffusion with cfg={cfg}") if verbose else None
        )
        for i, t in enumerate(timesteps):
            timestep = t.unsqueeze(0).to(dtype=z.dtype, device=device)

            # Inference
            noise_pred = sample_fn(x=samples, t=timestep, **model_kwargs)

            samples = self.scheduler.step(model_output=noise_pred, timestep=timestep, sample=samples)
            if not isinstance(samples, torch.Tensor):
                samples = samples.prev_sample

            if verbose:
                progress_bar.update(1)
        return samples


def modulate(x, shift, scale):
    return x * (1 + scale) + shift


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half).to(
            device=t.device
        )
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq.to(t.dtype))
        return t_emb


# Copied from transformers.models.llama.modeling_llama.LlamaRMSNorm with Llama->Qwen2
class Qwen2RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        Qwen2RMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        out = self.weight * hidden_states.to(input_dtype)
        return out


class ResBlock(nn.Module):
    """
    A residual block that can optionally change the number of channels.
    :param channels: the number of input channels.
    """

    def __init__(self, channels):
        super().__init__()
        self.channels = channels

        self.in_ln = nn.LayerNorm(channels, eps=1e-6)

        USE_MLP_NORM = os.environ.get('USE_MLP_NORM_IN_RESBLOCK_OF_FM', 'False').lower()
        if USE_MLP_NORM in ('true', '1'):
            self.mlp = nn.Sequential(
                nn.Linear(channels, channels, bias=True),
                nn.LayerNorm(channels, eps=1e-6),
                nn.SiLU(),
                nn.Linear(channels, channels, bias=True),
            )
        else:
            self.mlp = nn.Sequential(
                nn.Linear(channels, channels, bias=True),
                nn.SiLU(),
                nn.Linear(channels, channels, bias=True),
            )

        self.out_norm = None
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(channels, 3 * channels, bias=True))

    def forward(self, x, y):
        shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(y).chunk(3, dim=-1)
        h = modulate(self.in_ln(x), shift_mlp, scale_mlp)
        h = self.mlp(h)
        out = gate_mlp * h
        if self.out_norm is not None:
            out = self.out_norm(out)
        return x + out


class FinalLayer(nn.Module):
    """
    The final layer adopted from DiT.
    """

    def __init__(self, model_channels, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(model_channels, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(model_channels, out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(model_channels, 2 * model_channels, bias=True))

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class SimpleLinear(nn.Module):
    """
    :param in_channels: channels in the input Tensor.
    :param model_channels: base channel count for the model.
    :param out_channels: channels in the output Tensor.
    :param z_channels: channels in the condition.
    """

    def __init__(self, in_channels, out_channels, z_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.z_channels = z_channels

        self.Linear = nn.Linear(in_channels + z_channels, out_channels)
        self.initialize_weights()

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

    def forward(self, x, t, c):
        """
        Apply the model to an input batch.
        :param x: an [N x C] Tensor of inputs.
        :param t: a 1-D batch of timesteps.
        :param c: conditioning from AR transformer.
        :return: an [N x C] Tensor of outputs.
        """
        z = torch.cat([x, c], dim=-1)
        pred = self.Linear(z)
        return pred


class SimpleMLPAdaLN(nn.Module):
    """
    The MLP for Diffusion Loss.
    :param in_channels: channels in the input Tensor.
    :param model_channels: base channel count for the model.
    :param out_channels: channels in the output Tensor.
    :param z_channels: channels in the condition.
    :param num_res_blocks: number of residual blocks per downsample.
    """

    def __init__(self, in_channels, model_channels, out_channels, z_channels, num_res_blocks, grad_checkpointing=False):
        super().__init__()

        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.grad_checkpointing = grad_checkpointing

        self.time_embed = TimestepEmbedder(model_channels)
        self.cond_embed = nn.Linear(z_channels, model_channels)

        self.input_proj = nn.Linear(in_channels, model_channels)

        res_blocks = []
        for i in range(num_res_blocks):
            res_blocks.append(
                ResBlock(
                    model_channels,
                )
            )

        self.res_blocks = nn.ModuleList(res_blocks)
        self.final_layer = FinalLayer(model_channels, out_channels)

        self.initialize_weights()

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # Initialize timestep embedding MLP
        nn.init.normal_(self.time_embed.mlp[0].weight, std=0.02)
        nn.init.normal_(self.time_embed.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers
        for block in self.res_blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def forward(self, x, t, c):
        """
        Apply the model to an input batch.
        :param x: an [N x C] Tensor of inputs.
        :param t: a 1-D batch of timesteps.
        :param c: conditioning from AR transformer.
        :return: an [N x C] Tensor of outputs.
        """
        x = self.input_proj(x)
        t = self.time_embed(t)
        c = self.cond_embed(c)

        y = t + c

        if self.grad_checkpointing and not torch.jit.is_scripting():
            for block in self.res_blocks:
                x = checkpoint(block, x, y, use_reentrant=False, context_fn=recompute_all_context_fn)

            return checkpoint(
                self.final_layer,
                x,
                y,
                use_reentrant=False,
                context_fn=recompute_all_context_fn,
            )
        else:
            for block in self.res_blocks:
                x = block(x, y)

            return self.final_layer(x, y)

    def forward_with_cfg(self, x, t, c, cfg_scale):
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = self.forward(combined, t, c)
        eps, rest = model_out[:, : self.in_channels], model_out[:, self.in_channels :]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)
    
    def forward_with_txt_img_cfg(self, x, t, c, txt_cfg_scale, img_cfg_scale):
        part = x[: len(x) // 3]
        combined = torch.cat([part, part, part], dim=0)
        model_out = self.forward(combined, t, c)
        
        eps, rest = model_out[:, : self.in_channels], model_out[:, self.in_channels :]
        cond_eps, uncond_eps, imgcond_eps = torch.split(eps, len(eps) // 3, dim=0)
        part_eps = uncond_eps + \
            img_cfg_scale * (imgcond_eps - uncond_eps) + \
            txt_cfg_scale * (cond_eps - imgcond_eps)
        
        eps = torch.cat([part_eps, part_eps, part_eps], dim=0)
        return torch.cat([eps, rest], dim=1)
