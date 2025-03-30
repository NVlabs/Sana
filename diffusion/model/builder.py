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
from diffusers import AutoencoderDC
from diffusers.models import AutoencoderKL
from mmcv import Registry
from termcolor import colored
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, T5EncoderModel, T5Tokenizer
from transformers import logging as transformers_logging

from diffusion.model.dc_ae.efficientvit.ae_model_zoo import DCAE_HF
from diffusion.model.utils import set_fp32_attention, set_grad_checkpoint

MODELS = Registry("models")

transformers_logging.set_verbosity_error()


def build_model(cfg, use_grad_checkpoint=False, use_fp32_attention=False, gc_step=1, **kwargs):
    if isinstance(cfg, str):
        cfg = dict(type=cfg)
    model = MODELS.build(cfg, default_args=kwargs)

    if use_grad_checkpoint:
        set_grad_checkpoint(model, gc_step=gc_step)
    if use_fp32_attention:
        set_fp32_attention(model)
    return model


def get_tokenizer_and_text_encoder(name="T5", device="cuda"):
    text_encoder_dict = {
        "T5": "DeepFloyd/t5-v1_1-xxl",
        "T5-small": "google/t5-v1_1-small",
        "T5-base": "google/t5-v1_1-base",
        "T5-large": "google/t5-v1_1-large",
        "T5-xl": "google/t5-v1_1-xl",
        "T5-xxl": "google/t5-v1_1-xxl",
        "gemma-2b": "google/gemma-2b",
        "gemma-2b-it": "google/gemma-2b-it",
        "gemma-2-2b": "google/gemma-2-2b",
        "gemma-2-2b-it": "Efficient-Large-Model/gemma-2-2b-it",
        "gemma-2-9b": "google/gemma-2-9b",
        "gemma-2-9b-it": "google/gemma-2-9b-it",
        "Qwen2-0.5B-Instruct": "Qwen/Qwen2-0.5B-Instruct",
        "Qwen2-1.5B-Instruct": "Qwen/Qwen2-1.5B-Instruct",
    }
    assert name in list(text_encoder_dict.keys()), f"not support this text encoder: {name}"
    if "T5" in name:
        tokenizer = T5Tokenizer.from_pretrained(text_encoder_dict[name])
        text_encoder = T5EncoderModel.from_pretrained(text_encoder_dict[name], torch_dtype=torch.float16).to(device)
    elif "gemma" in name or "Qwen" in name:
        tokenizer = AutoTokenizer.from_pretrained(text_encoder_dict[name])
        tokenizer.padding_side = "right"
        text_encoder = (
            AutoModelForCausalLM.from_pretrained(text_encoder_dict[name], torch_dtype=torch.bfloat16)
            .get_decoder()
            .to(device)
        )
    else:
        print("error load text encoder")
        exit()

    return tokenizer, text_encoder


def get_vae(name, model_path, device="cuda"):
    if name == "sdxl" or name == "sd3":
        vae = AutoencoderKL.from_pretrained(model_path).to(device).to(torch.float16)
        if name == "sdxl":
            vae.config.shift_factor = 0
        return vae
    elif "dc-ae" in name:
        print(colored(f"[DC-AE] Loading model from {model_path}", attrs=["bold"]))
        dc_ae = DCAE_HF.from_pretrained(model_path).to(device).eval()
        return dc_ae
    elif "AutoencoderDC" in name:
        print(colored(f"[AutoencoderDC] Loading model from {model_path}", attrs=["bold"]))
        dc_ae = AutoencoderDC.from_pretrained(model_path).to(device).eval()
        return dc_ae
    else:
        print("error load vae")
        exit()


def vae_encode(name, vae, images, sample_posterior, device):
    if name == "sdxl" or name == "sd3":
        posterior = vae.encode(images.to(device)).latent_dist
        if sample_posterior:
            z = posterior.sample()
        else:
            z = posterior.mode()
        z = (z - vae.config.shift_factor) * vae.config.scaling_factor
    elif "dc-ae" in name:
        ae = vae
        scaling_factor = ae.cfg.scaling_factor if ae.cfg.scaling_factor else 0.41407
        z = ae.encode(images.to(device))
        z = z * scaling_factor
    elif "AutoencoderDC" in name:
        ae = vae
        scaling_factor = ae.config.scaling_factor if ae.config.scaling_factor else 0.41407
        z = ae.encode(images.to(device))[0]
        z = z * scaling_factor
    else:
        print("error load vae")
        exit()
    return z


def vae_decode(name, vae, latent):
    if name == "sdxl" or name == "sd3":
        latent = (latent.detach() / vae.config.scaling_factor) + vae.config.shift_factor
        samples = vae.decode(latent).sample
    elif "dc-ae" in name:
        ae = vae
        vae_scale_factor = (
            2 ** (len(ae.config.encoder_block_out_channels) - 1)
            if hasattr(ae, "config") and ae.config is not None
            else 32
        )
        scaling_factor = ae.cfg.scaling_factor if ae.cfg.scaling_factor else 0.41407
        if latent.shape[-1] * vae_scale_factor > 4000 or latent.shape[-2] * vae_scale_factor > 4000:
            from patch_conv import convert_model

            ae = convert_model(ae, splits=4)
        samples = ae.decode(latent.detach() / scaling_factor)
    elif "AutoencoderDC" in name:
        ae = vae
        scaling_factor = ae.config.scaling_factor if ae.config.scaling_factor else 0.41407
        try:
            samples = ae.decode(latent / scaling_factor, return_dict=False)[0]
        except torch.cuda.OutOfMemoryError as e:
            print("Warning: Ran out of memory when regular VAE decoding, retrying with tiled VAE decoding.")
            ae.enable_tiling(tile_sample_min_height=1024, tile_sample_min_width=1024)
            samples = ae.decode(latent / scaling_factor, return_dict=False)[0]
    else:
        print("error load vae")
        exit()
    return samples
