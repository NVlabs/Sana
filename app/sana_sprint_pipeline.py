# Copyright 2024 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

import os
import warnings
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import pyrallis
import torch
import torch.nn as nn
from tqdm import tqdm

warnings.filterwarnings("ignore")  # ignore warning
os.environ["DISABLE_XFORMERS"] = "1"


from diffusion import SCMScheduler
from diffusion.data.datasets.utils import ASPECT_RATIO_512_TEST, ASPECT_RATIO_1024_TEST
from diffusion.model.builder import build_model, get_tokenizer_and_text_encoder, get_vae, vae_decode
from diffusion.model.utils import get_weight_dtype, prepare_prompt_ar, resize_and_crop_tensor
from diffusion.utils.config import SanaConfig, model_init_config
from diffusion.utils.logger import get_root_logger
from tools.download import find_model


def classify_height_width_bin(height: int, width: int, ratios: dict) -> Tuple[int, int]:
    """Returns binned height and width."""
    ar = float(height / width)
    closest_ratio = min(ratios.keys(), key=lambda ratio: abs(float(ratio) - ar))
    default_hw = ratios[closest_ratio]
    return int(default_hw[0]), int(default_hw[1])


@dataclass
class SanaSprintInference(SanaConfig):
    config: Optional[str] = "configs/sana_sprint_config/1024ms/SanaSprint_1600M_1024px_allqknorm_bf16_scm_ladd.yaml"
    model_path: str = field(
        default="output/Sana_D20/SANA.pth", metadata={"help": "Path to the model file (positional)"}
    )
    output: str = "./output"
    bs: int = 1
    image_size: int = 1024
    cfg_scale: float = 5.0
    seed: int = 42
    step: int = -1
    max_timesteps: Optional[float] = 1.57080
    intermediate_timesteps: Optional[float] = 1.3
    timesteps: Optional[List[float]] = None
    custom_image_size: Optional[int] = None
    shield_model_path: str = field(
        default="google/shieldgemma-2b",
        metadata={"help": "The path to shield model, we employ ShieldGemma-2B by default."},
    )


class SanaSprintPipeline(nn.Module):
    def __init__(
        self,
        config: Optional[
            str
        ] = "configs/sana_sprint_config/1024ms/SanaSprint_1600M_1024px_allqknorm_bf16_scm_ladd.yaml",
    ):
        super().__init__()
        config = pyrallis.load(SanaSprintInference, open(config))
        self.args = self.config = config

        # set some hyper-parameters
        self.image_size = self.config.model.image_size

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        logger = get_root_logger()
        self.logger = logger
        self.progress_fn = lambda progress, desc: None

        self.latent_size = self.image_size // config.vae.vae_downsample_rate
        self.max_sequence_length = config.text_encoder.model_max_length
        self.guidance_type = "classifier-free"

        weight_dtype = get_weight_dtype(config.model.mixed_precision)
        self.weight_dtype = weight_dtype
        self.vae_dtype = get_weight_dtype(config.vae.weight_dtype)

        self.base_ratios = eval(f"ASPECT_RATIO_{self.image_size}_TEST")
        self.vis_sampler = self.config.scheduler.vis_sampler
        logger.info(f"Sampler {self.vis_sampler}")
        logger.info(f"Inference with {self.weight_dtype}")

        # 1. build vae and text encoder
        self.vae = self.build_vae(config.vae)
        self.tokenizer, self.text_encoder = self.build_text_encoder(config.text_encoder)

        # 2. build Sana model
        self.model = self.build_sana_model(config).to(self.device)

    def build_vae(self, config):
        vae = get_vae(config.vae_type, config.vae_pretrained, self.device).to(self.vae_dtype)
        return vae

    def build_text_encoder(self, config):
        tokenizer, text_encoder = get_tokenizer_and_text_encoder(name=config.text_encoder_name, device=self.device)
        return tokenizer, text_encoder

    def build_sana_model(self, config):
        # model setting
        model_kwargs = model_init_config(config, latent_size=self.latent_size)
        model = build_model(
            config.model.model,
            use_fp32_attention=config.model.get("fp32_attention", False) and config.model.mixed_precision != "bf16",
            cfg_embed=config.model.cfg_embed,
            cfg_embed_scale=config.model.cfg_embed_scale,
            **model_kwargs,
        )
        self.logger.info(f"use_fp32_attention: {model.fp32_attention}")
        self.logger.info(
            f"{model.__class__.__name__}:{config.model.model},"
            f"Model Parameters: {sum(p.numel() for p in model.parameters()):,}"
        )
        return model

    def from_pretrained(self, model_path):
        state_dict = find_model(model_path)
        state_dict = state_dict.get("state_dict", state_dict)
        if "pos_embed" in state_dict:
            del state_dict["pos_embed"]
        missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
        self.model.eval().to(self.weight_dtype)

        self.logger.info("Generating sample from ckpt: %s" % model_path)
        self.logger.warning(f"Missing keys: {missing}")
        self.logger.warning(f"Unexpected keys: {unexpected}")

    def register_progress_bar(self, progress_fn=None):
        self.progress_fn = progress_fn if progress_fn is not None else self.progress_fn

    @torch.inference_mode()
    def forward(
        self,
        prompt=None,
        height=1024,
        width=1024,
        num_inference_steps=20,
        guidance_scale=5,
        num_images_per_prompt=1,
        generator=None,
        latents=None,
        use_resolution_binning=True,
    ):
        self.ori_height, self.ori_width = height, width
        if use_resolution_binning:
            self.height, self.width = classify_height_width_bin(height, width, ratios=self.base_ratios)
        else:
            self.height, self.width = height, width
        self.latent_size_h, self.latent_size_w = (
            self.height // self.config.vae.vae_downsample_rate,
            self.width // self.config.vae.vae_downsample_rate,
        )
        sigma_data = self.config.scheduler.sigma_data

        # 1. pre-compute embedding
        if prompt is None:
            prompt = [""]
        prompts = prompt if isinstance(prompt, list) else [prompt]
        samples = []

        # set scheduler
        if self.vis_sampler == "scm":
            scheduler = SCMScheduler()
        else:
            raise ValueError(f"Unsupported sampling algorithm: {self.vis_sampler}")

        scheduler.set_timesteps(
            num_inference_steps=num_inference_steps,
            max_timesteps=self.config.max_timesteps,
            intermediate_timesteps=self.config.intermediate_timesteps,
            timesteps=self.config.timesteps,
        )
        timesteps = scheduler.timesteps

        for prompt in prompts:
            # data prepare
            prompts, hw, ar = (
                [],
                torch.tensor([[self.image_size, self.image_size]], dtype=torch.float, device=self.device).repeat(
                    num_images_per_prompt, 1
                ),
                torch.tensor([[1.0]], device=self.device).repeat(num_images_per_prompt, 1),
            )

            for _ in range(num_images_per_prompt):
                prompts.append(prepare_prompt_ar(prompt, self.base_ratios, device=self.device, show=False)[0].strip())

            with torch.no_grad():
                # prepare text feature
                if not self.config.text_encoder.chi_prompt:
                    max_length_all = self.config.text_encoder.model_max_length
                    prompts_all = prompts
                else:
                    chi_prompt = "\n".join(self.config.text_encoder.chi_prompt)
                    prompts_all = [chi_prompt + prompt for prompt in prompts]
                    num_chi_prompt_tokens = len(self.tokenizer.encode(chi_prompt))
                    max_length_all = (
                        num_chi_prompt_tokens + self.config.text_encoder.model_max_length - 2
                    )  # magic number 2: [bos], [_]

                caption_token = self.tokenizer(
                    prompts_all,
                    max_length=max_length_all,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt",
                ).to(device=self.device)
                select_index = [0] + list(range(-self.config.text_encoder.model_max_length + 1, 0))
                caption_embs = self.text_encoder(caption_token.input_ids, caption_token.attention_mask)[0][:, None][
                    :, :, select_index
                ].to(self.weight_dtype)
                emb_masks = caption_token.attention_mask[:, select_index]

                n = len(prompts)
                if latents is None:
                    latents = (
                        torch.randn(
                            n,
                            self.config.vae.vae_latent_dim,
                            self.latent_size_h,
                            self.latent_size_w,
                            generator=generator,
                            device=self.device,
                        )
                        * sigma_data
                    )
                else:
                    latents = latents.to(self.device)
                model_kwargs = dict(
                    data_info={
                        "img_hw": hw,
                        "aspect_ratio": ar,
                        "cfg_scale": torch.tensor([guidance_scale] * latents.shape[0]).to(self.device),
                    },
                    mask=emb_masks,
                )

                #  sCM MultiStep Sampling Loop:
                for i, t in tqdm(list(enumerate(timesteps[:-1]))):

                    timestep = t.expand(latents.shape[0]).to(self.device)

                    # model prediction
                    model_pred = sigma_data * self.model(
                        latents / sigma_data,
                        timestep,
                        caption_embs,
                        **model_kwargs,
                    )

                    # compute the previous noisy sample x_t -> x_t-1
                    latents, denoised = scheduler.step(
                        model_pred, i, t, latents, generator=generator, return_dict=False
                    )

            with torch.no_grad():
                sample = (denoised / sigma_data).to(self.vae_dtype)
                sample = vae_decode(self.config.vae.vae_type, self.vae, sample)
                torch.cuda.empty_cache()

            if use_resolution_binning:
                sample = resize_and_crop_tensor(sample, self.ori_width, self.ori_height)
            samples.append(sample)

            return sample

        return samples
