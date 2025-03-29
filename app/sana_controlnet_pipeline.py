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
import warnings
from dataclasses import dataclass, field
from typing import Optional, Tuple

import cv2
import numpy as np
import pyrallis
import torch
import torch.nn as nn
from PIL import Image

warnings.filterwarnings("ignore")  # ignore warning


from diffusion import DPMS, FlowEuler
from diffusion.data.datasets.utils import (
    ASPECT_RATIO_512_TEST,
    ASPECT_RATIO_1024_TEST,
    ASPECT_RATIO_2048_TEST,
    ASPECT_RATIO_4096_TEST,
)
from diffusion.model.builder import build_model, get_tokenizer_and_text_encoder, get_vae, vae_decode, vae_encode
from diffusion.model.utils import get_weight_dtype, prepare_prompt_ar, resize_and_crop_tensor
from diffusion.utils.config import SanaConfig, model_init_config
from diffusion.utils.logger import get_root_logger
from tools.controlnet.utils import get_scribble_map, transform_control_signal
from tools.download import find_model


def guidance_type_select(default_guidance_type, pag_scale, attn_type):
    guidance_type = default_guidance_type
    if not (pag_scale > 1.0 and attn_type == "linear"):
        guidance_type = "classifier-free"
    elif pag_scale > 1.0 and attn_type == "linear":
        guidance_type = "classifier-free_PAG"
    return guidance_type


def classify_height_width_bin(height: int, width: int, ratios: dict) -> Tuple[int, int]:
    """Returns binned height and width."""
    ar = float(height / width)
    closest_ratio = min(ratios.keys(), key=lambda ratio: abs(float(ratio) - ar))
    default_hw = ratios[closest_ratio]
    return int(default_hw[0]), int(default_hw[1])


def get_ar_from_ref_image(ref_image):
    def reduce_ratio(h, w):
        def gcd(a, b):
            while b:
                a, b = b, a % b
            return a

        divisor = gcd(h, w)
        return f"{h // divisor}:{w // divisor}"

    if isinstance(ref_image, str):
        ref_image = Image.open(ref_image)
    w, h = ref_image.size
    return reduce_ratio(h, w)


@dataclass
class SanaControlNetInference(SanaConfig):
    config: Optional[str] = "configs/sana_config/1024ms/Sana_1600M_img1024.yaml"  # config
    model_path: str = field(
        default="output/Sana_D20/SANA.pth", metadata={"help": "Path to the model file (positional)"}
    )
    output: str = "./output"
    bs: int = 1
    image_size: int = 1024
    cfg_scale: float = 5.0
    pag_scale: float = 2.0
    seed: int = 42
    step: int = -1
    custom_image_size: Optional[int] = None
    shield_model_path: str = field(
        default="google/shieldgemma-2b",
        metadata={"help": "The path to shield model, we employ ShieldGemma-2B by default."},
    )


class SanaControlNetPipeline(nn.Module):
    def __init__(
        self,
        config: Optional[str] = "configs/sana_config/1024ms/Sana_1600M_img1024.yaml",
    ):
        super().__init__()
        config = pyrallis.load(SanaControlNetInference, open(config))
        self.args = self.config = config

        # set some hyper-parameters
        self.image_size = self.config.model.image_size

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        logger = get_root_logger()
        self.logger = logger
        self.progress_fn = lambda progress, desc: None
        self.thickness = 2
        self.blend_alpha = 0.0

        self.latent_size = self.image_size // config.vae.vae_downsample_rate
        self.max_sequence_length = config.text_encoder.model_max_length
        self.flow_shift = config.scheduler.flow_shift
        guidance_type = "classifier-free_PAG"

        weight_dtype = get_weight_dtype(config.model.mixed_precision)
        self.weight_dtype = weight_dtype
        self.vae_dtype = get_weight_dtype(config.vae.weight_dtype)

        self.base_ratios = eval(f"ASPECT_RATIO_{self.image_size}_TEST")
        self.vis_sampler = self.config.scheduler.vis_sampler
        logger.info(f"Sampler {self.vis_sampler}, flow_shift: {self.flow_shift}")
        self.guidance_type = guidance_type_select(guidance_type, self.args.pag_scale, config.model.attn_type)
        logger.info(f"Inference with {self.weight_dtype}, PAG guidance layer: {self.config.model.pag_applied_layers}")

        # 1. build vae and text encoder
        self.vae = self.build_vae(config.vae)
        self.tokenizer, self.text_encoder = self.build_text_encoder(config.text_encoder)

        # 2. build Sana model
        self.model = self.build_sana_model(config).to(self.device)

        # 3. pre-compute null embedding
        with torch.no_grad():
            null_caption_token = self.tokenizer(
                "", max_length=self.max_sequence_length, padding="max_length", truncation=True, return_tensors="pt"
            ).to(self.device)
            self.null_caption_embs = self.text_encoder(null_caption_token.input_ids, null_caption_token.attention_mask)[
                0
            ]

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

    def set_blend_alpha(self, blend_alpha):
        self.blend_alpha = blend_alpha

    @torch.inference_mode()
    def forward(
        self,
        prompt=None,
        ref_image=None,
        negative_prompt="",
        num_inference_steps=20,
        guidance_scale=5,
        pag_guidance_scale=2.5,
        num_images_per_prompt=1,
        sketch_thickness=2,
        generator=torch.Generator().manual_seed(42),
        latents=None,
    ):
        self.ori_height, self.ori_width = ref_image.height, ref_image.width
        self.guidance_type = guidance_type_select(self.guidance_type, pag_guidance_scale, self.config.model.attn_type)

        # 1. pre-compute negative embedding
        if negative_prompt != "":
            null_caption_token = self.tokenizer(
                negative_prompt,
                max_length=self.max_sequence_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            ).to(self.device)
            self.null_caption_embs = self.text_encoder(null_caption_token.input_ids, null_caption_token.attention_mask)[
                0
            ]

        if prompt is None:
            prompt = [""]
        prompts = prompt if isinstance(prompt, list) else [prompt]
        samples = []

        for prompt in prompts:
            # data prepare
            prompts, hw, ar = (
                [],
                torch.tensor([[self.image_size, self.image_size]], dtype=torch.float, device=self.device).repeat(
                    num_images_per_prompt, 1
                ),
                torch.tensor([[1.0]], device=self.device).repeat(num_images_per_prompt, 1),
            )

            ar = get_ar_from_ref_image(ref_image)
            prompt += f" --ar {ar}"
            for _ in range(num_images_per_prompt):
                prompt_clean, _, hw, ar, custom_hw = prepare_prompt_ar(
                    prompt, self.base_ratios, device=self.device, show=False
                )
                prompts.append(prompt_clean.strip())

            self.latent_size_h, self.latent_size_w = (
                int(hw[0, 0] // self.config.vae.vae_downsample_rate),
                int(hw[0, 1] // self.config.vae.vae_downsample_rate),
            )

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
                null_y = self.null_caption_embs.repeat(len(prompts), 1, 1)[:, None].to(self.weight_dtype)

                n = len(prompts)
                if latents is None:
                    z = torch.randn(
                        n,
                        self.config.vae.vae_latent_dim,
                        self.latent_size_h,
                        self.latent_size_w,
                        generator=generator,
                        device=self.device,
                    )
                else:
                    z = latents.to(self.device)

                # control signal
                if isinstance(ref_image, str):
                    ref_image = cv2.imread(ref_image)
                elif isinstance(ref_image, Image.Image):
                    ref_image = np.array(ref_image)
                control_signal = get_scribble_map(
                    input_image=ref_image,
                    det="Scribble_HED",
                    detect_resolution=int(hw.min()),
                    thickness=sketch_thickness,
                )

                control_signal = transform_control_signal(control_signal, hw).to(self.device).to(self.weight_dtype)

                control_signal_latent = vae_encode(
                    self.config.vae.vae_type, self.vae, control_signal, self.config.vae.sample_posterior, self.device
                )

                model_kwargs = dict(
                    data_info={"img_hw": hw, "aspect_ratio": ar, "control_signal": control_signal_latent},
                    mask=emb_masks,
                )

                if self.vis_sampler == "flow_euler":
                    flow_solver = FlowEuler(
                        self.model,
                        condition=caption_embs,
                        uncondition=null_y,
                        cfg_scale=guidance_scale,
                        model_kwargs=model_kwargs,
                    )
                    sample = flow_solver.sample(
                        z,
                        steps=num_inference_steps,
                    )
                elif self.vis_sampler == "flow_dpm-solver":
                    scheduler = DPMS(
                        self.model.forward_with_dpmsolver,
                        condition=caption_embs,
                        uncondition=null_y,
                        guidance_type=self.guidance_type,
                        cfg_scale=guidance_scale,
                        model_type="flow",
                        model_kwargs=model_kwargs,
                        schedule="FLOW",
                    )
                    scheduler.register_progress_bar(self.progress_fn)
                    sample = scheduler.sample(
                        z,
                        steps=num_inference_steps,
                        order=2,
                        skip_type="time_uniform_flow",
                        method="multistep",
                        flow_shift=self.flow_shift,
                    )

            sample = sample.to(self.vae_dtype)
            with torch.no_grad():
                sample = vae_decode(self.config.vae.vae_type, self.vae, sample)

            if self.blend_alpha > 0:
                print(f"blend image and mask with alpha: {self.blend_alpha}")
                sample = sample * (1 - self.blend_alpha) + control_signal * self.blend_alpha

            sample = resize_and_crop_tensor(sample, self.ori_width, self.ori_height)
            samples.append(sample)

            return sample

        return samples
