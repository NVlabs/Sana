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

# from diffusion.utils.misc import read_config
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


def tensor_to_pil(tensor):
    # Ensure tensor is on CPU and convert to uint8
    tensor = (tensor / 2 + 0.5).clamp(0, 1).cpu()

    tensor = (tensor * 255).byte()  # Convert to [0, 255] and uint8

    # Convert CHW to HWC format
    if tensor.dim() == 3:
        tensor = tensor.permute(1, 2, 0)

    # Convert to numpy and create PIL Image
    numpy_array = tensor.numpy()
    return Image.fromarray(numpy_array)


@dataclass
class SanaInference(SanaConfig):
    config: Optional[str] = "configs/sana1-5_config/1024ms/Sana_1600M_1024px_allqknorm_bf16_lr2e5.yaml"  # config
    model_path: str = field(
        default="output/Sana_D20/SANA.pth", metadata={"help": "Path to the model file (positional)"}
    )
    output: str = "./output"
    bs: int = 1
    image_size: int = 1024
    cfg_scale: float = 4.5
    pag_scale: float = 1.0
    seed: int = 42
    step: int = -1
    custom_image_size: Optional[int] = None
    shield_model_path: str = field(
        default="google/shieldgemma-2b",
        metadata={"help": "The path to shield model, we employ ShieldGemma-2B by default."},
    )


class SanaPipelineInpaint(nn.Module):
    def __init__(
        self,
        config: Optional[str] = "configs/sana_config/1024ms/Sana_1600M_img1024.yaml",
    ):
        super().__init__()
        config = pyrallis.load(SanaInference, open(config))
        self.args = self.config = config

        # set some hyper-parameters
        self.image_size = self.config.model.image_size

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        logger = get_root_logger()
        self.logger = logger
        self.progress_fn = lambda progress, desc: None

        self.latent_size = self.image_size // config.vae.vae_downsample_rate
        self.max_sequence_length = config.text_encoder.model_max_length
        self.flow_shift = config.scheduler.flow_shift
        guidance_type = "classifier-free"

        weight_dtype = get_weight_dtype(config.model.mixed_precision)
        self.weight_dtype = weight_dtype
        self.vae_dtype = weight_dtype

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

    def _extract_into_tensor(self, arr, timesteps, broadcast_shape):
        """
        Extract values from a 1-D numpy array for a batch of indices.
        :param arr: the 1-D numpy array.
        :param timesteps: a tensor of indices into the array to extract.
        :param broadcast_shape: a larger shape of K dimensions with the batch
                                dimension equal to the length of timesteps.
        :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
        """
        res = torch.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
        while len(res.shape) < len(broadcast_shape):
            res = res[..., None]
        return res + torch.zeros(broadcast_shape, device=timesteps.device)

    def encode_latent(self, image):
        return vae_encode(self.config.vae.vae_type, self.vae, image, self.config.vae.sample_posterior, self.device)

    def decode_latent(self, latent):
        image_space = vae_decode(self.config.vae.vae_type, self.vae, latent)

        if image_space.dim() == 4:  # Batch dimension present [B, C, H, W]
            image_tensor = image_space[0]  # Take first image from batch
        else:
            image_tensor = image_space  # Already [C, H, W]

        image_tensor = image_tensor.cpu().float()  # Convert to float32 on CPU
        image_tensor = (image_tensor / 2 + 0.5).clamp(0, 1)  # Normalize from [-1,1] to [0,1]
        image_tensor = (image_tensor * 255).byte()  # Convert to uint8

        if image_tensor.dim() == 3:
            image_tensor = image_tensor.permute(1, 2, 0)

        numpy_array = image_tensor.numpy()
        img = Image.fromarray(numpy_array)
        return img

    def renoise_image(self, image, scheduler, total_steps, renoise_steps):
        latent_clean = vae_encode(
            self.config.vae.vae_type, self.vae, image, self.config.vae.sample_posterior, self.device
        )
        t_start = 1.0 / total_steps
        t_end = t_start * renoise_steps
        assert t_end < 1.0, "t_end should be less than or equal to 1.0"
        noised_latent = scheduler.inverse(latent_clean, steps=renoise_steps, t_end=t_end)

        noisy_image = self.decode_latent(noised_latent)
        noisy_image.save("renoised_image.png")
        return noised_latent

    def get_prompt_embed(self, prompt, num_images_per_prompt=1):
        if prompt is None:
            prompt = [""]
        prompts = prompt if isinstance(prompt, list) else [prompt]

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
                null_y = self.null_caption_embs.repeat(len(prompts), 1, 1)[:, None].to(self.weight_dtype)

        return caption_embs, emb_masks, null_y

    def get_mask_latent(self, mask, use_resolution_binning=True, height=1024, width=1024):
        """
        rescale the mask to latent space size according to latent blended diffusion
        """

        self.ori_height, self.ori_width = height, width
        if use_resolution_binning:
            self.height, self.width = classify_height_width_bin(height, width, ratios=self.base_ratios)
        else:
            self.height, self.width = height, width
        self.latent_size_h, self.latent_size_w = (
            self.height // self.config.vae.vae_downsample_rate,
            self.width // self.config.vae.vae_downsample_rate,
        )

        # Ensure mask is a tensor and has the right dimensions
        if isinstance(mask, Image.Image):
            mask = torch.from_numpy(np.array(mask)).float()
        elif not isinstance(mask, torch.Tensor):
            mask = torch.tensor(mask).float()

        # Add batch and channel dimensions if needed
        if mask.dim() == 2:  # [H, W]
            mask = mask.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
        elif mask.dim() == 3:  # [H, W, C] or [C, H, W]
            if mask.shape[-1] == 1 or mask.shape[-1] == 3:  # [H, W, C]
                mask = mask.permute(2, 0, 1).unsqueeze(0)  # [1, C, H, W]
            else:  # [C, H, W]
                mask = mask.unsqueeze(0)  # [1, C, H, W]
        elif mask.dim() == 4:  # [B, C, H, W]
            pass  # Already correct format

        # Rescale mask to latent dimensions
        mask = torch.nn.functional.interpolate(
            mask, size=(self.latent_size_h, self.latent_size_w), mode="bilinear", align_corners=False
        )

        # Convert to binary mask if needed (threshold at 0.5)
        mask = (mask > 0.5).float()

        return mask

    @torch.inference_mode()
    def forward(
        self,
        image,
        mask,
        prompt=None,
        height=1024,
        width=1024,
        negative_prompt="",
        num_inference_steps=20,
        guidance_scale=4.5,
        pag_guidance_scale=1.0,
        num_images_per_prompt=1,
        generator=torch.Generator().manual_seed(42),
        latents=None,
        use_resolution_binning=True,
        return_latent=False,
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
        self.guidance_type = guidance_type_select(self.guidance_type, pag_guidance_scale, self.config.model.attn_type)

        latent_mask = self.get_mask_latent(mask, use_resolution_binning, height, width)

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
                        # dtype=self.weight_dtype,
                    )
                else:
                    z = latents.to(self.device)
                model_kwargs = dict(data_info={"img_hw": hw, "aspect_ratio": ar}, mask=emb_masks)
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
                        self.model,
                        condition=caption_embs,
                        uncondition=null_y,
                        guidance_type=self.guidance_type,
                        cfg_scale=guidance_scale,
                        pag_scale=pag_guidance_scale,
                        pag_applied_layers=self.config.model.pag_applied_layers,
                        model_type="flow",
                        model_kwargs=model_kwargs,
                        schedule="FLOW",
                    )
                    scheduler.register_progress_bar(self.progress_fn)
                    hw = torch.tensor([[self.height, self.width]], dtype=torch.float, device=self.device).reshape(1, 2)
                    image = transform_control_signal(image, hw).to(self.device).to(self.weight_dtype)
                    noised_latent = vae_encode(
                        self.config.vae.vae_type, self.vae, image, self.config.vae.sample_posterior, self.device
                    )
                    latent_mask = (
                        latent_mask.repeat(1, noised_latent.shape[1], 1, 1).to(self.device).to(self.weight_dtype)
                    )
                    # noised_latent = self.renoise_image(image, scheduler, total_steps=num_inference_steps,renoise_steps=renoise_steps)  # Example usage of renoise_image
                    sample = scheduler.sample_mask(
                        x=z,
                        img_latent=noised_latent,
                        latent_mask=latent_mask,
                        steps=num_inference_steps,
                        order=2,
                        skip_type="time_uniform_flow",
                        method="multistep",
                        flow_shift=self.flow_shift,
                    )

            sample = sample.to(self.vae_dtype)
            if return_latent:
                return sample

            with torch.no_grad():
                sample = vae_decode(self.config.vae.vae_type, self.vae, sample)

            if use_resolution_binning:
                sample = resize_and_crop_tensor(sample, self.ori_width, self.ori_height)
            samples.append(sample)

            return sample

        return samples
