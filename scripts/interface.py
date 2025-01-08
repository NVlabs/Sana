#!/usr/bin/env python
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
import argparse
import gc
import os
import random
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Tuple, Union

import gradio as gr
import numpy as np
import pyrallis
import torch
from gradio.components import Image, Textbox
from torchvision.utils import _log_api_usage_once, make_grid, save_image

warnings.filterwarnings("ignore")  # ignore warning

from asset.examples import examples
from diffusion import DPMS, FlowEuler, SASolverSampler
from diffusion.data.datasets.utils import (
    ASPECT_RATIO_512_TEST,
    ASPECT_RATIO_1024_TEST,
    ASPECT_RATIO_2048_TEST,
    ASPECT_RATIO_4096_TEST,
)
from diffusion.model.builder import build_model, get_tokenizer_and_text_encoder, get_vae, vae_decode
from diffusion.model.utils import get_weight_dtype, prepare_prompt_ar, resize_and_crop_tensor
from diffusion.utils.config import SanaConfig, model_init_config
from diffusion.utils.dist_utils import flush
from tools.download import find_model

# from diffusion.utils.misc import read_config

MAX_SEED = np.iinfo(np.int32).max


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="config path")
    return parser.parse_known_args()[0]


@dataclass
class SanaInference(SanaConfig):
    config: Optional[str] = "configs/sana_config/1024ms/Sana_1600M_img1024.yaml"  # config
    model_path: str = field(
        default="output/Sana_1600M/SANA.pth", metadata={"help": "Path to the model file (positional)"}
    )
    output: str = "./output"
    bs: int = 1
    image_size: int = 1024
    cfg_scale: float = 5.0
    pag_scale: float = 2.0
    seed: int = 42
    step: int = -1
    port: int = 7788
    custom_image_size: Optional[int] = None
    shield_model_path: str = field(
        default="google/shieldgemma-2b",
        metadata={"help": "The path to shield model, we employ ShieldGemma-2B by default."},
    )


@torch.no_grad()
def ndarr_image(
    tensor: Union[torch.Tensor, List[torch.Tensor]],
    **kwargs,
) -> None:
    if not torch.jit.is_scripting() and not torch.jit.is_tracing():
        _log_api_usage_once(save_image)
    grid = make_grid(tensor, **kwargs)
    # Add 0.5 after unnormalizing to [0, 255] to round to the nearest integer
    ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
    return ndarr


def set_env(seed=0):
    torch.manual_seed(seed)
    torch.set_grad_enabled(False)
    for _ in range(30):
        torch.randn(1, 4, args.image_size, args.image_size)


def randomize_seed_fn(seed: int, randomize_seed: bool) -> int:
    if randomize_seed:
        seed = random.randint(0, MAX_SEED)
    return seed


def classify_height_width_bin(height: int, width: int, ratios: dict) -> Tuple[int, int]:
    """Returns binned height and width."""
    ar = float(height / width)
    closest_ratio = min(ratios.keys(), key=lambda ratio: abs(float(ratio) - ar))
    default_hw = ratios[closest_ratio]
    return int(default_hw[0]), int(default_hw[1])


@torch.inference_mode()
def generate_img(
    prompt,
    sampler,
    sample_steps,
    scale,
    pag_scale=1.0,
    guidance_type="classifier-free",
    seed=0,
    randomize_seed=False,
    base_size=1024,
    height=1024,
    width=1024,
):
    flush()
    gc.collect()
    torch.cuda.empty_cache()

    seed = int(randomize_seed_fn(seed, randomize_seed))
    set_env(seed)
    base_ratios = eval(f"ASPECT_RATIO_{base_size}_TEST")

    os.makedirs(f"output/demo/online_demo_prompts/", exist_ok=True)
    save_promt_path = f"output/demo/online_demo_prompts/tested_prompts{datetime.now().date()}.txt"
    with open(save_promt_path, "a") as f:
        f.write(f"{seed}: {prompt}" + "\n")
    print(f"{seed}: {prompt}")
    prompt_clean, prompt_show, _, _, _ = prepare_prompt_ar(prompt, base_ratios, device=device)  # ar for aspect ratio
    orig_height, orig_width = height, width
    height, width = classify_height_width_bin(height, width, ratios=base_ratios)

    prompt_show += (
        f"\n Sample steps: {sample_steps}, CFG Scale: {scale}, PAG Scale: {pag_scale}, flow_shift: {flow_shift}"
    )
    prompt_clean = prompt_clean.strip()
    if isinstance(prompt_clean, str):
        prompts = [prompt_clean]

    # prepare text feature
    if not config.text_encoder.chi_prompt:
        max_length_all = max_sequence_length
        prompts_all = prompts
    else:
        chi_prompt = "\n".join(config.text_encoder.chi_prompt)
        prompts_all = [chi_prompt + prompt for prompt in prompts]
        num_chi_prompt_tokens = len(tokenizer.encode(chi_prompt))
        max_length_all = num_chi_prompt_tokens + max_sequence_length - 2  # magic number 2: [bos], [_]

    caption_token = tokenizer(
        prompts_all, max_length=max_length_all, padding="max_length", truncation=True, return_tensors="pt"
    ).to(device)
    select_index = [0] + list(range(-max_sequence_length + 1, 0))
    caption_embs = text_encoder(caption_token.input_ids, caption_token.attention_mask)[0][:, None][:, :, select_index]
    emb_masks = caption_token.attention_mask[:, select_index]
    null_y = null_caption_embs.repeat(len(prompts), 1, 1)[:, None]

    n = len(prompts)
    latent_size_h, latent_size_w = height // config.vae.vae_downsample_rate, width // config.vae.vae_downsample_rate
    z = torch.randn(n, config.vae.vae_latent_dim, latent_size_h, latent_size_w, device=device)
    model_kwargs = dict(data_info={"img_hw": (latent_size_h, latent_size_w), "aspect_ratio": 1.0}, mask=emb_masks)
    print(f"Latent Size: {z.shape}")
    # Sample images:
    if sampler == "dpm-solver":
        # Create sampling noise:
        dpm_solver = DPMS(
            model.forward_with_dpmsolver,
            condition=caption_embs,
            uncondition=null_y,
            cfg_scale=scale,
            model_kwargs=model_kwargs,
        )
        samples = dpm_solver.sample(
            z,
            steps=sample_steps,
            order=2,
            skip_type="time_uniform",
            method="multistep",
        )
    elif sampler == "sa-solver":
        # Create sampling noise:
        sa_solver = SASolverSampler(model.forward_with_dpmsolver, device=device)
        samples = sa_solver.sample(
            S=sample_steps,
            batch_size=n,
            shape=(4, latent_size_h, latent_size_w),
            eta=1,
            conditioning=caption_embs,
            unconditional_conditioning=null_y,
            unconditional_guidance_scale=scale,
            model_kwargs=model_kwargs,
        )[0]
    elif sampler == "flow_euler":
        flow_solver = FlowEuler(
            model, condition=caption_embs, uncondition=null_y, cfg_scale=scale, model_kwargs=model_kwargs
        )
        samples = flow_solver.sample(
            z,
            steps=sample_steps,
        )
    elif sampler == "flow_dpm-solver":
        if not (pag_scale > 1.0 and config.model.attn_type == "linear"):
            guidance_type = "classifier-free"
        dpm_solver = DPMS(
            model,
            condition=caption_embs,
            uncondition=null_y,
            guidance_type=guidance_type,
            cfg_scale=scale,
            pag_scale=pag_scale,
            pag_applied_layers=pag_applied_layers,
            model_type="flow",
            model_kwargs=model_kwargs,
            schedule="FLOW",
        )
        samples = dpm_solver.sample(
            z,
            steps=sample_steps,
            order=2,
            skip_type="time_uniform_flow",
            method="multistep",
            flow_shift=flow_shift,
        )
    else:
        raise ValueError(f"{args.sampling_algo} is not defined")

    samples = samples.to(vae_dtype)
    samples = vae_decode(config.vae.vae_type, vae, samples)
    samples = resize_and_crop_tensor(samples, orig_width, orig_height)
    display_model_info = (
        f"Model path: {args.model_path},\nBase image size: {args.image_size}, \nSampling Algo: {sampler}"
    )
    return ndarr_image(samples, normalize=True, value_range=(-1, 1)), prompt_show, display_model_info, seed


if __name__ == "__main__":
    from diffusion.utils.logger import get_root_logger

    args = get_args()
    config = args = pyrallis.parse(config_class=SanaInference, config_path=args.config)
    # config = read_config(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger = get_root_logger()

    args.image_size = config.model.image_size
    assert args.image_size in [
        256,
        512,
        1024,
        2048,
        4096,
    ], "We only provide pre-trained models for 256x256, 512x512, 1024x1024, 2048x2048 and 4096x4096 resolutions."

    # only support fixed latent size currently
    latent_size = config.model.image_size // config.vae.vae_downsample_rate
    max_sequence_length = config.text_encoder.model_max_length
    pe_interpolation = config.model.pe_interpolation
    micro_condition = config.model.micro_condition
    pag_applied_layers = config.model.pag_applied_layers
    flow_shift = config.scheduler.flow_shift

    weight_dtype = get_weight_dtype(config.model.mixed_precision)
    logger.info(f"Inference with {weight_dtype}")

    vae_dtype = get_weight_dtype(config.vae.weight_dtype)
    vae = get_vae(config.vae.vae_type, config.vae.vae_pretrained, device).to(vae_dtype)
    tokenizer, text_encoder = get_tokenizer_and_text_encoder(name=config.text_encoder.text_encoder_name, device=device)

    # model setting
    model_kwargs = model_init_config(config, latent_size=latent_size)
    model = build_model(
        config.model.model, use_fp32_attention=config.model.get("fp32_attention", False), **model_kwargs
    ).to(device)
    # model = build_model(config.model, **model_kwargs).to(device)
    logger.info(
        f"{model.__class__.__name__}:{config.model.model}, Model Parameters: {sum(p.numel() for p in model.parameters()):,}"
    )
    logger.info("Generating sample from ckpt: %s" % args.model_path)
    state_dict = find_model(args.model_path)
    if "pos_embed" in state_dict["state_dict"]:
        del state_dict["state_dict"]["pos_embed"]

    missing, unexpected = model.load_state_dict(state_dict["state_dict"], strict=False)
    logger.warning(f"Missing keys: {missing}")
    logger.warning(f"Unexpected keys: {unexpected}")
    model.eval().to(weight_dtype)
    base_ratios = eval(f"ASPECT_RATIO_{args.image_size}_TEST")

    null_caption_token = tokenizer(
        "", max_length=max_sequence_length, padding="max_length", truncation=True, return_tensors="pt"
    ).to(device)
    null_caption_embs = text_encoder(null_caption_token.input_ids, attention_mask=null_caption_token.attention_mask)[0]

    model_size = "1.6" if "D20" in args.model_path else "0.6"
    title = f"""
        <div style='display: flex; align-items: center; justify-content: center; text-align: center;'>
            <img src="https://raw.githubusercontent.com/NVlabs/Sana/refs/heads/main/asset/logo.png" width="50%" alt="logo"/>
        </div>
    """
    DESCRIPTION = f"""
            <p><span style="font-size: 36px; font-weight: bold;">Sana-{model_size}B</span><span style="font-size: 20px; font-weight: bold;">{args.image_size}px</span></p>
            <p style="font-size: 16px; font-weight: bold;">Sana: Efficient High-Resolution Image Synthesis with Linear Diffusion Transformer</p>
            <p><span style="font-size: 16px;"><a href="https://arxiv.org/abs/2410.10629">[Paper]</a></span> <span style="font-size: 16px;"><a href="https://github.com/NVlabs/Sana">[Github]</a></span> <span style="font-size: 16px;"><a href="https://nvlabs.github.io/Sana">[Project]</a></span</p>
            <p style="font-size: 16px; font-weight: bold;">Powered by <a href="https://hanlab.mit.edu/projects/dc-ae">DC-AE</a> with 32x latent space</p>
            """
    if model_size == "0.6":
        DESCRIPTION += "\n<p>0.6B model's text rendering ability is limited.</p>"
    if not torch.cuda.is_available():
        DESCRIPTION += "\n<p>Running on CPU 🥶 This demo does not work on CPU.</p>"

    demo = gr.Interface(
        fn=generate_img,
        inputs=[
            Textbox(
                label="Note: If you want to specify a aspect ratio or determine a customized height and width, "
                "use --ar h:w (or --aspect_ratio h:w) or --hw h:w. If no aspect ratio or hw is given, all setting will be default.",
                placeholder="Please enter your prompt. \n",
            ),
            gr.Radio(
                choices=["dpm-solver", "sa-solver", "flow_dpm-solver", "flow_euler"],
                label=f"Sampler",
                interactive=True,
                value="flow_dpm-solver",
            ),
            gr.Slider(label="Sample Steps", minimum=1, maximum=100, value=20, step=1),
            gr.Slider(label="Guidance Scale", minimum=1.0, maximum=30.0, value=5.0, step=0.1),
            gr.Slider(label="PAG Scale", minimum=1.0, maximum=10.0, value=2.5, step=0.5),
            gr.Radio(
                choices=["classifier-free", "classifier-free_PAG", "classifier-free_PAG_seq"],
                label=f"Guidance Type",
                interactive=True,
                value="classifier-free_PAG_seq",
            ),
            gr.Slider(
                label="Seed",
                minimum=0,
                maximum=MAX_SEED,
                step=1,
                value=0,
            ),
            gr.Checkbox(label="Randomize seed", value=True),
            gr.Radio(
                choices=[256, 512, 1024, 2048, 4096],
                label=f"Base Size",
                interactive=True,
                value=args.image_size,
            ),
            gr.Slider(
                label="Height",
                minimum=256,
                maximum=6000,
                step=32,
                value=args.image_size,
            ),
            gr.Slider(
                label="Width",
                minimum=256,
                maximum=6000,
                step=32,
                value=args.image_size,
            ),
        ],
        outputs=[
            Image(type="numpy", label="Img"),
            Textbox(label="clean prompt"),
            Textbox(label="model info"),
            gr.Slider(label="seed"),
        ],
        title=title,
        description=DESCRIPTION,
        examples=examples,
    )
    demo.launch(server_name="0.0.0.0", server_port=args.port, debug=True, share=True)
