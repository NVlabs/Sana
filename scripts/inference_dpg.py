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

import argparse
import json
import os
import re
import time
import warnings
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pyrallis
import torch
from einops import rearrange
from PIL import Image
from torchvision.utils import _log_api_usage_once, make_grid, save_image
from tqdm import tqdm

warnings.filterwarnings("ignore")  # ignore warning

from diffusion import DPMS, FlowEuler, SASolverSampler
from diffusion.data.datasets.utils import (
    ASPECT_RATIO_512_TEST,
    ASPECT_RATIO_1024_TEST,
    ASPECT_RATIO_2048_TEST,
    ASPECT_RATIO_4096_TEST,
    get_chunks,
)
from diffusion.model.builder import build_model, get_tokenizer_and_text_encoder, get_vae, vae_decode
from diffusion.model.utils import get_weight_dtype, prepare_prompt_ar
from diffusion.utils.config import SanaConfig, model_init_config
from diffusion.utils.logger import get_root_logger

# from diffusion.utils.misc import read_config
from tools.download import find_model


@torch.no_grad()
def pil_image(
    tensor,
    **kwargs,
) -> Image:
    if not torch.jit.is_scripting() and not torch.jit.is_tracing():
        _log_api_usage_once(save_image)
    grid = make_grid(tensor, **kwargs)
    # Add 0.5 after unnormalizing to [0, 255] to round to the nearest integer
    ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
    img = Image.fromarray(ndarr)
    return img


def set_env(seed=0, latent_size=256):
    torch.manual_seed(seed)
    torch.set_grad_enabled(False)
    for _ in range(30):
        torch.randn(1, 4, latent_size, latent_size)


@torch.inference_mode()
def visualize(items, bs, sample_steps, cfg_scale, pag_scale=1.0):

    generator = torch.Generator(device=device).manual_seed(args.seed)
    tqdm_desc = f"{save_root.split('/')[-1]} Using GPU: {args.gpu_id}: {args.start_index}-{args.end_index}"
    assert bs == 1
    for chunk in tqdm(list(get_chunks(items, bs)), desc=tqdm_desc, unit="batch", position=args.gpu_id, leave=True):

        prompt = data_dict[chunk[0]]["prompt"]

        # Generate images
        with torch.no_grad():
            all_samples = list()
            for _ in range((args.n_samples + batch_size - 1) // batch_size):
                prompts, hw, ar = (
                    [],
                    torch.tensor([[args.image_size, args.image_size]], dtype=torch.float, device=device).repeat(
                        batch_size, 1
                    ),
                    torch.tensor([[1.0]], device=device).repeat(batch_size, 1),
                )

                for _ in range(batch_size):
                    prompts.append(prepare_prompt_ar(prompt, base_ratios, device=device, show=False)[0].strip())
                    latent_size_h, latent_size_w = latent_size, latent_size

                # check exists
                save_file_name = f"{chunk[0]}.jpg"
                save_path = os.path.join(save_root, save_file_name)
                if os.path.exists(save_path):
                    # make sure the noise is totally same
                    torch.randn(
                        len(prompts),
                        config.vae.vae_latent_dim,
                        latent_size,
                        latent_size,
                        device=device,
                        generator=generator,
                    )
                    continue

                # prepare text feature
                caption_token = tokenizer(
                    prompts, max_length=max_sequence_length, padding="max_length", truncation=True, return_tensors="pt"
                ).to(device)
                caption_embs = text_encoder(caption_token.input_ids, caption_token.attention_mask)[0][:, None]
                emb_masks, null_y = caption_token.attention_mask, null_caption_embs.repeat(len(prompts), 1, 1)[:, None]

                # start sampling
                with torch.no_grad():
                    n = len(prompts)
                    z = torch.randn(
                        n,
                        config.vae.vae_latent_dim,
                        latent_size,
                        latent_size,
                        device=device,
                        generator=generator,
                    )
                    model_kwargs = dict(data_info={"img_hw": hw, "aspect_ratio": ar}, mask=emb_masks)

                    if args.sampling_algo == "dpm-solver":
                        dpm_solver = DPMS(
                            model.forward_with_dpmsolver,
                            condition=caption_embs,
                            uncondition=null_y,
                            cfg_scale=cfg_scale,
                            model_kwargs=model_kwargs,
                        )
                        samples = dpm_solver.sample(
                            z,
                            steps=sample_steps,
                            order=2,
                            skip_type="time_uniform",
                            method="multistep",
                        )
                    elif args.sampling_algo == "sa-solver":
                        sa_solver = SASolverSampler(model.forward_with_dpmsolver, device=device)
                        samples = sa_solver.sample(
                            S=25,
                            batch_size=n,
                            shape=(config.vae.vae_latent_dim, latent_size_h, latent_size_w),
                            eta=1,
                            conditioning=caption_embs,
                            unconditional_conditioning=null_y,
                            unconditional_guidance_scale=cfg_scale,
                            model_kwargs=model_kwargs,
                        )[0]
                    elif args.sampling_algo == "flow_euler":
                        flow_solver = FlowEuler(
                            model,
                            condition=caption_embs,
                            uncondition=null_y,
                            cfg_scale=cfg_scale,
                            model_kwargs=model_kwargs,
                        )
                        samples = flow_solver.sample(
                            z,
                            steps=sample_steps,
                        )
                    elif args.sampling_algo == "flow_dpm-solver":
                        dpm_solver = DPMS(
                            model.forward_with_dpmsolver,
                            condition=caption_embs,
                            uncondition=null_y,
                            guidance_type=guidance_type,
                            cfg_scale=cfg_scale,
                            pag_scale=pag_scale,
                            pag_applied_layers=pag_applied_layers,
                            model_type="flow",
                            model_kwargs=model_kwargs,
                            schedule="FLOW",
                            interval_guidance=args.interval_guidance,
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
                    torch.cuda.empty_cache()

                    all_samples.append(samples)

            if all_samples:
                # additionally, save as grid
                grid = torch.stack(all_samples, 0)
                grid = rearrange(grid, "n b c h w -> (n b) c h w")
                grid = make_grid(grid, nrow=n_rows, normalize=True, value_range=(-1, 1))

                # to image
                grid = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
                grid = Image.fromarray(grid.astype(np.uint8))
                grid.save(save_path)
                del grid
        del all_samples

    print("Done.")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="config")
    parser.add_argument("--model_path", default=None, type=str, help="Path to the model file (optional)")

    return parser.parse_known_args()[0]


@dataclass
class SanaInference(SanaConfig):
    config: Optional[str] = "configs/sana_config/1024ms/Sana_1600M_img1024.yaml"  # config
    dataset: str = "DPG"
    outdir: str = "outputs"
    n_samples: int = 4
    batch_size: int = 1
    skip_grid: bool = False
    position_model_path: str = "output/pretrained_models/Sana.pth"
    model_path: str = None
    txt_file: str = "asset/samples/samples.txt"
    json_file: str = None
    sample_nums: int = 1065
    cfg_scale: float = 4.5
    pag_scale: float = 1.0
    sampling_algo: str = field(
        default="dpm-solver", metadata={"choices": ["dpm-solver", "sa-solver", "flow_euler", "flow_dpm-solver"]}
    )
    bs: int = 1
    seed: int = 0
    step: int = -1
    add_label: str = ""
    tar_and_del: bool = False
    exist_time_prefix: str = ""
    gpu_id: int = 0
    image_size: int = 512
    custom_image_size: int = None
    start_index: int = 0
    end_index: int = 553
    interval_guidance: list = field(
        default_factory=lambda: [0, 1], metadata={"help": "A list value, like [0, 1.] for use cfg"}
    )
    ablation_selections: list = None
    ablation_key: str = field(default=None, metadata={"choices": ["step", "cfg_scale", "pag_scale"]})
    if_save_dirname: bool = False


if __name__ == "__main__":

    args = get_args()
    config = args = pyrallis.parse(config_class=SanaInference, config_path=args.config)
    # config = read_config(args.config)

    args.image_size = config.model.image_size
    if args.custom_image_size:
        args.image_size = args.custom_image_size
        print(f"custom_image_size: {args.image_size}")

    set_env(args.seed, args.image_size // config.vae.vae_downsample_rate)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger = get_root_logger()

    n_rows = args.n_samples // 2
    batch_size = args.n_samples
    assert args.batch_size == 1, ValueError(f"{batch_size} > 1 is not available in DPG-bench")

    # only support fixed latent size currently
    latent_size = args.image_size // config.vae.vae_downsample_rate
    max_sequence_length = config.text_encoder.model_max_length
    pe_interpolation = config.model.pe_interpolation
    micro_condition = config.model.micro_condition
    flow_shift = config.scheduler.flow_shift
    pag_applied_layers = config.model.pag_applied_layers
    guidance_type = "classifier-free_PAG"
    # guidance_type = config.guidance_type
    assert (
        isinstance(args.interval_guidance, list)
        and len(args.interval_guidance) == 2
        and args.interval_guidance[0] <= args.interval_guidance[1]
    )
    args.interval_guidance = [max(0, args.interval_guidance[0]), min(1, args.interval_guidance[1])]
    sample_steps_dict = {"dpm-solver": 20, "sa-solver": 25, "flow_dpm-solver": 20, "flow_euler": 28}
    sample_steps = args.step if args.step != -1 else sample_steps_dict[args.sampling_algo]
    weight_dtype = get_weight_dtype(config.model.mixed_precision)
    logger.info(f"Inference with {weight_dtype}, default guidance_type: {guidance_type}, flow_shift: {flow_shift}")

    vae_dtype = get_weight_dtype(config.vae.weight_dtype)
    vae = get_vae(config.vae.vae_type, config.vae.vae_pretrained, device).to(vae_dtype)
    tokenizer, text_encoder = get_tokenizer_and_text_encoder(name=config.text_encoder.text_encoder_name, device=device)

    null_caption_token = tokenizer(
        "", max_length=max_sequence_length, padding="max_length", truncation=True, return_tensors="pt"
    ).to(device)
    null_caption_embs = text_encoder(null_caption_token.input_ids, null_caption_token.attention_mask)[0]

    # model setting
    model_kwargs = model_init_config(config, latent_size=latent_size)
    model = build_model(
        config.model.model, use_fp32_attention=config.model.get("fp32_attention", False), **model_kwargs
    ).to(device)
    # model = build_model(config.model, **model_kwargs).to(device)
    logger.info(
        f"{model.__class__.__name__}:{config.model.model}, Model Parameters: {sum(p.numel() for p in model.parameters()):,}"
    )
    args.model_path = args.model_path or args.position_model_path
    logger.info("Generating sample from ckpt: %s" % args.model_path)
    state_dict = find_model(args.model_path)
    if "pos_embed" in state_dict["state_dict"]:
        del state_dict["state_dict"]["pos_embed"]

    missing, unexpected = model.load_state_dict(state_dict["state_dict"], strict=False)
    logger.warning(f"Missing keys: {missing}")
    logger.warning(f"Unexpected keys: {unexpected}")
    model.eval().to(weight_dtype)
    base_ratios = eval(f"ASPECT_RATIO_{args.image_size}_TEST")
    args.sampling_algo = (
        args.sampling_algo
        if ("flow" not in args.model_path or args.sampling_algo == "flow_dpm-solver")
        else "flow_euler"
    )

    work_dir = (
        f"/{os.path.join(*args.model_path.split('/')[:-2])}"
        if args.model_path.startswith("/")
        else os.path.join(*args.model_path.split("/")[:-2])
    )

    # dataset
    dict_prompt = args.json_file is not None
    if dict_prompt:
        data_dict = json.load(open(args.json_file))
        items = list(data_dict.keys())
    else:
        with open(args.txt_file) as f:
            items = [item.strip() for item in f.readlines()]
    logger.info(f"Eval first {min(args.sample_nums, len(items))}/{len(items)} samples")
    items = items[: max(0, args.sample_nums)]
    items = items[max(0, args.start_index) : min(len(items), args.end_index)]  # save path

    match = re.search(r".*epoch_(\d+).*step_(\d+).*", args.model_path)
    epoch_name, step_name = match.groups() if match else ("unknown", "unknown")

    img_save_dir = os.path.join(str(work_dir), "vis")
    os.umask(0o000)
    os.makedirs(img_save_dir, exist_ok=True)
    logger.info(f"Sampler {args.sampling_algo}")

    def create_save_root(args, dataset, epoch_name, step_name, sample_steps, guidance_type):
        save_root = os.path.join(
            img_save_dir,
            # f"{datetime.now().date() if args.exist_time_prefix == '' else args.exist_time_prefix}_"
            f"{dataset}_epoch{epoch_name}_step{step_name}_scale{args.cfg_scale}"
            f"_step{sample_steps}_size{args.image_size}_bs{batch_size}_samp{args.sampling_algo}"
            f"_seed{args.seed}_{str(weight_dtype).split('.')[-1]}",
        )

        if args.pag_scale != 1.0:
            save_root = save_root.replace(f"scale{args.cfg_scale}", f"scale{args.cfg_scale}_pagscale{args.pag_scale}")
        if flow_shift != 1.0:
            save_root += f"_flowshift{flow_shift}"
        if guidance_type != "classifier-free":
            save_root += f"_{guidance_type}"
        if args.interval_guidance[0] != 0 and args.interval_guidance[1] != 1:
            save_root += f"_intervalguidance{args.interval_guidance[0]}{args.interval_guidance[1]}"

        save_root += f"_imgnums{args.sample_nums}" + args.add_label
        return save_root

    def guidance_type_select(default_guidance_type, pag_scale, attn_type):
        guidance_type = default_guidance_type
        if not (pag_scale > 1.0 and attn_type == "linear"):
            logger.info("Setting back to classifier-free")
            guidance_type = "classifier-free"
        return guidance_type

    dataset = "MJHQ-30K" if args.json_file and "MJHQ-30K" in args.json_file else args.dataset
    if args.ablation_selections and args.ablation_key:
        for ablation_factor in args.ablation_selections:
            setattr(args, args.ablation_key, eval(ablation_factor))
            print(f"Setting {args.ablation_key}={eval(ablation_factor)}")
            sample_steps = args.step if args.step != -1 else sample_steps_dict[args.sampling_algo]
            guidance_type = guidance_type_select(guidance_type, args.pag_scale, config.model.attn_type)

            save_root = create_save_root(args, dataset, epoch_name, step_name, sample_steps, guidance_type)
            os.makedirs(save_root, exist_ok=True)
            if args.if_save_dirname and args.gpu_id == 0:
                # save at work_dir/metrics/tmp_xxx.txt for metrics testing
                with open(f"{work_dir}/metrics/tmp_{dataset}_{time.time()}.txt", "w") as f:
                    print(f"save tmp file at {work_dir}/metrics/tmp_{dataset}_{time.time()}.txt")
                    f.write(os.path.basename(save_root))
            logger.info(f"Inference with {weight_dtype}, guidance_type: {guidance_type}, flow_shift: {flow_shift}")

            visualize(items, args.bs, sample_steps, args.cfg_scale, args.pag_scale)
    else:
        guidance_type = guidance_type_select(guidance_type, args.pag_scale, config.model.attn_type)
        logger.info(f"Inference with {weight_dtype}, guidance_type: {guidance_type}, flow_shift: {flow_shift}")

        save_root = create_save_root(args, dataset, epoch_name, step_name, sample_steps, guidance_type)
        os.makedirs(save_root, exist_ok=True)
        if args.if_save_dirname and args.gpu_id == 0:
            os.makedirs(f"{work_dir}/metrics", exist_ok=True)
            # save at work_dir/metrics/tmp_dpg_xxx.txt for metrics testing
            with open(f"{work_dir}/metrics/tmp_{dataset}_{time.time()}.txt", "w") as f:
                print(f"save tmp file at {work_dir}/metrics/tmp_{dataset}_{time.time()}.txt")
                f.write(os.path.basename(save_root))

        visualize(items, args.bs, sample_steps, args.cfg_scale, args.pag_scale)
