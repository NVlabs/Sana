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
import json
import os
import re
import subprocess
import tarfile
import time
import warnings
from dataclasses import dataclass, field
from typing import List, Optional

import pyrallis
import torch
from termcolor import colored
from torchvision.utils import save_image
from tqdm import tqdm

warnings.filterwarnings("ignore")  # ignore warning
os.environ["DISABLE_XFORMERS"] = "1"

from diffusion import SCMScheduler
from diffusion.data.datasets.utils import ASPECT_RATIO_512_TEST, ASPECT_RATIO_1024_TEST
from diffusion.model.builder import build_model, get_tokenizer_and_text_encoder, get_vae, vae_decode
from diffusion.model.utils import get_weight_dtype, prepare_prompt_ar
from diffusion.utils.config import SanaConfig, model_init_config
from diffusion.utils.logger import get_root_logger
from tools.download import find_model


def set_env(seed=0, latent_size=256):
    torch.manual_seed(seed)
    torch.set_grad_enabled(False)
    for _ in range(30):
        torch.randn(1, 4, latent_size, latent_size)


def get_dict_chunks(data, bs):
    keys = []
    for k in data:
        keys.append(k)
        if len(keys) == bs:
            yield keys
            keys = []
    if keys:
        yield keys


def create_tar(data_path):
    tar_path = f"{data_path}.tar"
    with tarfile.open(tar_path, "w") as tar:
        tar.add(data_path, arcname=os.path.basename(data_path))
    print(f"Created tar file: {tar_path}")
    return tar_path


def delete_directory(exp_name):
    if os.path.exists(exp_name):
        subprocess.run(["rm", "-r", exp_name], check=True)
        print(f"Deleted directory: {exp_name}")


@torch.inference_mode()
def visualize(config, args, model, items, bs, sample_steps, cfg_scale):
    if isinstance(items, dict):
        get_chunks = get_dict_chunks
    else:
        from diffusion.data.datasets.utils import get_chunks

    generator = torch.Generator(device=device).manual_seed(args.seed)

    # set scheduler
    if args.sampling_algo == "scm":
        scheduler = SCMScheduler()
    else:
        raise ValueError(f"Unsupported sampling algorithm: {args.sampling_algo}")

    scheduler.set_timesteps(
        num_inference_steps=sample_steps,
        max_timesteps=args.max_timesteps,
        intermediate_timesteps=args.intermediate_timesteps,
        timesteps=args.timesteps,
    )
    timesteps = scheduler.timesteps

    tqdm_desc = f"{save_root.split('/')[-1]} Using GPU: {args.gpu_id}: {args.start_index}-{args.end_index}"
    for chunk in tqdm(list(get_chunks(items, bs)), desc=tqdm_desc, unit="batch", position=args.gpu_id, leave=True):
        # data prepare
        prompts, hw, ar = (
            [],
            torch.tensor([[args.image_size, args.image_size]], dtype=torch.float, device=device).repeat(bs, 1),
            torch.tensor([[1.0]], device=device).repeat(bs, 1),
        )
        if bs == 1:
            prompt = data_dict[chunk[0]]["prompt"] if dict_prompt else chunk[0]
            prompt_clean, _, hw, ar, custom_hw = prepare_prompt_ar(prompt, base_ratios, device=device, show=False)
            latent_size_h, latent_size_w = (
                (int(hw[0, 0] // config.vae.vae_downsample_rate), int(hw[0, 1] // config.vae.vae_downsample_rate))
                if args.image_size == 1024
                else (latent_size, latent_size)
            )
            prompts.append(prompt_clean.strip())
        else:
            for data in chunk:
                prompt = data_dict[data]["prompt"] if dict_prompt else data
                prompts.append(prepare_prompt_ar(prompt, base_ratios, device=device, show=False)[0].strip())
            latent_size_h, latent_size_w = latent_size, latent_size

        # check exists
        save_file_name = f"{chunk[0]}.jpg" if dict_prompt else f"{prompts[0][:100]}.jpg"
        save_path = os.path.join(save_root, save_file_name)
        if os.path.exists(save_path):
            # make sure the noise is totally same
            torch.randn(bs, config.vae.vae_latent_dim, latent_size, latent_size, device=device, generator=generator)
            continue

        # prepare text feature
        if not config.text_encoder.chi_prompt:
            max_length_all = config.text_encoder.model_max_length
            prompts_all = prompts
        else:
            chi_prompt = "\n".join(config.text_encoder.chi_prompt)
            prompts_all = [chi_prompt + prompt for prompt in prompts]
            num_chi_prompt_tokens = len(tokenizer.encode(chi_prompt))
            max_length_all = (
                num_chi_prompt_tokens + config.text_encoder.model_max_length - 2
            )  # magic number 2: [bos], [_]

        caption_token = tokenizer(
            prompts_all, max_length=max_length_all, padding="max_length", truncation=True, return_tensors="pt"
        ).to(device)
        select_index = [0] + list(range(-config.text_encoder.model_max_length + 1, 0))
        caption_embs = text_encoder(caption_token.input_ids, caption_token.attention_mask)[0][:, None][
            :, :, select_index
        ]
        emb_masks = caption_token.attention_mask[:, select_index]

        # start sampling
        with torch.no_grad():
            n = len(prompts)
            latents = (
                torch.randn(
                    n,
                    config.vae.vae_latent_dim,
                    latent_size,
                    latent_size,
                    device=device,
                    generator=generator,
                )
                * sigma_data
            )
            model_kwargs = dict(
                data_info={
                    "img_hw": hw,
                    "aspect_ratio": ar,
                    "cfg_scale": torch.tensor([cfg_scale] * latents.shape[0]).to(device),
                },
                mask=emb_masks,
            )

            #  sCM MultiStep Sampling Loop:
            for i, t in enumerate(timesteps[:-1]):

                timestep = t.expand(latents.shape[0]).to(device)

                # model prediction
                model_pred = sigma_data * model(
                    latents / sigma_data,
                    timestep,
                    caption_embs,
                    **model_kwargs,
                )

                # compute the previous noisy sample x_t -> x_t-1
                latents, denoised = scheduler.step(model_pred, i, t, latents, return_dict=False)

        samples = (denoised / sigma_data).to(vae_dtype)
        samples = vae_decode(config.vae.vae_type, vae, samples)
        torch.cuda.empty_cache()

        os.umask(0o000)
        for i, sample in enumerate(samples):
            save_file_name = f"{chunk[i]}.jpg" if dict_prompt else f"{prompts[i][:100]}.jpg"
            save_path = os.path.join(save_root, save_file_name)
            save_image(sample, save_path, nrow=1, normalize=True, value_range=(-1, 1))


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="config")
    return parser.parse_known_args()[0]


@dataclass
class SanaInference(SanaConfig):
    config: Optional[
        str
    ] = "configs/sana_sprint_config/1024ms/SanaSprint_1600M_1024px_allqknorm_bf16_scm_ladd.yaml"  # config
    model_path: Optional[
        str
    ] = "hf://Efficient-Large-Model/Sana_Sprint_1.6B_1024px/checkpoints/Sana_Sprint_1.6B_1024px.pth"
    work_dir: Optional[str] = None
    txt_file: str = "asset/samples/samples_mini.txt"
    json_file: Optional[str] = None
    sample_nums: int = 100_000
    bs: int = 1
    cfg_scale: float = 1.0
    sampling_algo: str = "scm"
    max_timesteps: Optional[float] = 1.57080
    intermediate_timesteps: Optional[float] = 1.3
    timesteps: Optional[List[float]] = None
    seed: int = 0
    dataset: str = "custom"
    step: int = -1
    add_label: str = ""
    tar_and_del: bool = False
    exist_time_prefix: str = ""
    gpu_id: int = 0
    custom_image_size: Optional[int] = None
    start_index: int = 0
    end_index: int = 30_000
    interval_guidance: List[float] = field(default_factory=lambda: [0, 1])
    ablation_selections: Optional[List[float]] = None
    ablation_key: Optional[str] = None
    debug: bool = False
    if_save_dirname: bool = False


if __name__ == "__main__":

    args = get_args()
    config = args = pyrallis.parse(config_class=SanaInference, config_path=args.config)

    args.image_size = config.model.image_size
    if args.custom_image_size:
        args.image_size = args.custom_image_size
        print(f"custom_image_size: {args.image_size}")

    set_env(args.seed, args.image_size // config.vae.vae_downsample_rate)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger = get_root_logger()

    # only support fixed latent size currently
    latent_size = args.image_size // config.vae.vae_downsample_rate
    max_sequence_length = config.text_encoder.model_max_length
    guidance_type = "classifier-free"
    sigma_data = config.scheduler.sigma_data
    assert (
        isinstance(args.interval_guidance, list)
        and len(args.interval_guidance) == 2
        and args.interval_guidance[0] <= args.interval_guidance[1]
    )
    args.interval_guidance = [max(0, args.interval_guidance[0]), min(1, args.interval_guidance[1])]
    sample_steps_dict = {"scm": 2}
    sample_steps = args.step if args.step != -1 else sample_steps_dict[args.sampling_algo]

    weight_dtype = get_weight_dtype(config.model.mixed_precision)
    logger.info(f"Inference with {weight_dtype}, default guidance_type: {guidance_type}, ")

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
        config.model.model,
        use_fp32_attention=config.model.get("fp32_attention", False),
        logvar=config.model.logvar,
        cfg_embed=config.model.cfg_embed,
        cfg_embed_scale=config.model.cfg_embed_scale,
        **model_kwargs,
    ).to(device)
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

    if args.work_dir is None:
        work_dir = (
            f"/{os.path.join(*args.model_path.split('/')[:-2])}"
            if args.model_path.startswith("/")
            else os.path.join(*args.model_path.split("/")[:-2])
        )
    else:
        work_dir = args.work_dir
    config.work_dir = work_dir
    img_save_dir = os.path.join(str(work_dir), "vis")

    logger.info(colored(f"Saving images at {img_save_dir}", "green"))
    dict_prompt = args.json_file is not None
    if dict_prompt:
        data_dict = json.load(open(args.json_file))
        items = list(data_dict.keys())
    else:
        with open(args.txt_file) as f:
            items = [item.strip() for item in f.readlines()]
    logger.info(f"Eval first {min(args.sample_nums, len(items))}/{len(items)} samples")
    items = items[: max(0, args.sample_nums)]
    items = items[max(0, args.start_index) : min(len(items), args.end_index)]

    match = re.search(r".*epoch_(\d+).*step_(\d+).*", args.model_path)
    epoch_name, step_name = match.groups() if match else ("unknown", "unknown")

    os.umask(0o000)
    os.makedirs(img_save_dir, exist_ok=True)
    logger.info(f"Sampler {args.sampling_algo}")

    def create_save_root(args, dataset, epoch_name, step_name, sample_steps, guidance_type):
        save_root = os.path.join(
            img_save_dir,
            f"{dataset}_epoch{epoch_name}_step{step_name}_scale{args.cfg_scale}"
            f"_step{sample_steps}_size{args.image_size}_bs{args.bs}_samp{args.sampling_algo}"
            f"_seed{args.seed}_{str(weight_dtype).split('.')[-1]}",
        )

        save_root += f"_maxT{args.max_timesteps}"
        if args.intermediate_timesteps != 1.3:
            save_root += f"_midT{args.intermediate_timesteps}"
        if args.timesteps:
            save_root += f"_timesteps{args.timesteps}"
        save_root += f"_imgnums{args.sample_nums}" + args.add_label
        return save_root

    dataset = "MJHQ-30K" if args.json_file and "MJHQ-30K" in args.json_file else args.dataset
    if args.ablation_selections and args.ablation_key:
        for ablation_factor in args.ablation_selections:
            setattr(args, args.ablation_key, eval(ablation_factor))
            print(f"Setting {args.ablation_key}={eval(ablation_factor)}")
            sample_steps = args.step if args.step != -1 else sample_steps_dict[args.sampling_algo]

            save_root = create_save_root(args, dataset, epoch_name, step_name, sample_steps, guidance_type)
            os.makedirs(save_root, exist_ok=True)
            if args.if_save_dirname and args.gpu_id == 0:
                os.makedirs(f"{work_dir}/metrics", exist_ok=True)
                # save at work_dir/metrics/tmp_xxx.txt for metrics testing
                with open(f"{work_dir}/metrics/tmp_{dataset}_{time.time()}.txt", "w") as f:
                    print(f"save tmp file at {work_dir}/metrics/tmp_{dataset}_{time.time()}.txt")
                    f.write(os.path.basename(save_root))
            logger.info(f"Inference with {weight_dtype}, guidance_type: {guidance_type}")

            visualize(
                config=config,
                args=args,
                model=model,
                items=items,
                bs=args.bs,
                sample_steps=sample_steps,
                cfg_scale=args.cfg_scale,
            )
    else:
        logger.info(f"Inference with {weight_dtype}, guidance_type: {guidance_type}")

        save_root = create_save_root(args, dataset, epoch_name, step_name, sample_steps, guidance_type)
        os.makedirs(save_root, exist_ok=True)
        if args.if_save_dirname and args.gpu_id == 0:
            os.makedirs(f"{work_dir}/metrics", exist_ok=True)
            # save at work_dir/metrics/tmp_xxx.txt for metrics testing
            with open(f"{work_dir}/metrics/tmp_{dataset}_{time.time()}.txt", "w") as f:
                print(f"save tmp file at {work_dir}/metrics/tmp_{dataset}_{time.time()}.txt")
                f.write(os.path.basename(save_root))

        if args.debug:
            items = [
                "portrait photo of a girl, photograph, highly detailed face, depth of field",
                "Self-portrait oil painting, a beautiful cyborg with golden hair, 8k",
                "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k",
                "A photo of beautiful mountain with realistic sunset and blue lake, highly detailed, masterpiece",
            ]
        visualize(
            config=config,
            args=args,
            model=model,
            items=items,
            bs=args.bs,
            sample_steps=sample_steps,
            cfg_scale=args.cfg_scale,
        )

        if args.tar_and_del:
            create_tar(save_root)
            delete_directory(save_root)

    print(
        colored(f"Sana inference has finished. Results stored at ", "green"),
        colored(f"{img_save_dir}", attrs=["bold"]),
        ".",
    )
