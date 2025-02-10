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
import warnings
from dataclasses import dataclass, field
from typing import List, Optional

import pyrallis
import torch
from torchvision.utils import save_image
from tqdm import tqdm

warnings.filterwarnings("ignore")  # ignore warning

import cv2
from termcolor import colored

from diffusion import DPMS
from diffusion.data.datasets.utils import ASPECT_RATIO_512_TEST, ASPECT_RATIO_1024_TEST
from diffusion.model.builder import build_model, get_tokenizer_and_text_encoder, get_vae, vae_decode, vae_encode
from diffusion.model.utils import prepare_prompt_ar
from diffusion.utils.config import SanaConfig, model_init_config
from diffusion.utils.logger import get_root_logger
from tools.controlnet.utils import get_scribble_map, transform_control_signal
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


def create_save_root(args, dataset, epoch_name, step_name, sample_steps, guidance_type):
    save_root = os.path.join(
        img_save_dir,
        f"{dataset}_epoch{epoch_name}_step{step_name}_scale{args.cfg_scale}"
        f"_step{sample_steps}_size{args.image_size}_bs{args.bs}_samp{args.sampling_algo}"
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


def get_ar_from_ref_image(ref_image_path):
    def reduce_ratio(h, w):
        def gcd(a, b):
            while b:
                a, b = b, a % b
            return a

        divisor = gcd(h, w)
        return f"{h // divisor}:{w // divisor}"

    ref_image = cv2.imread(ref_image_path)
    h, w = ref_image.shape[:2]
    return reduce_ratio(h, w)


@torch.inference_mode()
def visualize(config, args, model, items, bs, sample_steps, cfg_scale, pag_scale=1.0):
    assert bs == 1, "only support batch size 1 currently"

    if isinstance(items, dict):
        get_chunks = get_dict_chunks
    else:
        from diffusion.data.datasets.utils import get_chunks

    generator = torch.Generator(device=device).manual_seed(args.seed)
    tqdm_desc = f"{save_root.split('/')[-1]} Using GPU: {args.gpu_id}: {args.start_index}-{args.end_index}"
    for chunk in tqdm(list(get_chunks(items, bs)), desc=tqdm_desc, unit="batch", position=args.gpu_id, leave=True):
        # data prepare
        prompts, hw, ar = (
            [],
            torch.tensor([[args.image_size, args.image_size]], dtype=torch.float, device=device).repeat(bs, 1),
            torch.tensor([[1.0]], device=device).repeat(bs, 1),
        )

        if "ref_image_path" in chunk[0]:
            prompt, ref_image_path = chunk[0]["prompt"], chunk[0]["ref_image_path"]
            args.reference_image_path = ref_image_path
            ar = get_ar_from_ref_image(args.reference_image_path)
        else:
            assert "ref_controlmap_path" in chunk[0], "neither ref_image_path nor ref_controlmap_path is provided"
            prompt, ref_controlmap_path = chunk[0]["prompt"], chunk[0]["ref_controlmap_path"]
            args.controlmap_path = ref_controlmap_path
            ar = get_ar_from_ref_image(args.controlmap_path)

        prompt += f" --ar {ar}"
        prompt_clean, _, hw, ar, custom_hw = prepare_prompt_ar(prompt, base_ratios, device=device, show=False)
        latent_size_h, latent_size_w = (
            (int(hw[0, 0] // config.vae.vae_downsample_rate), int(hw[0, 1] // config.vae.vae_downsample_rate))
            if args.image_size == 1024
            else (latent_size, latent_size)
        )
        prompts.append(prompt_clean.strip())

        # check exists
        save_file_name = f"{prompts[0]}.jpg"
        save_path = os.path.join(save_root, save_file_name)
        if os.path.exists(save_path):
            # make sure the noise is totally same
            torch.randn(bs, config.vae.vae_latent_dim, latent_size_h, latent_size_w, device=device, generator=generator)
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
        null_y = null_caption_embs.repeat(len(prompts), 1, 1)[:, None]

        # start sampling
        with torch.no_grad():
            n = len(prompts)
            z = torch.randn(
                n, config.vae.vae_latent_dim, latent_size_h, latent_size_w, device=device, generator=generator
            )
            model_kwargs = dict(data_info={"img_hw": hw, "aspect_ratio": ar}, mask=emb_masks)

            if args.reference_image_path is not None:
                input_image = cv2.imread(args.reference_image_path)
                control_signal = get_scribble_map(
                    input_image=input_image,
                    det="Scribble_HED",
                    detect_resolution=int(hw.min()),
                    thickness=int(args.thickness),
                )
                control_signal = transform_control_signal(control_signal, hw).to(device).to(weight_dtype)
            else:
                control_signal = transform_control_signal(args.controlmap_path, hw).to(device).to(weight_dtype)

            control_signal_latent = vae_encode(
                config.vae.vae_type, vae, control_signal, config.vae.sample_posterior, device
            )

            model_kwargs["control_signal"] = control_signal_latent
            if args.sampling_algo == "flow_dpm-solver":
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

        samples = samples.to(weight_dtype)
        samples = vae_decode(config.vae.vae_type, vae, samples)
        torch.cuda.empty_cache()

        return dict(samples=samples, control_signal=control_signal)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="config")
    parser.add_argument("--model_path", default=None, type=str, help="Path to the model file (optional)")

    return parser.parse_known_args()[0]


@dataclass
class SanaInference(SanaConfig):
    config: Optional[str] = ""
    model_path: Optional[str] = "output/pretrained_models/Sana_1600M_1024px.pth"
    work_dir: str = "output/inference"
    version: str = "sigma"
    txt_file: str = "asset/samples/samples_mini.txt"
    json_file: Optional[str] = None
    sample_nums: int = 100_000
    bs: int = 1
    cfg_scale: float = 4.5
    pag_scale: float = 1.0
    sampling_algo: str = "flow_dpm-solver"
    seed: int = 0
    dataset: str = "custom_controlnet"
    step: int = -1
    add_label: str = ""
    tar_and_del: bool = False
    exist_time_prefix: str = ""
    gpu_id: int = 0
    start_index: int = 0
    end_index: int = 30_000
    interval_guidance: List[float] = field(default_factory=lambda: [0, 1])
    ablation_selections: Optional[List[float]] = None
    ablation_key: Optional[str] = None
    debug: bool = False
    if_save_dirname: bool = False
    # controlnet
    reference_image_path: Optional[str] = None
    controlmap_path: Optional[str] = None
    thickness: int = 2
    blend_alpha: float = 0.0


if __name__ == "__main__":

    args = get_args()
    config = args = pyrallis.parse(config_class=SanaInference, config_path=args.config)

    args.image_size = config.model.image_size

    if args.json_file is None:
        assert (args.reference_image_path is None) != (
            args.controlmap_path is None
        ), "only one of reference_image_path/controlmap_path can be None"

    set_env(args.seed, args.image_size // config.vae.vae_downsample_rate)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger = get_root_logger()

    # only support fixed latent size currently
    latent_size = args.image_size // config.vae.vae_downsample_rate
    max_sequence_length = config.text_encoder.model_max_length
    pe_interpolation = config.model.pe_interpolation
    micro_condition = config.model.micro_condition
    flow_shift = config.scheduler.flow_shift
    pag_applied_layers = config.model.pag_applied_layers
    guidance_type = "classifier-free_PAG"
    assert (
        isinstance(args.interval_guidance, list)
        and len(args.interval_guidance) == 2
        and args.interval_guidance[0] <= args.interval_guidance[1]
    )
    args.interval_guidance = [max(0, args.interval_guidance[0]), min(1, args.interval_guidance[1])]
    sample_steps_dict = {"flow_dpm-solver": 20, "flow_euler": 28}
    sample_steps = args.step if args.step != -1 else sample_steps_dict[args.sampling_algo]
    if config.model.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif config.model.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    elif config.model.mixed_precision == "fp32":
        weight_dtype = torch.float32
    else:
        raise ValueError(f"weigh precision {config.model.mixed_precision} is not defined")
    logger.info(f"Inference with {weight_dtype}, default guidance_type: {guidance_type}, flow_shift: {flow_shift}")

    vae = get_vae(config.vae.vae_type, config.vae.vae_pretrained, device).to(weight_dtype)
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
    args.sampling_algo = (
        args.sampling_algo
        if ("flow" not in args.model_path or args.sampling_algo == "flow_dpm-solver")
        else "flow_euler"
    )

    if args.work_dir is None:
        work_dir = (
            f"/{os.path.join(*args.model_path.split('/')[:-2])}"
            if args.model_path.startswith("/")
            else os.path.join(*args.model_path.split("/")[:-2])
        )
        img_save_dir = os.path.join(str(work_dir), "vis")
    else:
        img_save_dir = args.work_dir

    dict_prompt = args.json_file is not None
    if dict_prompt:
        data_dict = json.load(open(args.json_file))
        items = data_dict
        args.sample_nums = len(items)
    else:
        raise ValueError("json_file is not provided")

    match = re.search(r".*epoch_(\d+).*step_(\d+).*", args.model_path)
    epoch_name, step_name = match.groups() if match else ("unknown", "unknown")

    os.umask(0o000)
    os.makedirs(img_save_dir, exist_ok=True)
    logger.info(f"Sampler {args.sampling_algo}")

    dataset = "MJHQ-30K" if args.json_file and "MJHQ-30K" in args.json_file else args.dataset

    guidance_type = guidance_type_select(guidance_type, args.pag_scale, config.model.attn_type)
    logger.info(f"Inference with {weight_dtype}, guidance_type: {guidance_type}, flow_shift: {flow_shift}")
    save_root = create_save_root(args, dataset, epoch_name, step_name, sample_steps, guidance_type)
    os.makedirs(save_root, exist_ok=True)

    if args.debug:
        print(f"debug mode, use fixed items")
        pass

    for idx, item in enumerate(items):
        # args.seed = idx
        results = visualize(
            config=config,
            args=args,
            model=model,
            items=[item],
            bs=args.bs,
            sample_steps=sample_steps,
            cfg_scale=args.cfg_scale,
            pag_scale=args.pag_scale,
        )
        os.umask(0o000)
        sample, control_signal = results["samples"][0], results["control_signal"][0]
        # 混合mask和image
        if args.blend_alpha > 0:
            print(f"blend image and mask with alpha: {args.blend_alpha}")
            sample = sample * (1 - args.blend_alpha) + control_signal * args.blend_alpha

        save_file_name = f"{idx}_{item['prompt'][:100]}.jpg"
        save_path = os.path.join(save_root, save_file_name)
        save_image(sample, save_path, nrow=1, normalize=True, value_range=(-1, 1))

    print(
        colored(f"Sana inference has finished. Results stored at ", "green"),
        colored(f"{img_save_dir}", attrs=["bold"]),
        ".",
    )
