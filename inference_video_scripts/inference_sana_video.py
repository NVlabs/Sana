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
import hashlib
import json
import os
import random
import re
import time
import warnings
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import imageio
import pyrallis
import torch
from accelerate import Accelerator
from termcolor import colored
from tqdm import tqdm

warnings.filterwarnings("ignore")  # ignore warning
os.environ["DISABLE_XFORMERS"] = "1"

from diffusion import DPMS, FlowEuler, LongLiveFlowEuler, LTXFlowEuler
from diffusion.data.datasets.utils import *
from diffusion.data.transforms import read_image_from_path
from diffusion.guiders import AdaptiveProjectedGuidance
from diffusion.model.builder import (
    build_model,
    encode_image,
    get_tokenizer_and_text_encoder,
    get_vae,
    vae_decode,
    vae_encode,
)
from diffusion.model.utils import get_weight_dtype, prepare_prompt_ar
from diffusion.utils.config import SanaVideoConfig, model_video_init_config
from diffusion.utils.logger import get_root_logger
from tools.download import find_model

os.environ["TOKENIZERS_PARALLELISM"] = "false"


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


class DistributePromptsDataset(torch.utils.data.Dataset):
    """Dataset for vbench inference.

    Args:
        prompts: Dictionary with keys and prompt tuples as values, or list of prompts
        original_indices: List of original indices from txt file corresponding to each prompt
    """

    def __init__(self, prompts, original_indices=None):
        if isinstance(prompts, dict):
            self.prompts = prompts
            self.keys_list = list(self.prompts.keys())
            self.original_indices = original_indices or list(range(len(prompts)))
        else:
            # Convert list to dict where key and value are the same
            self.prompts = {
                prompt[:50].split("/")[0] + str(hashlib.sha256(prompt.encode()).hexdigest())[:10]: prompt
                for prompt in prompts
            }
            self.keys_list = list(self.prompts.keys())
            self.original_indices = original_indices or list(range(len(prompts)))

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        key = self.keys_list[idx]
        prompt = self.prompts[key]
        txt_line_idx = self.original_indices[idx]
        return {
            "key": key,
            "prompt": prompt,
            "global_idx": txt_line_idx,
        }


@torch.inference_mode()
def visualize(config, args, model, items, bs, sample_steps, cfg_scale):

    cur_seed = args.seed + int(rank)
    generator = torch.Generator(device=device).manual_seed(cur_seed)
    tqdm_desc = f"{save_root.split('/')[-1]} Using GPU: {args.gpu_id}: {args.start_index}-{args.end_index}"
    for chunk in tqdm(prompts_dataloader, desc=tqdm_desc, unit="batch", position=rank, leave=True):
        # data prepare
        prompts, hw = (
            [],
            torch.tensor([[args.image_size, args.image_size]], dtype=torch.float, device=device).repeat(bs, 1),
        )
        images = []
        if bs == 1:
            prompt = chunk["prompt"][0]
            prompt_clean, _, hw, _, _ = prepare_prompt_ar(prompt, base_ratios, device=device, show=False)
            if config.task == "ti2v" or config.task == "ltx":
                prompt_clean, image_path = prompt_clean.split(image_split_token)
                images.append(image_path)
            if args.prompt_split_token in prompt_clean:
                prompt_clean = prompt_clean.split(args.prompt_split_token)
                prompts.extend([_prompt.strip() + motion_prompt for _prompt in prompt_clean])
            else:
                prompts.append(prompt_clean.strip() + motion_prompt)
        else:
            for prompt in chunk["prompt"]:
                prompt_clean, _, hw, _, _ = prepare_prompt_ar(prompt, base_ratios, device=device, show=False)
                if config.task == "ti2v" or config.task == "ltx":
                    prompt_clean, image_path = prompt_clean.split(image_split_token)
                    images.append(image_path)
                prompts.append(prompt_clean.strip() + motion_prompt)
        latent_size_h, latent_size_w = (
            int(hw[0, 0] // config.vae.vae_downsample_rate),
            int(hw[0, 1] // config.vae.vae_downsample_rate),
        )

        # check exists
        exist = False
        for i in range(bs):
            save_file_name = f"{chunk['global_idx'][i]}. {chunk['key'][i]}.mp4"
            save_path = os.path.join(save_root, save_file_name)
            exist = os.path.exists(save_path)
            if not exist:
                break
        if exist:
            # make sure the noise is totally same
            torch.randn(
                bs,
                config.vae.vae_latent_dim,
                latent_size_t,
                latent_size_h,
                latent_size_w,
                device=device,
                generator=generator,
            )
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

        if "Qwen" in config.text_encoder.text_encoder_name:
            with torch.no_grad():
                caption_embs, emb_masks = text_handler.get_prompt_embeds(prompts_all, max_length=max_length_all)
                caption_embs = caption_embs[:, None]
                negative_embs_mask = negative_caption_embs_mask.repeat(bs, 1)  # B, L
                negative_embs = negative_caption_embs.repeat(bs, 1, 1)[:, None]
        else:
            caption_token = tokenizer(
                prompts_all, max_length=max_length_all, padding="max_length", truncation=True, return_tensors="pt"
            ).to(device)
            select_index = [0] + list(
                range(-config.text_encoder.model_max_length + 1, 0)
            )  # first one is bos, the rest is the text after chi_prompt
            caption_embs = text_encoder(caption_token.input_ids, caption_token.attention_mask)[0][:, None][
                :, :, select_index
            ]
            emb_masks = caption_token.attention_mask[:, select_index]  # B, L
            negative_embs_mask = negative_caption_token.attention_mask.repeat(bs, 1)  # B, L
            negative_embs = negative_caption_embs.repeat(bs, 1, 1)[:, None]

        if cfg_scale > 1.0:
            emb_masks = torch.cat([negative_embs_mask, emb_masks], dim=0)  # 2B, L
        other_kwargs = dict(
            stg_applied_layers=args.stg_applied_layers,
            stg_scale=args.stg_scale,
        )
        # start sampling
        with torch.no_grad():
            n = bs
            z = torch.randn(
                n,
                config.vae.vae_latent_dim,
                latent_size_t,
                latent_size_h,
                latent_size_w,
                device=device,
                generator=generator,
            )
            model_kwargs = dict(data_info={"img_hw": hw}, mask=emb_masks)

            if config.task == "ltx":
                images = [
                    read_image_from_path(
                        imgp,
                        (
                            int(latent_size_h * config.vae.vae_downsample_rate),
                            int(latent_size_w * config.vae.vae_downsample_rate),
                        ),
                    )
                    for imgp in images
                ]  # C,H,W

                image_vae_embeds = vae_encode(
                    config.vae.vae_type, vae, torch.stack(images, dim=0)[:, :, None].to(vae_dtype), device=device
                )  # 1,C,1,H,W
                condition_frame_info = {
                    0: (
                        config.train.noise_multiplier if config.train.noise_multiplier is not None else 0.0
                    ),  # frame_idx: frame_weight, weight is used for timestep
                }
                for frame_idx in list(condition_frame_info.keys()):
                    z[:, :, frame_idx : frame_idx + 1] = image_vae_embeds  # 1,C,F,H,W, first frame is the image
                model_kwargs["data_info"].update({"condition_frame_info": condition_frame_info})  # B,C,F,H,W

                if image_encoder is not None:
                    image_embeds = encode_image(
                        name=config.image_encoder.image_encoder_name,
                        image_encoder=image_encoder,
                        image_processor=image_processor,
                        images=torch.stack(images, dim=0).to(device),
                        device=device,
                        dtype=weight_dtype,
                    )
                    if cfg_scale > 1.0:
                        image_embeds = torch.cat([image_embeds, image_embeds], dim=0)  # 2,C,1,H,W

                    model_kwargs["data_info"].update({"image_embeds": image_embeds})  # B,C,F,H,W

            if args.sampling_algo == "flow_euler_ltx":
                flow_solver = LTXFlowEuler(
                    model,
                    condition=caption_embs,
                    uncondition=negative_embs,
                    cfg_scale=cfg_scale,
                    flow_shift=flow_shift,
                    model_kwargs=model_kwargs,
                )
                samples = flow_solver.sample(
                    z,
                    steps=sample_steps,
                    generator=generator,
                )
            elif args.sampling_algo == "flow_euler":
                flow_solver = FlowEuler(
                    model,
                    condition=caption_embs,
                    uncondition=negative_embs,
                    cfg_scale=cfg_scale,
                    flow_shift=flow_shift,
                    model_kwargs=model_kwargs,
                    apg=apg,
                )
                samples = flow_solver.sample(z, steps=sample_steps)
            elif args.sampling_algo == "flow_dpm-solver":
                dpm_solver = DPMS(
                    model,
                    condition=caption_embs,
                    uncondition=negative_embs,
                    cfg_scale=cfg_scale,
                    model_type="flow",
                    guidance_type=guidance_type,
                    model_kwargs=model_kwargs,
                    schedule="FLOW",
                    apg=apg,
                    **other_kwargs,
                )
                samples = dpm_solver.sample(
                    z,
                    steps=sample_steps,
                    order=2,
                    skip_type=args.skip_type,
                    method="multistep",
                    flow_shift=flow_shift,
                )
            elif args.sampling_algo == "longlive_flow_euler":
                base_chunk_frames = base_model_frames // config.vae.vae_stride[0]
                flow_solver = LongLiveFlowEuler(
                    model,
                    condition=caption_embs,
                    flow_shift=flow_shift,
                    model_kwargs=model_kwargs,
                    base_chunk_frames=base_chunk_frames,
                    num_cached_blocks=args.num_cached_blocks,
                )
                samples = flow_solver.sample(
                    z,
                    steps=sample_steps,
                    generator=generator,
                )
            else:
                raise ValueError(f"{args.sampling_algo} is not defined")

        samples = samples.to(vae_dtype)
        samples = vae_decode(config.vae.vae_type, vae, samples)
        if isinstance(samples, list):
            samples = torch.stack(samples, dim=0)
        videos = (
            torch.clamp(127.5 * samples + 127.5, 0, 255).permute(0, 2, 3, 4, 1).to("cpu", dtype=torch.uint8)
        )  # B,C,T,H,W -> B,T,H,W,C
        torch.cuda.empty_cache()

        os.umask(0o000)
        for i, video in enumerate(videos):
            save_file_name = f"{chunk['global_idx'][i]:03d}. {chunk['key'][i]}.mp4"
            save_path = os.path.join(save_root, save_file_name)
            writer = imageio.get_writer(save_path, fps=args.fps, codec="libx264", quality=8)
            for frame in video.numpy():
                writer.append_data(frame)
            writer.close()


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="config")
    return parser.parse_known_args()[0]


@dataclass
class SanaInference(SanaVideoConfig):
    config: Optional[str] = "configs/sana_video_config/Sana_2000M_480px_AdamW_fsdp.yaml"  # config
    model_path: Optional[str] = "hf://Efficient-Large-Model/SANA-Video_2B_480p/checkpoints/SANA_Video_2B_480p.pth"
    work_dir: Optional[str] = None
    txt_file: str = "asset/samples/video_prompts_samples.txt"
    json_file: Optional[str] = None
    fps: int = 16
    sample_nums: int = 100_000
    bs: int = 1
    num_frames: int = -1
    cfg_scale: float = 6.0
    flow_shift: Optional[float] = None
    sampling_algo: Optional[str] = None
    skip_type: str = "time_uniform_flow"  # time_uniform_flow, linear_quadratic
    guidance_type: str = "classifier-free"  # [classifier-free, adaptive_projected_guidance, classifier-free_STG]
    seed: int = 0
    dataset: str = ""
    step: int = -1
    add_label: str = ""
    tar_and_del: bool = False
    exist_time_prefix: str = ""
    gpu_id: int = 0
    custom_height_width: Optional[Tuple[int, int]] = None
    use_resolution_binning: bool = True
    start_index: int = 0
    end_index: int = 30_000
    ablation_selections: Optional[List[float]] = None
    ablation_key: Optional[str] = None
    debug: bool = False
    if_save_dirname: bool = False
    image_split_token: str = "<image>"
    high_motion: bool = False
    prompt_split_token: str = "<split>"
    motion_score: int = 10
    negative_prompt: str = (
        "A chaotic sequence with misshapen, deformed limbs in heavy motion blur, sudden disappearance, jump cuts, jerky movements, rapid shot changes, frames out of sync, inconsistent character shapes, temporal artifacts, jitter, and ghosting effects, creating a disorienting visual experience."
    )
    interval_k: float = 0.0
    unified_noise: bool = False
    stg_applied_layers: List[int] = field(default_factory=list)
    stg_scale: float = 0.0
    apg_mode: str = "hw"
    num_cached_blocks: int = -1


if __name__ == "__main__":

    args = get_args()
    config = args = pyrallis.parse(config_class=SanaInference, config_path=args.config)

    args.image_size = config.model.image_size
    set_env(args.seed, args.image_size // config.vae.vae_downsample_rate)

    accelerator = Accelerator(mixed_precision=config.model.mixed_precision)
    device = accelerator.device
    rank = int(os.environ.get("RANK", 0))
    logger = get_root_logger()

    # only support fixed latent size currently
    if args.use_resolution_binning and args.custom_height_width is None:
        if config.vae.vae_downsample_rate in [16, 32]:
            base_ratios = eval(f"ASPECT_RATIO_VIDEO_{args.image_size}_TEST_DIV32")
        else:
            base_ratios = eval(f"ASPECT_RATIO_VIDEO_{args.image_size}_TEST")
        aspect_ratio_key = random.choice(list(base_ratios.keys()))
        video_height, video_width = map(int, base_ratios[aspect_ratio_key])
    elif args.custom_height_width is not None:
        video_height, video_width = args.custom_height_width
        base_ratios = {f"{video_height/video_width:.2f}": [float(video_height), float(video_width)]}
    else:
        logger.info(f"Using default height and width: 480, 832")
        video_height, video_width = 480, 832
        base_ratios = {f"{video_height/video_width:.2f}": [float(video_height), float(video_width)]}
    latent_size_w = int(video_width) // config.vae.vae_stride[2]
    latent_size_h = int(video_height) // config.vae.vae_stride[1]
    latent_size = args.image_size // config.vae.vae_downsample_rate
    base_model_frames = config.data.num_frames
    num_frames = config.data.num_frames if args.num_frames == -1 else args.num_frames
    latent_size_t = int(num_frames - 1) // config.vae.vae_stride[0] + 1
    logger.info(f"Latent size: {latent_size_t}t, {latent_size_h}h, {latent_size_w}w")
    max_sequence_length = config.text_encoder.model_max_length
    if args.flow_shift is not None:
        flow_shift = args.flow_shift
    else:
        flow_shift = (
            config.scheduler.inference_flow_shift
            if config.scheduler.inference_flow_shift is not None
            else config.scheduler.flow_shift
        )
    if args.motion_score > 0:
        motion_prompt = f" motion score: {int(args.motion_score)}."
    else:
        motion_prompt = " high motion" if args.high_motion else " low motion"
    if config.negative_prompt is None or config.negative_prompt == "None":
        config.negative_prompt = ""
    # if negative_prompt is a dict, convert it to a string
    elif config.negative_prompt.startswith("{") and config.negative_prompt.endswith("}"):
        negative_prompt_dict = eval(config.negative_prompt)
        negative_parts = []
        for key, value in negative_prompt_dict.items():
            negative_parts.append(f"{key}: {value}")
        config.negative_prompt = " ".join(negative_parts)
    logger.info(f"negative_prompt: {config.negative_prompt}")

    if args.sampling_algo == "longlive_flow_euler":
        assert args.cfg_scale == 1.0, "cfg_scale must be 1.0 for longlive_flow_euler"

    guidance_type = args.guidance_type
    if guidance_type == "adaptive_projected_guidance":
        apg = AdaptiveProjectedGuidance(
            guidance_scale=args.cfg_scale,
            adaptive_projected_guidance_momentum=-0.5,
            adaptive_projected_guidance_rescale=27,
            eta=1.0,
            mode=args.apg_mode,
        )
    else:
        apg = None
    sample_steps = args.step if args.step != -1 else 50
    image_split_token = args.image_split_token

    weight_dtype = get_weight_dtype(config.model.mixed_precision)
    logger.info(f"Inference with {weight_dtype}, default guidance_type: {guidance_type}, flow_shift: {flow_shift}")
    logger.info(f"motion_prompt: {motion_prompt}")

    vae_dtype = get_weight_dtype(config.vae.weight_dtype)
    vae = get_vae(config.vae.vae_type, config.vae.vae_pretrained, device=device, dtype=vae_dtype, config=config.vae)
    tokenizer, text_encoder = get_tokenizer_and_text_encoder(name=config.text_encoder.text_encoder_name, device=device)
    if "Qwen" in config.text_encoder.text_encoder_name:
        text_handler = text_encoder
        text_encoder = text_handler.text_encoder
        negative_caption_embs, negative_caption_embs_mask = text_handler.get_prompt_embeds(
            config.negative_prompt, max_length=max_sequence_length
        )
    else:
        negative_caption_token = tokenizer(
            config.negative_prompt,
            max_length=max_sequence_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).to(device)
        negative_caption_embs = text_encoder(negative_caption_token.input_ids, negative_caption_token.attention_mask)[0]

    image_encoder, image_processor = None, None

    # model setting
    model_kwargs = model_video_init_config(config, latent_size=latent_size)
    model = build_model(
        config.model.model,
        use_fp32_attention=config.model.get("fp32_attention", False),
        **model_kwargs,
    ).to(device)
    logger.info(
        f"{model.__class__.__name__}:{config.model.model}, Model Parameters: {sum(p.numel() for p in model.parameters()):,}"
    )

    logger.info(f"Generating sample from ckpt: {args.model_path}")
    state_dict = find_model(args.model_path)
    if "generator" in state_dict:  # used for loading LongSANA checkpoints
        state_dict = state_dict["generator"]
    if not "state_dict" in state_dict:
        new_state_dict = dict()
        for k, v in state_dict.items():
            if k.startswith("model."):
                k = k[len("model.") :]
            new_state_dict[k] = v
        state_dict = {"state_dict": new_state_dict}

    if args.model_path.endswith(".bin"):
        logger.info("Loading fsdp bin checkpoint....")
        old_state_dict = state_dict
        state_dict = dict()
        state_dict["state_dict"] = old_state_dict

    if "pos_embed" in state_dict["state_dict"]:
        del state_dict["state_dict"]["pos_embed"]

    missing, unexpected = model.load_state_dict(state_dict["state_dict"], strict=False)
    logger.warning(f"Missing keys: {missing}")
    logger.warning(f"Unexpected keys: {unexpected}")
    model.eval().to(weight_dtype)

    args.sampling_algo = config.scheduler.vis_sampler if args.sampling_algo is None else args.sampling_algo
    if config.task == "ltx" or config.task == "ti2v":
        args.sampling_algo = "flow_euler_ltx"

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

    logger.info(colored(f"Saving videos at {img_save_dir}", "green"))
    dict_prompt = args.json_file is not None
    if dict_prompt:
        data_dict = json.load(open(args.json_file))
        all_items = list(data_dict.keys())
        logger.info(f"Eval first {min(args.sample_nums, len(all_items))}/{len(all_items)} samples")
        total_items = all_items[: max(0, args.sample_nums)]
        start_idx = max(0, args.start_index)
        end_idx = min(len(total_items), args.end_index)
        items = total_items[start_idx:end_idx]
        original_indices = list(range(start_idx, end_idx))
    else:
        with open(args.txt_file) as f:
            all_items = [item.strip() for item in f.readlines()]
        logger.info(f"Eval first {min(args.sample_nums, len(all_items))}/{len(all_items)} samples")
        total_items = all_items[: max(0, args.sample_nums)]
        start_idx = max(0, args.start_index)
        end_idx = min(len(total_items), args.end_index)
        items = total_items[start_idx:end_idx]
        original_indices = list(range(start_idx, end_idx))

    match = re.search(r".*epoch_(\d+).*step_(\d+).*", args.model_path)
    epoch_name, step_name = match.groups() if match else ("unknown", "unknown")

    os.umask(0o000)
    os.makedirs(img_save_dir, exist_ok=True)
    logger.info(f"Sampler {args.sampling_algo}, t_type: {args.skip_type}")

    # prepare prompts dataset
    prompts_dataset = DistributePromptsDataset(items, original_indices)
    prompts_dataloader = torch.utils.data.DataLoader(prompts_dataset, batch_size=args.bs, shuffle=False)
    # prepare dataloader, model, text encoder
    prompts_dataloader, model, text_encoder = accelerator.prepare(prompts_dataloader, model, text_encoder)
    if num_frames > base_model_frames:
        assert args.sampling_algo == "longlive_flow_euler"

    def create_save_root(args, dataset, epoch_name, step_name, sample_steps, num_frames):
        save_root = os.path.join(
            img_save_dir,
            f"{dataset}_step{step_name}_scale{args.cfg_scale}"
            f"_step{sample_steps}_size{args.image_size}_numframes{num_frames}_bs{args.bs}_samp{args.sampling_algo}"
            f"_seed{args.seed}_{str(weight_dtype).split('.')[-1]}",
        )

        if args.skip_type != "time_uniform_flow":
            save_root += f"_skip-{args.skip_type}"
        if args.skip_type == "time_uniform_flow" and flow_shift != 1.0:
            save_root += f"_flowshift{flow_shift}"
        if args.interval_k > 0:
            save_root += f"_interval_k{int(args.interval_k*1000)}"
        if args.num_cached_blocks > 0:
            save_root += f"_numcb{args.num_cached_blocks}"
        if args.high_motion:
            save_root += f"_highmotion"
        if args.motion_score > 0:
            save_root += f"_motion{args.motion_score}"
        if args.negative_prompt != "":
            save_root += f"_negp{args.negative_prompt[:5].replace(' ', '')}"

        save_root += f"_gd{''.join(word[0] for word in args.guidance_type.split('_'))}"
        if args.guidance_type == "adaptive_projected_guidance":
            save_root += f"_{args.apg_mode}"
        if args.guidance_type == "classifier-free_STG":
            save_root += f"_stg{args.stg_scale}_stgl{''.join(str(layer) for layer in args.stg_applied_layers)}"
        save_root += f"_imgnums{args.sample_nums}" + args.add_label
        return save_root

    dataset = args.dataset

    logger.info(f"Inference with {weight_dtype}, flow_shift: {flow_shift}")

    save_root = create_save_root(args, dataset, epoch_name, step_name, sample_steps, num_frames)
    os.makedirs(save_root, exist_ok=True)
    if args.if_save_dirname and args.gpu_id == 0:
        os.makedirs(f"{work_dir}/metrics", exist_ok=True)
        # save at work_dir/metrics/tmp_xxx.txt for metrics testing
        with open(f"{work_dir}/metrics/tmp_{dataset}_{time.time()}.txt", "w") as f:
            print(f"save tmp file at {work_dir}/metrics/tmp_{dataset}_{time.time()}.txt")
            f.write(os.path.basename(save_root))

    if args.debug:
        items = [
            "A fashionable woman in a black leather jacket, long red dress, and black boots confidently strolls down a wet, reflective Tokyo street. Neon lights and animated signs glow warmly around her. She carries a black purse and wears red lipstick and sunglasses. Pedestrians move about in the bustling background. Medium shot, dynamic camera movement.",
            "Several giant woolly mammoths lumber through a snowy meadow, their long, fluffy fur gently swaying in the breeze. Snow-covered trees and snow-capped mountains loom in the background under mid-afternoon sunlight with wispy clouds, casting a warm glow. A low-angle camera captures the majestic creatures in stunning detail with a soft depth of field.",
            "A cinematic movie trailer in vibrant 35mm film style, featuring a 30-year-old space adventurer wearing a red wool knitted motorcycle helmet, exploring a vast blue-sky salt desert. Wide shots and dynamic camera movements.",
            "Drone view of waves crashing against rugged cliffs at Big Sur's Garrapata Point. Blue waves form white tips as the setting sun casts golden light on the rocky shore. A distant island with a lighthouse and green shrubs on the cliff edge add to the scene. Dramatic steep drops from the coastal road to the beach highlight the raw beauty of the Pacific Coast Highway. Medium shot, sweeping drone movement.",
            "Close-up of a cute, fluffy monster kneeling beside a melting red candle in a realistic 3D style. The monster gazes curiously at the flickering flame with wide eyes and an open mouth, expressing innocence and playfulness. Warm colors and dramatic lighting create a cozy, wondrous atmosphere.",
            "the opening scene begins with a dynamic view of a bustling cityscape captured in vibrant detail. towering skyscrapers dominate the skyline, while the streets below are alive with motion. people from diverse cultures fill the sidewalks, engaging in daily activities, their vibrant attire adding splashes of color to the scene. vehicles, including cars and buses, weave through the busy roads in a synchronized rhythm. bright billboards in various languages flash advertisements, reflecting the multicultural essence of the city. thecamera smoothly pans upward from the busy streets to focus on a sleek, modern office building. its reflective glass facade shimmers in the sunlight, hinting at its importance as a central location in the story. the atmosphere is energetic and cosmopolitan, setting the stage for an international narrative.",
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

    print(
        colored(f"Sana inference has finished. Results stored at ", "green"),
        colored(f"{img_save_dir}", attrs=["bold"]),
        ".",
    )
