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

import datetime
import gc
import getpass
import hashlib
import json
import os
import os.path as osp
import random
import time
import warnings
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path

warnings.filterwarnings("ignore")  # ignore warning

import imageio
import numpy as np
import pyrallis
import torch
from accelerate import Accelerator, InitProcessGroupKwargs, skip_first_batches
from PIL import Image
from termcolor import colored

from diffusion import DPMS, ChunkFlowEuler, LTXFlowEuler, Scheduler
from diffusion.data.builder import build_dataloader, build_dataset
from diffusion.data.transforms import read_image_from_path
from diffusion.data.wids import DistributedRangedSampler
from diffusion.model.builder import (
    build_model,
    encode_image,
    get_image_encoder,
    get_tokenizer_and_text_encoder,
    get_vae,
    vae_decode,
    vae_encode,
)
from diffusion.model.respace import IncrementalTimesteps, process_timesteps
from diffusion.model.utils import get_weight_dtype
from diffusion.utils.checkpoint import load_checkpoint, save_checkpoint
from diffusion.utils.config import SanaVideoConfig, model_video_init_config
from diffusion.utils.data_sampler import AspectRatioBatchSampler, AspectRatioBatchSamplerVideo
from diffusion.utils.dist_utils import flush, get_world_size
from diffusion.utils.git import save_git_snapshot
from diffusion.utils.logger import LogBuffer, get_root_logger
from diffusion.utils.lr_scheduler import build_lr_scheduler
from diffusion.utils.misc import DebugUnderflowOverflow, init_random_seed, set_random_seed
from diffusion.utils.optimizer import auto_scale_lr, build_optimizer

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def set_fsdp_env():
    # Basic FSDP settings
    os.environ["ACCELERATE_USE_FSDP"] = "true"

    # Auto wrapping policy
    os.environ["FSDP_AUTO_WRAP_POLICY"] = "TRANSFORMER_BASED_WRAP"
    os.environ["FSDP_TRANSFORMER_CLS_TO_WRAP"] = "SanaVideoMSBlock"  # Your transformer block name

    # Performance optimization settings
    os.environ["FSDP_BACKWARD_PREFETCH"] = "BACKWARD_PRE"
    os.environ["FSDP_FORWARD_PREFETCH"] = "false"

    # State dict settings
    os.environ["FSDP_STATE_DICT_TYPE"] = "FULL_STATE_DICT"
    os.environ["FSDP_SYNC_MODULE_STATES"] = "true"
    os.environ["FSDP_USE_ORIG_PARAMS"] = "true"

    # Sharding strategy
    os.environ["FSDP_SHARDING_STRATEGY"] = "HYBRID_SHARD"  # FULL_SHARD

    # Memory optimization settings (optional)
    os.environ["FSDP_CPU_RAM_EFFICIENT_LOADING"] = "false"
    os.environ["FSDP_OFFLOAD_PARAMS"] = "false"

    # Precision settings
    os.environ["FSDP_REDUCE_SCATTER_PRECISION"] = "fp32"
    os.environ["FSDP_ALL_GATHER_PRECISION"] = "fp32"
    os.environ["FSDP_OPTIMIZER_STATE_PRECISION"] = "fp32"


def ema_update(model_dest, model_src, rate):
    param_dict_src = dict(model_src.named_parameters())
    for p_name, p_dest in model_dest.named_parameters():
        p_src = param_dict_src[p_name]
        assert p_src is not p_dest
        p_dest.data.mul_(rate).add_((1 - rate) * p_src.data)


@torch.inference_mode()
def log_validation(accelerator, config, model, logger, step, device, vae=None, init_noise=None):
    torch.cuda.empty_cache()
    vis_sampler = config.scheduler.vis_sampler
    model = accelerator.unwrap_model(model).eval()
    hw = torch.tensor([[video_height, video_width]], dtype=torch.float, device=device).repeat(1, 1)
    ar = torch.tensor([[1.0]], device=device).repeat(1, 1)
    null_y = torch.load(null_embed_path, map_location="cpu")
    null_y = null_y["uncond_prompt_embeds"].to(device)
    cfg_scale = 4.5

    # Create sampling noise:
    logger.info("Running validation... ")
    video_logs = []

    def run_sampling(init_z=None, label_suffix="", vae=None, sampler="dpm-solver"):
        latents = []
        current_video_logs = []
        for prompt in validation_prompts:
            z = (
                torch.randn(1, config.vae.vae_latent_dim, latent_temp, latent_height, latent_width, device=device)
                if init_z is None
                else init_z
            )
            logger.info(f"Loading embedding for prompt from: {config.train.valid_prompt_embed_root}")
            embed = torch.load(
                osp.join(config.train.valid_prompt_embed_root, f"{prompt[:50]}_{valid_prompt_embed_suffix}"),
                map_location="cpu",
            )
            caption_embs, emb_masks = embed["caption_embeds"].to(device), embed["emb_mask"].to(device)
            model_kwargs = dict(
                data_info={"img_hw": hw, "aspect_ratio": ar},
                mask=emb_masks,
                chunk_index=config.model.get("chunk_index", None),
            )

            if config.task == "ti2v":
                image_vae_embeds = embed["image_vae_embeds"].to(device)  # 1,C,1,H,W
                if config.model.image_latent_mode == "repeat":
                    image_vae_embeds = image_vae_embeds.repeat(1, 1, latent_temp, 1, 1)  # B,C,F,H,W
                    z = (
                        0.99 * z + 0.01 * image_vae_embeds
                    )  # inversion trick for better consistency, this should not be used for video zero, since the later frames of video zero are grey
                if cfg_scale > 1.0:
                    image_vae_embeds = torch.cat([image_vae_embeds, image_vae_embeds], dim=0)  # 2,C,1,H,W

                model_kwargs["data_info"].update({"image_vae_embeds": image_vae_embeds})  # B,C,F,H,W

                if config.get("image_encoder", {}).get("image_encoder_name") == "flux-siglip":
                    image_embeds = embed["image_embeds"].to(device)
                    if cfg_scale > 1.0:
                        image_embeds = torch.cat([image_embeds, image_embeds], dim=0)  # 2,C,1,H,W
                    model_kwargs["data_info"].update({"image_embeds": image_embeds})  # B,C,F,H,W
            elif config.task == "df" or config.task == "ltx":
                # NOTE during inference, we do not use noise for the first frame, hard-coded here
                condition_frame_info = {
                    0: 0.0,  # frame_idx: frame_weight, weight is used for timestep
                }
                image_vae_embeds = embed["image_vae_embeds"].to(device)  # 1,C,1,H,W
                model_kwargs["data_info"].update({"condition_frame_info": condition_frame_info})  # B,C,F,H,W
                for frame_idx in list(condition_frame_info.keys()):
                    z[:, :, frame_idx : frame_idx + 1] = image_vae_embeds  # 1,C,F,H,W, first frame is the image

            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                flow_shift = (
                    config.scheduler.inference_flow_shift
                    if config.scheduler.inference_flow_shift is not None
                    else config.scheduler.flow_shift
                )
                if sampler == "chunk_flow_euler":
                    chunk_index = model_kwargs.get("chunk_index", None)
                    logger.info(f"chunk_index: {chunk_index}, interval_k: {1/len(chunk_index):.2f}")
                    assert chunk_index is not None

                    flow_solver = ChunkFlowEuler(
                        model,
                        condition=caption_embs,
                        uncondition=null_y,
                        cfg_scale=cfg_scale,
                        flow_shift=flow_shift,
                        model_kwargs=model_kwargs,
                    )
                    denoised = flow_solver.sample(
                        z,
                        steps=50,
                        generator=None,
                        chunk_index=chunk_index,
                        interval_k=1 / len(chunk_index),
                    )
                elif sampler == "flow_dpm-solver":
                    dpm_solver = DPMS(
                        model.forward_with_dpmsolver,
                        condition=caption_embs,
                        uncondition=null_y,
                        cfg_scale=cfg_scale,
                        model_type="flow",
                        model_kwargs=model_kwargs,
                        schedule="FLOW",
                    )
                    denoised = dpm_solver.sample(
                        z,
                        steps=50,
                        order=2,
                        skip_type="time_uniform_flow",
                        method="multistep",
                        flow_shift=flow_shift,
                    )
                elif sampler == "flow_euler_ltx":
                    ltx_flow_euler = LTXFlowEuler(
                        model,
                        condition=caption_embs,
                        uncondition=null_y,
                        cfg_scale=cfg_scale,
                        flow_shift=flow_shift,
                        model_kwargs=model_kwargs,
                    )
                    denoised = ltx_flow_euler.sample(
                        z,
                        steps=50,
                        generator=None,
                    )
                else:
                    raise ValueError(f"{sampler} not implemented")

            latents.append(denoised)
        torch.cuda.empty_cache()
        if vae is None:
            vae = get_vae(
                config.vae.vae_type, config.vae.vae_pretrained, accelerator.device, dtype=vae_dtype, config=config.vae
            )
        for prompt, latent in zip(validation_prompts, latents):
            latent = latent.to(vae_dtype)
            samples = vae_decode(config.vae.vae_type, vae, latent)
            video = (
                torch.clamp(127.5 * samples[0] + 127.5, 0, 255).permute(1, 0, 2, 3).to("cpu", dtype=torch.uint8).numpy()
            )  # C,T,H,W -> T,C,H,W
            current_video_logs.append({"validation_prompt": prompt + label_suffix, "videos": video})

        return current_video_logs

    # First run with original noise
    video_logs += run_sampling(init_z=None, label_suffix="", vae=vae, sampler=vis_sampler)

    # Second run with init_noise if provided
    if init_noise is not None:
        torch.cuda.empty_cache()
        gc.collect()
        init_noise = torch.clone(init_noise).to(device)
        video_logs += run_sampling(init_z=init_noise, label_suffix=" w/ init noise", vae=vae, sampler=vis_sampler)

    for tracker in accelerator.trackers:
        if tracker.name == "wandb":
            import wandb

            wandb_items = []
            for log_item in video_logs:
                wandb_items.append(
                    wandb.Video(log_item["videos"], caption=log_item["validation_prompt"], fps=16, format="mp4")
                )
            tracker.log({"validation": wandb_items})
        else:
            logger.warn(f"Video logging not implemented for {tracker.name}")

    def concatenate_videos(video_data, videos_per_row=3, video_format="mp4"):
        videos = [torch.from_numpy(log["videos"]).to(torch.uint8) for log in video_data]  # T,C,H,W

        num_videos = len(videos)
        num_rows = (num_videos + videos_per_row - 1) // videos_per_row
        num_frames, num_channels, height, width = videos[0].shape
        total_width = width * min(videos_per_row, num_videos)
        total_height = height * num_rows

        grid_video = torch.zeros((num_frames, num_channels, total_height, total_width), dtype=videos[0].dtype)

        for i, video in enumerate(videos):

            row = i // videos_per_row
            col = i % videos_per_row

            y_offset = row * height
            x_offset = col * width

            h, w = video.shape[2:]

            grid_video[:, :, y_offset : y_offset + h, x_offset : x_offset + w] = video

        return grid_video

    if config.train.local_save_vis:
        file_format = "mp4"
        local_vis_save_path = osp.join(config.work_dir, "log_vis")
        os.umask(0o000)
        os.makedirs(local_vis_save_path, exist_ok=True)
        concatenated_video = concatenate_videos(video_logs, videos_per_row=5, video_format=file_format)
        save_path = (
            osp.join(local_vis_save_path, f"vis_{step}.{file_format}")
            if init_noise is None
            else osp.join(local_vis_save_path, f"vis_{step}_w_init.{file_format}")
        )
        save_video = concatenated_video.permute(0, 2, 3, 1)
        writer = imageio.v2.get_writer(save_path, fps=16, format="FFMPEG", codec="libx264", quality=8)
        for frame in save_video.numpy():
            writer.append_data(frame)
        writer.close()

    model.train()
    del vae
    flush()
    return video_logs


def train(
    config,
    args,
    accelerator,
    model,
    model_ema,
    optimizer,
    lr_scheduler,
    train_dataloader,
    train_dataloader_image,
    train_diffusion,
    logger,
):
    if getattr(config.train, "debug_nan", False):
        DebugUnderflowOverflow(model, max_frames_to_save=100)
        logger.info("NaN debugger registered. Start to detect overflow during training.")
    log_buffer = LogBuffer()

    global_step = start_step
    video_step = start_video_step  # Track video steps separately
    image_step = start_image_step  # Track image steps separately

    skip_step = max(config.train.skip_step, video_step) % train_dataloader_len
    skip_step = skip_step if skip_step < (train_dataloader_len - 20) else 0
    skip_step_image = max(config.train.skip_step, image_step) % train_dataloader_image_len
    skip_step_image = skip_step_image if skip_step_image < (train_dataloader_image_len - 20) else 0
    loss_nan_timer = 0
    model_instance.to(accelerator.device)

    time_sampler = IncrementalTimesteps(
        F=config.model.get("chunk_index", None),
        T=config.scheduler.train_sampling_steps,
        device=accelerator.device,
        dtype=torch.float64,
    )
    # Now you train the model
    for epoch in range(start_epoch + 1, config.train.num_epochs + 1):
        time_start, last_tic = time.time(), time.time()
        sampler = (
            train_dataloader.batch_sampler.sampler
            if (num_replicas > 1 or config.model.multi_scale)
            else train_dataloader.sampler
        )
        image_sampler = (
            train_dataloader_image.batch_sampler.sampler
            if (num_replicas > 1 or config.model.multi_scale)
            else train_dataloader_image.sampler
        )
        if train_dataloader.dataset.shuffle_dataset:
            logger.info(f"Shuffled dataset, no skip step")
        else:
            set_start_value = max((skip_step - 1) * config.train.train_batch_size, 0)
            os.environ[f"CURRENT_VIDEO_STEP_START_RANK_{rank}"] = str(set_start_value)
            sampler.set_epoch(epoch)
            sampler.set_start(set_start_value)

            set_image_start_value = max((skip_step_image - 1) * config.train.train_batch_size_image, 0)
            os.environ[f"CURRENT_IMAGE_STEP_START_RANK_{rank}"] = str(set_image_start_value)
            image_sampler.set_epoch(epoch)
            image_sampler.set_start(set_image_start_value)

            if skip_step > 1 and accelerator.is_main_process:
                logger.info(f"Skipped video training Steps: {skip_step}")
                logger.info(f"Skipped image training Steps: {skip_step_image}")
        skip_step = 1
        data_time_start = time.time()
        data_time_all = 0
        lm_time_all = 0
        vae_time_all = 0
        model_time_all = 0

        # Create dataloader iterators for joint training
        video_dataloader_iter = iter(train_dataloader)
        image_dataloader_iter = iter(train_dataloader_image)

        # Use range instead of enumerating train_dataloader
        for step in range(train_dataloader_len):

            # Determine if this is an image training step
            is_image_step = (
                joint_training_interval > 0 and (global_step % joint_training_interval == 0) and (global_step > 0)
            )

            if is_image_step:
                # Get image batch for joint training
                try:
                    batch = next(image_dataloader_iter)
                except StopIteration:
                    # Reset image dataloader iterator if exhausted
                    image_dataloader_iter = iter(train_dataloader_image)
                    batch = next(image_dataloader_iter)
                is_video_data = False
                image_step += 1  # Increment image step counter
            else:
                # Get video batch
                try:
                    batch = next(video_dataloader_iter)
                except StopIteration:
                    # Reset video dataloader iterator if exhausted
                    logger.info(f"Reset video dataloader iterator")
                    sampler.set_start(0)
                    video_dataloader_iter = iter(train_dataloader)
                    batch = next(video_dataloader_iter)
                is_video_data = True
                video_step += 1  # Increment video step counter

            # image, json_info, key = batch
            accelerator.wait_for_everyone()
            data_time_all += time.time() - data_time_start
            vae_time_start = time.time()
            data_info = batch[3]
            with torch.no_grad():
                if is_video_data:
                    if load_vae_feat:  # feat is only stored for video data
                        z = batch[0].to(accelerator.device)
                    else:
                        # Video data processing (original code)
                        z = vae_encode(
                            config.vae.vae_type,
                            vae,
                            batch[0].permute(0, 2, 1, 3, 4).to(vae_dtype),
                            device=accelerator.device,
                            cache_key=data_info["cache_key"],
                            if_cache=config.vae.if_cache,
                            data_info=data_info,
                        )  # B,F,C,H,W -> B,C,F,H,W

                        if config.task == "ti2v":
                            if config.model.image_latent_mode == "repeat":
                                image_vae_embeds = vae_encode(
                                    config.vae.vae_type,
                                    vae,
                                    batch[0][:, :1].permute(0, 2, 1, 3, 4).to(vae_dtype),
                                    device=accelerator.device,
                                )  # B,1,C,H,W -> B,C,1,H,W -> B,C,1,H,W
                                image_vae_embeds = image_vae_embeds.repeat(1, 1, z.shape[2], 1, 1)
                            elif config.model.image_latent_mode == "video_zero":
                                pad_video = torch.zeros_like(batch[0])
                                pad_video[:, :1] = batch[0][:, :1]
                                image_vae_embeds = vae_encode(
                                    config.vae.vae_type,
                                    vae,
                                    pad_video.permute(0, 2, 1, 3, 4).to(vae_dtype),
                                    device=accelerator.device,
                                )

                            rand_null_image_vae_embeds = (
                                torch.rand(image_vae_embeds.size(0)) < config.model.class_dropout_prob
                            )
                            image_vae_embeds[rand_null_image_vae_embeds] = 0
                            data_info["image_vae_embeds"] = image_vae_embeds
                            # TODO: We may add some noise here.

                            if config.get("image_encoder", {}).get("image_encoder_name") == "flux-siglip":
                                image_embeds = encode_image(
                                    name=config.image_encoder.image_encoder_name,
                                    image_encoder=image_encoder,
                                    image_processor=image_processor,
                                    images=batch[0][:, 0],
                                    device=accelerator.device,
                                    dtype=image_encoder.dtype,
                                )
                                data_info["image_embeds"] = image_embeds

                else:
                    # Image data processing (similar to stage1)
                    if batch[0].dim() == 4:
                        batch[0] = batch[0][:, :, None]  # B,C,H,W -> B,C,1,H,W
                    z = vae_encode(
                        config.vae.vae_type,
                        vae,
                        batch[0].to(vae_dtype),
                        device=accelerator.device,
                    )
                if args.debug and step % 10 == 0 and accelerator.is_main_process:
                    # decode video to check
                    samples = vae_decode(config.vae.vae_type, vae, z)  # B,C,T,H,W
                    if not is_video_data:
                        samples = samples[0][:, :1]
                        original_video = batch[0][0]  # C,1,H,W
                    else:
                        samples = samples[0]
                        if load_vae_feat:
                            original_video = batch[0][0]
                            original_video = vae_decode(config.vae.vae_type, vae, original_video.cuda())[0]
                        else:
                            original_video = batch[0][0].permute(1, 0, 2, 3)  # F,C,H,W -> C,F,H,W

                    recon_video = (
                        torch.clamp(127.5 * samples + 127.5, 0, 255)
                        .permute(1, 2, 3, 0)
                        .to("cpu", dtype=torch.uint8)
                        .numpy()
                    )  # C,T,H,W -> T,H,W,C
                    # save video
                    original_video = (
                        torch.clamp(127.5 * original_video + 127.5, 0, 255)
                        .permute(1, 2, 3, 0)
                        .to("cpu", dtype=torch.uint8)
                        .numpy()
                    )  #  C,F,H,W -> F,H,W,C
                    save_video = np.concatenate([original_video, recon_video], axis=1)  # F,H,W,C -> F,2H,W,C
                    save_path = osp.join(config.work_dir, "log_vis", f"recon_video_{global_step}.mp4")
                    os.makedirs(osp.dirname(save_path), exist_ok=True)
                    if is_video_data:
                        writer = imageio.v2.get_writer(save_path, fps=16, format="FFMPEG", codec="libx264", quality=8)
                        for frame in save_video:
                            writer.append_data(frame)
                        writer.close()
                    else:
                        imageio.imwrite(save_path.replace(".mp4", ".png"), save_video[0])

            accelerator.wait_for_everyone()
            vae_time_all += time.time() - vae_time_start

            clean_images = z

            lm_time_start = time.time()
            do_i2v = is_video_data and random.random() < config.train.ltx_image_condition_prob
            loss_mask = None
            if do_i2v:
                loss_mask = torch.ones_like(clean_images)  # B,C,F,H,W
                loss_mask[:, :, 0] = 0  # first frame has no loss

            if load_text_feat:
                y = batch[1]  # bs, 1, N, C
                y_mask = batch[2]  # bs, 1, 1, N
            else:
                if "T5" in config.text_encoder.text_encoder_name:
                    with torch.no_grad():
                        txt_tokens = tokenizer(
                            batch[1], max_length=max_length, padding="max_length", truncation=True, return_tensors="pt"
                        ).to(accelerator.device)
                        y = text_encoder(txt_tokens.input_ids, attention_mask=txt_tokens.attention_mask)[0][:, None]
                        y_mask = txt_tokens.attention_mask[:, None, None]
                elif "gemma" in config.text_encoder.text_encoder_name:
                    with torch.no_grad():
                        if not config.text_encoder.chi_prompt:
                            max_length_all = config.text_encoder.model_max_length
                            prompt = batch[1]
                        else:
                            chi_prompt = "\n".join(config.text_encoder.chi_prompt)
                            prompt = [chi_prompt + i for i in batch[1]]
                            num_sys_prompt_tokens = len(tokenizer.encode(chi_prompt))
                            max_length_all = (
                                num_sys_prompt_tokens + config.text_encoder.model_max_length - 2
                            )  # magic number 2: [bos], [_]
                        txt_tokens = tokenizer(
                            prompt,
                            padding="max_length",
                            max_length=max_length_all,
                            truncation=True,
                            return_tensors="pt",
                        ).to(accelerator.device)
                        select_index = [0] + list(
                            range(-config.text_encoder.model_max_length + 1, 0)
                        )  # first bos and end N-1
                        y = text_encoder(txt_tokens.input_ids, attention_mask=txt_tokens.attention_mask)[0][:, None][
                            :, :, select_index
                        ]
                        y_mask = txt_tokens.attention_mask[:, None, None][:, :, :, select_index]
                elif "Qwen" in config.text_encoder.text_encoder_name:
                    with torch.no_grad():
                        if do_i2v and config.model.encode_image_prompt_embeds:
                            if load_vae_feat:
                                first_frame = batch[8][:, 0]
                            else:
                                first_frame = batch[0][:, 0]
                            y, y_mask = text_handler.get_image_prompt_embeds(
                                batch[1], first_frame, max_length=max_length
                            )
                        else:
                            y, y_mask = text_handler.get_prompt_embeds(batch[1], max_length=max_length)
                        y = y[:, None]
                        y_mask = y_mask[:, None, None]
                else:
                    print("error")
                    exit()

            # Sample a random timestep for each image
            bs = clean_images.shape[0]  # clean_images: B,C,F,H,W
            do_i2v = is_video_data and random.random() < config.train.ltx_image_condition_prob
            loss_mask = None
            if do_i2v:
                loss_mask = torch.ones_like(clean_images)  # B,C,F,H,W
                loss_mask[:, :, 0] = 0  # first frame has no loss

                if config.model.encode_image_prompt_embeds:
                    if load_vae_feat:
                        first_frame = batch[8][:, 0]
                    else:
                        first_frame = batch[0][:, 0]
                    y, y_mask = text_handler.get_image_prompt_embeds(
                        batch[1], first_frame, max_length=max_length
                    )  # +500 for image prompt
                    y = y[:, None]
                    y_mask = y_mask[:, None, None]

            chunk_index = config.model.get("chunk_index", None) if is_video_data else None

            timesteps = process_timesteps(
                weighting_scheme=config.scheduler.weighting_scheme,
                train_sampling_steps=config.scheduler.train_sampling_steps,
                size=(bs,),  # B,1,F
                device=clean_images.device,
                logit_mean=config.scheduler.logit_mean,
                logit_std=config.scheduler.logit_std,
                num_frames=clean_images.shape[2],
                do_i2v=do_i2v,
                noise_multiplier=config.train.noise_multiplier,  # add small random noise to the first frame
                chunk_index=chunk_index,
                chunk_sampling_strategy=config.train.get("chunk_sampling_strategy", "uniform"),
                same_timestep_prob=config.train.get("same_timestep_prob", 0.0),
                time_sampler=time_sampler,
                p_low=config.scheduler.p_low,
                p_high=config.scheduler.p_high,
            )

            grad_norm = None
            accelerator.wait_for_everyone()
            lm_time_all += time.time() - lm_time_start
            model_time_start = time.time()

            with accelerator.accumulate(model):
                # Predict the noise residual
                optimizer.zero_grad()
                loss_term = train_diffusion.training_losses(
                    model,
                    clean_images,
                    timesteps,
                    model_kwargs=dict(y=y, mask=y_mask, data_info=data_info, chunk_index=chunk_index),
                    timestep_weight=config.train.timestep_weight,
                    loss_mask=loss_mask,
                )
                loss = loss_term["loss"].mean()
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    grad_norm = accelerator.clip_grad_norm_(model.parameters(), config.train.gradient_clip)
                    if not config.train.use_fsdp and config.train.ema_update and model_ema is not None:
                        ema_update(model_ema, model, config.train.ema_rate)

                optimizer.step()
                lr_scheduler.step()
                accelerator.wait_for_everyone()
                model_time_all += time.time() - model_time_start

            if torch.any(torch.isnan(loss)):
                loss_nan_timer += 1
            lr = lr_scheduler.get_last_lr()[0]
            logs = {args.loss_report_name: accelerator.gather(loss).mean().item()}
            if grad_norm is not None:
                logs.update(grad_norm=accelerator.gather(grad_norm).mean().item())
            log_buffer.update(logs)
            if (global_step + 1) % config.train.log_interval == 0 or (step + 1) == 1:
                accelerator.wait_for_everyone()
                if args.debug:
                    print(f"Rank {rank}: current_batch_id: {batch[4]}")

                t = (time.time() - last_tic) / config.train.log_interval
                t_d = data_time_all / config.train.log_interval
                t_m = model_time_all / config.train.log_interval
                t_lm = lm_time_all / config.train.log_interval
                t_vae = vae_time_all / config.train.log_interval
                avg_time = (time.time() - time_start) / (step + 1)
                eta = str(datetime.timedelta(seconds=int(avg_time * (total_steps - global_step - 1))))
                eta_epoch = str(
                    datetime.timedelta(
                        seconds=int(
                            avg_time
                            * (train_dataloader_len - sampler.step_start // config.train.train_batch_size - step - 1)
                        )
                    )
                )
                log_buffer.average()

                current_step = (
                    global_step
                    - sampler.step_start // config.train.train_batch_size
                    - image_sampler.step_start // config.train.train_batch_size_image
                ) % train_dataloader_len
                current_step = train_dataloader_len if current_step == 0 else current_step

                data_type = "Image" if not is_video_data else "Video"
                id_info = (
                    f"{batch[4][-1]}:{'/'.join(data_info['zip_file'][-1].split('/')[-2:])}"
                    if "zip_file" in data_info
                    else f"{batch[4][-1]}"
                )
                info = (
                    f"Epoch: {epoch} | Global Step: {global_step + 1} / {train_dataloader_len}, "
                    f"Video Step: {video_step} | Image Step: {image_step} | id: {id_info}, "
                    f"total_eta: {eta}, epoch_eta:{eta_epoch}, time: all:{t:.3f}, model:{t_m:.3f}, data:{t_d:.3f}, "
                    f"lm:{t_lm:.3f}, vae:{t_vae:.3f}, lr:{lr:.3e}, DataType: {data_type}, Cap: {batch[5][0]}, "
                )
                info += (
                    f"s:({model.module.f}, {model.module.h}, {model.module.w}), "
                    if hasattr(model, "module")
                    else f"s:({model.f}, {model.h}, {model.w}), "
                )

                info += ", ".join([f"{k}:{v:.4f}" for k, v in log_buffer.output.items()])
                last_tic = time.time()
                log_buffer.clear()
                data_time_all = 0
                model_time_all = 0
                lm_time_all = 0
                vae_time_all = 0
                if accelerator.is_main_process:
                    logger.info(info)

            logs.update(lr=lr)
            if accelerator.is_main_process:
                accelerator.log(logs, step=global_step)

            global_step += 1

            if loss_nan_timer > 20:
                raise ValueError("Loss is NaN too much times. Break here.")
            if (
                global_step % config.train.save_model_steps == 0
                or (time.time() - training_start_time) / 3600 > config.train.early_stop_hours
            ):
                torch.cuda.synchronize()
                accelerator.wait_for_everyone()

                # Choose different saving methods based on whether FSDP is used
                if config.train.use_fsdp:
                    # FSDP mode
                    os.umask(0o000)
                    saved_info = {
                        "video_step": video_step,
                        "image_step": image_step,
                    }
                    ckpt_saved_path = save_checkpoint(
                        work_dir=osp.join(config.work_dir, "checkpoints"),
                        epoch=epoch,
                        model=model,
                        accelerator=accelerator,
                        optimizer=optimizer,
                        lr_scheduler=lr_scheduler,
                        step=global_step,
                        saved_info=saved_info,
                        add_symlink=True,
                    )
                else:
                    # DDP mode
                    if accelerator.is_main_process:
                        os.umask(0o000)
                        saved_info = {
                            "video_step": video_step,
                            "image_step": image_step,
                        }
                        ckpt_saved_path = save_checkpoint(
                            work_dir=osp.join(config.work_dir, "checkpoints"),
                            epoch=epoch,
                            model=accelerator.unwrap_model(model),
                            model_ema=accelerator.unwrap_model(model_ema) if model_ema is not None else None,
                            optimizer=optimizer,
                            lr_scheduler=lr_scheduler,
                            step=global_step,
                            saved_info=saved_info,
                            generator=generator,
                            add_symlink=True,
                        )

                if accelerator.is_main_process:
                    if config.train.online_metric and global_step % config.train.eval_metric_step == 0 and step > 1:
                        online_metric_monitor_dir = osp.join(config.work_dir, config.train.online_metric_dir)
                        os.makedirs(online_metric_monitor_dir, exist_ok=True)
                        with open(f"{online_metric_monitor_dir}/{ckpt_saved_path.split('/')[-1]}.txt", "w") as f:
                            f.write(osp.join(config.work_dir, "config.py") + "\n")
                            f.write(ckpt_saved_path)

                if (time.time() - training_start_time) / 3600 > config.train.early_stop_hours:
                    logger.info(f"Stopping training at epoch {epoch}, step {global_step} due to time limit.")
                    return

            if config.train.visualize and (global_step % config.train.eval_sampling_steps == 0 or (step + 1) == 1):
                if config.train.use_fsdp:
                    merged_state_dict = accelerator.get_state_dict(model)

                accelerator.wait_for_everyone()
                if accelerator.is_main_process:
                    if config.train.use_fsdp:
                        model_instance.load_state_dict(merged_state_dict)

                    if validation_noise is not None:
                        log_validation(
                            accelerator=accelerator,
                            config=config,
                            model=model_instance,
                            logger=logger,
                            step=global_step,
                            device=accelerator.device,
                            vae=vae,
                            init_noise=validation_noise,
                        )
                    else:
                        log_validation(
                            accelerator=accelerator,
                            config=config,
                            model=model_instance,
                            logger=logger,
                            step=global_step,
                            device=accelerator.device,
                            vae=vae,
                        )

            # avoid dead-lock of multiscale data batch sampler
            if (
                config.model.multi_scale
                and (train_dataloader_len - sampler.step_start // config.train.train_batch_size - step) < 30
            ):
                global_step = (
                    (global_step + train_dataloader_len - 1) // train_dataloader_len
                ) * train_dataloader_len + 1
                logger.info("Early stop current iteration")
                skip_first_batches(train_dataloader, True)
                break

            data_time_start = time.time()

        if epoch % config.train.save_model_epochs == 0 or epoch == config.train.num_epochs and not config.debug:
            accelerator.wait_for_everyone()
            torch.cuda.synchronize()

            # Choose different saving methods based on whether FSDP is used
            if config.train.use_fsdp:
                # FSDP mode
                os.umask(0o000)
                saved_info = {
                    "video_step": video_step,
                    "image_step": image_step,
                }
                ckpt_saved_path = save_checkpoint(
                    work_dir=osp.join(config.work_dir, "checkpoints"),
                    epoch=epoch,
                    model=model,
                    accelerator=accelerator,
                    optimizer=optimizer,
                    lr_scheduler=lr_scheduler,
                    step=global_step,
                    saved_info=saved_info,
                    add_symlink=True,
                )
            else:
                # DDP mode
                if accelerator.is_main_process:
                    os.umask(0o000)
                    saved_info = {
                        "video_step": video_step,
                        "image_step": image_step,
                    }
                    ckpt_saved_path = save_checkpoint(
                        osp.join(config.work_dir, "checkpoints"),
                        epoch=epoch,
                        step=global_step,
                        saved_info=saved_info,
                        model=accelerator.unwrap_model(model),
                        model_ema=accelerator.unwrap_model(model_ema) if model_ema is not None else None,
                        optimizer=optimizer,
                        lr_scheduler=lr_scheduler,
                        generator=generator,
                        add_symlink=True,
                    )

            if accelerator.is_main_process:
                online_metric_monitor_dir = osp.join(config.work_dir, config.train.online_metric_dir)
                os.makedirs(online_metric_monitor_dir, exist_ok=True)
                with open(f"{online_metric_monitor_dir}/{ckpt_saved_path.split('/')[-1]}.txt", "w") as f:
                    f.write(osp.join(config.work_dir, "config.py") + "\n")
                    f.write(ckpt_saved_path)


@pyrallis.wrap()
def main(cfg: SanaVideoConfig) -> None:
    global train_dataloader_len, start_epoch, start_step, start_video_step, start_image_step, vae, generator, num_replicas, rank, training_start_time
    global load_vae_feat, load_text_feat, validation_noise, text_encoder, tokenizer, text_handler
    global max_length, validation_prompts, latent_size, valid_prompt_embed_suffix, null_embed_path
    global image_size, cache_file, total_steps, vae_dtype, model_instance
    global video_width, video_height, num_frames, latent_temp, latent_height, latent_width
    global image_encoder, image_processor, joint_training_interval, train_dataloader_image_len

    config = cfg
    args = cfg

    # Record app start time at the beginning of main
    app_start_time = int(time.time() * 1000)

    # 1.Initialize training mode
    if config.train.use_fsdp:
        set_fsdp_env()
        init_train = "FSDP"
    else:
        init_train = "DDP"

    training_start_time = time.time()
    load_from = True

    if args.resume_from or config.model.resume_from:
        load_from = False
        config.model.resume_from = dict(
            checkpoint=args.resume_from or config.model.resume_from,
            load_ema=False,
            resume_optimizer=True,
            resume_lr_scheduler=config.train.resume_lr_scheduler,
        )

    if args.debug:
        config.train.train_batch_size = min(64, config.train.train_batch_size)
        if config.train.use_fsdp:
            os.environ["FSDP_SHARDING_STRATEGY"] = "FULL_SHARD"
        config.data.data_dir = {"video_toy_data": "data/video_toy_data"}
        config.train.validation_prompts = [
            "the opening scene begins with a dynamic view of a bustling cityscape captured in vibrant detail. towering skyscrapers dominate the skyline, while the streets below are alive with motion. people from diverse cultures fill the sidewalks, engaging in daily activities, their vibrant attire adding splashes of color to the scene. vehicles, including cars and buses, weave through the busy roads in a synchronized rhythm. bright billboards in various languages flash advertisements, reflecting the multicultural essence of the city. thecamera smoothly pans upward from the busy streets to focus on a sleek, modern office building. its reflective glass facade shimmers in the sunlight, hinting at its importance as a central location in the story. the atmosphere is energetic and cosmopolitan, setting the stage for an international narrative."
        ]

    os.umask(0o000)
    os.makedirs(config.work_dir, exist_ok=True)

    init_handler = InitProcessGroupKwargs()
    init_handler.timeout = datetime.timedelta(seconds=5400)  # change timeout to avoid a strange NCCL bug

    # Initialize accelerator and tensorboard logging
    accelerator = Accelerator(
        mixed_precision=config.model.mixed_precision,
        gradient_accumulation_steps=config.train.gradient_accumulation_steps,
        log_with=args.report_to,
        project_dir=osp.join(config.work_dir, "logs"),
        kwargs_handlers=[init_handler],
    )

    log_name = "train_log.log"
    logger = get_root_logger(osp.join(config.work_dir, log_name))
    logger.info(accelerator.state)

    # save git snapshot
    if not args.debug and accelerator.is_main_process:
        job_name = osp.basename(config.work_dir)
        save_git_snapshot(config.work_dir, job_name, logger)

    config.train.seed = init_random_seed(getattr(config.train, "seed", None))
    set_random_seed(config.train.seed + int(os.environ["LOCAL_RANK"]))
    generator = torch.Generator(device="cpu").manual_seed(config.train.seed)

    if accelerator.is_main_process:
        pyrallis.dump(config, open(osp.join(config.work_dir, "config.yaml"), "w"), sort_keys=False, indent=4)
        if args.report_to == "wandb":
            import wandb

            wandb.init(project=args.tracker_project_name, name=args.name, resume="allow", id=args.name)

    config.global_world_size = get_world_size()
    logger.info(f"Config: \n{config}")
    logger.info(f"World_size: {config.global_world_size}, seed: {config.train.seed}")
    logger.info(f"Initializing: {init_train} for training")

    # scheduler
    pred_sigma = getattr(config.scheduler, "pred_sigma", True)
    learn_sigma = getattr(config.scheduler, "learn_sigma", True) and pred_sigma

    # VAE
    vae = None
    vae_dtype = get_weight_dtype(config.vae.weight_dtype)
    # if not config.data.load_vae_feat:
    # VAE is always required for image and video joint training
    vae = get_vae(
        config.vae.vae_type, config.vae.vae_pretrained, accelerator.device, dtype=vae_dtype, config=config.vae
    )

    logger.info(f"vae type: {config.vae.vae_type}, path: {config.vae.vae_pretrained}, weight_dtype: {vae_dtype}")

    # Text encoder
    max_length = config.text_encoder.model_max_length
    tokenizer = text_encoder = text_handler = None
    if not config.data.load_text_feat:
        tokenizer, text_encoder = get_tokenizer_and_text_encoder(
            name=config.text_encoder.text_encoder_name, device=accelerator.device
        )
        if "Qwen" in config.text_encoder.text_encoder_name:
            text_handler = text_encoder
            text_encoder = text_handler.text_encoder
        text_embed_dim = text_encoder.config.hidden_size
    else:
        text_embed_dim = config.text_encoder.caption_channels

    if config.text_encoder.chi_prompt:
        chi_prompt = "\n".join(config.text_encoder.chi_prompt)
        logger.info(f"Complex Human Instruct: {chi_prompt}")
    if config.task == "ti2v":
        logger.info(f"Image latent mode: {config.model.image_latent_mode}")
    if config.train.ltx_image_condition_prob > 0:
        logger.info(f"Image condition probability: {config.train.ltx_image_condition_prob}")

    os.makedirs(config.train.null_embed_root, exist_ok=True)
    null_embed_path = osp.join(
        config.train.null_embed_root,
        f"null_embed_diffusers_{config.text_encoder.text_encoder_name}_{max_length}token_{text_embed_dim}.pth",
    )

    # Image Prior
    if config.task == "ti2v" and config.get("image_encoder", {}).get("image_encoder_name") == "flux-siglip":
        image_encoder_dtype = get_weight_dtype(config.image_encoder.weight_dtype)
        image_encoder, image_processor = get_image_encoder(
            name=config.image_encoder.image_encoder_name,
            model_path=config.image_encoder.image_encoder_path,
            device=accelerator.device,
            dtype=image_encoder_dtype,
        )
    else:
        image_encoder, image_processor = None, None

    # 2. build scheduler
    train_diffusion = Scheduler(
        str(config.scheduler.train_sampling_steps),
        noise_schedule=config.scheduler.noise_schedule,
        predict_flow_v=config.scheduler.predict_flow_v,
        learn_sigma=learn_sigma,
        pred_sigma=pred_sigma,
        snr=config.train.snr_loss,
        flow_shift=config.scheduler.flow_shift,
    )
    predict_info = (
        f"flow-prediction: {config.scheduler.predict_flow_v}, noise schedule: {config.scheduler.noise_schedule}"
    )
    if "flow" in config.scheduler.noise_schedule:
        predict_info += f", flow shift: {config.scheduler.flow_shift}"
        if config.scheduler.inference_flow_shift is not None:
            predict_info += f", inference flow shift: {config.scheduler.inference_flow_shift}"
    if config.scheduler.weighting_scheme in ["logit_normal", "mode"]:
        predict_info += (
            f", flow weighting: {config.scheduler.weighting_scheme}, "
            f"logit-mean: {config.scheduler.logit_mean}, logit-std: {config.scheduler.logit_std}"
        )
    logger.info(predict_info)

    # 3. build dataloader
    config.data.data_dir = (
        config.data.data_dir if isinstance(config.data.data_dir, dict) else {"default": config.data.data_dir}
    )
    config.data.data_dir = {
        k: data if data.startswith(("https://", "http://", "gs://", "/", "~")) else osp.abspath(osp.expanduser(data))
        for k, data in config.data.data_dir.items()
    }
    config.image_data.data_dir = (
        config.image_data.data_dir if isinstance(config.image_data.data_dir, list) else [config.image_data.data_dir]
    )
    config.image_data.data_dir = [
        data if data.startswith(("https://", "http://", "gs://", "/", "~")) else osp.abspath(osp.expanduser(data))
        for data in config.image_data.data_dir
    ]

    num_replicas = int(os.environ["WORLD_SIZE"])
    rank = int(os.environ["RANK"])
    joint_training_interval = config.train.joint_training_interval

    # video dataset
    set_random_seed(int(time.time()) % (2**31) + int(os.environ["LOCAL_RANK"]))
    if config.model.aspect_ratio_type is not None:
        config.data.aspect_ratio_type = config.model.aspect_ratio_type
    dataset = build_dataset(
        asdict(config.data),
        resolution=config.data.image_size,
        max_length=max_length,
        config=config,
        caption_proportion=config.data.caption_proportion,
        sort_dataset=config.data.sort_dataset,
        vae_downsample_rate=config.vae.vae_stride[-1],
        num_frames=config.data.num_frames,
    )
    sampler = DistributedRangedSampler(dataset, num_replicas=num_replicas, rank=rank)

    # image dataset
    if config.model.aspect_ratio_type is not None:
        config.image_data.aspect_ratio_type = config.model.aspect_ratio_type
    dataset_image = build_dataset(
        asdict(config.image_data),
        resolution=config.image_data.image_size,
        max_length=max_length,
        config=config,
        caption_proportion=config.image_data.caption_proportion,
        sort_dataset=config.image_data.sort_dataset,
        vae_downsample_rate=config.vae.vae_stride[-1],
        num_frames=config.image_data.num_frames,
    )

    image_sampler = DistributedRangedSampler(dataset_image, num_replicas=num_replicas, rank=rank)

    if config.model.multi_scale:
        batch_sampler = AspectRatioBatchSamplerVideo(
            sampler=sampler,
            dataset=dataset,
            batch_size=config.train.train_batch_size,
            aspect_ratios=dataset.aspect_ratio,
            drop_last=True,
            ratio_nums=dataset.ratio_nums,
            config=config,
            valid_num=config.data.valid_num,
        )
        train_dataloader = build_dataloader(
            dataset, batch_sampler=batch_sampler, num_workers=config.train.num_workers, dataloader_type="video"
        )
        train_dataloader_len = len(train_dataloader)

        batch_sampler_image = AspectRatioBatchSampler(
            sampler=image_sampler,
            dataset=dataset_image,
            batch_size=config.train.train_batch_size_image,
            aspect_ratios=dataset_image.aspect_ratio,
            drop_last=True,
            ratio_nums=dataset_image.ratio_nums,
            config=config,
            clipscore_filter_thres=args.data.del_img_clip_thr,
        )
        train_dataloader_image = build_dataloader(
            dataset_image,
            batch_sampler=batch_sampler_image,
            num_workers=config.train.num_workers,
            dataloader_type="image",
        )
        train_dataloader_image_len = len(train_dataloader_image)

    else:
        train_dataloader = build_dataloader(
            dataset,
            num_workers=config.train.num_workers,
            batch_size=config.train.train_batch_size,
            shuffle=False,
            sampler=sampler,
            dataloader_type="video",
        )
        train_dataloader_len = len(train_dataloader)

        # Build image dataloader for joint training
        train_dataloader_image = build_dataloader(
            dataset_image,
            num_workers=config.train.num_workers,
            batch_size=config.train.train_batch_size_image,
            shuffle=False,
            sampler=image_sampler,
            dataloader_type="image",
        )
        train_dataloader_image_len = len(train_dataloader_image)

    logger.info(
        f"Video set DataLoader length: {train_dataloader_len}, Image DataLoader length: {train_dataloader_image_len}"
    )
    if joint_training_interval > 0:
        logger.info(
            colored(
                f"Joint training mode enabled: Image data will be trained with every {joint_training_interval} video iterations",
                "red",
            )
        )
    else:
        logger.info(
            colored(
                f"Joint training mode disabled: Image data will not be trained with image data",
                "red",
            )
        )
    load_vae_feat = getattr(train_dataloader.dataset, "load_vae_feat", False)
    load_text_feat = getattr(train_dataloader.dataset, "load_text_feat", False)

    # prepare input for visualization during training
    # aspect_ratio_key = random.choice(list(dataset.aspect_ratio.keys()))
    aspect_ratio_key = "0.57"
    video_height, video_width = map(int, dataset.aspect_ratio[aspect_ratio_key])
    num_frames = config.data.num_frames
    latent_width = int(video_width) // config.vae.vae_stride[2]
    latent_height = int(video_height) // config.vae.vae_stride[1]
    latent_temp = int(num_frames - 1) // config.vae.vae_stride[0] + 1

    validation_noise = (
        torch.randn(
            1, config.vae.vae_latent_dim, latent_temp, latent_height, latent_width, device="cpu", generator=generator
        )
        if getattr(config.train, "deterministic_validation", False)
        else None
    )

    if not config.data.load_vae_feat and config.vae.cache_dir is not None:
        vae_cache_dir = os.path.join(
            config.vae.cache_dir,
            f"{config.vae.vae_type}_{num_frames}x{video_height}x{video_width}",
        )
        os.makedirs(vae_cache_dir, exist_ok=True)
        vae.cfg.cache_dir = vae_cache_dir
        logger.info(f"Cache VAE latent of {num_frames}x{video_height}x{video_width} to {vae_cache_dir}")

    # 4.preparing embeddings for visualization. We put it here for saving GPU memory
    if config.train.visualize and len(config.train.validation_prompts):
        valid_prompt_embed_suffix = f"{max_length}token_{config.text_encoder.text_encoder_name}_{text_embed_dim}.pth"
        validation_prompts = config.train.validation_prompts
        validation_images = config.train.validation_images
        skip = True
        if config.text_encoder.chi_prompt:
            uuid_sys_prompt = hashlib.sha256(chi_prompt.encode()).hexdigest()
        else:
            uuid_sys_prompt = hashlib.sha256(b"").hexdigest()
        config.train.valid_prompt_embed_root = osp.join(
            config.train.valid_prompt_embed_root,
            f"{uuid_sys_prompt}_{config.task}_{latent_height}x{latent_width}_{config.vae.vae_type}_{config.model.image_latent_mode}",
        )
        Path(config.train.valid_prompt_embed_root).mkdir(parents=True, exist_ok=True)

        if config.text_encoder.chi_prompt:
            # Save system prompt to a file
            system_prompt_file = osp.join(config.train.valid_prompt_embed_root, "system_prompt.txt")
            with open(system_prompt_file, "w", encoding="utf-8") as f:
                f.write(chi_prompt)

        for prompt in validation_prompts:
            prompt_embed_path = osp.join(
                config.train.valid_prompt_embed_root, f"{prompt[:50]}_{valid_prompt_embed_suffix}"
            )
            if not (osp.exists(prompt_embed_path) and osp.exists(null_embed_path)):
                skip = False
                logger.info(f"Preparing Visualization prompt embeddings at: {config.train.valid_prompt_embed_root}")
                break
        if accelerator.is_main_process and not skip:
            if config.data.load_text_feat and (tokenizer is None or text_encoder is None):
                logger.info(f"Loading text encoder and tokenizer from {config.text_encoder.text_encoder_name} ...")
                tokenizer, text_encoder = get_tokenizer_and_text_encoder(name=config.text_encoder.text_encoder_name)

            for i, prompt in enumerate(validation_prompts):
                prompt_embed_path = osp.join(
                    config.train.valid_prompt_embed_root, f"{prompt[:50]}_{valid_prompt_embed_suffix}"
                )
                if "T5" in config.text_encoder.text_encoder_name:
                    txt_tokens = tokenizer(
                        prompt, max_length=max_length, padding="max_length", truncation=True, return_tensors="pt"
                    ).to(accelerator.device)
                    caption_emb = text_encoder(txt_tokens.input_ids, attention_mask=txt_tokens.attention_mask)[0]
                    caption_emb_mask = txt_tokens.attention_mask
                elif "gemma" in config.text_encoder.text_encoder_name:
                    if not config.text_encoder.chi_prompt:
                        max_length_all = config.text_encoder.model_max_length
                    else:
                        chi_prompt = "\n".join(config.text_encoder.chi_prompt)
                        prompt = chi_prompt + prompt
                        num_sys_prompt_tokens = len(tokenizer.encode(chi_prompt))
                        max_length_all = (
                            num_sys_prompt_tokens + config.text_encoder.model_max_length - 2
                        )  # magic number 2: [bos], [_]

                    txt_tokens = tokenizer(
                        prompt,
                        max_length=max_length_all,
                        padding="max_length",
                        truncation=True,
                        return_tensors="pt",
                    ).to(accelerator.device)
                    select_index = [0] + list(range(-config.text_encoder.model_max_length + 1, 0))
                    caption_emb = text_encoder(txt_tokens.input_ids, attention_mask=txt_tokens.attention_mask)[0][
                        :, select_index
                    ]
                    caption_emb_mask = txt_tokens.attention_mask[:, select_index]
                elif "Qwen" in config.text_encoder.text_encoder_name:
                    with torch.no_grad():
                        caption_emb, caption_emb_mask = text_handler.get_prompt_embeds(prompt, max_length=max_length)
                else:
                    raise ValueError(f"{config.text_encoder.text_encoder_name} is not supported!!")

                save_dict = {"caption_embeds": caption_emb, "emb_mask": caption_emb_mask}
                if config.task == "ti2v" or config.task == "df" or config.task == "ltx":
                    image_path = validation_images[i]
                    image = read_image_from_path(image_path, (video_height, video_width))  # C,H,W

                    # image vae embeds
                    if config.model.image_latent_mode == "repeat" or config.task == "df" or config.task == "ltx":
                        image_vae_embeds = vae_encode(
                            config.vae.vae_type, vae, image[None, :, None].to(vae_dtype), device=accelerator.device
                        )  # 1,C,1,H,W
                    elif config.model.image_latent_mode == "video_zero":
                        dummy_vid = torch.zeros_like(image[None, :, None]).repeat(
                            1, 1, config.data.num_frames, 1, 1
                        )  # C,H,W -> 1,C,1,H,W -> 1,C,F,H,W
                        dummy_vid[:, :, :1] = image[None, :, None]
                        image_vae_embeds = vae_encode(
                            config.vae.vae_type, vae, dummy_vid.to(vae_dtype), device=accelerator.device
                        )  # 1,C,F,H,W
                    save_dict["image_vae_embeds"] = image_vae_embeds.cpu()

                    # image encoder embeds, diffusion forcing do not need image encoder embeds
                    if (
                        config.get("image_encoder", {}).get("image_encoder_name") == "flux-siglip"
                        and not config.task == "df"
                        and not config.task == "ltx"
                    ):
                        image_embeds = encode_image(
                            name=config.image_encoder.image_encoder_name,
                            image_encoder=image_encoder,
                            image_processor=image_processor,
                            images=image,
                            device=accelerator.device,
                            dtype=image_encoder.dtype,
                        )
                        save_dict["image_embeds"] = image_embeds.cpu()

                torch.save(save_dict, prompt_embed_path)

            if "T5" in config.text_encoder.text_encoder_name:
                null_tokens = tokenizer(
                    "", max_length=max_length, padding="max_length", truncation=True, return_tensors="pt"
                ).to(accelerator.device)
                null_token_emb = text_encoder(null_tokens.input_ids, attention_mask=null_tokens.attention_mask)[0]
                null_token_mask = null_tokens.attention_mask
            elif "gemma" in config.text_encoder.text_encoder_name:
                null_tokens = tokenizer(
                    "", max_length=max_length, padding="max_length", truncation=True, return_tensors="pt"
                ).to(accelerator.device)
                null_token_emb = text_encoder(null_tokens.input_ids, attention_mask=null_tokens.attention_mask)[0]
                null_token_mask = null_tokens.attention_mask
            elif "Qwen" in config.text_encoder.text_encoder_name:
                with torch.no_grad():
                    null_tokens = None
                    null_token_emb, null_token_mask = text_handler.get_prompt_embeds("", max_length=max_length)
            else:
                raise ValueError(f"{config.text_encoder.text_encoder_name} is not supported!!")
            torch.save(
                {"uncond_prompt_embeds": null_token_emb, "uncond_prompt_embeds_mask": null_token_mask},
                null_embed_path,
            )
            if config.data.load_text_feat:
                del tokenizer
                del text_encoder
            del null_token_emb
            del null_tokens
            flush()

    # 5. build models
    os.environ["AUTOCAST_LINEAR_ATTN"] = "true" if config.model.autocast_linear_attn else "false"
    image_size = config.model.image_size
    latent_size = int(image_size) // config.vae.vae_stride[-1]
    model_kwargs = model_video_init_config(config, latent_size=latent_size)
    model = build_model(
        config.model.model,
        config.train.grad_checkpointing,
        getattr(config.model, "fp32_attention", False),
        null_embed_path=null_embed_path,
        **model_kwargs,
    ).train()

    if (not config.train.use_fsdp) and config.train.ema_update:
        model_ema = deepcopy(model).eval()
        logger.info("Creating EMA model for DDP mode")
    elif config.train.use_fsdp and config.train.ema_update:
        logger.warning("EMA update is not supported in FSDP mode. Setting model_ema to None.")
        model_ema = None
    else:
        model_ema = None

    logger.info(
        colored(
            f"{model.__class__.__name__}:{config.model.model}, "
            f"Model Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M",
            "green",
            attrs=["bold"],
        )
    )

    if config.train.use_fsdp:
        model_instance = deepcopy(model)
    elif model_ema is not None:
        model_instance = deepcopy(model_ema)
    else:
        model_instance = model

    # 5-1. load model
    if args.load_from is not None:
        config.model.load_from = args.load_from
    if config.model.load_from is not None and load_from:

        load_result = load_checkpoint(
            checkpoint=config.model.load_from,
            model=model,
            model_ema=model_ema,
            FSDP=config.train.use_fsdp,
            load_ema=config.model.resume_from.get("load_ema", False),
            null_embed_path=null_embed_path,
        )

        _, missing, unexpected, _, _ = load_result
        logger.warning(colored(f"Missing keys: {missing}", "red"))
        logger.warning(colored(f"Unexpected keys: {unexpected}", "red"))

    if config.train.ema_update and not config.train.use_fsdp and model_ema is not None:
        ema_update(model_ema, model, 0.0)

    # 5-2. model growth
    if config.model_growth is not None:
        from diffusion.model.model_growth_utils import ModelGrowthInitializer

        assert config.model.load_from is None
        model_growth_initializer = ModelGrowthInitializer(model, config.model_growth)
        model = model_growth_initializer.initialize(
            strategy=config.model_growth.init_strategy, **config.model_growth.init_params
        )

    # 6. build optimizer and lr scheduler
    lr_scale_ratio = 1
    if getattr(config.train, "auto_lr", None):
        lr_scale_ratio = auto_scale_lr(
            config.train.train_batch_size * get_world_size() * config.train.gradient_accumulation_steps,
            config.train.optimizer,
            **config.train.auto_lr,
        )
    optimizer = build_optimizer(model, config.train.optimizer)

    if config.train.lr_schedule_args and config.train.lr_schedule_args.get("num_warmup_steps", None):
        config.train.lr_schedule_args["num_warmup_steps"] = (
            config.train.lr_schedule_args["num_warmup_steps"] * num_replicas
        )
    lr_scheduler = build_lr_scheduler(config.train, optimizer, train_dataloader, lr_scale_ratio)
    logger.warning(
        f"{colored(f'Basic Training Settings: ', 'green', attrs=['bold'])}"
        f"lr: {config.train.optimizer['lr']:.5f}, bs: {config.train.train_batch_size}, gc: {config.train.grad_checkpointing}, "
        f"gc_accum_step: {config.train.gradient_accumulation_steps}."
    )
    logger.info(
        f"{colored(f'Model Settings: ', 'green', attrs=['bold'])}"
        f"qk norm: {config.model.qk_norm}, fp32 attn: {config.model.fp32_attention}, attn type: {config.model.attn_type}, linear_head_dim: {config.model.linear_head_dim}, ffn type: {config.model.ffn_type}, "
        f"text encoder: {config.text_encoder.text_encoder_name}, captions: {config.data.caption_proportion}, precision: {config.model.mixed_precision}."
    )

    timestamp = time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())

    if accelerator.is_main_process:
        tracker_config = dict(vars(config))
        try:
            accelerator.init_trackers(args.tracker_project_name, tracker_config)
        except:
            accelerator.init_trackers(f"tb_{timestamp}")

    start_epoch = 0
    start_step = 0
    start_video_step = 0  # Initialize video step counter
    start_image_step = 0  # Initialize image step counter
    total_steps = train_dataloader_len * config.train.num_epochs

    # 7. Resume training
    if config.model.resume_from is not None and config.model.resume_from["checkpoint"] is not None:
        rng_state = None
        loaded_image_step = None
        loaded_video_step = None
        ckpt_path = osp.join(config.work_dir, "checkpoints")
        check_flag = osp.exists(ckpt_path) and len(os.listdir(ckpt_path)) != 0
        remove_state_dict_keys = config.model.remove_state_dict_keys

        if config.model.resume_from["checkpoint"] == "latest":
            if check_flag:
                remove_state_dict_keys = None
                config.model.resume_from["resume_optimizer"] = True
                config.model.resume_from["resume_lr_scheduler"] = True
                checkpoints = os.listdir(ckpt_path)
                if "latest.pth" in checkpoints and osp.exists(osp.join(ckpt_path, "latest.pth")):
                    config.model.resume_from["checkpoint"] = osp.realpath(osp.join(ckpt_path, "latest.pth"))
                else:
                    checkpoints = [i for i in checkpoints if i.startswith("epoch_")]
                    checkpoints = sorted(checkpoints, key=lambda x: int(x.replace(".pth", "").split("_")[3]))
                    config.model.resume_from["checkpoint"] = osp.join(ckpt_path, checkpoints[-1])
            else:
                config.model.resume_from["resume_optimizer"] = config.train.load_from_optimizer
                config.model.resume_from["resume_lr_scheduler"] = config.train.load_from_lr_scheduler
                config.model.resume_from["checkpoint"] = config.model.load_from

        if config.model.resume_from["checkpoint"] is not None:

            load_result = load_checkpoint(
                **config.model.resume_from,
                model=model,
                model_ema=model_ema if not config.train.use_fsdp else None,
                FSDP=config.train.use_fsdp,
                optimizer=optimizer,
                lr_scheduler=lr_scheduler,
                null_embed_path=null_embed_path,
                remove_state_dict_keys=remove_state_dict_keys,
            )

            # Handle both old and new return formats
            epoch, missing, unexpected, rng_state, saved_info = load_result
            loaded_video_step = saved_info.get("video_step", None)
            loaded_image_step = saved_info.get("image_step", None)

            logger.warning(colored(f"Missing keys: {missing}", "red"))
            logger.warning(colored(f"Unexpected keys: {unexpected}", "red"))

            path = osp.basename(config.model.resume_from["checkpoint"])
        try:
            start_epoch = int(path.replace(".pth", "").split("_")[1]) - 1
            start_step = int(path.replace(".pth", "").split("_")[3])
        except:
            pass

        # Set video_step and image_step based on availability
        if loaded_video_step is not None:
            start_video_step = loaded_video_step
            logger.info(f"Loaded video_step: {start_video_step} from checkpoint")
        else:
            # If no video_step in checkpoint, use global_step as video_step
            start_video_step = start_step
            logger.info(f"No video_step in checkpoint, using global_step as video_step: {start_video_step}")

        if loaded_image_step is not None:
            start_image_step = loaded_image_step
            logger.info(f"Loaded image_step: {start_image_step} from checkpoint")
        else:
            # If no image_step in checkpoint, start from 0
            start_image_step = 0
            logger.info(f"No image_step in checkpoint, starting image_step from 0")

    # 8. Prepare everything
    # There is no specific order to remember, you just need to unpack the
    # objects in the same order you gave them to the prepare method.
    model = accelerator.prepare(model)
    if model_ema is not None and not config.train.use_fsdp:
        model_ema = accelerator.prepare(model_ema)
    optimizer, lr_scheduler = accelerator.prepare(optimizer, lr_scheduler)

    # load everything except model when resume
    if (
        config.train.use_fsdp
        and config.model.resume_from is not None
        and config.model.resume_from["checkpoint"] is not None
        and config.model.resume_from["resume_optimizer"]
        and config.model.resume_from["resume_lr_scheduler"]
    ):
        logger.info(f"FSDP resume: Loading optimizer, scheduler, scaler, random_states...")
        accelerator.load_state(
            os.path.join(config.model.resume_from["checkpoint"], "model"),
            state_dict_key=["optimizer", "scheduler", "scaler", "random_states"],
        )

    set_random_seed((start_step + 1) // config.train.save_model_steps + int(os.environ["LOCAL_RANK"]))
    logger.info(f'Set seed: {(start_step + 1) // config.train.save_model_steps + int(os.environ["LOCAL_RANK"])}')

    # Start Training
    train(
        config=config,
        args=args,
        accelerator=accelerator,
        model=model,
        model_ema=model_ema,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        train_dataloader=train_dataloader,
        train_dataloader_image=train_dataloader_image,
        train_diffusion=train_diffusion,
        logger=logger,
    )


if __name__ == "__main__":
    main()
