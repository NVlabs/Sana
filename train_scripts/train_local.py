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
import hashlib
import os
import os.path as osp
import time
import types
import warnings
from pathlib import Path

import numpy as np
import pyrallis
import torch
import torch.utils
import torch.utils.data
from PIL import Image
from accelerate import Accelerator, InitProcessGroupKwargs
from accelerate.utils import DistributedType
from termcolor import colored

warnings.filterwarnings("ignore")  # ignore warning

import gc
import json
import math
import random

from diffusion import DPMS, FlowEuler, Scheduler
from diffusion.model.builder import build_model, get_tokenizer_and_text_encoder, get_vae, vae_decode
from diffusion.model.respace import compute_density_for_timestep_sampling
from diffusion.utils.checkpoint import load_checkpoint, save_checkpoint
from diffusion.utils.config import SanaConfig
from diffusion.utils.dist_utils import clip_grad_norm_, flush, get_world_size
from diffusion.utils.logger import LogBuffer, get_root_logger
from diffusion.utils.lr_scheduler import build_lr_scheduler
from diffusion.utils.misc import DebugUnderflowOverflow, init_random_seed, set_random_seed
from diffusion.utils.optimizer import build_optimizer

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def set_fsdp_env():
    os.environ["ACCELERATE_USE_FSDP"] = "true"
    os.environ["FSDP_AUTO_WRAP_POLICY"] = "TRANSFORMER_BASED_WRAP"
    os.environ["FSDP_BACKWARD_PREFETCH"] = "BACKWARD_PRE"
    os.environ["FSDP_TRANSFORMER_CLS_TO_WRAP"] = "SanaBlock"


image_index = 0


@torch.inference_mode()
def idation(accelerator, config, model, step, device, vae=None, init_noise=None):
    torch.cuda.empty_cache()
    vis_sampler = config.scheduler.vis_sampler
    model = accelerator.unwrap_model(model).eval()
    hw = torch.tensor([[image_size, image_size]], dtype=torch.float, device=device).repeat(1, 1)
    ar = torch.tensor([[1.0]], device=device).repeat(1, 1)
    null_y = torch.load(null_embed_path, map_location="cpu")
    null_y = null_y["uncond_prompt_embeds"].to(device)

    # Create sampling noise:
    logger.info("Running validation... ")
    image_logs = []

    del_vae = False
    if vae is None:
        vae = get_vae(config.vae.vae_type, config.vae.vae_pretrained, accelerator.device).to(torch.float16)
        del_vae = True

    def run_sampling(init_z=None, label_suffix="", sampler="dpm-solver"):
        latents = []
        current_image_logs = []

        for prompt in validation_prompts:
            logger.info(prompt)
            z = (
                torch.randn(1, config.vae.vae_latent_dim, latent_size, latent_size, device=device)
                if init_z is None
                else init_z
            )
            embed = torch.load(
                osp.join(config.train.valid_prompt_embed_root, f"{prompt[:50]}_{valid_prompt_embed_suffix}"),
                map_location="cpu",
            )
            caption_embs, emb_masks = embed["caption_embeds"].to(device), embed["emb_mask"].to(device)
            # caption_embs = caption_embs[:, None]
            # emb_masks = emb_masks[:, None]
            model_kwargs = dict(data_info={"img_hw": hw, "aspect_ratio": ar}, mask=emb_masks)

            if sampler == "dpm-solver":
                dpm_solver = DPMS(
                    model.forward_with_dpmsolver,
                    condition=caption_embs,
                    uncondition=null_y,
                    cfg_scale=4.5,
                    model_kwargs=model_kwargs,
                )
                denoised = dpm_solver.sample(
                    z,
                    steps=14,
                    order=2,
                    skip_type="time_uniform",
                    method="multistep",
                )
            elif sampler == "flow_euler":
                flow_solver = FlowEuler(
                    model, condition=caption_embs, uncondition=null_y, cfg_scale=5.5, model_kwargs=model_kwargs
                )
                denoised = flow_solver.sample(z, steps=28)
            elif sampler == "flow_dpm-solver":
                dpm_solver = DPMS(
                    model.forward_with_dpmsolver,
                    condition=caption_embs,
                    uncondition=null_y,
                    cfg_scale=5.5,
                    model_type="flow",
                    model_kwargs=model_kwargs,
                    schedule="FLOW",
                )

                denoised = dpm_solver.sample(
                    z,
                    steps=24,
                    order=2,
                    skip_type="time_uniform_flow",
                    method="multistep",
                    flow_shift=config.scheduler.flow_shift,
                )
            else:
                raise ValueError(f"{sampler} not implemented")

            latents.append(denoised)
        torch.cuda.empty_cache()

        for prompt, latent in zip(validation_prompts, latents):
            latent = latent.to(torch.float16)
            samples = vae_decode(config.vae.vae_type, vae, latent)
            samples = (
                torch.clamp(127.5 * samples + 128.0, 0, 255).permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()[0]
            )
            image = Image.fromarray(samples)
            current_image_logs.append({"validation_prompt": prompt + label_suffix, "images": [image]})

        return current_image_logs

    # First run with original noise
    image_logs += run_sampling(init_z=None, label_suffix="", sampler=vis_sampler)

    # Second run with init_noise if provided
    if init_noise is not None:
        init_noise = torch.clone(init_noise).to(device)
        image_logs += run_sampling(init_z=init_noise, label_suffix=" w/ init noise", sampler=vis_sampler)

    if del_vae:
        vae = None
        gc.collect()
        torch.cuda.empty_cache()

    formatted_images = []
    for log in image_logs:
        images = log["images"]
        validation_prompt = log["validation_prompt"]
        for image in images:
            formatted_images.append((validation_prompt, np.asarray(image)))

    for tracker in accelerator.trackers:
        if tracker.name == "tensorboard":
            for validation_prompt, image in formatted_images:
                tracker.writer.add_images(validation_prompt, image[None, ...], step, dataformats="NHWC")
        elif tracker.name == "wandb":
            import wandb

            wandb_images = []
            for validation_prompt, image in formatted_images:
                wandb_images.append(wandb.Image(image, caption=validation_prompt, file_type="jpg"))
            tracker.log({"validation": wandb_images})
        else:
            logger.warn(f"image logging not implemented for {tracker.name}")

    def concatenate_images(image_caption, images_per_row=5, image_format="webp"):
        import io

        images = [log["images"][0] for log in image_caption]
        if images[0].size[0] > 1024:
            images = [image.resize((1024, 1024)) for image in images]

        widths, heights = zip(*(img.size for img in images))
        max_width = max(widths)
        total_height = sum(heights[i: i + images_per_row][0] for i in range(0, len(images), images_per_row))

        new_im = Image.new("RGB", (max_width * images_per_row, total_height))

        y_offset = 0
        for i in range(0, len(images), images_per_row):
            row_images = images[i: i + images_per_row]
            x_offset = 0
            for img in row_images:
                new_im.paste(img, (x_offset, y_offset))
                x_offset += max_width
            y_offset += heights[i]
        webp_image_bytes = io.BytesIO()
        new_im.save(webp_image_bytes, format=image_format)
        webp_image_bytes.seek(0)
        new_im = Image.open(webp_image_bytes)

        return new_im

    if config.train.local_save_vis:
        file_format = "webp"
        local_vis_save_path = osp.join(config.work_dir, "log_vis")
        os.umask(0o000)
        os.makedirs(local_vis_save_path, exist_ok=True)
        concatenated_image = concatenate_images(image_logs, images_per_row=5, image_format=file_format)
        save_path = (
            osp.join(local_vis_save_path, f"vis_{step}.{file_format}")
            if init_noise is None
            else osp.join(local_vis_save_path, f"vis_{step}_w_init.{file_format}")
        )
        concatenated_image.save(save_path)

    del vae
    flush()
    return image_logs


class RatioBucketsDataset:
    def __init__(self, buckets_file):
        with open(buckets_file) as file:
            self.buckets = json.load(file)

    def __getitem__(self, idx):
        while True:
            loader = random.choice(self.loaders)

            try:
                return next(loader)
            except StopIteration:
                self.loaders.remove(loader)
                logger.info(f"bucket ended, {len(self.loaders)}")

    def __len__(self):
        return self.size

    def make_loaders(self, batch_size):
        self.loaders = []
        self.size = 0
        for bucket in self.buckets.keys():
            dataset = ImageDataset(self.buckets[bucket])

            loader = torch.utils.data.DataLoader(
                dataset, batch_size=batch_size, shuffle=True, pin_memory=False, drop_last=False
            )
            self.loaders.append(iter(loader))
            self.size += math.ceil(len(dataset) / batch_size)


class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, images):
        self.images = images

    def getdata(self, idx):
        path = self.images[idx]
        filename_wo_ext = os.path.splitext(os.path.basename(path))[0]

        text_file = os.path.join(os.path.dirname(path), f"{filename_wo_ext}.txt")
        with open(text_file) as file:
            prompt = file.read()

        cache_file = os.path.join(os.path.dirname(path), f"{filename_wo_ext}_img.npz")
        cached_data = torch.load(cache_file)

        size = cached_data["prefsize"]
        ratio = cached_data["ratio"]
        vae_embed = cached_data["img"]

        data_info = {
            "img_hw": size,
            "aspect_ratio": torch.tensor(ratio.item()),
        }

        return (
            vae_embed,
            data_info,
            prompt,
        )

    def __getitem__(self, idx):
        for _ in range(10):
            try:
                data = self.getdata(idx)
                return data
            except Exception as e:
                logger.error(f"Error details: {str(e)}")
                idx = idx + 1
        raise RuntimeError("Too many bad data.")

    def __len__(self):
        return len(self.images)


def train(config, args, accelerator, model, optimizer, lr_scheduler, dataset, train_diffusion):
    if getattr(config.train, "debug_nan", False):
        DebugUnderflowOverflow(model)
        logger.info("NaN debugger registered. Start to detect overflow during training.")
    log_buffer = LogBuffer()

    def check_nan_inf(model):
        for name, param in model.named_parameters():
            if torch.isnan(param).any() or torch.isinf(param).any():
                logger.error(f"NaN/Inf detected in {name}")

    check_nan_inf(model)

    def save_model(save_metric=True):
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            os.umask(0o000)
            checkpoints_dir = osp.join(config.work_dir, "checkpoints")

            # Remove all old checkpoint files in the directory
            if os.path.exists(checkpoints_dir):
                for filename in os.listdir(checkpoints_dir):
                    file_path = osp.join(checkpoints_dir, filename)
                    if os.path.isfile(file_path):
                        os.remove(file_path)

            ckpt_saved_path = save_checkpoint(
                checkpoints_dir,
                epoch=epoch,
                step=global_step,
                model=accelerator.unwrap_model(model),
                optimizer=optimizer,
                lr_scheduler=lr_scheduler,
                generator=generator,
                add_symlink=True,
            )
            if save_metric:
                online_metric_monitor_dir = osp.join(config.work_dir, config.train.online_metric_dir)
                os.makedirs(online_metric_monitor_dir, exist_ok=True)
                with open(f"{online_metric_monitor_dir}/{ckpt_saved_path.split('/')[-1]}.txt", "w") as f:
                    f.write(osp.join(config.work_dir, "config.py") + "\n")
                    f.write(ckpt_saved_path)

    global_step = start_step + 1
    skip_step = max(config.train.skip_step, global_step) % len(dataset)
    skip_step = skip_step if skip_step < (len(dataset) - 20) else 0
    loss_nan_timer = 0

    # Now you train the model
    for epoch in range(start_epoch + 1, config.train.num_epochs + 1):
        time_start, last_tic = time.time(), time.time()
        if skip_step > 1 and accelerator.is_main_process:
            logger.info(f"Skipped Steps: {skip_step}")
        skip_step = 1
        data_time_start = time.time()
        data_time_all = 0
        lm_time_all = 0
        model_time_all = 0
        optimizer_time_all = 0
        dataset.make_loaders(config.train.train_batch_size)
        for step, batch in enumerate(dataset):
            # image, json_info, key = batch
            accelerator.wait_for_everyone()
            data_time_all += time.time() - data_time_start
            z = batch[0].to(accelerator.device)

            accelerator.wait_for_everyone()

            clean_images = z
            data_info = batch[1]

            lm_time_start = time.time()
            prompts = list(batch[2])
            shuffled_prompts = []
            for prompt in prompts:
                tags = prompt.split(",")  # Split the string into a list of tags
                random.shuffle(tags)  # Shuffle the tags
                shuffled_prompts.append(",".join(tags))  # Join them back into a string

            if "T5" in config.text_encoder.text_encoder_name:
                with torch.no_grad():
                    txt_tokens = tokenizer(
                        shuffled_prompts,
                        max_length=max_length,
                        padding="max_length",
                        truncation=True,
                        return_tensors="pt",
                    ).to(accelerator.device)
                    y = text_encoder(txt_tokens.input_ids, attention_mask=txt_tokens.attention_mask)[0][:, None]
                    y_mask = txt_tokens.attention_mask[:, None, None]
            elif "gemma" in config.text_encoder.text_encoder_name or "Qwen" in config.text_encoder.text_encoder_name:
                with torch.no_grad():
                    if not config.text_encoder.chi_prompt:
                        max_length_all = config.text_encoder.model_max_length
                        prompt = shuffled_prompts
                    else:
                        chi_prompt = "\n".join(config.text_encoder.chi_prompt)
                        prompt = [chi_prompt + i for i in shuffled_prompts]
                        num_chi_prompt_tokens = len(tokenizer.encode(chi_prompt))
                        max_length_all = (
                                num_chi_prompt_tokens + config.text_encoder.model_max_length - 2
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
            else:
                logger.error("error")
                exit()

            # Sample a random timestep for each image
            bs = clean_images.shape[0]
            timesteps = torch.randint(
                0, config.scheduler.train_sampling_steps, (bs,), device=clean_images.device
            ).long()
            if config.scheduler.weighting_scheme in ["logit_normal"]:
                # adapting from diffusers.training_utils
                u = compute_density_for_timestep_sampling(
                    weighting_scheme=config.scheduler.weighting_scheme,
                    batch_size=bs,
                    logit_mean=config.scheduler.logit_mean,
                    logit_std=config.scheduler.logit_std,
                    mode_scale=None,  # not used
                )
                timesteps = (u * config.scheduler.train_sampling_steps).long().to(clean_images.device)
            grad_norm = None
            accelerator.wait_for_everyone()
            lm_time_all += time.time() - lm_time_start
            model_time_start = time.time()
            with accelerator.accumulate(model):
                # Predict the noise residual
                optimizer.zero_grad()
                loss_term = train_diffusion.training_losses(
                    model, clean_images, timesteps, model_kwargs=dict(y=y, mask=y_mask, data_info=data_info)
                )
                loss = loss_term["loss"].mean()

                # Check if the loss is NaN
                if torch.isnan(loss):
                    loss_nan_timer += 1
                    logger.warning(f"Skip nan: {loss_nan_timer}")
                    continue  # Skip the rest of the loop iteration if loss is NaN

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    grad_norm = accelerator.clip_grad_norm_(model.parameters(), config.train.gradient_clip)

                model_time_all += time.time() - model_time_start
                optimizer_time_start = time.time()
                optimizer.step()
                optimizer_time_all += time.time() - optimizer_time_start
                lr_scheduler.step()
                accelerator.wait_for_everyone()

            lr = lr_scheduler.get_last_lr()[0]
            logs = {args.loss_report_name: accelerator.gather(loss).mean().item()}
            if grad_norm is not None:
                logs.update(grad_norm=accelerator.gather(grad_norm).mean().item())
            log_buffer.update(logs)
            if (step + 1) % config.train.log_interval == 0 or (step + 1) == 1:
                accelerator.wait_for_everyone()
                t = (time.time() - last_tic) / config.train.log_interval
                t_d = data_time_all / config.train.log_interval
                t_m = model_time_all / config.train.log_interval
                t_opt = optimizer_time_all / config.train.log_interval
                t_lm = lm_time_all / config.train.log_interval
                avg_time = (time.time() - time_start) / (step + 1)
                eta = str(datetime.timedelta(seconds=int(avg_time * (total_steps - global_step - 1))))
                eta_epoch = str(
                    datetime.timedelta(
                        seconds=int(avg_time * (len(dataset) - step // config.train.train_batch_size - step - 1))
                    )
                )
                log_buffer.average()

                current_step = (global_step - step // config.train.train_batch_size) % len(dataset)
                current_step = len(dataset) if current_step == 0 else current_step
                info = (
                    f"Epoch: {epoch} | Global Step: {global_step} | Local Step: {current_step} // {len(dataset)}, "
                    f"total_eta: {eta}, epoch_eta:{eta_epoch}, time: all:{t:.3f}, model:{t_m:.3f}, optimizer:{t_opt:.3f}, data:{t_d:.3f}, "
                    f"lm:{t_lm:.3f}, lr:{lr:.3e}, "
                )
                info += (
                    f"s:({model.module.h}, {model.module.w}), "
                    if hasattr(model, "module")
                    else f"s:({model.h}, {model.w}), "
                )

                info += ", ".join([f"{k}:{v:.4f}" for k, v in log_buffer.output.items()])
                last_tic = time.time()
                log_buffer.clear()
                data_time_all = 0
                model_time_all = 0
                optimizer_time_all = 0
                lm_time_all = 0
                if accelerator.is_main_process:
                    logger.info(info)

            logs.update(lr=lr)
            if accelerator.is_main_process:
                accelerator.log(logs, step=global_step)

            global_step += 1

            if loss_nan_timer > 20:
                raise ValueError("Loss is NaN too much times. Break here.")
            if global_step % config.train.save_model_steps == 0:
                save_model(config.train.online_metric and global_step % config.train.eval_metric_step == 0 and step > 1)

                # if (time.time() - training_start_time) / 3600 > 3.8:
                #     logger.info(f"Stopping training at epoch {epoch}, step {global_step} due to time limit.")
                #     return
            if config.train.visualize and (global_step % config.train.eval_sampling_steps == 0 or (step + 1) == 1):
                accelerator.wait_for_everyone()
                if accelerator.is_main_process:
                    if validation_noise is not None:
                        idation(
                            accelerator=accelerator,
                            config=config,
                            model=model,
                            step=global_step,
                            device=accelerator.device,
                            vae=vae,
                            init_noise=validation_noise,
                        )
                    else:
                        idation(
                            accelerator=accelerator,
                            config=config,
                            model=model,
                            step=global_step,
                            device=accelerator.device,
                            vae=vae,
                        )

            data_time_start = time.time()

        if epoch % config.train.save_model_epochs == 0 or epoch == config.train.num_epochs and not config.debug:
            save_model()
        accelerator.wait_for_everyone()
    save_model()


@pyrallis.wrap()
def main(cfg: SanaConfig) -> None:
    global start_epoch, start_step, vae, generator, num_replicas, rank, training_start_time
    global load_vae_feat, load_text_feat, validation_noise, text_encoder, tokenizer, logger
    global max_length, validation_prompts, latent_size, valid_prompt_embed_suffix, null_embed_path
    global image_size, total_steps

    config = cfg
    args = cfg
    # config = read_config(args.config)

    training_start_time = time.time()
    load_from = True
    if args.resume_from or config.model.resume_from:
        load_from = False
        config.model.resume_from = dict(
            checkpoint=args.resume_from or config.model.resume_from,
            load_ema=False,
            resume_optimizer=False,
            resume_lr_scheduler=False,
        )

    if args.debug:
        config.train.log_interval = 1
        config.train.train_batch_size = min(64, config.train.train_batch_size)
        args.report_to = "tensorboard"

    os.umask(0o000)
    os.makedirs(config.work_dir, exist_ok=True)

    init_handler = InitProcessGroupKwargs()
    init_handler.timeout = datetime.timedelta(seconds=5400)  # change timeout to avoid a strange NCCL bug
    # Initialize accelerator and tensorboard logging
    if config.train.use_fsdp:
        init_train = "FSDP"
        from accelerate import FullyShardedDataParallelPlugin
        from torch.distributed.fsdp.fully_sharded_data_parallel import FullStateDictConfig

        set_fsdp_env()
        fsdp_plugin = FullyShardedDataParallelPlugin(
            state_dict_config=FullStateDictConfig(offload_to_cpu=False, rank0_only=False),
        )
    else:
        init_train = "DDP"
        fsdp_plugin = None

    accelerator = Accelerator(
        mixed_precision=config.model.mixed_precision,
        gradient_accumulation_steps=config.train.gradient_accumulation_steps,
        log_with=args.report_to,
        project_dir=osp.join(config.work_dir, "logs"),
        fsdp_plugin=fsdp_plugin,
        kwargs_handlers=[init_handler],
    )

    log_name = "train_log.log"
    logger = get_root_logger(osp.join(config.work_dir, log_name))
    logger.info(accelerator.state)

    config.train.seed = init_random_seed(getattr(config.train, "seed", None))
    set_random_seed(config.train.seed + int(os.environ["LOCAL_RANK"]))
    generator = torch.Generator(device="cpu").manual_seed(config.train.seed)

    if accelerator.is_main_process:
        pyrallis.dump(config, open(osp.join(config.work_dir, "config.yaml"), "w"), sort_keys=False, indent=4)
        if args.report_to == "wandb":
            import wandb

            wandb.init(project=args.tracker_project_name, name=args.name, resume="allow", id=args.name)

    logger.info(f"Config: \n{config}")
    logger.info(f"World_size: {get_world_size()}, seed: {config.train.seed}")
    logger.info(f"Initializing: {init_train} for training")
    image_size = config.model.image_size
    latent_size = int(image_size) // config.vae.vae_downsample_rate
    pred_sigma = getattr(config.scheduler, "pred_sigma", True)
    learn_sigma = getattr(config.scheduler, "learn_sigma", True) and pred_sigma
    max_length = config.text_encoder.model_max_length
    vae = None
    validation_noise = (
        torch.randn(1, config.vae.vae_latent_dim, latent_size, latent_size, device="cpu", generator=generator)
        if getattr(config.train, "deterministic_validation", False)
        else None
    )

    tokenizer = text_encoder = None
    if not config.data.load_text_feat:
        tokenizer, text_encoder = get_tokenizer_and_text_encoder(
            name=config.text_encoder.text_encoder_name, device=accelerator.device
        )
        text_encoder.requires_grad_(False)
        text_embed_dim = text_encoder.config.hidden_size
    else:
        text_embed_dim = config.text_encoder.caption_channels

    logger.info(f"vae type: {config.vae.vae_type}")
    if config.text_encoder.chi_prompt:
        chi_prompt = "\n".join(config.text_encoder.chi_prompt)
        logger.info(f"Complex Human Instruct: {chi_prompt}")

    os.makedirs(config.train.null_embed_root, exist_ok=True)
    null_embed_path = osp.join(
        config.train.null_embed_root,
        f"null_embed_diffusers_{config.text_encoder.text_encoder_name}_{max_length}token_{text_embed_dim}.pth",
    )
    if config.train.visualize and len(config.train.validation_prompts):
        # preparing embeddings for visualization. We put it here for saving GPU memory
        valid_prompt_embed_suffix = f"{max_length}token_{config.text_encoder.text_encoder_name}_{text_embed_dim}.pth"
        validation_prompts = config.train.validation_prompts
        skip = True
        if config.text_encoder.chi_prompt:
            uuid_chi_prompt = hashlib.sha256(chi_prompt.encode()).hexdigest()
        else:
            uuid_chi_prompt = hashlib.sha256(b"").hexdigest()
        config.train.valid_prompt_embed_root = osp.join(config.train.valid_prompt_embed_root, uuid_chi_prompt)
        Path(config.train.valid_prompt_embed_root).mkdir(parents=True, exist_ok=True)

        if config.text_encoder.chi_prompt:
            # Save complex human instruct to a file
            chi_prompt_file = osp.join(config.train.valid_prompt_embed_root, "chi_prompt.txt")
            with open(chi_prompt_file, "w", encoding="utf-8") as f:
                f.write(chi_prompt)

        for prompt in validation_prompts:
            prompt_embed_path = osp.join(
                config.train.valid_prompt_embed_root, f"{prompt[:50]}_{valid_prompt_embed_suffix}"
            )
            if not (osp.exists(prompt_embed_path) and osp.exists(null_embed_path)):
                skip = False
                logger.info("Preparing Visualization prompt embeddings...")
                break
        if accelerator.is_main_process and not skip:
            if config.data.load_text_feat and (tokenizer is None or text_encoder is None):
                logger.info(f"Loading text encoder and tokenizer from {config.text_encoder.text_encoder_name} ...")
                tokenizer, text_encoder = get_tokenizer_and_text_encoder(name=config.text_encoder.text_encoder_name)

            for prompt in validation_prompts:
                prompt_embed_path = osp.join(
                    config.train.valid_prompt_embed_root, f"{prompt[:50]}_{valid_prompt_embed_suffix}"
                )
                if "T5" in config.text_encoder.text_encoder_name:
                    txt_tokens = tokenizer(
                        prompt, max_length=max_length, padding="max_length", truncation=True, return_tensors="pt"
                    ).to(accelerator.device)
                    caption_emb = text_encoder(txt_tokens.input_ids, attention_mask=txt_tokens.attention_mask)[0]
                    caption_emb_mask = txt_tokens.attention_mask
                elif (
                        "gemma" in config.text_encoder.text_encoder_name or "Qwen" in config.text_encoder.text_encoder_name
                ):
                    if not config.text_encoder.chi_prompt:
                        max_length_all = config.text_encoder.model_max_length
                    else:
                        chi_prompt = "\n".join(config.text_encoder.chi_prompt)
                        prompt = chi_prompt + prompt
                        num_chi_prompt_tokens = len(tokenizer.encode(chi_prompt))
                        max_length_all = (
                                num_chi_prompt_tokens + config.text_encoder.model_max_length - 2
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
                else:
                    raise ValueError(f"{config.text_encoder.text_encoder_name} is not supported!!")

                torch.save({"caption_embeds": caption_emb, "emb_mask": caption_emb_mask}, prompt_embed_path)

            null_tokens = tokenizer(
                "bad artwork,ugly,sketch,poorly drawn,messy,noisy,score: 0/10,blurry,low quality,old",
                max_length=max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            ).to(accelerator.device)
            if "T5" in config.text_encoder.text_encoder_name:
                null_token_emb = text_encoder(null_tokens.input_ids, attention_mask=null_tokens.attention_mask)[0]
            elif "gemma" in config.text_encoder.text_encoder_name or "Qwen" in config.text_encoder.text_encoder_name:
                null_token_emb = text_encoder(null_tokens.input_ids, attention_mask=null_tokens.attention_mask)[0]
            else:
                raise ValueError(f"{config.text_encoder.text_encoder_name} is not supported!!")
            torch.save(
                {"uncond_prompt_embeds": null_token_emb, "uncond_prompt_embeds_mask": null_tokens.attention_mask},
                null_embed_path,
            )
            if config.data.load_text_feat:
                del tokenizer
                del text_encoder
            del null_token_emb
            del null_tokens
            flush()

    os.environ["AUTOCAST_LINEAR_ATTN"] = "true" if config.model.autocast_linear_attn else "false"

    # 1. build scheduler
    train_diffusion = Scheduler(
        str(config.scheduler.train_sampling_steps),
        noise_schedule=config.scheduler.noise_schedule,
        predict_v=config.scheduler.predict_v,
        learn_sigma=learn_sigma,
        pred_sigma=pred_sigma,
        snr=config.train.snr_loss,
        flow_shift=config.scheduler.flow_shift,
    )
    predict_info = f"v-prediction: {config.scheduler.predict_v}, noise schedule: {config.scheduler.noise_schedule}"
    if "flow" in config.scheduler.noise_schedule:
        predict_info += f", flow shift: {config.scheduler.flow_shift}"
    if config.scheduler.weighting_scheme in ["logit_normal", "mode"]:
        predict_info += (
            f", flow weighting: {config.scheduler.weighting_scheme}, "
            f"logit-mean: {config.scheduler.logit_mean}, logit-std: {config.scheduler.logit_std}"
        )
    logger.info(predict_info)

    # 2. build models
    model_kwargs = {
        "pe_interpolation": config.model.pe_interpolation,
        "config": config,
        "model_max_length": max_length,
        "qk_norm": config.model.qk_norm,
        "micro_condition": config.model.micro_condition,
        "caption_channels": text_embed_dim,
        "y_norm": config.text_encoder.y_norm,
        "attn_type": config.model.attn_type,
        "ffn_type": config.model.ffn_type,
        "mlp_ratio": config.model.mlp_ratio,
        "mlp_acts": list(config.model.mlp_acts),
        "in_channels": config.vae.vae_latent_dim,
        "y_norm_scale_factor": config.text_encoder.y_norm_scale_factor,
        "use_pe": config.model.use_pe,
        "linear_head_dim": config.model.linear_head_dim,
        "pred_sigma": pred_sigma,
        "learn_sigma": learn_sigma,
    }
    model = build_model(
        config.model.model,
        config.train.grad_checkpointing,
        getattr(config.model, "fp32_attention", False),
        input_size=latent_size,
        **model_kwargs,
    ).train()

    logger.info(
        colored(
            f"{model.__class__.__name__}:{config.model.model}, "
            f"Model Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M",
            "green",
            attrs=["bold"],
        )
    )
    # 2-1. load model
    if args.load_from is not None:
        config.model.load_from = args.load_from
    if config.model.load_from is not None and load_from:
        _, missing, unexpected, _ = load_checkpoint(
            config.model.load_from,
            model,
            load_ema=config.model.resume_from.get("load_ema", False),
            null_embed_path=null_embed_path,
        )
        if missing:
            logger.warning(f"Missing keys: {missing}")
        if unexpected:
            logger.warning(f"Unexpected keys: {unexpected}")

    # prepare for FSDP clip grad norm calculation
    if accelerator.distributed_type == DistributedType.FSDP:
        for m in accelerator._models:
            m.clip_grad_norm_ = types.MethodType(clip_grad_norm_, m)

    # 3. build dataloader
    config.data.data_dir = config.data.data_dir if isinstance(config.data.data_dir, list) else [config.data.data_dir]
    config.data.data_dir = [
        data if data.startswith(("https://", "http://", "gs://", "/", "~")) else osp.abspath(osp.expanduser(data))
        for data in config.data.data_dir
    ]
    num_replicas = int(os.environ["WORLD_SIZE"])
    rank = int(os.environ["RANK"])
    dataset = RatioBucketsDataset(config.data.buckets_file)
    accelerator.wait_for_everyone()
    dataset.make_loaders(batch_size=config.train.train_batch_size)

    load_vae_feat = getattr(dataset, "load_vae_feat", False)
    load_text_feat = getattr(dataset, "load_text_feat", False)

    optimizer = build_optimizer(model, config.train.optimizer)

    if config.train.lr_schedule_args and config.train.lr_schedule_args.get("num_warmup_steps", None):
        config.train.lr_schedule_args["num_warmup_steps"] = (
                config.train.lr_schedule_args["num_warmup_steps"] * num_replicas
        )
    lr_scheduler = build_lr_scheduler(config.train, optimizer, dataset, 1)

    logger.warning(
        f"{colored(f'Basic Setting: ', 'green', attrs=['bold'])}"
        f"lr: {config.train.optimizer['lr']:.9f}, bs: {config.train.train_batch_size}, gc: {config.train.grad_checkpointing}, "
        f"gc_accum_step: {config.train.gradient_accumulation_steps}, qk norm: {config.model.qk_norm}, "
        f"fp32 attn: {config.model.fp32_attention}, attn type: {config.model.attn_type}, ffn type: {config.model.ffn_type}, "
        f"text encoder: {config.text_encoder.text_encoder_name}, captions: {config.data.caption_proportion}, precision: {config.model.mixed_precision}"
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
    total_steps = len(dataset) * config.train.num_epochs

    # Resume training
    if config.model.resume_from is not None and config.model.resume_from["checkpoint"] is not None:
        rng_state = None
        ckpt_path = osp.join(config.work_dir, "checkpoints")
        check_flag = osp.exists(ckpt_path) and len(os.listdir(ckpt_path)) != 0
        if config.model.resume_from["checkpoint"] == "latest":
            if check_flag:
                checkpoints = os.listdir(ckpt_path)
                if "latest.pth" in checkpoints and osp.exists(osp.join(ckpt_path, "latest.pth")):
                    config.model.resume_from["checkpoint"] = osp.realpath(osp.join(ckpt_path, "latest.pth"))
                else:
                    checkpoints = [i for i in checkpoints if i.startswith("epoch_")]
                    checkpoints = sorted(checkpoints, key=lambda x: int(x.replace(".pth", "").split("_")[3]))
                    config.model.resume_from["checkpoint"] = osp.join(ckpt_path, checkpoints[-1])
            else:
                config.model.resume_from["checkpoint"] = config.model.load_from

        if config.model.resume_from["checkpoint"] is not None:
            _, missing, unexpected, rng_state = load_checkpoint(
                **config.model.resume_from,
                model=model,
                optimizer=optimizer if check_flag else None,
                lr_scheduler=lr_scheduler if check_flag else None,
                null_embed_path=null_embed_path,
            )

            if missing:
                logger.warning(f"Missing keys: {missing}")
            if unexpected:
                logger.warning(f"Unexpected keys: {unexpected}")

            path = osp.basename(config.model.resume_from["checkpoint"])
        try:
            start_epoch = int(path.replace(".pth", "").split("_")[1]) - 1
            start_step = int(path.replace(".pth", "").split("_")[3])
        except:
            pass

        # resume randomise
        if rng_state:
            logger.info("resuming randomise")
            torch.set_rng_state(rng_state["torch"])
            torch.cuda.set_rng_state_all(rng_state["torch_cuda"])
            np.random.set_state(rng_state["numpy"])
            random.setstate(rng_state["python"])
            generator.set_state(rng_state["generator"])  # resume generator status

    # Prepare everything
    # There is no specific order to remember, you just need to unpack the
    # objects in the same order you gave them to the prepare method.
    model = accelerator.prepare(model)
    optimizer, lr_scheduler = accelerator.prepare(optimizer, lr_scheduler)

    train(
        config=config,
        args=args,
        accelerator=accelerator,
        model=model,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        dataset=dataset,
        train_diffusion=train_diffusion
    )


if __name__ == "__main__":
    main()
