# Copyright 2026 NVIDIA CORPORATION & AFFILIATES
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

"""Minimal, production training entry point for Sana-Video 2.0."""

import argparse
import datetime
import os
import os.path as osp
import time
from dataclasses import asdict
from pathlib import Path

import pyrallis
import torch
from accelerate import Accelerator, InitProcessGroupKwargs
from termcolor import colored

from diffusion import Scheduler
from diffusion.data.builder import build_dataloader, build_dataset
from diffusion.data.wids import DistributedRangedSampler
from diffusion.model.builder import build_model, get_tokenizer_and_text_encoder, get_vae, vae_encode
from diffusion.model.respace import compute_density_for_timestep_sampling
from diffusion.model.utils import get_weight_dtype
from diffusion.utils.checkpoint import save_checkpoint
from diffusion.utils.config import SanaVideoConfig, model_video_init_config, validate_sana_video2_config
from diffusion.utils.logger import get_root_logger
from diffusion.utils.lr_scheduler import build_lr_scheduler
from diffusion.utils.misc import set_random_seed
from diffusion.utils.optimizer import build_optimizer
from tools.download import find_model

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def configure_fsdp() -> None:
    """Configure Accelerate FSDP for the released transformer block."""
    os.environ["ACCELERATE_USE_FSDP"] = "true"
    os.environ["FSDP_AUTO_WRAP_POLICY"] = "TRANSFORMER_BASED_WRAP"
    os.environ["FSDP_TRANSFORMER_CLS_TO_WRAP"] = "SanaVideo2Block"
    os.environ["FSDP_BACKWARD_PREFETCH"] = "BACKWARD_PRE"
    os.environ["FSDP_FORWARD_PREFETCH"] = "false"
    os.environ["FSDP_STATE_DICT_TYPE"] = "FULL_STATE_DICT"
    os.environ["FSDP_SYNC_MODULE_STATES"] = "true"
    os.environ["FSDP_USE_ORIG_PARAMS"] = "true"
    os.environ["FSDP_SHARDING_STRATEGY"] = "FULL_SHARD"
    os.environ["FSDP_CPU_RAM_EFFICIENT_LOADING"] = "false"
    os.environ["FSDP_OFFLOAD_PARAMS"] = "false"


def normalize_state_dict(checkpoint: dict) -> dict[str, torch.Tensor]:
    """Extract and normalize a model state dict from supported checkpoint layouts."""
    if "generator" in checkpoint:
        checkpoint = checkpoint["generator"]
    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    elif "state_dict_ema" in checkpoint:
        state_dict = checkpoint["state_dict_ema"]
    else:
        state_dict = checkpoint

    normalized = {}
    for key, value in state_dict.items():
        for prefix in ("module.", "model."):
            if key.startswith(prefix):
                key = key[len(prefix) :]
        if key != "pos_embed":
            normalized[key] = value
    return normalized


def load_initial_weights(model: torch.nn.Module, checkpoint_path: str, logger) -> None:
    """Load an initialization checkpoint and reject architecture mismatches."""
    state_dict = normalize_state_dict(find_model(checkpoint_path))
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    missing = [key for key in missing if key != "pos_embed"]
    if missing or unexpected:
        raise RuntimeError(
            f"Checkpoint does not match Sana-Video 2.0. Missing keys: {missing}; " f"unexpected keys: {unexpected}."
        )
    logger.info(f"Loaded initialization weights from {checkpoint_path}")


def encode_captions(
    captions: list[str],
    config: SanaVideoConfig,
    tokenizer,
    text_encoder,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode Gemma captions using the same CHI token selection as inference."""
    if "gemma" not in config.text_encoder.text_encoder_name.lower():
        raise ValueError("The Sana-Video 2.0 release recipe supports the Gemma 2 text encoder.")

    if config.text_encoder.chi_prompt:
        prefix = "\n".join(config.text_encoder.chi_prompt)
        prompts = [prefix + caption for caption in captions]
        max_length = len(tokenizer.encode(prefix)) + config.text_encoder.model_max_length - 2
    else:
        prompts = captions
        max_length = config.text_encoder.model_max_length

    tokens = tokenizer(
        prompts,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    ).to(device)
    selected = [0] + list(range(-config.text_encoder.model_max_length + 1, 0))
    with torch.no_grad():
        embeddings = text_encoder(tokens.input_ids, attention_mask=tokens.attention_mask)[0]
    return embeddings[:, None, selected], tokens.attention_mask[:, None, None, selected]


def encode_null_caption(
    config: SanaVideoConfig,
    tokenizer,
    text_encoder,
    device: torch.device,
) -> torch.Tensor:
    """Encode the empty prompt used by classifier-free training dropout."""
    tokens = tokenizer(
        "",
        max_length=config.text_encoder.model_max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    ).to(device)
    with torch.no_grad():
        return text_encoder(tokens.input_ids, attention_mask=tokens.attention_mask)[0][0]


def sample_timesteps(config: SanaVideoConfig, batch_size: int, device: torch.device) -> torch.Tensor:
    """Sample the scalar flow timestep distribution configured for training."""
    steps = config.scheduler.train_sampling_steps
    if config.scheduler.weighting_scheme in {"logit_normal", "mode"}:
        density = compute_density_for_timestep_sampling(
            weighting_scheme=config.scheduler.weighting_scheme,
            batch_size=batch_size,
            logit_mean=config.scheduler.logit_mean,
            logit_std=config.scheduler.logit_std,
            mode_scale=config.scheduler.mode_scale,
        )
        return (density * steps).long().clamp_(max=steps - 1).to(device)
    if config.scheduler.weighting_scheme not in {None, "uniform"}:
        raise ValueError(f"Unsupported weighting scheme: {config.scheduler.weighting_scheme!r}.")
    return torch.randint(0, steps, (batch_size,), device=device)


def save_training_checkpoint(
    config: SanaVideoConfig,
    accelerator: Accelerator,
    model: torch.nn.Module,
    optimizer,
    lr_scheduler,
    epoch: int,
    global_step: int,
    next_epoch: int,
    next_batch: int,
) -> None:
    """Save a resumable checkpoint and an inference-ready merged state dict."""
    accelerator.wait_for_everyone()
    checkpoint_root = osp.join(config.work_dir, "checkpoints")
    if config.train.use_fsdp:
        save_checkpoint(
            work_dir=checkpoint_root,
            epoch=epoch,
            model=model,
            accelerator=accelerator,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            step=global_step,
            saved_info={
                "global_step": global_step,
                "next_epoch": next_epoch,
                "next_batch": next_batch,
            },
            add_symlink=True,
        )
    elif accelerator.is_main_process:
        save_checkpoint(
            work_dir=checkpoint_root,
            epoch=epoch,
            model=accelerator.unwrap_model(model),
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            step=global_step,
            saved_info={
                "global_step": global_step,
                "next_epoch": next_epoch,
                "next_batch": next_batch,
            },
            add_symlink=True,
        )
    accelerator.wait_for_everyone()


def resume_training(
    checkpoint_path: str,
    accelerator: Accelerator,
    model: torch.nn.Module,
    optimizer,
    lr_scheduler,
) -> tuple[int, int, int]:
    """Restore model/optimizer state and return the next epoch, batch, and step."""
    checkpoint = Path(checkpoint_path)
    if checkpoint.is_dir():
        state_dir = checkpoint / "model" if (checkpoint / "model").is_dir() else checkpoint
        accelerator.load_state(str(state_dir))
        metadata_path = checkpoint / "metadata.pth"
        metadata = torch.load(metadata_path, map_location="cpu") if metadata_path.is_file() else {}
    else:
        if accelerator.distributed_type.name == "FSDP":
            raise ValueError("FSDP resume requires the checkpoint directory, not the merged .pth file.")
        metadata = find_model(checkpoint_path)
        state_dict = normalize_state_dict(metadata)
        missing, unexpected = accelerator.unwrap_model(model).load_state_dict(state_dict, strict=False)
        missing = [key for key in missing if key != "pos_embed"]
        if missing or unexpected:
            raise RuntimeError(f"Resume checkpoint mismatch: missing={missing}, unexpected={unexpected}.")
        if "optimizer" in metadata:
            optimizer.load_state_dict(metadata["optimizer"])
        if "scheduler" in metadata:
            lr_scheduler.load_state_dict(metadata["scheduler"])

    return (
        int(metadata.get("next_epoch", metadata.get("epoch", 1))),
        int(metadata.get("next_batch", 0)),
        int(metadata.get("global_step", metadata.get("step", 0))),
    )


def build_video_dataloader(config: SanaVideoConfig, accelerator: Accelerator):
    """Build the single video-only data path used by the public trainer."""
    if config.model.multi_scale:
        raise ValueError(
            "train_sana_video2.py intentionally exposes the fixed-resolution recipe only; "
            "set model.multi_scale=false."
        )

    data_dir = config.data.data_dir
    if not isinstance(data_dir, dict):
        data_dir = {"default": data_dir}
    config.data.data_dir = {
        name: path if path.startswith(("http://", "https://", "gs://", "/", "~")) else osp.abspath(path)
        for name, path in data_dir.items()
    }
    dataset = build_dataset(
        asdict(config.data),
        resolution=config.data.image_size,
        max_length=config.text_encoder.model_max_length,
        config=config,
        caption_proportion=config.data.caption_proportion,
        sort_dataset=config.data.sort_dataset,
        vae_downsample_rate=config.vae.vae_stride[-1],
        num_frames=config.data.num_frames,
    )
    sampler = DistributedRangedSampler(
        dataset,
        num_replicas=accelerator.num_processes,
        rank=accelerator.process_index,
    )
    dataloader = build_dataloader(
        dataset,
        num_workers=config.train.num_workers,
        batch_size=config.train.train_batch_size,
        shuffle=False,
        sampler=sampler,
        dataloader_type="video",
    )
    return dataloader


def train(cfg: SanaVideoConfig) -> None:
    """Train Sana-Video 2.0 from a validated YAML config."""
    validate_sana_video2_config(cfg)
    if cfg.task != "t2v":
        raise ValueError(f"The public Sana-Video 2.0 trainer supports task='t2v', got {cfg.task!r}.")
    if cfg.train.use_fsdp:
        configure_fsdp()

    process_group = InitProcessGroupKwargs(timeout=datetime.timedelta(minutes=90))
    log_with = None if cfg.report_to.lower() in {"", "none"} else cfg.report_to
    accelerator = Accelerator(
        mixed_precision=cfg.model.mixed_precision,
        gradient_accumulation_steps=cfg.train.gradient_accumulation_steps,
        log_with=log_with,
        kwargs_handlers=[process_group],
    )
    set_random_seed(cfg.train.seed)
    Path(cfg.work_dir).mkdir(parents=True, exist_ok=True)
    logger = get_root_logger(osp.join(cfg.work_dir, "train_log.log"))

    if cfg.data.load_text_feat or cfg.data.load_vae_feat:
        raise ValueError("The public Sana-Video 2.0 trainer uses online text and VAE encoding.")
    dataloader = build_video_dataloader(cfg, accelerator)

    tokenizer, text_encoder = get_tokenizer_and_text_encoder(
        name=cfg.text_encoder.text_encoder_name,
        device=accelerator.device,
    )
    text_encoder.eval().requires_grad_(False)
    vae_dtype = get_weight_dtype(cfg.vae.weight_dtype)
    vae = get_vae(
        cfg.vae.vae_type,
        cfg.vae.vae_pretrained,
        device=accelerator.device,
        dtype=vae_dtype,
        config=cfg.vae,
    )
    vae.eval().requires_grad_(False)

    latent_size = cfg.model.image_size // cfg.vae.vae_stride[-1]
    model = build_model(
        cfg.model.model,
        use_grad_checkpoint=cfg.train.grad_checkpointing,
        use_fp32_attention=cfg.model.fp32_attention,
        **model_video_init_config(cfg, latent_size=latent_size),
    ).train()
    null_caption = encode_null_caption(cfg, tokenizer, text_encoder, accelerator.device)
    with torch.no_grad():
        model.y_embedder.y_embedding.copy_(
            null_caption.to(
                device=model.y_embedder.y_embedding.device,
                dtype=model.y_embedder.y_embedding.dtype,
            )
        )
    if cfg.model.load_from:
        load_initial_weights(model, cfg.model.load_from, logger)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())

    diffusion = Scheduler(
        str(cfg.scheduler.train_sampling_steps),
        noise_schedule=cfg.scheduler.noise_schedule,
        predict_flow_v=cfg.scheduler.predict_flow_v,
        learn_sigma=bool(cfg.scheduler.learn_sigma and cfg.scheduler.pred_sigma),
        pred_sigma=cfg.scheduler.pred_sigma,
        snr=cfg.train.snr_loss,
        flow_shift=cfg.scheduler.flow_shift,
    )
    optimizer = build_optimizer(model, cfg.train.optimizer)
    lr_scheduler = build_lr_scheduler(cfg.train, optimizer, dataloader, lr_scale_ratio=1.0)
    model, optimizer, lr_scheduler = accelerator.prepare(model, optimizer, lr_scheduler)

    if accelerator.is_main_process and log_with is not None:
        accelerator.init_trackers(cfg.tracker_project_name, config=asdict(cfg))

    logger.info(
        colored(
            f"Training {cfg.model.model}: {parameter_count:,} parameters, {len(dataloader)} batches/epoch",
            "green",
        )
    )

    next_epoch, next_batch, global_step = 1, 0, 0
    if cfg.resume_from:
        next_epoch, next_batch, global_step = resume_training(
            cfg.resume_from,
            accelerator,
            model,
            optimizer,
            lr_scheduler,
        )
        logger.info(
            f"Resumed {cfg.resume_from}: next_epoch={next_epoch}, "
            f"next_batch={next_batch}, global_step={global_step}"
        )

    training_start = time.time()
    for epoch in range(next_epoch, cfg.train.num_epochs + 1):
        if hasattr(dataloader.sampler, "set_epoch"):
            dataloader.sampler.set_epoch(epoch)

        for batch_index, batch in enumerate(dataloader):
            if epoch == next_epoch and batch_index < next_batch:
                continue
            with torch.no_grad():
                clean_latents = vae_encode(
                    cfg.vae.vae_type,
                    vae,
                    batch[0].permute(0, 2, 1, 3, 4).to(dtype=vae_dtype),
                    sample_posterior=cfg.vae.sample_posterior,
                    device=accelerator.device,
                )
                caption_embeddings, caption_mask = encode_captions(
                    batch[1],
                    cfg,
                    tokenizer,
                    text_encoder,
                    accelerator.device,
                )

            timesteps = sample_timesteps(cfg, clean_latents.shape[0], clean_latents.device)
            with accelerator.accumulate(model):
                loss_terms = diffusion.training_losses(
                    model,
                    clean_latents,
                    timesteps,
                    model_kwargs={
                        "y": caption_embeddings,
                        "mask": caption_mask,
                    },
                )
                loss = loss_terms["loss"].mean()
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Non-finite training loss at step {global_step}: {loss.item()}")
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), cfg.train.gradient_clip)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            if not accelerator.sync_gradients:
                continue
            global_step += 1
            reduced_loss = accelerator.gather(loss.detach()).mean().item()
            if global_step == 1 or global_step % cfg.train.log_interval == 0:
                elapsed = time.time() - training_start
                logger.info(
                    f"epoch={epoch} step={global_step} loss={reduced_loss:.6f} "
                    f"lr={lr_scheduler.get_last_lr()[0]:.3e} elapsed={elapsed / 3600:.2f}h"
                )
                if log_with is not None:
                    accelerator.log(
                        {"train/loss": reduced_loss, "train/lr": lr_scheduler.get_last_lr()[0]},
                        step=global_step,
                    )

            if global_step % cfg.train.save_model_steps == 0:
                following_epoch = epoch
                following_batch = batch_index + 1
                if following_batch >= len(dataloader):
                    following_epoch, following_batch = epoch + 1, 0
                save_training_checkpoint(
                    cfg,
                    accelerator,
                    model,
                    optimizer,
                    lr_scheduler,
                    epoch,
                    global_step,
                    following_epoch,
                    following_batch,
                )

        next_batch = 0
        if epoch % cfg.train.save_model_epochs == 0:
            save_training_checkpoint(
                cfg,
                accelerator,
                model,
                optimizer,
                lr_scheduler,
                epoch,
                global_step,
                epoch + 1,
                0,
            )

    accelerator.end_training()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to a Sana-Video 2.0 YAML config.")
    parser.add_argument("--resume-from", help="Optional checkpoint directory or merged .pth file.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with open(args.config, encoding="utf-8") as config_file:
        config = pyrallis.load(SanaVideoConfig, config_file)
    if args.resume_from:
        config.resume_from = args.resume_from
    train(config)


if __name__ == "__main__":
    main()
