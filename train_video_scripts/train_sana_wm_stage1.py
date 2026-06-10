# Copyright 2024 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# SPDX-License-Identifier: Apache-2.0

import datetime
import os
import os.path as osp
import random
import time
from dataclasses import asdict, fields

os.environ.setdefault("DISABLE_XFORMERS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import pyrallis
import torch
from accelerate import Accelerator, InitProcessGroupKwargs
from accelerate.utils import FullyShardedDataParallelPlugin
from torch.utils.data import DataLoader, DistributedSampler

from diffusion import Scheduler
from diffusion.data.builder import build_dataset
from diffusion.model.builder import build_model, get_tokenizer_and_text_encoder
from diffusion.model.respace import IncrementalTimesteps, process_timesteps
from diffusion.model.utils import get_weight_dtype
from diffusion.utils.camctrl_config import SanaVideoCamCtrlConfig, model_video_camctrl_init_config
from diffusion.utils.checkpoint import load_checkpoint, save_checkpoint
from diffusion.utils.chunk_utils import chunk_index_from_chunk_size
from diffusion.utils.config import model_video_init_config
from diffusion.utils.dist_utils import dist, get_world_size
from diffusion.utils.logger import LogBuffer, get_root_logger
from diffusion.utils.lr_scheduler import build_lr_scheduler
from diffusion.utils.misc import init_random_seed, set_random_seed
from diffusion.utils.optimizer import auto_scale_lr, build_optimizer


def _cfg_get(obj, key: str, default=None):
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _parse_bool(value, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    return default


def _resolve_cp_size(config: SanaVideoCamCtrlConfig) -> int:
    cp_size = int(getattr(config.train, "cp_size", 0) or 0)
    if cp_size <= 1:
        env_cp_size = os.environ.get("SANA_CP_SIZE")
        if env_cp_size is not None:
            try:
                cp_size = int(env_cp_size)
            except ValueError:
                cp_size = 1
    return max(1, cp_size)


def _resolve_cp_runtime_config(config: SanaVideoCamCtrlConfig) -> object:
    from diffusion.distributed.context_parallel.config import CpRuntimeConfig

    cp_cfg = _cfg_get(getattr(config.train, "extra", None), "cp", None)
    values = {field.name: _cfg_get(cp_cfg, field.name, None) for field in fields(CpRuntimeConfig)}
    if values["triton_block_fusion"] is None:
        values["triton_block_fusion"] = True
    return CpRuntimeConfig(**values)


def _resolve_fsdp2_runtime_knobs(config: SanaVideoCamCtrlConfig) -> dict:
    cp_cfg = _cfg_get(getattr(config.train, "extra", None), "cp", None)
    fsdp2_cfg = _cfg_get(cp_cfg, "fsdp2", None)
    return {
        "reshard_after_forward": _parse_bool(
            _cfg_get(fsdp2_cfg, "reshard_after_forward", os.environ.get("SANA_FSDP2_RESHARD_AFTER_FORWARD")),
            False,
        ),
        "limit_all_gathers": _parse_bool(
            _cfg_get(fsdp2_cfg, "limit_all_gathers", os.environ.get("SANA_FSDP2_LIMIT_ALL_GATHERS")),
            False,
        ),
        "backward_prefetch": str(
            _cfg_get(fsdp2_cfg, "backward_prefetch", os.environ.get("SANA_FSDP2_BACKWARD_PREFETCH", "backward_pre"))
        ).lower(),
    }


def _build_fsdp2_plugin(config: SanaVideoCamCtrlConfig, cp_size: int, logger):
    from torch.distributed.fsdp import BackwardPrefetch, MixedPrecisionPolicy

    transformer_block_class = "SanaVideoMSCamCtrlBlock" if "CamCtrl" in config.model.model else "SanaVideoMSBlock"
    knobs = _resolve_fsdp2_runtime_knobs(config)
    backward_prefetch = None
    if knobs["backward_prefetch"] in {"backward_pre", "pre"}:
        backward_prefetch = BackwardPrefetch.BACKWARD_PRE
    elif knobs["backward_prefetch"] in {"backward_post", "post"}:
        backward_prefetch = BackwardPrefetch.BACKWARD_POST

    if int(os.environ.get("RANK", "0")) == 0:
        logger.info(
            "[FSDP2] "
            f"block={transformer_block_class}, cp_size={cp_size}, "
            f"reshard_after_forward={knobs['reshard_after_forward']}, "
            f"limit_all_gathers={knobs['limit_all_gathers']}, "
            f"backward_prefetch={knobs['backward_prefetch']}"
        )

    return FullyShardedDataParallelPlugin(
        fsdp_version=2,
        reshard_after_forward=knobs["reshard_after_forward"],
        auto_wrap_policy="TRANSFORMER_BASED_WRAP",
        transformer_cls_names_to_wrap=[transformer_block_class],
        mixed_precision_policy=MixedPrecisionPolicy(
            param_dtype=torch.bfloat16 if config.model.mixed_precision == "bf16" else torch.float32,
            reduce_dtype=torch.float32,
        ),
        limit_all_gathers=knobs["limit_all_gathers"],
        forward_prefetch=None,
        backward_prefetch=backward_prefetch,
        activation_checkpointing=bool(config.train.grad_checkpointing),
        state_dict_type="SHARDED_STATE_DICT",
    )


def _build_parallelism_config(config: SanaVideoCamCtrlConfig, cp_size: int):
    fsdp_version = int(getattr(config.train, "fsdp_version", 1) or os.environ.get("SANA_FSDP_VERSION", "1"))
    if fsdp_version != 2 or cp_size <= 1:
        return None
    from accelerate.utils import ParallelismConfig

    world = int(os.environ.get("WORLD_SIZE", "1"))
    if world % cp_size != 0:
        raise ValueError(f"WORLD_SIZE={world} must be divisible by cp_size={cp_size}.")
    return ParallelismConfig(dp_shard_size=world // cp_size, cp_size=cp_size)


def _register_cp_group(accelerator: Accelerator, cp_size: int, logger) -> None:
    if cp_size <= 1:
        return
    from diffusion.distributed.context_parallel import set_cp_group
    from diffusion.distributed.context_parallel.config import init_context_parallel

    if accelerator.torch_device_mesh is not None:
        cp_group = accelerator.torch_device_mesh["cp"].get_group()
        set_cp_group(cp_group)
        logger.info(
            f"CP group registered from Accelerate device mesh: "
            f"cp_rank={dist.get_rank(cp_group)}, cp_world={dist.get_world_size(cp_group)}"
        )
        return

    init_context_parallel(cp_size)
    logger.info(f"CP group registered manually: cp_size={cp_size}")


def _resolve_under_root(path: str, root: str) -> str:
    path = osp.expanduser(str(path))
    if osp.isabs(path):
        return path
    return osp.abspath(osp.join(root, path))


def _resolve_data_dirs_under_root(data_dir, root: str):
    if isinstance(data_dir, dict):
        return {name: _resolve_under_root(path, root) for name, path in data_dir.items()}
    if isinstance(data_dir, str):
        return _resolve_under_root(data_dir, root)
    return [_resolve_under_root(path, root) for path in data_dir or []]


def _prepare_hf_dataset(config: SanaVideoCamCtrlConfig, accelerator: Accelerator, logger) -> None:
    repo_id = getattr(config.data, "hf_dataset_repo", None)
    if not repo_id:
        return

    local_dir = osp.abspath(osp.expanduser(getattr(config.data, "hf_dataset_local_dir", ".") or "."))
    revision = getattr(config.data, "hf_dataset_revision", None)
    allow_patterns = getattr(config.data, "hf_dataset_allow_patterns", None)
    if isinstance(allow_patterns, str):
        allow_patterns = [allow_patterns]

    if accelerator.is_main_process:
        from huggingface_hub import snapshot_download

        logger.info(
            "Preparing HF dataset "
            f"repo_id={repo_id}, revision={revision}, local_dir={local_dir}, allow_patterns={allow_patterns}"
        )
        snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            revision=revision,
            local_dir=local_dir,
            allow_patterns=allow_patterns,
        )
    accelerator.wait_for_everyone()

    config.data.data_dir = _resolve_data_dirs_under_root(config.data.data_dir, local_dir)
    if config.data.vae_cache_dir:
        config.data.vae_cache_dir = _resolve_under_root(config.data.vae_cache_dir, local_dir)


def _remove_custom_module_call(model, logger) -> None:
    import torch.nn as nn

    for klass in type(model).__mro__:
        if klass is nn.Module:
            break
        if "__call__" in klass.__dict__:
            logger.info(f"FSDP2: removing custom __call__ from {klass.__name__}")
            delattr(klass, "__call__")


def _build_dataloader(config: SanaVideoCamCtrlConfig, max_length: int, dp_size: int, dp_rank: int) -> DataLoader:
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
    sampler = None
    if dp_size > 1:
        sampler = DistributedSampler(
            dataset,
            num_replicas=dp_size,
            rank=dp_rank,
            shuffle=bool(config.data.shuffle_dataset),
            drop_last=True,
        )

    kwargs = {}
    if int(config.train.num_workers) > 0 and getattr(config.train, "prefetch_factor", None):
        kwargs["prefetch_factor"] = int(config.train.prefetch_factor)

    return DataLoader(
        dataset,
        batch_size=config.train.train_batch_size,
        sampler=sampler,
        shuffle=sampler is None and bool(config.data.shuffle_dataset),
        num_workers=config.train.num_workers,
        pin_memory=True,
        persistent_workers=config.train.num_workers > 0,
        drop_last=True,
        **kwargs,
    )


def _right_pad_temporal_with_last_frame(tensor: torch.Tensor, dim: int, pad_size: int) -> torch.Tensor:
    if pad_size <= 0:
        return tensor
    index = [slice(None)] * tensor.ndim
    index[dim] = tensor.shape[dim] - 1
    last_frame = tensor[tuple(index)].unsqueeze(dim)
    repeat = [1] * tensor.ndim
    repeat[dim] = pad_size
    return torch.cat([tensor, last_frame.repeat(*repeat)], dim=dim)


def _map_global_chunk_index_to_local(
    chunk_index: list[int] | None,
    global_num_frames: int,
    local_start: int,
    local_end: int,
) -> list[int] | None:
    if chunk_index is None:
        return None
    starts = sorted({int(idx) for idx in chunk_index if 0 <= int(idx) < global_num_frames})
    boundaries = [0] + [idx for idx in starts if idx > 0]
    if boundaries[-1] != global_num_frames:
        boundaries.append(global_num_frames)

    local_starts = []
    for seg_start, seg_end in zip(boundaries[:-1], boundaries[1:]):
        inter_start = max(seg_start, local_start)
        inter_end = min(seg_end, local_end)
        if inter_end <= inter_start:
            continue
        local_idx = inter_start - local_start
        if not local_starts or local_starts[-1] != local_idx:
            local_starts.append(local_idx)
    if not local_starts or local_starts[0] != 0:
        local_starts = [0] + local_starts
    return local_starts


def _chunk_index_from_config(config: SanaVideoCamCtrlConfig, num_frames: int) -> list[int] | None:
    chunk_size = getattr(config.model, "chunk_size", None)
    if chunk_size is None or int(chunk_size) >= num_frames:
        return None
    return chunk_index_from_chunk_size(
        num_frames,
        int(chunk_size),
        getattr(config.model, "chunk_split_strategy", "uniform"),
    )


def _pad_and_split_cp(clean_images, noise, timesteps, camera_conditions, chunk_plucker, loss_mask, cp_size: int):
    if cp_size <= 1:
        return clean_images, noise, timesteps, camera_conditions, chunk_plucker, loss_mask

    from diffusion.distributed.context_parallel import (
        cp_broadcast_tensor,
        cp_build_frame_valid_mask,
        cp_right_pad_size,
        cp_right_pad_temporal,
        cp_split_temporal,
        get_cp_group,
    )

    cp_group = get_cp_group()
    if cp_group is None:
        raise RuntimeError("Context parallel group is not initialized.")

    local_world = dist.get_world_size(cp_group)
    pad_frames = cp_right_pad_size(clean_images.shape[2], local_world)
    if pad_frames > 0:
        clean_images = cp_right_pad_temporal(clean_images, dim=2, pad_size=pad_frames, value=0.0)
        noise = cp_right_pad_temporal(noise, dim=2, pad_size=pad_frames, value=0.0)
        if isinstance(timesteps, torch.Tensor) and timesteps.ndim >= 3:
            timesteps = cp_right_pad_temporal(timesteps, dim=2, pad_size=pad_frames, value=0.0)
        if camera_conditions is not None:
            camera_conditions = _right_pad_temporal_with_last_frame(camera_conditions, dim=1, pad_size=pad_frames)
        if chunk_plucker is not None:
            chunk_plucker = cp_right_pad_temporal(chunk_plucker, dim=2, pad_size=pad_frames, value=0.0)
        frame_valid_mask = cp_build_frame_valid_mask(clean_images, pad_frames)
        if loss_mask is None:
            loss_mask = frame_valid_mask
        else:
            loss_mask = cp_right_pad_temporal(loss_mask, dim=2, pad_size=pad_frames, value=0.0)
            loss_mask = loss_mask * frame_valid_mask

    cp_broadcast_tensor(clean_images, cp_group)
    cp_broadcast_tensor(noise, cp_group)
    if isinstance(timesteps, torch.Tensor) and timesteps.ndim >= 3:
        cp_broadcast_tensor(timesteps, cp_group)
    if camera_conditions is not None:
        cp_broadcast_tensor(camera_conditions, cp_group)
    if chunk_plucker is not None:
        cp_broadcast_tensor(chunk_plucker, cp_group)
    if loss_mask is not None:
        cp_broadcast_tensor(loss_mask, cp_group)

    clean_images = cp_split_temporal(clean_images, dim=2, group=cp_group).contiguous()
    noise = cp_split_temporal(noise, dim=2, group=cp_group).contiguous()
    if isinstance(timesteps, torch.Tensor) and timesteps.ndim >= 3:
        timesteps = cp_split_temporal(timesteps, dim=2, group=cp_group).contiguous()
    if camera_conditions is not None:
        camera_conditions = cp_split_temporal(camera_conditions, dim=1, group=cp_group).contiguous()
    if chunk_plucker is not None:
        chunk_plucker = cp_split_temporal(chunk_plucker, dim=2, group=cp_group).contiguous()
    if loss_mask is not None:
        loss_mask = cp_split_temporal(loss_mask, dim=2, group=cp_group).contiguous()
    return clean_images, noise, timesteps, camera_conditions, chunk_plucker, loss_mask


def _build_timesteps(
    config: SanaVideoCamCtrlConfig,
    clean_images,
    global_t: int,
    do_i2v: bool,
    time_sampler: IncrementalTimesteps | None = None,
):
    bs = clean_images.shape[0]
    chunk_index = _chunk_index_from_config(config, global_t)
    size = (bs, 1, global_t) if config.task == "df" else (bs,)
    timesteps = process_timesteps(
        weighting_scheme=config.scheduler.weighting_scheme,
        train_sampling_steps=config.scheduler.train_sampling_steps,
        size=size,
        device=clean_images.device,
        logit_mean=config.scheduler.logit_mean,
        logit_std=config.scheduler.logit_std,
        p_low=config.scheduler.p_low,
        p_high=config.scheduler.p_high,
        num_frames=global_t,
        chunk_index=chunk_index,
        chunk_sampling_strategy=getattr(config.train, "chunk_sampling_strategy", "uniform"),
        same_timestep_prob=getattr(config.train, "same_timestep_prob", 0.0),
        chunk_mixture_probs=getattr(config.train, "chunk_mixture_probs", None),
        time_sampler=time_sampler,
        do_i2v=do_i2v,
        noise_multiplier=config.train.noise_multiplier,
    )
    return timesteps, chunk_index


def _build_time_sampler(config: SanaVideoCamCtrlConfig, latent_t: int, device: torch.device) -> IncrementalTimesteps | None:
    chunk_index = _chunk_index_from_config(config, latent_t)
    if chunk_index is None:
        return None
    return IncrementalTimesteps(
        F=chunk_index,
        T=config.scheduler.train_sampling_steps,
        device=device,
        dtype=torch.float64,
    )


def _build_i2v_loss_mask(clean_images: torch.Tensor, do_i2v: bool) -> torch.Tensor | None:
    if not do_i2v:
        return None
    loss_mask = torch.ones(
        (clean_images.shape[0], 1, clean_images.shape[2], 1, 1),
        device=clean_images.device,
        dtype=clean_images.dtype,
    )
    loss_mask[:, :, 0] = 0
    return loss_mask


def _extract_batch(batch, device, weight_dtype, load_text_feat: bool, return_chunk_plucker: bool):
    clean_images = batch[0].to(device=device, dtype=weight_dtype, non_blocking=True)
    if load_text_feat:
        y = batch[1].to(device=device, dtype=weight_dtype, non_blocking=True)
        y_mask = batch[2].to(device=device, non_blocking=True)
    else:
        y = batch[1]
        y_mask = None
    data_info = batch[3]
    camera_conditions = None
    if len(batch) > 6:
        camera_conditions = batch[6].to(device=device, dtype=weight_dtype, non_blocking=True)
    chunk_plucker = None
    if return_chunk_plucker:
        chunk_plucker = batch[-1].to(device=device, dtype=weight_dtype, non_blocking=True)
    return clean_images, y, y_mask, data_info, camera_conditions, chunk_plucker


def _encode_prompts(prompts, tokenizer, text_encoder, config, device):
    if isinstance(prompts, str):
        prompts = [prompts]
    prompts = list(prompts)
    max_length = config.text_encoder.model_max_length
    with torch.no_grad():
        if "T5" in config.text_encoder.text_encoder_name:
            txt_tokens = tokenizer(
                prompts,
                max_length=max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            ).to(device)
            y = text_encoder(txt_tokens.input_ids, attention_mask=txt_tokens.attention_mask)[0][:, None]
            y_mask = txt_tokens.attention_mask[:, None, None]
        elif "gemma" in config.text_encoder.text_encoder_name:
            if not config.text_encoder.chi_prompt:
                max_length_all = max_length
                encoded_prompts = prompts
            else:
                chi_prompt = "\n".join(config.text_encoder.chi_prompt)
                encoded_prompts = [chi_prompt + prompt for prompt in prompts]
                num_sys_prompt_tokens = len(tokenizer.encode(chi_prompt))
                max_length_all = num_sys_prompt_tokens + max_length - 2
            txt_tokens = tokenizer(
                encoded_prompts,
                padding="max_length",
                max_length=max_length_all,
                truncation=True,
                return_tensors="pt",
            ).to(device)
            select_index = [0] + list(range(-max_length + 1, 0))
            y = text_encoder(txt_tokens.input_ids, attention_mask=txt_tokens.attention_mask)[0][:, None][
                :, :, select_index
            ]
            y_mask = txt_tokens.attention_mask[:, None, None][:, :, :, select_index]
        elif "Qwen" in config.text_encoder.text_encoder_name:
            y, y_mask = text_encoder.get_prompt_embeds(prompts, max_length=max_length)
            y_mask = y_mask[:, None, None]
        else:
            raise ValueError(f"{config.text_encoder.text_encoder_name} is not supported.")
    return y, y_mask


def _ensure_null_embed(null_embed_path, tokenizer, text_encoder, config, device, weight_dtype):
    if osp.exists(null_embed_path):
        return
    if tokenizer is not None or text_encoder is not None:
        null_embed, null_mask = _encode_prompts([""], tokenizer, text_encoder, config, device)
        null_embed = null_embed[:, 0].detach().cpu()
        null_mask = null_mask[:, 0].detach().cpu()
    else:
        null_embed = torch.zeros(
            1,
            int(config.text_encoder.model_max_length),
            int(config.text_encoder.caption_channels),
            dtype=weight_dtype,
        )
        null_mask = torch.ones(1, int(config.text_encoder.model_max_length), dtype=torch.int64)
    torch.save({"uncond_prompt_embeds": null_embed, "uncond_prompt_embeds_mask": null_mask}, null_embed_path)


@pyrallis.wrap()
def main(cfg: SanaVideoCamCtrlConfig) -> None:
    config = cfg
    os.umask(0o000)
    os.makedirs(config.work_dir, exist_ok=True)

    from diffusion.distributed.context_parallel.config import set_cp_runtime_config

    set_cp_runtime_config(_resolve_cp_runtime_config(config))
    cp_size = _resolve_cp_size(config)
    fsdp_version = int(getattr(config.train, "fsdp_version", 1) or os.environ.get("SANA_FSDP_VERSION", "1"))
    if config.train.use_fsdp and fsdp_version != 2:
        raise ValueError("train_sana_wm_stage1.py supports FSDP2 for FSDP runs; set train.fsdp_version=2.")

    init_handler = InitProcessGroupKwargs(timeout=datetime.timedelta(seconds=5400))
    bootstrap_logger = get_root_logger(osp.join(config.work_dir, "train_log.log"))
    fsdp_plugin = _build_fsdp2_plugin(config, cp_size, bootstrap_logger) if config.train.use_fsdp else None
    parallelism_config = _build_parallelism_config(config, cp_size)
    accelerator = Accelerator(
        mixed_precision=config.model.mixed_precision,
        gradient_accumulation_steps=config.train.gradient_accumulation_steps,
        log_with=None if config.report_to in {"none", "None", None} else config.report_to,
        project_dir=osp.join(config.work_dir, "logs"),
        kwargs_handlers=[init_handler],
        fsdp_plugin=fsdp_plugin,
        parallelism_config=parallelism_config,
    )
    logger = get_root_logger(osp.join(config.work_dir, "train_log.log"))
    logger.info(accelerator.state)
    logger.info(f"Initializing: {'FSDP2' if config.train.use_fsdp else 'DDP'} for SANA-WM stage-1 training")

    world = get_world_size()
    rank = int(accelerator.process_index)
    if world % cp_size != 0:
        raise ValueError(f"WORLD_SIZE={world} must be divisible by cp_size={cp_size}.")
    dp_size = world // cp_size
    dp_rank = rank // cp_size
    logger.info(f"Context Parallel config: cp_size={cp_size}, dp_size={dp_size}, dp_rank={dp_rank}")

    seed = init_random_seed(getattr(config.train, "seed", None))
    config.train.seed = seed
    set_random_seed(seed + dp_rank)
    _prepare_hf_dataset(config, accelerator, logger)

    if accelerator.is_main_process:
        pyrallis.dump(config, open(osp.join(config.work_dir, "config.yaml"), "w"), sort_keys=False, indent=4)

    pred_sigma = getattr(config.scheduler, "pred_sigma", True)
    learn_sigma = getattr(config.scheduler, "learn_sigma", True) and pred_sigma
    train_diffusion = Scheduler(
        str(config.scheduler.train_sampling_steps),
        noise_schedule=config.scheduler.noise_schedule,
        predict_flow_v=config.scheduler.predict_flow_v,
        learn_sigma=learn_sigma,
        pred_sigma=pred_sigma,
        snr=config.train.snr_loss,
        flow_shift=config.scheduler.flow_shift,
    )

    if not config.data.load_vae_feat:
        raise ValueError("train_sana_wm_stage1.py expects precomputed latents.")

    max_length = config.text_encoder.model_max_length
    train_dataloader = _build_dataloader(config, max_length=max_length, dp_size=dp_size, dp_rank=dp_rank)
    logger.info(f"Training dataloader length: {len(train_dataloader)}")

    tokenizer = text_encoder = None
    if not config.data.load_text_feat:
        logger.info(f"Loading text encoder and tokenizer from {config.text_encoder.text_encoder_name} ...")
        tokenizer, text_encoder = get_tokenizer_and_text_encoder(
            name=config.text_encoder.text_encoder_name,
            device=accelerator.device,
        )
        if text_encoder is not None:
            text_encoder.eval().requires_grad_(False)

    latent_size = int(config.model.image_size) // int(config.vae.vae_stride[-1])
    if "CamCtrl" in config.model.model:
        model_kwargs = model_video_camctrl_init_config(config, latent_size=latent_size)
    else:
        model_kwargs = model_video_init_config(config, latent_size=latent_size)
    null_embed_path = osp.join(config.work_dir, "null_embed.pth")
    weight_dtype = get_weight_dtype(config.vae.weight_dtype)
    _ensure_null_embed(null_embed_path, tokenizer, text_encoder, config, accelerator.device, weight_dtype)
    model = build_model(
        config.model.model,
        config.train.grad_checkpointing,
        getattr(config.model, "fp32_attention", False),
        null_embed_path=null_embed_path,
        **model_kwargs,
    ).train()

    if config.model.load_from:
        load_checkpoint(
            checkpoint=config.model.load_from,
            model=model,
            model_ema=None,
            FSDP=False,
            load_ema=False,
            null_embed_path=null_embed_path,
            remove_state_dict_keys=getattr(config.model, "remove_state_dict_keys", None),
        )

    lr_scale_ratio = 1
    if getattr(config.train, "auto_lr", None):
        lr_scale_ratio = auto_scale_lr(
            config.train.train_batch_size * dp_size * config.train.gradient_accumulation_steps,
            config.train.optimizer,
            **config.train.auto_lr,
        )
    optimizer = build_optimizer(model, config.train.optimizer)
    lr_scheduler = build_lr_scheduler(config.train, optimizer, train_dataloader, lr_scale_ratio)

    logger.info(f"[Rank {rank}] Entering accelerator.prepare(model, optimizer, lr_scheduler)...")
    model, optimizer, lr_scheduler = accelerator.prepare(model, optimizer, lr_scheduler)
    if config.train.use_fsdp:
        _remove_custom_module_call(model, logger)
    _register_cp_group(accelerator, cp_size, logger)
    logger.info(f"[Rank {rank}] Model, optimizer, and scheduler prepared successfully.")

    log_buffer = LogBuffer()
    max_steps = getattr(config.train, "max_steps", None)
    global_step = 0
    start_time = time.time()
    early_stop_hours = float(getattr(config.train, "early_stop_hours", 0) or 0)
    last_saved_step = 0
    time_sampler = None
    time_sampler_chunks = 0

    for epoch in range(1, config.train.num_epochs + 1):
        if isinstance(train_dataloader.sampler, DistributedSampler):
            train_dataloader.sampler.set_epoch(epoch)
        for batch in train_dataloader:
            clean_images, y, y_mask, data_info, camera_conditions, chunk_plucker = _extract_batch(
                batch,
                accelerator.device,
                weight_dtype,
                bool(config.data.load_text_feat),
                bool(getattr(config.data, "return_chunk_plucker", False)),
            )
            if not config.data.load_text_feat:
                y, y_mask = _encode_prompts(y, tokenizer, text_encoder, config, accelerator.device)
                y = y.to(device=accelerator.device, dtype=weight_dtype)
                y_mask = y_mask.to(device=accelerator.device, non_blocking=True)

            global_t = clean_images.shape[2]
            chunk_index_for_sampler = _chunk_index_from_config(config, global_t)
            num_sampler_chunks = len(chunk_index_for_sampler) if chunk_index_for_sampler is not None else 0
            if num_sampler_chunks > time_sampler_chunks:
                time_sampler = _build_time_sampler(config, global_t, accelerator.device)
                time_sampler_chunks = num_sampler_chunks
            do_i2v = config.task == "df" and random.random() < float(config.train.ltx_image_condition_prob)
            loss_mask = _build_i2v_loss_mask(clean_images, do_i2v)
            timesteps, _ = _build_timesteps(config, clean_images, global_t, do_i2v, time_sampler=time_sampler)
            noise = torch.randn_like(clean_images)
            clean_images, noise, timesteps, camera_conditions, chunk_plucker, loss_mask = _pad_and_split_cp(
                clean_images,
                noise,
                timesteps,
                camera_conditions,
                chunk_plucker,
                loss_mask,
                cp_size,
            )
            full_t = clean_images.shape[2] * cp_size if cp_size > 1 else clean_images.shape[2]
            chunk_index_global = _chunk_index_from_config(config, full_t)
            chunk_index = chunk_index_global
            if cp_size > 1 and chunk_index_global is not None:
                from diffusion.distributed.context_parallel import get_cp_group

                cp_group = get_cp_group()
                cp_rank = dist.get_rank(cp_group)
                local_t = clean_images.shape[2]
                chunk_index = _map_global_chunk_index_to_local(
                    chunk_index_global,
                    full_t,
                    cp_rank * local_t,
                    (cp_rank + 1) * local_t,
                )

            model_kwargs = {"y": y, "mask": y_mask, "data_info": data_info}
            chunk_size = getattr(config.model, "chunk_size", None)
            if chunk_size is not None:
                model_kwargs["chunk_size"] = int(chunk_size)
                model_kwargs["chunk_split_strategy"] = getattr(config.model, "chunk_split_strategy", "uniform")
            if chunk_index is not None:
                model_kwargs["chunk_index"] = chunk_index
            if chunk_index_global is not None:
                model_kwargs["chunk_index_global"] = chunk_index_global
            if camera_conditions is not None:
                model_kwargs["camera_conditions"] = camera_conditions
            if chunk_plucker is not None:
                model_kwargs["chunk_plucker"] = chunk_plucker

            with accelerator.accumulate(model):
                optimizer.zero_grad(set_to_none=True)
                loss_term = train_diffusion.training_losses(
                    model,
                    clean_images,
                    timesteps,
                    model_kwargs=model_kwargs,
                    noise=noise,
                    timestep_weight=config.train.timestep_weight,
                    loss_mask=loss_mask,
                )
                loss = loss_term["loss"].mean()
                if cp_size > 1:
                    from diffusion.distributed.context_parallel import cp_reduce_loss, get_cp_group

                    cp_group = get_cp_group()
                    if cp_group is not None:
                        if loss_mask is None:
                            valid_tokens = clean_images.numel()
                        else:
                            valid_tokens = (
                                loss_mask.sum()
                                * clean_images.shape[1]
                                * clean_images.shape[3]
                                * clean_images.shape[4]
                            )
                        loss = cp_reduce_loss(loss, cp_group, num_valid_tokens=valid_tokens)
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), config.train.gradient_clip)
                optimizer.step()
                lr_scheduler.step()

            logs = {config.loss_report_name: accelerator.gather(loss.detach()).mean().item()}
            log_buffer.update(logs)
            if accelerator.is_main_process and (global_step == 0 or (global_step + 1) % config.train.log_interval == 0):
                log_buffer.average()
                logger.info(
                    f"Epoch: {epoch} | Step: {global_step + 1} | "
                    f"lr: {lr_scheduler.get_last_lr()[0]:.3e} | "
                    + ", ".join(f"{k}: {v:.4f}" for k, v in log_buffer.output.items())
                )
                log_buffer.clear()

            global_step += 1
            if config.train.save_model_steps > 0 and global_step % config.train.save_model_steps == 0:
                accelerator.wait_for_everyone()
                if config.train.use_fsdp:
                    save_checkpoint(
                        work_dir=osp.join(config.work_dir, "checkpoints"),
                        epoch=epoch,
                        step=global_step,
                        model=model,
                        accelerator=accelerator,
                        optimizer=optimizer,
                        lr_scheduler=lr_scheduler,
                        add_symlink=True,
                    )
                elif accelerator.is_main_process:
                    save_checkpoint(
                        work_dir=osp.join(config.work_dir, "checkpoints"),
                        epoch=epoch,
                        step=global_step,
                        model=accelerator.unwrap_model(model),
                        model_ema=None,
                        optimizer=optimizer,
                        lr_scheduler=lr_scheduler,
                        add_symlink=True,
                    )
                last_saved_step = global_step
            if early_stop_hours > 0 and (time.time() - start_time) >= early_stop_hours * 3600:
                accelerator.wait_for_everyone()
                if last_saved_step != global_step:
                    if config.train.use_fsdp:
                        save_checkpoint(
                            work_dir=osp.join(config.work_dir, "checkpoints"),
                            epoch=epoch,
                            step=global_step,
                            model=model,
                            accelerator=accelerator,
                            optimizer=optimizer,
                            lr_scheduler=lr_scheduler,
                            add_symlink=True,
                        )
                    elif accelerator.is_main_process:
                        save_checkpoint(
                            work_dir=osp.join(config.work_dir, "checkpoints"),
                            epoch=epoch,
                            step=global_step,
                            model=accelerator.unwrap_model(model),
                            model_ema=None,
                            optimizer=optimizer,
                            lr_scheduler=lr_scheduler,
                            add_symlink=True,
                        )
                logger.info(
                    f"Reached train.early_stop_hours={early_stop_hours}; stopping at global_step={global_step}."
                )
                return
            if max_steps is not None and global_step >= int(max_steps):
                return


if __name__ == "__main__":
    main()
