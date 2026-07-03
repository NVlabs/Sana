# Copyright 2024 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# SPDX-License-Identifier: Apache-2.0

"""ODE and self-forcing distillation trainer for SANA-WM.

The trainer is selected by ``train_longsana.py`` through ``wm_ode`` or
``wm_self_forcing``. Self-forcing uses the streaming student's recurrent
KV cache; only the bidirectional score models shard their temporal input over
the context-parallel group.
"""

from __future__ import annotations

import datetime
import json
import os
import os.path as osp
import time
from contextlib import nullcontext
from dataclasses import asdict

import lmdb
import numpy as np
import torch
import torch.distributed as dist
from accelerate import Accelerator, InitProcessGroupKwargs
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, Dataset, DistributedSampler

from diffusion.data.builder import build_dataset
from diffusion.distributed.context_parallel import cp_reduce_loss, get_cp_group
from diffusion.distributed.context_parallel.config import set_cp_runtime_config
from diffusion.longsana.utils.lmdb import (
    get_array_shape_from_lmdb,
    retrieve_row_from_lmdb,
)
from diffusion.longsana.utils.model_wrapper import SanaTextEncoder
from diffusion.model import nets as _nets  # noqa: F401
from diffusion.model.builder import build_model
from diffusion.model.utils import get_weight_dtype
from diffusion.utils.camctrl_config import (
    ModelVideoCamCtrlConfig,
    model_video_camctrl_init_config,
)
from diffusion.utils.chunk_utils import chunk_index_from_chunk_size
from diffusion.utils.config import AEConfig, SchedulerConfig, TextEncoderConfig
from diffusion.utils.logger import get_root_logger
from diffusion.utils.misc import init_random_seed, set_random_seed
from tools.download import find_model
from train_video_scripts.train_sana_wm_stage1 import (
    _build_fsdp2_plugin,
    _build_parallelism_config,
    _prepare_hf_dataset,
    _register_cp_group,
    _remove_custom_module_call,
    _resolve_cp_runtime_config,
)


class SanaWMOdeTrajectoryDataset(Dataset):
    """Read sharded SANA-WM ODE trajectories.

    Every LMDB shard stores ``latents`` as ``(N, K, T, C, H, W)`` plus
    ``prompts``, ``camera_conditions`` (``N, T, 20``), and
    ``chunk_plucker`` (``N, 48, T, H, W``). ``first_frame`` is optional; when
    absent it is taken from the clean trajectory endpoint.
    """

    def __init__(self, data_path: str, num_frames: int | None = None, max_samples: int | None = None):
        if os.path.isfile(os.path.join(data_path, "data.mdb")):
            shard_paths = [data_path]
        else:
            shard_paths = [
                os.path.join(data_path, name)
                for name in sorted(os.listdir(data_path))
                if os.path.isfile(os.path.join(data_path, name, "data.mdb"))
            ]
        if not shard_paths:
            raise FileNotFoundError(f"No LMDB shard found under {data_path!r}.")

        self.envs = [lmdb.open(path, readonly=True, lock=False, readahead=False, meminit=False) for path in shard_paths]
        self.shapes = []
        self.index = []
        for shard_id, env in enumerate(self.envs):
            shapes = {"latents": get_array_shape_from_lmdb(env, "latents")}
            for name in ("camera_conditions", "chunk_plucker", "first_frame"):
                try:
                    shapes[name] = get_array_shape_from_lmdb(env, name)
                except Exception:
                    shapes[name] = None
            if shapes["camera_conditions"] is None or shapes["chunk_plucker"] is None:
                raise ValueError(
                    f"SANA-WM ODE shard {shard_paths[shard_id]!r} must contain " "camera_conditions and chunk_plucker."
                )
            self.shapes.append(shapes)
            self.index.extend((shard_id, row) for row in range(shapes["latents"][0]))

        self.num_frames = None if num_frames is None else int(num_frames)
        if max_samples is not None and int(max_samples) > 0:
            self.index = self.index[: int(max_samples)]

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str]:
        shard_id, row = self.index[idx]
        env = self.envs[shard_id]
        shapes = self.shapes[shard_id]

        trajectory = torch.from_numpy(
            retrieve_row_from_lmdb(env, "latents", np.float16, row, shape=shapes["latents"][1:]).copy()
        ).float()
        if trajectory.ndim == 4:
            trajectory = trajectory.unsqueeze(0)
        if trajectory.ndim != 5:
            raise ValueError(f"Expected trajectory (K,T,C,H,W), got {tuple(trajectory.shape)}")

        camera = torch.from_numpy(
            retrieve_row_from_lmdb(
                env,
                "camera_conditions",
                np.float16,
                row,
                shape=shapes["camera_conditions"][1:],
            ).copy()
        ).float()
        plucker = torch.from_numpy(
            retrieve_row_from_lmdb(
                env,
                "chunk_plucker",
                np.float16,
                row,
                shape=shapes["chunk_plucker"][1:],
            ).copy()
        ).float()

        if self.num_frames is not None:
            trajectory = trajectory[:, : self.num_frames]
            camera = camera[: self.num_frames]
            plucker = plucker[:, : self.num_frames]

        if shapes["first_frame"] is None:
            first_frame = trajectory[-1, :1].permute(1, 0, 2, 3).contiguous()
        else:
            first_frame = torch.from_numpy(
                retrieve_row_from_lmdb(
                    env,
                    "first_frame",
                    np.float16,
                    row,
                    shape=shapes["first_frame"][1:],
                ).copy()
            ).float()

        return {
            "trajectory": trajectory,
            "prompt": retrieve_row_from_lmdb(env, "prompts", str, row),
            "camera_conditions": camera,
            "chunk_plucker": plucker,
            "first_frame": first_frame,
        }


class SanaWMDistillTrainer:
    def __init__(self, config: DictConfig):
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        self.config = config
        self.mode = str(config.mode).lower()
        if self.mode not in {"ode", "self_forcing"}:
            raise ValueError(f"mode must be 'ode' or 'self_forcing', got {self.mode!r}")
        if int(config.train.gradient_accumulation_steps) != 1:
            raise ValueError("SANA-WM distillation currently requires gradient_accumulation_steps=1.")

        work_dir = str(config.get("logdir", "")).strip() or str(config.work_dir)
        self.work_dir = osp.abspath(osp.expanduser(work_dir))
        os.makedirs(self.work_dir, exist_ok=True)
        self.logger = get_root_logger(osp.join(self.work_dir, "train_log.log"))
        if not config.get("resume_from", None) and bool(config.get("auto_resume", False)):
            checkpoint_root = osp.join(self.work_dir, "checkpoints")
            if osp.isdir(checkpoint_root):
                checkpoints = [
                    osp.join(checkpoint_root, name)
                    for name in os.listdir(checkpoint_root)
                    if name.removeprefix("step_").isdigit()
                    and osp.isfile(osp.join(checkpoint_root, name, "trainer_state.json"))
                ]
                if checkpoints:
                    config.resume_from = max(
                        checkpoints,
                        key=lambda path: int(osp.basename(path).removeprefix("step_")),
                    )

        self.student_config = self._load_model_config(str(config.model_config))
        self.student_config.train = config.train
        self.student_config.work_dir = self.work_dir
        self.student_config.model.use_autograd_kernel = True

        self.cp_size = 1 if self.mode == "ode" else max(1, int(config.train.cp_size))
        if self.mode == "self_forcing":
            set_cp_runtime_config(_resolve_cp_runtime_config(self.student_config))
        fsdp_version = int(config.train.fsdp_version)
        if bool(config.train.use_fsdp) and fsdp_version != 2:
            raise ValueError("SANA-WM distillation supports FSDP2 only.")

        init_handler = InitProcessGroupKwargs(timeout=datetime.timedelta(seconds=5400))
        # The model already checkpoints each transformer block. Do not wrap
        # the same blocks in a second FSDP activation-checkpoint layer.
        fsdp_config = OmegaConf.create(OmegaConf.to_container(self.student_config, resolve=False))
        fsdp_config.train.grad_checkpointing = False
        fsdp_plugin = (
            _build_fsdp2_plugin(fsdp_config, self.cp_size, self.logger) if bool(config.train.use_fsdp) else None
        )
        parallelism_config = _build_parallelism_config(self.student_config, self.cp_size)
        report_to = config.get("report_to", "none")
        if bool(config.get("disable_wandb", False)) and report_to == "wandb":
            report_to = "none"
        tracking_dir = (
            str(config.get("wandb_save_dir", "./wandb")) if report_to == "wandb" else osp.join(self.work_dir, "logs")
        )
        if report_to == "wandb":
            os.makedirs(tracking_dir, exist_ok=True)
        self.accelerator = Accelerator(
            # The models are cast explicitly below. Accelerate's FSDP2 path
            # otherwise upcasts BF16 master parameters and Adam states to FP32.
            mixed_precision="no",
            gradient_accumulation_steps=1,
            log_with=None if report_to in {None, "none", "None"} else report_to,
            project_dir=tracking_dir,
            kwargs_handlers=[init_handler],
            fsdp_plugin=fsdp_plugin,
            parallelism_config=parallelism_config,
        )
        self.rank = int(self.accelerator.process_index)
        self.world_size = int(self.accelerator.num_processes)
        if self.world_size % self.cp_size:
            raise ValueError(f"WORLD_SIZE={self.world_size} must be divisible by cp_size={self.cp_size}")
        self.dp_size = self.world_size // self.cp_size
        self.dp_rank = self.rank // self.cp_size

        seed = init_random_seed(int(config.train.seed))
        self.config.train.seed = seed
        set_random_seed(seed + self.dp_rank)
        self.dtype = get_weight_dtype(str(self.student_config.model.mixed_precision))

        self.student = self._build_model(
            self.student_config,
            str(config.model_path),
            trainable=True,
            checkpoint_key="generator",
        )
        self.fake_score = self.real_score = None
        if self.mode == "self_forcing":
            self.fake_config = self._load_model_config(str(config.fake_model_config))
            self.fake_config.work_dir = self.work_dir
            self.fake_config.model.use_autograd_kernel = True
            self.real_config = self._load_model_config(str(config.real_model_config))
            self.real_config.work_dir = self.work_dir
            self.real_config.model.use_autograd_kernel = False
            self.fake_score = self._build_model(
                self.fake_config,
                str(config.fake_model_path),
                trainable=True,
                checkpoint_key="critic",
            )
            self.real_score = self._build_model(
                self.real_config,
                str(config.real_model_path),
                trainable=False,
            )

        self.student_optimizer = torch.optim.AdamW(
            self.student.parameters(),
            lr=float(config.optimizer.lr),
            betas=tuple(config.optimizer.betas),
            weight_decay=float(config.optimizer.weight_decay),
        )
        if self.mode == "ode":
            self.student, self.student_optimizer = self.accelerator.prepare(
                self.student,
                self.student_optimizer,
            )
        else:
            self.fake_optimizer = torch.optim.AdamW(
                self.fake_score.parameters(),
                lr=float(config.critic_optimizer.lr),
                betas=tuple(config.critic_optimizer.betas),
                weight_decay=float(config.critic_optimizer.weight_decay),
            )
            self.student, self.student_optimizer = self.accelerator.prepare(
                self.student,
                self.student_optimizer,
            )
            self.fake_score, self.fake_optimizer = self.accelerator.prepare(
                self.fake_score,
                self.fake_optimizer,
            )
            # Accelerate FSDP2 supports one model per prepare() call.  The
            # frozen teacher needs neither sharding nor checkpoint state.
            self.real_score = self.real_score.to(device=self.accelerator.device, dtype=self.dtype)

        if bool(config.train.use_fsdp):
            _remove_custom_module_call(self.student, self.logger)
        if self.mode == "self_forcing":
            _register_cp_group(self.accelerator, self.cp_size, self.logger)

        self.student.eval()
        if self.fake_score is not None:
            self.fake_score.eval()
            self.real_score.eval()

        self.text_encoder = SanaTextEncoder(
            self.student_config,
            device=self.accelerator.device,
            dtype=self.dtype,
        )
        self.text_encoder.eval().requires_grad_(False)
        self.dataloader = self._build_dataloader()
        set_random_seed(seed + self.dp_rank)
        self.global_step = 0

        resume_from = config.get("resume_from", None)
        if resume_from:
            state_path = osp.join(str(resume_from), "trainer_state.json")
            if osp.isfile(state_path):
                with open(state_path, encoding="utf-8") as f:
                    self.global_step = int(json.load(f).get("global_step", 0))

        steps_per_epoch = len(self.dataloader)
        if steps_per_epoch == 0:
            raise ValueError("The distillation dataloader has no complete batch.")
        batches_seen = self.global_step
        if self.mode == "self_forcing":
            interval = int(self.config.generator_update_interval)
            batches_seen += (self.global_step + interval - 1) // interval
        self.data_epoch = batches_seen // steps_per_epoch
        self.dataloader.sampler.set_epoch(self.data_epoch)
        self.data_iterator = iter(self.dataloader)
        for _ in range(batches_seen % steps_per_epoch):
            next(self.data_iterator)

        if resume_from:
            if self.accelerator.is_main_process:
                self.logger.info(f"Resuming from {resume_from} at step {self.global_step}")
            self._materialize_adam_state()
            # Load last so checkpointed Python/NumPy/Torch RNG states replace
            # any randomness consumed while positioning the dataloader.
            self.accelerator.load_state(str(resume_from))
            if self.accelerator.is_main_process:
                self.logger.info(f"Restored model, optimizer, and RNG state at step {self.global_step}")

        if self.accelerator.is_main_process:
            OmegaConf.save(config, osp.join(self.work_dir, "config.yaml"))
        if report_to not in {None, "none", "None"}:
            init_kwargs = {}
            if report_to == "wandb":
                init_kwargs["wandb"] = {"dir": tracking_dir}
                if config.get("wandb_name", None):
                    init_kwargs["wandb"]["name"] = str(config.wandb_name)
            self.accelerator.init_trackers(
                str(config.get("tracker_project_name", "sana-wm-distill")),
                config=OmegaConf.to_container(config, resolve=True),
                init_kwargs=init_kwargs,
            )

    @staticmethod
    def _load_model_config(path: str) -> DictConfig:
        config = OmegaConf.load(path)
        config.model = OmegaConf.merge(OmegaConf.create(asdict(ModelVideoCamCtrlConfig())), config.model)
        config.vae = OmegaConf.merge(OmegaConf.create(asdict(AEConfig())), config.vae)
        config.text_encoder = OmegaConf.merge(OmegaConf.create(asdict(TextEncoderConfig())), config.text_encoder)
        config.scheduler = OmegaConf.merge(OmegaConf.create(asdict(SchedulerConfig())), config.scheduler)
        return config

    def _build_model(
        self,
        model_config: DictConfig,
        checkpoint: str,
        trainable: bool,
        checkpoint_key: str | None = None,
    ) -> torch.nn.Module:
        latent_size = int(model_config.model.image_size) // int(model_config.vae.vae_stride[-1])
        model = build_model(
            str(model_config.model.model),
            use_grad_checkpoint=bool(self.config.train.grad_checkpointing and trainable),
            use_fp32_attention=bool(model_config.model.get("fp32_attention", False)),
            **model_video_camctrl_init_config(model_config, latent_size=latent_size),
        )

        state = find_model(checkpoint)
        if checkpoint_key is not None and checkpoint_key in state:
            state = state[checkpoint_key]
        elif "generator" in state:
            state = state["generator"]
        elif "model" in state:
            state = state["model"]
        if "state_dict" in state:
            state = state["state_dict"]
        state = {(key[6:] if key.startswith("model.") else key): value for key, value in state.items()}
        state.pop("pos_embed", None)
        missing, unexpected = model.load_state_dict(state, strict=False)
        if set(missing) != {"pos_embed"} or unexpected:
            raise RuntimeError(
                f"Checkpoint {checkpoint} does not match {model_config.model.model}: "
                f"missing={missing}, unexpected={unexpected}"
            )
        if self.accelerator.is_main_process:
            self.logger.info(
                f"Loaded {checkpoint}: model={model_config.model.model}, "
                f"missing={len(missing)}, unexpected={len(unexpected)}"
            )
        # Match the internal SANA-WM trainers: parameters and Adam states are
        # BF16, rather than FP32 parameters with BF16 forward autocasting.
        model.to(dtype=self.dtype)
        model.requires_grad_(trainable)
        return model.eval()

    def _build_dataloader(self) -> DataLoader:
        if self.mode == "ode":
            dataset = SanaWMOdeTrajectoryDataset(
                str(self.config.data_path),
                num_frames=int(self.config.num_latent_frames),
                max_samples=self.config.get("max_samples", None),
            )
        else:
            self.student_config.data = self.config.data
            _prepare_hf_dataset(self.student_config, self.accelerator, self.logger)
            data_cfg = OmegaConf.to_container(self.student_config.data, resolve=True)
            dataset = build_dataset(
                data_cfg,
                resolution=int(self.student_config.model.image_size),
                max_length=int(self.student_config.text_encoder.model_max_length),
                config=self.student_config,
                caption_proportion=self.student_config.data.caption_proportion,
                sort_dataset=bool(self.student_config.data.sort_dataset),
                vae_downsample_rate=int(self.student_config.vae.vae_stride[-1]),
                num_frames=int(self.student_config.data.num_frames),
            )

        sampler = DistributedSampler(
            dataset,
            num_replicas=self.dp_size,
            rank=self.dp_rank,
            shuffle=True,
            drop_last=True,
            seed=int(self.config.train.seed),
        )
        return DataLoader(
            dataset,
            batch_size=int(self.config.train.batch_size),
            sampler=sampler,
            num_workers=int(self.config.train.num_workers),
            pin_memory=True,
            persistent_workers=int(self.config.train.num_workers) > 0,
            drop_last=True,
            generator=torch.Generator().manual_seed(int(self.config.train.seed) + self.dp_rank),
        )

    def _next_batch(self):
        try:
            return next(self.data_iterator)
        except StopIteration:
            self.data_epoch += 1
            self.dataloader.sampler.set_epoch(self.data_epoch)
            self.data_iterator = iter(self.dataloader)
            return next(self.data_iterator)

    def _forward_score_cp(
        self,
        model: torch.nn.Module,
        x: torch.Tensor,
        timestep: torch.Tensor,
        y: torch.Tensor,
        y_mask: torch.Tensor,
        camera_conditions: torch.Tensor,
        chunk_plucker: torch.Tensor,
    ):
        """Run one bidirectional score-model forward on a temporal CP shard."""
        batch, _, valid_t, _, _ = x.shape
        pad = (-valid_t) % self.cp_size
        if pad:
            x = torch.cat([x, x.new_zeros(batch, x.shape[1], pad, x.shape[3], x.shape[4])], dim=2)
            timestep = torch.cat([timestep, timestep.new_zeros(batch, pad)], dim=1)
            camera_conditions = torch.cat(
                [camera_conditions, camera_conditions[:, -1:].expand(-1, pad, -1)],
                dim=1,
            )
            chunk_plucker = torch.cat(
                [
                    chunk_plucker,
                    chunk_plucker.new_zeros(
                        batch,
                        chunk_plucker.shape[1],
                        pad,
                        chunk_plucker.shape[3],
                        chunk_plucker.shape[4],
                    ),
                ],
                dim=2,
            )
        padded_t = x.shape[2]
        frame_valid = torch.ones(batch, padded_t, device=x.device, dtype=x.dtype)
        if pad:
            frame_valid[:, valid_t:] = 0

        cp_group = get_cp_group()
        cp_rank = dist.get_rank(cp_group) if self.cp_size > 1 else 0
        local_t = padded_t // self.cp_size
        local_start = cp_rank * local_t
        local_end = local_start + local_t
        x_local = x[:, :, local_start:local_end].contiguous()
        t_local = timestep[:, local_start:local_end].contiguous()
        camera_local = camera_conditions[:, local_start:local_end].contiguous()
        plucker_local = chunk_plucker[:, :, local_start:local_end].contiguous()
        valid_local = frame_valid[:, local_start:local_end].contiguous()
        kwargs = {
            "data_info": {},
            "camera_conditions": camera_local,
            "chunk_plucker": plucker_local,
            # This mask also opts the bidirectional scorer into its CP-aware
            # halo/all-gather path without changing ordinary LongSANA calls.
            "frame_valid_mask": valid_local,
        }

        flow = model(x_local, t_local[:, None, :], y, mask=y_mask, **kwargs)
        if isinstance(flow, tuple):
            flow = flow[0]
        if hasattr(flow, "sample"):
            flow = flow.sample

        return flow, {
            "x": x_local,
            "timestep": t_local,
            "valid": valid_local,
            "local_start": local_start,
            "local_end": local_end,
            "pad": pad,
        }

    def _init_kv_cache(self, num_chunks: int) -> list:
        model = self.accelerator.unwrap_model(self.student)
        return [[[None] * 10 for _ in model.blocks] for _ in range(num_chunks)]

    def _accumulate_kv_cache(self, caches: list, chunk_id: int) -> list:
        """Build the cached state consumed by one streaming generator chunk."""
        if chunk_id == 0:
            return caches[0]

        cached_chunks = int(self.config.num_cached_blocks)
        start = max(chunk_id - cached_chunks, 0) if cached_chunks > 0 else 0
        selected = list(range(start, chunk_id))
        if bool(self.config.sink_token) and cached_chunks > 0:
            recent_start = max(chunk_id - cached_chunks + 1, 0)
            if recent_start > 0:
                selected = [0, *range(recent_start, chunk_id)]

        current = caches[chunk_id]
        kept = set(selected)
        for block_id in range(len(current)):
            previous = caches[chunk_id - 1][block_id]
            type_flag = previous[6]
            is_state = (
                type_flag is not None and float(type_flag.item() if torch.is_tensor(type_flag) else type_flag) > 0.5
            )
            if is_state:
                current[block_id] = [
                    previous[0],
                    previous[1],
                    previous[2],
                    previous[3],
                    previous[4],
                    None,
                    previous[6],
                    None,
                    None,
                    previous[-1],
                ]
            else:
                accumulated = [None] * 4
                for cached_id in selected:
                    cached = caches[cached_id][block_id]
                    for slot in range(4):
                        if cached[slot] is None:
                            continue
                        accumulated[slot] = (
                            cached[slot]
                            if accumulated[slot] is None
                            else torch.cat([accumulated[slot], cached[slot]], dim=2)
                        )
                current[block_id] = [
                    *accumulated,
                    previous[4],
                    None,
                    previous[6],
                    None,
                    None,
                    previous[-1],
                ]

            if cached_chunks > 0:
                for cached_id in range(chunk_id):
                    if cached_id not in kept:
                        caches[cached_id][block_id] = [None] * 10

        return current

    def _masked_mse(self, prediction: torch.Tensor, target: torch.Tensor, frame_mask: torch.Tensor):
        mask = frame_mask[:, None, :, None, None].float()
        residual = torch.where(mask.bool(), prediction.float() - target.float(), 0.0)
        error = residual.square()
        local_count = mask.sum() * prediction.shape[1] * prediction.shape[3] * prediction.shape[4]
        loss = error.sum() / local_count.clamp_min(1)
        if self.cp_size > 1:
            loss = cp_reduce_loss(loss, get_cp_group(), num_valid_tokens=local_count)
        return loss

    def _encode(self, prompts, use_chi_prompt: bool = True):
        encoded = self.text_encoder.forward_chi(prompts, use_chi_prompt=use_chi_prompt)
        return encoded["prompt_embeds"][:, None], encoded["mask"][:, None, None]

    def _rollout(
        self,
        first_frame: torch.Tensor,
        camera_conditions: torch.Tensor,
        chunk_plucker: torch.Tensor,
        y: torch.Tensor,
        y_mask: torch.Tensor,
        requires_grad: bool,
    ) -> torch.Tensor:
        total_t = int(self.config.num_latent_frames)
        chunk_size = int(self.student_config.model.chunk_size)
        chunk_split_strategy = str(self.student_config.model.chunk_split_strategy)
        starts = chunk_index_from_chunk_size(total_t, chunk_size, chunk_split_strategy)
        ends = [*starts[1:], total_t]
        schedule = [int(value) for value in self.config.denoising_step_list][:-1]

        exit_step = torch.randint(0, len(schedule), (1,), device=self.accelerator.device)
        # This value changes the number of model calls, so every FSDP shard
        # must take the same branch.  Noise remains independent across DP
        # replicas and is synchronized only within each CP group below.
        if dist.is_initialized():
            dist.broadcast(exit_step, src=0)
        exit_step = int(exit_step.item())

        batch, channels, _, height, width = first_frame.shape
        initial_noise = torch.randn(
            batch,
            channels,
            total_t,
            height,
            width,
            device=self.accelerator.device,
            dtype=self.dtype,
        )
        if self.cp_size > 1:
            cp_group = get_cp_group()
            dist.broadcast(initial_noise, src=dist.get_global_rank(cp_group, 0), group=cp_group)
        initial_noise[:, :, :1] = first_frame
        generated_chunks: list[torch.Tensor] = []
        kv_cache = self._init_kv_cache(len(starts))

        for chunk_id, (start, end) in enumerate(zip(starts, ends)):
            current = initial_noise[:, :, start:end]
            chunk_cache = self._accumulate_kv_cache(kv_cache, chunk_id)
            chunk_camera = camera_conditions[:, start:end]
            current_plucker = chunk_plucker[:, :, start:end]
            frame_index = torch.arange(start, end, device=current.device)

            for step_id, current_t in enumerate(schedule[: exit_step + 1]):
                chunk_timestep = torch.full(
                    (batch, end - start),
                    float(current_t),
                    device=current.device,
                    dtype=torch.float32,
                )
                if chunk_id == 0:
                    chunk_timestep[:, 0] = 0

                track_grad = requires_grad and step_id == exit_step
                grad_context = nullcontext() if track_grad else torch.no_grad()
                # forward_long checkpoints every transformer block. Keep the
                # read-cache lists stable until backward; the t=0 cache-save
                # call below updates the original lists in place.
                read_cache = [list(block_cache) for block_cache in chunk_cache] if track_grad else chunk_cache
                with grad_context:
                    result = self.student(
                        current,
                        chunk_timestep[:, None, :],
                        y,
                        mask=y_mask,
                        data_info={},
                        camera_conditions=chunk_camera,
                        chunk_plucker=current_plucker,
                        chunk_size=chunk_size,
                        chunk_split_strategy=chunk_split_strategy,
                        start_f=start,
                        end_f=end,
                        frame_index=frame_index,
                        kv_cache=read_cache,
                        save_kv_cache=False,
                    )
                    flow = result[0] if isinstance(result, tuple) else result
                    if hasattr(flow, "sample"):
                        flow = flow.sample
                    sigma = chunk_timestep[:, None, :, None, None] / 1000.0
                    current_x0 = (current - sigma * flow).to(flow.dtype)
                    if chunk_id == 0:
                        current_x0 = torch.cat([first_frame, current_x0[:, :, 1:]], dim=2)

                if step_id < exit_step:
                    next_sigma = float(schedule[step_id + 1]) / 1000.0
                    step_noise = torch.randn_like(current_x0)
                    if self.cp_size > 1:
                        cp_group = get_cp_group()
                        dist.broadcast(step_noise, src=dist.get_global_rank(cp_group, 0), group=cp_group)
                    current = (1.0 - next_sigma) * current_x0.detach() + next_sigma * step_noise
                    if chunk_id == 0:
                        current[:, :, :1] = first_frame

            with torch.no_grad():
                cache_result = self.student(
                    current_x0.detach(),
                    torch.zeros(batch, 1, end - start, device=current.device),
                    y,
                    mask=y_mask,
                    data_info={},
                    camera_conditions=chunk_camera,
                    chunk_plucker=current_plucker,
                    chunk_size=chunk_size,
                    chunk_split_strategy=chunk_split_strategy,
                    start_f=start,
                    end_f=end,
                    frame_index=frame_index,
                    kv_cache=chunk_cache,
                    save_kv_cache=True,
                )
                if not isinstance(cache_result, tuple) or len(cache_result) != 2:
                    raise RuntimeError("Streaming SANA-WM generator did not return its KV cache.")
                kv_cache[chunk_id] = cache_result[1]

            generated_chunks.append(current_x0 if requires_grad else current_x0.detach())

        return torch.cat(generated_chunks, dim=2)

    def _materialize_adam_state(self):
        """Create step-0 Adam slots for parameters that have not received a gradient."""
        optimizers = [self.student_optimizer]
        if self.fake_score is not None:
            optimizers.append(self.fake_optimizer)
        for optimizer in optimizers:
            for group in optimizer.param_groups:
                for parameter in group["params"]:
                    if parameter in optimizer.state:
                        continue
                    state = optimizer.state[parameter]
                    step_device = parameter.device if group.get("capturable") or group.get("fused") else "cpu"
                    state["step"] = torch.zeros((), dtype=torch.float32, device=step_device)
                    state["exp_avg"] = torch.zeros_like(parameter)
                    state["exp_avg_sq"] = torch.zeros_like(parameter)
                    if group.get("amsgrad", False):
                        state["max_exp_avg_sq"] = torch.zeros_like(parameter)

    def _save(self):
        checkpoint_dir = osp.join(self.work_dir, "checkpoints", f"step_{self.global_step:06d}")
        self._materialize_adam_state()
        self.accelerator.wait_for_everyone()
        self.accelerator.save_state(checkpoint_dir)
        generator_state = self.accelerator.get_state_dict(self.student)
        critic_state = self.accelerator.get_state_dict(self.fake_score) if self.fake_score is not None else None
        if self.accelerator.is_main_process:
            payload = {
                "generator": generator_state,
                "step": self.global_step,
                "denoising_step_list": [int(value) for value in self.config.denoising_step_list],
            }
            if critic_state is not None:
                payload["critic"] = critic_state
            torch.save(payload, osp.join(checkpoint_dir, "model.pt"))
            with open(osp.join(checkpoint_dir, "trainer_state.json"), "w", encoding="utf-8") as f:
                json.dump({"global_step": self.global_step}, f)
        self.accelerator.wait_for_everyone()

    def _train_ode_step(self, batch) -> dict[str, float]:
        self.student_optimizer.zero_grad(set_to_none=True)
        trajectory = batch["trajectory"].to(self.accelerator.device, dtype=self.dtype)
        camera = batch["camera_conditions"].to(self.accelerator.device, dtype=self.dtype)
        plucker = batch["chunk_plucker"].to(self.accelerator.device, dtype=self.dtype)
        first_frame = batch["first_frame"].to(self.accelerator.device, dtype=self.dtype)
        y, y_mask = self._encode(batch["prompt"])

        batch_size, snapshots, total_t, channels, height, width = trajectory.shape
        if snapshots != len(self.config.denoising_step_list):
            raise ValueError(
                f"ODE trajectory has {snapshots} snapshots but denoising_step_list has "
                f"{len(self.config.denoising_step_list)} entries."
            )
        starts = chunk_index_from_chunk_size(
            total_t,
            int(self.student_config.model.chunk_size),
            str(self.student_config.model.chunk_split_strategy),
        )
        ends = [*starts[1:], total_t]
        snapshot_index = torch.empty(batch_size, total_t, device=trajectory.device, dtype=torch.long)
        for start, end in zip(starts, ends):
            snapshot_index[:, start:end] = torch.randint(
                0,
                snapshots,
                (batch_size, 1),
                device=trajectory.device,
            )
        snapshot_index[:, 0] = snapshots - 1
        noisy = torch.gather(
            trajectory,
            1,
            snapshot_index[:, None, :, None, None, None].expand(-1, 1, -1, channels, height, width),
        ).squeeze(1)
        noisy = noisy.permute(0, 2, 1, 3, 4).contiguous()
        noisy[:, :, :1] = first_frame
        target = trajectory[:, -1].permute(0, 2, 1, 3, 4).contiguous()
        schedule = torch.tensor(self.config.denoising_step_list, device=trajectory.device)
        timestep = schedule[snapshot_index].float()

        flow = self.student(
            noisy,
            timestep[:, None, :],
            y,
            mask=y_mask,
            data_info={},
            camera_conditions=camera,
            chunk_plucker=plucker,
            chunk_size=int(self.student_config.model.chunk_size),
            chunk_split_strategy=str(self.student_config.model.chunk_split_strategy),
            chunk_index=starts,
        )
        if isinstance(flow, tuple):
            flow = flow[0]
        if hasattr(flow, "sample"):
            flow = flow.sample
        sigma = timestep[:, None, :, None, None] / 1000.0
        prediction = (noisy - sigma * flow).to(flow.dtype)
        loss = self._masked_mse(prediction, target, timestep != 0)
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite ODE loss")

        self.accelerator.backward(loss)
        self.accelerator.unscale_gradients(self.student_optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.student.parameters(),
            float(self.config.train.gradient_clip),
        )
        if not torch.isfinite(grad_norm):
            raise FloatingPointError("Non-finite ODE gradient")
        self.student_optimizer.step()
        return {"ode_loss": float(loss.detach()), "generator_grad_norm": float(grad_norm)}

    def _self_forcing_batch(self, batch):
        latent = batch[0].to(self.accelerator.device, dtype=self.dtype)
        camera = batch[6].to(self.accelerator.device, dtype=self.dtype)
        plucker = batch[-1].to(self.accelerator.device, dtype=self.dtype)
        latent = latent[:, :, : int(self.config.num_latent_frames)]
        camera = camera[:, : int(self.config.num_latent_frames)]
        plucker = plucker[:, :, : int(self.config.num_latent_frames)]
        y, y_mask = self._encode(batch[1])
        negative = [str(self.config.negative_prompt)] * latent.shape[0]
        y_uncond, y_uncond_mask = self._encode(negative, use_chi_prompt=False)
        return latent[:, :, :1], camera, plucker, y, y_mask, y_uncond, y_uncond_mask

    def _train_generator_step(self, values) -> dict[str, float]:
        self.student_optimizer.zero_grad(set_to_none=True)
        first_frame, camera, plucker, y, y_mask, y_uncond, y_uncond_mask = values
        generated = self._rollout(first_frame, camera, plucker, y, y_mask, requires_grad=True)
        batch, _, total_t, _, _ = generated.shape

        timestep = torch.randint(
            0,
            1000,
            (batch, 1),
            device=generated.device,
        ).float()
        shift = float(self.config.timestep_shift)
        timestep = shift * (timestep / 1000.0) / (1.0 + (shift - 1.0) * (timestep / 1000.0)) * 1000.0
        timestep = timestep.clamp(
            min=float(self.config.min_score_timestep),
            max=float(self.config.max_score_timestep),
        )
        timestep = timestep.expand(-1, total_t).clone()
        timestep[:, 0] = 0
        noise = torch.randn_like(generated)
        if self.cp_size > 1:
            cp_group = get_cp_group()
            source = dist.get_global_rank(cp_group, 0)
            dist.broadcast(timestep, src=source, group=cp_group)
            dist.broadcast(noise, src=source, group=cp_group)
        sigma = timestep[:, None, :, None, None] / 1000.0
        noisy = ((1.0 - sigma) * generated + sigma * noise).to(generated.dtype)
        noisy[:, :, :1] = generated[:, :, :1]

        with torch.no_grad():
            fake_flow, meta = self._forward_score_cp(
                self.fake_score,
                noisy,
                timestep,
                y,
                y_mask,
                camera,
                plucker,
            )
            real_flow_cond, _ = self._forward_score_cp(
                self.real_score,
                noisy,
                timestep,
                y,
                y_mask,
                camera,
                plucker,
            )
            real_flow_uncond, _ = self._forward_score_cp(
                self.real_score,
                noisy,
                timestep,
                y_uncond,
                y_uncond_mask,
                camera,
                plucker,
            )
            local_sigma = meta["timestep"][:, None, :, None, None] / 1000.0
            pred_fake = (meta["x"] - local_sigma * fake_flow).to(fake_flow.dtype)
            pred_real_cond = (meta["x"] - local_sigma * real_flow_cond).to(real_flow_cond.dtype)
            pred_real_uncond = (meta["x"] - local_sigma * real_flow_uncond).to(real_flow_uncond.dtype)
            pred_real = pred_real_cond + float(self.config.real_guidance_scale) * (pred_real_cond - pred_real_uncond)

        generated_padded = generated
        if meta["pad"]:
            generated_padded = torch.cat(
                [
                    generated,
                    generated.new_zeros(
                        batch,
                        generated.shape[1],
                        meta["pad"],
                        generated.shape[3],
                        generated.shape[4],
                    ),
                ],
                dim=2,
            )
        generated_local = generated_padded[:, :, meta["local_start"] : meta["local_end"]]
        with torch.no_grad():
            valid = meta["valid"][:, None, :, None, None].float()
            normalizer_error = torch.where(
                valid.bool(),
                (generated_local - pred_real).abs(),
                0.0,
            )
            normalizer_sum = normalizer_error.sum(dim=(1, 2, 3, 4))
            normalizer_count = (
                meta["valid"].float().sum(dim=1)
                * generated_local.shape[1]
                * generated_local.shape[3]
                * generated_local.shape[4]
            )
            if self.cp_size > 1:
                dist.all_reduce(normalizer_sum, group=get_cp_group())
                dist.all_reduce(normalizer_count, group=get_cp_group())
            normalizer = (normalizer_sum / normalizer_count).clamp_min(1e-6)
            dmd_gradient = (pred_fake - pred_real) / normalizer[:, None, None, None, None]
            dmd_gradient = torch.nan_to_num(dmd_gradient)

        target = (generated_local - dmd_gradient).detach()
        frame_mask = meta["valid"].clone()
        if meta["local_start"] == 0:
            frame_mask[:, 0] = 0
        loss = 0.5 * self._masked_mse(generated_local, target, frame_mask)
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite generator loss")

        self.accelerator.backward(loss)
        self.accelerator.unscale_gradients(self.student_optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.student.parameters(),
            float(self.config.train.generator_gradient_clip),
        )
        if not torch.isfinite(grad_norm):
            raise FloatingPointError("Non-finite generator gradient")
        self.student_optimizer.step()
        return {
            "generator_loss": float(loss.detach()),
            "generator_grad_norm": float(grad_norm),
            "dmd_gradient_norm": float(dmd_gradient.abs().mean()),
        }

    def _train_critic_step(self, values) -> dict[str, float]:
        self.fake_optimizer.zero_grad(set_to_none=True)
        first_frame, camera, plucker, y, y_mask, _, _ = values
        with torch.no_grad():
            generated = self._rollout(first_frame, camera, plucker, y, y_mask, requires_grad=False)
        batch, _, total_t, _, _ = generated.shape
        timestep = torch.randint(
            0,
            1000,
            (batch, 1),
            device=generated.device,
        ).float()
        shift = float(self.config.timestep_shift)
        timestep = shift * (timestep / 1000.0) / (1.0 + (shift - 1.0) * (timestep / 1000.0)) * 1000.0
        timestep = timestep.clamp(
            min=float(self.config.min_score_timestep),
            max=float(self.config.max_score_timestep),
        )
        timestep = timestep.expand(-1, total_t).clone()
        timestep[:, 0] = 0
        noise = torch.randn_like(generated)
        if self.cp_size > 1:
            cp_group = get_cp_group()
            source = dist.get_global_rank(cp_group, 0)
            dist.broadcast(timestep, src=source, group=cp_group)
            dist.broadcast(noise, src=source, group=cp_group)
        sigma = timestep[:, None, :, None, None] / 1000.0
        noisy = ((1.0 - sigma) * generated + sigma * noise).to(generated.dtype)
        noisy[:, :, :1] = generated[:, :, :1]

        fake_flow, meta = self._forward_score_cp(
            self.fake_score,
            noisy,
            timestep,
            y,
            y_mask,
            camera,
            plucker,
        )
        generated_padded = generated
        noise_padded = noise
        if meta["pad"]:
            pad_shape = (batch, generated.shape[1], meta["pad"], generated.shape[3], generated.shape[4])
            generated_padded = torch.cat([generated, generated.new_zeros(pad_shape)], dim=2)
            noise_padded = torch.cat([noise, noise.new_zeros(pad_shape)], dim=2)
        local_slice = slice(meta["local_start"], meta["local_end"])
        generated_local = generated_padded[:, :, local_slice]
        noise_local = noise_padded[:, :, local_slice]
        target_flow = noise_local - generated_local
        frame_mask = meta["valid"].clone()
        if meta["local_start"] == 0:
            frame_mask[:, 0] = 0
        loss = self._masked_mse(fake_flow, target_flow, frame_mask)
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite critic loss; model_path must be a distilled generator checkpoint.")

        self.accelerator.backward(loss)
        self.accelerator.unscale_gradients(self.fake_optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.fake_score.parameters(),
            float(self.config.train.critic_gradient_clip),
        )
        if not torch.isfinite(grad_norm):
            raise FloatingPointError("Non-finite critic gradient")
        self.fake_optimizer.step()
        return {"critic_loss": float(loss.detach()), "critic_grad_norm": float(grad_norm)}

    def train(self):
        start_time = time.time()
        max_steps = min(int(self.config.train.max_steps), int(self.config.get("max_iters", 10**18)))
        save_steps = int(self.config.train.save_model_steps)
        save_enabled = not bool(self.config.get("no_save", False))
        log_interval = int(self.config.train.log_interval)
        early_stop_hours = float(self.config.train.early_stop_hours)
        if self.accelerator.is_main_process:
            self.logger.info(
                f"mode={self.mode} start_step={self.global_step} max_steps={max_steps} "
                f"save_enabled={save_enabled} cp_size={self.cp_size} dp_size={self.dp_size}"
            )

        while self.global_step < max_steps:
            batch = self._next_batch()
            if self.mode == "ode":
                logs = self._train_ode_step(batch)
            else:
                values = self._self_forcing_batch(batch)
                logs = {}
                if self.global_step % int(self.config.generator_update_interval) == 0:
                    logs.update(self._train_generator_step(values))
                    values = self._self_forcing_batch(self._next_batch())
                logs.update(self._train_critic_step(values))

            self.global_step += 1
            if self.accelerator.is_main_process and (self.global_step == 1 or self.global_step % log_interval == 0):
                self.logger.info(
                    f"mode={self.mode} step={self.global_step} "
                    + " ".join(f"{key}={value:.6f}" for key, value in logs.items())
                )
                if self.accelerator.trackers:
                    self.accelerator.log(logs, step=self.global_step)

            saved_this_step = save_enabled and save_steps > 0 and self.global_step % save_steps == 0
            if saved_this_step:
                self._save()
            should_stop = early_stop_hours > 0 and time.time() - start_time >= early_stop_hours * 3600
            if dist.is_initialized():
                stop_tensor = torch.tensor(int(should_stop), device=self.accelerator.device)
                dist.all_reduce(stop_tensor, op=dist.ReduceOp.MAX)
                should_stop = bool(stop_tensor.item())
            if should_stop:
                if save_enabled and not saved_this_step:
                    self._save()
                break

        if save_enabled and self.global_step >= max_steps and (save_steps <= 0 or self.global_step % save_steps):
            self._save()
        self.accelerator.end_training()
