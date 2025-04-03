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

import os
import random
import re

import numpy as np
import torch
from accelerate.state import DistributedType

from diffusion.utils.logger import get_root_logger
from tools.download import find_model


def save_checkpoint(
    work_dir,
    epoch,
    model,
    accelerator=None,
    model_ema=None,
    optimizer=None,
    lr_scheduler=None,
    generator=torch.Generator(device="cpu").manual_seed(42),
    keep_last=False,
    step=None,
    add_symlink=False,
    add_suffix=None,
):
    if accelerator is not None and accelerator.distributed_type == DistributedType.FSDP:
        return save_checkpoint_fsdp(
            work_dir=work_dir,
            epoch=epoch,
            accelerator=accelerator,
            lr_scheduler=lr_scheduler,
            generator=generator,
            keep_last=keep_last,
            step=step,
            add_symlink=add_symlink,
            add_suffix=add_suffix,
        )
    else:
        return save_checkpoint_ddp(
            work_dir=work_dir,
            epoch=epoch,
            model=model,
            model_ema=model_ema,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            generator=generator,
            keep_last=keep_last,
            step=step,
            add_symlink=add_symlink,
            add_suffix=add_suffix,
        )


def save_checkpoint_ddp(
    work_dir,
    epoch,
    model,
    model_ema=None,
    optimizer=None,
    lr_scheduler=None,
    generator=torch.Generator(device="cpu").manual_seed(42),
    keep_last=False,
    step=None,
    add_symlink=False,
    add_suffix=None,
):
    os.makedirs(work_dir, exist_ok=True)
    state_dict = dict(state_dict=model.state_dict())
    if model_ema is not None:
        state_dict["state_dict_ema"] = model_ema.state_dict()
    if optimizer is not None:
        state_dict["optimizer"] = optimizer.state_dict()
    if lr_scheduler is not None:
        state_dict["scheduler"] = lr_scheduler.state_dict()
    if epoch is not None:
        state_dict["epoch"] = epoch
        file_path = os.path.join(work_dir, f"epoch_{epoch}.pth")
        if step is not None:
            file_path = file_path.split(".pth")[0] + f"_step_{step}.pth"
    if add_suffix is not None:
        file_path = file_path.replace(".pth", f"_{add_suffix}.pth")
    rng_state = {
        "torch": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all(),
        "numpy": np.random.get_state(),
        "python": random.getstate(),
        "generator": generator.get_state(),
    }
    state_dict["rng_state"] = rng_state

    logger = get_root_logger()
    torch.save(state_dict, file_path)
    logger.info(f"Saved checkpoint of epoch {epoch} to {file_path.format(epoch)}.")
    if keep_last:
        for i in range(epoch):
            previous_ckgt = file_path.format(i)
            if os.path.exists(previous_ckgt):
                os.remove(previous_ckgt)
    if add_symlink:
        link_path = os.path.join(os.path.dirname(file_path), "latest.pth")
        if os.path.exists(link_path) or os.path.islink(link_path):
            os.remove(link_path)
        os.symlink(os.path.abspath(file_path), link_path)

    return file_path


def save_checkpoint_fsdp(
    work_dir,
    epoch,
    accelerator=None,
    lr_scheduler=None,
    generator=torch.Generator(device="cpu").manual_seed(42),
    keep_last=False,
    step=None,
    add_symlink=False,
    add_suffix=None,
):
    """FSDP checkpoint save function, sharding"""
    logger = get_root_logger()

    checkpoint_dir = os.path.join(work_dir, f"epoch_{epoch}")
    if step is not None:
        checkpoint_dir = checkpoint_dir + f"_step_{step}"
    if add_suffix is not None:
        checkpoint_dir = checkpoint_dir + f"_{add_suffix}"
    os.makedirs(checkpoint_dir, exist_ok=True)

    model_dir = os.path.join(checkpoint_dir, "model")
    os.makedirs(model_dir, exist_ok=True)

    accelerator.save_state(model_dir)

    if accelerator.is_main_process:
        metadata = dict()
        rng_state = {
            "torch": torch.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state_all(),
            "numpy": np.random.get_state(),
            "python": random.getstate(),
            "generator": generator.get_state(),
        }
        metadata["rng_state"] = rng_state
        if lr_scheduler is not None:
            metadata["scheduler"] = lr_scheduler.state_dict()
        if epoch is not None:
            metadata["epoch"] = epoch

        torch.save(metadata, os.path.join(checkpoint_dir, "metadata.pth"))

        if keep_last:
            checkpoints = sorted(
                [d for d in os.listdir(work_dir) if os.path.isdir(os.path.join(work_dir, d)) and d.startswith("epoch_")]
            )
            for old_ckpt in checkpoints[:-1]:
                old_path = os.path.join(work_dir, old_ckpt)
                if os.path.exists(old_path):
                    import shutil

                    shutil.rmtree(old_path)

        if add_symlink:
            link_path = os.path.join(work_dir, "latest.pth")
            if os.path.exists(link_path) or os.path.islink(link_path):
                os.remove(link_path)
            os.symlink(os.path.abspath(checkpoint_dir), link_path)

        logger.info(f"Saved checkpoint to {checkpoint_dir}")

        # add model symlink
        model_link_path = checkpoint_dir + ".pth"
        state_dict = torch.load(os.path.join(model_dir, "pytorch_model_fsdp.bin"), map_location="cpu")
        torch.save({"state_dict": state_dict}, model_link_path)

    accelerator.wait_for_everyone()
    return checkpoint_dir


def load_checkpoint(
    checkpoint,
    model,
    model_ema=None,
    optimizer=None,
    lr_scheduler=None,
    load_ema=False,
    resume_optimizer=True,
    resume_lr_scheduler=True,
    null_embed_path=None,
    FSDP=False,
):
    if FSDP:
        return load_checkpoint_fsdp(
            checkpoint=checkpoint,
            model=model,
        )
    else:
        return load_checkpoint_ddp(
            checkpoint=checkpoint,
            model=model,
            model_ema=model_ema,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            load_ema=load_ema,
            resume_optimizer=resume_optimizer,
            resume_lr_scheduler=resume_lr_scheduler,
            null_embed_path=null_embed_path,
        )


def load_checkpoint_ddp(
    checkpoint,
    model,
    model_ema=None,
    optimizer=None,
    lr_scheduler=None,
    load_ema=False,
    resume_optimizer=True,
    resume_lr_scheduler=True,
    null_embed_path=None,
):
    assert isinstance(checkpoint, str)
    logger = get_root_logger()
    ckpt_file = checkpoint
    checkpoint = find_model(ckpt_file)

    state_dict_keys = ["pos_embed", "base_model.pos_embed", "model.pos_embed"]
    for key in state_dict_keys:
        if key in checkpoint["state_dict"]:
            del checkpoint["state_dict"][key]
            if "state_dict_ema" in checkpoint and key in checkpoint["state_dict_ema"]:
                del checkpoint["state_dict_ema"][key]
            break

    if load_ema:
        state_dict = checkpoint["state_dict_ema"]
    else:
        state_dict = checkpoint.get("state_dict", checkpoint)  # to be compatible with the official checkpoint

    null_embed = torch.load(null_embed_path, map_location="cpu")
    state_dict["y_embedder.y_embedding"] = null_embed["uncond_prompt_embeds"][0]
    rng_state = checkpoint.get("rng_state", None)

    missing, unexpect = model.load_state_dict(state_dict, strict=False)
    if model_ema is not None:
        model_ema.load_state_dict(checkpoint["state_dict_ema"], strict=False)
    if optimizer is not None and resume_optimizer:
        optimizer.load_state_dict(checkpoint["optimizer"])
    if lr_scheduler is not None and resume_lr_scheduler:
        lr_scheduler.load_state_dict(checkpoint["scheduler"])

    epoch = 0
    if optimizer is not None and resume_optimizer:
        epoch_match = re.search(r"epoch_(\d+)", ckpt_file)
        epoch = int(epoch_match.group(1)) if epoch_match else 0
        logger.info(
            f"Resume checkpoint of epoch {epoch} from {ckpt_file}. Load ema: {load_ema}, "
            f"resume optimizer： {resume_optimizer}, resume lr scheduler: {resume_lr_scheduler}."
        )
        return epoch, missing, unexpect, rng_state
    logger.info(f"Load checkpoint from {ckpt_file}. Load ema: {load_ema}.")
    return epoch, missing, unexpect, None


def load_checkpoint_fsdp(
    checkpoint,
    model,
):
    assert isinstance(checkpoint, str)
    logger = get_root_logger()

    # 1 load model
    if ".pth" in checkpoint:
        state_dict_model = find_model(checkpoint)
        state_dict_model = state_dict_model.get("state_dict", state_dict_model)
    else:
        if os.path.isfile(checkpoint):
            checkpoint = os.path.dirname(checkpoint)
        assert os.path.isdir(checkpoint), f"Checkpoint directory {checkpoint} does not exist!"

        state_dict_model = find_model(os.path.join(checkpoint, "model", "pytorch_model_fsdp.bin"))

    state_dict_keys = ["pos_embed", "base_model.pos_embed", "model.pos_embed"]
    for key in state_dict_keys:
        if key in state_dict_model:
            del state_dict_model[key]
            break

    missing, unexpect = model.load_state_dict(state_dict_model, strict=False)
    logger.info(f"Load checkpoint of {checkpoint}.")

    return None, missing, unexpect, None
