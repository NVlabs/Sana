"""Shared utilities for Sol-RL training scripts (SD3, FLUX.1, SANA)."""

import json
import logging
import os
import random
import tempfile
from collections import defaultdict
from functools import wraps

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import wandb
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

_logger = logging.getLogger(__name__)

_TE_IMPORT_ERROR = None
try:
    import transformer_engine.pytorch as te
    from transformer_engine.common.recipe import DelayedScaling, Format

    NVFP4_RECIPE = DelayedScaling(
        fp8_format=Format.E2M1,
        amax_history_len=16,
        amax_compute_algo="max",
    )
    _HAS_TE = True
except (ImportError, OSError, RuntimeError) as exc:
    te = None
    NVFP4_RECIPE = None
    _HAS_TE = False
    _TE_IMPORT_ERROR = exc


def ensure_transformer_engine_available(feature="Transformer Engine"):
    if _HAS_TE:
        return
    detail = f": {_TE_IMPORT_ERROR}" if _TE_IMPORT_ERROR is not None else ""
    raise RuntimeError(
        f"{feature} requires a working `transformer_engine[pytorch]` installation{detail}"
    ) from _TE_IMPORT_ERROR


# ---------------------------------------------------------------------------
# Distributed helpers
# ---------------------------------------------------------------------------


def setup_distributed(rank, local_rank, world_size):
    os.environ["MASTER_ADDR"] = os.getenv("MASTER_ADDR", "localhost")
    os.environ["MASTER_PORT"] = os.getenv("MASTER_PORT", "12355")
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(local_rank)


def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process(rank):
    return rank == 0


def set_seed(seed, rank=0):
    random.seed(seed + rank)
    np.random.seed(seed + rank)
    torch.manual_seed(seed + rank)
    torch.cuda.manual_seed_all(seed + rank)


def gather_tensor_to_all(tensor, world_size):
    tensor = tensor.contiguous()
    gathered = [torch.zeros_like(tensor) for _ in range(world_size)]
    dist.all_gather(gathered, tensor)
    return torch.cat(gathered, dim=0).cpu()


# ---------------------------------------------------------------------------
# Time logging
# ---------------------------------------------------------------------------


class DistributedTimeLogger:
    def __init__(self, device):
        self.device = device
        self.starts = {}
        self.ends = {}
        self.results = defaultdict(float)
        self.is_distributed = dist.is_available() and dist.is_initialized()
        self.rank = dist.get_rank() if self.is_distributed else 0

    def start(self, name):
        torch.cuda.synchronize(self.device)
        ev = torch.cuda.Event(enable_timing=True)
        ev.record()
        self.starts[name] = ev

    def end(self, name):
        if name not in self.starts:
            return
        ev = torch.cuda.Event(enable_timing=True)
        ev.record()
        torch.cuda.synchronize(self.device)
        self.results[name] += self.starts[name].elapsed_time(ev) / 1000.0
        self.ends[name] = ev

    def get_results(self, aggregate=True):
        final = {}
        for name, dur in self.results.items():
            t = torch.tensor([dur], device=self.device)
            if self.is_distributed and aggregate:
                dist.reduce(t, dst=0, op=dist.ReduceOp.MAX)
            if self.rank == 0:
                final[name] = t.item()
        return final if self.rank == 0 else None

    def empty_cache(self):
        self.starts.clear()
        self.ends.clear()
        self.results.clear()


# ---------------------------------------------------------------------------
# JSON / reward trace serialization
# ---------------------------------------------------------------------------


def to_jsonable(value):
    if torch.is_tensor(value):
        if value.numel() == 1:
            return value.item()
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, dict):
        return {k: to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def slice_prompt_metadata(prompt_metadata, prompt_idx):
    if isinstance(prompt_metadata, (list, tuple)):
        if 0 <= prompt_idx < len(prompt_metadata):
            return prompt_metadata[prompt_idx]
        return None
    if isinstance(prompt_metadata, dict):
        sliced = {}
        for k, v in prompt_metadata.items():
            if torch.is_tensor(v) and v.ndim > 0 and prompt_idx < v.shape[0]:
                sliced[k] = v[prompt_idx]
            elif isinstance(v, np.ndarray) and v.ndim > 0 and prompt_idx < v.shape[0]:
                sliced[k] = v[prompt_idx]
            elif isinstance(v, (list, tuple)) and prompt_idx < len(v):
                sliced[k] = v[prompt_idx]
            else:
                sliced[k] = v
        return sliced
    return prompt_metadata


def extract_prompt_reward_group(prompt_idx, prompt_text, prompt_meta, intra_prompt_data_list):
    rollouts, rollout_idx = [], 0
    for chunk_idx, sample_item in enumerate(intra_prompt_data_list):
        rewards_dict = sample_item["rewards"]
        if "avg" not in rewards_dict:
            raise KeyError("Expected reward key 'avg' not found in rewards dict.")
        chunk_size = int(rewards_dict["avg"].shape[0])
        for row_idx in range(chunk_size):
            rollouts.append(
                {
                    "rollout_idx": rollout_idx,
                    "chunk_idx": chunk_idx,
                    "idx_in_chunk": row_idx,
                    "rewards": {rn: float(rt[row_idx].detach().cpu().item()) for rn, rt in rewards_dict.items()},
                }
            )
            rollout_idx += 1
    return {
        "prompt_idx_local": int(prompt_idx),
        "prompt_text": str(prompt_text),
        "prompt_metadata": to_jsonable(prompt_meta),
        "num_rollouts": len(rollouts),
        "rollouts": rollouts,
    }


def save_step_reward_groups(config, global_step, epoch, rank, world_size, prompt_reward_groups):
    reward_trace_dir = os.path.join(config.save_dir, "reward_traces")
    os.makedirs(reward_trace_dir, exist_ok=True)
    payload = {
        "global_step": int(global_step),
        "epoch": int(epoch),
        "rank": int(rank),
        "world_size": int(world_size),
        "num_prompt_groups": len(prompt_reward_groups),
        "total_rollouts": int(sum(g["num_rollouts"] for g in prompt_reward_groups)),
        "prompt_reward_groups": prompt_reward_groups,
    }
    output_path = os.path.join(reward_trace_dir, f"step_{int(global_step):08d}_rank_{int(rank)}.json")
    tmp = output_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=True, indent=2)
    os.replace(tmp, output_path)


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------


def return_decay(step, decay_type, custom_decay_step=0, custom_decay_value=0.0):
    if decay_type == 0:
        flat, uprate, uphold = 0, 0.0, 0.0
    elif decay_type == 1:
        flat, uprate, uphold = 0, 0.001, 0.5
    elif decay_type == 2:
        flat, uprate, uphold = 75, 0.0075, 0.999
    elif decay_type == 3:
        assert custom_decay_step > 0 and custom_decay_value > 0, (
            f"decay_type=3 requires custom_decay_step>0 and custom_decay_value>0, "
            f"got step={custom_decay_step}, value={custom_decay_value}"
        )
        flat, uprate, uphold = 0, custom_decay_value / custom_decay_step, custom_decay_value
    else:
        raise ValueError(f"Unsupported decay_type={decay_type}")
    if step < flat:
        return 0.0
    return min((step - flat) * uprate, uphold)


def calculate_zero_std_ratio(prompts, gathered_rewards):
    prompt_array = np.array(prompts)
    unique_prompts, inverse_indices, counts = np.unique(prompt_array, return_inverse=True, return_counts=True)
    if len(unique_prompts) == 0:
        return 0.0, 0.0
    grouped_rewards = gathered_rewards["avg"][np.argsort(inverse_indices), 0]
    split_indices = np.cumsum(counts)[:-1]
    reward_groups = np.split(grouped_rewards, split_indices)
    prompt_std_devs = np.array([np.std(group) for group in reward_groups])
    zero_std_count = np.count_nonzero(prompt_std_devs == 0)
    return zero_std_count / len(prompt_std_devs), float(prompt_std_devs.mean())


# ---------------------------------------------------------------------------
# Collation / filtering
# ---------------------------------------------------------------------------


def collate_dict_items(items):
    if not items:
        return {}
    return {
        key: (
            torch.cat([it[key] for it in items], dim=0)
            if not isinstance(items[0][key], dict)
            else {k2: torch.cat([it[key][k2] for it in items], dim=0) for k2 in items[0][key]}
        )
        for key in items[0].keys()
    }


def filter_by_indices(collated_samples, keep_indices):
    filtered = {}
    for key, value in collated_samples.items():
        if isinstance(value, torch.Tensor):
            filtered[key] = value[keep_indices]
        elif isinstance(value, dict):
            filtered[key] = {sk: (sv[keep_indices] if isinstance(sv, torch.Tensor) else sv) for sk, sv in value.items()}
        else:
            filtered[key] = value
    return filtered


def select_indices_by_mode(rewards, target_count, mode):
    total = rewards.shape[0]
    target_count = max(1, min(int(target_count), total))
    mode = str(mode).lower()

    if mode == "random":
        return torch.randperm(total, device=rewards.device)[:target_count]

    if mode == "mean_deviation":
        return torch.topk(torch.abs(rewards - rewards.mean()), target_count, largest=True).indices

    if mode == "extremes_random":
        if total == 1:
            return torch.tensor([0], dtype=torch.long, device=rewards.device)
        extremes = torch.unique(torch.stack([torch.argmax(rewards), torch.argmin(rewards)]))
        if target_count <= extremes.numel():
            return extremes[:target_count]
        mask = torch.ones(total, dtype=torch.bool, device=rewards.device)
        mask[extremes] = False
        remaining = torch.nonzero(mask, as_tuple=False).squeeze(1)
        rk = min(target_count - extremes.numel(), remaining.numel())
        return torch.cat([extremes, remaining[torch.randperm(remaining.numel(), device=rewards.device)[:rk]]])

    n_best = target_count // 2
    n_worst = target_count - n_best
    best = (
        torch.topk(rewards, n_best, largest=True).indices
        if n_best > 0
        else torch.empty(0, dtype=torch.long, device=rewards.device)
    )
    worst = (
        torch.topk(rewards, n_worst, largest=False).indices
        if n_worst > 0
        else torch.empty(0, dtype=torch.long, device=rewards.device)
    )
    return torch.cat([best, worst])


# ---------------------------------------------------------------------------
# Debug images
# ---------------------------------------------------------------------------


def save_debug_image_subset(images, prompts, save_root, prefix, resolution, rewards=None, max_images=6):
    os.makedirs(save_root, exist_ok=True)
    for idx in range(min(int(max_images), len(images))):
        pil = Image.fromarray((images[idx].float().cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8))
        pil = pil.resize((resolution, resolution))
        ps = str(prompts[idx]).replace("/", "_").replace("\\", "_").replace(":", "_")[:60]
        if rewards is not None and len(rewards) > idx:
            fn = f"{prefix}_{idx:02d}_r{float(rewards[idx]):.3f}_{ps}.jpg"
        else:
            fn = f"{prefix}_{idx:02d}_{ps}.jpg"
        pil.save(os.path.join(save_root, fn))


# ---------------------------------------------------------------------------
# Compile / LoRA / NVFP4 helpers
# ---------------------------------------------------------------------------


def unwrap_compiled(model):
    return model._orig_mod if hasattr(model, "_orig_mod") else model


@torch.no_grad()
def sync_lora_to_inference(peft_model, inference_model, adapter_name="old"):
    synced = 0
    for name, module in peft_model.base_model.model.named_modules():
        if not hasattr(module, "lora_A") or adapter_name not in module.lora_A:
            continue
        base_weight = module.base_layer.weight.data
        lora_A = module.lora_A[adapter_name].weight.data
        lora_B = module.lora_B[adapter_name].weight.data
        scaling = module.scaling[adapter_name]
        merged = base_weight + scaling * (lora_B @ lora_A)

        target = inference_model
        for part in name.split("."):
            target = getattr(target, part)
        target.weight.data.copy_(merged)
        synced += 1
    return synced


def wrap_forward_with_fp8(module):
    ensure_transformer_engine_available("NVFP4 quantization")
    original_forward = module.forward

    @wraps(original_forward)
    def wrapped(*args, **kwargs):
        with te.fp8_autocast(enabled=True, fp8_recipe=NVFP4_RECIPE):
            return original_forward(*args, **kwargs)

    module.forward = wrapped
    return original_forward


class BF16TELinear(nn.Module):
    """Wrapper around te.Linear that casts input to bfloat16."""

    def __init__(self, te_linear):
        super().__init__()
        self.te_linear = te_linear

    @property
    def weight(self):
        return self.te_linear.weight

    @property
    def bias(self):
        return self.te_linear.bias

    @property
    def in_features(self):
        return self.te_linear.in_features

    @property
    def out_features(self):
        return self.te_linear.out_features

    def forward(self, x):
        return self.te_linear(x.to(torch.bfloat16))


def replace_linear_with_te(model, skip_modules=None, min_dim=0, _prefix=""):
    ensure_transformer_engine_available("NVFP4 quantization")
    if skip_modules is None:
        skip_modules = []
    replaced = skipped = 0
    replaced_details, skipped_details = [], []

    for name, child in list(model.named_children()):
        fqn = f"{_prefix}.{name}" if _prefix else name

        if isinstance(child, nn.Linear):
            in_feat, out_feat = child.in_features, child.out_features
            has_bias = child.bias is not None

            skip_reason = None
            if any(pat in fqn for pat in skip_modules):
                skip_reason = "name_pattern"
            elif min_dim > 0 and max(in_feat, out_feat) <= min_dim:
                skip_reason = f"small_dim(<={min_dim})"

            info = (fqn, in_feat, out_feat, has_bias, skip_reason)
            if skip_reason:
                skipped += 1
                skipped_details.append(info)
            else:
                te_lin = te.Linear(in_feat, out_feat, bias=has_bias).to(
                    device=child.weight.device, dtype=child.weight.dtype
                )
                with torch.no_grad():
                    te_lin.weight.copy_(child.weight)
                    if has_bias and te_lin.bias is not None:
                        te_lin.bias.copy_(child.bias)
                setattr(model, name, BF16TELinear(te_lin))
                replaced += 1
                replaced_details.append(info)
        else:
            r, s, rd, sd = replace_linear_with_te(child, skip_modules, min_dim=min_dim, _prefix=fqn)
            replaced += r
            skipped += s
            replaced_details.extend(rd)
            skipped_details.extend(sd)

    return replaced, skipped, replaced_details, skipped_details


# ---------------------------------------------------------------------------
# Checkpoint save / resume
# ---------------------------------------------------------------------------


def save_ckpt(save_dir, transformer_ddp, global_step, rank, ema, config, optimizer, scaler):
    """Save LoRA adapters, EMA, optimizer and scaler to a checkpoint directory."""
    if not is_main_process(rank):
        return
    save_root = os.path.join(save_dir, "checkpoints", f"checkpoint-{global_step}")
    model_to_save = unwrap_compiled(transformer_ddp.module)
    save_root_lora = os.path.join(save_root, "lora")
    os.makedirs(save_root_lora, exist_ok=True)
    model_to_save.save_pretrained(save_root_lora)
    if getattr(config.train, "ema", False) and ema is not None:
        torch.save(ema.state_dict(), os.path.join(save_root, "ema.pt"))
    torch.save(optimizer.state_dict(), os.path.join(save_root, "optimizer.pt"))
    if scaler is not None:
        torch.save(scaler.state_dict(), os.path.join(save_root, "scaler.pt"))
    _logger.info("Saved checkpoint to %s", save_root)


def find_resume_candidates(config):
    """Return a list of ``(step, path)`` checkpoint candidates sorted by step descending."""
    if not (getattr(config, "resume_from", None) and getattr(config, "resume", False)):
        return []
    ckpt_dir = os.path.join(config.resume_from, "checkpoints")
    if not os.path.exists(ckpt_dir):
        _logger.warning("Resume path %s does not exist. Starting from scratch.", ckpt_dir)
        os.makedirs(ckpt_dir, exist_ok=True)
        return []
    candidates = []
    try:
        for d in os.listdir(ckpt_dir):
            full = os.path.join(ckpt_dir, d)
            if d.startswith("checkpoint-") and os.path.isdir(full):
                try:
                    candidates.append((int(d.split("-")[-1]), full))
                except ValueError:
                    continue
        candidates.sort(key=lambda x: x[0], reverse=True)
    except Exception as e:
        _logger.warning("Error searching for checkpoints: %s", e)
    try:
        if getattr(config, "resume_path", None):
            explicit_step = int(os.path.basename(config.resume_path).split("-")[-1])
            candidates.insert(0, (explicit_step, config.resume_path))
    except Exception:
        pass
    return candidates


def resume_from_checkpoint(candidates, peft_model, ema, optimizer, scaler, device):
    """Try to load training state from *candidates* (output of :func:`find_resume_candidates`).

    *peft_model* should be the unwrapped PeftModel
    (use ``unwrap_compiled(transformer_ddp.module)`` when the module may be compiled).

    Returns ``(global_step, resumed)`` where *resumed* is ``True`` on success.
    """
    for ckpt_step, ckpt_path in candidates:
        try:
            _logger.info("Attempting to resume from %s (step %d)", ckpt_path, ckpt_step)
            lora_path = os.path.join(ckpt_path, "lora")
            lora_path_old = os.path.join(lora_path, "old")
            peft_model.load_adapter(lora_path, adapter_name="default", is_trainable=True)
            peft_model.load_adapter(lora_path_old, adapter_name="old", is_trainable=False)
            ema_path = os.path.join(ckpt_path, "ema.pt")
            if os.path.exists(ema_path) and ema is not None:
                ema.load_state_dict(torch.load(ema_path, map_location=device))
            opt_path = os.path.join(ckpt_path, "optimizer.pt")
            if os.path.exists(opt_path):
                optimizer.load_state_dict(torch.load(opt_path, map_location=device))
            scaler_path = os.path.join(ckpt_path, "scaler.pt")
            if os.path.exists(scaler_path) and scaler is not None:
                scaler.load_state_dict(torch.load(scaler_path, map_location=device))
            _logger.info("Successfully resumed from step %d", ckpt_step)
            return ckpt_step, True
        except Exception as e:
            _logger.warning("Failed to load checkpoint %s: %s. Trying next...", ckpt_path, e)
            continue
    if candidates:
        _logger.warning("All checkpoints failed to load. Starting from scratch.")
    return 0, False


# ---------------------------------------------------------------------------
# Rollout image logging
# ---------------------------------------------------------------------------


def log_rollout_images(images, prompts, rewards, config, global_step, rank, max_wandb_images=12):
    """Save debug images to disk and log to wandb during rollout."""
    debug_image_every_steps = max(1, int(getattr(config, "debug_image_every_steps", 10)))
    enable_debug_image_save = bool(getattr(config, "enable_debug_image_save", True))
    if not (
        enable_debug_image_save
        and is_main_process(rank)
        and images is not None
        and global_step % debug_image_every_steps == 0
    ):
        return
    images_to_log = images.cpu()
    num_to_log = min(max_wandb_images, len(images_to_log))
    rollout_debug_dir = os.path.join(
        config.save_dir,
        "debug_images",
        "rollout",
        f"step_{global_step}",
    )
    save_debug_image_subset(
        images=images_to_log,
        prompts=prompts,
        save_root=rollout_debug_dir,
        prefix="rollout",
        resolution=config.resolution,
        rewards=rewards,
        max_images=getattr(config, "debug_image_subset_size", 6),
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        for idx in range(num_to_log):
            image = images_to_log[idx].float()
            pil = Image.fromarray((image.numpy().transpose(1, 2, 0) * 255).astype(np.uint8))
            pil = pil.resize((config.resolution, config.resolution))
            pil.save(os.path.join(tmpdir, f"{idx}.jpg"))
        wandb.log(
            {
                "images": [
                    wandb.Image(
                        os.path.join(tmpdir, f"{idx}.jpg"),
                        caption=f"{prompts[idx]:.100} | avg: {rewards[idx]:.2f}",
                    )
                    for idx in range(num_to_log)
                ],
            },
            commit=False,
        )


# ---------------------------------------------------------------------------
# Dataset / dataloader construction
# ---------------------------------------------------------------------------


def build_datasets_and_loaders(config, world_size, rank):
    """Build train/test datasets with distributed samplers and dataloaders.

    Returns ``(train_dataset, train_dataloader, train_sampler, test_dataset, test_dataloader)``.
    """
    from diffusion.post_training.prompt_dataset import (
        DistributedKRepeatSampler,
        GenevalPromptDataset,
        TextPromptDataset,
    )

    if config.prompt_fn == "general_ocr":
        train_dataset = TextPromptDataset(config.dataset, "train")
        test_dataset = TextPromptDataset(config.dataset, "test")
    elif config.prompt_fn == "geneval":
        train_dataset = GenevalPromptDataset(config.dataset, "train")
        test_dataset = GenevalPromptDataset(config.dataset, "test")
    else:
        raise NotImplementedError(f"Unsupported prompt_fn={config.prompt_fn}")

    train_sampler = DistributedKRepeatSampler(
        dataset=train_dataset,
        batch_size=config.sample.per_gpu_to_process_prompts,
        k=1,
        num_replicas=world_size,
        rank=rank,
        seed=config.seed,
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        num_workers=0,
        collate_fn=train_dataset.collate_fn,
        pin_memory=True,
    )
    test_sampler = (
        DistributedSampler(test_dataset, num_replicas=world_size, rank=rank, shuffle=False) if world_size > 1 else None
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=config.sample.test_batch_size,
        sampler=test_sampler,
        collate_fn=test_dataset.collate_fn,
        num_workers=0,
        pin_memory=True,
    )
    return train_dataset, train_dataloader, train_sampler, test_dataset, test_dataloader
