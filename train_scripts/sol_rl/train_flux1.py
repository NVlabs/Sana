import logging
import os
import random
import sys
import tempfile
import time
from collections import defaultdict
from concurrent import futures

_rank = int(os.environ.get("RANK", 0))
_cache_root = os.environ.get("CACHE_ROOT", os.path.expanduser("~/.cache/sol_rl"))
os.environ.setdefault("TRITON_CACHE_DIR", f"{_cache_root}/triton/rank_{_rank}")
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", f"{_cache_root}/torchinductor/rank_{_rank}")
os.environ.setdefault("TORCHINDUCTOR_FX_GRAPH_CACHE", "1")

import warnings

warnings.filterwarnings("ignore", message=".*truncated.*")

import transformers

transformers.logging.set_verbosity_error()

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import tqdm
import wandb
from absl import app, flags
from diffusers import FluxPipeline
from diffusers.models import FluxTransformer2DModel
from diffusers.pipelines.flux.pipeline_flux import retrieve_timesteps
from ml_collections import config_flags
from peft import LoraConfig, PeftModel, get_peft_model
from PIL import Image
from torch.cuda.amp import GradScaler
from torch.cuda.amp import autocast as torch_autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from train_utils import (
    _HAS_TE,
    NVFP4_RECIPE,
    DistributedTimeLogger,
    build_datasets_and_loaders,
    calculate_zero_std_ratio,
    cleanup_distributed,
    collate_dict_items,
    extract_prompt_reward_group,
    filter_by_indices,
    find_resume_candidates,
    gather_tensor_to_all,
    is_main_process,
    log_rollout_images,
    replace_linear_with_te,
    resume_from_checkpoint,
    return_decay,
    save_ckpt,
    save_debug_image_subset,
    save_step_reward_groups,
    select_indices_by_mode,
    set_seed,
    setup_distributed,
    slice_prompt_metadata,
    sync_lora_to_inference,
    te,
    unwrap_compiled,
    wrap_forward_with_fp8,
)

import diffusion.flow_grpo.rewards
from diffusion.flow_grpo.diffusers_patch.solver import run_sampling
from diffusion.flow_grpo.ema import EMAModuleWrapper
from diffusion.flow_grpo.stat_tracking import PerPromptStatTracker

tqdm = tqdm.tqdm
FLAGS = flags.FLAGS
config_flags.DEFINE_config_file(
    "config",
    "configs/sol_rl/flux1.py",
    "Training configuration.",
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

FLUX_NVFP4_DEFAULT_SKIP_MODULES = [
    "x_embedder",
    "context_embedder",
    "time_text_embed",
    "norm_out",
]

TEXT_ENCODER_MAX_SEQ_LEN = 128
TOKENIZER_MAX_LENGTH = 256
WANDB_MAX_LOG_IMAGES = 12


def calculate_shift(
    image_seq_len,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
):
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    return image_seq_len * m + b


def compute_text_embeddings(prompts, pipeline, max_sequence_length, device):
    with torch.no_grad():
        prompt_embeds, pooled_prompt_embeds, text_ids = pipeline.encode_prompt(
            prompt=prompts,
            prompt_2=prompts,
            prompt_embeds=None,
            pooled_prompt_embeds=None,
            device=device,
            num_images_per_prompt=1,
            max_sequence_length=max_sequence_length,
            lora_scale=None,
        )
    return prompt_embeds.to(device), pooled_prompt_embeds.to(device), text_ids.to(device)


def _build_generators_from_seeds(seed_list, device):
    return [torch.Generator(device=device).manual_seed(int(seed)) for seed in seed_list]


@torch.no_grad()
def pipeline_with_logprob_flux(
    pipeline,
    prompt=None,
    prompt_2=None,
    height=None,
    width=None,
    num_inference_steps=28,
    guidance_scale=3.5,
    num_images_per_prompt=1,
    generator=None,
    latents=None,
    prompt_embeds=None,
    pooled_prompt_embeds=None,
    text_ids=None,
    output_type="pt",
    joint_attention_kwargs=None,
    max_sequence_length=512,
    noise_level=0.7,
    deterministic=False,
    solver="flow",
    sequential_decode=False,
):
    height = height or pipeline.default_sample_size * pipeline.vae_scale_factor
    width = width or pipeline.default_sample_size * pipeline.vae_scale_factor

    pipeline.check_inputs(
        prompt,
        prompt_2,
        height,
        width,
        prompt_embeds=prompt_embeds,
        pooled_prompt_embeds=pooled_prompt_embeds,
        max_sequence_length=max_sequence_length,
    )

    if prompt is not None and isinstance(prompt, str):
        batch_size = 1
    elif prompt is not None and isinstance(prompt, list):
        batch_size = len(prompt)
    else:
        batch_size = prompt_embeds.shape[0]

    device = pipeline._execution_device
    lora_scale = joint_attention_kwargs.get("scale", None) if joint_attention_kwargs is not None else None

    if prompt_embeds is None or pooled_prompt_embeds is None or text_ids is None:
        prompt_embeds, pooled_prompt_embeds, text_ids = pipeline.encode_prompt(
            prompt=prompt,
            prompt_2=prompt_2,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
            lora_scale=lora_scale,
        )

    num_channels_latents = pipeline.transformer.config.in_channels // 4
    if latents is None:
        latents, latent_image_ids = pipeline.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )
    else:
        latents = latents.to(device)
        latent_image_ids = pipeline._prepare_latent_image_ids(
            batch_size * num_images_per_prompt,
            height // pipeline.vae_scale_factor,
            width // pipeline.vae_scale_factor,
            device,
            prompt_embeds.dtype,
        )

    sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
    if hasattr(pipeline.scheduler.config, "use_flow_sigmas") and pipeline.scheduler.config.use_flow_sigmas:
        sigmas = None

    image_seq_len = latents.shape[1]
    mu = calculate_shift(
        image_seq_len,
        pipeline.scheduler.config.get("base_image_seq_len", 256),
        pipeline.scheduler.config.get("max_image_seq_len", 4096),
        pipeline.scheduler.config.get("base_shift", 0.5),
        pipeline.scheduler.config.get("max_shift", 1.15),
    )
    _, num_inference_steps = retrieve_timesteps(
        pipeline.scheduler,
        num_inference_steps,
        device,
        sigmas=sigmas,
        mu=mu,
    )
    sigmas = pipeline.scheduler.sigmas.float()

    active_transformer = pipeline.transformer
    guidance_config = unwrap_compiled(active_transformer).config
    if guidance_config.guidance_embeds:
        guidance = torch.full([1], guidance_scale, device=device, dtype=torch.float32).expand(latents.shape[0])
    else:
        guidance = None

    def v_pred_fn(z, sigma):
        timestep = torch.full([z.shape[0]], float(sigma), device=z.device, dtype=z.dtype)
        noise_pred = active_transformer(
            hidden_states=z,
            timestep=timestep,
            guidance=guidance,
            pooled_projections=pooled_prompt_embeds,
            encoder_hidden_states=prompt_embeds,
            txt_ids=text_ids,
            img_ids=latent_image_ids,
            joint_attention_kwargs=joint_attention_kwargs,
            return_dict=False,
        )[0]
        return noise_pred

    all_latents = [latents]
    latents, all_latents, all_log_probs = run_sampling(v_pred_fn, latents, sigmas, solver, deterministic, noise_level)

    latents = pipeline._unpack_latents(latents, height, width, pipeline.vae_scale_factor)
    latents = (latents / pipeline.vae.config.scaling_factor) + pipeline.vae.config.shift_factor
    latents = latents.to(dtype=pipeline.vae.dtype)

    if sequential_decode and latents.shape[0] > 1:
        decoded_batches = []
        for idx in range(latents.shape[0]):
            decoded_batches.append(pipeline.vae.decode(latents[idx : idx + 1], return_dict=False)[0])
        image = torch.cat(decoded_batches, dim=0)
    else:
        image = pipeline.vae.decode(latents, return_dict=False)[0]

    image = pipeline.image_processor.postprocess(image, output_type=output_type)
    pipeline.maybe_free_model_hooks()

    return image, all_latents, latent_image_ids, text_ids, all_log_probs


def eval_fn(
    pipeline,
    test_dataloader,
    text_encoders,
    tokenizers,
    config,
    device,
    rank,
    world_size,
    global_step,
    reward_fn,
    executor,
    mixed_precision_dtype,
    ema,
    transformer_trainable_parameters,
):
    set_seed(config.seed + 1_000_000, rank)

    sequential_decode = bool(getattr(config, "sequential_decode", True))
    if config.train.ema and ema is not None:
        ema.copy_ema_to(transformer_trainable_parameters, store_temp=True)

    pipeline.transformer.eval()
    all_rewards = defaultdict(list)
    test_sampler = (
        DistributedSampler(test_dataloader.dataset, num_replicas=world_size, rank=rank, shuffle=False)
        if world_size > 1
        else None
    )
    eval_loader = DataLoader(
        test_dataloader.dataset,
        batch_size=config.sample.test_batch_size,
        sampler=test_sampler,
        collate_fn=test_dataloader.collate_fn,
        num_workers=test_dataloader.num_workers,
    )

    for prompts, prompt_metadata in tqdm(
        eval_loader,
        desc="Eval",
        disable=not is_main_process(rank),
        dynamic_ncols=True,
    ):
        prompt_embeds, pooled_prompt_embeds, text_ids = compute_text_embeddings(
            prompts, pipeline, max_sequence_length=TEXT_ENCODER_MAX_SEQ_LEN, device=device
        )
        len(prompts)
        with torch_autocast(enabled=(config.mixed_precision in ["fp16", "bf16"]), dtype=mixed_precision_dtype):
            with torch.no_grad():
                images, _, _, _, _ = pipeline_with_logprob_flux(
                    pipeline,
                    prompt_embeds=prompt_embeds,
                    pooled_prompt_embeds=pooled_prompt_embeds,
                    text_ids=text_ids,
                    num_inference_steps=config.sample.eval_num_steps,
                    guidance_scale=config.eval_sample_guidance_scale,
                    output_type="pt",
                    height=config.resolution,
                    width=config.resolution,
                    noise_level=config.sample.noise_level,
                    deterministic=True,
                    solver=config.sample.solver,
                    sequential_decode=sequential_decode,
                )
        rewards_future = executor.submit(reward_fn, images, prompts, prompt_metadata, only_strict=False)
        time.sleep(0)
        rewards, _ = rewards_future.result()
        for key, value in rewards.items():
            rewards_tensor = torch.as_tensor(value, device=device).float()
            all_rewards[key].append(gather_tensor_to_all(rewards_tensor, world_size).numpy())

    enable_debug_image_save = bool(getattr(config, "enable_debug_image_save", True))
    if is_main_process(rank):
        final_rewards = {key: np.concatenate(value_list) for key, value_list in all_rewards.items()}
        images_to_log = images.cpu()
        prompts_to_log = prompts
        if enable_debug_image_save:
            eval_debug_dir = os.path.join(config.save_dir, "debug_images", "eval", f"step_{global_step}")
            save_debug_image_subset(
                images=images_to_log,
                prompts=prompts_to_log,
                save_root=eval_debug_dir,
                prefix="eval",
                resolution=config.resolution,
                rewards=final_rewards.get("avg", None),
                max_images=getattr(config, "debug_image_subset_size", 6),
            )
        with tempfile.TemporaryDirectory() as tmpdir:
            num_to_log = min(WANDB_MAX_LOG_IMAGES, len(images_to_log))
            for idx in range(num_to_log):
                image = images_to_log[idx].float()
                pil = Image.fromarray((image.numpy().transpose(1, 2, 0) * 255).astype(np.uint8))
                pil = pil.resize((config.resolution, config.resolution))
                pil.save(os.path.join(tmpdir, f"{idx}.jpg"))

            sampled_prompts_log = [prompts_to_log[i] for i in range(num_to_log)]
            sampled_rewards_log = [{k: final_rewards[k][i] for k in final_rewards} for i in range(num_to_log)]

            wandb.log(
                {
                    "eval_images": [
                        wandb.Image(
                            os.path.join(tmpdir, f"{idx}.jpg"),
                            caption=f"{prompt:.1000} | "
                            + " | ".join(f"{k}: {v:.2f}" for k, v in reward.items() if v != -10),
                        )
                        for idx, (prompt, reward) in enumerate(zip(sampled_prompts_log, sampled_rewards_log))
                    ],
                    **{f"eval_reward_{k}": np.mean(v[v != -10]) for k, v in final_rewards.items()},
                },
                commit=False,
            )

    if config.train.ema and ema is not None:
        ema.copy_temp_to(transformer_trainable_parameters)
    if world_size > 1:
        dist.barrier()


def _swap_pipeline_model(pipeline, mode, inference_models, transformer_ddp, original_transformer):
    """Swap pipeline.transformer to the model specified by *mode*.

    mode: "compile_nvfp4" | "compile" | "peft"
    """
    if mode == "peft":
        pipeline.transformer = original_transformer
        transformer_ddp.module.set_adapter("old")
    else:
        pipeline.transformer = inference_models[mode]


def _rollout_for_one_prompt(
    pipeline,
    reward_fn,
    executor,
    prompt_text,
    prompt_meta,
    prompt_embed_single,
    pooled_embed_single,
    text_ids_single,
    prompt_token_ids_single,
    config,
    device,
    inference_models=None,
    transformer_ddp=None,
    original_transformer=None,
):
    sequential_decode = bool(getattr(config, "sequential_decode", True))
    amp_dtype = (
        torch.bfloat16
        if config.mixed_precision == "bf16"
        else (torch.float16 if config.mixed_precision == "fp16" else None)
    )
    enable_amp = amp_dtype is not None
    preview_step = int(getattr(config, "preview_step", 0))
    full_steps = int(getattr(config, "rollout_sample_num_steps", config.sample.num_steps))

    draft_total = int(config.sample.per_prompt_iter_num) * int(config.sample.rollout_batch_size)
    full_rollout_num = int(getattr(config.sample, "full_rollout_num", config.sample.best_of_n))
    full_rollout_num = max(1, min(full_rollout_num, draft_total))

    seed_pool = []
    draft_reward_pool = []
    prompt_samples = []
    final_images = None
    final_prompts = None
    full_chunks = int(config.sample.rollout_batch_size)

    preview_model_key = str(getattr(config, "preview_model", "peft"))
    fullrollout_model_key = str(getattr(config, "fullrollout_model", "peft"))
    _can_swap = inference_models is not None and transformer_ddp is not None and original_transformer is not None

    if preview_step > 0:
        # --- Stage 1: draft preview (fast screening) ---
        if _can_swap:
            _swap_pipeline_model(pipeline, preview_model_key, inference_models, transformer_ddp, original_transformer)

        with torch.no_grad():
            for iter_idx in range(config.sample.per_prompt_iter_num):
                batch_size = int(config.sample.rollout_batch_size)
                prompt_embeds = prompt_embed_single.repeat(batch_size, 1, 1)
                pooled_prompt_embeds = pooled_embed_single.repeat(batch_size, 1)
                text_ids = text_ids_single

                seed_list = torch.randint(
                    low=0,
                    high=2**31 - 1,
                    size=(batch_size,),
                    device="cpu",
                ).tolist()
                generators = _build_generators_from_seeds(seed_list, device=device)

                with torch_autocast(enabled=enable_amp, dtype=amp_dtype):
                    images, _, _, _, _ = pipeline_with_logprob_flux(
                        pipeline,
                        generator=generators,
                        prompt_embeds=prompt_embeds,
                        pooled_prompt_embeds=pooled_prompt_embeds,
                        text_ids=text_ids,
                        num_inference_steps=preview_step,
                        guidance_scale=config.rollout_sample_guidance_scale,
                        output_type="pt",
                        height=config.resolution,
                        width=config.resolution,
                        noise_level=config.sample.noise_level,
                        deterministic=True,
                        solver=config.sample.solver,
                        sequential_decode=sequential_decode,
                    )
                rewards, _ = reward_fn(
                    images,
                    [prompt_text] * batch_size,
                    [prompt_meta] * batch_size,
                    only_strict=True,
                )
                draft_avg = torch.as_tensor(rewards["avg"], device=device).float()
                seed_pool.extend(int(s) for s in seed_list)
                draft_reward_pool.extend(draft_avg.detach().cpu().tolist())

        draft_rewards = torch.as_tensor(draft_reward_pool, device=device).float()
        stage1_indices = select_indices_by_mode(
            draft_rewards,
            target_count=full_rollout_num,
            mode=getattr(config.sample, "stage1_select_mode", "best_worst"),
        )
        selected_seeds = [seed_pool[int(i)] for i in stage1_indices.detach().cpu().tolist()]

        # --- Stage 2: full rollout (may use a different model) ---
        if _can_swap and fullrollout_model_key != preview_model_key:
            _swap_pipeline_model(
                pipeline, fullrollout_model_key, inference_models, transformer_ddp, original_transformer
            )

        for start in range(0, len(selected_seeds), full_chunks):
            seed_chunk = selected_seeds[start : start + full_chunks]
            bs = len(seed_chunk)
            prompt_embeds = prompt_embed_single.repeat(bs, 1, 1)
            pooled_prompt_embeds = pooled_embed_single.repeat(bs, 1)
            text_ids = text_ids_single
            generators = _build_generators_from_seeds(seed_chunk, device=device)
            with torch.no_grad():
                with torch_autocast(enabled=enable_amp, dtype=amp_dtype):
                    images, latents, latent_image_ids, text_ids, _ = pipeline_with_logprob_flux(
                        pipeline,
                        generator=generators,
                        prompt_embeds=prompt_embeds,
                        pooled_prompt_embeds=pooled_prompt_embeds,
                        text_ids=text_ids,
                        num_inference_steps=full_steps,
                        guidance_scale=config.rollout_sample_guidance_scale,
                        output_type="pt",
                        height=config.resolution,
                        width=config.resolution,
                        noise_level=config.sample.noise_level,
                        deterministic=True,
                        solver=config.sample.solver,
                        sequential_decode=sequential_decode,
                    )
            timesteps = pipeline.scheduler.timesteps.repeat(bs, 1).to(device)
            latents = torch.stack(latents, dim=1)
            rewards_future = executor.submit(
                reward_fn,
                images,
                [prompt_text] * bs,
                [prompt_meta] * bs,
                True,
            )
            time.sleep(0)
            prompt_samples.append(
                {
                    "prompt_ids": prompt_token_ids_single.repeat(bs, 1),
                    "prompt_embeds": prompt_embeds,
                    "pooled_prompt_embeds": pooled_prompt_embeds,
                    "txt_ids": text_ids.repeat(bs, 1, 1),
                    "img_ids": latent_image_ids.repeat(bs, 1, 1),
                    "timesteps": timesteps,
                    "next_timesteps": torch.concatenate([timesteps[:, 1:], torch.zeros_like(timesteps[:, :1])], dim=1),
                    "latents_clean": latents[:, -1],
                    "rewards_future": rewards_future,
                }
            )
            final_images = images
            final_prompts = [prompt_text] * bs
    else:
        for iter_idx in range(config.sample.per_prompt_iter_num):
            batch_size = int(config.sample.rollout_batch_size)
            prompt_embeds = prompt_embed_single.repeat(batch_size, 1, 1)
            pooled_prompt_embeds = pooled_embed_single.repeat(batch_size, 1)
            text_ids = text_ids_single
            seed_list = torch.randint(
                low=0,
                high=2**31 - 1,
                size=(batch_size,),
                device="cpu",
            ).tolist()
            generators = _build_generators_from_seeds(seed_list, device=device)
            with torch.no_grad():
                with torch_autocast(enabled=enable_amp, dtype=amp_dtype):
                    images, latents, latent_image_ids, text_ids, _ = pipeline_with_logprob_flux(
                        pipeline,
                        generator=generators,
                        prompt_embeds=prompt_embeds,
                        pooled_prompt_embeds=pooled_prompt_embeds,
                        text_ids=text_ids,
                        num_inference_steps=full_steps,
                        guidance_scale=config.rollout_sample_guidance_scale,
                        output_type="pt",
                        height=config.resolution,
                        width=config.resolution,
                        noise_level=config.sample.noise_level,
                        deterministic=True,
                        solver=config.sample.solver,
                        sequential_decode=sequential_decode,
                    )
            timesteps = pipeline.scheduler.timesteps.repeat(batch_size, 1).to(device)
            latents = torch.stack(latents, dim=1)
            rewards_future = executor.submit(
                reward_fn,
                images,
                [prompt_text] * batch_size,
                [prompt_meta] * batch_size,
                True,
            )
            time.sleep(0)
            prompt_samples.append(
                {
                    "prompt_ids": prompt_token_ids_single.repeat(batch_size, 1),
                    "prompt_embeds": prompt_embeds,
                    "pooled_prompt_embeds": pooled_prompt_embeds,
                    "txt_ids": text_ids.repeat(batch_size, 1, 1),
                    "img_ids": latent_image_ids.repeat(batch_size, 1, 1),
                    "timesteps": timesteps,
                    "next_timesteps": torch.concatenate([timesteps[:, 1:], torch.zeros_like(timesteps[:, :1])], dim=1),
                    "latents_clean": latents[:, -1],
                    "rewards_future": rewards_future,
                }
            )
            final_images = images
            final_prompts = [prompt_text] * batch_size

    for item in prompt_samples:
        rewards, _ = item["rewards_future"].result()
        item["rewards"] = {k: torch.as_tensor(v, device=device).float() for k, v in rewards.items()}
        del item["rewards_future"]

    collated = collate_dict_items(prompt_samples)
    final_rewards = collated["rewards"]["avg"]
    keep_indices = select_indices_by_mode(
        final_rewards,
        target_count=config.sample.best_of_n,
        mode=getattr(config.sample, "stage2_select_mode", "best_worst"),
    )
    collated = filter_by_indices(collated, keep_indices)
    return collated, final_images, final_prompts


def main(_):
    config = FLAGS.config
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    setup_distributed(rank, local_rank, world_size)
    device = torch.device(f"cuda:{local_rank}")

    if is_main_process(rank):
        log_dir = os.path.join(config.logdir, config.run_name)
        os.makedirs(log_dir, exist_ok=True)
        wandb.init(
            project="sol-rl",
            name=config.run_name,
            config=config.to_dict(),
            resume="allow",
            dir=log_dir,
            id=config.run_name,
        )
        wandb.define_metric("global_step")
        wandb.define_metric("*", step_metric="global_step")
    logger.info("\n%s", config)

    set_seed(config.seed, rank)

    # cuDNN SDPA backward graph fails on Blackwell (sm_100); fall back to flash/math
    torch.backends.cuda.enable_cudnn_sdp(False)

    mixed_precision_dtype = None
    if config.mixed_precision == "fp16":
        mixed_precision_dtype = torch.float16
    elif config.mixed_precision == "bf16":
        mixed_precision_dtype = torch.bfloat16
    enable_amp = mixed_precision_dtype is not None
    scaler = GradScaler(enabled=enable_amp)

    pipeline = FluxPipeline.from_pretrained(config.pretrained.model)
    pipeline.vae.requires_grad_(False)
    pipeline.text_encoder.requires_grad_(False)
    pipeline.text_encoder_2.requires_grad_(False)
    pipeline.transformer.requires_grad_(not config.use_lora)
    pipeline.set_progress_bar_config(disable=True)
    text_encoders = [pipeline.text_encoder, pipeline.text_encoder_2]
    tokenizers = [pipeline.tokenizer, pipeline.tokenizer_2]

    text_encoder_dtype = mixed_precision_dtype if enable_amp else torch.float32
    pipeline.vae.to(device, dtype=torch.float32)
    pipeline.text_encoder.to(device, dtype=text_encoder_dtype)
    pipeline.text_encoder_2.to(device, dtype=text_encoder_dtype)

    transformer = pipeline.transformer.to(device, dtype=torch.bfloat16)
    transformer.enable_gradient_checkpointing()

    compile_mode = str(getattr(config, "compile_mode", "max-autotune-no-cudagraphs"))
    preview_step = int(getattr(config, "preview_step", 0))
    preview_model_key = str(getattr(config, "preview_model", "peft"))
    fullrollout_model_key = str(getattr(config, "fullrollout_model", "peft"))

    needed_model_types = set()
    if preview_step > 0:
        if preview_model_key != "peft":
            needed_model_types.add(preview_model_key)
        if fullrollout_model_key != "peft":
            needed_model_types.add(fullrollout_model_key)
    else:
        if fullrollout_model_key != "peft":
            needed_model_types.add(fullrollout_model_key)

    inference_models = {}
    nvfp4_skip_modules = list(getattr(config, "nvfp4_skip_modules", FLUX_NVFP4_DEFAULT_SKIP_MODULES))
    nvfp4_min_dim = int(getattr(config, "nvfp4_min_dim", 1000))

    for mtype in sorted(needed_model_types):
        logger.info(f"[INIT] Creating inference model: {mtype!r} ...")
        m = FluxTransformer2DModel.from_config(transformer.config).to(device, dtype=torch.bfloat16)
        m.load_state_dict(transformer.state_dict())
        m.requires_grad_(False)
        m.eval()

        if "nvfp4" in mtype:
            if not _HAS_TE:
                raise RuntimeError(f"model type {mtype!r} requires transformer_engine")
            n_rep, n_skip, rep_d, skip_d = replace_linear_with_te(
                m,
                skip_modules=nvfp4_skip_modules,
                min_dim=nvfp4_min_dim,
            )
            logger.info(f"[NVFP4] {mtype}: replaced {n_rep} nn.Linear -> te.Linear, skipped {n_skip}")
            wrap_forward_with_fp8(m)
            logger.info(f"[NVFP4] {mtype}: wrap_forward_with_fp8 applied")
            if is_main_process(rank):
                report_path = os.path.join(config.save_dir, f"nvfp4_quant_report_{mtype}.txt")
                os.makedirs(os.path.dirname(report_path), exist_ok=True)
                with open(report_path, "w") as f:
                    f.write(f"NVFP4 Quantization Report ({mtype})\n{'=' * 60}\n")
                    f.write(f"skip_modules: {nvfp4_skip_modules}\n")
                    f.write(f"min_dim: {nvfp4_min_dim}\n")
                    f.write(f"replaced: {n_rep}  skipped: {n_skip}\n\n")
                    f.write(f"Replaced (te.Linear + NVFP4):\n{'-' * 60}\n")
                    for fqn, inf, outf, bias, _ in rep_d:
                        f.write(f"  {fqn:60s}  in={inf:6d}  out={outf:6d}  bias={bias}\n")
                    f.write(f"\nSkipped (kept as nn.Linear):\n{'-' * 60}\n")
                    for fqn, inf, outf, bias, reason in skip_d:
                        f.write(f"  {fqn:60s}  in={inf:6d}  out={outf:6d}  bias={bias}  reason={reason}\n")
                logger.info(f"[NVFP4] Report saved to {report_path}")

        if world_size > 1:
            dist.barrier()
        logger.info(f"[COMPILE] torch.compile(mode={compile_mode!r}) on {mtype!r} ...")
        m = torch.compile(m, mode=compile_mode)
        inference_models[mtype] = m
        logger.info(f"[INIT] {mtype!r} ready")

    if inference_models:
        logger.info(f"[INIT] Inference models created: {list(inference_models.keys())}")
    else:
        logger.info("[INIT] No inference models needed, using PEFT model for all inference")

    if config.use_lora:
        init_lora_weights = getattr(config.train, "lora_init_mode", config.train.lora_init_weights)
        transformer_lora_config = LoraConfig(
            r=config.train.lora_rank,
            lora_alpha=config.train.lora_alpha,
            init_lora_weights=init_lora_weights,
            target_modules=list(
                getattr(config.train, "flux_lora_target_modules", ["to_k", "to_q", "to_v", "to_out.0"])
            ),
        )
        if config.train.lora_path:
            transformer = PeftModel.from_pretrained(transformer, config.train.lora_path)
            transformer.set_adapter("default")
        else:
            transformer = get_peft_model(transformer, transformer_lora_config)
        transformer.add_adapter("old", transformer_lora_config)
        transformer.set_adapter("default")

    transformer.set_adapter("default")
    transformer_trainable_parameters = list(filter(lambda p: p.requires_grad, transformer.parameters()))
    transformer.set_adapter("old")
    old_transformer_trainable_parameters = list(filter(lambda p: p.requires_grad, transformer.parameters()))
    for p in old_transformer_trainable_parameters:
        p.requires_grad_(False)
    transformer.set_adapter("default")
    transformer_ddp = DDP(transformer, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)

    if config.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    optimizer = torch.optim.AdamW(
        transformer_trainable_parameters,
        lr=config.train.learning_rate,
        betas=(config.train.adam_beta1, config.train.adam_beta2),
        weight_decay=config.train.adam_weight_decay,
        eps=config.train.adam_epsilon,
    )

    _, train_dataloader, train_sampler, _, test_dataloader = build_datasets_and_loaders(config, world_size, rank)

    if config.sample.best_of_n == 1:
        config.per_prompt_stat_tracking = False
    if config.per_prompt_stat_tracking:
        stat_tracker = PerPromptStatTracker(config.global_std)
    else:
        raise ValueError("per_prompt_stat_tracking must be enabled for this recipe")

    reward_fn = getattr(diffusion.flow_grpo.rewards, "multi_score")(device, config.reward_fn)
    eval_reward_fn = getattr(diffusion.flow_grpo.rewards, "multi_score")(device, config.reward_fn)
    executor = futures.ThreadPoolExecutor(max_workers=8)

    ema = None
    if config.train.ema:
        ema = EMAModuleWrapper(transformer_trainable_parameters, decay=0.9, update_step_interval=1, device=device)

    num_train_timesteps = int(config.rollout_sample_num_steps * config.train.timestep_fraction)
    train_iter = iter(train_dataloader)
    optimizer.zero_grad()

    # --- Resume from checkpoint ---
    first_epoch = 0
    global_step = 0
    candidates = find_resume_candidates(config)
    global_step, resume_parameters = resume_from_checkpoint(
        candidates,
        unwrap_compiled(transformer_ddp.module),
        ema,
        optimizer,
        scaler,
        device,
    )
    first_epoch = global_step

    if not resume_parameters:
        for src_param, tgt_param in zip(
            transformer_trainable_parameters, old_transformer_trainable_parameters, strict=True
        ):
            tgt_param.data.copy_(src_param.detach().data)

    for mtype, inf_model in inference_models.items():
        n_synced = sync_lora_to_inference(
            unwrap_compiled(transformer_ddp.module),
            unwrap_compiled(inf_model),
            adapter_name="old",
        )
        logger.info(f"[SYNC] Initial sync: merged {n_synced} LoRA layers -> {mtype!r}")

    if global_step != 0:
        for i in range(global_step):
            prompts, prompt_metadata = next(train_iter)

    if world_size > 1:
        dist.barrier()

    time_logger = DistributedTimeLogger(device)
    start_time = time.time()
    for epoch in range(first_epoch, config.num_epochs):
        time_logger.start("total_time")

        if hasattr(train_sampler, "set_epoch"):
            train_sampler.set_epoch(epoch)

        if epoch % config.save_freq == 0 and not config.debug:
            save_ckpt(config.save_dir, transformer_ddp, global_step, rank, ema, config, optimizer, scaler)

        time_logger.start("eval_time")
        if epoch % config.eval_freq == 0 and not config.debug:
            pipeline.text_encoder.to(device, dtype=text_encoder_dtype)
            pipeline.text_encoder_2.to(device, dtype=text_encoder_dtype)
            pipeline.vae.to(device, dtype=torch.float32)

            py_rng_state = random.getstate()
            np_rng_state = np.random.get_state()
            torch_rng_state = torch.random.get_rng_state()
            cuda_rng_state = torch.cuda.get_rng_state_all()

            eval_fn(
                pipeline,
                test_dataloader,
                text_encoders,
                tokenizers,
                config,
                device,
                rank,
                world_size,
                global_step,
                eval_reward_fn,
                executor,
                mixed_precision_dtype,
                ema,
                transformer_trainable_parameters,
            )

            random.setstate(py_rng_state)
            np.random.set_state(np_rng_state)
            torch.random.set_rng_state(torch_rng_state)
            torch.cuda.set_rng_state_all(cuda_rng_state)
        time_logger.end("eval_time")

        time_logger.start("rollout_time")
        pipeline.text_encoder.to(device, dtype=text_encoder_dtype)
        pipeline.text_encoder_2.to(device, dtype=text_encoder_dtype)
        pipeline.vae.to(device, dtype=torch.float32)
        pipeline.transformer.eval()
        prompts, prompt_metadata = next(train_iter)
        prompt_embeds_all, pooled_prompt_embeds_all, text_ids_all = compute_text_embeddings(
            prompts, pipeline, max_sequence_length=TEXT_ENCODER_MAX_SEQ_LEN, device=device
        )
        prompt_ids_all = tokenizers[0](
            prompts,
            padding="max_length",
            max_length=TOKENIZER_MAX_LENGTH,
            truncation=True,
            return_tensors="pt",
        ).input_ids.to(device)

        prompt_wise_samples = []
        step_prompt_reward_groups = []
        images_for_log = None
        prompts_for_log = None
        rewards_for_log = None

        _saved_pipeline_transformer = pipeline.transformer

        if preview_step == 0:
            if fullrollout_model_key != "peft" and fullrollout_model_key in inference_models:
                pipeline.transformer = inference_models[fullrollout_model_key]
            else:
                transformer_ddp.module.set_adapter("old")

        for _p in transformer_trainable_parameters:
            _p.requires_grad_(True)
        for prompt_idx in tqdm(
            range(config.sample.per_gpu_to_process_prompts),
            desc=f"Epoch {epoch}: rollout",
            disable=not is_main_process(rank),
            dynamic_ncols=True,
        ):
            collated_prompt_samples, final_images, final_prompts = _rollout_for_one_prompt(
                pipeline=pipeline,
                reward_fn=reward_fn,
                executor=executor,
                prompt_text=prompts[prompt_idx],
                prompt_meta=prompt_metadata[prompt_idx],
                prompt_embed_single=prompt_embeds_all[prompt_idx : prompt_idx + 1],
                pooled_embed_single=pooled_prompt_embeds_all[prompt_idx : prompt_idx + 1],
                text_ids_single=text_ids_all,
                prompt_token_ids_single=prompt_ids_all[prompt_idx : prompt_idx + 1],
                config=config,
                device=device,
                inference_models=inference_models,
                transformer_ddp=transformer_ddp,
                original_transformer=_saved_pipeline_transformer,
            )
            prompt_wise_samples.append(collated_prompt_samples)
            prompt_meta_i = slice_prompt_metadata(prompt_metadata, prompt_idx)
            step_prompt_reward_groups.append(
                extract_prompt_reward_group(
                    prompt_idx=prompt_idx,
                    prompt_text=prompts[prompt_idx],
                    prompt_meta=prompt_meta_i,
                    intra_prompt_data_list=[collated_prompt_samples],
                )
            )
            images_for_log = final_images
            prompts_for_log = final_prompts
            rewards_for_log = collated_prompt_samples["rewards"]["avg"]

        pipeline.transformer = _saved_pipeline_transformer
        transformer_ddp.module.set_adapter("default")
        for _p in transformer_trainable_parameters:
            _p.requires_grad_(True)

        save_step_reward_groups(
            config=config,
            global_step=global_step,
            epoch=epoch,
            rank=rank,
            world_size=world_size,
            prompt_reward_groups=step_prompt_reward_groups,
        )

        collated_samples = collate_dict_items(prompt_wise_samples)

        log_rollout_images(images_for_log, prompts_for_log, rewards_for_log, config, global_step, rank)

        collated_samples["rewards"]["avg"] = (
            collated_samples["rewards"]["avg"].unsqueeze(1).repeat(1, num_train_timesteps)
        )

        gathered_rewards_dict = {
            key: gather_tensor_to_all(value, world_size).numpy() for key, value in collated_samples["rewards"].items()
        }
        if is_main_process(rank):
            rewards_to_log = gathered_rewards_dict["avg"]
            rewards_to_log = rewards_to_log.reshape(
                world_size * config.sample.per_gpu_to_process_prompts, -1, num_train_timesteps
            )
            rewards_to_log = rewards_to_log.mean(axis=-1)
            wandb.log(
                {
                    "epoch": epoch,
                    "reward/mean": rewards_to_log.mean(),
                    "reward/max": rewards_to_log.max(axis=1).mean(),
                    "reward/min": rewards_to_log.min(axis=1).mean(),
                    "reward/range": rewards_to_log.max(axis=1).mean() - rewards_to_log.min(axis=1).mean(),
                },
                commit=False,
            )

        prompt_ids_all_global = gather_tensor_to_all(collated_samples["prompt_ids"], world_size)
        prompts_all_decoded = tokenizers[0].batch_decode(prompt_ids_all_global.cpu().numpy(), skip_special_tokens=True)
        advantages = stat_tracker.update(prompts_all_decoded, gathered_rewards_dict["avg"])

        if is_main_process(rank):
            group_size, trained_prompt_num = stat_tracker.get_stats()
            zero_std_ratio, reward_std_mean = calculate_zero_std_ratio(prompts_all_decoded, gathered_rewards_dict)
            wandb.log(
                {
                    "group_size": group_size,
                    "trained_prompt_num": trained_prompt_num,
                    "zero_std_ratio": zero_std_ratio,
                    "reward_std_mean": reward_std_mean,
                    "mean_reward_100": stat_tracker.get_mean_of_top_rewards(100),
                    "mean_reward_75": stat_tracker.get_mean_of_top_rewards(75),
                    "mean_reward_50": stat_tracker.get_mean_of_top_rewards(50),
                    "mean_reward_25": stat_tracker.get_mean_of_top_rewards(25),
                    "mean_reward_10": stat_tracker.get_mean_of_top_rewards(10),
                },
                commit=False,
            )
        stat_tracker.clear()

        samples_per_gpu = collated_samples["timesteps"].shape[0]
        if advantages.ndim == 1:
            advantages = advantages[:, None]
        if advantages.shape[0] != world_size * samples_per_gpu:
            raise RuntimeError("Unexpected advantage shape after all-gather")
        collated_samples["advantages"] = torch.from_numpy(
            advantages.reshape(world_size, samples_per_gpu, -1)[rank].astype("float32")
        ).to(device)

        del collated_samples["rewards"]
        del collated_samples["prompt_ids"]
        time_logger.end("rollout_time")

        pipeline.text_encoder.to("cpu")
        pipeline.text_encoder_2.to("cpu")
        pipeline.vae.to("cpu")
        torch.cuda.empty_cache()

        total_batch_size_filtered, num_timesteps_filtered = collated_samples["timesteps"].shape
        assert total_batch_size_filtered == config.sample.per_gpu_total_samples_to_train

        time_logger.start("train_time")
        transformer_ddp.train()
        effective_grad_accum_steps = config.train.gradient_accumulation_steps * num_train_timesteps
        current_accumulated_steps = 0
        gradient_update_times = 0

        for inner_epoch in range(config.train.num_inner_epochs):
            perm = torch.randperm(total_batch_size_filtered, device=device)
            shuffled_samples = {k: v[perm] for k, v in collated_samples.items()}
            perms_time = torch.stack(
                [torch.randperm(num_timesteps_filtered, device=device) for _ in range(total_batch_size_filtered)]
            )
            for key in ["timesteps", "next_timesteps"]:
                shuffled_samples[key] = shuffled_samples[key][
                    torch.arange(total_batch_size_filtered, device=device)[:, None],
                    perms_time,
                ]

            training_batch_size = config.train.batch_size
            batches = []
            for batch_idx in range(config.train.n_batch_per_epoch):
                start = batch_idx * training_batch_size
                end = (batch_idx + 1) * training_batch_size
                batches.append({k: v[start:end] for k, v in shuffled_samples.items()})

            info_accumulated = defaultdict(list)
            for train_batch in tqdm(
                batches,
                desc=f"Epoch {epoch}.{inner_epoch}: train",
                disable=not is_main_process(rank),
                dynamic_ncols=True,
            ):
                current_bs = len(train_batch["prompt_embeds"])
                embeds = train_batch["prompt_embeds"]
                pooled_embeds = train_batch["pooled_prompt_embeds"]
                txt_ids_batch = train_batch["txt_ids"][0]
                img_ids_batch = train_batch["img_ids"][0]
                if transformer_ddp.module.config.guidance_embeds:
                    guidance_batch = torch.full(
                        [current_bs],
                        float(config.train_sample_guidance_scale),
                        device=device,
                        dtype=torch.float32,
                    )
                else:
                    guidance_batch = None

                for j_idx in range(num_train_timesteps):
                    x0 = train_batch["latents_clean"]
                    t = torch.clamp(train_batch["timesteps"][:, j_idx] / 1000.0, 0.0, 1.0).to(x0.dtype)
                    t_expanded = t.view(-1, *([1] * (len(x0.shape) - 1)))
                    noise = torch.randn_like(x0)
                    xt = (1 - t_expanded) * x0 + t_expanded * noise
                    timestep_for_model = t

                    transformer_ddp.module.set_adapter("old")
                    for _p in transformer_trainable_parameters:
                        _p.requires_grad_(True)
                    with torch.no_grad():
                        with torch_autocast(enabled=enable_amp, dtype=mixed_precision_dtype):
                            old_prediction = transformer_ddp.module(
                                hidden_states=xt,
                                timestep=timestep_for_model,
                                guidance=guidance_batch,
                                encoder_hidden_states=embeds,
                                pooled_projections=pooled_embeds,
                                txt_ids=txt_ids_batch,
                                img_ids=img_ids_batch,
                                return_dict=False,
                            )[0].detach()
                    transformer_ddp.module.set_adapter("default")
                    for _p in transformer_trainable_parameters:
                        _p.requires_grad_(True)
                    with torch_autocast(enabled=enable_amp, dtype=mixed_precision_dtype):
                        forward_prediction = transformer_ddp(
                            hidden_states=xt,
                            timestep=timestep_for_model,
                            guidance=guidance_batch,
                            encoder_hidden_states=embeds,
                            pooled_projections=pooled_embeds,
                            txt_ids=txt_ids_batch,
                            img_ids=img_ids_batch,
                            return_dict=False,
                        )[0]
                    with torch.no_grad():
                        with torch_autocast(enabled=enable_amp, dtype=mixed_precision_dtype):
                            with transformer_ddp.module.disable_adapter():
                                ref_forward_prediction = transformer_ddp.module(
                                    hidden_states=xt,
                                    timestep=timestep_for_model,
                                    guidance=guidance_batch,
                                    encoder_hidden_states=embeds,
                                    pooled_projections=pooled_embeds,
                                    txt_ids=txt_ids_batch,
                                    img_ids=img_ids_batch,
                                    return_dict=False,
                                )[0]
                            transformer_ddp.module.set_adapter("default")

                    advantages_clip = torch.clamp(
                        train_batch["advantages"][:, j_idx],
                        -config.train.adv_clip_max,
                        config.train.adv_clip_max,
                    )
                    if hasattr(config.train, "adv_mode"):
                        if config.train.adv_mode == "positive_only":
                            advantages_clip = torch.clamp(advantages_clip, 0, config.train.adv_clip_max)
                        elif config.train.adv_mode == "negative_only":
                            advantages_clip = torch.clamp(advantages_clip, -config.train.adv_clip_max, 0)
                        elif config.train.adv_mode == "one_only":
                            advantages_clip = torch.where(
                                advantages_clip > 0, torch.ones_like(advantages_clip), torch.zeros_like(advantages_clip)
                            )
                        elif config.train.adv_mode == "binary":
                            advantages_clip = torch.sign(advantages_clip)

                    normalized_adv = (advantages_clip / config.train.adv_clip_max) / 2.0 + 0.5
                    r = torch.clamp(normalized_adv, 0, 1)

                    positive_prediction = config.beta * forward_prediction + (1 - config.beta) * old_prediction.detach()
                    implicit_negative_prediction = (
                        1.0 + config.beta
                    ) * old_prediction.detach() - config.beta * forward_prediction

                    x0_prediction = xt - t_expanded * positive_prediction
                    with torch.no_grad():
                        weight_factor = (
                            torch.abs(x0_prediction.double() - x0.double())
                            .mean(dim=tuple(range(1, x0.ndim)), keepdim=True)
                            .clip(min=1e-5)
                        )
                    positive_loss = ((x0_prediction - x0) ** 2 / weight_factor).mean(dim=tuple(range(1, x0.ndim)))

                    negative_x0_prediction = xt - t_expanded * implicit_negative_prediction
                    with torch.no_grad():
                        negative_weight_factor = (
                            torch.abs(negative_x0_prediction.double() - x0.double())
                            .mean(dim=tuple(range(1, x0.ndim)), keepdim=True)
                            .clip(min=1e-5)
                        )
                    negative_loss = ((negative_x0_prediction - x0) ** 2 / negative_weight_factor).mean(
                        dim=tuple(range(1, x0.ndim))
                    )

                    ori_policy_loss = r * positive_loss / config.beta + (1.0 - r) * negative_loss / config.beta
                    policy_loss = (ori_policy_loss * config.train.adv_clip_max).mean()
                    loss = policy_loss
                    loss_terms = {}
                    loss_terms["policy_loss"] = policy_loss.detach()
                    loss_terms["unweighted_policy_loss"] = ori_policy_loss.mean().detach()
                    loss_terms["x0_norm"] = torch.mean(x0**2).detach()
                    loss_terms["x0_norm_max"] = torch.max(x0**2).detach()
                    loss_terms["old_deviate"] = torch.mean((forward_prediction - old_prediction) ** 2).detach()
                    loss_terms["old_deviate_max"] = torch.max((forward_prediction - old_prediction) ** 2).detach()

                    kl_div_loss = ((forward_prediction - ref_forward_prediction) ** 2).mean(
                        dim=tuple(range(1, x0.ndim))
                    )
                    loss += config.train.beta * torch.mean(kl_div_loss)
                    kl_div_loss = torch.mean(kl_div_loss)

                    loss_terms["kl_div_loss"] = torch.mean(kl_div_loss).detach()
                    loss_terms["kl_div"] = torch.mean(
                        ((forward_prediction - ref_forward_prediction) ** 2).mean(dim=tuple(range(1, x0.ndim)))
                    ).detach()
                    loss_terms["old_kl_div"] = torch.mean(
                        ((old_prediction - ref_forward_prediction) ** 2).mean(dim=tuple(range(1, x0.ndim)))
                    ).detach()
                    loss_terms["total_loss"] = loss.detach()

                    scaled_loss = loss / effective_grad_accum_steps
                    if torch.isnan(scaled_loss) or torch.isinf(scaled_loss):
                        scaled_loss = scaled_loss * 0.0

                    if mixed_precision_dtype == torch.float16:
                        scaler.scale(scaled_loss).backward()
                    else:
                        scaled_loss.backward()
                    current_accumulated_steps += 1

                    for k_info, v_info in loss_terms.items():
                        info_accumulated[k_info].append(v_info)

                    if current_accumulated_steps % effective_grad_accum_steps == 0:
                        if mixed_precision_dtype == torch.float16:
                            scaler.unscale_(optimizer)
                        grad_norm = torch.nn.utils.clip_grad_norm_(
                            transformer_ddp.module.parameters(), config.train.max_grad_norm
                        )
                        if mixed_precision_dtype == torch.float16:
                            scaler.step(optimizer)
                            scaler.update()
                        else:
                            optimizer.step()
                        optimizer.zero_grad()
                        gradient_update_times += 1

                        log_info = {k: torch.mean(torch.stack(v)).item() for k, v in info_accumulated.items()}
                        if torch.is_tensor(grad_norm):
                            log_info["grad_norm"] = grad_norm.detach().float().item()
                        else:
                            log_info["grad_norm"] = float(grad_norm)
                        info_tensor = torch.tensor([log_info[k] for k in sorted(log_info)], device=device)
                        dist.all_reduce(info_tensor, op=dist.ReduceOp.AVG)
                        reduced_log = {k: info_tensor[i].item() for i, k in enumerate(sorted(log_info))}
                        if is_main_process(rank):
                            wandb.log(
                                {
                                    "global_step": global_step,
                                    "gradient_update_times": gradient_update_times,
                                    "epoch": epoch,
                                    "inner_epoch": inner_epoch,
                                    "current_time": time.time() - start_time,
                                    **reduced_log,
                                },
                                commit=False,
                            )
                        global_step += 1
                        info_accumulated = defaultdict(list)

                    if (
                        config.train.ema
                        and ema is not None
                        and (current_accumulated_steps % effective_grad_accum_steps == 0)
                    ):
                        ema.step(transformer_trainable_parameters, global_step)

        time_logger.end("train_time")

        if world_size > 1:
            dist.barrier()
        with torch.no_grad():
            decay = return_decay(
                global_step,
                config.decay_type,
                custom_decay_step=getattr(config, "custom_decay_step", 0),
                custom_decay_value=getattr(config, "custom_decay_value", 0.0),
            )
            for src_param, tgt_param in zip(
                transformer_trainable_parameters, old_transformer_trainable_parameters, strict=True
            ):
                tgt_param.data.copy_(tgt_param.detach().data * decay + src_param.detach().clone().data * (1.0 - decay))

        for mtype, inf_model in inference_models.items():
            sync_lora_to_inference(
                unwrap_compiled(transformer_ddp.module),
                unwrap_compiled(inf_model),
                adapter_name="old",
            )

        time_logger.end("total_time")
        stats = time_logger.get_results()

        if is_main_process(rank):
            time_logs = {f"time/{k}": v for k, v in stats.items()}
            wandb.log(time_logs, commit=True)
            logger.info("Step %d Time Report: %s", global_step, time_logs)

        time_logger.empty_cache()

    if is_main_process(rank):
        wandb.finish()
    cleanup_distributed()


if __name__ == "__main__":
    app.run(main)
