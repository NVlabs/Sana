"""
Sana post-training (BON + preview rollout) using diffusers SanaPipeline.
Structure mirrors train_sd3 (Sol-RL) with BON + preview rollout.
"""

import os
import sys
from collections import defaultdict

_rank = int(os.environ.get("RANK", 0))
_cache_root = os.environ.get("CACHE_ROOT", os.path.expanduser("~/.cache/sol_rl"))
os.environ.setdefault("TRITON_CACHE_DIR", f"{_cache_root}/triton/rank_{_rank}")
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", f"{_cache_root}/torchinductor/rank_{_rank}")
os.environ.setdefault("TORCHINDUCTOR_FX_GRAPH_CACHE", "1")

import logging
import random
import time
from concurrent import futures

import numpy as np
import torch
import torch.distributed as dist
import tqdm
import wandb
from absl import app, flags
from diffusers import SanaPipeline
from ml_collections import config_flags
from peft import LoraConfig, PeftModel, get_peft_model
from torch.cuda.amp import GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pyrallis
from train_utils import (
    _HAS_TE,
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
    save_step_reward_groups,
    select_indices_by_mode,
    set_seed,
    setup_distributed,
    slice_prompt_metadata,
    sync_lora_to_inference,
    unwrap_compiled,
    wrap_forward_with_fp8,
)

import diffusion.model.nets.sana_multi_scale  # noqa: F401
import diffusion.post_training.rewards
from diffusion.model.builder import MODELS
from diffusion.post_training.diffusers_patch.pipeline_with_logprob import pipeline_with_logprob_sana
from diffusion.post_training.diffusers_patch.text_encode import encode_sana_prompt
from diffusion.post_training.ema import EMAModuleWrapper
from diffusion.post_training.stat_tracking import PerPromptStatTracker
from diffusion.utils.config import SanaConfig, model_init_config
from tools.download import find_model

tqdm = tqdm.tqdm
FLAGS = flags.FLAGS
config_flags.DEFINE_config_file(
    "config",
    "configs/sol_rl/sana.py",
    "Training configuration.",
)
flags.DEFINE_string("native_config", "", "Optional override for native model YAML config path")

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

TEXT_ENCODER_MAX_SEQ_LEN = 300
TOKENIZER_MAX_LENGTH = 300
WANDB_MAX_LOG_IMAGES = 12


def compute_text_embeddings(prompts, pipeline, max_sequence_length, device):
    with torch.no_grad():
        return encode_sana_prompt(
            pipeline,
            prompts,
            max_sequence_length=max_sequence_length,
            device=device,
            negative_prompt="",
            do_classifier_free_guidance=True,
        )


def _resolve_native_checkpoint_source(config, native_cfg):
    native_model_path = str(getattr(config, "native_model_path", "") or "").strip()
    native_model_source = str(getattr(config, "native_model_source", "") or "").strip()

    if native_model_path:
        native_cfg.model.load_from = native_model_path
        if os.path.isfile(native_model_path):
            return native_model_path
        if native_model_source:
            logger.info(
                "[INIT] Native checkpoint missing at %s; falling back to %s", native_model_path, native_model_source
            )
            return native_model_source

    return native_cfg.model.load_from


def _prepare_latents_from_seeds(seed_list, num_channels, latent_h, latent_w, device, dtype):
    latents = []
    for seed in seed_list:
        g = torch.Generator(device=device).manual_seed(int(seed))
        latents.append(torch.randn(1, num_channels, latent_h, latent_w, device=device, dtype=dtype, generator=g))
    return torch.cat(latents, dim=0)


def _select_inference_transformer(mode, inference_models, transformer_ddp, peft_transformer):
    """Return the transformer to use for inference based on *mode*.
    mode: "compile_nvfp4" | "compile" | "peft"
    """
    if mode == "peft":
        transformer_ddp.module.set_adapter("old")
        return peft_transformer
    return inference_models[mode]


def _rollout_for_one_prompt(
    rollout_transformer,
    vae,
    num_channels,
    latent_size,
    reward_fn,
    executor,
    prompt_text,
    prompt_meta,
    prompt_embed_single,
    prompt_mask_single,
    neg_prompt_embed_single,
    neg_prompt_mask_single,
    prompt_token_ids_single,
    config,
    device,
    inference_models=None,
    transformer_ddp=None,
    peft_transformer=None,
):
    preview_step = int(getattr(config, "preview_step", 0))
    full_steps = int(getattr(config, "rollout_sample_num_steps", config.sample.num_steps))
    draft_total = int(config.sample.per_prompt_iter_num) * int(config.sample.rollout_batch_size)
    full_rollout_num = max(
        1, min(int(getattr(config.sample, "full_rollout_num", config.sample.best_of_n)), draft_total)
    )
    full_chunks = int(config.sample.rollout_batch_size)

    solver = str(getattr(config.sample, "solver", "flow"))
    noise_level = float(getattr(config.sample, "noise_level", 0.7))
    deterministic = True

    seed_pool, draft_reward_pool = [], []
    prompt_samples = []
    final_images = final_prompts = None

    preview_model_key = str(getattr(config, "preview_model", "peft"))
    fullrollout_model_key = str(getattr(config, "fullrollout_model", "peft"))
    _can_swap = inference_models is not None and transformer_ddp is not None and peft_transformer is not None

    if preview_step > 0:
        if _can_swap:
            current_transformer = _select_inference_transformer(
                preview_model_key, inference_models, transformer_ddp, peft_transformer
            )
        else:
            current_transformer = rollout_transformer

        with torch.no_grad():
            for _ in range(config.sample.per_prompt_iter_num):
                bs = int(config.sample.rollout_batch_size)
                p_emb = prompt_embed_single.repeat(bs, 1, 1)
                p_mask = prompt_mask_single.repeat(bs, 1)
                neg_emb = neg_prompt_embed_single.repeat(bs, 1, 1)
                neg_mask = neg_prompt_mask_single.repeat(bs, 1)
                seed_list = torch.randint(0, 2**31 - 1, (bs,), device="cpu").tolist()
                init_latents = _prepare_latents_from_seeds(
                    seed_list, num_channels, latent_size, latent_size, device, p_emb.dtype
                )
                images, _, _ = pipeline_with_logprob_sana(
                    current_transformer,
                    vae,
                    latents=init_latents,
                    prompt_embeds=p_emb,
                    prompt_attention_mask=p_mask,
                    negative_prompt_embeds=neg_emb,
                    negative_prompt_attention_mask=neg_mask,
                    num_inference_steps=preview_step,
                    guidance_scale=config.rollout_sample_guidance_scale,
                    noise_level=noise_level,
                    deterministic=deterministic,
                    solver=solver,
                )
                rewards, _ = reward_fn(images, [prompt_text] * bs, [prompt_meta] * bs, only_strict=True)
                seed_pool.extend(int(s) for s in seed_list)
                draft_reward_pool.extend(torch.as_tensor(rewards["avg"], device=device).float().detach().cpu().tolist())

        draft_rewards = torch.as_tensor(draft_reward_pool, device=device).float()
        stage1_idx = select_indices_by_mode(
            draft_rewards, full_rollout_num, getattr(config.sample, "stage1_select_mode", "best_worst")
        )
        selected_seeds = [seed_pool[int(i)] for i in stage1_idx.cpu().tolist()]

        if _can_swap and fullrollout_model_key != preview_model_key:
            current_transformer = _select_inference_transformer(
                fullrollout_model_key, inference_models, transformer_ddp, peft_transformer
            )

        for start in range(0, len(selected_seeds), full_chunks):
            chunk_seeds = selected_seeds[start : start + full_chunks]
            bs = len(chunk_seeds)
            p_emb = prompt_embed_single.repeat(bs, 1, 1)
            p_mask = prompt_mask_single.repeat(bs, 1)
            neg_emb = neg_prompt_embed_single.repeat(bs, 1, 1)
            neg_mask = neg_prompt_mask_single.repeat(bs, 1)
            init_latents = _prepare_latents_from_seeds(
                chunk_seeds, num_channels, latent_size, latent_size, device, p_emb.dtype
            )
            with torch.no_grad():
                images, all_latents, step_sigmas = pipeline_with_logprob_sana(
                    current_transformer,
                    vae,
                    latents=init_latents,
                    prompt_embeds=p_emb,
                    prompt_attention_mask=p_mask,
                    negative_prompt_embeds=neg_emb,
                    negative_prompt_attention_mask=neg_mask,
                    num_inference_steps=full_steps,
                    guidance_scale=config.rollout_sample_guidance_scale,
                    noise_level=noise_level,
                    deterministic=deterministic,
                    solver=solver,
                )
            timesteps = step_sigmas.unsqueeze(0).repeat(bs, 1)
            latents = torch.stack(all_latents, dim=1)
            rewards_future = executor.submit(reward_fn, images, [prompt_text] * bs, [prompt_meta] * bs, True)
            time.sleep(0)
            prompt_samples.append(
                {
                    "prompt_ids": prompt_token_ids_single.repeat(bs, 1),
                    "prompt_embeds": p_emb,
                    "prompt_attention_mask": p_mask,
                    "timesteps": timesteps,
                    "next_timesteps": torch.cat([timesteps[:, 1:], torch.zeros_like(timesteps[:, :1])], dim=1),
                    "latents_clean": latents[:, -1],
                    "rewards_future": rewards_future,
                }
            )
            final_images, final_prompts = images, [prompt_text] * bs
    else:
        for _ in range(config.sample.per_prompt_iter_num):
            bs = int(config.sample.rollout_batch_size)
            p_emb = prompt_embed_single.repeat(bs, 1, 1)
            p_mask = prompt_mask_single.repeat(bs, 1)
            neg_emb = neg_prompt_embed_single.repeat(bs, 1, 1)
            neg_mask = neg_prompt_mask_single.repeat(bs, 1)
            seed_list = torch.randint(0, 2**31 - 1, (bs,), device="cpu").tolist()
            init_latents = _prepare_latents_from_seeds(
                seed_list, num_channels, latent_size, latent_size, device, p_emb.dtype
            )
            with torch.no_grad():
                images, all_latents, step_sigmas = pipeline_with_logprob_sana(
                    rollout_transformer,
                    vae,
                    latents=init_latents,
                    prompt_embeds=p_emb,
                    prompt_attention_mask=p_mask,
                    negative_prompt_embeds=neg_emb,
                    negative_prompt_attention_mask=neg_mask,
                    num_inference_steps=full_steps,
                    guidance_scale=config.rollout_sample_guidance_scale,
                    noise_level=noise_level,
                    deterministic=deterministic,
                    solver=solver,
                )
            timesteps = step_sigmas.unsqueeze(0).repeat(bs, 1)
            latents = torch.stack(all_latents, dim=1)
            rewards_future = executor.submit(reward_fn, images, [prompt_text] * bs, [prompt_meta] * bs, True)
            time.sleep(0)
            prompt_samples.append(
                {
                    "prompt_ids": prompt_token_ids_single.repeat(bs, 1),
                    "prompt_embeds": p_emb,
                    "prompt_attention_mask": p_mask,
                    "timesteps": timesteps,
                    "next_timesteps": torch.cat([timesteps[:, 1:], torch.zeros_like(timesteps[:, :1])], dim=1),
                    "latents_clean": latents[:, -1],
                    "rewards_future": rewards_future,
                }
            )
            final_images, final_prompts = images, [prompt_text] * bs

    for item in prompt_samples:
        rewards, _ = item["rewards_future"].result()
        item["rewards"] = {k: torch.as_tensor(v, device=device).float() for k, v in rewards.items()}
        del item["rewards_future"]

    collated = collate_dict_items(prompt_samples)
    keep = select_indices_by_mode(
        collated["rewards"]["avg"],
        config.sample.best_of_n,
        getattr(config.sample, "stage2_select_mode", "best_worst"),
    )
    collated = filter_by_indices(collated, keep)
    return collated, final_images, final_prompts


def eval_fn(
    pipeline,
    eval_transformer,
    vae,
    num_channels,
    latent_size,
    test_dataloader,
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

    if config.train.ema and ema is not None:
        ema.copy_ema_to(transformer_trainable_parameters, store_temp=True)
    eval_transformer.eval()
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
        num_workers=0,
    )

    solver = str(getattr(config.sample, "solver", "flow"))
    for test_batch in tqdm(eval_loader, desc="Eval:", disable=not is_main_process(rank)):
        prompts, prompt_metadata = test_batch
        with torch.no_grad():
            prompt_embeds, prompt_mask, neg_embeds, neg_mask = compute_text_embeddings(
                prompts,
                pipeline,
                max_sequence_length=TEXT_ENCODER_MAX_SEQ_LEN,
                device=device,
            )
            images, _, _ = pipeline_with_logprob_sana(
                eval_transformer,
                vae,
                num_channels=num_channels,
                latent_size=latent_size,
                prompt_embeds=prompt_embeds,
                prompt_attention_mask=prompt_mask,
                negative_prompt_embeds=neg_embeds,
                negative_prompt_attention_mask=neg_mask,
                num_inference_steps=config.sample.eval_num_steps,
                guidance_scale=config.eval_sample_guidance_scale,
                solver=solver,
            )
        rewards_future = executor.submit(reward_fn, images, prompts, prompt_metadata, only_strict=False)
        time.sleep(0)
        rewards, _ = rewards_future.result()
        for key, value in rewards.items():
            all_rewards[key].extend(value)

    final_rewards = {k: np.array([x.cpu() if torch.is_tensor(x) else x for x in v]) for k, v in all_rewards.items()}
    if is_main_process(rank) and final_rewards:
        wandb.log(
            {**{f"eval_reward_{k}": np.mean(v[v != -10]) for k, v in final_rewards.items()}},
            commit=False,
        )
        for k, v in final_rewards.items():
            logger.info("eval_reward_%s: %.4f", k, np.mean(v[v != -10]))

    if config.train.ema and ema is not None:
        ema.copy_temp_to(transformer_trainable_parameters)
    if world_size > 1:
        dist.barrier()


def main(_):
    config = FLAGS.config
    if FLAGS.native_config:
        config.native_config = FLAGS.native_config

    # cuDNN SDPA backward graph fails on Blackwell (sm_100); fall back to flash/math
    torch.backends.cuda.enable_cudnn_sdp(False)

    start_time = time.time()

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
    logger.info(f"\n{config}")

    set_seed(config.seed, rank)

    mixed_precision_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}.get(config.mixed_precision)
    enable_amp = mixed_precision_dtype is not None
    scaler = GradScaler(enabled=enable_amp and mixed_precision_dtype == torch.float16)

    # Build native transformer from the Sana YAML config.
    with open(config.native_config, encoding="utf-8") as _f:
        native_cfg = pyrallis.load(SanaConfig, _f)
    ckpt_source = _resolve_native_checkpoint_source(config, native_cfg)

    # Keep the diffusers text encoder + VAE, but replace the native transformer.
    pipeline = SanaPipeline.from_pretrained(
        "Efficient-Large-Model/Sana_1600M_1024px_BF16_diffusers", torch_dtype=torch.bfloat16
    )
    pipeline.vae.requires_grad_(False)
    pipeline.text_encoder.requires_grad_(False)
    pipeline.set_progress_bar_config(disable=True)

    text_encoder_dtype = mixed_precision_dtype if enable_amp else torch.float32
    pipeline.vae.to(device, dtype=torch.bfloat16)
    pipeline.text_encoder.to(device, dtype=text_encoder_dtype)
    vae = pipeline.vae
    del pipeline.transformer
    torch.cuda.empty_cache()
    latent_size = config.resolution // 32
    if getattr(native_cfg, "work_dir", None):
        os.makedirs(native_cfg.work_dir, exist_ok=True)
    model_kwargs = model_init_config(native_cfg, latent_size=latent_size)
    transformer = MODELS.build(dict(type=native_cfg.model.model), default_args=model_kwargs)
    transformer.to(device, dtype=torch.bfloat16)

    logger.info(f"[INIT] Loading native checkpoint from {ckpt_source}")
    ckpt = find_model(ckpt_source)
    if isinstance(ckpt, dict):
        if "state_dict" in ckpt:
            state = ckpt["state_dict"]
        elif "model" in ckpt:
            state = ckpt["model"]
        else:
            state = ckpt
    else:
        state = ckpt
    missing, unexpected = transformer.load_state_dict(state, strict=False)
    if missing:
        logger.warning(f"[weights] missing {len(missing)} keys (showing up to 10): {missing[:10]}")

    for blk in transformer.blocks:
        if hasattr(blk.attn, "eps"):
            blk.attn.eps = 1e-15

    num_channels = native_cfg.vae.vae_latent_dim
    transformer.requires_grad_(not config.use_lora)

    # Create optional inference-only copies for compiled and/or NVFP4 rollout modes.
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
    nvfp4_skip_modules = list(getattr(config, "nvfp4_skip_modules", []))
    nvfp4_min_dim = int(getattr(config, "nvfp4_min_dim", 0))

    for mtype in sorted(needed_model_types):
        logger.info(f"[INIT] Creating inference model: {mtype!r} ...")
        m = MODELS.build(dict(type=native_cfg.model.model), default_args=model_kwargs)
        m.to(device, dtype=torch.bfloat16)
        m.load_state_dict(state, strict=False)
        for blk in m.blocks:
            if hasattr(blk.attn, "eps"):
                blk.attn.eps = 1e-15
        m.eval()
        m.requires_grad_(False)

        if "nvfp4" in mtype:
            if not _HAS_TE:
                raise RuntimeError(f"{mtype!r} requires transformer_engine")
            n_rep, n_skip, rep_d, skip_d = replace_linear_with_te(
                m, skip_modules=nvfp4_skip_modules, min_dim=nvfp4_min_dim
            )
            logger.info(f"[NVFP4] {mtype}: replaced {n_rep} -> te.Linear, skipped {n_skip}")
            if is_main_process(rank):
                os.makedirs(config.save_dir, exist_ok=True)
                report_path = os.path.join(config.save_dir, f"nvfp4_report_{mtype}.txt")
                with open(report_path, "w") as f:
                    f.write(f"NVFP4 Quantization Report: {mtype}\n")
                    f.write(f"skip_modules={nvfp4_skip_modules}, min_dim={nvfp4_min_dim}\n")
                    f.write(f"replaced={n_rep}, skipped={n_skip}\n\n")
                    f.write(f"REPLACED ({len(rep_d)}):\n")
                    for fqn, inf, outf, b, _ in rep_d:
                        f.write(f"  {fqn:60s} {inf:>5d} -> {outf:>5d}  bias={b}\n")
                    f.write(f"\nSKIPPED ({len(skip_d)}):\n")
                    for fqn, inf, outf, b, r in skip_d:
                        f.write(f"  {fqn:60s} {inf:>5d} -> {outf:>5d}  bias={b}  reason={r}\n")
                logger.info(f"[NVFP4] Report saved to {report_path}")
            wrap_forward_with_fp8(m)

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
        lora_cfg = LoraConfig(
            r=config.train.lora_rank,
            lora_alpha=config.train.lora_alpha,
            init_lora_weights=init_lora_weights,
            target_modules=list(config.train.lora_target_modules),
        )
        if config.train.lora_path:
            transformer = PeftModel.from_pretrained(transformer, config.train.lora_path)
            transformer.set_adapter("default")
        else:
            transformer = get_peft_model(transformer, lora_cfg)
        transformer.add_adapter("old", lora_cfg)
        transformer.set_adapter("default")
    peft_transformer = transformer

    transformer_ddp = DDP(transformer, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)
    transformer_ddp.module.set_adapter("default")
    transformer_trainable_parameters = list(filter(lambda p: p.requires_grad, transformer_ddp.module.parameters()))
    transformer_ddp.module.set_adapter("old")
    old_transformer_trainable_parameters = list(filter(lambda p: p.requires_grad, transformer_ddp.module.parameters()))
    transformer_ddp.module.set_adapter("default")

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

    ema = None
    if config.train.ema:
        ema = EMAModuleWrapper(
            transformer_trainable_parameters,
            decay=getattr(config.train, "ema_decay", 0.9),
            update_step_interval=getattr(config.train, "ema_update_step_interval", 1),
            device=device,
        )

    _, train_dataloader, train_sampler, _, test_dataloader = build_datasets_and_loaders(config, world_size, rank)
    train_iter = iter(train_dataloader)

    stat_tracker = PerPromptStatTracker(config.global_std) if config.per_prompt_stat_tracking else None
    executor = futures.ThreadPoolExecutor(max_workers=8)

    reward_fn = getattr(diffusion.post_training.rewards, "multi_score")(device, config.reward_fn)
    eval_reward_fn = getattr(diffusion.post_training.rewards, "multi_score")(device, config.reward_fn)

    timestep_clip = getattr(config, "timestep_clip", None)
    if timestep_clip is not None:
        _ts_start, _ts_end = int(timestep_clip[0]), int(timestep_clip[1])
        num_train_timesteps = _ts_end - _ts_start
    else:
        _ts_start = _ts_end = None
        num_train_timesteps = int(config.rollout_sample_num_steps * config.train.timestep_fraction)

    first_epoch = 0
    global_step = 0
    candidates = find_resume_candidates(config)
    global_step, _resumed = resume_from_checkpoint(
        candidates,
        transformer_ddp.module,
        ema,
        optimizer,
        scaler,
        device,
    )
    first_epoch = global_step

    if first_epoch == 0:
        for src_p, tgt_p in zip(transformer_trainable_parameters, old_transformer_trainable_parameters, strict=True):
            tgt_p.data.copy_(src_p.detach().data)

    # The rollout path reads from the "old" adapter weights.
    for mtype, inf_model in inference_models.items():
        sync_lora_to_inference(transformer_ddp.module, unwrap_compiled(inf_model), adapter_name="old")

    optimizer.zero_grad()
    time_logger = DistributedTimeLogger(device)

    if global_step != 0:
        for _ in range(global_step):
            next(train_iter)

    if world_size > 1:
        dist.barrier()

    for epoch in range(first_epoch, config.num_epochs):
        time_logger.start("total_time")

        if hasattr(train_sampler, "set_epoch"):
            train_sampler.set_epoch(epoch)

        if epoch % config.save_freq == 0 and not config.debug:
            save_ckpt(config.save_dir, transformer_ddp, global_step, rank, ema, config, optimizer, scaler)

        time_logger.start("eval_time")
        if epoch % config.eval_freq == 0 and not config.debug:
            torch.cuda.empty_cache()
            py_rng = random.getstate()
            np_rng = np.random.get_state()
            torch_rng = torch.random.get_rng_state()
            cuda_rng = torch.cuda.get_rng_state_all()
            eval_fn(
                pipeline,
                peft_transformer,
                vae,
                num_channels,
                latent_size,
                test_dataloader,
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
            random.setstate(py_rng)
            np.random.set_state(np_rng)
            torch.random.set_rng_state(torch_rng)
            torch.cuda.set_rng_state_all(cuda_rng)
        time_logger.end("eval_time")

        time_logger.start("rollout_time")
        peft_transformer.eval()
        prompts, prompt_metadata = next(train_iter)

        time_logger.start("text_tokenizer_time")
        with torch.no_grad():
            prompt_embeds, prompt_masks, neg_embeds, neg_masks = compute_text_embeddings(
                prompts,
                pipeline,
                max_sequence_length=TEXT_ENCODER_MAX_SEQ_LEN,
                device=device,
            )
        txt_tokens = pipeline.tokenizer(
            prompts,
            max_length=TOKENIZER_MAX_LENGTH,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        time_logger.end("text_tokenizer_time")

        prompt_ids_all = txt_tokens.input_ids.to(device)
        prompt_embeds_list = [prompt_embeds[i : i + 1] for i in range(len(prompts))]
        prompt_masks_list = [prompt_masks[i : i + 1] for i in range(len(prompts))]
        neg_embeds_single = neg_embeds[0:1]
        neg_masks_single = neg_masks[0:1]

        prompt_wise_samples = []
        step_prompt_reward_groups = []
        images_for_log = prompts_for_log = rewards_for_log = None

        if preview_step == 0:
            if fullrollout_model_key != "peft" and fullrollout_model_key in inference_models:
                default_rollout_transformer = inference_models[fullrollout_model_key]
            else:
                transformer_ddp.module.set_adapter("old")
                default_rollout_transformer = peft_transformer
        else:
            default_rollout_transformer = peft_transformer

        for prompt_idx in tqdm(
            range(config.sample.per_gpu_to_process_prompts),
            desc=f"Epoch {epoch}: rollout",
            disable=not is_main_process(rank),
            dynamic_ncols=True,
        ):
            collated_prompt_samples, final_images, final_prompts = _rollout_for_one_prompt(
                rollout_transformer=default_rollout_transformer,
                vae=vae,
                num_channels=num_channels,
                latent_size=latent_size,
                reward_fn=reward_fn,
                executor=executor,
                prompt_text=prompts[prompt_idx],
                prompt_meta=prompt_metadata[prompt_idx],
                prompt_embed_single=prompt_embeds_list[prompt_idx],
                prompt_mask_single=prompt_masks_list[prompt_idx],
                neg_prompt_embed_single=neg_embeds_single,
                neg_prompt_mask_single=neg_masks_single,
                prompt_token_ids_single=prompt_ids_all[prompt_idx : prompt_idx + 1],
                config=config,
                device=device,
                inference_models=inference_models,
                transformer_ddp=transformer_ddp,
                peft_transformer=peft_transformer,
            )
            prompt_wise_samples.append(collated_prompt_samples)
            prompt_meta_i = slice_prompt_metadata(prompt_metadata, prompt_idx)
            step_prompt_reward_groups.append(
                extract_prompt_reward_group(prompt_idx, prompts[prompt_idx], prompt_meta_i, [collated_prompt_samples])
            )
            images_for_log, prompts_for_log = final_images, final_prompts
            rewards_for_log = collated_prompt_samples["rewards"]["avg"]

        transformer_ddp.module.set_adapter("default")

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
            k: gather_tensor_to_all(v, world_size).numpy() for k, v in collated_samples["rewards"].items()
        }
        if is_main_process(rank):
            r2l = (
                gathered_rewards_dict["avg"]
                .reshape(world_size * config.sample.per_gpu_to_process_prompts, -1, num_train_timesteps)
                .mean(axis=-1)
            )
            wandb.log(
                {
                    "epoch": epoch,
                    "reward/mean": r2l.mean(),
                    "reward/max": r2l.max(axis=1).mean(),
                    "reward/min": r2l.min(axis=1).mean(),
                    "reward/range": r2l.max(axis=1).mean() - r2l.min(axis=1).mean(),
                },
                commit=False,
            )

        prompt_ids_global = gather_tensor_to_all(collated_samples["prompt_ids"], world_size)
        prompts_decoded = pipeline.tokenizer.batch_decode(prompt_ids_global.cpu().numpy(), skip_special_tokens=True)

        if stat_tracker is not None:
            advantages = stat_tracker.update(prompts_decoded, gathered_rewards_dict["avg"])
            if is_main_process(rank):
                gs, tp = stat_tracker.get_stats()
                zsr, rsm = calculate_zero_std_ratio(prompts_decoded, gathered_rewards_dict)
                wandb.log(
                    {
                        "group_size": gs,
                        "trained_prompt_num": tp,
                        "zero_std_ratio": zsr,
                        "reward_std_mean": rsm,
                        "mean_reward_100": stat_tracker.get_mean_of_top_rewards(100),
                        "mean_reward_50": stat_tracker.get_mean_of_top_rewards(50),
                    },
                    commit=False,
                )
            stat_tracker.clear()
        else:
            avg = gathered_rewards_dict["avg"]
            advantages = (avg - avg.mean()) / (avg.std() + 1e-4)

        samples_per_gpu = collated_samples["timesteps"].shape[0]
        if advantages.ndim == 1:
            advantages = advantages[:, None]
        collated_samples["advantages"] = torch.from_numpy(advantages.reshape(world_size, samples_per_gpu, -1)[rank]).to(
            device
        )
        del collated_samples["rewards"]
        del collated_samples["prompt_ids"]
        time_logger.end("rollout_time")

        total_batch_size_filtered, num_timesteps_filtered = collated_samples["timesteps"].shape

        time_logger.start("train_time")
        effective_grad_accum_steps = config.train.gradient_accumulation_steps * num_train_timesteps
        current_accumulated_steps = 0
        gradient_update_times = 0

        for inner_epoch in range(config.train.num_inner_epochs):
            perm = torch.randperm(total_batch_size_filtered, device=device)
            shuffled = {k: v[perm] for k, v in collated_samples.items()}
            perms_time = torch.stack(
                [torch.randperm(num_timesteps_filtered, device=device) for _ in range(total_batch_size_filtered)]
            )
            for key in ["timesteps", "next_timesteps"]:
                shuffled[key] = shuffled[key][
                    torch.arange(total_batch_size_filtered, device=device)[:, None], perms_time
                ]

            batches = []
            for bi in range(config.train.n_batch_per_epoch):
                s, e = bi * config.train.batch_size, (bi + 1) * config.train.batch_size
                batches.append({k: v[s:e] for k, v in shuffled.items()})

            info_accumulated = defaultdict(list)
            for train_batch in tqdm(
                batches,
                desc=f"Epoch {epoch}.{inner_epoch}: train",
                disable=not is_main_process(rank),
                dynamic_ncols=True,
            ):
                embeds = train_batch["prompt_embeds"]
                embeds_mask = train_batch["prompt_attention_mask"]
                embeds_4d = embeds.unsqueeze(1)
                mask_4d = embeds_mask.unsqueeze(1).unsqueeze(1).to(torch.int16) if embeds_mask is not None else None

                for j_idx in range(num_train_timesteps):
                    x0 = train_batch["latents_clean"]
                    sigma = train_batch["timesteps"][:, j_idx]
                    sigma_expanded = sigma.view(-1, *([1] * (len(x0.shape) - 1)))
                    noise = torch.randn_like(x0.float())
                    xt = ((1 - sigma_expanded) * x0 + sigma_expanded * noise).to(torch.bfloat16)

                    transformer_ddp.module.set_adapter("old")
                    with torch.no_grad():
                        old_prediction = transformer_ddp(xt, sigma, embeds_4d, mask=mask_4d).detach()
                    transformer_ddp.module.set_adapter("default")

                    forward_prediction = transformer_ddp(xt, sigma, embeds_4d, mask=mask_4d)

                    with torch.no_grad():
                        with transformer_ddp.module.disable_adapter():
                            ref_forward_prediction = transformer_ddp(xt, sigma, embeds_4d, mask=mask_4d)
                        transformer_ddp.module.set_adapter("default")

                    loss_terms = {}
                    advantages_clip = torch.clamp(
                        train_batch["advantages"][:, j_idx], -config.train.adv_clip_max, config.train.adv_clip_max
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

                    r = torch.clamp((advantages_clip / config.train.adv_clip_max) / 2.0 + 0.5, 0, 1)

                    positive_prediction = config.beta * forward_prediction + (1 - config.beta) * old_prediction.detach()
                    implicit_negative_prediction = (
                        1.0 + config.beta
                    ) * old_prediction.detach() - config.beta * forward_prediction

                    x0_prediction = xt - sigma_expanded * positive_prediction
                    with torch.no_grad():
                        weight_factor = (
                            torch.abs(x0_prediction.double() - x0.double())
                            .mean(dim=tuple(range(1, x0.ndim)), keepdim=True)
                            .clip(min=1e-5)
                        )
                    positive_loss = ((x0_prediction - x0) ** 2 / weight_factor).mean(dim=tuple(range(1, x0.ndim)))

                    negative_x0_prediction = xt - sigma_expanded * implicit_negative_prediction
                    with torch.no_grad():
                        neg_wf = (
                            torch.abs(negative_x0_prediction.double() - x0.double())
                            .mean(dim=tuple(range(1, x0.ndim)), keepdim=True)
                            .clip(min=1e-5)
                        )
                    negative_loss = ((negative_x0_prediction - x0) ** 2 / neg_wf).mean(dim=tuple(range(1, x0.ndim)))

                    ori_policy_loss = r * positive_loss / config.beta + (1.0 - r) * negative_loss / config.beta
                    policy_loss = (ori_policy_loss * config.train.adv_clip_max).mean()
                    loss = policy_loss
                    loss_terms["policy_loss"] = policy_loss.detach()
                    loss_terms["unweighted_policy_loss"] = ori_policy_loss.mean().detach()

                    kl_div_loss = ((forward_prediction - ref_forward_prediction) ** 2).mean(
                        dim=tuple(range(1, x0.ndim))
                    )
                    loss += config.train.beta * torch.mean(kl_div_loss)
                    loss_terms["kl_div_loss"] = torch.mean(kl_div_loss).detach()
                    loss_terms["kl_div"] = loss_terms["kl_div_loss"]
                    loss_terms["old_kl_div"] = torch.mean(
                        ((old_prediction - ref_forward_prediction) ** 2).mean(dim=tuple(range(1, x0.ndim)))
                    ).detach()
                    loss_terms["x0_norm"] = torch.mean(x0**2).detach()
                    loss_terms["x0_norm_max"] = torch.max(x0**2).detach()
                    loss_terms["old_deviate"] = torch.mean((forward_prediction - old_prediction) ** 2).detach()
                    loss_terms["old_deviate_max"] = torch.max((forward_prediction - old_prediction) ** 2).detach()
                    loss_terms["total_loss"] = loss.detach()

                    scaled_loss = loss / effective_grad_accum_steps
                    if torch.isnan(scaled_loss) or torch.isinf(scaled_loss):
                        scaled_loss = scaled_loss * 0.0

                    if mixed_precision_dtype == torch.float16:
                        scaler.scale(scaled_loss).backward()
                    else:
                        scaled_loss.backward()
                    current_accumulated_steps += 1

                    for ki, vi in loss_terms.items():
                        info_accumulated[ki].append(vi)

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
                        log_info["grad_norm"] = (
                            grad_norm.detach().float().item() if torch.is_tensor(grad_norm) else float(grad_norm)
                        )
                        info_tensor = torch.tensor([log_info[k] for k in sorted(log_info)], device=device)
                        dist.all_reduce(info_tensor, op=dist.ReduceOp.AVG)
                        reduced = {k: info_tensor[i].item() for i, k in enumerate(sorted(log_info))}
                        if is_main_process(rank):
                            _log_dict = {
                                "global_step": global_step,
                                "gradient_update_times": gradient_update_times,
                                "epoch": epoch,
                                "inner_epoch": inner_epoch,
                                "current_time": time.time() - start_time,
                                **reduced,
                            }
                            wandb.log(_log_dict, commit=False)
                            logger.info(
                                "[step %d] loss=%.6f policy=%.6f grad=%.6f kl=%.6f",
                                global_step,
                                reduced.get("total_loss", 0),
                                reduced.get("policy_loss", 0),
                                reduced.get("grad_norm", 0),
                                reduced.get("kl_div_loss", 0),
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
            for src_p, tgt_p in zip(
                transformer_trainable_parameters, old_transformer_trainable_parameters, strict=True
            ):
                tgt_p.data.copy_(tgt_p.detach().data * decay + src_p.detach().clone().data * (1.0 - decay))

        for mtype, inf_model in inference_models.items():
            sync_lora_to_inference(transformer_ddp.module, unwrap_compiled(inf_model), adapter_name="old")

        time_logger.end("total_time")
        stats = time_logger.get_results()
        if is_main_process(rank):
            wandb.log({f"time/{k}": v for k, v in stats.items()}, commit=True)
            logger.info("Step %d Time: %s", global_step, stats)
        time_logger.empty_cache()

    if is_main_process(rank):
        wandb.finish()
    cleanup_distributed()


if __name__ == "__main__":
    app.run(main)
