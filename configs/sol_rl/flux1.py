import imp
import os

base = imp.load_source("base", os.path.join(os.path.dirname(__file__), "base.py"))


def get_config(name):
    return globals()[name]()


def _get_config(n_gpus=8, dataset="pickscore", reward_fn=None):
    if reward_fn is None:
        reward_fn = {"pickscore": 1.0}

    config = base.get_config()
    config.dataset = os.path.join(os.getcwd(), f"dataset/{dataset}")
    config.pretrained.model = "black-forest-labs/FLUX.1-dev"
    config.resolution = 512

    config.train.lora_target_modules = ["to_k", "to_q", "to_v", "to_out.0"]
    config.train.flux_lora_target_modules = config.train.lora_target_modules
    config.train.lora_init_mode = "gaussian"

    config.sample.num_steps = 10
    config.sample.eval_num_steps = 28
    config.sample.noise_level = 0.7
    config.sample.solver = "dpm2"
    config.sample.stage1_select_mode = "best_worst"
    config.sample.stage2_select_mode = "best_worst"
    config.sample.test_batch_size = 16
    if n_gpus > 32:
        config.sample.test_batch_size = max(1, config.sample.test_batch_size // 2)

    config.prompt_fn = "geneval" if dataset == "geneval" else "general_ocr"
    config.reward_fn = reward_fn
    config.train.beta = 0.0001
    config.train.timestep_fraction = 0.4
    config.decay_type = 1
    config.beta = 1.0
    config.train.adv_mode = "all"
    config.rollout_sample_num_steps = 10
    config.rollout_sample_guidance_scale = 1.0
    config.train_sample_guidance_scale = 1.0
    config.eval_sample_guidance_scale = 1.0
    config.preview_step = 0
    config.preview_type = "flow"
    config.sequential_decode = True
    config.enable_debug_image_save = True
    config.debug_image_every_steps = 10
    config.debug_image_subset_size = 6

    config.compile_mode = "max-autotune-no-cudagraphs"
    config.nvfp4_skip_modules = ["x_embedder", "context_embedder", "time_text_embed", "norm_out"]
    config.nvfp4_min_dim = 1000

    config.preview_model = "peft"
    config.fullrollout_model = "peft"
    config.train.max_grad_norm = 1.0
    config.resume = True
    config.global_std = True
    config.after_adv = False

    return config


def _set_bon_combo(
    config, reward_name, best_of_n, num_image_per_prompt, n_gpus, grad_steps_per_epoch, rollout_bsz, train_bsz, seed=42
):
    config.sample.num_image_per_prompt = int(num_image_per_prompt)
    config.sample.best_of_n = int(best_of_n)
    config.sample.full_rollout_num = int(best_of_n)

    num_groups = 48
    assert num_groups % n_gpus == 0
    config.sample.per_gpu_to_process_prompts = num_groups // n_gpus
    config.sample.per_gpu_total_samples_to_train = num_groups * config.sample.best_of_n // n_gpus

    assert config.sample.num_image_per_prompt % rollout_bsz == 0
    config.sample.rollout_batch_size = rollout_bsz
    config.sample.per_prompt_iter_num = config.sample.num_image_per_prompt // rollout_bsz

    assert config.sample.per_gpu_total_samples_to_train % train_bsz == 0
    config.train.batch_size = int(train_bsz)
    n_batch_per_epoch = config.sample.per_gpu_total_samples_to_train // train_bsz
    assert n_batch_per_epoch % grad_steps_per_epoch == 0
    config.train.n_batch_per_epoch = n_batch_per_epoch
    config.train.gradient_accumulation_steps = n_batch_per_epoch // grad_steps_per_epoch

    config.global_std = True
    config.after_adv = False
    config.seed = int(seed)
    return config


def _auto_rollout_bsz(num_image_per_prompt):
    num_image_per_prompt = int(num_image_per_prompt)
    max_rollout_bsz = min(24, num_image_per_prompt)
    for candidate in range(max_rollout_bsz, 0, -1):
        if num_image_per_prompt % candidate == 0:
            return int(candidate)
    return 1


def _build_reward(reward_name, best_of_n, num_image_per_prompt):
    reward_map = {
        "pickscore": {"pickscore": 1.0},
        "clipscore": {"clipscore": 1.0},
        "imagereward": {"imagereward": 1.0},
        "hpsv2": {"hpsv2": 1.0},
    }
    cfg = _get_config(n_gpus=8, dataset="pickscore", reward_fn=reward_map[reward_name])
    rollout_bsz = _auto_rollout_bsz(num_image_per_prompt)
    return _set_bon_combo(
        cfg,
        reward_name,
        best_of_n,
        num_image_per_prompt,
        n_gpus=8,
        grad_steps_per_epoch=1,
        rollout_bsz=rollout_bsz,
        train_bsz=12,
        seed=42,
    )


def _set_run(cfg, name):
    cfg.run_name = name
    cfg.resume_from = f"logs/nft_slurm/{name}"
    cfg.save_dir = cfg.resume_from
    return cfg


# ============================================================================
# DiffusionNFT Baseline: 24-in-24, PEFT inference
# ============================================================================


def flux1_diffusionnft_pickscore():
    return _set_run(_build_reward("pickscore", 24, 24), "flux1_diffusionnft_pickscore")


def flux1_diffusionnft_clipscore():
    return _set_run(_build_reward("clipscore", 24, 24), "flux1_diffusionnft_clipscore")


def flux1_diffusionnft_hpsv2():
    return _set_run(_build_reward("hpsv2", 24, 24), "flux1_diffusionnft_hpsv2")


def flux1_diffusionnft_imagereward():
    return _set_run(_build_reward("imagereward", 24, 24), "flux1_diffusionnft_imagereward")


# ============================================================================
# Naive Scaling: 24-in-96, PEFT model
# ============================================================================


def flux1_naive_scaling_pickscore():
    return _set_run(_build_reward("pickscore", 24, 96), "flux1_naive_scaling_pickscore")


def flux1_naive_scaling_clipscore():
    return _set_run(_build_reward("clipscore", 24, 96), "flux1_naive_scaling_clipscore")


def flux1_naive_scaling_hpsv2():
    return _set_run(_build_reward("hpsv2", 24, 96), "flux1_naive_scaling_hpsv2")


def flux1_naive_scaling_imagereward():
    return _set_run(_build_reward("imagereward", 24, 96), "flux1_naive_scaling_imagereward")


# ============================================================================
# Compile: 24-in-96, BF16 compile model
# ============================================================================


def flux1_compile_pickscore():
    cfg = _build_reward("pickscore", 24, 96)
    cfg.fullrollout_model = "compile"
    return _set_run(cfg, "flux1_compile_pickscore")


def flux1_compile_clipscore():
    cfg = _build_reward("clipscore", 24, 96)
    cfg.fullrollout_model = "compile"
    return _set_run(cfg, "flux1_compile_clipscore")


def flux1_compile_hpsv2():
    cfg = _build_reward("hpsv2", 24, 96)
    cfg.fullrollout_model = "compile"
    return _set_run(cfg, "flux1_compile_hpsv2")


def flux1_compile_imagereward():
    cfg = _build_reward("imagereward", 24, 96)
    cfg.fullrollout_model = "compile"
    return _set_run(cfg, "flux1_compile_imagereward")


# ============================================================================
# Naive Quant: 24-in-96, NVFP4 compile model (direct quantized rollout)
# ============================================================================


def flux1_naive_quant_pickscore():
    cfg = _build_reward("pickscore", 24, 96)
    cfg.fullrollout_model = "compile_nvfp4"
    return _set_run(cfg, "flux1_naive_quant_pickscore")


def flux1_naive_quant_clipscore():
    cfg = _build_reward("clipscore", 24, 96)
    cfg.fullrollout_model = "compile_nvfp4"
    return _set_run(cfg, "flux1_naive_quant_clipscore")


def flux1_naive_quant_hpsv2():
    cfg = _build_reward("hpsv2", 24, 96)
    cfg.fullrollout_model = "compile_nvfp4"
    return _set_run(cfg, "flux1_naive_quant_hpsv2")


def flux1_naive_quant_imagereward():
    cfg = _build_reward("imagereward", 24, 96)
    cfg.fullrollout_model = "compile_nvfp4"
    return _set_run(cfg, "flux1_naive_quant_imagereward")


# ============================================================================
# Sol-RL: 24-in-96, Two-stage decoupled framework
#   Stage 1 (FP4 Exploration): NVFP4 compiled draft model, 6 steps
#   Stage 2 (BF16 Regeneration): BF16 compiled model, 10 steps
# ============================================================================


def flux1_sol_rl_pickscore():
    cfg = _build_reward("pickscore", 24, 96)
    cfg.preview_step = 6
    cfg.preview_model = "compile_nvfp4"
    cfg.fullrollout_model = "compile"
    return _set_run(cfg, "flux1_sol_rl_pickscore")


def flux1_sol_rl_clipscore():
    cfg = _build_reward("clipscore", 24, 96)
    cfg.preview_step = 6
    cfg.preview_model = "compile_nvfp4"
    cfg.fullrollout_model = "compile"
    return _set_run(cfg, "flux1_sol_rl_clipscore")


def flux1_sol_rl_hpsv2():
    cfg = _build_reward("hpsv2", 24, 96)
    cfg.preview_step = 6
    cfg.preview_model = "compile_nvfp4"
    cfg.fullrollout_model = "compile"
    return _set_run(cfg, "flux1_sol_rl_hpsv2")


def flux1_sol_rl_imagereward():
    cfg = _build_reward("imagereward", 24, 96)
    cfg.preview_step = 6
    cfg.preview_model = "compile_nvfp4"
    cfg.fullrollout_model = "compile"
    return _set_run(cfg, "flux1_sol_rl_imagereward")
