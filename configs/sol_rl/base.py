import ml_collections


def get_config():
    config = ml_collections.ConfigDict()

    # run name for wandb logging and checkpoint saving -- if not provided, will be auto-generated based on the datetime.
    config.run_name = ""
    config.debug = False
    config.sana_config_file = ""

    # random seed for reproducibility.
    config.seed = 42
    # top-level logging directory for checkpoint saving.
    config.logdir = "logs"
    # number of epochs to train for. each epoch is one round of sampling from the model followed by training on those
    # samples.
    config.num_epochs = 100000
    # number of epochs between saving model checkpoints.
    config.save_freq = 5
    config.eval_freq = 20
    # mixed precision training. options are "fp16", "bf16", and "no". half-precision speeds up training significantly.
    config.mixed_precision = "bf16"
    # allow tf32 on Ampere GPUs, which can speed up training.
    config.allow_tf32 = True
    # resume training from a checkpoint. either an exact checkpoint directory (e.g. checkpoint_50), or a directory
    # containing checkpoints, in which case the latest one will be used. `config.use_lora` must be set to the same value
    # as the run that generated the saved checkpoint.
    config.resume_from = None
    config.resume = False
    # whether or not to use LoRA.
    config.use_lora = True
    config.dataset = ""
    config.resolution = 768

    config.pretrained = pretrained = ml_collections.ConfigDict()
    # base model to load. either a path to a local directory, or a model name from the HuggingFace model hub.
    pretrained.model = ""

    config.sample = sample = ml_collections.ConfigDict()
    # number of sampler inference steps.
    sample.num_steps = 40
    sample.eval_num_steps = 40
    sample.num_image_per_prompt = 24
    sample.test_batch_size = 1
    # noise level
    sample.noise_level = 1.0
    # number of best of n samples to select from each prompt
    sample.best_of_n = 24

    config.train = train = ml_collections.ConfigDict()
    # batch size (per GPU!) to use for training.
    train.batch_size = 1
    # learning rate.
    train.learning_rate = 3e-4
    # Adam beta1.
    train.adam_beta1 = 0.9
    # Adam beta2.
    train.adam_beta2 = 0.999
    # Adam weight decay.
    train.adam_weight_decay = 1e-4
    # Adam epsilon.
    train.adam_epsilon = 1e-8
    # number of gradient accumulation steps. the effective batch size is `batch_size * num_gpus *
    # gradient_accumulation_steps`.
    train.gradient_accumulation_steps = 1
    # maximum gradient norm for gradient clipping.
    train.max_grad_norm = 0.002
    # number of inner epochs per outer epoch. each inner epoch is one iteration through the data collected during one
    # outer epoch's round of sampling.
    train.num_inner_epochs = 1
    # clip advantages to the range [-adv_clip_max, adv_clip_max].
    train.adv_clip_max = 5
    # the fraction of timesteps to train on. if set to less than 1.0, the model will be trained on a subset of the
    # timesteps for each sample. this will speed up training but reduce the accuracy of policy gradient estimates.
    train.timestep_fraction = 0.6
    # kl ratio
    train.beta = 0.0001
    # pretrained lora path
    train.lora_path = None
    # LoRA hyper-parameters; defaults match current train_nft_sana_baseline_bon.py behavior.
    train.lora_rank = 32
    train.lora_alpha = 64
    train.lora_init_weights = True
    train.lora_target_modules = None
    train.ema = True

    config.prompt_fn = ""

    # reward function to use. see `rewards.py` for available reward functions.
    config.reward_fn = ml_collections.ConfigDict()
    config.save_dir = ""

    config.per_prompt_stat_tracking = True

    config.global_std = True
    config.after_adv = False

    config.beta = 1.0
    config.decay_type = 1
    config.rollout_sample_guidance_scale = 1.0
    config.train_sample_guidance_scale = 1.0
    config.eval_sample_guidance_scale = 1.0
    config.rollout_sample_num_steps = 10
    config.preview_step = 0
    config.sequential_decode = True

    config.enable_debug_image_save = True
    config.debug_image_every_steps = 10
    config.debug_image_subset_size = 6

    config.compile_mode = "max-autotune-no-cudagraphs"
    config.preview_model = "peft"
    config.fullrollout_model = "peft"
    config.nvfp4_skip_modules = []
    config.nvfp4_min_dim = 0

    return config
