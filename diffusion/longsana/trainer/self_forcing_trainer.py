import gc
import logging
import os
import time

import torch
import torch.distributed as dist
import wandb
from omegaconf import OmegaConf
from termcolor import colored
from torch.distributed.fsdp import FullOptimStateDictConfig, FullStateDictConfig
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import StateDictType
from torchvision.io import write_video

from diffusion.longsana.model import DMDSana
from diffusion.longsana.pipeline import SanaInferencePipeline
from diffusion.longsana.utils.dataset import ShardingLMDBDataset, TextDataset, TwoTextDataset, cycle
from diffusion.longsana.utils.debug_option import DEBUG
from diffusion.longsana.utils.distributed import EMA_FSDP, fsdp_state_dict, fsdp_wrap, launch_distributed_job
from diffusion.longsana.utils.misc import merge_dict_list, set_seed
from tools.download import find_model


class Trainer:
    def __init__(self, config):
        self.config = config
        self.step = 0

        # Step 1: Initialize the distributed training environment (rank, seed, dtype, logging etc.)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        launch_distributed_job()
        global_rank = dist.get_rank()
        self.world_size = dist.get_world_size()

        self.dtype = torch.bfloat16 if config.mixed_precision else torch.float32
        self.device = torch.cuda.current_device()
        self.is_main_process = global_rank == 0
        self.causal = config.causal
        self.disable_wandb = config.disable_wandb

        # use a random seed for the training
        if config.seed == 0:
            random_seed = torch.randint(0, 10000000, (1,), device=self.device)
            dist.broadcast(random_seed, src=0)
            config.seed = random_seed.item()

        set_seed(config.seed + global_rank)

        if self.is_main_process and not self.disable_wandb:
            if not wandb.api.api_key:
                wandb.login(key=config.wandb_key)
            wandb.init(
                config=OmegaConf.to_container(config, resolve=True),
                name=config.config_name,
                id=config.config_name,
                mode="online",
                entity=config.wandb_entity if config.wandb_entity else None,
                project=config.wandb_project,
                dir=config.wandb_save_dir,
                resume="allow",
            )

        self.output_path = config.logdir

        # Step 2: Initialize the model
        self.output_path = config.logdir

        # Step 2: Initialize the model and optimizer
        self.model = DMDSana(config, device=self.device)

        # Save pretrained model state_dicts to CPU
        self.fake_score_state_dict_cpu = self.model.fake_score.state_dict()

        self.model.generator = fsdp_wrap(
            self.model.generator,
            sharding_strategy=config.sharding_strategy,
            mixed_precision=config.mixed_precision,
            wrap_strategy=config.generator_fsdp_wrap_strategy,
        )

        self.model.real_score = fsdp_wrap(
            self.model.real_score,
            sharding_strategy=config.sharding_strategy,
            mixed_precision=config.mixed_precision,
            wrap_strategy=config.real_score_fsdp_wrap_strategy,
        )

        self.model.fake_score = fsdp_wrap(
            self.model.fake_score,
            sharding_strategy=config.sharding_strategy,
            mixed_precision=config.mixed_precision,
            wrap_strategy=config.fake_score_fsdp_wrap_strategy,
        )

        self.model.text_encoder = fsdp_wrap(
            self.model.text_encoder,
            sharding_strategy=config.sharding_strategy,
            mixed_precision=config.mixed_precision,
            wrap_strategy=config.text_encoder_fsdp_wrap_strategy,
            cpu_offload=getattr(config, "text_encoder_cpu_offload", False),
        )

        if not config.no_visualize or config.load_raw_video:
            self.model.vae = self.model.vae.to(
                device=self.device, dtype=torch.bfloat16 if config.mixed_precision else torch.float32
            )
        self.model._initialize_inference_pipeline()

        ##############################################################################################################
        # Set up EMA parameter containers
        rename_param = (
            lambda name: name.replace("_fsdp_wrapped_module.", "")
            .replace("_checkpoint_wrapped_module.", "")
            .replace("_orig_mod.", "")
        )
        self.name_to_trainable_params = {}
        for n, p in self.model.generator.named_parameters():
            if not p.requires_grad:
                continue

            renamed_n = rename_param(n)
            self.name_to_trainable_params[renamed_n] = p
        ema_weight = config.ema_weight
        self.generator_ema = None
        if (ema_weight is not None) and (ema_weight > 0.0):
            print(f"Setting up EMA with weight {ema_weight}")
            self.generator_ema = EMA_FSDP(self.model.generator, decay=ema_weight)

        ##############################################################################################################
        # (If resuming) Load the model and optimizer, lr_scheduler, ema's statedicts
        if getattr(config, "generator_ckpt", False):
            print(f"Loading pretrained generator from {config.generator_ckpt}")
            state_dict = find_model(config.generator_ckpt)
            if "generator" in state_dict:
                state_dict = state_dict["generator"]
            elif "model" in state_dict:
                state_dict = state_dict["model"]
            self.model.generator.load_state_dict(state_dict, strict=True)

        # Step 4: Initialize the optimizer
        self.generator_optimizer = torch.optim.AdamW(
            [param for param in self.model.generator.parameters() if param.requires_grad],
            lr=config.lr,
            betas=(config.beta1, config.beta2),
            weight_decay=config.weight_decay,
        )

        self.critic_optimizer = torch.optim.AdamW(
            [param for param in self.model.fake_score.parameters() if param.requires_grad],
            lr=config.lr_critic if hasattr(config, "lr_critic") else config.lr,
            betas=(config.beta1_critic, config.beta2_critic),
            weight_decay=config.weight_decay,
        )
        # Step 5: Initialize the dataloader
        if self.config.i2v:
            dataset = ShardingLMDBDataset(config.data_path, max_pair=int(1e8))
        elif getattr(self.config, "switch_prompt_path", None) is not None:
            dataset = TwoTextDataset(config.data_path, config.switch_prompt_path)
        else:
            dataset = TextDataset(config.data_path)

        sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=True, drop_last=True)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=config.batch_size, sampler=sampler, num_workers=8)

        if dist.get_rank() == 0:
            print("DATASET SIZE %d" % len(dataset))
        self.dataloader = cycle(dataloader)

        # Step 6: Initialize the validation dataloader for visualization (fixed prompts)
        self.fixed_vis_batch = None
        self.vis_interval = getattr(config, "vis_interval", -1)
        if self.vis_interval > 0 and len(getattr(config, "vis_video_lengths", [])) > 0:
            val_data_path = getattr(config, "val_data_path", None) or config.data_path
            val_dataset = TextDataset(val_data_path)

            if dist.get_rank() == 0:
                print("VAL DATASET SIZE %d" % len(val_dataset))

            sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False, drop_last=False)
            # Sequential sampling to keep prompts fixed
            val_dataloader = torch.utils.data.DataLoader(
                val_dataset,
                batch_size=getattr(config, "val_batch_size", 1),
                sampler=sampler,
                num_workers=8,
            )

            # Take the first batch as fixed visualization batch
            try:
                self.fixed_vis_batch = next(iter(val_dataloader))
            except StopIteration:
                self.fixed_vis_batch = None

            # ----------------------------------------------------------------------------------------------------------
            # Visualization settings
            # ----------------------------------------------------------------------------------------------------------
            self.vis_video_lengths = getattr(config, "vis_video_lengths", [])

            if self.vis_interval > 0 and len(self.vis_video_lengths) > 0:
                self._setup_visualizer()
        auto_resume = True
        # ================================= Model logic =================================
        checkpoint_path = None

        if auto_resume and self.output_path:
            latest_checkpoint = self.find_latest_checkpoint(self.output_path)
            if latest_checkpoint:
                checkpoint_path = latest_checkpoint
                if self.is_main_process:
                    print(f"Auto resume: Found latest checkpoint at {checkpoint_path}")
            else:
                if self.is_main_process:
                    print("Auto resume: No checkpoint found in logdir, starting from scratch")
        elif auto_resume:
            if self.is_main_process:
                print("Auto resume enabled but no logdir specified, starting from scratch")
        else:
            if self.is_main_process:
                print("Auto resume disabled, starting from scratch")

        if checkpoint_path is None:
            if getattr(config, "generator_ckpt", False):
                checkpoint_path = config.generator_ckpt
                if self.is_main_process:
                    print(f"Using explicit checkpoint: {checkpoint_path}")

        if checkpoint_path:
            if self.is_main_process:
                print(f"Loading checkpoint from {checkpoint_path}")
            checkpoint = find_model(checkpoint_path)

            # Load generator
            if "generator" in checkpoint:
                if self.is_main_process:
                    print(f"Loading pretrained generator from {checkpoint_path}")
                self.model.generator.load_state_dict(checkpoint["generator"], strict=True)
            elif "model" in checkpoint:
                if self.is_main_process:
                    print(f"Loading pretrained generator from {checkpoint_path}")
                self.model.generator.load_state_dict(checkpoint["model"], strict=True)
            else:
                if self.is_main_process:
                    print("Warning: Generator checkpoint not found.")

            # Load critic
            if "critic" in checkpoint:
                if self.is_main_process:
                    print(f"Loading pretrained critic from {checkpoint_path}")
                self.model.fake_score.load_state_dict(checkpoint["critic"], strict=True)
            else:
                if self.is_main_process:
                    print("Warning: Critic checkpoint not found.")

            # Load EMA
            if "generator_ema" in checkpoint and self.generator_ema is not None:
                if self.is_main_process:
                    print(f"Loading pretrained EMA from {checkpoint_path}")
                self.generator_ema.load_state_dict(checkpoint["generator_ema"])
            else:
                if self.is_main_process:
                    print("Warning: EMA checkpoint not found or EMA not initialized.")

            # For auto resume, always resume full training state
            # Load optimizers
            if "generator_optimizer" in checkpoint:
                if self.is_main_process:
                    print("Resuming generator optimizer...")
                gen_osd = FSDP.optim_state_dict_to_load(
                    self.model.generator,  # FSDP root module
                    self.generator_optimizer,  # newly created optimizer
                    checkpoint["generator_optimizer"],  # OSD at the time of saving
                )
                self.generator_optimizer.load_state_dict(gen_osd)
            else:
                if self.is_main_process:
                    print("Warning: Generator optimizer checkpoint not found.")

            if "critic_optimizer" in checkpoint:
                if self.is_main_process:
                    print("Resuming critic optimizer...")
                crit_osd = FSDP.optim_state_dict_to_load(
                    self.model.fake_score, self.critic_optimizer, checkpoint["critic_optimizer"]
                )
                self.critic_optimizer.load_state_dict(crit_osd)
            else:
                if self.is_main_process:
                    print("Warning: Critic optimizer checkpoint not found.")

            # Load training step
            if "step" in checkpoint:
                self.step = checkpoint["step"]
                if self.is_main_process:
                    print(f"Resuming from step {self.step}")
            else:
                if self.is_main_process:
                    print("Warning: Step not found in checkpoint, starting from step 0.")

        ##############################################################################################################

        # Let's delete EMA params for early steps to save some computes at training and inference
        if self.step < config.ema_start_step:
            self.generator_ema = None

        self.max_grad_norm_generator = getattr(config, "max_grad_norm_generator", 10.0)
        self.max_grad_norm_critic = getattr(config, "max_grad_norm_critic", 10.0)
        self.previous_time = None

        self.motion_score = getattr(config, "motion_score", 0)

    def _move_optimizer_to_device(self, optimizer, device):
        """Move optimizer state to the specified device."""
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)

    def find_latest_checkpoint(self, logdir):
        """Find the latest checkpoint in the logdir."""
        if not os.path.exists(logdir):
            return None

        checkpoint_dirs = []
        for item in os.listdir(logdir):
            if item.startswith("checkpoint_model_") and os.path.isdir(os.path.join(logdir, item)):
                try:
                    # Extract step number from directory name
                    step_str = item.replace("checkpoint_model_", "")
                    step = int(step_str)
                    checkpoint_path = os.path.join(logdir, item, "model.pt")
                    if os.path.exists(checkpoint_path):
                        checkpoint_dirs.append((step, checkpoint_path))
                except ValueError:
                    continue

        if not checkpoint_dirs:
            return None

        # Sort by step number and return the latest one
        checkpoint_dirs.sort(key=lambda x: x[0])
        latest_step, latest_path = checkpoint_dirs[-1]
        return latest_path

    def get_all_checkpoints(self, logdir):
        """Get all checkpoints in the logdir sorted by step number."""
        if not os.path.exists(logdir):
            return []

        checkpoint_dirs = []
        for item in os.listdir(logdir):
            if item.startswith("checkpoint_model_") and os.path.isdir(os.path.join(logdir, item)):
                try:
                    # Extract step number from directory name
                    step_str = item.replace("checkpoint_model_", "")
                    step = int(step_str)
                    checkpoint_dir_path = os.path.join(logdir, item)
                    checkpoint_file_path = os.path.join(checkpoint_dir_path, "model.pt")
                    if os.path.exists(checkpoint_file_path):
                        checkpoint_dirs.append((step, checkpoint_dir_path, item))
                except ValueError:
                    continue

        # Sort by step number (ascending order)
        checkpoint_dirs.sort(key=lambda x: x[0])
        return checkpoint_dirs

    def cleanup_old_checkpoints(self, logdir, max_checkpoints):
        """Remove old checkpoints if the number exceeds max_checkpoints.

        Only the main process performs the actual deletion to avoid race conditions
        in distributed training.
        """
        if max_checkpoints <= 0:
            return

        # Only main process should perform cleanup to avoid race conditions
        if not self.is_main_process:
            return

        checkpoints = self.get_all_checkpoints(logdir)
        if len(checkpoints) > max_checkpoints:
            # Calculate how many to remove
            num_to_remove = len(checkpoints) - max_checkpoints
            checkpoints_to_remove = checkpoints[:num_to_remove]  # Remove oldest ones

            print(
                f"Checkpoint cleanup: Found {len(checkpoints)} checkpoints, removing {num_to_remove} oldest ones (keeping {max_checkpoints})"
            )

            import shutil

            removed_count = 0
            for step, checkpoint_dir_path, dir_name in checkpoints_to_remove:
                try:
                    print(f"  Removing: {dir_name} (step {step})")
                    shutil.rmtree(checkpoint_dir_path)
                    removed_count += 1
                except Exception as e:
                    print(f"  Warning: Failed to remove checkpoint {dir_name}: {e}")

            print(f"Checkpoint cleanup completed: removed {removed_count}/{num_to_remove} old checkpoints")
        else:
            if len(checkpoints) > 0:
                print(
                    f"Checkpoint cleanup: Found {len(checkpoints)} checkpoints (max: {max_checkpoints}, no cleanup needed)"
                )

    def save(self):
        print("Start gathering distributed model states...")
        generator_state_dict = fsdp_state_dict(self.model.generator)
        critic_state_dict = fsdp_state_dict(self.model.fake_score)

        if self.config.ema_start_step < self.step:
            state_dict = {
                "generator": generator_state_dict,
                "critic": critic_state_dict,
                "generator_ema": self.generator_ema.state_dict(),
            }
        else:
            state_dict = {
                "generator": generator_state_dict,
                "critic": critic_state_dict,
            }

        state_dict["step"] = self.step

        if self.is_main_process:
            os.makedirs(os.path.join(self.output_path, f"checkpoint_model_{self.step:06d}"), exist_ok=True)
            torch.save(state_dict, os.path.join(self.output_path, f"checkpoint_model_{self.step:06d}", "model.pt"))
            print("Model saved to", os.path.join(self.output_path, f"checkpoint_model_{self.step:06d}", "model.pt"))

    @torch.no_grad()
    def get_text_embeddings(self, text_prompts, use_chi_prompt=True):
        if use_chi_prompt and self.motion_score > 0:
            text_prompts = [f"{prompt} motion score: {self.motion_score}." for prompt in text_prompts]
        return self.model.text_encoder.forward_chi(text_prompts=text_prompts, use_chi_prompt=use_chi_prompt)

    def fwdbwd_one_step(self, batch, train_generator):
        self.model.eval()  # prevent any randomness (e.g. dropout)

        if self.step % 20 == 0:
            torch.cuda.empty_cache()

        # Step 1: Get the next batch of text prompts
        text_prompts = batch["prompts"]
        if self.config.i2v:
            clean_latent = None
            image_latent = batch["ode_latent"][:, -1][
                :,
                0:1,
            ].to(device=self.device, dtype=self.dtype)
        else:
            clean_latent = None
            image_latent = None

        batch_size = len(text_prompts)
        image_or_video_shape = list(self.config.image_or_video_shape)
        image_or_video_shape[0] = batch_size

        # Step 2: Extract the conditional infos for sana
        #
        with torch.no_grad():
            conditional_dict = self.get_text_embeddings(text_prompts=text_prompts, use_chi_prompt=True)
            if self.config.get("negative_prompt", None) is not None:
                unconditional_dict = self.get_text_embeddings(
                    text_prompts=self.config.negative_prompt, use_chi_prompt=False
                )
            else:
                unconditional_dict = None

            if self.config.real_name == "SANA":
                conditional_dict_real = conditional_dict
                unconditional_dict_real = unconditional_dict
            else:
                conditional_dict_real = self.model.text_encoder_real(text_prompts=text_prompts)
                if self.config.get("negative_prompt_real", None) is not None:
                    unconditional_dict_real = self.model.text_encoder_real(
                        text_prompts=[self.config.negative_prompt_real] * batch_size
                    )
                    unconditional_dict_real = {k: v.detach() for k, v in unconditional_dict_real.items()}
                    self.unconditional_dict_real = unconditional_dict_real
                    if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
                        print(f"[SeqTrain-Trainer] Created and cached unconditional_dict_real")
                else:
                    unconditional_dict_real = None

        # Step 3: Store gradients for the generator (if training the generator)
        if train_generator:
            generator_loss, generator_log_dict = self.model.generator_loss(
                image_or_video_shape=image_or_video_shape,
                conditional_dict=conditional_dict,
                unconditional_dict=unconditional_dict,
                conditional_dict_real=conditional_dict_real,
                unconditional_dict_real=unconditional_dict_real,
                clean_latent=clean_latent,
                initial_latent=image_latent if self.config.i2v else None,
            )

            generator_loss.backward()
            generator_grad_norm = self.model.generator.clip_grad_norm_(self.max_grad_norm_generator)

            generator_log_dict.update({"generator_loss": generator_loss, "generator_grad_norm": generator_grad_norm})

            return generator_log_dict
        else:
            generator_log_dict = {}

        # Step 4: Store gradients for the critic (if training the critic)
        critic_loss, critic_log_dict = self.model.critic_loss(
            image_or_video_shape=image_or_video_shape,
            conditional_dict=conditional_dict,
            unconditional_dict=unconditional_dict,
            conditional_dict_real=conditional_dict_real,
            unconditional_dict_real=unconditional_dict_real,
            clean_latent=clean_latent,
            initial_latent=image_latent if self.config.i2v else None,
        )

        critic_loss.backward()
        critic_grad_norm = self.model.fake_score.clip_grad_norm_(self.max_grad_norm_critic)

        critic_log_dict.update({"critic_loss": critic_loss, "critic_grad_norm": critic_grad_norm})

        return critic_log_dict

    def train(self):
        start_step = self.step
        while True:
            TRAIN_GENERATOR = self.step % self.config.dfake_gen_update_ratio == 0
            self.model.set_step(self.step)
            # Train the generator
            if TRAIN_GENERATOR:
                self.generator_optimizer.zero_grad(set_to_none=True)
                extras_list = []
                batch = next(self.dataloader)
                extra = self.fwdbwd_one_step(batch, True)
                extras_list.append(extra)
                generator_log_dict = merge_dict_list(extras_list)
                self.generator_optimizer.step()
                if self.generator_ema is not None:
                    self.generator_ema.update(self.model.generator)

            # Train the critic
            self.critic_optimizer.zero_grad(set_to_none=True)
            extras_list = []
            batch = next(self.dataloader)
            extra = self.fwdbwd_one_step(batch, False)
            extras_list.append(extra)
            critic_log_dict = merge_dict_list(extras_list)
            self.critic_optimizer.step()

            # Increment the step since we finished gradient update
            self.step += 1

            # Create EMA params (if not already created)
            if (
                (self.step >= self.config.ema_start_step)
                and (self.generator_ema is None)
                and (self.config.ema_weight > 0)
            ):
                self.generator_ema = EMA_FSDP(self.model.generator, decay=self.config.ema_weight)

            # Save the model
            if (not self.config.no_save) and (self.step - start_step) > 0 and self.step % self.config.log_iters == 0:
                torch.cuda.empty_cache()
                self.save()
                torch.cuda.empty_cache()

            # Logging
            if self.is_main_process:
                wandb_loss_dict = {}
                if TRAIN_GENERATOR:
                    wandb_loss_dict.update(
                        {
                            "generator_loss": f"{generator_log_dict['generator_loss'].mean().item():.4f}",
                            "generator_grad_norm": f"{generator_log_dict['generator_grad_norm'].mean().item():.4f}",
                            "dmdtrain_gradient_norm": f"{generator_log_dict['dmdtrain_gradient_norm'].mean().item():.4f}",
                        }
                    )

                wandb_loss_dict.update(
                    {
                        "critic_loss": f"{critic_log_dict['critic_loss'].mean().item():.4f}",
                        "critic_grad_norm": f"{critic_log_dict['critic_grad_norm'].mean().item():.4f}",
                    }
                )

                if not self.disable_wandb:
                    wandb.log(wandb_loss_dict, step=self.step)
                wandb_loss_dict["step"] = self.step
                print(wandb_loss_dict)

            if self.step % self.config.gc_interval == 0:
                if dist.get_rank() == 0:
                    logging.info("DistGarbageCollector: Running GC.")
                gc.collect()
                torch.cuda.empty_cache()

            if self.is_main_process:
                current_time = time.time()
                if self.previous_time is None:
                    self.previous_time = current_time
                else:
                    if not self.disable_wandb:
                        wandb.log({"per iteration time": current_time - self.previous_time}, step=self.step)
                    self.previous_time = current_time

            if self.vis_interval > 0 and (self.step % self.vis_interval == 0 or (self.step - start_step) == 1):
                self._visualize()

            # Check if we've reached max iterations
            if self.step > self.config.max_iters:
                print(f"Reached max iterations: {self.step} > {self.config.max_iters}, stopping training")
                break

    def generate_video(self, pipeline, num_frames, prompts, image=None):
        batch_size = len(prompts)
        channel, h, w = self.config.image_or_video_shape[-3:]
        generator = torch.Generator(device=self.device).manual_seed(self.config.seed)
        if image is not None:
            image = image.squeeze(0).unsqueeze(0).unsqueeze(2).to(device=self.device, dtype=torch.bfloat16)

            # Encode the input image as the first latent
            initial_latent = pipeline.vae.encode_to_latent(image).to(device=self.device, dtype=torch.bfloat16)
            initial_latent = initial_latent.repeat(batch_size, 1, 1, 1, 1)
            sampled_noise = torch.randn(
                [batch_size, channel, num_frames - 1, h, w], device=self.device, dtype=self.dtype, generator=generator
            )
        else:
            initial_latent = None
            sampled_noise = torch.randn(
                [batch_size, channel, num_frames, h, w], device=self.device, dtype=self.dtype, generator=generator
            )
        with torch.no_grad():
            video_latent_btchw, _ = pipeline.inference(
                noise=sampled_noise,
                text_prompts=prompts,
                return_latents=True,
                initial_latent=initial_latent,
            )
            # B,T,C,H,W
            video_latent_bcthw = video_latent_btchw.permute(0, 2, 1, 3, 4)
            pixel_bcthw = pipeline.vae.decode_to_pixel(video_latent_bcthw)
            if isinstance(pixel_bcthw, list):
                pixel_bcthw = torch.stack(pixel_bcthw, dim=0)
            pixel_btchw = (
                torch.clamp(127.5 * pixel_bcthw + 127.5, 0, 255).permute(0, 2, 3, 4, 1).to(torch.uint8).cpu().numpy()
            )
        current_video = pixel_btchw
        # clear VAE cache
        try:
            if hasattr(pipeline, "vae"):
                if hasattr(pipeline.vae, "model") and hasattr(pipeline.vae.model, "clear_cache"):
                    pipeline.vae.model.clear_cache()
                elif hasattr(pipeline.vae, "vae") and hasattr(pipeline.vae.vae, "clear_cache"):
                    pipeline.vae.vae.clear_cache()
                elif hasattr(pipeline.vae, "clear_cache"):
                    pipeline.vae.clear_cache()
        except Exception as _e:
            if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
                print(f"[Trainer] VAE cache clear skipped: {_e}")
        return current_video

    def _setup_visualizer(self):
        """Initialize the inference pipeline for visualization on CPU, to be moved to GPU only when needed."""

        # use SANA inference pipeline for visualization
        self.vis_pipeline = SanaInferencePipeline(
            args=self.config,
            device=self.device,
            generator=self.model.generator,
            text_encoder=self.model.text_encoder,
            vae=self.model.vae,
        )

        self.vis_output_dir = os.path.join(self.output_path, "vis")
        os.makedirs(self.vis_output_dir, exist_ok=True)
        if self.config.vis_ema:
            raise NotImplementedError("Visualization with EMA is not implemented")

    def _visualize(self):
        """Generate and save sample videos to monitor training progress."""
        if self.vis_interval <= 0 or not hasattr(self, "vis_pipeline"):
            return

        # Use the fixed batch of prompts/images prepared from val_loader
        if not getattr(self, "fixed_vis_batch", None):
            print("[Warning] No fixed validation batch available for visualization.")
            return

        step_vis_dir = os.path.join(self.vis_output_dir, f"step_{self.step:07d}")
        os.makedirs(step_vis_dir, exist_ok=True)
        batch = self.fixed_vis_batch
        prompts = batch["prompts"]
        mode_info = ""

        for vid_len in self.vis_video_lengths:
            print(f"Generating video of length {vid_len}")
            videos = self.generate_video(self.vis_pipeline, vid_len, prompts)

            # Save each sample
            for idx, video_np in enumerate(videos):
                video_name = f"step_{self.step:07d}_rank_{dist.get_rank()}_sample_{idx}_len_{vid_len}{mode_info}.mp4"
                out_path = os.path.join(
                    step_vis_dir,
                    video_name,
                )
                video_tensor = torch.from_numpy(video_np.astype("uint8"))
                write_video(out_path, video_tensor, fps=16)

            # After saving current length videos, release related tensors to reduce peak memory
            del videos, video_np, video_tensor
            torch.cuda.empty_cache()

        torch.cuda.empty_cache()
        import gc

        gc.collect()
