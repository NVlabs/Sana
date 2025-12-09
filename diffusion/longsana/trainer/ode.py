import gc
import logging
import os
import shutil
import time
from collections import defaultdict

import imageio.v3 as iio
import numpy as np
import torch
import torch.distributed as dist
import wandb
from omegaconf import OmegaConf
from torch.distributed.fsdp import FullOptimStateDictConfig, FullStateDictConfig
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import StateDictType
from torchvision.io import write_video

from diffusion.longsana.model import ODERegressionSana
from diffusion.longsana.pipeline.sana_inference_pipeline import SanaInferencePipeline
from diffusion.longsana.utils.dataset import ODERegressionLMDBDataset, TextDataset, cycle
from diffusion.longsana.utils.distributed import barrier, fsdp_wrap, launch_distributed_job
from diffusion.longsana.utils.misc import set_seed
from tools.download import find_model


class ODESANATrainer:
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
        self.global_rank = global_rank
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

        # Step 2: Initialize the model and optimizer
        assert config.distribution_loss == "ode", "Only ODE loss is supported for ODE training"
        self.model = ODERegressionSana(config, device=self.device)

        self.model.generator = fsdp_wrap(
            self.model.generator,
            sharding_strategy=config.sharding_strategy,
            mixed_precision=config.mixed_precision,
            wrap_strategy=config.generator_fsdp_wrap_strategy,
        )
        self.model.text_encoder = fsdp_wrap(
            self.model.text_encoder,
            sharding_strategy=config.sharding_strategy,
            mixed_precision=config.mixed_precision,
            wrap_strategy=config.text_encoder_fsdp_wrap_strategy,
            cpu_offload=getattr(config, "text_encoder_cpu_offload", False),
        )

        if not config.no_visualize:
            self.model.vae = self.model.vae.to(
                device=self.device, dtype=torch.bfloat16 if config.mixed_precision else torch.float32
            )

        # Step 4: Initialize the optimizer
        self.generator_optimizer = torch.optim.AdamW(
            [param for param in self.model.generator.parameters() if param.requires_grad],
            lr=config.lr,
            betas=(config.beta1, config.beta2),
            weight_decay=config.weight_decay,
        )

        # Step 5: Initialize the dataloader
        dataset = ODERegressionLMDBDataset(config.data_path, max_pair=getattr(config, "max_pair", int(1e8)))
        sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=True, drop_last=True)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=config.batch_size, sampler=sampler, num_workers=8)
        total_batch_size = getattr(config, "total_batch_size", None)
        if total_batch_size is not None:
            assert (
                total_batch_size == config.batch_size * self.world_size
            ), "Gradient accumulation is not supported for ODE training"
        self.dataloader = cycle(dataloader)

        # Step 6: Initialize the validation dataloader for visualization (fixed prompts)
        self.fixed_vis_batch = None
        self.vis_interval = getattr(config, "vis_interval", -1)
        if self.vis_interval > 0 and len(getattr(config, "vis_video_lengths", [])) > 0:
            # Determine validation data path
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

        self.step = 0

        ##############################################################################################################
        # Auto resume configuration
        auto_resume = getattr(config, "auto_resume", True)  # Default to True

        checkpoint_path = None

        if auto_resume and self.output_path:
            # Auto resume: find latest checkpoint in logdir
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

            # Load optimizer state
            if "generator_optimizer" in checkpoint:
                if self.is_main_process:
                    print("Resuming generator optimizer...")
                gen_osd = FSDP.optim_state_dict_to_load(
                    self.model.generator,
                    self.generator_optimizer,
                    checkpoint["generator_optimizer"],
                )
                self.generator_optimizer.load_state_dict(gen_osd)
            else:
                if self.is_main_process:
                    print("Warning: Generator optimizer checkpoint not found.")

            # Load training step
            if "step" in checkpoint:
                self.step = checkpoint["step"]
                if self.is_main_process:
                    print(f"Resuming from step {self.step}")
            else:
                if self.is_main_process:
                    print("Warning: Step not found in checkpoint, starting from step 0.")

        ##############################################################################################################

        self.max_grad_norm = 10.0
        self.previous_time = None

        self.motion_score = getattr(config, "motion_score", 0)

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
        # save the inference model

        # Gather full state dict with optimizer support
        with FSDP.state_dict_type(
            self.model.generator,
            StateDictType.FULL_STATE_DICT,
            FullStateDictConfig(rank0_only=True, offload_to_cpu=True),
            FullOptimStateDictConfig(rank0_only=True),
        ):
            generator_state_dict = self.model.generator.state_dict()
            generator_optim_state_dict = FSDP.optim_state_dict(self.model.generator, self.generator_optimizer)

        state_dict = {
            "generator": generator_state_dict,
            "generator_optimizer": generator_optim_state_dict,
            "step": self.step,
        }

        if self.is_main_process:
            checkpoint_dir = os.path.join(self.output_path, f"checkpoint_model_{self.step:06d}")
            os.makedirs(checkpoint_dir, exist_ok=True)
            checkpoint_file = os.path.join(checkpoint_dir, "model.pt")
            torch.save(state_dict, checkpoint_file)
            print("Model saved to", checkpoint_file)
            generator_checkpoint_file = os.path.join(checkpoint_dir, "generator.pth")
            torch.save(state_dict["generator"], generator_checkpoint_file)
            print("Generator saved to", generator_checkpoint_file)
            inference_model_state_dict = self.model.inference_model.state_dict()
            inference_model_checkpoint_file = os.path.join(checkpoint_dir, "inference_model.pth")
            torch.save(inference_model_state_dict, inference_model_checkpoint_file)
            print("Inference model saved to", inference_model_checkpoint_file)

            # Cleanup old checkpoints if max_checkpoints is set
            max_checkpoints = getattr(self.config, "max_checkpoints", 0)
            if max_checkpoints > 0:
                self.cleanup_old_checkpoints(self.output_path, max_checkpoints)

        torch.cuda.empty_cache()
        gc.collect()

    @torch.no_grad()
    def get_text_embeddings(self, text_prompts, use_chi_prompt=True):
        if use_chi_prompt and self.motion_score > 0:
            text_prompts = [f"{prompt} motion score: {self.motion_score}." for prompt in text_prompts]
        return self.model.text_encoder.forward_chi(text_prompts=text_prompts, use_chi_prompt=use_chi_prompt)

    def _setup_visualizer(self):
        """Initialize the inference pipeline for visualization on CPU, to be moved to GPU only when needed."""

        # Use SANA inference pipeline for visualization
        self.vis_pipeline = SanaInferencePipeline(
            args=self.config,
            device=self.device,
            generator=self.model.inference_model,
            text_encoder=self.model.text_encoder,
            vae=self.model.vae,
            num_cached_blocks=self.config.get("num_cached_blocks", -1),
        )

        self.vis_output_dir = os.path.join(self.output_path, "vis")
        os.makedirs(self.vis_output_dir, exist_ok=True)

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
            # B,T,C,H,W
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
        try:
            if hasattr(pipeline, "vae"):
                if hasattr(pipeline.vae, "model") and hasattr(pipeline.vae.model, "clear_cache"):
                    pipeline.vae.model.clear_cache()
                elif hasattr(pipeline.vae, "vae") and hasattr(pipeline.vae.vae, "clear_cache"):
                    pipeline.vae.vae.clear_cache()
                elif hasattr(pipeline.vae, "clear_cache"):
                    pipeline.vae.clear_cache()
        except Exception as _e:
            if not dist.is_initialized() or dist.get_rank() == 0:
                print(f"[Trainer] VAE cache clear skipped: {_e}")
        return current_video

    def _visualize(self):
        """Generate and save sample videos to monitor training progress."""
        if self.vis_interval <= 0:
            return

        # Use the fixed batch of prompts/images prepared from val_loader
        if not getattr(self, "fixed_vis_batch", None):
            print("[Warning] No fixed validation batch available for visualization.")
            return

        self._setup_visualizer()
        step_vis_dir = os.path.join(self.vis_output_dir, f"step_{self.step:07d}")
        if self.is_main_process:
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

            del videos, video_np, video_tensor
            torch.cuda.empty_cache()

        torch.cuda.empty_cache()
        gc.collect()

    def train_one_step(self):
        VISUALIZE = self.step % self.config.vis_interval == 0
        self.model.eval()  # prevent any randomness (e.g. dropout)

        # Step 1: Get the next batch of text prompts
        batch = next(self.dataloader)
        text_prompts = batch["prompts"]
        ode_latent = batch["ode_latent"].to(device=self.device, dtype=self.dtype)

        # Step 2: Extract the conditional infos
        with torch.no_grad():
            conditional_dict = self.get_text_embeddings(text_prompts=text_prompts)

        # Step 3: Train the generator
        generator_loss, log_dict = self.model.generator_loss(ode_latent=ode_latent, conditional_dict=conditional_dict)

        unnormalized_loss = log_dict["unnormalized_loss"]
        timestep = log_dict["timestep"]

        if self.world_size > 1:
            gathered_unnormalized_loss = torch.zeros(
                [self.world_size, *unnormalized_loss.shape], dtype=unnormalized_loss.dtype, device=self.device
            )
            gathered_timestep = torch.zeros(
                [self.world_size, *timestep.shape], dtype=timestep.dtype, device=self.device
            )

            dist.all_gather_into_tensor(gathered_unnormalized_loss, unnormalized_loss)
            dist.all_gather_into_tensor(gathered_timestep, timestep)
        else:
            gathered_unnormalized_loss = unnormalized_loss
            gathered_timestep = timestep

        loss_breakdown = defaultdict(list)
        stats = {}

        for index, t in enumerate(timestep):
            loss_breakdown[str(int(t.item()) // 250 * 250)].append(unnormalized_loss[index].item())

        for key_t in loss_breakdown.keys():
            stats["loss_at_time_" + key_t] = sum(loss_breakdown[key_t]) / len(loss_breakdown[key_t])

        if self.is_main_process and self.step % 10 == 0:
            print(f"step {self.step}, generator_loss {generator_loss}")

        self.generator_optimizer.zero_grad()
        generator_loss.backward()
        generator_grad_norm = self.model.generator.clip_grad_norm_(self.max_grad_norm)
        self.generator_optimizer.step()

        # Step 4: Visualization
        if VISUALIZE and not self.config.no_visualize:
            # Gather full state dict with optimizer support
            with FSDP.state_dict_type(
                self.model.generator,
                StateDictType.FULL_STATE_DICT,
                FullStateDictConfig(rank0_only=False, offload_to_cpu=True),
            ):
                generator_state_dict = self.model.generator.state_dict()
            self.model.inference_model.load_state_dict(generator_state_dict, strict=True)
            self._visualize()

            local_save_dir = os.path.join(self.output_path, f"log_vis")
            if self.is_main_process:
                os.makedirs(local_save_dir, exist_ok=True)
            # Visualize the input, output, and ground truth
            input = log_dict["input"]  # B, T, C, H, W
            output = log_dict["output"]  # B, T, C, H, W
            timestep_list = log_dict["timestep_list"]  # B, T
            ground_truth = ode_latent[:, -1]  # B, T, C, H, W
            with torch.no_grad():
                input_video = self.model.vae.decode_to_pixel(input.permute(0, 2, 1, 3, 4))
                output_video = self.model.vae.decode_to_pixel(output.permute(0, 2, 1, 3, 4))
                ground_truth_video = self.model.vae.decode_to_pixel(ground_truth.permute(0, 2, 1, 3, 4))

            input_video = (255.0 * (input_video[0].permute(1, 2, 3, 0).cpu().numpy() * 0.5 + 0.5)).astype(
                np.uint8
            )  # T, H, W, C
            output_video = (255.0 * (output_video[0].permute(1, 2, 3, 0).cpu().numpy() * 0.5 + 0.5)).astype(
                np.uint8
            )  # T, H, W, C
            ground_truth_video = (255.0 * (ground_truth_video[0].permute(1, 2, 3, 0).cpu().numpy() * 0.5 + 0.5)).astype(
                np.uint8
            )  # T, H, W, C
            rank = dist.get_rank()
            if rank < 8:
                iio.imwrite(
                    os.path.join(
                        local_save_dir,
                        f"step_{self.step:06d}_rank_{rank}_input_{int(timestep_list[0,0].item())}_{int(timestep_list[0,-1].item())}.mp4",
                    ),
                    input_video,
                    fps=16,
                )
                iio.imwrite(
                    os.path.join(local_save_dir, f"step_{self.step:06d}_rank_{rank}_output.mp4"), output_video, fps=16
                )
                iio.imwrite(
                    os.path.join(local_save_dir, f"step_{self.step:06d}_rank_{rank}_ground_truth.mp4"),
                    ground_truth_video,
                    fps=16,
                )

        # Step 5: Logging
        if self.is_main_process and not self.disable_wandb:
            wandb_loss_dict = {
                "generator_loss": generator_loss.item(),
                "generator_grad_norm": generator_grad_norm.item(),
                **stats,
            }
            wandb.log(wandb_loss_dict, step=self.step)

        if self.step % self.config.gc_interval == 0:
            if dist.get_rank() == 0:
                logging.info("DistGarbageCollector: Running GC.")
            gc.collect()

    def train(self):
        while True:
            self.train_one_step()
            if (not self.config.no_save) and self.step % self.config.log_iters == 0 and self.step != 0:
                self.save()
                torch.cuda.empty_cache()

            barrier()
            if self.is_main_process:
                current_time = time.time()
                if self.previous_time is None:
                    self.previous_time = current_time
                else:
                    if not self.disable_wandb:
                        wandb.log({"per iteration time": current_time - self.previous_time}, step=self.step)
                    self.previous_time = current_time

            self.step += 1
            if self.step > self.config.max_iters:
                break
