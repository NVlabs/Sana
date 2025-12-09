import gc
import logging
import random
import time

import numpy as np
import torch
import torch.distributed as dist
import wandb
from einops import rearrange
from omegaconf import OmegaConf
from termcolor import colored
from torch.distributed.fsdp import FullOptimStateDictConfig, FullStateDictConfig
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import StateDictType
from torchvision.io import write_video

from diffusion.longsana.model import StreamingSANATrainingModel
from diffusion.longsana.utils.debug_option import DEBUG, DEBUG_GRADIENT, LOG_GPU_MEMORY
from diffusion.longsana.utils.distributed import EMA_FSDP, fsdp_state_dict, fsdp_wrap, launch_distributed_job
from diffusion.longsana.utils.misc import merge_dict_list, set_seed

from .self_forcing_trainer import Trainer as SelfForcingScoreDistillationTrainer


class LongSANATrainer(SelfForcingScoreDistillationTrainer):
    def __init__(self, config):
        super().__init__(config)
        # streaming training configuration
        self.streaming_training = getattr(config, "streaming_training", True)
        self.streaming_chunk_size = getattr(config, "streaming_chunk_size", 21)
        self.streaming_max_length = getattr(config, "streaming_max_length", 63)

        # Create streaming training model if enabled
        if self.streaming_training:
            self.streaming_model = StreamingSANATrainingModel(self.model, config)
            if self.is_main_process:
                print(
                    f"streaming training enabled: chunk_size={self.streaming_chunk_size}, max_length={self.streaming_max_length}"
                )
        else:
            self.streaming_model = None

        # streaming training state (simplified)
        self.streaming_active = False  # Whether we're currently in a sequence
        self.gradient_accumulation_steps = getattr(config, "gradient_accumulation_steps", 1)
        self.is_lora_enabled = getattr(config, "is_lora_enabled", False)

    def start_new_sequence(self):
        if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
            print(f"[SeqTrain-Trainer] start_new_sequence called")

        # Fetch a new batch
        if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
            print(f"[SeqTrain-Trainer] start_new_sequence: fetch new batch")
        batch = next(self.dataloader)

        # Prepare conditional information
        text_prompts = batch["prompts"]
        print(f"[SeqTrain-Trainer] text_prompts={text_prompts}")
        if self.config.i2v:
            image_latent = batch["ode_latent"][:, -1][
                :,
                0:1,
            ].to(device=self.device, dtype=self.dtype)
        else:
            image_latent = None

        batch_size = len(text_prompts)
        image_or_video_shape = list(self.config.image_or_video_shape)
        image_or_video_shape[0] = batch_size
        self.current_text_prompts = text_prompts

        if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
            print(f"[SeqTrain-Trainer] Setting up sequence: batch_size={batch_size}, i2v={self.config.i2v}")
            print(f"[SeqTrain-Trainer] image_or_video_shape={image_or_video_shape}")

        with torch.no_grad():
            primary_conditional_dict = self.get_text_embeddings(text_prompts=text_prompts, use_chi_prompt=True)
            if self.config.get("negative_prompt", None) is not None:
                primary_unconditional_dict = self.get_text_embeddings(
                    text_prompts=self.config.negative_prompt, use_chi_prompt=False
                )
            else:
                primary_unconditional_dict = None

            if self.config.real_name == "SANA":
                primary_conditional_dict_real = primary_conditional_dict
                primary_unconditional_dict_real = primary_unconditional_dict
            else:
                primary_conditional_dict_real = self.model.text_encoder_real(text_prompts=text_prompts)
                if self.config.get("negative_prompt_real", None) is not None:
                    primary_unconditional_dict_real = self.model.text_encoder_real(
                        text_prompts=[self.config.negative_prompt_real] * batch_size
                    )
                    primary_unconditional_dict_real = {
                        k: v.detach() for k, v in primary_unconditional_dict_real.items()
                    }
                    self.unconditional_dict_real = primary_unconditional_dict_real
                    if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
                        print(f"[SeqTrain-Trainer] Created and cached unconditional_dict_real")
                else:
                    primary_unconditional_dict_real = None

        temp_max_length = self.streaming_model.max_length

        if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
            print(
                f"[SeqTrain-Model] Selected temporary max length: {temp_max_length} (from {self.streaming_model.possible_max_length})"
            )

        # Handle DMD Switch related information
        switch_conditional_dict = None
        switch_frame_index = None
        if getattr(self.config, "switch_prompt_path", None) is not None:
            if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
                print(f"[SeqTrain-Trainer] Processing Switch info")

            switch_text_prompts = batch["switch_prompts"]
            print(f"[SeqTrain-Trainer] switch_text_prompts={switch_text_prompts}")
            with torch.no_grad():
                switch_conditional_dict = self.get_text_embeddings(
                    text_prompts=switch_text_prompts, use_chi_prompt=True
                )

            switch_frame_index = self._get_switch_frame_index(temp_max_length)

            if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
                print(f"[SeqTrain-Trainer] switch_frame_index={switch_frame_index}")

        # Set up the sequence
        if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
            print(f"[SeqTrain-Trainer] Calling streaming_model.setup_sequence")

        self.streaming_model.setup_sequence(
            conditional_dict=primary_conditional_dict,
            unconditional_dict=primary_unconditional_dict,
            conditional_dict_real=primary_conditional_dict_real,
            unconditional_dict_real=primary_unconditional_dict_real,
            initial_latent=image_latent,
            switch_conditional_dict=switch_conditional_dict,
            switch_frame_index=switch_frame_index,
            temp_max_length=temp_max_length,
        )

        self.streaming_active = True

        if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
            print(f"[SeqTrain-Trainer] streaming training sequence setup completed")

    def fwdbwd_one_step_streaming(self, train_generator):
        """Forward/backward pass using the new StreamingTrainingModel for serialized training"""
        self.model.eval()  # prevent any randomness (e.g. dropout)

        if self.step % 5 == 0:
            torch.cuda.empty_cache()

        # If no active sequence, start a new one
        if not self.streaming_active:
            if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
                print(f"[SeqTrain-Trainer] No active sequence, starting new one")
            self.start_new_sequence()

        # Check whether we can generate more chunks
        if not self.streaming_model.can_generate_more():
            # Current sequence is finished; start a new one
            if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
                print(f"[SeqTrain-Trainer] Current sequence completed, starting new one")
            self.streaming_active = False
            self.start_new_sequence()

        self.kv_cache_before_generator_rollout = None
        self.kv_cache_after_generator_rollout = None
        self.kv_cache_after_generator_backward = None
        self.kv_cache_before_critic_rollout = None
        self.kv_cache_after_critic_rollout = None
        self.kv_cache_after_critic_backward = None

        if train_generator:
            if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
                print(f"[SeqTrain-Trainer] Training generator: generating next chunk")

            train_first_chunk = getattr(self.config, "train_first_chunk", False)
            current_seq_length = self.streaming_model.state.get("current_length")
            if train_first_chunk:
                generated_chunk, chunk_info = self.streaming_model.generate_next_chunk(requires_grad=True)
            else:
                current_seq_length = self.streaming_model.state.get("current_length")
                if current_seq_length == 0:
                    generated_chunk, chunk_info = self.streaming_model.generate_next_chunk(requires_grad=False)

                generated_chunk, chunk_info = self.streaming_model.generate_next_chunk(requires_grad=True)

            if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
                print(
                    colored(
                        f"[SeqTrain-Trainer] Generator: train_first_chunk={train_first_chunk}, current_seq_length={current_seq_length}, chunk shape: {generated_chunk.shape}",
                        "yellow",
                    )
                )

            # Compute generator loss
            generator_loss, generator_log_dict = self.streaming_model.compute_generator_loss(
                chunk=generated_chunk, chunk_info=chunk_info
            )

            # Scale loss for gradient accumulation and backward
            scaled_generator_loss = generator_loss / self.gradient_accumulation_steps

            if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
                print(f"[DEBUG] Scaled generator loss: {scaled_generator_loss.item()}")

            try:
                scaled_generator_loss.backward()
            except RuntimeError:
                raise

            generator_log_dict.update(
                {
                    "generator_loss": generator_loss,
                    "generator_grad_norm": torch.tensor(0.0, device=self.device),
                }
            )

            return generator_log_dict
        else:
            if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
                print(f"[SeqTrain-Trainer] Training critic: generating next chunk")

            train_first_chunk = getattr(self.config, "train_first_chunk", False)
            with torch.no_grad():
                current_seq_length = self.streaming_model.state.get("current_length")
                if train_first_chunk:
                    generated_chunk, chunk_info = self.streaming_model.generate_next_chunk(requires_grad=False)
                else:
                    current_seq_length = self.streaming_model.state.get("current_length")
                    if current_seq_length == 0:
                        generated_chunk, chunk_info = self.streaming_model.generate_next_chunk(requires_grad=False)

                    generated_chunk, chunk_info = self.streaming_model.generate_next_chunk(requires_grad=False)

            if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
                print(
                    colored(
                        f"[SeqTrain-Trainer] Critic: train_first_chunk={train_first_chunk}, current_seq_length={current_seq_length}, chunk shape: {generated_chunk.shape}",
                        "red",
                    )
                )

            if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
                print(f"[SeqTrain-Trainer] Critic: Generated chunk shape: {generated_chunk.shape}")
                print(f"[SeqTrain-Trainer] Critic: Generated chunk requires_grad: {generated_chunk.requires_grad}")

            if generated_chunk.requires_grad:
                generated_chunk = generated_chunk.detach()

            # Compute critic loss
            critic_loss, critic_log_dict = self.streaming_model.compute_critic_loss(
                chunk=generated_chunk, chunk_info=chunk_info
            )

            if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
                print(f"[SeqTrain-Trainer] Critic loss: {critic_loss.item()}")

            # Scale loss for gradient accumulation and backward
            scaled_critic_loss = critic_loss / self.gradient_accumulation_steps
            scaled_critic_loss.backward()

            critic_log_dict.update(
                {
                    "critic_loss": critic_loss,
                    "critic_grad_norm": torch.tensor(0.0, device=self.device),
                }
            )

            return critic_log_dict

    def train(self):
        start_step = self.step
        try:
            while True:
                # Check if we should train generator on this optimization step
                TRAIN_GENERATOR = self.step % self.config.dfake_gen_update_ratio == 0
                self.model.set_step(self.step)

                if dist.get_rank() == 0 and DEBUG:
                    print(f"[Debug] Step {self.step}: switch_mode={getattr(self.config,'switch_mode','fixed')}")

                if self.streaming_training:
                    if TRAIN_GENERATOR:
                        self.generator_optimizer.zero_grad(set_to_none=True)
                    self.critic_optimizer.zero_grad(set_to_none=True)

                    accumulated_generator_logs = []
                    accumulated_critic_logs = []

                    for accumulation_step in range(self.gradient_accumulation_steps):
                        if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
                            print(
                                f"[SeqTrain-Trainer] Whole-cycle accumulation step {accumulation_step + 1}/{self.gradient_accumulation_steps}"
                            )

                        # Train generator (if needed)
                        if TRAIN_GENERATOR:
                            if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
                                print(
                                    f"[SeqTrain-Trainer] Accumulation step {accumulation_step + 1}: Training generator"
                                )
                            extra_gen = self.fwdbwd_one_step_streaming(True)
                            accumulated_generator_logs.append(extra_gen)

                        # Train critic
                        if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
                            print(f"[SeqTrain-Trainer] Accumulation step {accumulation_step + 1}: Training critic")
                        extra_crit = self.fwdbwd_one_step_streaming(False)
                        accumulated_critic_logs.append(extra_crit)

                    # Compute grad norm and update parameters
                    if TRAIN_GENERATOR:
                        generator_grad_norm = self.model.generator.clip_grad_norm_(self.max_grad_norm_generator)
                        generator_log_dict = merge_dict_list(accumulated_generator_logs)
                        generator_log_dict["generator_grad_norm"] = generator_grad_norm

                        if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
                            print(
                                f"[SeqTrain-Trainer] Generator training completed, grad_norm={generator_grad_norm.item()}"
                            )

                        self.generator_optimizer.step()
                        if self.generator_ema is not None:
                            self.generator_ema.update(self.model.generator)
                    else:
                        generator_log_dict = {}

                    critic_grad_norm = self.model.fake_score.clip_grad_norm_(self.max_grad_norm_critic)
                    critic_log_dict = merge_dict_list(accumulated_critic_logs)
                    critic_log_dict["critic_grad_norm"] = critic_grad_norm

                    if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
                        print(f"[SeqTrain-Trainer] Critic training completed, grad_norm={critic_grad_norm.item()}")

                    self.critic_optimizer.step()

                    # Increase step count
                    self.step += 1

                    if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
                        print(f"[SeqTrain-Trainer] streaming training step completed: step={self.step}")
                        if hasattr(self, "streaming_model") and self.streaming_model is not None:
                            current_seq_length = self.streaming_model.state.get("current_length", 0)
                            print(
                                f"[SeqTrain-Trainer] Current sequence length: {current_seq_length}/{self.streaming_model.max_length}"
                            )

                else:
                    if TRAIN_GENERATOR:
                        self.generator_optimizer.zero_grad(set_to_none=True)
                    self.critic_optimizer.zero_grad(set_to_none=True)

                    # Whole-cycle gradient accumulation loop
                    accumulated_generator_logs = []
                    accumulated_critic_logs = []

                    for accumulation_step in range(self.gradient_accumulation_steps):
                        batch = next(self.dataloader)

                        # Train generator (if needed)
                        if TRAIN_GENERATOR:
                            extra_gen = self.fwdbwd_one_step(batch, True)
                            accumulated_generator_logs.append(extra_gen)

                        # Train critic
                        extra_crit = self.fwdbwd_one_step(batch, False)
                        accumulated_critic_logs.append(extra_crit)

                    # Compute grad norm and update parameters
                    if TRAIN_GENERATOR:
                        generator_grad_norm = self.model.generator.clip_grad_norm_(self.max_grad_norm_generator)
                        generator_log_dict = merge_dict_list(accumulated_generator_logs)
                        generator_log_dict["generator_grad_norm"] = generator_grad_norm

                        self.generator_optimizer.step()
                        if self.generator_ema is not None:
                            self.generator_ema.update(self.model.generator)
                    else:
                        generator_log_dict = {}

                    critic_grad_norm = self.model.fake_score.clip_grad_norm_(self.max_grad_norm_critic)
                    critic_log_dict = merge_dict_list(accumulated_critic_logs)
                    critic_log_dict["critic_grad_norm"] = critic_grad_norm

                    self.critic_optimizer.step()

                    # Increment the step since we finished gradient update
                    self.step += 1

                # Create EMA params (if not already created)
                if (
                    (self.step >= self.config.ema_start_step)
                    and (self.generator_ema is None)
                    and (self.config.ema_weight > 0)
                ):
                    if not self.is_lora_enabled:
                        self.generator_ema = EMA_FSDP(self.model.generator, decay=self.config.ema_weight)
                        if self.is_main_process:
                            print(f"EMA created at step {self.step} with weight {self.config.ema_weight}")
                    else:
                        if self.is_main_process:
                            print(f"EMA creation skipped at step {self.step} (disabled in LoRA mode)")

                # Save the model
                if (
                    (not self.config.no_save)
                    and (self.step - start_step) > 0
                    and self.step % self.config.log_iters == 0
                ):
                    torch.cuda.empty_cache()
                    self.save()
                    torch.cuda.empty_cache()

                # Logging
                if self.is_main_process:
                    wandb_loss_dict = {}
                    if TRAIN_GENERATOR and generator_log_dict:
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
                    iteration_time = 0 if self.previous_time is None else current_time - self.previous_time
                    if not self.disable_wandb:
                        wandb.log({"per iteration time": iteration_time}, step=self.step)
                    self.previous_time = current_time
                    # Log training progress
                    if TRAIN_GENERATOR and generator_log_dict:
                        print(
                            f"step {self.step}, per iteration time {iteration_time}, generator_loss {generator_log_dict['generator_loss'].mean().item()}, generator_grad_norm {generator_log_dict['generator_grad_norm'].mean().item()}, dmdtrain_gradient_norm {generator_log_dict['dmdtrain_gradient_norm'].mean().item()}, critic_loss {critic_log_dict['critic_loss'].mean().item()}, critic_grad_norm {critic_log_dict['critic_grad_norm'].mean().item()}"
                        )
                    else:
                        print(
                            f"step {self.step}, per iteration time {iteration_time}, critic_loss {critic_log_dict['critic_loss'].mean().item()}, critic_grad_norm {critic_log_dict['critic_grad_norm'].mean().item()}"
                        )

                # ---------------------------------------- Visualization ---------------------------------------------------
                if self.vis_interval > 0 and ((self.step % self.vis_interval == 0) or (self.step - start_step) == 1):
                    try:
                        self._visualize()
                    except Exception as e:
                        print(f"[Warning] Visualization failed at step {self.step}: {e}")

                if self.step > self.config.max_iters:
                    break

        except Exception as e:
            if self.is_main_process:
                print(f"[ERROR] Training crashed at step {self.step} with exception: {e}")
                print(f"[ERROR] Exception traceback:", flush=True)
                import traceback

                traceback.print_exc()

    def _get_switch_frame_index(self, max_length=None):
        if getattr(self.config, "switch_mode", "fixed") == "random":
            block = self.config.num_frame_per_block
            min_idx = self.config.min_switch_frame_index
            max_idx = self.config.max_switch_frame_index
            if min_idx == max_idx:
                switch_idx = min_idx
            else:
                choices = list(range(min_idx, max_idx, block))
                if max_length is not None:
                    choices = [choice for choice in choices if choice < max_length]

                if len(choices) == 0:
                    if max_length is not None:
                        raise ValueError(f"No valid switch choices available (all choices >= max_length {max_length})")
                    else:
                        switch_idx = block
                else:
                    if dist.is_initialized():
                        if dist.get_rank() == 0:
                            switch_idx = random.choice(choices)
                        else:
                            switch_idx = 0
                        switch_idx_tensor = torch.tensor(switch_idx, device=self.device)
                        dist.broadcast(switch_idx_tensor, src=0)
                        switch_idx = switch_idx_tensor.item()
                    else:
                        switch_idx = random.choice(choices)
        elif getattr(self.config, "switch_mode", "fixed") == "fixed":
            switch_idx = getattr(self.config, "fixed_switch_index", 21)
            if max_length is not None:
                assert max_length > switch_idx, f"max_length {max_length} is not greater than switch_idx {switch_idx}"
        elif getattr(self.config, "switch_mode", "fixed") == "random_choice":
            switch_choices = getattr(self.config, "switch_choices", [])
            if len(switch_choices) == 0:
                raise ValueError("switch_choices is empty")
            else:
                if max_length is not None:
                    switch_choices = [choice for choice in switch_choices if choice < max_length]
                    if len(switch_choices) == 0:
                        raise ValueError(f"No valid switch choices available (all choices >= max_length {max_length})")

                if dist.is_initialized():
                    if dist.get_rank() == 0:
                        switch_idx = random.choice(switch_choices)
                    else:
                        switch_idx = 0
                    switch_idx_tensor = torch.tensor(switch_idx, device=self.device)
                    dist.broadcast(switch_idx_tensor, src=0)
                    switch_idx = switch_idx_tensor.item()
                else:
                    switch_idx = random.choice(switch_choices)
        else:
            raise ValueError(f"Invalid switch_mode: {getattr(self.config, 'switch_mode', 'fixed')}")
        return switch_idx
