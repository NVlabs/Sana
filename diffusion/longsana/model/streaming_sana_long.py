# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0).
# You may not use this file except in compliance with the License.
# To view a copy of this license, visit https://creativecommons.org/licenses/by-nc-sa/4.0/deed.en
#
# No warranties are given. The work is provided "AS IS", without warranty of any kind, express or implied.
#
# SPDX-License-Identifier: CC-BY-NC-SA-4.0
import time
from typing import Any, Dict, Optional, Tuple

import torch
import torch.distributed as dist
from einops import rearrange
from termcolor import colored

from diffusion.longsana.pipeline.sana_switch_training_pipeline import SanaSwitchTrainingPipeline
from diffusion.longsana.pipeline.sana_training_pipeline import SanaTrainingPipeline
from diffusion.longsana.utils.debug_option import DEBUG, DEBUG_GRADIENT, LOG_GPU_MEMORY


class StreamingSANATrainingModel:
    """
    A model wrapper specifically for streaming/serialized training.

    This class wraps existing models (DMD, DMDSwitch, etc.) and provides a unified
    interface for streaming training. Main features:
    1. Manage streaming generation state
    2. Reuse KV cache and cross-attention cache
    3. Support prompt switching for DMD Switch
    4. Provide chunk-wise loss computation
    5. Support overlapping frames to ensure continuity
    """

    def __init__(self, base_model, config):
        """
        Initialize the streaming training model.

        Args:
            base_model: underlying model (DMD, DMDSwitch, etc.)
            config: configuration object
        """
        self.base_model = base_model
        self.config = config
        self.device = base_model.device
        self.dtype = base_model.dtype
        self.image_or_video_shape = getattr(config, "image_or_video_shape", None)

        # Streaming training configuration
        self.chunk_size = getattr(
            config, "streaming_chunk_size", 21
        )  # Fixed chunk size used for loss computation, for Wan Real Model
        self.max_length = getattr(config, "streaming_max_length", 141)  # 141 for 35s, 261 for 60s
        self.possible_max_length = getattr(config, "streaming_possible_max_length", None)
        self.min_new_frame = getattr(config, "streaming_min_new_frame", 20)
        self.independent_first_frame = self.chunk_size % 20 == 1

        # Get required components from the underlying model
        self.generator = base_model.generator
        self.fake_score = base_model.fake_score
        self.scheduler = base_model.scheduler
        self.denoising_loss_func = base_model.denoising_loss_func

        # Fetch model configuration
        self.num_frame_per_block = base_model.num_frame_per_block
        self.frame_seq_length = getattr(base_model.inference_pipeline, "frame_seq_length", 1560)

        # Initialize inference pipeline
        self.inference_pipeline = base_model.inference_pipeline
        if self.inference_pipeline is None:
            base_model._initialize_inference_pipeline()
            self.inference_pipeline = base_model.inference_pipeline

        # Streaming state
        self.reset_state()

        if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
            print(f"[StreamingTrain-Model] streamingTrainingModel initialized:")
            print(f"[StreamingTrain-Model] chunk_size={self.chunk_size}, max_length={self.max_length}")
            print(f"[StreamingTrain-Model] min_new_frame={self.min_new_frame}")
            print(f"[StreamingTrain-Model] base_model type: {type(self.base_model).__name__}")

    def _process_first_frame_encoding(self, frames: torch.Tensor) -> torch.Tensor:
        """
        Apply special encoding to the first frame, following the logic in _run_generator.

        Args:
            frames: frame sequence [batch_size, num_frames, C, H, W]

        Returns:
            processed_frames: processed frame sequence where the first frame is re-encoded as an image latent
        """
        total_frames = frames.shape[1]

        if total_frames <= 1:
            # Only one or zero frames, return as is
            return frames

        # Determine the range to process: last 21 frames
        process_frames = min(self.chunk_size, total_frames)

        if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
            print(
                f"[StreamingTrain-Model] Processing first frame encoding for loss: total_frames={total_frames}, processing last {process_frames} frames"
            )

        with torch.no_grad():
            # Decode the frames to be processed into pixels
            frames_to_decode = frames[:, : -(process_frames - 1), ...]  # B,F-20,C,H,W
            pixels = self.base_model.vae.decode_to_pixel(rearrange(frames_to_decode, "b f c h w -> b c f h w"))

            # Take the last frame's pixel representation
            last_frame_pixel = pixels[:, :, -1:, ...].to(self.dtype)  # b,c,1,h,w

            # Re-encode as image latent
            image_latent = self.base_model.vae.encode_to_latent(last_frame_pixel).to(self.dtype)
            image_latent = rearrange(image_latent, "b c f h w -> b f c h w")

        remaining_frames = frames[:, -(process_frames - 1) :, ...]
        processed_frames = torch.cat([image_latent, remaining_frames], dim=1)

        if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
            print(f"[StreamingTrain-Model] Processed first frame encoding: {frames.shape} -> {processed_frames.shape}")

        return processed_frames

    def reset_state(self):
        """Reset streaming training state"""
        if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
            print(f"[StreamingTrain-Model] Resetting streaming training state")

        self.state = {
            "current_length": 0,
            "conditional_info": None,
            "has_switched": False,  # Track whether prompt has been switched
            "previous_frames": None,  # Store last generated frames for overlap (up to 21)
            "temp_max_length": None,  # Temporary max length for the current sequence
        }

        self.inference_pipeline.clear_kv_cache()

    def _should_switch_prompt(self, chunk_start_frame: int, chunk_size: int) -> bool:
        # If already switched, do not switch again
        if self.state.get("has_switched", False):
            if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
                print(f"[StreamingTrain-Model] Already switched, not switching again")
            return False

        switch_info = self.state.get("conditional_info", {}).get("switch_info", None)
        if not switch_info:
            raise ValueError("No switch_info found")

        switch_frame_index = switch_info.get("switch_frame_index", None)
        if switch_frame_index is None:
            raise ValueError("switch_frame_index is None")

        # if the switch point is within the current chunk ([start, end)), switch in the current chunk
        chunk_end_frame = chunk_start_frame + chunk_size
        should_switch = chunk_start_frame <= switch_frame_index < chunk_end_frame

        if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
            print(
                f"[StreamingTrain-Model] Switch check: switch_frame={switch_frame_index}, chunk_start={chunk_start_frame}, should_switch={should_switch}"
            )

        return should_switch

    def _get_current_conditional_dict(self, chunk_start_frame: int) -> dict:
        """Get the conditional_dict to use for the current chunk"""
        cond_info = self.state["conditional_info"]

        # Check whether it has switched already or should switch now
        switch_info = cond_info.get("switch_info", {})
        if switch_info:
            switch_frame_index = switch_info.get("switch_frame_index")
            if switch_frame_index is not None:
                if self.state.get("has_switched", False) or chunk_start_frame >= switch_frame_index:
                    # If already switched, or current frame has reached the switch point, use the switched prompt
                    if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
                        print(
                            f"[StreamingTrain-Model] Using switch conditional_dict for chunk starting at frame {chunk_start_frame}"
                        )
                    return switch_info.get("switch_conditional_dict", cond_info["conditional_dict"])

        # Otherwise use the original prompt
        if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
            print(
                f"[StreamingTrain-Model] Using original conditional_dict for chunk starting at frame {chunk_start_frame}"
            )
        return cond_info["conditional_dict"]

    def _generate_chunk(
        self,
        noise_chunk: torch.Tensor,
        chunk_start_frame: int,
        requires_grad: bool = True,
    ) -> Tuple[torch.Tensor, Optional[int], Optional[int]]:
        """
        Generate a single chunk.

        Args:
            noise_chunk: noise input [batch_size, chunk_frames, C, H, W]
            chunk_start_frame: start frame index of the chunk in the full sequence
            requires_grad: whether gradients are required

        Returns:
            generated_chunk: generated chunk [batch_size, chunk_frames, C, H, W]
            denoised_timestep_from: starting timestep for denoising
            denoised_timestep_to: ending timestep for denoising
        """

        # Get the conditional_dict to use now
        current_conditional_dict = self._get_current_conditional_dict(chunk_start_frame)
        # For switch pipeline, always pass the original prompt as cond1, and provide switch_prompt_embeds via kwargs
        if isinstance(self.inference_pipeline, SanaSwitchTrainingPipeline) and "switch_info" in self.state.get(
            "conditional_info", {}
        ):
            base_conditional_dict = self.state["conditional_info"]["conditional_dict"]
            prompt_embeds = base_conditional_dict["prompt_embeds"]
            mask = base_conditional_dict.get("mask", current_conditional_dict.get("mask", None))
        else:
            prompt_embeds = current_conditional_dict["prompt_embeds"]
            mask = current_conditional_dict.get("mask", None)
        noise_chunk = rearrange(noise_chunk, "b f c h w -> b c f h w")
        # Prepare generation parameters
        kwargs = {
            "noise": noise_chunk,
            "prompt_embeds": prompt_embeds,
            "mask": mask,
            "current_start_frame": chunk_start_frame,
            "requires_grad": requires_grad,
            "return_sim_step": False,
        }

        # Add switching logic for SanaSwitchTrainingPipeline
        if isinstance(self.inference_pipeline, SanaSwitchTrainingPipeline):
            switch_info = self.state.get("conditional_info", {}).get("switch_info", {})
            if switch_info:
                # pass the absolute switch frame index, same as the logic in SanaSwitchTrainingPipeline
                kwargs["switch_frame_index"] = int(switch_info["switch_frame_index"])  # absolute
                # pass the second segment's prompt_embeds, named switch_prompt_embeds
                if (
                    "switch_conditional_dict" in switch_info
                    and "prompt_embeds" in switch_info["switch_conditional_dict"]
                ):
                    kwargs["switch_prompt_embeds"] = switch_info["switch_conditional_dict"]["prompt_embeds"]
                if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
                    has_switch = "switch_prompt_embeds" in kwargs
                    print(
                        f"[StreamingTrain-Model] Switch params set: switch_frame_index={kwargs['switch_frame_index']}, has switch_prompt_embeds={has_switch}"
                    )

        output, denoised_timestep_from, denoised_timestep_to = self.inference_pipeline.generate_chunk_with_cache(
            **kwargs
        )
        output = rearrange(output, "b c f h w -> b f c h w")

        return output, denoised_timestep_from, denoised_timestep_to

    def setup_sequence(
        self,
        conditional_dict: Dict,
        unconditional_dict: Dict,
        conditional_dict_real: Dict,
        unconditional_dict_real: Dict,
        initial_latent: Optional[torch.Tensor] = None,
        switch_conditional_dict: Optional[Dict] = None,
        switch_frame_index: Optional[int] = None,
        temp_max_length: Optional[int] = None,
    ):
        """Set up a new sequence"""

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        batch_size = self.image_or_video_shape[0]
        if self.inference_pipeline.kv_cache is None:
            self.inference_pipeline._initialize_kv_cache(batch_size=batch_size, dtype=self.dtype, device=self.device)
            if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
                print(f"[StreamingTrain-Model] init kv_cache: chunks={len(self.inference_pipeline.kv_cache)}")

        # Reset state
        self.reset_state()
        self.state["temp_max_length"] = temp_max_length

        # Prepare initial sequence
        if initial_latent is not None:
            self.state["current_length"] = initial_latent.shape[1]
            if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
                print(f"[StreamingTrain-Model] Starting with initial_latent, length={self.state['current_length']}")
        else:
            self.state["current_length"] = 0
            if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
                print(f"[StreamingTrain-Model] Starting with empty sequence")

        # Save conditional information
        self.state["conditional_info"] = {
            "conditional_dict": conditional_dict,
            "unconditional_dict": unconditional_dict,
            "conditional_dict_real": conditional_dict_real,
            "unconditional_dict_real": unconditional_dict_real,
        }

        # DMDSwitch related information
        if switch_conditional_dict is not None and switch_frame_index is not None:
            self.state["conditional_info"]["switch_info"] = {
                "switch_conditional_dict": switch_conditional_dict,
                "switch_frame_index": switch_frame_index,
            }
            if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
                print(f"[StreamingTrain-Model] DMDSwitch info saved: switch_frame_index={switch_frame_index}")

        if initial_latent is not None and DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
            print(f"[StreamingTrain-Model] initial_latent provided; kv_cache will be updated during generation")

        else:
            if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
                print(f"[StreamingTrain-Model] No initial latent")

    def can_generate_more(self) -> bool:
        """Check whether more chunks can be generated"""
        current_length = self.state["current_length"]
        temp_max_length = self.state.get("temp_max_length")
        can_generate = current_length < temp_max_length and (current_length + self.min_new_frame) <= temp_max_length

        if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
            print(
                f"[StreamingTrain-Model] can_generate_more: current_length={current_length}, temp_max_length={temp_max_length}, global_max_length={self.max_length}, can_generate={can_generate}"
            )

        return can_generate

    def check_current_kv_cache(self):
        """Check whether the current kv_cache is valid"""
        if self.inference_pipeline.kv_cache is None:
            return 0

        previous_chunk_index = 0
        for _kv_cache in self.inference_pipeline.kv_cache:
            if _kv_cache[0][-1] is not None:
                previous_chunk_index += 1

        return previous_chunk_index

    def generate_next_chunk(self, requires_grad: bool = True) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Generate the next chunk, supporting overlap to ensure temporal continuity.

        Args:
            requires_grad: whether gradients are required

        Returns:
            generated_chunk: the full generated chunk (including overlap frames)
            info: generation info (including timestep, gradient_mask, etc.)
        """
        if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
            print(f"[StreamingTrain-Model] generate_next_chunk called: requires_grad={requires_grad}")

        if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
            gen_training_mode = self.generator.training
            gen_params_requiring_grad = sum(1 for p in self.generator.parameters() if p.requires_grad)
            gen_params_total = sum(1 for p in self.generator.parameters())
            print(f"[DEBUG-SeqModel] Generator training mode: {gen_training_mode}")
            print(f"[DEBUG-SeqModel] Generator params requiring grad: {gen_params_requiring_grad}/{gen_params_total}")

        if not self.can_generate_more():
            raise ValueError("Cannot generate more chunks")

        current_length = self.state["current_length"]
        batch_size = self.image_or_video_shape[0]

        if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
            print(f"[StreamingTrain-Model] Generating chunk: current_length={current_length}")

        # Check if previous_frames can be used for overlap and auto-compute overlap frame count
        previous_frames = self.state.get("previous_frames")
        if previous_frames is not None:
            # Randomly select number of new frames (min=min_new_frame, max=chunk_size, step=3)
            max_new_frames = min(self.state["temp_max_length"] - current_length + 1, self.chunk_size)
            num_frame_per_block = getattr(self.base_model, "num_frame_per_block", 10)
            possible_new_frames = (
                list(range(self.min_new_frame, max_new_frames, num_frame_per_block))
                if max_new_frames > self.min_new_frame
                else [self.min_new_frame]
            )
            # Ensure all processes choose the same random value
            if dist.is_initialized():
                if dist.get_rank() == 0:
                    import random

                    selected_idx = random.randint(0, len(possible_new_frames) - 1)
                else:
                    selected_idx = 0
                selected_idx_tensor = torch.tensor(selected_idx, device=self.device, dtype=torch.int32)
                dist.broadcast(selected_idx_tensor, src=0)
                selected_idx = selected_idx_tensor.item()
            else:
                import random

                selected_idx = random.randint(0, len(possible_new_frames) - 1)

            new_frames_to_generate = possible_new_frames[selected_idx]

            # Auto-compute required overlap frames to ensure the final chunk has 21 frames
            overlap_frames = self.chunk_size - new_frames_to_generate
            if overlap_frames > 0 and overlap_frames <= previous_frames.shape[1]:
                overlap_frames_to_use = overlap_frames
            else:
                # If overlap can't be used, generate a full chunk_size without overlap
                overlap_frames_to_use = 0
                new_frames_to_generate = self.chunk_size

            if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
                print(
                    f"[StreamingTrain-Model] With auto overlap: generating {new_frames_to_generate} new frames, reusing {overlap_frames_to_use} overlap frames"
                )
        else:
            overlap_frames_to_use = 0
            new_frames_to_generate = self.chunk_size
            if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
                print(f"[StreamingTrain-Model] First chunk: generating {new_frames_to_generate} frames (no overlap)")
        if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
            print(f"[StreamingTrain-Model] Random frame selection: selected={new_frames_to_generate}")
            print(f"[StreamingTrain-Model] Auto overlap calculation: overlap_frames={overlap_frames_to_use}")

        # Sample noise for new frames
        noise_chunk = torch.randn(
            [batch_size, new_frames_to_generate, *self.image_or_video_shape[2:]], device=self.device, dtype=self.dtype
        )

        # Generate new frames - note chunk_start_frame should consider overlap
        generated_new_frames, denoised_timestep_from, denoised_timestep_to = self._generate_chunk(
            noise_chunk=noise_chunk,
            chunk_start_frame=current_length,
            requires_grad=requires_grad,
        )

        # Build the full chunk for loss computation
        if previous_frames is not None and overlap_frames_to_use > 0:
            # Concatenate specified overlap frames and newly generated frames
            full_chunk = torch.cat([previous_frames, generated_new_frames], dim=1)
        else:
            full_chunk = generated_new_frames  # B,F,C,H,W

        # Update state - save the last 21 frames as previous_frames for the next chunk
        # The frames saved here should be those before _process_first_frame_encoding
        frames_to_save = full_chunk.detach().clone()[:, -self.chunk_size :, ...]
        if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
            print(f"[StreamingTrain-Model] Saved last {frames_to_save.shape[1]} frames as previous_frames")

        # Process first-frame encoding (if there is overlap)
        if previous_frames is not None and self.independent_first_frame:
            full_chunk = self._process_first_frame_encoding(full_chunk)

        if previous_frames is not None:
            # Create gradient_mask: only newly generated frames require gradients
            gradient_mask = torch.zeros_like(full_chunk, dtype=torch.bool)
            # Overlap frames do not compute gradients; new frames do
            # TODO: only one overlap, either 1 or 11
            gradient_mask[:, overlap_frames_to_use : overlap_frames_to_use + new_frames_to_generate, ...] = True
            if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
                print(f"[StreamingTrain-Model] Built chunk with auto overlap: shape={full_chunk.shape}")
                print(
                    f"[StreamingTrain-Model] Gradient mask: {new_frames_to_generate} frames will have gradients out of {full_chunk.shape[1]}"
                )
        else:
            # For the first chunk, all frames are newly generated
            gradient_mask = torch.ones_like(full_chunk, dtype=torch.bool)

        self.state["current_length"] += new_frames_to_generate  # Increase only by newly generated frames

        if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
            print(f"[StreamingTrain-Model] Updated state: current_length={self.state['current_length']}")
            if self.state["previous_frames"] is not None:
                print(
                    f"[StreamingTrain-Model] Saved {self.state['previous_frames'].shape[1]} frames as previous_frames for next chunk"
                )

        self.state["previous_frames"] = frames_to_save

        # Return info
        info = {
            "denoised_timestep_from": denoised_timestep_from,
            "denoised_timestep_to": denoised_timestep_to,
            "chunk_start_frame": current_length,  # Start frame position in the full sequence
            "chunk_frames": full_chunk.shape[1],  # Chunk size used for loss (fixed 21 frames)
            "new_frames_generated": new_frames_to_generate,
            "current_length": self.state["current_length"],
            "gradient_mask": gradient_mask,  # Mask frames that do not require gradients for loss computation
            "overlap_frames_used": overlap_frames_to_use,
        }
        if not dist.is_initialized() or dist.get_rank() == 0:
            print(
                f"[StreamingTrain-Model] current_training_chunk: ({self.state['current_length'] - new_frames_to_generate} -> {self.state['current_length']})/{self.state['temp_max_length']}"
            )
        return full_chunk, info

    def compute_generator_loss(
        self, chunk: torch.Tensor, chunk_info: Dict[str, Any]
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Compute the generator loss.

        Args:
            chunk: generated chunk
            chunk_info: chunk metadata

        Returns:
            loss: loss value
            log_dict: log dictionary
        """
        _t_loss_start = time.time()

        # Fetch conditional_dict for loss computation
        chunk_start_frame = chunk_info["chunk_start_frame"]
        conditional_dict = self._get_current_conditional_dict(chunk_start_frame)
        unconditional_dict = self.state["conditional_info"]["unconditional_dict"]
        conditional_dict_real = self.state["conditional_info"]["conditional_dict_real"]
        unconditional_dict_real = self.state["conditional_info"]["unconditional_dict_real"]

        # Fetch gradient_mask to compute loss only on newly generated frames
        gradient_mask = chunk_info.get("gradient_mask", None)

        if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
            print(
                f"[StreamingTrain-Model] Using conditional_dict and unconditional_dict for loss calculation at frame {chunk_start_frame}"
            )

        self.base_model.decode_and_save_clip(
            chunk, f"generator_gen_f{self.state['current_length']}_t{int(chunk_info['denoised_timestep_to'])}"
        )

        # Compute DMD loss
        dmd_loss, dmd_log_dict = self.base_model.compute_distribution_matching_loss(
            image_or_video=chunk,
            conditional_dict=conditional_dict,
            unconditional_dict=unconditional_dict,
            conditional_dict_real=conditional_dict_real,
            unconditional_dict_real=unconditional_dict_real,
            gradient_mask=gradient_mask,  # Pass gradient_mask
            denoised_timestep_from=chunk_info["denoised_timestep_from"],
            denoised_timestep_to=chunk_info["denoised_timestep_to"],
            current_length=self.state["current_length"],
        )

        # Update log dict
        dmd_log_dict.update(
            {
                "loss_time": time.time() - _t_loss_start,
                "new_frames_supervised": chunk_info.get("new_frames_generated", chunk.shape[1]),
            }
        )

        return dmd_loss, dmd_log_dict

    def _clear_cache_gradients(self):
        """
        Clear possible gradient references in KV cache and cross-attention cache.
        This is important for preventing memory leaks, especially before critic training.
        """
        if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
            print(f"[StreamingTrain-Model] Clearing cache gradients")

        self.inference_pipeline._clear_cache_gradients()

        if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
            print(f"[StreamingTrain-Model] Cache gradients cleared")

    def compute_critic_loss(
        self, chunk: torch.Tensor, chunk_info: Dict[str, Any]
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Compute critic loss.

        Args:
            chunk: generated chunk
            chunk_info: chunk metadata

        Returns:
            loss: loss value
            log_dict: log dictionary
        """
        _t_loss_start = time.time()

        if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
            print(f"[StreamingTrain-Model] compute_critic_loss: chunk_shape={chunk.shape}")
            for k, v in chunk_info.items():
                if k == "gradient_mask":
                    print(f"[StreamingTrain-Model] chunk_info {k}: {v[0, :, 0, 0, 0]}")
                else:
                    print(f"[StreamingTrain-Model] chunk_info {k}: {v}")
            print(f"[StreamingTrain-Model] chunk requires_grad: {chunk.requires_grad}")

        # Critical fix: ensure chunk has no gradient connections
        if chunk.requires_grad:
            chunk = chunk.detach()

        # Critical fix: clear gradient references inside caches
        self._clear_cache_gradients()

        # Force clear CUDA cache to ensure previous graphs are released
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Fetch conditional_dict for loss computation
        chunk_start_frame = chunk_info["chunk_start_frame"]
        conditional_dict = self._get_current_conditional_dict(chunk_start_frame)

        # Fetch gradient_mask to compute loss only on newly generated frames
        gradient_mask = chunk_info.get("gradient_mask", None)

        batch_size, num_frame = chunk.shape[:2]

        if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
            print(f"[StreamingTrain-Model] Preparing critic loss: batch_size={batch_size}, num_frame={num_frame}")

        # Use the same timestep range logic as non-streaming training
        denoised_timestep_from = chunk_info.get("denoised_timestep_from", None)
        denoised_timestep_to = chunk_info.get("denoised_timestep_to", None)

        self.base_model.decode_and_save_clip(
            chunk, f"critic_gen_f{self.state['current_length']}_t{int(denoised_timestep_to)}"
        )

        min_timestep = (
            denoised_timestep_to
            if (getattr(self.base_model, "ts_schedule", False) and denoised_timestep_to is not None)
            else getattr(self.base_model, "min_score_timestep")
        )
        max_timestep = (
            denoised_timestep_from
            if (getattr(self.base_model, "ts_schedule_max", False) and denoised_timestep_from is not None)
            else getattr(self.base_model, "num_train_timestep")
        )

        # Randomly select time steps
        critic_timestep = self.base_model._get_timestep(
            min_timestep=min_timestep,
            max_timestep=max_timestep,
            batch_size=batch_size,
            num_frame=num_frame,
            num_frame_per_block=getattr(self.base_model, "num_frame_per_block", 10),
            uniform_timestep=True,  # Set to True to match non-streaming training
        ).to(self.device)

        # Apply the same timestep shift logic as non-streaming training
        if getattr(self.base_model, "timestep_shift") > 1:
            timestep_shift = self.base_model.timestep_shift
            critic_timestep = (
                timestep_shift * (critic_timestep / 1000) / (1 + (timestep_shift - 1) * (critic_timestep / 1000)) * 1000
            )

        critic_timestep = critic_timestep.clamp(self.base_model.min_step, self.base_model.max_step)

        # Sample noise
        critic_noise = torch.randn_like(chunk)

        # Add noise to chunk
        noisy_chunk = self.scheduler.add_noise(
            chunk.flatten(0, 1), critic_noise.flatten(0, 1), critic_timestep.flatten(0, 1)
        ).unflatten(0, (batch_size, num_frame))

        if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
            print(
                f"[StreamingTrain-Model] Added noise, timestep range: [{critic_timestep.min().item()}, {critic_timestep.max().item()}]"
            )

        # Compute fake prediction
        if self.base_model.fake_name == "SANA":
            assert "mask" in conditional_dict
            condition = conditional_dict["prompt_embeds"].clone()
            mask = conditional_dict.get("mask", None)
            # BFCHW -> BCFHW
            noisy_bcfhw = rearrange(noisy_chunk, "b f c h w -> b c f h w")
            _, pred_fake_image_bcfhw, _ = self.fake_score(
                noisy_image_or_video=noisy_bcfhw, condition=condition, timestep=critic_timestep, mask=mask
            )
            # BCFHW -> BFCHW
            pred_fake_image = rearrange(pred_fake_image_bcfhw, "b c f h w -> b f c h w")
        else:
            _, pred_fake_image = self.fake_score(
                noisy_image_or_video=noisy_chunk, conditional_dict=conditional_dict, timestep=critic_timestep
            )

        self.base_model.decode_and_save_clip(
            pred_fake_image,
            f"critic_fake_f{self.state['current_length']}_t{int(critic_timestep.reshape(-1)[0].item())}",
        )

        # Compute denoising loss
        denoising_loss_type = getattr(self.base_model.args, "denoising_loss_type", "mse")
        if denoising_loss_type == "flow":
            from diffusion.longsana.utils.model_wrapper import SanaModelWrapper

            flow_pred = SanaModelWrapper._convert_x0_to_flow_pred(
                scheduler=self.scheduler,
                x0_pred=pred_fake_image.flatten(0, 1),
                xt=noisy_chunk.flatten(0, 1),
                timestep=critic_timestep.flatten(0, 1),
            )
            pred_fake_noise = None
            if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
                print(f"[StreamingTrain-Model] Using flow-based denoising loss")
        else:
            flow_pred = None
            pred_fake_noise = self.scheduler.convert_x0_to_noise(
                x0=pred_fake_image.flatten(0, 1), xt=noisy_chunk.flatten(0, 1), timestep=critic_timestep.flatten(0, 1)
            ).unflatten(0, (batch_size, num_frame))
            if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
                print(f"[StreamingTrain-Model] Using MSE-based denoising loss")

        gradient_mask_flat = gradient_mask.flatten(0, 1) if gradient_mask is not None else None
        denoising_loss = self.denoising_loss_func(
            x=chunk.flatten(0, 1),
            x_pred=pred_fake_image.flatten(0, 1),
            noise=critic_noise.flatten(0, 1),
            noise_pred=pred_fake_noise,
            alphas_cumprod=self.scheduler.alphas_cumprod,
            timestep=critic_timestep.flatten(0, 1),
            flow_pred=flow_pred,
            gradient_mask=gradient_mask_flat,  # Pass gradient_mask
        )

        if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
            print(f"[StreamingTrain-Model] Critic loss computed: {denoising_loss.item()}")

        # Critical: clean up intermediate variables after critic loss
        del conditional_dict, critic_noise, noisy_chunk, pred_fake_image
        if "flow_pred" in locals():
            del flow_pred
        if "pred_fake_noise" in locals():
            del pred_fake_noise

        # Build log dict
        critic_log_dict = {
            "loss_time": time.time() - _t_loss_start,
            "new_frames_supervised": chunk_info.get("new_frames_generated", num_frame),
        }

        return denoising_loss, critic_log_dict

    def get_sequence_length(self) -> int:
        """Get current sequence length"""
        return self.state.get("current_length", 0)
