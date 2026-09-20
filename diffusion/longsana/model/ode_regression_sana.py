import os
from typing import Tuple

import imageio
import pyrallis
import torch
import torch.distributed as dist
import torch.nn.functional as F
from einops import rearrange
from termcolor import colored

from diffusion.longsana.pipeline.sana_training_pipeline import SanaTrainingPipeline
from diffusion.longsana.sana_video_pipeline import LongSANAVideoInference
from diffusion.longsana.utils.model_wrapper import SanaModelWrapper, SanaTextEncoder, SanaVAEWrapper
from diffusion.model.builder import build_model, get_tokenizer_and_text_encoder, get_vae, vae_decode, vae_encode
from diffusion.model.utils import get_weight_dtype
from diffusion.utils.config import model_video_init_config
from tools.download import find_model


class SanaModelChunkWrapper(SanaModelWrapper):
    def forward(
        self,
        noisy_image_or_video: torch.Tensor,
        condition: torch.Tensor,
        timestep: torch.Tensor,
        chunk_index: list,
        mask: torch.Tensor = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        forward pass, compatible with WanDiffusionWrapper interface
        """
        if condition.dim() == 3:
            condition = condition.unsqueeze(1)
        elif condition.dim() == 2:
            condition = condition.unsqueeze(0).unsqueeze(0)

        model = self.model
        input_t = timestep

        model_out = model(
            noisy_image_or_video,
            input_t,
            condition,
            chunk_index=chunk_index,
            mask=mask,
            **kwargs,
        )
        if isinstance(model_out, tuple) and len(model_out) == 2:
            model_out, kv_cache_ret = model_out
        else:
            pass

        try:
            from diffusers.models.modeling_outputs import Transformer2DModelOutput

            if isinstance(model_out, Transformer2DModelOutput):
                model_out = model_out[0]
        except Exception:
            pass

        if isinstance(model_out, Transformer2DModelOutput):
            model_out = model_out[0]

        flow_pred_bcfhw = model_out
        flow_pred = rearrange(flow_pred_bcfhw, "b c f h w -> b f c h w")  # (B, F, C, H, W)
        noisy_image_or_video = rearrange(noisy_image_or_video, "b c f h w -> b f c h w")  # (B, F, C, H, W)
        # input_t, B,1,F
        input_t = input_t.reshape(-1)
        pred_x0 = self._convert_flow_pred_to_x0(
            flow_pred=flow_pred.flatten(0, 1), xt=noisy_image_or_video.flatten(0, 1), timestep=input_t
        ).unflatten(
            0, flow_pred.shape[:2]
        )  # (B, F, C, H, W)
        pred_x0_bcfhw = rearrange(pred_x0, "b f c h w -> b c f h w")  # (B, C, F, H, W)

        return flow_pred_bcfhw, pred_x0_bcfhw


class ODERegressionSana(torch.nn.Module):
    def __init__(self, args, device):
        """
        Initialize the ODERegression module.
        This class is self-contained and compute generator losses
        in the forward pass given precomputed ode solution pairs.
        This class supports the ode regression loss for both causal and bidirectional models.
        See Sec 4.3 of CausVid https://arxiv.org/abs/2412.07772 for details
        """
        super().__init__()
        self.dtype = torch.bfloat16 if args.mixed_precision else torch.float32
        self.device = device

        self._initialize_models(args, device)

        self.device = device
        self.args = args
        self.dtype = torch.bfloat16 if args.mixed_precision else torch.float32
        if hasattr(args, "denoising_step_list"):
            self.denoising_step_list = torch.tensor(args.denoising_step_list, dtype=torch.long)
            if args.warp_denoising_step:
                timesteps = torch.cat((self.scheduler.timesteps.cpu(), torch.tensor([0], dtype=torch.float32)))
                self.denoising_step_list = timesteps[1000 - self.denoising_step_list]

        # Step 1: Initialize all models
        if getattr(args, "generator_ckpt", False):
            print(f"Loading pretrained generator from {args.generator_ckpt}")
            state_dict = find_model(args.generator_ckpt)
            state_dict = state_dict.get("state_dict", state_dict)
            try:
                self.generator.model.load_state_dict(state_dict, strict=True)
            except Exception:
                self.generator.load_state_dict(state_dict, strict=True)

        self.num_frame_per_block = getattr(args, "num_frame_per_block", 1)

        if self.num_frame_per_block > 1:
            self.generator.model.num_frame_per_block = self.num_frame_per_block

        self.independent_first_frame = getattr(args, "independent_first_frame", False)
        if self.independent_first_frame:
            self.generator.model.independent_first_frame = True
        if args.gradient_checkpointing:
            self.generator.enable_gradient_checkpointing()

        # Step 2: Initialize all hyperparameters
        self.timestep_shift = getattr(args, "timestep_shift", 1.0)
        self.denoising_step_list = self.denoising_step_list.to(self.device)
        self.inference_pipeline: SanaTrainingPipeline = None
        self.build_inference_model(args, device)

    def build_inference_model(self, args, device):

        self.is_main_process = (dist.get_rank() == 0) if dist.is_initialized() else True
        # sana config(generator)
        sana_config_path = getattr(
            args, "sana_inference_config", "configs/sana_video_config/longsana/480ms/self_forcing.yaml"
        )
        print(f"init sana config (self forcing): {sana_config_path}")
        sana_cfg = pyrallis.load(LongSANAVideoInference, open(sana_config_path))
        work_dir = "output/sana_logs"
        try:
            os.makedirs(work_dir, exist_ok=True)
        except Exception:
            pass
        latent_size = sana_cfg.model.image_size // sana_cfg.vae.vae_downsample_rate
        model_kwargs = model_video_init_config(sana_cfg, latent_size=latent_size)
        if self.is_main_process:
            print(colored(f"init sana inference model", "green"))
        sana_model_gen = build_model(
            sana_cfg.model.model,
            use_grad_checkpoint=True,
            use_fp32_attention=sana_cfg.model.get("fp32_attention", False) and sana_cfg.model.mixed_precision != "bf16",
            **model_kwargs,
        )
        if self.is_main_process:
            print(
                colored(
                    f"{sana_model_gen.__class__.__name__}:{sana_cfg.model.model},"
                    f"Model Parameters: {sum(p.numel() for p in sana_model_gen.parameters()):,}"
                ),
                "green",
            )

        if sana_cfg.flow_shift is not None:
            flow_shift = sana_cfg.flow_shift
        else:
            flow_shift = (
                sana_cfg.scheduler.inference_flow_shift
                if sana_cfg.scheduler.inference_flow_shift is not None
                else sana_cfg.scheduler.flow_shift
            )

        # generator
        self.inference_model = SanaModelWrapper(sana_model_gen, flow_shift=flow_shift)
        self.inference_model.model.requires_grad_(False)
        self.inference_model = self.inference_model.to(device=device, dtype=self.dtype)

    def _initialize_models(self, args, device):
        """initialize Sana models"""
        self.is_main_process = (dist.get_rank() == 0) if dist.is_initialized() else True
        # sana config(generator)
        sana_config_path = getattr(args, "sana_config", "configs/sana_video_config/longsana/480ms/self_forcing.yaml")
        print(f"init sana config (generator): {sana_config_path}")
        sana_cfg = pyrallis.load(LongSANAVideoInference, open(sana_config_path))
        work_dir = "output/sana_logs"
        self.chunk_index = sana_cfg.model.chunk_index
        try:
            os.makedirs(work_dir, exist_ok=True)
        except Exception:
            pass
        # infer latent_size (480p: image_size // vae_downsample_rate)
        latent_size = sana_cfg.model.image_size // sana_cfg.vae.vae_downsample_rate
        model_kwargs = model_video_init_config(sana_cfg, latent_size=latent_size)
        if self.is_main_process:
            print(colored(f"init sana generator, chunk_index: {self.chunk_index}", "green"))
        sana_model_gen = build_model(
            sana_cfg.model.model,
            use_grad_checkpoint=True,
            use_fp32_attention=sana_cfg.model.get("fp32_attention", False) and sana_cfg.model.mixed_precision != "bf16",
            **model_kwargs,
        )
        if self.is_main_process:
            print(
                colored(
                    f"{sana_model_gen.__class__.__name__}:{sana_cfg.model.model},"
                    f"Model Parameters: {sum(p.numel() for p in sana_model_gen.parameters()):,}"
                ),
                "green",
            )
        if sana_cfg.flow_shift is not None:
            flow_shift = sana_cfg.flow_shift
        else:
            flow_shift = (
                sana_cfg.scheduler.inference_flow_shift
                if sana_cfg.scheduler.inference_flow_shift is not None
                else sana_cfg.scheduler.flow_shift
            )

        # generator
        self.generator = SanaModelChunkWrapper(sana_model_gen, flow_shift=flow_shift)
        self.generator.model.requires_grad_(True)
        self.generator = self.generator.to(device=device, dtype=self.dtype)

        self.text_encoder = SanaTextEncoder(sana_cfg=sana_cfg, device=device, dtype=self.dtype)
        self.text_encoder.requires_grad_(False)

        self.vae = SanaVAEWrapper(sana_cfg=sana_cfg, device=device, dtype=self.dtype)
        self.vae.requires_grad_(False)

        self.scheduler = self.generator.get_scheduler()
        self.scheduler.timesteps = self.scheduler.timesteps.to(device)

    def _get_timestep(
        self,
        min_timestep: int,
        max_timestep: int,
        batch_size: int,
        num_frame: int,
        num_frame_per_block: int,
        uniform_timestep: bool = False,
    ) -> torch.Tensor:
        """
        Randomly generate a timestep tensor based on the generator's task type. It uniformly samples a timestep
        from the range [min_timestep, max_timestep], and returns a tensor of shape [batch_size, num_frame].
        - If uniform_timestep, it will use the same timestep for all frames.
        - If not uniform_timestep, it will use a different timestep for each block.
        """
        if uniform_timestep:
            timestep = torch.randint(
                min_timestep, max_timestep, [batch_size, 1], device=self.device, dtype=torch.long
            ).repeat(1, num_frame)
            return timestep
        else:
            timestep = torch.randint(
                min_timestep, max_timestep, [batch_size, num_frame], device=self.device, dtype=torch.long
            )
            # make the noise level the same within every block
            if self.independent_first_frame:
                # the first frame is always kept the same
                timestep_from_second = timestep[:, 1:]
                timestep_from_second = timestep_from_second.reshape(
                    timestep_from_second.shape[0], -1, num_frame_per_block
                )
                timestep_from_second[:, :, 1:] = timestep_from_second[:, :, 0:1]
                timestep_from_second = timestep_from_second.reshape(timestep_from_second.shape[0], -1)
                timestep = torch.cat([timestep[:, 0:1], timestep_from_second], dim=1)
            else:
                timestep = timestep.reshape(timestep.shape[0], -1, num_frame_per_block)
                timestep[:, :, 1:] = timestep[:, :, 0:1]
                timestep = timestep.reshape(timestep.shape[0], -1)
            return timestep

    @torch.no_grad()
    def _prepare_generator_input(self, ode_latent: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Given a tensor containing the whole ODE sampling trajectories,
        randomly choose an intermediate timestep and return the latent as well as the corresponding timestep.
        Input:
            - ode_latent: a tensor containing the whole ODE sampling trajectories [batch_size, num_denoising_steps, num_frames, num_channels, height, width].
        Output:
            - noisy_input: a tensor containing the selected latent [batch_size, num_frames, num_channels, height, width].
            - timestep: a tensor containing the corresponding timestep [batch_size].
        """
        batch_size, num_denoising_steps, num_frames, num_channels, height, width = ode_latent.shape

        num_chunks = len(self.chunk_index)
        index_chunk = self._get_timestep(
            0,
            len(self.denoising_step_list),
            batch_size * num_chunks,
            1,
            self.num_frame_per_block,
            uniform_timestep=True,
        ).reshape(batch_size, num_chunks)
        chunk_start_end = self.chunk_index[:] + [num_frames]
        index = torch.zeros(batch_size, num_frames, device=self.device, dtype=torch.long)
        for i in range(num_chunks):
            index[:, chunk_start_end[i] : chunk_start_end[i + 1]] = index_chunk[:, i]

        if self.args.i2v:
            index[:, 0] = len(self.denoising_step_list) - 1

        noisy_input = torch.gather(
            ode_latent,
            dim=1,
            index=index.reshape(batch_size, 1, num_frames, 1, 1, 1)
            .expand(-1, -1, -1, num_channels, height, width)
            .to(self.device),
        ).squeeze(1)
        timestep = self.denoising_step_list[index].to(self.device)

        return noisy_input, timestep

    def generator_loss(self, ode_latent: torch.Tensor, conditional_dict: dict) -> Tuple[torch.Tensor, dict]:
        """
        Generate image/videos from noisy latents and compute the ODE regression loss.
        Input:
            - ode_latent: a tensor containing the ODE latents [batch_size, num_denoising_steps, num_frames, num_channels, height, width].
            They are ordered from most noisy to clean latents.
            - conditional_dict: a dictionary containing the conditional information (e.g. text embeddings, image embeddings).
        Output:
            - loss: a scalar tensor representing the generator loss.
            - log_dict: a dictionary containing additional information for loss timestep breakdown.
        """

        # Step 1: Run generator on noisy latents
        target_latent = ode_latent[:, -1]

        noisy_input, timestep = self._prepare_generator_input(ode_latent=ode_latent)

        noisy_bcfhw = noisy_input.permute(0, 2, 1, 3, 4)
        _, pred_image_or_video = self.generator(
            noisy_image_or_video=noisy_bcfhw,
            condition=conditional_dict.get("prompt_embeds", None),
            mask=conditional_dict.get("mask", None),
            timestep=timestep[:, None],
            chunk_index=self.chunk_index,
        )
        pred_image_or_video = pred_image_or_video.permute(0, 2, 1, 3, 4)  # B, T, C, H, W

        # Step 2: Compute the regression loss
        mask = timestep != 0

        if mask.sum() == 0:
            loss = F.mse_loss(pred_image_or_video, target_latent, reduction="mean") * 0
        else:
            loss = F.mse_loss(pred_image_or_video[mask], target_latent[mask], reduction="mean")

        log_dict = {
            "unnormalized_loss": F.mse_loss(pred_image_or_video, target_latent, reduction="none")
            .mean(dim=[1, 2, 3, 4])
            .detach(),
            "timestep": timestep.float().mean(dim=1).detach(),
            "timestep_list": timestep.float().detach(),
            "input": noisy_input.detach(),
            "output": pred_image_or_video.detach(),
        }

        return loss, log_dict
