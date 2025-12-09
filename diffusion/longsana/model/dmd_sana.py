import os
import time
from typing import Optional, Tuple

import imageio
import pyrallis
import torch
import torch.distributed as dist
import torch.nn.functional as F
from einops import rearrange
from termcolor import colored

from diffusion.longsana.pipeline.sana_inference_pipeline import SanaInferencePipeline
from diffusion.longsana.sana_video_pipeline import LongSANAVideoInference
from diffusion.longsana.utils.debug_option import DEBUG, LOG_GPU_MEMORY
from diffusion.longsana.utils.loss import get_denoising_loss
from diffusion.longsana.utils.model_wrapper import SanaModelWrapper, SanaTextEncoder, SanaVAEWrapper
from diffusion.model.builder import build_model, get_tokenizer_and_text_encoder, get_vae, vae_decode, vae_encode
from diffusion.utils.config import model_video_init_config
from tools.download import find_model


class DMDSana(torch.nn.Module):
    def __init__(self, args, device):
        """
        Initialize the DMD (Distribution Matching Distillation) module.
        This class is self-contained and compute generator and fake score losses
        in the forward pass.
        """
        super().__init__()

        self.device = device
        self.args = args
        self.dtype = torch.bfloat16 if args.mixed_precision else torch.float32

        self._initialize_sana_models(args, device)

        if hasattr(args, "denoising_step_list"):
            self.denoising_step_list = torch.tensor(args.denoising_step_list, dtype=torch.long)
            if args.warp_denoising_step:
                timesteps = torch.cat((self.scheduler.timesteps.cpu(), torch.tensor([0], dtype=torch.float32)))
                self.denoising_step_list = timesteps[1000 - self.denoising_step_list]

        # initialize denoising loss function
        self.denoising_loss_func = get_denoising_loss(args.denoising_loss_type)()

        self.num_frame_per_block = getattr(args, "num_frame_per_block", 10)
        self.same_step_across_blocks = getattr(args, "same_step_across_blocks", True)
        self.min_num_training_frames = getattr(args, "min_num_training_frames", 21)
        self.num_training_frames = getattr(args, "num_training_frames", 21)
        self.student_max_frame = getattr(args, "student_max_frame", 21)

        if self.num_frame_per_block > 1 and hasattr(self.generator, "model"):
            self.generator.model.num_frame_per_block = self.num_frame_per_block

        if args.gradient_checkpointing:
            if hasattr(self.generator, "enable_gradient_checkpointing"):
                self.generator.enable_gradient_checkpointing()
            if hasattr(self.fake_score, "enable_gradient_checkpointing"):
                self.fake_score.enable_gradient_checkpointing()

        self.inference_pipeline: SanaInferencePipeline = None

        self.image_or_video_shape = args.image_or_video_shape
        self.batch_size = args.image_or_video_shape[0]

        self.num_train_timestep = args.num_train_timestep
        self.min_step = int(0.02 * self.num_train_timestep)
        self.max_step = int(0.98 * self.num_train_timestep)
        if hasattr(args, "real_guidance_scale"):
            self.real_guidance_scale = args.real_guidance_scale
            self.fake_guidance_scale = args.fake_guidance_scale
        else:
            self.real_guidance_scale = args.guidance_scale
            self.fake_guidance_scale = 0.0
        self.timestep_shift = getattr(args, "timestep_shift", 1.0)
        self.ts_schedule = getattr(args, "ts_schedule", True)
        self.ts_schedule_max = getattr(args, "ts_schedule_max", False)
        self.min_score_timestep = getattr(args, "min_score_timestep", 0)

        if hasattr(self, "scheduler") and getattr(self.scheduler, "alphas_cumprod", None) is not None:
            self.scheduler.alphas_cumprod = self.scheduler.alphas_cumprod.to(device)
        else:
            if hasattr(self, "scheduler"):
                self.scheduler.alphas_cumprod = None
        self.step = 0
        self.output_path = os.path.join(args.logdir, "debug_save")
        os.makedirs(self.output_path, exist_ok=True)

        self.vis_interval = getattr(args, "vis_interval", 50)
        self.train_vis_interval = getattr(args, "train_vis_interval", self.vis_interval)
        self.train_vis_rank = getattr(args, "train_vis_rank", 8)

    def from_pretrained(self, model_path: str, strict: bool = False, fake_ckpt: str = None, real_ckpt: str = None):
        """
        load pretrained SANA weights:
        - always load model_path to generator
        - when fake_sana=True and fake is SANA, load fake_ckpt extra to fake_score
        - otherwise only load generator
        """
        state_dict = find_model(model_path)
        state_dict = state_dict.get("generator", state_dict)
        state_dict = state_dict.get("state_dict", state_dict)

        try:
            # load ode init model
            missing_g, unexpected_g = self.generator.load_state_dict(state_dict, strict=True)

        except Exception:
            missing_g, unexpected_g = self.generator.model.load_state_dict(state_dict, strict=True)

        # eval mode and align dtype/device
        self.generator.model.eval()
        self.generator = self.generator.to(device=self.device, dtype=self.dtype)
        ret = {
            "generator": {"missing": missing_g, "unexpected": unexpected_g},
        }
        if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
            print(f"[StreamingDMDSana] Loaded generator pretrained weights from {model_path}")
            print(f"[StreamingDMDSana] Generator missing={len(missing_g)}, unexpected={len(unexpected_g)}")

        fake_load_info = None
        if fake_ckpt is not None:
            fake_state = find_model(fake_ckpt)
            fake_state = fake_state.get("critic", fake_state)
            fake_state = fake_state.get("state_dict", fake_state)
            try:
                missing_f_fake, unexpected_f_fake = self.fake_score.load_state_dict(fake_state, strict=strict)
            except Exception:
                missing_f_fake, unexpected_f_fake = self.fake_score.model.load_state_dict(fake_state, strict=strict)
            fake_load_info = {"missing": missing_f_fake, "unexpected": unexpected_f_fake, "path": fake_ckpt}
            self.fake_score.model.eval()
            self.fake_score = self.fake_score.to(device=self.device, dtype=self.dtype)
            # load fake ckpt to real score
            if real_ckpt is None:
                real_ckpt = fake_ckpt

        if self.real_name == "SANA":
            real_state = find_model(real_ckpt)
            real_state = real_state.get("state_dict", real_state)
            try:
                missing_f_real, unexpected_f_real = self.real_score.load_state_dict(real_state, strict=strict)
            except Exception:
                missing_f_real, unexpected_f_real = self.real_score.model.load_state_dict(real_state, strict=strict)
            self.real_score.model.eval()
            self.real_score = self.real_score.to(device=self.device, dtype=self.dtype)

        if not dist.is_initialized() or dist.get_rank() == 0:
            print(colored(f"[StreamingDMDSana] Loaded Generator SANA from: {model_path}", "green"))
            print(colored(f"[StreamingDMDSana] Loaded fake SANA from: {fake_ckpt}", "green"))
            if self.real_name == "SANA":
                print(colored(f"[StreamingDMDSana] Loaded real SANA from: {real_ckpt}", "green"))

        if DEBUG and (not dist.is_initialized() or dist.get_rank() == 0):
            print(f"[StreamingDMDSana] Loaded fake SANA from: {fake_ckpt}")
            print(f"[StreamingDMDSana] FakeScore missing={len(missing_f_fake)}, unexpected={len(unexpected_f_fake)}")

        if fake_load_info is not None:
            ret["fake_score"] = fake_load_info
        return ret

    def set_step(self, step: int):
        self.step = step

    def _initialize_sana_models(self, args, device):
        """initialize Sana models"""
        self.is_main_process = dist.get_rank() == 0
        self.real_name = getattr(args, "real_name", "SANA")
        if self.is_main_process:
            print(colored(f"init real model: {self.real_name}", "green"))

        if "sana" in self.real_name.lower():
            # use independent sana config (if not provided, fallback to generator's config)
            sana_fake_config_path = getattr(args, "sana_fake_config", None)
            if self.is_main_process:
                print(f"init sana config (fake) for real model: {sana_fake_config_path}")
            sana_cfg_fake = pyrallis.load(LongSANAVideoInference, open(sana_fake_config_path))
            latent_size_fake = sana_cfg_fake.model.image_size // sana_cfg_fake.vae.vae_downsample_rate
            model_kwargs_fake = model_video_init_config(sana_cfg_fake, latent_size=latent_size_fake)
            if self.is_main_process:
                print(colored(f"init sana fake for real model", "green"))
            sana_model_real = build_model(
                sana_cfg_fake.model.model,
                use_grad_checkpoint=True,
                use_fp32_attention=sana_cfg_fake.model.get("fp32_attention", False)
                and sana_cfg_fake.model.mixed_precision != "bf16",
                **model_kwargs_fake,
            )
            if self.is_main_process:
                print(
                    colored(
                        f"{sana_model_real.__class__.__name__}:{sana_cfg_fake.model.model},"
                        f"Model Parameters: {sum(p.numel() for p in sana_model_real.parameters()):,}"
                    ),
                    "green",
                )
            # flow shift for fake
            if sana_cfg_fake.flow_shift is not None:
                flow_shift_fake = sana_cfg_fake.flow_shift
            else:
                flow_shift_fake = (
                    sana_cfg_fake.scheduler.inference_flow_shift
                    if sana_cfg_fake.scheduler.inference_flow_shift is not None
                    else sana_cfg_fake.scheduler.flow_shift
                )
            self.real_score = SanaModelWrapper(sana_model_real, flow_shift=flow_shift_fake)
            self.real_score = self.real_score.to(device=device, dtype=self.dtype)

            self.text_encoder_real = SanaTextEncoder(sana_cfg=sana_cfg_fake, device=device, dtype=self.dtype)
            self.text_encoder_real.requires_grad_(False)
        else:
            raise ValueError(f"Invalid real model: {self.real_name}")

        self.real_score.model.requires_grad_(False)

        # sana config(generator)
        # TODO: Need to make the path more robust.
        sana_config_path = getattr(args, "sana_config", "sana/configs/Sana_2B_480p_self_forcing.yaml")
        print(f"init sana config (generator): {sana_config_path}")
        sana_cfg = pyrallis.load(LongSANAVideoInference, open(sana_config_path))
        work_dir = "output/sana_logs"
        try:
            os.makedirs(work_dir, exist_ok=True)
        except Exception:
            pass
        latent_size = sana_cfg.model.image_size // sana_cfg.vae.vae_downsample_rate
        model_kwargs = model_video_init_config(sana_cfg, latent_size=latent_size)
        if self.is_main_process:
            print(colored(f"init sana generator", "green"))
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

        # fake_model
        fake_name = getattr(args, "fake_name", "SANA")
        self.fake_name = fake_name
        if fake_name == "SANA":
            # use independent sana config (if not provided, fallback to generator's config)
            sana_fake_config_path = getattr(args, "sana_fake_config", None)
            print(f"init sana config (fake): {sana_fake_config_path}")
            sana_cfg_fake = pyrallis.load(LongSANAVideoInference, open(sana_fake_config_path))
            latent_size_fake = sana_cfg_fake.model.image_size // sana_cfg_fake.vae.vae_downsample_rate
            if latent_size_fake != latent_size:
                raise ValueError(
                    f"Generator and fake SANA latent_size mismatch: gen={latent_size}, fake={latent_size_fake}. "
                    f"Please ensure compatible VAE/image_size settings."
                )
            model_kwargs_fake = model_video_init_config(sana_cfg_fake, latent_size=latent_size_fake)
            if self.is_main_process:
                print(colored(f"init sana fake", "green"))
            sana_model_fake = build_model(
                sana_cfg_fake.model.model,
                use_grad_checkpoint=True,
                use_fp32_attention=sana_cfg_fake.model.get("fp32_attention", False)
                and sana_cfg_fake.model.mixed_precision != "bf16",
                **model_kwargs_fake,
            )
            if self.is_main_process:
                print(
                    colored(
                        f"{sana_model_fake.__class__.__name__}:{sana_cfg_fake.model.model},"
                        f"Model Parameters: {sum(p.numel() for p in sana_model_fake.parameters()):,}"
                    ),
                    "green",
                )
            # flow shift for fake
            if sana_cfg_fake.flow_shift is not None:
                flow_shift_fake = sana_cfg_fake.flow_shift
            else:
                flow_shift_fake = (
                    sana_cfg_fake.scheduler.inference_flow_shift
                    if sana_cfg_fake.scheduler.inference_flow_shift is not None
                    else sana_cfg_fake.scheduler.flow_shift
                )
            self.fake_score = SanaModelWrapper(sana_model_fake, flow_shift=flow_shift_fake)
            self.fake_score.model.requires_grad_(True)
            self.fake_score = self.fake_score.to(device=device, dtype=self.dtype)

        # generator
        self.generator = SanaModelWrapper(sana_model_gen, flow_shift=flow_shift)
        self.generator.model.requires_grad_(True)
        self.generator = self.generator.to(device=device, dtype=self.dtype)

        self.text_encoder = SanaTextEncoder(sana_cfg=sana_cfg, device=device, dtype=self.dtype)
        self.text_encoder.requires_grad_(False)

        self.vae = SanaVAEWrapper(sana_cfg=sana_cfg, device=device, dtype=self.dtype)
        self.vae.requires_grad_(False)

        self.scheduler = self.generator.get_scheduler()
        self.scheduler.timesteps = self.scheduler.timesteps.to(device)

        if self.fake_name == "SANA":
            load_info = self.from_pretrained(
                args.generator_ckpt, strict=True, fake_ckpt=args.fake_ckpt, real_ckpt=args.get("real_ckpt", None)
            )
        else:
            load_info = self.from_pretrained(args.generator_ckpt, strict=True, real_ckpt=args.get("real_ckpt", None))
        print(f"[Load] SANA weights loaded: {load_info}")

    def _initialize_inference_pipeline(self):
        """
        initialize training pipeline for DMDSana, using SanaTrainingPipeline
        """
        from diffusion.longsana.pipeline.sana_switch_training_pipeline import SanaSwitchTrainingPipeline
        from diffusion.longsana.pipeline.sana_training_pipeline import SanaTrainingPipeline

        if getattr(self.args, "switch_prompt_path", None) is not None:
            self.inference_pipeline = SanaSwitchTrainingPipeline(
                denoising_step_list=self.denoising_step_list,
                scheduler=self.scheduler,
                generator=self.generator,
                same_step_across_blocks=self.args.same_step_across_blocks,
                last_step_only=self.args.last_step_only,
                num_max_frames=self.student_max_frame,
                context_noise=self.args.get("context_noise", 0),
                num_frame_per_block=self.num_frame_per_block,
                num_chunks_per_clip=self.args.num_chunks_per_clip,
                batch_size=self.batch_size,
                update_kv_cache_by_end=self.args.get("update_kv_cache_by_end", False),
                num_cached_blocks=self.args.get("num_cached_blocks", -1),
            )
        else:
            self.inference_pipeline = SanaTrainingPipeline(
                denoising_step_list=self.denoising_step_list,
                scheduler=self.scheduler,
                generator=self.generator,
                same_step_across_blocks=self.args.same_step_across_blocks,
                last_step_only=self.args.last_step_only,
                num_max_frames=self.student_max_frame,
                context_noise=self.args.get("context_noise", 0),
                num_frame_per_block=self.num_frame_per_block,
                num_chunks_per_clip=self.args.num_chunks_per_clip,
                batch_size=self.batch_size,
                update_kv_cache_by_end=self.args.get("update_kv_cache_by_end", False),
                num_cached_blocks=self.args.get("num_cached_blocks", -1),
            )

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

    def _compute_kl_grad(
        self,
        noisy_image_or_video: torch.Tensor,  # B,F,C,H,W
        estimated_clean_image_or_video: torch.Tensor,  # B,F,C,H,W
        timestep: torch.Tensor,
        conditional_dict: dict,
        unconditional_dict: dict,
        conditional_dict_real: dict,
        unconditional_dict_real: dict,
        normalization: bool = True,
        current_length: int = 0,
    ) -> Tuple[torch.Tensor, dict]:
        """
        compute KL gradient (reference DMD implementation).
        """
        # compute fake conditional prediction (CFG requires cond/uncond)
        if self.fake_name == "SANA":
            assert "mask" in conditional_dict
            condition = conditional_dict["prompt_embeds"].clone()
            mask = conditional_dict.get("mask", None)
            # BFCHW -> BCFHW
            noisy_bcfhw = rearrange(noisy_image_or_video, "b f c h w -> b c f h w")
            _, pred_fake_image_cond_bcfhw, _ = self.fake_score(
                noisy_image_or_video=noisy_bcfhw, condition=condition, timestep=timestep, mask=mask
            )
            # BCFHW -> BFCHW
            pred_fake_image_cond = rearrange(pred_fake_image_cond_bcfhw, "b c f h w -> b f c h w")
        else:
            _, pred_fake_image_cond = self.fake_score(
                noisy_image_or_video=noisy_image_or_video,
                conditional_dict=conditional_dict_real,
                timestep=timestep,
            )
        if getattr(self, "fake_guidance_scale", 0.0) != 0.0:
            raise NotImplementedError("Fake guidance scale is not supported for SANA")
        else:
            pred_fake_image = pred_fake_image_cond

        if self.real_name == "SANA":
            assert "mask" in conditional_dict_real
            condition = conditional_dict_real["prompt_embeds"].clone()
            mask = conditional_dict_real.get("mask", None)
            # BFCHW -> BCFHW
            noisy_bcfhw = rearrange(noisy_image_or_video, "b f c h w -> b c f h w")
            _, pred_real_image_cond_bcfhw, _ = self.real_score(
                noisy_image_or_video=noisy_bcfhw, condition=condition, timestep=timestep, mask=mask
            )
            # BCFHW -> BFCHW
            pred_real_image_cond = rearrange(pred_real_image_cond_bcfhw, "b c f h w -> b f c h w")

            # unconditional
            assert "mask" in unconditional_dict_real
            condition = unconditional_dict_real["prompt_embeds"].clone()
            mask = unconditional_dict_real.get("mask", None)
            _, pred_real_image_uncond_bcfhw, _ = self.real_score(
                noisy_image_or_video=noisy_bcfhw, condition=condition, timestep=timestep, mask=mask
            )
            pred_real_image_uncond = rearrange(pred_real_image_uncond_bcfhw, "b c f h w -> b f c h w")
        else:
            # real score  predict x0 and do CFG
            _, pred_real_image_cond = self.real_score(
                noisy_image_or_video=noisy_image_or_video, conditional_dict=conditional_dict_real, timestep=timestep
            )
            _, pred_real_image_uncond = self.real_score(
                noisy_image_or_video=noisy_image_or_video, conditional_dict=unconditional_dict_real, timestep=timestep
            )

        pred_real_image = (
            pred_real_image_cond + (pred_real_image_cond - pred_real_image_uncond) * self.real_guidance_scale
        )

        # DMD gradient
        grad = pred_fake_image - pred_real_image

        self.decode_and_save_clip(
            pred_real_image, f"generator_real_f{current_length}_t{int(timestep.reshape(-1)[0].item())}"
        )
        self.decode_and_save_clip(
            pred_fake_image, f"generator_fake_f{current_length}_t{int(timestep.reshape(-1)[0].item())}"
        )

        if normalization:
            p_real = estimated_clean_image_or_video - pred_real_image
            normalizer = torch.abs(p_real).mean(dim=[1, 2, 3, 4], keepdim=True)
            grad = grad / normalizer
        grad = torch.nan_to_num(grad)
        return grad, {"dmdtrain_gradient_norm": torch.mean(torch.abs(grad)).detach(), "timestep": timestep.detach()}

    def compute_distribution_matching_loss(
        self,
        image_or_video: torch.Tensor,
        conditional_dict: dict,
        unconditional_dict: dict,
        conditional_dict_real: dict,
        unconditional_dict_real: dict,
        gradient_mask: Optional[torch.Tensor] = None,
        denoised_timestep_from: int = 0,
        denoised_timestep_to: int = 0,
        current_length: int = 0,
    ) -> Tuple[torch.Tensor, dict]:
        """
        compute DMD loss (consistent with DMD class).
        """
        # image_or_video: B,F,C,H,W
        original_latent = image_or_video
        batch_size, num_frame = image_or_video.shape[:2]

        with torch.no_grad():
            # sample timestep and add noise
            min_timestep = (
                denoised_timestep_to
                if self.ts_schedule and denoised_timestep_to is not None
                else self.min_score_timestep
            )
            max_timestep = (
                denoised_timestep_from
                if self.ts_schedule_max and denoised_timestep_from is not None
                else self.num_train_timestep
            )
            timestep = self._get_timestep(
                min_timestep, max_timestep, batch_size, num_frame, self.num_frame_per_block, uniform_timestep=True
            )
            if self.timestep_shift > 1:
                timestep = (
                    self.timestep_shift * (timestep / 1000) / (1 + (self.timestep_shift - 1) * (timestep / 1000)) * 1000
                )
            timestep = timestep.clamp(self.min_step, self.max_step)

            noise = torch.randn_like(image_or_video)
            noisy_latent = (
                self.scheduler.add_noise(image_or_video.flatten(0, 1), noise.flatten(0, 1), timestep.flatten(0, 1))
                .detach()
                .unflatten(0, (batch_size, num_frame))
            )

            # KL gradient
            grad, dmd_log_dict = self._compute_kl_grad(
                noisy_image_or_video=noisy_latent,
                estimated_clean_image_or_video=original_latent,
                timestep=timestep,
                conditional_dict=conditional_dict,
                unconditional_dict=unconditional_dict,
                conditional_dict_real=conditional_dict_real,
                unconditional_dict_real=unconditional_dict_real,
                current_length=current_length,
            )

        if gradient_mask is not None:
            dmd_loss = 0.5 * F.mse_loss(
                original_latent.double()[gradient_mask],
                (original_latent.double() - grad.double()).detach()[gradient_mask],
                reduction="mean",
            )
        else:
            dmd_loss = 0.5 * F.mse_loss(
                original_latent.double(), (original_latent.double() - grad.double()).detach(), reduction="mean"
            )
        return dmd_loss, dmd_log_dict

    def _run_generator(
        self, image_or_video_shape, prompt_embeds, mask, initial_latent=None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Optionally simulate the generator's input from noise using backward simulation
        and then run the generator for one-step.
        Input:
            - image_or_video_shape: a list containing the shape of the image or video [B, F, C, H, W].
            - conditional_dict: a dictionary containing the conditional information (e.g. text embeddings, image embeddings).
            - unconditional_dict: a dictionary containing the unconditional information (e.g. null/negative text embeddings, null/negative image embeddings).
            - clean_latent: a tensor containing the clean latents [B, F, C, H, W]. Need to be passed when no backward simulation is used.
            - initial_latent: a tensor containing the initial latents [B, F, C, H, W].
        Output:
            - pred_image: a tensor with shape [B, F, C, H, W].
            - denoised_timestep: an integer
        """
        # Step 1: Sample noise and backward simulate the generator's input
        assert getattr(self.args, "backward_simulation", True), "Backward simulation needs to be enabled"
        conditional_dict = {"prompt_embeds": prompt_embeds, "mask": mask}
        if initial_latent is not None:
            conditional_dict["initial_latent"] = initial_latent
        if self.args.i2v:
            noise_shape = [image_or_video_shape[0], image_or_video_shape[1] - 1, *image_or_video_shape[2:]]
        else:
            noise_shape = image_or_video_shape.copy()

        b, f, c, h, w = image_or_video_shape
        noise_shape = [b, c, f, h, w]

        # During training, the number of generated frames should be uniformly sampled from
        # [21, self.num_training_frames], but still being a multiple of self.num_frame_per_block
        self.args.independent_first_frame = (
            f % 20 == 1
        )  # if 21 frames, independent first frame is True, otherwise False
        # assert self.args.independent_first_frame, "Independent first frame is required for SANA"
        min_num_frames = (
            self.min_num_training_frames - 1 if self.args.independent_first_frame else self.min_num_training_frames
        )
        max_num_frames = self.num_training_frames - 1 if self.args.independent_first_frame else self.num_training_frames
        assert max_num_frames % self.num_frame_per_block == 0
        assert min_num_frames % self.num_frame_per_block == 0
        max_num_blocks = max_num_frames // self.num_frame_per_block
        min_num_blocks = min_num_frames // self.num_frame_per_block
        num_generated_blocks = torch.randint(min_num_blocks, max_num_blocks + 1, (1,), device=self.device)
        dist.broadcast(num_generated_blocks, src=0)
        num_generated_blocks = num_generated_blocks.item()
        num_generated_frames = num_generated_blocks * self.num_frame_per_block
        if self.args.independent_first_frame and initial_latent is None:
            num_generated_frames += 1
            min_num_frames += 1
        # Sync num_generated_frames across all processes
        noise_shape = [b, c, num_generated_frames, h, w]

        pred_image_or_video, denoised_timestep_from, denoised_timestep_to = self._consistency_backward_simulation(
            noise=torch.randn(noise_shape, device=self.device, dtype=self.dtype),
            **conditional_dict,
        )  # B,F,C,H,W
        # Slice last 21 frames
        if pred_image_or_video.shape[1] > 21:
            with torch.no_grad():
                # Reencode to get image latent
                latent_to_decode = pred_image_or_video[:, :-20, ...]  # B,F-20,C,H,W

                # Deccode to video
                pixels = self.vae.decode_to_pixel(rearrange(latent_to_decode, "b f c h w -> b c f h w"))  # b,c,f,h,w
                pixels = torch.stack(pixels, dim=0)
                frame = pixels[:, :, -1:, ...].to(self.dtype)  # b,c,1,h,w
                # Encode frame to get image latent
                image_latent = self.vae.encode_to_latent(frame).to(self.dtype)
                image_latent = rearrange(image_latent, "b c f h w -> b f c h w")
            pred_image_or_video_last_21 = torch.cat(
                [image_latent, pred_image_or_video[:, -20:, ...]], dim=1
            )  # B,F,C,H,W
        else:
            pred_image_or_video_last_21 = pred_image_or_video

        if num_generated_frames != min_num_frames:
            # Currently, we do not use gradient for the first chunk, since it contains image latents
            gradient_mask = torch.ones_like(pred_image_or_video_last_21, dtype=torch.bool)
            if self.args.independent_first_frame:
                gradient_mask[:, :1] = False
            else:
                gradient_mask[:, : self.num_frame_per_block] = False
        else:
            gradient_mask = None

        pred_image_or_video_last_21 = pred_image_or_video_last_21.to(self.dtype)
        return pred_image_or_video_last_21, gradient_mask, denoised_timestep_from, denoised_timestep_to

    def _consistency_backward_simulation(self, noise, prompt_embeds, mask, initial_latent=None):

        if self.inference_pipeline is None:
            self._initialize_inference_pipeline()

        output, denoised_timestep_from, denoised_timestep_to = self.inference_pipeline.inference_with_trajectory(
            noise=noise,
            prompt_embeds=prompt_embeds,
            mask=mask,
        )  # B,C,F,H,W

        return rearrange(output, "b c f h w -> b f c h w"), denoised_timestep_from, denoised_timestep_to

    def generator_loss(
        self,
        image_or_video_shape,
        conditional_dict: dict,
        unconditional_dict: dict,
        conditional_dict_real: dict,
        unconditional_dict_real: dict,
        clean_latent: torch.Tensor,
        initial_latent: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, dict]:
        """
        same as DMD: unfold generator to get fake video and compute DMD loss.
        """

        _t_gen_start = time.time()
        if DEBUG and dist.get_rank() == 0:
            print(f"generator_rollout")
        pred_image, gradient_mask, denoised_timestep_from, denoised_timestep_to = self._run_generator(
            image_or_video_shape=image_or_video_shape,
            prompt_embeds=conditional_dict["prompt_embeds"],
            mask=conditional_dict.get("mask", None),
            initial_latent=initial_latent,
        )  # B,F,C,H,W
        if dist.get_rank() == 0 and DEBUG:
            print(f"pred_image: {pred_image.shape}")
        gen_time = time.time() - _t_gen_start
        _t_loss_start = time.time()
        self.decode_and_save_clip(pred_image, f"generator_gen_to_t{int(denoised_timestep_to)}")

        dmd_loss, dmd_log_dict = self.compute_distribution_matching_loss(
            image_or_video=pred_image,
            conditional_dict=conditional_dict,
            unconditional_dict=unconditional_dict,
            conditional_dict_real=conditional_dict_real,
            unconditional_dict_real=unconditional_dict_real,
            gradient_mask=gradient_mask,
            denoised_timestep_from=denoised_timestep_from,
            denoised_timestep_to=denoised_timestep_to,
        )
        loss_time = time.time() - _t_loss_start
        dmd_log_dict.update({"gen_time": gen_time, "loss_time": loss_time})
        return dmd_loss, dmd_log_dict

    def critic_loss(
        self,
        image_or_video_shape,
        conditional_dict: dict,
        unconditional_dict: dict,
        conditional_dict_real: dict,
        unconditional_dict_real: dict,
        clean_latent: torch.Tensor,
        initial_latent: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, dict]:
        """
        consistent with DMD: generate samples and train fake_score (SANA).
        """
        _t_gen_start = time.time()
        with torch.no_grad():
            if DEBUG and dist.get_rank() == 0:
                print(f"critic_rollout")
            generated_image, gradient_mask, denoised_timestep_from, denoised_timestep_to = self._run_generator(
                image_or_video_shape=image_or_video_shape,
                prompt_embeds=conditional_dict["prompt_embeds"],
                mask=conditional_dict.get("mask", None),
                initial_latent=initial_latent,
            )  # B,F,C,H,W
            self.decode_and_save_clip(generated_image, f"critic_gen_t{int(denoised_timestep_to)}")

        if dist.get_rank() == 0 and DEBUG:
            print(f"pred_image: {generated_image.shape}")
        gen_time = time.time() - _t_gen_start
        batch_size, num_frame = generated_image.shape[:2]

        _t_loss_start = time.time()
        min_timestep = (
            denoised_timestep_to if self.ts_schedule and denoised_timestep_to is not None else self.min_score_timestep
        )
        max_timestep = (
            denoised_timestep_from
            if self.ts_schedule_max and denoised_timestep_from is not None
            else self.num_train_timestep
        )
        critic_timestep = self._get_timestep(
            min_timestep, max_timestep, batch_size, num_frame, self.num_frame_per_block, uniform_timestep=True
        )
        if self.timestep_shift > 1:
            critic_timestep = (
                self.timestep_shift
                * (critic_timestep / 1000)
                / (1 + (self.timestep_shift - 1) * (critic_timestep / 1000))
                * 1000
            )
        critic_timestep = critic_timestep.clamp(self.min_step, self.max_step)

        critic_noise = torch.randn_like(generated_image)
        noisy_generated_image = self.scheduler.add_noise(
            generated_image.flatten(0, 1), critic_noise.flatten(0, 1), critic_timestep.flatten(0, 1)
        ).unflatten(
            0, (batch_size, num_frame)
        )  # B,F,C,H,W

        if self.fake_name == "SANA":
            assert "mask" in conditional_dict
            condition = conditional_dict["prompt_embeds"].clone()
            mask = conditional_dict.get("mask", None)
            # BFCHW -> BCFHW
            noisy_bcfhw = rearrange(noisy_generated_image, "b f c h w -> b c f h w")
            _, pred_fake_image_bcfhw, _ = self.fake_score(
                noisy_image_or_video=noisy_bcfhw, condition=condition, timestep=critic_timestep, mask=mask
            )
            # BCFHW -> BFCHW
            pred_fake_image = rearrange(pred_fake_image_bcfhw, "b c f h w -> b f c h w")
        else:
            _, pred_fake_image = self.fake_score(
                noisy_image_or_video=noisy_generated_image,
                conditional_dict=conditional_dict_real,
                timestep=critic_timestep,
            )

        # compute denoising loss (support flow/mse)
        denoising_loss_type = getattr(self.args, "denoising_loss_type", "mse")
        if denoising_loss_type == "flow":
            flow_pred = SanaModelWrapper._convert_x0_to_flow_pred(
                scheduler=self.scheduler,
                x0_pred=pred_fake_image.flatten(0, 1),
                xt=noisy_generated_image.flatten(0, 1),
                timestep=critic_timestep.flatten(0, 1),
            )
            pred_fake_noise = None
        else:
            flow_pred = None
            pred_fake_noise = self.scheduler.convert_x0_to_noise(
                x0=pred_fake_image.flatten(0, 1),
                xt=noisy_generated_image.flatten(0, 1),
                timestep=critic_timestep.flatten(0, 1),
            ).unflatten(0, (batch_size, num_frame))

        denoising_loss = self.denoising_loss_func(
            x=generated_image.flatten(0, 1),
            x_pred=pred_fake_image.flatten(0, 1),
            noise=critic_noise.flatten(0, 1),
            noise_pred=pred_fake_noise,
            alphas_cumprod=self.scheduler.alphas_cumprod,
            timestep=critic_timestep.flatten(0, 1),
            flow_pred=flow_pred,
        )
        loss_time = time.time() - _t_loss_start

        critic_log_dict = {"critic_timestep": critic_timestep.detach(), "gen_time": gen_time, "loss_time": loss_time}

        self.decode_and_save_clip(pred_fake_image, f"critic_fake_t{int(critic_timestep.reshape(-1)[0].item())}")
        return denoising_loss, critic_log_dict

    @torch.no_grad()
    def decode_and_save_clip(self, clip_btchw: torch.Tensor, save_name: str, fps: int = 16):
        rank = dist.get_rank() if dist.is_initialized() else 0
        if self.train_vis_interval > 0 and self.step % self.train_vis_interval == 0 and rank < self.train_vis_rank:
            clip_bcthw = clip_btchw.permute(0, 2, 1, 3, 4)
            pixel_bcthw = self.vae.decode_to_pixel(clip_bcthw)
            if isinstance(pixel_bcthw, list):
                pixel_bcthw = torch.stack(pixel_bcthw, dim=0)
            pixel_btchw = (
                torch.clamp(127.5 * pixel_bcthw + 127.5, 0, 255).permute(0, 2, 3, 4, 1).to("cpu", dtype=torch.uint8)
            )  # B,T,H,W,C -> B,T,H,W,C
            if dist.is_initialized():
                save_path = f"{self.output_path}/step_{self.step:05d}_rank{rank}_{save_name}.mp4"
            else:
                save_path = f"{self.output_path}/step_{self.step:05d}_{save_name}.mp4"
            os.makedirs(os.path.dirname(save_path), exist_ok=True)

            writer = imageio.get_writer(save_path, fps=16, codec="libx264", quality=8)
            for video in pixel_btchw:
                for frame in video.numpy():
                    writer.append_data(frame)
            writer.close()
