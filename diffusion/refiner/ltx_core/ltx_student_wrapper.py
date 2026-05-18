"""Student wrapper: merged teacher checkpoint + fresh causal LoRA on self-attention qkv.

Only the LoRA parameters are trainable.  The base weights are frozen and
identical to the teacher initialization.
"""

from __future__ import annotations

import logging
from typing import Iterator

import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusion.refiner.vendor.ltx_core.components.patchifiers import VideoLatentPatchifier
from diffusion.refiner.vendor.ltx_core.loader import LoraPathStrengthAndSDOps
from diffusion.refiner.vendor.ltx_core.loader.single_gpu_model_builder import SingleGPUModelBuilder as Builder
from diffusion.refiner.vendor.ltx_core.model.transformer import (
    LTXV_MODEL_COMFY_RENAMING_MAP,
    LTXModelConfigurator,
    X0Model,
)
from diffusion.refiner.vendor.ltx_core.types import LatentState, VideoLatentShape
from diffusion.refiner.vendor.ltx_pipelines.utils.helpers import modality_from_latent_state

from .causal_attention import apply_causal_mask

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# LoRA adapter
# ---------------------------------------------------------------------------


class LoRALinear(nn.Module):
    """Wraps a frozen ``nn.Linear`` with a low-rank adapter."""

    # Class-level switch: set False to bypass LoRA (teacher forward uses this).
    enabled = True

    def __init__(self, base: nn.Linear, rank: int, alpha: float) -> None:
        super().__init__()
        self.base = base
        self.lora_A = nn.Parameter(
            torch.zeros(rank, base.in_features, device=base.weight.device, dtype=base.weight.dtype)
        )
        self.lora_B = nn.Parameter(
            torch.zeros(base.out_features, rank, device=base.weight.device, dtype=base.weight.dtype)
        )
        self.scaling = alpha / rank
        nn.init.kaiming_uniform_(self.lora_A)
        # B is zero-initialized so LoRA output is initially zero (base + 0 = base)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        if not LoRALinear.enabled:
            return out
        # Memory-efficient LoRA: in-place add avoids materialising a second
        # full-size (base_out + lora_out) tensor, and F.linear uses BLAS
        # kernels directly so the implicit .to(x.dtype) copy on lora_A / lora_B
        # is gone (FSDP2 already gives us bf16 params for a bf16 input).
        # Rank-space intermediate is tiny: [*, rank] ≈ 78 MB vs 3.3 GB hidden.
        out += F.linear(F.linear(x, self.lora_A), self.lora_B) * self.scaling
        return out


def inject_lora(
    model: nn.Module,
    target_modules: list[str],
    rank: int,
    alpha: float,
) -> dict[str, LoRALinear]:
    """Replace target ``nn.Linear`` modules with ``LoRALinear`` wrappers.

    Returns a dict of ``{dotted_path: LoRALinear}`` for the injected modules.
    """
    injected: dict[str, LoRALinear] = {}
    for name, module in list(model.named_modules()):
        # Match: name must end with a target AND the segment before the target
        # must not be an audio module (e.g. "audio_attn1.to_q" should NOT match "attn1.to_q").
        matched = False
        for t in target_modules:
            if not name.endswith(t):
                continue
            prefix = name[: len(name) - len(t)]
            if prefix.endswith("audio_") or prefix.endswith("audio_to_video_") or prefix.endswith("video_to_audio_"):
                continue
            matched = True
            break
        if not matched:
            continue
        if not isinstance(module, nn.Linear):
            continue
        for p in module.parameters():
            p.requires_grad_(False)
        lora = LoRALinear(module, rank, alpha)
        parts = name.split(".")
        parent = model
        for part in parts[:-1]:
            parent = getattr(parent, part)
        setattr(parent, parts[-1], lora)
        injected[name] = lora
    return injected


# ---------------------------------------------------------------------------
# Student wrapper
# ---------------------------------------------------------------------------

DEFAULT_LORA_TARGETS = ["attn1.to_q", "attn1.to_k", "attn1.to_v"]


class LTXStudentWrapper:
    """Student model: frozen base (same as teacher) + trainable causal LoRA.

    Parameters
    ----------
    checkpoint_path : str
        Pre-merged (fused) teacher checkpoint.  If you already merged the
        distilled LoRA into the base model (e.g. ``ltx-2-19b-refiner-fused.safetensors``),
        pass that path here and leave ``loras`` empty.
    lora_rank : int
        Rank for the **new** student LoRA adapters (trained from scratch).
    lora_alpha : float
        Alpha scaling for the new student LoRA adapters.
    lora_target_modules : list[str], optional
        Which modules to attach LoRA to.  Default: self-attention qkv.
    loras : list[LoraPathStrengthAndSDOps], optional
        Additional LoRAs to merge at load time (usually empty when using
        a pre-fused checkpoint).
    """

    def __init__(
        self,
        checkpoint_path: str,
        lora_rank: int = 384,
        lora_alpha: float = 384.0,
        lora_target_modules: list[str] | None = None,
        loras: list[LoraPathStrengthAndSDOps] | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        self.device = device or (
            torch.device("cuda", torch.cuda.current_device()) if torch.cuda.is_available() else torch.device("cpu")
        )
        self.dtype = dtype
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.lora_target_modules = lora_target_modules or DEFAULT_LORA_TARGETS

        # Build from fused checkpoint (distilled LoRA already merged in)
        builder = Builder(
            model_path=checkpoint_path,
            model_class_configurator=LTXModelConfigurator,
            model_sd_ops=LTXV_MODEL_COMFY_RENAMING_MAP,
            loras=tuple(loras or ()),
        )
        self._transformer: X0Model = X0Model(builder.build(device=self.device)).to(self.device).eval()

        # Freeze everything first
        for p in self._transformer.parameters():
            p.requires_grad_(False)

        # Inject LoRA on target modules (only these params will be trainable).
        # Passing rank <= 0 (or an empty target list) intentionally disables
        # LoRA — used for partial-FT inference paths (e.g. the chunk-causal AR
        # refiner trained with ``train_mode=attn_qkvo`` and ``lora_rank=0``)
        # where the on-disk DCP shards expect the *base* parameter keys, not
        # ``lora_A`` / ``lora_B``. Mirrors the dev/tian wrapper.
        if self.lora_rank <= 0 or not self.lora_target_modules:
            self._lora_modules = {}
        else:
            self._lora_modules = inject_lora(
                self._transformer,
                self.lora_target_modules,
                self.lora_rank,
                self.lora_alpha,
            )

        # Set student to train mode (for dropout etc.) but base stays frozen
        self._transformer.train()

        n_lora = sum(p.numel() for p in self.lora_parameters())
        n_total = sum(p.numel() for p in self._transformer.parameters())
        logger.info(
            "[LTXStudentWrapper] %d LoRA modules injected, %d trainable params (%.2f%% of %d total)",
            len(self._lora_modules),
            n_lora,
            100.0 * n_lora / n_total,
            n_total,
        )

    @classmethod
    def from_teacher(
        cls,
        teacher: LTXTeacherWrapper,
        lora_rank: int = 384,
        lora_alpha: float = 384.0,
        lora_target_modules: list[str] | None = None,
    ) -> LTXStudentWrapper:
        """Create student by injecting LoRA onto the teacher's transformer.

        Shares the same base weights — no extra GPU memory for a second model.
        The teacher must not be used for inference after this (its modules are
        wrapped with LoRA).  Use the student's ``forward_denoise_step`` for the
        student path, and call the teacher wrapper's ``forward_denoise_step``
        (which uses ``@torch.no_grad``) for the teacher path — LoRA's B=0 init
        means the teacher output is unchanged at the start.
        """
        from .ltx_teacher_wrapper import LTXTeacherWrapper  # avoid circular at module level

        obj = cls.__new__(cls)
        obj.device = teacher.device
        obj.dtype = teacher.dtype
        obj.lora_rank = lora_rank
        obj.lora_alpha = lora_alpha
        obj.lora_target_modules = lora_target_modules or DEFAULT_LORA_TARGETS
        obj._transformer = teacher.transformer  # shared reference

        # Inject LoRA (base stays frozen, only LoRA params trainable)
        obj._lora_modules = inject_lora(
            obj._transformer,
            obj.lora_target_modules,
            obj.lora_rank,
            obj.lora_alpha,
        )
        obj._transformer.train()

        n_lora = sum(p.numel() for p in obj.lora_parameters())
        n_total = sum(p.numel() for p in obj._transformer.parameters())
        logger.info(
            "[LTXStudentWrapper.from_teacher] %d LoRA modules, %d trainable params (%.2f%%)",
            len(obj._lora_modules),
            n_lora,
            100.0 * n_lora / n_total,
        )
        return obj

    # --- properties ---

    @property
    def transformer(self) -> X0Model:
        return self._transformer

    # --- parameter access ---

    def lora_parameters(self) -> Iterator[nn.Parameter]:
        """Yield only the trainable LoRA parameters."""
        for lora in self._lora_modules.values():
            yield lora.lora_A
            yield lora.lora_B

    def lora_state_dict(self) -> dict[str, torch.Tensor]:
        """Export LoRA weights only."""
        sd = {}
        for name, lora in self._lora_modules.items():
            sd[f"{name}.lora_A"] = lora.lora_A.data.clone()
            sd[f"{name}.lora_B"] = lora.lora_B.data.clone()
        return sd

    def load_lora_state_dict(self, sd: dict[str, torch.Tensor]) -> None:
        """Load LoRA weights from a state dict."""
        for name, lora in self._lora_modules.items():
            lora.lora_A.data.copy_(sd[f"{name}.lora_A"])
            lora.lora_B.data.copy_(sd[f"{name}.lora_B"])

    # --- frozen check ---

    def base_is_frozen(self) -> bool:
        """Return True if all non-LoRA parameters are frozen."""
        lora_ids = {id(p) for p in self.lora_parameters()}
        return all(not p.requires_grad for p in self._transformer.parameters() if id(p) not in lora_ids)

    def lora_is_trainable(self) -> bool:
        """Return True if all LoRA parameters require grad."""
        return all(p.requires_grad for p in self.lora_parameters())

    # --- state construction (same logic as teacher) ---

    def create_noised_video_state(
        self,
        stage2_latent: torch.Tensor,
        *,
        noise_scale: float | None = None,
        seed: int = 0,
        fps: float = 24.0,
    ) -> tuple[LatentState, VideoLatentTools]:
        """Patchified noised state from a raw stage-2 latent [B,C,F,H,W]."""
        from diffusion.refiner.vendor.ltx_core.components.noisers import GaussianNoiser
        from diffusion.refiner.vendor.ltx_core.tools import VideoLatentTools
        from diffusion.refiner.vendor.ltx_pipelines.utils.constants import STAGE_2_DISTILLED_SIGMA_VALUES

        B, C, F, H, W = stage2_latent.shape
        tools = VideoLatentTools(
            patchifier=VideoLatentPatchifier(patch_size=1),
            target_shape=VideoLatentShape(batch=B, channels=C, frames=F, height=H, width=W),
            fps=fps,
        )
        state = tools.create_initial_state(
            device=self.device,
            dtype=self.dtype,
            initial_latent=stage2_latent.to(device=self.device, dtype=self.dtype),
        )
        default_scale = float(STAGE_2_DISTILLED_SIGMA_VALUES[0])
        scale = default_scale if noise_scale is None else float(noise_scale)
        noiser = GaussianNoiser(torch.Generator(device=self.device).manual_seed(seed))
        return noiser(state, scale), tools

    # --- forward ---

    def forward_denoise(
        self,
        *,
        video_state: LatentState,
        v_context: torch.Tensor,
        sigma: torch.Tensor | float,
        prompt_sigma: torch.Tensor | float | None = None,
        apply_causal: bool = True,
        chunk_sizes: list[int] | None = None,
        attention_mode: str = "causal",
        window_radius: int | None = None,
    ) -> torch.Tensor:
        """Student forward pass: predict denoised x0 at the given sigma.

        This is the fundamental forward method. ``forward_denoise_step``
        is a convenience wrapper that indexes into a sigma schedule.

        Parameters
        ----------
        sigma : Tensor or float
            Per-token noise level for the patchified ``denoise_mask``. Scalar
            or shape ``(B,)``. ``timesteps_from_mask`` multiplies the mask by
            this value to yield per-token timesteps.
        prompt_sigma : Tensor, float, or None
            Optional separate scalar sigma routed through ``Modality.sigma``
            to drive the prompt-AdalN path (``X0Model`` reads it once for
            the cross-attn modulation). Defaults to ``sigma``. Pass the
            mean active target-frame sigma when ``denoise_mask`` encodes a
            mixed (progressive) schedule — otherwise the prompt sees the
            global maximum and over-conditions clean tokens.
        apply_causal : bool
            If True, apply causal / sliding-window attention mask.
        chunk_sizes, attention_mode, window_radius
            Causal mask configuration (see ``apply_causal_mask``).

        Returns denoised video prediction [B, T, D].  Gradients flow
        through the LoRA parameters only.
        """
        if apply_causal:
            video_state = apply_causal_mask(
                video_state,
                chunk_sizes=chunk_sizes,
                attention_mode=attention_mode,
                window_radius=window_radius,
            )

        if not torch.is_tensor(sigma):
            sigma = torch.tensor(sigma, device=video_state.latent.device)
        video_mod = modality_from_latent_state(video_state, v_context, sigma)
        if prompt_sigma is not None:
            from dataclasses import replace as _dc_replace

            if not torch.is_tensor(prompt_sigma):
                prompt_sigma = torch.tensor(prompt_sigma, device=video_mod.latent.device, dtype=torch.float32)
            prompt_sigma = prompt_sigma.to(device=video_mod.latent.device, dtype=torch.float32)
            if prompt_sigma.dim() == 0:
                prompt_sigma = prompt_sigma.expand(video_mod.latent.shape[0])
            if prompt_sigma.ndim != 1 or prompt_sigma.shape[0] != video_mod.latent.shape[0]:
                raise ValueError(
                    "prompt_sigma must be scalar or shape [B], "
                    f"got {tuple(prompt_sigma.shape)} for B={video_mod.latent.shape[0]}"
                )
            video_mod = _dc_replace(video_mod, sigma=prompt_sigma)
        denoised_video, _ = self._transformer(video=video_mod, audio=None, perturbations=None)
        return denoised_video

    def forward_denoise_step(
        self,
        *,
        video_state: LatentState,
        v_context: torch.Tensor,
        sigmas: torch.Tensor,
        step_index: int,
        apply_causal: bool = True,
        chunk_sizes: list[int] | None = None,
        attention_mode: str = "causal",
        window_radius: int | None = None,
    ) -> torch.Tensor:
        """Convenience wrapper: index ``sigmas[step_index]`` then call ``forward_denoise``."""
        return self.forward_denoise(
            video_state=video_state,
            v_context=v_context,
            sigma=sigmas[step_index],
            apply_causal=apply_causal,
            chunk_sizes=chunk_sizes,
            attention_mode=attention_mode,
            window_radius=window_radius,
        )

    # --- lifecycle ---

    def close(self) -> None:
        if hasattr(self, "_transformer"):
            self._transformer.to("meta")
            del self._transformer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
