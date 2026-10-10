# SPDX-License-Identifier: Apache-2.0
"""Expose MiniMax-H3 as an x0 model while adapting its timestep convention.

H3 predicts data-ward velocity ``v = x0 - eps``, the exact opposite of the
repo's noise-ward ``v_lerp = eps - x0``. Passing raw H3 velocity to
``convert_from_pred`` silently gives an x0 wrong by ``2 * t * v``; converting
inside this wrapper makes that sign error unrepresentable.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import torch

from ..scheduler import minimax_h3_rf_v_to_x0
from .causal_model import CausalMiniMaxH3DiTModel
from .causal_model_sp import CausalMiniMaxH3DiTModelSP
from .model import MiniMaxH3DiTModel
from .model_sp import MiniMaxH3DiTModelSP

# Checkpoint fidelity: keyframe conditioning is attested at 0.999; do not tune.
MINIMAX_H3_VIDEO_CLEAN_TIMESTEP = 0.999
# Text natively inherits the checkpoint's video timestep.
MINIMAX_H3_TEXT_CLEAN_TIMESTEP = 0.999
# Our choice: tune this first if a causal student's predicted audio context drifts.
MINIMAX_H3_AUDIO_CLEAN_TIMESTEP = 0.999


def minimax_h3_repo_to_h3_timesteps(
    repo_timesteps: torch.Tensor,
    token_tags: torch.Tensor,
    clean_timesteps_by_tag: Sequence[float] | None = None,
) -> torch.Tensor:
    """Map row noise levels to H3 anchors, including clean rows and padding."""
    timesteps = repo_timesteps.float()
    anchors = timesteps.new_tensor(
        clean_timesteps_by_tag
        if clean_timesteps_by_tag is not None
        else (
            MINIMAX_H3_VIDEO_CLEAN_TIMESTEP,
            MINIMAX_H3_TEXT_CLEAN_TIMESTEP,
            MINIMAX_H3_AUDIO_CLEAN_TIMESTEP,
        )
    )
    return torch.where(
        timesteps == 0,
        anchors[token_tags.clamp(min=0)],
        1.0 - timesteps,
    )


def build_nested_module(entry: Any, class_map: Mapping[str, type]) -> Any:
    """Instantiate a ``_class_name``-tagged config tree against ``class_map``.

    A copy of ltx2's helper rather than an import of it: projects are
    self-contained, so a second family reaching into the first would be the
    repo's only cross-family dependency.
    """
    if entry is None or isinstance(entry, torch.nn.Module):
        return entry
    if isinstance(entry, list):
        return [build_nested_module(item, class_map) for item in entry]
    if isinstance(entry, tuple):
        return tuple(build_nested_module(item, class_map) for item in entry)
    if not isinstance(entry, Mapping):
        return entry

    if "_class_name" not in entry:
        return {key: build_nested_module(value, class_map) for key, value in entry.items()}

    class_name = entry["_class_name"]
    if class_name not in class_map:
        raise KeyError(f"Unknown nested module class: {class_name}")

    config = entry.get("_config")
    if config is None:
        config = {key: value for key, value in entry.items() if key != "_class_name"}
    elif isinstance(config, Mapping):
        config = dict(config)
    else:
        raise TypeError(f"Nested config for {class_name} must be a mapping, got {type(config).__name__}")

    config = {key: build_nested_module(value, class_map) for key, value in config.items()}
    return class_map[class_name](**config)


class MiniMaxH3X0Model(torch.nn.Module):
    """Expose an H3 DiT through the repo's timestep and x0 contract."""

    def __init__(
        self,
        dit: torch.nn.Module | dict[str, Any],
        *,
        timestep_conditioning_scale: float = 1.0,
    ) -> None:
        super().__init__()
        self.timestep_conditioning_scale = float(timestep_conditioning_scale)
        if (
            not math.isfinite(self.timestep_conditioning_scale)
            or self.timestep_conditioning_scale <= 0
        ):
            raise ValueError("timestep_conditioning_scale must be finite and positive")
        self.dit = build_nested_module(
            dit,
            {
                "MiniMaxH3DiTModel": MiniMaxH3DiTModel,
                "MiniMaxH3DiTModelSP": MiniMaxH3DiTModelSP,
                "CausalMiniMaxH3DiTModel": CausalMiniMaxH3DiTModel,
                "CausalMiniMaxH3DiTModelSP": CausalMiniMaxH3DiTModelSP,
            },
        )
        # ``(noisy, condition)`` widths of a video row when the DiT takes clean
        # condition channels, else None. A DiT without ``arch`` takes none.
        arch = getattr(self.dit, "arch", None)
        self.video_row_split = None
        if arch is not None and arch.video_condition_channels:
            patch_volume = math.prod(arch.patch_size)
            self.video_row_split = (
                arch.latents_dim * patch_volume,
                arch.video_condition_channels * patch_volume,
            )

    def _conditioning_timesteps(
        self,
        repo_timesteps: torch.Tensor,
        token_tags: torch.Tensor,
        clean_timesteps_by_tag: Sequence[float] | None = None,
        timestep_conditioning_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Map scaled repo noise levels to H3's opposite time direction.

        The optional row mask preserves reference and history conditioning.
        Without one, every nonzero repo timestep is scaled. Clean repo-zero
        anchors are unchanged in both cases.
        """
        timesteps = repo_timesteps.float()
        h3_timesteps = minimax_h3_repo_to_h3_timesteps(
            timesteps, token_tags, clean_timesteps_by_tag
        )
        if self.timestep_conditioning_scale == 1.0:
            return h3_timesteps
        selected = timesteps != 0
        if timestep_conditioning_mask is not None:
            if timestep_conditioning_mask.dtype != torch.bool:
                raise TypeError("timestep_conditioning_mask must be a bool tensor")
            if timestep_conditioning_mask.shape != timesteps.shape:
                raise ValueError("timestep_conditioning_mask must match row timesteps")
            selected = selected & timestep_conditioning_mask
        return torch.where(
            selected,
            1.0 - timesteps * self.timestep_conditioning_scale,
            h3_timesteps,
        )

    # No load_state_dict/state_dict overrides, deliberately. This module's state
    # dict is ``dit.*``, exactly as the tree says, and the released checkpoint's
    # unprefixed keys are reconciled where that difference belongs -- in
    # ``MiniMaxH3WeightLoader``, which is the only thing that reads that file.
    #
    # Delegating either one is a trap. Delegating only load_state_dict left the
    # weight loader's missing/unexpected report calling all 535 keys both missing
    # AND unexpected, so it could not tell a complete load from an empty one.
    # Delegating both made state_dict report unprefixed names while the modules
    # were still nested, and DCP -- whose get_model_state_dict calls
    # ``model.state_dict()`` rather than walking the tree itself, contrary to
    # what a comment here once claimed -- then failed to resolve them:
    # ``AttributeError: 'MiniMaxH3CausalX0DiT' object has no attribute
    # 'video_patch_proj'``, on the first checkpoint save, 50 steps in.

    def set_gradient_checkpointing(
        self,
        enabled: bool = True,
        gc_start_idx: int = 0,
        gc_step: int = 1,
    ) -> None:
        """Mirror the wrapped DiT's signature exactly, defaults included.

        ``NativeGradientCheckpointing`` forwards the YAML plugin config as
        ``**kwargs``, so the parameter NAME and the presence of a default are
        both part of the contract: H3's DiT takes ``enabled=True``, which lets a
        bare plugin entry work. Naming it ``enable`` here -- ltx2's spelling,
        whose trials pass ``enable: true`` explicitly -- made a bare entry a
        TypeError raised only at launch.
        """
        self.dit.set_gradient_checkpointing(
            enabled,
            gc_start_idx=gc_start_idx,
            gc_step=gc_step,
        )

    def forward(
        self,
        *,
        x: torch.Tensor,
        audio_x: torch.Tensor,
        unique_timesteps: torch.Tensor,
        inverse_indices: torch.Tensor,
        token_tags: torch.Tensor,
        eps: torch.Tensor,
        audio_eps: torch.Tensor,
        timestep_conditioning_mask: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Translate repo timesteps, noise clean context, and return x0.

        With condition channels, each ``x`` row is ``[noisy | condition]``.
        Only the noisy part is context-noised (with the matching leading
        columns of ``eps``) and converted to x0; the condition part reaches the
        DiT unchanged. ``timestep_conditioning_mask`` selects rows whose
        network time embeddings are scaled; latent noising and x0 conversion
        always use the original noise levels.
        """
        condition = None
        if self.video_row_split is not None:
            assert x.shape[-1] == sum(self.video_row_split), (
                f"video rows are {x.shape[-1]} wide; the DiT takes "
                f"{self.video_row_split[0]} noisy + {self.video_row_split[1]} "
                "condition columns"
            )
            x, condition = x.split(self.video_row_split, dim=-1)
            eps = eps[..., : self.video_row_split[0]]
        repo_t = unique_timesteps.float()[inverse_indices]
        clean_timesteps_by_tag = kwargs.pop("clean_timesteps_by_tag", None)
        h3_t = minimax_h3_repo_to_h3_timesteps(
            repo_t, token_tags, clean_timesteps_by_tag
        )
        conditioning_t = h3_t
        if self.timestep_conditioning_scale != 1.0:
            conditioning_t = self._conditioning_timesteps(
                repo_t, token_tags, clean_timesteps_by_tag, timestep_conditioning_mask
            )
        h3_unique, h3_inverse = torch.unique(conditioning_t, return_inverse=True)

        # Mask on repo t, not H3 t: a noisy sigma=0.001 row also maps to 0.999
        # and must not be noised a second time.
        context = repo_t == 0
        t = h3_t.view(1, -1, 1)
        context = context.view(1, -1, 1)
        # This is LinearInterpolationSchedule.forward at sigma=1-t_h3, not a
        # separate H3-specific noising rule.
        x = torch.where(context, t * x + (1 - t) * eps, x)
        audio_x = torch.where(
            context,
            t * audio_x + (1 - t) * audio_eps,
            audio_x,
        )
        # Both where calls write every row, but _embed reads x only at img_pos
        # and audio_x only at audio_pos, so untouched-modality rows are never read.
        v_video, v_audio = self.dit(
            x=x if condition is None else torch.cat((x, condition), dim=-1),
            audio_x=audio_x,
            unique_timesteps=h3_unique,
            inverse_indices=h3_inverse,
            token_tags=token_tags,
            **kwargs,
        )
        if kwargs.get("classify_mode"):
            # The DiT returns (taps, t_emb) here, not a velocity pair. Pass it
            # straight through: MiniMaxH3GANX0DiTSP is what turns it into logits,
            # and it is the only caller that sets classify_mode.
            return v_video, v_audio

        video_pos = MiniMaxH3DiTModel._pos_ids(
            kwargs["img_pos_for_infer_output_info"],
            "img_pos_for_infer_output_info",
        )
        audio_pos = MiniMaxH3DiTModel._pos_ids(
            kwargs["audio_pos_info"],
            "audio_pos_info",
        )
        video_xt = x[0].index_select(0, video_pos.to(x.device))
        audio_xt = audio_x[0].index_select(0, audio_pos.to(audio_x.device))
        video_h3_t = h3_t.index_select(0, video_pos.to(h3_t.device))
        audio_h3_t = h3_t.index_select(0, audio_pos.to(h3_t.device))

        # The scheduler helper applies x0 = x_t + (1 - t_h3) * v_h3.
        return (
            minimax_h3_rf_v_to_x0(video_xt, v_video, video_h3_t),
            minimax_h3_rf_v_to_x0(audio_xt, v_audio, audio_h3_t),
        )


EntryClass = MiniMaxH3X0Model

__all__ = [
    "MINIMAX_H3_AUDIO_CLEAN_TIMESTEP",
    "minimax_h3_repo_to_h3_timesteps",
    "MINIMAX_H3_TEXT_CLEAN_TIMESTEP",
    "MINIMAX_H3_VIDEO_CLEAN_TIMESTEP",
    "MiniMaxH3X0Model",
]
