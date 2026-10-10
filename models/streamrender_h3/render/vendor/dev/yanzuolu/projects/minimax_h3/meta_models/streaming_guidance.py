# SPDX-License-Identifier: Apache-2.0
"""Conditioning branches for streaming supervision."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
import math
from typing import Any

import torch

from dev.yanzuolu.projects.minimax_h3.modeling.streaming import StreamingInputs


@dataclass(frozen=True)
class StreamingGuidanceBranch:
    """One reduced-conditioning prediction and its independent supervision."""

    name: str
    drop: frozenset[str]
    loss_weight: float = 0.0


@dataclass(frozen=True)
class StreamingGuidanceStage:
    """Guidance on the edge from one named state to the next state."""

    branch: str
    scale: float


@dataclass(frozen=True)
class CompiledStreamingGuidance:
    """Per-sample coefficients and auxiliary weights after equivalent states merge.

    Every fit reads ``(P - sum(c_i * stopgrad(N_i))) / scale`` for its own
    positive prediction P. ``branch_scales`` and ``branch_coefficients`` give
    each branch's supervision fit, the identity outside guided anchors.
    """

    full_scale: float
    coefficients: tuple[float, ...]
    loss_weights: tuple[float, ...]
    branch_scales: tuple[float, ...]
    branch_coefficients: tuple[tuple[float, ...], ...]

    @property
    def used(self) -> tuple[bool, ...]:
        """Branches whose prediction enters a fit or carries supervision."""
        return tuple(
            coefficient != 0.0 or weight > 0.0 or any(fit[index] != 0.0 for fit in self.branch_coefficients)
            for index, (coefficient, weight) in enumerate(zip(self.coefficients, self.loss_weights, strict=True))
        )


GUIDANCE_ANCHORS = ("posterior", "guided")
GUIDANCE_CONDITIONS = ("text", "reference", "history")


def parse_guidance_branches(
    options: Mapping[str, Any], conditions: Sequence[str] = GUIDANCE_CONDITIONS,
) -> tuple[StreamingGuidanceBranch, ...] | None:
    """Validate explicit branches while leaving the legacy configuration intact.

    ``conditions`` are the names a branch may drop, the host's conditioning.
    """
    configured = options.get("guidance_branches")
    if configured is None:
        return None
    if not isinstance(configured, (list, tuple)):
        raise ValueError("guidance_branches must be a list of conditioning branches")
    if (float(options.get("guidance_scale", 1.0)) != 1.0
            or float(options.get("negative_loss_weight", 0.0)) != 0.0
            or not bool(options.get("keep_negative_reference", True))):
        raise ValueError("guidance_branches cannot be mixed with nondefault legacy fitting options")
    branches, names = [], set()
    for value in configured:
        if set(value) - {"name", "drop", "loss_weight"}:
            raise ValueError("guidance branches accept name, drop and loss_weight; scales belong to guidance_fitting.stages")
        name, drop = value["name"], value["drop"]
        if not isinstance(name, str) or not name or name in names:
            raise ValueError("guidance branch names must be nonempty and unique")
        if not isinstance(drop, (list, tuple)) or not drop or not set(drop) <= set(conditions):
            raise ValueError(f"guidance branch drop must select from {', '.join(conditions)}")
        weight = float(value.get("loss_weight", 0.0))
        if not math.isfinite(weight) or weight < 0.0:
            raise ValueError("guidance branch loss_weight must be finite and nonnegative")
        branches.append(StreamingGuidanceBranch(name, frozenset(drop), weight))
        names.add(name)
    return tuple(branches)


def parse_guidance_fitting(
    options: Mapping[str, Any], branches: tuple[StreamingGuidanceBranch, ...] | None,
) -> tuple[tuple[StreamingGuidanceStage, ...], str]:
    """Read a sequential path of increasingly complete conditioning states and its anchor reading."""
    configured = options.get("guidance_fitting")
    if configured is None:
        return (), "posterior"
    if branches is None:
        raise ValueError("guidance_fitting requires guidance_branches")
    if (not isinstance(configured, Mapping) or "stages" not in configured
            or set(configured) - {"stages", "anchors"}):
        raise ValueError("guidance_fitting requires a stages list and accepts only anchors besides it")
    anchors = configured.get("anchors", "posterior")
    if anchors not in GUIDANCE_ANCHORS:
        raise ValueError("guidance_fitting.anchors must be posterior or guided")
    values = configured["stages"]
    if not isinstance(values, (list, tuple)):
        raise ValueError("guidance_fitting.stages must be a list")
    definitions = {branch.name: branch for branch in branches}
    stages, previous = [], None
    for value in values:
        if set(value) != {"branch", "scale"}:
            raise ValueError("each guidance fitting stage requires branch and scale")
        name, scale = value["branch"], float(value["scale"])
        if name not in definitions:
            raise ValueError(f"unknown guidance fitting branch {name!r}")
        dropped = definitions[name].drop
        if previous is not None and not dropped < previous:
            raise ValueError("guidance fitting stages must strictly add conditions by reducing drop sets")
        if not math.isfinite(scale) or scale < 1.0:
            raise ValueError("guidance fitting scale must be finite and at least one")
        stages.append(StreamingGuidanceStage(name, scale))
        previous = dropped
    chain = {definitions[stage.branch].drop for stage in stages}
    if anchors == "guided" and any(branch.loss_weight > 0.0 and branch.drop not in chain for branch in branches):
        raise ValueError("guided anchors fit each supervised branch along its chain, so it must lie on guidance_fitting.stages")
    return tuple(stages), anchors


def _guided_fit(
    states: Sequence[frozenset[str]], scales: Sequence[float],
) -> tuple[float, dict[frozenset[str], float]]:
    """Recover the posterior mean at the last state from outputs guided along the chain up to each state.

    With ``u_0 = f_0`` and ``u_k = u_{k-1} + (f_k - f_{k-1}) / s_k``, the last
    mean is ``(f_last - sum(c_j * f_j)) / s_last``.
    """
    if not scales:
        return 1.0, {}
    last, bounds = scales[-1], [1.0, *scales]
    return last, {
        state: last / following - last / current
        for state, current, following in zip(states[:-1], bounds[:-1], bounds[1:], strict=True)
    }


def compile_guidance(
    branches: Sequence[StreamingGuidanceBranch], stages: Sequence[StreamingGuidanceStage],
    available: frozenset[str], *, anchors: str = "posterior",
) -> CompiledStreamingGuidance:
    """Compile a telescoping guidance path and merge equivalent auxiliary states.

    The full state is implicit. Edges whose endpoint conditions coincide do
    nothing, and states equivalent to full conditioning receive no auxiliary
    supervision. The first configured branch with each effective drop set
    supplies the shared prediction for that state.

    ``posterior`` anchors read every reduced prediction as its posterior mean
    and supervise branches on their raw outputs. ``guided`` anchors read each
    state's output as guided along the chain truncated at that state. The
    full fit and each supervised branch's fit then recover the posterior mean
    of their own state from the lower states on the per-sample chain.
    """
    effective = [branch.drop & available for branch in branches]
    owners, definitions = {}, {}
    for index, (branch, dropped) in enumerate(zip(branches, effective, strict=True)):
        definitions[branch.name] = dropped
        if dropped:
            owners.setdefault(dropped, index)
    empty = frozenset()
    path = [definitions[stage.branch] for stage in stages] + [empty]

    def branch_values(values: Mapping[frozenset[str], float]) -> tuple[float, ...]:
        merged = [0.0] * len(branches)
        for dropped, coefficient in values.items():
            if dropped:
                merged[owners[dropped]] += coefficient
        return tuple(merged)

    loss_weights = [0.0] * len(branches)
    for branch, dropped in zip(branches, effective, strict=True):
        if dropped:
            loss_weights[owners[dropped]] += branch.loss_weight
    branch_scales, branch_coefficients = [1.0] * len(branches), [(0.0,) * len(branches)] * len(branches)
    if anchors == "guided":
        states, scales = [path[0]], []
        for stage, following in zip(stages, path[1:], strict=True):
            if following != states[-1]:
                states.append(following)
                scales.append(stage.scale)
        full_scale, state_coefficients = _guided_fit(states, scales)
        for index, dropped in enumerate(effective):
            if loss_weights[index] > 0.0:
                position = states.index(dropped)
                branch_scales[index], fit = _guided_fit(states[:position + 1], scales[:position])
                branch_coefficients[index] = branch_values(fit)
    else:
        state_coefficients = {path[0]: 1.0}
        for stage, current, following in zip(stages, path[:-1], path[1:], strict=True):
            if current != following:
                state_coefficients[current] = state_coefficients.get(current, 0.0) - stage.scale
                state_coefficients[following] = state_coefficients.get(following, 0.0) + stage.scale
        full_scale = state_coefficients[empty]
    return CompiledStreamingGuidance(
        full_scale, branch_values(state_coefficients), tuple(loss_weights),
        tuple(branch_scales), tuple(branch_coefficients),
    )


def possible_guidance_uses(
    branches: Sequence[StreamingGuidanceBranch], stages: Sequence[StreamingGuidanceStage],
    *, anchors: str = "posterior", conditions: Sequence[str] = GUIDANCE_CONDITIONS,
) -> tuple[bool, ...]:
    """Find prediction states needed for any availability of ``conditions``."""
    used = [False] * len(branches)
    for bits in range(1 << len(conditions)):
        available = frozenset(name for index, name in enumerate(conditions) if bits & (1 << index))
        compiled = compile_guidance(branches, stages, available, anchors=anchors)
        used = [before or now for before, now in zip(used, compiled.used, strict=True)]
    return tuple(used)


def without_target_history(
    inputs: StreamingInputs,
    video_xts: Sequence[torch.Tensor],
    audio_xts: Sequence[torch.Tensor],
) -> tuple[StreamingInputs, tuple[list[torch.Tensor], list[torch.Tensor]]]:
    """Remove clean target AV rows while preserving every surviving RoPE coordinate.

    Reference rows and the conditioning prefix are retained. The noisy selectors
    address the same global targets, including the newly generated audio tail.
    """
    plans, packs, videos, audios = [], [], [], []
    for plan, pack, video, audio in zip(inputs.plans, inputs.packs, video_xts, audio_xts, strict=True):
        video_mask, audio_mask = plan["video_noisy_mask"], plan["audio_noisy_mask"]
        video_keep, audio_keep = video_mask.nonzero().flatten(), audio_mask.nonzero().flatten()
        frame_rows = (plan["target_video_shape"][2] // 2) * (plan["target_video_shape"][3] // 2)
        prefix_rows = plan["text_len"] + pack["reference_rows"]
        device = pack["position_ids"].device
        keep = torch.cat((
            torch.ones(prefix_rows, dtype=torch.bool, device=device),
            audio_mask.to(device).repeat(2),
            video_mask.to(device).repeat_interleave(frame_rows),
        ))
        rows = keep.nonzero().flatten()
        row_map = torch.full((pack["sample_lens"],), -1, dtype=torch.long, device=device)
        row_map[rows] = torch.arange(rows.numel(), device=device)
        new_plan = dict(plan)
        for prefix, mask in (("video", video_mask), ("audio", audio_mask)):
            new_plan[f"{prefix}_indices"] = plan[f"{prefix}_indices"][mask]
            new_plan[f"{prefix}_noisy_mask"] = torch.ones_like(mask[mask])
        for field in ("audio_retained_mask", "audio_commit_mask", "audio_lookahead_mask"):
            new_plan[field] = plan[field][audio_mask]
        new_plan["audio_commit_indices"] = new_plan["audio_indices"][new_plan["audio_commit_mask"]]
        new_plan["packing_rows"] = plan["packing_rows"] - int((~keep).sum())
        new_pack = dict(pack)
        for field in ("position_ids", "token_tags"):
            new_pack[field] = pack[field].index_select(0, rows)
        for field in ("text_pos", "img_pos", "audio_pos", "noisy_img_pos"):
            old_positions = pack[field]
            new_pack[field] = row_map[old_positions[keep[old_positions]]]
        new_pack.update(
            sample_lens=rows.numel(),
            video_rows=video_keep.numel() * frame_rows,
            audio_rows=2 * audio_keep.numel(),
            video_noisy_mask=new_plan["video_noisy_mask"].to(device),
            audio_noisy_mask=new_plan["audio_noisy_mask"].to(device),
            audio_noisy_sel=torch.arange(2 * audio_keep.numel(), device=device),
        )
        plans.append(new_plan)
        packs.append(new_pack)
        videos.append(video.index_select(1, video_keep.to(video.device)))
        audios.append(audio.index_select(2, audio_keep.to(audio.device)))
    return replace(
        inputs, plans=plans, packs=packs,
        token_tags=torch.cat([pack["token_tags"] for pack in packs]),
        seqlens=inputs.seqlens.new_tensor([pack["sample_lens"] for pack in packs]),
    ), (videos, audios)


def noised_target_history(
    inputs: StreamingInputs,
    video_xts: Sequence[torch.Tensor],
    audio_xts: Sequence[torch.Tensor],
    noises: tuple[Sequence[torch.Tensor], Sequence[torch.Tensor]],
    *,
    level: float = 1.0,
) -> tuple[StreamingInputs, tuple[list[torch.Tensor], list[torch.Tensor]]]:
    """Noise clean target AV rows to ``level`` in place of removing them.

    Every row, its RoPE coordinate and the packed layout stay as they are.
    History rows read ``(1 - level) * x + level * noise`` at noise level
    ``level``, so 1.0 masks them with complete noise. ``noises`` holds one
    tensor per sample and modality shaped like its values. Noisy target rows,
    the reference rows and the conditioning prefix are unchanged, and a window
    without history returns its inputs as they are.
    """
    plans, videos, audios = [], [], []
    for plan, video, audio, video_noise, audio_noise in zip(
        inputs.plans, video_xts, audio_xts, *noises, strict=True,
    ):
        video_history = ~plan["video_noisy_mask"].to(video.device)
        audio_history = ~plan["audio_noisy_mask"].to(audio.device)
        if not (video_history.any() or audio_history.any()):
            plans.append(plan)
            videos.append(video)
            audios.append(audio)
            continue
        plans.append(dict(plan, history_timestep=float(level)))
        videos.append(torch.where(video_history.view(1, -1, 1, 1),
                                  (1 - level) * video + level * video_noise.to(video), video))
        audios.append(torch.where(audio_history.view(1, 1, -1),
                                  (1 - level) * audio + level * audio_noise.to(audio), audio))
    return replace(inputs, plans=plans), (videos, audios)


__all__ = [
    "StreamingGuidanceBranch", "StreamingGuidanceStage", "CompiledStreamingGuidance", "GUIDANCE_ANCHORS",
    "GUIDANCE_CONDITIONS",
    "parse_guidance_branches", "parse_guidance_fitting", "compile_guidance", "possible_guidance_uses",
    "without_target_history", "noised_target_history",
]
