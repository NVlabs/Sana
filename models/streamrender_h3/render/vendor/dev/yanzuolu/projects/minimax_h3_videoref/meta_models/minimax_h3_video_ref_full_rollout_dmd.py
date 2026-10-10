# SPDX-License-Identifier: Apache-2.0
"""DMD over independently sampled windows of a complete fresh student rollout.

Each dataloader emission is a complete latent pack from a non-chain dataset
with fixed crop shapes and streaming policy, such as
``VideoRefStreamingLongLatentT2AVDataset``. The current student generates
every window before either optimizer runs. The resulting AV clip is stored
once, beside each window's detached noisy-row trajectory and predictions.
FAKE and GEN then independently sample windows without replacement per clip.
Their ``meta_model.inner_batch`` entries each specify ``size`` and
``include_first``. The latter reserves slot zero for the bootstrap and samples
the remaining slots from continuations. Each slot packs one window per clip.

The engine backpropagates each slot immediately, divides by the number of
slots, and keeps its FAKE-before-GEN optimizer barrier. GEN replays its sampled
student stage against that updated fake model. Both ODE and renoising use the
ordinary streaming DMD score-point primitives. ``engine.offline`` must be 1
so an optimizer step never consumes an earlier step's complete trajectory.
Gradient accumulation may prepare several complete rollouts before the step.

The full rollout has no persistent chain or mid-clip checkpoint. Checkpoints
at the engine's drained-pool boundaries retain the ordinary dataset state.
The chain-only ``meta_model.history_refresh`` option is not supported here.
``meta_model.fake_history_refresh`` reconstructs history with the fake model
for each selected FAKE inner batch. ``meta_model.score_history_refresh`` uses
the real teacher for each selected GEN inner batch, and for FAKE when fake
refresh is off. These phase-local passes leave the student unchanged.

Configuration additions::

    data:
      module: dev.yanzuolu.projects.minimax_h3_videoref.data.video_ref_streaming_long_latent
      class_name: VideoRefStreamingLongLatentT2AVDataset
    meta_model:
      module: dev.yanzuolu.projects.minimax_h3_videoref.meta_models.minimax_h3_video_ref_full_rollout_dmd
      class_name: MiniMaxH3VideoRefFullRolloutDMD
      inner_batch:
        fake: {size: 4, include_first: true}
        gen: {size: 2, include_first: false}
    engine:
      offline: 1
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Any

import torch

from dev.yanzuolu.common.distributed.ops import get_device, get_rank
from dev.yanzuolu.common.distributed.unified_parallel import SPDistForward
from dev.yanzuolu.common.phase import ExecutionPhase, execution_phase
from dev.yanzuolu.projects.minimax_h3.data.streaming import streaming_window_starts
from dev.yanzuolu.projects.minimax_h3.meta_models.minimax_h3_streaming_dmd import (
    MiniMaxH3StreamingDMD,
    StreamingDMDRollout,
)
from dev.yanzuolu.projects.minimax_h3.modeling.streaming import StreamingInputs
from dev.yanzuolu.projects.minimax_h3_videoref.meta_models.minimax_h3_video_ref_streaming_dmd import (
    DMDHistoryRefreshInputs,
    MiniMaxH3VideoRefStreamingDMD,
    VideoRefStreamingDMDInputs,
)
from dev.yanzuolu.projects.minimax_h3_videoref.meta_models.minimax_h3_video_ref_streaming_rf import _ChainState

_Pair = tuple[list[torch.Tensor], list[torch.Tensor]]


@dataclass(frozen=True)
class FullRolloutInputs:
    """One shared reference/geometry payload and the conditioning of every window."""

    shared: dict[str, Any]
    windows: tuple[dict[str, Any], ...]

    @property
    def batch_size(self) -> int:
        return len(self.shared["latent_shapes"])


@dataclass(frozen=True)
class FullRolloutCache:
    """One canonical AV clip, with only noisy-row values stored at each stage."""

    source: FullRolloutInputs
    canonical: _Pair
    rollouts: tuple[StreamingDMDRollout, ...]
    trajectories: tuple[list[_Pair], ...]


class MiniMaxH3VideoRefFullRolloutDMD(MiniMaxH3VideoRefStreamingDMD):
    """Generate complete clips, then train independently sampled FAKE and GEN windows."""

    checkpoint_chains = False

    def __init__(self, config: Any) -> None:
        if config.meta_model.get("history_refresh", False):
            raise ValueError("full-rollout DMD does not use chain history_refresh")
        if int(config.get("engine", {}).get("offline", 1)) != 1:
            raise ValueError("full-rollout DMD requires engine.offline 1")
        if "resume_mid_chain" in config.data.args:
            raise ValueError("full-rollout DMD uses a non-chain dataset; remove data.args.resume_mid_chain")
        super().__init__(config)
        options = config.meta_model.get("inner_batch", {})
        if set(options) - {"fake", "gen"}:
            raise ValueError("meta_model.inner_batch accepts fake and gen")
        self.inner_batch: dict[str, tuple[int, bool]] = {}
        for phase in ("fake", "gen"):
            item = options.get(phase, {})
            if set(item) - {"size", "include_first"}:
                raise ValueError(f"meta_model.inner_batch.{phase} accepts size and include_first")
            size, include_first = item.get("size", 1), item.get("include_first", False)
            if not isinstance(size, int) or isinstance(size, bool) or size < 1:
                raise ValueError(f"meta_model.inner_batch.{phase}.size must be a positive integer")
            if not isinstance(include_first, bool):
                raise ValueError(f"meta_model.inner_batch.{phase}.include_first must be a boolean")
            self.inner_batch[phase] = size, include_first

    @execution_phase(ExecutionPhase.PREPARE)
    @torch.no_grad()
    def prepare_inputs(self, ctx: dict[str, Any]) -> dict[str, Any]:
        """Encode a complete pack, retaining no chain state between engine iterations."""
        batch = ctx["batch"]
        if any(key in batch for key in ("chain_id", "chain_index", "chain_length", "chain_windows")):
            raise ValueError("full-rollout DMD requires complete packs from a non-chain latent dataset")
        counts = {
            len(streaming_window_starts(shape[1], **self._streaming_policy(policy),
                                       video_temporal_mapping=self.video_temporal_mapping))
            for shape, policy in zip(batch["latent_shapes"], batch["streaming_config"], strict=True)
        }
        if len(counts) != 1:
            raise ValueError("packed clips must have the same complete-rollout window count")
        count = counts.pop()
        for phase, (size, _) in self.inner_batch.items():
            if size > count:
                raise ValueError(f"inner_batch.{phase}.size {size} exceeds the clip's {count} windows")
        head = dict(batch, chain_id=f"full:{get_rank()}:{ctx['iter']}", chain_index=0,
                    chain_length=count, chain_windows=count)
        view = dict(ctx, batch=head)
        entry = self._start_chain(view)
        try:
            windows = []
            for window in range(count):
                plans, conditioning = self._chain_window_conditioning(view, entry, window)
                windows.append(dict(conditioning, preselected_plans=plans, chain_window=window,
                                    **self._chain_window_payload(entry, window)))
            shared = dict(
                chain_id=entry.chain_id, seed=entry.seed, reference_latents=entry.reference,
                latent_shapes=entry.latent_shapes, audio_shapes=entry.audio_shapes,
                streaming_config=entry.streaming_config, packing_rows=entry.packing_rows,
            )
            ctx["inputs"] = FullRolloutInputs(self._to_device(shared), tuple(self._to_device(windows)))
            ctx["neg_inputs"] = None
        finally:
            del self._chains[entry.chain_id]
        return ctx

    @torch.no_grad()
    def sync_inputs(self, ctx: dict[str, Any]) -> Iterator[dict[str, Any]]:
        """Broadcast each source's complete payload once, with no repeated full AV copies."""
        source = ctx["inputs"]
        sync = SPDistForward(name="video_ref_full_rollout_dmd_inputs", comm_shape=True, device=get_device())
        for payload in sync({"shared": source.shared, "windows": source.windows}):
            yield dict(ctx, inputs=FullRolloutInputs(payload["shared"], tuple(payload["windows"])), neg_inputs=None)

    def _full_rollout_window(
        self, source: FullRolloutInputs, canonical: _Pair, window: int, *, chain: _ChainState | None = None,
    ) -> VideoRefStreamingDMDInputs:
        payload = {**source.shared, **source.windows[window]}
        return self._dmd_window(dict(encoded_batch=payload, clean_latents=canonical, chain=chain),
                                prepare_score_history=chain is None)

    @execution_phase(ExecutionPhase.ROLLOUT)
    @torch.no_grad()
    def rollout(self, ctx: dict[str, Any]) -> dict[str, Any]:
        """Generate every window with fixed student weights and cache detached chunk trajectories."""
        source = ctx["inputs"]
        shared, count = source.shared, len(source.windows)
        canonical = tuple([torch.zeros(tuple(shape), dtype=torch.float32, device=get_device()) for shape in shapes]
                          for shapes in (shared["latent_shapes"], shared["audio_shapes"]))
        entry = _ChainState(
            chain_id=shared["chain_id"], chain_length=count, video=canonical[0], audio=canonical[1],
            reference=shared["reference_latents"],
            plans=[[window["preselected_plans"][index] for window in source.windows] for index in range(source.batch_size)],
            latent_shapes=shared["latent_shapes"], audio_shapes=shared["audio_shapes"],
            streaming_config=shared["streaming_config"], packing_rows=shared["packing_rows"], seed=shared["seed"],
        )
        self._chains[entry.chain_id] = entry
        backbone = ctx["models"]["backbone"]
        was_training = backbone.training
        backbone.eval()
        rollouts, trajectories = [], []
        try:
            for window in range(count):
                inputs = self._full_rollout_window(source, canonical, window, chain=entry)
                view = dict(ctx, inputs=inputs, rng=ctx["rng"].fork("full_rollout_window", window))
                view = MiniMaxH3StreamingDMD.rollout(self, view)
                rollouts.append(view["rollout_x0s"])
                trajectories.append(view["trajectory_xts"])
        finally:
            backbone.train(was_training)
            self._chains.pop(entry.chain_id, None)
        ctx["rollout_x0s"] = FullRolloutCache(source, canonical, tuple(rollouts), tuple(trajectories))
        ctx["trajectory_xts"] = None
        return ctx

    def _pack_selected_windows(self, windows: Sequence[StreamingInputs]) -> StreamingInputs:
        """Take sample i from selected window i and build one ordinary packed forward."""
        return self._streaming_inputs_from_payload(dict(
            plans=[window.plans[index] for index, window in enumerate(windows)],
            prompt_embeds=[window.prompt_embeds[index] for index, window in enumerate(windows)],
            reference_latents=(None if windows[0].reference_latents is None
                               else [window.reference_latents[index] for index, window in enumerate(windows)]),
            text_token_tags=(None if windows[0].text_token_tags is None
                             else [window.text_token_tags[index] for index, window in enumerate(windows)]),
        ))

    def _pack_selected_inputs(self, windows: Sequence[VideoRefStreamingDMDInputs]) -> VideoRefStreamingDMDInputs:
        """Preserve each clip's student/score layouts, conditioning and original window history."""
        first = windows[0]
        return VideoRefStreamingDMDInputs(
            chain_id=first.chain_id, window=tuple(window.window for window in windows),
            commit=tuple(window.commit[index] for index, window in enumerate(windows)),
            self_history=tuple(window.self_history[index] for index, window in enumerate(windows)),
            window_inputs=self._pack_selected_windows([window.window_inputs for window in windows]),
            negative_window_inputs=(None if first.negative_window_inputs is None else self._pack_selected_windows(
                [window.negative_window_inputs for window in windows])),
            history=tuple([window.history[modality][index] for index, window in enumerate(windows)] for modality in (0, 1)),
            score_window=(None if first.score_window is None else self._pack_selected_inputs(
                [window.score_window for window in windows])),
            branch_windows=(None if first.branch_windows is None else {
                name: self._pack_selected_windows([window.branch_windows[name] for window in windows])
                for name in first.branch_windows
            }),
            guidance_available=(None if first.guidance_available is None else tuple(
                window.guidance_available[index] for index, window in enumerate(windows))),
            score_history_refresh=(None if first.score_history_refresh is None else DMDHistoryRefreshInputs(
                window=self._pack_selected_inputs([window.score_history_refresh.window for window in windows]),
                clean_x0s=tuple([window.score_history_refresh.clean_x0s[modality][index]
                                for index, window in enumerate(windows)] for modality in (0, 1)),
                timesteps=tuple(torch.stack([window.score_history_refresh.timesteps[modality][index]
                                             for index, window in enumerate(windows)]) for modality in (0, 1)),
            )),
        )

    def _selected_window_context(self, ctx: dict[str, Any], indices: tuple[int, ...]) -> dict[str, Any]:
        cache = ctx["rollout_x0s"]
        windows = {window: self._full_rollout_window(cache.source, cache.canonical, window) for window in sorted(set(indices))}
        inputs = self._pack_selected_inputs([windows[window] for window in indices])
        selected = [cache.rollouts[window] for window in indices]
        stages = len(selected[0].stage_video)
        rollout = StreamingDMDRollout(
            video=[value.video[index] for index, value in enumerate(selected)],
            audio=[value.audio[index] for index, value in enumerate(selected)],
            stage_video=[[value.stage_video[stage][index] for index, value in enumerate(selected)] for stage in range(stages)],
            stage_audio=[[value.stage_audio[stage][index] for index, value in enumerate(selected)] for stage in range(stages)],
            video_grid=torch.stack([value.video_grid[index] for index, value in enumerate(selected)]),
            audio_grid=torch.stack([value.audio_grid[index] for index, value in enumerate(selected)]),
        )
        trajectory = [tuple([cache.trajectories[window][stage][modality][index]
                             for index, window in enumerate(indices)] for modality in (0, 1)) for stage in range(stages)]
        return dict(ctx, inputs=inputs, neg_inputs=None, rollout_x0s=rollout, trajectory_xts=trajectory)

    def inner_batches(self, ctx: dict[str, Any], *, phase: str) -> tuple[int, Iterator[dict[str, Any]]]:
        """Select each clip independently, then materialize one detached inner batch at a time."""
        size, include_first = self.inner_batch[phase]
        cache, rng = ctx["rollout_x0s"], ctx["rng"]
        count = len(cache.rollouts)
        selections = []
        for sample in range(cache.source.batch_size):
            draw = rng.fork("window_selection", phase, sample).python_generator
            selections.append(([0] + draw.sample(range(1, count), size - 1)) if include_first
                              else draw.sample(range(count), size))

        def batches() -> Iterator[dict[str, Any]]:
            for slot in range(size):
                indices = tuple(selection[slot] for selection in selections)
                sub_ctx = self._selected_window_context(ctx, indices)
                sub_ctx["rng"] = rng.fork("inner_batch", phase, slot)
                yield sub_ctx
                del sub_ctx

        return size, batches()


EntryClass = MiniMaxH3VideoRefFullRolloutDMD

__all__ = ["MiniMaxH3VideoRefFullRolloutDMD", "FullRolloutInputs", "FullRolloutCache", "EntryClass"]
