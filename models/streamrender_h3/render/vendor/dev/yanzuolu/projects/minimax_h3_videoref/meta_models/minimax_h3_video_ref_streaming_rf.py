# SPDX-License-Identifier: Apache-2.0
"""Resampling forcing for the bidirectional streaming window model.

``MiniMaxH3VideoRefStreamingSFT`` is teacher forcing: every iteration draws one
window of a clip, the sink and recent history rows are corpus latents, and the
current chunk is noised and regressed to the corpus. When the model streams,
the history is its own output, so the history distribution seen in training
never matches the one seen at inference. The default ``resample_mode: chain``
uses resampling forcing without touching the target: history rows come from the model's one-step x0
reconstruction of the earlier windows, and the chunk is still regressed to
the corpus latents with the parent's unchanged loss and fitting.

A companion dataset turns each pack of clips into a chain of consecutive
engine iterations, one per streaming window in order. The head batch is the
usual collated pack plus ``chain_id``, ``chain_index`` (0 on a head unless
it resumes a chain from a checkpoint), ``chain_length`` (how many leading
windows the chain trains) and ``chain_windows`` (the clip's window count).
Every later batch is light: only those four scalars. Per iteration this meta
trains exactly window ``chain_index`` of every clip in the chain against a
per-rank pool entry that persists across iterations in process memory:

* float32 device working copies of the video and audio latents, initialised
  to the corpus at the head and overwritten chunk by chunk with the model's
  reconstruction;
* the reference latents, every window's plan, one resampling timestep pair
  ``t_s`` per clip, the packing budgets, and the conditioning the host needs
  (per-window Qwen presentations built once from the reference pixels, or the
  prompt token ids when ``qwen_visual_context`` is off). A clip with a picture
  also keeps that picture's clean latent, encoded once at the head with the
  unsharded ``video_vae``, and every window's presentation starts with it.

In chain mode each iteration runs two DiT forwards over the same window. The resample pass
(``eval()`` + ``no_grad``) noises the chunk at ``t_s`` with its own forked RNGs
and predicts x0. The training pass is the parent's forward with the
captionless negative and fitting. The reconstruction is written back only
after the training pass, so nothing consumed in this iteration -- targets,
noisy inputs, reference perturbation -- observes the new values, and the next
window reads them as history. The last window's resample forward still runs
and is discarded so that ranks sitting at different chain phases issue the
same collectives every iteration: one text-encoder call on the current
window's presentations or on the prompt token ids, one no-grad resample
forward, one training forward, the configured negative forward, one backward.

Working copies live on the rank that loaded the chain. With a training UP
size above 1 every rank of an SP group still owns its own chains, and the
group trains one source rank's window per iteration from the SP input
broadcast, which carries that window's working copies, plans, conditioning and
``t_s``. The prepare seed is shared within a group, so a chain head on any
source rank but the first forks it by that rank. Every group rank computes the
same full reconstruction, and only the source writes it back.
``bootstrap_probability`` is rejected because every window of a chain is
trained exactly once, and ``selective_video_encoding`` because the chain
consumes the latent corpus only.

Checkpoints keep the unfinished chains. The meta registers itself as a
persistence plugin, and before every checkpoint save each rank writes
``resampling_chains/rank_<rank>.pt`` into the checkpoint directory. Per
chain it holds the chain's progress, the prepare seed of its head and the
working copies. The meta sets ``data.args.resume_mid_chain``, so a resumed
dataset delivers each interrupted chain's pack again as a head at its
``chain_index``. The entry is rebuilt from that head with the saved seed,
which reproduces the plans, ``t_s``, pictures and conditioning, and continues
on the saved working copies. A resumed head without saved state, because the
checkpoint has none or was written for another world size, train UP size or
dataloader worker count, restarts its chain at window 0 for the remaining
windows and logs a warning.

``resample_mode: history_only`` keeps the chain's corpus latents immutable.
At each training forward the current backbone jointly reconstructs the needed
sink and recent AV history from noised GT. The current target AV rows never
enter that resample forward. Picture, text and the history's semantic
conditions remain, with the window's RoPE template preserved. The detached
prediction replaces history only in this step's training inputs, leaving the
already prepared current targets and target noise unchanged. This is local history
reconstruction, not a replay of sequential generation. Every sample executes
one resample forward even when its bootstrap has no history. Chain progress
is saved as usual, but its corpus remains GT; resuming a checkpoint made in
chain mode reloads the head's GT instead of using its saved generated values.

Expected configuration additions::

    meta_model:
      module: dev.yanzuolu.projects.minimax_h3_videoref.meta_models.minimax_h3_video_ref_streaming_rf
      class_name: MiniMaxH3VideoRefStreamingRF
      resample_mode: chain
      resample_timesteps:
        module: dev.yanzuolu.projects.minimax_h3.modeling.training_timesteps
        class_name: PairedLogitNormalTrainingTimesteps
        args: {T: 1.0, loc: 0.0, scale: 1.0, shift: 0.6, audio_shift: 0.6}
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch.distributed.fsdp import FSDPModule

from dev.yanzuolu.common.diffusion import build_diffusion
from dev.yanzuolu.common.distributed.ops import get_device, get_rank, get_world_size
from dev.yanzuolu.common.distributed.unified_parallel import (
    get_unified_parallel_rank,
    get_unified_parallel_world_size,
    is_unified_parallel_initialized,
)
from dev.yanzuolu.common.logging import get_logger
from dev.yanzuolu.common.phase import ExecutionPhase, execution_phase
from dev.yanzuolu.common.seed import RandomState, combine_seed, local_seed
from dev.yanzuolu.projects.minimax_h3.data.streaming import build_history_resample_plan, streaming_window_starts
from dev.yanzuolu.projects.minimax_h3.meta_models.minimax_h3_streaming_sft import StreamingBatch
from dev.yanzuolu.projects.minimax_h3.modeling.time_request import VideoTemporalMapping
from dev.yanzuolu.projects.minimax_h3.modeling.transformer.x0_model import MINIMAX_H3_VIDEO_CLEAN_TIMESTEP
from dev.yanzuolu.projects.minimax_h3_videoref.meta_models.minimax_h3_video_ref_streaming_sft import (
    MiniMaxH3VideoRefStreamingSFT,
)
from dev.yanzuolu.projects.minimax_h3_videoref.modeling.ref2va_encoder import (
    Ref2VAPresentation,
    encode_ref2va_presentations,
    text_only_ref2va_presentation,
)
from dev.yanzuolu.projects.minimax_h3_videoref.modeling.ref2va_reference import encode_reference_image
from dev.yanzuolu.projects.minimax_h3_videoref.modeling.streaming_qwen import build_streaming_ref_presentation

logger = get_logger()


def _build_resample_timesteps(options: Any) -> Any:
    """Build the paired timestep sampler shared by clip-head resampling hosts."""
    node = options.get("resample_timesteps")
    if node is None:
        raise ValueError("resampling forcing requires meta_model.resample_timesteps")
    if (node.get("args") or {}).get("shift_schedule") is not None:
        raise ValueError("resample_timesteps draws one pair per clip and does not support shift_schedule")
    sampler = build_diffusion({"resample_timesteps": node})["resample_timesteps"]
    if not callable(getattr(sampler, "sample_pair", None)):
        raise ValueError("resample_timesteps must provide sample_pair for the video and audio timestep pair")
    return sampler


def _sample_chain_resample_timesteps(
    sampler: Any, entry: _ChainState, batch: dict[str, Any], rng: Any,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Draw one paired time per clip from its original chain-head seed."""
    device, video_times, audio_times = get_device(), [], []
    for index in range(entry.batch_size):
        with local_seed(combine_seed(rng.seed, "resample_timesteps", index) % 2**31):
            video_t, audio_t = sampler.sample_pair(
                (1,), torch.tensor([int(batch["packing_rows"][index])], device=device), device,
            )
        video_times.append(video_t.float().reshape(()))
        audio_times.append(audio_t.float().reshape(()))
    return torch.stack(video_times), torch.stack(audio_times)


@dataclass
class _ChainState:
    """One pack's chain as this rank sees it between iterations.

    ``video`` and ``audio`` are the float32 working copies the windows read
    history from and write reconstructions into. ``plans[index][window]`` is
    fixed at the head for every window the chain trains, and ``conditioning``
    is whatever the host prepared there to condition those windows. A
    distillation host whose teacher reads its own window policy keeps the
    teacher's plans for the same starts in ``teacher_plans``, and a host whose
    discriminator reads the clip's corpus keeps the head's latents, which no
    write-back touches, in ``corpus``. ``seed`` is the head's prepare seed.
    ``offset`` is the dataset ``chain_index`` of the
    entry's window 0, nonzero only when a resumed head restarted its chain.
    ``resample_timesteps`` holds the optional resampling ``t_s`` pair per clip.
    """

    chain_id: str
    chain_length: int
    video: list[torch.Tensor]
    audio: list[torch.Tensor]
    reference: list[torch.Tensor] | None
    plans: list[list[dict[str, Any]]]
    latent_shapes: list[Any]
    audio_shapes: list[Any]
    streaming_config: list[Any]
    packing_rows: list[int]
    seed: int
    resample_timesteps: tuple[torch.Tensor, torch.Tensor] | None = None
    conditioning: Any = None
    teacher_plans: list[list[dict[str, Any]]] | None = None
    corpus: tuple[list[torch.Tensor], list[torch.Tensor]] | None = None
    next_index: int = 0
    offset: int = 0

    @property
    def batch_size(self) -> int:
        return len(self.video)

    def window_plans(self, window: int) -> list[dict[str, Any]]:
        return [windows[window] for windows in self.plans]


class StreamingChainPoolMixin:
    """Keep a per-rank pool of streaming window chains whose working copies survive checkpoints.

    Hosts implement three hooks. ``_chain_head_inputs`` returns the pack's
    window geometry and optional reference latents, ``_chain_head_conditioning``
    prepares what conditions every window of the chain, and
    ``_chain_window_conditioning`` returns one window's conditioned plans and
    payload keys. Training algorithms add their per-clip head draws through
    ``_chain_head_draws`` and their per-window payload keys through
    ``_chain_window_payload``. The pool, the plans, the SP broadcast, the
    write-back and the checkpoints do not depend on how windows are
    conditioned or trained.
    """

    _chains: dict[str, _ChainState]
    checkpoint_chains: bool = True

    def __init__(self, config: Any) -> None:
        super().__init__(config)
        options = config.meta_model
        if "bootstrap_probability" in options:
            raise ValueError("chains train every window exactly once; remove meta_model.bootstrap_probability")
        if self.selective_video_encoding:
            raise ValueError("chains consume the latent corpus and cannot use selective_video_encoding")
        if self.checkpoint_chains:
            # The meta, not the trial, decides how the dataset resumes a chain.
            if not config.data.args.get("resume_mid_chain", True):
                raise ValueError("the chain pool checkpoints its chains, so data.args.resume_mid_chain must stay true")
            config.data.args.resume_mid_chain = True
        self._chain_topology = {
            "world_size": get_world_size(),
            "up_size": int(config.get("engine", {}).get("up_size", {}).get("train", 1)),
            "num_workers": int(config.data.get("num_workers", 0)),
        }
        self._chains = {}
        self._saved_chains: dict[str, dict[str, Any]] = {}
        plugins = config.get("persistence_plugins")
        if self.checkpoint_chains and plugins is not None:
            plugins.plugins.append(self)

    # ------------------------------------------------------------- hooks --
    def _chain_head_inputs(self, batch: dict[str, Any]) -> tuple[StreamingBatch, list[torch.Tensor] | None]:
        """Return the pack's window geometry and its optional reference latents."""
        raise NotImplementedError

    def _chain_head_conditioning(self, ctx: dict[str, Any], entry: _ChainState) -> Any:
        """Prepare what the host needs to condition every window of the chain."""
        raise NotImplementedError

    def _chain_window_conditioning(
        self, ctx: dict[str, Any], entry: _ChainState, window: int,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """Return one window's conditioned plans and the payload keys conditioning them."""
        raise NotImplementedError

    def _chain_head_draws(self, entry: _ChainState, batch: dict[str, Any], rng: Any) -> None:
        """Draw the training algorithm's per-clip values from the head's prepare stream."""

    def _chain_window_payload(self, entry: _ChainState, window: int) -> dict[str, Any]:
        """Payload keys the training algorithm adds to every window."""
        return {}

    # -------------------------------------------------------------- pool --
    @staticmethod
    def _chain_fields(batch: dict[str, Any]) -> tuple[str, int, int, int]:
        return str(batch["chain_id"]), int(batch["chain_index"]), int(batch["chain_length"]), int(batch["chain_windows"])

    def _start_chain(self, ctx: dict[str, Any]) -> _ChainState:
        """Plan every window of a new chain and take working copies of its corpus latents.

        A head at a later ``chain_index`` resumes a chain from a checkpoint.
        With saved state the entry is rebuilt from the saved head seed and
        continues on the saved working copies. Without it the chain restarts
        at window 0 for its remaining windows.
        """
        chain_id, chain_index, chain_length, chain_windows = self._chain_fields(ctx["batch"])
        assert 0 <= chain_index < chain_length <= chain_windows
        saved = self._saved_chains.pop(chain_id, None) if chain_index else None
        if saved is not None:
            ctx = dict(ctx, rng=RandomState(saved["seed"]))
        elif chain_index:
            logger.warning("Chain %s resumes at window %d without saved state: restarting it at window 0 for %d windows",
                           chain_id, chain_index, chain_length - chain_index)
        offset = chain_index if saved is None else saved["offset"]
        seed = ctx["rng"].seed
        source = get_unified_parallel_rank() if is_unified_parallel_initialized() else 0
        if source:
            ctx = dict(ctx, rng=ctx["rng"].fork("chain_source", source))
        batch, models, rng = ctx["batch"], ctx["models"], ctx["rng"]
        self._check_codec(models["video_vae"])
        for descriptor in batch["video_temporal_mapping"]:
            if VideoTemporalMapping.from_dict(descriptor) != self.video_temporal_mapping:
                raise ValueError("dataset temporal mapping disagrees with video_vae")
        geometry, reference = self._chain_head_inputs(batch)
        device = get_device()
        video = [value.to(device=device, dtype=torch.float32, copy=True) for value in batch["video_latents"]]
        audio = [value.to(device=device, dtype=torch.float32, copy=True) for value in batch["audio_latents"]]
        for values, shapes in ((video, geometry.video_shapes), (audio, geometry.audio_shapes)):
            assert all(tuple(value.shape) == tuple(shape) for value, shape in zip(values, shapes, strict=True))
        plans = []
        for index in range(geometry.batch_size):
            starts = streaming_window_starts(
                geometry.video_shapes[index][1], **self._streaming_policy(geometry.streaming_configs[index]),
                video_temporal_mapping=self.video_temporal_mapping,
            )
            assert len(starts) == chain_windows, "chain_windows disagrees with the clip's streaming windows"
            plans.append([self._plan(geometry, index, start) for start in starts[:chain_length - offset]])
        entry = _ChainState(
            chain_id=chain_id, chain_length=chain_length - offset, video=video, audio=audio,
            reference=None if reference is None else [value.to(device) for value in reference], plans=plans,
            latent_shapes=list(batch["latent_shapes"]), audio_shapes=list(batch["audio_shapes"]),
            streaming_config=list(batch["streaming_config"]), packing_rows=[int(rows) for rows in batch["packing_rows"]],
            seed=seed, offset=offset,
        )
        # Drawn from the head's prepare seed, which a chain resumed from its saved state reuses.
        self._chain_head_draws(entry, batch, rng)
        entry.conditioning = self._chain_head_conditioning(ctx, entry)
        if saved is not None:
            assert saved["chain_length"] == entry.chain_length and saved["next_index"] == chain_index - offset
            for values, copies in ((entry.video, saved["video"]), (entry.audio, saved["audio"])):
                for index, value in enumerate(copies):
                    assert value.shape == values[index].shape
                    values[index] = value.to(device=device, dtype=torch.float32, copy=True)
            entry.next_index = saved["next_index"]
        self._chains[chain_id] = entry
        return entry

    def _continue_chain(self, ctx: dict[str, Any]) -> _ChainState:
        chain_id, chain_index, chain_length, _ = self._chain_fields(ctx["batch"])
        entry = self._chains.get(chain_id)
        if entry is None:
            raise RuntimeError(f"chain {chain_id!r} has no pool entry: the dataloader and the resampling pool desynchronised")
        assert chain_index - entry.offset == entry.next_index and chain_length - entry.offset == entry.chain_length
        return entry

    # ------------------------------------------------------- persistence --
    @staticmethod
    def _chain_state_path(checkpoint_dir: str | Path) -> Path:
        return Path(checkpoint_dir) / "resampling_chains" / f"rank_{get_rank()}.pt"

    def before_checkpoint_saved(self, *, checkpoint_dir: str | Path, **_: Any) -> None:
        """Write this rank's unfinished chains next to the checkpoint.

        Saves run after the last training pass of an iteration, so every
        chain has written back its latest window.
        """
        chains = {
            chain_id: {
                "offset": entry.offset, "chain_length": entry.chain_length, "next_index": entry.next_index,
                "seed": entry.seed, "video": [value.cpu() for value in entry.video],
                "audio": [value.cpu() for value in entry.audio],
            }
            for chain_id, entry in self._chains.items()
        }
        path = self._chain_state_path(checkpoint_dir)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"topology": self._chain_topology, "chains": chains}, path)

    def after_checkpoint_loaded(self, *, checkpoint_dir: str | Path, **_: Any) -> None:
        """Keep this rank's saved chains until their resumed heads arrive.

        The file is memory-mapped, so chains whose heads never come back,
        for example after a dataloader reset, cost no memory.
        """
        self._saved_chains = {}
        path = self._chain_state_path(checkpoint_dir)
        if not path.exists():
            logger.warning("No resampling chain state at %s: resumed chains restart at window 0", path)
            return
        state = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
        if state["topology"] != self._chain_topology:
            logger.warning("Resampling chain state at %s was saved for %s, not %s: resumed chains restart at window 0",
                           path, state["topology"], self._chain_topology)
            return
        self._saved_chains = state["chains"]
        logger.info("Loaded %d resampling chains from %s", len(self._saved_chains), path)

    @execution_phase(ExecutionPhase.PREPARE)
    @torch.no_grad()
    def prepare_inputs(self, ctx: dict[str, Any]) -> dict[str, Any]:
        """Reads batch, models, rng. Writes encoded_batch, inputs, clean_latents, chain, chain_window.

        ``clean_latents`` are the pool's working copies themselves, so the
        history rows ``add_noise`` selects are the reconstructions written
        back by the earlier windows of this chain.
        """
        entry = self._start_chain(ctx) if "video_latents" in ctx["batch"] else self._continue_chain(ctx)
        window = entry.next_index
        plans, conditioning = self._chain_window_conditioning(ctx, entry, window)
        payload = {
            **conditioning,
            "reference_latents": entry.reference, "video_latents": entry.video, "audio_latents": entry.audio,
            "latent_shapes": entry.latent_shapes, "audio_shapes": entry.audio_shapes,
            "streaming_config": entry.streaming_config, "packing_rows": entry.packing_rows,
            "preselected_plans": plans, "chain_window": window, **self._chain_window_payload(entry, window),
        }
        ctx["encoded_batch"] = self._to_device(payload)
        ctx["inputs"] = self._inputs_from_payload(ctx["encoded_batch"])
        ctx["clean_latents"] = (entry.video, entry.audio)
        ctx["chain"], ctx["chain_window"] = entry, window
        return ctx

    def sync_inputs(self, ctx: dict[str, Any]) -> Iterator[dict[str, Any]]:
        """Train each source rank's window in turn. Only the source keeps ``chain`` and writes back."""
        if not is_unified_parallel_initialized() or get_unified_parallel_world_size() <= 1:
            yield from super().sync_inputs(ctx)
            return
        rank = get_unified_parallel_rank()
        # The SP input broadcast visits the group's source ranks in order from 0.
        for source, sub_ctx in enumerate(super().sync_inputs(ctx)):
            yield dict(sub_ctx, chain=ctx["chain"] if source == rank else None,
                       chain_window=sub_ctx["encoded_batch"]["chain_window"])

    def _prepared_streaming_plans(self, ctx: dict[str, Any]) -> list[dict[str, Any]]:
        """Every window of a chain is fixed at its head, so no bootstrap draw happens."""
        return [{key: value.cpu() if isinstance(value, torch.Tensor) else value for key, value in plan.items()}
                for plan in ctx["encoded_batch"]["preselected_plans"]]

    @staticmethod
    def _chunk_rows(plan: dict[str, Any], *, audio: bool) -> torch.Tensor:
        prefix = "audio" if audio else "video"
        return plan[f"{prefix}_indices"][plan[f"{prefix}_noisy_mask"]]

    def _write_back(
        self, entry: _ChainState, plans: list[dict[str, Any]], values: tuple[list[torch.Tensor], list[torch.Tensor]],
    ) -> None:
        """Overwrite the current chunk in every clip's working copies and advance the chain."""
        for index, plan in enumerate(plans):
            video, audio = entry.video[index], entry.audio[index]
            video.index_copy_(1, self._chunk_rows(plan, audio=False).to(video.device), values[0][index].to(video.dtype))
            audio.index_copy_(2, self._chunk_rows(plan, audio=True).to(audio.device), values[1][index].to(audio.dtype))
        entry.next_index += 1
        if entry.next_index == entry.chain_length:
            del self._chains[entry.chain_id]


class StreamingResamplingChainMixin(StreamingChainPoolMixin):
    """Train with chain write-back or current-weight local history reconstruction.

    Every clip draws one ``t_s`` pair from ``resample_timesteps`` at its head.
    The resample pass, the write-back and the metrics do not depend on how
    windows are conditioned.
    """

    def __init__(self, config: Any) -> None:
        self.resample_mode = str(config.meta_model.get("resample_mode", "chain"))
        if self.resample_mode not in {"chain", "history_only"}:
            raise ValueError("meta_model.resample_mode must be chain or history_only")
        if self.resample_mode == "history_only":
            if config.data.args.get("reference_only", False):
                raise ValueError("history_only resampling requires ground-truth target latents")
        super().__init__(config)
        self.resample_timesteps = _build_resample_timesteps(config.meta_model)

    def _start_chain(self, ctx: dict[str, Any]) -> _ChainState:
        entry = super()._start_chain(ctx)
        if self.resample_mode == "history_only":
            # A resumed chain may have been saved with generated working copies.
            # Its head supplies the original corpus again, while progress and
            # the original head seed continue to come from the saved entry.
            entry.video = [value.to(device=get_device(), dtype=torch.float32, copy=True)
                           for value in ctx["batch"]["video_latents"]]
            entry.audio = [value.to(device=get_device(), dtype=torch.float32, copy=True)
                           for value in ctx["batch"]["audio_latents"]]
        return entry

    def _chain_head_draws(self, entry: _ChainState, batch: dict[str, Any], rng: Any) -> None:
        """One ``t_s`` pair per clip for the whole chain."""
        super()._chain_head_draws(entry, batch, rng)
        entry.resample_timesteps = _sample_chain_resample_timesteps(self.resample_timesteps, entry, batch, rng)

    def _chain_window_payload(self, entry: _ChainState, window: int) -> dict[str, Any]:
        return {**super()._chain_window_payload(entry, window), "resample_timesteps": entry.resample_timesteps}

    # ---------------------------------------------------------- resample --

    @torch.no_grad()
    def _prepare_resampling(self, ctx: dict[str, Any]) -> dict[str, Any]:
        """Prepare chain write-back or rebuild this step's complete history from GT."""
        if self.resample_mode == "chain":
            prediction = self._resample_window(ctx)
            ctx["resample_prediction"] = prediction
            ctx["chain_metrics"] = self._resample_metrics(ctx, prediction)
            return ctx

        ctx.update(self._resample_history(ctx, ctx["plans"], model_name="backbone", role="student"))
        ctx["noisy_latents"] = self._history_conditioned_values(ctx, ctx["clean_latents"], role="student")
        return ctx

    @torch.no_grad()
    def _resample_history(
        self, ctx: dict[str, Any], window_plans: list[dict[str, Any]], *,
        model_name: str, role: str, teacher_conditions: bool = False,
    ) -> dict[str, Any]:
        """Reconstruct one role's history from GT without changing the caller's inputs."""
        plans = [build_history_resample_plan(plan) for plan in window_plans]
        inputs = ctx["inputs"]
        if teacher_conditions:
            inputs = replace(inputs, reference_latents=self._teacher_references(ctx))
        embeddings = list(inputs.prompt_embeds)
        tags = None if inputs.text_token_tags is None else list(inputs.text_token_tags)
        for index, plan in enumerate(plans):
            has_reference = (plan["reference_video_shape"] is not None
                             and plan["reference_video_indices"].numel() > 0)
            if (plan["video_indices"].numel() == 0 and plan["audio_indices"].numel() == 0
                    and embeddings[index].shape[0] == 0 and not has_reference and plan.get("picture_shape") is None):
                # An empty bootstrap still executes one collective-compatible
                # forward. This constant prefix carries no target information.
                embeddings[index] = embeddings[index].new_zeros((1, embeddings[index].shape[1]))
                plans[index] = dict(plan, text_len=1, packing_rows=plan["packing_rows"] + 1)
                if tags is not None:
                    tags[index] = tags[index].new_ones(1)
        view = dict(ctx, inputs=replace(inputs, prompt_embeds=embeddings, text_token_tags=tags), plans=plans,
                    train_timesteps=ctx["encoded_batch"]["resample_timesteps"],
                    sample_rngs=[ctx["rng"].fork("history_resample", role, index) for index in range(len(plans))])
        # A Ref2VA teacher must not receive a pixel/source student's input
        # channels. Its own plans already carry its reference and picture modes.
        view = (MiniMaxH3VideoRefStreamingSFT.add_noise(self, view) if teacher_conditions else self.add_noise(view))
        backbone = ctx["models"][model_name]
        was_training = backbone.training
        backbone.eval()
        try:
            prediction = self._streaming_forward(
                backbone, view["window_inputs"], video_xts=view["noisy_latents"][0], audio_xts=view["noisy_latents"][1],
                video_timesteps=view["train_timesteps"][0], audio_timesteps=view["train_timesteps"][1],
            )
        finally:
            backbone.train(was_training)
        rebuilt = ([], [])
        for modality, is_audio in enumerate((False, True)):
            prefix, dim = ("audio", 2) if is_audio else ("video", 1)
            for corpus, generated, plan in zip(
                ctx["clean_latents"][modality], prediction[modality], plans, strict=True,
            ):
                rebuilt[modality].append(corpus.index_copy(
                    dim, plan[f"{prefix}_indices"].to(corpus.device), generated.detach().to(corpus)
                ))
        return dict(clean_latents=rebuilt, resample_prediction=prediction,
                    chain_metrics=self._resample_metrics(view, prediction))

    @torch.no_grad()
    def _history_conditioned_values(
        self, ctx: dict[str, Any], clean: tuple[list[torch.Tensor], list[torch.Tensor]], *, role: str,
        values: tuple[list[torch.Tensor], list[torch.Tensor]] | None = None,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """Replace history only, preserving every already prepared noisy target row."""
        values = ctx["noisy_latents"] if values is None else values
        noisy = ([], [])
        for modality, is_audio in enumerate((False, True)):
            prefix, dim = ("audio", 2) if is_audio else ("video", 1)
            for index, (reconstructed, current, xt) in enumerate(zip(
                clean[modality], ctx["plans"], values[modality], strict=True,
            )):
                history = ~current[f"{prefix}_noisy_mask"]
                selected = self._select(reconstructed, current[f"{prefix}_indices"][history], audio=is_audio)
                if not is_audio:
                    anchor = MINIMAX_H3_VIDEO_CLEAN_TIMESTEP
                    selected = anchor * selected + (1 - anchor) * self._noise(
                        selected, ctx["rng"].fork("resampled_history_condition", role, index)
                    )
                noisy[modality].append(xt.index_copy(dim, history.nonzero().flatten().to(xt.device), selected.to(xt)))
        return noisy

    def _resample_window(self, ctx: dict[str, Any]) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """Predict x0 for the current chunk at ``t_s`` without touching the training RNGs."""
        backbone = ctx["models"]["backbone"]
        was_training = backbone.training
        backbone.eval()
        try:
            with torch.no_grad():
                view = dict(ctx)
                view["train_timesteps"] = ctx["encoded_batch"]["resample_timesteps"]
                view["sample_rngs"] = [ctx["rng"].fork("chain_resample", index) for index in range(len(ctx["plans"]))]
                view = self.add_noise(view)
                return self._streaming_forward(
                    backbone, view["window_inputs"], video_xts=view["noisy_latents"][0], audio_xts=view["noisy_latents"][1],
                    video_timesteps=view["train_timesteps"][0], audio_timesteps=view["train_timesteps"][1],
                )
        finally:
            backbone.train(was_training)

    def _resample_metrics(
        self, ctx: dict[str, Any], prediction: tuple[list[torch.Tensor], list[torch.Tensor]],
    ) -> dict[str, float]:
        """Mean resampling times and reconstruction errors against the selected corpus rows."""
        video_t, audio_t = ctx["encoded_batch"]["resample_timesteps"]
        errors: tuple[list[torch.Tensor], list[torch.Tensor]] = ([], [])
        for index, plan in enumerate(ctx["plans"]):
            for modality, working in enumerate(ctx["clean_latents"]):
                is_audio = modality == 1
                target = self._select(working[index], self._chunk_rows(plan, audio=is_audio), audio=is_audio)
                value = prediction[modality][index].float()
                errors[modality].append((value - target).square().mean() if target.numel() else value.new_zeros(()))
        return {"train/resample_video_timestep": float(video_t.mean()),
                "train/resample_audio_timestep": float(audio_t.mean()),
                "train/resample_video_error": float(torch.stack(errors[0]).mean()),
                "train/resample_audio_error": float(torch.stack(errors[1]).mean())}

    def _commit_resampled_window(
        self, ctx: dict[str, Any], prediction: tuple[list[torch.Tensor], list[torch.Tensor]],
    ) -> None:
        """Advance the source chain, writing predictions only in chain mode."""
        entry = ctx["chain"]
        if entry is None:
            return
        if self.resample_mode == "history_only":
            entry.next_index += 1
            if entry.next_index == entry.chain_length:
                del self._chains[entry.chain_id]
            return
        self._write_back(entry, ctx["plans"], prediction)

    @execution_phase(ExecutionPhase.TRAIN_FORWARD)
    def forward(self, ctx: dict[str, Any]) -> dict[str, Any]:
        """Reads the parent's forward keys plus chain, rng. Writes the parent's keys plus chain_metrics.

        Chain mode writes back after the training pass. History-only mode
        reconstructs before that pass and advances without changing the GT pool.
        """
        ctx = self._prepare_resampling(ctx)
        ctx = super().forward(ctx)
        self._commit_resampled_window(ctx, ctx["resample_prediction"])
        return ctx

    @execution_phase(ExecutionPhase.TRAIN_FORWARD)
    def compute_loss(self, ctx: dict[str, Any]) -> dict[str, Any]:
        ctx = super().compute_loss(ctx)
        ctx["metrics"].update(ctx["chain_metrics"])
        ctx["metrics"][f"train/window_loss/{ctx['chain_window']:02d}"] = float(ctx["loss"].detach())
        return ctx


@dataclass
class _VideoRefChainConditioning:
    """What a chain conditions its windows with: prompt token ids or per-window presentations.

    ``pictures`` holds each clip's clean picture latent when the pack carries pictures.
    """

    text_input_ids: list[list[torch.Tensor]] | None = None
    text_lens: list[list[int]] | None = None
    prompts: list[list[str]] | None = None
    presentations: list[list[Ref2VAPresentation]] | None = None
    text_presentations: list[list[Ref2VAPresentation]] | None = None
    pictures: list[torch.Tensor] | None = None


class VideoRefChainHostMixin:
    """Condition every chain window with the paired reference, optional picture and Qwen context."""

    def _chain_head_inputs(self, batch: dict[str, Any]) -> tuple[StreamingBatch, list[torch.Tensor] | None]:
        reference = list(batch["reference_video_latents"])
        assert all(tuple(value.shape) == tuple(shape)
                   for value, shape in zip(reference, batch["reference_latent_shapes"], strict=True))
        geometry = StreamingBatch(
            prompt_embeds=[torch.empty((0 if self.qwen_visual_context else int(length), 0)) for length in batch["text_lens"]],
            reference_latents=[torch.empty(tuple(shape), device="meta") for shape in batch["reference_latent_shapes"]],
            video_shapes=[tuple(shape) for shape in batch["latent_shapes"]],
            audio_shapes=[tuple(shape) for shape in batch["audio_shapes"]],
            streaming_configs=[self._streaming_policy(policy) for policy in batch["streaming_config"]],
        )
        return geometry, reference

    def _chain_head_conditioning(self, ctx: dict[str, Any], entry: _ChainState) -> _VideoRefChainConditioning:
        """Keep the prompt token ids, or present every window's picture and reference frames once.

        The text encoder may be FSDP-sharded, so it is called once per
        iteration in both modes rather than once per chain: ranks sit at
        different chain phases and would otherwise desynchronise.
        """
        batch = ctx["batch"]
        selected = [self._selected_caption_inputs(batch, entry.window_plans(window))
                    for window in range(entry.chain_length)]
        prompts = [[values[0][index] for values in selected] for index in range(entry.batch_size)]
        token_ids = [[values[1][index] for values in selected] for index in range(entry.batch_size)]
        text_lens = [[int(values[2][index]) for values in selected] for index in range(entry.batch_size)]
        if not self.qwen_visual_context:
            return _VideoRefChainConditioning(text_input_ids=token_ids, text_lens=text_lens, prompts=prompts)
        processor = self._qwen_processor()
        images = pictures = None
        if "picture" in batch:
            video_vae = ctx["models"]["video_vae"]
            if any(isinstance(module, FSDPModule) for module in video_vae.modules()):
                raise ValueError("chain heads differ across ranks, so picture encoding requires an unsharded video_vae")
            images = [Image.fromarray(value.numpy()) for value in batch["picture"]]
            pictures = [encode_reference_image(video_vae, image, seed=ctx["rng"].fork("picture_encode", index).seed).to(get_device())
                        for index, image in enumerate(images)]
        references = batch["reference_video_pixels"] if self.qwen_reference_video else [None] * entry.batch_size
        presentations = [
            [build_streaming_ref_presentation(processor, prompt, pixels, plan, fixed_window_rope=self.fixed_window_rope,
                                              picture=None if images is None else images[index])
             for prompt, plan in zip(captions, windows, strict=True)]
            for index, (captions, pixels, windows) in enumerate(zip(prompts, references, entry.plans, strict=True))
        ]
        text_presentations = None
        if self.needs_reference_free_text_conditioning:
            text_presentations = [[text_only_ref2va_presentation(ids) for ids in windows] for windows in token_ids]
            assert all(item.input_ids.numel() == length
                       for values, lengths in zip(text_presentations, text_lens, strict=True)
                       for item, length in zip(values, lengths, strict=True))
        return _VideoRefChainConditioning(prompts=prompts, presentations=presentations,
                                          text_presentations=text_presentations, pictures=pictures)

    def _chain_window_conditioning(
        self, ctx: dict[str, Any], entry: _ChainState, window: int,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """One encoder call on this window's presentations, with the parent's payload keys."""
        plans, state = entry.window_plans(window), entry.conditioning
        if not self.qwen_visual_context:
            embeddings = self._encode_prompts(ctx["models"], [values[window] for values in state.text_input_ids],
                                              [values[window] for values in state.text_lens])
            conditioning = {"prompt_embeds": embeddings}
            if self.needs_reference_free_text_conditioning:
                conditioning["text_only_prompt_embeds"] = embeddings
            return [self._conditioned_plan(plan, value.shape[0]) for plan, value in zip(plans, embeddings, strict=True)], conditioning
        presentations = [windows[window] for windows in state.presentations]
        text_only_embeddings = None
        if state.text_presentations is not None:
            text_presentations = [values[window] for values in state.text_presentations]
            encoded = encode_ref2va_presentations(ctx["models"]["text_encoder"], [*presentations, *text_presentations])
            embeddings, tags, negative, negative_tags = self._window_conditioning(presentations, encoded[:len(presentations)])
            text_only_embeddings = encoded[len(presentations):]
        else:
            embeddings, tags, negative, negative_tags = self._encode_window_presentations(ctx["models"], presentations)
        conditioning = dict(
            prompt_embeds=embeddings, prompts=[values[window] for values in state.prompts], text_token_tags=tags,
            negative_prompt_embeds=negative, negative_text_token_tags=negative_tags,
            media_prefix_lengths=[item.media_prefix_length for item in presentations],
        )
        if text_only_embeddings is not None:
            conditioning["text_only_prompt_embeds"] = text_only_embeddings
        plans = [self._conditioned_plan(plan, value.shape[0]) for plan, value in zip(plans, embeddings, strict=True)]
        if state.pictures is not None:
            plans = [self._with_picture(plan, picture, keyframe=self.picture_mode == "keyframe")
                     for plan, picture in zip(plans, state.pictures, strict=True)]
        return plans, conditioning


class MiniMaxH3VideoRefStreamingRF(VideoRefChainHostMixin, StreamingResamplingChainMixin, MiniMaxH3VideoRefStreamingSFT):
    """Resampling forcing over paired-reference streaming windows of the latent corpus."""


EntryClass = MiniMaxH3VideoRefStreamingRF

__all__ = ["MiniMaxH3VideoRefStreamingRF", "StreamingChainPoolMixin", "StreamingResamplingChainMixin",
           "VideoRefChainHostMixin", "EntryClass"]
