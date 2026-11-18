# SPDX-License-Identifier: Apache-2.0
"""CausalMiniMaxH3Base: the chunk-causal MiniMax-H3 mechanism, and its validation.

Everything a chunk-causal MiniMax-H3 T2AV student needs in order to *run*:
derive the packed layout from a dataset batch, pack rows for the causal ABI and
for the native bidirectional one, drive the KV-cached chunk-by-chunk rollout,
decode a clip through both VAEs and log it, and reduce a per-modality loss. No
training algorithm is attached -- those live in the subclasses (DMD in
``causal_minimax_h3_dmd.py``, and DMD2/TF/TSCD on top of it).

Not to be confused with ``projects/minimax_h3/meta_models/minimax_h3_base.py``,
which samples the *released bidirectional* checkpoint with full attention over
the whole clip in one dense forward per step. This file is the causal student's
world: chunked attention, a KV cache, clean history rows written by the
student's own x0, and a serving grid of a handful of steps.

This class is concrete, not abstract. ``common.engine.BaseEngine``'s val-only
path builds the requested models, calls ``validate`` once at step 0 and exits,
so a validation-only trial names it directly and carries no training knobs at
all::

    meta_model:
      module: dev.yanzuolu.projects.minimax_h3.meta_models.causal_minimax_h3_base
      class_name: CausalMiniMaxH3Base

The optional ``meta_model.video_chunk_renorm`` generation setting defaults to
false. No training settings are required by this base. ``config.diffusion``
supplies the four components validation needs -- ``sampling_timesteps`` and
``audio_sampling_timesteps`` (same step count, each with its own shift),
``schedule`` and ``sampler``. A training subclass adds its own nodes to that
same block and picks them out of ``self._diffusion``, which is built once here.
Each validation mapping may override ``sampling_timesteps``,
``audio_sampling_timesteps`` and ``sampler`` independently. Missing or null
nodes use the training diffusion configuration. Validation builds separate
sampling objects around the training schedule and restores the training
objects on exit, leaving their timestep grids untouched.

Validation selects ``backbone`` by default. ``validation.validate_backbone``
and ``validation.validate_ema`` independently enable the live and EMA models,
and at least one must be enabled. ``models`` must provide the selected models,
``text_encoder`` and the ``video_vae``/``audio_vae`` used for media decoding.

The layout authority is ``projects/minimax_h3/data/causal_text_only.py`` (which
ships the causal packed layout) and ``projects/minimax_h3/modeling/packing.py``
(which ships the native bidirectional layout), so this file derives, never
re-invents. Validation packs its prompts through the very dataset class the
trial trains on, for the same reason.

Four H3 facts drive everything below.

1. **Caption-only validation CFG.** Training uses one randomly dropped caption
   branch, while validation can combine positive and empty-caption predictions.

2. **The repo timestep convention never leaves this file.** ``0`` means clean,
   exactly; noisy rows carry the repo sigma. ``MiniMaxH3X0Model`` owns the
   translation to H3's inverted ``t_h3 = 1 - t_repo``, the per-modality clean
   constant (0.999) and the velocity sign. This file must therefore write a
   *literal* ``0.0`` for every context row -- the wrapper's mask is an exact
   equality test -- and must pass full-shape ``eps``/``audio_eps`` so the
   wrapper can apply the attested clean augmentation ``0.999*x0 + 0.001*eps``.

3. **Video and audio share a step index but not a timestep.** H3's reference
   sampler advances one grid index at a time with two shifts (video 12, audio 3),
   so rollout drives ``sampling_timesteps`` and ``audio_sampling_timesteps`` in
   lockstep and asserts they have the same length.

4. **Audio is 0.4% of the packed elements.** At the 192-frame profile video
   contributes 5,253,120 elements against audio's 20,480 -- 256:1. Reducing both
   modalities against one joint element count would hand audio 0.4% of the
   gradient, so each modality is reduced against its *own* element count and the
   two are combined with ``audio_loss_weight`` (default 1.0 == equal weight).

Rollout shape. The cache's chunk boundaries must line up with the training
mask's non-noise splits, and split 0 is the text. So the text gets its own
cache-fill forward (folding it into the first media chunk would let text rows
attend media rows and poison the cached text keys), and then each chunk is
``denoise (cache read-only) -> cache-fill on the student's own detached x0``.
The clean ``eps`` rows are drawn once per rollout and carried on ``_RolloutX0s``
so a later teacher-forced forward can reuse the exact rows the cache-fill saw.

The dataset owns tokenization and ships ``text_input_ids``/``text_lens``; the
text encoder is called **once per batch** as::

    text_encoder(input_ids=[B, L], attention_mask=[B, L]) -> [B, L, D]

whose rows become ``prompt_embeds``. Which decoder layer that is comes from the
encoder node (``MiniMaxH3TextEncoder.hidden_layer``), not from here. Rows are
right-padded to the longest prompt and sliced back by ``text_lens``, so **the
encoder must honour the mask** -- one that ignores it lets the padding
contaminate the live prompt rows through attention, which degrades quality
without ever failing. One call per batch rather than one per sample is not a
performance choice: the number of samples the dataset packs varies per rank, and
a per-sample loop would make the collective count rank-dependent.

Sequence parallelism is supported on every path. The models are the ``...SP``
classes chosen in the trial, and every packed sequence this file builds is
padded to a multiple of the Ulysses world size -- the H3 SP models shard rows
and refuse a length they cannot divide. The padding always travels as its own
attention document (and its own zero-retention cache sample), so it is invisible
to every real sample.
"""

from __future__ import annotations

import importlib
import math
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator, Sequence

import torch
import torch.nn.functional as F

from dev.yanzuolu.common import media
from dev.yanzuolu.common.data import WorkerResumeContext
from dev.yanzuolu.common.diffusion import build_diffusion
from dev.yanzuolu.common.diffusion.schedule import PredictionType
from dev.yanzuolu.common.distributed import ops
from dev.yanzuolu.common.distributed.ops import all_reduce_sum, get_device, get_world_size
from dev.yanzuolu.common.distributed.unified_parallel import (
    get_unified_parallel_world_size,
    is_unified_parallel_initialized,
)
from dev.yanzuolu.common.logging import get_logger
from dev.yanzuolu.common.meter import get_running_average_meter
from dev.yanzuolu.common.phase import ExecutionPhase, execution_phase
from dev.yanzuolu.common.seed import RandomState, combine_seed, local_seed, yield_seed
from dev.yanzuolu.projects.minimax_h3.data.causal_text_only import _chunk_position_ids
from dev.yanzuolu.projects.minimax_h3.modeling.constants import MINIMAX_H3_SUPPORTED_FPS
from dev.yanzuolu.projects.minimax_h3.modeling.packed_tokens import (
    minimax_h3_patchify_video_latent,
    minimax_h3_unpack_audio_tokens,
    minimax_h3_unpatchify_video_tokens,
)
from dev.yanzuolu.projects.minimax_h3.modeling.packing import minimax_h3_packed_sequence
from dev.yanzuolu.projects.minimax_h3.modeling.video_chunk_renorm import (
    VideoChunkAnchor,
    renorm_video_chunk,
)
from dev.yanzuolu.utils.flex_attn import _prepare_flex_attention_mask, create_sparse_mask
from dev.yanzuolu.utils.naive_cache import NaiveCache

logger = get_logger()

# The BlockMask's internal tiling for the torch flex-attention backend. It does
# not constrain the sequence length -- see causal_wan_t2v's _block_mask and
# tests/test_flex_mask_padding_necessity.py, which measures that rather than
# assuming it.
FLEX_ATTN_BLOCK_SIZE = 128

# H3 token tags, from dev.yanzuolu.projects/minimax_h3/modeling/packing.py.
_VIDEO_TAG = 0
_TEXT_TAG = 1
_AUDIO_TAG = 2
# Stereo, and the DiT patch: both are family constants, not per-trial knobs --
# packing.py hard-codes _PATCH_H/_PATCH_W = 2 and the dataset's frame_rows
# arithmetic divides by 2 on both spatial axes.
_AUDIO_CHANNELS = 2
_PATCH_SIZE = (1, 2, 2)


def _validation_sample_seed(
    validation_seed: int, prompt_index: int, *, shift_seed: bool
) -> int:
    """Derive a validation sample stream independently of output variants."""
    return combine_seed(
        validation_seed,
        "causal_minimax_h3.validation.sample",
        prompt_index if shift_seed else 0,
    )


@dataclass(frozen=True)
class _Chunk:
    """One time chunk of a sample, as laid out by the causal dataset.

    ``noise_start``/``clean_start`` are sample-local row offsets used for the
    noisy query and cache commit. ``has_clean_copy`` records whether the packed
    sample physically contains a second clean split. Both physical splits are
    row-identical in structure: audio rows first, then video rows.
    """

    noise_start: int
    clean_start: int | None
    has_clean_copy: bool
    audio_rows: int
    video_rows: int
    video_start: int
    video_stop: int
    audio_start: int
    audio_stop: int

    @property
    def rows(self) -> int:
        return self.audio_rows + self.video_rows


@dataclass(frozen=True)
class _Layout:
    """Everything about one sample's geometry, derived from the dataset batch."""

    text_len: int
    chunks: tuple[_Chunk, ...]
    # Chunk-order audio row k lands at native row audio_row_perm[k]. Video needs
    # no such map: concatenating chunks in order already reproduces the native
    # frame-major order, whereas audio chunks are [L_c | R_c] and native is
    # [L(all) | R(all)].
    audio_row_perm: torch.Tensor
    latent_shape: tuple[int, int, int, int]
    audio_shape: tuple[int, int, int]

    @property
    def video_patch_grid(self) -> tuple[int, int, int]:
        _, latent_t, latent_h, latent_w = self.latent_shape
        return latent_t, latent_h // _PATCH_SIZE[1], latent_w // _PATCH_SIZE[2]


@dataclass(frozen=True)
class ForwardInput:
    """Static per-micro-iteration payload.

    Deliberately immutable: unlike wan's ``ForwardInput`` this carries no
    phase-varying latents/timesteps, so no clone discipline is needed and no
    primitive can silently observe another's mutation. Phase tensors travel as
    explicit arguments to the private forward helpers.
    """

    batch_size: int
    prompt_embeds: list[torch.Tensor]
    text_lens: list[int]
    seqlens: torch.Tensor
    layouts: list[_Layout]
    sinks: list[int]
    window_sizes: list[int | None]
    # causal packed layout, already offset into the packed coordinate frame
    sample_lens: list[int]
    split_lens: list[list[int]]
    attn_modes: list[list[str]]
    position_ids: torch.Tensor
    token_tags: torch.Tensor
    img_pos: torch.Tensor
    audio_pos: torch.Tensor
    text_pos: torch.Tensor
    q_ranges: torch.Tensor
    k_ranges: torch.Tensor
    attn_type_map: torch.Tensor
    attn_workloads: list[int]
    # video output rows worth computing: the clean copies are context, never loss
    noisy_img_pos: torch.Tensor
    # audio has no separate infer-output selector in the model, so its clean
    # copies always come back and are filtered here instead
    audio_noisy_sel: torch.Tensor
    # native bidirectional layout, one dict per sample (packing.py's builder)
    native: list[dict[str, Any]]
    # normalized corpus x0, carried inside inputs so the rollout payload keeps it
    # available to a DMD2 FAKE step; absent for text-only and non-DMD2 callers
    clean_latents: tuple[list[torch.Tensor], list[torch.Tensor]] | None = None


@dataclass(frozen=True)
class _RolloutX0s:
    """The student's rollout, its cache-fill noise and fixed video anchors."""

    video: list[torch.Tensor]
    audio: list[torch.Tensor]
    video_eps: list[torch.Tensor]
    audio_eps: list[torch.Tensor]
    video_anchors: list[VideoChunkAnchor | None] | None = field(
        default=None, kw_only=True
    )



class CausalMiniMaxH3Base:
    """The chunk-causal mechanism plus validation; see the module docstring."""

    # Subclasses enable this when corpus conditioning or training objectives
    # require clean video/audio latents in ForwardInput.
    _carry_clean_latents = False
    # Equal weight between the two modalities. A training subclass overrides the
    # instance value from its own meta_model block; validation never reads it.
    audio_loss_weight = 1.0
    # Temporal reweighting is a training-algorithm knob, off unless a subclass
    # sets it.
    chunk_wise_weighting: dict[str, Any] | None = None

    def __init__(self, config: Any) -> None:
        """Build the shared diffusion nodes and validation state."""
        self.config = config

        # ONE build for the whole hierarchy. A subclass indexes its own nodes
        # out of this dict rather than calling build_diffusion again on the same
        # config block, which would hand the sampler and the training path two
        # different schedule objects that only look identical.
        self._diffusion = build_diffusion(config.diffusion)
        self.sampling_timesteps = self._diffusion["sampling_timesteps"]
        self.audio_sampling_timesteps = self._diffusion["audio_sampling_timesteps"]
        self.schedule = self._diffusion["schedule"]
        self.sampler = self._diffusion["sampler"]

        # Every model node is a MiniMaxH3X0Model, so the schedule must agree
        # that the prediction already IS x0: with any other pred_type
        # convert_from_pred would reinterpret an x0 as a velocity and be wrong by
        # a plausible-looking amount. ``schedule`` is read nowhere else in this
        # file -- build_diffusion injects that same object into the sampler,
        # whose step_to converts through it on every rollout step -- so this
        # assert is what keeps it from reading as dead config.
        assert self.schedule.pred_type == PredictionType.x_0, (
            f"diffusion.schedule.pred_type must be x_0 because every MiniMax-H3 "
            f"model node returns x0, got {self.schedule.pred_type}"
        )

        # Built lazily on the first validation: the dataset owns the packed
        # layout, so validation packs its prompts through the very same class
        # rather than re-deriving the chunk structure here.
        self._validation_packer_cache: Any = None
        # Warn once per loss key, not once per micro-iteration.
        self._warned_weighting: set[str] = set()

    # ------------------------------------------------------------------
    # Inputs
    # ------------------------------------------------------------------

    def _build_inputs(
        self, batch: dict[str, Any], prompt_embeds: list[torch.Tensor]
    ) -> ForwardInput:
        """Derive the whole static payload from one dataset batch.

        Split out of ``prepare_inputs`` so ``sync_inputs`` can rebuild it from a
        peer's batch without re-running the text encoder: everything here is a
        pure function of the batch plus those embeddings.
        """
        device = get_device()
        batch_size = len(batch["sample_lens"])
        text_lens = [int(value) for value in batch["text_lens"]]
        latent_shapes = [tuple(int(dim) for dim in shape) for shape in batch["latent_shapes"]]
        audio_shapes = [tuple(int(dim) for dim in shape) for shape in batch["audio_shapes"]]
        sample_lens = [int(value) for value in batch["sample_lens"]]

        layouts = [
            self._build_layout(
                split_lens=[int(value) for value in batch["split_lens"][i]],
                attn_modes=list(batch["attn_modes"][i]),
                token_tags=batch["token_tags"][i],
                text_len=text_lens[i],
                sample_len=sample_lens[i],
                latent_shape=latent_shapes[i],
                audio_shape=audio_shapes[i],
            )
            for i in range(batch_size)
        ]

        # Offset every per-sample row index into the packed coordinate frame.
        offsets = self._offsets(sample_lens)
        img_pos = torch.cat(
            [batch["img_pos"][i] + offsets[i] for i in range(batch_size)]
        ).to(device)
        audio_pos = torch.cat(
            [batch["audio_pos"][i] + offsets[i] for i in range(batch_size)]
        ).to(device)
        text_pos = torch.cat(
            [batch["text_pos"][i] + offsets[i] for i in range(batch_size)]
        ).to(device)
        q_ranges = torch.cat(
            [batch["q_ranges"][i] + offsets[i] for i in range(batch_size)]
        ).to(device)
        k_ranges = torch.cat(
            [batch["k_ranges"][i] + offsets[i] for i in range(batch_size)]
        ).to(device)
        attn_type_map = torch.cat(batch["attn_type_map"]).to(device)
        # One workload count per sample: _prepare_flex_attention_mask returns an
        # int, and FlexAttention sums the list.
        attn_workloads = [int(value) for value in batch["attn_workloads"]]
        position_ids = torch.cat(batch["position_ids"], dim=0).to(device)
        token_tags = torch.cat(batch["token_tags"]).to(device)

        # The dataset numbers clean rows into slot 0 and noisy rows into slot 1;
        # only the membership is meaningful here, the values are ours to pick.
        clean_rows = torch.cat(batch["inverse_indices"]).to(device) == 0
        noisy_img_pos = img_pos[~clean_rows[img_pos]]
        audio_noisy_sel = torch.nonzero(~clean_rows[audio_pos], as_tuple=False).view(-1)

        self._check_layout_agrees(layouts, offsets, img_pos, audio_pos, token_tags)

        native = [
            minimax_h3_packed_sequence(
                text_len=text_lens[i],
                latent_t=latent_shapes[i][1],
                latent_h=latent_shapes[i][2],
                latent_w=latent_shapes[i][3],
                audio_t=audio_shapes[i][2],
                audio_channel=_AUDIO_CHANNELS,
                include_keyframe_cond=False,
            )
            for i in range(batch_size)
        ]

        assert [embed.shape[0] for embed in prompt_embeds] == text_lens, (
            "prompt embeddings must have one row per live text token"
        )
        return ForwardInput(
            batch_size=batch_size,
            prompt_embeds=[embed.to(device) for embed in prompt_embeds],
            text_lens=text_lens,
            seqlens=torch.tensor(
                [int(value) for value in batch["seqlens"]], dtype=torch.int32, device=device
            ),
            layouts=layouts,
            sinks=[int(value) for value in batch["sinks"]],
            window_sizes=[
                None if value is None else int(value) for value in batch["window_sizes"]
            ],
            sample_lens=sample_lens,
            split_lens=[[int(v) for v in sample] for sample in batch["split_lens"]],
            attn_modes=[list(sample) for sample in batch["attn_modes"]],
            position_ids=position_ids,
            token_tags=token_tags,
            img_pos=img_pos,
            audio_pos=audio_pos,
            text_pos=text_pos,
            q_ranges=q_ranges,
            k_ranges=k_ranges,
            attn_type_map=attn_type_map,
            attn_workloads=attn_workloads,
            noisy_img_pos=noisy_img_pos,
            audio_noisy_sel=audio_noisy_sel,
            native=[self._native_to_device(entry, device) for entry in native],
            clean_latents=(
                (
                    [tensor.to(device) for tensor in batch["video_latents"]],
                    [tensor.to(device) for tensor in batch["audio_latents"]],
                )
                if self._carry_clean_latents
                and "video_latents" in batch
                and "audio_latents" in batch
                else None
            ),
        )

    def _empty_text_inputs(self, inputs: ForwardInput) -> ForwardInput:
        """Repack ``inputs`` as the dataset's true zero-text-row condition.

        Empty text is structural in the causal ABI. Removing only embeddings
        would leave stale row offsets, temporal origins and attention
        rectangles. Rebuild every text-dependent field from the authoritative
        causal and native packing rules while preserving the media geometry and
        sample order.
        """
        device = inputs.token_tags.device
        split_lens = [[0, *sample[1:]] for sample in inputs.split_lens]
        attn_modes = [list(sample) for sample in inputs.attn_modes]
        sample_lens = [sum(sample) for sample in split_lens]
        offsets = self._offsets(sample_lens)

        position_parts: list[torch.Tensor] = []
        tag_parts: list[torch.Tensor] = []
        img_parts: list[torch.Tensor] = []
        audio_parts: list[torch.Tensor] = []
        noisy_img_parts: list[torch.Tensor] = []
        audio_noisy_parts: list[torch.Tensor] = []
        q_parts: list[torch.Tensor] = []
        k_parts: list[torch.Tensor] = []
        type_parts: list[torch.Tensor] = []
        workloads: list[int] = []
        rebuilt_layouts: list[_Layout] = []
        native: list[dict[str, Any]] = []
        audio_index_offset = 0

        for index, source_layout in enumerate(inputs.layouts):
            _, _, latent_h, latent_w = source_layout.latent_shape
            local_positions: list[torch.Tensor] = []
            local_tags: list[torch.Tensor] = []
            local_img: list[torch.Tensor] = []
            local_audio: list[torch.Tensor] = []
            local_noisy_img: list[torch.Tensor] = []
            local_audio_noisy: list[torch.Tensor] = []
            cursor = 0
            audio_cursor = 0
            for chunk in source_layout.chunks:
                chunk_positions = _chunk_position_ids(
                    video_start=chunk.video_start,
                    video_stop=chunk.video_stop,
                    audio_start=chunk.audio_start,
                    audio_stop=chunk.audio_stop,
                    latent_h=latent_h,
                    latent_w=latent_w,
                    origin=0,
                ).to(device)
                chunk_tags = torch.cat(
                    (
                        torch.full(
                            (chunk.audio_rows,),
                            _AUDIO_TAG,
                            dtype=torch.long,
                            device=device,
                        ),
                        torch.full(
                            (chunk.video_rows,),
                            _VIDEO_TAG,
                            dtype=torch.long,
                            device=device,
                        ),
                    )
                )
                copies = 2 if chunk.has_clean_copy else 1
                for copy_index in range(copies):
                    local_positions.append(chunk_positions)
                    local_tags.append(chunk_tags)
                    local_audio.append(
                        torch.arange(
                            cursor,
                            cursor + chunk.audio_rows,
                            dtype=torch.long,
                            device=device,
                        )
                    )
                    local_img.append(
                        torch.arange(
                            cursor + chunk.audio_rows,
                            cursor + chunk.rows,
                            dtype=torch.long,
                            device=device,
                        )
                    )
                    if copy_index == 0:
                        local_noisy_img.append(local_img[-1])
                        local_audio_noisy.append(
                            torch.arange(
                                audio_index_offset + audio_cursor,
                                audio_index_offset + audio_cursor + chunk.audio_rows,
                                dtype=torch.long,
                                device=device,
                            )
                        )
                    cursor += chunk.rows
                    audio_cursor += chunk.audio_rows

            sample_positions = torch.cat(local_positions)
            sample_tags = torch.cat(local_tags)
            position_parts.append(sample_positions)
            tag_parts.append(sample_tags)
            img_parts.extend(value + offsets[index] for value in local_img)
            audio_parts.extend(value + offsets[index] for value in local_audio)
            noisy_img_parts.extend(value + offsets[index] for value in local_noisy_img)
            audio_noisy_parts.extend(local_audio_noisy)
            audio_index_offset += audio_cursor

            layout = self._build_layout(
                split_lens=split_lens[index],
                attn_modes=attn_modes[index],
                token_tags=sample_tags,
                text_len=0,
                sample_len=sample_lens[index],
                latent_shape=source_layout.latent_shape,
                audio_shape=source_layout.audio_shape,
            )
            rebuilt_layouts.append(layout)
            sample_q, sample_k, sample_type, workload = _prepare_flex_attention_mask(
                split_lens[index],
                attn_modes[index],
                sink=inputs.sinks[index],
                window_size=inputs.window_sizes[index],
                device=device,
            )
            q_parts.append(sample_q + offsets[index])
            k_parts.append(sample_k + offsets[index])
            type_parts.append(sample_type)
            workloads.append(int(workload))
            native.append(
                self._native_to_device(
                    minimax_h3_packed_sequence(
                        text_len=0,
                        latent_t=source_layout.latent_shape[1],
                        latent_h=latent_h,
                        latent_w=latent_w,
                        audio_t=source_layout.audio_shape[2],
                        audio_channel=_AUDIO_CHANNELS,
                        include_keyframe_cond=False,
                    ),
                    device,
                )
            )

        position_ids = torch.cat(position_parts)
        token_tags = torch.cat(tag_parts)
        img_pos = torch.cat(img_parts)
        audio_pos = torch.cat(audio_parts)
        self._check_layout_agrees(
            rebuilt_layouts, offsets, img_pos, audio_pos, token_tags
        )
        return ForwardInput(
            batch_size=inputs.batch_size,
            prompt_embeds=[embedding[:0] for embedding in inputs.prompt_embeds],
            text_lens=[0] * inputs.batch_size,
            seqlens=inputs.seqlens,
            layouts=rebuilt_layouts,
            sinks=list(inputs.sinks),
            window_sizes=list(inputs.window_sizes),
            sample_lens=sample_lens,
            split_lens=split_lens,
            attn_modes=attn_modes,
            position_ids=position_ids,
            token_tags=token_tags,
            img_pos=img_pos,
            audio_pos=audio_pos,
            text_pos=torch.empty(0, dtype=torch.long, device=device),
            q_ranges=torch.cat(q_parts),
            k_ranges=torch.cat(k_parts),
            attn_type_map=torch.cat(type_parts),
            attn_workloads=workloads,
            noisy_img_pos=torch.cat(noisy_img_parts),
            audio_noisy_sel=torch.cat(audio_noisy_parts),
            native=native,
            clean_latents=inputs.clean_latents,
        )

    # ------------------------------------------------------------------
    # Rollout
    # ------------------------------------------------------------------

    @property
    def video_chunk_renorm(self) -> bool:
        """Whether generated video chunks share the first generated statistics."""
        return bool(self.config.get("meta_model", {}).get("video_chunk_renorm", False))

    def _video_renorm_start_chunks(self, inputs: ForwardInput) -> tuple[int, ...]:
        """Locate each sample's first generated chunk after any corpus prefix."""
        return (0,) * inputs.batch_size

    def _renorm_completed_video_chunks(
        self,
        inputs: ForwardInput,
        videos: list[torch.Tensor],
        anchors: list[VideoChunkAnchor | None] | None,
        *,
        chunk_index: int,
    ) -> list[torch.Tensor]:
        """Align completed chunks before output assembly and clean cache writes."""
        if anchors is None:
            return videos
        starts = self._video_renorm_start_chunks(inputs)
        for index, (video, start) in enumerate(zip(videos, starts, strict=True)):
            if chunk_index < start:
                continue
            if chunk_index > start:
                assert anchors[index] is not None
            videos[index], anchors[index] = renorm_video_chunk(video, anchors[index])
        return videos

    def _renorm_video_chunks(
        self,
        inputs: ForwardInput,
        videos: list[torch.Tensor],
        anchors: Sequence[VideoChunkAnchor | None] | None,
        *,
        chunk_offset: int = 0,
    ) -> list[torch.Tensor]:
        """Match full prediction chunks to the fixed anchors of their rollout."""
        if not self.video_chunk_renorm:
            return videos
        assert anchors is not None
        starts = self._video_renorm_start_chunks(inputs)
        result = []
        for video, layout, anchor, start in zip(
            videos, inputs.layouts, anchors, starts, strict=True
        ):
            chunks = []
            for chunk_index, chunk in enumerate(layout.chunks):
                value = video[:, chunk.video_start : chunk.video_stop]
                if chunk_index + chunk_offset > start:
                    assert anchor is not None
                    value, _ = renorm_video_chunk(value, anchor)
                chunks.append(value)
            result.append(torch.cat(chunks, dim=1))
        return result

    @torch.no_grad()
    def _rollout_latents(
        self,
        backbone: Any,
        inputs: ForwardInput,
        rng: Any,
        *,
        keep_trajectory: bool,
        rngs: Sequence[Any] | None = None,
        trajectory_x0_chunks: list[
            tuple[list[torch.Tensor], list[torch.Tensor]]
        ] | None = None,
    ) -> tuple[_RolloutX0s, list[tuple[list[torch.Tensor], list[torch.Tensor]]]]:
        """Chunk-by-chunk causal sampling against a KV cache.

        Shared by training rollout and validation; ``keep_trajectory`` controls
        whether DMD retains the intermediate ``x_t`` of every sampling step.
        ``trajectory_x0_chunks`` optionally captures each step's x0 prediction in
        chunk-major, step-major order for DMD2 FAKE training.

        ``rngs`` (optional, one stream per sample) gives every sample its own
        deterministic stream inside ONE batched rollout -- the validation
        pattern, matching wan/crafter: per-prompt seeds, batch is just a
        shared forward, so changing batch_size never perturbs a prompt's
        noise. The training rollout draws all samples from ``rng``.
        """
        device = get_device()
        sample_rngs = rngs if rngs is not None else [rng] * inputs.batch_size
        assert len(sample_rngs) == inputs.batch_size, (
            f"rngs must have one stream per sample: {len(sample_rngs)} vs "
            f"batch_size {inputs.batch_size}"
        )

        video_noises = [
            torch.empty(layout.latent_shape, device=device).normal_(
                generator=self._generator(sample_rngs[i])
            )
            for i, layout in enumerate(inputs.layouts)
        ]
        audio_noises = [
            torch.empty(layout.audio_shape, device=device).normal_(
                generator=self._generator(sample_rngs[i])
            )
            for i, layout in enumerate(inputs.layouts)
        ]
        # Clean-context noise, drawn once per sample and carried on the payload:
        # gen_forward's clean splits must reuse the exact rows the cache-fill
        # forward baked into the KV cache, otherwise the packed teacher-forced
        # forward conditions on a different context than the rollout did.
        video_eps = [
            torch.empty(layout.latent_shape, device=device).normal_(
                generator=self._generator(sample_rngs[i])
            )
            for i, layout in enumerate(inputs.layouts)
        ]
        audio_eps = [
            torch.empty(layout.audio_shape, device=device).normal_(
                generator=self._generator(sample_rngs[i])
            )
            for i, layout in enumerate(inputs.layouts)
        ]

        self.sampling_timesteps.set_timesteps(seqlen=inputs.seqlens, device=device)
        self.audio_sampling_timesteps.set_timesteps(seqlen=inputs.seqlens, device=device)
        video_grid = self.sampling_timesteps.timesteps
        audio_grid = self.audio_sampling_timesteps.timesteps
        assert video_grid.dim() == 1 and audio_grid.dim() == 1, (
            "dynamic-shift sampling grids are not supported: rollout advances the "
            "video and audio grids by a shared step index"
        )
        assert video_grid.numel() == audio_grid.numel(), (
            "sampling_timesteps and audio_sampling_timesteps must have the same "
            f"number of steps, got {video_grid.numel()} and {audio_grid.numel()}"
        )
        num_steps = int(video_grid.numel())
        # The sampler natively accepts a per-sample rng list; the training
        # path keeps drawing the whole batch from the engine's single rng.
        step_rng = rngs if rngs is not None else rng

        # Under SP each cached segment is padded to a multiple of the group size,
        # and that padding travels as an extra cache sample. sink=0 with
        # window_size=0 evicts it on the very step it is written, so it costs one
        # varlen segment per forward and nothing in the cache.
        pad_sample = self._sp_size() > 1
        cache = NaiveCache(
            self._num_layers(backbone),
            inputs.batch_size + (1 if pad_sample else 0),
            sink=list(inputs.sinks) + ([0] if pad_sample else []),
            window_size=list(inputs.window_sizes) + ([0] if pad_sample else []),
        )
        # Cache chunk 0 is the text, matching non-noise split 0 of the training
        # mask. It gets its own forward: folding it into the first media chunk
        # would let the text rows attend media rows, and the cached text keys
        # would no longer be the ones the training mask assumes.
        self._text_cache_fill(backbone, inputs, cache)

        video_x0s = [torch.zeros_like(noise) for noise in video_noises]
        audio_x0s = [torch.zeros_like(noise) for noise in audio_noises]
        video_anchors: list[VideoChunkAnchor | None] | None = (
            [None] * inputs.batch_size if self.video_chunk_renorm else None
        )
        trajectory_xts: list[tuple[list[torch.Tensor], list[torch.Tensor]]] = [
            (
                [torch.zeros_like(noise) for noise in video_noises],
                [torch.zeros_like(noise) for noise in audio_noises],
            )
            for _ in range(num_steps if keep_trajectory else 0)
        ]

        num_chunks = len(inputs.layouts[0].chunks)
        for chunk_index in range(num_chunks):
            chunks = [layout.chunks[chunk_index] for layout in inputs.layouts]
            video_xts = [
                video_noises[i][:, chunk.video_start : chunk.video_stop]
                for i, chunk in enumerate(chunks)
            ]
            audio_xts = [
                audio_noises[i][:, :, chunk.audio_start : chunk.audio_stop]
                for i, chunk in enumerate(chunks)
            ]

            for step in range(num_steps):
                if keep_trajectory:
                    for i, chunk in enumerate(chunks):
                        trajectory_xts[step][0][i][
                            :, chunk.video_start : chunk.video_stop
                        ] = video_xts[i]
                        trajectory_xts[step][1][i][
                            :, :, chunk.audio_start : chunk.audio_stop
                        ] = audio_xts[i]

                video_t = video_grid[step].expand(inputs.batch_size).to(device)
                audio_t = audio_grid[step].expand(inputs.batch_size).to(device)
                video_s = self.sampling_timesteps.get_next_timesteps(video_t)
                audio_s = self.audio_sampling_timesteps.get_next_timesteps(audio_t)

                video_pred, audio_pred = self._chunk_forward(
                    backbone,
                    inputs,
                    chunk_index=chunk_index,
                    role="noise",
                    video_rows_source=video_xts,
                    audio_rows_source=audio_xts,
                    video_timesteps=video_t,
                    audio_timesteps=audio_t,
                    cache=cache,
                    update_cache=False,
                )
                if trajectory_x0_chunks is not None:
                    trajectory_x0_chunks.append(
                        (
                            [value.detach().clone() for value in video_pred],
                            [value.detach().clone() for value in audio_pred],
                        )
                    )
                video_xts = self.sampler.step_to(
                    pred=video_pred, x_t=video_xts, t=video_t, s=video_s,
                    rng=step_rng, seqlens=inputs.seqlens,
                )
                audio_xts = self.sampler.step_to(
                    pred=audio_pred, x_t=audio_xts, t=audio_t, s=audio_s,
                    rng=step_rng, seqlens=inputs.seqlens,
                )

            video_xts = self._renorm_completed_video_chunks(
                inputs, video_xts, video_anchors, chunk_index=chunk_index
            )
            for i, chunk in enumerate(chunks):
                video_x0s[i][:, chunk.video_start : chunk.video_stop] = video_xts[i]
                audio_x0s[i][:, :, chunk.audio_start : chunk.audio_stop] = audio_xts[i]

            if chunks[0].clean_start is None:
                continue  # last chunk: nothing after it will read this history
            self._chunk_forward(
                backbone,
                inputs,
                chunk_index=chunk_index,
                role="clean",
                video_rows_source=video_xts,
                audio_rows_source=audio_xts,
                video_timesteps=torch.zeros(inputs.batch_size, device=device),
                audio_timesteps=torch.zeros(inputs.batch_size, device=device),
                cache=cache,
                update_cache=True,
                video_eps_source=[
                    video_eps[i][:, chunk.video_start : chunk.video_stop]
                    for i, chunk in enumerate(chunks)
                ],
                audio_eps_source=[
                    audio_eps[i][:, :, chunk.audio_start : chunk.audio_stop]
                    for i, chunk in enumerate(chunks)
                ],
            )

        return (
            _RolloutX0s(
                video=video_x0s, audio=audio_x0s, video_eps=video_eps,
                audio_eps=audio_eps, video_anchors=video_anchors,
            ),
            trajectory_xts,
        )

    @torch.no_grad()
    def _guided_rollout_latents(
        self,
        backbone: Any,
        branch_inputs: Sequence[tuple[float, ForwardInput]],
        rngs: Sequence[Any],
        *,
        first_branch_without_history: bool = False,
    ) -> _RolloutX0s:
        """Sample caption branches on one shared causal trajectory."""
        inputs = branch_inputs[0][1]
        device = get_device()
        assert len(rngs) == inputs.batch_size, (
            f"rngs must have one stream per sample: {len(rngs)} vs "
            f"batch_size {inputs.batch_size}"
        )

        video_noises = [
            torch.empty(layout.latent_shape, device=device).normal_(
                generator=self._generator(rngs[i])
            )
            for i, layout in enumerate(inputs.layouts)
        ]
        audio_noises = [
            torch.empty(layout.audio_shape, device=device).normal_(
                generator=self._generator(rngs[i])
            )
            for i, layout in enumerate(inputs.layouts)
        ]
        video_eps = [
            torch.empty(layout.latent_shape, device=device).normal_(
                generator=self._generator(rngs[i])
            )
            for i, layout in enumerate(inputs.layouts)
        ]
        audio_eps = [
            torch.empty(layout.audio_shape, device=device).normal_(
                generator=self._generator(rngs[i])
            )
            for i, layout in enumerate(inputs.layouts)
        ]

        self.sampling_timesteps.set_timesteps(seqlen=inputs.seqlens, device=device)
        self.audio_sampling_timesteps.set_timesteps(seqlen=inputs.seqlens, device=device)
        video_grid = self.sampling_timesteps.timesteps
        audio_grid = self.audio_sampling_timesteps.timesteps
        assert video_grid.dim() == 1 and audio_grid.dim() == 1, (
            "dynamic-shift sampling grids are not supported: rollout advances the "
            "video and audio grids by a shared step index"
        )
        assert video_grid.numel() == audio_grid.numel(), (
            "sampling_timesteps and audio_sampling_timesteps must have the same "
            f"number of steps, got {video_grid.numel()} and {audio_grid.numel()}"
        )

        pad_sample = self._sp_size() > 1
        caches = [
            NaiveCache(
                self._num_layers(backbone),
                branch.batch_size + (1 if pad_sample else 0),
                sink=list(branch.sinks) + ([0] if pad_sample else []),
                window_size=list(branch.window_sizes) + ([0] if pad_sample else []),
            )
            for _, branch in branch_inputs
        ]
        for branch_index, ((_, branch), cache) in enumerate(
            zip(branch_inputs, caches, strict=True)
        ):
            if first_branch_without_history and branch_index == 0:
                continue
            self._text_cache_fill(backbone, branch, cache)

        video_x0s = [torch.zeros_like(noise) for noise in video_noises]
        audio_x0s = [torch.zeros_like(noise) for noise in audio_noises]
        video_anchors: list[VideoChunkAnchor | None] | None = (
            [None] * inputs.batch_size if self.video_chunk_renorm else None
        )
        num_chunks = len(inputs.layouts[0].chunks)
        for chunk_index in range(num_chunks):
            chunks = [layout.chunks[chunk_index] for layout in inputs.layouts]
            video_xts = [
                video_noises[i][:, chunk.video_start : chunk.video_stop]
                for i, chunk in enumerate(chunks)
            ]
            audio_xts = [
                audio_noises[i][:, :, chunk.audio_start : chunk.audio_stop]
                for i, chunk in enumerate(chunks)
            ]

            for step in range(int(video_grid.numel())):
                video_t = video_grid[step].expand(inputs.batch_size).to(device)
                audio_t = audio_grid[step].expand(inputs.batch_size).to(device)
                video_s = self.sampling_timesteps.get_next_timesteps(video_t)
                audio_s = self.audio_sampling_timesteps.get_next_timesteps(audio_t)

                video_pred: list[torch.Tensor] | None = None
                audio_pred: list[torch.Tensor] | None = None
                for (weight, branch), cache in zip(branch_inputs, caches, strict=True):
                    branch_video, branch_audio = self._chunk_forward(
                        backbone,
                        branch,
                        chunk_index=chunk_index,
                        role="noise",
                        video_rows_source=video_xts,
                        audio_rows_source=audio_xts,
                        video_timesteps=video_t,
                        audio_timesteps=audio_t,
                        cache=cache,
                        update_cache=False,
                    )
                    if video_pred is None:
                        video_pred = [float(weight) * value for value in branch_video]
                        audio_pred = [float(weight) * value for value in branch_audio]
                    else:
                        video_pred = [
                            pred + float(weight) * value
                            for pred, value in zip(
                                video_pred, branch_video, strict=True
                            )
                        ]
                        assert audio_pred is not None
                        audio_pred = [
                            pred + float(weight) * value
                            for pred, value in zip(
                                audio_pred, branch_audio, strict=True
                            )
                        ]

                assert video_pred is not None and audio_pred is not None
                video_xts = self.sampler.step_to(
                    pred=video_pred,
                    x_t=video_xts,
                    t=video_t,
                    s=video_s,
                    rng=list(rngs),
                    seqlens=inputs.seqlens,
                )
                audio_xts = self.sampler.step_to(
                    pred=audio_pred,
                    x_t=audio_xts,
                    t=audio_t,
                    s=audio_s,
                    rng=list(rngs),
                    seqlens=inputs.seqlens,
                )

            video_xts = self._renorm_completed_video_chunks(
                inputs, video_xts, video_anchors, chunk_index=chunk_index
            )
            for i, chunk in enumerate(chunks):
                video_x0s[i][:, chunk.video_start : chunk.video_stop] = video_xts[i]
                audio_x0s[i][:, :, chunk.audio_start : chunk.audio_stop] = audio_xts[i]

            if chunks[0].clean_start is None:
                continue
            for branch_index, ((_, branch), cache) in enumerate(
                zip(branch_inputs, caches, strict=True)
            ):
                if first_branch_without_history and branch_index == 0:
                    continue
                self._chunk_forward(
                    backbone,
                    branch,
                    chunk_index=chunk_index,
                    role="clean",
                    video_rows_source=video_xts,
                    audio_rows_source=audio_xts,
                    video_timesteps=torch.zeros(inputs.batch_size, device=device),
                    audio_timesteps=torch.zeros(inputs.batch_size, device=device),
                    cache=cache,
                    update_cache=True,
                    video_eps_source=[
                        video_eps[i][:, chunk.video_start : chunk.video_stop]
                        for i, chunk in enumerate(chunks)
                    ],
                    audio_eps_source=[
                        audio_eps[i][:, :, chunk.audio_start : chunk.audio_stop]
                        for i, chunk in enumerate(chunks)
                    ],
                )

        return _RolloutX0s(
            video=video_x0s,
            audio=audio_x0s,
            video_eps=video_eps,
            audio_eps=audio_eps,
            video_anchors=video_anchors,
        )

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    @contextmanager
    def _validation_sampling(self, validation: Any) -> Iterator[None]:
        """Apply one validation entry's sampling nodes and restore training state."""
        timestep_names = ("sampling_timesteps", "audio_sampling_timesteps")
        saved = {
            name: getattr(self, name) for name in (*timestep_names, "sampler")
        }
        nodes = {
            name: (
                validation[name]
                if validation.get(name) is not None
                else self.config.diffusion[name]
            )
            for name in saved
        }
        sampling = build_diffusion({name: nodes[name] for name in timestep_names})
        sampler_node = nodes["sampler"]
        sampler_cls = getattr(
            importlib.import_module(sampler_node["module"]),
            sampler_node["class_name"],
        )
        sampling["sampler"] = sampler_cls(
            schedule=self.schedule,
            **dict(sampler_node.get("args", {}) or {}),
        )
        try:
            for name, node in sampling.items():
                setattr(self, name, node)
            yield
        finally:
            for name, node in saved.items():
                setattr(self, name, node)

    def _validation_frames(
        self, frames: torch.Tensor, inputs: Any, *, index: int
    ) -> torch.Tensor:
        """Frames written to one validation mp4; subclasses may add panels beside the sample."""
        del inputs, index
        return frames

    @execution_phase(ExecutionPhase.VALIDATION)
    @torch.no_grad()
    def validate(self, ctx: dict[str, Any]) -> dict[str, Any]:
        """Run one validation mapping or each mapping in a validation list."""
        config = ctx["config"]
        validation = config.validation
        if not isinstance(validation, list):
            with self._validation_sampling(validation):
                return self._validate_once(ctx)

        original_cache = self._validation_packer_cache
        try:
            for entry in validation:
                config["validation"] = entry
                self._validation_packer_cache = None
                with self._validation_sampling(entry):
                    self._validate_once(ctx)
            return ctx
        finally:
            config["validation"] = validation
            self._validation_packer_cache = original_cache

    def _validate_once(self, ctx: dict[str, Any]) -> dict[str, Any]:
        """Reads: config, models, step. Writes: nothing (media is logged externally).

        Roll out each prompt, then export normalized AV latents or a muxed mp4. Every
        rank walks the same number of entries -- the split pads by repeating the
        last one with ``should_save=False`` -- because the rollout is collective
        under FSDP and sequence parallelism, and a rank that ran a different
        number of forwards would hang the group.
        """
        config = ctx["config"]
        models = ctx["models"]
        step = int(ctx["step"])
        validation = config.validation
        requests = self._validation_requests(validation)
        assert requests, "validation requires at least one prompt"

        rank = ops.get_rank()
        sp_size = self._sp_size()
        group_index = rank // sp_size
        num_groups = ops.get_world_size() // sp_size
        is_group_main = rank % sp_size == 0

        suffix = f"_{validation.name}" if "name" in validation else ""
        variants = self._validation_model_variants(models, validation, suffix=suffix)
        media_dir = logger.iter_dir("media", step)
        save_latent = bool(validation.get("save_latent", False))
        out_dir = media_dir
        if save_latent and validation.get("latent_dir", None):
            out_dir = Path(validation.latent_dir).expanduser()

        # Resume: drop prompts already on disk, and adopt rank 0's view of which
        # those are. A divergent view gives ranks different per-rank counts and
        # desynchronises the collective forwards.
        completed = ops.all_gather_object(
            self._completed_prompt_ids(
                out_dir, [name for name, _ in variants], "pt" if save_latent else "mp4"
            )
        )[0]
        pending = [
            (index, request)
            for index, request in enumerate(requests)
            if index not in completed
        ]
        entries = self._split_validation_prompts(
            pending,
            group_index,
            num_groups,
            self._validation_request_prompt,
        )

        original_training = [(model, model.training) for _, model in variants]
        saved_items: list[tuple[str, str, str, str]] = []
        fps = int(validation.get("fps", MINIMAX_H3_SUPPORTED_FPS))
        try:
            for _, model in variants:
                model.eval()
            # Validation batch_size packs that many prompts into ONE rollout
            # forward (training parity: the dataset packs the same way under
            # its max_seqlen budget). 1 keeps the historical per-prompt path,
            # including the exact per-prompt rng seed.
            batch_size = int(validation.get("batch_size", 1))
            assert batch_size >= 1, f"validation.batch_size must be >= 1, got {batch_size}"
            guidance_scale = float(validation.get("guidance_scale", 1.0))
            previous_guidance_scale = (
                float(validation.get("previous_guidance_scale", 1.0))
                if getattr(self, "validation_mode", "causal") == "causal"
                else 1.0
            )
            for batch_start in range(0, len(entries), batch_size):
                batch_entries = entries[batch_start : batch_start + batch_size]
                batch_requests = [entry["request"] for entry in batch_entries]
                batch_prompts = [entry["prompt"] for entry in batch_entries]
                if previous_guidance_scale == 1.0:
                    if guidance_scale == 1.0:
                        branch_inputs = [
                            (
                                1.0,
                                self._validation_inputs_for_requests(
                                    config,
                                    models,
                                    batch_requests,
                                    prompts=batch_prompts,
                                ),
                            )
                        ]
                    else:
                        branch_specs = [
                            (guidance_scale, batch_prompts),
                            (1.0 - guidance_scale, [""] * len(batch_prompts)),
                        ]
                        branch_inputs = [
                            (
                                weight,
                                self._validation_inputs_for_requests(
                                    config,
                                    models,
                                    batch_requests,
                                    prompts=prompts,
                                ),
                            )
                            for weight, prompts in branch_specs
                            if weight != 0.0
                        ]
                else:
                    empty_inputs = self._validation_inputs_for_requests(
                        config,
                        models,
                        batch_requests,
                        prompts=[""] * len(batch_prompts),
                    )
                    branch_inputs = [
                        (weight, empty_inputs)
                        for weight in (
                            1.0 - previous_guidance_scale,
                            previous_guidance_scale - guidance_scale,
                        )
                        if weight != 0.0
                    ]
                    if guidance_scale != 0.0:
                        branch_inputs.append(
                            (
                                guidance_scale,
                                self._validation_inputs_for_requests(
                                    config,
                                    models,
                                    batch_requests,
                                    prompts=batch_prompts,
                                ),
                            )
                        )
                # Per-prompt seeds (the wan/crafter pattern): each prompt keeps
                # its own deterministic stream inside the batched rollout, so
                # changing batch_size never perturbs a prompt's noise. Output
                # variants rebuild fresh RNGs from these same seeds, keeping
                # CFG scales and backbone/EMA samples on identical initial
                # noise. shift_seed: false pins every prompt to the same seed
                # (ltx2's option); true (default) shifts by prompt index.
                shift_seed = bool(validation.get("shift_seed", True))
                sample_seeds = [
                    _validation_sample_seed(
                        int(validation.seed),
                        entry["prompt_idx"],
                        shift_seed=shift_seed,
                    )
                    for entry in batch_entries
                ]
                for variant_name, backbone in variants:
                    sample_rngs = [RandomState(seed) for seed in sample_seeds]
                    latents = self._validation_rollout(
                        models, backbone, branch_inputs, sample_rngs,
                        guidance_scale=guidance_scale,
                        previous_guidance_scale=previous_guidance_scale,
                    )
                    for index, entry in enumerate(batch_entries):
                        if not (is_group_main and entry["should_save"]):
                            continue
                        if save_latent:
                            name = f"validation/{variant_name}/prompt{entry['prompt_idx']:04d}.pt"
                            path = out_dir / name
                            self._save_validation_latents(
                                path, latents, index=index,
                                prompt=entry["prompt"], prompt_idx=entry["prompt_idx"],
                                metadata=self._validation_latent_metadata(
                                    config, branch_inputs[0][1], entry["request"], index=index,
                                ),
                            )
                            saved_items.append((variant_name, name, str(path), ""))
                            continue
                        frames, waveform = self._decode_latents(models, latents, index=index)
                        frames = self._validation_frames(frames, branch_inputs[0][1], index=index)
                        name = f"validation/{variant_name}/prompt{entry['prompt_idx']:04d}.mp4"
                        path = media_dir / name
                        path.parent.mkdir(parents=True, exist_ok=True)
                        media.save_video(
                            frames,
                            path,
                            fps=fps,
                            nrow=int(validation.get("nrow", 1)),
                            crf=int(validation.get("crf", 18)),
                            audio_tensor=waveform,
                            audio_sample_rate=int(models["audio_vae"].sample_rate),
                            normalize=True,
                            # revert_tensor already clamped the decoded pixels.
                            value_range=(0, 1),
                        )
                        saved_items.append(
                            (variant_name, name, str(path), f"[{variant_name}] {entry['prompt']}")
                        )
                        del frames, waveform
                    del latents
                del branch_inputs

            gathered = ops.all_gather_object(saved_items)
            try:
                if rank == 0 and save_latent:
                    logger.info(
                        "saved %d latents under %s",
                        sum(len(rank_items) for rank_items in gathered),
                        out_dir / "validation",
                    )
                elif rank == 0:
                    all_items = [item for rank_items in gathered for item in rank_items]
                    for variant_name, _ in variants:
                        variant_items = [item for item in all_items if item[0] == variant_name]
                        if not variant_items:
                            continue
                        logger.log_video(
                            {name: Path(path) for _, name, path, _ in variant_items},
                            step=step,
                            collection=f"validation/{variant_name}/videos",
                            captions={name: caption for _, name, _, caption in variant_items},
                            fps=fps,
                        )
                        if not validation.get("save_grid", True):
                            continue
                        grid_name = f"validation/{variant_name}/grid.mp4"
                        grid_path = media_dir / grid_name
                        try:
                            self._save_grid(
                                [Path(path) for _, _, path, _ in variant_items],
                                grid_path,
                                fps=fps,
                                nrow=int(validation.get("nrow", 4)),
                                crf=int(validation.get("crf", 18)),
                            )
                            logger.log_video(
                                {grid_name: grid_path},
                                step=step,
                                collection=f"validation/{variant_name}/grid",
                                fps=fps,
                            )
                        except Exception as exc:
                            # The grid is a preview; the per-prompt files are the
                            # real artifact. Losing a run at step N because a
                            # convenience encode hiccuped is the worse failure.
                            logger.warning("validation grid build failed: %s", exc)
            finally:
                ops.barrier()
        finally:
            for model, training in original_training:
                model.train(training)
        return ctx

    # ------------------------------------------------------------------
    # Validation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _validation_model_variants(
        models: dict[str, Any], validation: Any, *, suffix: str = "",
    ) -> list[tuple[str, Any]]:
        """Select the models required by validation and artifact completion."""
        names = []
        if validation.get("validate_backbone", True):
            names.append("backbone")
        if validation.get("validate_ema", False):
            names.append("backbone_ema")
        if not names:
            raise ValueError("validation must enable validate_backbone or validate_ema")
        return [(f"{name}{suffix}", models[name]) for name in names]

    @staticmethod
    def _save_validation_latents(
        path: Path, latents: Any, *, index: int, prompt: str, prompt_idx: int,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Atomically export one complete normalized sample before codec decoding."""
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + ".tmp")
        torch.save(
            {
                **(metadata or {}),
                "video": latents.video[index].to(torch.bfloat16).cpu(),
                "audio": latents.audio[index].to(torch.bfloat16).cpu(),
                "prompt": prompt,
                "prompt_idx": prompt_idx,
            },
            temporary,
        )
        temporary.replace(path)

    def _validation_latent_metadata(
        self, config: Any, inputs: Any, request: Any, *, index: int,
    ) -> dict[str, Any]:
        """Attach source conditions needed to reuse an exported sample for training."""
        return {}

    def _validation_rollout(self, models: dict[str, Any], backbone: Any, branch_inputs: Any,
                            rngs: Sequence[Any], *, guidance_scale: float, previous_guidance_scale: float) -> Any:
        """Expose runtime conditioning models without changing the default sampler."""
        if previous_guidance_scale != 1.0:
            return self._guided_rollout_latents(
                backbone, branch_inputs, rngs, first_branch_without_history=True,
            )
        if guidance_scale == 1.0:
            latents, _ = self._rollout_latents(
                backbone, branch_inputs[0][1], rngs[0], keep_trajectory=False, rngs=rngs,
            )
            return latents
        return self._guided_rollout_latents(backbone, branch_inputs, rngs)

    @staticmethod
    def _validation_prompts(validation: Any) -> list[str]:
        prompts = validation.get("prompts", None)
        if prompts is None:
            path = Path(validation.prompt_path).expanduser()
            with path.open("r", encoding="utf8") as handle:
                prompts = [line.strip() for line in handle if line.strip()]
        prompts = [str(prompt) for prompt in prompts]
        limit = int(validation.get("num_prompts", 0))
        return prompts[:limit] if limit > 0 else prompts

    def _validation_requests(self, validation: Any) -> list[Any]:
        """Build validation requests. Prompt-only validation uses raw strings."""
        return self._validation_prompts(validation)

    @staticmethod
    def _validation_request_prompt(request: Any) -> str:
        """Return the caption prompt carried by one validation request."""
        return str(request)

    @staticmethod
    def _read_video_frames(path: Path) -> torch.Tensor:
        """Read every MP4 frame as a uint8 THWC tensor."""
        import av

        with av.open(str(path)) as container:
            frames = [
                torch.from_numpy(frame.to_ndarray(format="rgb24"))
                for frame in container.decode(video=0)
            ]
        return torch.stack(frames)

    def _save_grid(
        self, paths: list[Path], out_path: Path, *, fps: int, nrow: int, crf: int
    ) -> None:
        """Tile the saved clips into one silent preview.

        Silent by omission, exactly as ltx2 does it: ``media.save_video`` writes a
        video-only MP4 when no audio argument is given. There is nothing to mix
        anyway -- sixteen soundtracks cannot be listened to at once, so the audio
        stays in the per-prompt files, which are the artifact that matters for a
        T2AV model.
        """
        videos = [
            self._read_video_frames(path).permute(3, 0, 1, 2).float().div(127.5).sub(1.0)
            for path in paths
        ]
        shapes = {tuple(video.shape) for video in videos}
        assert len(shapes) == 1, (
            f"validation clips must share one geometry to tile, got {shapes}"
        )
        grid = torch.stack(videos)

        # Bound the whole tiling to a 1080p budget shared across cells, so the
        # preview cost does not scale with the trial's resolution.
        rows = math.ceil(grid.shape[0] / nrow)
        max_pixels = 1920 * 1080 // (nrow * rows)
        batch, channels, frames, height, width = grid.shape
        if height * width > max_pixels:
            scale = (max_pixels / (height * width)) ** 0.5
            target = (
                int(height * scale) - int(height * scale) % 2,
                int(width * scale) - int(width * scale) % 2,
            )
            flat = grid.permute(0, 2, 1, 3, 4).reshape(batch * frames, channels, height, width)
            flat = F.interpolate(flat, size=target, mode="bilinear", align_corners=False)
            grid = flat.reshape(batch, frames, channels, *target).permute(0, 2, 1, 3, 4)

        out_path.parent.mkdir(parents=True, exist_ok=True)
        media.save_video(
            grid, out_path, fps=fps, nrow=nrow, crf=crf,
            normalize=True, value_range=(-1, 1),
        )

    @staticmethod
    def _completed_prompt_ids(
        media_dir: Path, variant_names: list[str], suffix: str = "mp4"
    ) -> set[int]:
        """Prompt indices already rendered for every variant."""
        completed: set[int] | None = None
        for name in variant_names:
            ids = {
                int(path.name[6:10])
                for path in (media_dir / "validation" / name).glob(f"prompt*.{suffix}")
            }
            completed = ids if completed is None else completed & ids
        return completed or set()

    @staticmethod
    def _split_validation_prompts(
        pending: list[tuple[int, Any]],
        group_index: int,
        num_groups: int,
        prompt_from_request: Callable[[Any], str] | None = None,
    ) -> list[dict[str, Any]]:
        """Equal-length per-group slices; the padding entries are not saved."""
        if not pending:
            return []
        per_group = (len(pending) + num_groups - 1) // num_groups
        padded = pending + [pending[-1]] * (per_group * num_groups - len(pending))
        start = group_index * per_group
        return [
            {
                "prompt": (
                    request
                    if prompt_from_request is None
                    else prompt_from_request(request)
                ),
                "request": request,
                "prompt_idx": index,
                "should_save": start + offset < len(pending),
            }
            for offset, (index, request) in enumerate(padded[start : start + per_group])
        ]

    def _validation_packer(self, config: Any) -> Any:
        """The training dataset class, reused purely as the layout packer.

        Validation must produce byte-identical chunk structure to training, and
        the dataset is where that structure is defined. Rebuilding it here would
        be a second implementation of the same rule, free to drift.
        """
        if self._validation_packer_cache is None:
            module = importlib.import_module(config.data.module)
            dataset_cls = getattr(module, config.data.class_name)
            args = dict(config.data.get("args", {}))
            validation = config.validation
            args.update(
                prompts=[str(validation.get("packer_prompt", "packer"))],
                paths=None,
                height=int(validation.height),
                width=int(validation.width),
                num_frames=int(validation.num_frames),
            )
            self._validation_packer_cache = dataset_cls(
                seed=int(validation.seed),
                resume_context=WorkerResumeContext(
                    rank=0, world_size=1, num_workers=1,
                    next_logical_worker_id=0, committed_states={},
                ),
                **args,
            )
        return self._validation_packer_cache

    def _validation_inputs_for_requests(
        self,
        config: Any,
        models: dict[str, Any],
        requests: Sequence[Any],
        *,
        prompts: Sequence[str] | None = None,
    ) -> ForwardInput:
        """Build inputs for requests while allowing caption-only branches."""
        if prompts is None:
            prompts = [self._validation_request_prompt(request) for request in requests]
        return self._validation_inputs(config, models, prompts)

    def _validation_inputs(
        self, config: Any, models: dict[str, Any], prompts: Sequence[str]
    ) -> ForwardInput:
        """Pack ``prompts`` into ONE ForwardInput (batch_size = len(prompts)).

        Validation batch_size is a training-parity feature: the dataset's
        ``__iter__`` packs as many samples as the ``max_seqlen`` budget allows
        and the rollout paths are batch-aware throughout, so validation packs
        the same way. A single rollout forward then covers the whole batch,
        one latents entry per prompt (``_decode_latents(..., index=)``).
        """
        packer = self._validation_packer(config)
        samples = [packer._pack_sample(prompt=prompt) for prompt in prompts]
        batch = {key: [sample[key] for sample in samples] for key in samples[0]}
        budget = int(packer.max_seqlen)
        assert sum(int(v) for v in batch["seqlens"]) <= budget, (
            f"validation batch of {len(prompts)} prompts needs "
            f"{sum(int(v) for v in batch['seqlens'])} rows but the packer's "
            f"max_seqlen budget is {budget}"
        )
        prompt_embeds = self._encode_prompts(
            models, batch["text_input_ids"], [int(v) for v in batch["text_lens"]]
        )
        return self._build_inputs(batch, prompt_embeds)

    @staticmethod
    def _decode_latents(
        models: dict[str, Any], latents: _RolloutX0s, index: int = 0
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Normalized latents -> ``([1,3,F,H,W]`` in [0,1], ``[2, samples]``).

        Both VAEs carry their per-channel statistics as non-persistent buffers
        and apply neither: upstream un-normalizes outside the module, so this is
        where ``z * std + mean`` belongs. Video then needs ``decode_base`` (not
        ``decode``) so the released clip_length/token_drop/tiling knobs are
        honoured, and ``revert_tensor`` to leave ImageNet-normalized space.
        Audio needs no vocoder node: ``DacAudioVAE.decode`` runs BigVGAN itself.
        """
        video_vae = models["video_vae"]
        audio_vae = models["audio_vae"]

        video = latents.video[index].unsqueeze(0)
        # Cast to the VAE's runtime param dtype (bf16 in the trials): without
        # autocast -- the inference path -- the fp32-drawn latents would
        # promote the whole VAE compute to fp32; under the training autocast
        # wrapper this is the same cast it would make anyway. The attr is
        # annotated by apply_runtime from the runtime.param_dtype config.
        video = video.to(dtype=video_vae.param_dtype)
        video = video * video_vae.latents_std.to(video).view(
            1, -1, 1, 1, 1
        ) + video_vae.latents_mean.to(video).view(1, -1, 1, 1, 1)
        frames = video_vae.processor.revert_tensor(video_vae.decode_base(video))

        # Stereo is two mono batch items, so the statistics broadcast over dim 1.
        audio = latents.audio[index]
        audio = audio.to(dtype=audio_vae.param_dtype)
        audio = audio * audio_vae.latents_std.to(audio).view(
            1, -1, 1
        ) + audio_vae.latents_mean.to(audio).view(1, -1, 1)
        waveform = audio_vae.decode(audio).squeeze(1)
        return frames, waveform

    # ------------------------------------------------------------------
    # Layout derivation
    # ------------------------------------------------------------------

    @staticmethod
    def _offsets(sample_lens: Sequence[int]) -> list[int]:
        offsets, running = [], 0
        for length in sample_lens:
            offsets.append(running)
            running += int(length)
        return offsets

    @staticmethod
    def _build_layout(
        *,
        split_lens: Sequence[int],
        attn_modes: Sequence[str],
        token_tags: torch.Tensor,
        text_len: int,
        sample_len: int,
        latent_shape: tuple[int, int, int, int],
        audio_shape: tuple[int, int, int],
    ) -> _Layout:
        """Recover the chunk table from the dataset's split structure.

        Everything here is *derived*, never re-derived from geometry: the
        dataset owns the chunking rule and duplicating it would let the two
        drift apart silently.
        """
        assert len(split_lens) == len(attn_modes), "split_lens/attn_modes disagree"
        assert attn_modes[0] == "full" and split_lens[0] == text_len, (
            "split 0 must be the text split, which is what makes the cache's "
            "chunk 0 the text and requires sink >= 1"
        )
        assert bool((token_tags[:text_len] == _TEXT_TAG).all()), "split 0 is not text rows"

        _, latent_t, latent_h, latent_w = latent_shape
        frame_rows = (latent_h // _PATCH_SIZE[1]) * (latent_w // _PATCH_SIZE[2])
        audio_t = audio_shape[2]

        media_splits = list(zip(split_lens[1:], attn_modes[1:]))
        teacher_forcing = media_splits[0][1] == "noise"
        chunks: list[dict[str, int | bool | None]] = []
        offset = text_len
        video_cursor = 0
        audio_cursor = 0
        for split_index, (length, mode) in enumerate(media_splits):
            length = int(length)
            tags = token_tags[offset : offset + length]
            audio_rows = int((tags == _AUDIO_TAG).sum())
            video_rows = length - audio_rows
            # The dataset lays a chunk out audio-first, then video. Assert it
            # rather than assume it: the row writers below depend on the order.
            assert bool((tags[:audio_rows] == _AUDIO_TAG).all()) and bool(
                (tags[audio_rows:] == _VIDEO_TAG).all()
            ), "a chunk split is not [audio rows | video rows]"
            assert video_rows % frame_rows == 0, (
                f"chunk video rows {video_rows} is not a multiple of frame rows {frame_rows}"
            )
            assert audio_rows % _AUDIO_CHANNELS == 0, (
                f"chunk audio rows {audio_rows} is not a multiple of {_AUDIO_CHANNELS}"
            )

            if teacher_forcing and mode != "noise":
                assert chunks and chunks[-1]["clean_start"] is None, (
                    "a clean split must directly follow its own noisy split"
                )
                assert (
                    chunks[-1]["audio_rows"] == audio_rows
                    and chunks[-1]["video_rows"] == video_rows
                ), "a clean split must mirror its noisy split row for row"
                chunks[-1]["clean_start"] = offset
                chunks[-1]["has_clean_copy"] = True
            else:
                video_len = video_rows // frame_rows
                audio_len = audio_rows // _AUDIO_CHANNELS
                chunks.append(
                    {
                        "noise_start": offset,
                        "clean_start": (
                            offset
                            if not teacher_forcing and split_index < len(media_splits) - 1
                            else None
                        ),
                        "has_clean_copy": False,
                        "audio_rows": audio_rows,
                        "video_rows": video_rows,
                        "video_start": video_cursor,
                        "video_stop": video_cursor + video_len,
                        "audio_start": audio_cursor,
                        "audio_stop": audio_cursor + audio_len,
                    }
                )
                video_cursor += video_len
                audio_cursor += audio_len
            offset += length

        assert offset == sample_len, f"split lengths sum to {offset}, expected {sample_len}"
        assert video_cursor == latent_t, (
            f"chunks cover {video_cursor} video latents, expected {latent_t}"
        )
        assert audio_cursor == audio_t, (
            f"chunks cover {audio_cursor} audio latents, expected {audio_t}"
        )
        assert chunks[-1]["clean_start"] is None, (
            "the last chunk must have no clean copy: nothing after it reads that history"
        )

        built = tuple(_Chunk(**chunk) for chunk in chunks)
        audio_row_perm = torch.cat(
            [
                torch.cat(
                    (
                        torch.arange(chunk.audio_start, chunk.audio_stop, dtype=torch.long),
                        torch.arange(
                            audio_t + chunk.audio_start,
                            audio_t + chunk.audio_stop,
                            dtype=torch.long,
                        ),
                    )
                )
                for chunk in built
            ]
        )
        return _Layout(
            text_len=text_len,
            chunks=built,
            audio_row_perm=audio_row_perm,
            latent_shape=latent_shape,
            audio_shape=audio_shape,
        )

    @staticmethod
    def _check_layout_agrees(
        layouts: Sequence[_Layout],
        offsets: Sequence[int],
        img_pos: torch.Tensor,
        audio_pos: torch.Tensor,
        token_tags: torch.Tensor,
    ) -> None:
        """Pin the derived chunk table against the dataset's own row indices.

        The row writers below reconstruct ``img_pos``/``audio_pos`` from the
        chunk table. If the dataset ever reorders a split, that reconstruction
        would silently write latents to the wrong rows, so compare once.
        """
        derived_img, derived_audio = [], []
        for layout, offset in zip(layouts, offsets):
            for chunk in layout.chunks:
                starts = (
                    (chunk.noise_start, chunk.clean_start)
                    if chunk.has_clean_copy
                    else (chunk.noise_start,)
                )
                for start in starts:
                    base = offset + start
                    derived_audio.append(
                        torch.arange(base, base + chunk.audio_rows, dtype=torch.long)
                    )
                    derived_img.append(
                        torch.arange(
                            base + chunk.audio_rows, base + chunk.rows, dtype=torch.long
                        )
                    )
        derived_img = torch.cat(derived_img).to(img_pos.device)
        derived_audio = torch.cat(derived_audio).to(audio_pos.device)
        assert torch.equal(derived_img, img_pos.to(torch.long)), (
            "derived video rows disagree with the dataset's img_pos"
        )
        assert torch.equal(derived_audio, audio_pos.to(torch.long)), (
            "derived audio rows disagree with the dataset's audio_pos"
        )
        assert bool((token_tags[img_pos.to(torch.long)] == _VIDEO_TAG).all())
        assert bool((token_tags[audio_pos.to(torch.long)] == _AUDIO_TAG).all())

    @classmethod
    def _to_device(cls, value: Any) -> Any:
        """Move every tensor leaf of a nested batch payload onto the run device."""
        if isinstance(value, torch.Tensor):
            return value.to(get_device())
        if isinstance(value, list):
            return [cls._to_device(item) for item in value]
        if isinstance(value, dict):
            return {key: cls._to_device(item) for key, item in value.items()}
        return value

    @staticmethod
    def _native_to_device(entry: dict[str, Any], device: torch.device) -> dict[str, Any]:
        return {
            key: value.to(device) if isinstance(value, torch.Tensor) else value
            for key, value in entry.items()
        }

    # ------------------------------------------------------------------
    # Row packing
    # ------------------------------------------------------------------

    @staticmethod
    def _video_rows(latent: torch.Tensor, start: int, stop: int) -> torch.Tensor:
        """[C, T, H, W] -> [(stop-start)*frame_rows, C*patch]."""
        return minimax_h3_patchify_video_latent(
            latent[:, start:stop].unsqueeze(0), patch_size=_PATCH_SIZE
        )

    @staticmethod
    def _audio_rows(latent: torch.Tensor, start: int, stop: int) -> torch.Tensor:
        """[2, C, A] -> [2*(stop-start), C] in channel-major [L | R] order."""
        return latent[:, :, start:stop].permute(0, 2, 1).reshape(-1, latent.shape[1])

    @staticmethod
    def _video_from_rows(rows: torch.Tensor, layout: _Layout) -> torch.Tensor:
        latent_t, patch_h, patch_w = layout.video_patch_grid
        return minimax_h3_unpatchify_video_tokens(
            rows,
            latent_shape=(latent_t, patch_h, patch_w, layout.latent_shape[0]),
            patch_size=_PATCH_SIZE,
        )[0]

    @staticmethod
    def _audio_from_rows(rows: torch.Tensor, layout: _Layout) -> torch.Tensor:
        return minimax_h3_unpack_audio_tokens(
            rows,
            audio_t=layout.audio_shape[2] * _AUDIO_CHANNELS,
            audio_channel=_AUDIO_CHANNELS,
        )

    def _chunk_audio_to_native(self, rows: torch.Tensor, layout: _Layout) -> torch.Tensor:
        """Undo the chunk-order [L_c | R_c] interleave into native [L(all) | R(all)]."""
        native = rows.new_empty(rows.shape)
        return native.index_copy(0, layout.audio_row_perm.to(rows.device), rows)

    # ------------------------------------------------------------------
    # Forward helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _num_layers(model: Any) -> int:
        assert hasattr(model, "dit"), (
            "MiniMax-H3 model nodes must be MiniMaxH3X0Model wrappers; the raw DiT "
            "returns velocity and would silently flip the x0 sign"
        )
        return len(model.dit.blocks)

    @staticmethod
    def _sp_size() -> int:
        """The active Ulysses group size, or 1 when SP is off."""
        if is_unified_parallel_initialized():
            return get_unified_parallel_world_size()
        return 1

    @classmethod
    def _pad_rows(cls, rows: int) -> int:
        """Rows to append so a packed sequence shards evenly across the SP group.

        Both H3 SP models refuse a ``seq_len`` that the Ulysses world size does
        not divide -- they shard rows and never pad on the caller's behalf. The
        native bidirectional layout is already 64-aligned, but the causal
        layouts come from the dataset's chunk structure and are arbitrary.

        When SP is active this is deliberately never zero: the padding travels
        as its own attention document (and, on the cached path, its own cache
        sample), and a zero-length varlen segment is a needless edge case for
        ``alignment`` rows of savings.
        """
        alignment = cls._sp_size()
        if alignment <= 1:
            return 0
        remainder = rows % alignment
        return alignment - remainder if remainder else alignment

    def _pad_for_sp(
        self,
        rows: int,
        *,
        video_blocks: list[torch.Tensor],
        audio_blocks: list[torch.Tensor],
        video_eps_blocks: list[torch.Tensor],
        audio_eps_blocks: list[torch.Tensor],
        row_timesteps: list[torch.Tensor],
        position_ids: torch.Tensor,
        token_tags: torch.Tensor,
        sample_lens: list[int],
        video_dim: int,
        audio_dim: int,
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        """Append the alignment document every *causal* packed forward needs under SP.

        The native bidirectional layout is already 64-aligned and needs none of
        this. The three causal paths must do it identically -- the one that
        forgets fails only under SP, and only at the rank count that happens to
        be unlucky -- so it lives here rather than three times over.

        The block lists are appended in place; the returned tensors replace the
        caller's position ids and token tags. Pad rows inherit dtype from the
        blocks they join: a hardcoded float32 would silently promote a whole
        bf16 pack through ``torch.cat``.
        """
        pad = self._pad_rows(rows)
        if pad == 0:
            return position_ids, token_tags, 0
        video_blocks.append(video_blocks[0].new_zeros((pad, video_dim)))
        audio_blocks.append(audio_blocks[0].new_zeros((pad, audio_dim)))
        video_eps_blocks.append(video_eps_blocks[0].new_zeros((pad, video_dim)))
        audio_eps_blocks.append(audio_eps_blocks[0].new_zeros((pad, audio_dim)))
        row_timesteps.append(row_timesteps[0].new_zeros(pad))
        sample_lens.append(pad)
        return (
            torch.cat([position_ids, position_ids.new_zeros((pad, 3))]),
            # -1 is the packer's padding tag; the model clamps it for AdaLN.
            torch.cat([token_tags, token_tags.new_full((pad,), -1)]),
            pad,
        )

    @staticmethod
    def _block_mask(
        inputs: ForwardInput,
        device: torch.device,
        *,
        sample_lens: list[int],
        split_lens: list[int],
        attn_modes: list[str],
    ) -> Any:
        """The torch flex-attention BlockMask, or None when magi consumes rectangles.

        Both backends are wired on purpose: ``q_ranges``/``k_ranges`` drive magi's
        ``flex_flash_attn_func`` and this mask drives ``torch.flex_attention``.
        Building the mask is the meta model's job -- ``create_sparse_mask`` owns
        the inductor fusion knob and has to be the one calling
        ``create_block_mask``.

        ``sink``/``window_size`` are passed unextended even when a padding
        document was appended: ``_packed_sparse_mask`` asserts they are uniform
        and reads element 0, so their length is not a document count.
        """
        from dev.yanzuolu.utils.flex_attn import FLEX_FLASH_ATTN_AVAILABLE

        if FLEX_FLASH_ATTN_AVAILABLE:
            return None
        return create_sparse_mask(
            sample_lens,
            split_lens,
            attn_modes,
            device,
            sink=inputs.sinks,
            window_size=inputs.window_sizes,
            block_size=FLEX_ATTN_BLOCK_SIZE,
        )

    def _refiner_params(self, inputs: ForwardInput) -> dict[str, Any]:
        live_lens = [length for length in inputs.text_lens if length > 0]
        cu = [0]
        for length in live_lens:
            cu.append(cu[-1] + length)
        return {
            "cu_seqlens_q": torch.tensor(
                cu, dtype=torch.int32, device=inputs.token_tags.device
            ),
            "max_seqlen_q": max(live_lens, default=0),
        }

    def _causal_packed_forward(
        self,
        model: Any,
        inputs: ForwardInput,
        *,
        video_xts: list[torch.Tensor],
        audio_xts: list[torch.Tensor],
        video_context: list[torch.Tensor],
        audio_context: list[torch.Tensor],
        video_eps: list[torch.Tensor],
        audio_eps: list[torch.Tensor],
        video_timesteps: list[torch.Tensor],
        audio_timesteps: list[torch.Tensor],
        text_timesteps: torch.Tensor | None = None,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """One packed teacher-forced causal forward over the dataset's layout.

        Chunks alternate ``noisy`` (queried, supervised) and ``clean`` (context,
        supplied by the caller -- rollout x0 for DMD, corpus x0 for a
        teacher-forced fit). Rows are assembled by concatenation rather than
        in-place writes so the generator graph is never routed through an
        index_put_.

        Video and audio timesteps are per sample AND per chunk: entry ``i``
        holds one value for each chunk of layout ``i``. Every noisy chunk of a
        sample is an independent denoising query, so nothing here requires them
        to agree; a caller that wants one timestep per sample expands it.
        ``text_timesteps`` optionally carries one live video timestep per sample.
        Its default keeps text rows at the repo clean timestep ``0``.
        """
        device = inputs.token_tags.device
        video_dim = inputs.layouts[0].latent_shape[0] * _PATCH_SIZE[1] * _PATCH_SIZE[2]
        audio_dim = inputs.layouts[0].audio_shape[1]
        # The row table is model conditioning, not schedule math: pin it to one
        # dtype so torch.cat cannot fail on a float64 timestep component.
        video_timesteps = [t.to(device=device, dtype=torch.float32) for t in video_timesteps]
        audio_timesteps = [t.to(device=device, dtype=torch.float32) for t in audio_timesteps]
        if text_timesteps is not None:
            text_timesteps = text_timesteps.to(device=device, dtype=torch.float32)

        video_blocks, audio_blocks = [], []
        video_eps_blocks, audio_eps_blocks = [], []
        row_timesteps = []

        def push(audio_rows: torch.Tensor, video_rows: torch.Tensor,
                 audio_eps_rows: torch.Tensor, video_eps_rows: torch.Tensor,
                 audio_t: torch.Tensor, video_t: torch.Tensor) -> None:
            n_audio = audio_rows.shape[0]
            n_video = video_rows.shape[0]
            video_blocks.append(video_rows.new_zeros((n_audio, video_dim)))
            video_blocks.append(video_rows)
            audio_blocks.append(audio_rows)
            audio_blocks.append(audio_rows.new_zeros((n_video, audio_dim)))
            video_eps_blocks.append(video_eps_rows.new_zeros((n_audio, video_dim)))
            video_eps_blocks.append(video_eps_rows)
            audio_eps_blocks.append(audio_eps_rows)
            audio_eps_blocks.append(audio_eps_rows.new_zeros((n_video, audio_dim)))
            row_timesteps.append(audio_t.expand(n_audio))
            row_timesteps.append(video_t.expand(n_video))

        zero = torch.zeros((), device=device, dtype=torch.float32)
        for i, layout in enumerate(inputs.layouts):
            # Text rows are embeddings rather than latents. Their row timestep
            # controls AdaLN conditioning without making these zero blocks live.
            text_rows = layout.text_len
            video_blocks.append(video_xts[i].new_zeros((text_rows, video_dim)))
            audio_blocks.append(audio_xts[i].new_zeros((text_rows, audio_dim)))
            video_eps_blocks.append(video_xts[i].new_zeros((text_rows, video_dim)))
            audio_eps_blocks.append(audio_xts[i].new_zeros((text_rows, audio_dim)))
            text_timestep = zero if text_timesteps is None else text_timesteps[i]
            row_timesteps.append(text_timestep.expand(text_rows))

            for chunk_index, chunk in enumerate(layout.chunks):
                push(
                    self._audio_rows(audio_xts[i], chunk.audio_start, chunk.audio_stop),
                    self._video_rows(video_xts[i], chunk.video_start, chunk.video_stop),
                    audio_xts[i].new_zeros((chunk.audio_rows, audio_dim)),
                    video_xts[i].new_zeros((chunk.video_rows, video_dim)),
                    audio_timesteps[i][chunk_index],
                    video_timesteps[i][chunk_index],
                )
                if not chunk.has_clean_copy:
                    continue
                push(
                    self._audio_rows(audio_context[i], chunk.audio_start, chunk.audio_stop),
                    self._video_rows(video_context[i], chunk.video_start, chunk.video_stop),
                    self._audio_rows(audio_eps[i], chunk.audio_start, chunk.audio_stop),
                    self._video_rows(video_eps[i], chunk.video_start, chunk.video_stop),
                    zero,
                    zero,
                )

        rows = sum(inputs.sample_lens)
        sample_lens = list(inputs.sample_lens)
        split_lens = [length for sample in inputs.split_lens for length in sample]
        attn_modes = [mode for sample in inputs.attn_modes for mode in sample]
        q_ranges, k_ranges = inputs.q_ranges, inputs.k_ranges
        attn_type_map, attn_workloads = inputs.attn_type_map, inputs.attn_workloads
        # The alignment rows are their own document, so they are invisible to
        # every real sample and never enter img_pos/audio_pos/text_pos.
        position_ids, token_tags, pad = self._pad_for_sp(
            rows,
            video_blocks=video_blocks, audio_blocks=audio_blocks,
            video_eps_blocks=video_eps_blocks, audio_eps_blocks=audio_eps_blocks,
            row_timesteps=row_timesteps,
            position_ids=inputs.position_ids, token_tags=inputs.token_tags,
            sample_lens=sample_lens,
            video_dim=video_dim, audio_dim=audio_dim,
        )
        if pad:
            split_lens.append(pad)
            attn_modes.append("causal")
            pad_q, pad_k, pad_type, pad_work = _prepare_flex_attention_mask(
                [pad], ["causal"], sink=inputs.sinks[0], window_size=inputs.window_sizes[0]
            )
            q_ranges = torch.cat([q_ranges, pad_q.to(device) + rows])
            k_ranges = torch.cat([k_ranges, pad_k.to(device) + rows])
            attn_type_map = torch.cat([attn_type_map, pad_type.to(device)])
            attn_workloads = list(attn_workloads) + [int(pad_work)]

        kwargs = self._common_kwargs(
            inputs,
            x=torch.cat(video_blocks).unsqueeze(0),
            audio_x=torch.cat(audio_blocks).unsqueeze(0),
            eps=torch.cat(video_eps_blocks).unsqueeze(0),
            audio_eps=torch.cat(audio_eps_blocks).unsqueeze(0),
            row_timesteps=torch.cat(row_timesteps),
            position_ids=position_ids,
            token_tags=token_tags,
            img_pos=inputs.img_pos,
            audio_pos=inputs.audio_pos,
            text_pos=inputs.text_pos,
            infer_out_pos=inputs.noisy_img_pos,
        )
        kwargs.update(
            sample_lens=sample_lens,
            q_ranges=q_ranges,
            k_ranges=k_ranges,
            attn_type_map=attn_type_map,
            attn_workloads=attn_workloads,
            attention_mask=self._block_mask(
                inputs, device,
                sample_lens=sample_lens, split_lens=split_lens, attn_modes=attn_modes,
            ),
        )
        video_logits, audio_logits = model(**kwargs)
        # Video came back already narrowed to the noisy rows; audio has no
        # separate output selector in the model, so filter it here.
        return self._split_causal_output(
            inputs, video_logits, audio_logits.index_select(0, inputs.audio_noisy_sel)
        )

    def _split_causal_output(
        self,
        inputs: ForwardInput,
        video_logits: torch.Tensor,
        audio_logits: torch.Tensor,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """Rows in chunk order -> per-sample native latents."""
        video_out, audio_out = [], []
        video_cursor, audio_cursor = 0, 0
        for layout in inputs.layouts:
            video_rows = sum(chunk.video_rows for chunk in layout.chunks)
            audio_rows = sum(chunk.audio_rows for chunk in layout.chunks)
            # Concatenating chunks in order already reproduces native frame-major
            # video; audio needs the stereo de-interleave.
            video_out.append(
                self._video_from_rows(
                    video_logits[video_cursor : video_cursor + video_rows], layout
                )
            )
            audio_out.append(
                self._audio_from_rows(
                    self._chunk_audio_to_native(
                        audio_logits[audio_cursor : audio_cursor + audio_rows], layout
                    ),
                    layout,
                )
            )
            video_cursor += video_rows
            audio_cursor += audio_rows
        return video_out, audio_out

    def _bidirectional_kwargs(
        self,
        model: Any,
        inputs: ForwardInput,
        *,
        video_xts: list[torch.Tensor],
        audio_xts: list[torch.Tensor],
        video_timesteps: torch.Tensor,
        audio_timesteps: torch.Tensor,
    ) -> tuple[dict[str, Any], torch.Tensor, torch.Tensor, torch.Tensor]:
        """Pack the native bidirectional layout into model kwargs."""
        device = inputs.token_tags.device
        video_dim = inputs.layouts[0].latent_shape[0] * _PATCH_SIZE[1] * _PATCH_SIZE[2]
        audio_dim = inputs.layouts[0].audio_shape[1]

        video_blocks, audio_blocks, row_timesteps = [], [], []
        img_pos, audio_pos, text_pos, cu = [], [], [], [0]
        offset = 0
        for i, native in enumerate(inputs.native):
            seq_len = int(native["seq_len"])
            sample_img = native["img_pos"].to(torch.long)
            sample_audio = native["audio_pos"].to(torch.long)
            video_blocks.append(
                video_xts[i]
                .new_zeros((seq_len, video_dim))
                .index_copy(0, sample_img, self._video_rows(video_xts[i], 0, video_xts[i].shape[1]))
            )
            audio_blocks.append(
                audio_xts[i]
                .new_zeros((seq_len, audio_dim))
                .index_copy(0, sample_audio, self._audio_rows(audio_xts[i], 0, audio_xts[i].shape[2]))
            )
            # Text (and padding) rows inherit the current video timestep,
            # matching sglang's denoise loop (_expand_step_timesteps leaves
            # every non-audio-target row on t_video). The released checkpoint
            # was trained with text AdaLN modulated by the live noise level --
            # pinning them to the clean 0.999 from step 0 (what a zero-filled
            # row would hit in the wrapper) degrades the text conditioning the
            # media rows attend to, and audio -- 640 rows leaning on text
            # alone -- collapses to a near-constant latent (narrowband tone).
            sample_t = torch.full(
                (seq_len,), float(video_timesteps[i]), dtype=torch.float32, device=device
            )
            sample_t[sample_img] = video_timesteps[i].to(torch.float32)
            sample_t[sample_audio] = audio_timesteps[i].to(torch.float32)
            row_timesteps.append(sample_t)

            img_pos.append(sample_img + offset)
            audio_pos.append(sample_audio + offset)
            text_pos.append(native["text_pos"].to(torch.long) + offset)
            # Each sample contributes its live document and its alignment
            # padding as two varlen segments, so padding never attends live rows.
            cu.extend((native["cu_seqlens"][1:].to(torch.long) + offset).tolist())
            offset += seq_len

        row_timesteps = torch.cat(row_timesteps)
        img_pos = torch.cat(img_pos)
        audio_pos = torch.cat(audio_pos)
        x = torch.cat(video_blocks).unsqueeze(0)
        audio_x = torch.cat(audio_blocks).unsqueeze(0)
        # Cast media inputs to the model's runtime param dtype (bf16 in the
        # trials): on the no-autocast inference path fp32-drawn latents would
        # promote the whole qkv/FFN compute to fp32. Timesteps stay fp32 --
        # H3's time embedding is explicitly fp32 (the vendored model builds it
        # in _FP32_DTYPE), so only the media rows are touched. Under the
        # training autocast wrapper this is the same cast it would make anyway.
        x = x.to(dtype=model.param_dtype)
        audio_x = audio_x.to(dtype=model.param_dtype)
        cu_tensor = torch.tensor(cu, dtype=torch.int32, device=device)
        kwargs = self._common_kwargs(
            inputs,
            x=x,
            audio_x=audio_x,
            eps=torch.zeros_like(x),
            audio_eps=torch.zeros_like(audio_x),
            row_timesteps=row_timesteps,
            position_ids=torch.cat(
                [native["img_position_ids"] for native in inputs.native], dim=0
            ),
            token_tags=torch.cat([native["token_tags"] for native in inputs.native]),
            img_pos=img_pos,
            audio_pos=audio_pos,
            text_pos=torch.cat(text_pos),
            infer_out_pos=img_pos,
        )
        kwargs["packed_seq_params"] = {
            "cu_seqlens_q": cu_tensor,
            "max_seqlen_q": int((cu_tensor[1:] - cu_tensor[:-1]).max()),
        }
        return kwargs, row_timesteps, img_pos, audio_pos

    def _bidirectional_forward(
        self,
        model: Any,
        inputs: ForwardInput,
        *,
        video_xts: list[torch.Tensor],
        audio_xts: list[torch.Tensor],
        video_timesteps: torch.Tensor,
        audio_timesteps: torch.Tensor,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """One dense forward per critic/teacher over the native H3 layout.

        Causality is a generation-time constraint, not a scoring one: the whole
        clip, including its optional ragged 2-latent tail, is scored at once
        with full attention over ``[text | all audio | all video]``.
        """
        kwargs, row_timesteps, img_pos, audio_pos = self._bidirectional_kwargs(
            model,
            inputs,
            video_xts=video_xts,
            audio_xts=audio_xts,
            video_timesteps=video_timesteps,
            audio_timesteps=audio_timesteps,
        )
        # Nothing read by this forward is a context row, so the clean-noise path
        # inside the wrapper is dead here and a zero eps is honest rather than
        # merely convenient. Assert it instead of assuming it.
        assert bool((row_timesteps[img_pos] != 0).all()) and bool(
            (row_timesteps[audio_pos] != 0).all()
        ), "a bidirectional score row carries the clean timestep 0"
        video_logits, audio_logits = model(**kwargs)

        video_out, audio_out = [], []
        video_cursor, audio_cursor = 0, 0
        for i, layout in enumerate(inputs.layouts):
            video_rows = int(inputs.native[i]["img_pos"].numel())
            audio_rows = int(inputs.native[i]["audio_pos"].numel())
            video_out.append(
                self._video_from_rows(
                    video_logits[video_cursor : video_cursor + video_rows], layout
                )
            )
            audio_out.append(
                self._audio_from_rows(
                    audio_logits[audio_cursor : audio_cursor + audio_rows], layout
                )
            )
            video_cursor += video_rows
            audio_cursor += audio_rows
        return video_out, audio_out

    def _text_cache_fill(
        self, model: Any, inputs: ForwardInput, cache: NaiveCache
    ) -> None:
        """Write text as cache chunk 0 and remove temporary empty-sample rows."""
        device = inputs.token_tags.device
        video_dim = inputs.layouts[0].latent_shape[0] * _PATCH_SIZE[1] * _PATCH_SIZE[2]
        audio_dim = inputs.layouts[0].audio_shape[1]

        position_ids, text_pos, tag_blocks = [], [], []
        sample_offsets = self._offsets(inputs.sample_lens)
        sample_lens = []
        empty_samples = []
        offset = 0
        for i, layout in enumerate(inputs.layouts):
            start = sample_offsets[i]
            if layout.text_len > 0:
                position_ids.append(
                    inputs.position_ids[start : start + layout.text_len]
                )
                text_pos.append(
                    torch.arange(
                        offset,
                        offset + layout.text_len,
                        device=device,
                        dtype=torch.long,
                    )
                )
                tag_blocks.append(
                    torch.full(
                        (layout.text_len,), _TEXT_TAG, device=device, dtype=torch.long
                    )
                )
                sample_len = layout.text_len
            else:
                position_ids.append(inputs.position_ids.new_zeros((1, 3)))
                text_pos.append(torch.empty(0, device=device, dtype=torch.long))
                tag_blocks.append(torch.full((1,), -1, device=device, dtype=torch.long))
                empty_samples.append(i)
                sample_len = 1
            sample_lens.append(sample_len)
            offset += sample_len

        video_blocks = [torch.zeros((offset, video_dim), device=device)]
        audio_blocks = [torch.zeros((offset, audio_dim), device=device)]
        video_eps_blocks = [torch.zeros((offset, video_dim), device=device)]
        audio_eps_blocks = [torch.zeros((offset, audio_dim), device=device)]
        row_timesteps = [torch.zeros(offset, device=device, dtype=torch.float32)]
        packed_position_ids, token_tags, _ = self._pad_for_sp(
            offset,
            video_blocks=video_blocks, audio_blocks=audio_blocks,
            video_eps_blocks=video_eps_blocks, audio_eps_blocks=audio_eps_blocks,
            row_timesteps=row_timesteps,
            position_ids=torch.cat(position_ids, dim=0),
            token_tags=torch.cat(tag_blocks),
            sample_lens=sample_lens,
            video_dim=video_dim, audio_dim=audio_dim,
        )

        kwargs = self._common_kwargs(
            inputs,
            x=torch.cat(video_blocks).unsqueeze(0),
            audio_x=torch.cat(audio_blocks).unsqueeze(0),
            eps=torch.cat(video_eps_blocks).unsqueeze(0),
            audio_eps=torch.cat(audio_eps_blocks).unsqueeze(0),
            row_timesteps=torch.cat(row_timesteps),
            position_ids=packed_position_ids,
            token_tags=token_tags,
            img_pos=torch.empty(0, device=device, dtype=torch.long),
            audio_pos=torch.empty(0, device=device, dtype=torch.long),
            text_pos=torch.cat(text_pos),
            infer_out_pos=torch.empty(0, device=device, dtype=torch.long),
        )
        kwargs.update(
            sample_lens=sample_lens,
            past_key_values=cache,
            update_past_key_values=True,
        )
        model(**kwargs)

        if not empty_samples:
            return
        # Eviction may have removed text or dummy rows during this prefill.
        cache_offsets = self._offsets(cache.kvlens)
        dummy_rows = [
            cache_offsets[sample] for sample in empty_samples if cache.kvlens[sample]
        ]
        if dummy_rows:
            for layer in range(cache.num_layers):
                keys = cache.key_cache[layer]
                values = cache.value_cache[layer]
                keep = torch.ones(keys.shape[0], dtype=torch.bool, device=keys.device)
                keep[torch.tensor(dummy_rows, device=keys.device)] = False
                cache.key_cache[layer] = keys[keep]
                cache.value_cache[layer] = values[keep]
        for sample in empty_samples:
            cache.kvlens[sample] = 0
            cache.chunk_lens[-1][sample] = 0

    def _chunk_forward(
        self,
        model: Any,
        inputs: ForwardInput,
        *,
        chunk_index: int,
        role: str,
        video_rows_source: list[torch.Tensor],
        audio_rows_source: list[torch.Tensor],
        video_timesteps: torch.Tensor,
        audio_timesteps: torch.Tensor,
        cache: NaiveCache,
        update_cache: bool,
        video_eps_source: list[torch.Tensor] | None = None,
        audio_eps_source: list[torch.Tensor] | None = None,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """One cached forward over chunk ``chunk_index`` only.

        ``role='noise'`` denoises against the cache without touching it;
        ``role='clean'`` re-runs the same rows carrying the student's own x0 at
        the repo clean timestep and writes their K/V into the cache.

        ``text_pos`` is empty here because the text lives in the cache, but
        ``_embed`` still runs the text refiner over the whole prompt on every
        call -- it derives the live length from ``refiner_cu_seqlens``, not from
        the row map. Two layers over the prompt against fifty over the chunk, so
        it is left alone; it is a real all-gather per call if the refiner is
        FSDP-sharded.
        """
        device = inputs.token_tags.device
        video_dim = inputs.layouts[0].latent_shape[0] * _PATCH_SIZE[1] * _PATCH_SIZE[2]
        audio_dim = inputs.layouts[0].audio_shape[1]
        video_timesteps = video_timesteps.to(device=device, dtype=torch.float32)
        audio_timesteps = audio_timesteps.to(device=device, dtype=torch.float32)

        video_blocks, audio_blocks = [], []
        video_eps_blocks, audio_eps_blocks = [], []
        row_timesteps, position_ids, token_tags = [], [], []
        img_pos, audio_pos, sample_lens = [], [], []
        sample_offsets = self._offsets(inputs.sample_lens)
        cursor = 0
        for i, layout in enumerate(inputs.layouts):
            chunk = layout.chunks[chunk_index]
            start = chunk.noise_start if role == "noise" else chunk.clean_start
            assert start is not None, "requested the clean copy of a chunk that has none"
            base = sample_offsets[i] + start

            audio_rows = self._audio_rows(
                audio_rows_source[i], 0, audio_rows_source[i].shape[2]
            )
            video_rows = self._video_rows(video_rows_source[i], 0, video_rows_source[i].shape[1])
            assert audio_rows.shape[0] == chunk.audio_rows
            assert video_rows.shape[0] == chunk.video_rows

            video_blocks.append(video_rows.new_zeros((chunk.audio_rows, video_dim)))
            video_blocks.append(video_rows)
            audio_blocks.append(audio_rows)
            audio_blocks.append(audio_rows.new_zeros((chunk.video_rows, audio_dim)))
            if video_eps_source is None:
                video_eps_blocks.append(video_rows.new_zeros((chunk.rows, video_dim)))
                audio_eps_blocks.append(audio_rows.new_zeros((chunk.rows, audio_dim)))
            else:
                video_eps_rows = self._video_rows(video_eps_source[i], 0, video_eps_source[i].shape[1])
                audio_eps_rows = self._audio_rows(
                    audio_eps_source[i], 0, audio_eps_source[i].shape[2]
                )
                video_eps_blocks.append(video_eps_rows.new_zeros((chunk.audio_rows, video_dim)))
                video_eps_blocks.append(video_eps_rows)
                audio_eps_blocks.append(audio_eps_rows)
                audio_eps_blocks.append(audio_eps_rows.new_zeros((chunk.video_rows, audio_dim)))

            row_timesteps.append(audio_timesteps[i].expand(chunk.audio_rows))
            row_timesteps.append(video_timesteps[i].expand(chunk.video_rows))
            position_ids.append(inputs.position_ids[base : base + chunk.rows])
            token_tags.append(inputs.token_tags[base : base + chunk.rows])
            audio_pos.append(
                torch.arange(cursor, cursor + chunk.audio_rows, device=device, dtype=torch.long)
            )
            img_pos.append(
                torch.arange(
                    cursor + chunk.audio_rows, cursor + chunk.rows, device=device, dtype=torch.long
                )
            )
            sample_lens.append(chunk.rows)
            cursor += chunk.rows

        img_pos = torch.cat(img_pos)
        audio_pos = torch.cat(audio_pos)
        packed_position_ids, packed_token_tags, _ = self._pad_for_sp(
            cursor,
            video_blocks=video_blocks, audio_blocks=audio_blocks,
            video_eps_blocks=video_eps_blocks, audio_eps_blocks=audio_eps_blocks,
            row_timesteps=row_timesteps,
            position_ids=torch.cat(position_ids, dim=0),
            token_tags=torch.cat(token_tags),
            sample_lens=sample_lens,
            video_dim=video_dim, audio_dim=audio_dim,
        )
        kwargs = self._common_kwargs(
            inputs,
            x=torch.cat(video_blocks).unsqueeze(0),
            audio_x=torch.cat(audio_blocks).unsqueeze(0),
            eps=torch.cat(video_eps_blocks).unsqueeze(0),
            audio_eps=torch.cat(audio_eps_blocks).unsqueeze(0),
            row_timesteps=torch.cat(row_timesteps),
            position_ids=packed_position_ids,
            token_tags=packed_token_tags,
            img_pos=img_pos,
            audio_pos=audio_pos,
            # The text lives in the cache, not in this forward's rows.
            text_pos=torch.empty(0, device=device, dtype=torch.long),
            infer_out_pos=img_pos,
        )
        kwargs.update(
            sample_lens=sample_lens,
            past_key_values=cache,
            update_past_key_values=update_cache,
        )
        video_logits, audio_logits = model(**kwargs)
        if role == "clean":
            # A cache-fill pass exists only for the K/V it writes; unpatchifying
            # its logits would be an einsum + contiguous per sample per chunk
            # that the caller drops on the floor.
            return [], []

        video_out, audio_out = [], []
        video_cursor, audio_cursor = 0, 0
        for i, layout in enumerate(inputs.layouts):
            chunk = layout.chunks[chunk_index]
            video_out.append(
                minimax_h3_unpatchify_video_tokens(
                    video_logits[video_cursor : video_cursor + chunk.video_rows],
                    latent_shape=(
                        chunk.video_stop - chunk.video_start,
                        layout.video_patch_grid[1],
                        layout.video_patch_grid[2],
                        layout.latent_shape[0],
                    ),
                    patch_size=_PATCH_SIZE,
                )[0]
            )
            audio_out.append(
                minimax_h3_unpack_audio_tokens(
                    audio_logits[audio_cursor : audio_cursor + chunk.audio_rows],
                    audio_t=chunk.audio_rows,
                    audio_channel=_AUDIO_CHANNELS,
                )
            )
            video_cursor += chunk.video_rows
            audio_cursor += chunk.audio_rows
        return video_out, audio_out

    def _common_kwargs(
        self,
        inputs: ForwardInput,
        *,
        x: torch.Tensor,
        audio_x: torch.Tensor,
        eps: torch.Tensor,
        audio_eps: torch.Tensor,
        row_timesteps: torch.Tensor,
        position_ids: torch.Tensor,
        token_tags: torch.Tensor,
        img_pos: torch.Tensor,
        audio_pos: torch.Tensor,
        text_pos: torch.Tensor,
        infer_out_pos: torch.Tensor,
    ) -> dict[str, Any]:
        """The kwargs every H3 forward shares, with the repo timestep table.

        ``unique_timesteps``/``inverse_indices`` are built by laying the repo
        timestep out per row and then de-duplicating. The slots carry no
        semantics -- they are just the distinct values -- which is exactly what
        lets one pack hold a clean 0, a per-sample video sigma and a different
        per-sample audio sigma at the same time.
        """
        unique_timesteps, inverse_indices = torch.unique(
            row_timesteps, return_inverse=True
        )
        return {
            "x": x,
            "audio_x": audio_x,
            "eps": eps,
            "audio_eps": audio_eps,
            "img_position_ids": position_ids.unsqueeze(0),
            "unique_timesteps": unique_timesteps,
            "inverse_indices": inverse_indices,
            "token_tags": token_tags,
            "update_mask": torch.ones(
                infer_out_pos.numel(), dtype=torch.bool, device=x.device
            ),
            "prompt_embeds": torch.cat(inputs.prompt_embeds, dim=0),
            "img_pos_info": {"position_ids": img_pos},
            "audio_pos_info": {"position_ids": audio_pos},
            "text_pos_info": {"position_ids": text_pos},
            "img_pos_for_infer_output_info": {"position_ids": infer_out_pos},
            "refiner_packed_seq_params": self._refiner_params(inputs),
        }

    # ------------------------------------------------------------------
    # Text, noise, timesteps
    # ------------------------------------------------------------------

    def _encode_prompts(
        self,
        models: dict[str, Any],
        text_input_ids: Sequence[torch.Tensor],
        text_lens: Sequence[int],
    ) -> list[torch.Tensor]:
        """Exactly one encoder call per batch, right-padded and masked.

        Not one call per sample: the dataset packs a variable number of samples,
        so a per-sample loop makes the number of encoder forwards rank-dependent,
        and an FSDP-sharded text encoder would deadlock on the first
        micro-iteration where two ranks disagree. One call per batch is
        rank-invariant however the packs differ.

        Each zero-length caption is encoded as one masked-in stand-in token and
        sliced back to zero rows before the DiT. Which hidden layer conditions
        the DiT is the encoder node's business, not this one's:
        ``MiniMaxH3TextEncoder`` returns it directly.
        """
        device = get_device()
        lens = [int(length) for length in text_lens]
        safe_lens = [max(length, 1) for length in lens]
        longest = max(safe_lens)
        ids = torch.zeros((len(lens), longest), dtype=torch.long, device=device)
        mask = torch.zeros((len(lens), longest), dtype=torch.long, device=device)
        for index, (sample_ids, length, safe_length) in enumerate(
            zip(text_input_ids, lens, safe_lens, strict=True)
        ):
            sample_ids = sample_ids.view(-1)
            assert int(sample_ids.numel()) == length, (
                f"text_input_ids has {int(sample_ids.numel())} tokens but "
                f"text_lens says {length}"
            )
            ids[index, :length] = sample_ids.to(device)
            mask[index, :safe_length] = 1
        hidden = models["text_encoder"](input_ids=ids, attention_mask=mask)
        # Right padding plus the mask keeps the live rows identical to what a
        # single-sample call would have produced. .contiguous() rather than a
        # view: the view would pin the whole [B, longest, D] storage for the life
        # of the offline pool, and sync_inputs broadcasts these through NCCL.
        return [
            hidden[index, :length].contiguous()
            for index, length in enumerate(lens)
        ]

    def _sample_noises(
        self, inputs: ForwardInput, rng: Any
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        device = get_device()
        video = [
            torch.empty(layout.latent_shape, device=device).normal_(
                generator=self._generator(rng)
            )
            for layout in inputs.layouts
        ]
        audio = [
            torch.empty(layout.audio_shape, device=device).normal_(
                generator=self._generator(rng)
            )
            for layout in inputs.layouts
        ]
        return video, audio

    @staticmethod
    def _generator(rng: Any) -> torch.Generator:
        """The rng's generator for the current device.

        ``RandomState.torch_cuda_generator`` is None without CUDA, and
        ``normal_(generator=None)`` would silently fall back to the global
        stream and break resume determinism.
        """
        device = get_device()
        return rng.torch_cuda_generator if device.type == "cuda" else rng.torch_generator

    @staticmethod
    def _sample_paired_timesteps(
        batch_size: int, seqlens: torch.Tensor, video: Any, audio: Any, rng: Any
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """One uniform draw, pushed through each modality's own shift.

        Takes the two numbers it needs rather than a ``ForwardInput``: a caller
        that draws one timestep per chunk has no such object to hand over.

        Neither two independent draws nor one shared sigma. H3's sampler advances
        a shared step index through two shifted grids, so along its own
        trajectory ``sigma_video != sigma_audio`` everywhere except the
        endpoints -- a shared sigma is just as far off that curve as an
        independent draw is. Sharing the underlying uniform and letting each
        component apply its own shift lands exactly on the pairs the frozen
        teacher has actually seen.

        Drawing the uniform here rather than calling ``sample`` twice is what
        makes the sharing explicit: two ``sample`` calls would consume the RNG
        twice and silently decouple.
        """
        with local_seed(rng.seed % 2**31):
            uniform = torch.rand((batch_size,), dtype=torch.float64)
        rng.seed = yield_seed(rng.seed)
        device = get_device()
        return (
            video.postprocess_sample(uniform.clone(), seqlens, device),
            audio.postprocess_sample(uniform.clone(), seqlens, device),
        )

    @staticmethod
    def _sample_timesteps(inputs: ForwardInput, timesteps: Any, rng: Any) -> torch.Tensor:
        """Draw per-sample timesteps, advancing the phase rng deterministically."""
        with local_seed(rng.seed % 2**31):
            drawn = timesteps.sample(
                size=(inputs.batch_size,), seqlens=inputs.seqlens, device=get_device()
            )
        rng.seed = yield_seed(rng.seed)
        return drawn

    # ------------------------------------------------------------------
    # Losses
    # ------------------------------------------------------------------

    @staticmethod
    def _chunk_slice(chunk: _Chunk, modality: str) -> tuple[slice, ...]:
        """Index one chunk's own entries out of a native latent tensor."""
        if modality == "video":  # [C, T, H, W]
            return (slice(None), slice(chunk.video_start, chunk.video_stop))
        return (slice(None), slice(None), slice(chunk.audio_start, chunk.audio_stop))

    def _loss_fn(
        self,
        video_preds: list[torch.Tensor],
        audio_preds: list[torch.Tensor],
        *,
        inputs: ForwardInput,
        video_targets: list[torch.Tensor] | None = None,
        audio_targets: list[torch.Tensor] | None = None,
        key_prefix: str,
    ) -> torch.Tensor:
        """World-normalized per-modality reduction, combined by audio_loss_weight.

        Each modality is divided by its OWN global element count. Dividing both
        by a joint count is what would hand audio 0.4% of the gradient at H3's
        256:1 element ratio.

        ``chunk_wise_weighting`` is orthogonal to that: it scales chunks, never
        modalities. One weight curve is derived per sample from the COMBINED
        video+audio element count of each chunk and applied to both, so the
        temporal shape is identical across modalities and only the per-modality
        denominator differs.
        """
        device = get_device()
        alpha = (
            None
            if self.chunk_wise_weighting is None
            else self.chunk_wise_weighting.get(key_prefix)
        )
        if self.chunk_wise_weighting is not None and alpha is None:
            if key_prefix not in self._warned_weighting:
                self._warned_weighting.add(key_prefix)
                logger.warning("chunk_wise_weighting has no entry for %s", key_prefix)

        weights = (
            None
            if alpha is None
            else [self._chunk_weights(layout, float(alpha), device) for layout in inputs.layouts]
        )
        video_sum, video_count = self._reduce_modality(
            video_preds, video_targets, inputs=inputs, modality="video", weights=weights
        )
        audio_sum, audio_count = self._reduce_modality(
            audio_preds, audio_targets, inputs=inputs, modality="audio", weights=weights
        )

        counts = torch.stack([video_count, audio_count])
        all_reduce_sum(counts)
        world_size = get_world_size()
        video_loss = video_sum / counts[0] * world_size
        audio_loss = audio_sum / counts[1] * world_size

        meter = get_running_average_meter()
        meter.put_scalar(f"running/{key_prefix}/video", float(video_loss.detach().item()))
        meter.put_scalar(f"running/{key_prefix}/audio", float(audio_loss.detach().item()))
        return video_loss + self.audio_loss_weight * audio_loss

    @staticmethod
    def _chunk_masses_and_participation(
        layout: _Layout, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return H3 chunk element masses and normalized suffix participation."""
        masses = torch.tensor(
            [
                (chunk.video_stop - chunk.video_start)
                * layout.latent_shape[0]
                * layout.latent_shape[2]
                * layout.latent_shape[3]
                + (chunk.audio_stop - chunk.audio_start)
                * _AUDIO_CHANNELS
                * layout.audio_shape[1]
                for chunk in layout.chunks
            ],
            dtype=torch.float64,
            device=device,
        )
        participation = torch.flip(
            torch.cumsum(torch.flip(masses, dims=[0]), dim=0), dims=[0]
        )
        participation = participation / participation[0].clamp_min(1.0)
        return masses, participation

    def _chunk_weights(
        self, layout: _Layout, alpha: float, device: torch.device
    ) -> torch.Tensor:
        """Per-chunk weights from participation, renormalized to conserve mass.

        Same curve as wan's: a chunk's ``participation`` is the share of the clip
        that still lies at or after it, and a negative alpha reverses it so later
        chunks -- the ones with the least history to lean on -- weigh more. The
        rescale keeps the weighted element count equal to the raw one, so the
        knob changes the distribution over time and not the loss magnitude.
        """
        masses, participation = self._chunk_masses_and_participation(layout, device)
        if alpha > 0:
            raw = alpha * participation / (1 + (alpha - 1) * participation)
        elif alpha < 0:
            reverse = participation[-1] / participation.clamp_min(1e-6)
            shift = -alpha
            raw = shift * reverse / (1 + (shift - 1) * reverse)
        else:
            raw = torch.ones_like(participation)
        weighted = (raw * masses).sum()
        if weighted <= 0:
            raise ValueError("chunk_wise_weighting produced a non-positive total weight")
        return raw * (masses.sum() / weighted)

    def _reduce_modality(
        self,
        preds: list[torch.Tensor],
        targets: list[torch.Tensor] | None,
        *,
        inputs: ForwardInput,
        modality: str,
        weights: list[torch.Tensor] | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """(summed loss, element count) for one modality, optionally chunk-weighted."""
        device = get_device()
        total = torch.zeros((), dtype=torch.float64, device=device)
        count = torch.zeros((), dtype=torch.float64, device=device)
        for index, pred in enumerate(preds):
            target = None if targets is None else targets[index]
            if weights is None:
                total = total + self._elementwise(pred, target).sum().double()
                count = count + pred.numel()
                continue
            for weight, chunk in zip(weights[index], inputs.layouts[index].chunks):
                chunk_slice = self._chunk_slice(chunk, modality)
                piece = pred[chunk_slice]
                term = self._elementwise(
                    piece, None if target is None else target[chunk_slice]
                )
                total = total + weight * term.sum().double()
                count = count + weight * piece.numel()
        return total, count

    @staticmethod
    def _elementwise(
        pred: torch.Tensor, target: torch.Tensor | None
    ) -> torch.Tensor:
        """Precomputed per-element loss (target None) or a supervised MSE."""
        if target is None:
            return pred
        return F.mse_loss(pred.float(), target.float().detach(), reduction="none")
