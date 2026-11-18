# SPDX-License-Identifier: Apache-2.0
"""Bidirectional streaming-window supervision and stateful T2AV sampling."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, replace
from fractions import Fraction
import math
from typing import Any

import torch

from dev.yanzuolu.common.diffusion.schedule.lerp import LinearInterpolationSchedule
from dev.yanzuolu.common.distributed.ops import all_reduce_max, all_reduce_sum, get_device, get_world_size
from dev.yanzuolu.common.distributed.unified_parallel import (
    SPDistForward,
    get_unified_parallel_world_size,
    is_unified_parallel_initialized,
)
from dev.yanzuolu.common.phase import ExecutionPhase, execution_phase
from dev.yanzuolu.common.seed import RandomState, combine_seed, local_seed, yield_seed
from dev.yanzuolu.projects.minimax_h3.meta_models.causal_minimax_h3_base import (
    CausalMiniMaxH3Base,
)
from dev.yanzuolu.projects.minimax_h3.meta_models.streaming_guidance import (
    GUIDANCE_CONDITIONS,
    StreamingGuidanceBranch,
    StreamingGuidanceStage,
    compile_guidance,
    noised_target_history,
    parse_guidance_branches,
    parse_guidance_fitting,
    possible_guidance_uses,
    without_target_history,
)
from dev.yanzuolu.projects.minimax_h3.modeling.time_request import (
    VideoTemporalMapping,
    video_temporal_mapping_from_model_config,
)
from dev.yanzuolu.projects.minimax_h3.modeling.transformer.x0_model import (
    MINIMAX_H3_VIDEO_CLEAN_TIMESTEP,
)
from dev.yanzuolu.projects.minimax_h3.data.streaming import (
    build_streaming_plan,
    streaming_target_shapes,
    streaming_window_starts,
    streaming_window_stop,
)
from dev.yanzuolu.projects.minimax_h3.data.caption_timeline import caption_index_for_interval
from dev.yanzuolu.projects.minimax_h3.modeling.streaming import (
    StreamingForwardMixin,
    StreamingInputs,
)
from dev.yanzuolu.projects.minimax_h3.modeling.streaming_output import (
    StreamingAVChunk,
    StreamingAVDecoder,
)
from dev.yanzuolu.projects.minimax_h3.modeling.tokenizer import MiniMaxH3Tokenizer


@dataclass(frozen=True)
class StreamingBatch:
    """Complete AV geometry and optional reference timelines before selection."""

    prompt_embeds: list[torch.Tensor]
    reference_latents: list[torch.Tensor] | None
    video_shapes: list[tuple[int, int, int, int]]
    audio_shapes: list[tuple[int, int, int]]
    streaming_configs: list[dict[str, int | None]]
    prompts: list[str] | None = None
    reference_pixels: list[torch.Tensor] | None = None
    text_token_tags: list[torch.Tensor] | None = None
    negative_prompt_embeds: list[torch.Tensor] | None = None
    negative_text_token_tags: list[torch.Tensor] | None = None
    text_only_prompt_embeds: list[torch.Tensor] | None = None
    caption_segments: list[list[dict[str, Any]] | None] | None = None
    caption_embeds: list[list[torch.Tensor] | None] | None = None

    @property
    def batch_size(self) -> int:
        return len(self.prompt_embeds)


@dataclass(frozen=True)
class StreamingLatentChunk:
    """New frames for active samples, addressed on their complete AV timelines."""

    sample_indices: list[int]
    video_indices: list[torch.Tensor]
    audio_indices: list[torch.Tensor]
    video: list[torch.Tensor]
    audio: list[torch.Tensor]
    is_last: list[bool]
    audio_lookahead: list[torch.Tensor] | None = None
    audio_lookahead_indices: list[torch.Tensor] | None = None
    audio_new_indices: list[torch.Tensor] | None = None
    decoded: list[list[StreamingAVChunk]] | None = None


@dataclass(frozen=True)
class StreamingRollout:
    """Complete normalized video and stereo audio in original sample order."""

    video: list[torch.Tensor]
    audio: list[torch.Tensor]
    chunks: list[StreamingLatentChunk] | None = None
    audio_lookahead_latents: list[int] | None = None
    audio_right_lookahead_latents: list[int] | None = None


@dataclass
class StreamingState:
    """Bounded AV history and a continuous uncommitted audio tail."""

    sample_index: int
    prompt_branches: list[tuple[float, torch.Tensor]]
    video_geometry: tuple[int, int, int]
    reference_geometry: tuple[int, int, int] | None
    audio_channels: int
    streaming_config: dict[str, int | None]
    rng: Any
    audio_length: int | None
    video_history: torch.Tensor
    audio_history: torch.Tensor
    audio_future: torch.Tensor
    reference_history: torch.Tensor | None
    video_history_indices: torch.Tensor
    audio_history_indices: torch.Tensor
    reference_history_indices: torch.Tensor | None
    next_video: int = 0
    next_audio: int = 0
    audio_generated_stop: int = 0
    finished: bool = False
    video_length: int | None = None
    prompt_texts: list[str] | None = None
    prompt_branch_tags: list[torch.Tensor] | None = None
    reference_pixel_history: torch.Tensor | None = None
    reference_pixel_indices: torch.Tensor | None = None


class MiniMaxH3StreamingSFT(StreamingForwardMixin, CausalMiniMaxH3Base):
    """Fit dense windows of clean AV history, noisy targets and optional references.

    The first window supervises ``bootstrap_size`` target units, W + C when
    omitted. Later windows read the available sink prefix and latest W clean
    units, without duplicates, and supervise the next C. Each unit is one
    native latent by default. Optional single-frame merging groups the
    native H3 cadence into (2, 1, 1, 1) latent spans without changing codec
    geometry or public latent indices.
    Optional sink reduction begins at a configured
    1-based continuation index. Completed windows retain the next window's
    required history after their own denoising finishes.
    Optional references follow the selected target timeline with their own spatial grid.
    Audio is generated once, up to ``audio_right_lookahead_latents`` ahead of
    the current video boundary. When omitted, it follows ``audio_lookahead_latents``,
    the decoder's left context length. Later windows reuse those values unchanged and
    supervise only the missing right tail. Retained audio is supplied directly
    at the clean timestep. The selected DiT window is bounded by S/W/C plus lookahead,
    independently of how much continuous audio the output decoder retains.
    Text-only conditioning is encoded once before sampling. Reference hosts
    may instead encode the selected visual context once per window. Every
    window is a bidirectional document and no transformer KV state crosses windows.
    Reference and target use separate temporal origins by default.
    ``separate_reference_rope=False`` shares their origin with target audio,
    while spatial coordinates retain each input's grid.
    ``fixed_window_rope=True`` anchors the sink and rebases each continuation's
    recent/current AV window onto an S/W/C template with separate reference
    and target domains. ``keep_sink_reference`` independently controls whether
    continuation references retain sink frames and defaults to True.

    The backbone returns x0. ``loss_prediction_type`` selects x0 MSE
    or native data-ward ``flow_v`` MSE, with the latter explicitly converting
    x0 at positive noise levels. Each sample is averaged over its original
    supervised element count per modality.
    A window without new audio contributes zero audio loss and still counts
    toward the batch denominator.
    ``bootstrap_loss_weight`` scales noisy AV rows in bootstrap windows.
    An omitted or null ``first_frame_loss_weight`` inherits the bootstrap
    weight. An explicit value overrides it at each modality's own global
    latent index 0. ``continuation_loss_weight`` scales only noisy AV rows in
    continuation windows. Bootstrap and continuation weights default to 1.0,
    without changing the original element-count denominator.
    ``audio_loss_weight`` multiplies the resulting audio loss, including its
    window and first-latent weights.
    ``guidance_scale`` enables detached-negative guidance-aware supervision
    only when it differs from one. Validation CFG is configured separately.
    ``keep_negative_reference`` defaults to True, so the captionless branch
    still reads the reference latents and any visual prefix. False removes
    both from that branch, leaving only the target AV window, in training
    fitting and in sampling CFG alike.
    ``negative_loss_weight`` defaults to 0.0, so the captionless branch only
    serves as a detached anchor. A positive weight runs that branch with
    gradients, even at ``guidance_scale`` 1, and adds the weighted window MSE
    of its own prediction, calibrating it toward the posterior mean of its
    reduced conditioning while the fitting term keeps the detached anchor.
    Explicit ``guidance_branches`` replace those legacy fitting options with
    named condition removals and independent supervision weights. A branch
    drops any of ``guidance_conditions``, ``text``, ``reference`` and
    ``history`` here, and hosts with further conditions extend it. Optional
    ``guidance_fitting.stages`` selects one sequential path through them to
    full conditioning. Equivalent states merge per sample before compiling
    a single fitted prediction, and no-op states add neither guidance nor loss.
    ``guidance_fitting.anchors`` defaults to ``posterior``, which reads each
    reduced prediction as its posterior mean. ``guided`` reads every state's
    output as guided along the chain up to it. The full and each supervised
    branch's fit then recover their own posterior mean from detached lower
    states, and every supervised branch must lie on the chain.
    ``history_drop`` sets how a branch drops ``history``, the clean target AV
    rows of a continuation window. ``remove``, the default, deletes them and
    keeps every surviving RoPE coordinate. ``noise`` keeps every row and masks
    the history with complete noise at noise level 1.0, drawn once per sample
    and step and shared by every branch. Only noisy target rows enter a loss in
    either mode, and a bootstrap window has no history to drop.
    ``bootstrap_probability`` defaults to 0.5. Clips without a continuation
    window always use the bootstrap, regardless of that probability.
    """

    qwen_visual_context: bool = False
    selective_video_encoding: bool = False
    guidance_branches: tuple[StreamingGuidanceBranch, ...] | None = None
    guidance_fitting: tuple[StreamingGuidanceStage, ...] = ()
    guidance_anchors: str = "posterior"
    history_drop: str = "remove"
    guidance_conditions: tuple[str, ...] = GUIDANCE_CONDITIONS
    _guidance_possible_uses: tuple[bool, ...] = ()
    validation_mode = "streaming"
    _sync_inputs_name = "h3_streaming_sft_inputs"

    def __init__(self, config: Any) -> None:
        self.guidance_branches = parse_guidance_branches(config.meta_model, self.guidance_conditions)
        self.guidance_fitting, self.guidance_anchors = parse_guidance_fitting(config.meta_model, self.guidance_branches)
        self._guidance_possible_uses = possible_guidance_uses(
            self.guidance_branches or (), self.guidance_fitting, anchors=self.guidance_anchors,
            conditions=self.guidance_conditions,
        )
        super().__init__(config)
        self._configure_streaming(config)
        options = config.meta_model
        self.training_timesteps = self._diffusion["training_timesteps"]
        self.guidance_scale = float(options.get("guidance_scale", 1.0))
        self.negative_loss_weight = float(options.get("negative_loss_weight", 0.0))
        self.loss_prediction_type = str(options.get("loss_prediction_type", "x0"))
        self.history_drop = str(options.get("history_drop", "remove"))
        if self.history_drop not in {"remove", "noise"}:
            raise ValueError("history_drop must be remove or noise")
        if not math.isfinite(self.guidance_scale) or self.guidance_scale <= 0:
            raise ValueError("guidance_scale must be finite and positive")
        if not math.isfinite(self.negative_loss_weight) or self.negative_loss_weight < 0:
            raise ValueError("negative_loss_weight must be finite and nonnegative")
        if self.loss_prediction_type not in {"x0", "flow_v"}:
            raise ValueError("loss_prediction_type must be x0 or flow_v")
        if self.loss_prediction_type == "flow_v" and not isinstance(self.schedule, LinearInterpolationSchedule):
            raise ValueError("flow_v supervision requires LinearInterpolationSchedule")

    @property
    def needs_reference_free_text_conditioning(self) -> bool:
        """Request a genuine text-only encoder pass only when a branch uses it."""
        return any(
            needed and "reference" in branch.drop and "text" not in branch.drop
            for branch, needed in zip(self.guidance_branches or (), self._guidance_possible_uses, strict=True)
        )

    def _configure_streaming(self, config: Any) -> None:
        """Configure shared window geometry, conditioning and loss reduction."""
        options = config.meta_model
        self.audio_loss_weight = float(options.get("audio_loss_weight", 1.0))
        self.bootstrap_loss_weight = float(options.get("bootstrap_loss_weight", 1.0))
        first_frame_loss_weight = options.get("first_frame_loss_weight")
        self.first_frame_loss_weight = (
            self.bootstrap_loss_weight if first_frame_loss_weight is None else float(first_frame_loss_weight)
        )
        self.continuation_loss_weight = float(options.get("continuation_loss_weight", 1.0))
        self.bootstrap_probability = float(options.get("bootstrap_probability", 0.5))
        self.separate_reference_rope = bool(options.get("separate_reference_rope", True))
        self.fixed_window_rope = bool(options.get("fixed_window_rope", False))
        self.keep_sink_reference = bool(options.get("keep_sink_reference", True))
        self.keep_negative_reference = bool(options.get("keep_negative_reference", True))
        self.qwen_visual_context = bool(options.get("qwen_visual_context", False))
        if self.qwen_visual_context:
            self._configure_window_context(config)
        self.selective_video_encoding = bool(options.get("selective_video_encoding", False))
        if self.selective_video_encoding:
            self._configure_selective_video_encoding(config)
        self.loss_weighting = str(options.get("loss_weighting", "uniform"))
        if not math.isfinite(self.audio_loss_weight) or self.audio_loss_weight < 0:
            raise ValueError("audio_loss_weight must be finite and nonnegative")
        if not math.isfinite(self.bootstrap_loss_weight) or self.bootstrap_loss_weight < 0:
            raise ValueError("bootstrap_loss_weight must be finite and nonnegative")
        if not math.isfinite(self.first_frame_loss_weight) or self.first_frame_loss_weight < 0:
            raise ValueError("first_frame_loss_weight must be finite and nonnegative")
        if not math.isfinite(self.continuation_loss_weight) or self.continuation_loss_weight < 0:
            raise ValueError("continuation_loss_weight must be finite and nonnegative")
        if not 0 <= self.bootstrap_probability <= 1:
            raise ValueError("bootstrap_probability must be in [0, 1]")
        if self.loss_weighting != "uniform":
            raise ValueError("streaming losses support uniform per-sample loss weighting")
        self.video_temporal_mapping = video_temporal_mapping_from_model_config(config.models.video_vae)
        if "video_decoder" in config.models:
            decoder_mapping = video_temporal_mapping_from_model_config(config.models.video_decoder)
            if decoder_mapping != self.video_temporal_mapping:
                raise ValueError("models.video_decoder temporal mapping must match models.video_vae")
        configured = config.data.args.get("video_temporal_mapping")
        if configured is not None and VideoTemporalMapping.from_dict(configured) != self.video_temporal_mapping:
            raise ValueError("data video_temporal_mapping must match models.video_vae")
        config.data.args.video_temporal_mapping = self.video_temporal_mapping.to_dict()
        self.streaming_config = {
            name: config.data.args[name] for name in ("sink_size", "window_size", "chunk_size")
        }
        self.audio_lookahead_latents = config.data.args.get("audio_lookahead_latents", 17)
        self.streaming_config["audio_lookahead_latents"] = self.audio_lookahead_latents
        self.streaming_config["sink_switch_at"] = config.data.args.get("sink_switch_at", 1)
        self.streaming_config["sink_size_after_switch"] = config.data.args.get("sink_size_after_switch")
        self.streaming_config["merge_single_frame_units"] = config.data.args.get("merge_single_frame_units", False)
        if config.data.args.get("audio_right_lookahead_latents") is not None:
            self.streaming_config["audio_right_lookahead_latents"] = config.data.args.audio_right_lookahead_latents
        if config.data.args.get("bootstrap_size") is not None:
            self.streaming_config["bootstrap_size"] = config.data.args.bootstrap_size
        streaming_window_starts(1, **self.streaming_config, video_temporal_mapping=self.video_temporal_mapping)
        self._validation_tokenizer_cache = None

    def _configure_window_context(self, config: Any) -> None:
        raise ValueError("qwen_visual_context requires a reference-enabled meta model")

    def _configure_selective_video_encoding(self, config: Any) -> None:
        raise ValueError("selective_video_encoding requires raw VideoRef streaming SFT inputs")

    def _prepare_step_context(self, states: Sequence[StreamingState], plans: list[dict[str, Any]], *,
                              models: dict[str, Any] | None, reference_pixels: Sequence[torch.Tensor] | None) -> list[dict[str, Any]]:
        raise NotImplementedError("visual streaming must supply original reference pixels")

    def _commit_reference_pixels(self, state: StreamingState, plan: dict[str, Any]) -> None:
        raise NotImplementedError("visual streaming must retain the next reference window")

    def _offline_reference_pixels(self, inputs: StreamingBatch, states: Sequence[StreamingState], counts: Sequence[int]) -> list[torch.Tensor]:
        if inputs.reference_pixels is None:
            raise ValueError("qwen_visual_context requires original reference_pixels, not latent-only input")
        boundary = self.video_temporal_mapping.decode_timeline.boundary
        return [inputs.reference_pixels[state.sample_index][:, boundary(state.next_video):boundary(state.next_video + count)]
                for state, count in zip(states, counts, strict=True)]

    def _validation_requests(self, validation: Any) -> list[Any]:
        if "requests" not in validation:
            return super()._validation_requests(validation)
        requests = list(validation.requests)
        limit = int(validation.get("num_prompts", 0))
        return requests[:limit] if limit > 0 else requests

    @staticmethod
    def _validation_request_prompt(request: Any) -> str:
        return str(request["prompt"]) if isinstance(request, dict) else str(request)

    def _validation_tokenizer(self, config: Any) -> MiniMaxH3Tokenizer:
        if self._validation_tokenizer_cache is None:
            self._validation_tokenizer_cache = MiniMaxH3Tokenizer(str(config.data.args.tokenizer_path))
        return self._validation_tokenizer_cache

    def _check_codec(self, codec: Any) -> None:
        if codec.video_temporal_mapping != self.video_temporal_mapping:
            raise ValueError("runtime video_vae temporal mapping disagrees with its configuration")

    def _streaming_decoder(
        self, models: dict[str, Any], *, audio_lookahead_latents: int,
        audio_right_lookahead_latents: int | None = None,
    ) -> StreamingAVDecoder:
        """Decode normalized model latents using the selected media codec's statistics."""
        video_vae = models["video_vae"]
        video_decoder = models.get("video_decoder", video_vae)
        if "video_decoder" in models:
            self._check_codec(video_vae)
            if video_decoder.video_temporal_mapping != self.video_temporal_mapping:
                raise ValueError("runtime video_decoder temporal mapping must match video_vae")
            channels = video_vae.latents_mean.numel()
            if video_decoder.latents_mean.numel() != channels or video_decoder.latents_std.numel() != channels:
                raise ValueError("video_decoder latent channels must match video_vae")
            if video_decoder.vae_ratio != video_vae.vae_ratio:
                raise ValueError("video_decoder spatial ratio must match video_vae")
        return StreamingAVDecoder(
            video_decoder, models["audio_vae"],
            audio_lookahead_latents=audio_lookahead_latents,
            audio_right_lookahead_latents=audio_right_lookahead_latents,
        )

    def _bootstrap_size(self, policy: dict[str, int | None]) -> int:
        return streaming_window_stop(0, policy, video_temporal_mapping=self.video_temporal_mapping)

    def _streaming_policy(self, overrides: dict[str, int | None]) -> dict[str, int | None]:
        """Merge options while retaining explicit requests for inherited defaults."""
        policy = {**self.streaming_config, **overrides}
        if (policy.get("audio_right_lookahead_latents") is None
                and self.streaming_config.get("audio_right_lookahead_latents") is None):
            policy.pop("audio_right_lookahead_latents", None)
        return policy

    @staticmethod
    def _audio_right_lookahead(policy: dict[str, int | None]) -> int:
        right = policy.get("audio_right_lookahead_latents")
        return policy["audio_lookahead_latents"] if right is None else right

    def _validation_streaming_policy(self, validation: Any, request: Any) -> dict[str, int | None]:
        policy = {name: request.get(name, validation.get(name, value))
                  for name, value in self.streaming_config.items()}
        if "audio_right_lookahead_latents" in request:
            policy["audio_right_lookahead_latents"] = request["audio_right_lookahead_latents"]
        elif "audio_right_lookahead_latents" in validation:
            policy["audio_right_lookahead_latents"] = validation["audio_right_lookahead_latents"]
        return self._streaming_policy(policy)

    def _inputs_from_payload(self, payload: dict[str, Any]) -> StreamingBatch:
        payload = self._to_device(payload)
        batch = StreamingBatch(
            prompt_embeds=payload["prompt_embeds"],
            reference_latents=payload.get("reference_latents"),
            video_shapes=[tuple(shape) for shape in payload["latent_shapes"]],
            audio_shapes=[tuple(shape) for shape in payload["audio_shapes"]],
            streaming_configs=[self._streaming_policy(policy) for policy in payload["streaming_config"]],
            prompts=payload.get("prompts"), text_token_tags=payload.get("text_token_tags"),
            negative_prompt_embeds=payload.get("negative_prompt_embeds"),
            negative_text_token_tags=payload.get("negative_text_token_tags"),
            text_only_prompt_embeds=payload.get("text_only_prompt_embeds"),
            caption_segments=payload.get("caption_segments"), caption_embeds=payload.get("caption_embeds"),
        )
        for index in range(batch.batch_size):
            self._plan(batch, index, 0)
        return batch

    def _plan(self, inputs: StreamingBatch, index: int, start: int, *, text_len: int | None = None) -> dict[str, Any]:
        return build_streaming_plan(
            target_video_shape=inputs.video_shapes[index],
            target_audio_shape=inputs.audio_shapes[index],
            reference_video_shape=(
                tuple(inputs.reference_latents[index].shape)
                if inputs.reference_latents is not None else None
            ),
            start=start, **self._streaming_policy(inputs.streaming_configs[index]),
            video_temporal_mapping=self.video_temporal_mapping,
            fixed_window_rope=self.fixed_window_rope,
            keep_sink_reference=self.keep_sink_reference,
            text_len=inputs.prompt_embeds[index].shape[0] if text_len is None else text_len,
        )

    @staticmethod
    def _plan_without_reference(plan: dict[str, Any]) -> dict[str, Any]:
        """Remove reference rows from a plan while keeping its target window."""
        shape = plan["reference_video_shape"]
        if shape is None:
            return plan
        rows = plan["reference_video_indices"].numel() * (shape[2] // 2) * (shape[3] // 2)
        return dict(plan, reference_video_shape=None, reference_video_indices=None,
                    packing_rows=plan["packing_rows"] - rows)

    @staticmethod
    def _branch_removes_language(embeddings: Sequence[torch.Tensor]) -> bool:
        """Identify the captionless branch, which must be uniform across samples."""
        rows = {int(value.shape[0]) for value in embeddings}
        if len(rows) > 1 and 0 in rows:
            raise ValueError("a conditioning branch must remove language for every sample or for none")
        return rows == {0}

    @execution_phase(ExecutionPhase.PREPARE)
    @torch.no_grad()
    def prepare_inputs(self, ctx: dict[str, Any]) -> dict[str, Any]:
        """Read normalized T2AV corpus latents and encode language conditioning."""
        batch, models = ctx["batch"], ctx["models"]
        for descriptor in batch["video_temporal_mapping"]:
            if VideoTemporalMapping.from_dict(descriptor) != self.video_temporal_mapping:
                raise ValueError("dataset temporal mapping disagrees with video_vae")
        for values, shapes in (
            (batch["video_latents"], batch["latent_shapes"]),
            (batch["audio_latents"], batch["audio_shapes"]),
        ):
            assert all(tuple(value.shape) == tuple(shape) for value, shape in zip(values, shapes, strict=True))
        payload = {
            "prompt_embeds": self._encode_prompts(models, batch["text_input_ids"], batch["text_lens"]),
            "reference_latents": None,
            "video_latents": batch["video_latents"], "audio_latents": batch["audio_latents"],
            "latent_shapes": batch["latent_shapes"], "audio_shapes": batch["audio_shapes"],
            "streaming_config": batch["streaming_config"], "packing_rows": batch["packing_rows"],
        }
        ctx["encoded_batch"] = self._to_device(payload)
        ctx["inputs"] = self._inputs_from_payload(ctx["encoded_batch"])
        ctx["clean_latents"] = (ctx["encoded_batch"]["video_latents"], ctx["encoded_batch"]["audio_latents"])
        return ctx

    def sync_inputs(self, ctx: dict[str, Any]) -> Iterator[dict[str, Any]]:
        if not is_unified_parallel_initialized() or get_unified_parallel_world_size() <= 1:
            yield ctx
            return
        sync = SPDistForward(name=self._sync_inputs_name, comm_shape=True, device=get_device())
        for payload in sync(ctx["encoded_batch"]):
            sub_ctx = dict(ctx, encoded_batch=payload, inputs=self._inputs_from_payload(payload))
            sub_ctx["clean_latents"] = (payload["video_latents"], payload["audio_latents"])
            yield sub_ctx

    def _sample_streaming_plan(self, inputs: StreamingBatch, index: int, rng: Any) -> dict[str, Any]:
        """Select one sample's window before that sample's timestep draw."""
        starts = streaming_window_starts(
            inputs.video_shapes[index][1], **self._streaming_policy(inputs.streaming_configs[index]),
            video_temporal_mapping=self.video_temporal_mapping,
        )
        bootstrap = len(starts) == 1 or self.bootstrap_probability == 1
        if not bootstrap and self.bootstrap_probability > 0:
            bootstrap = rng.python_generator.random() < self.bootstrap_probability
        start = 0 if bootstrap else rng.python_generator.choice(starts[1:])
        return self._plan(inputs, index, start)

    def _prepared_streaming_plans(self, ctx: dict[str, Any]) -> list[dict[str, Any]] | None:
        """Reuse the windows selected before visual conditioning or sparse encoding."""
        if not (self.qwen_visual_context or self.selective_video_encoding):
            return None
        plans = [{key: value.cpu() if isinstance(value, torch.Tensor) else value
                  for key, value in plan.items()} for plan in ctx["encoded_batch"]["preselected_plans"]]
        if self.selective_video_encoding:
            ctx["selected_video_indices"] = [plan["video_indices"].clone() for plan in plans]
            ctx["selected_reference_indices"] = [plan["reference_video_indices"].clone() for plan in plans]
        return plans

    @execution_phase(ExecutionPhase.TRAIN_FORWARD)
    def sample_timesteps(self, ctx: dict[str, Any]) -> dict[str, Any]:
        inputs, rng = ctx["inputs"], ctx["rng"]
        rngs = ctx.get("sample_rngs")
        if rngs is None:
            rngs = [rng.fork("streaming_sample", index) for index in range(inputs.batch_size)]
        assert len(rngs) == inputs.batch_size
        prepared_plans = self._prepared_streaming_plans(ctx)
        plans, video_times, audio_times = [], [], []
        for index, sample_rng in enumerate(rngs):
            plan = self._sample_streaming_plan(inputs, index, sample_rng) if prepared_plans is None else prepared_plans[index]
            plans.append(plan)
            with local_seed(combine_seed(sample_rng.seed, "timesteps") % 2**31):
                video_t, audio_t = self.training_timesteps.sample_pair(
                    (1,), torch.tensor([plan["packing_rows"]], device=get_device()), get_device(),
                    positions=torch.tensor([(inputs.video_shapes[index][1] - plan["start"]) / inputs.video_shapes[index][1]]),
                )
            sample_rng.seed = yield_seed(sample_rng.seed)
            video_times.append(video_t.float().reshape(()))
            audio_times.append(audio_t.float().reshape(()))
        rng.seed = yield_seed(rng.seed)
        if "encoded_batch" in ctx:
            assert all(plan["packing_rows"] <= int(budget) for plan, budget in zip(
                plans, ctx["encoded_batch"]["packing_rows"], strict=True
            )), "selected streaming window exceeds the dataset packing budget"
        ctx["plans"], ctx["sample_rngs"] = plans, list(rngs)
        ctx["train_timesteps"] = (torch.stack(video_times), torch.stack(audio_times))
        return ctx

    @staticmethod
    def _select(value: torch.Tensor, indices: torch.Tensor, *, audio: bool = False) -> torch.Tensor:
        return value.index_select(2 if audio else 1, indices.to(value.device))

    def _noisy_rows(self, values: Any, plans: list[dict[str, Any]]) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        return tuple([
            self._select(value, plan["audio_noisy_mask" if modality else "video_noisy_mask"].nonzero().flatten(),
                         audio=bool(modality))
            for value, plan in zip(modality_values, plans, strict=True)
        ] for modality, modality_values in enumerate(values))

    def _fitted_prediction(
        self, model: Any, inputs: StreamingInputs, negative: Callable[[], StreamingInputs], *,
        scale: float, **kwargs: Any,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """Apply optional CFG fitting with gradients only through the positive prediction.

        ``negative`` builds the captionless window only when a fitting scale
        other than one needs it.
        """
        prediction = self._streaming_forward(model, inputs, **kwargs)
        if scale != 1.0:
            with torch.no_grad():
                unconditional_prediction = self._streaming_forward(model, negative(), **kwargs)
            prediction = tuple([
                (positive + (scale - 1) * unconditional.detach()) / scale
                for positive, unconditional in zip(positives, negatives, strict=True)
            ] for positives, negatives in zip(prediction, unconditional_prediction, strict=True))
        return prediction

    def _guided_prediction(
        self, model: Any, inputs: StreamingInputs, negative: Callable[[], StreamingInputs], *,
        guidance: tuple[float, float], fitting_scale: float, rng: Any, **kwargs: Any,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """Combine optional fitting with external CFG drawn per sample from ``guidance``.

        A fitted model's conditional output already carries guidance at
        ``fitting_scale``, so the external scale is divided by it. One
        conditional forward suffices only when both are static and equal.
        """
        lower, upper = guidance
        if lower == upper == fitting_scale:
            return self._streaming_forward(model, inputs, **kwargs)
        unconditional = self._streaming_forward(model, negative(), **kwargs)
        conditional = self._streaming_forward(model, inputs, **kwargs)
        scales = [lower] * inputs.batch_size if lower == upper else lower + (upper - lower) * torch.rand(
            (inputs.batch_size,), device=get_device(), generator=self._generator(rng)
        )
        if fitting_scale != 1.0:
            scales = [scale / fitting_scale for scale in scales]
        return tuple([
            uncond + scale * (cond - uncond)
            for cond, uncond, scale in zip(positive, negative_values, scales, strict=True)
        ] for positive, negative_values in zip(conditional, unconditional, strict=True))

    @staticmethod
    def _noise(value: torch.Tensor, rng: Any) -> torch.Tensor:
        return torch.empty_like(value, dtype=torch.float32).normal_(generator=CausalMiniMaxH3Base._generator(rng))

    def _window_inputs(self, inputs: StreamingBatch, plans: list[dict[str, Any]], references: list[torch.Tensor] | None,
                       indices: Sequence[int] | None = None) -> StreamingInputs:
        indices = list(range(inputs.batch_size) if indices is None else indices)
        return self._streaming_inputs_from_payload({
            "plans": plans, "reference_latents": references,
            "prompt_embeds": [inputs.prompt_embeds[index] for index in indices],
            "text_token_tags": None if inputs.text_token_tags is None else [inputs.text_token_tags[index] for index in indices],
        })

    @execution_phase(ExecutionPhase.TRAIN_FORWARD)
    def add_noise(self, ctx: dict[str, Any]) -> dict[str, Any]:
        inputs, plans = ctx["inputs"], ctx["plans"]
        if self.selective_video_encoding:
            for plan, video_indices, reference_indices in zip(
                plans, ctx["selected_video_indices"], ctx["selected_reference_indices"], strict=True,
            ):
                assert torch.equal(plan["video_indices"].cpu(), video_indices), "video selection changed after encoding"
                assert torch.equal(plan["reference_video_indices"].cpu(), reference_indices), "reference selection changed after encoding"
        video_t, audio_t = ctx["train_timesteps"]
        noisy = ([], [])
        targets, noises = ([], []), ([], [])
        references = [] if inputs.reference_latents is not None else None
        for index, (plan, rng) in enumerate(zip(plans, ctx["sample_rngs"], strict=True)):
            for modality, timestep in enumerate((video_t[index], audio_t[index])):
                is_audio = modality == 1
                selected = self._select(ctx["clean_latents"][modality][index],
                                        plan["audio_indices" if is_audio else "video_indices"], audio=is_audio).detach().float()
                mask = plan["audio_noisy_mask" if is_audio else "video_noisy_mask"].to(selected.device)
                target = self._select(selected, mask.nonzero().flatten(), audio=is_audio)
                if is_audio:
                    epsilon = self._noise(target, rng)
                    mixed = self.schedule.forward(x_0=target, x_T=epsilon, t=timestep)
                    values = selected.clone()
                    values.index_copy_(2, mask.nonzero().flatten(), mixed)
                    noisy[modality].append(values)
                    targets[modality].append(target)
                    noises[modality].append(epsilon)
                    continue
                epsilon = self._noise(selected, rng)
                mixed = self.schedule.forward(x_0=selected, x_T=epsilon, t=timestep)
                anchor = MINIMAX_H3_VIDEO_CLEAN_TIMESTEP
                noisy[modality].append(torch.where(mask.view(1, -1, 1, 1), mixed,
                                                    anchor * selected + (1 - anchor) * epsilon))
                targets[modality].append(target)
                noises[modality].append(self._select(epsilon, mask.nonzero().flatten(), audio=is_audio))
            if inputs.reference_latents is not None:
                reference = self._select(inputs.reference_latents[index], plan["reference_video_indices"]).detach().float()
                references.append(MINIMAX_H3_VIDEO_CLEAN_TIMESTEP * reference +
                                  (1 - MINIMAX_H3_VIDEO_CLEAN_TIMESTEP) * self._noise(reference, rng))
        ctx["noisy_latents"], ctx["window_targets"], ctx["target_noises"] = noisy, targets, noises
        ctx["window_inputs"] = self._window_inputs(inputs, plans, references)
        return ctx

    def _negative_window_inputs(self, ctx: dict[str, Any]) -> StreamingInputs:
        """Prepare one shared negative layout from encoded prefixes and the same AV context.

        The captionless branch keeps the reference latents and any visual
        prefix by default. With ``keep_negative_reference`` disabled it drops
        both and conditions only on the target AV window.
        """
        if "negative_window_inputs" not in ctx:
            inputs, source = ctx["window_inputs"], ctx["inputs"]
            if self.keep_negative_reference:
                embeds, tags = source.negative_prompt_embeds, source.negative_text_token_tags
                if embeds is None:
                    assert not self.qwen_visual_context, "visual conditioning requires a prepared negative prefix"
                    embeds, tags = [value[:0] for value in inputs.prompt_embeds], None
                plans, references = inputs.plans, inputs.reference_latents
            else:
                embeds, tags = [value[:0] for value in inputs.prompt_embeds], None
                plans, references = [self._plan_without_reference(plan) for plan in inputs.plans], None
            negative = self._streaming_inputs_from_payload({
                "plans": [dict(plan, text_len=value.shape[0],
                               packing_rows=plan["packing_rows"] - plan["text_len"] + value.shape[0])
                          for plan, value in zip(plans, embeds, strict=True)],
                "prompt_embeds": embeds, "text_token_tags": tags,
                "reference_latents": references,
            })
            ctx["negative_window_inputs"] = replace(negative, reference_latents=references)
        return ctx["negative_window_inputs"]

    def _guidance_available_conditions(self, ctx: dict[str, Any]) -> tuple[frozenset[str], ...]:
        """Describe actual conditioning presence independently of branch definitions."""
        inputs, source = ctx["window_inputs"], ctx["inputs"]
        available = []
        for index, plan in enumerate(inputs.plans):
            media_length = (0 if source.negative_prompt_embeds is None
                            else source.negative_prompt_embeds[index].shape[0])
            has_text = inputs.text_lens[index] > media_length
            has_reference = plan["reference_video_shape"] is not None
            has_history = bool((~plan["video_noisy_mask"]).any() or (~plan["audio_noisy_mask"]).any())
            available.append(frozenset(
                name
                for name, present in (("text", has_text), ("reference", has_reference), ("history", has_history))
                if present
            ))
        return tuple(available)

    def _guidance_branch_active_samples(
        self, ctx: dict[str, Any], branch: StreamingGuidanceBranch,
    ) -> tuple[bool, ...]:
        """Identify samples where the requested removal changes conditioning."""
        return tuple(bool(branch.drop & available) for available in self._guidance_available_conditions(ctx))

    def _guidance_window_inputs(
        self, ctx: dict[str, Any], branch: StreamingGuidanceBranch,
    ) -> tuple[StreamingInputs, tuple[list[torch.Tensor], list[torch.Tensor]]]:
        """Reuse prepared noise and conditioning for one reduced-condition branch."""
        inputs, source = ctx["window_inputs"], ctx["inputs"]
        drop_text, drop_reference = "text" in branch.drop, "reference" in branch.drop
        if drop_text or drop_reference:
            if drop_text:
                if drop_reference or not self.qwen_visual_context:
                    embeds, tags = [value[:0] for value in inputs.prompt_embeds], None
                else:
                    embeds, tags = source.negative_prompt_embeds, source.negative_text_token_tags
                    assert embeds is not None, "visual conditioning requires a prepared media-only prefix"
            elif self.qwen_visual_context:
                embeds = source.text_only_prompt_embeds
                assert embeds is not None, "reference-free conditioning requires text-only prompt embeddings"
                tags = [torch.ones(value.shape[0], dtype=torch.long, device=value.device) for value in embeds]
            else:
                embeds, tags = inputs.prompt_embeds, inputs.text_token_tags
            references = None if drop_reference else inputs.reference_latents
            plans = ([self._plan_without_reference(plan) for plan in inputs.plans]
                     if drop_reference else inputs.plans)
            inputs = self._streaming_inputs_from_payload({
                "plans": [dict(plan, text_len=value.shape[0],
                               packing_rows=plan["packing_rows"] - plan["text_len"] + value.shape[0])
                          for plan, value in zip(plans, embeds, strict=True)],
                "prompt_embeds": embeds, "text_token_tags": tags, "reference_latents": references,
            })
        values = ctx["noisy_latents"]
        if "history" in branch.drop:
            return self._drop_target_history(ctx, inputs, values)
        return inputs, values

    def _drop_target_history(
        self, ctx: dict[str, Any], inputs: StreamingInputs, values: tuple[list[torch.Tensor], list[torch.Tensor]],
    ) -> tuple[StreamingInputs, tuple[list[torch.Tensor], list[torch.Tensor]]]:
        """Remove the target history rows, or mask them with the step's shared complete noise.

        The noise forks each sample's stream, which every rank of an SP group
        shares, and is cached in ``ctx`` for the step's other branches.
        """
        if self.history_drop == "remove":
            return without_target_history(inputs, *values)
        if "history_noise" not in ctx:
            ctx["history_noise"] = tuple(
                [self._noise(value, rng.fork("history_drop", modality))
                 for value, rng in zip(values[modality], ctx["sample_rngs"], strict=True)]
                for modality in range(2)
            )
        return noised_target_history(inputs, *values, ctx["history_noise"])

    def _forward_guidance_branches(self, ctx: dict[str, Any]) -> None:
        """Run each useful branch on every rank when any rank has active samples."""
        branches = self.guidance_branches
        compiled = [compile_guidance(branches, self.guidance_fitting, available, anchors=self.guidance_anchors)
                    for available in self._guidance_available_conditions(ctx)]
        ctx["compiled_guidance"] = compiled
        ctx["guidance_active_samples"] = {
            branch.name: tuple(item.used[index] for item in compiled)
            for index, branch in enumerate(branches)
        }
        ctx["guidance_loss_weights"] = {
            branch.name: tuple(item.loss_weights[index] for item in compiled)
            for index, branch in enumerate(branches)
        }
        ctx["guidance_loss_branches"] = []
        ctx["guidance_predictions"] = {}
        if not any(self._guidance_possible_uses):
            return
        needed = torch.tensor([
            [any(ctx["guidance_active_samples"][branch.name]), any(ctx["guidance_loss_weights"][branch.name])]
            for branch in branches
        ], dtype=torch.int32, device=get_device())
        all_reduce_max(needed)
        for branch, (run, with_loss) in zip(branches, needed.tolist(), strict=True):
            if not run:
                continue
            inputs, values = self._guidance_window_inputs(ctx, branch)
            if with_loss:
                ctx["guidance_loss_branches"].append(branch.name)
            with torch.set_grad_enabled(torch.is_grad_enabled() and bool(with_loss)):
                ctx["guidance_predictions"][branch.name] = self._streaming_forward(
                    ctx["models"]["backbone"], inputs, video_xts=values[0], audio_xts=values[1],
                    video_timesteps=ctx["train_timesteps"][0], audio_timesteps=ctx["train_timesteps"][1],
                )

    def _fitted_guidance_prediction(
        self, ctx: dict[str, Any], prediction: tuple[list[torch.Tensor], list[torch.Tensor]],
        fits: Sequence[tuple[float, tuple[float, ...]]] | None = None,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """Invert one sequential guidance combination per sample using detached anchors.

        ``fits`` holds each sample's scale and branch coefficients and
        defaults to the full-condition fit.
        """
        if fits is None:
            fits = [(item.full_scale, item.coefficients) for item in ctx["compiled_guidance"]]
        fitted = ([], [])
        for modality, values in enumerate(prediction):
            for sample_index, (positive, (scale, coefficients)) in enumerate(zip(values, fits, strict=True)):
                value = positive
                for branch, coefficient in zip(self.guidance_branches, coefficients, strict=True):
                    if coefficient != 0.0:
                        anchor = ctx["guidance_predictions"][branch.name][modality][sample_index]
                        value = value + (-coefficient) * anchor.detach()
                if scale != 1.0:
                    value = value / scale
                fitted[modality].append(value)
        return fitted

    def _compute_guidance_branch_losses(self, ctx: dict[str, Any]) -> None:
        """Add reduced-condition supervision of each branch's fit, with inactive samples contributing zero."""
        for index, branch in enumerate(self.guidance_branches):
            name = branch.name
            if name not in ctx["guidance_loss_branches"]:
                continue
            fitted = self._fitted_guidance_prediction(ctx, ctx["guidance_predictions"][name], [
                (item.branch_scales[index], item.branch_coefficients[index]) for item in ctx["compiled_guidance"]
            ])
            predictions, targets = self._supervised_pairs(ctx, fitted)
            video, audio = self._reduce_window_losses(
                ctx, predictions, targets, sample_weights=ctx["guidance_loss_weights"][name],
            )
            ctx["loss"] = ctx["loss"] + video + self.audio_loss_weight * audio
            ctx["metrics"].update({f"train/branches/{name}/video_loss": float(video.detach()),
                                   f"train/branches/{name}/audio_loss": float(audio.detach())})

    @execution_phase(ExecutionPhase.TRAIN_FORWARD)
    def forward(self, ctx: dict[str, Any]) -> dict[str, Any]:
        inputs = ctx["window_inputs"]
        kwargs = dict(video_xts=ctx["noisy_latents"][0], audio_xts=ctx["noisy_latents"][1],
                      video_timesteps=ctx["train_timesteps"][0], audio_timesteps=ctx["train_timesteps"][1])
        prediction = self._streaming_forward(ctx["models"]["backbone"], inputs, **kwargs)
        if self.guidance_branches is not None:
            self._forward_guidance_branches(ctx)
            ctx["pred"] = self._fitted_guidance_prediction(ctx, prediction)
            return ctx
        if self.guidance_scale != 1 or self.negative_loss_weight > 0:
            with torch.set_grad_enabled(self.negative_loss_weight > 0):
                ctx["negative_pred"] = self._streaming_forward(
                    ctx["models"]["backbone"], self._negative_window_inputs(ctx), **kwargs,
                )
        if self.guidance_scale != 1:
            prediction = tuple([
                (positive + (self.guidance_scale - 1) * unconditional.detach()) / self.guidance_scale
                for positive, unconditional in zip(positive_values, negative_values, strict=True)
            ] for positive_values, negative_values in zip(prediction, ctx["negative_pred"], strict=True))
        ctx["pred"] = prediction
        return ctx

    @execution_phase(ExecutionPhase.TRAIN_FORWARD)
    def compute_loss(self, ctx: dict[str, Any]) -> dict[str, Any]:
        ctx = self._compute_window_loss(ctx, *self._supervised_pairs(ctx, ctx["pred"]))
        if self.guidance_branches is not None:
            self._compute_guidance_branch_losses(ctx)
            return ctx
        if self.negative_loss_weight > 0:
            video, audio = self._reduce_window_losses(ctx, *self._supervised_pairs(ctx, ctx["negative_pred"]))
            ctx["loss"] = ctx["loss"] + self.negative_loss_weight * (video + self.audio_loss_weight * audio)
            ctx["metrics"].update({"train/negative_video_loss": float(video.detach()),
                                   "train/negative_audio_loss": float(audio.detach())})
        return ctx

    def _supervised_pairs(
        self, ctx: dict[str, Any], predictions: tuple[list[torch.Tensor], list[torch.Tensor]],
    ) -> tuple[tuple[list[torch.Tensor], list[torch.Tensor]], tuple[list[torch.Tensor], list[torch.Tensor]]]:
        """Pair predictions with the window targets in the configured supervision space."""
        if self.loss_prediction_type != "flow_v":
            return predictions, ctx["window_targets"]
        converted, targets = ([], []), ([], [])
        for modality, (values, clean) in enumerate(zip(predictions, ctx["window_targets"], strict=True)):
            for index, (prediction, target) in enumerate(zip(values, clean, strict=True)):
                if target.numel():
                    sigma = self.schedule.B(ctx["train_timesteps"][modality][index])
                    if not bool(sigma > 0):
                        raise ValueError("flow_v supervision requires positive noisy target timesteps")
                    epsilon = ctx["target_noises"][modality][index]
                    xt = (1 - sigma) * target + sigma * epsilon
                    prediction = (prediction.float() - xt) / sigma
                    target = target - epsilon
                converted[modality].append(prediction)
                targets[modality].append(target)
        return converted, targets

    def _compute_window_loss(
        self, ctx: dict[str, Any], predictions: tuple[list[torch.Tensor], list[torch.Tensor]],
        targets: tuple[list[torch.Tensor], list[torch.Tensor]],
    ) -> dict[str, Any]:
        """Store the weighted window loss with its metrics."""
        losses = self._reduce_window_losses(ctx, predictions, targets)
        ctx["loss"] = losses[0] + self.audio_loss_weight * losses[1]
        ctx["metrics"] = {"train/video_loss": float(losses[0].detach()), "train/audio_loss": float(losses[1].detach()),
                          "train/bootstrap_fraction": sum(plan["is_bootstrap"] for plan in ctx["plans"]) / len(ctx["plans"])}
        return ctx

    def _reduce_window_losses(
        self, ctx: dict[str, Any], predictions: tuple[list[torch.Tensor], list[torch.Tensor]],
        targets: tuple[list[torch.Tensor], list[torch.Tensor]],
        *, active_samples: Sequence[bool] | None = None, sample_weights: Sequence[float] | None = None,
    ) -> list[torch.Tensor]:
        """Reduce noisy AV MSE with window and overriding first-latent weights."""
        sums, counts = [], []
        for modality, (values, clean) in enumerate(zip(predictions, targets, strict=True)):
            total = values[0].float().sum() * 0
            for index, (prediction, target) in enumerate(zip(values, clean, strict=True)):
                if ((active_samples is not None and not active_samples[index])
                        or (sample_weights is not None and sample_weights[index] == 0)):
                    total = total + prediction.float().sum() * 0
                    continue
                if target.numel() == 0:
                    continue
                prediction = prediction.float()
                squared_error = (prediction - target.detach()).square()
                if (
                    self.bootstrap_loss_weight != 1.0 or self.first_frame_loss_weight != 1.0
                ) and ctx["plans"][index]["is_bootstrap"]:
                    plan = ctx["plans"][index]
                    prefix, time_dim = ("audio", 2) if modality else ("video", 1)
                    noisy_indices = plan[f"{prefix}_indices"][plan[f"{prefix}_noisy_mask"]]
                    latent_weights = squared_error.new_full(
                        (squared_error.shape[time_dim],), self.bootstrap_loss_weight,
                    )
                    latent_weights[noisy_indices.to(squared_error.device) == 0] = self.first_frame_loss_weight
                    weight_shape = (1, 1, -1) if modality else (1, -1, 1, 1)
                    squared_error = squared_error * latent_weights.view(weight_shape)
                elif self.continuation_loss_weight != 1.0 and not ctx["plans"][index]["is_bootstrap"]:
                    squared_error = squared_error * self.continuation_loss_weight
                sample_loss = squared_error.mean()
                if sample_weights is not None:
                    sample_loss = sample_loss * sample_weights[index]
                total = total + sample_loss
            sums.append(total)
            counts.append(len(clean))
        global_counts = torch.tensor(counts, device=get_device(), dtype=torch.float64)
        all_reduce_sum(global_counts)
        return [total * get_world_size() / count.clamp_min(1) for total, count in zip(sums, global_counts, strict=True)]


    @torch.no_grad()
    def _validation_inputs_for_requests(self, config: Any, models: dict[str, Any], requests: Sequence[Any],
                                        *, prompts: Sequence[str] | None = None) -> StreamingBatch:
        """Prepare prompt-only streams from requested target geometry."""
        self._check_codec(models["video_vae"])
        validation, data = config.validation, config.data.args
        video_shapes, audio_shapes, policies = [], [], []
        for raw_request in requests:
            request = raw_request if isinstance(raw_request, dict) else {"prompt": str(raw_request)}
            frames = int(request.get("num_frames", validation.num_frames))
            height, width = int(request.get("height", validation.height)), int(request.get("width", validation.width))
            fps = int(request.get("fps", validation.get("fps", 24)))
            if fps != 24:
                raise ValueError("streaming AV geometry requires 24 FPS video and 40 Hz audio latents")
            video_shape, audio_shape = streaming_target_shapes(
                height=height, width=width, num_frames=frames,
                video_latent_channels=int(data.get("video_latent_channels", 24)),
                audio_latent_channels=int(data.get("audio_latent_channels", 32)),
                spatial_vae_stride=int(data.get("spatial_vae_stride", 16)),
                video_temporal_mapping=self.video_temporal_mapping,
            )
            video_shapes.append(video_shape)
            audio_shapes.append(audio_shape)
            policies.append(self._validation_streaming_policy(validation, request))
        prompts = [self._validation_request_prompt(request) for request in requests] if prompts is None else prompts
        tokens = [self._validation_tokenizer(config).encode(str(prompt)) for prompt in prompts]
        embeds = self._encode_prompts(models, [item[0] for item in tokens], [item[1] for item in tokens])
        return self._inputs_from_payload(dict(prompt_embeds=embeds, reference_latents=None,
                                             latent_shapes=video_shapes, audio_shapes=audio_shapes,
                                             streaming_config=policies))

    def _sampling_grid(self, node: Any, seqlens: Sequence[int], device: torch.device) -> torch.Tensor:
        grids = []
        for length in seqlens:
            node.set_timesteps(seqlen=torch.tensor([length], device=device), device=device)
            grid = node.timesteps
            assert grid.ndim == 1
            grids.append(grid.to(device=device, dtype=torch.float32).clone())
        return torch.stack(grids)

    def start_stream(
        self, *, prompt_embeds: Sequence[torch.Tensor] | None = None, video_geometries: Sequence[tuple[int, int, int]],
        audio_channels: Sequence[int], rngs: Sequence[Any],
        reference_geometries: Sequence[tuple[int, int, int]] | None = None,
        streaming_configs: Sequence[dict[str, int | None]] | None = None,
        audio_lengths: Sequence[int | None] | None = None,
        video_lengths: Sequence[int | None] | None = None,
        guidance_scale: float = 1.0,
        conditioning_branches: Sequence[tuple[float, Sequence[torch.Tensor]]] | None = None,
        prompts: Sequence[str] | None = None,
        conditioning_prompts: Sequence[tuple[float, Sequence[str]]] | None = None,
    ) -> list[StreamingState]:
        """Create streams with optional reference geometry and known AV lengths.

        Geometries are (channels, height, width), and lengths count each
        modality's latents. T2AV streams advance from their window policy or
        explicit video_counts. Reference streams receive new latents at step.
        Known video lengths also bound audio lookahead when audio lengths
        are omitted, using the codec's decoded frame count.
        """
        if self.qwen_visual_context:
            if prompts is None or len(prompts) != len(video_geometries):
                raise ValueError("qwen_visual_context requires the original prompt for each sample")
            if prompt_embeds is None:
                prompt_embeds = [torch.empty((0, 0), device=get_device()) for _ in prompts]
        if prompt_embeds is None:
            raise ValueError("prompt_embeds must be provided")
        count = len(prompt_embeds)
        if count == 0 or not math.isfinite(guidance_scale):
            raise ValueError("streaming requires a nonempty batch and finite guidance scale")
        policies = list(streaming_configs) if streaming_configs is not None else [self.streaming_config] * count
        lengths = list(audio_lengths) if audio_lengths is not None else [None] * count
        video_limits = list(video_lengths) if video_lengths is not None else [None] * count
        references = list(reference_geometries) if reference_geometries is not None else [None] * count
        if conditioning_branches is None:
            conditioning_branches = [(guidance_scale, prompt_embeds)]
            if guidance_scale != 1:
                conditioning_branches.append((1 - guidance_scale, [value[:0] for value in prompt_embeds]))
        branches = [(weight, embeddings) for weight, embeddings in conditioning_branches if weight != 0]
        text_branches = None
        if self.qwen_visual_context:
            if conditioning_prompts is None:
                conditioning_prompts = [(guidance_scale, prompts)]
                if guidance_scale != 1:
                    conditioning_prompts.append((1 - guidance_scale, [""] * count))
            text_branches = [(weight, list(texts)) for weight, texts in conditioning_prompts if weight != 0]
            if [weight for weight, _ in text_branches] != [weight for weight, _ in branches] or any(len(texts) != count for _, texts in text_branches):
                raise ValueError("conditioning prompts must align with the sampling branches")
        assert branches and all(len(embeddings) == count for _, embeddings in branches)
        states = []
        for index, (embedding, video, reference, channels, rng, policy, length, video_length) in enumerate(zip(
            prompt_embeds, video_geometries, references, audio_channels, rngs, policies, lengths, video_limits, strict=True
        )):
            policy = self._streaming_policy(policy)
            policy.setdefault("audio_lookahead_latents", self.audio_lookahead_latents)
            streaming_window_starts(1, **policy, video_temporal_mapping=self.video_temporal_mapping)
            if length is not None and (isinstance(length, bool) or not isinstance(length, int) or length < 0):
                raise ValueError("audio_length must be a nonnegative integer or None")
            if video_length is not None and (isinstance(video_length, bool) or not isinstance(video_length, int) or video_length <= 0):
                raise ValueError("video_length must be a positive integer or None")
            if length is None and video_length is not None:
                length = round(Fraction(self.video_temporal_mapping.frame_count(video_length) * 40, 24))
            device = embedding.device
            states.append(StreamingState(
                sample_index=index, prompt_branches=[(weight, values[index]) for weight, values in branches],
                video_geometry=tuple(video), reference_geometry=None if reference is None else tuple(reference),
                audio_channels=channels, streaming_config=dict(policy), rng=rng, audio_length=length,
                video_history=torch.empty((video[0], 0, *video[1:]), device=device),
                audio_history=torch.empty((2, channels, 0), device=device),
                audio_future=torch.empty((2, channels, 0), device=device),
                reference_history=(None if reference is None else torch.empty((reference[0], 0, *reference[1:]), device=device)),
                video_history_indices=torch.empty(0, dtype=torch.long),
                audio_history_indices=torch.empty(0, dtype=torch.long), video_length=video_length,
                reference_history_indices=(None if reference is None else torch.empty(0, dtype=torch.long)),
                prompt_texts=None if text_branches is None else [texts[index] for _, texts in text_branches],
            ))
        return states

    def _state_plan(self, state: StreamingState, stop: int, audio_stop: int, text_len: int) -> dict[str, Any]:
        return build_streaming_plan(
            target_video_shape=(state.video_geometry[0], stop, *state.video_geometry[1:]),
            target_audio_shape=(2, state.audio_channels, audio_stop),
            reference_video_shape=(
                (state.reference_geometry[0], stop, *state.reference_geometry[1:])
                if state.reference_geometry is not None else None
            ),
            start=state.next_video, **state.streaming_config,
            video_temporal_mapping=self.video_temporal_mapping, text_len=text_len,
            fixed_window_rope=self.fixed_window_rope,
            keep_sink_reference=self.keep_sink_reference,
            audio_previous_prediction_stop=min(state.audio_generated_stop, audio_stop),
        )

    def _commit_stream_state(self, state: StreamingState, plan: dict[str, Any], video: torch.Tensor,
                             selected_audio: torch.Tensor, reference: torch.Tensor | None, is_last: bool) -> None:
        full_video = torch.cat((state.video_history, video), dim=1)
        sink_stop, recent_start = plan["next_sink_stop"], plan["next_recent_start"]
        video_indices, audio_indices = plan["video_indices"], plan["audio_indices"]
        keep_video = (video_indices < sink_stop) | (video_indices >= recent_start)
        keep_audio = torch.zeros_like(audio_indices, dtype=torch.bool)
        boundary = self.video_temporal_mapping.decode_timeline.clock_boundary_ceil
        for index in video_indices[keep_video].tolist():
            keep_audio |= (audio_indices >= boundary(index)) & (audio_indices < boundary(index + 1))
        keep_audio &= audio_indices < plan["audio_commit_stop"]
        state.video_history = self._select(full_video, keep_video.nonzero().flatten()).detach().contiguous()
        if reference is not None:
            reference_indices = plan["reference_video_indices"]
            keep_reference = reference_indices >= recent_start
            if self.keep_sink_reference:
                keep_reference |= reference_indices < sink_stop
            state.reference_history = self._select(reference, keep_reference.nonzero().flatten()).detach().contiguous()
            state.reference_history_indices = reference_indices[keep_reference]
        state.audio_history = self._select(selected_audio, keep_audio.nonzero().flatten(), audio=True).detach().contiguous()
        future = plan["audio_lookahead_mask"]
        state.audio_future = self._select(selected_audio, future.nonzero().flatten(), audio=True).detach().contiguous()
        state.video_history_indices = video_indices[keep_video]
        state.audio_history_indices = audio_indices[keep_audio]
        state.next_video = plan["stop"]
        state.next_audio = plan["audio_commit_stop"]
        state.audio_generated_stop = plan["audio_prediction_stop"]
        state.finished = is_last
        if self.qwen_visual_context:
            self._commit_reference_pixels(state, plan)

    def _denoise_window(
        self, backbone: Any, branches: Sequence[tuple[float, StreamingInputs]], plans: list[dict[str, Any]],
        histories: tuple[list[torch.Tensor], list[torch.Tensor]], video_xts: list[torch.Tensor],
        audio_xts: list[torch.Tensor], *, rngs: list[Any],
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """Denoise the noisy rows of complete windows over the sampling grid.

        ``histories`` carry the conditioned rows, and each weighted branch
        contributes its prediction. Returns the final noisy-row latents.
        """
        device = video_xts[0].device
        seqlens = [plan["packing_rows"] for plan in plans]
        video_grid = self._sampling_grid(self.sampling_timesteps, seqlens, device)
        audio_grid = self._sampling_grid(self.audio_sampling_timesteps, seqlens, device)
        if video_grid.shape != audio_grid.shape:
            raise ValueError("video and audio sampling grids must have equal step counts")
        for step_index in range(video_grid.shape[1]):
            vt, at = video_grid[:, step_index], audio_grid[:, step_index]
            vs = video_grid[:, step_index + 1] if step_index + 1 < video_grid.shape[1] else torch.zeros_like(vt)
            audio_s = audio_grid[:, step_index + 1] if step_index + 1 < audio_grid.shape[1] else torch.zeros_like(at)
            selected_video, selected_audio = [], []
            for index, plan in enumerate(plans):
                v, a = histories[0][index].clone(), histories[1][index].clone()
                v[:, plan["video_noisy_mask"].to(device)] = video_xts[index]
                a[:, :, plan["audio_noisy_mask"].to(device)] = audio_xts[index]
                selected_video.append(v)
                selected_audio.append(a)
            vp, ap = [torch.zeros_like(value) for value in video_xts], [torch.zeros_like(value) for value in audio_xts]
            for weight, branch in branches:
                prediction = self._streaming_forward(backbone, branch, video_xts=selected_video, audio_xts=selected_audio,
                                                     video_timesteps=vt, audio_timesteps=at)
                for accumulators, values in zip((vp, ap), prediction, strict=True):
                    for accumulator, value in zip(accumulators, values, strict=True):
                        accumulator.add_(value, alpha=weight)
            lengths = torch.tensor(seqlens, device=device)
            video_xts = self.sampler.step_to(pred=vp, x_t=video_xts, t=vt, s=vs, rng=rngs, seqlens=lengths)
            audio_xts = self.sampler.step_to(pred=ap, x_t=audio_xts, t=at, s=audio_s, rng=rngs, seqlens=lengths)
        return video_xts, audio_xts

    @torch.no_grad()
    def step(self, backbone: Any, states: Sequence[StreamingState],
             reference_latents: Sequence[torch.Tensor] | None = None, *,
             video_counts: Sequence[int] | None = None,
             is_last: Sequence[bool] | None = None,
             reference_pixels: Sequence[torch.Tensor] | None = None,
             models: dict[str, Any] | None = None) -> StreamingLatentChunk:
        """Denoise one batch of bounded windows and commit new AV latents.

        T2AV uses the bootstrap units initially and C thereafter. video_counts remains
        a native latent count and can specify a shorter final piece. Known video
        lengths truncate the last piece and mark completion. Reference streams retain their positional
        reference_latents argument. Visual conditioning additionally requires
        models and new reference_pixels on the original RGB timeline.
        Distributed callers invoke steps collectively.
        """
        assert states
        if self.qwen_visual_context and (models is None or reference_pixels is None):
            raise ValueError("visual streaming requires models and original reference_pixels")
        has_reference = states[0].reference_geometry is not None
        assert all((state.reference_geometry is not None) == has_reference for state in states)
        if has_reference:
            assert reference_latents is not None and len(states) == len(reference_latents)
        else:
            assert reference_latents is None
        if video_counts is not None:
            assert len(video_counts) == len(states)
        final_flags = [False] * len(states) if is_last is None else list(is_last)
        assert len(final_flags) == len(states)
        device = states[0].video_history.device
        weights = [weight for weight, _ in states[0].prompt_branches]
        assert all([weight for weight, _ in state.prompt_branches] == weights for state in states)
        plans, video_xts, audio_xts = [], [], []
        raw_references = [] if has_reference else None
        references = [] if has_reference else None
        histories = ([], [])
        boundary = self.video_temporal_mapping.decode_timeline.clock_boundary_ceil
        for index, state in enumerate(states):
            if state.finished:
                raise ValueError("cannot append to a finished stream")
            expected = streaming_window_stop(
                state.next_video, state.streaming_config, video_temporal_mapping=self.video_temporal_mapping,
            ) - state.next_video
            if has_reference:
                new_reference = reference_latents[index]
                channels, height, width = state.reference_geometry
                assert new_reference.ndim == 4 and (new_reference.shape[0], *new_reference.shape[2:]) == (channels, height, width)
                count = new_reference.shape[1]
                if video_counts is not None:
                    assert count == video_counts[index]
            else:
                count = expected if video_counts is None else video_counts[index]
                if isinstance(count, bool) or not isinstance(count, int):
                    raise ValueError("video_counts must contain positive integers")
            if state.video_length is not None:
                remaining = state.video_length - state.next_video
                if has_reference and count > remaining:
                    raise ValueError("reference chunk exceeds the known video length")
                count = min(count, remaining)
                final_flags[index] = final_flags[index] or count == remaining
            final = final_flags[index]
            if not 0 < count <= expected or (count != expected and not final):
                raise ValueError("video chunks must span the bootstrap units initially or C subsequently, except a short final chunk")
            stop = state.next_video + count
            audio_stop = boundary(stop)
            if state.audio_length is not None:
                audio_stop = min(audio_stop, state.audio_length)
            elif final:
                audio_stop = round(Fraction(self.video_temporal_mapping.frame_count(stop) * 40, 24))
            if audio_stop < state.next_audio:
                raise ValueError("final audio length precedes already generated audio")
            prediction_stop = audio_stop if final else audio_stop + self._audio_right_lookahead(state.streaming_config)
            if state.audio_length is not None:
                prediction_stop = min(prediction_stop, state.audio_length)
            plan = self._state_plan(state, stop, prediction_stop, state.prompt_branches[0][1].shape[0])
            assert torch.equal(plan["video_indices"][~plan["video_noisy_mask"]], state.video_history_indices)
            plans.append(plan)
            v = torch.zeros((state.video_geometry[0], count, *state.video_geometry[1:]), device=device)
            a = torch.zeros((2, state.audio_channels, int(plan["audio_noisy_mask"].sum())), device=device)
            video_xts.append(self._noise(v, state.rng))
            audio_xts.append(self._noise(a, state.rng))
            video_clean = torch.cat((state.video_history, v), dim=1)
            histories[0].append(MINIMAX_H3_VIDEO_CLEAN_TIMESTEP * video_clean +
                                (1 - MINIMAX_H3_VIDEO_CLEAN_TIMESTEP) * self._noise(video_clean, state.rng))
            audio_values = torch.zeros((2, state.audio_channels, plan["audio_indices"].numel()), device=device)
            retained = ~plan["audio_noisy_mask"]
            cached_indices = torch.cat((state.audio_history_indices, torch.arange(state.next_audio, state.audio_generated_stop)))
            cached_audio = torch.cat((state.audio_history, state.audio_future), dim=2)
            requested_indices = plan["audio_indices"][retained]
            cached_positions = torch.searchsorted(cached_indices, requested_indices)
            assert torch.equal(cached_indices[cached_positions], requested_indices)
            audio_values.index_copy_(2, retained.nonzero().flatten().to(device),
                                     self._select(cached_audio, cached_positions, audio=True))
            histories[1].append(audio_values)
            if has_reference:
                reference_indices = plan["reference_video_indices"]
                assert torch.equal(reference_indices[reference_indices < state.next_video], state.reference_history_indices)
                assert torch.equal(reference_indices[reference_indices >= state.next_video], torch.arange(state.next_video, stop))
                raw = torch.cat((state.reference_history, new_reference.to(device=device, dtype=torch.float32)), dim=1)
                raw_references.append(raw)
                references.append(MINIMAX_H3_VIDEO_CLEAN_TIMESTEP * raw +
                                  (1 - MINIMAX_H3_VIDEO_CLEAN_TIMESTEP) * self._noise(raw, state.rng))
        if self.qwen_visual_context:
            plans = self._prepare_step_context(states, plans, models=models, reference_pixels=reference_pixels)
        branches = []
        for branch_index, weight in enumerate(weights):
            embeddings = [state.prompt_branches[branch_index][1] for state in states]
            branch_plans = [self._branch_state_plan(state, plan, embedding.shape[0], branch_index)
                            for state, plan, embedding in zip(states, plans, embeddings, strict=True)]
            branch_references = references
            if has_reference and not self.keep_negative_reference and self._branch_removes_language(embeddings):
                branch_plans = [self._plan_without_reference(plan) for plan in branch_plans]
                branch_references = None
            branches.append((weight, self._streaming_inputs_from_payload({
                "plans": branch_plans, "prompt_embeds": embeddings, "reference_latents": branch_references,
                "text_token_tags": ([state.prompt_branch_tags[branch_index] for state in states]
                                    if self.qwen_visual_context else None),
            })))
        video_xts, audio_xts = self._denoise_window(
            backbone, branches, plans, histories, video_xts, audio_xts, rngs=[state.rng for state in states],
        )
        selected_audios = []
        for plan, retained, new in zip(plans, histories[1], audio_xts, strict=True):
            values = retained.clone()
            values.index_copy_(2, plan["audio_noisy_mask"].nonzero().flatten().to(device), new)
            selected_audios.append(values)
        commit_references = raw_references if raw_references is not None else [None] * len(states)
        for state, plan, video, audio, reference, final in zip(states, plans, video_xts, selected_audios, commit_references, final_flags, strict=True):
            self._commit_stream_state(state, plan, video, audio, reference, final)
        committed_audio, lookahead_audio = [], []
        for audio, plan in zip(selected_audios, plans, strict=True):
            committed = plan["audio_commit_mask"]
            future = plan["audio_lookahead_mask"]
            committed_audio.append(self._select(audio, committed.nonzero().flatten(), audio=True))
            lookahead_audio.append(self._select(audio, future.nonzero().flatten(), audio=True))
        return StreamingLatentChunk(
            [state.sample_index for state in states],
            [plan["video_indices"][plan["video_noisy_mask"]] for plan in plans],
            [plan["audio_commit_indices"] for plan in plans],
            video_xts, committed_audio, final_flags,
            audio_lookahead=lookahead_audio,
            audio_lookahead_indices=[plan["audio_indices"][plan["audio_lookahead_mask"]] for plan in plans],
            audio_new_indices=[plan["audio_indices"][plan["audio_noisy_mask"]] for plan in plans],
        )

    def _branch_state_plan(self, state: StreamingState, plan: dict[str, Any], text_len: int,
                           branch_index: int) -> dict[str, Any]:
        """The window plan sampling branch ``branch_index`` reads: the shared window at its own text length."""
        return self._state_plan(state, plan["stop"], plan["target_audio_shape"][2], text_len)

    @torch.no_grad()
    def _select_stream_captions(
        self, states: Sequence[StreamingState], counts: Sequence[int],
        branch_inputs: Sequence[tuple[float, StreamingBatch]],
    ) -> None:
        """Select each branch's caption once per window using its true noisy frame interval."""
        boundary = self.video_temporal_mapping.decode_timeline.boundary
        branches = [(weight, branch) for weight, branch in branch_inputs if weight != 0]
        for state, count in zip(states, counts, strict=True):
            for branch_index, (weight, branch) in enumerate(branches):
                segments = None if branch.caption_segments is None else branch.caption_segments[state.sample_index]
                if segments is None:
                    continue
                selected = caption_index_for_interval(segments, boundary(state.next_video), boundary(state.next_video + count))
                if self.qwen_visual_context:
                    assert state.prompt_texts is not None
                    state.prompt_texts[branch_index] = segments[selected]["prompt"]
                else:
                    assert branch.caption_embeds is not None and branch.caption_embeds[state.sample_index] is not None
                    state.prompt_branches[branch_index] = (weight, branch.caption_embeds[state.sample_index][selected])

    @torch.no_grad()
    def _iter_stream_latents(self, backbone: Any, branch_inputs: Sequence[tuple[float, StreamingBatch]],
                            rngs: Sequence[Any], *, models: dict[str, Any] | None = None) -> Iterator[StreamingLatentChunk]:
        inputs = branch_inputs[0][1]
        assert len(rngs) == inputs.batch_size and inputs.batch_size > 0
        device = inputs.prompt_embeds[0].device
        references = inputs.reference_latents
        states = self.start_stream(
            prompt_embeds=inputs.prompt_embeds,
            video_geometries=[(shape[0], *shape[2:]) for shape in inputs.video_shapes],
            reference_geometries=(None if references is None else [(value.shape[0], *value.shape[2:]) for value in references]),
            audio_channels=[shape[1] for shape in inputs.audio_shapes], rngs=rngs,
            audio_lengths=[shape[2] for shape in inputs.audio_shapes],
            video_lengths=[shape[1] for shape in inputs.video_shapes],
            streaming_configs=inputs.streaming_configs,
            conditioning_branches=[(weight, branch.prompt_embeds) for weight, branch in branch_inputs],
            **({"prompts": inputs.prompts, "conditioning_prompts": [(weight, branch.prompts) for weight, branch in branch_inputs]}
               if self.qwen_visual_context else {}),
        )
        starts = [streaming_window_starts(
            shape[1], **state.streaming_config, video_temporal_mapping=self.video_temporal_mapping,
        ) for shape, state in zip(inputs.video_shapes, states, strict=True)]
        rounds = torch.tensor(max(map(len, starts)), device=device)
        all_reduce_max(rounds)
        for round_index in range(int(rounds)):
            active = [index for index, values in enumerate(starts) if round_index < len(values)]
            if active:
                selected_states, counts, final_flags = [], [], []
                selected_references = [] if references is not None else None
                for index in active:
                    state = states[index]
                    stop = min(streaming_window_stop(
                        state.next_video, state.streaming_config, video_temporal_mapping=self.video_temporal_mapping,
                    ), inputs.video_shapes[index][1])
                    counts.append(stop - state.next_video)
                    if references is not None:
                        selected_references.append(references[index][:, state.next_video:stop])
                    selected_states.append(state)
                    final_flags.append(round_index + 1 == len(starts[index]))
                self._select_stream_captions(selected_states, counts, branch_inputs)
                yield self.step(backbone, selected_states, selected_references, video_counts=counts, is_last=final_flags,
                                **({"models": models, "reference_pixels": self._offline_reference_pixels(inputs, selected_states, counts)}
                                   if self.qwen_visual_context else {}))
            else:
                # Finished groups preserve collective order without advancing real RNGs.
                dummy = self.start_stream(
                    prompt_embeds=inputs.prompt_embeds[:1], video_geometries=[(inputs.video_shapes[0][0], *inputs.video_shapes[0][2:])],
                    reference_geometries=(None if references is None else [(references[0].shape[0], *references[0].shape[2:])]),
                    audio_channels=[inputs.audio_shapes[0][1]], audio_lengths=[inputs.audio_shapes[0][2]],
                    streaming_configs=inputs.streaming_configs[:1],
                    rngs=[RandomState(combine_seed(rngs[0].seed, "padding_round", round_index))],
                    conditioning_branches=[(weight, branch.prompt_embeds[:1]) for weight, branch in branch_inputs],
                    **({"prompts": inputs.prompts[:1], "conditioning_prompts": [(weight, branch.prompts[:1]) for weight, branch in branch_inputs]}
                       if self.qwen_visual_context else {}),
                )
                count = min(self._bootstrap_size(dummy[0].streaming_config), inputs.video_shapes[0][1])
                dummy_references = None if references is None else [references[0][:, :count]]
                self._select_stream_captions(dummy, [count], branch_inputs)
                self.step(backbone, dummy, dummy_references, video_counts=[count], is_last=[True],
                          **({"models": models, "reference_pixels": self._offline_reference_pixels(inputs, dummy, [count])}
                             if self.qwen_visual_context else {}))

    def stream_latents(self, backbone: Any, inputs: StreamingBatch, rngs: Sequence[Any], *,
                       guidance_scale: float = 1.0,
                       models: dict[str, Any] | None = None) -> Iterator[StreamingLatentChunk]:
        """Yield new AV latents with fixed text or freshly encoded window conditions."""
        if not math.isfinite(guidance_scale):
            raise ValueError("sampling guidance_scale must be finite")
        branches = [(guidance_scale, inputs)]
        if guidance_scale != 1:
            branches.append((1 - guidance_scale, self._captionless_batch(inputs)))
        yield from self._iter_stream_latents(backbone, branches, rngs, **({"models": models} if self.qwen_visual_context else {}))

    @staticmethod
    def _captionless_batch(inputs: StreamingBatch) -> StreamingBatch:
        """The same streams without language: empty prompts, embeddings and caption segments."""
        return replace(
            inputs, prompt_embeds=[value[:0] for value in inputs.prompt_embeds],
            prompts=None if inputs.prompts is None else [""] * inputs.batch_size,
            caption_segments=(None if inputs.caption_segments is None else [
                None if segments is None else [dict(item, prompt="") for item in segments]
                for segments in inputs.caption_segments
            ]),
            caption_embeds=(None if inputs.caption_embeds is None else [
                None if values is None else [value[:0] for value in values] for values in inputs.caption_embeds
            ]),
        )

    def stream(self, models: dict[str, Any], inputs: StreamingBatch, rngs: Sequence[Any], *,
               guidance_scale: float = 1.0) -> Iterator[StreamingLatentChunk]:
        """Yield new AV latents and decoded outputs with explicit timeline offsets.

        Each sample owns an independent codec adapter. Its outputs can be
        delayed or empty, and its final chunk contains both push and flush
        events. A codec without incremental decoding emits its complete RGB
        sequence at flush. Transformer history is unrelated to codec state.
        """
        self._check_codec(models["video_vae"])
        policies = [self._streaming_policy(policy) for policy in inputs.streaming_configs]
        decoders = [self._streaming_decoder(
            models,
            audio_lookahead_latents=policy["audio_lookahead_latents"],
            audio_right_lookahead_latents=policy.get("audio_right_lookahead_latents"),
        ) for policy in policies]
        for chunk in self.stream_latents(models["backbone"], inputs, rngs, guidance_scale=guidance_scale,
                                         **({"models": models} if self.qwen_visual_context else {})):
            decoded = []
            for index, video, audio, lookahead, is_last in zip(chunk.sample_indices, chunk.video, chunk.audio, chunk.audio_lookahead, chunk.is_last, strict=True):
                events = [decoders[index].push(video=video, audio=audio, audio_lookahead=lookahead)]
                if is_last:
                    events.append(decoders[index].flush())
                decoded.append(events)
            yield replace(chunk, decoded=decoded)

    @torch.no_grad()
    def _guided_rollout_latents(self, backbone: Any, branch_inputs: Sequence[tuple[float, StreamingBatch]],
                                rngs: Sequence[Any], *, first_branch_without_history: bool = False,
                                models: dict[str, Any] | None = None) -> StreamingRollout:
        assert not first_branch_without_history
        inputs = branch_inputs[0][1]
        device = inputs.prompt_embeds[0].device
        video = [torch.zeros(shape, device=device) for shape in inputs.video_shapes]
        audio = [torch.zeros(shape, device=device) for shape in inputs.audio_shapes]
        chunks = []
        for chunk in self._iter_stream_latents(backbone, branch_inputs, rngs, **({"models": models} if self.qwen_visual_context else {})):
            chunks.append(chunk)
            for index, vi, ai, v, a in zip(chunk.sample_indices, chunk.video_indices, chunk.audio_indices, chunk.video, chunk.audio, strict=True):
                video[index].index_copy_(1, vi.to(device), v)
                audio[index].index_copy_(2, ai.to(device), a)
        policies = [self._streaming_policy(policy) for policy in inputs.streaming_configs]
        return StreamingRollout(
            video, audio, chunks=chunks,
            audio_lookahead_latents=[policy["audio_lookahead_latents"] for policy in policies],
            audio_right_lookahead_latents=(
                [self._audio_right_lookahead(policy) for policy in policies]
                if any(policy.get("audio_right_lookahead_latents") is not None for policy in policies)
                else None
            ),
        )

    def _validation_rollout(self, models: dict[str, Any], backbone: Any, branch_inputs: Any,
                            rngs: Sequence[Any], *, guidance_scale: float, previous_guidance_scale: float) -> StreamingRollout:
        if not self.qwen_visual_context:
            return super()._validation_rollout(models, backbone, branch_inputs, rngs,
                                              guidance_scale=guidance_scale, previous_guidance_scale=previous_guidance_scale)
        assert previous_guidance_scale == 1.0
        return self._guided_rollout_latents(backbone, branch_inputs, rngs, models=models)

    def _decode_latents(self, models: dict[str, Any], latents: StreamingRollout, index: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
        """Replay the same current-plus-future codec calls used by live output."""
        assert latents.chunks is not None and latents.audio_lookahead_latents is not None
        decoder = self._streaming_decoder(
            models, audio_lookahead_latents=latents.audio_lookahead_latents[index],
            audio_right_lookahead_latents=(
                None if latents.audio_right_lookahead_latents is None
                else latents.audio_right_lookahead_latents[index]
            ),
        )
        frames, waveforms = [], []
        video_cursor = audio_cursor = 0
        for chunk in latents.chunks:
            if index not in chunk.sample_indices:
                continue
            local = chunk.sample_indices.index(index)
            events = [decoder.push(video=chunk.video[local], audio=chunk.audio[local],
                                   audio_lookahead=chunk.audio_lookahead[local])]
            if chunk.is_last[local]:
                events.append(decoder.flush())
            for event in events:
                if event.video is not None:
                    assert event.video_start == video_cursor
                    video_cursor += event.video.shape[2]
                    frames.append(event.video)
                if event.audio is not None:
                    assert event.audio_start == audio_cursor
                    audio_cursor += event.audio.shape[1]
                    waveforms.append(event.audio)
        assert frames
        audio = torch.cat(waveforms, dim=1) if waveforms else latents.audio[index].new_empty((2, 0))
        return torch.cat(frames, dim=2), audio

    @torch.no_grad()
    def _rollout_latents(self, backbone: Any, inputs: StreamingBatch, rng: Any, *, keep_trajectory: bool,
                         rngs: Sequence[Any] | None = None, trajectory_x0_chunks: Any = None) -> tuple[StreamingRollout, list[Any]]:
        assert not keep_trajectory and trajectory_x0_chunks is None
        if rngs is None:
            rngs = [rng.fork("streaming_sample", index) for index in range(inputs.batch_size)]
        return self._guided_rollout_latents(backbone, [(1.0, inputs)], rngs), []


EntryClass = MiniMaxH3StreamingSFT

__all__ = ["MiniMaxH3StreamingSFT", "StreamingBatch", "StreamingState", "StreamingLatentChunk", "StreamingRollout", "EntryClass"]
