# SPDX-License-Identifier: Apache-2.0
"""Native bidirectional Ref2VA validation for MiniMax-H3."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from dev.yanzuolu.common.distributed.ops import get_device
from dev.yanzuolu.common.seed import combine_seed
from dev.yanzuolu.projects.minimax_h3.meta_models.minimax_h3_base import MiniMaxH3Base
from dev.yanzuolu.projects.minimax_h3.modeling.time_request import (
    H3_VIDEO_TEMPORAL_MAPPING,
    VideoTemporalMapping,
)
from dev.yanzuolu.projects.minimax_h3.modeling.packed_tokens import minimax_h3_unpatchify_video_tokens
from dev.yanzuolu.projects.minimax_h3_videoref.modeling.packing import (
    minimax_h3_packed_sequence_ref2va_blocks,
)
from dev.yanzuolu.projects.minimax_h3_videoref.modeling.ref2va_encoder import (
    MINIMAX_H3_QWEN3VL_HIDDEN_DIM,
    MiniMaxH3Ref2VAPresentationProcessor,
    Ref2VAPresentation,
)
from dev.yanzuolu.projects.minimax_h3_videoref.modeling.ref2va_reference import (
    EncodedReferencePlan,
    encode_ref_block_plan,
    parse_ref_block_plan,
)
from dev.yanzuolu.projects.minimax_h3_videoref.modeling.video_ref_conditions import video_ref_corpus_metadata

_AUDIO_CHANNELS = 2
_VIDEO_PATCH_SIZE = (1, 2, 2)
_URI_SCHEME = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*:")


@dataclass(frozen=True)
class _Ref2VALayout:
    latent_shape: tuple[int, int, int, int]
    audio_shape: tuple[int, int, int]
    video_temporal_mapping: VideoTemporalMapping = field(
        default=H3_VIDEO_TEMPORAL_MAPPING, kw_only=True
    )

    @property
    def video_patch_grid(self) -> tuple[int, int, int]:
        _, latent_t, latent_h, latent_w = self.latent_shape
        return latent_t, latent_h // _VIDEO_PATCH_SIZE[1], latent_w // _VIDEO_PATCH_SIZE[2]


@dataclass(frozen=True)
class _Ref2VAInputs:
    batch_size: int
    prompt_embeds: list[torch.Tensor]
    text_lens: list[int]
    seqlens: torch.Tensor
    layouts: list[_Ref2VALayout]
    native: list[dict[str, Any]]
    reference_plans: list[EncodedReferencePlan]
    token_tags: torch.Tensor


def _processor_path(validation: Any) -> str:
    configured = validation.get("processor_path", None)
    if configured is None:
        raise ValueError("validation.processor_path must be supplied explicitly")
    return str(configured)


def _local_media_path(value: str, *, base: Path, location: str) -> str:
    if "\x00" in value:
        raise ValueError(f"{location} local media path must not contain NUL")
    path = Path(value)
    candidate = path if path.is_absolute() else base / path
    if candidate.is_symlink():
        raise ValueError(
            f"{location} local media path must not be a symlink: {value!r}"
        )
    if not candidate.exists():
        raise ValueError(f"{location} local media path does not exist: {value!r}")
    if not candidate.is_file():
        raise ValueError(
            f"{location} local media path must be a regular file: {value!r}"
        )
    return str(candidate.resolve(strict=True))


def _normalize_jsonl_request(
    request: Any,
    *,
    jsonl_parent: Path,
    line_number: int,
) -> dict[str, Any]:
    location = f"validation.requests_jsonl line {line_number}"
    if not isinstance(request, Mapping):
        raise ValueError(f"{location} must be a JSON object")

    prompt = request.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError(f"{location} prompt must be a non-empty string")
    conditions = request.get("conditions")
    if not isinstance(conditions, Sequence) or isinstance(
        conditions, (str, bytes, bytearray)
    ):
        raise ValueError(f"{location} conditions must be a non-string sequence")

    normalized_conditions: list[dict[str, Any]] = []
    for index, condition in enumerate(conditions):
        condition_location = f"{location} conditions[{index}]"
        if not isinstance(condition, Mapping):
            raise ValueError(f"{condition_location} must be a mapping")
        field = (
            "path"
            if "path" in condition
            else "uri"
            if "uri" in condition
            else None
        )
        if field is None:
            raise ValueError(f"{condition_location} requires path or uri")
        value = condition[field]
        if not isinstance(value, str) or not value:
            raise ValueError(
                f"{condition_location}.{field} must be a non-empty string"
            )

        normalized = dict(condition)
        if _URI_SCHEME.match(value) is None:
            normalized[field] = _local_media_path(
                value,
                base=jsonl_parent,
                location=f"{condition_location}.{field}",
            )
        normalized_conditions.append(normalized)

    normalized_request = dict(request)
    normalized_request["conditions"] = normalized_conditions
    return normalized_request


def _requests_from_jsonl(configured_path: Any) -> list[dict[str, Any]]:
    configured = Path(configured_path)
    candidate = configured if configured.is_absolute() else Path.cwd() / configured
    if candidate.is_symlink():
        raise ValueError(
            "validation.requests_jsonl must not be a symlink: "
            f"{str(configured)!r}"
        )
    if not candidate.exists():
        raise ValueError(
            f"validation.requests_jsonl does not exist: {str(configured)!r}"
        )
    if not candidate.is_file():
        raise ValueError(
            "validation.requests_jsonl must be a regular file: "
            f"{str(configured)!r}"
        )

    jsonl_path = candidate.resolve(strict=True)
    requests: list[dict[str, Any]] = []
    with jsonl_path.open("rb") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            location = f"validation.requests_jsonl line {line_number}"
            try:
                line = raw_line.decode("utf-8")
            except UnicodeDecodeError as error:
                raise ValueError(f"{location} must be valid UTF-8") from error
            if not line.strip():
                raise ValueError(f"{location} must not be empty")
            try:
                request = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"{location} contains invalid JSON: {error.msg}"
                ) from error
            requests.append(
                _normalize_jsonl_request(
                    request,
                    jsonl_parent=jsonl_path.parent,
                    line_number=line_number,
                )
            )

    if not requests:
        raise ValueError("validation.requests_jsonl contains no requests")
    return requests


def _load_validation_requests(validation: Any) -> list[Any]:
    inline_config = validation.get("requests", None)
    inline_requests = [] if inline_config is None else list(inline_config)
    jsonl_config = validation.get("requests_jsonl", None)
    if jsonl_config is not None and inline_requests:
        raise ValueError(
            "validation.requests_jsonl and non-empty validation.requests are "
            "mutually exclusive"
        )

    requests = (
        _requests_from_jsonl(jsonl_config)
        if jsonl_config is not None
        else inline_requests
    )
    if not requests:
        raise ValueError(
            "validation requires non-empty requests or validation.requests_jsonl"
        )
    limit = int(validation.get("num_prompts", 0))
    return requests[:limit] if limit > 0 else requests


def _strict_suffix_mask(
    mask: torch.Tensor,
    *,
    target_rows: int,
    name: str,
) -> None:
    assert mask.dtype == torch.bool and mask.ndim == 1, (
        f"{name} must be a one-dimensional bool tensor"
    )
    assert int(mask.sum().item()) == int(target_rows), (
        f"{name} has {int(mask.sum().item())} target rows but expected {target_rows}"
    )
    prefix = int(mask.numel()) - int(target_rows)
    assert not bool(mask[:prefix].any()), f"{name} reference prefix must be false"
    assert bool(mask[prefix:].all()), f"{name} target suffix must be true"


def _target_row_counts(layout: _Ref2VALayout) -> tuple[int, int]:
    latent_t, patch_h, patch_w = layout.video_patch_grid
    return latent_t * patch_h * patch_w, layout.audio_shape[2] * _AUDIO_CHANNELS


def _native_with_qwen_tags(
    *,
    presentation: Ref2VAPresentation,
    plan: EncodedReferencePlan,
    layout: _Ref2VALayout,
    device: torch.device,
) -> dict[str, Any]:
    latent_channels, latent_t, latent_h, latent_w = layout.latent_shape
    audio_channels, _, audio_t = layout.audio_shape
    assert latent_channels * _VIDEO_PATCH_SIZE[1] * _VIDEO_PATCH_SIZE[2] == 96
    assert audio_channels == _AUDIO_CHANNELS

    text_len = int(presentation.input_ids.numel())
    native = minimax_h3_packed_sequence_ref2va_blocks(
        text_len=text_len,
        latent_t=latent_t,
        latent_h=latent_h,
        latent_w=latent_w,
        audio_t=audio_t,
        ref_blocks=plan.ref_blocks,
        audio_channel=_AUDIO_CHANNELS,
        video_temporal_mapping=layout.video_temporal_mapping,
    )
    text_pos = native["text_pos"].to(torch.long)
    qwen_tags = presentation.text_token_tags.to(dtype=torch.long, device="cpu")
    assert list(qwen_tags.shape) == [text_len]
    native["token_tags"][text_pos] = qwen_tags
    native = {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in native.items()
    }

    target_visual_rows, target_audio_rows = _target_row_counts(layout)
    _strict_suffix_mask(
        native["update_mask"],
        target_rows=target_visual_rows,
        name="update_mask",
    )
    _strict_suffix_mask(
        native["audio_update_mask"],
        target_rows=target_audio_rows,
        name="audio_update_mask",
    )
    assert int(native["img_pos"].numel()) == (
        int(plan.visual_rows.shape[0]) + target_visual_rows
    )
    assert int(native["audio_pos"].numel()) == (
        int(plan.audio_rows.shape[0]) + target_audio_rows
    )
    return native


class MiniMaxH3Ref2VABase(MiniMaxH3Base):
    """Thin native Ref2VA specialization of the released dense H3 sampler."""

    def _validation_requests(self, validation: Any) -> list[Any]:
        return _load_validation_requests(validation)

    @staticmethod
    def _validation_request_prompt(request: Any) -> str:
        return request["prompt"]

    def _validation_inputs_for_requests(
        self,
        config: Any,
        models: dict[str, Any],
        requests: Sequence[Any],
        *,
        prompts: Sequence[str] | None = None,
    ) -> _Ref2VAInputs:
        validation = config.validation
        request_prompts = (
            [self._validation_request_prompt(request) for request in requests]
            if prompts is None
            else prompts
        )

        packer = self._validation_packer(config)
        latent_shape = tuple(int(value) for value in packer.latent_shape)
        audio_shape = tuple(int(value) for value in packer.audio_shape)
        if len(latent_shape) != 4 or len(audio_shape) != 3:
            raise ValueError(
                f"unexpected target geometry {latent_shape!r} and {audio_shape!r}"
            )
        layout = _Ref2VALayout(latent_shape=latent_shape, audio_shape=audio_shape)
        processor = MiniMaxH3Ref2VAPresentationProcessor.from_pretrained(
            _processor_path(validation)
        )
        target_frame_count = int(validation.num_frames)
        default_ref_seed = int(validation.get("ref_noise_seed", validation.seed))
        visual_anchor = float(
            validation.get(
                "visual_anchor",
                validation.get("imgvid_cond_noise_aug_for_inference", 0.999),
            )
        )
        audio_anchor = float(validation.get("audio_anchor", 1.0))

        plans: list[EncodedReferencePlan] = []
        presentations: list[Ref2VAPresentation] = []
        native: list[dict[str, Any]] = []
        device = get_device()
        for request, prompt in zip(requests, request_prompts, strict=True):
            specs = parse_ref_block_plan(
                request["conditions"],
                request["media_probes"],
                resolved_shapes=request["resolved_shapes"],
                target_frame_count=target_frame_count,
            )
            ref_noise_seed = int(
                request.get(
                    "ref_noise_seed",
                    request.get("seed", default_ref_seed),
                )
            )
            plan = encode_ref_block_plan(
                specs,
                video_vae=models["video_vae"],
                audio_vae=models["audio_vae"],
                target_latent_t=latent_shape[1],
                encode_seed=combine_seed(
                    int(request.get("seed", validation.seed)), "reference_encode"
                ),
                noise_seed=ref_noise_seed,
                visual_anchor=visual_anchor,
                audio_anchor=audio_anchor,
                **({"retain_clean_visual": True} if validation.get("save_latent", False) else {}),
            )
            presentation = processor.build(prompt, plan.qwen_media)
            plans.append(plan)
            presentations.append(presentation)
            native.append(
                _native_with_qwen_tags(
                    presentation=presentation,
                    plan=plan,
                    layout=layout,
                    device=device,
                )
            )

        text_lens = [int(item.input_ids.numel()) for item in presentations]
        prompt_embeds = self._encode_prompts(models, presentations, text_lens)
        seq_lens = [int(entry["seq_len"]) for entry in native]
        rows = sum(seq_lens)
        budget = int(packer.max_seqlen)
        assert rows <= budget, (
            f"native Ref2VA batch of {len(requests)} requests needs {rows} rows "
            f"but target packer budget is {budget}"
        )
        return _Ref2VAInputs(
            batch_size=len(requests),
            prompt_embeds=[embedding.to(device) for embedding in prompt_embeds],
            text_lens=text_lens,
            seqlens=torch.tensor(seq_lens, dtype=torch.int32, device=device),
            layouts=[layout for _ in requests],
            native=native,
            reference_plans=plans,
            token_tags=torch.cat([entry["token_tags"] for entry in native]),
        )

    def _validation_frames(
        self, frames: torch.Tensor, inputs: _Ref2VAInputs, *, index: int
    ) -> torch.Tensor:
        """Reference videos in request order on the left, the sample on the right."""
        panels = []
        height = int(frames.shape[3])
        for block in inputs.reference_plans[index].blocks:
            if block.spec.kind not in ("video", "video_audio"):
                continue
            # The frames the reference encoder saw: 24 fps, resized, [T, H, W, 3] uint8.
            panel = torch.from_numpy(block.visual_media).permute(3, 0, 1, 2)[None]
            assert panel.shape[2] == frames.shape[2], (
                f"reference has {panel.shape[2]} frames, sample has {frames.shape[2]}"
            )
            if panel.shape[3] != height:
                width = round(panel.shape[4] * height / panel.shape[3])
                panel = torch.nn.functional.interpolate(
                    panel[0].permute(1, 0, 2, 3).float(), size=(height, width),
                    mode="bilinear", antialias=True,
                ).permute(1, 0, 2, 3)[None]
            panels.append(panel.to(frames.dtype) / 255)
        if not panels:
            return frames
        return torch.cat([*panels, frames.cpu()], dim=4)

    def _validation_latent_metadata(
        self, config: Any, inputs: _Ref2VAInputs, request: Any, *, index: int,
    ) -> dict[str, Any]:
        blocks = inputs.reference_plans[index].blocks
        if len(blocks) != 1:
            return {}
        block = blocks[0]
        spec = block.spec
        if (spec.kind != "video" or block.audio_t != 0
                or spec.resolved_frame_count != int(config.validation.num_frames)):
            return {}
        assert block.clean_visual_rows is not None
        reference = minimax_h3_unpatchify_video_tokens(
            block.clean_visual_rows,
            latent_shape=(block.latent_t, block.latent_h // 2, block.latent_w // 2, 24),
            patch_size=_VIDEO_PATCH_SIZE,
        )[0]
        return video_ref_corpus_metadata(
            reference, path=spec.path, num_frames=spec.resolved_frame_count,
            height=spec.resolved_height, width=spec.resolved_width,
            video_temporal_mapping=spec.video_temporal_mapping,
            start_time_seconds=float(spec.start_time_seconds),
        )

    def _encode_prompts(
        self,
        models: dict[str, Any],
        text_input_ids: Sequence[Any],
        text_lens: Sequence[int],
    ) -> list[torch.Tensor]:
        assert len(text_input_ids) == len(text_lens)
        encoded: list[torch.Tensor] = []
        for presentation, text_len in zip(text_input_ids, text_lens):
            if not isinstance(presentation, Ref2VAPresentation):
                raise TypeError("native Ref2VA prompt encoding requires presentations")
            assert int(presentation.input_ids.numel()) == int(text_len)
            hidden = models["text_encoder"].encode(presentation)
            assert list(hidden.shape) == [
                int(presentation.input_ids.numel()),
                MINIMAX_H3_QWEN3VL_HIDDEN_DIM,
            ], f"unexpected Ref2VA Qwen hidden shape {list(hidden.shape)}"
            encoded.append(hidden)
        return encoded

    def _bidirectional_kwargs(
        self,
        model: Any,
        inputs: _Ref2VAInputs,
        *,
        video_xts: list[torch.Tensor],
        audio_xts: list[torch.Tensor],
        video_timesteps: torch.Tensor,
        audio_timesteps: torch.Tensor,
    ) -> tuple[dict[str, Any], torch.Tensor, torch.Tensor, torch.Tensor]:
        """Pack native reference rows while selecting only target output rows."""
        device = inputs.token_tags.device
        assert len(video_xts) == len(audio_xts) == inputs.batch_size
        assert len(inputs.native) == len(inputs.reference_plans) == inputs.batch_size
        video_timesteps = video_timesteps.to(device=device, dtype=torch.float32)
        audio_timesteps = audio_timesteps.to(device=device, dtype=torch.float32)

        video_blocks: list[torch.Tensor] = []
        audio_blocks: list[torch.Tensor] = []
        row_timesteps: list[torch.Tensor] = []
        img_pos: list[torch.Tensor] = []
        target_img_pos: list[torch.Tensor] = []
        audio_pos: list[torch.Tensor] = []
        text_pos: list[torch.Tensor] = []
        update_masks: list[torch.Tensor] = []
        audio_update_masks: list[torch.Tensor] = []
        cu_host: list[int] = [0]
        offset = 0

        for index, (_layout, native, plan) in enumerate(
            zip(inputs.layouts, inputs.native, inputs.reference_plans)
        ):
            seq_len = int(native["seq_len"])
            sample_img = native["img_pos"].to(torch.long)
            sample_audio = native["audio_pos"].to(torch.long)
            visual_mask = native["update_mask"]
            audio_mask = native["audio_update_mask"]
            target_visual_rows = int(visual_mask.sum().item())
            target_audio_rows = int(audio_mask.sum().item())
            _strict_suffix_mask(
                visual_mask,
                target_rows=target_visual_rows,
                name=f"native[{index}].update_mask",
            )
            _strict_suffix_mask(
                audio_mask,
                target_rows=target_audio_rows,
                name=f"native[{index}].audio_update_mask",
            )

            target_video = self._video_rows(
                video_xts[index], 0, video_xts[index].shape[1]
            )
            target_audio = self._audio_rows(
                audio_xts[index], 0, audio_xts[index].shape[2]
            )
            assert int(target_video.shape[0]) == target_visual_rows
            assert int(target_audio.shape[0]) == target_audio_rows
            visual_rows = torch.cat(
                [
                    plan.visual_rows.to(
                        device=device, dtype=target_video.dtype
                    ),
                    target_video,
                ],
                dim=0,
            )
            audio_rows = torch.cat(
                [
                    plan.audio_rows.to(
                        device=device, dtype=target_audio.dtype
                    ),
                    target_audio,
                ],
                dim=0,
            )
            assert int(visual_rows.shape[0]) == int(sample_img.numel())
            assert int(audio_rows.shape[0]) == int(sample_audio.numel())

            video_blocks.append(
                target_video.new_zeros((seq_len, int(target_video.shape[1]))).index_copy(
                    0, sample_img, visual_rows
                )
            )
            audio_blocks.append(
                target_audio.new_zeros((seq_len, int(target_audio.shape[1]))).index_copy(
                    0, sample_audio, audio_rows
                )
            )

            video_t = video_timesteps[index]
            audio_t = audio_timesteps[index]
            sample_t = video_t.expand(seq_len).clone()
            ref_visual_rows = int(plan.visual_rows.shape[0])
            ref_audio_rows = int(plan.audio_rows.shape[0])
            if ref_visual_rows:
                visual_anchors = plan.visual_row_anchors.to(device=device)
                assert list(visual_anchors.shape) == [ref_visual_rows]
                sample_t[sample_img[:ref_visual_rows]] = torch.minimum(
                    video_t.expand(ref_visual_rows), 1.0 - visual_anchors
                )
            sample_t[sample_img[ref_visual_rows:]] = video_t
            if ref_audio_rows:
                audio_anchors = plan.audio_row_anchors.to(device=device)
                assert list(audio_anchors.shape) == [ref_audio_rows]
                sample_t[sample_audio[:ref_audio_rows]] = torch.minimum(
                    audio_t.expand(ref_audio_rows), 1.0 - audio_anchors
                )
            sample_t[sample_audio[ref_audio_rows:]] = audio_t
            row_timesteps.append(sample_t)

            img_pos.append(sample_img + offset)
            target_img_pos.append(sample_img[visual_mask] + offset)
            audio_pos.append(sample_audio + offset)
            text_pos.append(native["text_pos"].to(torch.long) + offset)
            update_masks.append(visual_mask)
            audio_update_masks.append(audio_mask)
            cu_host.extend(
                int(value) + offset for value in native["cu_seqlens"][1:].tolist()
            )
            offset += seq_len

        packed_cu_host = tuple(cu_host)
        packed_cu = torch.tensor(packed_cu_host, dtype=torch.int32, device=device)
        max_seqlen = max(
            stop - start
            for start, stop in zip(packed_cu_host[:-1], packed_cu_host[1:])
        )
        all_img_pos = torch.cat(img_pos)
        infer_out_pos = torch.cat(target_img_pos)
        packed_update_mask = torch.cat(update_masks)
        assert int(infer_out_pos.numel()) == int(packed_update_mask.sum().item()), (
            "target-only infer_out_pos must contain exactly the update_mask target rows"
        )
        if bool((~packed_update_mask).any()):
            assert int(infer_out_pos.numel()) != int(packed_update_mask.numel()), (
                "target-only infer_out_pos must exclude Ref2VA visual reference rows"
            )
        all_audio_pos = torch.cat(audio_pos)
        x = torch.cat(video_blocks).unsqueeze(0).to(dtype=model.param_dtype)
        audio_x = torch.cat(audio_blocks).unsqueeze(0).to(dtype=model.param_dtype)
        kwargs = self._common_kwargs(
            inputs,
            x=x,
            audio_x=audio_x,
            eps=torch.zeros_like(x),
            audio_eps=torch.zeros_like(audio_x),
            row_timesteps=torch.cat(row_timesteps),
            position_ids=torch.cat(
                [entry["img_position_ids"] for entry in inputs.native], dim=0
            ),
            token_tags=torch.cat(
                [entry["token_tags"] for entry in inputs.native], dim=0
            ),
            img_pos=all_img_pos,
            audio_pos=all_audio_pos,
            text_pos=torch.cat(text_pos),
            infer_out_pos=infer_out_pos,
        )
        kwargs.update(
            update_mask=packed_update_mask,
            update_audio_mask=torch.cat(audio_update_masks),
            # Target-only infer_out_pos has a different length from the full
            # golden-builder update_mask and therefore requires skip masking.
            skip_mask_out_condition=True,
            clean_timesteps_by_tag=(1.0, 0.999, 1.0),
            packed_seq_params={
                "cu_seqlens_q": packed_cu,
                "cu_seqlens_q_host": packed_cu_host,
                "max_seqlen_q": max_seqlen,
            },
        )
        target_audio_pos = all_audio_pos[torch.cat(audio_update_masks)]
        return kwargs, torch.cat(row_timesteps), infer_out_pos, target_audio_pos

    def _bidirectional_forward(
        self,
        model: Any,
        inputs: _Ref2VAInputs,
        *,
        video_xts: list[torch.Tensor],
        audio_xts: list[torch.Tensor],
        video_timesteps: torch.Tensor,
        audio_timesteps: torch.Tensor,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        kwargs, _, _, _ = self._bidirectional_kwargs(
            model,
            inputs,
            video_xts=video_xts,
            audio_xts=audio_xts,
            video_timesteps=video_timesteps,
            audio_timesteps=audio_timesteps,
        )
        video_logits, audio_logits = model(**kwargs)

        video_out: list[torch.Tensor] = []
        audio_out: list[torch.Tensor] = []
        video_cursor = 0
        audio_cursor = 0
        for index, (layout, native) in enumerate(zip(inputs.layouts, inputs.native)):
            visual_mask = native["update_mask"]
            audio_mask = native["audio_update_mask"]
            target_video_rows = int(visual_mask.sum().item())
            target_audio_rows = int(audio_mask.sum().item())
            _strict_suffix_mask(
                visual_mask,
                target_rows=target_video_rows,
                name=f"output[{index}].update_mask",
            )
            _strict_suffix_mask(
                audio_mask,
                target_rows=target_audio_rows,
                name=f"output[{index}].audio_update_mask",
            )
            video_rows = video_logits[
                video_cursor : video_cursor + target_video_rows
            ]
            sample_audio_rows = int(audio_mask.numel())
            all_sample_audio = audio_logits[
                audio_cursor : audio_cursor + sample_audio_rows
            ]
            target_audio = all_sample_audio[
                sample_audio_rows - target_audio_rows :
            ]
            assert int(video_rows.shape[0]) == target_video_rows
            assert int(target_audio.shape[0]) == target_audio_rows
            video_out.append(self._video_from_rows(video_rows, layout))
            audio_out.append(self._audio_from_rows(target_audio, layout))
            video_cursor += target_video_rows
            audio_cursor += sample_audio_rows

        assert video_cursor == int(video_logits.shape[0])
        assert audio_cursor == int(audio_logits.shape[0])
        return video_out, audio_out


__all__ = ["MiniMaxH3Ref2VABase"]
