# Copyright 2026 MiniMax
#
# Licensed under the Apache License, Version 2.0 (the "License")
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Native MiniMax-H3 Ref2VA presentation and Qwen3-VL layer-50 encoder."""

from __future__ import annotations

import math
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from os import PathLike
from typing import TYPE_CHECKING, Any, Literal

import torch
from torch import nn

from dev.yanzuolu.common.distributed.ops import get_device

if TYPE_CHECKING:
    from transformers import Qwen3VLModel

VISION_START = "<|vision_start|>"
VISION_END = "<|vision_end|>"
IMAGE_PAD = "<|image_pad|>"
VIDEO_PAD = "<|video_pad|>"

TEXT_TOKEN_TAG = 1
VIDEO_TOKEN_TAG = 0
MINIMAX_H3_QWEN3VL_HIDDEN_DIM = 5120
MINIMAX_H3_QWEN3VL_SELECTED_LM_LAYER = 50
_MINIMAX_H3_QWEN3VL_CHECKPOINT_LAYERS = 64
_QWEN_VIDEO_SOURCE_FPS = 24.0
_QWEN_VIDEO_SAMPLE_FPS = 2.0
assert _QWEN_VIDEO_SOURCE_FPS % _QWEN_VIDEO_SAMPLE_FPS == 0
_QWEN_VIDEO_SAMPLE_STRIDE = int(
    _QWEN_VIDEO_SOURCE_FPS / _QWEN_VIDEO_SAMPLE_FPS
)
_QWEN_TEMPORAL_PATCH = 2

_LAYER_WEIGHT_RE = re.compile(r"^model\.language_model\.layers\.(\d+)\.")

MediaKind = Literal["image", "audio", "video", "video_audio"]
Ref2VAMediaKind = MediaKind
ConditionKind = Literal["image", "audio", "video"]
VisionKind = Literal["image", "video"]


@dataclass(frozen=True)
class Ref2VAPresentationMedia:
    """One request-ordered Ref2VA condition.

    ``media`` is an image for ``image``, an opaque audio payload for ``audio``,
    and the shared 24 FPS ``[T,H,W,3]`` RGB frames for either video kind. Audio
    payloads never enter Qwen. Plain ``video`` requires the upstream probe's
    explicit ``has_audio`` bool. The ``video_audio`` kind is itself the upstream
    request specification that audio is present.

    Indexed videos supply one global 24 FPS frame index and one presentation
    timestamp per RGB row. Each continuous source interval is sampled and
    temporally padded independently while retaining a single video label.
    """

    kind: MediaKind
    media: Any = None
    has_audio: bool | None = None
    video_frame_indices: Sequence[int] | None = field(default=None, kw_only=True)
    video_frame_timestamps: Sequence[float] | None = field(default=None, kw_only=True)

    def __post_init__(self) -> None:
        if self.kind not in ("image", "audio", "video", "video_audio"):
            raise ValueError(f"unsupported Ref2VA media kind {self.kind!r}")
        if self.kind in ("image", "audio"):
            if self.video_frame_indices is not None or self.video_frame_timestamps is not None:
                raise ValueError("only video media can carry frame metadata")
            if self.has_audio is not None:
                raise ValueError(f"{self.kind} media must not set has_audio")
            if self.kind == "image" and self.media is None:
                raise ValueError("image media must carry an image")
            return
        if self.kind == "video" and not isinstance(self.has_audio, bool):
            raise TypeError(
                "video has_audio must be an explicit bool supplied by the "
                "upstream probe"
            )
        if self.kind == "video_audio" and self.has_audio not in (None, True):
            raise ValueError("video_audio media cannot declare has_audio=False")
        if self.media is None:
            raise ValueError(f"{self.kind} media must carry shared 24 FPS RGB frames")
        if (self.video_frame_indices is None) != (self.video_frame_timestamps is None):
            raise ValueError("video frame indices and timestamps must be supplied together")


@dataclass(frozen=True)
class Ref2VAConditionLabel:
    """One emitted label in presentation order."""

    kind: ConditionKind
    ordinal: int
    request_index: int


@dataclass(frozen=True)
class Ref2VAVisionMedia:
    """One Qwen vision input in request presentation order."""

    kind: VisionKind
    ordinal: int
    request_index: int


@dataclass(frozen=True)
class Ref2VAVideoBlock:
    """One timestamped Qwen temporal vision block."""

    request_index: int
    video_ordinal: int
    block_index: int
    pad_token_count: int
    timestamp_seconds: float


@dataclass(frozen=True)
class Ref2VAPresentation:
    """Aligned Qwen inputs and immutable Ref2VA ordering metadata.

    media_prefix_length includes all condition labels, timestamps and vision
    rows before the caption. It identifies the complete empty-language prefix.
    """

    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    position_ids: torch.Tensor | None
    text_token_tags: torch.Tensor
    pixel_values: torch.Tensor | None
    image_grid_thw: torch.Tensor | None
    pixel_values_videos: torch.Tensor | None
    video_grid_thw: torch.Tensor | None
    condition_order: tuple[Ref2VAConditionLabel, ...]
    vision_media_order: tuple[Ref2VAVisionMedia, ...]
    video_blocks: tuple[Ref2VAVideoBlock, ...]
    media_prefix_length: int = field(default=0, kw_only=True)

    def __post_init__(self) -> None:
        if self.input_ids.ndim != 1:
            raise ValueError(f"input_ids must be 1-D, got {list(self.input_ids.shape)}")
        sequence_length = int(self.input_ids.shape[0])
        if list(self.attention_mask.shape) != [sequence_length]:
            raise ValueError("attention_mask must align with input_ids")
        if list(self.text_token_tags.shape) != [sequence_length]:
            raise ValueError("text_token_tags must align with input_ids")
        if not 0 <= self.media_prefix_length <= sequence_length:
            raise ValueError("media_prefix_length must lie inside the presentation")
        if (self.pixel_values is None) != (self.image_grid_thw is None):
            raise ValueError("pixel_values and image_grid_thw must be given together")
        if (self.pixel_values_videos is None) != (self.video_grid_thw is None):
            raise ValueError(
                "pixel_values_videos and video_grid_thw must be given together"
            )

    def encoder_inputs(self) -> dict[str, torch.Tensor | None]:
        """Return the exact keyword payload accepted by the encoder forward."""

        return {
            "input_ids": self.input_ids,
            "attention_mask": self.attention_mask,
            "position_ids": self.position_ids,
            "pixel_values": self.pixel_values,
            "image_grid_thw": self.image_grid_thw,
            "pixel_values_videos": self.pixel_values_videos,
            "video_grid_thw": self.video_grid_thw,
        }


class _PresentationAccumulator:
    """Accumulate ids and tags together so their sequence order cannot drift."""

    def __init__(self) -> None:
        self.ids: list[int] = []
        self.tags: list[int] = []

    def text(self, token_ids: Sequence[int]) -> None:
        ids = [int(token_id) for token_id in token_ids]
        self.ids.extend(ids)
        self.tags.extend([TEXT_TOKEN_TAG] * len(ids))

    def vision(self, token_ids: Sequence[int]) -> None:
        ids = [int(token_id) for token_id in token_ids]
        self.ids.extend(ids)
        self.tags.extend([VIDEO_TOKEN_TAG] * len(ids))

    def build(self) -> tuple[torch.Tensor, torch.Tensor]:
        assert len(self.ids) == len(self.tags)
        return (
            torch.tensor(self.ids, dtype=torch.long),
            torch.tensor(self.tags, dtype=torch.long),
        )


def _text_ids(tokenizer: Any, text: str) -> list[int]:
    return [
        int(token_id)
        for token_id in tokenizer(text, add_special_tokens=False)["input_ids"]
    ]


def _vision_block_ids(tokenizer: Any, pad_token: str, count: int) -> list[int]:
    if int(count) <= 0:
        raise ValueError("vision pad token count must be positive")
    return [
        int(tokenizer.convert_tokens_to_ids(VISION_START)),
        *[int(tokenizer.convert_tokens_to_ids(pad_token))] * int(count),
        int(tokenizer.convert_tokens_to_ids(VISION_END)),
    ]


def _indexed_video_samples(
    frame_indices: Sequence[int], frame_timestamps: Sequence[float],
) -> tuple[list[int], list[float]]:
    """Sample disjoint intervals and include the final frame without crossing gaps."""
    indices = [int(index) for index in frame_indices]
    timestamps = [float(timestamp) for timestamp in frame_timestamps]
    if not indices or len(indices) != len(timestamps):
        raise ValueError("video indices and timestamps must be nonempty and aligned")
    if indices[0] < 0 or any(right <= left for left, right in zip(indices, indices[1:])):
        raise ValueError("video frame indices must be strictly increasing and nonnegative")
    if not all(math.isfinite(timestamp) for timestamp in timestamps):
        raise ValueError("video timestamps must be finite")
    starts = [0] + [
        index for index in range(1, len(indices)) if indices[index] != indices[index - 1] + 1
    ]
    selected: list[int] = []
    for start, stop in zip(starts, [*starts[1:], len(indices)], strict=True):
        interval = list(range(start, stop, _QWEN_VIDEO_SAMPLE_STRIDE))
        interval.extend([interval[-1]] * ((-len(interval)) % _QWEN_TEMPORAL_PATCH))
        selected.extend(interval)
    selected[-1] = len(indices) - 1
    block_timestamps = [
        (timestamps[selected[index]] + timestamps[selected[index + _QWEN_TEMPORAL_PATCH - 1]]) / 2
        for index in range(0, len(selected), _QWEN_TEMPORAL_PATCH)
    ]
    return selected, block_timestamps


def _sample_video_frames(
    frames: Any, *, frame_indices: Sequence[int] | None = None,
    frame_timestamps: Sequence[float] | None = None,
) -> tuple[torch.Tensor, list[float]]:
    """Sample native or indexed video frames with aligned temporal timestamps."""

    if not hasattr(frames, "shape") or len(frames.shape) != 4:
        raise ValueError("video frames must be a [T,H,W,C] RGB array or tensor")
    if int(frames.shape[0]) <= 0 or int(frames.shape[-1]) != 3:
        raise ValueError("video frames must be non-empty [T,H,W,3] RGB data")
    if frame_indices is not None:
        if frame_timestamps is None or len(frame_indices) != int(frames.shape[0]):
            raise ValueError("frame metadata must describe every supplied RGB row")
        selected, block_timestamps = _indexed_video_samples(frame_indices, frame_timestamps)
        sampled_frames = torch.as_tensor(frames).index_select(0, torch.tensor(selected))
        return sampled_frames.permute(0, 3, 1, 2).contiguous(), block_timestamps
    sampled_frames = frames[::_QWEN_VIDEO_SAMPLE_STRIDE]
    sampled_count = int(sampled_frames.shape[0])
    timestamps = [index / _QWEN_VIDEO_SAMPLE_FPS for index in range(sampled_count)]
    pad = (-len(timestamps)) % _QWEN_TEMPORAL_PATCH
    timestamps = timestamps + [timestamps[-1]] * pad
    block_timestamps = [
        (
            timestamps[index]
            + timestamps[index + _QWEN_TEMPORAL_PATCH - 1]
        )
        / 2
        for index in range(0, len(timestamps), _QWEN_TEMPORAL_PATCH)
    ]
    channels_first = torch.as_tensor(sampled_frames).permute(0, 3, 1, 2).contiguous()
    return channels_first, block_timestamps


def _append_video_presentation(
    presentation: _PresentationAccumulator, tokenizer: Any, ordinal: int,
    counts: Sequence[int], timestamps: Sequence[float],
) -> None:
    presentation.text(_text_ids(tokenizer, f"<Video {ordinal}>: "))
    for count, timestamp in zip(counts, timestamps, strict=True):
        presentation.text(_text_ids(tokenizer, f"<{float(timestamp):.1f} seconds>"))
        presentation.vision(_vision_block_ids(tokenizer, VIDEO_PAD, count))


def _append_image_presentation(
    presentation: _PresentationAccumulator, tokenizer: Any, ordinal: int, count: int,
) -> None:
    presentation.text(_text_ids(tokenizer, f"<Picture {ordinal}>: "))
    presentation.vision(_vision_block_ids(tokenizer, IMAGE_PAD, count))


class MiniMaxH3Ref2VAPresentationProcessor:
    """Build the exact SGLang Ref2VA Qwen presentation for RAVEN."""

    def __init__(self, tokenizer: Any, processor: Any) -> None:
        if tokenizer is None or processor is None:
            raise ValueError("tokenizer and processor are required")
        self.tokenizer = tokenizer
        self.processor = processor

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str | PathLike[str],
        **kwargs: Any,
    ) -> MiniMaxH3Ref2VAPresentationProcessor:
        """Load a local Qwen processor while keeping construction injectable."""

        from transformers import AutoProcessor

        kwargs.setdefault("local_files_only", True)
        processor = AutoProcessor.from_pretrained(
            pretrained_model_name_or_path,
            **kwargs,
        )
        return cls(processor.tokenizer, processor)

    def video_presentation_length(
        self, prompt: str, *, frame_indices: Sequence[int],
        frame_timestamps: Sequence[float], height: int, width: int,
    ) -> int:
        """Count an indexed video presentation using its actual processor grid."""
        from transformers.models.qwen3_vl.video_processing_qwen3_vl import smart_resize

        selected, timestamps = _indexed_video_samples(frame_indices, frame_timestamps)
        processor = self.processor.video_processor
        if int(processor.temporal_patch_size) != _QWEN_TEMPORAL_PATCH:
            raise ValueError("Ref2VA video presentations require temporal patch size 2")
        patch_size, merge_size = int(processor.patch_size), int(processor.merge_size)
        if processor.do_resize:
            height, width = smart_resize(
                num_frames=len(selected), height=height, width=width,
                temporal_factor=int(processor.temporal_patch_size),
                factor=patch_size * merge_size,
                min_pixels=int(processor.size["shortest_edge"]),
                max_pixels=int(processor.size["longest_edge"]),
            )
        if height % (patch_size * merge_size) or width % (patch_size * merge_size):
            raise ValueError("video dimensions must align with the processor's merged patches")
        per_block = (height // patch_size) * (width // patch_size) // merge_size**2
        presentation = _PresentationAccumulator()
        _append_video_presentation(presentation, self.tokenizer, 1,
                                   [per_block] * len(timestamps), timestamps)
        presentation.text(_text_ids(self.tokenizer, prompt))
        return len(presentation.ids)

    def image_presentation_length(self, *, height: int, width: int) -> int:
        """Count one leading image presentation using its actual processor grid."""
        from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize

        processor = self.processor.image_processor
        patch_size, merge_size = int(processor.patch_size), int(processor.merge_size)
        if processor.do_resize:
            height, width = smart_resize(
                height, width, factor=patch_size * merge_size,
                min_pixels=int(processor.size["shortest_edge"]),
                max_pixels=int(processor.size["longest_edge"]),
            )
        presentation = _PresentationAccumulator()
        _append_image_presentation(presentation, self.tokenizer, 1,
                                   (height // patch_size) * (width // patch_size) // merge_size**2)
        return len(presentation.ids)

    def build(
        self,
        prompt: str,
        ordered_media: Sequence[Ref2VAPresentationMedia],
    ) -> Ref2VAPresentation:
        """Build ids, synchronized tags, and Qwen pixels in request order."""

        media = tuple(ordered_media)
        if not prompt and not media:
            raise ValueError("prompt or reference media must be non-empty")

        counters: dict[ConditionKind, int] = {"image": 0, "audio": 0, "video": 0}
        condition_order: list[Ref2VAConditionLabel] = []
        vision_media_order: list[Ref2VAVisionMedia] = []
        images: list[Any] = []
        image_request_indices: list[int] = []
        videos: list[torch.Tensor] = []
        video_request_indices: list[int] = []
        video_timestamps_by_request: dict[int, list[float]] = {}

        for request_index, item in enumerate(media):
            if not isinstance(item, Ref2VAPresentationMedia):
                raise TypeError(
                    "ordered_media entries must be Ref2VAPresentationMedia instances"
                )
            if item.kind == "image":
                counters["image"] += 1
                condition_order.append(
                    Ref2VAConditionLabel("image", counters["image"], request_index)
                )
                vision_media_order.append(
                    Ref2VAVisionMedia("image", counters["image"], request_index)
                )
                images.append(item.media)
                image_request_indices.append(request_index)
                continue
            if item.kind == "audio":
                counters["audio"] += 1
                condition_order.append(
                    Ref2VAConditionLabel("audio", counters["audio"], request_index)
                )
                continue

            contributes_audio = item.kind == "video_audio" or bool(item.has_audio)
            if contributes_audio:
                counters["audio"] += 1
                condition_order.append(
                    Ref2VAConditionLabel("audio", counters["audio"], request_index)
                )
            counters["video"] += 1
            condition_order.append(
                Ref2VAConditionLabel("video", counters["video"], request_index)
            )
            vision_media_order.append(
                Ref2VAVisionMedia("video", counters["video"], request_index)
            )
            sampled, block_timestamps = _sample_video_frames(
                item.media, frame_indices=item.video_frame_indices,
                frame_timestamps=item.video_frame_timestamps,
            )
            videos.append(sampled)
            video_request_indices.append(request_index)
            video_timestamps_by_request[request_index] = block_timestamps

        pixel_values = None
        image_grid_thw = None
        image_counts_by_request: dict[int, int] = {}
        if images:
            image_output = self.processor.image_processor(
                images=images,
                return_tensors="pt",
            )
            pixel_values = image_output["pixel_values"]
            image_grid_thw = image_output["image_grid_thw"]
            if int(image_grid_thw.shape[0]) != len(images):
                raise ValueError(
                    f"expected {len(images)} image grids, got "
                    f"{list(image_grid_thw.shape)}"
                )
            merge_area = int(self.processor.image_processor.merge_size) ** 2
            for image_index, request_index in enumerate(image_request_indices):
                grid_tokens = int(image_grid_thw[image_index].prod().item())
                if grid_tokens % merge_area:
                    raise ValueError("image grid is not divisible by merge_size squared")
                image_counts_by_request[request_index] = grid_tokens // merge_area

        pixel_values_videos = None
        video_grid_thw = None
        video_counts_by_request: dict[int, list[int]] = {}
        video_blocks: list[Ref2VAVideoBlock] = []
        if videos:
            video_output = self.processor.video_processor(
                videos=videos,
                do_sample_frames=False,
                input_data_format="channels_first",
                return_tensors="pt",
            )
            pixel_values_videos = video_output["pixel_values_videos"]
            video_grid_thw = video_output["video_grid_thw"]
            if int(video_grid_thw.shape[0]) != len(videos):
                raise ValueError(
                    f"expected {len(videos)} video grids, got "
                    f"{list(video_grid_thw.shape)}"
                )
            merge_area = int(self.processor.image_processor.merge_size) ** 2
            video_ordinals = {
                vision.request_index: vision.ordinal
                for vision in vision_media_order
                if vision.kind == "video"
            }
            for video_index, request_index in enumerate(video_request_indices):
                temporal_blocks = int(video_grid_thw[video_index, 0].item())
                spatial_tokens = int(video_grid_thw[video_index, 1].item()) * int(
                    video_grid_thw[video_index, 2].item()
                )
                if spatial_tokens % merge_area:
                    raise ValueError("video grid is not divisible by merge_size squared")
                per_block = spatial_tokens // merge_area
                timestamps = video_timestamps_by_request[request_index]
                if len(timestamps) != temporal_blocks:
                    raise ValueError(
                        "video block count mismatch between Qwen processor and "
                        f"timestamps for request media {request_index}: "
                        f"{temporal_blocks} vs {len(timestamps)}"
                    )
                video_counts_by_request[request_index] = [per_block] * temporal_blocks
                for block_index, timestamp in enumerate(timestamps):
                    video_blocks.append(
                        Ref2VAVideoBlock(
                            request_index=request_index,
                            video_ordinal=video_ordinals[request_index],
                            block_index=block_index,
                            pad_token_count=per_block,
                            timestamp_seconds=float(timestamp),
                        )
                    )

        presentation = _PresentationAccumulator()
        rendered_conditions: list[Ref2VAConditionLabel] = []
        rendered_vision: list[Ref2VAVisionMedia] = []
        rendered_blocks: list[tuple[int, int]] = []
        vision_by_request = {
            vision.request_index: vision for vision in vision_media_order
        }
        for condition in condition_order:
            rendered_conditions.append(condition)
            if condition.kind == "audio":
                presentation.text(
                    _text_ids(self.tokenizer, f"<Audio {condition.ordinal}>: ")
                )
                continue
            if condition.kind == "image":
                _append_image_presentation(
                    presentation, self.tokenizer, condition.ordinal,
                    image_counts_by_request[condition.request_index],
                )
                rendered_vision.append(vision_by_request[condition.request_index])
                continue

            counts = video_counts_by_request[condition.request_index]
            timestamps = video_timestamps_by_request[condition.request_index]
            if not counts or len(counts) != len(timestamps):
                raise ValueError("video block token counts and timestamps must align")
            rendered_vision.append(vision_by_request[condition.request_index])
            _append_video_presentation(
                presentation, self.tokenizer, condition.ordinal, counts, timestamps,
            )
            for block_index in range(len(counts)):
                rendered_blocks.append((condition.request_index, block_index))

        media_prefix_length = len(presentation.ids)
        presentation.text(_text_ids(self.tokenizer, prompt))
        input_ids, text_token_tags = presentation.build()

        expected_blocks = [
            (block.request_index, block.block_index) for block in video_blocks
        ]
        assert rendered_conditions == condition_order
        assert rendered_vision == vision_media_order
        assert image_request_indices == [
            item.request_index for item in vision_media_order if item.kind == "image"
        ]
        assert video_request_indices == [
            item.request_index for item in vision_media_order if item.kind == "video"
        ]
        assert rendered_blocks == expected_blocks

        return Ref2VAPresentation(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            position_ids=None,
            text_token_tags=text_token_tags,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            pixel_values_videos=pixel_values_videos,
            video_grid_thw=video_grid_thw,
            condition_order=tuple(condition_order),
            vision_media_order=tuple(vision_media_order),
            video_blocks=tuple(video_blocks),
            media_prefix_length=media_prefix_length,
        )


class MiniMaxH3Ref2VAEncoder(nn.Module):
    """Qwen3-VL core ending at unnormalized language layer 49 output.

    Golden-source mapping is direct. Hugging Face ``Qwen3VLModel`` supplies
    SGLang ``Qwen3VLModel``'s visual tower, multimodal scatter, mRoPE indexing,
    and deepstack visual injection. Trimming ``num_hidden_layers`` to 50 maps
    to SGLang ``MINIMAX_H3_QWEN3VL_SELECTED_LM_LAYER``. Replacing
    ``language_model.norm`` with ``Identity`` returns hidden_states[50] before
    the final norm. The base model has no ``lm_head``.
    """

    def __init__(self, model: Qwen3VLModel) -> None:
        super().__init__()
        self.model = model
        layers = self.model.language_model.layers
        if len(layers) != MINIMAX_H3_QWEN3VL_SELECTED_LM_LAYER:
            raise ValueError(
                "MiniMax H3 Ref2VA encoder requires exactly 50 live language "
                f"layers, got {len(layers)}"
            )
        self.model.language_model.norm = nn.Identity()
        self.image_token_id = int(self.model.config.image_token_id)
        self.video_token_id = int(self.model.config.video_token_id)
        self.selected_lm_layer = MINIMAX_H3_QWEN3VL_SELECTED_LM_LAYER
        self.hidden_dim = MINIMAX_H3_QWEN3VL_HIDDEN_DIM

        forbidden_live_keys = [
            name
            for name in self.state_dict()
            if self.is_excluded_checkpoint_weight(name)
        ]
        if forbidden_live_keys:
            raise RuntimeError(
                "truncated Qwen3-VL core retained excluded checkpoint keys: "
                f"{forbidden_live_keys[:4]}"
            )

    @staticmethod
    def is_excluded_checkpoint_weight(name: str) -> bool:
        """Identify full-DCP keys intentionally absent from this core."""

        if name == "lm_head.weight" or name.startswith("model.language_model.norm."):
            return True
        match = _LAYER_WEIGHT_RE.match(name)
        return bool(
            match
            and int(match.group(1)) >= MINIMAX_H3_QWEN3VL_SELECTED_LM_LAYER
        )

    @classmethod
    def should_materialize_checkpoint_weight(cls, name: str) -> bool:
        return (
            "rotary_emb.inv_freq" not in name
            and not cls.is_excluded_checkpoint_weight(name)
        )

    @property
    def device(self) -> torch.device:
        return get_device()

    @torch.no_grad()
    def forward(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        pixel_values: torch.Tensor | None = None,
        image_grid_thw: torch.Tensor | None = None,
        pixel_values_videos: torch.Tensor | None = None,
        video_grid_thw: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Encode a presentation or right-padded multimodal batch to BF16 states.

        Visual pixels and grids follow sample-major order, matching the flattened
        placeholder traversal used by Qwen's scatter and mRoPE implementations.
        """

        if input_ids.ndim not in (1, 2):
            raise ValueError(
                f"input_ids must have shape [L] or [B,L], got {list(input_ids.shape)}"
            )
        is_batched = input_ids.ndim == 2
        if (pixel_values is None) != (image_grid_thw is None):
            raise ValueError("pixel_values and image_grid_thw must be given together")
        if (pixel_values_videos is None) != (video_grid_thw is None):
            raise ValueError(
                "pixel_values_videos and video_grid_thw must be given together"
            )

        host_ids = input_ids.to(device="cpu", dtype=torch.long)
        if not is_batched:
            host_ids = host_ids[None]
        if attention_mask is None:
            host_attention_mask = torch.ones_like(host_ids)
        else:
            if list(attention_mask.shape) != list(input_ids.shape):
                raise ValueError("attention_mask must have the same shape as input_ids")
            host_attention_mask = attention_mask.to(device="cpu", dtype=torch.long)
            if not is_batched:
                host_attention_mask = host_attention_mask[None]
        host_image_grid_thw = (
            image_grid_thw.to(device="cpu", dtype=torch.long)
            if image_grid_thw is not None
            else None
        )
        host_video_grid_thw = (
            video_grid_thw.to(device="cpu", dtype=torch.long)
            if video_grid_thw is not None
            else None
        )

        resolved_position_ids = position_ids
        if resolved_position_ids is None and (
            host_image_grid_thw is not None or host_video_grid_thw is not None
        ):
            resolved_position_ids, _ = self.model.get_rope_index(
                host_ids,
                host_image_grid_thw,
                host_video_grid_thw,
                attention_mask=host_attention_mask,
            )
        elif resolved_position_ids is not None:
            resolved_position_ids = resolved_position_ids.to(
                device="cpu", dtype=torch.long
            )
            if is_batched:
                expected_position_shape = [
                    3,
                    int(input_ids.shape[0]),
                    int(input_ids.shape[1]),
                ]
                if list(resolved_position_ids.shape) != expected_position_shape:
                    raise ValueError("batched position_ids must have shape [3,B,L]")
            else:
                sequence_length = int(input_ids.shape[0])
                if list(resolved_position_ids.shape) == [3, sequence_length]:
                    resolved_position_ids = resolved_position_ids[:, None, :]
                elif list(resolved_position_ids.shape) != [3, 1, sequence_length]:
                    raise ValueError(
                        "single-sample position_ids must have shape [3,L] or [3,1,L]"
                    )

        device = self.device
        call_kwargs: dict[str, Any] = {
            "input_ids": host_ids.to(device),
            "attention_mask": host_attention_mask.to(device),
            "use_cache": False,
            "output_attentions": False,
            "output_hidden_states": False,
            "return_dict": True,
        }
        if resolved_position_ids is not None:
            call_kwargs["position_ids"] = resolved_position_ids.to(device)
        if pixel_values is not None:
            call_kwargs["pixel_values"] = pixel_values.to(device, torch.bfloat16)
            call_kwargs["image_grid_thw"] = host_image_grid_thw.to(device)
        if pixel_values_videos is not None:
            call_kwargs["pixel_values_videos"] = pixel_values_videos.to(
                device, torch.bfloat16
            )
            call_kwargs["video_grid_thw"] = host_video_grid_thw.to(device)

        outputs = self.model(**call_kwargs)
        hidden = outputs.last_hidden_state
        if not is_batched:
            hidden = hidden[0]
        hidden = hidden.to(torch.bfloat16)
        expected_shape = [*list(input_ids.shape), self.hidden_dim]
        if list(hidden.shape) != expected_shape:
            raise ValueError(
                f"unexpected hidden shape {list(hidden.shape)}, "
                f"expected {expected_shape}"
            )
        return hidden

    @torch.no_grad()
    def encode(self, presentation: Ref2VAPresentation) -> torch.Tensor:
        """Encode a presentation object without repacking any conditioning."""

        return self(**presentation.encoder_inputs())


def text_only_ref2va_presentation(input_ids: torch.Tensor) -> Ref2VAPresentation:
    """Present already-tokenized caption rows without any reference context."""
    input_ids = input_ids.reshape(-1).to(device="cpu", dtype=torch.long)
    return Ref2VAPresentation(
        input_ids=input_ids,
        attention_mask=torch.ones_like(input_ids),
        position_ids=None,
        text_token_tags=torch.ones_like(input_ids),
        pixel_values=None,
        image_grid_thw=None,
        pixel_values_videos=None,
        video_grid_thw=None,
        condition_order=(),
        vision_media_order=(),
        video_blocks=(),
    )


def encode_ref2va_presentations(
    encoder: Any,
    presentations: Sequence[Ref2VAPresentation],
) -> list[torch.Tensor]:
    """Encode a ragged presentation batch with one top-level model forward.

    Right padding preserves each sample's causal hidden states. Pixels and grids
    are concatenated in sample order, while mRoPE is resolved jointly using the
    padded attention mask. Calling the model rather than its ``encode`` method
    preserves placement and runtime forward hooks. FSDP callers must invoke the
    same visual branches on every rank, such as a non-empty depth-video batch
    per rank. Different sample counts are supported, but mixing video-bearing
    and entirely text-only ranks does not align their visual-tower collectives.
    Empty captions use one masked-in stand-in token and return zero rows.
    """
    lengths = [int(item.input_ids.numel()) for item in presentations]
    longest = max(max(length, 1) for length in lengths)
    ids = torch.zeros((len(lengths), longest), dtype=torch.long)
    mask = torch.zeros_like(ids)
    for index, (item, length) in enumerate(zip(presentations, lengths, strict=True)):
        ids[index, :length] = item.input_ids.cpu()
        mask[index, :length] = item.attention_mask.cpu()
        if length == 0:
            mask[index, 0] = 1

    kwargs: dict[str, Any] = {"input_ids": ids, "attention_mask": mask}
    for key in (
        "pixel_values", "image_grid_thw", "pixel_values_videos", "video_grid_thw"
    ):
        values = [getattr(item, key) for item in presentations]
        parts = [value for value in values if value is not None]
        kwargs[key] = torch.cat(parts, dim=0) if parts else None
    positions = [item.position_ids for item in presentations]
    if any(value is not None for value in positions):
        assert all(value is not None for value in positions)
        padded = torch.ones((3, len(lengths), longest), dtype=torch.long)
        for index, (value, length) in enumerate(zip(positions, lengths, strict=True)):
            padded[:, index, :length] = value.reshape(3, length).cpu()
        kwargs["position_ids"] = padded

    hidden = encoder(**kwargs)
    return [hidden[index, :length].contiguous() for index, length in enumerate(lengths)]


def build_ref2va_encoder(
    pretrained_model_name_or_path: str | PathLike[str],
    **kwargs: Any,
) -> MiniMaxH3Ref2VAEncoder:
    """Construct the 50-layer Qwen3-VL core from local configuration only."""

    from transformers import AutoConfig, Qwen3VLModel

    kwargs.setdefault("local_files_only", True)
    config = AutoConfig.from_pretrained(pretrained_model_name_or_path, **kwargs)
    original_layers = int(config.text_config.num_hidden_layers)
    if original_layers != _MINIMAX_H3_QWEN3VL_CHECKPOINT_LAYERS:
        raise ValueError(
            "MiniMax H3 Ref2VA requires a full 64-layer Qwen3-VL config, "
            f"got {original_layers}"
        )
    config.text_config.num_hidden_layers = MINIMAX_H3_QWEN3VL_SELECTED_LM_LAYER
    config.text_config.use_cache = False
    model = Qwen3VLModel(config)
    return MiniMaxH3Ref2VAEncoder(model)


EntryClass = MiniMaxH3Ref2VAEncoder

__all__ = [
    "EntryClass",
    "IMAGE_PAD",
    "MINIMAX_H3_QWEN3VL_HIDDEN_DIM",
    "MINIMAX_H3_QWEN3VL_SELECTED_LM_LAYER",
    "MiniMaxH3Ref2VAEncoder",
    "MiniMaxH3Ref2VAPresentationProcessor",
    "Ref2VAConditionLabel",
    "Ref2VAMediaKind",
    "Ref2VAPresentation",
    "Ref2VAPresentationMedia",
    "Ref2VAVideoBlock",
    "Ref2VAVisionMedia",
    "TEXT_TOKEN_TAG",
    "VIDEO_PAD",
    "VIDEO_TOKEN_TAG",
    "VISION_END",
    "VISION_START",
    "build_ref2va_encoder",
    "encode_ref2va_presentations",
    "text_only_ref2va_presentation",
]
