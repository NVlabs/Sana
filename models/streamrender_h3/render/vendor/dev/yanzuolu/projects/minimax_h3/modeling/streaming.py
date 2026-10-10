# SPDX-License-Identifier: Apache-2.0
"""Dense AV streaming windows with optional aligned video references."""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Any

import torch

from dev.yanzuolu.projects.minimax_h3.data.streaming import _streaming_window_time_origins
from dev.yanzuolu.projects.minimax_h3.modeling.packing import _axis_from_sqrt_area
from dev.yanzuolu.projects.minimax_h3.modeling.packed_tokens import (
    minimax_h3_unpatchify_video_tokens,
)
from dev.yanzuolu.projects.minimax_h3.modeling.time_request import VideoTemporalMapping


def _spatial_grid(height: int, width: int) -> torch.Tensor:
    area = sqrt(height * width)
    hh, ww = torch.meshgrid(
        _axis_from_sqrt_area(height, 2, area),
        _axis_from_sqrt_area(width, 2, area),
        indexing="ij",
    )
    return torch.stack((hh.flatten(), ww.flatten()), dim=-1)


def _streaming_layout(
    plan: dict[str, Any], text_len: int, *, separate_reference_rope: bool = True,
    fixed_window_rope: bool = False,
    text_token_tags: torch.Tensor | None = None,
) -> dict[str, Any]:
    """Place AV clocks globally or with fixed sink and recent-window origins.

    An optional ``picture_shape`` places one image latent between the prefix and
    the reference rows. Like a native image reference it occupies one time slot at
    ``text_len`` on its own spatial grid, and every later origin advances by one.
    With ``picture_keyframe`` it is instead a native first-frame keyframe on the
    target grid. It shares target latent 0's time, which fixed-window RoPE keeps
    in the unshifted sink domain, and no origin advances.
    """
    mapping = VideoTemporalMapping.from_dict(plan["video_temporal_mapping"])
    video_indices = plan["video_indices"].cpu()
    audio_indices = plan["audio_indices"].cpu()
    video_noisy = plan["video_noisy_mask"].cpu()
    audio_noisy = plan["audio_noisy_mask"].cpu()
    reference_shape = plan["reference_video_shape"]
    if reference_shape is not None and fixed_window_rope and not separate_reference_rope:
        raise ValueError("fixed_window_rope requires separate_reference_rope with reference video")
    reference_indices = (
        None if reference_shape is None else plan["reference_video_indices"].cpu()
    )
    picture_shape = plan.get("picture_shape")
    picture_grid = None if picture_shape is None else _spatial_grid(*picture_shape[2:])
    picture_keyframe = picture_grid is not None and bool(plan.get("picture_keyframe", False))
    if picture_keyframe and tuple(picture_shape[2:]) != tuple(plan["target_video_shape"][2:]):
        raise ValueError("a keyframe picture must share the target latent grid")
    picture_rows = 0 if picture_grid is None else picture_grid.shape[0]
    reference_grid = None if reference_shape is None else _spatial_grid(*reference_shape[2:])
    target_grid = _spatial_grid(*plan["target_video_shape"][2:])
    reference_start = text_len + picture_rows
    reference_rows = 0 if reference_grid is None else reference_indices.numel() * reference_grid.shape[0]
    video_rows = video_indices.numel() * target_grid.shape[0]
    audio_rows = 2 * audio_indices.numel()
    audio_start = reference_start + reference_rows
    video_start = audio_start + audio_rows
    sample_len = video_start + video_rows
    positions = torch.zeros((sample_len, 3), dtype=torch.float64)
    positions[:text_len, 0] = torch.arange(text_len, dtype=torch.float64)
    video_times = torch.tensor(
        [mapping.position_start(int(index)) for index in video_indices],
        dtype=torch.float64,
    )
    reference_times = (
        None if reference_indices is None else torch.tensor(
            [mapping.position_start(int(index)) for index in reference_indices],
            dtype=torch.float64,
        )
    )
    audio_times = audio_indices.to(torch.float64)
    reference_origin = float(text_len) + (1.0 if picture_grid is not None and not picture_keyframe else 0.0)
    target_origin = reference_origin
    if fixed_window_rope and not plan["is_bootstrap"]:
        sink_history_stop = int(plan["sink_history_stop"])
        sink_end, recent_origin = _streaming_window_time_origins(
            plan, video_temporal_mapping=mapping, fixed_window_rope=True,
        )
        video_times = torch.where(
            video_indices < sink_history_stop, video_times, sink_end + video_times - recent_origin,
        )
        audio_times = torch.where(
            audio_indices < mapping.decode_timeline.clock_boundary_ceil(sink_history_stop),
            audio_times, sink_end + audio_times - recent_origin,
        )
        if reference_grid is not None:
            reference_times = torch.where(
                reference_indices < sink_history_stop,
                reference_times, sink_end + reference_times - recent_origin,
            )
            target_origin += mapping.position_start(int(plan["rope_window_stop"]))
    elif reference_grid is not None and separate_reference_rope:
        target_origin += mapping.position_start(int(plan["stop"]))
    if picture_grid is not None:
        positions[text_len:reference_start, 0] = (
            target_origin + mapping.position_start(0) if picture_keyframe else float(text_len)
        )
        positions[text_len:reference_start, 1:] = picture_grid
    if reference_grid is not None:
        reference_positions = positions[reference_start:audio_start].view(
            reference_indices.numel(), reference_grid.shape[0], 3
        )
        reference_positions[:, :, 0] = (reference_origin + reference_times)[:, None]
        reference_positions[:, :, 1:] = reference_grid[None]
    target_positions = positions[video_start:].view(
        video_indices.numel(), target_grid.shape[0], 3
    )
    target_positions[:, :, 0] = (target_origin + video_times)[:, None]
    target_positions[:, :, 1:] = target_grid[None]
    audio_positions = positions[audio_start:video_start].view(2, -1, 3)
    audio_positions[:, :, 0] = target_origin + audio_times
    audio_positions[0, :, 2] = target_grid[:, 1].min()
    audio_positions[1, :, 2] = target_grid[:, 1].max()
    tags = torch.zeros(sample_len, dtype=torch.long)
    tags[:text_len] = 1
    if text_token_tags is not None:
        if text_token_tags.shape != (text_len,):
            raise ValueError("text_token_tags must match the complete conditioning prefix")
        tags[:text_len] = text_token_tags.to(device="cpu", dtype=torch.long)
    tags[audio_start:video_start] = 2
    target_img = torch.arange(video_start, sample_len)
    return {
        "sample_lens": sample_len,
        "position_ids": positions,
        "token_tags": tags,
        "text_pos": torch.arange(text_len),
        "img_pos": torch.cat((torch.arange(text_len, audio_start), target_img)),
        "audio_pos": torch.arange(audio_start, video_start),
        "noisy_img_pos": target_img[video_noisy.repeat_interleave(target_grid.shape[0])],
        "audio_noisy_sel": audio_noisy.repeat(2).nonzero().flatten(),
        "video_noisy_mask": video_noisy,
        "audio_noisy_mask": audio_noisy,
        "reference_rows": picture_rows + reference_rows,
        "video_rows": video_rows,
        "audio_rows": audio_rows,
        "noisy_video_frames": int(video_noisy.sum()),
        "noisy_audio_frames": int(audio_noisy.sum()),
    }


@dataclass(frozen=True)
class StreamingInputs:
    """One dense document per sample with optional prepared reference latents."""

    plans: list[dict[str, Any]]
    prompt_embeds: list[torch.Tensor]
    reference_latents: list[torch.Tensor] | None
    packs: list[dict[str, Any]]
    token_tags: torch.Tensor
    seqlens: torch.Tensor
    text_token_tags: list[torch.Tensor] | None = None

    @property
    def batch_size(self) -> int:
        return len(self.plans)

    @property
    def text_lens(self) -> list[int]:
        return [int(embedding.shape[0]) for embedding in self.prompt_embeds]


class StreamingForwardMixin:
    """Materialize bidirectional windows without persistent DiT state.

    Hosts supply the shared H3 row, padding and timestep helpers. Callers prepare
    optional near-clean reference and video history, plus unchanged retained audio.
    A plan may carry a near-clean ``picture_latents`` image, packed with the reference.
    Every call recomputes the full bidirectional window, assigns conditioned
    AV rows noise level 0.001 and preserves all supplied target gradients.
    A plan's ``history_timestep`` instead sets the noise level of its target
    AV history rows, those outside the noisy masks, whose supplied values must
    then be noised to that level. Reference and picture rows keep 0.001.
    """

    separate_reference_rope: bool = True
    fixed_window_rope: bool = False

    def _streaming_inputs_from_payload(self, payload: dict[str, Any]) -> StreamingInputs:
        payload = self._to_device(payload)
        prefix_tags = payload.get("text_token_tags")
        packs = [
            self._to_device(_streaming_layout(
                plan, int(embedding.shape[0]),
                separate_reference_rope=self.separate_reference_rope,
                fixed_window_rope=self.fixed_window_rope,
                text_token_tags=None if prefix_tags is None else prefix_tags[index],
            ))
            for index, (plan, embedding) in enumerate(zip(
                payload["plans"], payload["prompt_embeds"],
                strict=True,
            ))
        ]
        tags = torch.cat([pack["token_tags"] for pack in packs])
        return StreamingInputs(
            plans=payload["plans"], prompt_embeds=payload["prompt_embeds"],
            reference_latents=payload["reference_latents"], packs=packs,
            token_tags=tags,
            seqlens=torch.tensor(
                [pack["sample_lens"] for pack in packs], dtype=torch.int32, device=tags.device
            ),
            text_token_tags=prefix_tags,
        )

    def _streaming_kwargs(
        self, model: Any, inputs: StreamingInputs, *,
        video_xts: list[torch.Tensor], audio_xts: list[torch.Tensor],
        video_timesteps: torch.Tensor, audio_timesteps: torch.Tensor,
    ) -> dict[str, Any]:
        """Build native dense H3 kwargs from selected target and reference AV rows."""
        device = inputs.token_tags.device
        video_timesteps = video_timesteps.to(device=device, dtype=torch.float32)
        audio_timesteps = audio_timesteps.to(device=device, dtype=torch.float32)
        video_blocks, audio_blocks, video_eps, audio_eps, times = [], [], [], [], []
        conditioning_masks = []
        positions, tags, text_pos, img_pos, audio_pos, infer_pos = [], [], [], [], [], []
        sample_lens = []
        offset = 0
        if inputs.reference_latents is not None:
            assert len(inputs.reference_latents) == inputs.batch_size
        for index, (plan, pack, video, audio) in enumerate(zip(
            inputs.plans, inputs.packs, video_xts, audio_xts, strict=True,
        )):
            reference = None if inputs.reference_latents is None else inputs.reference_latents[index]
            video_count = plan["video_indices"].numel()
            audio_count = plan["audio_indices"].numel()
            channels, _, height, width = plan["target_video_shape"]
            audio_channels = plan["target_audio_shape"][1]
            assert tuple(video.shape) == (channels, video_count, height, width)
            assert tuple(audio.shape) == (2, audio_channels, audio_count)
            video_rows = self._video_rows(video, 0, video_count)
            if plan["reference_video_shape"] is None:
                assert reference is None
                reference_rows = video_rows.new_empty((0, video_rows.shape[1]))
            else:
                ref_channels, _, ref_height, ref_width = plan["reference_video_shape"]
                reference_count = plan["reference_video_indices"].numel()
                assert reference is not None
                assert tuple(reference.shape) == (ref_channels, reference_count, ref_height, ref_width)
                reference_rows = self._video_rows(reference, 0, reference_count)
            if plan.get("picture_shape") is not None:
                picture = plan["picture_latents"]
                assert tuple(picture.shape) == tuple(plan["picture_shape"])
                reference_rows = torch.cat((self._video_rows(picture.to(reference_rows), 0, 1), reference_rows))
            sound_rows = self._audio_rows(audio, 0, audio_count)
            text_len = inputs.text_lens[index]
            video_block = torch.cat((
                video_rows.new_zeros((text_len, video_rows.shape[1])), reference_rows,
                video_rows.new_zeros((pack["audio_rows"], video_rows.shape[1])), video_rows,
            ))
            audio_block = torch.cat((
                sound_rows.new_zeros((text_len + pack["reference_rows"], audio_channels)),
                sound_rows, sound_rows.new_zeros((pack["video_rows"], audio_channels)),
            ))
            clean_t = video_timesteps.new_tensor(0.001)
            history_t = video_timesteps.new_tensor(float(plan.get("history_timestep", 0.001)))
            video_t = torch.where(pack["video_noisy_mask"], video_timesteps[index], history_t)
            audio_t = torch.where(pack["audio_noisy_mask"], audio_timesteps[index], history_t)
            frame_rows = (height // 2) * (width // 2)
            times.append(torch.cat((
                video_timesteps[index].expand(text_len), clean_t.expand(pack["reference_rows"]),
                audio_t.repeat(2), video_t.repeat_interleave(frame_rows),
            )))
            conditioning_masks.append(torch.cat((
                torch.ones(text_len, dtype=torch.bool, device=device),
                torch.zeros(pack["reference_rows"], dtype=torch.bool, device=device),
                pack["audio_noisy_mask"].repeat(2),
                pack["video_noisy_mask"].repeat_interleave(frame_rows),
            )))
            video_blocks.append(video_block)
            audio_blocks.append(audio_block)
            video_eps.append(torch.zeros_like(video_block))
            audio_eps.append(torch.zeros_like(audio_block))
            positions.append(pack["position_ids"])
            tags.append(pack["token_tags"])
            text_pos.append(pack["text_pos"] + offset)
            img_pos.append(pack["img_pos"] + offset)
            audio_pos.append(pack["audio_pos"] + offset)
            infer_pos.append(pack["noisy_img_pos"] + offset)
            sample_lens.append(int(pack["sample_lens"]))
            offset += int(pack["sample_lens"])
        position_ids, token_tags, pad = self._pad_for_sp(
            offset, video_blocks=video_blocks, audio_blocks=audio_blocks,
            video_eps_blocks=video_eps, audio_eps_blocks=audio_eps,
            row_timesteps=times, position_ids=torch.cat(positions), token_tags=torch.cat(tags),
            sample_lens=sample_lens, video_dim=video_blocks[0].shape[1],
            audio_dim=audio_blocks[0].shape[1],
        )
        if pad:
            conditioning_masks.append(torch.zeros(pad, dtype=torch.bool, device=device))
        kwargs = self._common_kwargs(
            inputs, x=torch.cat(video_blocks).unsqueeze(0),
            audio_x=torch.cat(audio_blocks).unsqueeze(0),
            eps=torch.cat(video_eps).unsqueeze(0), audio_eps=torch.cat(audio_eps).unsqueeze(0),
            row_timesteps=torch.cat(times), position_ids=position_ids, token_tags=token_tags,
            img_pos=torch.cat(img_pos), audio_pos=torch.cat(audio_pos),
            text_pos=torch.cat(text_pos), infer_out_pos=torch.cat(infer_pos),
        )
        cumulative = [0]
        for length in sample_lens:
            cumulative.append(cumulative[-1] + length)
        kwargs.update(
            timestep_conditioning_mask=torch.cat(conditioning_masks),
            clean_timesteps_by_tag=(1.0, 0.999, 1.0),
            packed_seq_params={
                "cu_seqlens_q": torch.tensor(cumulative, dtype=torch.int32, device=device),
                "cu_seqlens_q_host": cumulative,
                "max_seqlen_q": max(sample_lens),
            },
        )
        return kwargs

    def _streaming_forward(
        self, model: Any, inputs: StreamingInputs, *,
        video_xts: list[torch.Tensor], audio_xts: list[torch.Tensor],
        video_timesteps: torch.Tensor, audio_timesteps: torch.Tensor,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """Return noisy target video and audio, including predicted audio lookahead."""
        video_logits, audio_logits = model(**self._streaming_kwargs(
            model, inputs, video_xts=video_xts, audio_xts=audio_xts,
            video_timesteps=video_timesteps, audio_timesteps=audio_timesteps,
        ))
        videos, audios = [], []
        video_cursor = audio_cursor = 0
        for plan, pack in zip(inputs.plans, inputs.packs, strict=True):
            channels, _, height, width = plan["target_video_shape"]
            video_frames = pack["noisy_video_frames"]
            video_rows = video_frames * (height // 2) * (width // 2)
            videos.append(
                minimax_h3_unpatchify_video_tokens(
                    video_logits[video_cursor:video_cursor + video_rows],
                    latent_shape=(video_frames, height // 2, width // 2, channels),
                    patch_size=(1, 2, 2),
                )[0]
                if video_frames else video_logits.new_empty((channels, 0, height, width))
            )
            sound = audio_logits[audio_cursor:audio_cursor + pack["audio_rows"]]
            sound = sound.index_select(0, pack["audio_noisy_sel"])
            audios.append(sound.reshape(
                2, pack["noisy_audio_frames"], plan["target_audio_shape"][1]
            ).permute(0, 2, 1))
            video_cursor += video_rows
            audio_cursor += pack["audio_rows"]
        return videos, audios


__all__ = ["StreamingInputs", "StreamingForwardMixin"]
