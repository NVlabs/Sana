# SPDX-License-Identifier: Apache-2.0
"""Shared packed forwards for native reference prefixes and causal targets."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import torch

from dev.yanzuolu.common.distributed.ops import get_device
from dev.yanzuolu.projects.minimax_h3.data.causal_text_only import _stereo_chunk_indices
from dev.yanzuolu.projects.minimax_h3.meta_models.causal_minimax_h3_base import (
    _Chunk,
    _Layout,
)
from dev.yanzuolu.projects.minimax_h3.modeling.time_request import (
    H3_VIDEO_TEMPORAL_MAPPING,
    VideoTemporalMapping,
)
from dev.yanzuolu.projects.minimax_h3.modeling.transformer.x0_model import (
    MINIMAX_H3_AUDIO_CLEAN_TIMESTEP,
    MINIMAX_H3_VIDEO_CLEAN_TIMESTEP,
)
from dev.yanzuolu.projects.minimax_h3_videoref.data.video_ref_prefix_tf import (
    build_video_ref_prefix_tf_plan,
)
from dev.yanzuolu.projects.minimax_h3_videoref.modeling.prefix_tf import (
    build_video_ref_prefix_tf_layout,
)
from dev.yanzuolu.utils.flex_attn import _prepare_flex_attention_mask


@dataclass(frozen=True)
class PrefixTFInputs:
    """Encoded prefix conditions and target geometry shared by training and sampling."""

    prompt_embeds: list[torch.Tensor]
    native: list[dict[str, Any]]
    references: list[dict[str, torch.Tensor]]
    prefix_plans: list[dict[str, Any]]
    layouts: list[_Layout]
    packs: list[dict[str, Any]]
    seqlens: torch.Tensor
    token_tags: torch.Tensor

    @property
    def batch_size(self) -> int:
        return len(self.layouts)

    @property
    def text_lens(self) -> list[int]:
        return [int(value.shape[0]) for value in self.prompt_embeds]

    @property
    def sinks(self) -> list[int]:
        return [int(plan["sink"]) for plan in self.prefix_plans]

    @property
    def window_sizes(self) -> list[int | None]:
        return [plan["window_size"] for plan in self.prefix_plans]


class PrefixForwardMixin:
    """Share prefix packing between TF, DF and conditional score models.

    Hosts provide the causal H3 geometry, device, attention and padding helpers.
    Clean target rows require explicit context latents and endpoint noise.
    Their gradient policy is controlled by the caller, and reference rows are
    used as encoded.
    """

    _prefix_plan_factory = staticmethod(build_video_ref_prefix_tf_plan)

    def _prefix_inputs_from_payload(self, payload: dict[str, Any]) -> PrefixTFInputs:
        payload = self._to_device(payload)
        packs, layouts = [], []
        for native, video_shape, audio_shape, embedding, plan in zip(
            payload["native"], payload["latent_shapes"], payload["audio_shapes"],
            payload["prompt_embeds"], payload["prefix_plans"], strict=True,
        ):
            video_shape = tuple(int(value) for value in video_shape)
            audio_shape = tuple(int(value) for value in audio_shape)
            assert video_shape == tuple(plan["target_video_shape"])
            assert audio_shape == tuple(plan["target_audio_shape"])
            pack = self._to_device(build_video_ref_prefix_tf_layout(
                native=native,
                plan=plan,
            ))
            frame_rows = (video_shape[2] // 2) * (video_shape[3] // 2)
            chunks = tuple(
                _Chunk(
                    noise_start=int(chunk["noise_start"]),
                    clean_start=chunk["clean_start"],
                    has_clean_copy=chunk["clean_start"] is not None,
                    audio_rows=2 * (chunk["audio_stop"] - chunk["audio_start"]),
                    video_rows=frame_rows * (chunk["video_stop"] - chunk["video_start"]),
                    video_start=chunk["video_start"], video_stop=chunk["video_stop"],
                    audio_start=chunk["audio_start"], audio_stop=chunk["audio_stop"],
                )
                for chunk in pack["chunks"]
            )
            layouts.append(_Layout(
                text_len=int(embedding.shape[0]),
                chunks=chunks,
                audio_row_perm=torch.cat([
                    _stereo_chunk_indices(audio_shape[2], chunk.audio_start, chunk.audio_stop)
                    for chunk in chunks
                ]).to(get_device()),
                latent_shape=video_shape,
                audio_shape=audio_shape,
            ))
            packs.append(pack)
        return PrefixTFInputs(
            prompt_embeds=payload["prompt_embeds"],
            native=payload["native"],
            references=payload["references"],
            prefix_plans=payload["prefix_plans"],
            layouts=layouts,
            packs=packs,
            seqlens=torch.tensor([
                sum(chunk.video_rows + chunk.audio_rows for chunk in layout.chunks)
                for layout in layouts
            ], dtype=torch.int32, device=get_device()),
            token_tags=torch.cat([pack["token_tags"] for pack in packs]),
        )

    def _prefix_kwargs(
        self, model: Any, inputs: PrefixTFInputs, *,
        video_xts: list[torch.Tensor], audio_xts: list[torch.Tensor],
        video_timesteps: torch.Tensor, audio_timesteps: torch.Tensor,
        video_context: list[torch.Tensor] | None = None,
        audio_context: list[torch.Tensor] | None = None,
        video_eps: list[torch.Tensor] | None = None,
        audio_eps: list[torch.Tensor] | None = None,
    ) -> dict[str, Any]:
        """Materialize native rows at each sample's shared video/audio times.

        Text and Qwen rows follow the video time. Reference noise levels are
        capped by their encoded anchors, and clean target history uses the
        existing H3 clean endpoint.
        """
        device = inputs.token_tags.device
        video_timesteps = video_timesteps.to(device=device, dtype=torch.float32)
        audio_timesteps = audio_timesteps.to(device=device, dtype=torch.float32)
        video_dim = inputs.layouts[0].latent_shape[0] * 4
        audio_dim = inputs.layouts[0].audio_shape[1]
        video_blocks, audio_blocks, video_eps_blocks, audio_eps_blocks, row_timesteps = [], [], [], [], []
        positions, tags, img_pos, audio_pos, text_pos, noisy_img_pos = [], [], [], [], [], []
        q_ranges, k_ranges, attn_types, workloads, sample_lens, split_lens, modes = [], [], [], [], [], [], []
        offset = 0

        for index, (native, reference, layout, pack) in enumerate(zip(
            inputs.native, inputs.references, inputs.layouts, inputs.packs, strict=True
        )):
            native_len = int(native["seq_len"])
            source = pack["source_row_indices"].to(device=device, dtype=torch.long)
            source_img = native["img_pos"].to(device=device, dtype=torch.long)
            source_audio = native["audio_pos"].to(device=device, dtype=torch.long)
            video_rows = self._video_rows(video_xts[index], 0, layout.latent_shape[1])
            audio_rows = self._audio_rows(audio_xts[index], 0, layout.audio_shape[2])
            native_video = video_rows.new_zeros((native_len, video_dim)).index_copy(
                0, source_img, torch.cat((reference["visual_rows"].to(video_rows), video_rows))
            )
            native_audio = audio_rows.new_zeros((native_len, audio_dim)).index_copy(
                0, source_audio, torch.cat((reference["audio_rows"].to(audio_rows), audio_rows))
            )
            has_clean_rows = any(chunk["clean_start"] is not None for chunk in pack["chunks"])
            if has_clean_rows:
                assert video_context is not None and audio_context is not None
                assert video_eps is not None and audio_eps is not None
                target_img = source_img[native["update_mask"]]
                target_audio = source_audio[native["audio_update_mask"]]
                # Apply the clean-endpoint mapping to history once. Native reference
                # plans already contain augmentation, including the anchor=1 case.
                video_anchor = video_timesteps.new_tensor(MINIMAX_H3_VIDEO_CLEAN_TIMESTEP)
                audio_anchor = audio_timesteps.new_tensor(MINIMAX_H3_AUDIO_CLEAN_TIMESTEP)
                clean_video = video_anchor * video_context[index].float() + (1.0 - video_anchor) * video_eps[index].float()
                clean_audio = audio_anchor * audio_context[index].float() + (1.0 - audio_anchor) * audio_eps[index].float()
                native_clean_video = clean_video.new_zeros(native_video.shape).index_copy(
                    0, target_img, self._video_rows(clean_video, 0, layout.latent_shape[1])
                )
                native_clean_audio = clean_audio.new_zeros(native_audio.shape).index_copy(
                    0, target_audio, self._audio_rows(clean_audio, 0, layout.audio_shape[2])
                )
                clean_rows = pack["row_roles"].eq(1)
                video_block = torch.where(clean_rows[:, None], native_clean_video[source], native_video[source])
                audio_block = torch.where(clean_rows[:, None], native_clean_audio[source], native_audio[source])
            else:
                # Packed inputs retain the precision used alongside FP32 history.
                video_block = native_video[source].to(torch.promote_types(native_video.dtype, torch.float32))
                audio_block = native_audio[source].to(torch.promote_types(native_audio.dtype, torch.float32))
            video_blocks.append(video_block)
            audio_blocks.append(audio_block)
            video_eps_blocks.append(torch.zeros_like(video_block))
            audio_eps_blocks.append(torch.zeros_like(audio_block))

            native_t = video_timesteps[index].expand(native_len).clone()
            native_t[source_audio] = audio_timesteps[index]
            ref_video_count = reference["visual_rows"].shape[0]
            ref_audio_count = reference["audio_rows"].shape[0]
            native_t[source_img[:ref_video_count]] = torch.minimum(
                video_timesteps[index].expand(ref_video_count),
                1.0 - reference["visual_row_anchors"].to(native_t),
            )
            native_t[source_audio[:ref_audio_count]] = torch.minimum(
                audio_timesteps[index].expand(ref_audio_count),
                1.0 - reference["audio_row_anchors"].to(native_t),
            )
            packed_t = native_t[source]
            if has_clean_rows:
                clean_t = torch.where(
                    pack["token_tags"].eq(2),
                    1.0 - audio_anchor,
                    1.0 - video_anchor,
                )
                packed_t = torch.where(clean_rows, clean_t, packed_t)
            row_timesteps.append(packed_t)
            positions.append(pack["position_ids"])
            tags.append(pack["token_tags"])
            img_pos.append(pack["img_pos"] + offset)
            audio_pos.append(pack["audio_pos"] + offset)
            text_pos.append(pack["text_pos"] + offset)
            noisy_img_pos.append(pack["noisy_img_pos"] + offset)
            q_ranges.append(pack["q_ranges"] + offset)
            k_ranges.append(pack["k_ranges"] + offset)
            attn_types.append(pack["attn_type_map"])
            workloads.append(int(pack["attn_workloads"]))
            sample_lens.append(int(pack["sample_lens"]))
            split_lens.extend(pack["split_lens"])
            modes.extend(pack["attn_modes"])
            offset += int(pack["sample_lens"])

        position_ids, token_tags, pad = self._pad_for_sp(
            offset, video_blocks=video_blocks, audio_blocks=audio_blocks,
            video_eps_blocks=video_eps_blocks, audio_eps_blocks=audio_eps_blocks,
            row_timesteps=row_timesteps, position_ids=torch.cat(positions),
            token_tags=torch.cat(tags), sample_lens=sample_lens,
            video_dim=video_dim, audio_dim=audio_dim,
        )
        if pad:
            split_lens.append(pad)
            modes.append("causal")
            pad_q, pad_k, pad_types, pad_work = _prepare_flex_attention_mask(
                [pad], ["causal"], sink=inputs.sinks[0],
                window_size=inputs.window_sizes[0], device=device
            )
            q_ranges.append(pad_q + offset)
            k_ranges.append(pad_k + offset)
            attn_types.append(pad_types)
            workloads.append(int(pad_work))
        kwargs = self._common_kwargs(
            inputs, x=torch.cat(video_blocks).unsqueeze(0),
            audio_x=torch.cat(audio_blocks).unsqueeze(0),
            eps=torch.cat(video_eps_blocks).unsqueeze(0),
            audio_eps=torch.cat(audio_eps_blocks).unsqueeze(0),
            row_timesteps=torch.cat(row_timesteps), position_ids=position_ids,
            token_tags=token_tags, img_pos=torch.cat(img_pos),
            audio_pos=torch.cat(audio_pos), text_pos=torch.cat(text_pos),
            infer_out_pos=torch.cat(noisy_img_pos),
        )
        kwargs.update(
            clean_timesteps_by_tag=(1.0, 0.999, 1.0),
            sample_lens=sample_lens, q_ranges=torch.cat(q_ranges),
            k_ranges=torch.cat(k_ranges), attn_type_map=torch.cat(attn_types),
            attn_workloads=workloads,
            attention_mask=self._block_mask(
                inputs, device, sample_lens=sample_lens,
                split_lens=split_lens, attn_modes=modes,
            ),
        )
        return kwargs

    def _prefix_forward(
        self, model: Any, inputs: PrefixTFInputs, *,
        video_xts: list[torch.Tensor], audio_xts: list[torch.Tensor],
        video_timesteps: torch.Tensor, audio_timesteps: torch.Tensor,
        video_context: list[torch.Tensor] | None = None,
        audio_context: list[torch.Tensor] | None = None,
        video_eps: list[torch.Tensor] | None = None,
        audio_eps: list[torch.Tensor] | None = None,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """Return target-only x0 with native video and stereo audio geometry."""
        kwargs = self._prefix_kwargs(
            model, inputs, video_xts=video_xts, audio_xts=audio_xts,
            video_timesteps=video_timesteps, audio_timesteps=audio_timesteps,
            video_context=video_context, audio_context=audio_context,
            video_eps=video_eps, audio_eps=audio_eps,
        )
        video_logits, audio_logits = model(**kwargs)
        video_dim = inputs.layouts[0].latent_shape[0] * 4
        audio_dim = inputs.layouts[0].audio_shape[1]
        noisy_audio_sel = []
        audio_offset = 0
        for pack in inputs.packs:
            noisy_audio_sel.append(pack["audio_noisy_sel"] + audio_offset)
            audio_offset += int(pack["audio_pos"].numel())
        audio_logits = audio_logits.index_select(0, torch.cat(noisy_audio_sel))
        videos, audios = [], []
        video_cursor = audio_cursor = 0
        for layout, pack in zip(inputs.layouts, inputs.packs, strict=True):
            video_sel = pack["noisy_video_native_sel"]
            audio_sel = pack["noisy_audio_native_sel"]
            full_video_rows = sum(chunk.video_rows for chunk in layout.chunks)
            full_audio_rows = sum(chunk.audio_rows for chunk in layout.chunks)
            native_video = video_logits.new_zeros((full_video_rows, video_dim)).index_copy(
                0, video_sel, video_logits[video_cursor:video_cursor + video_sel.numel()]
            )
            native_audio = audio_logits.new_zeros((full_audio_rows, audio_dim)).index_copy(
                0, audio_sel, audio_logits[audio_cursor:audio_cursor + audio_sel.numel()]
            )
            videos.append(self._video_from_rows(native_video, layout))
            audios.append(
                self._audio_from_rows(native_audio, layout)
                if layout.audio_shape[2]
                else native_audio.reshape(layout.audio_shape)
            )
            video_cursor += video_sel.numel()
            audio_cursor += audio_sel.numel()
        return videos, audios

    @staticmethod
    def _prefix_gan_kwargs(inputs: PrefixTFInputs) -> dict[str, torch.Tensor]:
        """Select each noisy target AV block and its audio outputs for GAN heads."""
        ranges, sample_indices, audio_masks = [], [], []
        offset = 0
        for sample_index, (layout, pack) in enumerate(zip(
            inputs.layouts, inputs.packs, strict=True
        )):
            frame_rows = (layout.latent_shape[2] // 2) * (layout.latent_shape[3] // 2)
            for chunk in pack["chunks"]:
                if chunk["noise_start"] is None:
                    continue
                start = offset + int(chunk["noise_start"])
                audio_rows = 2 * (chunk["audio_stop"] - chunk["audio_start"])
                video_rows = frame_rows * (chunk["video_stop"] - chunk["video_start"])
                ranges.append((start, start + audio_rows + video_rows))
                sample_indices.append(sample_index)
            audio_masks.append(pack["row_roles"][pack["audio_pos"]].eq(2))
            offset += int(pack["sample_lens"])
        device = inputs.token_tags.device
        return {
            "gan_chunk_ranges": torch.tensor(ranges, dtype=torch.long, device=device).reshape(-1, 2),
            "gan_chunk_sample_indices": torch.tensor(sample_indices, dtype=torch.long, device=device),
            "update_audio_mask": torch.cat(audio_masks),
        }

    def _prefix_chunk_inputs(self, inputs: PrefixTFInputs, chunk_index: int) -> PrefixTFInputs:
        plans = [self._prefix_plan_factory(
            target_video_shape=plan["target_video_shape"],
            target_audio_shape=plan["target_audio_shape"],
            video_chunk_ranges=plan["video_chunk_ranges"],
            audio_chunk_ranges=plan["audio_chunk_ranges"],
            target_chunk_index=chunk_index,
            video_temporal_mapping=VideoTemporalMapping.from_dict(
                plan.get("video_temporal_mapping", H3_VIDEO_TEMPORAL_MAPPING.to_dict())
            ),
        ) for plan in inputs.prefix_plans]
        packs = [self._to_device(build_video_ref_prefix_tf_layout(native=native, plan=plan))
                 for native, plan in zip(inputs.native, plans, strict=True)]
        return replace(inputs, prefix_plans=plans, packs=packs,
                       token_tags=torch.cat([pack["token_tags"] for pack in packs]))


__all__ = ["PrefixTFInputs", "PrefixForwardMixin"]
