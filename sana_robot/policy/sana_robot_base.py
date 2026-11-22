# Copyright 2026 NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math

import torch
from torch.utils.checkpoint import checkpoint as torch_checkpoint

from sana_robot.policy.attention.condition import (
    ConditionChunkLayout,
    ConditionGatedDeltaNet,
    ConditionPass,
    ConditionSoftmaxAttention,
    condition_chunk_layout,
)
from sana_robot.policy.attention.cross_attention import MultiHeadCrossAttention, PackedGroupedText
from sana_robot.policy.backbone.block import PolicyBlock
from sana_robot.policy.backbone.dit import SanaRobotBackbone
from sana_robot.policy.layout.multiview import (
    batch_view_count,
    batch_view_shapes,
    strip_to_view_tokens,
    view_to_strip_tokens,
)


def text_group_spans(view_token_counts: list, robot_rows: int) -> tuple:
    """Return the (offset, length) token span that reads each text group: one span per view, then one for the robot rows."""
    spans = []
    offset = 0
    for count in view_token_counts:
        spans.append((offset, int(count)))
        offset += int(count)
    spans.append((offset, robot_rows))
    return tuple(spans)


class SanaRobotBase(SanaRobotBackbone):
    """SANA-Robot-base, the policy without memory.

    From the current observation (the first latent frame of every view), the robot state and the text it denoises
    the future latent frames and one action chunk. Tokens are ordered
    [video of view 1; ...; video of view V; state; actions]; the observation and the state form the condition
    chunk, which never reads the noisy prediction.
    """

    def __init__(self, **backbone_arguments) -> None:
        super().__init__(**backbone_arguments)
        self.grad_checkpointing = False
        self.packed_text_cache = None
        self._layout_cache = {}

    def build_block(self, softmax: bool) -> PolicyBlock:
        """Build one layer: softmax attention or a Gated DeltaNet over the condition and prediction chunks, with text cross-attention."""
        if softmax:
            self_attention = ConditionSoftmaxAttention(self.hidden_size, self.hidden_size // self.softmax_head_dim)
        else:
            self_attention = ConditionGatedDeltaNet(self.hidden_size, self.hidden_size // self.linear_head_dim)
        cross_attention = MultiHeadCrossAttention(self.hidden_size, self.cross_attention_heads)
        return PolicyBlock(self.hidden_size, self_attention, cross_attention, self.mlp_ratio)

    def forward(self, video_latents, timestep, text_embeddings, text_mask=None, data_info=None, condition_prefix=None):
        """Predict the video flow and the action flow of one denoising step.

        Args:
            video_latents: Video latents. One view: [B, C, F, H, W]. Several views: the strip
                [B, C, F, 1, sum(H_v * W_v)] that concatenates the flattened views of every latent frame. Frame 0
                is the clean observation.
            timestep: Video timestep per batch, per latent frame [B, F] or per video token; 0 on the observation.
            text_embeddings: Text embeddings [B, V + 1, 1, L, C_text]: one group per view, then one for the robot
                rows.
            text_mask: Text token mask [B, V + 1, L], or None.
            data_info: Dict with ``view_count``, ``view_latent_shape`` and ``view_slot_ids`` (several views),
                ``model_fps``, ``initial_state``, ``initial_state_condition_mask``, ``action`` (the noisy actions
                [B, A, 80]), ``action_mask`` and ``action_timestep``; optionally ``video_frame_stride``.
            condition_prefix: Prefix caching for inference, ``(ConditionPrefix, phase)``: ``PREFILL`` runs the
                condition chunk alone and fills the cache, ``TARGET`` runs the prediction chunk alone against it.
                The output of the tokens a pass skips is zero.

        Returns:
            Dict with ``video_pred``, the video flow in the layout of ``video_latents``, and ``action_pred``
            [B, A, 80].
        """
        self.set_latent_grid(video_latents)
        if batch_view_count(data_info) == 1:
            view_shapes = ((self.h, self.w),)
        else:
            view_shapes = batch_view_shapes(data_info)
        batch = video_latents.shape[0]
        video_latents = video_latents.to(self.dtype)
        text_embeddings = text_embeddings.to(self.dtype)
        timestep = timestep.long().float()
        video = strip_to_view_tokens(self.x_embedder(video_latents), self.f, view_shapes)
        view_token_counts = [self.f * height * width for (height, width) in view_shapes]
        video_tokens = video.shape[1]
        (robot, action_rows) = self.embed_state_and_actions(data_info, video)
        hidden = torch.cat((video, robot), dim=1)

        video_timestep = strip_to_view_tokens(self.video_token_timesteps(timestep, video_tokens), self.f, view_shapes)
        action_timestep = self.action_row_timesteps(data_info, batch, action_rows, video_latents.device)
        robot_timestep = torch.cat((video_timestep.new_zeros(batch, 1), action_timestep), dim=1)
        token_timestep = torch.cat((video_timestep, robot_timestep), dim=1).unsqueeze(1)
        time_embedding = self.embed_timesteps(token_timestep)
        modulation = self.t_block(time_embedding)

        rope_linear = self._rope(self.rope_linear, data_info, batch, view_shapes, action_rows, video_latents.device)
        rope_softmax = self._rope(self.rope_softmax, data_info, batch, view_shapes, action_rows, video_latents.device)

        layout = self._condition_chunk_layout(view_shapes, action_rows, video_latents.device)
        text_source = (text_embeddings, text_mask)
        (text, text_mask) = self.embed_text(text_embeddings, text_mask)
        spans = text_group_spans(view_token_counts, 1 + action_rows)
        if condition_prefix is not None:
            layout = ConditionPass(layout, condition_prefix[1], condition_prefix[0])
            (text, text_mask, spans) = self._text_of_pass(layout, text, text_mask, spans)
            hidden = hidden.index_select(1, layout.keep)
            modulation = modulation.index_select(2, layout.keep)
            rope_linear = rope_linear.index_select(2, layout.keep)
            rope_softmax = rope_softmax.index_select(2, layout.keep)
        if self.packed_text_cache is not None and batch == 1 and not self.training:
            (text, text_mask, spans) = self._pack_text(text, text_mask, spans, text_source)
        text_arguments = {"text_mask": text_mask, "prompt_group_spans": spans}
        run_layers = self._run_layers if self.training else self._run_layers_inference
        hidden = run_layers(hidden, text, modulation, text_arguments, rope_linear, rope_softmax, layout)
        if condition_prefix is not None:
            full = hidden.new_zeros(batch, layout.full_tokens, hidden.shape[-1])
            hidden = full.index_copy(1, layout.keep, hidden)

        video_hidden = self.final_layer(hidden[:, :video_tokens], time_embedding[:, :, :video_tokens])
        video_flow = self.unpatchify(view_to_strip_tokens(video_hidden, self.f, view_shapes))
        action_flow = self.predict_actions(
            hidden[:, video_tokens + 1 :], time_embedding[:, :, video_tokens + 1 :], data_info
        )
        return {"video_pred": video_flow, "action_pred": action_flow}

    def _rope(self, rope, data_info: dict, batch: int, view_shapes: tuple, action_rows: int, device):
        """Return the rotary phases of every token: the video on its view's tile at its frame's time, the state at time 0 and action row i at the time of control step i."""
        fps = self.batch_fps(data_info["model_fps"], batch, device)
        video_fps = fps / self.video_frame_stride(data_info)
        video_ids = self.video_position_ids(rope, data_info, view_shapes, video_fps, self.f, device)
        action_steps = torch.arange(1, action_rows + 1, device=device, dtype=torch.float64)
        action_time = rope.base_fps * action_steps[None] / (self.vae_temporal_stride * fps[:, None])
        robot_time = torch.cat((torch.zeros(batch, 1, device=device), action_time), dim=1)
        robot_ids = torch.stack((robot_time, torch.zeros_like(robot_time), torch.zeros_like(robot_time)), dim=-1)
        return torch.cat((rope.from_position_ids(video_ids), rope.from_position_ids(robot_ids)), dim=2)

    def _condition_chunk_layout(self, view_shapes: tuple, action_rows: int, device) -> ConditionChunkLayout:
        """Return the cached condition / prediction layout of this token grid."""
        key = (self.f, view_shapes, action_rows, str(device))
        layout = self._layout_cache.get(key)
        if layout is None:
            with torch.inference_mode(False):
                frame_ids = torch.cat(
                    [torch.arange(self.f).repeat_interleave(height * width) for (height, width) in view_shapes]
                )
                layout = condition_chunk_layout(frame_ids, action_rows, device=device)
            self._layout_cache[key] = layout
        return layout

    @staticmethod
    def _text_of_pass(layout: ConditionPass, text, text_mask, spans):
        """Cut the text routing to the tokens of a prefix-caching pass, dropping the groups none of its tokens read."""
        cut = [layout.kept(offset, length) for (offset, length) in spans]
        groups = [group for (group, (_offset, length)) in enumerate(cut) if length]
        if len(groups) < len(cut):
            text = torch.cat([text[:, group : group + 1] for group in groups], dim=1)
            if text_mask is not None:
                text_mask = torch.cat([text_mask[:, group : group + 1] for group in groups], dim=1)
        return (text, text_mask, tuple((cut[group] for group in groups)))

    def _pack_text(self, text, text_mask, spans, source):
        """Pack the grouped text of one sample for a single variable-length cross-attention call: the real tokens of every group end to end [1, T, hidden], no mask, and the ``PackedGroupedText`` of the token spans. The result is remembered per caption tensor and token spans."""
        (source_text, source_mask) = source
        cache = self.packed_text_cache
        cache_key = (
            source_text.data_ptr(),
            tuple(source_text.shape),
            0 if source_mask is None else source_mask.data_ptr(),
            () if source_mask is None else tuple(source_mask.shape),
            str(text.device),
            spans,
        )
        entry = cache.get(cache_key)
        if entry is not None:
            return entry["packed"]
        if text_mask is None:
            group_text = list(text[0].unbind(0))
        else:
            group_text = [tokens[mask.to(torch.bool)] for (tokens, mask) in zip(text[0], text_mask[0])]

        def offsets(lengths):
            return torch.tensor([sum(lengths[:index]) for index in range(len(lengths) + 1)], dtype=torch.int32)

        query_lengths = [length for (_offset, length) in spans]
        text_lengths = [int(tokens.shape[0]) for tokens in group_text]
        packed = PackedGroupedText(
            spans=spans,
            kv_lens=tuple(text_lengths),
            cu_seqlens_q=offsets(query_lengths).to(text.device),
            cu_seqlens_k=offsets(text_lengths).to(text.device),
            max_seqlen_q=max(query_lengths),
            max_seqlen_k=max(text_lengths),
        )
        if len(cache) >= 16:
            cache.clear()
        cache[cache_key] = {"packed": (torch.cat(group_text, dim=0).unsqueeze(0), None, packed), "source": source}
        return cache[cache_key]["packed"]

    def _layer_step(self, layer, sources, partial_block, text, modulation, text_arguments, rotary_emb, layout):
        """Run both sublayers of one layer on their AttnRes inputs and return the two residual updates."""
        block = self.blocks[layer]
        attention_input = self.attn_res.attend(self.attn_res.attn_proj, list(sources), partial_block)
        attention_delta = block(
            attention_input,
            text,
            modulation,
            text_arguments,
            sublayer="attention",
            rotary_emb=rotary_emb,
            layout=layout,
        )
        partial_block = attention_delta if partial_block is None else partial_block + attention_delta
        mlp_input = self.attn_res.attend(self.attn_res.mlp_proj, list(sources), partial_block)
        return (attention_delta, block(mlp_input, modulation, sublayer="mlp"))

    def _run_layers(self, hidden, text, modulation, text_arguments, rope_linear, rope_softmax, layout):
        """Run the layer stack with autograd; every finished group of ``attn_res_block_size`` layers becomes an AttnRes source."""
        depth = len(self.blocks)
        sources = [hidden]
        for first_layer in range(0, depth, self.attn_res_block_size):
            partial_block = None
            for layer in range(first_layer, min(first_layer + self.attn_res_block_size, depth)):
                rotary_emb = rope_softmax if layer in self.softmax_layer_indices else rope_linear
                step = (layer, tuple(sources), partial_block, text, modulation, text_arguments, rotary_emb, layout)
                if self.grad_checkpointing:
                    (attention_delta, mlp_delta) = torch_checkpoint(self._layer_step, *step, use_reentrant=False)
                else:
                    (attention_delta, mlp_delta) = self._layer_step(*step)
                partial_block = attention_delta if partial_block is None else partial_block + attention_delta
                partial_block = partial_block + mlp_delta
            sources.append(partial_block)
        return self.attn_res.attend(self.attn_res.final_proj, sources)

    @torch.no_grad()
    def _run_layers_inference(self, hidden, text, modulation, text_arguments, rope_linear, rope_softmax, layout):
        """Run the layer stack without autograd, keeping the AttnRes sources in preallocated buffers."""
        depth = len(self.blocks)
        attn_res = self.attn_res
        value_buffer = hidden.new_empty(math.ceil(depth / self.attn_res_block_size) + 2, *hidden.shape)
        key_buffer = torch.empty_like(value_buffer)
        value_buffer[0] = hidden
        key_buffer[0] = attn_res.key_norm(hidden.unsqueeze(0)).squeeze(0)
        source_count = 1
        for first_layer in range(0, depth, self.attn_res_block_size):
            partial_block = None
            for layer in range(first_layer, min(first_layer + self.attn_res_block_size, depth)):
                block = self.blocks[layer]
                rotary_emb = rope_softmax if layer in self.softmax_layer_indices else rope_linear
                attention_input = attn_res.attend_buffer(
                    attn_res.attn_proj, value_buffer, key_buffer, source_count, partial_block
                )
                attention_delta = block(
                    attention_input,
                    text,
                    modulation,
                    text_arguments,
                    sublayer="attention",
                    rotary_emb=rotary_emb,
                    layout=layout,
                )
                partial_block = attention_delta if partial_block is None else partial_block + attention_delta
                mlp_input = attn_res.attend_buffer(
                    attn_res.mlp_proj, value_buffer, key_buffer, source_count, partial_block
                )
                partial_block = partial_block + block(mlp_input, modulation, sublayer="mlp")
            value_buffer[source_count] = partial_block
            key_buffer[source_count] = attn_res.key_norm(partial_block.unsqueeze(0)).squeeze(0)
            source_count += 1
        return attn_res.attend_buffer(attn_res.final_proj, value_buffer, key_buffer, source_count, None)
