"""MiniMax-H3 block stack over N GPUs: Ulysses on the packed sequence, weights gathered per layer.

The kernels are the GB200 `fast_forward` v2 bf16 line (fused norm/modulate, residual-gate-norm, SwiGLU,
q/k-norm + RoPE).  What changes for N ranks:

  * after the embeddings the packed sequence is split into contiguous row shards (`tensor_split` sizes,
    so 38247 rows over 4 ranks are [9562, 9562, 9562, 9561]); RoPE tables and AdaLN row indices follow;
  * attention trades rows for heads with the packed Ulysses all-to-all (`ulysses_custom`), runs the
    attention core (Sol-Attn or dense) on the whole sequence for this rank's heads, and trades back;
  * `block_pre(i)` runs before block i, which is where `shard.LayerShards` rebuilds the block's weights;
  * after the stack the row shards are gathered back, and the output head runs on the full sequence.
"""
import torch
import torch.distributed as dist
from diffusers.models.transformers.transformer_minimax_h3 import (
    MINIMAX_H3_MODALITY_NUM, MiniMaxH3TransformerOutput,
)


def install(tr, attn_core, group=None, block_pre=None):
    import kern
    import ulysses_custom as U
    from fusions import fused_rmsnorm_modulate

    world = dist.get_world_size(group) if dist.is_initialized() else 1
    rank = dist.get_rank(group) if dist.is_initialized() else 0

    def rows_of(n):
        sizes = [n // world + (1 if r < n % world else 0) for r in range(world)]
        return sizes, sum(sizes[:rank])

    def take_rows(t, r0, r1):                 # RoPE tables are (S, d) or (1, S, d)
        return t[r0:r1] if t.dim() == 2 else t[:, r0:r1]

    def attention(blk, normed, rotary_emb):
        a = blk.attn
        q, k, v = a.to_q(normed), a.to_k(normed), a.to_v(normed)
        b, s = q.shape[0], q.shape[1]
        q = q.view(b, s, a.heads, a.head_dim)
        k = k.view(b, s, a.heads, a.head_dim)
        cos, sin = rotary_emb
        q, k = kern.qknorm_rope_v3(q, k, a.norm_q.weight, a.norm_k.weight, cos, sin, a.norm_q.eps,
                                   tok=8, num_warps=4)
        rows, hd = b * s, a.head_dim
        q, k, v = q.view(rows, a.heads, hd), k.view(rows, a.heads, hd), v.view(rows, a.heads, hd)
        if world > 1:
            x = U._packed_qkv_all_to_all(q, k, v, world, group)          # (rows_full, heads/N, 3*hd)
            o = attn_core(x[..., :hd], x[..., hd:2 * hd], x[..., 2 * hd:])
            out = U._packed_out_all_to_all(o.contiguous(), rows, world, group)
        else:
            out = attn_core(q, k, v)
        return a.to_out[0](out.reshape(b, s, a.heads * hd))

    @torch.no_grad()
    def forward(hidden_states, audio_hidden_states, encoder_hidden_states, timestep, timestep_indices,
                token_tags, position_ids, video_indices, audio_indices, text_indices,
                attention_kwargs=None, return_dict=True):
        self = tr
        S = position_ids.shape[0]
        rotary_emb = self.rope(position_ids)
        video_embeds = self.proj_in(hidden_states.to(self.proj_in.weight.dtype))
        audio_embeds = self.audio_proj_in(audio_hidden_states.to(self.audio_proj_in.weight.dtype))
        text_embeds = self.context_embedder(encoder_hidden_states.to(self.context_embedder.weight.dtype))
        text_embeds = self.token_refiner(text_embeds)
        h = text_embeds.new_zeros((text_embeds.shape[0], S, text_embeds.shape[-1]))
        h = h.index_copy(1, text_indices, text_embeds)
        h = h.index_copy(1, video_indices, video_embeds.to(text_embeds.dtype))
        h = h.index_copy(1, audio_indices, audio_embeds.to(text_embeds.dtype))
        temb = self.time_proj(timestep)
        temb = self.time_embedder(temb.to(self.time_embedder.linear_1.weight.dtype))
        idx = timestep_indices * MINIMAX_H3_MODALITY_NUM + token_tags.clamp(min=0)
        if bool((token_tags < 0).any()):
            raise RuntimeError("forward_mp assumes a padless packed sequence")

        if world > 1:                                                     # my contiguous row shard
            sizes, r0 = rows_of(S)
            r1 = r0 + sizes[rank]
            h = h[:, r0:r1].contiguous()
            idx = idx[r0:r1].contiguous()
            rotary_emb = tuple(take_rows(t, r0, r1).contiguous() for t in rotary_emb)
            U.reset_row_counts()

        pending = None
        for i, blk in enumerate(self.transformer_blocks):
            if block_pre is not None:
                block_pre(i)
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = blk.adaln_proj(temb)
            if pending is None:
                normed = fused_rmsnorm_modulate(h, blk.norm1.weight, scale_msa, shift_msa, idx, blk.norm1.eps)
            else:
                h, normed = kern.resnorm_bf16_v2(h, pending[0], pending[1], blk.norm1.weight, scale_msa,
                                                 shift_msa, idx, blk.norm1.eps)
            attn_out = attention(blk, normed, rotary_emb)
            h, normed = kern.resnorm_bf16_v2(h, attn_out, gate_msa, blk.norm2.weight, scale_mlp, shift_mlp,
                                             idx, blk.norm2.eps)
            up = blk.ff.net[0].proj(normed)
            pending = (blk.ff.net[2](kern.swiglu_bf16_v2(up)), gate_mlp)
        h = kern.residual_gate(h, pending[0], pending[1], idx)

        if world > 1:                                                     # rows back in order
            sizes, _ = rows_of(S)
            pad = max(sizes) - h.shape[1]
            hp = torch.nn.functional.pad(h, (0, 0, 0, pad)) if pad else h
            buf = torch.empty((world,) + tuple(hp.shape), dtype=hp.dtype, device=hp.device)
            dist.all_gather_into_tensor(buf, hp.contiguous(), group=group)
            h = torch.cat([buf[r][:, :sizes[r]] for r in range(world)], dim=1)

        h = self.norm_out(h, temb, timestep_indices).to(self.proj_out.weight.dtype)
        video_output = self.proj_out(h).index_select(1, video_indices)
        audio_output = self.audio_proj_out(h).index_select(1, audio_indices)
        if not return_dict:
            return (video_output, audio_output)
        return MiniMaxH3TransformerOutput(sample=video_output, audio_sample=audio_output)

    tr.forward = forward
    return {"world": world, "ulysses": world > 1, "block_pre": block_pre is not None}
