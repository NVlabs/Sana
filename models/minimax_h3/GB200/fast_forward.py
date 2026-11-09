"""A replacement for `MiniMaxH3Transformer3DModel.forward` that owns the block loop.

Owning the loop is what allows the block boundary to be fused: block i ends with
`h + gate_mlp_i[idx] * ff_i`, and block i+1 starts with `rmsnorm_modulate(h)` -- one fused
residual+gate+norm+modulate kernel instead of an eager gather/mul/add followed by a norm kernel.

kernels="v1"  the shipped Triton fusions, attention through the installed `attn.forward`
kernels="v2"  the tuned kernels in `kern.py`, and the single-GPU attention wrapper owned here:
              three projections, q and k normalized+rotated in one launch, the attention core
              (`tr._h3_attn_core`, Sol-Attn), output projection.

`fp8_blocks` names the blocks whose four linears run as MXFP8 (E4M3 values, one E8M0 scale per 32
along K); in those blocks the norm/modulate and SwiGLU kernels emit the quantized activation
directly. Everything outside the block stack is upstream code, unchanged.

`cp=(world, rank, group)` runs the same loop under Ulysses context parallelism (kernels="v2" only).
The diffusers CP hooks still split `rope`'s output and `norm_out`'s index and gather at the heads,
but they sit on the block modules this loop never calls, so the loop takes its own rows of the
packed sequence and its own rows of the AdaLN index, with the same `tensor_split` the hooks use.
Attention trades rows for heads in one packed all-to-all (`ulysses_custom`), runs the attention
core on the whole sequence and its local heads, and trades back.
"""
import torch

from diffusers.models.transformers.transformer_minimax_h3 import (
    MINIMAX_H3_MODALITY_NUM, MiniMaxH3TransformerOutput,
)


def install(tr, fp8_blocks=(), kernels="v1", cp=None):
    import kern
    from fusions import fused_residual_gate_rmsnorm_modulate, fused_rmsnorm_modulate, fused_swiglu

    fp8_blocks = set(fp8_blocks)
    if fp8_blocks:
        import mxfp8
        for i in sorted(fp8_blocks):
            blk = tr.transformer_blocks[i]
            a = blk.attn
            a.to_q, a.to_k, a.to_v = (mxfp8.MXFP8Linear(m) for m in (a.to_q, a.to_k, a.to_v))
            a.to_out[0] = mxfp8.MXFP8Linear(a.to_out[0])
            blk.ff.net[0].proj = mxfp8.MXFP8Linear(blk.ff.net[0].proj)
            blk.ff.net[2] = mxfp8.MXFP8Linear(blk.ff.net[2])
        torch.cuda.empty_cache()
    v2 = kernels == "v2"
    world, rank, group = cp if cp is not None else (1, 0, None)
    if world > 1:
        if not v2:
            raise ValueError("fast_forward under context parallelism needs kernels='v2'")
        import ulysses_custom

    def norm_mod(i, h, w, scale, shift, idx, eps):
        if i in fp8_blocks:
            return mxfp8.fused_rmsnorm_modulate_mxfp8(h, w, scale, shift, idx, eps)
        return fused_rmsnorm_modulate(h, w, scale, shift, idx, eps)

    def res_norm_mod(i, h, branch, gate, w, scale, shift, idx, eps):
        if i in fp8_blocks:
            if v2:
                return kern.resnorm_mx_v2(h, branch, gate, w, scale, shift, idx, eps, chunk=2048, num_warps=8)
            return mxfp8.fused_residual_gate_rmsnorm_modulate_mxfp8(h, branch, gate, w, scale, shift, idx, eps)
        if v2:
            return kern.resnorm_bf16_v2(h, branch, gate, w, scale, shift, idx, eps)
        return fused_residual_gate_rmsnorm_modulate(h, branch, gate, w, scale, shift, idx, eps)

    def swiglu(i, x):
        if i in fp8_blocks:
            return kern.swiglu_mx_v2(x, block_rows=16, groups=16, num_warps=4) if v2 else mxfp8.fused_swiglu_mxfp8(x)
        return kern.swiglu_bf16_v2(x) if v2 else fused_swiglu(x)

    attn_core = getattr(tr, "_h3_attn_core", None)

    def attention(blk, normed, rotary_emb):
        if not v2:
            return blk.attn(normed, rotary_emb, None)
        a = blk.attn
        q, k, v = a.to_q(normed), a.to_k(normed), a.to_v(normed)
        b, s = q.shape[0], q.shape[1]
        q = q.view(b, s, a.heads, a.head_dim)
        k = k.view(b, s, a.heads, a.head_dim)
        cos, sin = rotary_emb
        q, k = kern.qknorm_rope_v3(q, k, a.norm_q.weight, a.norm_k.weight, cos, sin, a.norm_q.eps,
                                   tok=8, num_warps=4)
        rows = b * s
        q, k, v = (x.reshape(rows, a.heads, a.head_dim) for x in (q, k, v))
        if world > 1:
            packed = ulysses_custom._packed_qkv_all_to_all(q, k, v, world=world, group=group)
            q, k, v = packed.split(a.head_dim, dim=-1)
            out = attn_core(q, k, v).contiguous()
            out = ulysses_custom._packed_out_all_to_all(out, rows, world, group)
        else:
            out = attn_core(q, k, v)
        return a.to_out[0](out.reshape(b, s, a.heads * a.head_dim))

    def forward(hidden_states, audio_hidden_states, encoder_hidden_states, timestep, timestep_indices,
                token_tags, position_ids, video_indices, audio_indices, text_indices,
                attention_kwargs=None, return_dict=True):
        self = tr
        sequence_length = position_ids.shape[0]
        rotary_emb = self.rope(position_ids)
        video_embeds = self.proj_in(hidden_states.to(self.proj_in.weight.dtype))
        audio_embeds = self.audio_proj_in(audio_hidden_states.to(self.audio_proj_in.weight.dtype))
        text_embeds = self.context_embedder(encoder_hidden_states.to(self.context_embedder.weight.dtype))
        text_embeds = self.token_refiner(text_embeds)
        h = text_embeds.new_zeros((text_embeds.shape[0], sequence_length, text_embeds.shape[-1]))
        h = h.index_copy(1, text_indices, text_embeds)
        h = h.index_copy(1, video_indices, video_embeds.to(text_embeds.dtype))
        h = h.index_copy(1, audio_indices, audio_embeds.to(text_embeds.dtype))
        temb = self.time_proj(timestep)
        temb = self.time_embedder(temb.to(self.time_embedder.linear_1.weight.dtype))
        idx = timestep_indices * MINIMAX_H3_MODALITY_NUM + token_tags.clamp(min=0)
        if bool((token_tags < 0).any()):
            raise RuntimeError("fast_forward assumes a padless packed sequence")
        if world > 1:                       # this rank's rows; `rope` is already split by its hook
            h = h.tensor_split(world, dim=1)[rank]
            idx = idx.tensor_split(world)[rank]
            if rotary_emb[0].shape[0] != h.shape[1]:
                raise RuntimeError(f"rope rows {rotary_emb[0].shape[0]} != local rows {h.shape[1]}")

        pending = None      # (ff_output, gate_mlp) of the previous block, not yet added to h
        for i, blk in enumerate(self.transformer_blocks):
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = blk.adaln_proj(temb)
            if pending is None:
                normed = norm_mod(i, h, blk.norm1.weight, scale_msa, shift_msa, idx, blk.norm1.eps)
            else:
                h, normed = res_norm_mod(i, h, pending[0], pending[1], blk.norm1.weight, scale_msa,
                                         shift_msa, idx, blk.norm1.eps)
            attn_out = attention(blk, normed, rotary_emb)
            h, normed = res_norm_mod(i, h, attn_out, gate_msa, blk.norm2.weight, scale_mlp, shift_mlp,
                                     idx, blk.norm2.eps)
            up = blk.ff.net[0].proj(normed)
            pending = (blk.ff.net[2](swiglu(i, up)), gate_mlp)
        h = kern.residual_gate(h, pending[0], pending[1], idx)

        h = self.norm_out(h, temb, timestep_indices).to(self.proj_out.weight.dtype)
        video_output = self.proj_out(h).index_select(1, video_indices)
        audio_output = self.audio_proj_out(h).index_select(1, audio_indices)
        if not return_dict:
            return (video_output, audio_output)
        return MiniMaxH3TransformerOutput(sample=video_output, audio_sample=audio_output)

    tr.forward = forward
    return {"fp8_blocks": sorted(fp8_blocks), "kernels": kernels, "ulysses_degree": world}
