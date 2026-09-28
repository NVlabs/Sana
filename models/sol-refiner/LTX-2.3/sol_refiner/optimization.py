"""Instance-local SOL-Attn and fused operations for the video-only refiner."""

import torch
import torch.nn.functional as F
from diffusers.models.transformers.transformer_ltx2 import (
    LTX2AudioVideoAttnProcessor,
    LTX2VideoTransformerBlock,
    apply_split_rotary_emb,
)

from . import fusion
from .sol_attention import _morton3d_perm, solattn_sm90_attention


def reference_rotary(x, frequencies):
    """Apply split RoPE with the reference BF16 intermediate rounding."""
    cos, sin = frequencies
    b, h, t, half = cos.shape
    x = x.reshape(b, t, h, 2, half).transpose(1, 2)
    out = x * cos.unsqueeze(-2)
    out[..., :1, :].addcmul_(-sin.unsqueeze(-2), x[..., 1:, :])
    out[..., 1:, :].addcmul_(sin.unsqueeze(-2), x[..., :1, :])
    return out.flatten(-2).transpose(1, 2).reshape(b, t, -1)


class RefinerAttention(LTX2AudioVideoAttnProcessor):
    def __init__(self, engine, layer, sparse):
        super().__init__()
        self.engine, self.layer, self.sparse = engine, layer, sparse

    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        query_rotary_emb=None,
        key_rotary_emb=None,
        perturbation_mask=None,
        all_perturbed=None,
    ):
        if perturbation_mask is not None or all_perturbed:
            raise ValueError("Perturbed attention is not used by the refiner")
        context = (
            hidden_states if encoder_hidden_states is None else encoder_hidden_states
        )
        q, k, v = attn.to_q(hidden_states), attn.to_k(context), attn.to_v(context)
        nq = fusion.rmsnorm_w(q, attn.norm_q) if self.engine else None
        nk = fusion.rmsnorm_w(k, attn.norm_k) if self.engine else None
        q = attn.norm_q(q) if nq is None else nq
        k = attn.norm_k(k) if nk is None else nk
        if query_rotary_emb is not None:
            if attn.rope_type != "split":
                raise ValueError("The LTX-2.3 refiner requires split RoPE")
            krot = query_rotary_emb if key_rotary_emb is None else key_rotary_emb
            if self.engine:
                qr, kr = (
                    fusion.rope_split(q, *query_rotary_emb),
                    fusion.rope_split(k, *krot),
                )
            else:
                qr, kr = (
                    reference_rotary(q, query_rotary_emb),
                    reference_rotary(k, krot),
                )
            q = apply_split_rotary_emb(q, query_rotary_emb) if qr is None else qr
            k = apply_split_rotary_emb(k, krot) if kr is None else kr
        q, k, v = [x.unflatten(2, (attn.heads, -1)).transpose(1, 2) for x in (q, k, v)]
        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(
                attention_mask, context.shape[1], context.shape[0]
            )
            attention_mask = attention_mask.view(
                context.shape[0], attn.heads, -1, attention_mask.shape[-1]
            )
        sparse = (
            self.sparse
            and self.layer not in {0, self.engine.layers - 1}
            and self.engine.sparse_shape
        )
        if sparse and attention_mask is None:
            out = solattn_sm90_attention(
                q,
                k,
                v,
                tau=self.engine.tau,
                target_density=self.engine.density,
                cache=self.engine.tau_cache,
            )
            self.engine.sparse_calls += 1
        else:
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=attention_mask)
        out = out.transpose(1, 2)
        if attn.to_gate_logits is not None:
            out = out * (2 * attn.to_gate_logits(hidden_states).sigmoid()).unsqueeze(-1)
        return attn.to_out[1](attn.to_out[0](out.flatten(2)))


class FusedVideoBlock(LTX2VideoTransformerBlock):
    def forward(
        self,
        hidden_states,
        audio_hidden_states,
        *,
        encoder_hidden_states,
        temb,
        temb_prompt=None,
        video_rotary_emb=None,
        encoder_attention_mask=None,
        self_attention_mask=None,
        **kwargs,
    ):
        if kwargs.get("use_a2v_cross_attention", False) or kwargs.get(
            "use_v2a_cross_attention", False
        ):
            raise ValueError("FusedVideoBlock only supports isolated video refinement")
        batch = hidden_states.shape[0]
        (
            shift,
            scale,
            gate,
            ff_shift,
            ff_scale,
            ff_gate,
            text_shift,
            text_scale,
            text_gate,
        ) = self.get_mod_params(self.scale_shift_table, temb, batch)
        x = fusion.rmsnorm_adaln(hidden_states, scale, shift, self.norm1.eps)
        x = self.attn1(
            x, query_rotary_emb=video_rotary_emb, attention_mask=self_attention_mask
        )
        hidden_states = hidden_states + x * gate
        pshift, pscale = self.get_mod_params(
            self.prompt_scale_shift_table, temb_prompt, batch
        )
        context = encoder_hidden_states * (1 + pscale) + pshift
        x = fusion.rmsnorm_adaln(hidden_states, text_scale, text_shift, self.norm2.eps)
        x = self.attn2(
            x, encoder_hidden_states=context, attention_mask=encoder_attention_mask
        )
        hidden_states = hidden_states + x * text_gate
        x = fusion.rmsnorm_adaln(hidden_states, ff_scale, ff_shift, self.norm3.eps)
        hidden_states = fusion.gate_residual(hidden_states, self.ff(x), ff_gate)
        return hidden_states, audio_hidden_states


class SolEngine:
    def __init__(self, transformer, tau=None, density=None):
        if tau is not None and density is not None:
            raise ValueError("Choose tau or density, not both")
        if density is not None and not 0 < density <= 1:
            raise ValueError("Density must be in (0, 1]")
        self.tau = 1.5 if tau is None and density is None else tau
        if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 9:
            raise ValueError("This SOL-Attn backend requires an SM90 GPU (H100/H200)")

        self.layers = len(transformer.transformer_blocks)
        self.tau_cache = {}
        self.density, self.step, self.sparse_calls = density, 0, 0
        self.perm = self.inverse = None
        self.sparse_shape = False
        for i, block in enumerate(transformer.transformer_blocks):
            block.__class__ = FusedVideoBlock
            block.attn1.set_processor(RefinerAttention(self, i, sparse=True))
            block.attn2.set_processor(RefinerAttention(self, i, sparse=False))
            block.register_forward_pre_hook(self._positions, with_kwargs=True)
        transformer.transformer_blocks[0].register_forward_pre_hook(
            self._reorder, with_kwargs=True, prepend=True
        )
        transformer.transformer_blocks[-1].register_forward_hook(self._restore)

    def begin(self, grid, device):
        self.perm, self.inverse = _morton3d_perm(grid, device)
        self.sparse_shape = True
        self.step = self.sparse_calls = 0

    def _reorder(self, module, args, kwargs):
        kwargs["hidden_states"] = kwargs["hidden_states"].index_select(1, self.perm)
        return args, kwargs

    def _positions(self, module, args, kwargs):
        # The transformer passes the same original positional tensors to every block.
        for name in ["temb"]:
            if kwargs[name].shape[1] == self.perm.numel():
                kwargs[name] = kwargs[name].index_select(1, self.perm)
        if kwargs.get("video_rotary_emb") is not None:
            kwargs["video_rotary_emb"] = tuple(
                t.index_select(2, self.perm) for t in kwargs["video_rotary_emb"]
            )
        return args, kwargs

    def _restore(self, module, args, output):
        return output[0].index_select(1, self.inverse), output[1]
