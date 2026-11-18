"""Fused CuTe DSL DiffusionNABlock for deferred stage-4 inputs."""

from __future__ import annotations

import torch

from ltx_core.model.video_vae.transformer.blocks import DiffusionNABlock
from ltx_core.model.video_vae.transformer.dsl_kernels.attn import require_softmax_bound
from ltx_core.model.video_vae.transformer.layers import AdaLNZero, LinearPixelShuffleUpsample

#: The stage-4 upsample fused into ``context_proj``, packed by SDOps as one Linear per
#: pixel-shuffle subpixel: ``(P * C, Ctx)`` / ``(P * C,)``. Registered empty by
#: BLACKWELL_DSL apply, which runs on the meta model before any weight exists.
FUSED_CTX_WEIGHT_BUFFER = "fused_ctx_weight"
FUSED_CTX_BIAS_BUFFER = "fused_ctx_bias"


def fused_ctx_weights(block: DiffusionNABlock) -> "tuple[torch.Tensor, torch.Tensor] | None":
    """This block's packed fused-context weights, or ``None`` if they never landed.
    A load that skipped the DSL SDOps leaves the buffers apply registered untouched, and
    since that apply runs on the meta model and the load assigns rather than copies, what
    is left is still on meta -- which is the check.
    """
    weight = getattr(block, FUSED_CTX_WEIGHT_BUFFER, None)
    bias = getattr(block, FUSED_CTX_BIAS_BUFFER, None)
    if weight is None or bias is None or weight.is_meta or weight.numel() == 0:
        return None
    return weight, bias


def fused_na_block_supported(block: DiffusionNABlock, x: torch.Tensor) -> bool:
    """Whether the fused kernel serves this block at this volume."""
    from ltx_kernels.vae import fna_supported  # noqa: PLC0415

    fused = fused_ctx_weights(block)
    upsample = block.stage4_upsample
    if fused is None or upsample is None:
        return False
    _, t, h, w, channels = x.shape
    return fna_supported(
        C=channels,
        Ctx=int(fused[0].shape[1]),
        Hidd=int(block.mlp.w_gate.weight.shape[0]),
        num_heads=block.attn.num_heads,
        kernel_size=block.attn.kernel_size,
        T=t,
        H=h,
        W=w,
        upsample_stride=tuple(upsample.stride),  # type: ignore[arg-type]
    )


class DSLDiffusionNABlock(DiffusionNABlock):
    """Deferred-context diffusion block that runs as one fused CuTe DSL launch.
    Installed via ``__class__`` swap. Signature matches deferred decode:
    ``(x, stage4_feat, modulation)``, and ``forward`` / ``forward_x_ctx`` are the same
    entry (Chunked naming).
    There is no host fallback: a shape the kernel declines raises rather than quietly
    running a slower route, because in this pathway that is a decode the caller should
    have tiled differently, not a shape to absorb. Use another
    :class:`~ltx_core.model.video_vae.transformer.config.DiffVAEMode` for volumes the
    kernel does not serve.
    """

    stage4_upsample: LinearPixelShuffleUpsample | None

    def forward_x_ctx(
        self,
        x: torch.Tensor,
        stage4_feat: torch.Tensor,
        modulation: tuple[torch.Tensor, ...],
        *,
        drop_leading_frame: bool = True,
        out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self._fused_forward_x_ctx(x, stage4_feat, modulation, drop_leading_frame=drop_leading_frame, out=out)

    def forward(
        self,
        x: torch.Tensor,
        stage4_feat: torch.Tensor,
        modulation: tuple[torch.Tensor, ...],
        *,
        drop_leading_frame: bool = True,
        out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.forward_x_ctx(x, stage4_feat, modulation, drop_leading_frame=drop_leading_frame, out=out)

    @torch.compiler.disable
    def _fused_forward_x_ctx(
        self,
        x: torch.Tensor,
        stage4_feat: torch.Tensor,
        modulation: tuple[torch.Tensor, ...],
        *,
        drop_leading_frame: bool,
        out: torch.Tensor | None,
    ) -> torch.Tensor:
        """Marshal weights into :func:`ltx_kernels.vae.run_block_fna_dsl`."""
        from ltx_kernels.vae import run_block_fna_dsl  # noqa: PLC0415

        upsample = self.stage4_upsample
        if upsample is None:
            raise RuntimeError("stage4_upsample not configured; apply DiffVAEConfig with block=BLACKWELL_DSL")
        fused = fused_ctx_weights(self)
        if fused is None:
            raise RuntimeError(
                f"{FUSED_CTX_WEIGHT_BUFFER} unset: apply DiffVAEMode.BLACKWELL_DSL with "
                "video_decoder_sd_ops_for_checkpoint(na_dsl_kernel=True)"
            )
        w_fused, b_fused = fused
        scale_msa, shift_msa, _, scale_mlp, shift_mlp, _, _ = [
            modulation[i] + self.scale_shift_table[i].view(1, 1, 1, 1, -1) for i in range(AdaLNZero.NUM_CHUNKS)
        ]
        attn = self.attn
        return run_block_fna_dsl(
            x,
            stage4_feat,
            w_ctx=w_fused,
            b_ctx=b_fused,
            upsample_stride=tuple(upsample.stride),  # type: ignore[arg-type]
            drop_leading_frame=drop_leading_frame,
            norm1_weight=self.norm1.weight,
            norm2_weight=self.norm2.weight,
            q_norm_weight=attn.q_norm.weight,
            k_norm_weight=attn.k_norm.weight,
            softmax_bound=require_softmax_bound(attn),
            scale_msa=scale_msa,
            shift_msa=shift_msa,
            scale_mlp=scale_mlp,
            shift_mlp=shift_mlp,
            w_q=attn.qkv.to_q.weight,
            w_k=attn.qkv.to_k.weight,
            w_v=attn.qkv.to_v.weight,
            w_o=attn.proj.weight,
            b_q=attn.qkv.to_q.bias,
            b_k=attn.qkv.to_k.bias,
            b_v=attn.qkv.to_v.bias,
            b_o=attn.proj.bias,
            w_gate=self.mlp.w_gate.weight,
            w_up=self.mlp.w_up.weight,
            w_down=self.mlp.w_down.weight,
            inv_t=attn.rope_inv_t.to(device=x.device),
            inv_h=attn.rope_inv_h.to(device=x.device),
            inv_w=attn.rope_inv_w.to(device=x.device),
            num_heads=attn.num_heads,
            attn_scale=attn.scale,
            rope_dim_split=tuple(attn.rope_dim_split),
            kernel_size=tuple(attn.kernel_size),
            out=out,
        ).to(dtype=x.dtype)
