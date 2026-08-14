"""ChunkedDiffusionNABlock: deferred inject + W-chunked attn + modulating MLP."""

from __future__ import annotations

import torch

from ltx_core.model.video_vae.transformer.blocks import DiffusionNABlock
from ltx_core.model.video_vae.transformer.chunked.attn import chunked as residual_attn
from ltx_core.model.video_vae.transformer.chunked.context import deferred as inject_context
from ltx_core.model.video_vae.transformer.chunked.mlp import residual_modulating_mlp
from ltx_core.model.video_vae.transformer.layers import LinearPixelShuffleUpsample


class ChunkedDiffusionNABlock(DiffusionNABlock):
    """Deferred-context diffusion block.
    ``forward`` / ``forward_x_ctx``: in-place inject (eager) then attn+mlp.
    ``forward_attn_mlp``: opaque void-mutate residual + SwiGLU — this is what
    ChunkedCompile ``torch.compile``s (do not compile inject).
    ``stage4_upsample`` is wired by ``apply`` to ``decoder.upsamples[3]`` so
    inject runs sequential upsample Linear then ``context_proj``.
    """

    stage4_upsample: LinearPixelShuffleUpsample | None

    def forward_attn_mlp(
        self,
        x: torch.Tensor,
        scale_msa: torch.Tensor,
        shift_msa: torch.Tensor,
        scale_mlp: torch.Tensor,
        shift_mlp: torch.Tensor,
    ) -> torch.Tensor:
        """In-place chunked NA residual + modulating SwiGLU (compile target)."""
        x = residual_attn(x, self.attn, self.norm1, scale_msa, shift_msa)
        x = residual_modulating_mlp(x, self.mlp, self.norm2, scale_mlp, shift_mlp, self.mlp.tile)
        return x

    def forward_x_ctx(
        self,
        x: torch.Tensor,
        stage4_feat: torch.Tensor,
        modulation: tuple[torch.Tensor, ...],
        *,
        drop_leading_frame: bool = True,
    ) -> torch.Tensor:
        if self.stage4_upsample is None:
            raise RuntimeError("stage4_upsample not configured; apply DiffVAEConfig with block=CHUNKED")

        scale_msa, shift_msa, scale_mlp, shift_mlp = self._modulation(modulation)
        upsample = self.stage4_upsample
        x = inject_context(
            x,
            stage4_feat,
            upsample.proj,
            self.context_proj,
            tuple(upsample.stride),  # type: ignore[arg-type]
            w_chunks=self.attn.w_chunks,
            drop_leading_frame=drop_leading_frame,
        )
        return self.forward_attn_mlp(x, scale_msa, shift_msa, scale_mlp, shift_mlp)

    def forward(
        self,
        x: torch.Tensor,
        stage4_feat: torch.Tensor,
        modulation: tuple[torch.Tensor, ...],
        *,
        drop_leading_frame: bool = True,
    ) -> torch.Tensor:
        return self.forward_x_ctx(x, stage4_feat, modulation, drop_leading_frame=drop_leading_frame)
