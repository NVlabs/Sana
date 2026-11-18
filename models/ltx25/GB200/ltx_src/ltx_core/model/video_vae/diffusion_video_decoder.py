"""Diffusion (NATTEN) video VAE decoder."""

from __future__ import annotations

from typing import Iterator, List, Literal, Tuple

import torch
from torch import nn

from ltx_core.devices import cuda_activation_budget_bytes
from ltx_core.model.disposable import Disposable
from ltx_core.model.transformer.timestep_embedding import PixArtAlphaCombinedTimestepSizeEmbeddings
from ltx_core.model.video_vae import diffusion_tiling
from ltx_core.model.video_vae.ops import PerChannelStatistics, patchify, unpatchify
from ltx_core.model.video_vae.transformer import (
    AdaLNZero,
    ChannelLinear,
    CombinedDiffusionNABlock,
    LinearPixelShuffleUpsample,
    NABlock,
)
from ltx_core.model.video_vae.transformer.config import DiffVAEMode
from ltx_core.model.video_vae.video_vae import VideoDecoder
from ltx_core.tiling import (
    Tile,
    TileSizeConfig,
    TilingConfig,
    group_tiles_by_temporal_slice,
    masks_are_complementary,
    scale_by_masks_1d,
)
from ltx_core.types import SpatioTemporalScaleFactors, VideoLatentShape
from ltx_core.utils import to_velocity

# Production CausalVideoAutoencoderL decoder layout: channel width per stage.
# Mirrors the (non-diffusion) NA decoder's stage spec.
_L_STAGE_CHANNELS: Tuple[int, ...] = (1024, 512, 256, 256, 128)
_L_STAGE_DEPTHS: Tuple[int, ...] = (4, 6, 4, 2, 2)
# (stride, out_channels_reduction_factor) per upsample, in stage order.
_L_UPSAMPLES: Tuple[Tuple[Tuple[int, int, int], int], ...] = (
    ((1, 2, 2), 2),  # compress_space x2
    ((2, 1, 1), 2),  # compress_time x2
    ((2, 2, 2), 1),  # compress_all x1 (channel-preserving)
    ((2, 2, 2), 2),  # compress_all x2
)
# Per-stage 3D neighborhood (K_t, K_h, K_w).
_L_STAGE_KERNELS: Tuple[Tuple[int, int, int], ...] = (
    (3, 7, 7),
    (3, 7, 7),
    (3, 5, 5),
    (3, 5, 5),
    (3, 3, 3),
)

# Stage-5 (diffusion stage) defaults: wider kernel + more blocks than the
# deterministic stages, since it carries the entire per-step diffusion compute.
_DIFF_STAGE5_KERNEL_DEFAULT: Tuple[int, int, int] = (3, 7, 7)
_DIFF_STAGE5_DEPTH_DEFAULT: int = 8
_DIFF_STAGE_DEPTHS_DEFAULT: Tuple[int, ...] = (*_L_STAGE_DEPTHS[:-1], _DIFF_STAGE5_DEPTH_DEFAULT)


class DiffusionVideoDecoder(nn.Module, Disposable, VideoDecoder):
    """Diffusion-based video VAE decoder (Neighborhood-Attention backbone).
    Minimal port of the reference ``NADiffusionDecoder``.
    Stages 1-4 deterministically upsample the latent into a context volume
    (same NA-upsample path as the non-diffusion NA decoder). Stage 5 runs
    ``DiffusionNABlock``s that denoise the patchified noised pixels ``x_t``,
    guided by that context via AdaLN-Zero scale/shift (ungated residuals;
    legacy static gates are folded into Linear weights at load time).
    Last-frame NATTEN window-shift is mitigated by temporarily replicating the
    last latent frame ``(stage1_K_t // 2) * 2`` times through stages 1-4, then
    cropping that appendix from context before stage 5 - but only down to
    ``max(original_context_T, stage5_kernel[0])`` so undersized clips (e.g. a
    single latent frame) still satisfy NATTEN's kernel floor. Latents / tiles
    below ``stage_min_tile_sizes`` are edge-padded first via ``diffusion_tiling``;
    leftover pad is cropped from the final pixels.
    """

    def __init__(  # noqa: PLR0913
        self,
        in_channels: int = 128,
        out_channels: int = 3,
        patch_size: int = 4,
        head_dim: int = 64,
        rope_dim_split: Tuple[int, int, int] | None = None,
        stage_channels: Tuple[int, ...] = _L_STAGE_CHANNELS,
        stage_depths: Tuple[int, ...] = _DIFF_STAGE_DEPTHS_DEFAULT,
        stage_kernels: Tuple[Tuple[int, int, int], ...] = _L_STAGE_KERNELS,
        upsamples: Tuple[Tuple[Tuple[int, int, int], int], ...] = _L_UPSAMPLES,
        stage5_kernel: Tuple[int, int, int] = _DIFF_STAGE5_KERNEL_DEFAULT,
        stage5_channels: int | None = None,
        t_emb_dim: int = 384,
        default_num_inference_steps: int = 2,
        timestep_scale_multiplier: float = 1.0,
        model_output_type: Literal["v", "x0"] = "v",
    ) -> None:
        super().__init__()
        assert len(stage_channels) == len(stage_depths) == len(stage_kernels)
        assert len(upsamples) == len(stage_channels) - 1
        for c in stage_channels:
            assert c % head_dim == 0, f"stage_channels {stage_channels} must each be a multiple of head_dim={head_dim}"

        self.patch_size = patch_size
        self.register_buffer(
            "default_inference_timesteps",
            torch.linspace(1.0, 1.0 / default_num_inference_steps, default_num_inference_steps, device="cpu"),
            persistent=False,
        )
        self.out_channels = out_channels
        self.stage_channels = stage_channels
        self.stage_depths = stage_depths
        self.base_channels = stage_channels[-1]
        self.causal = False
        self.timestep_conditioning = True
        self.video_downscale_factors = SpatioTemporalScaleFactors.default()
        self.stage5_kernel: Tuple[int, int, int] = tuple(stage5_kernel)  # type: ignore[assignment]
        # NATTEN last-frame border workaround: replicate last latent frame
        # ``(K_t // 2) * 2`` times through stages 1-4, then crop the appendix
        # off context before stage 5 down to at least ``stage5_kernel[0]``.
        self._natten_trailing_pad_latent_frames = (stage_kernels[0][0] // 2) * 2

        # Encoder output is per-channel normalized; undo before conv_in (same as ConvVideoDecoder).
        self.per_channel_statistics = PerChannelStatistics(latent_channels=in_channels)

        self.conv_in = ChannelLinear(in_channels, stage_channels[0], bias=True)

        self.det_stages = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        n_det_stages = len(stage_channels) - 1
        for stage_i in range(n_det_stages):
            c = stage_channels[stage_i]
            depth = stage_depths[stage_i]
            kernel = stage_kernels[stage_i]
            self.det_stages.append(
                nn.ModuleList(
                    [
                        NABlock(dim=c, kernel_size=kernel, head_dim=head_dim, rope_dim_split=rope_dim_split)
                        for _ in range(depth)
                    ]
                )
            )
            stride, reduction = upsamples[stage_i]
            self.upsamples.append(
                LinearPixelShuffleUpsample(in_channels=c, stride=stride, out_channels_reduction_factor=reduction)
            )

        self.t_embedder = PixArtAlphaCombinedTimestepSizeEmbeddings(embedding_dim=t_emb_dim, size_emb_dim=0)

        c_ctx = stage_channels[-1]
        self.context_channels = c_ctx
        c5 = stage5_channels if stage5_channels is not None else c_ctx
        d5 = stage_depths[-1]
        assert c5 % head_dim == 0, f"stage5_channels {c5} must be a multiple of head_dim={head_dim}"
        noised_pixel_channels = out_channels * (patch_size**2)

        # Latent-grid floor so stages 1-3 (full volume) never undershoot NA.
        self.stage_min_tile_sizes: Tuple[int, int, int] = diffusion_tiling.all_stages_min_tile_size(
            stage_kernels, upsamples, stage5_kernel
        )
        # Stage-4-input tile floor / overlap halos (only stages 4-5 are tiled).
        up3_stride = upsamples[3][0]
        self.tile_min_sizes: Tuple[int, int, int] = diffusion_tiling.compute_tile_min_size(
            stage_kernels[3], stage5_kernel, up3_stride
        )
        self.tile_halos: Tuple[Tuple[int, int, int], Tuple[int, int, int]] = diffusion_tiling.compute_tile_halos(
            stage_kernels[3],
            stage_depths[3],
            stage5_kernel,
            stage_depths[-1],
            up3_stride,
        )
        self.conv_in_x_t = ChannelLinear(noised_pixel_channels, c5, bias=True)

        # Shared AdaLN-Zero (7-chunk for shape compat; gate slots unused in block).
        self.shared_adaln = AdaLNZero(dim=c5, t_emb_dim=t_emb_dim)

        self.diff_blocks = nn.ModuleList(
            [
                CombinedDiffusionNABlock(
                    dim=c5,
                    kernel_size=stage5_kernel,
                    context_channels=c_ctx,
                    head_dim=head_dim,
                    rope_dim_split=rope_dim_split,
                )
                for _ in range(d5)
            ]
        )

        self.norm_out = nn.RMSNorm(c5, eps=1e-6)
        self.conv_out = ChannelLinear(c5, noised_pixel_channels, bias=True)

        self.timestep_scale_multiplier = timestep_scale_multiplier
        self.model_output_type = model_output_type
        # Set True by ``compile_diffusion_decoder`` so decode marks T/H/W dynamic.
        self.mark_dynamic_shapes = False
        # When True, skip stage-4 upsample and inject via deferred sequential upsample+proj.
        # Default False = combined pathway (``CombinedDiffusionNABlock``). Chunked DiffVAE
        # modes flip this via ``apply_diffvae_config``.
        self.deferred_stage4_upsample = False

    def _run_det_stage(self, x: torch.Tensor, stage_i: int, drop_leading_frame: bool) -> torch.Tensor:
        """One deterministic stage: NA blocks + upsample."""
        if self.mark_dynamic_shapes:
            for dim in (1, 2, 3):
                torch._dynamo.mark_dynamic(x, dim)
        for block in self.det_stages[stage_i]:
            x = block(x)
        return self.upsamples[stage_i](x, drop_leading_frame=drop_leading_frame)

    def forward_stages_1_to_3(
        self,
        z_noisy: torch.Tensor,
        drop_leading_frame: bool = True,
    ) -> torch.Tensor:
        """Stages 1-3 on a full (or already ghost-padded) latent → stage-4 input feature.
        Output is channels-last ``(B, T, H, W, C)`` at stage-4 input resolution.
        Callers that want NATTEN trailing ghosting should pad the latent first via
        ``diffusion_tiling.pad_trailing_latent_for_natten_border``.
        """
        z_noisy = self.per_channel_statistics.un_normalize(z_noisy)
        x = z_noisy.permute(0, 2, 3, 4, 1)
        x = self.conv_in(x)
        for stage_i in range(3):
            x = self._run_det_stage(x, stage_i, drop_leading_frame)
        return x

    def forward_stage_4(
        self,
        x: torch.Tensor,
        drop_leading_frame: bool = True,
        pad_trailing: bool = True,
    ) -> torch.Tensor:
        """Stage 4 on a stage-4-input feature tile → stage-5 context (or pre-upsample feat).
        ``x`` is channels-last. When ``pad_trailing``, soft-crop the ghosting
        appendix before returning (ghost pad must already be present upstream).
        When ``deferred_stage4_upsample`` is set, runs NA blocks only (no
        ``upsamples[3]``) and crops ghost at pre-upsample temporal resolution.
        """
        if self.deferred_stage4_upsample:
            if self.mark_dynamic_shapes:
                for dim in (1, 2, 3):
                    torch._dynamo.mark_dynamic(x, dim)
            for block in self.det_stages[3]:
                x = block(x)
            if pad_trailing:
                up_t = int(self.upsamples[3].stride[0])
                x = diffusion_tiling.crop_trailing_context_natten_pad(
                    x,
                    n_latent_frames=self._natten_trailing_pad_latent_frames,
                    time_scale=self.video_downscale_factors.time // up_t,
                    stage5_kernel_t=max(1, -(-self.stage5_kernel[0] // up_t)),
                )
            return x

        x = self._run_det_stage(x, 3, drop_leading_frame)
        if pad_trailing:
            x = diffusion_tiling.crop_trailing_context_natten_pad(
                x,
                n_latent_frames=self._natten_trailing_pad_latent_frames,
                time_scale=self.video_downscale_factors.time,
                stage5_kernel_t=self.stage5_kernel[0],
            )
        return x

    def _context_and_x_for_diff_step(self, context: torch.Tensor, x_t: torch.Tensor) -> torch.Tensor:
        """Build block-ready ``[context | conv_in_x_t(patched x)]`` for ``forward_diff_step``."""
        noised_pixels_patched = patchify(x_t, patch_size_hw=self.patch_size, patch_size_t=1)
        x = self.conv_in_x_t(noised_pixels_patched.permute(0, 2, 3, 4, 1))
        return torch.cat([context, x], dim=-1)

    def _x_for_diff_step(self, x_t: torch.Tensor) -> torch.Tensor:
        """Conv-processed noised pixels only (deferred-context path)."""
        noised_pixels_patched = patchify(x_t, patch_size_hw=self.patch_size, patch_size_t=1)
        return self.conv_in_x_t(noised_pixels_patched.permute(0, 2, 3, 4, 1))

    def forward_diff_step(
        self,
        context_and_x: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """One stage-5 diffusion step. Returns the model prediction in pixel space.
        ``context_and_x`` is ``[latent_context | conv_in_x_t(x)]`` (channels-last), built
        at the call site via ``_context_and_x_for_diff_step``. That single buffer is
        reused across ``diff_blocks``: each block writes its output x-half back
        with ``copy_`` (no per-block ``cat``). One-tensor layout keeps Dynamo
        T/H/W symbols identical under ``mark_dynamic``.
        """
        x_half = context_and_x[..., self.context_channels :]
        t_emb = self.t_embedder(self.timestep_scale_multiplier * t, hidden_dtype=x_half.dtype)
        modulation = self.shared_adaln(t_emb)

        if self.mark_dynamic_shapes:
            for dim in (1, 2, 3):
                torch._dynamo.mark_dynamic(context_and_x, dim)

        for block in self.diff_blocks:
            x_half.copy_(block.forward_combined(context_and_x, modulation))
        x = x_half

        x = self.norm_out(x)
        x = self.conv_out(x)
        x = x.permute(0, 4, 1, 2, 3).contiguous()
        return unpatchify(x, patch_size_hw=self.patch_size, patch_size_t=1)

    def forward_diff_step_deferred(
        self,
        x: torch.Tensor,
        stage4_feat: torch.Tensor,
        t: torch.Tensor,
        *,
        drop_leading_frame: bool = True,
    ) -> torch.Tensor:
        """Stage-5 step with deferred context: only ``x`` + low-res ``stage4_feat``.
        Marks T/H/W dynamic on both tensors. CHUNKED blocks upsample then
        ``context_proj`` on the host before attn+mlp; BLACKWELL_DSL
        (``DSLDiffusionBlockChain``) folds that hop into the fused kernel and never
        materialises full-resolution context. ``drop_leading_frame`` must match the
        flag used for this tile's stage-4 path (origin tile vs non-origin).
        """
        t_emb = self.t_embedder(self.timestep_scale_multiplier * t, hidden_dtype=x.dtype)
        modulation = self.shared_adaln(t_emb)

        if self.mark_dynamic_shapes:
            for dim in (1, 2, 3):
                torch._dynamo.mark_dynamic(x, dim)
                torch._dynamo.mark_dynamic(stage4_feat, dim)

        from ltx_core.model.video_vae.transformer.dsl_kernels import DSLDiffusionBlockChain  # noqa: PLC0415

        if isinstance(self.diff_blocks, DSLDiffusionBlockChain):
            # Ping-pong fused launches; same deferred (x, stage4_feat) contract.
            x = self.diff_blocks(x, stage4_feat, modulation, drop_leading_frame=drop_leading_frame)
        else:
            for block in self.diff_blocks:
                x = block.forward_x_ctx(x, stage4_feat, modulation, drop_leading_frame=drop_leading_frame)

        x = self.norm_out(x)
        x = self.conv_out(x)
        x = x.permute(0, 4, 1, 2, 3).contiguous()
        return unpatchify(x, patch_size_hw=self.patch_size, patch_size_t=1)

    def _euler_step(
        self, x_t: torch.Tensor, model_out: torch.Tensor, t_now: torch.Tensor, t_next: torch.Tensor
    ) -> torch.Tensor:
        """One reverse-diffusion Euler update: advance ``x_t`` from ``t_now`` to
        ``t_next`` given the model's prediction at ``t_now``.
        """
        compute_dtype = x_t.dtype
        dt = (t_now - t_next).view(-1, *([1] * (x_t.ndim - 1))).to(torch.float32)
        x_t_fp32 = x_t.to(torch.float32)
        v_pred = model_out if self.model_output_type == "v" else to_velocity(x_t_fp32, t_now, model_out)
        return (x_t_fp32 - dt * v_pred).to(compute_dtype)

    def _decode_one_tile(
        self,
        feat_tile: torch.Tensor,
        x_t_tile_init: torch.Tensor,
        *,
        is_origin: bool,
        timestep: torch.Tensor,
        pad_trailing: bool,
    ) -> torch.Tensor:
        """Run stage 4 + diffusion on one stage-4 feature tile (isolation)."""
        context_tile = self.forward_stage_4(
            feat_tile,
            drop_leading_frame=is_origin,
            pad_trailing=pad_trailing,
        )

        x_t = x_t_tile_init
        _, num_steps = timestep.shape
        for i in range(num_steps - 1):
            t_now = timestep[:, i]
            t_next = timestep[:, i + 1]
            if self.deferred_stage4_upsample:
                x = self._x_for_diff_step(x_t)
                model_out = self.forward_diff_step_deferred(x, context_tile, t_now, drop_leading_frame=is_origin).to(
                    torch.float32
                )
            else:
                context_and_x = self._context_and_x_for_diff_step(context_tile, x_t)
                model_out = self.forward_diff_step(context_and_x, t_now).to(torch.float32)
            x_t = self._euler_step(x_t, model_out, t_now, t_next)

        t_now = timestep[:, -1]
        if self.deferred_stage4_upsample:
            x = self._x_for_diff_step(x_t)
            model_out = self.forward_diff_step_deferred(x, context_tile, t_now, drop_leading_frame=is_origin)
        else:
            context_and_x = self._context_and_x_for_diff_step(context_tile, x_t)
            model_out = self.forward_diff_step(context_and_x, t_now)
        if self.model_output_type == "x0":
            return model_out
        return self._euler_step(x_t, model_out.to(torch.float32), t_now, torch.zeros_like(t_now))

    def _decode_temporal_group_isolated(
        self,
        tiles: List[Tile],
        feat_s4: torch.Tensor,
        content_s4_frames: int,
        x_t_init: torch.Tensor | None,
        timestep: torch.Tensor,
        full_video_shape: VideoLatentShape,
        curr_temporal_slice: slice,
        generator: torch.Generator | None,
        *,
        complementary: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor | None]:
        """Decode every tile of one temporal group in isolation and blend."""
        group_temporal_len = curr_temporal_slice.stop - curr_temporal_slice.start
        group_shape = full_video_shape._replace(frames=group_temporal_len)
        full_torch_shape = full_video_shape.to_torch_shape()
        accum_dtype = torch.float16 if feat_s4.dtype == torch.bfloat16 else feat_s4.dtype
        buffer = torch.zeros(group_shape.to_torch_shape(), device=feat_s4.device, dtype=accum_dtype)
        weights: torch.Tensor | None = None if complementary else torch.zeros_like(buffer)
        local_temporal_slice = slice(0, group_temporal_len)

        compute_dtype = feat_s4.dtype
        randn_device = generator.device if generator is not None else feat_s4.device
        up3_stride = tuple(self.upsamples[3].stride)

        for tile in tiles:
            feat_tile, is_origin, pad_trailing, content_thw = diffusion_tiling.slice_stage4_tile(
                feat_s4, tile, content_frames=content_s4_frames
            )
            content_pixel_shape = diffusion_tiling.pixel_tile_shape(full_torch_shape, tile.out_coords)
            stage5_f, stage5_h, stage5_w = diffusion_tiling.stage5_pixel_shape_from_stage4(
                content_thw[0],
                content_thw[1],
                content_thw[2],
                upsample_stride=up3_stride,  # type: ignore[arg-type]
                patch_size=self.patch_size,
                stage5_kernel_t=self.stage5_kernel[0],
                drop_leading_frame=is_origin,
                pad_trailing=pad_trailing,
            )

            if x_t_init is None:
                x_t_tile_init = torch.randn(
                    (content_pixel_shape[0], content_pixel_shape[1], stage5_f, stage5_h, stage5_w),
                    dtype=compute_dtype,
                    generator=generator,
                    device=randn_device,
                ).to(feat_s4.device)
            else:
                # Expand/crop to stage-5 canvas with the same edge policy as latent
                # size-floor / ghost pad (not fresh noise - NA mixes padded values
                # into kept pixels near the boundary).
                x_t_tile_init = x_t_init[tile.out_coords]
                x_t_tile_init, _ = diffusion_tiling.resize_axis(x_t_tile_init, 2, stage5_f, mode="repeat_last")
                x_t_tile_init, _ = diffusion_tiling.resize_axis(x_t_tile_init, 3, stage5_h, mode="symmetric")
                x_t_tile_init, _ = diffusion_tiling.resize_axis(x_t_tile_init, 4, stage5_w, mode="symmetric")

            pixel_tile = self._decode_one_tile(
                feat_tile,
                x_t_tile_init,
                is_origin=is_origin,
                timestep=timestep,
                pad_trailing=pad_trailing,
            )
            pixel_tile = diffusion_tiling.crop_pixels_to_content(
                pixel_tile,
                content_pixel_shape[2],
                content_pixel_shape[3],
                content_pixel_shape[4],
            ).to(buffer.dtype)

            masks = tuple(m.to(device=buffer.device, dtype=torch.float32) for m in tile.masks_1d)
            local_coords = (
                tile.out_coords[0],
                tile.out_coords[1],
                local_temporal_slice,
                tile.out_coords[3],
                tile.out_coords[4],
            )
            buffer[local_coords] += scale_by_masks_1d(pixel_tile, masks)
            if weights is not None:
                strength = torch.ones(pixel_tile.shape, device=buffer.device, dtype=buffer.dtype)
                weights[local_coords] += scale_by_masks_1d(strength, masks)

        return buffer, weights

    def _decode_pixels(  # noqa: PLR0912, PLR0915
        self,
        latent: torch.Tensor,
        tiling_config: TilingConfig | None = None,
        generator: torch.Generator | None = None,
        *,
        as_fhwc: bool = False,
    ) -> Iterator[torch.Tensor]:
        """Decode latent to pixels, yielding temporal chunks.
        Default yields raw ``(B, C, F, H, W)`` in ``[-1, 1]``. With ``as_fhwc=True``
        (used by :meth:`decode_video`), each chunk is materialized once as
        contiguous ``[F, H, W, C]`` still in ``[-1, 1]`` - layout copy only;
        range mapping stays in ``to_rgb``.
        Stages 1-3 run once on the full volume; stages 4-5 run per tile with
        pixel blend (one tile / one group when untiled or no real split).
        Across temporal groups only the trailing overlap is retained between
        iterations; exclusive frames are yielded before the next group decodes.
        Peak residency is ~two tile extents (current buffer + still-live emit /
        overlap stub), not a single ``tile + overlap`` slab.
        """
        content_shape = VideoLatentShape.from_torch_shape(latent.shape)
        content_pixel = content_shape.upscale(self.video_downscale_factors)._replace(channels=self.out_channels)

        latent, (_t_pad, h_pad, w_pad) = diffusion_tiling.ensure_min_latent_shape(latent, self.stage_min_tile_sizes)
        spatial_scale = (self.video_downscale_factors.height, self.video_downscale_factors.width)
        work_shape = VideoLatentShape.from_torch_shape(latent.shape)
        full_video_shape = work_shape.upscale(self.video_downscale_factors)._replace(channels=self.out_channels)
        target_shape = full_video_shape.to_torch_shape()

        strides = [tuple(u.stride) for u in self.upsamples]
        s4_t, s4_h, s4_w = diffusion_tiling.stage4_thw_from_latent(
            strides, latent.shape[2], latent.shape[3], latent.shape[4], drop_leading_frame=True
        )
        tiles = diffusion_tiling.prepare_tile_schedule(
            torch.Size([latent.shape[0], latent.shape[1], s4_t, s4_h, s4_w]),
            tiling_config,
            upsample3_stride=tuple(self.upsamples[3].stride),  # type: ignore[arg-type]
            patch_size=self.patch_size,
            min_tile_size=self.tile_min_sizes,
            tile_halos=self.tile_halos,
        )

        latent_padded = diffusion_tiling.pad_trailing_latent_for_natten_border(
            latent, self._natten_trailing_pad_latent_frames
        )
        if self.mark_dynamic_shapes:
            for dim in (2, 3, 4):
                torch._dynamo.mark_dynamic(latent_padded, dim)

        feat_s4 = self.forward_stages_1_to_3(latent_padded, drop_leading_frame=True)

        batch = latent.shape[0]
        timestep = self.default_inference_timesteps.to(latent.device).unsqueeze(0).expand(batch, -1)
        single_step_x0 = timestep.shape[1] == 1 and self.model_output_type == "x0"

        x_t_init: torch.Tensor | None = None
        if not single_step_x0:
            compute_dtype = latent.dtype
            randn_device = generator.device if generator is not None else latent.device
            x_t_init = torch.randn(
                tuple(target_shape), dtype=compute_dtype, generator=generator, device=randn_device
            ).to(latent.device)

        complementary = masks_are_complementary(tiles, target_shape)
        groups = group_tiles_by_temporal_slice(tiles)
        group_slices = [slice(*group[0].out_coords[2].indices(target_shape[2])[:2]) for group in groups]

        # Keep only the trailing temporal overlap of the previous group (not the full
        # chunk). Exclusive frames are yielded before the next group is decoded; the
        # consumer may still hold that emit while the next buffer is live (~2x tile).
        overlap_stub: torch.Tensor | None = None
        overlap_stub_weights: torch.Tensor | None = None

        def _finalize(buf: torch.Tensor, wts: torch.Tensor | None) -> torch.Tensor:
            if complementary:
                return buf.to(latent.dtype)
            assert wts is not None
            wts = wts.clamp(min=diffusion_tiling._weight_floor(wts.dtype))
            return (buf / wts).to(latent.dtype)

        def _narrow_content_cfhw(t: torch.Tensor, frames_keep: int) -> torch.Tensor:
            """Spatial/temporal content crop as views (no ``.contiguous()``)."""
            x = t[:, :, :frames_keep]
            th, tw = content_pixel.height, content_pixel.width
            scale_h, scale_w = spatial_scale
            if h_pad is not None:
                before = diffusion_tiling.scale_axis_pad(h_pad, scale_h).before
                x = x.narrow(3, before, th)
            else:
                need = x.shape[3] - th
                if need > 0:
                    x = x.narrow(3, need // 2, th)
                elif need < 0:
                    x, _ = diffusion_tiling.resize_axis(x, 3, th, mode="symmetric")
            if w_pad is not None:
                before = diffusion_tiling.scale_axis_pad(w_pad, scale_w).before
                x = x.narrow(4, before, tw)
            else:
                need = x.shape[4] - tw
                if need > 0:
                    x = x.narrow(4, need // 2, tw)
                elif need < 0:
                    x, _ = diffusion_tiling.resize_axis(x, 4, tw, mode="symmetric")
            return x

        def _crop_emit(buf: torch.Tensor, wts: torch.Tensor | None, global_start: int) -> torch.Tensor | None:
            if global_start >= content_pixel.frames or buf.shape[2] < 1:
                return None
            frames_keep = min(buf.shape[2], content_pixel.frames - global_start)
            if frames_keep < 1:
                return None
            if not as_fhwc:
                chunk = _finalize(buf[:, :, :frames_keep], None if wts is None else wts[:, :, :frames_keep])
                return diffusion_tiling.crop_pixels_to_content(
                    chunk,
                    frames_keep,
                    content_pixel.height,
                    content_pixel.width,
                    h_pad=h_pad,
                    w_pad=w_pad,
                    spatial_scale=spatial_scale,
                )

            # One materialize: contiguous FHWC in latent.dtype, still [-1, 1].
            # CFHW→FHWC cannot be inplace; range mapping is left to to_rgb.
            cfhw = _narrow_content_cfhw(buf, frames_keep)
            src = cfhw[0]  # C, F, H, W (view into accumulator)
            video = torch.empty(
                src.shape[1],
                src.shape[2],
                src.shape[3],
                src.shape[0],
                dtype=latent.dtype,
                device=src.device,
            )
            video.copy_(src.permute(1, 2, 3, 0))
            if not complementary:
                assert wts is not None
                w_cfhw = _narrow_content_cfhw(wts, frames_keep)
                wview = w_cfhw[0].permute(1, 2, 3, 0)
                # Inplace floor on exclusive weight region only (discarded after emit).
                wview.clamp_min_(diffusion_tiling._weight_floor(w_cfhw.dtype))
                video.div_(wview)
            return video

        for gi, group in enumerate(groups):
            curr_temporal_slice = group_slices[gi]
            buffer, weights = self._decode_temporal_group_isolated(
                group,
                feat_s4,
                s4_t,
                x_t_init,
                timestep,
                full_video_shape,
                curr_temporal_slice,
                generator=generator,
                complementary=complementary,
            )

            if overlap_stub is not None:
                overlap_len = int(overlap_stub.shape[2])
                if overlap_len > 0:
                    # Stub is exactly the region overlapping this group (cloned when
                    # the previous group finished); blend then write back into buffer.
                    overlap_stub += buffer[:, :, :overlap_len]
                    if complementary:
                        buffer[:, :, :overlap_len] = overlap_stub
                    else:
                        assert overlap_stub_weights is not None
                        assert weights is not None
                        overlap_stub_weights += weights[:, :, :overlap_len]
                        buffer[:, :, :overlap_len] = overlap_stub
                        weights[:, :, :overlap_len] = overlap_stub_weights
                overlap_stub = None
                overlap_stub_weights = None

            if gi + 1 < len(groups):
                next_start = group_slices[gi + 1].start
                exclusive_len = min(max(0, next_start - curr_temporal_slice.start), buffer.shape[2])
                emitted = _crop_emit(
                    buffer[:, :, :exclusive_len],
                    None if weights is None else weights[:, :, :exclusive_len],
                    curr_temporal_slice.start,
                )
                if emitted is not None:
                    yield emitted
                # Retain only the trailing overlap for the next handoff.
                overlap_stub = buffer[:, :, exclusive_len:].clone()
                if not complementary:
                    assert weights is not None
                    overlap_stub_weights = weights[:, :, exclusive_len:].clone()
                del buffer, weights
            else:
                emitted = _crop_emit(buffer, weights, curr_temporal_slice.start)
                if emitted is not None:
                    yield emitted

    def forward(
        self,
        sample: torch.Tensor,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Decode via ``_decode_pixels`` with ``tiling_config=None`` (single full tile)."""
        return next(self._decode_pixels(sample, tiling_config=None, generator=generator))

    def recommended_tiling_config(
        self,
        *,
        height: int,
        width: int,
        num_frames: int,
        mode: DiffVAEMode = DiffVAEMode.CHUNKED_EAGER,
        free_bytes: int | None = None,
        model_bytes: int | None = None,
    ) -> TileSizeConfig:
        """DiffVAE-aware tiling: stage-4/5 overlaps + memory-capped tile sizes."""
        if free_bytes is None:
            device = next(self.parameters()).device
            free_bytes = cuda_activation_budget_bytes(device) if device.type == "cuda" else 0
        if model_bytes is None:
            model_bytes = sum(p.numel() * p.element_size() for p in self.parameters())
        pixel_scale = diffusion_tiling.stage4_to_pixel_scale_factors(
            tuple(self.upsamples[3].stride),  # type: ignore[arg-type]
            self.patch_size,
        )
        return diffusion_tiling.recommended_decode_tiling_config(
            tile_halos=self.tile_halos,
            pixel_scale=pixel_scale,
            min_tile_size_s4=self.tile_min_sizes,
            patch_size=self.patch_size,
            height=height,
            width=width,
            num_frames=num_frames,
            mode=mode,
            free_bytes=free_bytes,
            model_bytes=model_bytes,
        )

    def tiled_decode(
        self,
        latent: torch.Tensor,
        tiling_config: TilingConfig,
        generator: torch.Generator | None = None,
    ) -> Iterator[torch.Tensor]:
        """Tiled decode: stages 1-3 once, stages 4-5 per tile, pixel blend."""
        yield from self._decode_pixels(latent, tiling_config, generator=generator)

    def decode_video(
        self,
        latent: torch.Tensor,
        tiling_config: TilingConfig | None = None,
        generator: torch.Generator | None = None,
    ) -> Iterator[torch.Tensor]:
        """Decode latent video, yielding float chunk(s) ``[f, h, w, c]`` in ``[0, 1]``.
        Untiled and tiled both go through ``_decode_pixels``. Tiled decode may yield
        multiple times when ``tiling_config.frames`` splits the video.
        Layout is packed once to contiguous FHWC on emit; ``to_rgb`` only does
        inplace ``[-1, 1]→[0, 1]`` (no second realloc).
        """

        def to_rgb(frames: torch.Tensor) -> torch.Tensor:
            return frames.add_(1).mul_(0.5).clamp_(0, 1)

        for chunk in self._decode_pixels(latent, tiling_config, generator=generator, as_fhwc=True):
            yield to_rgb(chunk)
