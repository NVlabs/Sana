"""Diffusion (NATTEN) video VAE decoder."""

from __future__ import annotations

import itertools
import logging
from typing import Iterator, List, Literal, Sequence, Tuple

import torch
from einops import rearrange
from torch import nn

from ltx_core.model.disposable import Disposable
from ltx_core.model.transformer.timestep_embedding import PixArtAlphaCombinedTimestepSizeEmbeddings
from ltx_core.model.video_vae.ops import PerChannelStatistics, patchify, unpatchify
from ltx_core.model.video_vae.tiling import TilingConfig
from ltx_core.model.video_vae.transformer import (
    AdaLNZero,
    ChannelLinear,
    DiffusionNABlock,
    LinearPixelShuffleUpsample,
    NABlock,
)
from ltx_core.model.video_vae.video_vae import VideoDecoder, latent_tile_splitters
from ltx_core.tiling import (
    DEFAULT_SPLIT_OPERATION,
    DimensionInterval,
    Tile,
    compute_trapezoidal_mask_1d,
    group_tiles_by_temporal_slice,
    masks_are_complementary,
    scale_by_masks_1d,
    untiled_mask_1d,
)
from ltx_core.types import SpatioTemporalScaleFactors, VideoLatentShape
from ltx_core.utils import to_denoised, to_velocity

logger: logging.Logger = logging.getLogger(__name__)


# Production CausalVideoAutoencoderL decoder layout: channel width per stage.
# Mirrors the (non-diffusion) NA decoder's stage spec.
_L_STAGE_CHANNELS: Tuple[int, ...] = (1024, 512, 256, 256, 128)
# Doubled-width variant used by some diffusion-decoder checkpoints.
_L_STAGE_CHANNELS_2X: Tuple[int, ...] = tuple(c * 2 for c in _L_STAGE_CHANNELS)
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


def _propagate_interval_through_upsample_hops(
    interval: DimensionInterval,
    strides: Sequence[int],
    causal: bool,
) -> DimensionInterval:
    """Forward-propagate one latent-space ``DimensionInterval`` through a
    sequence of upsample hops on one axis.
    Mirrors ``LinearPixelShuffleUpsample.forward``'s exact index math: a plain
    multiply by ``stride``, plus -- for the causal temporal axis, whenever
    ``stride == 2`` -- the duplicate-frame drop. That drop happens exactly
    once, globally, inside whichever hop's tile owns the true temporal origin
    (``interval.start == 0``, which stays invariant across hops by
    construction: only non-origin intervals' ``start`` is shifted below);
    every other tile's coordinates shift back by 1 at that hop to stay in
    step with it. Left/right ramp widths propagate losslessly (pure multiply)
    since ``LinearPixelShuffleUpsample`` is a per-frame reshape with no
    cross-frame mixing -- confirmed exact against the real module in
    ``test_diffusion_decoder_tiling.py``.
    """
    x = interval
    for stride in strides:
        start = x.start * stride
        end = x.end * stride
        left_ramp = x.left_ramp * stride
        right_ramp = x.right_ramp * stride
        if causal and stride == 2:
            end -= 1
            if x.start != 0:
                start -= 1
        x = DimensionInterval(start=start, end=end, left_ramp=left_ramp, right_ramp=right_ramp)
    return x


def _weight_floor(dtype: torch.dtype) -> float:
    """Smallest divisor that safely guards ``buffer / weights`` in ``dtype``.
    The blend normalizes by an accumulated weight that is ~1 wherever any tile
    covered a voxel, so this only ever has to stop a division by zero. A bare
    ``1e-8`` is not enough once the accumulators are 16-bit: fp16's smallest
    subnormal is ~5.96e-8, so ``clamp(min=1e-8)`` rounds the floor itself to
    zero and silently stops guarding anything. Never go below the dtype's own
    representable floor, but never raise the existing fp32 floor either.
    """
    return max(1e-8, torch.finfo(dtype).tiny)


def _cumulative_upsample_strides(
    upsamples: Sequence[Tuple[Tuple[int, int, int], int]],
) -> List[Tuple[int, int, int]]:
    """Per-axis product of hop strides for ``upsamples[:i]``, one entry per
    stage boundary (``cumulative[0] = (1, 1, 1)`` at stage 1, growing through
    each hop) -- how much a stage-1 latent-grid distance is multiplied by to
    land at stage ``i+1``'s own resolution.
    """
    cumulative = [(1, 1, 1)]
    t, h, w = 1, 1, 1
    for stride, _ in upsamples:
        t, h, w = t * stride[0], h * stride[1], w * stride[2]
        cumulative.append((t, h, w))
    return cumulative


def _all_stages_min_tile_size(
    stage_kernels: Sequence[Tuple[int, int, int]],
    upsamples: Sequence[Tuple[Tuple[int, int, int], int]],
    stage5_kernel: Tuple[int, int, int],
) -> Tuple[int, int, int]:
    """Per-axis latent-grid floor so every stage's NA sees dims ``>= kernel_size``.
    A stage-1 tile of length ``S`` lands at stage ``i`` with length
    ``S * cumulative_stride[i]`` (ignoring the causal temporal drop of 1, which
    only shrinks an already-large extent). Requiring
    ``S >= ceil(kernel_axis_i / stride_to_stage_i)`` for every stage keeps
    ``natten.na3d`` from rejecting undersized remnant tiles.
    """
    cumulative = _cumulative_upsample_strides(upsamples)
    mins = [1, 1, 1]
    for stage_i in range(len(upsamples)):
        strides = cumulative[stage_i]
        for axis in range(3):
            mins[axis] = max(mins[axis], -(-stage_kernels[stage_i][axis] // strides[axis]))
    strides5 = cumulative[len(upsamples)]
    for axis in range(3):
        mins[axis] = max(mins[axis], -(-stage5_kernel[axis] // strides5[axis]))
    return (mins[0], mins[1], mins[2])


def _all_stages_halo(
    stage_kernels: Sequence[Tuple[int, int, int]],
    stage_depths: Sequence[int],
    upsamples: Sequence[Tuple[Tuple[int, int, int], int]],
    stage5_kernel: Tuple[int, int, int],
) -> Tuple[Tuple[int, int, int], ...]:
    """Per-stage (all 5: det stages 1-4, then stage 5), one-sided halo in
    *latent* (stage-1) grid units that a tile boundary chosen at stage 1
    needs as overlap so that stage's own ``NeighborhoodAttention3D``
    boundary effect is fully covered once propagated forward.
    Each stage's halo is ``depth_i * (kernel_axis_i // 2)`` at *that*
    stage's own resolution, converted back to latent-grid units by dividing
    by the cumulative upstream stride to that stage (stage 1 itself gets no
    such shrinkage, so it dominates in practice). ``ceil`` (not floor)
    division, since a fractional latent-frame requirement must round up to
    still cover the halo.
    Returns one ``(T,H,W)`` tuple per stage (index 0..3 = det stages 1-4,
    index 4 = stage 5) rather than a single combined value, so a caller
    enforcing a tiling minimum can report *which* stage is the bottleneck --
    stage 1 dominates for typical depth/kernel choices, but this makes that
    an observed fact, not an assumption baked into the return type.
    Stage 5's own kernel/depth (``stage5_kernel``, ``stage_depths[-1]``) are
    tracked separately from the det stages' ``stage_kernels``/``stage_depths``
    entries, matching ``DiffusionVideoDecoder.__init__``'s own convention
    (``stage_kernels[-1]`` is never actually used there either -- stage 5's
    kernel always comes from the dedicated ``stage5_kernel`` parameter).
    """
    cumulative = _cumulative_upsample_strides(upsamples)
    per_stage: List[Tuple[int, int, int]] = []
    for stage_i in range(len(upsamples)):
        stride_to_stage = cumulative[stage_i]
        per_stage.append(
            tuple(
                -(-(stage_depths[stage_i] * (stage_kernels[stage_i][axis] // 2)) // stride_to_stage[axis])
                for axis in range(3)
            )
        )
    stage5_depth = stage_depths[-1]
    stride_to_stage5 = cumulative[-1]
    per_stage.append(
        tuple(-(-(stage5_depth * (stage5_kernel[axis] // 2)) // stride_to_stage5[axis]) for axis in range(3))
    )
    return tuple(per_stage)


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
    cropping that appendix from context before stage 5 (see
    ``forward_pre_diffusion``).
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
        strict_tiling_overlap: bool = False,
    ) -> None:
        super().__init__()
        assert len(stage_channels) == len(stage_depths) == len(stage_kernels)
        assert len(upsamples) == len(stage_channels) - 1
        for c in stage_channels:
            assert c % head_dim == 0, f"stage_channels {stage_channels} must each be a multiple of head_dim={head_dim}"

        # Local import: video_vae <-> transformer package boundary.
        from ltx_core.model.video_vae.transformer.attention import natten_available  # noqa: PLC0415

        if not natten_available():
            raise ImportError(
                "DiffusionVideoDecoder requires natten (NeighborhoodAttention3D uses natten.na3d). "
                "Install with: uv sync --package ltx-core --extra natten "
                '(or: uv pip install "natten==0.21.5+torch290cu128" -f https://whl.natten.org)'
            )

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
        # NATTEN last-frame border workaround: replicate last latent frame
        # ``(K_t // 2) * 2`` times through stages 1-4 only, then crop the
        # appendix off context before stage 5. Moves the window-shift border
        # past kept frames without changing attention kernels.
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

        # Per-stage, one-sided halo (all 5 stages, in *latent*-grid units --
        # see _all_stages_halo's docstring) that a tile boundary chosen at
        # stage 1 needs to overlap so every stage's own NeighborhoodAttention3D
        # boundary window-shift (see its docstring) is fully covered once
        # propagated forward: positions within this margin of a *tile* edge
        # diverge from the full-tensor computation, and that divergence
        # compounds linearly with block depth. The det-stage entries are
        # empirically confirmed exact (tight, not just an upper bound) for
        # NABlock in test_diffusion_decoder_tiling.py.
        self.stage_halos: Tuple[Tuple[int, int, int], ...] = _all_stages_halo(
            stage_kernels, stage_depths, upsamples, stage5_kernel
        )
        # Latent-grid (T,H,W) floor for tile splitters — max over stages of
        # ceil(kernel / stride_to_stage) so remnant tiles never undershoot NA.
        self.stage_min_tile_sizes: Tuple[int, int, int] = _all_stages_min_tile_size(
            stage_kernels, upsamples, stage5_kernel
        )
        # If True, _validate_min_overlap raises instead of just warning when
        # tiling_config's overlap is below what stage kernels recommend --
        # the requested overlap is always honored as-is either way, this
        # only controls whether an insufficient one is treated as fatal.
        self.strict_tiling_overlap = strict_tiling_overlap

        self.conv_in_x_t = ChannelLinear(noised_pixel_channels, c5, bias=True)

        # Shared AdaLN-Zero (7-chunk for shape compat; gate slots unused in block).
        self.shared_adaln = AdaLNZero(dim=c5, t_emb_dim=t_emb_dim)

        self.diff_blocks = nn.ModuleList(
            [
                DiffusionNABlock(
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
        # NOTE: the four call sites use maybe_mark_dynamic, not mark_dynamic:
        # the decoder's attention block specialises T/H/W, and the strict form
        # turns that into a ConstraintViolationError that kills --vae-compile.
        self.mark_dynamic_shapes = False

    def _pad_trailing_latent_for_natten_border(self, latent: torch.Tensor) -> torch.Tensor:
        """Replicate the last latent frame ``_natten_trailing_pad_latent_frames`` times."""
        n = self._natten_trailing_pad_latent_frames
        if n <= 0:
            return latent
        return torch.cat([latent, latent[:, :, -1:].expand(-1, -1, n, -1, -1)], dim=2)

    def _crop_trailing_context_natten_pad(self, context: torch.Tensor) -> torch.Tensor:
        """Drop the stage-5-resolution appendix produced by the stage-1 latent pad."""
        n = self._natten_trailing_pad_latent_frames
        if n <= 0:
            return context
        crop = n * self.video_downscale_factors.time
        return context[:, :-crop]

    def forward_pre_diffusion(
        self,
        z_noisy: torch.Tensor,
        drop_leading_frame: bool = True,
        pad_trailing: bool = True,
    ) -> torch.Tensor:
        """Stages 1..4: produce the latent context at stage-5 spatial resolution.
        Deterministic NA upsample. Output is (B, T_5, H_5, W_5, C_5) channels-last.
        ``drop_leading_frame`` must be ``True`` only when ``z_noisy`` contains the
        latent's true temporal origin (t=0). Tiled callers decoding a later
        temporal chunk in isolation must pass ``False`` -- see
        ``LinearPixelShuffleUpsample.forward``. This is the only thing that
        depends on this tile's position in the full latent: RoPE itself needs
        no absolute origin (see ``rope.py``'s module docstring), so no offset
        is threaded through the block calls below.
        ``pad_trailing``: when True (default; always for untiled decode,
        and for tiles that include the full latent's last frame), append a
        short replicate of the last latent frame before stages 1-4 and crop it
        off the returned context so stage 5 never sees the NATTEN border.
        Non-trailing temporal tiles must pass False.
        """
        if pad_trailing:
            z_noisy = self._pad_trailing_latent_for_natten_border(z_noisy)
        z_noisy = self.per_channel_statistics.un_normalize(z_noisy)
        x = z_noisy.permute(0, 2, 3, 4, 1)
        x = self.conv_in(x)
        for stage_i, blocks in enumerate(self.det_stages):
            # compile_diffusion_decoder compiles each block individually (not
            # this loop), so a fresh eager tensor is what a compiled block
            # actually sees -- mark it dynamic here, every call, same as
            # _decode_tile_isolated does for its own per-tile tensors.
            # Shape is constant across a stage's blocks, so this only costs a
            # real recompile once per stage (4 total), not once per call.
            if self.mark_dynamic_shapes:
                for dim in (1, 2, 3):
                    torch._dynamo.maybe_mark_dynamic(x, dim)
            for block in blocks:
                x = block(x)
            x = self.upsamples[stage_i](x, drop_leading_frame=drop_leading_frame)
        if pad_trailing:
            x = self._crop_trailing_context_natten_pad(x)
        return x

    def _combined_for_diff_step(self, context: torch.Tensor, x_t: torch.Tensor) -> torch.Tensor:
        """Build block-ready ``[context | conv_in_x_t(patched x)]`` for ``forward_diff_step``."""
        noised_pixels_patched = patchify(x_t, patch_size_hw=self.patch_size, patch_size_t=1)
        x = self.conv_in_x_t(noised_pixels_patched.permute(0, 2, 3, 4, 1))
        return torch.cat([context, x], dim=-1)

    def forward_diff_step(
        self,
        combined: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """One stage-5 diffusion step. Returns the model prediction in pixel space.
        ``combined`` is ``[latent_context | conv_in_x_t(x)]`` (channels-last), built
        at the call site via ``_combined_for_diff_step``. That single buffer is
        reused across ``diff_blocks``: each block writes its output x-half back
        with ``copy_`` (no per-block ``cat``). One-tensor layout keeps Dynamo
        T/H/W symbols identical under ``mark_dynamic``.
        """
        x_half = combined[..., self.context_channels :]
        t_emb = self.t_embedder(self.timestep_scale_multiplier * t, hidden_dtype=x_half.dtype)
        modulation = self.shared_adaln(t_emb)

        if self.mark_dynamic_shapes:
            for dim in (1, 2, 3):
                torch._dynamo.maybe_mark_dynamic(combined, dim)

        for block in self.diff_blocks:
            x_half.copy_(block.forward_combined(combined, modulation))
        x = x_half

        x = self.norm_out(x)
        x = self.conv_out(x)
        x = x.permute(0, 4, 1, 2, 3).contiguous()
        return unpatchify(x, patch_size_hw=self.patch_size, patch_size_t=1)

    def _validate_min_overlap(self, tiling_config: TilingConfig, *, strict: bool | None = None) -> TilingConfig:
        """Warn (or, if ``strict``, raise) when ``tiling_config``'s overlap is
        below what ANY of the 5 stages' own ``NeighborhoodAttention3D``
        boundary effects recommend (``self.stage_halos``) -- below this, a
        tile edge is measurably wrong once propagated to that stage, not
        just lower-quality.
        Returns ``tiling_config`` **unchanged**: the caller's requested
        overlap is honored exactly, at whatever tile boundaries/memory
        footprint that implies -- this deliberately does NOT silently widen
        it to the recommended minimum. Silently substituting a larger,
        decoder-derived overlap changes the effective tile size ratio (and
        therefore both memory and, since stage 5's tiling reruns every Euler
        step, runtime) in a way the caller never asked for and has no way to
        see except by noticing the change in behavior.
        One combined computation across all 5 stages (not two independent
        ones per stage group), so this can report *which* stage is actually
        the bottleneck instead of just a bare number: kernel half-width at
        that stage's own resolution, converted to pixel/frame units, next to
        what was actually requested.
        ``strict`` (defaults to ``self.strict_tiling_overlap``): if ``True``,
        raise instead of warning -- for callers that want an insufficient
        overlap treated as an error, not just a logged recommendation.
        """
        if strict is None:
            strict = self.strict_tiling_overlap
        stage_names = [f"stage {i + 1}" for i in range(len(self.stage_halos))]
        scale = self.video_downscale_factors

        def round_up(value: int, multiple: int) -> int:
            return -(-value // multiple) * multiple

        def dominant(axis: int) -> Tuple[int, str]:
            halo, name = max((self.stage_halos[i][axis], stage_names[i]) for i in range(len(self.stage_halos)))
            return halo, name

        temporal_config = tiling_config.temporal_config
        if temporal_config is not None:
            halo_t, name_t = dominant(0)
            recommended = round_up(halo_t * scale.time, 8)
            if temporal_config.tile_overlap_in_frames < recommended:
                message = (
                    f"Temporal overlap {temporal_config.tile_overlap_in_frames} frames is below the "
                    f"recommended {recommended} frames: {name_t}'s kernel half-width is {halo_t} latent "
                    f"frame(s) there, which is {recommended} pixel frames after upscaling (x{scale.time}). "
                    f"Proceeding with the requested {temporal_config.tile_overlap_in_frames} frames -- "
                    "tile edges near that stage's boundary will diverge from the untiled reference."
                )
                if strict:
                    raise ValueError(message)
                logger.warning(message)

        spatial_config = tiling_config.spatial_config
        if spatial_config is not None:
            halo_h, name_h = dominant(1)
            halo_w, name_w = dominant(2)
            halo_hw, name_hw = (halo_h, name_h) if halo_h >= halo_w else (halo_w, name_w)
            recommended_px = round_up(halo_hw * scale.height, 32)
            if spatial_config.tile_overlap_in_pixels < recommended_px:
                message = (
                    f"Spatial overlap {spatial_config.tile_overlap_in_pixels}px is below the recommended "
                    f"{recommended_px}px: {name_hw}'s kernel half-width is {halo_hw} latent pixel(s) there, "
                    f"which is {recommended_px}px after upscaling (x{scale.height}). Proceeding with the "
                    f"requested {spatial_config.tile_overlap_in_pixels}px -- tile edges near that stage's "
                    "boundary will diverge from the untiled reference."
                )
                if strict:
                    raise ValueError(message)
                logger.warning(message)

        return tiling_config

    def _prepare_tile_schedule(
        self,
        latent_shape: torch.Size,
        tiling_config: TilingConfig,
    ) -> Tuple[List[Tile], List[Tile]]:
        """Compute stage 1-4 tiles AND stage 5 tiles together, as one tiling
        schedule spanning latent -> stage-5 -> pixel resolution, from a
        single, combined receptive-field overlap enforcement
        (``_validate_min_overlap``, which considers all 5 stages at once).
        Replaces what used to be two fully independent methods
        (``_prepare_stage1_tiles``/``_prepare_stage5_tiles``), each calling
        ``latent_tile_splitters`` and enforcing its own, separately-derived
        overlap minimum -- which could produce inconsistent tile boundaries
        between the two (different bump amounts from different halo
        sources) for the same ``tiling_config``, and gave no single place to
        reason about "how does this latent get carved up end to end". Now
        computed once, per axis, in one pass through latent -> stage-5 ->
        pixel; stage 1-4's tiles and stage 5's tiles are two views built
        from that one pass, not two independent computations.
        Returns ``(pre_diffusion_tiles, diffusion_tiles)`` -- named after
        what consumes them (``forward_pre_diffusion``/``forward_diff_step``),
        not their stage index, since "stage 1" / "stage 5" are internal
        implementation details callers of this tiling schedule shouldn't
        need to know:
        - ``pre_diffusion_tiles``: for ``forward_pre_diffusion``. ``in_coords``
          land at latent (stage-1) grid resolution, in the latent's own real
          ``(B,C,T,H,W)`` layout -- usable directly as
          ``latent[tile.in_coords]``. ``out_coords`` land at stage-5 grid
          resolution, in the context accumulation buffer's own real
          channels-last ``(B,T,H,W,C)`` layout -- usable directly as
          ``buffer[tile.out_coords]``.
        - ``diffusion_tiles``: for ``forward_diff_step``. ``in_coords`` land
          at stage-5 grid resolution (T matches pixel T since ``patch_size_t=1``;
          H/W are pre-unpatchify), for slicing ``latent_context`` /
          projected ``x``. ``out_coords`` land at final pixel resolution, for
          writing into the output buffer.
        Note ``pre_diffusion_tiles``' in/out axis orders genuinely differ
        (latent is channels-first, the context buffer is channels-last)
        while ``diffusion_tiles``' in/out share one generic
        ``(B,C,T,H,W)``-shaped order -- each is only ever consumed by
        directly indexing a tensor of the matching layout, never by
        cross-layout reuse.
        """
        tiling_config = self._validate_min_overlap(tiling_config)
        splitters = latent_tile_splitters(
            latent_shape,
            tiling_config,
            self.video_downscale_factors,
            min_tile_size=self.stage_min_tile_sizes,
        )
        na_strides = [tuple(u.stride) for u in self.upsamples]  # 4 hops, each a (T, H, W) stride tuple

        def axis_specs(
            axis_idx: int, stride_component: int, causal: bool
        ) -> list[tuple[slice, slice, slice, torch.Tensor, torch.Tensor]]:
            if splitters[axis_idx] is DEFAULT_SPLIT_OPERATION:
                return [(slice(None), slice(None), slice(None), untiled_mask_1d(), untiled_mask_1d())]
            intervals = splitters[axis_idx](latent_shape[axis_idx]).intervals
            strides = [s[stride_component] for s in na_strides]
            specs = []
            for iv in intervals:
                stage5 = _propagate_interval_through_upsample_hops(iv, strides, causal)
                if axis_idx == 2:  # T: patch_size_t=1, so stage-5 T *is* pixel T.
                    pixel = stage5
                else:  # H/W: one more exact x patch_size hop to reach pixel space.
                    pixel = _propagate_interval_through_upsample_hops(stage5, [self.patch_size], causal=False)
                mask_stage5 = compute_trapezoidal_mask_1d(
                    stage5.end - stage5.start, stage5.left_ramp, stage5.right_ramp, left_starts_from_0=causal
                )
                mask_pixel = compute_trapezoidal_mask_1d(
                    pixel.end - pixel.start, pixel.left_ramp, pixel.right_ramp, left_starts_from_0=causal
                )
                specs.append(
                    (
                        slice(iv.start, iv.end),
                        slice(stage5.start, stage5.end),
                        slice(pixel.start, pixel.end),
                        mask_stage5,
                        mask_pixel,
                    )
                )
            return specs

        t_specs = axis_specs(2, 0, True)
        h_specs = axis_specs(3, 1, False)
        w_specs = axis_specs(4, 2, False)

        pre_diffusion_tiles: List[Tile] = []
        diffusion_tiles: List[Tile] = []
        for t_spec, h_spec, w_spec in itertools.product(t_specs, h_specs, w_specs):
            t_lat, t_s5, t_px, t_mask5, t_maskpx = t_spec
            h_lat, h_s5, h_px, h_mask5, h_maskpx = h_spec
            w_lat, w_s5, w_px, w_mask5, w_maskpx = w_spec

            pre_diffusion_tiles.append(
                Tile(
                    in_coords=(slice(None), slice(None), t_lat, h_lat, w_lat),
                    out_coords=(slice(None), t_s5, h_s5, w_s5, slice(None)),
                    masks_1d=(untiled_mask_1d(), t_mask5, h_mask5, w_mask5, untiled_mask_1d()),
                )
            )
            diffusion_tiles.append(
                Tile(
                    in_coords=(slice(None), slice(None), t_s5, h_s5, w_s5),
                    out_coords=(slice(None), slice(None), t_px, h_px, w_px),
                    masks_1d=(untiled_mask_1d(), untiled_mask_1d(), t_maskpx, h_maskpx, w_maskpx),
                )
            )
        return pre_diffusion_tiles, diffusion_tiles

    @staticmethod
    def _tile_is_origin(tile: Tile) -> bool:
        """Whether ``tile`` contains the latent's true temporal origin (t=0) --
        only that tile may drop its leading frame in ``forward_pre_diffusion``'s
        upsample hops; see ``LinearPixelShuffleUpsample.forward``.
        """
        return tile.in_coords[2].start in (0, None)

    @staticmethod
    def _tile_includes_temporal_end(tile: Tile, latent_frames: int) -> bool:
        """Whether ``tile`` includes the full latent's last frame -- only those
        tiles need the trailing NATTEN border pad in ``forward_pre_diffusion``.
        """
        _, stop, _ = tile.in_coords[2].indices(latent_frames)
        return stop == latent_frames

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

    def _decode_tile_isolated(
        self,
        latent_tile: torch.Tensor,
        x_t_tile_init: torch.Tensor,
        is_origin: bool,
        timestep: torch.Tensor,
        pad_trailing: bool = True,
    ) -> torch.Tensor:
        """Run stages 1-4 once, then the diffusion step(s), for exactly one tile.
        Fully isolated from every other tile: this tile's context and ``x_t`` never
        touch a shared buffer and are discarded before the next tile starts, so
        peak activation memory stays tile-bounded.
        Single-step ``x0`` checkpoints skip Euler: one ``forward_diff_step`` and
        return the prediction (same algebra as untiled ``forward``). Multi-step
        keeps the Euler loop.
        """
        context_tile = self.forward_pre_diffusion(
            latent_tile,
            drop_leading_frame=is_origin,
            pad_trailing=pad_trailing,
        )
        if self.mark_dynamic_shapes:
            for dim in (1, 2, 3):
                torch._dynamo.maybe_mark_dynamic(context_tile, dim)

        _, num_steps = timestep.shape
        if num_steps == 1 and self.model_output_type == "x0":
            t_now = timestep[:, 0]
            combined = self._combined_for_diff_step(context_tile, x_t_tile_init)
            if self.mark_dynamic_shapes:
                for dim in (1, 2, 3):
                    torch._dynamo.maybe_mark_dynamic(combined, dim)
            return self.forward_diff_step(combined, t_now)

        x_t = x_t_tile_init
        for i in range(num_steps):
            t_now = timestep[:, i]
            t_next = timestep[:, i + 1] if i + 1 < num_steps else torch.zeros_like(t_now)
            combined = self._combined_for_diff_step(context_tile, x_t)
            if self.mark_dynamic_shapes:
                for dim in (1, 2, 3):
                    torch._dynamo.maybe_mark_dynamic(combined, dim)
            model_out = self.forward_diff_step(combined, t_now).to(torch.float32)
            x_t = self._euler_step(x_t, model_out, t_now, t_next)
        return x_t

    def forward(
        self,
        sample: torch.Tensor,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Single-step decode from noise (no Euler loop).
        Samples ``x_t ~ N(0, I)`` at full pixel resolution, runs stages 1-4 once,
        then one stage-5 step at the first (only) default timestep. Returns raw
        pixels ``(B, C, F, H, W)`` in the diffusion range (not ``[0, 1]`` RGB).
        Used by ``decode_video`` when untiled and ``num_steps == 1`` (bundled n1
        DiffVAE). Multi-step untiled decode keeps ``_decode_video_loop``.
        """
        latent_shape = VideoLatentShape.from_torch_shape(sample.shape)
        full_shape = latent_shape.upscale(self.video_downscale_factors)._replace(channels=self.out_channels)
        target_shape = full_shape.to_torch_shape()

        batch = sample.shape[0]
        t_now = self.default_inference_timesteps.to(sample.device)[:1].expand(batch)

        compute_dtype = sample.dtype
        randn_device = generator.device if generator is not None else sample.device
        x_t = torch.randn(tuple(target_shape), dtype=compute_dtype, generator=generator, device=randn_device).to(
            sample.device
        )

        if self.mark_dynamic_shapes:
            for dim in (2, 3, 4):
                torch._dynamo.maybe_mark_dynamic(sample, dim)

        context = self.forward_pre_diffusion(sample)
        combined = self._combined_for_diff_step(context, x_t)
        if self.mark_dynamic_shapes:
            for dim in (1, 2, 3):
                torch._dynamo.maybe_mark_dynamic(combined, dim)
        model_out = self.forward_diff_step(combined, t_now)

        if self.model_output_type == "x0":
            return model_out
        return to_denoised(x_t, model_out, t_now)

    @staticmethod
    def _pixel_tile_shape(full_shape: tuple[int, ...], out_coords: tuple[slice, ...]) -> tuple[int, ...]:
        """Resolve ``(B, C, F, H, W)`` for a tile from full-video shape + out slices."""
        dims: list[int] = []
        for size, coord in zip(full_shape, out_coords, strict=True):
            start, stop, step = coord.indices(size)
            dims.append(len(range(start, stop, step)))
        return tuple(dims)

    def _decode_temporal_group_isolated(
        self,
        pre_group: List[Tile],
        diff_group: List[Tile],
        latent: torch.Tensor,
        x_t_init: torch.Tensor | None,
        timestep: torch.Tensor,
        full_video_shape: VideoLatentShape,
        curr_temporal_slice: slice,
        generator: torch.Generator | None = None,
        *,
        complementary: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor | None]:
        """Decode every tile of one temporal group in isolation
        (``_decode_tile_isolated``) and blend into a group-sized (not
        full-video-sized) buffer, rebased so the group starts at local index 0.
        ``x_t_init`` is the full-video noise buffer for multi-step Euler. For
        single-step ``x0`` it is ``None``: noise is sampled per tile so peak
        VRAM does not keep a full-canvas ``noisy_x``.
        Returns ``(buffer, weights)`` -- unnormalized, for the caller to
        blend against a neighboring group's carry-over before normalizing.
        ``weights`` is ``None`` when ``complementary``.
        """
        group_temporal_len = curr_temporal_slice.stop - curr_temporal_slice.start
        group_shape = full_video_shape._replace(frames=group_temporal_len)
        full_torch_shape = full_video_shape.to_torch_shape()
        # ``buffer``/``weights`` are full spatial resolution and tiling never shrinks
        # them; with the caller's carry-over pair that is four live group-sized
        # tensors, so fp32 cost ~3.6 GB extra at 1088x1920 just to sum a handful of
        # masked tiles per voxel.
        # Accumulate in the latent's dtype, except bf16 which is promoted to fp16 --
        # same 2 bytes, 3 more mantissa bits. The promotion is what makes 16-bit
        # accumulation viable: error compounds across the masked adds before the
        # divide, and for a voxel covered by 8 tiles that measures 0.50 LSB of 8-bit
        # output in fp16 versus 1.99 LSB in bf16. fp16 is therefore indistinguishable
        # from the fp32 this replaces, while bf16 would put visible error in exactly
        # the overlap seams the blend exists to hide. fp16's narrower exponent range
        # is not a concern: values are pixels in [-1, 1] and weights summing to ~1.
        accum_dtype = torch.float16 if latent.dtype == torch.bfloat16 else latent.dtype
        buffer = torch.zeros(group_shape.to_torch_shape(), device=latent.device, dtype=accum_dtype)
        weights: torch.Tensor | None = None if complementary else torch.zeros_like(buffer)
        # Every tile in a group shares the same temporal out_coords by
        # construction (that's what group_tiles_by_temporal_slice groups on),
        # so the group-local temporal slice is always (0, group_temporal_len)
        # -- never re-derived per tile from diff_tile.out_coords[2], which can
        # be slice(None) (not a concrete start/stop) when temporal_config is
        # unset.
        local_temporal_slice = slice(0, group_temporal_len)

        compute_dtype = latent.dtype
        randn_device = generator.device if generator is not None else latent.device

        for pre_tile, diff_tile in zip(pre_group, diff_group, strict=True):
            is_origin = self._tile_is_origin(pre_tile)
            pad_trailing = self._tile_includes_temporal_end(pre_tile, latent.shape[2])
            if x_t_init is None:
                tile_shape = self._pixel_tile_shape(full_torch_shape, diff_tile.out_coords)
                x_t_tile_init = torch.randn(
                    tile_shape, dtype=compute_dtype, generator=generator, device=randn_device
                ).to(latent.device)
            else:
                x_t_tile_init = x_t_init[diff_tile.out_coords]

            pixel_tile = self._decode_tile_isolated(
                latent[pre_tile.in_coords],
                x_t_tile_init,
                is_origin,
                timestep,
                pad_trailing=pad_trailing,
            ).to(buffer.dtype)

            # Float32 masks so bf16/fp16 pixels promote on the separable multiply.
            masks = tuple(m.to(device=buffer.device, dtype=torch.float32) for m in diff_tile.masks_1d)
            local_coords = (
                diff_tile.out_coords[0],
                diff_tile.out_coords[1],
                local_temporal_slice,
                diff_tile.out_coords[3],
                diff_tile.out_coords[4],
            )
            buffer[local_coords] += scale_by_masks_1d(pixel_tile, masks)
            if weights is not None:
                strength = torch.ones(pixel_tile.shape, device=buffer.device, dtype=buffer.dtype)
                weights[local_coords] += scale_by_masks_1d(strength, masks)

        return buffer, weights

    def tiled_decode(  # noqa: PLR0912, PLR0915
        self,
        latent: torch.Tensor,
        tiling_config: TilingConfig,
        generator: torch.Generator | None = None,
    ) -> Iterator[torch.Tensor]:
        """Tiled decode: each tile runs stages 1-4 + diffusion in isolation
        (``_decode_tile_isolated``), then blends once at the end.
        Multi-step checkpoints keep a full-canvas ``noisy_x`` for the Euler
        trajectory. Single-step ``x0`` does **not** -- noise is sampled per
        tile only (saves ~0.5-1.5 GiB of resident VRAM vs the full-canvas
        buffer).
        Grouped and progressively yielded by temporal slice like
        ``ConvVideoDecoder.tiled_decode``. Yields raw chunks ``(B, C, F, H, W)``;
        ``decode_video`` applies the ``[0, 1]`` RGB mapping.
        """
        latent_shape = VideoLatentShape.from_torch_shape(latent.shape)
        full_video_shape = latent_shape.upscale(self.video_downscale_factors)._replace(channels=self.out_channels)
        target_shape = full_video_shape.to_torch_shape()

        batch = latent.shape[0]
        timestep = self.default_inference_timesteps.to(latent.device).unsqueeze(0).expand(batch, -1)
        single_step_x0 = timestep.shape[1] == 1 and self.model_output_type == "x0"

        # Full-canvas noisy_x only for multi-step Euler. N=1 x0 samples per tile.
        x_t_init: torch.Tensor | None = None
        if not single_step_x0:
            compute_dtype = latent.dtype
            randn_device = generator.device if generator is not None else latent.device
            x_t_init = torch.randn(
                tuple(target_shape), dtype=compute_dtype, generator=generator, device=randn_device
            ).to(latent.device)

        if self.mark_dynamic_shapes:
            for dim in (2, 3, 4):
                torch._dynamo.maybe_mark_dynamic(latent, dim)

        pre_diffusion_tiles, diffusion_tiles = self._prepare_tile_schedule(latent.shape, tiling_config)
        complementary = masks_are_complementary(diffusion_tiles, target_shape)
        diff_groups = group_tiles_by_temporal_slice(diffusion_tiles)
        pre_groups: List[List[Tile]] = []
        idx = 0
        for group in diff_groups:
            pre_groups.append(pre_diffusion_tiles[idx : idx + len(group)])
            idx += len(group)

        previous_chunk: torch.Tensor | None = None
        previous_weights: torch.Tensor | None = None
        previous_temporal_slice: slice | None = None

        for pre_group, diff_group in zip(pre_groups, diff_groups, strict=True):
            # .indices(...) normalizes slice(None) (no temporal_config, so no
            # real split -- a single group spanning the whole video) into a
            # concrete (start, stop), which the blend/rebase arithmetic below
            # needs; group_tiles_by_temporal_slice's equality-based grouping
            # already works fine on slice(None) since slice(None) == slice(None).
            start, stop, _ = diff_group[0].out_coords[2].indices(target_shape[2])
            curr_temporal_slice = slice(start, stop)
            buffer, weights = self._decode_temporal_group_isolated(
                pre_group,
                diff_group,
                latent,
                x_t_init,
                timestep,
                full_video_shape,
                curr_temporal_slice,
                generator=generator,
                complementary=complementary,
            )

            if previous_chunk is not None:
                assert previous_temporal_slice is not None
                if previous_temporal_slice.stop > curr_temporal_slice.start:
                    overlap_len = previous_temporal_slice.stop - curr_temporal_slice.start
                    temporal_overlap_slice = slice(curr_temporal_slice.start - previous_temporal_slice.start, None)
                    previous_chunk[:, :, temporal_overlap_slice] += buffer[:, :, :overlap_len]
                    if complementary:
                        buffer[:, :, :overlap_len] = previous_chunk[:, :, temporal_overlap_slice]
                    else:
                        assert previous_weights is not None
                        assert weights is not None
                        previous_weights[:, :, temporal_overlap_slice] += weights[:, :, :overlap_len]
                        buffer[:, :, :overlap_len] = previous_chunk[:, :, temporal_overlap_slice]
                        weights[:, :, :overlap_len] = previous_weights[:, :, temporal_overlap_slice]

                yield_len = curr_temporal_slice.start - previous_temporal_slice.start
                if complementary:
                    yield previous_chunk[:, :, :yield_len].to(latent.dtype)
                else:
                    assert previous_weights is not None
                    previous_weights = previous_weights.clamp(min=_weight_floor(previous_weights.dtype))
                    # Back to the latent's dtype: the 16-bit accumulators are an internal
                    # memory optimization, so yield what the untiled path and ConvVideoDecoder
                    # yield rather than leaking the accumulator dtype to callers.
                    yield (previous_chunk / previous_weights)[:, :, :yield_len].to(latent.dtype)

            previous_chunk = buffer
            previous_weights = weights
            previous_temporal_slice = curr_temporal_slice

        if previous_chunk is not None:
            if complementary:
                yield previous_chunk.to(latent.dtype)
            else:
                assert previous_weights is not None
                previous_weights = previous_weights.clamp(min=_weight_floor(previous_weights.dtype))
                yield (previous_chunk / previous_weights).to(latent.dtype)

    def decode_video(
        self,
        latent: torch.Tensor,
        tiling_config: TilingConfig | None = None,
        generator: torch.Generator | None = None,
    ) -> Iterator[torch.Tensor]:
        """Decode latent video, yielding float chunk(s) ``[f, h, w, c]`` in ``[0, 1]``.
        Untiled + single default step calls ``forward`` (no Euler loop), mirroring
        ``ConvVideoDecoder.decode_video`` → ``self(...)``. Untiled multi-step uses
        ``_decode_video_loop``. Tiled decode (``tiling_config`` set) delegates to
        ``tiled_decode``, which may yield multiple times -- one per temporal group
        -- when ``tiling_config.temporal_config`` splits the video; callers must
        drain the full iterator. If ``compile_diffusion_decoder`` was applied,
        ``mark_dynamic_shapes`` is already set and T/H/W are marked dynamically.
        """

        def to_rgb(frames: torch.Tensor) -> torch.Tensor:
            video = rearrange(frames[0], "c f h w -> f h w c")
            return video.add(1.0).mul(0.5).clamp(0.0, 1.0)

        if tiling_config is not None:
            for chunk in self.tiled_decode(latent, tiling_config, generator=generator):
                yield to_rgb(chunk)
            return
        if self.default_inference_timesteps.numel() == 1:
            yield to_rgb(self(latent, generator=generator))
            return
        yield to_rgb(self._decode_video_loop(latent, generator=generator))

    def _decode_video_loop(
        self,
        latent: torch.Tensor,
        generator: torch.Generator | None,
    ) -> torch.Tensor:
        """Untiled multi-step Euler decode. Returns the final ``x_t``
        (``B, C, F, H, W``, raw diffusion range) -- ``decode_video`` applies
        the ``[0, 1]`` RGB mapping.
        """
        latent_shape = VideoLatentShape.from_torch_shape(latent.shape)
        full_shape = latent_shape.upscale(self.video_downscale_factors)._replace(channels=self.out_channels)
        target_shape = full_shape.to_torch_shape()

        batch = latent.shape[0]
        timestep = self.default_inference_timesteps.to(latent.device).unsqueeze(0).expand(batch, -1)
        _, num_steps = timestep.shape

        compute_dtype = latent.dtype
        randn_device = generator.device if generator is not None else latent.device
        x_t = torch.randn(tuple(target_shape), dtype=compute_dtype, generator=generator, device=randn_device).to(
            latent.device
        )

        if self.mark_dynamic_shapes:
            for dim in (2, 3, 4):
                torch._dynamo.maybe_mark_dynamic(latent, dim)

        context = self.forward_pre_diffusion(latent)

        for i in range(num_steps):
            t_now = timestep[:, i]
            t_next = timestep[:, i + 1] if i + 1 < num_steps else torch.zeros_like(t_now)
            combined = self._combined_for_diff_step(context, x_t)
            if self.mark_dynamic_shapes:
                for dim in (1, 2, 3):
                    torch._dynamo.maybe_mark_dynamic(combined, dim)
            model_out = self.forward_diff_step(combined, t_now).to(torch.float32)
            x_t = self._euler_step(x_t, model_out, t_now, t_next)

        return x_t
