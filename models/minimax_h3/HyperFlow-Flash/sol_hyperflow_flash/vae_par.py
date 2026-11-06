"""Parallel H3 video VAE for multi-GPU (and the compiled decoder on 1 GPU): Sol-H3's own implementation, switchable per preset.

Same weights and the same diffusers AutoencoderKLMiniMaxH3.

decode "solh3"   Sol-H3 `h3_runtime/vae_parallel.install(vae, batched=True, compile_mode="default", encode_parallel=True)`
                 (the call of Sol-H3 engine.py and of the HyperFlow runner):
                 the ViT decoder compiled (torch.compile mode "default", dynamic=False) on every world size; world > 1:
                 the equal-size (temporal clip x spatial tile) decode tiles of the whole clip split contiguously over the
                 ranks, decoded as ONE batch per rank, all_gather_into_tensor, blended by the diffusers _stitch_tiles / _blend
                 (vae_parallel.py:222-323). 1 GPU: compiled decoder, diffusers per-tile loop (vae_parallel.py:84-100).
decode "diffusers"   the diffusers decode, replicated on every rank.

reference encode (video VAE encode of image / video references; ref2va only):
  "diffusers"        diffusers: every rank encodes every tile of every 17-frame clip.
  "sharded"      Sol-H3 encode_parallel (vae_parallel.py:170-185): the spatial tiles of ONE clip split over the ranks, per-tile
                 encoder loop as diffusers, per-clip all-gather, diffusers stitch. Images (one clip) the same.
  "global"       this runner (Sol-H3's tiling + blending rules, its global-decode scheme applied to the encoder): every (clip, tile)
                 pair of the whole reference video split contiguously over the ranks, per-tile encoder loop as diffusers, ONE
                 all-gather, then the diffusers per-clip stitch, concatenation and token drop of `_encode`. Images go through
                 "sharded" (one clip).
  encode_compile this runner: the encoder compiled (torch.compile mode "default", dynamic=False); Sol-H3 compiles only the decoder.
Every tile is encoded by the same encoder call on the same input view as diffusers, so the eager sharded / global encodes are
expected to be bitwise equal to the single-GPU encode (checked by the encode gate); the compiled encoder is not bitwise.
"""
from __future__ import annotations

import math

ENCODE_MODES = ("diffusers", "sharded", "global")
DECODE_MODES = ("diffusers", "solh3")
_PATCHED = ("_decode", "_decode_clip", "_encode_clip")


def encode_geometry(vae, height, width):
    """Spatial tiles of `_encode_clip` (pixel coords) and the latent-space overlaps of `_stitch_tiles`: the diffusers rules."""
    yi, yl, yo = vae._split_tiles(height, vae.tile_sample_min_height, vae.tile_sample_min_overlap_height)
    xi, xl, xo = vae._split_tiles(width, vae.tile_sample_min_width, vae.tile_sample_min_overlap_width)
    coords = [(yi[i], yl[i], xi[j], xl[j]) for i in range(len(yi)) for j in range(len(xi))]
    r = vae.spatial_compression_ratio
    return coords, yi, [o // r for o in yo], xi, [o // r for o in xo]


def stitch(vae, flat, start, y_indices, y_overlaps, x_indices, x_overlaps):
    rows, idx = [], start
    for _ in y_indices:
        row = []
        for _ in x_indices:
            row.append(flat[idx].unsqueeze(0))
            idx += 1
        rows.append(row)
    return vae._stitch_tiles(rows, y_overlaps, x_overlaps)


def global_encode_local(torch, vae, x, rank, world):
    """This rank's part of `vae._encode(x)` with every (clip, tile) pair split contiguously over `world` ranks (padded with
    duplicates of the last pair to an equal count per rank). Returns (local stack, plan) for `global_encode_assemble`."""
    clip_length = vae.config.clip_length
    T = x.shape[2]
    if T % clip_length:
        x = torch.cat([x, x[:, :, -1:].repeat(1, 1, (-T) % clip_length, 1, 1)], dim=2)
    n_clips = x.shape[2] // clip_length
    coords, yi, yo, xi, xo = encode_geometry(vae, x.shape[-2], x.shape[-1])
    items = [(c, t) for c in range(n_clips) for t in coords]
    per_rank = -(-len(items) // world)
    padded = items + [items[-1]] * (per_rank * world - len(items))
    local = []
    for c, (y, y_len, x0, x_len) in padded[rank * per_rank:(rank + 1) * per_rank]:
        tile = x[:, :, c * clip_length:(c + 1) * clip_length][..., y:y + y_len, x0:x0 + x_len]
        local.append(vae.quant_conv(vae.encoder(tile)))
    return torch.cat(local, dim=0), dict(n=len(items), n_clips=n_clips, tiles=len(coords), geom=(yi, yo, xi, xo))


def global_encode_assemble(torch, vae, flat, plan):
    """Diffusers stitch of every clip from the rank-major gathered tiles, concatenated in time, `token_drop` applied (= `_encode`)."""
    yi, yo, xi, xo = plan["geom"]
    moments = torch.cat([stitch(vae, flat, c * plan["tiles"], yi, yo, xi, xo) for c in range(plan["n_clips"])], dim=2)
    if vae.config.token_drop > 0 and plan.get("token_drop", True):
        moments = moments[:, :, :-vae.config.token_drop]
    return moments


def clip_encode_local(torch, vae, x, rank, world):
    """This rank's tiles of `vae._encode_clip(x)` (one temporal clip, e.g. an image reference) split contiguously over `world`."""
    coords, yi, yo, xi, xo = encode_geometry(vae, x.shape[-2], x.shape[-1])
    per_rank = -(-len(coords) // world)
    padded = coords + [coords[-1]] * (per_rank * world - len(coords))
    local = [vae.quant_conv(vae.encoder(x[..., y:y + y_len, x0:x0 + x_len]))
             for y, y_len, x0, x_len in padded[rank * per_rank:(rank + 1) * per_rank]]
    return torch.cat(local, dim=0), dict(n=len(coords), n_clips=1, tiles=len(coords), geom=(yi, yo, xi, xo), token_drop=False)


def global_encode(torch, vae, x, rank, world, gather):
    """`vae._encode(x)` sharded over (clip, tile) pairs. `gather(local_stack, n)` = rank-major all-gather truncated to n."""
    local, plan = global_encode_local(torch, vae, x, rank, world)
    return global_encode_assemble(torch, vae, gather(local, plan["n"]), plan)


class ParallelVAE:
    """Installs Sol-H3's vae_parallel once (its compiled decoder + patched methods are kept) and switches the VAE between
    diffusers and the parallel paths per preset without recompiling (Sol-H3's own uninstall resets dynamo)."""

    def __init__(self, torch, vae, vae_parallel_mod, *, compile_mode="default", deterministic=False):
        import torch.distributed as dist

        self.torch, self.vae, self.compile_mode = torch, vae, compile_mode
        self.dist = dist
        # inductor's deterministic mode (no on-device benchmarking of kernel configs): every rank compiles the SAME kernels, so a
        # tile gives the same bits whichever rank encodes / decodes it (needed for the bitwise text / encode split; without it
        # ranks may pick different reduction configs by timing noise)
        if deterministic:
            import torch._inductor.config as inductor_config

            inductor_config.deterministic = True
        self.deterministic = bool(deterministic)
        self.eager_decoder, self.eager_encoder = vae.decoder, vae.encoder
        before = {n: vae.__dict__.get(n) for n in _PATCHED}
        self._uninstall = vae_parallel_mod.install(vae, batched=True, compile_mode=compile_mode, encode_parallel=True)
        self.compiled_decoder = vae.decoder
        self.patched = {n: vae.__dict__[n] for n in _PATCHED if vae.__dict__.get(n) is not None and vae.__dict__[n] is not before[n]}
        self.world = dist.get_world_size() if (dist.is_available() and dist.is_initialized()) else 1
        self.rank = dist.get_rank() if self.world > 1 else 0
        self._compiled_encoder = None
        self.group = None             # encode subgroup (text / encode split: ranks 1..W-1), None = every rank
        self.decode_mode, self.encode_mode, self.encode_compile = None, None, None
        self.set("diffusers", "diffusers", False)

    def compiled_encoder(self):
        if self._compiled_encoder is None:
            self._compiled_encoder = self.torch.compile(self.eager_encoder, mode=self.compile_mode, dynamic=False)
        return self._compiled_encoder

    def _grp(self):
        """(rank, world) inside the encode group (the whole world unless a subgroup is set)."""
        if self.group is None:
            return self.rank, self.world
        return self.dist.get_rank(self.group), self.dist.get_world_size(self.group)

    def _gather(self, local_stack, n):
        _, w = self._grp()
        local_stack = local_stack.contiguous()
        flat = self.torch.empty((w * local_stack.shape[0], *local_stack.shape[1:]), dtype=local_stack.dtype,
                                device=local_stack.device)
        self.dist.all_gather_into_tensor(flat, local_stack, group=self.group)
        return flat[:n]

    def _global_encode(self, x):
        vae = self.vae
        r, w = self._grp()
        if not vae.use_tiling or x.shape[0] != 1 or w == 1:
            return type(vae)._encode(vae, x)
        return global_encode(self.torch, vae, x, r, w, self._gather)

    def _group_encode_clip(self, x):
        """`_encode_clip` (one clip: image reference) sharded over the encode subgroup (Sol-H3's patched version is bound to
        the world group, so it cannot run while rank 0 encodes the text)."""
        vae = self.vae
        r, w = self._grp()
        if not vae.use_tiling or x.shape[0] != 1 or w == 1:
            return type(vae)._encode_clip(vae, x)
        local, plan = clip_encode_local(self.torch, vae, x, r, w)
        return global_encode_assemble(self.torch, vae, self._gather(local, plan["n"]), plan)

    def set_group(self, group):
        """Encode on a process subgroup (or None = world). With a subgroup, image references use the group-aware clip encode."""
        self.group = group
        self.set(self.decode_mode, self.encode_mode, self.encode_compile)

    def _set_attr(self, name, fn):
        if fn is None:
            if name in self.vae.__dict__:
                delattr(self.vae, name)
        else:
            setattr(self.vae, name, fn)

    def set(self, decode="diffusers", encode="diffusers", encode_compile=False):
        if decode not in DECODE_MODES or encode not in ENCODE_MODES:
            raise ValueError(f"vae decode {decode!r} / encode {encode!r}")
        v = self.vae
        v.decoder = self.compiled_decoder if decode == "solh3" else self.eager_decoder
        for n in ("_decode", "_decode_clip"):
            self._set_attr(n, self.patched.get(n) if decode == "solh3" else None)
        if self.group is not None:                   # text / encode split: every encode on the subgroup (global scheme)
            self._set_attr("_encode_clip", self._group_encode_clip)
            self._set_attr("_encode", self._global_encode)
        else:
            self._set_attr("_encode_clip", self.patched.get("_encode_clip") if encode in ("sharded", "global") else None)
            self._set_attr("_encode", self._global_encode if (encode == "global" and self.world > 1) else None)
        v.encoder = self.compiled_encoder() if encode_compile else self.eager_encoder
        self.decode_mode, self.encode_mode, self.encode_compile = decode, encode, bool(encode_compile)

    def describe(self):
        return dict(decode=self.decode_mode, encode=self.encode_mode, encode_compile=self.encode_compile, world=self.world,
                    compile_mode=self.compile_mode, patched=sorted(self.patched),
                    source="Sol-H3 h3_runtime/vae_parallel.py; 'global' encode + encoder compile: this runner")


def psnr_range2(diff_mse):
    """Sol-H3's decoder-space PSNR: peak-to-peak 2 (vae_parallel.py:302)."""
    return math.inf if diff_mse == 0.0 else 10.0 * math.log10(4.0 / diff_mse)


# ---- gates (GPU validation; never inside a timed request) ------------------------------------------------------------
def _diff(torch, a, ref):
    d = a.float() - ref.float()
    mse = float(d.square().mean())
    return dict(max_abs=float(d.abs().max()), rel_l2=float(d.norm() / ref.float().norm().clamp_min(1e-30)), mse=mse,
                bitwise_equal=bool(torch.equal(a, ref)))


def _sync_ms(torch, fn):
    import time

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        out = fn()
    torch.cuda.synchronize()
    return out, (time.perf_counter() - t0) * 1e3


def decode_gate(torch, pvae, vae, z):
    """Decoder input `z` (exactly what the pipeline handed to vae.decode) decoded by the diffusers path and twice by Sol-H3's path,
    under the pipeline's fp16 autocast. Decoder-space deviation (Sol-H3's metrics: max_abs, rel L2, PSNR with peak-to-peak 2)
    and the uint8 video after the pipeline's de-normalisation + clamp; determinism = the two Sol-H3 decodes bitwise equal."""
    from diffusers.modular_pipelines.minimax_h3 import decoders as D

    state = (pvae.decode_mode, pvae.encode_mode, pvae.encode_compile)
    mean = torch.tensor(D.MINIMAX_H3_PIXEL_MEAN, device=z.device).view(1, -1, 1, 1, 1)
    std = torch.tensor(D.MINIMAX_H3_PIXEL_STD, device=z.device).view(1, -1, 1, 1, 1)

    def dec(mode):
        pvae.set(mode, state[1], state[2])
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            return _sync_ms(torch, lambda: vae.decode(z, return_dict=False)[0])

    try:
        ref, t_ref = dec("diffusers")
        a, t_a = dec("solh3")
        b, t_b = dec("solh3")
    finally:
        pvae.set(*state)
    r = dict(_diff(torch, a, ref), psnr_range2_db=None, deterministic=bool(torch.equal(a, b)), diffusers_ms=t_ref,
             solh3_first_ms=t_a, solh3_second_ms=t_b, shape=list(a.shape), dtype=str(a.dtype))
    r["psnr_range2_db"] = psnr_range2(r["mse"])
    r["where_max"] = locate_max(torch, vae, a, ref)

    def u8(x):
        return ((x.float() * std + mean).clamp(0, 1) * 255).round().to(torch.uint8)[0].permute(1, 0, 2, 3)
    ua, uref = u8(a), u8(ref)
    from .media import psnr_u8

    r["u8_max_abs"] = int((ua.int() - uref.int()).abs().max())
    r["u8_psnr_mean_db"], r["u8_psnr_min_db"] = psnr_u8(ua, uref, torch)
    r["u8_frac_pixels_differ"] = float((ua != uref).float().mean())
    return r


def locate_max(torch, vae, a, ref):
    """Where the largest |a - ref| sits: (channel, frame, y, x), the decode tiles (pixel rows / cols of the 256-px tiles with
    their blend overlaps) and temporal chunk covering it, the value range there, and how many values exceed 0.02 / 0.05."""
    d = (a.float() - ref.float()).abs()[0]                    # [C, T, H, W]
    flat = int(d.argmax())
    C, T, H, W = d.shape
    c, rem = divmod(flat, T * H * W)
    t, rem = divmod(rem, H * W)
    y, x = divmod(rem, W)
    yi, yl, yo = vae._split_tiles(H, vae.tile_sample_min_height, vae.tile_sample_min_overlap_height)
    xi, xl, xo = vae._split_tiles(W, vae.tile_sample_min_width, vae.tile_sample_min_overlap_width)
    tiles_y = [i for i, (s0, n) in enumerate(zip(yi, yl)) if s0 <= y < s0 + n]
    tiles_x = [j for j, (s0, n) in enumerate(zip(xi, xl)) if s0 <= x < s0 + n]
    chunk = vae.tokens_chunk_size * vae.temporal_compression_ratio - vae.frame_pre_padding
    q = torch.quantile(d.flatten()[::97].float(), torch.tensor([0.5, 0.999, 0.999999], device=d.device))
    return dict(channel=c, frame=t, y=y, x=x, tile_rows=tiles_y, tile_cols=tiles_x, in_spatial_overlap=len(tiles_y) > 1 or
                len(tiles_x) > 1, temporal_chunk_approx=t // max(chunk, 1), diffusers_value=float(ref[0, c, t, y, x]),
                ours_value=float(a[0, c, t, y, x]), n_over_0p02=int((d > 0.02).sum()), n_over_0p05=int((d > 0.05).sum()),
                n_values=int(d.numel()), abs_diff_quantiles_sampled={"p50": float(q[0]), "p99.9": float(q[1]),
                                                                     "p99.9999": float(q[2])},
                frame_max=[float(v) for v in d.amax(dim=(0, 2, 3))])


def encode_compare(torch, pvae, encode_references, args, modes):
    """Run the pipeline's reference encode (`encode_references(*args)`, i.e. MiniMaxH3Ref2VAReferenceEncoderStep with the
    prepared references of a real request) under each (mode, compile) of `modes`: the first call per mode is untimed (warms /
    compiles), the second one is timed; outputs (video rows, audio rows: posterior-sampled, fp16-rounded, normalised) compared
    with the first mode (diffusers)."""
    state = (pvae.decode_mode, pvae.encode_mode, pvae.encode_compile)
    out, ref = [], None
    try:
        for mode, comp in modes:
            pvae.set(state[0], mode, comp)
            with torch.no_grad():
                encode_references(*args)
            (vr, ar), ms = _sync_ms(torch, lambda: encode_references(*args))
            row = dict(mode=mode, compiled=bool(comp), ms=ms)
            if ref is None:
                ref = (vr, ar)
            else:
                if vr is not None:
                    row["video_rows"] = _diff(torch, vr, ref[0])
                    row["video_rows"]["frac_elements_differ"] = float((vr != ref[0]).float().mean())
                if ar is not None:
                    row["audio_rows"] = _diff(torch, ar, ref[1])
            out.append(row)
    finally:
        pvae.set(*state)
    return out


def encode_moments_compare(torch, pvae, vae, pixels, modes):
    """Raw video-VAE moments (before posterior sampling) of one reference video under each (mode, compile) vs the first mode."""
    state = (pvae.decode_mode, pvae.encode_mode, pvae.encode_compile)
    out, ref = [], None
    try:
        for mode, comp in modes:
            pvae.set(state[0], mode, comp)
            with torch.no_grad():
                vae._encode(pixels)
            m, ms = _sync_ms(torch, lambda: vae._encode(pixels))
            row = dict(mode=mode, compiled=bool(comp), ms=ms, shape=list(m.shape))
            if ref is None:
                ref = m
            else:
                row.update(_diff(torch, m, ref))
            out.append(row)
    finally:
        pvae.set(*state)
    return out
