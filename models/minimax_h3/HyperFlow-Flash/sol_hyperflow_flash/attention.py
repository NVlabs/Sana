"""Attention of the live HyperFlow H3 Flash blocks: a dense / sparse schedule over Sol-Attn (Sol-H3 ``sol_bsa``).

Schedule (preset.sparse): steps < dense_steps -> every live block dense attention; later steps -> blocks in edge_blocks dense
attention, every other live block Sol-Attn via h3_runtime.sparse_attention._sol_bsa_attention with the packed text + audio
prefix as an exact KV sink (dense query rows), thresh_type "diag", 64-token blocks, tau = preset.tau. Dropped blocks stay
identity. Dense attention = diffusers dispatch_attention_fn (backend None -> cuDNN SDPA on GB200, BF16).

Under Ulysses sequence parallelism (`sp`) the processor exchanges q / k / v to (full sequence, H / W heads) so every back-end
call sees exactly the full-sequence inputs of its heads, then exchanges back. Presets with `transport` != "bf16" use Sol-H3's
compressed Ulysses transport instead (h3_runtime/ulysses.py + relayout.py + comm_quant.py): QK RMSNorm +
partial RoPE + destination-major pack + INT8 encode (group-32 UE5M3 scales, 400-byte records) in one kernel, one all-to-all,
dequantised into the padded BSA buffer; attention output sent back as saturated raw FP8 E4M3. "int8fp8" = Sol-H3's default
scope (every call, ulysses.py:54 / :76-80); "int8fp8-sparse" = only the Sol-Attn calls (Sol-H3's H3_ULYSSES_INT8_SCOPE=sparse;
the dense calls keep BF16 both ways).

Exact prefix (text, every ref2va reference / fl2va keyframe, target audio): the packed layout is [text | conditioning | target audio
| target video] and the exact KV sink is [0, video_start) with video_start = first target-video row. Sol-H3's route policy
(bsa_metadata.route_and_compact) selects every (query block, key block) pair with the key OR the query inside the sink, so the
prefix keys are attended exactly by every query and the prefix rows attend every key (no omitted block, no Sol residual); the
density statistics start at the target video, so the prefix never counts against the density budget. `check_layout` refuses any
request whose non-target rows are not all inside that prefix.
"""
from __future__ import annotations

import math


def target_video_start(video_indices, S):
    """First row of the contiguous target-video tail = start of the LAST contiguous run of video rows (the rule of Sol-H3
    sparse_attention._target_video_start, which the sparse back-end uses)."""
    if video_indices is None or video_indices.numel() == 0:
        return int(S)
    breaks = ((video_indices[1:] - video_indices[:-1]) != 1).nonzero()
    start = int(breaks[-1].item()) + 1 if breaks.numel() else 0
    return int(video_indices[start].item())


def check_layout(torch, video_indices, audio_indices, text_indices, n_cond_video, n_cond_audio, S, video_start):
    """Every row that is not target video must lie in the exact prefix [0, video_start), and the target video must be exactly
    the tail [video_start, S). Returns the row accounting; raises RuntimeError otherwise."""
    vs = int(video_start)
    tgt = video_indices[int(n_cond_video):]
    if tgt.numel() != S - vs or not bool(torch.equal(tgt, torch.arange(vs, S, device=tgt.device, dtype=tgt.dtype))):
        raise RuntimeError(f"target video rows are not the tail [{vs}, {S})")
    groups = dict(text=text_indices, cond_video=video_indices[:int(n_cond_video)], cond_audio=audio_indices[:int(n_cond_audio)],
                  target_audio=audio_indices[int(n_cond_audio):])
    for name, idx in groups.items():
        if idx is not None and idx.numel() and int(idx.max()) >= vs:
            raise RuntimeError(f"{name} rows reach {int(idx.max())} >= target-video start {vs}: not inside the exact prefix")
    out = {f"n_{k}": int(0 if v is None else v.numel()) for k, v in groups.items()}
    out.update(S=int(S), video_start=vs, n_target_video=int(S - vs), sink_blocks=math.ceil(vs / 64))
    return out


def dense_fp32(torch, q, k, v, chunk=4096):
    """softmax(q k^T / sqrt(D)) v in fp32 for [1, T, H, D] inputs, one head and `chunk` query rows at a time (explicit math: no
    attention back-end, no [T, T] tensor)."""
    T, H, D = q.shape[1], q.shape[2], q.shape[3]
    out = torch.empty((1, T, H, D), dtype=torch.float32, device=q.device)
    for h in range(H):
        kh, vh = k[0, :, h].float(), v[0, :, h].float()
        for s0 in range(0, T, chunk):
            p = torch.softmax((q[0, s0:s0 + chunk, h].float() @ kh.t()) * D ** -0.5, dim=-1)
            out[0, s0:s0 + chunk, h] = p @ vh
    return out


class Controller:
    """Per-request / per-step state shared by the sampler, the attention processors and the block fusions."""

    def __init__(self, torch, *, sol_bucket: int = 4096):
        self.torch = torch
        self.preset = None
        self.step = 0
        self.video_start = None
        self.num_frames = None
        self.counts = {}
        self.bucket = sol_bucket
        self.sa = None             # h3_runtime.sparse_attention
        self.fusions = None        # h3_runtime.fusions
        self.sp = None             # sp.SP when world > 1
        self.fused_qknorm_rope = False
        self.block_fused = True
        self.collect = False       # density statistics (one extra prepare() per sparse call; never in timed requests)
        self.calls = []
        self.capture_step = None   # gate helper: clone latents after this step
        self.captured_latents = None
        self.layout = None         # check_layout() accounting of the current request
        self.check_exact = False   # gate: route mask + exact rows vs dense fp32 on the first sparse call of a request
        self.exact_report = None
        self.merge_ref = None      # reference Sol residual merge (the exact gate compares the tuned launch against it)
        self.sink_idx = None       # sink "text_audio": token rows of the exact sink (text + all audio) and its 64-block mask
        self.sink_block_mask = None
        self.visual_ref_idx = None
        self.q8 = None             # Sol-H3 transport modules (relayout, comm_quant, ulysses) once a transport preset is applied
        self.comm_log = None       # stage audit: list of (name, start event, end event) of the SP collectives, else None
        self.check_transport = False   # gate: compare the compressed transport with the BF16 one on one real call
        self.transport_report = None

    # ---- sampler hooks -------------------------------------------------------------------------------------------------------
    def on_request(self, video_indices, S):
        self.exact_report = None
        self.transport_report = None
        if self.q8 is not None:
            self.q8.ulysses.reset_row_counts()     # Sol-H3 caches the per-rank row counts per request (every rank calls this)
        self.video_start = target_video_start(video_indices, S)
        if self.sa is not None and int(self.sa._target_video_start(video_indices, S)) != self.video_start:
            raise RuntimeError("Sol-H3 _target_video_start disagrees with attention.target_video_start")

    def on_layout(self, video_indices, audio_indices, text_indices, n_cond_video, n_cond_audio, S):
        self.layout = check_layout(self.torch, video_indices, audio_indices, text_indices, n_cond_video, n_cond_audio, S,
                                   self.video_start)
        self.sink_idx = self.sink_block_mask = self.visual_ref_idx = None
        if self.sink_mode() == "text_audio":
            self.sink_idx, self.sink_block_mask = text_audio_sink(self.torch, text_indices, audio_indices, S)
            self.visual_ref_idx = video_indices[:int(n_cond_video)]
            self.layout = dict(self.layout, sink="text_audio", sink_rows=int(self.sink_idx.numel()),
                               sink_blocks_mask=int(self.sink_block_mask.sum()))

    def sink_mode(self):
        return getattr(self.preset, "sink", "prefix") if self.preset is not None else "prefix"

    def on_step(self, i):
        self.step = i

    def after_step(self, i, bs):
        if self.capture_step is not None and i == self.capture_step:
            self.captured_latents = (bs.latents.detach().clone(), bs.audio_latents.detach().clone())

    # ---- schedule -------------------------------------------------------------------------------------------------------------
    def backend(self, block: int) -> str:
        p = self.preset
        if p is None or not p.sparse:
            return "dense"
        if self.step < p.dense_steps or block in p.edge_blocks:
            return "dense"
        return "solbsa"

    def _sol(self, q, k, v, collect, valid_tokens=None):
        tau = self.preset.tau
        if valid_tokens is not None:                       # pre-bucketed (transport receive buffer), T = valid_tokens
            if valid_tokens % self.bucket == 0:
                raise RuntimeError(f"T={valid_tokens} is a multiple of the bucket {self.bucket}; choose another bucket")
            kw = dict(valid_tokens=valid_tokens, inputs_bucketed=True)
            if self.sink_mode() == "text_audio":
                return self.sa._sol_bsa_attention(q, k, v, tau=tau, thresh_type="diag", sink_start=0, sink_tokens=0,
                                                  sink_block_mask=self.sink_block_mask, collect_stats=collect,
                                                  compile_bucket_size=self.bucket, **kw)
            return self.sa._sol_bsa_attention(q, k, v, tau=tau, thresh_type="diag", sink_start=0,
                                              sink_tokens=int(self.video_start), collect_stats=collect,
                                              compile_bucket_size=self.bucket, **kw)
        if q.shape[1] % self.bucket == 0:
            # sol_attn's _pad_to_bucket returns independent unpadded tensors then; its own comment: only the packed layout is
            # validated on the SM100 TMA path for long sequences -> refuse instead of risking invalid results
            raise RuntimeError(f"T={q.shape[1]} is a multiple of the bucket {self.bucket}; choose another bucket")
        if self.sink_mode() == "text_audio":
            return self.sa._sol_bsa_attention(q, k, v, tau=tau, thresh_type="diag", sink_start=0, sink_tokens=0,
                                              sink_block_mask=self.sink_block_mask, collect_stats=collect,
                                              compile_bucket_size=self.bucket)
        return self.sa._sol_bsa_attention(q, k, v, tau=tau, thresh_type="diag", sink_start=0,
                                          sink_tokens=int(self.video_start), collect_stats=collect,
                                          compile_bucket_size=self.bucket)

    def attend(self, q, k, v, block, mask, dense_fn):
        if self.sp is not None and self.sp.W > 1:
            return self.attend_sp(q, k, v, block, mask)
        be = self.backend(block)
        self.counts[be] = self.counts.get(be, 0) + 1
        if be == "dense":
            return dense_fn()
        if mask is not None:
            raise RuntimeError("attention mask given to the mask-free sparse back-end")
        if self.check_exact and self.exact_report is None:
            self.exact_report = dict(step=self.step, block=block, **self.exact_check(q, k, v))
        out, dens = self._sol(q, k, v, self.collect)
        if self.collect:
            self.calls.append(dict(step=self.step, block=block, effective_density=dens["effective_density"],
                                   threshold_density=dens["threshold_density"], **self.vv_stats(q, k, v)))
        return out

    def transport(self):
        p = self.preset
        return "bf16" if (p is None or self.sp is None or self.sp.W == 1) else getattr(p, "transport", "bf16")

    def _timed(self, name, fn):
        if self.comm_log is None:
            return fn()
        from .media import event

        b = event(self.torch)
        out = fn()
        self.comm_log.append((name, b, event(self.torch)))
        return out

    def attend_sp_q8(self, q, k, v, attn, rot, block):
        """Sol-H3's compressed Ulysses transport (see the module docstring). q / k / v: [1, S_r, H, D] projections BEFORE the
        QK norm and RoPE (the pack kernel applies them, same arithmetic as the fused_qknorm_rope kernel)."""
        torch = self.torch
        be = self.backend(block)
        self.counts[be] = self.counts.get(be, 0) + 1
        S, W, D = self.sp.S, self.sp.W, q.shape[-1]
        rl, cq, ul = self.q8.relayout, self.q8.comm_quant, self.q8.ulysses
        lossy = self.transport() == "int8fp8" or be == "solbsa"
        wire = "int8_qkv" if lossy else "bf16"
        cos, sin = rot
        q3, k3, v3 = q[0], k[0], v[0]
        rows_local, hl = q3.shape[0], q3.shape[1] // W

        def run(wire, owire):
            packet = rl.qknorm_rope_pack_qkv_destination_major(q3, k3, v3, attn.norm_q.weight, attn.norm_k.weight, cos, sin,
                                                               attn.norm_q.eps, attn.norm_k.eps, W, wire_dtype=wire)
            packed = self._timed("a2a_qkv", lambda: ul._packed_qkv_all_to_all(q3, k3, v3, world=W, group=None,
                                                                              packed_send=packet, wire_dtype=wire))
            if wire == "int8_qkv":
                cap = self.sa._compile_capacity(S, self.bucket) if be == "solbsa" else S
                qb, kb, vb = cq.dequantize_qkv_padded(packed, S, hl, cap)
            else:
                qb, kb, vb = (x.unsqueeze(0) for x in packed.split(D, dim=-1))
            if be == "dense":
                from diffusers.models.attention_dispatch import dispatch_attention_fn

                of = dispatch_attention_fn(qb, kb, vb, attn_mask=None, dropout_p=0.0, is_causal=False, backend=None)
            else:
                if self.check_exact and self.exact_report is None:
                    self.exact_report = dict(step=self.step, block=block, heads_local=hl, transport=wire,
                                             **self.exact_check(qb[:, :S], kb[:, :S], vb[:, :S]))
                of, _ = self._sol(qb, kb, vb, False, valid_tokens=S if qb.shape[1] != S else None)
            out = of.to(q.dtype)[0].contiguous()                               # [S, H/W, D]
            back = self._timed("a2a_out", lambda: ul._packed_out_all_to_all(out, rows_local, W, None, wire_dtype=owire))
            return back.reshape(1, rows_local, W * hl, D)

        res = run(wire, "fp8" if lossy else "bf16")
        if self.check_transport and self.transport_report is None and lossy:
            ref = run("bf16", "bf16")
            num = (res.float() - ref.float()).square().sum()
            den = ref.float().square().sum()
            tot = torch.stack([num, den])
            torch.distributed.all_reduce(tot)
            self.transport_report = dict(step=self.step, block=block, backend=be, wire=wire, out_wire="fp8",
                                         rel_l2_vs_bf16_transport=float((tot[0] / tot[1].clamp_min(1e-30)).sqrt()),
                                         max_abs_vs_bf16_transport_rank0=float((res.float() - ref.float()).abs().max()))
        return res

    def attend_sp(self, q, k, v, block, mask):
        torch = self.torch
        if mask is not None:
            raise RuntimeError("attention mask under sequence parallelism")
        be = self.backend(block)
        self.counts[be] = self.counts.get(be, 0) + 1
        S, D = self.sp.S, q.shape[-1]
        xf = self.sp.seq_to_head(torch.cat([q, k, v], -1), S)          # [1, S, H/W, 3D]; q / k / v = packed views
        qf, kf, vf = xf[..., :D], xf[..., D:2 * D], xf[..., 2 * D:]
        if be == "dense":
            from diffusers.models.attention_dispatch import dispatch_attention_fn

            of = dispatch_attention_fn(qf, kf, vf, attn_mask=None, dropout_p=0.0, is_causal=False, backend=None)
        else:
            if self.check_exact and self.exact_report is None:
                self.exact_report = dict(step=self.step, block=block, heads_local=int(qf.shape[2]), **self.exact_check(qf, kf, vf))
            of, _ = self._sol(qf, kf, vf, False)
        return self.sp.head_to_seq(of.to(q.dtype), S)

    # ---- exact-prefix gate ----------------------------------------------------------------------------------------------------
    def exact_check(self, q, k, v):
        """Structural + numerical check of the exact sink on real tensors [1, T, H, D] (one call, never in a timed request): the
        Sol route mask (the one the BSA kernel runs) must select every sink key for every query block and every key for every
        sink query block; the Sol output of the sink rows is compared with dense attention (fp32 math on the same BF16 inputs),
        next to the error of the routed rows for scale. Sink = prefix [0, video_start) (default) or the text + audio blocks
        (sink "text_audio"; the routed visual reference rows are reported separately)."""
        torch = self.torch

        T, vs = q.shape[1], int(self.video_start)
        ta = self.sink_mode() == "text_audio"
        qp, kp, vp = self.sa._bucketed_qkv(q, k, v, self.bucket)
        meta = self.sa._build_bsa_metadata(qp, kp, vp, tau=self.preset.tau, thresh_type="diag", sink_start=0,
                                           sink_tokens=0 if ta else vs, sink_block_mask=self.sink_block_mask if ta else None,
                                           collect_stats=False, valid_tokens=T, metadata_blocks=qp.shape[1] // 64)
        route = meta[4]                                   # [B, H, query blocks, key blocks], bool
        if ta:
            sinkb = self.sink_block_mask
            exact_rows = self.sink_idx
        else:
            sinkb = torch.zeros(route.shape[-1], dtype=torch.bool, device=q.device)
            sinkb[:math.ceil(vs / 64)] = True
            exact_rows = torch.arange(vs, device=q.device)
        cols = bool(route[:, :, :, sinkb].all())
        rows = bool(route[:, :, sinkb, :].all())
        out, _ = self._sol(q, k, v, False)
        ref = dense_fp32(torch, q, k, v)
        routed = torch.ones(T, dtype=torch.bool, device=q.device)
        routed[exact_rows] = False

        def rel(a, b):
            return float((a.float() - b).norm() / b.norm().clamp_min(1e-30))

        r = dict(sink_mode="text_audio" if ta else "prefix", T=int(T), video_start=vs, sink_blocks=int(sinkb.sum()),
                 route_blocks=int(route.shape[-1]), n_exact_rows=int(exact_rows.numel()),
                 prefix_keys_routed_for_all_queries=cols, prefix_queries_route_all_keys=rows,
                 prefix_rel_l2_vs_dense_fp32=rel(out[:, exact_rows], ref[:, exact_rows]),
                 prefix_bf16_rounding_floor=rel(ref[:, exact_rows].to(q.dtype), ref[:, exact_rows]),
                 target_rel_l2_vs_dense_fp32=rel(out[:, routed], ref[:, routed]))
        if ta and self.visual_ref_idx is not None and self.visual_ref_idx.numel():
            r["visual_ref_rel_l2_vs_dense_fp32"] = rel(out[:, self.visual_ref_idx], ref[:, self.visual_ref_idx])
        if getattr(self.preset, "merge_tuned", False) and self.merge_ref is not None:
            cur = self.sa.merge_sol_residual
            self.sa.merge_sol_residual = self.merge_ref
            try:
                out_ref, _ = self._sol(q, k, v, False)
            finally:
                self.sa.merge_sol_residual = cur
            r["merge_tuned_rel_l2_vs_reference"] = rel(out, out_ref.float())
            r["merge_tuned_max_abs_vs_reference"] = float((out.float() - out_ref.float()).abs().max())
        return r

    # ---- density accounting ---------------------------------------------------------------------------------------------------
    def vv_stats(self, q, k, v):
        """Block density of the Sol route over the ROUTABLE blocks only (sink rows and columns excluded, +-1 block band counted,
        as route_and_compact does): prefix sink -> target video x target video; text_audio sink -> every block without text /
        audio rows (visual references + target video). thr(tau) = mean + tau * std is linear in tau, so two prepare() calls give
        it exactly."""
        torch = self.torch
        tau = self.preset.tau
        _, prepare = self.sa._import_preprocess()
        T = q.shape[1]
        qp, kp, vp = self.sa._bucketed_qkv(q, k, v, self.bucket)
        scale = q.shape[-1] ** -0.5
        kc, _, thr0, q_bar = prepare(qp, kp, vp, scale=scale, tau=0.0, thresh_type="diag", valid_tokens=T, return_q_bar=True)
        _, _, thr1 = prepare(qp, kp, vp, scale=scale, tau=1.0, thresh_type="diag", valid_tokens=T)
        nb = math.ceil(T / 64)
        ids = routable_blocks(torch, nb, int(self.video_start), self.sink_block_mask if self.sink_mode() == "text_audio" else None,
                              q.device)
        kc, thr0, thr1, q_bar = kc[:, ids], thr0[:, ids], thr1[:, ids], q_bar[:, ids]
        s = torch.einsum("bqhd,bkhd->bqkh", q_bar, kc.float()) * (scale * math.log2(math.e))
        z = (s - thr0[:, :, None, :]) / (thr1 - thr0)[:, :, None, :].clamp_min(1e-12)
        band = ((ids[:, None] - ids[None, :]).abs() <= 1)[None, :, :, None]
        z = torch.where(band, torch.full_like(z, float("inf")), z)
        return dict(vv_density=float((z > tau).float().mean()), vv_blocks=int(ids.numel()))


def text_audio_sink(torch, text_indices, audio_indices, S):
    """Sink "text_audio": token rows of text + every audio row, and the bool mask of the 64-blocks that contain any of them
    (the policy of Sol-H3 sparse_attention._sink_blocks)."""
    idx = torch.cat([i.to(torch.long) for i in (text_indices, audio_indices) if i is not None and i.numel()])
    if idx.numel() == 0 or int(idx.min()) < 0 or int(idx.max()) >= S:
        raise RuntimeError("text / audio rows missing or outside the packed sequence")
    mask = torch.zeros(math.ceil(S / 64), dtype=torch.bool, device=idx.device)
    mask[torch.div(idx, 64, rounding_mode="floor").unique()] = True
    return idx.sort().values, mask.contiguous()


def routable_blocks(torch, nb, video_start, sink_block_mask, device):
    """Block ids the density counts: [ceil(video_start / 64), nb) for the prefix sink, ~sink_block_mask for text_audio."""
    if sink_block_mask is None:
        return torch.arange(math.ceil(video_start / 64), nb, device=device)
    return (~sink_block_mask[:nb]).nonzero().flatten().to(device)


def make_processor(ctl, block_idx, rope_fn):
    """Attention processor of one live block: MiniMaxH3AttnProcessor's math up to the attention back-end, plus the optional
    fused QK-norm / RoPE (ctl.fused_qknorm_rope)."""

    class FlashAttnProcessor:
        _attention_backend = None
        _parallel_config = None

        def __call__(self, attn, hidden_states, rotary_emb=None, attention_mask=None):
            if getattr(attn, "fused_projections", False):
                # HyperFlow H3 Flash quantizes to_q / to_k / to_v separately; this runner never fuses them
                raise RuntimeError("fused QKV projections are not supported by this processor; do not call fuse_projections()")
            query = attn.to_q(hidden_states).unflatten(-1, (attn.heads, -1))
            key = attn.to_k(hidden_states).unflatten(-1, (attn.heads, -1))
            value = attn.to_v(hidden_states).unflatten(-1, (attn.heads, -1))
            tr = ctl.transport()
            if (tr == "int8fp8" or (tr == "int8fp8-sparse" and ctl.backend(block_idx) == "solbsa")) and rotary_emb is not None \
                    and attention_mask is None:
                if not ctl.fused_qknorm_rope:
                    raise RuntimeError("the compressed transport fuses QK-norm + RoPE: needs a preset with fusions")
                out = ctl.attend_sp_q8(query, key, value, attn, rotary_emb, block_idx)
                out = out.flatten(2, 3).type_as(query)
                out = attn.to_out[0](out)
                return attn.to_out[1](out)
            if ctl.fused_qknorm_rope and rotary_emb is not None:
                cos, sin = rotary_emb
                query = ctl.fusions.fused_qknorm_rope(query, attn.norm_q.weight, cos, sin, attn.norm_q.eps)
                key = ctl.fusions.fused_qknorm_rope(key, attn.norm_k.weight, cos, sin, attn.norm_k.eps)
            else:
                query = attn.norm_q(query)
                key = attn.norm_k(key)
                if rotary_emb is not None:
                    query = rope_fn(query, *rotary_emb)
                    key = rope_fn(key, *rotary_emb)

            def dense():
                from diffusers.models.attention_dispatch import dispatch_attention_fn

                return dispatch_attention_fn(query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False,
                                             backend=self._attention_backend, parallel_config=self._parallel_config)

            out = ctl.attend(query, key, value, block_idx, attention_mask, dense)
            out = out.flatten(2, 3).type_as(query)
            out = attn.to_out[0](out)
            return attn.to_out[1](out)

    return FlashAttnProcessor()
