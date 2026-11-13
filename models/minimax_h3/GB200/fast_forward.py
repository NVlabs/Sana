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
import os

import torch
import torch.distributed as dist
import torch.nn.functional as F

from diffusers.models.transformers.transformer_minimax_h3 import (
    MINIMAX_H3_MODALITY_NUM, MiniMaxH3TransformerOutput,
)


# quack GemmConfig kwargs per GEMM kind (H3_QUACK_FFN / H3_QUACK_GEMM), from the E15c autotune on GB200
# (scripts/qall_bench.py, M=38247, power-steady medians vs nvjet: qkv 2.05 vs 2.14 ms, o 2.08 vs 2.21,
# down 4.17 vs 4.30, ffn_up+swiglu 8.17 vs 9.08); a missing kind uses quack's heuristic config.
_Q = dict(tile_m=256, tile_n=512, pingpong=False, is_dynamic_persistent=True, cluster_m=2, cluster_n=1,
          device_capacity=10)
QUACK_CONFIGS = {"qkv": dict(_Q, swap_ab=False), "o": dict(_Q, swap_ab=True), "down": dict(_Q, swap_ab=True),
                 "gated": dict(_Q, tile_n=256, swap_ab=False)}


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
    # Head-group pipelined exchange on the copy engine (`ulysses_pipe`) instead of the two blocking
    # NCCL all-to-alls, 7 groups by default (GB200, same-node A/B, bit-identical mp4s: 2 GPU 70.2 ->
    # 68.2 s, 4 GPU 37.1 -> 35.6 s). Dense attention core only. H3_ULYSSES_PIPE=<groups> overrides;
    # 0 = off.
    pipe = None
    heads, head_dim = tr.transformer_blocks[0].attn.heads, tr.transformer_blocks[0].attn.head_dim
    dense = getattr(getattr(tr, "_h3_attn_core", None), "__name__", "") == "_dense"
    default = "7" if dense and heads % (world * 7) == 0 else "0"
    pipe_groups = int(os.environ.get("H3_ULYSSES_PIPE", default))
    if world > 1 and pipe_groups > 0:
        import ulysses_pipe
        pipe = ulysses_pipe.UlyssesPipe(world, rank, group or torch.distributed.group.WORLD, heads,
                                        head_dim, pipe_groups)

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

    # quack (CuTe-DSL SM100) GEMMs for the bf16 blocks, default on (same-node A/B, inference p50:
    # 1 GPU 127.92 -> 125.82 s, 4 GPU 33.26 -> 33.15 s). H3_QUACK_FFN (default 1): ffn_up with
    # SwiGLU in the epilogue (fp32 accumulator -> silu(gate) * value -> bf16; the separate path
    # rounds the pre-activation to bf16 first, so rounding differs). The ffn_up weight rows are
    # interleaved once in place to quack's gated layout (gate even, value odd; H3: value = rows
    # [:F], gate = rows [F:]). H3_QUACK_GEMM (default qkv,o,down; "+" also separates, "" or 0 =
    # off): plain quack GEMMs, bitwise equal to nvjet. Configs are pinned (`tuned=False` + config),
    # so the server never autotunes, and quack compiles with a symbolic M (no recompile per length).
    quack_ffn = v2 and os.environ.get("H3_QUACK_FFN", "1") != "0"
    quack_plain = set(filter(None, os.environ.get("H3_QUACK_GEMM", "qkv,o,down").replace("+", ",").split(",")))
    quack_plain = quack_plain if v2 else set()
    if quack_ffn or quack_plain:
        from quack.gemm_config import GemmConfig
        from quack.gemm_interface import gemm_act, gemm_tuned
        qcfg = {k: GemmConfig(**v) for k, v in QUACK_CONFIGS.items()}
    if quack_ffn:
        for i, blk in enumerate(tr.transformer_blocks):
            if i in fp8_blocks or getattr(blk.ff.net[0].proj, "_h3_gated_interleaved", False):
                continue
            w = blk.ff.net[0].proj.weight
            n = w.shape[0] // 2
            w.data = torch.stack([w.data[n:], w.data[:n]], dim=1).reshape(w.shape)
            blk.ff.net[0].proj._h3_gated_interleaved = True
        torch.cuda.empty_cache()

    def ffn(i, blk, x):
        if quack_ffn and i not in fp8_blocks:
            w = blk.ff.net[0].proj.weight
            x2 = x.reshape(-1, x.shape[-1])
            act = torch.empty(x2.shape[0], w.shape[0] // 2, dtype=x.dtype, device=x.device)
            gemm_act(x2, w.T, activation="swiglu", postact_out=act, store_preact=False,
                     config=qcfg.get("gated"), tuned=False)
            act = act.view(*x.shape[:-1], act.shape[-1])
        else:
            act = swiglu(i, blk.ff.net[0].proj(x))
        return linear(i, "down", blk.ff.net[2], act)

    def linear(i, kind, mod, x):
        if kind not in quack_plain or i in fp8_blocks:
            return mod(x)
        x2 = x.reshape(-1, x.shape[-1])
        out = torch.empty(x2.shape[0], mod.weight.shape[0], dtype=x.dtype, device=x.device)
        gemm_tuned.fn(x2, mod.weight.T, out, config=qcfg.get(kind))
        return out.view(*x.shape[:-1], out.shape[-1])

    attn_core = getattr(tr, "_h3_attn_core", None)

    # Sync-free forward (H3_SYNC_FREE=0 restores the hooks): with no host sync inside the forward the
    # CPU queues the next step's boundary work (scheduler, text stream, scatter) while the GPU is still
    # in the block stack. Under CP the forward had three syncs: the padless check, the diffusers gather
    # at `proj_out` (`gather_size_by_comm` copies the shard sizes to the host), and the token refiner,
    # whose attention got the Ulysses-anything config like every processor although its input is the
    # whole (unsharded) text stream on every rank, so it gathered sizes with `.item()` and attended over
    # `world` copies of the same keys. Here the refiner attends locally (plain attention over its own
    # tokens, as on one GPU), and the heads gather with shard sizes known on the host (`tensor_split`).
    sync_free = os.environ.get("H3_SYNC_FREE", "1") != "0"
    if sync_free and world > 1:
        import copy
        block_procs = {id(b.attn.processor) for b in tr.transformer_blocks}
        for m in tr.token_refiner.modules():
            proc = getattr(m, "processor", None)
            if proc is not None and getattr(proc, "_parallel_config", None) is not None:
                if id(proc) in block_procs:
                    proc = copy.copy(proc)
                    m.processor = proc
                proc._parallel_config = None
    cp_group = group or (dist.group.WORLD if world > 1 else None)

    # Row balance (default on; H3_ROW_BALANCE=0 disables): GPUs of one node differ 7-10% in sustained speed under the power
    # cap, and every layer waits for the slowest rank twice (exchange in, exchange out). The row phase
    # (out-proj, norms, FFN, q/k/v projections, ~40% of a step) scales with a rank's rows, so after
    # timing it on forwards [4, 16) of the first request (CUDA events, no waits on peers inside it) all
    # ranks gather the per-row times and take rows proportional to speed. Every row's arithmetic is
    # the same; only which rank computes it changes (GEMM shapes, so rounding, differ).
    # H3_ROW_BALANCE_LOG=1 times the same window of every later request too and logs it (one sync).
    balance = (world > 1 and pipe is not None and sync_free
               and os.environ.get("H3_ROW_BALANCE", "1") != "0")
    bal_log = os.environ.get("H3_ROW_BALANCE_LOG", "0") != "0"
    cal_from, cal_to, per_req = 4, 16, 49
    bal = {"w": None, "counts": None, "fwd": 0, "pairs": [], "ev": None, "on": False}

    def bal_decide(rows_now):
        torch.cuda.synchronize()
        t = sum(a.elapsed_time(b) for a, b in bal["pairs"])
        bal["pairs"] = []
        per_row = torch.tensor([t / rows_now], dtype=torch.float64, device="cuda")
        parts = [torch.empty_like(per_row) for _ in range(world)]
        dist.all_gather(parts, per_row, group=cp_group)
        per_row = [float(x) for x in torch.cat(parts).tolist()]
        speed = [1.0 / x for x in per_row]
        w = [min(max(world * v / sum(speed), 0.85), 1.15) for v in speed]
        w = [x / sum(w) for x in w]
        if rank == 0:
            print(f"[row_balance] fwd {bal['fwd']} row-phase ms/forward per rank "
                  f"{[round(x * c / (cal_to - cal_from), 1) for x, c in zip(per_row, cur_counts())]} "
                  f"rows {cur_counts()} -> {split_counts(bal['seq'], w)}"
                  f"{'' if bal['w'] is None else ' (log only)'}", flush=True)
        if bal["w"] is None:
            bal["w"] = w

    def split_counts(total, w):
        """`total` rows over ranks by weights `w` (sum 1); same integers on every rank."""
        want = [total * x for x in w]
        counts = [int(x) for x in want]
        for i in sorted(range(world), key=lambda i: (counts[i] - want[i], i))[:total - sum(counts)]:
            counts[i] += 1
        return counts

    def cur_counts():
        """Row counts for the current sequence length (balanced weights once decided; the sequence
        length changes per request with the prompt's text length)."""
        if bal["w"] is not None:
            return split_counts(bal["seq"], bal["w"])
        return [len(r) for r in torch.arange(bal["seq"]).tensor_split(world)]

    def gather_rows(x, counts):
        """`(b, counts[rank], c)` on every rank -> `(b, sum(counts), c)` in rank order."""
        x = x.contiguous()
        m = max(counts)
        if x.shape[1] < m:
            x = torch.cat([x, x.new_zeros(x.shape[0], m - x.shape[1], x.shape[2])], dim=1)
        parts = [torch.empty_like(x) for _ in range(world)]
        dist.all_gather(parts, x, group=cp_group)
        return torch.cat([t[:, :n] for t, n in zip(parts, counts)], dim=1)

    def attention(i, blk, normed, rotary_emb):
        if not v2:
            return blk.attn(normed, rotary_emb, None)
        a = blk.attn
        q, k, v = (linear(i, "qkv", m, normed) for m in (a.to_q, a.to_k, a.to_v))
        b, s = q.shape[0], q.shape[1]
        q = q.view(b, s, a.heads, a.head_dim)
        k = k.view(b, s, a.heads, a.head_dim)
        cos, sin = rotary_emb
        q, k = kern.qknorm_rope_v3(q, k, a.norm_q.weight, a.norm_k.weight, cos, sin, a.norm_q.eps,
                                   tok=8, num_warps=4)
        rows = b * s
        q, k, v = (x.reshape(rows, a.heads, a.head_dim) for x in (q, k, v))
        if pipe is not None:
            if bal["counts"] is not None:
                counts = bal["counts"]
            else:
                counts = ulysses_custom._row_counts(rows, world, group)
            if bal["on"]:
                ev = torch.cuda.Event(enable_timing=True)
                ev.record()
                if bal["ev"] is not None:
                    bal["pairs"].append((bal["ev"], ev))
            out = pipe.attention(q, k, v, attn_core, counts)
            if bal["on"]:
                bal["ev"] = torch.cuda.Event(enable_timing=True)
                bal["ev"].record()
        elif world > 1:
            packed = ulysses_custom._packed_qkv_all_to_all(q, k, v, world=world, group=group)
            q, k, v = packed.split(a.head_dim, dim=-1)
            out = attn_core(q, k, v).contiguous()
            out = ulysses_custom._packed_out_all_to_all(out, rows, world, group)
        else:
            out = attn_core(q, k, v)
        return linear(i, "o", a.to_out[0], out.reshape(b, s, a.heads * a.head_dim))

    def forward(hidden_states, audio_hidden_states, encoder_hidden_states, timestep, timestep_indices,
                token_tags, position_ids, video_indices, audio_indices, text_indices,
                attention_kwargs=None, return_dict=True):
        self = tr
        sequence_length = position_ids.shape[0]
        if balance:
            bal["seq"] = sequence_length
            k = bal["fwd"] % per_req
            if k == cal_to and bal["pairs"]:
                bal_decide(cur_counts()[rank])
            bal["on"] = cal_from <= k < cal_to and (bal["w"] is None or bal_log)
            bal["counts"] = cur_counts() if bal["w"] is not None else None
            bal["ev"] = None
            bal["fwd"] += 1
        if bal["counts"] is not None:       # own rows of the full rope (bypassing its split hook)
            off, n_loc = sum(bal["counts"][:rank]), bal["counts"][rank]
            rotary_emb = tuple(x.narrow(0, off, n_loc)
                               for x in type(self.rope).forward(self.rope, position_ids))
        else:
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
        if sync_free:
            torch._assert_async((token_tags >= 0).all(), "fast_forward assumes a padless packed sequence")
        elif bool((token_tags < 0).any()):
            raise RuntimeError("fast_forward assumes a padless packed sequence")
        if bal["counts"] is not None:
            h = h.narrow(1, off, n_loc)
            idx = idx.narrow(0, off, n_loc)
        elif world > 1:                     # this rank's rows; `rope` is already split by its hook
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
            attn_out = attention(i, blk, normed, rotary_emb)
            h, normed = res_norm_mod(i, h, attn_out, gate_msa, blk.norm2.weight, scale_mlp, shift_mlp,
                                     idx, blk.norm2.eps)
            pending = (ffn(i, blk, normed), gate_mlp)
        h = kern.residual_gate(h, pending[0], pending[1], idx)

        if sync_free and world > 1:
            # the module bodies without their CP hooks (`norm_out` splits `timestep_indices`, the
            # heads gather), so the same rows and the same arithmetic
            if bal["counts"] is not None:
                counts = bal["counts"]
                t_idx = timestep_indices.narrow(0, off, n_loc)
            else:
                counts = [len(r) for r in torch.arange(sequence_length).tensor_split(world)]
                t_idx = timestep_indices.tensor_split(world)[rank]
            h = type(self.norm_out).forward(self.norm_out, h, temb, t_idx)
            h = h.to(self.proj_out.weight.dtype)
            video_output = gather_rows(F.linear(h, self.proj_out.weight, self.proj_out.bias), counts)
            audio_output = gather_rows(F.linear(h, self.audio_proj_out.weight, self.audio_proj_out.bias),
                                       counts)
            video_output = video_output.index_select(1, video_indices)
            audio_output = audio_output.index_select(1, audio_indices)
        else:
            h = self.norm_out(h, temb, timestep_indices).to(self.proj_out.weight.dtype)
            video_output = self.proj_out(h).index_select(1, video_indices)
            audio_output = self.audio_proj_out(h).index_select(1, audio_indices)
        if not return_dict:
            return (video_output, audio_output)
        return MiniMaxH3TransformerOutput(sample=video_output, audio_sample=audio_output)

    tr.forward = forward
    return {"fp8_blocks": sorted(fp8_blocks), "kernels": kernels, "ulysses_degree": world,
            "ulysses_pipe_groups": pipe_groups if pipe is not None else 0, "sync_free": sync_free,
            "row_balance": balance}
