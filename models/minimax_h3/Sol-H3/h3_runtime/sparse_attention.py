"""Sol-Attn for MiniMax-H3, wired inside the Ulysses exchange.

This is a rewrite against the *released* kernel (`sol-attn` 0.5.0, NVlabs/Sana `sol-engine`
@cee25847a). The previous version targeted a pre-release copy of the same package and reinvented,
badly, three things the release had already settled. Recording them, because each one was a
measurement that looked plausible and was wrong:

  * **Morton reordering per attention call.** The released `install_wan_morton_forward` docstring
    says why not: only self-attention is order-sensitive, so the permutation belongs once at the
    block stack, and doing it per call cost more than the kernel saved. What the H3 integration
    then shows is that H3 needs *no* reordering at all — the packed video tail is already a
    contiguous grid-ordered block, and the routing works on it directly.
  * **A joint `[video | text]` path with an fp32 log-sum-exp merge.** The release replaces it with
    an exact-KV sink: `sink_start`/`sink_tokens` mark a contiguous KV range that every query
    attends exactly, at 64-token block granularity, inside the same single kernel pass. No
    second attention, no merge, no `[1, heads, video, prefix]` fp32 temporary.
  * **Calibrating `tau` per shape.** The validated H3 policy passes `tau=1.0` directly. The
    calibration this file used to call returned an empty route set on H3's shape and the
    configuration silently fell back to dense.

The released README states the integration contract in one sentence, and it is the whole design:

    "The sink does not change query routing: an MMDiT integration should still compute valid text
     query rows with dense attention and use Sol-Attn for image/video query rows."

So: one `sol_attn` call over the packed sequence with the prefix marked as an exact KV sink, then
the prefix's own query rows overwritten with dense SDPA. H3 packs

    [ text | conditioning video | audio | target video ]

and the prefix — everything before the target video tail — is contiguous, which is all the sink
needs. It is not only text: the audio rows carry the soundtrack and are themselves *generated*
(the model returns an audio velocity for them). The H100 handoff recorded one prompt whose picture
scored best of its set while its dialogue fell apart, so `sink_mode` defaults to `prefix` — sink
and densely recompute all 951 prefix rows, not just the 537 text rows. Against the reference's
text-only policy that is 6 extra exact KV blocks out of 598 (~1% density) and 414 extra dense
query rows out of 38247 (~1% of the attention). Set `H3_SOL_SINK_MODE=text` to reproduce the
reference policy exactly.

**Where this goes under Ulysses.** After `_packed_qkv_all_to_all` every rank holds the whole
sequence for its own heads — the only point in the model where a sequence-level operator can be
correct — so this plugs into `ulysses_custom`'s attention slot, not into a diffusers processor.
Nothing is reconciled across ranks: routing is decided per (query, head) and Ulysses gives each
rank a disjoint set of heads.

This is an approximation, and a visual metric alone will rate it too highly on this model.
"""

from __future__ import annotations

import math
import os
import sys

import torch
import torch.nn.functional as F

from .sol_residual import merge_sol_residual


_RELEASE_ROOT = os.environ.get(
    "H3_SOL_ATTN_ROOT",
    os.path.join(os.path.dirname(__file__), "third_party"),
)


def _import_kernel():
    """`sol_attn(q, k, v, *, scale, tau, thresh_type, kv_splits, sink_tokens, sink_start)`.

    The release vendors its CuTe dependencies under `sol_attn._vendor.flash_attn`, a private
    namespace, so the environment's real flash-attn can no longer shadow them — the failure that
    needed a `sol_attn_colmask` entry on PYTHONPATH before. CuTe DSL still has to be >= 4.5 and
    still has to win from process start, which the sbatch handles.
    """
    if _RELEASE_ROOT not in sys.path:
        sys.path.insert(0, _RELEASE_ROOT)
    _install_fused_preprocess()
    from sol_attn import sol_attn

    return sol_attn


def _import_preprocess():
    """Resolve Sol's block summaries/router without requiring the kernel import first."""
    if _RELEASE_ROOT not in sys.path:
        sys.path.insert(0, _RELEASE_ROOT)
    _install_fused_preprocess()
    from sol_attn.preprocess import BLOCK_SIZE, prepare
    return BLOCK_SIZE, prepare


def _install_fused_preprocess() -> None:
    """Opt into the measured bit-exact K/V summary fusion."""
    if os.environ.get("H3_SOL_FUSED_PREPROCESS", "0") != "1":
        return
    from sol_preprocess_fused import install

    if install():
        print("[sol_attn_h3] installed fused Sol K/V preprocess", flush=True)


def _import_bsa():
    """Resolve the pinned cuDNN frontend BSA API only when selected."""
    from cudnn import BSA
    return BSA


class H3SparseAttention:
    """The attention slot of the packed Ulysses path.

    Call shape is what `ulysses_custom` produces after the exchange: `(rows_full, heads_local,
    head_dim)` — the whole packed sequence, this rank's heads.
    """

    def __init__(self, backend: str = "sol", tau: float = 1.0,
                 thresh_type: str = "diag", dense_steps: int = 1,
                 dense_layers: int = 2, sink_mode: str = "prefix", gate: bool = False):
        self.tau = float(
            os.environ.get("H3_SOL_TAU", os.environ.get("SOL_ATTN_TAU", tau))
        )
        self.thresh_type = os.environ.get("SOL_ATTN_THRESH_TYPE", thresh_type)
        self.dense_steps = int(os.environ.get("SOL_ATTN_FIRST_DENSE_STEPS", dense_steps))
        self.dense_layers = int(os.environ.get("SOL_ATTN_FIRST_DENSE_LAYERS", dense_layers))
        self.sink_mode = os.environ.get("H3_SOL_SINK_MODE", sink_mode).lower()
        self.gate_enabled = os.environ.get("SOL_ATTN_CORRECTNESS_GATE", "1" if gate else "0") == "1"
        self.packed_input = os.environ.get("H3_SOL_PACKED_INPUT", "1") == "1"
        self.backend = os.environ.get("H3_SOL_BACKEND", backend).lower()
        bucket_override = os.environ.get("H3_SOL_COMPILE_BUCKET_SIZE")
        self.compile_bucket_size = int(
            bucket_override if bucket_override is not None
            else "4096"
        )
        if self.backend not in {"sol", "sol_bsa"}:
            raise ValueError(
                f"H3_SOL_BACKEND must be 'sol' or 'sol_bsa', got {self.backend!r}"
            )
        if self.sink_mode not in {"prefix", "text", "text_audio"}:
            raise ValueError(
                "H3_SOL_SINK_MODE must be 'prefix', 'text', or 'text_audio', "
                f"got {self.sink_mode!r}"
            )
        if self.sink_mode == "text_audio" and self.backend != "sol_bsa":
            raise ValueError("H3_SOL_SINK_MODE=text_audio requires H3_SOL_BACKEND=sol_bsa")
        if self.compile_bucket_size < 0 or self.compile_bucket_size % 64:
            raise ValueError("H3_SOL_COMPILE_BUCKET_SIZE must be zero or a multiple of 64")
        if (self.backend == "sol" and self.compile_bucket_size
                and os.environ.get("H3_SOL_FUSED_PREPROCESS", "0") == "1"):
            raise ValueError(
                "H3_SOL_FUSED_PREPROCESS=1 does not support prompt-stable compile buckets"
            )

        self.video_start: int | None = None
        self.sink_start = 0
        self.sink_tokens = 0
        self.sink_block_mask: torch.Tensor | None = None
        self.sequence_length: int | None = None
        self._layout_key: tuple | None = None

        self.step = -1
        self.layer = 0
        self.request = -1
        self._prev_timestep: float | None = None
        self._direction = 0
        self._sparse_at_request_start = 0
        self.last_request_sparse_calls: int | None = None

        self.sparse_calls = 0
        self.dense_calls = 0
        self.declined: dict[str, int] = {}
        self._gated_shapes: set = set()
        self._density: dict | None = None
        self.gate_stats: dict | None = None
        # A configuration that asked for sparse attention and silently got dense is a wrong
        # measurement wearing the right label.
        self.strict = os.environ.get("SOL_ATTN_STRICT", "1") == "1"

    @property
    def native_int8_qkv(self) -> bool:
        """Whether the currently selected backend consumes compressed QKV directly."""
        # Resident benchmarks change backend after construction, so this must
        # follow the live backend rather than cache its initial value.
        return self.backend == "sol_bsa"

    def use_int8_qkv_for_next_call(self) -> bool:
        """Whether the upcoming call is genuinely sparse and should use compressed QKV."""
        if not self.native_int8_qkv:
            return False
        rows = self.sequence_length if self.sequence_length is not None else -1
        return self._declined_contract(rows, torch.bfloat16, 128, self.layer) is None

    # -- request-static layout, read off the model's own tensors -------------------------

    def observe(self, video_indices, text_indices, audio_indices, position_ids, timestep) -> None:
        """Resolve the prefix/video split, and detect the start of a new request.

        The step counter must reset per request. It did not before: the clock only ever
        incremented, and each configuration runs a warmup request and then a measured one, so the
        measured request began at step 49 and ran fully sparse while `stats()` still reported ten
        dense steps. Two signals are available here as forward kwargs — the packed layout, and the
        timestep, which is monotone within a request and jumps back when the next one starts.
        """
        sequence_length = int(position_ids.shape[0])
        timestep_max = float(timestep.detach().float().max().item()) if timestep is not None else None

        layout_key = (
            sequence_length,
            id(video_indices),
            id(text_indices),
            id(audio_indices),
            id(position_ids),
        )
        layout_changed = layout_key != self._layout_key
        reversal = self._is_reversal(timestep_max)

        if layout_changed:
            self._layout_key = layout_key
            self.sequence_length = sequence_length
            self.video_start = _target_video_start(video_indices, sequence_length)
            self.sink_start, self.sink_tokens = self._sink_range(text_indices, audio_indices)
            self.sink_block_mask = self._sink_blocks(
                text_indices, audio_indices, sequence_length, position_ids.device
            )
            self._density = None

        if layout_changed or reversal:
            self._close_request()
            self.request += 1
            self.step = 0
            self._direction = 0
        else:
            self.step += 1
        self.layer = 0

    def _is_reversal(self, timestep_max: float | None) -> bool:
        """True when the timestep moves against the direction *this request* established.

        The direction is measured, not assumed. H3's scheduler builds `sigmas = linspace(1, 0, N)`
        and then `timesteps = 1 - sigmas`, so its timestep *rises* from 0 toward 1 across a
        request — the opposite of the falling convention the reference runtime relies on. Hardcoding
        "a rise means a new request" made the reset fire on every single step, pinning `step` at 0
        so that every call declined as `warmup_step`: both sparse configurations would have
        measured dense attention and reported it under a sparse label, with the strict guard
        silent because `warmup_step` is a legitimate decline. Reading the direction off the first
        transition costs nothing and cannot be wrong about which way this model counts.
        """
        prev, self._prev_timestep = self._prev_timestep, timestep_max
        if timestep_max is None or prev is None:
            return False
        delta = timestep_max - prev
        if abs(delta) <= 1e-6:
            return False
        if self._direction == 0:
            self._direction = 1 if delta > 0 else -1
            return False
        return (delta > 0) != (self._direction > 0)

    def _close_request(self) -> None:
        """Refuse to let a request finish having never reached the kernel.

        Every failure this file has had ends the same way: the configuration runs, produces a
        plausible latency, and reports it as `sparse`. The per-call decline reasons cannot catch
        it, because the decline that does the damage — `warmup_step` — is a legitimate one. It is
        only wrong in aggregate, so the check has to be in aggregate.
        """
        if self.request < 0:
            return
        # Recorded unconditionally: `stats()` otherwise reports totals summed over the warmup and
        # the measured request, and only the measured one is what the latency describes.
        self.last_request_sparse_calls = self.sparse_calls - self._sparse_at_request_start
        self._sparse_at_request_start = self.sparse_calls
        if self.step + 1 <= self.dense_steps or self.last_request_sparse_calls > 0:
            return
        message = (f"request {self.request} ran {self.step + 1} forwards past dense_steps="
                   f"{self.dense_steps} without one sparse call; declines={self.declined}")
        if self.strict:
            raise RuntimeError(message)
        print(f"[sol_attn_h3] WARNING: {message}", flush=True)

    def _sink_range(self, text_indices, audio_indices) -> tuple[int, int]:
        """The contiguous KV range every query keeps exact.

        `prefix` is everything before the target-video tail — text, any conditioning video, and
        audio. `text` is the reference's policy. The kernel takes any contiguous range inside
        `[0, T]` and applies exactness at 64-token block granularity, rounding outward: a
        951-token sink covers blocks [0, 15), so nine target-video keys become exact too.
        """
        if self.sink_mode == "text_audio":
            return 0, 0
        if self.sink_mode == "text":
            if text_indices is None or text_indices.numel() == 0:
                return 0, 0
            lo, hi = int(text_indices.min().item()), int(text_indices.max().item())
            if hi - lo + 1 != text_indices.numel():
                raise ValueError("MiniMax-H3 text rows are not contiguous; cannot form a KV sink")
            return lo, hi - lo + 1
        return 0, int(self.video_start or 0)

    def _sink_blocks(self, text_indices, audio_indices, sequence_length: int,
                     device: torch.device) -> torch.Tensor | None:
        """BSA blocks containing text or audio, excluding Ref2VA visual references."""
        if self.sink_mode != "text_audio":
            return None
        parts = [
            indices.to(device=device, dtype=torch.long)
            for indices in (text_indices, audio_indices)
            if indices is not None and indices.numel()
        ]
        if not parts:
            raise ValueError("text_audio sink mode received no text or audio indices")
        token_indices = torch.cat(parts)
        if bool(((token_indices < 0) | (token_indices >= sequence_length)).any()):
            raise ValueError("text/audio indices fall outside the packed sequence")
        block_indices = torch.div(token_indices, 64, rounding_mode="floor").unique()
        mask = torch.zeros(
            (math.ceil(sequence_length / 64),), dtype=torch.bool, device=device
        )
        mask[block_indices] = True
        return mask.contiguous()

    # -- the attention itself -----------------------------------------------------------

    def _declined(self, q, layer: int) -> str | None:
        """Why this call cannot take the sparse path, or None if it can.

        A reason rather than a boolean, deliberately. Silent conditions once produced a row
        labelled `sparse` that ran entirely dense and still read like a plausible measurement.
        """
        return self._declined_contract(q.shape[0], q.dtype, q.shape[-1], layer)

    def _declined_contract(self, rows_full: int, dtype: torch.dtype,
                           head_dim: int, layer: int) -> str | None:
        """The sparse contract without requiring materialized BF16 Q/K/V views."""
        if self.video_start is None:
            return "no video_start: the transformer pre-hook never saw video_indices"
        if self.sequence_length != rows_full:
            return f"sequence length {self.sequence_length} != attention rows {rows_full}"
        if not 0 < self.video_start < rows_full:
            return f"video_start {self.video_start} outside (0, {rows_full})"
        # The kernel's own contract, checked here so the reason names the cause.
        if dtype != torch.bfloat16:
            return f"kernel requires bfloat16, got {dtype}"
        if head_dim != 128:
            return f"kernel requires head_dim 128, got {head_dim}"
        # These two are the configuration asking for dense, not failures.
        if self.step < self.dense_steps:
            return "warmup_step"
        # `layer` is the caller's captured pre-increment index, not `self.layer`. Reading the
        # member here counted from 1, so `dense_layers=2` left only block 0 dense.
        if layer < self.dense_layers:
            return "dense_layer"
        return None

    def __call__(self, q, k, v):
        """`(rows_full, heads_local, head_dim)` in, same shape out."""
        layer = self.layer
        self.layer += 1

        reason = self._declined(q, layer)
        if reason is not None:
            if reason not in ("warmup_step", "dense_layer"):
                if self.strict:
                    raise RuntimeError(f"sparse attention declined: {reason}")
                if reason not in self.declined:
                    print(f"[sol_attn_h3] running dense: {reason}", flush=True)
            self.declined[reason] = self.declined.get(reason, 0) + 1
            self.dense_calls += 1
            return _dense(q, k, v)

        # On SM100/103 the CuTe TMA descriptors can consume these views directly from the packed
        # Ulysses receive buffer.  This removes three copy kernels and their Q/K/V intermediates;
        # keep a switch for a clean end-to-end A/B and for architectures whose backend has not yet
        # been validated with the interleaved stride.
        if self.packed_input:
            qb, kb, vb = (x.unsqueeze(0) for x in (q, k, v))
        else:
            qb, kb, vb = (x.unsqueeze(0).contiguous() for x in (q, k, v))

        if self.backend == "sol_bsa":
            if self.gate_enabled:
                self._run_bsa_gate(qb, kb, vb)
            out, density = _sol_bsa_attention(
                qb, kb, vb,
                tau=self.tau,
                thresh_type=self.thresh_type,
                sink_start=self.sink_start,
                sink_tokens=self.sink_tokens,
                sink_block_mask=self.sink_block_mask,
                collect_stats=self._density is None,
                compile_bucket_size=self.compile_bucket_size,
            )
            if density is not None:
                self._density = density
                print(f"[sol_attn_h3] SOL/BSA route density {self._density}", flush=True)
        else:
            sol_attn = _import_kernel()
            if self.gate_enabled:
                self._run_gate(sol_attn, qb, kb, vb)
            if self._density is None:
                self._density = _estimate_density(qb, kb, vb, tau=self.tau,
                                                  thresh_type=self.thresh_type,
                                                  sink_start=self.sink_start,
                                                  sink_tokens=self.sink_tokens,
                                                  compile_bucket_size=self.compile_bucket_size)
                print(f"[sol_attn_h3] route density {self._density}", flush=True)

            out = sol_attn(
                qb, kb, vb,
                tau=self.tau,
                thresh_type=self.thresh_type,
                kv_splits=1,                 # B200/SM100 supports 1 only
                sink_start=self.sink_start,
                sink_tokens=self.sink_tokens,
                compile_bucket_size=self.compile_bucket_size or None,
            )
        # SOL keeps the released sparse-query policy and recomputes sink queries
        # densely.  The BSA route selects every KV block for sink query blocks,
        # so its existing output is already exact and needs no second SDPA.
        if self.sink_tokens and self.backend == "sol":
            lo, hi = self.sink_start, self.sink_start + self.sink_tokens
            out[:, lo:hi] = _dense(qb[0, lo:hi], kb[0], vb[0]).unsqueeze(0)

        self.sparse_calls += 1
        return out.squeeze(0)

    def int8_qkv(self, packet):
        """Consume aligned int8 QKV packets without a logical-size BF16 copy."""
        if not self.native_int8_qkv:
            raise RuntimeError("native int8 QKV is implemented for SOL/BSA only")

        tokens, heads = int(packet.shape[0]), int(packet.shape[1])
        layer = self.layer
        self.layer += 1
        reason = self._declined_contract(tokens, torch.bfloat16, 128, layer)

        from .comm_quant import dequantize_qkv_padded

        if reason is not None:
            if reason not in ("warmup_step", "dense_layer"):
                if self.strict:
                    raise RuntimeError(f"sparse attention declined: {reason}")
                if reason not in self.declined:
                    print(f"[sol_attn_h3] running dense: {reason}", flush=True)
            self.declined[reason] = self.declined.get(reason, 0) + 1
            self.dense_calls += 1
            qb, kb, vb = dequantize_qkv_padded(packet, tokens, heads, tokens)
            return _dense(qb[0], kb[0], vb[0])

        capacity = _compile_capacity(tokens, self.compile_bucket_size)
        qb, kb, vb = dequantize_qkv_padded(packet, tokens, heads, capacity)
        if self.gate_enabled:
            self._run_bsa_gate(qb[:, :tokens], kb[:, :tokens], vb[:, :tokens])
        out, density = _sol_bsa_attention(
            qb, kb, vb,
            tau=self.tau,
            thresh_type=self.thresh_type,
            sink_start=self.sink_start,
            sink_tokens=self.sink_tokens,
            sink_block_mask=self.sink_block_mask,
            collect_stats=self._density is None,
            compile_bucket_size=self.compile_bucket_size,
            valid_tokens=tokens,
            inputs_bucketed=True,
        )
        if density is not None:
            self._density = density
            print(f"[sol_attn_h3] SOL/BSA route density {self._density}", flush=True)
        self.sparse_calls += 1
        return out.squeeze(0)

    def _run_bsa_gate(self, qb, kb, vb) -> None:
        """Check block64 BSA against dense SDPA with every block selected."""
        key = (_compile_capacity(qb.shape[1], self.compile_bucket_size),
               *tuple(int(x) for x in qb.shape[2:]))
        if key in self._gated_shapes:
            return
        BSA = _import_bsa()
        heads = min(int(os.environ.get("SOL_ATTN_GATE_HEADS", str(qb.shape[2]))), qb.shape[2])
        qs, ks, vs = (x[:, :, :heads].contiguous() for x in (qb, kb, vb))
        blocks = math.ceil(qs.shape[1] / 64)
        ids = torch.arange(blocks, device=qs.device, dtype=torch.int32)
        indices = ids.view(1, 1, 1, blocks).expand(1, heads, blocks, blocks).contiguous()
        sizes = torch.full((blocks,), 64, device=qs.device, dtype=torch.int32)
        sizes[-1] = qs.shape[1] - (blocks - 1) * 64
        got = BSA.block_sparse_attention_forward(
            qs, ks, vs, indices,
            block_sparse_num=blocks,
            block_sizes=sizes,
            sparse_block_size=64,
            layout="bshd",
            kv_splits=1,
        )["o_tensor"]
        want = _dense(qs[0], ks[0], vs[0]).unsqueeze(0)
        torch.cuda.synchronize(qb.device)

        diff = (got.float() - want.float()).abs()
        stats = {
            "max_abs": float(diff.max().item()),
            "mean_abs": float(diff.mean().item()),
            "rel_l2": float((torch.linalg.vector_norm(got.float() - want.float())
                             / torch.linalg.vector_norm(want.float()).clamp_min(1e-12)).item()),
        }
        limits = {
            "max_abs": float(os.environ.get("SOL_ATTN_GATE_MAX_ABS",
                                            "0.15" if qb.shape[1] >= 32768 else "0.08")),
            "mean_abs": float(os.environ.get("SOL_ATTN_GATE_MEAN_ABS", "0.002")),
            "rel_l2": float(os.environ.get("SOL_ATTN_GATE_REL_L2", "0.005")),
        }
        passed = _agree(all(stats[name] <= limit for name, limit in limits.items()))
        self.gate_stats = {"backend": "sol_bsa", "passed": passed, "shape": list(qs.shape), **stats}
        print(f"[sol_attn_h3] BSA correctness gate {'PASS' if passed else 'FAIL'} {stats} "
              f"limits {limits}", flush=True)
        if not passed:
            raise RuntimeError(f"cuDNN BSA correctness gate failed on real QKV: {stats} > {limits}")
        self._gated_shapes.add(key)

    def _run_gate(self, sol_attn, qb, kb, vb) -> None:
        """Route everything and check the kernel against SDPA, on the real QKV, once per shape.

        `tau=-1000` drives every block above threshold, so the kernel's own arithmetic is what is
        being measured rather than the routing policy. A random-tensor probe cannot do this: it
        answers a question about the kernel, not about this model's tensors at this shape.
        """
        key = tuple(int(x) for x in qb.shape[1:])
        if key in self._gated_shapes:
            return
        # All heads by default, not one. `preprocess.prepare` autotunes its Triton kernels on a key
        # of `T` alone, so a first call at one head would cache a configuration chosen for a 1-head
        # grid and every production 7-head call at the same `T` would inherit it. Gating at the
        # production head count also means the kernel's own compile cache — keyed on head count —
        # is warmed by the gate instead of being paid twice, and the check covers every head
        # rather than head 0.
        heads = min(int(os.environ.get("SOL_ATTN_GATE_HEADS", str(qb.shape[2]))), qb.shape[2])
        qs, ks, vs = (x[:, :, :heads].contiguous() for x in (qb, kb, vb))
        got = sol_attn(
            qs, ks, vs,
            tau=-1000.0,
            thresh_type=self.thresh_type,
            kv_splits=1,
            compile_bucket_size=self.compile_bucket_size or None,
        )
        want = _dense(qs[0], ks[0], vs[0]).unsqueeze(0)
        torch.cuda.synchronize(qb.device)

        diff = (got.float() - want.float()).abs()
        stats = {
            "max_abs": float(diff.max().item()),
            "mean_abs": float(diff.mean().item()),
            "rel_l2": float((torch.linalg.vector_norm(got.float() - want.float())
                             / torch.linalg.vector_norm(want.float()).clamp_min(1e-12)).item()),
        }
        limits = {
            "max_abs": float(os.environ.get("SOL_ATTN_GATE_MAX_ABS",
                                            "0.15" if qb.shape[1] >= 32768 else "0.08")),
            "mean_abs": float(os.environ.get("SOL_ATTN_GATE_MEAN_ABS", "0.002")),
            "rel_l2": float(os.environ.get("SOL_ATTN_GATE_REL_L2", "0.005")),
        }
        passed = all(stats[name] <= limit for name, limit in limits.items())
        # Ulysses gives each rank a disjoint set of heads, so this verdict is genuinely per-rank
        # and the ranks can disagree. Raising on one alone would leave the other seven blocked in
        # the next all-to-all: `bench_scale` catches the exception and moves to the next config, so
        # no process exits non-zero and `--kill-on-bad-exit` never fires — the allocation would sit
        # there until the three-hour walltime expired. Agree first, then fail together.
        passed = _agree(passed)
        self.gate_stats = {"passed": passed, "shape": list(qs.shape), **stats}
        print(f"[sol_attn_h3] correctness gate {'PASS' if passed else 'FAIL'} {stats} "
              f"limits {limits}", flush=True)
        if not passed:
            raise RuntimeError(f"Sol-Attn correctness gate failed on real QKV: {stats} > {limits}")
        self._gated_shapes.add(key)

    def stats(self) -> dict:
        total = self.sparse_calls + self.dense_calls
        default_wire = "int8_qkv" if self.native_int8_qkv else "bf16"
        wire_dtype = os.environ.get("H3_ULYSSES_COMM_DTYPE", default_wire).lower()
        default_output_wire = "fp8" if self.native_int8_qkv else "bf16"
        output_wire_dtype = os.environ.get(
            "H3_ULYSSES_OUTPUT_DTYPE", default_output_wire
        ).lower()
        int8_scope = os.environ.get("H3_ULYSSES_INT8_SCOPE", "all").lower()
        measured_total = max(self.step + 1, 0) * self.layer
        measured_sparse = self.last_request_sparse_calls or 0
        measured_int8 = (
            (measured_total if int8_scope == "all" else measured_sparse)
            if wire_dtype == "int8_qkv" else 0
        )
        measured_bf16 = measured_total - measured_int8
        measured_output_fp8 = 0
        if output_wire_dtype == "int8":
            measured_output_int8 = measured_total
        elif output_wire_dtype == "fp8":
            measured_output_int8 = 0
            measured_output_fp8 = measured_total
        elif output_wire_dtype == "match_qkv":
            measured_output_int8 = measured_int8
        else:
            measured_output_int8 = 0
        measured_output_bf16 = (
            measured_total - measured_output_int8 - measured_output_fp8
        )
        if measured_total:
            from .comm_quant import FP8_OUTPUT_PACKET, OUTPUT_PACKET

            qkv_wire_ratio = (
                measured_int8 * 400 + measured_bf16 * 768
            ) / (measured_total * 768)
            output_wire_ratio = (
                measured_output_int8 * OUTPUT_PACKET
                + measured_output_fp8 * FP8_OUTPUT_PACKET
                + measured_output_bf16 * 256
            ) / (measured_total * 256)
            attention_comm_ratio = (
                measured_int8 * 400
                + measured_bf16 * 768
                + measured_output_int8 * OUTPUT_PACKET
                + measured_output_fp8 * FP8_OUTPUT_PACKET
                + measured_output_bf16 * 256
            ) / (measured_total * (768 + 256))
        else:
            qkv_wire_ratio = 1.0
            output_wire_ratio = 1.0
            attention_comm_ratio = 1.0
        return {
            "sparse_calls": self.sparse_calls,
            "dense_calls": self.dense_calls,
            # Totals cover the warmup request too; this is the measured one alone. With the cache
            # active most block-stack calls never happen, so the totals stop being a proxy.
            "sparse_calls_measured_request": self.last_request_sparse_calls,
            "sparse_fraction": round(self.sparse_calls / total, 4) if total else None,
            "requests": self.request + 1,
            "last_step": self.step,
            "video_start": self.video_start,
            "sink_mode": self.sink_mode,
            "sink_start": self.sink_start,
            "sink_tokens": self.sink_tokens,
            "sink_blocks": (
                int(self.sink_block_mask.sum().item())
                if self.sink_block_mask is not None else None
            ),
            "sequence_length": self.sequence_length,
            "tau": self.tau,
            "thresh_type": self.thresh_type,
            "backend": self.backend,
            "qkv_wire_dtype": wire_dtype,
            "qkv_wire_policy": (
                f"int8_qkv_{int8_scope}" if wire_dtype == "int8_qkv" else "bf16"
            ),
            "qkv_int8_calls_measured_request": measured_int8,
            "qkv_bf16_calls_measured_request": measured_bf16,
            "qkv_wire_ratio": round(qkv_wire_ratio, 6),
            "output_wire_dtype": output_wire_dtype,
            "output_int8_calls_measured_request": measured_output_int8,
            "output_fp8_calls_measured_request": measured_output_fp8,
            "output_bf16_calls_measured_request": measured_output_bf16,
            "output_wire_ratio": round(output_wire_ratio, 6),
            "attention_comm_ratio": round(attention_comm_ratio, 6),
            "packed_input": self.packed_input,
            "compile_bucket_size": self.compile_bucket_size,
            "dense_steps": self.dense_steps,
            "dense_layers": self.dense_layers,
            "route_density": self._density,
            "gate": self.gate_stats,
            "declined": self.declined,
        }


def _agree(passed: bool) -> bool:
    """The gate's verdict, reduced across ranks so a failure aborts all of them or none."""
    import torch.distributed as dist

    if not (dist.is_available() and dist.is_initialized()):
        return passed
    flag = torch.tensor([1 if passed else 0], device="cuda", dtype=torch.int32)
    dist.all_reduce(flag, op=dist.ReduceOp.MIN)
    return bool(flag.item())


def _dense(q, k, v):
    """SDPA on `(rows, heads, dim)`, same layout out. The kernel's scale default matches."""
    out = F.scaled_dot_product_attention(
        q.transpose(0, 1).unsqueeze(0), k.transpose(0, 1).unsqueeze(0),
        v.transpose(0, 1).unsqueeze(0), dropout_p=0.0, is_causal=False)
    return out.squeeze(0).transpose(0, 1)


@torch.no_grad()
def _sol_bsa_attention(qb, kb, vb, *, tau, thresh_type, sink_start, sink_tokens,
                       collect_stats: bool, sink_block_mask: torch.Tensor | None = None,
                       compile_bucket_size: int = 0,
                       valid_tokens: int | None = None,
                       inputs_bucketed: bool = False):
    """Run Sol routing through cuDNN BSA and restore Sol's omitted-block residual."""
    BSA = _import_bsa()
    tokens = qb.shape[1] if valid_tokens is None else int(valid_tokens)
    if not 0 < tokens <= qb.shape[1]:
        raise ValueError(f"valid_tokens must be in [1, {qb.shape[1]}], got {tokens}")
    if inputs_bucketed:
        expected = _compile_capacity(tokens, compile_bucket_size)
        if qb.shape[1] != expected:
            raise ValueError(
                f"prebucketed QKV capacity {qb.shape[1]} does not match {expected}"
            )
        q_run, k_run, v_run = qb, kb, vb
    else:
        q_run, k_run, v_run = _bucketed_qkv(qb, kb, vb, compile_bucket_size)
    indices, nums, sizes, density, exact_mask, kc, vc = _build_bsa_metadata(
        q_run, k_run, v_run,
        tau=tau,
        thresh_type=thresh_type,
        sink_start=sink_start,
        sink_tokens=sink_tokens,
        sink_block_mask=sink_block_mask,
        collect_stats=collect_stats,
        valid_tokens=tokens,
        metadata_blocks=q_run.shape[1] // 64,
    )
    # Enter the kernel in native BHSD using views of the packed Ulysses buffer.
    # The patched rank-5 K/V TMA path preserves these strides and the true
    # partial-tail extent, so neither input nor output materialization is needed.
    qh, kh, vh = (tensor.transpose(1, 2) for tensor in (q_run, k_run, v_run))
    result = BSA.block_sparse_attention_forward(
        qh, kh, vh, indices,
        block_sparse_num=indices.shape[-1],
        block_sizes=sizes,
        q2k_block_nums=nums,
        sparse_block_size=64,
        layout="bhsd",
        kv_splits=1,
    )
    return merge_sol_residual(
        qb[:, :tokens],
        kc,
        vc,
        exact_mask,
        result["o_tensor"][:, :, :tokens],
        result["lse_tensor"][:, :, :tokens],
    ), density


@torch.no_grad()
def _build_bsa_metadata(qb, kb, vb, *, tau, thresh_type, sink_start, sink_tokens,
                        collect_stats: bool, sink_block_mask: torch.Tensor | None = None,
                        valid_tokens: int | None = None,
                        metadata_blocks: int | None = None):
    """Turn Sol's threshold route into cuDNN's per-query-block compact KV lists."""
    BLOCK_SIZE, prepare = _import_preprocess()

    batch, capacity_tokens, heads, head_dim = qb.shape
    tokens = capacity_tokens if valid_tokens is None else int(valid_tokens)
    if not 0 < tokens <= capacity_tokens:
        raise ValueError(f"valid_tokens must be in [1, {capacity_tokens}], got {tokens}")
    blocks = math.ceil(tokens / BLOCK_SIZE)
    scale = head_dim ** -0.5
    kc, vc, threshold, q_bar = prepare(
        qb,
        kb,
        vb,
        scale=scale,
        tau=tau,
        thresh_type=thresh_type,
        valid_tokens=tokens,
        return_q_bar=True,
    )
    kc = kc[:, :blocks]
    vc = vc[:, :blocks]
    threshold = threshold[:, :blocks]

    if q_bar is None:
        # The exact-threshold path retains its original BF16 pooled query, so
        # use the existing FP32 reduction for route scoring in that mode.
        pad = blocks * BLOCK_SIZE - tokens
        q_logical = qb[:, :tokens]
        q_pad = F.pad(q_logical, (0, 0, 0, 0, 0, pad)) if pad else q_logical
        block_lengths = torch.full(
            (blocks,), float(BLOCK_SIZE), device=qb.device, dtype=torch.float32
        )
        block_lengths[-1] = tokens - (blocks - 1) * BLOCK_SIZE
        q_bar = q_pad.view(batch, blocks, BLOCK_SIZE, heads, head_dim).float().sum(2)
        q_bar.div_(block_lengths.view(1, blocks, 1, 1))
    else:
        q_bar = q_bar[:, :blocks]

    scores = torch.einsum("bqhd,bkhd->bqkh", q_bar, kc.float())
    if sink_block_mask is not None:
        if tuple(sink_block_mask.shape) != (blocks,):
            raise ValueError(
                f"sink_block_mask must have shape {(blocks,)}, got {tuple(sink_block_mask.shape)}"
            )
        sink_blocks = int(sink_block_mask.sum().item())
        first = last = 0
    elif sink_tokens:
        first = sink_start // BLOCK_SIZE
        last = math.ceil((sink_start + sink_tokens) / BLOCK_SIZE)
        sink_blocks = last - first
    else:
        first = last = 0

    # Keep cuDNN's Q/K/V and metadata descriptors prompt-stable. Only the logical
    # route participates in the residual; padded query blocks are throwaway work
    # and receive one harmless key so every BSA row remains well formed.
    metadata_blocks = blocks if metadata_blocks is None else int(metadata_blocks)
    if metadata_blocks < blocks:
        raise ValueError(
            f"metadata_blocks={metadata_blocks} is smaller than logical blocks={blocks}"
        )
    from .bsa_metadata import route_and_compact

    scale_log2 = scale * math.log2(math.e)
    route, indices, nums = route_and_compact(
        scores,
        threshold,
        scale_log2=scale_log2,
        sink_first=first,
        sink_last=last,
        metadata_blocks=metadata_blocks,
        sink_block_mask=sink_block_mask,
    )
    sizes = torch.full((metadata_blocks,), BLOCK_SIZE, device=qb.device, dtype=torch.int32)
    sizes[blocks - 1] = tokens - (blocks - 1) * BLOCK_SIZE

    density = None
    if collect_stats:
        threshold_density = float(
            (scores * scale_log2 > threshold[:, :, None, :]).float().mean().item()
        )
        density = {
            "blocks": blocks,
            "sink_blocks": sink_blocks,
            "threshold_density": round(threshold_density, 5),
            "effective_density": round(float(route.float().mean().item()), 5),
            "route_count_min": int(route.sum(-1).min().item()),
            "route_count_mean": round(float(route.sum(-1).float().mean().item()), 2),
            "route_count_max": int(route.sum(-1).max().item()),
            "metadata_capacity": int(indices.shape[-1]),
        }
    return indices, nums, sizes, density, route, kc, vc


@torch.no_grad()
def _estimate_density(qb, kb, vb, *, tau, thresh_type, sink_start, sink_tokens,
                      compile_bucket_size: int = 0) -> dict:
    """What fraction of KV blocks the routing keeps, on one head of the real tensors.

    Reported because the failure this file exists to avoid is a sparse configuration that quietly
    computes dense attention: a density near 1.0 says the routing is not routing, and a density of
    0.0 says it collapsed. Sampled from the first sparse call, so it is a route statistic and not
    an all-layer average.
    """
    BLOCK_SIZE, prepare = _import_preprocess()

    heads = min(int(os.environ.get("SOL_ATTN_DENSITY_SAMPLE_HEADS", str(qb.shape[2]))),
                qb.shape[2])
    q, k, v = (x[:, :, :heads].contiguous() for x in (qb, kb, vb))
    tokens = q.shape[1]
    q_pre, k_pre, v_pre = _bucketed_qkv(q, k, v, compile_bucket_size)
    scale = q.shape[-1] ** -0.5
    kc, _, threshold = prepare(
        q_pre,
        k_pre,
        v_pre,
        scale=scale,
        tau=tau,
        thresh_type=thresh_type,
        valid_tokens=tokens,
    )

    blocks = math.ceil(tokens / BLOCK_SIZE)
    kc = kc[:, :blocks]
    threshold = threshold[:, :blocks]
    padded = F.pad(q, (0, 0, 0, 0, 0, blocks * BLOCK_SIZE - tokens))
    counts = torch.full((blocks,), float(BLOCK_SIZE), device=q.device, dtype=torch.float32)
    counts[-1] = tokens - (blocks - 1) * BLOCK_SIZE
    q_bar = padded.view(q.shape[0], blocks, BLOCK_SIZE, heads, q.shape[3]).float().sum(2)
    q_bar = q_bar / counts.view(1, blocks, 1, 1)

    scores = torch.einsum("bqhd,bkhd->bqkh", q_bar, kc.float()).mul_(scale * math.log2(math.e))
    routed = scores > threshold[:, :, None, :]
    threshold_density = float(routed.float().mean().item())

    ids = torch.arange(blocks, device=q.device)
    routed |= ((ids[:, None] - ids[None, :]).abs() <= 1)[None, :, :, None]   # local band
    sink_blocks = 0
    if sink_tokens:
        first = sink_start // BLOCK_SIZE
        last = math.ceil((sink_start + sink_tokens) / BLOCK_SIZE)
        routed[:, :, first:last, :] = True
        sink_blocks = last - first
    return {
        "blocks": blocks,
        "sink_blocks": sink_blocks,
        "threshold_density": round(threshold_density, 5),
        "effective_density": round(float(routed.float().mean().item()), 5),
    }


def _compile_capacity(tokens: int, bucket_size: int) -> int:
    """Stable descriptor length for prompt-dependent packed sequences."""
    if bucket_size <= 0:
        return int(tokens)
    return ((int(tokens) + bucket_size - 1) // bucket_size) * bucket_size


def _bucketed_qkv(q, k, v, bucket_size: int):
    """Pad only descriptor capacity; callers continue to pass the logical token count."""
    capacity = _compile_capacity(q.shape[1], bucket_size)
    if capacity == q.shape[1]:
        return q, k, v
    if _RELEASE_ROOT not in sys.path:
        sys.path.insert(0, _RELEASE_ROOT)
    from sol_attn.interface import _pad_to_bucket

    # H3 supplies packed Q/K/V views, and the patched BSA descriptors preserve
    # their strides.  Reuse Sol's one-allocation, one-copy padding path instead
    # of clearing and copying three independent tensors.
    return tuple(_pad_to_bucket(q, k, v, bucket_size))


def _target_video_start(video_indices, sequence_length: int) -> int:
    """First row of the contiguous target-video tail.

    `video_indices` is ascending and lists the conditioning rows before the target rows, with the
    audio block in between, so the tail is the last contiguous run.
    """
    if video_indices is None or video_indices.numel() == 0:
        return sequence_length
    steps = video_indices[1:] - video_indices[:-1]
    breaks = (steps != 1).nonzero()
    start = int(breaks[-1].item()) + 1 if breaks.numel() else 0
    return int(video_indices[start].item())


def install(transformer, backend: str = "sol", tau: float = 1.0,
            thresh_type: str = "diag", dense_steps: int = 1,
            dense_layers: int = 2, sink_mode: str = "prefix", **_ignored) -> H3SparseAttention:
    """Attach the layout probe and the step clock. Returns the attention callable.

    Hand the result to `ulysses_custom.install(..., attention_fn=...)`; this does not install
    itself into the model's attention, because under context parallelism the only correct place
    for it is inside the exchange.
    """
    sparse = H3SparseAttention(
        backend=backend,
        tau=tau,
        thresh_type=thresh_type,
        dense_steps=dense_steps,
        dense_layers=dense_layers,
        sink_mode=sink_mode,
    )

    def pre_hook(_module, _args, kwargs):
        if kwargs.get("video_indices") is not None and kwargs.get("position_ids") is not None:
            sparse.observe(kwargs["video_indices"], kwargs.get("text_indices"),
                           kwargs.get("audio_indices"), kwargs["position_ids"],
                           kwargs.get("timestep"))
        return None

    sparse._hook_handle = transformer.register_forward_pre_hook(pre_hook, with_kwargs=True)
    transformer._h3_sparse_attention = sparse
    return sparse


def uninstall(transformer) -> None:
    sparse = getattr(transformer, "_h3_sparse_attention", None)
    if sparse is not None:
        handle = getattr(sparse, "_hook_handle", None)
        if handle is not None:
            handle.remove()
        transformer._h3_sparse_attention = None
