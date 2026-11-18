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


# The released package. Kept out of the deployed the SGLang runtime checkout on purpose: that tree is
# pinned at b42e35a7 (2026-07-21), which is what the lossless 22.482 s baseline was measured on,
# and the Sol-Attn kernel landed a week later. Only the sparse path resolves through here.
_RELEASE_ROOT = os.environ.get(
    "H3_SOL_ATTN_ROOT",
    "/home/yitongl/code/agent_deploy/sol-attn-release/techniques/sparse_backends",
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
    from sol_attn import sol_attn

    return sol_attn


class H3SparseAttention:
    """The attention slot of the packed Ulysses path.

    Call shape is what `ulysses_custom` produces after the exchange: `(rows_full, heads_local,
    head_dim)` — the whole packed sequence, this rank's heads.
    """

    def __init__(self, tau: float = 1.0, thresh_type: str = "diag", dense_steps: int = 10,
                 dense_layers: int = 2, sink_mode: str = "prefix", gate: bool = True):
        self.tau = float(os.environ.get("SOL_ATTN_TAU", tau))
        self.thresh_type = os.environ.get("SOL_ATTN_THRESH_TYPE", thresh_type)
        self.dense_steps = int(os.environ.get("SOL_ATTN_FIRST_DENSE_STEPS", dense_steps))
        self.dense_layers = int(os.environ.get("SOL_ATTN_FIRST_DENSE_LAYERS", dense_layers))
        self.sink_mode = os.environ.get("H3_SOL_SINK_MODE", sink_mode)
        self.gate_enabled = os.environ.get("SOL_ATTN_CORRECTNESS_GATE", "1" if gate else "0") == "1"

        self.video_start: int | None = None
        self.sink_start = 0
        self.sink_tokens = 0
        self.sequence_length: int | None = None

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
        self.strict = os.environ.get("SOL_ATTN_STRICT", "0") == "1"

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

        layout_changed = sequence_length != self.sequence_length
        reversal = self._is_reversal(timestep_max)

        if layout_changed:
            self.sequence_length = sequence_length
            self.video_start = _target_video_start(video_indices, sequence_length)
            self.sink_start, self.sink_tokens = self._sink_range(text_indices, audio_indices)
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
        if self.sink_mode == "text":
            if text_indices is None or text_indices.numel() == 0:
                return 0, 0
            lo, hi = int(text_indices.min().item()), int(text_indices.max().item())
            if hi - lo + 1 != text_indices.numel():
                raise ValueError("MiniMax-H3 text rows are not contiguous; cannot form a KV sink")
            return lo, hi - lo + 1
        return 0, int(self.video_start or 0)

    # -- the attention itself -----------------------------------------------------------

    def _declined(self, q, layer: int) -> str | None:
        """Why this call cannot take the sparse path, or None if it can.

        A reason rather than a boolean, deliberately. Silent conditions once produced a row
        labelled `sparse` that ran entirely dense and still read like a plausible measurement.
        """
        rows_full = q.shape[0]
        if self.video_start is None:
            return "no video_start: the transformer pre-hook never saw video_indices"
        if self.sequence_length != rows_full:
            return f"sequence length {self.sequence_length} != attention rows {rows_full}"
        if not 0 < self.video_start < rows_full:
            return f"video_start {self.video_start} outside (0, {rows_full})"
        # The kernel's own contract, checked here so the reason names the cause.
        if q.dtype != torch.bfloat16:
            return f"kernel requires bfloat16, got {q.dtype}"
        if q.shape[-1] != 128:
            return f"kernel requires head_dim 128, got {q.shape[-1]}"
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

        sol_attn = _import_kernel()
        # The kernel wants contiguous BTHD. `ulysses_custom` hands over strided views into the
        # packed QKV buffer (SDPA needs only stride(-1) == 1), so this copies: three times
        # rows x heads_local x 128 bf16, ~68 MB each at CP=8, which is under 0.1 ms against a
        # 6.7 ms attention.
        qb, kb, vb = (x.unsqueeze(0).contiguous() for x in (q, k, v))

        if self.gate_enabled:
            self._run_gate(sol_attn, qb, kb, vb)
        if self._density is None:
            self._density = _estimate_density(qb, kb, vb, tau=self.tau,
                                              thresh_type=self.thresh_type,
                                              sink_start=self.sink_start,
                                              sink_tokens=self.sink_tokens)
            print(f"[sol_attn_h3] route density {self._density}", flush=True)

        out = sol_attn(
            qb, kb, vb,
            tau=self.tau,
            thresh_type=self.thresh_type,
            kv_splits=1,                     # B200/SM100 supports 1 only
            sink_start=self.sink_start,
            sink_tokens=self.sink_tokens,
        )
        # The sink makes the prefix exact as *keys*. Its own *queries* still route sparsely, and
        # the release is explicit that an MMDiT integration must recompute them densely.
        if self.sink_tokens:
            lo, hi = self.sink_start, self.sink_start + self.sink_tokens
            out[:, lo:hi] = _dense(qb[0, lo:hi], kb[0], vb[0]).unsqueeze(0)

        self.sparse_calls += 1
        return out.squeeze(0)

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
        got = sol_attn(qs, ks, vs, tau=-1000.0, thresh_type=self.thresh_type, kv_splits=1)
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
            "sequence_length": self.sequence_length,
            "tau": self.tau,
            "thresh_type": self.thresh_type,
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
def _estimate_density(qb, kb, vb, *, tau, thresh_type, sink_start, sink_tokens) -> dict:
    """What fraction of KV blocks the routing keeps, on one head of the real tensors.

    Reported because the failure this file exists to avoid is a sparse configuration that quietly
    computes dense attention: a density near 1.0 says the routing is not routing, and a density of
    0.0 says it collapsed. Sampled from the first sparse call, so it is a route statistic and not
    an all-layer average.
    """
    from sol_attn.preprocess import BLOCK_SIZE, prepare

    heads = min(int(os.environ.get("SOL_ATTN_DENSITY_SAMPLE_HEADS", str(qb.shape[2]))),
                qb.shape[2])
    q, k, v = (x[:, :, :heads].contiguous() for x in (qb, kb, vb))
    scale = q.shape[-1] ** -0.5
    kc, _, threshold = prepare(q, k, v, scale=scale, tau=tau, thresh_type=thresh_type)

    tokens = q.shape[1]
    blocks = math.ceil(tokens / BLOCK_SIZE)
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


def install(transformer, tau: float = 1.0, thresh_type: str = "diag", dense_steps: int = 10,
            dense_layers: int = 2, sink_mode: str = "prefix", **_ignored) -> H3SparseAttention:
    """Attach the layout probe and the step clock. Returns the attention callable.

    Hand the result to `ulysses_custom.install(..., attention_fn=...)`; this does not install
    itself into the model's attention, because under context parallelism the only correct place
    for it is inside the exchange.
    """
    sparse = H3SparseAttention(tau=tau, thresh_type=thresh_type, dense_steps=dense_steps,
                               dense_layers=dense_layers, sink_mode=sink_mode)

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
