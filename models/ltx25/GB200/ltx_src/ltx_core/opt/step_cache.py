"""Stage-1 step caches for LTX-2.5: fbcache / teacache / easycache.

All three share one mechanism -- "decide, then maybe skip the remaining blocks and
reuse the cached whole-stack residual" -- and differ only in the *signal*:

* ``fbcache``   -- First-Block Cache: run block 0, use its residual as the signal.
* ``teacache``  -- timestep-modulated input proxy: the step-entry hidden state.
                   (Canonical TeaCache uses the AdaLN-modulated input *inside* block 0;
                   this uses the pre-block state, identical up to the per-block affine.
                   Documented deviation.)
* ``easycache`` -- accumulate input drift scaled by the last measured output/input
                   transformation rate.

Installing patches ``LTXModel._process_transformer_blocks``. The loop body is
REIMPLEMENTED rather than wrapped, because slicing ``transformer_blocks`` to skip
ahead would restart ``enumerate``'s ``block_idx`` -- and perturbations are indexed by
block (STG is configured on ``stg_blocks=[29]``), so a sliced loop would silently
apply STG to the wrong layer.

CFG note: stage 1 runs CFG+STG. State is keyed by pass ordinal within a step, reset
whenever the denoiser reports a new ``step_index``, so cond/uncond/perturbed passes
never share a cache slot.

Stage scoping: armed only around the stage-1 loop; stage 2 is untouched.
"""

from __future__ import annotations

import os
from dataclasses import replace

import torch

from ltx_core.opt import _stubs as _st
_bg = _st.BlockGraphStub()
_nv = _st.NVFP4Stub()
import torch.distributed as dist

from ltx_core.guidance.perturbations import PerturbationType


_BGRAPH = os.environ.get("LTX_BGRAPH", "0") == "1"


def _rel_l1(a: torch.Tensor, b: torch.Tensor) -> float:
    denom = b.abs().mean()
    if denom == 0:
        return float("inf")
    return float(((a - b).abs().mean() / denom).item())


class _PassState:
    __slots__ = ("prev_signal", "res_v", "res_a", "prev_in", "prev_out", "rate", "accum")

    def __init__(self) -> None:
        self.prev_signal: torch.Tensor | None = None
        self.res_v: torch.Tensor | None = None
        self.res_a: torch.Tensor | None = None
        self.prev_in: torch.Tensor | None = None
        self.prev_out: torch.Tensor | None = None
        self.rate: float | None = None
        self.accum: float = 0.0


class CacheController:
    def __init__(self) -> None:
        self.policy: str | None = None
        self.threshold = 0.0
        self.warmup = 1
        self.max_consecutive = 3
        self.armed = False
        # Explicit denoising-stage scope.  Token count is not sufficient under the
        # 4-GPU stack: Stage 1 CFGP keeps the full sequence on every rank while
        # Stage 2 2x2 TDP shards the 4x-larger latent, so both stages can expose the
        # exact same local token count.
        self.stage: int | None = None
        self.step_index = -1
        self._pass = 0
        self._states: dict[int, _PassState] = {}
        self._consec: dict[int, int] = {}
        self.n_calls = 0
        self.n_skipped = 0
        self.stage1_tokens: int | None = None
        self.n_local_skip = 0
        self.n_bypass = 0

    def configure(self, policy, threshold, warmup=1, max_consecutive=3) -> None:
        self.policy = policy
        self.threshold = float(threshold)
        self.warmup = int(warmup)
        self.max_consecutive = int(max_consecutive)

    def reset(self) -> None:
        self.stage = None
        self.step_index = -1
        self._pass = 0
        self._states.clear()
        self._consec.clear()
        self.n_calls = 0
        self.n_skipped = 0
        self.n_local_skip = 0
        self.n_bypass = 0
        self.stage1_tokens = None

    def begin_step(self, step_index: int, *, stage: int | None = None) -> None:
        if stage is not None:
            self.stage = int(stage)
        self.step_index = step_index
        self._pass = 0

    def stats(self) -> str:
        rate = 100.0 * self.n_skipped / max(1, self.n_calls)
        return (
            f"policy={self.policy} thresh={self.threshold} "
            f"calls={self.n_calls} skipped={self.n_skipped} ({rate:.1f}%) "
            f"local_wanted={self.n_local_skip} bypass={self.n_bypass}"
        )

    def take_pass(self):
        p = self._pass
        self._pass += 1
        st = self._states.get(p)
        if st is None:
            st = _PassState()
            self._states[p] = st
        return p, st

    def _unified(self, local: bool) -> bool:
        """Make every rank agree on the skip decision.

        Under CFG parallelism each rank owns one guidance pass and decides from its
        OWN context, so the ranks skip on different steps. Two consequences, measured:
        per-rank skip was 32.5% but the wall-clock gain was only 1.18x (a step is only
        cheap when ALL ranks skip), and -- worse -- the guidance combination mixed a
        freshly computed pass with another pass's stale residual, which is simply the
        wrong direction. AND-reduce so the ranks skip together or not at all.

        `local` already encodes "wants to AND is able to" (decide() returns False when
        there is no cached residual yet), so a rank can never be forced to reuse a
        residual it does not have.
        """
        self.n_local_skip += int(local)
        if not (dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1):
            return local
        t = torch.tensor([1 if local else 0], device="cuda", dtype=torch.int32)
        dist.all_reduce(t, op=dist.ReduceOp.MIN)   # MIN over {0,1} == logical AND
        return bool(t.item())

    def decide(self, p: int, st: _PassState, signal: torch.Tensor) -> bool:
        if self.step_index < self.warmup or st.prev_signal is None or st.res_v is None:
            return False
        if self._consec.get(p, 0) >= self.max_consecutive:
            return False
        d = _rel_l1(signal, st.prev_signal)
        if self.policy == "easycache":
            st.accum += d * (st.rate if st.rate is not None else 1.0)
        else:
            st.accum += d
        return st.accum < self.threshold


CTRL = CacheController()


def install(model_module) -> None:
    LTXModel = model_module.LTXModel
    orig = LTXModel._process_transformer_blocks
    if getattr(orig, "_ltx_cache_patched", False):
        return

    def patched(self, video, audio, perturbations):
        if not CTRL.armed or CTRL.policy in (None, "off"):
            return orig(self, video, audio, perturbations)

        # The requested optimization is Stage-1-only.  This explicit marker is
        # driven by the denoiser wrapper and remains correct when CFGP Stage 1 and
        # 2x2-TDP Stage 2 happen to have identical per-rank token counts.
        if CTRL.stage != 1:
            CTRL.n_bypass += 1
            return orig(self, video, audio, perturbations)

        # Stage scoping. The single-GPU runner armed the controller around the stage-1
        # loop only; the mgpu install has no such boundary, so stage-2 tensors reached
        # the cache and _rel_l1 compared 8208 tokens against stage-1's 6144. Latch the
        # first shape seen (that is stage 1) and bypass anything else.
        _tok = (video.x if video is not None else audio.x).shape[1]
        if CTRL.stage1_tokens is None:
            CTRL.stage1_tokens = _tok
        elif _tok != CTRL.stage1_tokens:
            # this return skips _unified() -> no all_reduce on this rank
            CTRL.n_bypass += 1
            return orig(self, video, audio, perturbations)

        p, st = CTRL.take_pass()
        CTRL.n_calls += 1
        v_in = video.x.clone() if video is not None else None
        a_in = audio.x.clone() if audio is not None else None
        entry = v_in if v_in is not None else a_in

        skipped = False
        decided = False
        if CTRL.policy != "fbcache":
            # Signal available before any block runs.
            decided = True
            skipped = CTRL._unified(CTRL.decide(p, st, entry))
            signal = entry

        _nv.GUARD["step"] = CTRL.step_index
        if not skipped and _BGRAPH:
            # Per-block graphs: the cache still picks what runs, it just picks
            # which graph to replay instead of which python loop iteration.
            try:
                store = self.__dict__.setdefault("_ltx_bgraphs", {})
                key = _bg._sig(video, audio, perturbations)
                ent = store.get(key)
                if ent is None:
                    ent = _bg.build(self, video, audio, perturbations)
                    store[key] = ent
                _bg.load_step(ent, video, audio, perturbations)
                _dense = _nv.step_is_dense()
                _bg.replay(ent, 0, _dense)
                video, audio = _bg.current(ent, video, audio)
                if not decided:
                    decided = True
                    cur = (video.x - v_in) if video is not None else (audio.x - a_in)
                    signal = cur
                    skipped = CTRL._unified(CTRL.decide(p, st, cur))
                if not skipped:
                    for _i in range(1, len(self.transformer_blocks)):
                        _bg.replay(ent, _i, _dense)
                    video, audio = _bg.current(ent, video, audio)
            except Exception as _e:
                # never silently degrade to eager while reporting as graphed
                if _bg.STATS["fail"] is None:
                    _bg.STATS["fail"] = f"{type(_e).__name__}: {str(_e)[:200]}"
                    print(f"[bgraph] capture/replay FAILED, eager from here: "
                          f"{_bg.STATS['fail']}", flush=True)
                raise

        elif not skipped:
            for block_idx, block in enumerate(self.transformer_blocks):
                if video is not None:
                    video = self.block_input_processor(
                        video,
                        perturbations,
                        block_idx,
                        self_attn_type=PerturbationType.SKIP_VIDEO_SELF_ATTN,
                        cross_attn_type=PerturbationType.SKIP_A2V_CROSS_ATTN,
                    )
                if audio is not None:
                    audio = self.block_input_processor(
                        audio,
                        perturbations,
                        block_idx,
                        self_attn_type=PerturbationType.SKIP_AUDIO_SELF_ATTN,
                        cross_attn_type=PerturbationType.SKIP_V2A_CROSS_ATTN,
                    )
                video, audio = block(video=video, audio=audio)

                if not decided and block_idx == 0:
                    # fbcache: block 0's residual is the signal.
                    decided = True
                    cur = (video.x - v_in) if video is not None else (audio.x - a_in)
                    signal = cur
                    if CTRL._unified(CTRL.decide(p, st, cur)):
                        skipped = True
                        break

        if skipped:
            CTRL.n_skipped += 1
            CTRL._consec[p] = CTRL._consec.get(p, 0) + 1
            if video is not None and st.res_v is not None:
                video = replace(video, x=v_in + st.res_v)
            if audio is not None and st.res_a is not None:
                audio = replace(audio, x=a_in + st.res_a)
            st.prev_signal = signal.detach().clone()
            return video, audio

        CTRL._consec[p] = 0
        if video is not None:
            st.res_v = (video.x - v_in).detach()
        if audio is not None:
            st.res_a = (audio.x - a_in).detach()
        if CTRL.policy == "easycache" and st.prev_in is not None:
            out_now = video.x if video is not None else audio.x
            # rate = how much the OUTPUT moved between steps, per unit of INPUT
            # movement. Both terms must be step-to-step deltas; comparing the
            # output against the previous INPUT folds in the per-step denoising
            # transformation itself, which is far larger than the drift and pins
            # rate >> 1 forever.
            din = _rel_l1(entry, st.prev_in)
            if st.prev_out is not None:
                dout = _rel_l1(out_now, st.prev_out)
            else:
                dout = din  # first step with no previous output: rate 1.0
            st.rate = (dout / din) if din > 0 else 1.0
        st.prev_in = entry.detach().clone()
        if CTRL.policy == "easycache":
            st.prev_out = (video.x if video is not None else audio.x).detach().clone()
        st.prev_signal = signal.detach().clone()
        st.accum = 0.0
        return video, audio

    patched._ltx_cache_patched = True
    LTXModel._process_transformer_blocks = patched


def configure_from_env() -> None:
    CTRL.configure(
        policy=os.environ.get("LTX_CACHE", "off"),
        threshold=float(os.environ.get("LTX_CACHE_THRESH", "0.06")),
        warmup=int(os.environ.get("LTX_CACHE_WARMUP", "1")),
        max_consecutive=int(os.environ.get("LTX_CACHE_MAXCONSEC", "3")),
    )
