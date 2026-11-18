"""4-GPU hybrid + kernel fusion + FB Cache, all in the spawned mgpu workers.

Installed from inside blocks.py (see the tail of that file) because mgpu runs its
ranks as SPAWNED workers -- anything patched in the parent never reaches them.

Stacks three orthogonal things:
  parallelism : CFG-parallel S1 + TDP S2 + distributed VAE   (measured 2.9x, lossless)
  kernels     : vendor rms_fma/gated_attention + our AdaLN/RoPE (1.19-1.23x, lossless)
  cache       : FB Cache at the LOWER threshold 0.06          (LOSSY, quality unverified)
"""
import os

_DONE = {"k": False, "c": False}


def install_kernels(model) -> dict:
    if _DONE["k"]:
        return {}
    _DONE["k"] = True
    from ltx_core.opt.bf16_fusion import FusedGatedAttention, FusedPostSA
    from ltx_core.opt.ours import FusedAdaZero, install_ada_cache, install_rope_index_cache, make_tables_resident
    # Vendor CUDA rms_norm_split_rope measured 1.088x vs our Triton 1.044x, and
    # ops_cpp builds now, so take theirs. Falls back to eager on shape mismatch.
    from ltx_core.opt.ours2 import FusedPreAttention, VendorPreAttention
    import ltx_core.model.transformer.transformer as _t
    import ltx_core.model.transformer.rope as _r

    # The two vendor ops below account for every SIGSEGV observed (rms_fma x3,
    # gated_attention x1) across configs and ranks. They ship coupled to fp8 and
    # are being driven on a bf16 path; rms_fma also mutates its input in place,
    # so surviving runs are not proof of correctness. Off by default.
    safe = os.environ.get("LTX_SAFE_KERNELS", "1") == "1"
    n = 0
    for mod in model.modules():
        if not safe and hasattr(mod, "post_sa_function"):
            mod.post_sa_function = FusedPostSA(); n += 1
        if not safe and hasattr(mod, "gated_attention_function"):
            mod.gated_attention_function = FusedGatedAttention(); n += 1
        if hasattr(mod, "ada_zero_function"):
            mod.ada_zero_function = FusedAdaZero(); n += 1
        if hasattr(mod, "preattention_function"):
            mod.preattention_function = (VendorPreAttention()
                if os.environ.get("LTX_ROPE", "vendor") == "vendor"
                else FusedPreAttention()); n += 1
    out = {"ops_swapped": n, "vendor_fma_gate": not safe,
           "ada_cache": install_ada_cache(_t),
           "rope_idx": install_rope_index_cache(_r), "tables": make_tables_resident(model)}
    _log(f"[stack] kernels {out}")
    assert n > 0, "no ops swapped"
    return out


def install_cache() -> dict:
    if _DONE["c"]:
        return {}
    _DONE["c"] = True
    import ltx_core.model.transformer.model as _m
    from ltx_core.opt.step_cache import CTRL, install
    install(_m)
    _policy = os.environ.get("LTX_CACHE_POLICY", "fbcache")
    assert _policy in ("fbcache", "teacache", "easycache"), (
        f"unknown LTX_CACHE_POLICY={_policy!r}; step_cache implements only "
        "fbcache / teacache / easycache and would use a meaningless signal")
    CTRL.configure(
        policy=_policy,
        threshold=float(os.environ.get("LTX_CACHE_THRESH", "0.06")),
        warmup=int(os.environ.get("LTX_CACHE_WARMUP", "1")),
        max_consecutive=int(os.environ.get("LTX_CACHE_MAXCONSEC", "3")),
    )
    CTRL.reset()
    CTRL.armed = True

    # begin_step() is normally driven by the denoiser wrapper, which the mgpu path
    # does not have. Without it the pass counter never resets, every call allocates a
    # fresh slot and the cache can never hit. Under CFG parallelism each rank does
    # exactly ONE forward per step, so advancing the step per forward is exact.
    orig_proc = _m.LTXModel._process_transformer_blocks

    def stepped(self, video, audio, perturbations):
        CTRL.begin_step(CTRL.step_index + 1)
        out = orig_proc(self, video, audio, perturbations)
        if CTRL.step_index in (10, 39):
            _log(f"[stack] step{CTRL.step_index} {CTRL.stats()}")
        return out

    _m.LTXModel._process_transformer_blocks = stepped
    _log(f"[stack] cache armed policy={CTRL.policy} thresh={CTRL.threshold} "
         f"maxconsec={CTRL.max_consecutive} warmup={CTRL.warmup}")

    # Log the skip count per rank. Single-GPU fb-006 skipped 30% and gained 1.40x
    # (1/(1-0.30)=1.43, consistent). The 4-GPU stack gained only 1.18x, implying a
    # ~15% effective skip rate -- so either the per-rank rate dropped, or the ranks
    # DISAGREE and a step is only cheap when all four skip together. Without the
    # counts that is a guess, so count.
    import atexit
    atexit.register(lambda: _log(f"[stack] cache stats: {CTRL.stats()}"))
    return {"cache": "fbcache"}


def _log(msg):
    print(msg, flush=True)
    f = os.environ.get("STAGE_TIME_FILE")
    if f:
        with open(f, "a") as fh:
            fh.write(msg + "\n")
