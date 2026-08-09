"""MiniMax-H3 acceleration line #3: diffusion step caching, made safe under context parallelism.

Two things have to be supplied before any of diffusers' caches can run on MiniMax-H3, and the
second one is a correctness issue rather than a convenience.

1. Block registration
---------------------
The cache hooks look their host block up in `TransformerBlockRegistry`, and upstream registers no
entry for `MiniMaxH3TransformerBlock`. The registry exposes a public `register`, so the entry is
added at runtime here instead of patching the vendored source — the same discipline the CP plan
follows. `MiniMaxH3TransformerBlock.forward` returns a bare tensor rather than a tuple, which the
cache hooks already branch on, and its first parameter is `hidden_states`, which is what
`_get_parameter_from_args_kwargs` needs.

2. The skip decision must be collective
---------------------------------------
FirstBlockCache decides whether to skip the remaining blocks from
`(residual - prev_residual).abs().mean() / prev_residual.abs().mean()`, evaluated on whatever
`hidden_states_residual` the hook sees. Under context parallelism that tensor is the rank's **local
shard**, so each rank computes a different ratio and they can land on opposite sides of the
threshold. The ranks that skip stop calling attention while the ranks that continue enter the
Ulysses all-to-all — a collective with missing participants, i.e. a hang, or silent divergence if
it happens not to hang.

`make_decision_collective` replaces that ratio with the global one. Both means divide by the same
element count, so the global ratio is just `sum|residual - prev| / sum|prev|` reduced over ranks —
one all-reduce of two scalars per step. That makes the decision both *identical* across ranks and
*correct*: it is the metric FirstBlockCache intends, measured over the whole sequence rather than
over one shard.

The caches themselves are approximations, unlike lines #1 and #2, so this line owns a quality gate:
every configuration is compared against the lossless CP output produced in the same job, on the
same node, from the same seed.
"""

from __future__ import annotations

import torch
import torch.distributed as dist


def register_h3_blocks() -> bool:
    """Teach the cache hooks about `MiniMaxH3TransformerBlock`. Idempotent."""
    from diffusers.hooks._helpers import TransformerBlockMetadata, TransformerBlockRegistry
    from diffusers.models.transformers.transformer_minimax_h3 import MiniMaxH3TransformerBlock

    try:
        TransformerBlockRegistry.get(MiniMaxH3TransformerBlock)
        return False
    except Exception:
        pass

    TransformerBlockRegistry.register(
        model_class=MiniMaxH3TransformerBlock,
        metadata=TransformerBlockMetadata(
            # The block returns a bare tensor, so the index is only consulted on the tuple path.
            # There is no separate encoder stream: MiniMax-H3 packs text into the one sequence.
            return_hidden_states_index=0,
            return_encoder_hidden_states_index=None,
        ),
    )
    return True


def make_decision_collective() -> None:
    """Reduce FirstBlockCache's skip metric over the context-parallel ranks. Idempotent.

    Without this, `_should_compute_remaining_blocks` compares a per-shard ratio against the
    threshold and ranks can disagree — see the module docstring.
    """
    from diffusers.hooks.first_block_cache import FBCHeadBlockHook

    if getattr(FBCHeadBlockHook, "_h3_collective_decision", False):
        return

    def _should_compute_remaining_blocks(self, hidden_states_residual: torch.Tensor) -> bool:
        shared_state = self.state_manager.get_state()
        if shared_state.head_block_residual is None:
            return True

        prev = shared_state.head_block_residual
        # Sums rather than means: the two means share a denominator, so the ratio of the summed
        # numerator to the summed denominator is exactly the global ratio once reduced.
        stats = torch.stack([(hidden_states_residual - prev).abs().sum(), prev.abs().sum()])
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(stats, op=dist.ReduceOp.SUM)
        diff = (stats[0] / stats[1].clamp(min=1e-8)).item()
        return diff > self.threshold

    FBCHeadBlockHook._should_compute_remaining_blocks = _should_compute_remaining_blocks
    FBCHeadBlockHook._h3_collective_decision = True


def with_cp_reapplied(transformer, action):
    """Run `action` with the context-parallel hooks temporarily removed, then put them back on top.

    `HookRegistry.register_hook` wraps whatever `forward` is currently installed, so the hook
    registered *last* ends up outermost. Enabling a cache after context parallelism therefore puts
    the cache outside the CP split: the cache hook reads the caller's full-length `hidden_states`
    (38247) while the inner forward returns the rank's shard (9562), and `output - original` fails
    on the shape. Job 5815878 died exactly there, on all four FirstBlockCache thresholds.

    The order that works is cache first, CP outermost — CP splits the inputs, then the cache sees a
    consistent 9562 in and 9562 out. Since CP is enabled once at load and caches are toggled per
    configuration, the hooks are lifted and re-applied around every toggle. Only the hooks move;
    `_parallel_config` on the model and its attention processors is left in place.
    """
    from diffusers.hooks.context_parallel import apply_context_parallel, remove_context_parallel

    from cp_plan import MINIMAX_H3_CP_PLAN

    parallel_config = getattr(transformer, "_parallel_config", None)
    cp_config = getattr(parallel_config, "context_parallel_config", None) if parallel_config else None

    if cp_config is not None:
        remove_context_parallel(transformer, MINIMAX_H3_CP_PLAN)
    try:
        return action()
    finally:
        if cp_config is not None:
            apply_context_parallel(transformer, cp_config, MINIMAX_H3_CP_PLAN)


def bind_cache_context(transformer, name: str = "cond") -> None:
    """Give the cache hooks a context to store their state under. Idempotent.

    Caches that keep per-inference state look it up through `HookRegistry._set_context`, and
    upstream pipelines establish it by wrapping each denoiser call in
    `with transformer.cache_context("cond"):`. MiniMax-H3's modular denoise loop does not — it was
    never wired for caching upstream — so TaylorSeer failed with "No context is set" for every
    configuration in job 5815878. One transformer call is one denoising step, so entering the
    context around `forward` is exactly the scope upstream uses.
    """
    if getattr(transformer, "_h3_cache_context_bound", False):
        return

    original = transformer.forward

    def wrapped(*args, **kwargs):
        with transformer.cache_context(name):
            return original(*args, **kwargs)

    transformer.forward = wrapped
    transformer._h3_cache_context_bound = True


def invalidate_child_registry_cache(transformer) -> None:
    """Drop `HookRegistry`'s memoized child-registry list after the hook tree changes.

    `_set_context` reaches submodule state through `_get_child_registries()`, which walks
    `named_modules()` once and then caches the result forever — a deliberate optimization (the walk
    costs milliseconds per call on a large model) with no invalidation.

    That is fine when a model is configured once, and wrong for a sweep. The first `cache_context`
    call snapshots whichever modules already carried a `_diffusers_hook`; a cache enabled *later*
    registers hooks on modules that were not in that snapshot, so `_set_context` never reaches them
    and their state managers raise "No context is set". TaylorSeer failed exactly this way in jobs
    5815878 and 5816129, after FirstBlockCache had already populated the snapshot.
    """
    from diffusers.hooks import HookRegistry

    registry = HookRegistry.check_if_exists_or_initialize(transformer)
    if getattr(registry, "_child_registries_cache", None) is not None:
        registry._child_registries_cache = None


def enable_cache(transformer, method: str, **kwargs) -> dict:
    """Turn on one caching technique. Returns the settings actually applied."""
    from diffusers.hooks import FirstBlockCacheConfig, TaylorSeerCacheConfig

    register_h3_blocks()
    bind_cache_context(transformer)

    if method == "first_block_cache":
        make_decision_collective()
        threshold = float(kwargs.get("threshold", 0.05))
        with_cp_reapplied(transformer, lambda: transformer.enable_cache(FirstBlockCacheConfig(threshold=threshold)))
        invalidate_child_registry_cache(transformer)
        return {"method": method, "threshold": threshold}

    if method == "taylorseer":
        # Interval-driven rather than signal-driven: every rank refreshes on the same step indices,
        # so it needs no reduction to stay consistent across the context-parallel group.
        config = TaylorSeerCacheConfig(
            cache_interval=int(kwargs.get("cache_interval", 4)),
            disable_cache_before_step=int(kwargs.get("disable_cache_before_step", 3)),
            max_order=int(kwargs.get("max_order", 1)),
        )
        with_cp_reapplied(transformer, lambda: transformer.enable_cache(config))
        invalidate_child_registry_cache(transformer)
        return {
            "method": method,
            "cache_interval": config.cache_interval,
            "disable_cache_before_step": config.disable_cache_before_step,
            "max_order": config.max_order,
        }

    raise ValueError(f"unknown cache method: {method}")


def disable_cache(transformer) -> None:
    """Remove the cache, keeping the context-parallel hooks outermost for the next configuration."""
    if getattr(transformer, "is_cache_enabled", False):
        with_cp_reapplied(transformer, lambda: transformer.disable_cache())
        invalidate_child_registry_cache(transformer)


def video_psnr(reference: list, config: list) -> dict:
    """PSNR of a generated clip against the lossless reference from the same seed.

    Frames arrive as PIL images. A cache that skips too aggressively shows up here as a low PSNR,
    but the reverse does not hold: diffusion trajectories that diverge can still be clean, equally
    valid samples, so this number gates regressions rather than certifying quality on its own.
    """
    import numpy as np

    if reference is None or config is None or len(reference) != len(config):
        return {"frames": None, "psnr_db": None, "note": "frame count mismatch"}

    total, worst = 0.0, float("inf")
    for ref_frame, cand_frame in zip(reference, config):
        ref = np.asarray(ref_frame, dtype=np.float64)
        cand = np.asarray(cand_frame, dtype=np.float64)
        mse = float(((ref - cand) ** 2).mean())
        psnr = float("inf") if mse == 0 else 10.0 * float(np.log10(255.0**2 / mse))
        total += psnr
        worst = min(worst, psnr)
    n = len(reference)
    return {
        "frames": n,
        "psnr_db": round(total / n, 3) if total != float("inf") else None,
        "min_frame_psnr_db": round(worst, 3) if worst != float("inf") else None,
    }


def audio_mse(reference: torch.Tensor, config: torch.Tensor) -> dict:
    """Audio is generated jointly with the video, so it needs its own check.

    A cache that perturbs the packed sequence can leave every per-frame video metric clean while
    desynchronizing or degrading the soundtrack — the failure mode the other registered models in
    this repository cannot have.
    """
    if reference is None or config is None:
        return {"samples": None, "mse": None}
    ref = reference.detach().to(torch.float64).cpu()
    cand = config.detach().to(torch.float64).cpu()
    if ref.shape != cand.shape:
        return {"samples": None, "mse": None, "note": f"shape mismatch {tuple(ref.shape)} vs {tuple(cand.shape)}"}
    return {"samples": int(ref.numel()), "mse": round(float(((ref - cand) ** 2).mean()), 12)}
