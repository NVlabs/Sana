"""Utilities for installing optimized forwards under Diffusers CP hooks."""

from __future__ import annotations


def with_cp_reapplied(transformer, action):
    from diffusers.hooks.context_parallel import apply_context_parallel, remove_context_parallel

    from .cp_plan import MINIMAX_H3_CP_PLAN

    parallel_config = getattr(transformer, "_parallel_config", None)
    cp_config = getattr(parallel_config, "context_parallel_config", None) if parallel_config else None
    if cp_config is not None:
        remove_context_parallel(transformer, MINIMAX_H3_CP_PLAN)
    try:
        return action()
    finally:
        if cp_config is not None:
            apply_context_parallel(transformer, cp_config, MINIMAX_H3_CP_PLAN)
