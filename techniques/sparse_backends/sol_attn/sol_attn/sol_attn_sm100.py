"""Stable public wrapper around the evidence-bound SM100 SOL Attention runner."""

from experiments.sol_attn.native_bf16_lean6_routeidx_g512_cursor_ballotscatter_fusedroute_n128_pair_tmemp_hybrid_massreuse_terminalo_runner import (
    make_native_bf16_lean6_routeidx_g512_cursor_ballotscatter_fusedroute_n128_pair_tmemp_hybrid_massreuse_terminalo_runner as make_sol_attn_sm100,
)

__all__ = ["make_sol_attn_sm100"]
