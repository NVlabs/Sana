"""Stable public wrapper around the evidence-bound SM100 PISA2 runner."""

from experiments.pisa2.native_bf16_lean6_routeidx_g512_cursor_ballotscatter_fusedroute_n128_pair_tmemp_hybrid_massreuse_terminalo_runner import (
    make_native_bf16_lean6_routeidx_g512_cursor_ballotscatter_fusedroute_n128_pair_tmemp_hybrid_massreuse_terminalo_runner as make_pisa2_sm100,
)

__all__ = ["make_pisa2_sm100"]
