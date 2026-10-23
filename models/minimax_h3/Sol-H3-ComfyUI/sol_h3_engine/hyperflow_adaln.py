"""Exact AdaLN modulation table for HyperFlow-8 MiniMax-H3: drop the 26 GB of `adaln_proj` weights.

Every block's `adaln_proj(temb)` depends only on the step's distinct (timestep, endpoint) pairs, and HyperFlow's
8-step grid makes that set finite and known in advance. scripts/hyperflow/build_adaln_table.py records the pair
set of every forward the real pipeline runs and evaluates the original projection on each set with one call of the
same batch, in the original precision recipe (silu at temb's float32, cast to the BF16 projection after), so every
row is bit-identical to the live module (Sol-H3's install_exact_t2va_lookup rule). At run time each `adaln_proj`
becomes an AdaLNTable and `route_rows` selects the forward's rows by its exact float32 pair set; an unseen set
raises, so a schedule the table does not cover can never silently fall back to wrong modulation.
"""
import torch
from torch import nn


class AdaLNTable(nn.Module):
    """Drop-in for MiniMaxH3AdaLayerNormModulation reading precomputed outputs for the current forward's pair set.

    Rows are stored per *forward* (the full ordered tuple of distinct (t, r) pairs a step hands the transformer), each
    computed with one linear call of exactly that batch, as Sol-H3's exact lookup does: cuBLAS picks a different
    kernel per M, so evaluating all pairs as one batch differs by 1 BF16 ulp and that drift compounds over 8 steps.
    The table is a buffer, so block-level group offload moves it with its block like any weight.
    """

    def __init__(self, hidden_size, starts, table):
        super().__init__()
        self.hidden_size = hidden_size
        self.starts = starts  # {pair-tuple: (start row, count)}
        self.register_buffer("table", table.contiguous())  # [sum of rows, 6*H*M] BF16
        self.rows = None  # LongTensor of this forward's table rows on the table's device, set by route_rows()

    def forward(self, temb):
        # index_select with a tensor (not a Python slice) keeps a compiled block graph independent of the step.
        return self.table.index_select(0, self.rows).reshape(-1, 6 * self.hidden_size).chunk(6, dim=-1)


def route_rows(transformer):
    """Before each forward, find its exact pair tuple in the table; an unseen tuple is an error."""
    tables = [b.adaln_proj for b in transformer.transformer_blocks]
    starts = tables[0].starts
    embedder = transformer.time_embedder
    cache = {}

    def pre_forward(module, args, kwargs):
        key = tuple(zip(map(float, kwargs["timestep"].float().cpu().tolist()),
                        map(float, embedder._endpoint_timesteps.float().cpu().tolist())))
        if key not in starts:
            raise RuntimeError(f"AdaLN table has no entry for the pair set {key}; rebuild it for this schedule")
        start, count = starts[key]
        rows = cache.get(key)
        if rows is None:
            rows = cache[key] = torch.arange(start, start + count, device="cuda")
        for m in tables:
            m.rows = rows
        return None

    transformer.register_forward_pre_hook(pre_forward, with_kwargs=True)
