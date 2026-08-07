"""The rank-8 AdaLN factorisation the pruned checkpoint ships, and the re-point onto it.

Sibling of `optimized/adaln.py`. The released model reaches its modulation through
`t -> time_proj -> time_embedder -> silu -> Linear(2688 -> 96768)`, once per block. The pruned
file deletes `time_embedder` outright, ships a shared `adaln_t_table` of shape `(1025, 8)` and
narrows every AdaLN projection to `Linear(8 -> 96768)`.

That is a rank-8 factorisation of the timestep -> modulation map, and it is exact enough to
treat as free: every AdaLN consumes the *same* `silu(temb)`, and a scalar `t` only traces a
one-dimensional curve through it, so a small shared basis reproduces the whole trajectory.
`checks/validate_adaln.py` measures 1.98e-4 mean relative error against the released BF16
weights — finer than BF16's own spacing.
"""

from __future__ import annotations

import torch
from torch import nn

FP8_DTYPE = torch.float8_e4m3fn
FP8_MAX = 448.0

class _DtypeCarrier(nn.Module):
    """Exposes a `.weight` so the model's `time_embedder.linear_1.weight.dtype` still resolves.

    `MiniMaxH3Transformer3DModel.forward` reads that attribute to decide what to cast the
    sinusoidal embedding to. Carrying a scalar here keeps the replacement drop-in and leaves
    the model's own `forward` unedited.
    """

    def __init__(self, dtype: torch.dtype) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.zeros((), dtype=dtype), requires_grad=False)


class AdaLnTableLookup(nn.Module):
    """Stands in for `time_embedder`: maps `t` in `[0, 1]` to its row of `adaln_t_table`.

    The table has 1025 rows, i.e. a uniform grid of step `1/1024` over the closed unit
    interval. `round(t * 1024)` is the indexing convention: measured against the released
    BF16 weights it reproduces block 0's modulation to a mean relative error of 2.0e-4,
    where `floor`, a 1000-point grid and a reversed grid give 2.3e-4, 3.4e-3 and 1.1e-1.
    The residual is storage precision, not the rank-8 truncation — the true timestep ->
    modulation map has its 8th singular value at 3.9e-2 against a leading 1.5e+3, so rank 8
    captures it to 1.6e-10 and the pruned path is in fact finer than the BF16 original.
    """

    def __init__(self, table: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("table", table.float(), persistent=False)
        self.linear_1 = _DtypeCarrier(torch.float32)

    def forward(self, timestep: torch.Tensor) -> torch.Tensor:
        rows = self.table.shape[0] - 1
        index = (timestep.float() * rows).round().long().clamp_(0, rows)
        return self.table.index_select(0, index.reshape(-1))


def patch_pruned_adaln(transformer: nn.Module) -> None:
    """Re-point the model's timestep path at the table and drop the absorbed `silu`.

    `time_proj` becomes a pass-through and `time_embedder` becomes the table lookup, so the
    model's own `forward` is left untouched: it still computes `temb` once and hands the same
    tensor to all 51 AdaLN projections, which are now `Linear(8 -> ...)`.
    """
    from diffusers.models.transformers import transformer_minimax_h3 as h3

    table = transformer._adaln_t_table
    transformer.time_proj = nn.Identity()
    transformer.time_embedder = AdaLnTableLookup(table)

    def modulation_forward(self, temb: torch.Tensor):
        # No `silu`: the reference applied it to the 2688-dim embedding, and the factorisation
        # that produced `adaln_t_table` absorbed it into the table.
        #
        # The projection is float32 (39M parameters across the model, so the precision is
        # free), but its result has to come back to the block stack's dtype — leaving it
        # float32 would promote every `hidden_states * scale` downstream and silently run the
        # whole block in float32.
        out_dtype = getattr(self.linear, "_out_dtype", temb.dtype)
        temb = self.linear(temb.to(self.linear.weight.dtype)).to(out_dtype)
        temb = temb.view(-1, 6 * self.hidden_size)
        return temb.chunk(6, dim=-1)

    def out_forward(self, hidden_states, temb, timestep_indices):
        out_dtype = getattr(self.linear, "_out_dtype", hidden_states.dtype)
        modulation = self.linear(temb.to(self.linear.weight.dtype)).to(out_dtype)
        shift, scale = modulation.chunk(2, dim=-1)
        hidden_states = self.norm(hidden_states)
        return hidden_states * (1.0 + scale.index_select(0, timestep_indices)) + shift.index_select(
            0, timestep_indices
        )

    h3.MiniMaxH3AdaLayerNormModulation.forward = modulation_forward
    h3.MiniMaxH3AdaLayerNormOut.forward = out_forward


