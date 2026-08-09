"""Precompute MiniMax-H3's AdaLN modulation and drop the projection weights.

Roughly 13B of MiniMax-H3's 33B parameters sit in the per-block `adaln_proj`, a
`Linear(2688 -> 6 * 5376 * 3)` that every one of the 50 blocks evaluates on every denoising
step. Its input is the timestep embedding alone: a `(num_timesteps, 2688)` tensor with a
handful of rows that depends on nothing but the sampling schedule, which the pipeline fixes
before the loop starts. So the whole modulation table for the whole trajectory is knowable
up front, and the model card says as much -- "the AdaLN modulation outputs can be
precomputed and cached, these parameters do not need to be loaded for inference-only
deployment" -- but the diffusers reference recomputes it 2500 times per video.

What this buys, in order of size:

* **~24 GB of GPU memory.** 26 GB of `adaln_proj` weights are replaced by a ~1.5 GB table
  (50 steps x 50 blocks x 9 rows x 32256 values, bfloat16). The denoiser drops from 61.7 GB
  to roughly 37 GB, which is what lets it share one 80 GB H100 with the VAEs instead of
  needing a card to itself.
* **~26 GB of HBM reads per step.** Each block streamed 520 MB of weights to produce nine
  rows of output -- a pure bandwidth tax on an operation with no arithmetic intensity.

The precompute runs one GEMM per (block, step) with exactly the shapes the reference uses,
rather than one batched GEMM per block over all steps. That is slower by a few tens of
milliseconds, once, and in exchange the cached values are bitwise identical to what the
unmodified model would have computed -- so this technique needs no quality gate.
"""

from __future__ import annotations

import torch
from torch import nn

from diffusers.models.transformers.transformer_minimax_h3 import MINIMAX_H3_MODALITY_NUM


class _StepCursor:
    """Which denoising step the block stack is currently evaluating.

    The index lives in a device tensor rather than a Python int so that `torch.compile` sees
    one graph for the whole trajectory. A plain int would be specialized on by dynamo and
    recompile the entire block stack once per denoising step.
    """

    __slots__ = ("step",)

    def __init__(self, device: torch.device) -> None:
        self.step = torch.zeros((), dtype=torch.long, device=device)

    def set(self, index: int) -> None:
        self.step.fill_(index)


class PrecomputedModulation(nn.Module):
    """Drop-in replacement for `MiniMaxH3AdaLayerNormModulation` that indexes a table.

    Holds `(num_steps, num_rows, 6 * hidden_size)` and returns the six chunks of the row
    block belonging to the current step, matching the module it replaces exactly.
    """

    def __init__(self, table: torch.Tensor, cursor: _StepCursor) -> None:
        super().__init__()
        self.register_buffer("table", table, persistent=False)
        self.cursor = cursor

    def forward(self, temb: torch.Tensor) -> tuple[torch.Tensor, ...]:
        # `table[step_tensor]` would be advanced indexing on a data-dependent value, which
        # `torch.compile(fullgraph=True)` rejects: it cannot prove the result's shape.
        # `index_select` with a length-1 index tensor has a statically known shape, so the
        # step stays a runtime value and the graph stays whole.
        rows = self.table.index_select(0, self.cursor.step.reshape(1))[0]
        return rows.chunk(6, dim=-1)


def _timestep_embedding(transformer, timestep: torch.Tensor) -> torch.Tensor:
    """The `temb` the block stack would have been handed for this step."""
    temb = transformer.time_proj(timestep)
    return transformer.time_embedder(temb.to(transformer.time_embedder.linear_1.weight.dtype))


@torch.no_grad()
def precompute(transformer, row_timestep_plan: list, free_weights: bool = True) -> dict:
    """Build every block's modulation table for the whole trajectory.

    `row_timestep_plan` is the pipeline's own `[(timestep, timestep_indices), ...]`, one
    entry per step. Only the timestep values are read here; the indices stay with the caller.
    """
    if getattr(transformer, "_h3opt_adaln_cursor", None) is not None:
        raise RuntimeError("AdaLN precompute is already installed on this transformer.")

    device = next(transformer.parameters()).device
    embeddings = [_timestep_embedding(transformer, ts.to(device)) for ts, _ in row_timestep_plan]

    cursor = _StepCursor(device)
    freed_bytes = 0
    table_bytes = 0

    # A step's table has one row per (timestep, modality) pair, and the number of *distinct*
    # timesteps a step carries is not constant: the video and audio schedules run at different
    # shifts and coincide on some steps, so a plan mixes 1-row and 2-row steps. Pad every step
    # out to the widest one. A block reads row `timestep_indices * 3 + tag`, which never
    # reaches past that step's own rows, so the padding is written and never read.
    max_rows = max(int(timestep.numel()) for timestep, _ in row_timestep_plan) * MINIMAX_H3_MODALITY_NUM

    def padded(rows: torch.Tensor) -> torch.Tensor:
        if rows.shape[0] == max_rows:
            return rows
        return torch.cat([rows, rows.new_zeros(max_rows - rows.shape[0], rows.shape[1])])

    for block in transformer.transformer_blocks:
        projection = block.adaln_proj
        # One GEMM per step at the reference's own shape keeps the result bitwise identical.
        table = torch.stack([padded(torch.cat(projection(temb), dim=-1)) for temb in embeddings])
        table_bytes += table.numel() * table.element_size()

        for parameter in projection.linear.parameters():
            freed_bytes += parameter.numel() * parameter.element_size()

        block.adaln_proj = PrecomputedModulation(table, cursor)
        del projection

    transformer._h3opt_adaln_cursor = cursor
    torch.cuda.empty_cache()

    return {
        "steps": len(row_timestep_plan),
        "blocks": len(transformer.transformer_blocks),
        "table_gb": table_bytes / 1024**3,
        "freed_gb": freed_bytes / 1024**3,
    }


def enable_adaln_precompute(transformer, verbose: bool = True) -> None:
    """Arm the precompute: it fires when the pipeline enters its denoising loop.

    The schedule is not known until the pipeline has built `row_timestep_plan`, so the work
    is hung off the first iteration of the loop denoiser rather than done here. The cursor is
    advanced from the same place, which is the only spot that knows the step index.
    """
    from diffusers.modular_pipelines.minimax_h3 import denoise as h3_denoise

    if getattr(h3_denoise.MiniMaxH3LoopDenoiser, "_h3opt_patched", False):
        transformer._h3opt_adaln_wanted = True
        return

    original_call = h3_denoise.MiniMaxH3LoopDenoiser.__call__

    @torch.no_grad()
    def call_with_precomputed_adaln(self, components, block_state, i: int, t):
        transformer_component = components.transformer
        if getattr(transformer_component, "_h3opt_adaln_wanted", False) and i == 0:
            stats = precompute(transformer_component, block_state.row_timestep_plan)
            transformer_component._h3opt_adaln_wanted = False
            if verbose:
                print(
                    f"[h3opt.adaln] cached {stats['blocks']} blocks x {stats['steps']} steps: "
                    f"table {stats['table_gb']:.2f} GB, freed {stats['freed_gb']:.2f} GB of weights",
                    flush=True,
                )
        cursor = getattr(transformer_component, "_h3opt_adaln_cursor", None)
        if cursor is not None:
            cursor.set(i)
        return original_call(self, components, block_state, i, t)

    h3_denoise.MiniMaxH3LoopDenoiser.__call__ = call_with_precomputed_adaln
    h3_denoise.MiniMaxH3LoopDenoiser._h3opt_patched = True
    transformer._h3opt_adaln_wanted = True
