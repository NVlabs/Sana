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
from diffusers.modular_pipelines.minimax_h3.packing import MINIMAX_H3_KEYFRAME_NOISE_AUG


class _StepCursor:
    """Which denoising step the block stack is currently evaluating.

    The index lives in a device tensor rather than a Python int so that `torch.compile` sees
    one graph for the whole trajectory. A plain int would be specialized on by dynamo and
    recompile the entire block stack once per denoising step.
    """

    __slots__ = ("step", "schedule", "step_index", "schedule_index")

    def __init__(self, device: torch.device) -> None:
        self.step = torch.zeros((), dtype=torch.long, device=device)
        self.schedule = torch.zeros((), dtype=torch.long, device=device)
        self.step_index = 0
        self.schedule_index = 0

    def set(self, index: int, schedule: int) -> None:
        self.step_index = index
        self.schedule_index = int(schedule)
        self.step.fill_(index)
        self.schedule.fill_(int(schedule))


class PrecomputedModulation(nn.Module):
    """Drop-in replacement for `MiniMaxH3AdaLayerNormModulation` that indexes a table.

    Holds one table per conditioning schedule and returns the six chunks of the active row
    block. T2V uses video/audio noise levels, I2V adds the fixed visual-reference level,
    and Ref2VA can additionally carry a fixed audio-reference level.
    """

    def __init__(self, table: torch.Tensor, cursor: _StepCursor) -> None:
        super().__init__()
        self.register_buffer("table", table, persistent=False)
        self.cursor = cursor

    def forward(self, temb: torch.Tensor) -> tuple[torch.Tensor, ...]:
        if not torch.compiler.is_compiling():
            # The production transformer runs eager. Integer indexing returns a view of the
            # already-materialized table and avoids two tiny GPU index_select kernels per block.
            rows = self.table[self.cursor.schedule_index, self.cursor.step_index]
            return rows.chunk(6, dim=-1)
        # `table[step_tensor]` would be advanced indexing on a data-dependent value, which
        # `torch.compile(fullgraph=True)` rejects: it cannot prove the result's shape.
        # `index_select` with a length-1 index tensor has a statically known shape, so the
        # step stays a runtime value and the graph stays whole.
        schedule = self.table.index_select(0, self.cursor.schedule.reshape(1))[0]
        rows = schedule.index_select(0, self.cursor.step.reshape(1))[0]
        return rows.chunk(6, dim=-1)


def _timestep_embedding(transformer, timestep: torch.Tensor) -> torch.Tensor:
    """The `temb` the block stack would have been handed for this step."""
    temb = transformer.time_proj(timestep)
    return transformer.time_embedder(temb.to(transformer.time_embedder.linear_1.weight.dtype))


@torch.no_grad()
def precompute(
    transformer,
    video_timesteps: torch.Tensor,
    audio_timesteps: torch.Tensor,
    *,
    include_audio_condition: bool = False,
) -> dict:
    """Build every block's modulation table for the whole trajectory.

    T2V and visual-conditioning schedules are cached before the projection weights are
    freed. Ref2VA also caches the audio-conditioning schedule when requested. Merely
    padding a smaller table is insufficient because each extra noise level has a distinct
    projection value.
    """
    if getattr(transformer, "_h3opt_adaln_cursor", None) is not None:
        raise RuntimeError("AdaLN precompute is already installed on this transformer.")

    device = next(transformer.parameters()).device
    video_timesteps = video_timesteps.to(device)
    audio_timesteps = audio_timesteps.to(device)
    if video_timesteps.shape != audio_timesteps.shape:
        raise ValueError(
            "video and audio schedules must have the same shape, got "
            f"{tuple(video_timesteps.shape)} and {tuple(audio_timesteps.shape)}"
        )

    base_timesteps = []
    conditioned_timesteps = []
    audio_conditioned_timesteps = []
    for video_timestep, audio_timestep in zip(video_timesteps, audio_timesteps):
        # Match build_row_timesteps exactly: construct float32 values and sort/deduplicate.
        video_value = float(video_timestep)
        audio_value = float(audio_timestep)
        base = torch.tensor([video_value, audio_value], dtype=torch.float32, device=device).unique(sorted=True)
        conditioned = torch.tensor(
            [video_value, audio_value, max(video_value, MINIMAX_H3_KEYFRAME_NOISE_AUG)],
            dtype=torch.float32,
            device=device,
        ).unique(sorted=True)
        base_timesteps.append(base)
        conditioned_timesteps.append(conditioned)
        if include_audio_condition:
            audio_conditioned_timesteps.append(
                torch.tensor(
                    [video_value, audio_value, max(video_value, MINIMAX_H3_KEYFRAME_NOISE_AUG), 1.0],
                    dtype=torch.float32,
                    device=device,
                ).unique(sorted=True)
            )

    schedule_timesteps = [base_timesteps, conditioned_timesteps]
    if include_audio_condition:
        schedule_timesteps.append(audio_conditioned_timesteps)
    schedule_embeddings = [
        [_timestep_embedding(transformer, timesteps) for timesteps in schedule]
        for schedule in schedule_timesteps
    ]

    cursor = _StepCursor(device)
    freed_bytes = 0
    table_bytes = 0

    # A step's table has one row per (timestep, modality) pair. T2V carries at most two
    # distinct timesteps, I2V three, and Ref2VA with audio references four. Pad the
    # requested variants to one static shape; a block never addresses padding rows.
    max_rows = max(
        int(timestep.numel())
        for schedule in schedule_timesteps
        for timestep in schedule
    ) * MINIMAX_H3_MODALITY_NUM

    def padded(rows: torch.Tensor) -> torch.Tensor:
        if rows.shape[0] == max_rows:
            return rows
        return torch.cat([rows, rows.new_zeros(max_rows - rows.shape[0], rows.shape[1])])

    for block in transformer.transformer_blocks:
        projection = block.adaln_proj
        # One GEMM per step and schedule variant at the reference's own shape keeps every
        # live row bitwise identical to the unmodified projection.
        table = torch.stack(
            [
                torch.stack(
                    [padded(torch.cat(projection(temb), dim=-1)) for temb in embeddings]
                )
                for embeddings in schedule_embeddings
            ]
        )
        table_bytes += table.numel() * table.element_size()

        for parameter in projection.linear.parameters():
            freed_bytes += parameter.numel() * parameter.element_size()

        block.adaln_proj = PrecomputedModulation(table, cursor)
        del projection

    transformer._h3opt_adaln_cursor = cursor
    torch.cuda.empty_cache()

    return {
        "steps": len(video_timesteps),
        "blocks": len(transformer.transformer_blocks),
        "table_gb": table_bytes / 1024**3,
        "freed_gb": freed_bytes / 1024**3,
    }


def enable_adaln_precompute(
    transformer,
    verbose: bool = True,
    *,
    component_name: str = "transformer",
) -> None:
    """Arm the precompute: it fires when the pipeline enters its denoising loop.

    The schedule is not known until the pipeline has built `row_timestep_plan`, so the work
    is hung off the first iteration of the loop denoiser rather than done here. The cursor is
    advanced from the same place, which is the only spot that knows the step index.
    """
    from diffusers.modular_pipelines.minimax_h3 import denoise as h3_denoise

    if component_name == "transformer":
        loop_denoiser = h3_denoise.MiniMaxH3LoopDenoiser
        include_audio_condition = False
    elif component_name == "transformer_ref":
        loop_denoiser = h3_denoise.MiniMaxH3Ref2VALoopDenoiser
        include_audio_condition = True
    else:
        raise ValueError(f"unsupported MiniMax-H3 transformer component: {component_name!r}")

    if getattr(loop_denoiser, "_h3opt_patched", False):
        transformer._h3opt_adaln_wanted = True
        return

    original_call = loop_denoiser.__call__

    @torch.no_grad()
    def call_with_precomputed_adaln(self, components, block_state, i: int, t):
        transformer_component = getattr(components, component_name)
        if getattr(transformer_component, "_h3opt_adaln_wanted", False) and i == 0:
            stats = precompute(
                transformer_component,
                block_state.timesteps,
                block_state.audio_timesteps,
                include_audio_condition=include_audio_condition,
            )
            transformer_component._h3opt_adaln_wanted = False
            if verbose:
                print(
                    f"[h3opt.adaln] cached {stats['blocks']} blocks x {stats['steps']} steps: "
                    f"table {stats['table_gb']:.2f} GB, freed {stats['freed_gb']:.2f} GB of weights",
                    flush=True,
                )
        cursor = getattr(transformer_component, "_h3opt_adaln_cursor", None)
        if cursor is not None:
            schedule = int(bool(getattr(block_state, "num_condition_video_rows", 0)))
            if include_audio_condition and getattr(block_state, "num_condition_audio_rows", 0):
                schedule = 2
            cursor.set(i, schedule)
        return original_call(self, components, block_state, i, t)

    loop_denoiser.__call__ = call_with_precomputed_adaln
    loop_denoiser._h3opt_patched = True
    transformer._h3opt_adaln_wanted = True
