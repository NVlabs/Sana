"""Precompute all supported HyperFlow conditioning variants before dropping AdaLN weights."""
import torch
from .bootstrap import ensure_sol_h3

ensure_sol_h3()

from h3_runtime.adaln import PrecomputedModulation, _StepCursor, _timestep_embedding
from hyperflow_h3.blocks import HyperFlowLoopDenoiser
from hyperflow_h3.schedule import build_row_time_pairs, shift_sigmas, endpoints_from_sigmas


def signature(plan):
    return tuple((tuple(t.detach().cpu().tolist()), tuple(r.detach().cpu().tolist())) for t, r, _ in plan)


def make_plans(pipe, sigmas):
    """Use the actual schedulers, noise level and HyperFlow's own pair builder.

    Row counts do not affect the unique pair vectors, but their presence does.
    Keep a generated video row, a generated audio row and a text row in every plan.
    """
    video = shift_sigmas(sigmas, float(pipe.scheduler.shift))
    audio = shift_sigmas(sigmas, float(pipe.audio_scheduler.shift))
    pipe.scheduler.set_timesteps(sigmas=video, device='cpu')
    pipe.audio_scheduler.set_timesteps(sigmas=audio, device='cpu')
    vt, at = pipe.scheduler.timesteps, pipe.audio_scheduler.timesteps
    vr, ar = endpoints_from_sigmas(video), endpoints_from_sigmas(audio)
    plans = {}
    for label, nv, na in [('none', 0, 0), ('image', 1, 0), ('audio', 0, 1), ('image_audio', 1, 1)]:
        vi = torch.arange(1, 2 + nv)
        ai = torch.arange(2 + nv, 3 + nv + na)
        plans[label] = [build_row_time_pairs(
            video_indices=vi, audio_indices=ai, num_condition_video_rows=nv,
            num_condition_audio_rows=na, num_text_tokens=1,
            video_timestep=float(vt[i]), video_endpoint=float(vr[i]),
            audio_timestep=float(at[i]), audio_endpoint=float(ar[i]),
            condition_video_timestep=max(float(vt[i]), float(pipe.keyframe_noise_aug)),
            condition_audio_timestep=1.0) for i in range(len(vr))]
    return plans


@torch.no_grad()
def precompute(model, plans):
    if hasattr(model, '_multi_pair_cursor'):
        raise RuntimeError('HyperFlow modulation tables are already installed.')
    device = next(model.parameters()).device
    signatures = {signature(plan): i for i, plan in enumerate(plans.values())}
    assert len(signatures) == len(plans), 'Unexpected duplicate conditioning variant'
    step_counts = {len(plan) for plan in plans.values()}
    assert len(step_counts) == 1
    embeddings = []
    for plan in plans.values():
        row = []
        for t, r, _ in plan:
            with model.time_embedder.endpoint_context(r.to(device)):
                row.append(_timestep_embedding(model, t.to(device)))
        embeddings.append(row)
    max_rows = max(t.numel() for plan in plans.values() for t, _, _ in plan) * 3
    cursor = _StepCursor(device)
    table_bytes = freed_bytes = 0
    for block in model.transformer_blocks:
        projection = block.adaln_proj
        tables = []
        for schedule in embeddings:
            rows = []
            for embedding in schedule:
                value = torch.cat(projection(embedding), dim=-1)
                if value.shape[0] < max_rows:
                    value = torch.cat([value, value.new_zeros(max_rows-value.shape[0], value.shape[1])])
                rows.append(value)
            tables.append(torch.stack(rows))
        table = torch.stack(tables)
        table_bytes += table.numel() * table.element_size()
        freed_bytes += sum(p.numel() * p.element_size() for p in projection.parameters())
        block.adaln_proj = PrecomputedModulation(table, cursor)
        del projection
    model._multi_pair_cursor = cursor
    model._multi_pair_signatures = signatures
    model._multi_pair_labels = list(plans)
    model._multi_pair_report = dict(steps=step_counts.pop(), variants=list(plans), rows=max_rows,
        blocks=len(model.transformer_blocks), table_bytes=table_bytes, freed_bytes=freed_bytes,
        hits={name: 0 for name in plans})
    torch.cuda.empty_cache()
    return model._multi_pair_report


def select_plan(model, plan):
    key = signature(plan)
    if key not in model._multi_pair_signatures:
        raise RuntimeError('This schedule / conditioning noise level was not precomputed; reload with its plan')
    model._multi_pair_active = model._multi_pair_signatures[key]
    label = model._multi_pair_labels[model._multi_pair_active]
    model._multi_pair_report['hits'][label] += 1
    return model._multi_pair_active


def install_hook():
    if getattr(HyperFlowLoopDenoiser, '_sol_h3_multi_pair_hook', False):
        return
    original = HyperFlowLoopDenoiser.__call__

    def wrapped(self, components, block_state, i, t):
        model = getattr(components, self.transformer_name)
        if hasattr(model, '_multi_pair_cursor'):
            if i == 0:
                select_plan(model, block_state.row_timestep_plan)
            model._multi_pair_cursor.set(i, model._multi_pair_active)
        return original(self, components, block_state, i, t)

    HyperFlowLoopDenoiser.__call__ = wrapped
    HyperFlowLoopDenoiser._sol_h3_multi_pair_hook = True
