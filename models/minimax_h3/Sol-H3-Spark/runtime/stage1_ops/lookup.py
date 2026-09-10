"""Exact merged BF16 AdaLN projections for the native four-update schedule."""
import struct


def conditioning_lookup_plan(video_timesteps, audio_timesteps, *, video_condition=False, audio_condition=False):
    """Match native build_row_timesteps, including its FP32 assignment rounding."""
    base = t2va_lookup_plan(video_timesteps, audio_timesteps)
    if audio_condition and not video_condition:
        raise ValueError("reference audio requires a visual condition")
    fp32 = lambda value: struct.unpack("f", struct.pack("f", value))[0]
    return [sorted(set(row + ([fp32(max(video, 0.999))] if video_condition else [])
                       + ([1.0] if audio_condition else [])))
            for row, video in zip(base, video_timesteps)]

def quantized_linear_name(name):
    return name.endswith((".to_q", ".to_k", ".to_v", ".to_out", ".ff.fc_in", ".ff.fc_out"))


def t2va_lookup_plan(video_timesteps, audio_timesteps):
    # Native packing.py:489-493: text uses video time; no external condition
    # video/audio rows exist. Keep each actual GEMM's batch size (1/2/2/2),
    # rather than evaluating all states as one different-sized GEMM.
    if len(video_timesteps) != 4 or len(audio_timesteps) != 4:
        raise RuntimeError("exact T2VA lookup requires the original four updates")
    rows = [sorted(set((video, audio))) for video, audio in zip(video_timesteps, audio_timesteps)]
    if [len(row) for row in rows] != [1, 2, 2, 2] or rows[0] != [0.0]:
        raise RuntimeError(f"unexpected native T2VA timestep rows: {rows}")
    return rows


def require_lookup_step(actual, plan, calls):
    expected = plan[calls % 4]
    if actual != expected:
        raise RuntimeError(f"unexpected conditioning/timestep state: {actual}, expected {expected}")
    return calls % 4


def install_exact_t2va_lookup(model, pipe, task="t2va"):
    """Precompute exact merged BF16 projections at the native timestep rows."""
    import copy
    import torch
    import torch.nn.functional as F
    if model.adaln_basis is not None or len(model.transformer_blocks) != 50:
        raise RuntimeError("exact lookup requires the original full-rank 50-block H3")
    schedules = []
    for name, shift in (("scheduler", 12.0), ("audio_scheduler", 3.0)):
        scheduler = copy.deepcopy(pipe.modules[name])
        if scheduler.shift != shift:
            raise RuntimeError(f"unexpected {name} shift: {scheduler.shift}")
        scheduler.set_timesteps(5, device="cpu")
        schedules.append(scheduler.timesteps.tolist())
    if task not in ("t2va", "fl2va", "ref2va"):
        raise ValueError(f"unsupported lookup task: {task}")
    plans = [conditioning_lookup_plan(*schedules, video_condition=task != "t2va")]
    if task == "ref2va":
        # Audio-bearing references are optional per request. Compute each
        # native GEMM shape independently, not a larger superset GEMM.
        plans.append(conditioning_lookup_plan(*schedules, video_condition=True, audio_condition=True))
    plan = plans[0]
    device = model.time_embedder.fc_in.weight.device
    if any(p.dtype != torch.float32 for p in model.time_embedder.parameters()):
        raise RuntimeError("native time embedding must retain its original FP32 math")
    record = {"mode": "native_T2VA_exact_merged_BF16_four_update_7_row_lookup" if task == "t2va"
                       else "native_conditioned_exact_merged_BF16_four_update_lookup",
              "task": task, "native_timestep_plans": plans,
              "native_timestep_plan": plan, "video_timesteps": schedules[0],
              "audio_timesteps": schedules[1], "calls": 0,
              "time_embedding_dtype": "torch.float32", "projection_dtype": "torch.bfloat16",
              "prompt_or_output_cache": False, "verification": []}

    class FixedProjection(torch.nn.Module):
        def __init__(self, outputs):
            super().__init__()
            self.register_buffer("outputs", outputs)
            # Existing native AdaLN reads linear.weight.dtype before forward.
            self.register_buffer("weight", torch.empty(0, device=device, dtype=torch.float32))

        def forward(self, indices):
            return self.outputs.index_select(0, indices[:, 0].to(torch.long)), None

    class FixedSchedule(torch.nn.Module):
        def __init__(self):
            super().__init__()
            offset = 0
            for mode, rows in enumerate(plans):
                for step, row in enumerate(rows):
                    self.register_buffer(f"indices_{mode}_{step}", torch.arange(offset, offset + len(row),
                                         device=device, dtype=torch.float32).reshape(-1, 1))
                    offset += len(row)

        def forward(self, timestep):
            if torch.compiler.is_compiling() or timestep.dtype != torch.float32 or timestep.ndim != 1:
                raise RuntimeError("fixed timestep validation belongs to the original eager H3 driver")
            actual, step = timestep.detach().cpu().tolist(), record["calls"] % 4
            matches = [mode for mode, rows in enumerate(plans) if rows[step] == actual]
            if len(matches) != 1:
                raise RuntimeError(f"unexpected native conditioning timestep rows: {actual}")
            record["calls"] += 1
            return getattr(self, f"indices_{matches[0]}_{step}")

    # Retain the native time_embedder.fc_in.weight.dtype interface, without
    # retaining its now precomputed parameter storage.
    identity = torch.nn.Identity()
    identity.fc_in = torch.nn.Module()
    identity.fc_in.register_buffer("weight", torch.empty(0, device=device, dtype=torch.float32))
    lookup = FixedSchedule()
    with torch.inference_mode():
        embeddings = [model.time_embedder(model.time_proj(torch.tensor(row, device=device, dtype=torch.float32)))
                      for rows in plans for row in rows]
        targets = [(f"transformer_blocks.{index}.adaln_proj", block.adaln_proj)
                   for index, block in enumerate(model.transformer_blocks)] + [("norm_out", model.norm_out)]
        removed_bytes = sum(p.numel() * p.element_size() for p in model.time_embedder.parameters())
        table_bytes = 0
        for name, target in targets:
            linear = target.linear
            if not target.apply_silu or linear.weight.dtype != torch.bfloat16 or linear.bias.dtype != torch.bfloat16:
                raise RuntimeError(f"unexpected full BF16 native projection: {name}")
            reference = [linear(F.silu(temb).to(torch.bfloat16))[0] for temb in embeddings]
            replacement = FixedProjection(torch.cat(reference, dim=0))
            passed = all(torch.equal(replacement(getattr(lookup, f"indices_{index // 4}_{index % 4}"))[0], output)
                         and bool(torch.isfinite(output).all()) for index, output in enumerate(reference))
            if not passed:
                raise RuntimeError(f"exact merged BF16 lookup replay failed: {name}")
            removed_bytes += sum(p.numel() * p.element_size() for p in linear.parameters())
            table_bytes += replacement.outputs.numel() * replacement.outputs.element_size()
            record["verification"].append({"name": name, "step_rows": [len(row) for row in plan],
                                           "all_mode_step_rows": [[len(row) for row in rows] for rows in plans],
                                           "all_four_steps_bitwise_equal": True, "max_abs": 0.0})
            target.linear = replacement
            target.apply_silu = False  # SiLU is already included in the exact reference above.
    model.time_proj = lookup
    model.time_embedder = identity
    record.update(removed_parameter_bytes=removed_bytes, lookup_table_bytes=table_bytes,
                  all_51_lookup_projections_bitwise_bf16=len(record["verification"]) == 51)
    return record
