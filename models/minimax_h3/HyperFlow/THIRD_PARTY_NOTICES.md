# Source notices

The six files under `hyperflow_h3/` retain their original notices:
Copyright 2026 The HyperFlow authors, Apache License 2.0. Their exact input
hashes and snapshot provenance are recorded in `PROVENANCE.json`.

The retained headers identify these derived portions of Diffusers:

- `schedule.py`: `MiniMaxH3Scheduler.set_timesteps`, Copyright 2025 The MiniMax
  authors and The HuggingFace Team; and
  `MiniMaxH3SetTimestepsStep.build_row_timesteps`, Copyright 2026 The MiniMax and
  HuggingFace Teams. Apache License 2.0.
- `blocks.py`: subclasses/reworks `MiniMaxH3SetTimestepsStep` and
  `MiniMaxH3LoopDenoiser`, Copyright 2026 The MiniMax and HuggingFace Teams.
  Apache License 2.0.
- `sol_attn.py`: derives its attention processor from
  `MiniMaxH3AttnProcessor.__call__`, Copyright 2025 The MiniMax Team and The
  HuggingFace Team. Apache License 2.0.

The two-time conditioning follows AnyFlow (Gu et al., 2026), as noted in
`embedder.py`. The source's optional sparse attention interface acknowledges
NVIDIA Sol-Attn (Li et al., 2026).

The optimized entry point reuses `../Sol-H3/h3_runtime/`, including the existing
LoRA consumer fusion from NVlabs/Sana PR #503. Its kernels and third-party
notices remain in their existing locations; they are not re-vendored here.
The small `packing` import compatibility fallback is the sole change to that
shared runtime in this proposal.

`LICENSE` contains Apache License 2.0. This notice records attribution visible
in the supplied source headers; no separate original `THIRD_PARTY_NOTICES.md`
was present in the supplied snapshot. Model/adaptor weights are not included;
their own upstream terms remain applicable.
