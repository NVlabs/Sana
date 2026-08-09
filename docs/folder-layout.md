# Folder Layout

This repo is the control plane. `Sol-LTX-Infer` is the execution plane.

```text
autovideo/
  README.md
  Sol-LTX-Infer/
  agents/
    launch-agent.md
  transfeat/
    README.md
    schema.md
    baseline.toml
  evals/
    README.md
    profiles/
      official_video_t2v.toml
    rubrics/
    README.md
  search_space/
    README.md
    01_cache.md
  loops/
    README.md
    TEMPLATE/
    baseline-eval/
  docs/
    artifact-contract.md
    folder-layout.md
    orchestration.md
  scripts/
    launch_transfeat.py
  tools/
    vision/
      README.md
      nvidia_gemini_judge.py
    symposium/
      README.md
      vendor/Symposium/
      install_project_skills.py
      probe_goal_mode.py
      prepare_goal.py
      start_codex_goal.sh
  goals/
    README.md
  runs/
    README.md
```

## Ownership

`autovideo` owns:

- experiment manifests
- launch orchestration
- artifact conventions
- top-level agent instructions
- cross-transfeat reports
- eval profiles and promotion gates
- independent sub-loop folders
- search-space method-family docs

`Sol-LTX-Infer` owns:

- SGLang diffusion runtime
- Cosmos3 model code
- acceleration framework implementation
- Slurm scripts used by the implementation repo
- kernels, quantization, cache, and sparse-attention code

## Run Directory Shape

Each launch creates:

```text
runs/<timestamp>-<transfeat-id>/
  metadata.json
  manifest.resolved.toml
  launch.sh
  job.sbatch
  outputs/
    run.log
    out.mp4
    benchmark.json
    frames/
    quality.json
    risk_notes.md
    collection.json
    patch_summary.md
```

Only `runs/README.md` is tracked. Everything else is generated.

## Transfeat Naming

Use stable, searchable IDs:

- `baseline`
- `sparse_attention_pisa_090`
- `step_cache_late_reuse`
- `token_prune_feat_norm_075`
- `kwl_qk_rope_ffn`
- `nvfp4_ffn`

Keep one manifest per transfeat. Do not pack multiple ablations into one file;
the launcher and reports should be able to reason about a transfeat as a single
unit.
