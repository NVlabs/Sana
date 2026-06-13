# Folder Layout

This repo is the control plane. `Sol-LTX-Infer` is the execution plane.

```text
autovideo/
  README.md
  Sol-LTX-Infer/
  agents/
    launch-agent.md
  candidates/
    README.md
    schema.md
    baseline.toml
  evals/
    README.md
    profiles/
      official_video_t2v.toml
    rubrics/
  snippets/
    README.md
    sol-ltx-infer-reference.md
  loops/
    README.md
    TEMPLATE/
    baseline-eval/
  docs/
    artifact-contract.md
    folder-layout.md
    orchestration.md
  scripts/
    launch_candidate.py
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
- cross-candidate reports
- eval profiles and promotion gates
- independent sub-loop folders
- reference snippets copied from successful prior branches/reports

`Sol-LTX-Infer` owns:

- SGLang diffusion runtime
- Cosmos3 and LTX model code
- acceleration framework implementation
- Slurm scripts used by the implementation repo
- kernels, quantization, cache, and sparse-attention code

## Run Directory Shape

Each launch creates:

```text
runs/<timestamp>-<candidate-id>/
  metadata.json
  manifest.resolved.toml
  launch.sh
  job.sbatch
  outputs/
    run.log
    out.mp4
    perf.json
    frames/
    report.md
```

Only `runs/README.md` is tracked. Everything else is generated.

## Candidate Naming

Use stable, searchable IDs:

- `baseline`
- `sparse_attention_pisa_090`
- `step_cache_late_reuse`
- `token_prune_feat_norm_075`
- `kwl_qk_rope_ffn`
- `nvfp4_ffn`

Keep one manifest per candidate. Do not pack multiple ablations into one file;
the launcher and reports should be able to reason about a candidate as a single
unit.
