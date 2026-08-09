## Baseline And Kernel Delivery Contract

`BASELINE-LOCK.json` is created exactly once by the graph before executor work.
It is the immutable timing denominator and terminal visual gold standard. Do not
rerun, replace, edit, chmod, or regenerate the baseline or any locked artifact.

Kernel optimization has no quality-loss/speed frontier. It must deliver exactly
one point, and that point must be the fastest measured composed implementation
that preserves the model's mathematical algorithm and fixed workload. It is
not permissible to publish conservative, balanced, aggressive, fallback, or
alternative points. The one tier name is `exact_fastest`.

Before requesting terminal acceptance, write `DELIVERY-DRAFT.json`; do not write
`DELIVERY.json` directly. Required shape:

```json
{
  "component": "kernel",
  "model_id": "<experiment model_id>",
  "implementation_package": {
    "files": ["<target-model source file in the experiment worktree>"],
    "build_smoke": {"status": "passed", "evidence": "<path>"}
  },
  "frontier_points": [
    {
      "tier": "exact_fastest",
      "config_id": "<composed canonical id>",
      "run_dir": "runs/<terminal full run>",
      "implementation_manifest": "config/<canonical manifest>.toml",
      "activation": {"env": {"<kernel guard>": "1"}},
      "compute_budget": {"dit_calls_preserved": true, "denoising_steps": "<official step count>"},
      "quality": {
        "lossless": true,
        "config_relation": "mathematically equivalent; terminal visual gate passed",
        "max_artifact_severity": "none"
      },
      "runtime_evidence": {"assessment_path": "runs/<run>/assess_verdict.json"},
      "artifacts": ["runs/<run>/assess_verdict.json", "<full-DiT gate>"]
    }
  ],
  "pareto_assessment": {
    "status": "nondominated",
    "objective": "maximize_quality_subject_to_measured_compute_budget",
    "evidence": ["<canonical composed full-DiT gate>"]
  }
}
```

The generic objective string is part of the shared schema; for kernel, quality
is fixed at lossless and optimization reduces to selecting the minimum measured
time. The workflow gate rejects zero points, multiple points, any other tier,
or `lossless != true`.
