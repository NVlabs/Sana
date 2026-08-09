## Baseline And Delivery Contract

`BASELINE-LOCK.json` is created once by the workflow before the executor starts.
It is the only timing denominator and visual gold standard for this experiment.
Do not rerun, replace, edit, chmod, or regenerate the baseline, its source, its
five videos, or its lock. An invalid lock is infrastructure, not permission to
create another baseline.

The final PISA result is a measured speed/quality frontier with exactly three
distinct points:

- `conservative`: highest quality at a modest useful compute reduction, with at
  most low-severity visual change;
- `balanced`: highest quality at a middle budget, with at most medium-severity
  change;
- `aggressive`: highest quality at the most aggressive useful budget; a
  documented high-severity tradeoff is allowed, but critical failure is not.

Visual difference is not an automatic rejection. It determines tier placement.
At each budget, optimize layer, step, attention-type, and density policy for the
highest retained quality. Points must use distinct measured runs and strictly
increasing speedups. A density near 0.10 is a reasonable initial probe for many
PISA workloads, not a default answer or acceptance criterion. The target model's
useful density can be above or below 0.10 and must be established case by case.

When complete, write `DELIVERY-DRAFT.json`; never write `DELIVERY.json`
directly. Use this shape and expand it to conservative, balanced, and aggressive:

```json
{
  "component": "pisa",
  "model_id": "<experiment model_id>",
  "implementation_package": {
    "files": ["<target-model source file in the experiment worktree>"],
    "build_smoke": {"status": "passed", "evidence": "<path>"}
  },
  "frontier_points": [
    {
      "tier": "conservative",
      "config_id": "<distinct id>",
      "run_dir": "runs/<completed run>",
      "implementation_manifest": "config/<manifest>.toml",
      "activation": {"env": {"<activation env var for this technique>": "1"}},
      "compute_budget": {"density": 0.1, "layer_step_policy": "<path-or-id>"},
      "quality": {
        "config_relation": "<blind-review relation to locked baseline>",
        "max_artifact_severity": "none | low"
      },
      "runtime_evidence": {"assessment_path": "runs/<run>/assess_verdict.json"},
      "artifacts": ["runs/<run>/assess_verdict.json"]
    }
  ],
  "pareto_assessment": {
    "status": "nondominated",
    "objective": "maximize_quality_subject_to_measured_compute_budget",
    "evidence": ["PISA-SEARCH-STATE.json", "PISA-RECIPES.json"]
  }
}
```

Every assessment must use the locked workload and timing scope. Set
`AGENT-STATUS.json.status=complete` only after the draft and evidence are valid.
