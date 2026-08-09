## Baseline And Delivery Contract

`BASELINE-LOCK.json` is created once by the workflow before the executor starts.
It is the only timing denominator and visual gold standard for this experiment.
Do not rerun, replace, edit, chmod, or regenerate the baseline, its source, its
five videos, or its lock. If the lock or a locked artifact is invalid, report an
infrastructure blocker instead of creating another baseline.

The final cache result is a measured speed/quality frontier, not one selected
implementation. Deliver exactly three distinct points:

- `conservative`: best measured quality at the least aggressive useful compute
  reduction; no worse than low-severity visible change;
- `balanced`: best measured quality at a middle compute budget; no worse than
  medium-severity change;
- `aggressive`: best measured quality at the most aggressive useful budget; a
  visible high-severity tradeoff may be documented, but critical failure is not
  deliverable.

Visual difference is not an automatic rejection. It locates an operating point
on the frontier. At each measured compute budget, optimize for the highest
quality available among TeaCache, EasyCache, and TaylorSeer. Points must use
distinct transfeat/runs and strictly increasing measured speedups. A rough
engineering prior is that the aggressive practical ceiling may be around 2x to
3x for full inference, but this is not a target, guarantee, acceptance
threshold, or universal property; determine the target model's actual frontier from data.

When the search is complete, write `DELIVERY-DRAFT.json`. Do not write
`DELIVERY.json`; the workflow-owned delivery gate publishes it. The draft must
contain:

```json
{
  "component": "cache",
  "model_id": "<experiment model_id>",
  "implementation_package": {
    "files": ["<target-model source file in the experiment worktree>"],
    "build_smoke": {"status": "passed", "evidence": "<path>"}
  },
  "frontier_points": [
    {
      "tier": "conservative",
      "transfeat_id": "<distinct id>",
      "run_dir": "runs/<completed run>",
      "implementation_manifest": "transfeat/<manifest>.toml",
      "activation": {"env": {"FORWARD_CACHE_METHOD": "<family>"}},
      "compute_budget": {"time_ratio": 0.0, "target": "<measured budget>"},
      "quality": {
        "transfeat_relation": "<blind-review relation to locked baseline>",
        "max_artifact_severity": "none | low"
      },
      "runtime_evidence": {"assessment_path": "runs/<run>/assess_verdict.json"},
      "artifacts": ["runs/<run>/assess_verdict.json"]
    }
  ],
  "pareto_assessment": {
    "status": "nondominated",
    "objective": "maximize_quality_subject_to_measured_compute_budget",
    "evidence": ["<matched-budget comparison or search-state path>"]
  }
}
```

Expand the example to all three tiers. Every assessment must reference the
locked workload and timing scope. Set `AGENT-STATUS.json.status=complete` only
after the draft and all referenced evidence are complete.
