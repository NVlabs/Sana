# Unified Integration Delivery Contract

## Source Lock

The workflow source gate accepts only donor `DELIVERY.json` files and writes
`state/integration-source-inventory.json`. Do not parse or depend on donor
`AGENT-STATUS.json`, `PISA-RECIPES.json`, private workflow state, or moving donor
paths. Kernel exposes exactly one `exact_fastest` point. PISA and cache each
expose conservative, balanced, and aggressive points.

Write `INTEGRATION-SOURCES.lock.json` with this shape:

```json
{
  "schema_version": 2,
  "workflow_uid": "integrator_ia",
  "inventory_sha256": "<sha256 of state/integration-source-inventory.json>",
  "sources": {
    "kernel": {"candidate_ids": ["<exact_fastest id>"], "delivery_sha256": "<hash>"},
    "pisa": {"candidate_ids": ["<conservative>", "<balanced>", "<aggressive>"], "delivery_sha256": "<hash>"},
    "cache": {"candidate_ids": ["<conservative>", "<balanced>", "<aggressive>"], "delivery_sha256": "<hash>"}
  },
  "files": [
    {
      "component": "kernel | pisa | cache",
      "source": "state/integration-source-snapshots/<component>/<path>",
      "source_sha256": "<hash>",
      "destination": "<target-model source path in the experiment worktree>",
      "destination_sha256": "<hash>"
    }
  ]
}
```

Every snapshotted implementation file from every component must be materialized
and listed, even when a final recipe disables that component.

## One-Shot Baseline

`BASELINE-LOCK.json` was generated once by the graph before executor work. It is
the only condition `000`, absolute timing denominator, and five-video visual
gold standard. Do not run, replace, or synthesize a second baseline. Every
candidate benchmark and assessment must match its `workload_id`, timing scope,
and `baseline_total_s`.

## Integration State

`INTEGRATION-STATUS.json` and `COMPOSITION-MATRIX.json` must retain the existing
three-recipe and seven-new-toggle-measurement contracts. For each enabled PISA
or cache component, `candidate_id` may select any point exposed by that donor's
frontier. Kernel always uses its sole delivered id. Component counters must
prove actual dispatch; disabled components must report zero activity.

## Final Draft

After graph-owned blind visual review completes, write `DELIVERY-DRAFT.json`.
Do not write `DELIVERY.json` or `INTEGRATION-DELIVERY.json`. Expand this example
to exactly three distinct points with strictly increasing measured speedups:

```json
{
  "component": "integrator",
  "model_id": "<experiment model_id>",
  "implementation_package": {
    "files": ["<integrated target-model source file in the experiment worktree>"],
    "build_smoke": {"status": "passed", "evidence": "<path>"}
  },
  "frontier_points": [
    {
      "tier": "conservative",
      "candidate_id": "<distinct integrated id>",
      "run_dir": "runs/<completed run>",
      "implementation_manifest": "candidates/<integrated manifest>.toml",
      "activation": {"env": {"<integrated-recipe env var>": "conservative"}},
      "compute_budget": {"components": {}, "settings": {}},
      "quality": {
        "candidate_relation": "<blind-review relation to locked baseline>",
        "max_artifact_severity": "none | low"
      },
      "runtime_evidence": {"assessment_path": "runs/<run>/assess_verdict.json"},
      "artifacts": ["runs/<run>/assess_verdict.json", "runs/<run>/outputs/integration_stats.json"]
    }
  ],
  "pareto_assessment": {
    "status": "nondominated",
    "objective": "maximize_quality_subject_to_measured_compute_budget",
    "evidence": ["COMPOSITION-MATRIX.json", "INTEGRATION-SUMMARY.md"]
  }
}
```

Conservative permits severity through low, balanced through medium, and
aggressive through high; critical or incomplete visual evidence is not
deliverable. Visual difference positions a point on the frontier and is not an
automatic rejection. The workflow-owned delivery gate computes hashes and
publishes the stable `DELIVERY.json`.
