Use this callable node contract after a full diffusion run has
`outputs/benchmark.json` and frames or `out.mp4`.

Required command shape:

```bash
python search/plan_eval.py --assess <run_dir> \
  --baseline-frames <canonical_baseline_frames> \
  --model hunyuan_diffusers \
  --out <run_dir>/assess_verdict.json
```

The workflow eval node will trust only durable JSON artifacts with
`baseline_total_s`, `candidate_total_s`, `speedup`, and no infrastructure
blockers.
