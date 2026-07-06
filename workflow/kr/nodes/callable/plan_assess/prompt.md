Use this callable node contract only for terminal validation after a full
diffusion run has `outputs/benchmark.json` and frames or `out.mp4`.

Required command shape:

```bash
python search/plan_eval.py --assess <run_dir> \
  --baseline-frames <canonical_baseline_frames> \
  --model hunyuan_diffusers \
  --out <run_dir>/assess_verdict.json
```

The terminal `final_full_eval` node will trust only durable JSON artifacts with
`baseline_total_s`, `candidate_total_s`, `speedup`, `gemini_overall=pass`, and
no infrastructure blockers.

Assessment failure is not a discard decision. Missing baseline frames, missing
Gemini/API credentials, missing videos, collector failures, or incomplete
quality artifacts must be repaired or recorded as infrastructure blockers.
Reviewer may still request executor resume instead of discarding the method.
