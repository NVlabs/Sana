## Cache Scope

Optimize the target model by caching repeated denoising, transformer, attention,
block, residual, feature, or forecast work across denoising steps. This workflow
is for cache methods only; do not spend transfeat work on kernel fusion,
quantization, token pruning, VAE decode, text encoders, scheduler changes,
prompt changes, or benchmark shape changes.

The editable target model source must be the copy in your experiment worktree;
locate it within the worktree and only edit inside the worktree. Do not patch
the shared reference bundle under `/lustre/.../code` or any shared checkpoint,
VAE, Hugging Face cache, or baseline run.

## Fixed Evaluation Contract

Every cache transfeat must be judged by a full target-model run and aligned
quality assessment. Microbench, single-DiT, or module-only evidence may be used
for debugging, but it is never sufficient for workflow progress or method
judgment.

The full run must preserve the target model's official evaluation contract:

- model profile: the experiment's `model_id` (see goal.md / experiment.json);
- prompt file: the target model's validation prompt file (see the model profile
  / baseline manifest);
- prompt count: the first five prompts of the target model's validation set;
- resolution, duration, frame count, and fps: the target model's official eval
  profile (from the model profile / baseline manifest);
- denoising steps, guidance, flow shift, and motion score: the target model's
  official generation config (from the model profile / baseline manifest);
- checkpoint, VAE, scheduler, resolution, frame count, prompt text, and seed
  policy must not be changed to obtain speed or quality.

After a full run completes, preserve its benchmark, five videos or grouped
frames, and exit the executor invocation. The explicit workflow graph then runs
an independent blind Codex visual reviewer; the executor must not write that
reviewer's verdict or call Gemini itself.

The workflow gate accepts only durable `assess_verdict.json` evidence merged by
that reviewer node with
numeric `baseline_total_s`, `transfeat_total_s`, `speedup`, numeric
`lpips_max`, `visual_provider=codex`, a complete pass-or-fail
`codex_visual_overall`, a valid `codex_visual_verdict.json`, no infrastructure
blockers, and a matching full-run config. A complete visual fail is usable
quality evidence for frontier placement; it is not automatically a workflow or
method failure.

## Closed Method Scope

The transfeat set for this workflow is closed and contains exactly three cache
families:

1. TeaCache
2. EasyCache
3. TaylorSeer

Implement, compare, and tune only these three families. Do not introduce PAB,
DeepCache, FasterCache, Cache-DiT, generic fixed-step reuse, attention
broadcast, token pruning, or another cache family as a transfeat. The
cache-disabled target-model run is the control, not a fourth transfeat. A fixed
call-index reuse schedule may be used briefly to debug integration or calibrate
timing, but it cannot be retained, ranked, or presented as a workflow result.

Every measured transfeat must belong to exactly one of the three families.
Cross-family hybrids are outside this workflow because they make attribution
and matched-time comparison ambiguous. Implementation work within a family is
allowed and expected: faithfully integrate the method into the target model, remove its
bookkeeping overhead, choose its cache payload and placement, and tune its
native refresh, threshold, history, correction, layer, and timestep controls.
Do not add an independent optimization such as kernel fusion, precision change,
quantization, step reduction, or scheduler change to make one family appear
faster.

## Matched-Time Quality Objective

The objective is not to find the fastest isolated point from each family. Find
which of TeaCache, EasyCache, and TaylorSeer preserves the most quality at the
same measured inference-time compression, and find the best implementation and
parameters for each useful shared time budget.

Use full-run end-to-end wall time from the fixed evaluation contract:

```text
time_ratio = transfeat_total_s / baseline_total_s
time_compression = 1 - time_ratio
speedup = baseline_total_s / transfeat_total_s
```

All runs in one comparison must use the same baseline, hardware/job shape, and
fixed inference contract. Two operating points are matched only when their
measured `time_ratio` values differ by at most 2% relative. If they are not
matched, tune and rerun the family at the target budget; do not compare quality
at unequal speeds and do not estimate full-run speed by multiplying module or
microbenchmark speedups. For noisy or close results, repeat the same point and
use the median full-run time.

Build an evidence-backed speed/quality curve for each family, then choose shared
time targets from the overlap of the three feasible ranges. At every shared
target, compare complete outputs over all five prompts using aligned LPIPS and
the independent blind Codex visual assessment, including temporal consistency
and visible artifacts. Rank quality only after the matched-time condition is
met. Report the Codex visual judgment and prompt-level failures together with
LPIPS mean/max when available;
do not hide a bad prompt behind an aggregate. Time is a constraint for this
comparison, not a quality score or a tie-breaking advantage.

The final result must identify the quality winner and exact recipe at each
matched target, plus the overall Pareto frontier. If a family cannot reach a
target faithfully, record its measured feasible range and do not claim an
unmatched winner.

The repo already has cache-method search material:

- `search_space/01_cache.md` describes several cache directions. Only its
  TeaCache-style timestep-aware reuse, EasyCache-style runtime-adaptive
  transform-vector reuse, and Taylor-style forecasting material is in scope.
  Ignore the other families in that document for this workflow.
- `transfeat/step_cache/` contains existing cache family manifests such as
  TeaCache-style signal reuse and several out-of-scope methods. Existing code
  is reference material only and does not expand the closed transfeat set.
- Historical Hunyuan/Cosmos artifacts contain a generic TeaCache controller and
  prior quality-gated runs, but those are references only. Adapt the idea to
  the target model's actual denoising/transformer path instead of copying a
  different model adapter blindly.

For each transfeat, record the signal source, reuse payload, refresh rule,
hit/recompute pattern, OFF path, full-run runtime, normalized `time_ratio`,
LPIPS, Codex visual verdict, and failure mode. Also record the complete method
parameter point, its parent trial, target time budget, and why the previous
full-evaluation evidence selected the next adjustment. A cache method is not
proven by speed alone, and a quality winner is not proven at unmatched speed.

## Execution round limit

You have a hard budget of **20 optimization rounds** (transfeat attempts) for this workflow. This is an execution-round limit, not a suggestion: pace your search so that by round 20 you have finalized and delivered your best retained frontier. Do not plan for more than 20 rounds. One round = one transfeat implemented, launched, evaluated, and gated.
