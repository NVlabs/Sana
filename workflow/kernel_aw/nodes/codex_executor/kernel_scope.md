## Transformer Optimization Objective

Optimize the target model's transformer/DiT denoising path. New candidates should
target kernel-level or runtime-level work inside repeated transformer blocks,
attention paths, FFN paths, and transformer glue code.

The editable target model source must be the copy in your experiment worktree;
locate it within the worktree and only edit inside the worktree. Do not patch
the shared reference bundle under `/lustre/.../code` or any shared checkpoint,
VAE, or Hugging Face cache path.

This workflow does maintain a technique denylist for methods that change the
mathematical or algorithmic semantics of the model. The cache prohibition is
specific: do not use diffusion skip-step cache, stale denoiser-output reuse, or
any cross-step approximation that skips or reduces the required DiT forwards,
attention work, FFN work, or the fixed denoising-step count. This is not
a blanket ban on caches. Mathematically lossless implementation caches are in
scope, including reusable cross-attention K/V from invariant conditioning,
RoPE/position tensors, static masks and shape metadata, packed/transformed
weights, compiled artifacts, allocator buffers, and other values proven
invariant for the exact operation. Such caches must preserve every required
DiT call and the same mathematical result.

**Exact redundant-computation elimination IS allowed** (it is not "reducing
required work"). A forward/computation is *required* only if it actually
contributes to the result. If a forward is provably REDUNDANT — its output is
bit-identical to another already-computed forward AND its contribution to the
final result is provably zero (e.g. a guidance branch that, under the current
workload, duplicates the base branch so its delta term is identically 0) — then
eliminating it is exact common-subexpression / dead-computation elimination: the
result is bit-identical and the algorithm's semantics are unchanged. This IS
allowed and encouraged. The prohibition targets APPROXIMATE skipping or reuse of
forwards that DO contribute (trading accuracy for speed); it does not oblige you
to recompute a provably-redundant duplicate. When you eliminate such a forward,
prove the redundancy (identical inputs/conditioning ⇒ identical output;
provably-zero contribution) in your method/semantics argument.

Do not propose or implement attention sparsification, low-precision rewrites
outside the stated exception, 8-bit or lower quantization behavior, sparsity,
or other methods that reduce effective model work or change the algorithm being
evaluated. Reducing 32-bit arithmetic to 16-bit arithmetic is allowed; 16-bit
execution is not considered quantization for this policy. Only 8-bit and lower
quantization behavior is prohibited.
Rank allowed work by its measured contribution to the transformer bottleneck and
the official end-to-end objective.

### What "lossless" means here (mathematical / algorithmic correctness ONLY)

"Lossless" here means the candidate is a **mathematically valid, semantics-
preserving implementation of the same algorithm**, judged at the level of
METHOD, RULE, and REASONING — **never** by comparing outputs. Do NOT gate on any
result artifact whatsoever: not bit-identity, not a latent/tensor difference, not
a floating-point tolerance, not LPIPS / PSNR / visual similarity. None of them.

Why: two correct implementations of the same algorithm can differ numerically —
sometimes substantially — purely from implementation roughness, operation order,
or precision. That divergence does NOT make either one wrong. You must not call
an implementation incorrect merely because its output moved relative to another
(rougher, or "reference") implementation. Correctness is a property of the
METHOD, not of how close the output lands.

A candidate is admissible iff, by reasoning about the method, it:
- computes the *same mathematical function / algorithm* — the change is an
  implementation transformation (e.g. fusing, reordering, compiling with any
  aggressiveness, re-laying-out, re-placing / residency, caching a provably
  step-invariant quantity, reorganizing communication, staying within 16-bit
  precision), NOT an algorithmic approximation; and
- preserves the algorithm's semantics and work: the SAME denoising-step count,
  the SAME model/DiT-call count, with nothing skipped, dropped, rank-reduced,
  sparsified, or sub-16-bit-quantized (the denylist above).

Establish and record this as a **method / semantics argument** — what you changed
and why it is the same mathematics — together with the structural invariants
(denoising-step count and DiT-call count unchanged). That argument, NOT any output
measurement, is the retention and delivery criterion here. Do NOT run, cite, or
gate on an output-difference check of any kind (no bit diff, no latent diff, no fp
tolerance, no LPIPS). NEVER reject a candidate because its numbers moved — reject
ONLY if the method introduces a real algorithmic approximation or changes the
work. A faster implementation whose output differs but whose algorithm is provably
unchanged is CORRECT and MUST be retained. (Earlier candidates rejected only for
numeric divergence — e.g. compiler floating-point contraction changing rounding —
were WRONGLY rejected; do not repeat that: aggressive compilation and fp
reordering are the same algorithm and are allowed.)

Separate method admissibility from benchmark comparability:

- keep the official prompts, resolution, frame count, denoising-step count,
  scheduler, guidance, checkpoint, and output contract fixed for the canonical
  same-workload comparison;
- use speed as the evaluation criterion once mathematical correctness is
  preserved.

The primary search objective remains the repeated transformer/DiT path. If an
allowed candidate also changes an adjacent stage or runtime mechanism, measure
and label that contribution separately so the kernel attribution remains
intelligible. Ordinary loop evaluation stays at single-DiT or module level. Full
diffusion evaluation is reserved for terminal validation after reviewer exit
intent because that is the authoritative end-to-end visual quality gate.

## Where The Recoverable Time Hides — Profile First, You Choose

Do not assume the dominant recoverable time lives in fusing per-operator math.
Operator fusion is one family of lossless speedups, not the boundary of what is
allowed or available. Profile the **full warm end-to-end hot path** — not only the
isolated DiT forward — and let your own measurements decide where the largest
*mathematically lossless* wall-time reductions actually are for this specific
model on this specific hardware. It is expected that a meaningful fraction of the
recoverable time lies outside the operator-fusion work already sketched below.

Recoverable lossless time is frequently spent outside core operator compute.
Treat the following as open lines of inquiry to profile and pursue — not a
checklist, not exhaustive, and none guaranteed to apply here:

- work that is provably invariant across denoising steps (or across sequence
  segments, or across guidance branches) yet recomputed every time;
- avoidable movement, eviction, or re-materialization of large invariant
  tensors/parameters as execution crosses the pipeline's stages — including where
  those tensors physically reside during the hot loop;
- whether every core exact operation is dispatched through the fastest available
  *equivalent* implementation/primitive for the real shapes, dtype, and hardware,
  rather than a slower default that returns the same result;
- communication, gather/scatter, and intermediate-tensor overhead when the exact
  computation is executed across multiple devices;
- kernel-launch volume and host↔device synchronization stalls surrounding
  otherwise-cheap work.

You decide the targets from evidence: discover the real hotspots yourself from
your profile and rank levers by their measured end-to-end contribution, wherever
they turn out to be. The reference directions elsewhere in this document are a
starting set, not a boundary.

Heuristic prompts (NOT named methods — investigate whichever your profile
justifies):
- Revisit any transformation you (or a previous run) abandoned ONLY because its
  numeric output moved — under the correctness rule above that is not a defect, so
  it may be a valid, retained speedup now.
- Ask whether an equivalent-but-more-aggressive form of a transformation is being
  needlessly throttled by an over-strict self-imposed constraint rather than by an
  actual change to the algorithm.
- Consider the FULL set of large state that persists across a request — not only
  the single dominant consumer — when hunting avoidable inter-stage movement.

The only hard limit is the denylist above: no approximation, no reduced or skipped
model work, no sub-16-bit quantization, no sparsity, no step-skipping. Any change
that keeps the same algorithm (regardless of numeric output movement) while
reducing measured wall time is in scope.

## Execution Order And Frontier Policy

Do not begin with an isolated operator chosen only because it is easy to
microbenchmark. Use this order:

1. Read and verify the graph-created `BASELINE-LOCK.json` before implementing
   candidate changes. The graph already ran the experiment's one allowed
   baseline. Never rerun or modify it.
2. Establish a durable `KERNEL-PREFLIGHT.json` from the active, registry-resolved
   target model path. Record the official block mix, tensor shapes, dtype, dominant
   kernel families, launch/layout costs, and a warm repeated full-DiT profile.
3. Use the live full-DiT profile to choose candidates by expected integrated
   contribution. A module microbench is a screening surface, not the portfolio
   objective.
4. Keep a cumulative canonical ON manifest and accumulated acceleration stack.
   After every two positive component candidates, or after any material
   attention/FFN/AttnRes change, rerun a warm paired full-DiT OFF/ON benchmark
   for the composed frontier.

Once a method is judged effective by the current gate, add it to the accumulated
acceleration stack and record it in `canonical_on_manifest`. Later candidate
experiments must run with that accumulated ON stack enabled unless they are
explicitly isolating or debugging a stack interaction. Report both the new
candidate's incremental contribution on top of the stack and the cumulative
OFF/ON speed of the full stack.

Adding a module-level method to the stack does not close that module. The
executor may continue refining the same module later; each later refinement is
measured as an additional change on top of the accumulated stack, not as a
replacement for the earlier effective method.

Retaining a candidate does not require continuing to refine it immediately. A
positive or unresolved candidate may be recorded as `retained_parked` while the
executor moves to a higher-impact method family. Return to it only when profile
evidence ranks its next refinement above the alternatives. This workflow must
maintain breadth across major hotspots without discarding preserved work.

Implement one concrete candidate per iteration, but keep at least the following
portfolio fields current in `AGENT-STATUS.json`:

- `active_candidate_id` and `active_gate` for the candidate evaluated by the
  current executor invocation;
- `candidate_iteration`, which counts candidate iterations rather than workflow
  node transitions;
- `frontier_candidates`, including retained and retained-parked methods;
- `canonical_on_manifest` and the latest integrated full-DiT gate;
- `ranked_next_families`, with estimated full-DiT contribution and evidence.

The terminal kernel delivery is not a loss/speed frontier. It contains exactly
one `exact_fastest` point: the fastest measured composed canonical stack that
is mathematically lossless and passes terminal validation. Do not publish
multiple quality tiers or alternative kernel recipes.

## Kernel Technique References

Use the target model's Sol-Engine kernel-fusion examples as reference starting
points, not as a fixed roadmap. First consider these directions, then adapt or
extend them according to the target model's actual transformer structure, tensor
shapes, dtype, attention backend, and profiler evidence:

- AdaLN and residual gate fusion: normalize, scale, shift, gate, and residual
  glue around DiT blocks.
- GEMM epilogues: fuse memory-bound work after GEMMs, such as bias, activation,
  FFN output glue, residual updates, or normalization-adjacent epilogues.
- QK-norm plus RoPE fusion on attention Q/K paths.
- Attention output gate fusion after attention value aggregation and output
  projection.
- Residual and modulation glue fusion around transformer block boundaries.
- QKV merge when model layout allows equivalent merged projection execution.
- `torch.compile` or compiler fusion for stable, repeated transformer regions;
  record cold compile, warm timing, cache behavior, and failure modes separately.

## Timing And Contribution Accounting

Every timing claim must use a matching denominator and must report warmup-after
inference speed. Exclude load/setup time and the first two warmup rounds from
the measured inference speed. Do not label a subprocess wall timer as isolated
denoise or DiT time, and do not mix startup, model/text encoder loading,
generation, VAE decode, or video writing with kernel-only attribution.

Every contribution artifact must record:

- timing scope: operator, module, full DiT, warm single-prompt generation, or
  bundled process wall;
- warmup policy, including the first two warmup rounds excluded from measured
  inference speed;
- `prompt_count`, `steps_per_prompt`, model calls per step, and calls per DiT;
- recurring savings multiplied by every prompt/step/call in the measured run;
- one-time compile and initialization cost exactly once per process, reported
  separately from warm steady state;
- stage-isolated DiT/denoise time when computing percent-of-DiT contribution;
- warm and cold-amortized estimates without mixing their denominators.

If stage-isolated time is unavailable, set the field to unknown and report only
the matching total-wall comparison. Do not derive `average_dit_s` by dividing a
bundled process wall timer by the diffusion-step count. Module extrapolations
are hypotheses until the cumulative ON path passes a real full-DiT benchmark.

Likely target-model source regions include the transformer network files in
your experiment worktree: block-definition modules, attention output or AttnRes
paths, QK norm plus RoPE, linear/softmax attention, SwiGLU/FFN, and
residual/modulation glue. Treat this as a search boundary, not a fixed checklist.

Reference source: the target model's published kernel-fusion technique
documentation, when available.

## Execution round limit

You have a hard budget of **40 optimization rounds** (candidate attempts) for this
workflow. This budget is deliberately larger than a fusion-only search needs,
because the lossless space above spans several distinct families and its
highest-impact levers (data placement/movement, primitive/backend selection,
cross-step or cross-segment invariant reuse, multi-device communication) usually
require a **full-inference** validation rather than a cheap module microbenchmark,
so they cost more per round. Spend the budget in proportion to profiled impact:
screen broadly and cheaply where you can, invest full-inference rounds on the
ranked high-impact levers, and **deliver early if the lossless frontier plateaus** —
do not burn rounds refining a converged operator-fusion stack when your profile
shows the remaining recoverable time is elsewhere. One round = one candidate
implemented, launched, evaluated, and gated; do not plan beyond 40 rounds. This
scope's round budget governs for this workflow.

## Goal Target And Exit Conditions

This workflow runs against an **aspirational stretch target** — currently
**2.5× over the frozen baseline** — used only to create search pressure and
direction. Treat it as a direction, **NOT a delivery gate**: it is likely beyond
the lossless ceiling for this model (the fastest known mathematically-lossless
path for this model is well short of 2.5×), and you are not required to reach it.

**Stop and deliver your best retained frontier as soon as ANY of these is true —
whichever comes first:**

1. **Round cap** — you reach the hard round budget above (40 rounds).
2. **No improvement (plateau)** — your retained lossless frontier has not improved
   for several consecutive rounds (use judgment; roughly 3–4 rounds with no new
   best after genuinely different hypotheses).
3. **Target reached** — a candidate meets or exceeds the stretch target *and*
   passes the full lossless correctness gate.

Do not keep searching past a plateau merely because the target is unmet — an
unmet aspirational target is an expected, acceptable outcome, not a failure.

When you exit without reaching the target (the expected case), deliver your real
best and include an honest **gap report** in the delivery: the best speedup you
actually achieved, how far it is from the target, and why the remaining gap is
what it is — which lossless levers you exhausted, which you judged unavailable or
out of budget, and whether the remainder looks like a genuine lossless ceiling.

**Never** fabricate or overstate a number, and **never** relax the lossless
correctness definition, drop model work, or approximate to close the gap. A
smaller *true* lossless speedup always beats a larger number obtained by breaking
correctness — the master independently re-verifies both the speedup and the
computation-equivalence, so any such attempt is rejected and resumed.
