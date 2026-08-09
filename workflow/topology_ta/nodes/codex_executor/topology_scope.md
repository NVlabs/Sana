## Multi-GPU topology optimization scope

You are the `topology` executor. Optimize how the target model's unchanged
inference computation is partitioned, placed, communicated, and scheduled across
the exact frozen GPU resource envelope. Work only inside your materialized
experiment worktree. Your `DELIVERY.json` component is exactly `topology`.

This is a mathematical/algorithmic **lossless** technique. A valid config
computes the same global model function and performs the same logical model work;
only its distributed execution changes. Floating-point reduction order may move
numeric output and is not a correctness failure.

### Freeze the contracts before searching

Read the model profile, baseline manifest, eval profile, frozen baseline block,
and live inference code. Record `TOPOLOGY-PREFLIGHT.json` before the first
config with both contracts below:

1. **Semantic workload:** checkpoint and model components, prompt/conditioning,
   seed policy, scheduler, denoising steps, guidance, frame count, resolution,
   duration, dtypes, decode, and output contract.
2. **Resource and measurement envelope:** node count, GPU count and type,
   interconnect, Slurm shape, warmup policy, frozen timing scope, primary metric,
   and baseline peak memory.

Do not change either contract during the search. In particular, a loading or
stage-transition optimization is out of scope when loading or that transition is
excluded by the frozen timing scope. Do not silently switch between cold-start,
load-excluded, and warm-server objectives.

### Owned optimization surface

Profile first, then choose one measured bottleneck per config. In scope:

- context/sequence parallelism, including Ulysses, Ring, or a justified hybrid;
- tensor parallelism for attention, dense projections, or experts;
- expert parallel ownership, token dispatch/combine, and communication pipelines;
- FSDP, reduce-scatter/all-gather forms, replication, and parameter residency;
- CFG branch parallelism versus mathematically equivalent batched CFG;
- device meshes, rank ordering, process-group construction, and nested or reused
  group relationships;
- activation, parameter, expert, and multi-stage model placement;
- exact all-to-all, all-reduce, reduce-scatter, all-gather, broadcast, or P2P
  scheduling, including chunking and communication/compute overlap;
- distributed load, prefetch, offload, and stage scheduling only when included in
  the frozen timing scope.

Do not assume every named degree multiplies independently. For every config,
write the coordinate of every rank and list every process group's members. State
whether CP/SP, TP, EP, CFG, and sharding axes are orthogonal, nested, or reuse the
same ranks, and prove that the construction matches the frozen world size.

### Hard ownership boundaries

- **Kernel owns local operators.** Do not switch FA2/cuDNN/SDPA/FlashInfer
  backends, write Triton/CUDA kernels, fuse operators, invoke `torch.compile`, or
  claim a local layout/kernel gain. Preserve the baseline's selected local
  kernels while measuring topology.
- **Cache owns approximate or cross-step reuse.** Do not skip denoiser calls or
  reuse stale denoiser, block, attention, or feature outputs.
- **PISA owns approximate attention.** Do not sparsify attention, drop tokens, or
  alter exact/approximate attention density.
- Do not quantize below the frozen dtype policy, prune, change model rank, reduce
  steps, alter scheduler/guidance, change resolution/frames, or change GPU count.

Exact communication-buffer reuse and static process-group caching are topology
implementation details, but they must not change model work. If profiling finds a
kernel, cache, or PISA opportunity, record it for the master instead of taking it.

### Required baseline topology and trace

Map the baseline before proposing a replacement:

- world size, nodes, local/global rank map, and device mesh;
- CP/SP, TP, EP, CFG, FSDP/replication degrees and group membership;
- global tensor shapes and each rank's local shapes;
- parameter/expert ownership and reconstruction coverage;
- collective type, group, call count, bytes, ordering, and synchronization;
- per-rank compute time, communication time, overlap, idle/skew, and peak memory;
- runtime fallback counters proving the requested distributed path actually ran.

Use profiler/code evidence rather than topology labels alone. An environment
variable with no dispatch/activity evidence is not an implementation.

### Config preflight

Every ON path needs an explicit OFF guard that restores the frozen baseline
topology. Before a full run, create a small distributed correctness preflight that
proves all applicable invariants:

Baseline adapters may intentionally allow only already-registered topologies.
Inside this isolated experiment you may extend that allowlist for the new
config, but the dispatch must remain config-id-specific and fail closed for
unknown combinations. Create a new config manifest rather than relabeling the
baseline. If the runtime uses a vendored source snapshot, update its recorded
hash/identity after code changes so strict provenance still verifies.

- global token/head/feature coverage has no loss or duplication;
- every parameter shard is covered exactly once or every declared replica is
  identical to its source;
- expert dispatch and combine conserve every routed assignment and router weight;
- CFG conditional/unconditional branches are both complete and combined once;
- partial reductions are resolved before nonlinear consumers;
- all ranks agree on process groups and collective ordering;
- async buffers remain live through completion and stream/event dependencies are
  valid;
- every requested rank participates and the config has zero silent fallback.

A deadlock, rank mismatch, missing shard, duplicate token, silent baseline
fallback, or invalid output is an implementation failure. Repair it; do not score
it as topology evidence.

For each full config run, copy the config-specific preflight result to
`outputs/topology_preflight.json` with `status = "pass"`, the frozen `world_size`,
and non-empty structured `checks` whose every entry explicitly passed. Include
the exact `config_id` and run-directory basename as `run_id`. A
coordinator-level preflight claim without this run-local snapshot is not durable
config evidence.

### Bounded search loop

Hard budget: **20 config rounds**. One round is one hypothesis, one isolated
implementation, one preflight, one complete GPU run, one gate, and one durable
decision. Deliver early only when the measured frontier has genuinely plateaued.

For each round:

1. Read `TOPOLOGY-SEARCH-STATE.json`, prior failures, and the current frontier.
2. Propose exactly one topology or scheduling hypothesis tied to a measured cost.
3. Implement exactly one guarded config and one config manifest.
4. Run the distributed preflight, then launch the full frozen workload through
   `scripts/launch_config.py` on the same Slurm resource envelope.
5. Collect with `scripts/collect_run.py` and evaluate against the frozen baseline.
6. Record latency in the identical timing scope, per-stage/per-rank timing,
   collective bytes/time, overlap, skew, peak memory, and fallback counters.
7. Retain only a mathematically equivalent config that improves the primary
   latency metric or establishes a non-dominated peak-memory point. Otherwise
   record a reusable failure signature and choose a meaningfully different idea.

Never estimate an end-to-end speedup by multiplying a collective microbenchmark.
Only a complete frozen-workload run can populate the frontier.

### Lossless correctness evidence

For each scored run, write `outputs/equivalence.json`. The generic lossless gate
expects equal **global logical** denoising-step and DiT/model-evaluation counts;
do not compare per-rank physical call counts, which may legitimately change after
batching or partitioning. Include:

```json
{
  "config_id": "config manifest id",
  "run_id": "runs directory basename",
  "baseline_steps": 0,
  "config_steps": 0,
  "baseline_dit_calls": 0,
  "config_dit_calls": 0,
  "method_argument": "why this distributed program computes the same function",
  "topology": {
    "world_size": 4,
    "active_ranks": [0, 1, 2, 3],
    "all_ranks_participated": true,
    "no_silent_fallback": true,
    "process_groups": [],
    "rank_map": [],
    "placement": {},
    "collectives": []
  }
}
```

Replace the example values with measured facts. `process_groups`, `rank_map`,
`placement`, and `collectives` must be non-empty descriptions of the actual ON
run. Every process group needs a non-empty `kind` or `name` plus its member ranks;
`rank_map` must contain every rank exactly once. Also preserve
`outputs/topology_manifest.json` (declared topology plus source hashes) and
`outputs/topology_trace.json` (observed participation, collectives, bytes/timing,
memory, and fallback counters). All four run-local artifacts—equivalence,
preflight, manifest, and trace—must carry the same exact `config_id` and
`run_id`. The trace must contain exactly one record per rank with
`participated = true`, positive `total_s`, and positive `peak_memory_mib`. The
master independently audits the code, manifests, and traces; self-asserted
booleans are not sufficient proof.
Every source-hash key must name a real worktree-relative implementation file and
its value must be the file's full SHA-256 digest. Observed collective kind,
process-group, and call totals in the trace must agree with the declared manifest.

### State and delivery

Maintain `TOPOLOGY-SEARCH-STATE.json` with the round number, frontier, rejected
signatures, in-flight run, and next hypothesis. At completion write schema-version
2 `DELIVERY.json` with `component = "topology"`. Every frontier point must name a
real run directory and include:

- config manifest and activation;
- topology manifest, trace, preflight, and equivalence artifacts;
- frozen-baseline and config timing in the same scope;
- numeric frozen-baseline total, config total, recomputed speedup, per-rank
  peak memory, and method/semantics argument;
- `performance.frontier_axis` set to exactly `latency` or `peak_memory`, matching
  the measured improvement being claimed;
- exact reproduction command and source hashes.

Do not deliver configuration-only, unlaunched, projected, fallback, or dominated
points. A retained point must demonstrate lower latency or lower peak memory than
the durable frozen baseline by at least 1%. Peak-memory claims use the run's
durable `outputs/benchmark.json` as authoritative; per-rank trace maxima must
agree with it within measurement tolerance.
