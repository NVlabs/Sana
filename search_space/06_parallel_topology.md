# Search Space: Multi-GPU Parallel Topology

Goal: reduce inference latency or peak device memory by changing only the exact
distributed execution of a frozen model workload on a frozen resource envelope.

This is a method-family and evidence contract, not a fixed topology grid. The
target model's shapes, divisibility, process groups, current kernels, and measured
communication/compute balance determine useful candidates.

## Method families

- Context and sequence parallelism: Ulysses, Ring, or justified hybrids.
- Tensor parallelism: attention heads, dense/FFN dimensions, and expert-internal
  tensor parallelism.
- Expert parallelism: expert ownership, token dispatch/combine, load balance,
  and exact communication/compute pipelining.
- Parameter sharding and residency: FSDP, reduce-scatter/all-gather variants,
  full replication, staged materialization, and exact prefetch/offload.
- CFG execution: branch process groups, branch placement, or equivalent batched
  guidance without changing guidance semantics.
- Device mesh and rank mapping: process-group factorization, nesting/reuse,
  rank ordering, NUMA/NVLink/fabric-aware placement, and multi-stage placement.
- Collective scheduling: exact all-to-all, all-reduce, reduce-scatter,
  all-gather, broadcast, and P2P Ring chunking or overlap.

## Search axes

- Fixed resource envelope: world size, nodes, GPUs per node, device type, fabric,
  Slurm allocation, warmup policy, and timing scope.
- Parallel degrees and group relationships: CP/SP, TP, EP, CFG, DP, and sharding
  groups, including whether axes are orthogonal, nested, or share ranks.
- Tensor placement: global/local shapes, shard dimension, padding, parameter and
  expert ownership, activation residency, and stage placement.
- Communication: collective type, algorithm, bucket/chunk size, stream, async
  launch point, wait point, buffer lifetime, bytes, call count, and rank skew.
- Scheduling: prefetch distance, layer pipeline, dispatch/compute/combine
  overlap, stage transition, and offload timing when inside the measured scope.
- Fallback: shape/dtype/world-size guards, explicit OFF path, zero silent
  fallback, and recovery behavior.

## Required proof and measurements

- Preserve the global logical workload: checkpoint, prompts, seed, scheduler,
  steps, guidance, frames, resolution, dtypes, and output contract.
- Prove complete token/head/feature, parameter-shard, expert-assignment, and CFG
  branch coverage with no loss or duplication.
- Record every rank's mesh coordinate and every process group's membership.
- Record collective order, bytes, calls, time, overlap, rank skew, and per-rank
  peak memory from the actual candidate run.
- Require a full frozen-workload latency measurement; microbench projections are
  screening evidence only.
- Preserve a real OFF path that restores the baseline topology.

## Ownership boundaries

Local attention/GEMM backend selection, operator fusion, custom kernels, and
compile belong to kernel optimization. Cross-step model-output reuse belongs to
cache optimization. Approximate attention belongs to PISA/sparse-attention
optimization. Quantization, pruning, step reduction, and workload changes are
outside this lossless topology space.
