#!/usr/bin/env python3
"""Time one DiT forward in BF16 and in FP8, on the same recorded inputs, in one process.

The BF16 model is the released `MiniMaxAI/MiniMax-H3` transformer, loaded by diffusers with
nothing done to it. The FP8 model is this port of ComfyUI's pruned checkpoint. Both are handed
the identical tensors captured by `capture_dit_inputs.py`: the official forward derives
`adaln_indices` from `timestep_indices` and `token_tags` itself, so the recorded blob feeds
either model without translation.

One model per process, by `--only`. Two designs were tried first and both were wrong in ways
that looked like results:

* Holding both at once is 85 GB on a part whose 121 GB is shared with the host. It did not
  fail cleanly — the machine paged for an hour and stopped answering ssh.
* Alternating inside one process is worse than it sounds. Freeing the BF16 model returns its
  memory to the caching allocator, not to the kernel: with 29.1 GiB reported by torch the
  system still showed 116 GB used and 5 GB available, and the FP8 configuration measured under
  that pressure came out at 20,746 ms against the 9,790 ms it takes from a clean start. That
  is a plausible-looking number produced entirely by the previous model.

So each measurement gets a fresh process that exits. That reintroduces cross-process drift,
which runs up to 15% here, and the answer to it is alternation rather than isolation: run
bf16 / fp8 / bf16 / fp8 and take the minimum of each, so a slow drift in clocks or temperature
lands on both.

One hazard the code has to work around: `make_switcher` patches *module-level* attributes
(`MiniMaxH3TransformerBlock.forward`, `_apply_rotary_emb`, `SwiGLU.forward`), which both models
share, and the fused versions assume the FP8 port's structures. So the switcher is put back to
all-eager before the FP8 model is dropped, leaving the module in the state the next round's
BF16 model expects.

What the ratio does and does not isolate:

* It is not precision alone. The released checkpoint is unpruned and computes AdaLN modulation
  from a full-width time embedding; the ComfyUI one carries a rank-8 factorisation of the same
  thing, measured elsewhere at 1.98e-4 mean relative error — finer than BF16 itself, so it is a
  faithful factorisation, but it is still less arithmetic.
* The FP8 side is this port, so it is reported at both ends: unfused, which is closest to "same
  model, narrower numbers", and fully fused, which is what actually runs.

The honest reading is "released BF16 model against the FP8 stack in use", not "FP8 is N times
faster than BF16 at identical work".
"""

from __future__ import annotations

import paths

paths.setup(need_sol_engine=False)

import argparse

import torch

from bench_dit import load_inputs, load_transformer, make_switcher, time_forward


def record(tag: str, name: str, result: dict) -> None:
    """Append one measurement, so a caller alternating processes can pool them afterwards."""
    import json

    path = paths.ROOT / "precision_samples.jsonl"
    with open(path, "a") as handle:
        handle.write(json.dumps({"model": tag, "config": name,
                                 "min_ms": result["min_ms"], "ms": result["ms"]}) + "\n")


def free(model) -> None:
    """Drop a model and give the memory back before loading the next one."""
    import gc

    del model
    gc.collect()
    torch.cuda.empty_cache()


EAGER = {"qkv": False, "quantizer": "eager", "adaln": False, "rope": False, "swiglu": False}
FUSED = {"qkv": True, "quantizer": "triton", "adaln": True, "rope": True, "swiglu": True}


def load_bf16(device: str = "cuda"):
    """The released transformer, as diffusers builds it."""
    from diffusers.models.transformers.transformer_minimax_h3 import (
        MiniMaxH3Transformer3DModel,
    )

    model = MiniMaxH3Transformer3DModel.from_pretrained(
        paths.h3_snapshot(), subfolder="transformer", torch_dtype=torch.bfloat16
    )
    return model.to(device).eval()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cell", default="832x480")
    parser.add_argument("--step", type=int, default=24)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iters", type=int, default=2)
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--only", choices=("bf16", "fp8"),
                        help="one model per process; the caller alternates them")
    parser.add_argument("--summarize", action="store_true",
                        help="pool precision_samples.jsonl and print the ratio")
    args = parser.parse_args()
    if args.summarize:
        return summarize()
    if args.only is None:
        parser.error("--only is required unless --summarize")

    blob = load_inputs(args.step, cell=args.cell)
    inputs = blob["tensors"]
    print(f"inputs: {args.cell} step {args.step}, "
          f"{inputs['hidden_states'].shape[1]} packed rows\n", flush=True)

    if args.only == "bf16":
        for round_index in range(args.rounds):
            print(f"round {round_index}: loading BF16 (61.7 GiB)...", flush=True)
            model = load_bf16()
            print(f"  {torch.cuda.memory_allocated() / 2**30:.1f} GiB", flush=True)
            result = time_forward(model, inputs, args.warmup, args.iters)
            print(f"  bf16 (released)      {result['min_ms']:9.0f} ms", flush=True)
            record(args.only, "bf16 (released)", result)
            free(model)
        return 0

    print("loading FP8 (23.3 GiB)...", flush=True)
    # Only the QKV fusion, which is what `make_switcher` expects: it captures the current
    # block forward, rope and SwiGLU as its "eager" versions at construction, so loading with
    # the fusions already applied makes it capture the fused ones and patch on top of them.
    # `keep_split_qkv` follows `fuse_qkv`, so both projection forms stay resident.
    model, info = load_transformer(fuse_qkv=True, quantizer="eager", fuse_adaln=False,
                                   fuse_rope=False, fuse_swiglu=False)
    print(f"  {torch.cuda.memory_allocated() / 2**30:.1f} GiB, "
          f"{info['quantized_layers']} quantised linears", flush=True)
    switch = make_switcher(model)
    for round_index in range(args.rounds):
        for name, config in (("fp8 (unfused)", EAGER), ("fp8 (all lossless)", FUSED)):
            switch(config)
            result = time_forward(model, inputs, args.warmup, args.iters)
            print(f"  {name:20s} {result['min_ms']:9.0f} ms", flush=True)
            record(args.only, name, result)
    return 0


def summarize() -> int:
    """Pool the alternated runs and report the ratio.

    Minimum rather than mean: every source of noise here is additive — a scheduler tick, a
    page fault, another tenant — so the fastest observation is the closest to the model's own
    cost. The spread is printed alongside so a run where drift dominated is visible rather
    than averaged away.
    """
    import json
    from collections import defaultdict

    path = paths.ROOT / "precision_samples.jsonl"
    if not path.exists():
        raise SystemExit(f"no samples at {path}")
    samples = defaultdict(list)
    with open(path) as handle:
        for line in handle:
            row = json.loads(line)
            samples[row["config"]].append(row["min_ms"])

    base = min(samples["bf16 (released)"])
    print(f"{'config':22s} {'min ms':>9s} {'vs bf16':>9s} {'spread':>8s}  observations")
    for name in ("bf16 (released)", "fp8 (unfused)", "fp8 (all lossless)"):
        if name not in samples:
            continue
        values = samples[name]
        best = min(values)
        print(f"{name:22s} {best:9.0f} {base / best:8.2f}x "
              f"{(max(values) - best) / best:7.1%}  {[f'{v:.0f}' for v in values]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
