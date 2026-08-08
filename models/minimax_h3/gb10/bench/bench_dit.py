#!/usr/bin/env python3
"""Time one MiniMax-H3 DiT forward on a recorded latent, and check it stayed lossless.

This is the unit every optimisation is measured on. It loads the denoiser alone — no
conditioner, no VAEs — feeds it inputs captured from the official cell by
`capture_dit_inputs.py`, and times a single `forward` after warmup.

Two things it deliberately does not do:

* **No random inputs.** Sparsity is a property of the real activations, and a numerical
  check is only meaningful against outputs the model actually produces.
* **No reduced geometry.** 1344x768 / 124 frames, so 38,247 packed rows, exactly the cell
  Sol-Engine reports against.

`--save-reference` records the baseline output; every later run compares against it, so a
"lossless" claim is checked rather than asserted.

    python bench_dit.py --save-reference     # establish the baseline
    python bench_dit.py                      # time a variant, verify it matches
    python bench_dit.py --breakdown          # where the time goes, by module
    python bench_dit.py --profile            # where the time goes, by CUDA kernel
"""

from __future__ import annotations

import paths

paths.setup(need_sol_engine=False)

import argparse
import glob
import json
import os
import sys
from collections import defaultdict


os.environ.setdefault("HF_HOME", str(paths.HF_CACHE))

import torch

CAPTURE_DIR = str(paths.CAPTURE_DIR)
REFERENCE_DIR = str(paths.REFERENCE_DIR)


def load_transformer(device: str = "cuda", fuse_qkv: bool = False, quantizer: str = "eager",
                     fuse_adaln: bool = False, fuse_rope: bool = False,
                     fuse_swiglu: bool = False, sol_attn_tau: float | None = None):
    from h3_fp8 import build_pruned_fp8_transformer

    snapshot = paths.h3_snapshot()
    checkpoint = paths.dit_checkpoint()
    config = json.load(open(f"{snapshot}/transformer/config.json"))
    # The sweep needs both QKV forms resident to toggle between them in one process.
    return build_pruned_fp8_transformer(checkpoint, config, device=device, fuse_qkv=fuse_qkv,
                                       keep_split_qkv=fuse_qkv,
                                       quantizer=quantizer, fuse_adaln=fuse_adaln,
                                       fuse_rope=fuse_rope, fuse_swiglu=fuse_swiglu,
                                       sol_attn_tau=sol_attn_tau)


def load_inputs(step: int, device: str = "cuda", cell: str | None = None) -> dict:
    directory = CAPTURE_DIR if cell is None else os.path.join(CAPTURE_DIR, cell)
    path = os.path.join(directory, f"step_{step:02d}.pt")
    if not os.path.exists(path):
        available = sorted(os.path.basename(p) for p in glob.glob(f"{directory}/step_*.pt"))
        raise SystemExit(f"{path} not found; captured steps: {available or 'none — run capture_dit_inputs.py'}")
    blob = torch.load(path, weights_only=False)
    tensors = {k: v.to(device) for k, v in blob["tensors"].items() if k != "self"}
    return {"tensors": tensors, "meta": blob}


@torch.no_grad()
def time_forward(model, inputs: dict, warmup: int, iters: int) -> dict:
    for _ in range(warmup):
        model(**inputs, return_dict=False)
    torch.cuda.synchronize()

    samples = []
    for _ in range(iters):
        start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
        start.record()
        out = model(**inputs, return_dict=False)
        end.record()
        torch.cuda.synchronize()
        samples.append(start.elapsed_time(end))
    return {"ms": samples, "mean_ms": sum(samples) / len(samples), "min_ms": min(samples),
            "output": out}


@torch.no_grad()
def module_breakdown(model, inputs: dict) -> list[tuple[str, float, int]]:
    """Per-module CUDA time for one forward, bucketed by role.

    Events are recorded around each module and read once at the end, so the hooks do not
    serialise the stream the way a per-module `synchronize()` would.
    """
    def bucket(name: str) -> str | None:
        if ".attn.to_qkv" in name:
            return "attn: qkv proj (fused)"
        if ".attn.to_q" in name or ".attn.to_k" in name or ".attn.to_v" in name:
            return "attn: qkv proj"
        if ".attn.to_out" in name:
            return "attn: out proj"
        if name.endswith(".attn"):
            return "attn: total (incl. proj)"
        if ".ff.net.0" in name:
            return "ffn: up (SwiGLU)"
        if ".ff.net.2" in name:
            return "ffn: down"
        if "adaln_proj" in name:
            return "adaln"
        if ".norm1" in name or ".norm2" in name:
            return "norm"
        if name.startswith("token_refiner"):
            return "token refiner"
        if name in ("proj_in", "audio_proj_in", "context_embedder", "proj_out",
                    "audio_proj_out", "norm_out", "rope", "time_embedder"):
            return "io / embed"
        return None

    spans: dict[str, list] = defaultdict(list)
    handles = []
    for name, module in model.named_modules():
        role = bucket(name)
        if role is None:
            continue
        pending = {}

        def pre(mod, _args, _role=role, _pending=pending):
            event = torch.cuda.Event(enable_timing=True)
            event.record()
            _pending["start"] = event

        def post(mod, _args, _out, _role=role, _pending=pending):
            event = torch.cuda.Event(enable_timing=True)
            event.record()
            spans[_role].append((_pending["start"], event))

        handles.append(module.register_forward_pre_hook(pre))
        handles.append(module.register_forward_hook(post))

    model(**inputs, return_dict=False)
    torch.cuda.synchronize()
    for handle in handles:
        handle.remove()

    rows = [(role, sum(s.elapsed_time(e) for s, e in pairs), len(pairs))
            for role, pairs in spans.items()]
    return sorted(rows, key=lambda r: -r[1])


@torch.no_grad()
def kernel_profile(model, inputs: dict, top: int = 25):
    from torch.profiler import ProfilerActivity, profile

    model(**inputs, return_dict=False)
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        model(**inputs, return_dict=False)
        torch.cuda.synchronize()
    print(prof.key_averages().table(sort_by="self_device_time_total", row_limit=top))


def compare(output, reference) -> list[str]:
    lines = []
    for name, got, want in zip(("video", "audio"), output, reference):
        got, want = got.float(), want.to(got.device).float()
        diff = (got - want).abs()
        denom = want.abs().mean().clamp_min(1e-12)
        cos = torch.nn.functional.cosine_similarity(
            got.flatten().double(), want.flatten().double(), dim=0
        )
        lines.append(
            f"  {name}: max|d|={diff.max():.3e}  mean|d|/mean|ref|={diff.mean() / denom:.3e}  "
            f"cos={cos:.9f}  {'EXACT' if torch.equal(got, want) else ''}"
        )
    return lines


SWEEP_TOGGLES = ("quantizer", "adaln", "rope")


def make_switcher(model):
    """Return `apply(config)`, flipping the runtime-patchable optimisations in place.

    Everything except the fused QKV is a patch rather than a structural change, so one model
    can serve every configuration — which is the point: variants have to be interleaved inside
    a single process. Across processes this machine's clocks drift up to 15% (the same
    attention call measured 23.6 s and 26.4 s in two runs), while three iterations inside one
    process agree to 0.7%. Comparing 40.1 s from one run against 43.4 s from another says
    almost nothing.
    """
    import h3_fp8
    from diffusers.models.transformers import transformer_minimax_h3 as h3

    eager_block_forward = h3.MiniMaxH3TransformerBlock.forward
    eager_rope = h3._apply_rotary_emb
    from diffusers.models.activations import SwiGLU
    eager_swiglu = SwiGLU.forward
    h3_fp8.patch_fused_swiglu(model)
    fused_swiglu_forward = SwiGLU.forward
    SwiGLU.forward = eager_swiglu
    h3_fp8.patch_fused_adaln(model)
    fused_block_forward = h3.MiniMaxH3TransformerBlock.forward
    h3.MiniMaxH3TransformerBlock.forward = eager_block_forward

    linears = [m for m in model.modules()
               if isinstance(m, h3_fp8.Fp8Linear) and m.quantized_activations]
    attentions = [m for m in model.modules() if hasattr(m, "to_qkv")]
    dense_attn = h3.dispatch_attention_fn
    # `dense_first_steps` / `dense_last_steps` are deliberately left at zero here. They are
    # indexed by denoising step, and the benchmark calls the model repeatedly, so the step
    # counter would keep climbing and the first forwards would be measured dense. Their effect
    # on a trajectory is a weighted sum of the two per-step times, computed after the fact.
    sol_dispatch = {
        (tau, blocks): h3_fp8.make_sol_attn_dispatch(tau=tau, dense_blocks=blocks)
        for tau in (1.0, 2.0, 4.0) for blocks in (0, 2)
    }

    def apply(config):
        for module in attentions:
            module.fused_projections = config["qkv"]
        for module in linears:
            module._quantize = h3_fp8.QUANTIZERS[config["quantizer"]]
        h3.MiniMaxH3TransformerBlock.forward = (
            fused_block_forward if config["adaln"] else eager_block_forward
        )
        h3._apply_rotary_emb = (
            h3_fp8.fused_apply_rotary_emb if config["rope"] else eager_rope
        )
        SwiGLU.forward = fused_swiglu_forward if config["swiglu"] else eager_swiglu
        tau = config.get("sol_tau")
        if tau is None:
            h3.dispatch_attention_fn = dense_attn
        else:
            dispatch = sol_dispatch[(tau, config.get("dense_blocks", 2))]
            dispatch.reset()
            h3.dispatch_attention_fn = dispatch

    return apply


@torch.no_grad()
def sweep(model, inputs, configs, rounds: int):
    """Interleave the configurations round-robin so drift lands on all of them equally."""
    apply = make_switcher(model)
    samples = {name: [] for name in configs}
    outputs = {}

    for name, config in configs.items():          # warm every variant first
        apply(config)
        model(**inputs, return_dict=False)
    torch.cuda.synchronize()

    for _ in range(rounds):
        for name, config in configs.items():
            apply(config)
            start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
            start.record()
            out = model(**inputs, return_dict=False)
            end.record()
            torch.cuda.synchronize()
            samples[name].append(start.elapsed_time(end))
            outputs[name] = [t.detach().clone() for t in out]
    return samples, outputs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--step", type=int, default=24, help="which captured step to run on")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iters", type=int, default=3)
    parser.add_argument("--save-reference", action="store_true")
    parser.add_argument("--breakdown", action="store_true")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--fuse-qkv", action="store_true",
                        help="one QKV GEMM behind `attn.fused_projections` instead of three")
    parser.add_argument("--quantizer", choices=("eager", "compiled", "triton"), default="eager",
                        help="how activations are cast to FP8; only `triton` is bit-exact")
    parser.add_argument("--fuse-adaln", action="store_true",
                        help="gather the modulation tables inside the elementwise kernels")
    parser.add_argument("--fuse-swiglu", action="store_true",
                        help="fuse chunk/silu/mul after fc1")
    parser.add_argument("--fuse-rope", action="store_true",
                        help="one kernel for rotary instead of neg/cat/mul/mul/add/cat")
    parser.add_argument("--sol-attn-tau", type=float, default=None,
                        help="route the block stack through Sol-Attn's Triton reference")
    parser.add_argument("--cell", default=None,
                        help="captured cell to run on, e.g. 832x480; omit for 1344x768")
    parser.add_argument("--sweep", action="store_true",
                        help="interleave every optimisation combination in one process")
    parser.add_argument("--rounds", type=int, default=4, help="sweep rounds per config")
    parser.add_argument("--tag", default="baseline")
    args = parser.parse_args()

    blob = load_inputs(args.step, cell=args.cell)
    inputs, meta = blob["tensors"], blob["meta"]
    model, info = load_transformer(fuse_qkv=args.fuse_qkv, quantizer=args.quantizer,
                                   fuse_adaln=args.fuse_adaln, fuse_rope=args.fuse_rope,
                                   fuse_swiglu=args.fuse_swiglu, sol_attn_tau=args.sol_attn_tau)
    print(f"[bench] DiT {torch.cuda.memory_allocated() / 2**30:.2f} GiB, "
          f"{info['quantized_layers']} quantised linears, "
          f"qkv={'fused' if info['fused_qkv'] else 'split'}, "
          f"quant={info['quantizer']}, adaln={'fused' if info['fused_adaln_blocks'] else 'eager'}, "
          f"rope={'fused' if info['fused_rope'] else 'eager'}",
          flush=True)
    print(f"[bench] step {meta['call_index']} of {meta['num_scheduler_steps']}, "
          f"timestep {meta['timestep']}, {meta['width']}x{meta['height']} "
          f"{meta['num_frames']}f, {inputs['position_ids'].shape[0]} packed rows", flush=True)

    if args.sweep:
        configs = {
            "baseline":       {"qkv": False, "quantizer": "eager",  "adaln": False, "rope": False, "swiglu": False},
            "+ fused qkv":    {"qkv": True,  "quantizer": "eager",  "adaln": False, "rope": False, "swiglu": False},
            "+ triton quant": {"qkv": True,  "quantizer": "triton", "adaln": False, "rope": False, "swiglu": False},
            "+ fused adaln":  {"qkv": True,  "quantizer": "triton", "adaln": True,  "rope": False, "swiglu": False},
            "+ fused rope":   {"qkv": True,  "quantizer": "triton", "adaln": True,  "rope": True,  "swiglu": False},
            "+ fused swiglu": {"qkv": True,  "quantizer": "triton", "adaln": True,  "rope": True,  "swiglu": True},
            "sol tau=1":         {"qkv": True, "quantizer": "triton", "adaln": True, "rope": True, "swiglu": True, "sol_tau": 1.0},
            "sol tau=2":         {"qkv": True, "quantizer": "triton", "adaln": True, "rope": True, "swiglu": True, "sol_tau": 2.0},
            "sol tau=4":         {"qkv": True, "quantizer": "triton", "adaln": True, "rope": True, "swiglu": True, "sol_tau": 4.0},
            "sol tau=4":         {"qkv": True, "quantizer": "triton", "adaln": True, "rope": True, "swiglu": True, "sol_tau": 4.0},
            "sol tau=2, blk=0":  {"qkv": True, "quantizer": "triton", "adaln": True, "rope": True, "swiglu": True, "sol_tau": 2.0, "dense_blocks": 0},
        }
        samples, outputs = sweep(model, inputs, configs, args.rounds)
        first = next(iter(configs))
        base = min(samples[first])
        print(f"\n{'config':22s} {'min ms':>9s} {'median':>9s} {'vs first':>9s} {'vs prev':>8s}  match")
        previous = None
        for name in configs:
            values = sorted(samples[name])
            mid = values[len(values) // 2]
            exact = all(torch.equal(a, b) for a, b in zip(outputs[name], outputs[first]))
            prev = f"{previous / values[0]:7.3f}x" if previous else "     -"
            print(f"{name:22s} {values[0]:9.0f} {mid:9.0f} {base / values[0]:8.3f}x {prev}  "
                  f"{'EXACT' if exact else 'DIFFERS'}")
            previous = values[0]
        print(f"\nraw samples (round-robin, {args.rounds} rounds):")
        for name in configs:
            print(f"  {name:22s} {[f'{v:.0f}' for v in samples[name]]}")

        # Interleaving only cancels interference that is uniform across the round-robin. When
        # another job on this shared GPU is bursty it is not, and the giveaway is a config that
        # adds a strictly-less-work, bit-exact optimisation coming out slower. That happened
        # once with `+ fused qkv` at 125,295 ms against a 59,225 ms baseline; the run was
        # unusable and nothing in the output said so.
        lossless = [n for n in configs if configs[n].get("sol_tau") is None]
        regressions = [
            (a, b, min(samples[a]), min(samples[b]))
            for a, b in zip(lossless, lossless[1:])
            if min(samples[b]) > min(samples[a]) * 1.05
        ]
        if regressions:
            print("\n*** RUN INVALID: a strictly-cheaper bit-exact config measured slower ***")
            for a, b, ta, tb in regressions:
                print(f"    {b!r} {tb:.0f} ms > {a!r} {ta:.0f} ms")
            print("    another process was almost certainly contending; re-run when idle")
        return 0

    if args.profile:
        kernel_profile(model, inputs)
        return 0

    if args.breakdown:
        rows = module_breakdown(model, inputs)
        total = sum(r[1] for r in rows if "total" not in r[0])
        print(f"\n{'role':28s} {'ms':>9s} {'%':>7s} {'calls':>7s}")
        for role, ms, calls in rows:
            share = "" if "total" in role else f"{100 * ms / total:6.1f}%"
            print(f"{role:28s} {ms:9.1f} {share:>7s} {calls:7d}")
        print(f"{'sum (excl. attn total)':28s} {total:9.1f}")
        return 0

    result = time_forward(model, inputs, args.warmup, args.iters)
    print(f"\n[bench] {args.tag}: mean {result['mean_ms']:.1f} ms, min {result['min_ms']:.1f} ms "
          f"over {args.iters} iters ({[f'{v:.0f}' for v in result['ms']]})", flush=True)

    # The reference has to be per-cell: 832x480 produces 14,430 video rows against the
    # official cell's 37,296.
    reference_dir = REFERENCE_DIR if args.cell is None else os.path.join(REFERENCE_DIR, args.cell)
    os.makedirs(reference_dir, exist_ok=True)
    reference_path = os.path.join(reference_dir, f"step_{args.step:02d}.pt")
    if args.save_reference:
        torch.save([t.detach().cpu() for t in result["output"]], reference_path)
        print(f"[bench] wrote reference {reference_path}")
    elif os.path.exists(reference_path):
        print("[bench] against reference:")
        for line in compare(result["output"], torch.load(reference_path, weights_only=False)):
            print(line)
    else:
        print(f"[bench] no reference at {reference_path}; run --save-reference first")

    record = {"tag": args.tag, "fused_qkv": args.fuse_qkv,
              "quantizer": args.quantizer, "fused_adaln": args.fuse_adaln,
              "fused_rope": args.fuse_rope,
              "step": args.step, "mean_ms": result["mean_ms"],
              "min_ms": result["min_ms"], "ms": result["ms"],
              "packed_rows": int(inputs["position_ids"].shape[0])}
    with open(str(paths.ROOT / "dit_bench_log.jsonl"), "a") as handle:
        handle.write(json.dumps(record) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
