"""Paired public-API SM100 timings against a separate unchanged SOL checkout."""

import argparse
import hashlib
import importlib.util
import json
import statistics
import sys
import tempfile
import time
from pathlib import Path

import torch
from sol_attn import interface, sol_attn
from torch.nn.attention import SDPBackend, sdpa_kernel


def module(name, path, source):
    path.write_text(source)
    spec = importlib.util.spec_from_file_location(name, path)
    loaded = importlib.util.module_from_spec(spec)
    sys.modules[name] = loaded
    spec.loader.exec_module(loaded)
    return loaded


def baseline(root, generated):
    """Isolate baseline caches; change import names only, never kernel bodies."""
    root = root / "sol_attn"
    sources = {
        name: (root / name).read_text()
        for name in ("interface.py", "sm100/softmax.py", "sm100/mainloop.py")
    }
    name = "_benchmark_baseline"
    module(
        f"sol_attn.sm100.{name}_softmax",
        generated / "softmax.py",
        sources["sm100/softmax.py"],
    )
    text = sources["sm100/mainloop.py"]
    assert text.count("from .softmax import (") == 1
    module(
        f"sol_attn.sm100.{name}_mainloop",
        generated / "mainloop.py",
        text.replace("from .softmax import (", f"from .{name}_softmax import ("),
    )
    text = sources["interface.py"]
    assert text.count("from .sm100 import forward") == 1
    loaded = module(
        f"sol_attn.{name}_interface",
        generated / "interface.py",
        text.replace(
            "from .sm100 import forward", f"from .sm100.{name}_mainloop import forward"
        ),
    )
    hashes = {
        name: hashlib.sha256(text.encode()).hexdigest()
        for name, text in sources.items()
    }
    return loaded.sol_attn, hashes


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n")
    temporary.replace(path)


def tensors(case):
    batch, tokens, heads, protection, tau = case
    generator = torch.Generator(device="cuda").manual_seed(71)
    q, k, v = [
        torch.randn(
            batch,
            tokens,
            heads,
            128,
            device="cuda",
            dtype=torch.bfloat16,
            generator=generator,
        )
        for _ in range(3)
    ]
    mask = torch.zeros((tokens + 63) // 64, device="cuda", dtype=torch.bool)
    if protection == "all":
        mask.fill_(True)
    elif protection == "ends":
        mask[0] = mask[-1] = True
    return q, k, v, mask, dict(tau=tau, thresh_type="diag", scale=128**-0.5)


@torch.inference_mode()
def measure(q, k, v, mask, options, old, modes, warmup, repeats):
    def dense():
        with sdpa_kernel(SDPBackend.CUDNN_ATTENTION):
            return torch.nn.functional.scaled_dot_product_attention(
                q.transpose(1, 2),
                k.transpose(1, 2),
                v.transpose(1, 2),
                scale=options.get("scale", 128**-0.5),
            ).transpose(1, 2)

    functions = {
        "baseline": lambda: old(q, k, v, **options, exact_block_mask=mask),
        "candidate": lambda: sol_attn(q, k, v, **options, exact_block_mask=mask),
        "dense": dense,
    }
    reference = functions["baseline"]()
    torch.testing.assert_close(functions["candidate"](), reference, atol=0, rtol=0)
    report = dict(
        shape=list(q.shape),
        options=options,
        bitwise_equal=True,
        protected_blocks=int(mask.sum()) if mask is not None else 0,
        modes={},
    )
    for mode in modes:
        calls, graphs = {}, []
        for name, fn in functions.items():
            for _ in range(warmup):
                fn()
            torch.cuda.synchronize()
            if mode == "graph":
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    output = fn()
                graph.replay()
                torch.cuda.synchronize()
                if name != "dense":
                    torch.testing.assert_close(output, reference, atol=0, rtol=0)
                graphs.append((graph, output))
                calls[name] = graph.replay
            else:
                calls[name] = fn
        samples = []
        names = list(calls)
        for iteration in range(repeats):
            offset = iteration % len(names)
            order = names[offset:] + names[:offset]
            if iteration % 2:
                order.reverse()
            for name in order:
                torch.cuda.synchronize()
                begin, end = (
                    torch.cuda.Event(enable_timing=True),
                    torch.cuda.Event(enable_timing=True),
                )
                unix_s, start = time.time(), time.perf_counter()
                begin.record()
                calls[name]()
                end.record()
                torch.cuda.synchronize()
                samples.append(
                    dict(
                        round=iteration,
                        variant=name,
                        start_unix_s=unix_s,
                        wall_ms=1000 * (time.perf_counter() - start),
                        event_ms=begin.elapsed_time(end),
                    )
                )
        metrics = {}
        for name in names:
            values = [s["event_ms"] for s in samples if s["variant"] == name]
            metrics[name] = dict(
                median_ms=statistics.median(values),
                cv_percent=100 * statistics.stdev(values) / statistics.mean(values),
            )
        ratios = []
        for iteration in range(repeats):
            pair = {
                s["variant"]: s["event_ms"] for s in samples if s["round"] == iteration
            }
            ratios.append(pair["baseline"] / pair["candidate"])
        report["modes"][mode] = dict(
            samples=samples,
            metrics=metrics,
            paired_speedup_median=statistics.median(ratios),
            paired_speedups=ratios,
        )
    torch.testing.assert_close(functions["candidate"](), reference, atol=0, rtol=0)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline-root",
        type=Path,
        required=True,
        help="Unmodified techniques/sparse_backends directory",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--input", type=Path, help="Optional frozen H3 Q/K/V capture")
    parser.add_argument(
        "--stress-only",
        action="store_true",
        help="Check tiny values and changing maxima against baseline without timing",
    )
    parser.add_argument("--heads", type=int, default=28)
    parser.add_argument("--modes", default="eager,graph")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=16)
    args = parser.parse_args()
    if torch.cuda.get_device_capability() not in ((10, 0), (10, 3)):
        parser.error("SM100/SM103 required")
    if args.repeats < 2 or args.warmup < 1:
        parser.error("at least two repeats and one warmup are required")
    modes = args.modes.split(",")
    if any(mode not in ("eager", "graph") for mode in modes):
        parser.error("modes must be eager and/or graph")
    with tempfile.TemporaryDirectory(prefix="sol-baseline-") as directory:
        old, hashes = baseline(args.baseline_root, Path(directory))
        root = Path(interface.__file__).parent
        report = dict(
            gpu=torch.cuda.get_device_name(),
            capability=list(torch.cuda.get_device_capability()),
            torch=torch.__version__,
            cuda=torch.version.cuda,
            warmup=args.warmup,
            repeats=args.repeats,
            baseline_sha256=hashes,
            candidate_sha256={
                n: hashlib.sha256((root / n).read_bytes()).hexdigest() for n in hashes
            },
            cases=[],
        )
        if args.stress_only:
            with torch.inference_mode():
                for tokens in (257, 4097):
                    for kind in (
                        "random",
                        "mixed_maxima",
                        "zero_q",
                        "zero_v",
                        "tiny_v",
                        "subnormal_v",
                    ):
                        q, k, v, mask, options = tensors((2, tokens, 3, "all", 1.0))
                        if kind == "mixed_maxima":
                            q[:, ::2].mul_(3)
                            q[:, 1::4].zero_()
                            k[:, 64::128].mul_(3)
                        elif kind == "zero_q":
                            q.zero_()
                        elif kind == "zero_v":
                            v.zero_()
                        elif kind == "tiny_v":
                            v.mul_(2.0**-120)
                        elif kind == "subnormal_v":
                            v.mul_(2.0**-130)
                        expected = old(q, k, v, **options, exact_block_mask=mask)
                        actual = sol_attn(q, k, v, **options, exact_block_mask=mask)
                        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
                        report["cases"].append(
                            dict(tokens=tokens, kind=kind, bitwise_equal=True)
                        )
                        save(args.output, report)
            print("STRESS_PASSED", len(report["cases"]), flush=True)
            return
        if args.input:
            data = torch.load(args.input, map_location="cpu", weights_only=True)
            for name in ("q", "k", "v", "exact_block_mask"):
                tensor = data[name]
                if tensor is not None:
                    digest = hashlib.sha256(
                        tensor.contiguous().view(torch.uint8).numpy().tobytes()
                    ).hexdigest()
                    assert digest == data["tensor_sha256"][name], name
            values = [
                data[name][:, :, : args.heads].contiguous().cuda()
                for name in ("q", "k", "v")
            ]
            mask = data["exact_block_mask"]
            values += [mask.cuda() if mask is not None else None, data["options"]]
            report["capture_sha256"] = data["tensor_sha256"]
            cases = [(f"h3-layer-{data['layer']}", values)]
        else:
            cases = [
                (str(case), case)
                for case in (
                    (1, 257, 2, "empty", 1.0),
                    (2, 1024, 3, "all", 1.0),
                    (1, 4096, 2, "all", 1.0),
                    (2, 4097, 3, "ends", 1.0),
                    (1, 16384, 8, "empty", 1.5),
                    (1, 16384, 8, "empty", 1e6),
                    (1, 32768, 8, "ends", 1.0),
                    (1, 32768, 28, "ends", 1.0),
                )
            ]
        for name, case in cases:
            values = case if args.input else tensors(case)
            result = measure(*values, old, modes, args.warmup, args.repeats)
            result["case"] = name
            report["cases"].append(result)
            save(args.output, report)
            print(
                name,
                json.dumps(
                    {
                        mode: values["metrics"]
                        for mode, values in result["modes"].items()
                    }
                ),
                flush=True,
            )


if __name__ == "__main__":
    main()
