#!/usr/bin/env python3
"""Optional eight-GPU functional/parity checks; never run by the CPU test suite."""
import argparse
import hashlib
import json
from pathlib import Path


def digest(value):
    import torch

    if value is None:
        return None
    if torch.is_tensor(value):
        data = value.detach().contiguous().view(torch.uint8).cpu().numpy()
        return dict(shape=list(value.shape), dtype=str(value.dtype),
                    sha256=hashlib.sha256(data).hexdigest())
    if isinstance(value, (list, tuple)):
        return [digest(item) for item in value]
    if isinstance(value, dict):
        return {key: digest(item) for key, item in value.items()}
    raise TypeError(type(value))


def verify_parity(engine, reference):
    import torch
    import torch.distributed as dist
    from sol_hyperflow.config import Request
    from sol_hyperflow.runtime import run

    rows = []
    for task in ("t2v", "i2v", "ref2va"):
        request = Request(
            "A continuous view of the subject in natural light.", task=task, seed=42,
            image=reference if task == "i2v" else None,
            references=(reference,) if task == "ref2va" else (),
        ).payload()
        pipe = engine.pipes[request["task"]]
        model = pipe.transformer_ref if task == "ref2va" else pipe.transformer
        sparse = engine.attention["ref2v" if task == "ref2va" else "t2v"]
        fusion = model._consumer_lora_fusion
        previous = fusion.enabled
        traces = []
        handle = model.register_forward_hook(
            lambda _model, _args, output: traces.append(digest(output)))
        try:
            fusion.enabled = False
            native, _, _ = run(pipe, request, sparse, expect_fused=False)
            native_trace = list(traces)
            traces.clear()
            native_media = {key: digest(native[key]) for key in ("videos", "audio")}
            del native
            fusion.enabled = True
            fused, _, _ = run(pipe, request, sparse, expect_fused=True)
            fused_trace = list(traces)
            fused_media = {key: digest(fused[key]) for key in ("videos", "audio")}
            del fused
        finally:
            fusion.enabled = previous
            handle.remove()
        flags = torch.tensor([
            int(len(native_trace) == len(fused_trace) == 8 and native_trace == fused_trace),
            int(native_media == fused_media),
        ], device=engine.device)
        dist.all_reduce(flags, op=dist.ReduceOp.MIN)
        if not bool(flags.all().item()):
            raise RuntimeError(f"{task}: native/fused parity failed: {flags.cpu().tolist()}")
        rows.append(dict(task=task, all_ranks_dit_bitwise_equal=True,
                         all_ranks_decoded_media_bitwise_equal=True))
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--adapter", required=True, type=Path)
    parser.add_argument("--reference-image", required=True, type=Path)
    parser.add_argument("--durations", nargs="+", type=int, choices=(5, 10, 15), default=[5])
    parser.add_argument("--compare-native", action="store_true")
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args(argv)
    from infer import load_image
    from sol_hyperflow import HyperFlowInference

    image = load_image(args.reference_image)
    report = dict(complete=False, world_size=8, steps=8, cases=[],
                  timing_scope="Functional checks may include compilation; not a warmed latency benchmark.")
    with HyperFlowInference(args.model, args.adapter) as engine:
        for task in ("t2v", "i2v", "ref2va"):
            for duration in args.durations:
                media = engine.generate(
                    "A continuous view of the subject in natural light.", task=task,
                    duration=duration, seed=42, image=image if task == "i2v" else None,
                    references=[image] if task == "ref2va" else None,
                )
                if engine.rank == 0:
                    report["cases"].append(dict(engine.last_request,
                                                video=digest(media.video), audio=digest(media.audio)))
                    del media
        if args.compare_native:
            report["native_fused_parity"] = verify_parity(engine, image)
        if engine.rank == 0:
            report["complete"] = True
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(json.dumps(report, indent=2) + "\n")
            print("HYPERFLOW_VALIDATION_PASSED", args.report, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
