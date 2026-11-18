#!/usr/bin/env python3
"""Time official and sol-engine in one process (same request, same node), one JSON line per timed request plus a summary per preset.

    python benchmark.py --task t2v --prompt-file prompt.txt --duration 5 --base-model checkpoints/MiniMax-H3 --out runs/t2v.jsonl
    torchrun --standalone --nproc_per_node 4 benchmark.py --task ref2va --request examples/ref2va_request.json ...

Protocol: presets alternate in the order given; each runs --warmup untimed requests, then --reps timed requests; the summary
holds the median end-to-end and DiT seconds. --dump DIR additionally saves, on rank 0, the decoder inputs (video and audio
latents) and the uint8 video of one extra request per preset to DIR/<task>_<preset>_<frames>f_w<world>.pt (numerics gate).
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sol_hyperflow_flash import checkpoint as CK  # noqa: E402
from sol_hyperflow_flash import config as C  # noqa: E402
from sol_hyperflow_flash.references import resolve_seed  # noqa: E402
from sol_hyperflow_flash.runtime import DURATION_FRAMES, HyperFlowFlashRuntime, per_rank_caches  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", default=CK.HF_REPO)
    ap.add_argument("--revision", default=CK.HF_REVISION)
    ap.add_argument("--base-model", required=True)
    ap.add_argument("--sol-h3-dir", default=os.environ.get("SOL_H3_DIR"))
    ap.add_argument("--task", default="t2v", choices=("t2v", "ref2va"))
    ap.add_argument("--request")
    ap.add_argument("--media-dir", default="")
    ap.add_argument("--prompt")
    ap.add_argument("--prompt-file")
    ap.add_argument("--presets", default="official,sol-engine")
    ap.add_argument("--attention-backend", default="sol", choices=("sol", "dense"))
    ap.add_argument("--duration", type=int, choices=sorted(DURATION_FRAMES), default=5)
    ap.add_argument("--seed", type=int, default=None, help="default: the request's seed (ref2va), else 0")
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--dump", default="", help="directory for the numerics-gate dump (rank 0)")
    ap.add_argument("--out", required=True, help="JSON lines file (rank 0)")
    a = ap.parse_args()
    task = "t2va" if a.task == "t2v" else "ref2va"
    refs = None
    if task == "ref2va":
        from sol_hyperflow_flash.references import build_references, parse_request

        spec = parse_request(a.request, a.media_dir or None)
        prompt, refs = spec["prompt"], build_references(spec)
        a.seed = resolve_seed(a.seed, spec)
    elif a.prompt is not None:
        prompt = a.prompt
    else:
        raw = Path(a.prompt_file).read_text(encoding="utf-8")
        prompt = json.loads(raw)["prompt"] if a.prompt_file.endswith(".json") else raw.strip()
    a.seed = resolve_seed(a.seed)
    per_rank_caches()
    presets = [C.get(p, task=task, attention_backend=a.attention_backend) for p in a.presets.split(",")]
    need = ("sparse", "fusions") if any(p.sparse or p.fusions for p in presets) else ()
    rt = HyperFlowFlashRuntime(a.weights, a.base_model, a.sol_h3_dir, task=task, revision=a.revision, need=need)
    frames = DURATION_FRAMES[a.duration]
    out = None
    if rt.rank == 0:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        out = open(a.out, "a", encoding="utf-8")

    def emit(rec):
        if out is not None:
            out.write(json.dumps(rec) + "\n")
            out.flush()
            print(json.dumps(rec)[:400], flush=True)

    emit(dict(event="load", task=task, world=rt.world, load_s=rt.load_s, checkpoint=rt.ckpt.info, int8_path=rt.int8_path(),
              time=time.strftime("%Y-%m-%dT%H:%M:%S")))
    for p in presets:
        rt.apply(p)
        for _ in range(a.warmup):
            rt.generate(prompt, frames, a.seed, references=refs)
        recs = []
        for i in range(a.reps):
            rec, _ = rt.generate(prompt, frames, a.seed, references=refs)
            recs.append(rec)
            emit(dict(event="request", rep=i, **rec))
        if recs:
            emit(dict(event="summary", task=task, preset=p.name, frames=frames, world=rt.world, reps=len(recs),
                      e2e_s=statistics.median(r["e2e_ms"] for r in recs) / 1e3,
                      dit_s=statistics.median(r["dit_ms"] for r in recs) / 1e3,
                      peak_alloc_gb=max(r["peak_alloc_gb"] for r in recs), counts=recs[-1]["counts"],
                      finite=all(r["finite"] for r in recs), preset_config=p.as_dict()))
        if a.dump:
            import torch

            rec, (vid, aud, sr, lat) = rt.generate(prompt, frames, a.seed, keep=True, capture_latents=True, references=refs)
            if rt.rank == 0:
                Path(a.dump).mkdir(parents=True, exist_ok=True)
                path = Path(a.dump) / f"{task}_{p.name}_{frames}f_w{rt.world}.pt"
                torch.save(dict(latents={k: v.cpu() for k, v in rt.captured_raw.items()},
                                video_u8=(vid.float().clamp(0, 1) * 255).round().to(torch.uint8).cpu(),
                                audio=None if aud is None else aud.float().cpu(), record=rec), path)
                emit(dict(event="dump", preset=p.name, path=str(path)))
    if out is not None:
        out.close()
    if rt.world > 1:
        rt.torch.distributed.barrier()
        rt.torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
