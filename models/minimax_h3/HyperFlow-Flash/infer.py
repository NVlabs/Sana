#!/usr/bin/env python3
"""Generate one video + audio clip with HyperFlow H3 Flash (6 steps, 1344x768, 24 fps) on Sol-Engine.

    python infer.py --prompt "A red fox trots through fresh snow at dawn" --duration 5 --out fox.mp4 \
        --base-model /path/to/MiniMax-H3
    torchrun --standalone --nproc_per_node 4 infer.py ...            # Ulysses sequence parallelism over 4 GPUs
    (8 GPUs = two 4-GPU nodes: see the two-node torchrun example in README.md)
    python infer.py --task ref2va --request examples/ref2va_request.json --duration 5 --out ref.mp4 --base-model ...

--weights is the checkpoint: a Hub repo id (default, downloaded at the pinned revision) or a local folder (the repository
root or one partition folder). --base-model is the MiniMax-H3 snapshot (text encoder, VAEs, tokenizer, processors).
Presets: sol-engine (Sol-Engine recipe, default) | official (the checkpoint as released). --attention-backend dense keeps the rest of the preset and turns Sol-Attn
off. ref2va reads the prompt and the ordered references from a request json (sol_hyperflow_flash/references.py).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sol_hyperflow_flash import checkpoint as CK  # noqa: E402
from sol_hyperflow_flash import config as C  # noqa: E402
from sol_hyperflow_flash.references import resolve_seed  # noqa: E402
from sol_hyperflow_flash.runtime import DURATION_FRAMES, HyperFlowFlashRuntime, per_rank_caches  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", default=CK.HF_REPO, help=f"Hub repo id or local folder (default {CK.HF_REPO})")
    ap.add_argument("--revision", default=CK.HF_REVISION, help="Hub revision of --weights (default: the validated one)")
    ap.add_argument("--base-model", required=True, help="MiniMax-H3 snapshot directory (or Hub repo id)")
    ap.add_argument("--sol-h3-dir", default=os.environ.get("SOL_H3_DIR"), help="default: ../Sol-H3 of this checkout")
    ap.add_argument("--task", default="t2v", choices=("t2v", "ref2va"))
    ap.add_argument("--request", help="ref2va request json (prompt + references)")
    ap.add_argument("--media-dir", default="", help="ref2va: base of relative media paths (default: the request's directory)")
    ap.add_argument("--preset", default="sol-engine", choices=("sol-engine", "official"))
    ap.add_argument("--attention-backend", default="sol", choices=("sol", "dense"))
    text = ap.add_mutually_exclusive_group()
    text.add_argument("--prompt")
    text.add_argument("--prompt-file", help="UTF-8 text file, or a json with a 'prompt' field")
    ap.add_argument("--duration", type=int, choices=sorted(DURATION_FRAMES), default=5)
    ap.add_argument("--seed", type=int, default=None, help="default: the request's seed (ref2va), else 0")
    ap.add_argument("--warmup", action="store_true", help="run one untimed request first (kernel compilation)")
    ap.add_argument("--te-offload", action="store_true", help="keep the text encoder on the host between requests")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    task = "t2va" if a.task == "t2v" else "ref2va"
    refs = None
    if task == "ref2va":
        if not a.request or a.prompt or a.prompt_file:
            ap.error("--task ref2va takes --request (and no --prompt / --prompt-file)")
        from sol_hyperflow_flash.references import build_references, parse_request

        spec = parse_request(a.request, a.media_dir or None)
        a.prompt, refs = spec["prompt"], build_references(spec)
        a.seed = resolve_seed(a.seed, spec)
    elif a.prompt is None and a.prompt_file is None:
        ap.error("--prompt or --prompt-file is required")
    elif a.prompt is None:
        raw = Path(a.prompt_file).read_text(encoding="utf-8")
        a.prompt = json.loads(raw)["prompt"] if a.prompt_file.endswith(".json") else raw.strip()
    a.seed = resolve_seed(a.seed)
    per_rank_caches()
    preset = C.get(a.preset, task=task, attention_backend=a.attention_backend)
    need = ("sparse", "fusions") if (preset.sparse or preset.fusions) else ()
    rt = HyperFlowFlashRuntime(a.weights, a.base_model, a.sol_h3_dir, task=task, revision=a.revision,
                               te_offload=a.te_offload, need=need)
    rt.apply(preset)
    frames = DURATION_FRAMES[a.duration]
    if a.warmup:
        rt.generate(a.prompt, frames, a.seed, references=refs)
    rec, (vid, aud, sr, _) = rt.generate(a.prompt, frames, a.seed, keep=True, references=refs)
    if rt.rank == 0:
        from sol_hyperflow_flash.media import save_mp4

        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        save_mp4(vid, aud, sr, a.out)
        print(f"{a.out}: {rec['frames']} frames, e2e {rec['e2e_ms'] / 1e3:.2f} s (DiT {rec['dit_ms'] / 1e3:.2f} s), "
              f"peak {rec['peak_alloc_gb']:.1f} GiB, preset {preset.name}, {rt.world} GPU", flush=True)
    if rt.world > 1:
        rt.torch.distributed.barrier()
        rt.torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
