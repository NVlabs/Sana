#!/usr/bin/env python3
"""Stage-validate + run the newly-ported SANA-Video model in the sglang
multimodal_gen runtime. Each stage prints a marker so the slurm log pinpoints
where any wiring bug is, before/after the heavy model load."""

from __future__ import annotations

import argparse
import sys
import time
import traceback


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--model", default="Efficient-Large-Model/SANA-Video_2B_480p_diffusers"
    )
    ap.add_argument(
        "--prompt", default=None, help="prompt text; if omitted, read --prompt-file"
    )
    ap.add_argument(
        "--prompt-file",
        default=None,
        help="prompt file (default: prompts/sana/default.txt in the repo)",
    )
    ap.add_argument("--frames", type=int, default=81)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--width", type=int, default=832)
    ap.add_argument("--guidance-scale", type=float, default=6.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--output",
        default="sglang_sana_480p",
        help="output file BASENAME (no slashes — the runtime sanitizes "
        "slashes in output_file_name and writes outputs/<name>.mp4)",
    )
    ap.add_argument("--label", default="")
    ap.add_argument(
        "--compile",
        action="store_true",
        help="enable torch.compile (kernel-fusion toggle)",
    )
    ap.add_argument(
        "--compile-mode",
        default="default",
        help="torch.compile mode for the SANA DiT (default: 'default'). "
        "The generic path defaults to 'max-autotune-no-cudagraphs', "
        "whose in-process GEMM autotune deadlocks at cuda.synchronize() "
        "on GB200/cu130 for the full DiT; 'default' avoids it. An "
        "explicit SGLANG_TORCH_COMPILE_MODE env still wins.",
    )
    ap.add_argument(
        "--max-autotune",
        action="store_true",
        help="fast compile path (~2.56x once warm; overrides --compile-mode): "
        "max-autotune + subprocess autotune (per-choice timeout skips the "
        "grouped-conv Triton templates that deadlock at cuda.synchronize on "
        "GB200/cu130) + persistent inductor cache. First (cold) run pays "
        "autotune; warm runs reuse it. Default (off) = safe 'default' mode (~2.10x).",
    )
    ap.add_argument(
        "--linattn-bf16",
        action="store_true",
        help="bf16 linear-attention KV aggregation (fusion; part of fullopt stack)",
    )
    ap.add_argument(
        "--qkv-merge",
        action="store_true",
        help="lossless merged QKV projection for self-attention (fusion; part of fullopt stack)",
    )
    ap.add_argument(
        "--no-warmup",
        action="store_true",
        help="disable warmup (warmup is ON by default for clean steady-state timing)",
    )
    ap.add_argument(
        "--easycache",
        type=float,
        default=0.0,
        help="EasyCache reuse threshold (0=off; higher=more skips); "
        "calibration-free adaptive step skipping",
    )
    ap.add_argument(
        "--ec-warmup",
        type=int,
        default=3,
        help="EasyCache warmup steps (always computed)",
    )
    ap.add_argument(
        "--ec-subsample",
        type=int,
        default=8,
        help="EasyCache spatial subsample stride for the rel-change estimate",
    )
    args = ap.parse_args()
    if args.prompt is None:
        import os as _osp

        prompt_file = args.prompt_file or _osp.path.join(
            _osp.path.dirname(
                _osp.path.dirname(_osp.path.dirname(_osp.path.abspath(__file__)))
            ),
            "prompts",
            "sana",
            "default.txt",
        )
        with open(prompt_file) as _pf:
            args.prompt = _pf.read().strip()
        print(f"prompt from: {prompt_file}", flush=True)
    if args.label:
        args.output = f"{args.output}_{args.label}"
    # Set technique env BEFORE building the model (DiT reads them in __init__ /
    # post_load_weights); inherited by the local scheduler subprocess.
    import os as _os

    if args.compile:
        if args.max_autotune:
            # Fast path (~2.56x once warm). The SANA grouped-conv (GROUPS=13440,
            # 3x3) Triton autotune templates deadlock at cuda.synchronize during
            # IN-PROCESS autotune on GB200/cu130; run autotune in a subprocess
            # (per-choice timeout skips the hanging conv) and PERSIST the inductor
            # cache so only the first (cold) run pays autotune — warm runs reuse it.
            _os.environ.setdefault(
                "SGLANG_TORCH_COMPILE_MODE", "max-autotune-no-cudagraphs"
            )
            _os.environ.setdefault("TORCHINDUCTOR_AUTOTUNE_IN_SUBPROC", "1")
            _os.environ.setdefault(
                "TORCHINDUCTOR_CACHE_DIR",
                _os.path.expanduser("~/.cache/sgl_torchinductor"),
            )
        else:
            # Safe default (~2.10x): plain inductor 'default' mode avoids the
            # max-autotune grouped-conv deadlock; runs cold/in-process anywhere.
            _os.environ.setdefault("SGLANG_TORCH_COMPILE_MODE", args.compile_mode)
    if args.linattn_bf16:
        _os.environ["SGLANG_SANA_LINATTN_BF16"] = "1"
    if args.qkv_merge:
        _os.environ["SGLANG_SANA_QKV_MERGE"] = "1"
    if args.easycache:
        _os.environ["SGLANG_SANA_EASYCACHE_THRESH"] = str(args.easycache)
        _os.environ["SGLANG_SANA_EASYCACHE_WARMUP"] = str(args.ec_warmup)
        _os.environ["SGLANG_SANA_EASYCACHE_SUBSAMPLE"] = str(args.ec_subsample)

    print(f"==== STAGE 1: import new SANA-Video modules ====", flush=True)
    try:
        from sglang.multimodal_gen.configs.pipeline_configs.sana_video import (  # noqa: F401
            SanaVideoPipelineConfig,
        )
        from sglang.multimodal_gen.configs.sample.sana_video import (  # noqa: F401
            SanaVideoSamplingParams,
        )
        from sglang.multimodal_gen.runtime.models.dits.sana_video import (  # noqa: F401
            SanaVideoTransformer3DModel,
        )
        from sglang.multimodal_gen.runtime.pipelines.sana_video import (  # noqa: F401
            SanaVideoPipeline,
        )

        print("IMPORT_OK", flush=True)
    except Exception:
        traceback.print_exc()
        print("IMPORT_FAIL", flush=True)
        sys.exit(1)

    print(f"==== STAGE 2: resolve local snapshot + pipeline registry ====", flush=True)
    import glob
    import os

    def _resolve_local(model):
        if os.path.isdir(model):
            return model
        cache = os.environ.get(
            "HF_HUB_CACHE", os.path.expanduser("~/.cache/huggingface/hub")
        )
        safe = "models--" + model.replace("/", "--")
        snaps = sorted(glob.glob(os.path.join(cache, safe, "snapshots", "*")))
        return snaps[-1] if snaps else model

    model_path = _resolve_local(args.model)
    print("model_path:", model_path, flush=True)
    try:
        from sglang.multimodal_gen.registry import (
            _PIPELINE_REGISTRY,
            _discover_and_register_pipelines,
        )

        _discover_and_register_pipelines()
        print(
            "SanaVideoPipeline registered:",
            "SanaVideoPipeline" in _PIPELINE_REGISTRY,
            flush=True,
        )
        print(
            "sana pipelines:",
            [k for k in _PIPELINE_REGISTRY if "ana" in k.lower()],
            flush=True,
        )
    except Exception:
        traceback.print_exc()
        print("REGISTRY_WARN", flush=True)

    print(f"==== STAGE 3: build DiffGenerator ====", flush=True)
    try:
        from sglang.multimodal_gen.runtime.entrypoints.diffusion_generator import (
            DiffGenerator,
        )

        gen = DiffGenerator.from_pretrained(
            model_path=model_path,
            pipeline_class_name="SanaVideoPipeline",
            num_gpus=1,
            dit_cpu_offload=False,
            enable_torch_compile=args.compile,
            warmup=not args.no_warmup,
        )
        print(
            f"BUILD_OK (compile={args.compile}, warmup={not args.no_warmup})",
            flush=True,
        )
    except Exception:
        traceback.print_exc()
        print("BUILD_FAIL", flush=True)
        sys.exit(3)

    print(f"==== STAGE 4: generate ====", flush=True)
    try:
        t = time.time()
        res = gen.generate(
            sampling_params_kwargs=dict(
                prompt=args.prompt,
                num_frames=args.frames,
                num_inference_steps=args.steps,
                height=args.height,
                width=args.width,
                guidance_scale=args.guidance_scale,
                seed=args.seed,
                output_file_name=args.output,
                save_output=True,
                fps=16,
            )
        )
        # Guard against silent failure: a swallowed worker-side error (e.g. a
        # failed JIT build) can return None / write no file while still exiting 0.
        # Trust GenerationResult.output_file_path (the runtime sanitizes/prefixes
        # the requested name, so it rarely equals args.output) rather than args.output.
        items = (
            res
            if isinstance(res, (list, tuple))
            else ([res] if res is not None else [])
        )
        produced_paths = [
            p for r in items if (p := getattr(r, "output_file_path", None))
        ]
        if not items:
            print(f"GENERATE_FAIL: no result returned (res={res!r})", flush=True)
            sys.exit(4)
        if produced_paths and not any(os.path.exists(p) for p in produced_paths):
            print(
                f"GENERATE_FAIL: result paths missing on disk: {produced_paths}",
                flush=True,
            )
            sys.exit(4)
        print(
            f"GENERATE_OK in {time.time() - t:.1f}s: paths={produced_paths or res}",
            flush=True,
        )
        print("RUN_COMPLETE_MARKER", flush=True)
    except Exception:
        traceback.print_exc()
        print("GENERATE_FAIL", flush=True)
        sys.exit(4)


if __name__ == "__main__":
    main()
