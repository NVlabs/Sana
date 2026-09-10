"""Load the pinned public FastVideo entry and observe its native FA4 calls."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

EXAMPLE_SHAS = {
    "basic_fasth3.py": "db08012699362fbb65afc81b459e5dec500a7871a3ff06e14cbc0fcd64f26719",
}
def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n")
    os.replace(temporary, path)


def load_official_entry(root: Path):
    examples = root / "examples/inference/basic"
    for name, expected in EXAMPLE_SHAS.items():
        if sha256(examples / name) != expected:
            raise RuntimeError(f"official example changed: {examples / name}")
    sys.path.insert(0, str(root))
    sys.path.insert(0, str(examples))
    name = "basic_fasth3"
    spec = importlib.util.spec_from_file_location(name, examples / f"{name}.py")
    if spec is None or spec.loader is None:
        raise ImportError(name)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def install_fa4_audit(worker):
    """RPC: observe real CUDA FA4 calls below the compile-safe custom-op seam."""
    import os
    import traceback
    import diffusers
    import fastvideo.attention.utils.flash_attn_cute as cute
    from fastvideo.attention.utils.flash_attn_default import fa_version

    if fa_version != "4" or os.environ.get("FASTVIDEO_NVFP4_FA4") != "0":
        raise RuntimeError("worker did not select unquantized FA4")
    pipeline_config = worker.fastvideo_args.pipeline_config
    precision = {"dit": pipeline_config.dit_precision,
                 "text_encoder": list(pipeline_config.text_encoder_precisions),
                 "vae": pipeline_config.vae_precision,
                 "vae_decode_override": pipeline_config.vae_decode_precision}
    if precision != {"dit": "bf16", "text_encoder": ["bf16"], "vae": "fp32",
                     "vae_decode_override": None}:
        raise RuntimeError(f"official native H3 component precision drift: {precision!r}")
    state = {"rank": int(os.environ.get("RANK", "0")), "fa_version": fa_version,
             "calls": 0, "max_query_tokens": 0, "q_dtypes": [], "fallback_calls": 0,
             "resolved_precision": precision,
             "runtime_diffusers": {"version": diffusers.__version__, "file": diffusers.__file__},
             "first_fp16_call": None}
    original = cute._flash_attn_fwd

    def observed(*args, **kwargs):
        result = original(*args, **kwargs)
        q = args[0] if args else kwargs["q"]
        if q.device.type != "cuda":
            raise RuntimeError("FA4 actual-call receipt requires CUDA tensors")
        state["calls"] += 1
        state["max_query_tokens"] = max(state["max_query_tokens"], int(q.shape[-3]))
        dtype = str(q.dtype)
        if dtype not in state["q_dtypes"]:
            state["q_dtypes"].append(dtype)
        if dtype == "torch.float16" and state["first_fp16_call"] is None:
            # The generic FA4 seam also sees native decoder/compile calls.
            # Preserve the origin instead of inferring component precision.
            state["first_fp16_call"] = {"shape": list(q.shape),
                                        "stack": traceback.format_stack(limit=24)}
        return result

    def forbidden_fallback(*args, **kwargs):
        state["fallback_calls"] += 1
        raise RuntimeError("Stage1 FA4 audit forbids an FA2 fallback")

    cute._flash_attn_fwd = observed
    cute._fa2_or_raise = forbidden_fallback
    worker._sol_h3_fa4_audit = state
    return dict(state)


def read_fa4_audit(worker):
    """RPC: return call counters and the actual post-request scheduler grid."""
    import copy
    state = copy.deepcopy(worker._sol_h3_fa4_audit)
    stage = worker.pipeline._stage_name_mapping.get("denoising_stage")
    if stage is not None:
        for name in ("scheduler", "audio_scheduler"):
            scheduler = getattr(stage, name)
            for field in ("timesteps", "sigmas"):
                value = getattr(scheduler, field, None)
                if value is not None:
                    state[f"{name}_{field}"] = value.detach().cpu().tolist()
    return state
