"""Reuse a prepared 5090 environment and write host-local ComfyUI profiles."""
import argparse
import json
from pathlib import Path

REQUIRED = ("stage1_python", "qwen_python", "h3_model", "fastvideo_root",
            "fa4_root", "ltx_root", "qwen_checkpoint", "comfy_root")


def validate_model(root):
    """Check actual files: glob() alone also returns broken symlinks."""
    root = Path(root)
    for component in ("transformer", "vae", "audio_vae"):
        folder = root / component
        index = folder / "diffusion_pytorch_model.safetensors.index.json"
        files = (set(json.loads(index.read_text())["weight_map"].values())
                 if index.is_file() else {"diffusion_pytorch_model.safetensors"})
        for name in sorted(files):
            path = folder / name
            if not path.is_file() or path.stat().st_size == 0:
                raise FileNotFoundError("Missing H3 checkpoint file: " + str(path))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reuse-paths", type=Path, required=True)
    parser.add_argument("--dense-lora", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--hyperflow-weights", type=Path,
                        help="minimax_h3_hyperflow_8step_v1.0.safetensors (videorebirth/hyperflow)")
    parser.add_argument("--hyperflow-deps", type=Path,
                        help="overlay with diffusers>=0.40, peft, accelerate and hyperflow_h3")
    parser.add_argument("--hyperflow-fp8-dir", type=Path,
                        help="merged FP8 HyperFlow checkpoint from scripts/hyperflow/convert_fp8.py")
    parser.add_argument("--hyperflow-adaln-table", type=Path,
                        help="exact AdaLN table from scripts/hyperflow/build_adaln_table.py")
    parser.add_argument("--hyperflow-fast-vae-weights", type=Path,
                        help="fast_h3_vae_noncausal_24l.safetensors for the fast HyperFlow profile's video decode")
    parser.add_argument("--hyperflow-ref-fp8-dir", type=Path,
                        help="transformer_ref (reference images) FP8 checkpoint + adaln_table.pt from scripts/hyperflow/convert_ref.py")
    parser.add_argument("--one-stage-resolutions", default="768p,480p",
                        help="resolution ratios the one-stage profile offers (comma-separated)")
    parser.add_argument("--two-stage-package", type=Path,
                        help="NVlabs/Sana sol-engine models/minimax_h3/Sol-H3-RTX5090 (or Sol-H3-Spark) directory")
    parser.add_argument("--two-stage-paths", type=Path, help="that package's prepared paths JSON (prepare.py output)")
    parser.add_argument("--device", choices=("rtx5090", "spark"), default="rtx5090",
                        help="device the profiles run on; two stage runs without CPU offload on DGX Spark")
    parser.add_argument("--gallery", type=Path, help="showcase clips for the Studio gallery (scripts/build_gallery.py)")
    parser.add_argument("--fast-vae-code", type=Path)
    parser.add_argument("--fast-vae-weights", type=Path)
    parser.add_argument("--fast-vae-diffusers-source", type=Path)
    args = parser.parse_args()
    source = args.reuse_paths.resolve(strict=True)
    old = json.loads(source.read_text())
    paths = {}
    for key in REQUIRED + ("stage1_dependencies", "fa4_dependencies", "qwen_dependencies"):
        if key not in old:
            if key in REQUIRED:
                raise ValueError("Missing %s in source manifest" % key)
            continue
        value = Path(old[key]).expanduser()
        if not value.is_absolute():
            value = source.parent / value
        if not value.exists():
            raise FileNotFoundError(value)
        paths[key] = str(value.absolute())
    paths["dense_lora"] = str(args.dense_lora.resolve(strict=True))
    validate_model(paths["h3_model"])
    destination = args.output_dir.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    paths_file = destination / "paths.json"
    config_file = destination / "config.json"
    if paths_file.exists() or config_file.exists():
        raise FileExistsError("Choose a fresh output directory")
    profiles = {}
    for precision in ("fp8", "bf16"):
        profiles["H3 %s — single RTX 5090" % precision.upper()] = {
            "precision": precision, "python": paths["stage1_python"],
            "paths": str(paths_file), "gpu": str(args.gpu), "timeout_s": 3600}
    # Same Qwen, VAE, offload and GPU; original weights and official 50-point schedule.
    profiles["Original H3 50-step baseline — RTX 5090"] = {
        **profiles["H3 BF16 — single RTX 5090"], "precision": "baseline", "timeout_s": 14400}
    if args.hyperflow_weights or args.hyperflow_deps:
        if not (args.hyperflow_weights and args.hyperflow_deps):
            raise ValueError("Provide --hyperflow-weights and --hyperflow-deps together")
        # HyperFlow-8 is the default Studio profile when available: listed first.
        profiles = {"Sol-H3 HyperFlow · 8 steps — RTX 5090": {
            "precision": "hyperflow", "python": paths["stage1_python"], "paths": str(paths_file),
            "gpu": str(args.gpu), "timeout_s": 3600, "label": "HyperFlow · 8 steps", "steps": 8,
            "modes": ["text"],
            "hyperflow": {"weights": str(args.hyperflow_weights.resolve(strict=True)),
                          "deps": str(args.hyperflow_deps.resolve(strict=True)), "workflows": ["t2va"]}},
            **profiles}
    fast = (args.hyperflow_fp8_dir, args.hyperflow_adaln_table, args.hyperflow_fast_vae_weights, args.fast_vae_code)
    if any(fast[:3]):
        if not all(fast) or not (args.hyperflow_weights and args.hyperflow_deps):
            raise ValueError("The fast HyperFlow profile needs --hyperflow-weights/--hyperflow-deps, "
                             "--hyperflow-fp8-dir, --hyperflow-adaln-table, --hyperflow-fast-vae-weights and --fast-vae-code")
        base = profiles["Sol-H3 HyperFlow · 8 steps — RTX 5090"]
        hf = dict(base["hyperflow"])
        hf.update(
            fp8_dir=str(args.hyperflow_fp8_dir.resolve(strict=True)),
            adaln_table=str(args.hyperflow_adaln_table.resolve(strict=True)),
            sol_attn=True, dense_attn="sage2pp", compile=True, pin_unmatched_group=True, qwen_resident=True,
            fast_vae={"code": str(args.fast_vae_code.resolve(strict=True)),
                      "vae_config": str(Path(paths["h3_model"]) / "vae/config.json"),
                      "weights": str(args.hyperflow_fast_vae_weights.resolve(strict=True))})
        hf["workflows"] = ["t2va", "fl2va"]
        modes = ["text", "first_frame"]
        if args.hyperflow_ref_fp8_dir:
            ref = args.hyperflow_ref_fp8_dir.resolve(strict=True)
            # swap_ref_dit: one FP8 DiT in host RAM at a time (both together exceed a 90 GB host).
            hf.update(ref_fp8_dir=str(ref), ref_adaln_table=str((ref / "adaln_table.pt").resolve(strict=True)),
                      swap_ref_dit=True)
            hf["workflows"].append("ref2va")
            modes.append("reference")
        resolutions = [a for a in args.one_stage_resolutions.split(",") if a]
        # The fast one-stage profile is the default when available: listed first.
        profiles = {"Sol-H3 one stage — RTX 5090": dict(base, label="One stage", modes=modes, resolutions=resolutions,
                                                        device=args.device, hyperflow=hf), **profiles}
    if args.two_stage_package or args.two_stage_paths:
        if not (args.two_stage_package and args.two_stage_paths):
            raise ValueError("Provide --two-stage-package and --two-stage-paths together")
        package = args.two_stage_package.resolve(strict=True)
        if not (package / "runtime" / "pipeline.py").is_file():
            raise FileNotFoundError(package / "runtime" / "pipeline.py")
        two = json.loads(args.two_stage_paths.read_text())
        name = "Sol-H3 two stage — " + ("DGX Spark" if args.device == "spark" else "RTX 5090")
        profiles[name] = {"precision": "two_stage", "python": two["stage1_python"],
                          "paths": str(args.two_stage_paths.resolve(strict=True)), "gpu": str(args.gpu),
                          "timeout_s": 3600, "label": "Two stage", "steps": 7, "modes": ["text"], "resolutions": ["768p"],
                          "device": args.device,
                          # A resident Qwen session: the package's per-request load rereads 15.7 GB each time.
                          "two_stage": {"package": str(package), "offload": "none" if args.device == "spark" else "cpu",
                                        "resident_qwen": args.device != "spark"}}
    optional = (args.fast_vae_code, args.fast_vae_weights, args.fast_vae_diffusers_source)
    # --fast-vae-code alone serves the fast HyperFlow profile; the FastH3 noncausal profile needs all three.
    if args.fast_vae_weights or args.fast_vae_diffusers_source:
        if not all(optional):
            raise ValueError("Provide all three --fast-vae-* paths together")
        code, weights, diffusers = (str(p.resolve(strict=True)) for p in optional)
        profiles["H3 FP8 + noncausal VAE — RTX 5090"] = {
            **profiles["H3 FP8 — single RTX 5090"], "decoder": "fast_noncausal",
            "fast_vae": {"code": code, "weights": weights, "diffusers_source": diffusers}}
    paths_file.write_text(json.dumps(paths, indent=2) + "\n")
    settings = {"profiles": profiles}
    if args.gallery:
        settings["gallery"] = str(args.gallery.resolve(strict=True))
    config_file.write_text(json.dumps(settings, indent=2, ensure_ascii=False) + "\n")
    print("SOL_ENGINE_CONFIG=" + str(config_file))


if __name__ == "__main__":
    main()
