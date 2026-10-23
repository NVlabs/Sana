"""Host configuration is kept outside workflows and model dependencies."""
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def config_path():
    return Path(os.environ.get("SOL_ENGINE_CONFIG", ROOT / "config.json")).expanduser()


def profile_names():
    path = config_path()
    if not path.is_file():
        return ["Configure SOL_ENGINE_CONFIG first"]
    return list(json.loads(path.read_text())["profiles"])


def load_profile(name):
    path = config_path().resolve(strict=True)
    profile = dict(json.loads(path.read_text())["profiles"][name])
    if profile.get("precision") not in ("bf16", "fp8", "baseline", "hyperflow", "two_stage"):
        raise ValueError("precision must be bf16, fp8, baseline, hyperflow or two_stage")
    if profile["precision"] == "hyperflow":
        flow = profile.get("hyperflow") or {}
        for key in ("deps", "weights"):
            target = Path(flow.get(key, "")).expanduser()
            if not target.is_absolute():
                target = path.parent / target
            if not target.exists():
                raise FileNotFoundError("HyperFlow %s not found: %s" % (key, target))
            flow[key] = str(target.resolve())
        profile["hyperflow"] = flow
    if profile["precision"] == "two_stage":
        package = Path((profile.get("two_stage") or {}).get("package", "")).expanduser()
        if not (package / "runtime" / "pipeline.py").is_file():
            raise FileNotFoundError("Sol-H3 two-stage package not found: %s" % package)
        if profile["two_stage"].get("offload", "cpu") not in ("cpu", "none"):
            raise ValueError("two_stage.offload must be cpu or none")
    if profile["precision"] == "baseline" and profile.get("decoder", "original") != "original":
        raise ValueError("the original-H3 baseline uses the original H3 VAE")
    for key in ("python", "paths"):
        value = Path(profile[key]).expanduser()
        if not value.is_absolute():
            value = path.parent / value
        # Preserve venv interpreter symlinks: realpath changes environment.
        value = value.absolute()
        if not value.is_file():
            raise ValueError("Missing %s: %s" % (key, value))
        profile[key] = str(value)
    if not os.access(profile["python"], os.X_OK):
        raise ValueError("Runtime Python is not executable")
    gpu = str(profile.get("gpu", "0"))
    if not gpu.isdigit():
        raise ValueError("gpu must be a single numeric device index")
    profile["gpu"] = gpu
    profile["name"] = name
    if profile.get("decoder", "original") not in ("original", "fast_noncausal"):
        raise ValueError("Unknown H3 video decoder")
    if profile.get("decoder") == "fast_noncausal":
        for key in ("code", "weights", "diffusers_source"):
            target = Path(profile["fast_vae"][key]).expanduser()
            if not target.is_absolute():
                target = path.parent / target
            if not target.exists():
                raise FileNotFoundError(target)
            profile["fast_vae"][key] = str(target.resolve())
    return profile


# Denoising updates per precision, for the Studio's "step k/N" when a profile omits "steps".
DEFAULT_STEPS = {"fp8": 4, "bf16": 4, "baseline": 50, "hyperflow": 8, "two_stage": 7}
RESOLUTIONS = ("768p", "480p")
DEVICES = {"rtx5090": "RTX 5090", "spark": "DGX Spark"}


def _profiles():
    path = config_path()
    return json.loads(path.read_text())["profiles"] if path.is_file() else {}


def profile_modes(name):
    """Conditioning modes a profile declares; profiles that declare none are text-only."""
    return list(_profiles().get(name, {}).get("modes", ["text"]))


def profile_resolutions(name):
    """Output resolutions a profile declares (16:9: 768p = 1344x768, 480p = 864x480); default 768p only."""
    return list(_profiles().get(name, {}).get("resolutions", ["768p"]))


def capabilities():
    """Studio-facing description of configured profiles; never validates runtime paths."""
    profiles = _profiles()
    rows = [{"name": name, "label": data.get("label", name),
             "steps": data.get("steps", DEFAULT_STEPS.get(data.get("precision"))),
             "pipeline": "two_stage" if data.get("precision") == "two_stage" else "one_stage",
             "modes": list(data.get("modes", ["text"])), "resolutions": list(data.get("resolutions", ["768p"])),
             "latency_s": data.get("latency_s"), "default": index == 0}
            for index, (name, data) in enumerate(profiles.items())]
    device = next((data.get("device") for data in profiles.values() if data.get("device")), "rtx5090")
    return {"profiles": rows, "device": {"id": device, "label": DEVICES.get(device, device)},
            "gallery": gallery_dir() is not None}


def gallery_dir():
    """Showcase clips for the Studio gallery: config.json's "gallery" directory, if any."""
    path = config_path()
    if not path.is_file():
        return None
    folder = json.loads(path.read_text()).get("gallery")
    if not folder:
        return None
    folder = Path(folder).expanduser()
    folder = folder if folder.is_absolute() else path.parent / folder
    return folder.resolve() if folder.is_dir() else None
