"""Check local deployment assets without loading models."""
import argparse
import hashlib
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from runtime.config import load_config, ROOT

p = argparse.ArgumentParser()
p.add_argument("--config", default=str(ROOT / "configs/runtime.json"))
p.add_argument("--assets", default=str(ROOT / "configs/assets.local.json"))
p.add_argument("--sha256", action="store_true")
args = p.parse_args()
config, assets = load_config(args.config, args.assets)
missing = []
for name, value in assets.items():
    if name == "prompt":
        continue
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = Path(args.assets).resolve().parent / path
    if not path.exists():
        missing.append(name)
print(json.dumps({"missing_assets": missing, "layers": config["inference"]["models"]["backbone"]["args"]["num_layers"],
                  "nfe": config["inference"]["diffusion"]["sampling_timesteps"]["args"]["num_sampling_steps"]}, indent=2))
if missing:
    raise SystemExit(1)
if args.sha256:
    path = Path(config["inference"]["models"]["backbone"]["weight"]["path"])
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while data := handle.read(16 * 1024 * 1024):
            digest.update(data)
    print("backbone_sha256", digest.hexdigest())
