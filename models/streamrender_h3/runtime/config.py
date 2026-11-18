"""Resolve deployment assets without embedding cluster paths in source."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_config(config_path, assets_path):
    config = json.loads(Path(config_path).read_text())
    document = json.loads(Path(assets_path).read_text())
    assets = document["assets"]
    base = Path(assets_path).resolve().parent

    def resolve(value):
        if isinstance(value, dict):
            return {key: resolve(item) for key, item in value.items()}
        if isinstance(value, list):
            return [resolve(item) for item in value]
        if isinstance(value, str) and value.startswith("asset://"):
            key = value.removeprefix("asset://")
            if key not in assets:
                raise ValueError(f"Missing deployment asset: {key}")
            path = Path(assets[key]).expanduser()
            return str((base / path).resolve() if not path.is_absolute() else path)
        return value

    return resolve(config), assets


def mount_vendor():
    # Explicit, package-local source snapshot; never an old experiment directory.
    for directory in (ROOT / "render/vendor", ROOT / "render/decoder"):
        sys.path.insert(0, str(directory))
