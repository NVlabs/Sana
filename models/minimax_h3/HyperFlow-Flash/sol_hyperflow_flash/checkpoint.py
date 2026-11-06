"""The only module that knows the HyperFlow H3 Flash checkpoint layout.

The checkpoint ships its own model code next to the weights (diffusers remote code): one folder per partition with
`config.json`, the safetensors shards, the transformer module named by `config.json` `auto_map` and the INT8 module it
imports. This runner loads that code from the folder itself and finds what it needs by feature, not by file name:

* the transformer class: `auto_map["AutoModel"]` (falls back to `_class_name` in the module of the same stem);
* the INT8 module: the module of that package exposing `set_kernel_options` and `resolve_int8_gemm`;
* `DroppedBlock`, `set_lowrank_rank_multiple` and `set_ff_chunk` from the transformer module.

Run-time settings applied after loading (no weight change):

* low-rank linears zero-padded to a rank multiple of 64 (aligned GEMMs; the same function, see the model code);
* the feed-forward runs over the whole sequence (`set_ff_chunk(model, 0)`), as in every measurement reported here;
* INT8 kernel options `int8_gemm="auto"` (cuBLASLt on compute capability 10.x, Triton on Hopper) and
  `rotate_quant="dense"`.
"""
from __future__ import annotations

import hashlib
import importlib
import json
import sys
import types
from dataclasses import dataclass, field
from pathlib import Path

HF_REPO = "VideoRebirthAI/hyperflow-h3-flash"
HF_REVISION = "9e9ed14b8be4b751d7517b8ca58dd199d509720c"
PARTITIONS = {"t2va": "fl2va_int8_convrot", "ref2va": "ref2va_int8_convrot"}
RANK_MULTIPLE = 64
KERNEL_OPTIONS = {"int8_gemm": "auto", "rotate_quant": "dense"}


def resolve(weights: str, task: str, *, revision: str | None = HF_REVISION, local_files_only: bool = False) -> Path:
    """Local folder of the `task` partition. `weights` is a local checkpoint root, a partition folder, or a Hub repo id
    (downloaded at `revision`, only the partition's files)."""
    if task not in PARTITIONS:
        raise ValueError(f"task {task!r} not in {sorted(PARTITIONS)}")
    sub = PARTITIONS[task]
    path = Path(weights).expanduser()
    if path.is_dir():
        for cand in (path / sub, path):
            if (cand / "config.json").is_file():
                return cand
        raise FileNotFoundError(f"{path} holds neither {sub}/config.json nor config.json")
    from huggingface_hub import snapshot_download

    root = snapshot_download(weights, revision=revision, allow_patterns=[f"{sub}/*"], local_files_only=local_files_only)
    return Path(root) / sub


def _class_ref(cfg: dict) -> tuple[str, str]:
    ref = (cfg.get("auto_map") or {}).get("AutoModel")
    if isinstance(ref, (list, tuple)):
        ref = ref[0]
    if ref:
        module, _, cls = ref.rpartition(".")
        return module, cls
    raise KeyError("config.json has no auto_map.AutoModel entry naming the transformer module")


def load_code(folder: Path):
    """Import the folder's model code as a package (its modules import each other relatively). Returns (transformer
    module, transformer class, INT8 module)."""
    folder = Path(folder).resolve()
    cfg = json.loads((folder / "config.json").read_text())
    module_name, cls_name = _class_ref(cfg)
    if not (folder / f"{module_name}.py").is_file():
        raise FileNotFoundError(f"{folder}/{module_name}.py (named by config.json auto_map) is missing")
    pkg = "sol_hyperflow_flash_model_" + hashlib.sha1(str(folder).encode()).hexdigest()[:12]
    if pkg not in sys.modules:
        mod = types.ModuleType(pkg)
        mod.__path__ = [str(folder)]
        sys.modules[pkg] = mod
    tmod = importlib.import_module(f"{pkg}.{module_name}")
    cls = getattr(tmod, cls_name)
    int8 = [m for n, m in list(sys.modules.items())
            if n.startswith(pkg + ".") and hasattr(m, "set_kernel_options") and hasattr(m, "resolve_int8_gemm")]
    if len(int8) != 1:
        raise RuntimeError(f"expected one INT8 module with set_kernel_options in {folder}, found {len(int8)}")
    for name in ("DroppedBlock", "set_lowrank_rank_multiple", "set_ff_chunk"):
        if not hasattr(tmod, name):
            raise AttributeError(f"{tmod.__name__} has no {name}: unsupported model code revision")
    return tmod, cls, int8[0]


@dataclass
class Loaded:
    model: object
    folder: Path
    module: object
    int8: object
    dropped_cls: type
    info: dict = field(default_factory=dict)


def load_transformer(folder: Path, torch, *, rank_multiple: int = RANK_MULTIPLE, ff_chunk: int = 0,
                     kernel_options: dict | None = None) -> Loaded:
    tmod, cls, int8 = load_code(folder)
    opts = dict(KERNEL_OPTIONS, **(kernel_options or {}))
    int8.set_kernel_options(**opts)
    model = cls.from_pretrained(str(folder), torch_dtype=torch.bfloat16)
    model.eval()
    repadded = tmod.set_lowrank_rank_multiple(model, rank_multiple)
    tmod.set_ff_chunk(model, ff_chunk)
    info = dict(folder=str(folder), transformer_class=cls.__name__, kernel_options=opts, rank_multiple=rank_multiple,
                repadded_linears=int(repadded), ff_chunk=ff_chunk)
    return Loaded(model=model, folder=Path(folder), module=tmod, int8=int8, dropped_cls=tmod.DroppedBlock, info=info)
