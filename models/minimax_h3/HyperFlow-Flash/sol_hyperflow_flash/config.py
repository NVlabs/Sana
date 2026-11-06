"""Presets: `official` (the checkpoint on the MiniMax-H3 modular pipeline, as released) and `sol-engine` (the Sol-Engine recipe).

Every optimisation is an independent switch:

  sparse        Sol-Attn through the Sol-H3 cuDNN block-sparse backend (``sol_bsa``) in every live DiT block except
                ``edge_blocks``, at every step from ``dense_steps`` on; every other (step, block) pair keeps dense attention
                (diffusers dispatch, BF16). ``tau`` sets the density (higher = sparser).
  sink          exact KV sink of the sparse backend. "prefix": every row before the target video (text, target audio) is
                exact and outside the density budget (t2v). "text_audio": the 64-blocks holding text or audio rows are exact;
                visual reference rows are routed like the target video (Sol-H3's own ref2va policy).
  fusions       Sol-H3 block fusions on the live blocks: RMSNorm + AdaLN modulate, residual + gate + RMSNorm + modulate,
                SwiGLU, per-head QK RMSNorm + partial RoPE.
  merge_tuned   Sol residual merge launched with block_n 128 / 4 warps / 1 stage (same kernel and math).
  vae           "stock": diffusers decode on every rank. "solh3": Sol-H3's compiled tile-parallel decode of the same H3 VAE
                (1 GPU: compiled decoder only).
  ref_encode    ref2va reference encode: "stock" | "sharded" (Sol-H3 encode_parallel, one clip's tiles over the ranks) |
                "global" (every clip's tiles over the ranks, one gather); encode_compile: compiled encoder.
  prep_threads  ref2va: reference video frames resized on this many host threads (same PIL calls; 0 = stock).
  te_trim       Qwen3-VL conditioner run only up to the layer whose pre-norm output MiniMax-H3 reads (bitwise).
  transport     Ulysses all-to-all format (world > 1): "bf16" | "int8fp8" (Sol-H3 compressed transport on every call) |
                "int8fp8-sparse" (only on the Sol-Attn calls).

The H3 VAE weights and code are always the base model's (diffusers AutoencoderKLMiniMaxH3).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace


@dataclass(frozen=True)
class Preset:
    name: str
    sparse: bool = False
    tau: float = 1.25
    dense_steps: int = 2
    edge_blocks: tuple = (0, 1, 48, 49)
    fusions: bool = False
    sink: str = "prefix"
    merge_tuned: bool = False
    vae: str = "stock"
    ref_encode: str = "stock"
    encode_compile: bool = False
    prep_threads: int = 0
    transport: str = "bf16"
    te_trim: bool = False
    notes: str = field(default="", compare=False)

    def as_dict(self):
        d = asdict(self)
        d["edge_blocks"] = list(self.edge_blocks)
        return d


_SOL_ENGINE = dict(sparse=True, fusions=True, merge_tuned=True, vae="solh3", te_trim=True, transport="int8fp8-sparse")

PRESETS = {
    "official": Preset("official", notes="checkpoint on the MiniMax-H3 modular pipeline, dense attention"),
    "sol-engine": Preset("sol-engine", **_SOL_ENGINE, notes="Sol-Engine recipe"),
}
REF2VA = {
    "official": PRESETS["official"],
    "sol-engine": Preset("sol-engine", **_SOL_ENGINE, sink="text_audio", ref_encode="global", encode_compile=True, prep_threads=32,
                   notes="Sol-Engine recipe + sharded compiled reference encode + threaded reference prep"),
}


def get(name: str, task: str = "t2va", attention_backend: str = "sol", **overrides) -> Preset:
    """Preset `name` of `task`; attention_backend "dense" keeps every other switch of the preset and turns Sol-Attn off
    (ablation)."""
    table = REF2VA if task == "ref2va" else PRESETS
    if name not in table:
        raise KeyError(f"unknown {task} preset {name!r}; choose from {sorted(table)}")
    p = table[name]
    if attention_backend not in ("sol", "dense"):
        raise ValueError(f"attention_backend must be 'sol' or 'dense', got {attention_backend!r}")
    if attention_backend == "dense" and p.sparse:
        p = replace(p, name=p.name + "-dense", sparse=False)
    overrides = {k: v for k, v in overrides.items() if v is not None}
    if overrides:
        d = p.as_dict()
        d.update(overrides)
        d["edge_blocks"] = tuple(d["edge_blocks"])
        p = Preset(**d)
    return p
