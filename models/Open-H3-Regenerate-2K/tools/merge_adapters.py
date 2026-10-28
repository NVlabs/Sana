"""Merge the three Open-H3-Regenerate-2K adapters into one LoRA with their strengths folded in.

Adapters: refiner step5000 EMA (rank384, alpha384), LightX2V FL2VA Turbo v1.2 (rank128, alpha8), LMS Sharp v1.0
(rank64), each at its own strength s_i (given on the command line). The runtime adds these deltas to the
FP32 base weight and casts once. Concatenating the factors along the rank axis gives exactly the same delta:
    B_m @ A_m = sum_i s_i * B_i @ A_i,   A_m = cat(A_i), B_m = cat(s_i * B_i)
so the merged file is a plain LoRA with per-module rank (sum of ranks) and alpha == rank (scale 1).

Sharp's 50 curve AdaLN branches take an 8-D timestep curve, not the native input, so they cannot be folded into a
linear delta; they are stored with the Sharp strength folded in, in a separate small file. The 1025x8 curve table itself
is base-derived and is fetched at setup, not redistributed.

Usage: python tools/merge_adapters.py --refiner R.safetensors --turbo T.safetensors --sharp S.safetensors \
           --refiner-strength S_R --turbo-strength S_T --sharp-strength S_S --out lora/
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


SOURCES = {
    "refiner": {"sha256": "519367bec51915929806bcd313384a4df9c05588d3db9e9564fe3de2ba8604a3"},
    "turbo": {"sha256": "c3d4a2cf618efea71b9e21a4baaa12d412f1eb6c2b6f86efacaf0ebb6814b689",
              "repo": "lightx2v/Minimax-h3-Turbo", "revision": "3ec17a324ced54151364f24f8b5fb6bf7e26414f",
              "upstream_file": "minimax_h3_fl2v_turbo_4step_v1.2_768p_bf16.safetensors"},
    "sharp": {"sha256": "16f3195bc6bffa431c0598dc2031e520ba058b4fcbc9f104db9e3fdfdeacf60c",
              "repo": "Alissonerdx/Minimax-H3-ComfyUI", "revision": "849fb3d4f434f2e9b4a4d4f7253471bf5e6719f9",
              "upstream_file": "loras/minimax_h3_lms_v1.0_r64.safetensors"},
}
CURVE_TABLE = {"repo": "Comfy-Org/MiniMax-H3", "revision": "e5eb578a89295337b8ff433a035929ce0279e0b6",
               "file": "diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors",
               "sha256": "c02a6c11888297688c1e6278185ea1f947023acfc69f9003bbcdcec9a229a8e7"}


def convert_external(path):
    """LMS Sharp (original H3 key layout) -> {Diffusers module: (A, B)} plus the 8-D curve AdaLN branches."""
    raw = load_file(str(path), device="cpu")
    out, curve = {}, {}
    roots = sorted(k[:-14] for k in raw if k.endswith(".lora_A.weight"))
    assert set(raw) == {r + s for r in roots for s in [".lora_A.weight", ".lora_B.weight"]}
    for src in roots:
        a, b = raw[src + ".lora_A.weight"].float(), raw[src + ".lora_B.weight"].float()
        name = src.removeprefix("diffusion_model.")
        name = name.replace("token_refiner.blocks.", "token_refiner.refiner_blocks.")
        if name.startswith("blocks."):
            name = "transformer_blocks." + name[7:]
        name = name.replace("final_layer.adaln_proj.linear", "norm_out.linear")
        name = name.replace(".attn.out_proj", ".attn.to_out.0")
        if name.endswith(".attn.qkv_proj"):
            for tag, bb in zip(["to_q", "to_k", "to_v"], b.chunk(3, dim=0)):
                out[name[:-8] + tag] = (a, bb.contiguous())
        elif name.endswith(".mlp.fc1"):
            gate, value = b.chunk(2, dim=0)
            out[name[:-8] + ".ff.net.0.proj"] = (a, torch.cat([value, gate], dim=0))
        elif name.endswith(".mlp.fc2"):
            out[name[:-8] + ".ff.net.2"] = (a, b)
        elif ".adaln_proj.linear" in name and a.shape[1] == 8:
            curve[name] = (a, b)
        else:
            out[name] = (a, b)
    return out, curve, len(roots)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
for role in ("refiner", "turbo", "sharp"):
    parser.add_argument(f"--{role}", type=Path, required=True)
    parser.add_argument(f"--{role}-strength", type=float, required=True)
parser.add_argument("--out", type=Path, required=True)
args = parser.parse_args()
STRENGTH = {role: getattr(args, f"{role}_strength") for role in ("refiner", "turbo", "sharp")}
OUT = args.out
t0 = time.time()
sources = {}
for role in ("refiner", "turbo", "sharp"):
    path = getattr(args, role)
    digest = sha256(path)
    assert digest == SOURCES[role]["sha256"], (role, digest)
    sources[role] = {"file": path.name, "bytes": path.stat().st_size, **SOURCES[role]}

# --- load exactly as runtime/adapters.attach does (factor pairs per Diffusers module name, alpha/rank folded)
raw = load_file(str(args.refiner), device="cpu")
refiner = {k[:-7]: (v.float(), raw[k[:-7] + ".lora_B"].float()) for k, v in raw.items() if k.endswith(".lora_A")}
assert len(refiner) == 312 and len(raw) == 624
assert all(a.shape[0] == b.shape[1] == 384 for a, b in refiner.values())     # alpha384 / rank384 = 1
del raw
turbo_raw = load_file(str(args.turbo), device="cpu")
suffix = ".lora_A.default.weight"
roots = sorted(k[:-len(suffix)] for k in turbo_raw if k.endswith(suffix))
assert len(roots) == 312
turbo = {n: (turbo_raw[n + suffix].float(), turbo_raw[n + ".lora_B.default.weight"].float() * (8.0 / 128)) for n in roots}
assert all(a.shape[0] == 128 for a, _ in turbo.values())
del turbo_raw
sharp, curves, sharp_pairs = convert_external(args.sharp)
assert len(curves) == 50
adapters = {"refiner": refiner, "turbo": turbo, "sharp": sharp}

# --- merge
merged, ranks, checks = {}, {}, []
gen = torch.Generator().manual_seed(0)
names = sorted(set().union(*(set(a) for a in adapters.values())))
for name in names:
    parts = [(role, adapters[role][name]) for role in ("refiner", "turbo", "sharp") if name in adapters[role]]
    a_m = torch.cat([a for _, (a, _) in parts], dim=0).contiguous()
    b_m = torch.cat([STRENGTH[role] * b for role, (_, b) in parts], dim=1).contiguous()
    assert torch.isfinite(a_m).all() and torch.isfinite(b_m).all()
    x = torch.randn(a_m.shape[1], 4, generator=gen, dtype=torch.float64)
    got = b_m.double() @ (a_m.double() @ x)
    want = sum(STRENGTH[role] * (b.double() @ (a.double() @ x)) for role, (a, b) in parts)
    rel = float((got - want).norm() / want.norm().clamp_min(1e-30))
    checks.append(rel)
    merged[f"transformer.{name}.lora_A.weight"] = a_m
    merged[f"transformer.{name}.lora_B.weight"] = b_m
    ranks[name] = {"rank": a_m.shape[0], "parts": {role: a.shape[0] for role, (a, _) in parts}}

# exact full-matrix check on a few modules (FP32 merged factors vs FP64 sum of individually scaled deltas)
full = []
for name in names[:: max(1, len(names) // 6)][:6]:
    parts = [(role, adapters[role][name]) for role in ("refiner", "turbo", "sharp") if name in adapters[role]]
    want = sum(STRENGTH[role] * (b.double() @ a.double()) for role, (a, b) in parts)
    got = merged[f"transformer.{name}.lora_B.weight"].double() @ merged[f"transformer.{name}.lora_A.weight"].double()
    full.append({"module": name, "max_abs_err": float((got - want).abs().max()), "max_abs_delta": float(want.abs().max())})

curve_tensors = {}
for name, (a, b) in sorted(curves.items()):
    curve_tensors[f"transformer.{name}.curve_A.weight"] = a.contiguous()
    curve_tensors[f"transformer.{name}.curve_B.weight"] = (STRENGTH["sharp"] * b).contiguous()

OUT.mkdir(parents=True, exist_ok=True)
lora_file, curve_file = OUT / "open_h3_regenerate_2k_lora.safetensors", OUT / "sharp_curve_branches.safetensors"
meta = {"format": "lora", "scale": "alpha == rank for every module (strengths folded)", "dtype": "float32"}
save_file(merged, str(lora_file), metadata=meta)
save_file(curve_tensors, str(curve_file), metadata={"format": "lms-sharp-curve-adaln", "strength": "folded"})

config = {
    "base_model": "MiniMaxAI/MiniMax-H3@6818f6c32d12b210915e44ad56a4228c2608f160 (transformer, Diffusers layout)",
    "turbo_alpha_over_rank_folded": 8.0 / 128,
    "apply": "W = cast(W_fp32 + B @ A) once per targeted linear; equivalent to the reference ordered sum of deltas",
    "modules": len(names),
    "rank_histogram": {str(r): sum(1 for v in ranks.values() if v["rank"] == r) for r in sorted({v["rank"] for v in ranks.values()})},
    "curve_branches": {"count": len(curves), "file": curve_file.name, "input": "8-D timestep curve from adaln_t_table",
                       "table": {**CURVE_TABLE, "tensor": "adaln_t_table", "shape": [1025, 8],
                                 "note": "base-derived; fetched at setup, not redistributed"}},
    "sources": sources,
    "outputs": {},
}
for f in (lora_file, curve_file):
    config["outputs"][f.name] = {"bytes": f.stat().st_size, "sha256": sha256(f)}
(OUT / "lora_config.json").write_text(json.dumps(config, indent=1))
(OUT / "module_ranks.json").write_text(json.dumps(ranks, indent=1))
evidence = {"matvec_rel_err_max": max(checks), "matvec_rel_err_mean": sum(checks) / len(checks), "modules_checked": len(checks),
            "full_matrix_checks": full, "sharp_source_pairs": sharp_pairs, "seconds": time.time() - t0}
(OUT / "MERGE_VERIFY.json").write_text(json.dumps(evidence, indent=1))
print(json.dumps({**evidence, "outputs": config["outputs"], "rank_histogram": config["rank_histogram"]}, indent=1))
assert evidence["matvec_rel_err_max"] < 1e-5, "merge mismatch"
print("MERGE_OK")
