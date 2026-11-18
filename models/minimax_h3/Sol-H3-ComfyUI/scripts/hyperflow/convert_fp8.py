"""HyperFlow-8 MiniMax-H3 DiT → merged FP8 checkpoint for 90 GB-RAM / 32 GB-VRAM hosts.

Shard by shard (never the whole BF16 DiT in RAM):
  * the 300 block linears HyperFlow adapts (50 blocks x attn.to_q/k/v/out.0, ff.net.0.proj, ff.net.2) get the
    LoRA merged exactly (scale = alpha/rank = 1, fp32 accumulate: W + B @ A) and are then quantised per tensor to
    float8_e4m3fn with an fp32 `<name>.scale` (absmax / 448);
  * the 12 token-refiner linears HyperFlow also adapts get the same exact merge but stay BF16;
  * every other tensor is copied unchanged (norms, AdaLN, embedders, text refiners, proj_out ...).
The two time embedders keep their LoRA unmerged: load_hyperflow_lora still wraps them (TwoTimeEmbedder, gate).
Writes OUT_DIR/{config.json, diffusion_pytorch_model-*.safetensors, index, hyperflow_fp8.json}.
The same LoRA serves `transformer_ref` (ref2va): its metadata lists both subfolders.
  --drop-adaln  omit the 26 GB of adaln_proj weights (the engine replaces them with build_adaln_table.py's table)
  --consume     wait for each source shard to appear and delete it once converted (a download landing shard by
                shard into H3_TRANSFORMER_DIR never needs the whole BF16 checkpoint on disk)
Usage: convert_fp8.py H3_TRANSFORMER_DIR HYPERFLOW_LORA OUT_DIR [--drop-adaln] [--consume]
"""
import json
import shutil
import sys
import time
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

TARGETS = (".attn.to_q.weight", ".attn.to_k.weight", ".attn.to_v.weight", ".attn.to_out.0.weight",
           ".ff.net.0.proj.weight", ".ff.net.2.weight")
FP8_MAX = torch.finfo(torch.float8_e4m3fn).max  # 448


def main():
    src, lora_path, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    drop_adaln, consume = "--drop-adaln" in sys.argv, "--consume" in sys.argv
    out.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda")
    with safe_open(str(lora_path), "pt") as f:
        meta = f.metadata()
    assert int(meta["lora_alpha"]) == int(meta["lora_rank"]), "merge assumes alpha == rank (scale 1)"
    lora = safe_open(str(lora_path), "pt")
    lora_keys = set(lora.keys())
    index = json.loads((src / "diffusion_pytorch_model.safetensors.index.json").read_text())
    shards = sorted(set(index["weight_map"].values()))
    weight_map, merged, quantized, refiner, report = {}, 0, 0, 0, {"max_rel_quant_err": 0.0}
    t0 = time.monotonic()
    for shard in shards:
        if (out / shard).exists() and consume:  # resumed run: this shard is already converted
            with safe_open(str(out / shard), "pt") as f:
                for key in f.keys():
                    weight_map[key] = shard
                    merged += key.startswith("transformer_blocks.") and key.endswith(TARGETS)
                    quantized += key.startswith("transformer_blocks.") and key.endswith(TARGETS)
                    refiner += key.startswith("token_refiner.") and key.endswith(TARGETS)
            continue
        while consume and not (src / (shard + ".done")).exists():
            time.sleep(10)
        tensors = {}
        with safe_open(str(src / shard), "pt") as f:
            for key in f.keys():
                if drop_adaln and ".adaln_proj." in key:
                    continue
                t = f.get_tensor(key)
                if key.startswith("token_refiner.") and key.endswith(TARGETS):
                    base = key[: -len(".weight")]
                    a = lora.get_tensor(f"transformer.{base}.lora_A.weight").to(dev, torch.float32)
                    b = lora.get_tensor(f"transformer.{base}.lora_B.weight").to(dev, torch.float32)
                    tensors[key] = (t.to(dev, torch.float32) + b @ a).to(t.dtype).cpu()
                    weight_map[key] = shard
                    refiner += 1
                    continue
                if key.startswith("transformer_blocks.") and key.endswith(TARGETS):
                    base = key[: -len(".weight")]
                    a_key, b_key = f"transformer.{base}.lora_A.weight", f"transformer.{base}.lora_B.weight"
                    if a_key not in lora_keys or b_key not in lora_keys:
                        raise KeyError(f"no HyperFlow LoRA for {key}")
                    w = t.to(dev, torch.float32)
                    w += lora.get_tensor(b_key).to(dev, torch.float32) @ lora.get_tensor(a_key).to(dev, torch.float32)
                    merged += 1
                    amax = w.abs().max().clamp(min=1e-12)
                    scale = (amax / FP8_MAX).to(torch.float32)
                    q = (w / scale).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
                    err = float(((q.to(torch.float32) * scale) - w).norm() / w.norm())
                    report["max_rel_quant_err"] = max(report["max_rel_quant_err"], err)
                    tensors[key] = q.cpu()
                    tensors[f"{base}.scale"] = scale.reshape(1).cpu()
                    weight_map[f"{base}.scale"] = shard
                    quantized += 1
                else:
                    tensors[key] = t
                weight_map[key] = shard
        save_file(tensors, str(out / shard), metadata={"format": "pt"})
        print("SHARD", shard, round(time.monotonic() - t0, 1), flush=True)
        del tensors
        if consume:
            (src / shard).unlink()
    if merged != 300 or refiner != 12:
        raise RuntimeError(f"expected 300 block + 12 refiner merges, got {merged} + {refiner}")
    (out / "diffusion_pytorch_model.safetensors.index.json").write_text(json.dumps(
        {"metadata": {"total_size": sum((out / s).stat().st_size for s in shards)}, "weight_map": weight_map}, indent=2))
    report["adaln_proj"] = "dropped (AdaLN table)" if drop_adaln else "kept"
    shutil.copy(src / "config.json", out / "config.json")
    report.update(merged=merged, refiner_merged=refiner, quantized=quantized, lora=str(lora_path), base=str(src),
                  recipe="W+B@A fp32, per-tensor e4m3 absmax/448; time embedders unmerged")
    (out / "hyperflow_fp8.json").write_text(json.dumps(report, indent=2))
    print("DONE", json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
