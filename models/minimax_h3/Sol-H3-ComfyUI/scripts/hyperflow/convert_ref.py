"""HyperFlow ref2va DiT (MiniMax-H3 `transformer_ref`) → FP8 checkpoint + exact AdaLN table, one shard at a time.

The HyperFlow LoRA was trained on `transformer` and declares `transformer_ref` compatible (its metadata). This applies
convert_fp8.py's recipe (LoRA merged exactly, 300 block linears per-tensor e4m3, 12 token-refiner linears merged in
BF16) and drops adaln_proj, whose rows are computed here instead, at the forward pair sets of an existing fl2va table:
image references are held at the keyframe noise level exactly like fl2va keyframes (hyperflow_h3.schedule), so an
image-only ref2va request runs the same (t, r) sets. The rows follow build_adaln_table.py: temb from this checkpoint's
own time embedder with the LoRA (TwoTimeEmbedder), then one official MiniMaxH3AdaLayerNormModulation call per pair set.

Each BF16 shard is fetched from SOURCE_URL/<shard>, converted and deleted, so the 66 GB checkpoint never sits on disk; a
background thread downloads up to AHEAD shards beyond the one being converted.
Usage: convert_ref.py SOURCE_URL HYPERFLOW_LORA FL2VA_TABLE.pt OUT_DIR
  SOURCE_URL  e.g. https://www.modelscope.cn/models/MiniMax/MiniMax-H3/resolve/master/transformer_ref
"""
import json
import subprocess
import sys
import threading
import time
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

TARGETS = (".attn.to_q.weight", ".attn.to_k.weight", ".attn.to_v.weight", ".attn.to_out.0.weight",
           ".ff.net.0.proj.weight", ".ff.net.2.weight")
FP8_MAX = torch.finfo(torch.float8_e4m3fn).max
AHEAD = 2


def fetch(url, path):
    for attempt in range(8):
        if subprocess.run(["aria2c", "-q", "-x", "16", "-s", "16", "-k", "64M", "--file-allocation=none",
                           "--continue=true", "-d", str(path.parent), "-o", path.name, url]).returncode == 0:
            return
        time.sleep(10)
    raise RuntimeError("download failed: " + url)


def main():
    source, lora_path, table_path, out = sys.argv[1].rstrip("/"), Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4])
    out.mkdir(parents=True, exist_ok=True)
    stage = out / "bf16"
    stage.mkdir(exist_ok=True)
    dev = torch.device("cuda")
    from accelerate import init_empty_weights
    from diffusers import MiniMaxH3Transformer3DModel
    from diffusers.models.transformers.transformer_minimax_h3 import MiniMaxH3AdaLayerNormModulation
    from hyperflow_h3 import TwoTimeEmbedder, read_metadata

    for name in ("config.json", "diffusion_pytorch_model.safetensors.index.json"):
        if not (out / name).exists():
            fetch(f"{source}/{name}", out / name)
    index = json.loads((out / "diffusion_pytorch_model.safetensors.index.json").read_text())
    shards = sorted(set(index["weight_map"].values()))
    metadata = read_metadata(lora_path)
    assert metadata.lora_alpha == metadata.lora_rank, "merge assumes alpha == rank (scale 1)"
    lora = safe_open(str(lora_path), "pt")
    lora_keys = set(lora.keys())
    keys = torch.load(table_path, map_location="cpu", weights_only=True)["keys"]
    keys = [k for k in keys if len(k) == 3]  # fl2va sets: (video, condition 0.999, audio)
    if not keys:
        raise ValueError("no fl2va pair sets in " + str(table_path))

    # The engine's time embedder for this checkpoint: official modules, the LoRA merged as build_fp8_transformer does.
    config = MiniMaxH3Transformer3DModel.load_config(str(out))
    with init_empty_weights():
        model = MiniMaxH3Transformer3DModel.from_config(config)
    first = stage / shards[0]
    if not (out / shards[0]).exists() and not (stage / (shards[0] + ".done")).exists():
        fetch(f"{source}/{shards[0]}", first)
        (stage / (shards[0] + ".done")).touch()
    src = first if first.exists() else out / shards[0]
    with safe_open(str(src), "pt") as f:
        state = {k: f.get_tensor(k) for k in f.keys() if k.startswith("time_embedder.")}
    model.time_embedder.load_state_dict({k.split(".", 1)[1]: v for k, v in state.items()}, assign=True)
    model.time_proj.to("cpu")
    wrapper = TwoTimeEmbedder.wrap(model, metadata.gate)
    with torch.no_grad():
        for prefix, module in (("time_embedder", wrapper.base), ("endpoint_time_embedder", wrapper.endpoint)):
            for lin in ("linear_1", "linear_2"):
                layer = getattr(module, lin)
                a = lora.get_tensor(f"transformer.{prefix}.{lin}.lora_A.weight").float()
                b = lora.get_tensor(f"transformer.{prefix}.{lin}.lora_B.weight").float()
                layer.weight.copy_((layer.weight.float() + metadata.lora_alpha / metadata.lora_rank * (b @ a))
                                   .to(layer.weight.dtype))
    wrapper.to(dev)
    model.time_proj.to(dev)
    with torch.no_grad():
        tembs = []
        for key in keys:
            t = torch.tensor([p[0] for p in key], dtype=torch.float32, device=dev)
            r = torch.tensor([p[1] for p in key], dtype=torch.float32, device=dev)
            with wrapper.endpoint_context(r):
                tembs.append(wrapper(model.time_proj(t).to(next(wrapper.parameters()).dtype)))

    # Downloads run ahead of conversion (bounded by AHEAD to cap disk use); a shard is ready when <shard>.done exists.
    pending = [s for s in shards if not ((out / s).exists() and (out / (s + ".adaln.pt")).exists())]
    converted = threading.Semaphore(AHEAD)

    queue, lock = list(pending), threading.Lock()

    def prefetch():
        while True:
            converted.acquire()
            with lock:
                if not queue:
                    return
                shard = queue.pop(0)
            if not (stage / (shard + ".done")).exists():
                fetch(f"{source}/{shard}", stage / shard)
                (stage / (shard + ".done")).touch()
    for _ in range(AHEAD):  # AHEAD shards download in parallel
        threading.Thread(target=prefetch, daemon=True).start()

    hidden = config["hidden_size"]
    tables = {}
    weight_map, merged, refiner = {}, 0, 0
    t0 = time.monotonic()
    for shard in shards:
        rows_file = out / (shard + ".adaln.pt")
        if (out / shard).exists() and rows_file.exists():  # resumed run
            with safe_open(str(out / shard), "pt") as f:
                for key in f.keys():
                    weight_map[key] = shard
                    merged += key.startswith("transformer_blocks.") and key.endswith(TARGETS)
                    refiner += key.startswith("token_refiner.") and key.endswith(TARGETS)
            tables.update(torch.load(rows_file, weights_only=True))
            (stage / shard).unlink(missing_ok=True)
            continue
        path = stage / shard
        while not (stage / (shard + ".done")).exists():
            time.sleep(2)
        tensors, adaln = {}, {}
        with safe_open(str(path), "pt") as f:
            for key in f.keys():
                t = f.get_tensor(key)
                if ".adaln_proj." in key:
                    adaln[key] = t
                    continue
                if key.startswith("token_refiner.") and key.endswith(TARGETS):
                    base = key[: -len(".weight")]
                    a = lora.get_tensor(f"transformer.{base}.lora_A.weight").to(dev, torch.float32)
                    b = lora.get_tensor(f"transformer.{base}.lora_B.weight").to(dev, torch.float32)
                    tensors[key] = (t.to(dev, torch.float32) + b @ a).to(t.dtype).cpu()
                    refiner += 1
                elif key.startswith("transformer_blocks.") and key.endswith(TARGETS):
                    base = key[: -len(".weight")]
                    a_key, b_key = f"transformer.{base}.lora_A.weight", f"transformer.{base}.lora_B.weight"
                    if a_key not in lora_keys or b_key not in lora_keys:
                        raise KeyError(f"no HyperFlow LoRA for {key}")
                    w = t.to(dev, torch.float32)
                    w += lora.get_tensor(b_key).to(dev, torch.float32) @ lora.get_tensor(a_key).to(dev, torch.float32)
                    scale = (w.abs().max().clamp(min=1e-12) / FP8_MAX).to(torch.float32)
                    tensors[key] = (w / scale).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn).cpu()
                    tensors[f"{base}.scale"] = scale.reshape(1).cpu()
                    weight_map[f"{base}.scale"] = shard
                    merged += 1
                else:
                    tensors[key] = t
                weight_map[key] = shard
        # This shard's AdaLN rows: one official call per pair set, at its live batch size.
        rows = {}
        blocks = sorted({int(k.split(".")[1]) for k in adaln})
        with torch.no_grad():
            for i in blocks:
                prefix = f"transformer_blocks.{i}.adaln_proj.linear."
                if prefix + "weight" not in adaln or prefix + "bias" not in adaln:
                    raise RuntimeError(f"block {i} adaln_proj split across shards")
                module = MiniMaxH3AdaLayerNormModulation(tembs[0].shape[-1], hidden)
                module.linear.weight.data = adaln[prefix + "weight"].to(dev)
                module.linear.bias.data = adaln[prefix + "bias"].to(dev)
                rows[i] = torch.cat([torch.cat(module(temb), dim=-1).reshape(temb.shape[0], -1) for temb in tembs]).cpu()
        torch.save(rows, rows_file)
        tables.update(rows)
        save_file(tensors, str(out / shard), metadata={"format": "pt"})
        path.unlink()
        (stage / (shard + ".done")).unlink()
        converted.release()
        print("SHARD", shard, "blocks", blocks, round(time.monotonic() - t0, 1), flush=True)
        del tensors, adaln
    if merged != 300 or refiner != 12:
        raise RuntimeError(f"expected 300 block + 12 refiner merges, got {merged} + {refiner}")
    if sorted(tables) != list(range(config["num_layers"])):
        raise RuntimeError(f"AdaLN rows for blocks {sorted(tables)}")
    index["weight_map"] = weight_map
    index["metadata"] = {"total_size": sum((out / s).stat().st_size for s in shards)}
    (out / "diffusion_pytorch_model.safetensors.index.json").write_text(json.dumps(index, indent=2))
    torch.save({"keys": keys, "tables": [tables[i] for i in range(config["num_layers"])]}, out / "adaln_table.pt")
    for shard in shards:
        (out / (shard + ".adaln.pt")).unlink()
    for left in stage.iterdir():
        left.unlink()
    stage.rmdir()
    report = {"merged": merged, "refiner_merged": refiner, "lora": str(lora_path), "base": source,
              "adaln_proj": "dropped (AdaLN table)", "adaln_forwards": len(keys),
              "recipe": "W+B@A fp32, per-tensor e4m3 absmax/448; time embedders unmerged"}
    (out / "hyperflow_fp8.json").write_text(json.dumps(report, indent=2))
    print("DONE", json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
