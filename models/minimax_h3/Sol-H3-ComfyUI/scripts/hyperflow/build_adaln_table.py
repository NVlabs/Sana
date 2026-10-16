"""Build the exact HyperFlow AdaLN table (replaces 26 GB of adaln_proj weights).

1. Record the ordered (timestep, endpoint) pair set of every forward the *real* HyperFlow pipeline runs on t2va / fl2va
   payloads (DiT forward replaced by a recorder, no denoising compute), so the values are bit-exact with generation.
2. For each pair set, compute temb with the transformer's own TwoTimeEmbedder and each block's adaln_proj from the
   checkpoint weights with ONE linear call of exactly that batch (Sol-H3 install_exact_t2va_lookup rule: cuBLAS picks
   a different kernel per M, so a single big batch differs by 1 BF16 ulp), and save {keys, tables}.
3. Self-check: block 0's official MiniMaxH3AdaLayerNormModulation on every set equals the table bit for bit.
Run it with the runtime Python. PROFILE is a HyperFlow profile from SOL_ENGINE_CONFIG (the BF16 one configure.py
writes from --hyperflow-weights/--hyperflow-deps); FP8_DIR is convert_fp8.py's output *with* adaln_proj (the default).
One task per run (host RAM), then merge_adaln_tables.py. Payloads come from save_payload.py.
Usage: SOL_ENGINE_CONFIG=config.json build_adaln_table.py PROFILE FP8_DIR OUT.pt PAYLOAD.pt [PAYLOAD.pt ...]
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "sol_h3_engine"))
sys.path.insert(0, str(ROOT))


def main():
    profile, fp8_dir, out, payloads = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3]), sys.argv[4:]
    from sol_h3_engine.config import load_profile
    options = load_profile(profile)
    if options.get("precision") != "hyperflow":
        raise ValueError("%s is not a HyperFlow profile" % profile)
    options["hyperflow"]["fp8_dir"] = str(fp8_dir.resolve(strict=True))
    import torch
    from torch import nn
    from safetensors import safe_open
    from hyperflow_worker import HyperFlow
    import torch as _t
    tasks = sorted({_t.load(p, map_location="cpu", weights_only=False).get("task", "t2va") for p in payloads})
    # t2va and fl2va must be separate pipelines here: a shared fl2va pipe cannot run t2va (keyframes=[] → torch.cat).
    options["hyperflow"]["workflows"] = tasks
    options["hyperflow"]["share_fl2va_pipe"] = False
    options["hyperflow"].pop("adaln_table", None)
    options["hyperflow"].pop("ref_adaln_table", None)
    options["work_dir"] = str(out.parent / "adaln-build")
    paths = json.loads(Path(options["paths"]).read_text())
    engine = HyperFlow(options, paths)
    # Separate t2va and fl2va pipelines share nothing but share the same weights on disk: record both.
    dits = {task: pipe.transformer_ref if task == "ref2va" else pipe.transformer for task, pipe in engine.pipes.items()}
    dit = dits[tasks[0]]
    seen = []

    def recorder(*args, **kwargs):
        embedder = next(d for d in dits.values() if d.time_embedder._endpoint_timesteps is not None).time_embedder
        key = tuple(zip(kwargs["timestep"].float().cpu().tolist(),
                        embedder._endpoint_timesteps.float().cpu().tolist()))
        if key not in seen:
            seen.append(key)
        return torch.zeros_like(kwargs["hidden_states"]), torch.zeros_like(kwargs["audio_hidden_states"])

    originals = {task: d.forward for task, d in dits.items()}
    for d in dits.values():
        d.forward = recorder
    try:
        for path in payloads:
            payload = torch.load(path, map_location="cpu", weights_only=False)
            task = payload.get("task", "t2va")
            try:
                engine.generate({"seed": 42, "task": task}, payload, out.parent / ("adaln-" + Path(path).stem),
                                lambda *a: None)
            except Exception as error:  # shapes of zero predictions vs decode etc.: the pairs are already recorded
                print("NOTE", task, type(error).__name__, str(error)[:160], flush=True)
            print("PAIRS", task, len(seen), flush=True)
    finally:
        for task, d in dits.items():
            d.forward = originals[task]
    keys = seen
    dev = torch.device("cuda")
    from diffusers.models.transformers.transformer_minimax_h3 import MiniMaxH3AdaLayerNormModulation
    embedder = dit.time_embedder
    for m in (embedder, dit.time_proj):
        m.to(dev)
    # adaln_proj weights: the original checkpoint, or an FP8 directory converted without --drop-adaln (it keeps
    # them in BF16, unchanged) — the only copy left once convert_fp8.py --consume has deleted the BF16 shards.
    ref = tasks == ["ref2va"]
    fp8 = options["hyperflow"].get("ref_fp8_dir" if ref else "fp8_dir")
    ckpt = Path(fp8) if fp8 else Path(paths["h3_model"]) / ("transformer_ref" if ref else "transformer")
    index = json.loads((ckpt / "diffusion_pytorch_model.safetensors.index.json").read_text())["weight_map"]
    handles, tables, exact = {}, [], True
    with torch.no_grad():
        tembs = []
        for key in keys:
            t = torch.tensor([p[0] for p in key], dtype=torch.float32, device=dev)
            r = torch.tensor([p[1] for p in key], dtype=torch.float32, device=dev)
            with embedder.endpoint_context(r):
                tembs.append(embedder(dit.time_proj(t).to(next(embedder.parameters()).dtype)))
        for i in range(len(dit.transformer_blocks)):
            module = MiniMaxH3AdaLayerNormModulation(tembs[0].shape[-1], dit.transformer_blocks[0].adaln_proj.hidden_size)
            for leaf in ("weight", "bias"):
                name = f"transformer_blocks.{i}.adaln_proj.linear.{leaf}"
                f = index[name]
                if f not in handles:
                    handles[f] = safe_open(str(ckpt / f), "pt")
                getattr(module.linear, leaf).data = handles[f].get_tensor(name).to(dev)
            rows = []
            for temb in tembs:  # one official call per forward's batch → the live kernel and rounding
                out_i = torch.cat(module(temb), dim=-1).reshape(temb.shape[0], -1)
                rows.append(out_i)
                if i == 0:
                    again = nn.functional.linear(nn.functional.silu(temb).to(module.linear.weight.dtype),
                                                 module.linear.weight, module.linear.bias)
                    exact &= torch.equal(again, out_i)
            tables.append(torch.cat(rows).cpu())
    data = {"keys": [list(map(list, k)) for k in keys], "tables": tables}
    torch.save(data, out)
    print("DONE", json.dumps({"forwards": len(keys), "rows": int(tables[0].shape[0]), "blocks": len(tables),
                              "mbytes": round(sum(x.numel() * x.element_size() for x in tables) / 1e6, 1),
                              "block0_replay_exact": bool(exact)}), flush=True)


if __name__ == "__main__":
    main()
