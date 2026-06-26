"""Build a selective LTX-2 ModelOpt NVFP4 transformer checkpoint.

This utility quantizes only the linear modules selected by glob patterns and
keeps every other linear in BF16 through the ModelOpt ``ignore`` list. It is
meant for fixed-shape performance experiments where FP4 linear microbenchmarks
first decide which module families are profitable.
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import re
import shutil
import struct
from collections import defaultdict
from pathlib import Path
from typing import Mapping

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

INDEX_FILENAMES = (
    "model.safetensors.index.json",
    "diffusion_pytorch_model.safetensors.index.json",
)
BLOCK_SIZE = 16
FLOAT4_E2M1_MAX = 6.0
FLOAT8_E4M3_MAX = torch.finfo(torch.float8_e4m3fn).max
DEFAULT_INPUT_GLOBAL_SCALE = 512.0
DEFAULT_INCLUDE_PATTERNS = (
    "transformer_blocks.*.ff.net.0.proj",
    "transformer_blocks.*.ff.net.2",
    "transformer_blocks.*.attn1.to_q",
    "transformer_blocks.*.attn1.to_k",
    "transformer_blocks.*.attn1.to_v",
    "transformer_blocks.*.attn1.to_out.0",
    "transformer_blocks.*.attn2.to_q",
    "transformer_blocks.*.attn2.to_k",
    "transformer_blocks.*.attn2.to_v",
    "transformer_blocks.*.attn2.to_out.0",
)
LTX2_EXTRA_RUNTIME_IGNORE_MODULES = (
    "caption_projection.linear_1",
    "caption_projection.linear_2",
    "audio_caption_projection.linear_1",
    "audio_caption_projection.linear_2",
)


LTX2_HF_TO_RUNTIME_PARAM_PATTERNS = (
    (re.compile(r"^model\.diffusion_model\.(.*)$"), r"\1"),
    (re.compile(r"^proj_in\.(.*)$"), r"patchify_proj.\1"),
    (re.compile(r"^time_embed\.(.*)$"), r"adaln_single.\1"),
    (re.compile(r"^audio_proj_in\.(.*)$"), r"audio_patchify_proj.\1"),
    (re.compile(r"^audio_time_embed\.(.*)$"), r"audio_adaln_single.\1"),
    (re.compile(r"(.*)ff\.net\.0\.proj\.(.*)$"), r"\1ff.proj_in.\2"),
    (re.compile(r"(.*)ff\.net\.2\.(.*)$"), r"\1ff.proj_out.\2"),
    (re.compile(r"(.*)\.norm_q\.(.*)$"), r"\1.q_norm.\2"),
    (re.compile(r"(.*)\.norm_k\.(.*)$"), r"\1.k_norm.\2"),
    (
        re.compile(r"^av_cross_attn_video_scale_shift\.(.*)$"),
        r"av_ca_video_scale_shift_adaln_single.\1",
    ),
    (
        re.compile(r"^av_cross_attn_audio_scale_shift\.(.*)$"),
        r"av_ca_audio_scale_shift_adaln_single.\1",
    ),
    (
        re.compile(r"^av_cross_attn_video_a2v_gate\.(.*)$"),
        r"av_ca_a2v_gate_adaln_single.\1",
    ),
    (
        re.compile(r"^av_cross_attn_audio_v2a_gate\.(.*)$"),
        r"av_ca_v2a_gate_adaln_single.\1",
    ),
    (
        re.compile(r"(.*)scale_shift_table_a2v_ca_video$"),
        r"\1video_a2v_cross_attn_scale_shift_table",
    ),
    (
        re.compile(r"(.*)scale_shift_table_a2v_ca_audio$"),
        r"\1audio_a2v_cross_attn_scale_shift_table",
    ),
)


def _resolve_transformer_dir(path: str) -> Path:
    candidate = Path(path).expanduser().resolve()
    if (candidate / "config.json").is_file():
        return candidate
    transformer_dir = candidate / "transformer"
    if (transformer_dir / "config.json").is_file():
        return transformer_dir
    raise FileNotFoundError(f"Could not resolve transformer dir from {path!r}")


def _find_index_file(model_dir: Path) -> str | None:
    for filename in INDEX_FILENAMES:
        if (model_dir / filename).is_file():
            return filename
    matches = sorted(p.name for p in model_dir.glob("*.safetensors.index.json"))
    return matches[0] if matches else None


def _load_weight_map(model_dir: Path) -> tuple[dict[str, str], str | None]:
    index_filename = _find_index_file(model_dir)
    if index_filename is not None:
        with open(model_dir / index_filename, encoding="utf-8") as f:
            return dict(json.load(f)["weight_map"]), index_filename

    shards = sorted(p.name for p in model_dir.glob("*.safetensors"))
    if len(shards) != 1:
        raise ValueError(
            f"Expected index file or one safetensors shard in {model_dir}, got {len(shards)} shard(s)."
        )
    shard_name = shards[0]
    with safe_open(model_dir / shard_name, framework="pt", device="cpu") as f:
        return {key: shard_name for key in f.keys()}, None


def _read_safetensors_header(path: Path) -> dict:
    with open(path, "rb") as f:
        header_len = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(header_len))


def _module_name_for_tensor(name: str) -> str:
    for suffix in (
        ".weight",
        ".bias",
        ".weight_scale",
        ".weight_scale_2",
        ".input_scale",
    ):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


def _copy_non_shard_files(source_dir: Path, output_dir: Path) -> None:
    ignored = set(INDEX_FILENAMES)
    for entry in source_dir.iterdir():
        if entry.name.endswith(".safetensors") or entry.name in ignored:
            continue
        dst = output_dir / entry.name
        if entry.is_dir():
            shutil.copytree(entry, dst, dirs_exist_ok=True)
        else:
            shutil.copy2(entry, dst)


def _is_selected(module_name: str, patterns: list[str]) -> bool:
    return any(fnmatch.fnmatchcase(module_name, pattern) for pattern in patterns)


def _map_ltx2_param_name_for_runtime(param_name: str) -> str:
    for pattern, replacement in LTX2_HF_TO_RUNTIME_PARAM_PATTERNS:
        mapped = pattern.sub(replacement, param_name)
        if mapped != param_name:
            return mapped
    return param_name


def _map_ltx2_module_name_for_runtime(module_name: str) -> str:
    mapped = _map_ltx2_param_name_for_runtime(f"{module_name}.weight")
    return mapped[: -len(".weight")] if mapped.endswith(".weight") else mapped


def _make_global_scale(weight: torch.Tensor) -> torch.Tensor:
    max_abs = torch.amax(weight.abs()).clamp_min_(1e-6)
    return (FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX / max_abs).to(torch.float32)


def _apply_lora_delta_(
    weight: torch.Tensor,
    lora_a: torch.Tensor,
    lora_b: torch.Tensor,
    *,
    strength: float,
) -> torch.Tensor:
    if lora_a.ndim != 2 or lora_b.ndim != 2:
        raise ValueError(
            f"Expected 2D LoRA tensors, got A={tuple(lora_a.shape)} B={tuple(lora_b.shape)}"
        )
    if lora_a.shape[0] != lora_b.shape[1]:
        raise ValueError(
            f"LoRA rank mismatch: A={tuple(lora_a.shape)} B={tuple(lora_b.shape)}"
        )
    if weight.shape != (lora_b.shape[0], lora_a.shape[1]):
        raise ValueError(
            f"LoRA delta shape {(lora_b.shape[0], lora_a.shape[1])} does not match "
            f"weight shape {tuple(weight.shape)}"
        )
    weight.addmm_(lora_b.to(weight), lora_a.to(weight), beta=1.0, alpha=strength)
    return weight


def _lora_keys_for_module(module_name: str, prefix: str) -> tuple[str, str]:
    base = f"{prefix}{module_name}" if prefix else module_name
    return f"{base}.lora_A.weight", f"{base}.lora_B.weight"


def _maybe_merge_lora(
    weight_cpu: torch.Tensor,
    module_name: str,
    *,
    lora_tensors: Mapping[str, torch.Tensor] | None,
    lora_key_prefix: str,
    lora_strength: float,
    device: str,
) -> tuple[torch.Tensor, bool]:
    if not lora_tensors:
        return weight_cpu, False

    candidate_modules = [module_name]
    runtime_module_name = _map_ltx2_module_name_for_runtime(module_name)
    if runtime_module_name not in candidate_modules:
        candidate_modules.append(runtime_module_name)

    for candidate_module in candidate_modules:
        lora_a_key, lora_b_key = _lora_keys_for_module(
            candidate_module, lora_key_prefix
        )
        lora_a = lora_tensors.get(lora_a_key)
        lora_b = lora_tensors.get(lora_b_key)
        if lora_a is None and lora_b is None:
            continue
        if lora_a is None or lora_b is None:
            raise KeyError(
                f"Incomplete LoRA pair for {candidate_module}: {lora_a_key}, {lora_b_key}"
            )
        weight = weight_cpu.to(
            device=device, dtype=torch.bfloat16, non_blocking=False
        ).contiguous()
        _apply_lora_delta_(weight, lora_a, lora_b, strength=lora_strength)
        return weight, True

    return weight_cpu, False


def _quantize_weight_nvfp4(weight_cpu: torch.Tensor, device: str):
    import flashinfer

    weight = weight_cpu.to(
        device=device, dtype=torch.bfloat16, non_blocking=False
    ).contiguous()
    weight_global_scale = _make_global_scale(weight)
    weight_fp4, weight_scale = flashinfer.fp4_quantize(
        weight,
        weight_global_scale,
        is_sf_swizzled_layout=False,
    )
    if weight_scale.dtype == torch.uint8:
        weight_scale = weight_scale.view(torch.float8_e4m3fn)
    weight_scale_2 = (1.0 / weight_global_scale).reshape(1).cpu()
    return (
        weight_fp4.cpu().contiguous(),
        weight_scale.cpu().contiguous(),
        weight_scale_2.contiguous(),
    )


def _build_output_config(
    base_config: Mapping[str, object], ignore_modules: list[str]
) -> dict:
    output_config = json.loads(json.dumps(base_config))
    output_config["quantization_config"] = {
        "quant_method": "modelopt",
        "quant_algo": "NVFP4",
        "quant_type": "NVFP4",
        "group_size": BLOCK_SIZE,
        "ignore": ignore_modules,
        "ignore_is_authoritative": True,
        "swap_weight_nibbles": False,
        "weight_scale_layout": "linear",
    }
    return output_config


def build_selective_nvfp4_transformer(
    *,
    base_transformer_dir: str,
    output_dir: str,
    include_patterns: list[str],
    input_global_scale: float,
    lora_path: str | None,
    lora_key_prefix: str,
    lora_strength: float,
    device: str,
    overwrite: bool,
) -> dict[str, object]:
    base_dir = _resolve_transformer_dir(base_transformer_dir)
    output_path = Path(output_dir).expanduser().resolve()
    if output_path.exists():
        if not overwrite:
            raise FileExistsError(f"Output directory already exists: {output_path}")
        shutil.rmtree(output_path)
    output_path.mkdir(parents=True)
    _copy_non_shard_files(base_dir, output_path)

    with open(base_dir / "config.json", encoding="utf-8") as f:
        base_config = json.load(f)

    weight_map, index_filename = _load_weight_map(base_dir)
    headers_by_file = {
        filename: _read_safetensors_header(base_dir / filename)
        for filename in sorted(set(weight_map.values()))
    }

    linear_modules: dict[str, tuple[int, int]] = {}
    for filename, header in headers_by_file.items():
        del filename
        for name, meta in header.items():
            if name == "__metadata__" or not name.endswith(".weight"):
                continue
            shape = tuple(meta.get("shape", ()))
            if len(shape) == 2 and shape[1] % BLOCK_SIZE == 0:
                linear_modules[_module_name_for_tensor(name)] = shape

    selected_modules = sorted(
        module for module in linear_modules if _is_selected(module, include_patterns)
    )
    unselected_modules = sorted(
        module for module in linear_modules if module not in selected_modules
    )
    ignore_module_set = set(unselected_modules)
    ignore_module_set.update(
        _map_ltx2_module_name_for_runtime(module) for module in unselected_modules
    )
    ignore_modules = sorted(ignore_module_set)

    output_config = _build_output_config(base_config, ignore_modules)
    with open(output_path / "config.json", "w", encoding="utf-8") as f:
        json.dump(output_config, f, indent=2, sort_keys=True)
        f.write("\n")

    quant_config_text = json.dumps(output_config["quantization_config"], sort_keys=True)
    input_scale_tensor = torch.tensor([1.0 / input_global_scale], dtype=torch.float32)
    lora_tensors = load_file(lora_path, device="cpu") if lora_path else None

    weights_by_file: dict[str, list[str]] = defaultdict(list)
    for tensor_name, filename in weight_map.items():
        weights_by_file[filename].append(tensor_name)

    updated_weight_map: dict[str, str] = {}
    total_size = 0
    quantized_weights = 0
    copied_tensors = 0
    lora_merged_weights = 0

    for filename, tensor_names in sorted(weights_by_file.items()):
        print(f"Processing {filename} ({len(tensor_names)} tensor(s))", flush=True)
        shard_path = base_dir / filename
        shard_tensors = load_file(shard_path, device="cpu")
        with safe_open(shard_path, framework="pt", device="cpu") as f:
            metadata = dict(f.metadata() or {})
        metadata.setdefault("format", "pt")
        metadata["_class_name"] = str(
            output_config.get("_class_name", metadata.get("_class_name", ""))
        )
        metadata["config"] = json.dumps(output_config, sort_keys=True)
        metadata["quantization_config"] = quant_config_text
        metadata["_quantization_metadata"] = quant_config_text

        output_tensors: dict[str, torch.Tensor] = {}
        for name in tensor_names:
            tensor = shard_tensors[name]
            module_name = _module_name_for_tensor(name)
            if name.endswith(".weight") and module_name in selected_modules:
                merged_tensor, lora_merged = _maybe_merge_lora(
                    tensor,
                    module_name,
                    lora_tensors=lora_tensors,
                    lora_key_prefix=lora_key_prefix,
                    lora_strength=lora_strength,
                    device=device,
                )
                lora_merged_weights += int(lora_merged)
                weight_fp4, weight_scale, weight_scale_2 = _quantize_weight_nvfp4(
                    merged_tensor, device
                )
                output_tensors[name] = weight_fp4
                output_tensors[f"{module_name}.weight_scale"] = weight_scale
                output_tensors[f"{module_name}.weight_scale_2"] = weight_scale_2
                output_tensors[f"{module_name}.input_scale"] = (
                    input_scale_tensor.clone()
                )
                quantized_weights += 1
            else:
                if name.endswith(".weight"):
                    merged_tensor, lora_merged = _maybe_merge_lora(
                        tensor,
                        module_name,
                        lora_tensors=lora_tensors,
                        lora_key_prefix=lora_key_prefix,
                        lora_strength=lora_strength,
                        device=device,
                    )
                    if lora_merged:
                        lora_merged_weights += 1
                        tensor = merged_tensor.cpu()
                output_tensors[name] = tensor.contiguous()
                copied_tensors += 1

        save_file(output_tensors, output_path / filename, metadata=metadata)
        for name, tensor in output_tensors.items():
            updated_weight_map[name] = filename
            total_size += tensor.element_size() * tensor.numel()
        del shard_tensors, output_tensors
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if index_filename is not None:
        with open(output_path / index_filename, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "metadata": {"total_size": total_size},
                    "weight_map": updated_weight_map,
                },
                f,
                indent=2,
                sort_keys=True,
            )
            f.write("\n")

    stats = {
        "base_transformer_dir": str(base_dir),
        "output_dir": str(output_path),
        "linear_modules": len(linear_modules),
        "selected_modules": len(selected_modules),
        "ignored_modules": len(ignore_modules),
        "quantized_weights": quantized_weights,
        "copied_tensors": copied_tensors,
        "include_patterns": include_patterns,
        "input_global_scale": input_global_scale,
        "lora_key_prefix": lora_key_prefix,
        "lora_merged_weights": lora_merged_weights,
        "lora_path": str(Path(lora_path).expanduser().resolve()) if lora_path else None,
        "lora_strength": lora_strength,
        "total_size": total_size,
    }
    (output_path / "selective_nvfp4_stats.json").write_text(
        json.dumps(stats, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return stats


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-transformer-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--include-pattern", action="append", default=[])
    parser.add_argument(
        "--input-global-scale", type=float, default=DEFAULT_INPUT_GLOBAL_SCALE
    )
    parser.add_argument("--lora-path")
    parser.add_argument("--lora-key-prefix", default="diffusion_model.")
    parser.add_argument("--lora-strength", type=float, default=1.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    include_patterns = args.include_pattern or list(DEFAULT_INCLUDE_PATTERNS)
    stats = build_selective_nvfp4_transformer(
        base_transformer_dir=args.base_transformer_dir,
        output_dir=args.output_dir,
        include_patterns=include_patterns,
        input_global_scale=args.input_global_scale,
        lora_path=args.lora_path,
        lora_key_prefix=args.lora_key_prefix,
        lora_strength=args.lora_strength,
        device=args.device,
        overwrite=args.overwrite,
    )
    print(json.dumps(stats, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
