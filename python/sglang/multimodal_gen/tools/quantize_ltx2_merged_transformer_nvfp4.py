"""Quantize an LTX-2 merged BF16 transformer into SGLang NVFP4 format.

This tool is intended for the LTX-2 two-stage path where the distilled LoRA is
merged offline into a Diffusers transformer checkpoint. It uses an existing
SGLang-loadable LTX-2 NVFP4 transformer as a template for:

- tensor names and the FP4/BF16 fallback split
- quantization config and safetensors metadata
- activation input scales

Quantized weights are regenerated from the merged BF16 weights with
``flashinfer.fp4_quantize``. The saved scale-factor tensor uses FlashInfer's
swizzled layout, matching NVIDIA's original LTX-2 FP4 checkpoint format. BF16
fallback weights are copied from the merged checkpoint using the same
runtime-key layout as the template.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Mapping

import torch
from safetensors import safe_open
from safetensors.torch import save_file

INDEX_FILENAMES = (
    "model.safetensors.index.json",
    "diffusion_pytorch_model.safetensors.index.json",
)
RUNTIME_PREFIX = "model.diffusion_model."
TENSOR_MODULE_SUFFIXES = (
    ".weight_scale_2",
    ".weight_scale",
    ".input_scale",
    ".weight",
    ".bias",
)
BLOCK_SIZE = 16
FLOAT4_E2M1_MAX = 6.0

LTX2_REVERSE_PARAM_NAMES_MAPPING = {
    r"^model\.diffusion_model\.(.*)$": r"\1",
    r"^patchify_proj\.(.*)$": r"proj_in.\1",
    r"^adaln_single\.(.*)$": r"time_embed.\1",
    r"^audio_patchify_proj\.(.*)$": r"audio_proj_in.\1",
    r"^audio_adaln_single\.(.*)$": r"audio_time_embed.\1",
    r"(.*)ff\.proj_in\.(.*)$": r"\1ff.net.0.proj.\2",
    r"(.*)ff\.proj_out\.(.*)$": r"\1ff.net.2.\2",
    r"(.*)\.q_norm\.(.*)$": r"\1.norm_q.\2",
    r"(.*)\.k_norm\.(.*)$": r"\1.norm_k.\2",
    r"^av_ca_video_scale_shift_adaln_single\.(.*)$": (
        r"av_cross_attn_video_scale_shift.\1"
    ),
    r"^av_ca_audio_scale_shift_adaln_single\.(.*)$": (
        r"av_cross_attn_audio_scale_shift.\1"
    ),
    r"^av_ca_a2v_gate_adaln_single\.(.*)$": (r"av_cross_attn_video_a2v_gate.\1"),
    r"^av_ca_v2a_gate_adaln_single\.(.*)$": (r"av_cross_attn_audio_v2a_gate.\1"),
    r"(.*)video_a2v_cross_attn_scale_shift_table": (
        r"\1scale_shift_table_a2v_ca_video"
    ),
    r"(.*)audio_a2v_cross_attn_scale_shift_table": (
        r"\1scale_shift_table_a2v_ca_audio"
    ),
}


def _resolve_transformer_dir(path: str) -> str:
    candidate = Path(path).expanduser().resolve()
    if (candidate / "config.json").is_file():
        return str(candidate)
    transformer_dir = candidate / "transformer"
    if (transformer_dir / "config.json").is_file():
        return str(transformer_dir)
    raise FileNotFoundError(f"Could not resolve a transformer directory from: {path}")


def _find_index_file(model_dir: str) -> str | None:
    for filename in INDEX_FILENAMES:
        candidate = os.path.join(model_dir, filename)
        if os.path.isfile(candidate):
            return filename

    matches = sorted(
        filename
        for filename in os.listdir(model_dir)
        if filename.endswith(".safetensors.index.json")
    )
    return matches[0] if matches else None


def _load_weight_map(model_dir: str) -> tuple[dict[str, str], str | None]:
    index_filename = _find_index_file(model_dir)
    if index_filename is not None:
        with open(os.path.join(model_dir, index_filename), encoding="utf-8") as f:
            index_data = json.load(f)
        return dict(index_data["weight_map"]), index_filename

    safetensors_files = sorted(
        filename
        for filename in os.listdir(model_dir)
        if filename.endswith(".safetensors")
    )
    if len(safetensors_files) != 1:
        raise ValueError(
            f"Expected an index file or a single safetensors shard in {model_dir}, "
            f"found {len(safetensors_files)} shard(s)."
        )

    shard_name = safetensors_files[0]
    with safe_open(
        os.path.join(model_dir, shard_name), framework="pt", device="cpu"
    ) as f:
        weight_map = {key: shard_name for key in f.keys()}
    return weight_map, None


def _copy_non_shard_files(source_dir: str, output_dir: str) -> None:
    ignored = set(INDEX_FILENAMES)
    for entry in os.listdir(source_dir):
        if entry.endswith(".safetensors") or entry in ignored:
            continue
        source_path = os.path.join(source_dir, entry)
        output_path = os.path.join(output_dir, entry)
        if os.path.isdir(source_path):
            shutil.copytree(source_path, output_path, dirs_exist_ok=True)
        else:
            shutil.copy2(source_path, output_path)


def _get_param_names_mapping(mapping_dict: Mapping[str, str]):
    def mapping_fn(name: str) -> str:
        max_steps = max(8, len(mapping_dict) * 2)
        applied_patterns: set[str] = set()
        visited_names: set[str] = {name}

        for _ in range(max_steps):
            transformed = False
            for pattern, replacement in mapping_dict.items():
                if pattern in applied_patterns or re.match(pattern, name) is None:
                    continue
                new_name = re.sub(pattern, replacement, name)
                if new_name == name:
                    continue
                name = new_name
                applied_patterns.add(pattern)
                if name in visited_names:
                    transformed = False
                    break
                visited_names.add(name)
                transformed = True
                break
            if not transformed:
                break
        return name

    return mapping_fn


_runtime_to_hf_name = _get_param_names_mapping(LTX2_REVERSE_PARAM_NAMES_MAPPING)


def _base_key_for_template_key(template_key: str) -> str:
    return _runtime_to_hf_name(template_key)


def _module_name_for_tensor(tensor_name: str) -> str:
    for suffix in TENSOR_MODULE_SUFFIXES:
        if tensor_name.endswith(suffix):
            return tensor_name[: -len(suffix)]
    return tensor_name


def _load_all_base_tensors(
    base_dir: str,
    base_weight_map: Mapping[str, str],
    needed_base_keys: set[str],
) -> dict[str, torch.Tensor]:
    missing = sorted(needed_base_keys - set(base_weight_map))
    if missing:
        raise KeyError(
            f"{len(missing)} template tensor(s) could not be mapped to the merged "
            f"BF16 checkpoint. First examples: {missing[:10]}"
        )

    names_by_file: dict[str, list[str]] = defaultdict(list)
    for name in sorted(needed_base_keys):
        names_by_file[base_weight_map[name]].append(name)

    tensors: dict[str, torch.Tensor] = {}
    for filename, names in sorted(names_by_file.items()):
        print(
            f"Loading merged BF16 shard {filename} ({len(names)} tensor(s))",
            flush=True,
        )
        shard_path = os.path.join(base_dir, filename)
        with safe_open(shard_path, framework="pt", device="cpu") as f:
            for name in names:
                tensors[name] = f.get_tensor(name).contiguous()
    return tensors


def _copy_template_metadata(template_dir: str, template_weight_map: Mapping[str, str]):
    if not template_weight_map:
        return {"format": "pt"}
    first_shard = next(iter(template_weight_map.values()))
    with safe_open(
        os.path.join(template_dir, first_shard), framework="pt", device="cpu"
    ) as f:
        metadata = dict(f.metadata() or {})
    metadata.setdefault("format", "pt")
    return metadata


def _load_template_tensors(
    template_dir: str,
    template_weight_map: Mapping[str, str],
    tensor_names: set[str],
) -> dict[str, torch.Tensor]:
    names_by_file: dict[str, list[str]] = defaultdict(list)
    for name in sorted(tensor_names):
        names_by_file[template_weight_map[name]].append(name)

    tensors: dict[str, torch.Tensor] = {}
    for filename, names in sorted(names_by_file.items()):
        shard_path = os.path.join(template_dir, filename)
        with safe_open(shard_path, framework="pt", device="cpu") as f:
            for name in names:
                tensors[name] = f.get_tensor(name).contiguous()
    return tensors


def _discover_template_modules(
    template_dir: str,
    template_weight_map: Mapping[str, str],
) -> tuple[dict[str, torch.dtype], set[str]]:
    modules: dict[str, torch.dtype] = {}
    names_by_file: dict[str, list[str]] = defaultdict(list)
    for name, filename in template_weight_map.items():
        names_by_file[filename].append(name)

    for filename, names in sorted(names_by_file.items()):
        shard_path = os.path.join(template_dir, filename)
        with safe_open(shard_path, framework="pt", device="cpu") as f:
            for name in names:
                if not name.endswith(".weight"):
                    continue
                module_name = name[: -len(".weight")]
                modules[module_name] = f.get_tensor(name).dtype

    direct_copy_names = {
        name
        for name in template_weight_map
        if _module_name_for_tensor(name) not in modules
    }
    return modules, direct_copy_names


def _make_global_scale(weight: torch.Tensor) -> torch.Tensor:
    max_abs = torch.amax(weight.abs()).clamp_min_(1e-6)
    return (torch.finfo(torch.float8_e4m3fn).max * FLOAT4_E2M1_MAX / max_abs).to(
        torch.float32
    )


def _swap_fp4_nibbles(packed: torch.Tensor) -> torch.Tensor:
    return ((packed >> 4) | (packed << 4)).contiguous()


@torch.no_grad()
def _quantize_weight_for_checkpoint(
    weight: torch.Tensor,
    *,
    device: str,
    swap_weight_nibbles: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    import flashinfer

    if weight.dim() != 2:
        raise ValueError(
            f"Expected a 2D linear weight, got shape {tuple(weight.shape)}"
        )
    if int(weight.shape[1]) % BLOCK_SIZE != 0:
        raise ValueError(
            f"Expected input dimension divisible by {BLOCK_SIZE}, got "
            f"shape {tuple(weight.shape)}"
        )

    weight_device = weight.to(device=device, dtype=torch.bfloat16, non_blocking=False)
    weight_global_scale = _make_global_scale(weight_device)
    weight_fp4, weight_scale = flashinfer.fp4_quantize(
        weight_device,
        weight_global_scale,
        is_sf_swizzled_layout=True,
    )
    if weight_scale.dtype == torch.uint8:
        weight_scale = weight_scale.view(torch.float8_e4m3fn)
    if swap_weight_nibbles:
        weight_fp4 = _swap_fp4_nibbles(weight_fp4)
    return (
        weight_fp4.contiguous().cpu(),
        weight_scale.contiguous().cpu(),
        (1.0 / weight_global_scale).to(torch.float32).cpu(),
    )


def _load_swap_weight_nibbles(template_dir: str) -> bool:
    config_path = os.path.join(template_dir, "config.json")
    with open(config_path, encoding="utf-8") as f:
        config = json.load(f)
    quant_config = config.get("quantization_config") or {}
    return bool(quant_config.get("swap_weight_nibbles", True))


def _copy_template_config(template_dir: str, output_dir: Path) -> None:
    shutil.copy2(os.path.join(template_dir, "config.json"), output_dir / "config.json")


def quantize_ltx2_merged_transformer_nvfp4(
    *,
    base_transformer_dir: str,
    template_transformer_dir: str,
    output_dir: str,
    output_filename: str | None = None,
    overwrite: bool = False,
    device: str = "cuda",
) -> dict[str, int | str | bool]:
    base_dir = _resolve_transformer_dir(base_transformer_dir)
    template_dir = _resolve_transformer_dir(template_transformer_dir)
    base_weight_map, _ = _load_weight_map(base_dir)
    template_weight_map, template_index_filename = _load_weight_map(template_dir)
    if template_index_filename is None:
        template_index_filename = "diffusion_pytorch_model.safetensors.index.json"

    output_path = Path(output_dir).expanduser().resolve()
    if output_path.exists():
        if not overwrite:
            raise FileExistsError(
                f"Output directory already exists: {output_path}. "
                "Use --overwrite to replace it."
            )
        shutil.rmtree(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    _copy_non_shard_files(template_dir, str(output_path))
    _copy_template_config(template_dir, output_path)

    output_filename = output_filename or "diffusion_pytorch_model.safetensors"
    template_modules, direct_copy_names = _discover_template_modules(
        template_dir, template_weight_map
    )
    quantized_modules = {
        module_name
        for module_name, weight_dtype in template_modules.items()
        if weight_dtype == torch.uint8
    }
    bf16_modules = set(template_modules) - quantized_modules
    template_direct_tensors = _load_template_tensors(
        template_dir, template_weight_map, direct_copy_names
    )

    needed_base_keys: set[str] = set()
    for module_name in template_modules:
        for suffix in (".weight", ".bias"):
            template_key = f"{module_name}{suffix}"
            base_key = _base_key_for_template_key(template_key)
            if base_key in base_weight_map:
                needed_base_keys.add(base_key)
            elif suffix == ".weight":
                raise KeyError(
                    f"Mapped template weight {template_key!r} to missing base key "
                    f"{base_key!r}."
                )

    base_tensors = _load_all_base_tensors(base_dir, base_weight_map, needed_base_keys)
    template_aux_names: set[str] = set()
    for module_name in quantized_modules:
        template_aux_names.add(f"{module_name}.input_scale")
    template_aux_tensors = _load_template_tensors(
        template_dir,
        template_weight_map,
        {name for name in template_aux_names if name in template_weight_map},
    )

    swap_weight_nibbles = _load_swap_weight_nibbles(template_dir)
    output_tensors: dict[str, torch.Tensor] = {}
    output_tensors.update(template_direct_tensors)
    replaced_bf16_tensors = 0
    quantized_tensor_count = 0

    for index, module_name in enumerate(sorted(template_modules), start=1):
        weight_key = f"{module_name}.weight"
        bias_key = f"{module_name}.bias"
        base_weight_key = _base_key_for_template_key(weight_key)
        base_bias_key = _base_key_for_template_key(bias_key)

        if module_name in quantized_modules:
            print(
                f"Quantizing {index}/{len(template_modules)}: {module_name}",
                flush=True,
            )
            weight, weight_scale, weight_scale_2 = _quantize_weight_for_checkpoint(
                base_tensors[base_weight_key],
                device=device,
                swap_weight_nibbles=swap_weight_nibbles,
            )
            output_tensors[weight_key] = weight
            output_tensors[f"{module_name}.weight_scale"] = weight_scale
            output_tensors[f"{module_name}.weight_scale_2"] = weight_scale_2
            if f"{module_name}.input_scale" in template_aux_tensors:
                output_tensors[f"{module_name}.input_scale"] = template_aux_tensors[
                    f"{module_name}.input_scale"
                ]
            else:
                output_tensors[f"{module_name}.input_scale"] = torch.ones(
                    (), dtype=torch.float32
                )
            quantized_tensor_count += 1
        else:
            output_tensors[weight_key] = base_tensors[base_weight_key]
            replaced_bf16_tensors += 1

        if base_bias_key in base_tensors:
            output_tensors[bias_key] = base_tensors[base_bias_key]
            replaced_bf16_tensors += 1

    metadata = _copy_template_metadata(template_dir, template_weight_map)
    output_file = output_path / output_filename
    save_file(output_tensors, str(output_file), metadata=metadata)

    total_size = sum(
        tensor.element_size() * tensor.numel() for tensor in output_tensors.values()
    )
    updated_weight_map = {name: output_filename for name in sorted(output_tensors)}
    with open(output_path / template_index_filename, "w", encoding="utf-8") as f:
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

    return {
        "base_tensors_loaded": len(base_tensors),
        "bf16_modules": len(bf16_modules),
        "copied_template_tensors": len(template_direct_tensors),
        "output_tensors": len(output_tensors),
        "quantized_modules": len(quantized_modules),
        "replaced_bf16_tensors": replaced_bf16_tensors,
        "swap_weight_nibbles": swap_weight_nibbles,
        "template_tensors": len(template_weight_map),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Quantize a merged LTX-2 BF16 transformer into an SGLang-loadable "
            "NVFP4 transformer using an existing NVFP4 transformer as template."
        )
    )
    parser.add_argument(
        "--base-transformer-dir",
        required=True,
        help="Merged Diffusers transformer directory or parent model directory.",
    )
    parser.add_argument(
        "--template-transformer-dir",
        required=True,
        help="Existing SGLang LTX-2 NVFP4 transformer directory used as template.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory to write the regenerated NVFP4 transformer.",
    )
    parser.add_argument(
        "--output-filename",
        help=(
            "Optional safetensors filename. Defaults to "
            "diffusion_pytorch_model.safetensors."
        ),
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Torch device used for flashinfer.fp4_quantize.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace --output-dir if it already exists.",
    )
    parser.add_argument(
        "--stats-json",
        help="Optional path to write the JSON stats in addition to stdout.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    stats = quantize_ltx2_merged_transformer_nvfp4(
        base_transformer_dir=args.base_transformer_dir,
        template_transformer_dir=args.template_transformer_dir,
        output_dir=args.output_dir,
        output_filename=args.output_filename,
        overwrite=args.overwrite,
        device=args.device,
    )
    stats_text = json.dumps(stats, indent=2, sort_keys=True)
    print(stats_text)
    if args.stats_json:
        stats_path = Path(args.stats_json).expanduser().resolve()
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        stats_path.write_text(stats_text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
