"""Merge an LTX-2 distilled LoRA into a Diffusers transformer checkpoint.

This is intended for building an explicit stage-2 transformer for LTX-2
two-stage inference. The output keeps the original Diffusers shard layout and
HF parameter names, so it can be passed as component_paths.transformer_2 and
then converted by downstream quantization/export tooling.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

INDEX_FILENAMES = (
    "model.safetensors.index.json",
    "diffusion_pytorch_model.safetensors.index.json",
)
LORA_MERGE_CHUNK_BYTES = 32 * 1024 * 1024
LTX2_PARAM_NAMES_MAPPING = {
    r"^model\.diffusion_model\.(.*)$": r"\1",
    r"^proj_in\.(.*)$": r"patchify_proj.\1",
    r"^time_embed\.(.*)$": r"adaln_single.\1",
    r"^audio_proj_in\.(.*)$": r"audio_patchify_proj.\1",
    r"^audio_time_embed\.(.*)$": r"audio_adaln_single.\1",
    r"(.*)ff\.net\.0\.proj\.(.*)$": r"\1ff.proj_in.\2",
    r"(.*)ff\.net\.2\.(.*)$": r"\1ff.proj_out.\2",
    r"(.*)\.norm_q\.(.*)$": r"\1.q_norm.\2",
    r"(.*)\.norm_k\.(.*)$": r"\1.k_norm.\2",
    r"^av_cross_attn_video_scale_shift\.(.*)$": (
        r"av_ca_video_scale_shift_adaln_single.\1"
    ),
    r"^av_cross_attn_audio_scale_shift\.(.*)$": (
        r"av_ca_audio_scale_shift_adaln_single.\1"
    ),
    r"^av_cross_attn_video_a2v_gate\.(.*)$": (r"av_ca_a2v_gate_adaln_single.\1"),
    r"^av_cross_attn_audio_v2a_gate\.(.*)$": (r"av_ca_v2a_gate_adaln_single.\1"),
    r"(.*)scale_shift_table_a2v_ca_video": (
        r"\1video_a2v_cross_attn_scale_shift_table"
    ),
    r"(.*)scale_shift_table_a2v_ca_audio": (
        r"\1audio_a2v_cross_attn_scale_shift_table"
    ),
}
LTX2_REVERSE_PARAM_NAMES_MAPPING = {
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
LTX2_LORA_PARAM_NAMES_MAPPING: dict[str, str] = {}


@dataclass
class LoRAPair:
    base_weight_name: str
    lora_a_name: str | None = None
    lora_b_name: str | None = None
    alpha_name: str | None = None
    lora_a: torch.Tensor | None = None
    lora_b: torch.Tensor | None = None
    alpha: int | None = None

    @property
    def complete(self) -> bool:
        return self.lora_a_name is not None and self.lora_b_name is not None


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
    for entry in os.listdir(source_dir):
        if entry.endswith(".safetensors"):
            continue
        source_path = os.path.join(source_dir, entry)
        output_path = os.path.join(output_dir, entry)
        if os.path.isdir(source_path):
            shutil.copytree(source_path, output_path, dirs_exist_ok=True)
        else:
            shutil.copy2(source_path, output_path)


def _strip_lora_prefix(name: str) -> str:
    return re.sub(r"^(?:model\.)?diffusion_model\.", "", name)


def _parse_lora_key(name: str) -> tuple[str, str] | None:
    name = _strip_lora_prefix(name)
    if name.endswith(".lora_A.weight"):
        return name[: -len(".lora_A.weight")], "A"
    if name.endswith(".lora_B.weight"):
        return name[: -len(".lora_B.weight")], "B"
    if name.endswith(".alpha"):
        return name[: -len(".alpha")], "alpha"
    return None


def _get_param_names_mapping(mapping_dict: Mapping[str, str]):
    def mapping_fn(name: str) -> tuple[str, None, None]:
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
        return name, None, None

    return mapping_fn


def _build_name_mappers():
    hf_to_runtime = _get_param_names_mapping(LTX2_PARAM_NAMES_MAPPING)
    runtime_to_hf = _get_param_names_mapping(LTX2_REVERSE_PARAM_NAMES_MAPPING)
    lora_to_hf = _get_param_names_mapping(LTX2_LORA_PARAM_NAMES_MAPPING)
    return hf_to_runtime, runtime_to_hf, lora_to_hf


def _lora_module_to_base_weight_name(
    raw_module_name: str,
    *,
    hf_to_runtime,
    runtime_to_hf,
    lora_to_hf,
) -> str:
    hf_lora_module_name, _, _ = lora_to_hf(raw_module_name)
    runtime_module_name, _, _ = hf_to_runtime(hf_lora_module_name)
    base_weight_name, _, _ = runtime_to_hf(f"{runtime_module_name}.weight")
    return base_weight_name


def _load_lora_pairs(lora_path: str, *, load_tensors: bool) -> dict[str, LoRAPair]:
    hf_to_runtime, runtime_to_hf, lora_to_hf = _build_name_mappers()
    pairs: dict[str, LoRAPair] = {}

    with safe_open(lora_path, framework="pt", device="cpu") as f:
        for raw_key in f.keys():
            parsed = _parse_lora_key(raw_key)
            if parsed is None:
                continue
            raw_module_name, kind = parsed
            base_weight_name = _lora_module_to_base_weight_name(
                raw_module_name,
                hf_to_runtime=hf_to_runtime,
                runtime_to_hf=runtime_to_hf,
                lora_to_hf=lora_to_hf,
            )
            pair = pairs.setdefault(
                base_weight_name, LoRAPair(base_weight_name=base_weight_name)
            )
            if kind == "A":
                pair.lora_a_name = raw_key
            elif kind == "B":
                pair.lora_b_name = raw_key
            else:
                pair.alpha_name = raw_key

        if not load_tensors:
            return pairs

        for pair in pairs.values():
            if pair.lora_a_name is not None:
                pair.lora_a = f.get_tensor(pair.lora_a_name).contiguous()
            if pair.lora_b_name is not None:
                pair.lora_b = f.get_tensor(pair.lora_b_name).contiguous()
            if pair.alpha_name is not None:
                pair.alpha = int(f.get_tensor(pair.alpha_name).item())

    return pairs


def _chunk_rows_for_merge(
    input_dim: int, *, compute_dtype: torch.dtype, merge_chunk_bytes: int
) -> int:
    element_size = torch.empty((), dtype=compute_dtype).element_size()
    return max(1, merge_chunk_bytes // (int(input_dim) * max(1, element_size)))


@torch.no_grad()
def _merge_lora_pair_into_weight(
    weight: torch.Tensor,
    pair: LoRAPair,
    *,
    strength: float,
    merge_dtype: str,
    merge_chunk_bytes: int,
) -> None:
    if pair.lora_a is None or pair.lora_b is None:
        raise ValueError(f"Incomplete LoRA pair for {pair.base_weight_name}")

    data_2d = weight.reshape(-1, weight.shape[-1]) if weight.dim() > 2 else weight
    lora_a = pair.lora_a
    lora_b = pair.lora_b
    if lora_a.dim() > 2:
        lora_a = lora_a.reshape(-1, lora_a.shape[-1])
    if lora_b.dim() > 2:
        lora_b = lora_b.reshape(-1, lora_b.shape[-1])

    if lora_a.dim() != 2 or lora_b.dim() != 2 or data_2d.dim() != 2:
        raise ValueError(
            f"Expected 2D LoRA/base tensors for {pair.base_weight_name}, got "
            f"A={tuple(lora_a.shape)}, B={tuple(lora_b.shape)}, "
            f"weight={tuple(weight.shape)}"
        )
    expected = (int(data_2d.shape[0]), int(data_2d.shape[1]))
    actual = (int(lora_b.shape[0]), int(lora_a.shape[1]))
    if actual != expected or int(lora_b.shape[1]) != int(lora_a.shape[0]):
        raise ValueError(
            f"LoRA shape mismatch for {pair.base_weight_name}: "
            f"weight={tuple(data_2d.shape)}, A={tuple(lora_a.shape)}, "
            f"B={tuple(lora_b.shape)}"
        )

    rank = int(lora_a.shape[0])
    scale = float(strength)
    if pair.alpha is not None and pair.alpha != rank:
        scale *= float(pair.alpha) / float(rank)

    compute_dtype = torch.float32 if merge_dtype == "float32" else weight.dtype
    lora_a = lora_a.to(dtype=compute_dtype)
    lora_b = lora_b.to(dtype=compute_dtype)

    chunk_rows = _chunk_rows_for_merge(
        data_2d.shape[-1],
        compute_dtype=compute_dtype,
        merge_chunk_bytes=merge_chunk_bytes,
    )
    for start in range(0, int(lora_b.shape[0]), chunk_rows):
        end = min(start + chunk_rows, int(lora_b.shape[0]))
        delta = lora_b[start:end] @ lora_a
        data_2d[start:end].add_(delta.to(dtype=data_2d.dtype), alpha=scale)


def _validate_lora_pairs(
    pairs: Mapping[str, LoRAPair],
    base_weight_map: Mapping[str, str],
    *,
    allow_missing: bool,
) -> tuple[list[str], list[str]]:
    incomplete = sorted(name for name, pair in pairs.items() if not pair.complete)
    missing = sorted(name for name in pairs if name not in base_weight_map)
    errors = []
    if incomplete:
        errors.append(
            f"{len(incomplete)} LoRA target(s) are missing A or B weights; "
            f"first examples: {incomplete[:5]}"
        )
    if missing and not allow_missing:
        errors.append(
            f"{len(missing)} LoRA target(s) are not present in the base checkpoint; "
            f"first examples: {missing[:5]}"
        )
    if errors:
        raise ValueError("; ".join(errors))
    return incomplete, missing


def merge_ltx2_lora_transformer(
    *,
    base_transformer_dir: str,
    lora_path: str,
    output_dir: str,
    strength: float = 1.0,
    merge_dtype: str = "base",
    merge_chunk_bytes: int = LORA_MERGE_CHUNK_BYTES,
    overwrite: bool = False,
    allow_missing: bool = False,
    dry_run: bool = False,
) -> dict[str, int | float | str]:
    if merge_dtype not in ("base", "float32"):
        raise ValueError(f"Unsupported merge_dtype={merge_dtype!r}")

    base_dir = _resolve_transformer_dir(base_transformer_dir)
    base_weight_map, index_filename = _load_weight_map(base_dir)
    lora_pairs = _load_lora_pairs(
        str(Path(lora_path).expanduser().resolve()),
        load_tensors=not dry_run,
    )
    incomplete, missing = _validate_lora_pairs(
        lora_pairs, base_weight_map, allow_missing=allow_missing
    )

    output_path = Path(output_dir).expanduser().resolve()
    if dry_run:
        return {
            "base_tensors": len(base_weight_map),
            "complete_lora_targets": sum(pair.complete for pair in lora_pairs.values()),
            "incomplete_lora_targets": len(incomplete),
            "missing_lora_targets": len(missing),
            "merge_dtype": merge_dtype,
            "strength": float(strength),
        }

    if output_path.exists():
        if not overwrite:
            raise FileExistsError(
                f"Output directory already exists: {output_path}. "
                "Use --overwrite to replace it."
            )
        shutil.rmtree(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    _copy_non_shard_files(base_dir, str(output_path))

    pairs_by_file: dict[str, list[LoRAPair]] = defaultdict(list)
    for name, pair in lora_pairs.items():
        if name in base_weight_map and pair.complete:
            pairs_by_file[base_weight_map[name]].append(pair)

    merged_count = 0
    copied_shard_count = 0
    shard_filenames = sorted(set(base_weight_map.values()))
    for shard_index, filename in enumerate(shard_filenames, start=1):
        print(
            f"Merging shard {shard_index}/{len(shard_filenames)}: "
            f"{filename} ({len(pairs_by_file.get(filename, []))} LoRA target(s))",
            flush=True,
        )
        shard_path = os.path.join(base_dir, filename)
        shard_tensors = load_file(shard_path, device="cpu")
        with safe_open(shard_path, framework="pt", device="cpu") as f:
            metadata = dict(f.metadata() or {})
        metadata.setdefault("format", "pt")

        for pair in pairs_by_file.get(filename, []):
            _merge_lora_pair_into_weight(
                shard_tensors[pair.base_weight_name],
                pair,
                strength=strength,
                merge_dtype=merge_dtype,
                merge_chunk_bytes=merge_chunk_bytes,
            )
            merged_count += 1

        save_file(shard_tensors, str(output_path / filename), metadata=metadata)
        copied_shard_count += 1

    return {
        "base_tensors": len(base_weight_map),
        "copied_shards": copied_shard_count,
        "complete_lora_targets": sum(pair.complete for pair in lora_pairs.values()),
        "incomplete_lora_targets": len(incomplete),
        "merged_lora_targets": merged_count,
        "missing_lora_targets": len(missing),
        "index_file": index_filename or "",
        "merge_dtype": merge_dtype,
        "strength": float(strength),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge an LTX-2 distilled LoRA into a BF16 transformer checkpoint."
    )
    parser.add_argument(
        "--base-transformer-dir",
        required=True,
        help="Original Diffusers transformer directory or parent model directory.",
    )
    parser.add_argument(
        "--lora-path",
        required=True,
        help="Distilled LoRA safetensors file.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory to write the merged transformer checkpoint.",
    )
    parser.add_argument(
        "--strength",
        type=float,
        default=1.0,
        help="LoRA strength to merge. LTX2 two-stage defaults to 1.0.",
    )
    parser.add_argument(
        "--merge-dtype",
        choices=["base", "float32"],
        default="base",
        help=(
            "Matmul dtype for the merge. 'base' matches runtime distilled-LoRA "
            "merge semantics; 'float32' is available for diagnostics."
        ),
    )
    parser.add_argument(
        "--merge-chunk-bytes",
        type=int,
        default=LORA_MERGE_CHUNK_BYTES,
        help="Approximate maximum temporary delta bytes per merged tensor chunk.",
    )
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="Skip LoRA targets that are absent from the base checkpoint.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate mappings and print stats without writing output shards.",
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
    stats = merge_ltx2_lora_transformer(
        base_transformer_dir=args.base_transformer_dir,
        lora_path=args.lora_path,
        output_dir=args.output_dir,
        strength=args.strength,
        merge_dtype=args.merge_dtype,
        merge_chunk_bytes=args.merge_chunk_bytes,
        overwrite=args.overwrite,
        allow_missing=args.allow_missing,
        dry_run=args.dry_run,
    )
    stats_text = json.dumps(stats, indent=2, sort_keys=True)
    print(stats_text)
    if args.stats_json:
        stats_path = Path(args.stats_json).expanduser().resolve()
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        stats_path.write_text(stats_text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
