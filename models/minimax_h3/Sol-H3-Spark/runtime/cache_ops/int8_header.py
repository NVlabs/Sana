#!/usr/bin/env python3
"""Read and validate Comfy INT8 ConvRot layer metadata without ML imports."""

from __future__ import annotations

import json
import math
import struct
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any


RAW_TRANSFORMER_PREFIX = "model.diffusion_model."
COMFY_QUANT_SUFFIX = ".comfy_quant"
TRANSFORMER_SPEC_PREFIX = "transformer_blocks."
AUDIO_CONNECTOR_SPEC_PREFIX = "audio_embeddings_connector."
VIDEO_CONNECTOR_SPEC_PREFIX = "video_embeddings_connector."
EXPECTED_TRANSFORMER_SPEC_COUNT = 1_344
EXPECTED_AUDIO_CONNECTOR_SPEC_COUNT = 48
EXPECTED_VIDEO_CONNECTOR_SPEC_COUNT = 48
OFFICIAL_DEV_INT8_FILENAME = (
    "ltx-2.5-22b-dev-transformer-comfy-int8-convrot.safetensors"
)
OFFICIAL_GEMMA_INT8_FILENAME = (
    "gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors"
)


@dataclass(frozen=True)
class CheckpointContract:
    sha256: str
    size_bytes: int
    header_bytes: int
    tensor_count: int
    dtype_counts: tuple[tuple[str, int], ...]
    quantized_linear_count: int


OFFICIAL_DEV_INT8_CONTRACT = CheckpointContract(
    sha256="2edbdb4465cd6c3b532cd67a31ddb38a63e97dcad20be3729675e2a4e8caf92b",
    size_bytes=21_504_034_224,
    header_bytes=1_086_120,
    tensor_count=7_229,
    dtype_counts=(("BF16", 2_619), ("F32", 1_730), ("I8", 1_440), ("U8", 1_440)),
    quantized_linear_count=1_440,
)

OFFICIAL_GEMMA_INT8_CONTRACT = CheckpointContract(
    sha256="6ce688a0aa98a5fa36a9f1e6c3f42152a498cc2b53ee8c15674c64244f91487f",
    size_bytes=15_372_969_374,
    header_bytes=156_928,
    tensor_count=1_342,
    dtype_counts=(("BF16", 353), ("F32", 328), ("I8", 328), ("U8", 333)),
    quantized_linear_count=328,
)


@dataclass(frozen=True)
class Int8ConvRotLayerSpec:
    """The on-disk contract required by one quantized LTX Linear."""

    module_name: str
    weight_shape: tuple[int, int]
    scale_shape: tuple[int, ...]
    convrot_groupsize: int


def partition_dev_int8_convrot_specs(
    specs: tuple[Int8ConvRotLayerSpec, ...],
) -> tuple[
    tuple[Int8ConvRotLayerSpec, ...],
    tuple[Int8ConvRotLayerSpec, ...],
]:
    """Split the exact Full Dev schema across its two owning LTX modules.

    Transformer specs retain their checkpoint-relative names. Connector specs
    are renamed to the names created by ``EmbeddingsProcessorConfigurator``.
    The exact per-component cardinalities are part of the official artifact
    contract; unknown, duplicate, or partial schemas fail closed.
    """

    transformer: list[Int8ConvRotLayerSpec] = []
    connectors: list[Int8ConvRotLayerSpec] = []
    source_counts = {
        TRANSFORMER_SPEC_PREFIX: 0,
        AUDIO_CONNECTOR_SPEC_PREFIX: 0,
        VIDEO_CONNECTOR_SPEC_PREFIX: 0,
    }
    seen: set[str] = set()
    connector_targets = {
        AUDIO_CONNECTOR_SPEC_PREFIX: "audio_connector.",
        VIDEO_CONNECTOR_SPEC_PREFIX: "video_connector.",
    }

    for spec in specs:
        if spec.module_name in seen:
            raise ValueError(f"duplicate Full Dev INT8 module {spec.module_name!r}")
        seen.add(spec.module_name)
        if spec.module_name.startswith(TRANSFORMER_SPEC_PREFIX):
            source_counts[TRANSFORMER_SPEC_PREFIX] += 1
            transformer.append(spec)
            continue
        for source_prefix, target_prefix in connector_targets.items():
            if spec.module_name.startswith(source_prefix):
                source_counts[source_prefix] += 1
                connectors.append(
                    Int8ConvRotLayerSpec(
                        module_name=target_prefix
                        + spec.module_name.removeprefix(source_prefix),
                        weight_shape=spec.weight_shape,
                        scale_shape=spec.scale_shape,
                        convrot_groupsize=spec.convrot_groupsize,
                    )
                )
                break
        else:
            raise ValueError(
                f"Full Dev INT8 module has unknown owner {spec.module_name!r}"
            )

    expected_counts = {
        TRANSFORMER_SPEC_PREFIX: EXPECTED_TRANSFORMER_SPEC_COUNT,
        AUDIO_CONNECTOR_SPEC_PREFIX: EXPECTED_AUDIO_CONNECTOR_SPEC_COUNT,
        VIDEO_CONNECTOR_SPEC_PREFIX: EXPECTED_VIDEO_CONNECTOR_SPEC_COUNT,
    }
    if source_counts != expected_counts:
        raise ValueError(
            "official Full Dev INT8 component counts mismatch: "
            f"{source_counts} != {expected_counts}"
        )
    if len(transformer) + len(connectors) != len(specs):
        raise RuntimeError("Full Dev INT8 spec partition lost one or more modules")
    return tuple(transformer), tuple(connectors)


def uncompiled_module_name(module_name: str) -> str:
    """Map torch.compile state-dict names back to stock LTX module names."""

    return ".".join(
        segment for segment in module_name.split(".") if segment != "_orig_mod"
    )


def _product(shape: tuple[int, ...]) -> int:
    return math.prod(shape) if shape else 1


def _read_header(path: Path) -> tuple[dict[str, Any], int, int]:
    if not path.is_file() or path.stat().st_size <= 8:
        raise FileNotFoundError(path)
    with path.open("rb") as handle:
        raw_size = handle.read(8)
        if len(raw_size) != 8:
            raise ValueError(f"truncated safetensors header size: {path}")
        header_size = struct.unpack("<Q", raw_size)[0]
        if header_size <= 0 or header_size > min(path.stat().st_size - 8, 256 << 20):
            raise ValueError(f"invalid safetensors header size {header_size}: {path}")
        raw_header = handle.read(header_size)
    try:
        header = json.loads(raw_header.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid safetensors JSON header: {path}") from exc
    if not isinstance(header, dict):
        raise ValueError(f"safetensors header must be an object: {path}")
    return header, 8 + header_size, header_size


def _tensor_bytes(path: Path, data_start: int, entry: Any, key: str) -> bytes:
    if not isinstance(entry, dict) or entry.get("dtype") != "U8":
        raise ValueError(f"{key} must be a U8 tensor")
    shape = entry.get("shape")
    offsets = entry.get("data_offsets")
    if (
        not isinstance(shape, list)
        or not all(isinstance(value, int) and value >= 0 for value in shape)
        or not isinstance(offsets, list)
        or len(offsets) != 2
        or not all(isinstance(value, int) and value >= 0 for value in offsets)
    ):
        raise ValueError(f"invalid tensor descriptor for {key}")
    start, end = offsets
    if end < start or end - start != _product(tuple(shape)):
        raise ValueError(f"invalid U8 payload extent for {key}")
    if data_start + end > path.stat().st_size:
        raise ValueError(f"tensor payload escapes file for {key}")
    with path.open("rb") as handle:
        handle.seek(data_start + start)
        payload = handle.read(end - start)
    if len(payload) != end - start:
        raise ValueError(f"truncated tensor payload for {key}")
    return payload


def _shape(entry: Any, key: str, dtype: str) -> tuple[int, ...]:
    if not isinstance(entry, dict) or entry.get("dtype") != dtype:
        raise ValueError(f"{key} must have safetensors dtype {dtype}")
    value = entry.get("shape")
    if not isinstance(value, list) or not all(
        isinstance(dimension, int) and dimension > 0 for dimension in value
    ):
        raise ValueError(f"invalid shape for {key}")
    return tuple(value)


def _require_full_dev_filename(path: Path) -> None:
    if path.name != OFFICIAL_DEV_INT8_FILENAME:
        raise ValueError(
            "the offline INT8 cache builder accepts only the official Full Dev checkpoint; "
            f"refusing {path.name!r}"
        )


def _require_gemma_filename(path: Path) -> None:
    if path.name != OFFICIAL_GEMMA_INT8_FILENAME:
        raise ValueError(
            "the offline INT8 cache builder accepts only the official Gemma checkpoint; "
            f"refusing {path.name!r}"
        )


def _validate_global_contract(
    path: Path,
    header: dict[str, Any],
    header_bytes: int,
    contract: CheckpointContract,
) -> None:
    if path.stat().st_size != contract.size_bytes:
        raise ValueError(
            f"official Full Dev checkpoint size mismatch: "
            f"{path.stat().st_size} != {contract.size_bytes}"
        )
    if header_bytes != contract.header_bytes:
        raise ValueError(
            f"official Full Dev safetensors header mismatch: "
            f"{header_bytes} != {contract.header_bytes}"
        )
    tensors = {key: value for key, value in header.items() if key != "__metadata__"}
    if len(tensors) != contract.tensor_count:
        raise ValueError(
            f"official Full Dev tensor count mismatch: "
            f"{len(tensors)} != {contract.tensor_count}"
        )
    dtype_counts = Counter(
        entry.get("dtype") for entry in tensors.values() if isinstance(entry, dict)
    )
    if dtype_counts != Counter(dict(contract.dtype_counts)):
        raise ValueError(
            f"official Full Dev dtype counts mismatch: {dict(dtype_counts)}"
        )

    metadata = header.get("__metadata__")
    required_keys = {"config", "gemma_source_checkpoint", "license", "model_version"}
    if not isinstance(metadata, dict) or set(metadata) != required_keys:
        raise ValueError("official Full Dev __metadata__ keys mismatch")
    if metadata["model_version"] != "2.5.0":
        raise ValueError("official Full Dev model_version must be 2.5.0")
    if not isinstance(metadata["license"], str) or not metadata["license"]:
        raise ValueError("official Full Dev checkpoint has no license text")
    try:
        config = json.loads(metadata["config"])
        gemma_source = json.loads(metadata["gemma_source_checkpoint"])
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError("official Full Dev metadata JSON is invalid") from exc
    if not isinstance(config, dict) or set(config) != {"transformer", "scheduler"}:
        raise ValueError("official Full Dev config top-level keys mismatch")
    transformer = config.get("transformer")
    if not isinstance(transformer, dict) or (
        transformer.get("_class_name") != "AVTransformer3DModel"
        or transformer.get("num_layers") != 48
    ):
        raise ValueError("official Full Dev transformer identity mismatch")
    if gemma_source != {
        "ltx_version": "2.5.0",
        "gemma_version": "gemma4-12b-ltx-v1",
    }:
        raise ValueError("official Full Dev Gemma source identity mismatch")


def read_dev_int8_convrot_specs(
    checkpoint_path: str | Path,
    *,
    contract: CheckpointContract = OFFICIAL_DEV_INT8_CONTRACT,
) -> tuple[Int8ConvRotLayerSpec, ...]:
    """Validate a Full Dev Comfy checkpoint and return its quantized Linears.

    Only the safetensors header and tiny ``*.comfy_quant`` tensors are read.
    Weight payloads are never materialized by this inspection.
    """

    path = Path(checkpoint_path)
    _require_full_dev_filename(path)
    header, data_start, header_bytes = _read_header(path)
    _validate_global_contract(path, header, header_bytes, contract)
    quant_keys = sorted(
        key
        for key in header
        if key != "__metadata__" and key.endswith(COMFY_QUANT_SUFFIX)
    )
    if not quant_keys:
        raise ValueError(f"checkpoint has no per-layer Comfy quant metadata: {path}")
    if len(quant_keys) != contract.quantized_linear_count:
        raise ValueError(
            f"official Full Dev quantized Linear count mismatch: "
            f"{len(quant_keys)} != {contract.quantized_linear_count}"
        )

    specs: list[Int8ConvRotLayerSpec] = []
    for quant_key in quant_keys:
        if not quant_key.startswith(RAW_TRANSFORMER_PREFIX):
            raise ValueError(
                f"quantized layer is outside {RAW_TRANSFORMER_PREFIX!r}: {quant_key}"
            )
        raw_module = quant_key[: -len(COMFY_QUANT_SUFFIX)]
        module_name = raw_module.removeprefix(RAW_TRANSFORMER_PREFIX)
        weight_key = f"{raw_module}.weight"
        scale_key = f"{raw_module}.weight_scale"
        if weight_key not in header or scale_key not in header:
            raise ValueError(f"{quant_key} lacks matching weight/weight_scale tensors")

        payload = _tensor_bytes(path, data_start, header[quant_key], quant_key)
        if header[quant_key].get("shape") != [72]:
            raise ValueError(f"{quant_key} marker shape must be [72]")
        try:
            config = json.loads(payload.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid Comfy quant metadata in {quant_key}") from exc
        expected_marker = {
            "format": "int8_tensorwise",
            "convrot": True,
            "convrot_groupsize": 256,
        }
        if config != expected_marker:
            raise ValueError(f"{quant_key} is not exact INT8 ConvRot group-256 metadata")
        groupsize = 256

        weight_shape = _shape(header[weight_key], weight_key, "I8")
        scale_shape = _shape(header[scale_key], scale_key, "F32")
        if len(weight_shape) != 2:
            raise ValueError(f"INT8 Linear weight must be rank two: {weight_key}")
        out_features, in_features = weight_shape
        if scale_shape != (out_features, 1):
            raise ValueError(
                f"{scale_key} shape {scale_shape} must be ({out_features}, 1)"
            )
        if in_features % groupsize != 0:
            raise ValueError(
                f"{weight_key} input width {in_features} is not divisible by {groupsize}"
            )
        specs.append(
            Int8ConvRotLayerSpec(
                module_name=module_name,
                weight_shape=(out_features, in_features),
                scale_shape=scale_shape,
                convrot_groupsize=groupsize,
            )
        )

    return tuple(specs)


def _expected_gemma_modules() -> frozenset[str]:
    full_attention_layers = frozenset({5, 11, 17, 23, 29, 35, 41, 47})
    modules: set[str] = set()
    for layer in range(48):
        prefix = f"model.layers.{layer}"
        modules.update(
            {
                f"{prefix}.mlp.down_proj",
                f"{prefix}.mlp.gate_proj",
                f"{prefix}.mlp.up_proj",
                f"{prefix}.self_attn.k_proj",
                f"{prefix}.self_attn.o_proj",
                f"{prefix}.self_attn.q_proj",
            }
        )
        if layer not in full_attention_layers:
            modules.add(f"{prefix}.self_attn.v_proj")
    if len(modules) != OFFICIAL_GEMMA_INT8_CONTRACT.quantized_linear_count:
        raise RuntimeError("internal official Gemma INT8 module contract is inconsistent")
    return frozenset(modules)


def _validate_gemma_global_contract(
    path: Path,
    header: dict[str, Any],
    header_bytes: int,
    contract: CheckpointContract,
) -> None:
    _validate_global_contract_shape(path, header, header_bytes, contract, "Gemma")
    metadata = header.get("__metadata__")
    if not isinstance(metadata, dict) or set(metadata) != {"format", "gemma_config"}:
        raise ValueError("official Gemma __metadata__ keys mismatch")
    if metadata["format"] != "pt":
        raise ValueError("official Gemma safetensors format must be pt")
    try:
        config = json.loads(metadata["gemma_config"])
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError("official Gemma config metadata is invalid") from exc
    if not isinstance(config, dict) or (
        config.get("model_type") != "gemma4_unified"
        or config.get("gemma_version") != "gemma4-12b-ltx-v1"
    ):
        raise ValueError("official Gemma model identity mismatch")
    text_config = config.get("text_config")
    if not isinstance(text_config, dict) or text_config.get("num_hidden_layers") != 48:
        raise ValueError("official Gemma text topology mismatch")

    projections = {
        "text_embedding_projection.audio_aggregate_embed.bias": (2048,),
        "text_embedding_projection.audio_aggregate_embed.weight": (2048, 188160),
        "text_embedding_projection.video_aggregate_embed.bias": (4096,),
        "text_embedding_projection.video_aggregate_embed.weight": (4096, 188160),
    }
    for key, expected_shape in projections.items():
        if _shape(header.get(key), key, "BF16") != expected_shape:
            raise ValueError(f"official Gemma projection shape mismatch for {key}")

    expected_assets = {
        "hf_asset__chat_template.jinja",
        "hf_asset__generation_config.json",
        "hf_asset__processor_config.json",
        "hf_asset__tokenizer_config.json",
        "tokenizer_json",
    }
    actual_assets = {
        key
        for key in header
        if key == "tokenizer_json" or key.startswith("hf_asset__")
    }
    if actual_assets != expected_assets:
        raise ValueError("official Gemma embedded asset set mismatch")
    for key in actual_assets:
        _shape(header[key], key, "U8")


def _validate_global_contract_shape(
    path: Path,
    header: dict[str, Any],
    header_bytes: int,
    contract: CheckpointContract,
    component: str,
) -> None:
    if path.stat().st_size != contract.size_bytes:
        raise ValueError(
            f"official {component} checkpoint size mismatch: "
            f"{path.stat().st_size} != {contract.size_bytes}"
        )
    if header_bytes != contract.header_bytes:
        raise ValueError(
            f"official {component} safetensors header mismatch: "
            f"{header_bytes} != {contract.header_bytes}"
        )
    tensors = {key: value for key, value in header.items() if key != "__metadata__"}
    if len(tensors) != contract.tensor_count:
        raise ValueError(
            f"official {component} tensor count mismatch: "
            f"{len(tensors)} != {contract.tensor_count}"
        )
    dtype_counts = Counter(
        entry.get("dtype") for entry in tensors.values() if isinstance(entry, dict)
    )
    if dtype_counts != Counter(dict(contract.dtype_counts)):
        raise ValueError(
            f"official {component} dtype counts mismatch: {dict(dtype_counts)}"
        )


def read_gemma_int8_convrot_specs(
    checkpoint_path: str | Path,
    *,
    contract: CheckpointContract = OFFICIAL_GEMMA_INT8_CONTRACT,
) -> tuple[Int8ConvRotLayerSpec, ...]:
    """Validate the official Gemma 4 INT8 pack and return its 328 Linears."""

    path = Path(checkpoint_path)
    _require_gemma_filename(path)
    header, data_start, header_bytes = _read_header(path)
    _validate_gemma_global_contract(path, header, header_bytes, contract)
    quant_keys = sorted(
        key
        for key in header
        if key != "__metadata__" and key.endswith(COMFY_QUANT_SUFFIX)
    )
    if len(quant_keys) != contract.quantized_linear_count:
        raise ValueError(
            "official Gemma quantized Linear count mismatch: "
            f"{len(quant_keys)} != {contract.quantized_linear_count}"
        )
    raw_modules = frozenset(key[: -len(COMFY_QUANT_SUFFIX)] for key in quant_keys)
    expected_modules = _expected_gemma_modules()
    if raw_modules != expected_modules:
        missing = sorted(expected_modules - raw_modules)
        unexpected = sorted(raw_modules - expected_modules)
        raise ValueError(
            "official Gemma INT8 module set mismatch: "
            f"missing={missing[:4]}, unexpected={unexpected[:4]}"
        )

    specs: list[Int8ConvRotLayerSpec] = []
    for quant_key in quant_keys:
        raw_module = quant_key[: -len(COMFY_QUANT_SUFFIX)]
        weight_key = f"{raw_module}.weight"
        scale_key = f"{raw_module}.weight_scale"
        if weight_key not in header or scale_key not in header:
            raise ValueError(f"{quant_key} lacks matching weight/weight_scale tensors")
        payload = _tensor_bytes(path, data_start, header[quant_key], quant_key)
        if header[quant_key].get("shape") != [72]:
            raise ValueError(f"{quant_key} marker shape must be [72]")
        try:
            config = json.loads(payload.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid Comfy quant metadata in {quant_key}") from exc
        if config != {
            "format": "int8_tensorwise",
            "convrot": True,
            "convrot_groupsize": 256,
        }:
            raise ValueError(f"{quant_key} is not exact INT8 ConvRot group-256 metadata")

        weight_shape = _shape(header[weight_key], weight_key, "I8")
        scale_shape = _shape(header[scale_key], scale_key, "F32")
        if len(weight_shape) != 2:
            raise ValueError(f"INT8 Linear weight must be rank two: {weight_key}")
        out_features, in_features = weight_shape
        if scale_shape != (out_features, 1):
            raise ValueError(
                f"{scale_key} shape {scale_shape} must be ({out_features}, 1)"
            )
        if in_features % 256 != 0:
            raise ValueError(
                f"{weight_key} input width {in_features} is not divisible by 256"
            )
        specs.append(
            Int8ConvRotLayerSpec(
                module_name=raw_module,
                weight_shape=(out_features, in_features),
                scale_shape=scale_shape,
                convrot_groupsize=256,
            )
        )
    return tuple(specs)


__all__ = [
    "CheckpointContract",
    "Int8ConvRotLayerSpec",
    "OFFICIAL_DEV_INT8_CONTRACT",
    "OFFICIAL_GEMMA_INT8_CONTRACT",
    "partition_dev_int8_convrot_specs",
    "read_dev_int8_convrot_specs",
    "read_gemma_int8_convrot_specs",
    "uncompiled_module_name",
]
