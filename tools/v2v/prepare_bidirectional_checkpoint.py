# Copyright 2024 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

"""Prepare a SANA-Video checkpoint for bidirectional V2V training.

The bidirectional V2V model concatenates the noisy target latent and the clean
source latent along the channel dimension. Its patch embedder therefore needs
twice as many input channels as the base SANA-Video model. The V2V GDN blocks
also use separate Q/K/V projections while the base model stores one fused QKV
projection.

This tool performs both deterministic conversions while preserving the
checkpoint's outer wrapper and metadata. Inputs may be local paths or
``hf://<owner>/<repo>/<path>`` URIs.
"""

from __future__ import annotations

import argparse
import os
import re
import tempfile
from collections.abc import Mapping, MutableMapping
from pathlib import Path
from typing import Any

import torch

from sana.tools import resolve_hf_path

X_EMBEDDER_WEIGHT_SUFFIX = "x_embedder.proj.weight"
X_EMBEDDER_BIAS_SUFFIX = "x_embedder.proj.bias"
QKV_PATTERN = re.compile(r"^(?P<prefix>.*\.attn)\.qkv\.(?P<kind>weight|bias)$")


def _copy_mapping(mapping: Mapping[str, Any]) -> MutableMapping[str, Any]:
    """Return a shallow copy without changing an OrderedDict-like type."""

    copied = mapping.copy()
    if not isinstance(copied, MutableMapping):
        copied = dict(mapping)
    return copied


def _find_unique_key(state_dict: Mapping[str, Any], suffix: str, *, required: bool) -> str | None:
    matches = [key for key in state_dict if key == suffix or key.endswith(f".{suffix}")]
    if len(matches) > 1:
        raise ValueError(f"Expected one key ending in {suffix!r}, found {matches}.")
    if not matches:
        if required:
            raise KeyError(f"Checkpoint does not contain a key ending in {suffix!r}.")
        return None
    return matches[0]


def _duplicate_x_embedder(
    state_dict: MutableMapping[str, Any], expected_base_in_channels: int
) -> tuple[str, tuple[int, ...], tuple[int, ...], bool]:
    weight_key = _find_unique_key(state_dict, X_EMBEDDER_WEIGHT_SUFFIX, required=True)
    bias_key = _find_unique_key(state_dict, X_EMBEDDER_BIAS_SUFFIX, required=False)
    weight = state_dict[weight_key]
    if not isinstance(weight, torch.Tensor):
        raise TypeError(f"{weight_key} must be a tensor, got {type(weight).__name__}.")
    if weight.ndim < 2:
        raise ValueError(f"{weight_key} must have at least two dimensions, got shape {tuple(weight.shape)}.")

    expected_base_in_channels = int(expected_base_in_channels)
    if expected_base_in_channels <= 0:
        raise ValueError(f"expected_base_in_channels must be positive, got {expected_base_in_channels}.")
    expected_v2v_in_channels = expected_base_in_channels * 2
    source_shape = tuple(weight.shape)

    if weight.shape[1] == expected_base_in_channels:
        state_dict[weight_key] = torch.cat((weight, weight), dim=1).contiguous()
        duplicated = True
    elif weight.shape[1] == expected_v2v_in_channels:
        duplicated = False
    else:
        raise ValueError(
            f"{weight_key} has {weight.shape[1]} input channels; expected either "
            f"{expected_base_in_channels} (base) or {expected_v2v_in_channels} (already prepared)."
        )

    target_shape = tuple(state_dict[weight_key].shape)
    if target_shape[1] != expected_v2v_in_channels:
        raise AssertionError(
            f"Internal error: {weight_key} conversion produced shape {target_shape}, "
            f"expected input-channel dimension {expected_v2v_in_channels}."
        )
    if target_shape[0] != source_shape[0] or target_shape[2:] != source_shape[2:]:
        raise AssertionError(
            f"Internal error: {weight_key} changed non-input dimensions: {source_shape} -> {target_shape}."
        )

    if bias_key is not None:
        bias = state_dict[bias_key]
        if not isinstance(bias, torch.Tensor):
            raise TypeError(f"{bias_key} must be a tensor, got {type(bias).__name__}.")
        if bias.ndim != 1 or bias.shape[0] != target_shape[0]:
            raise ValueError(
                f"{bias_key} shape {tuple(bias.shape)} is incompatible with {weight_key} output dimension "
                f"{target_shape[0]}."
            )

    return weight_key, source_shape, target_shape, duplicated


def _qkv_destination_keys(qkv_key: str) -> tuple[str, str, str]:
    match = QKV_PATTERN.match(qkv_key)
    if match is None:
        raise ValueError(f"Not a supported QKV key: {qkv_key}")
    prefix = match.group("prefix")
    kind = match.group("kind")
    return tuple(f"{prefix}.{component}.{kind}" for component in ("q", "k", "v"))


def _convert_qkv_projections(state_dict: MutableMapping[str, Any]) -> list[dict[str, Any]]:
    converted: list[dict[str, Any]] = []
    qkv_keys = [key for key in state_dict if QKV_PATTERN.match(key)]

    # Weights and biases are handled independently because the public base
    # checkpoint normally has bias-free QKV projections.
    for qkv_key in qkv_keys:
        tensor = state_dict[qkv_key]
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{qkv_key} must be a tensor, got {type(tensor).__name__}.")
        match = QKV_PATTERN.match(qkv_key)
        kind = match.group("kind")
        expected_ndim = 2 if kind == "weight" else 1
        if tensor.ndim != expected_ndim:
            raise ValueError(
                f"{qkv_key} must be {expected_ndim}D for a fused Linear QKV projection, "
                f"got shape {tuple(tensor.shape)}."
            )
        if tensor.shape[0] % 3 != 0:
            raise ValueError(f"{qkv_key} output dimension {tensor.shape[0]} is not divisible by three.")

        q_key, k_key, v_key = _qkv_destination_keys(qkv_key)
        collisions = [key for key in (q_key, k_key, v_key) if key in state_dict]
        if collisions:
            raise ValueError(
                f"Refusing to overwrite existing split projection keys derived from {qkv_key}: {collisions}."
            )

        q, k, v = tensor.chunk(3, dim=0)
        if kind == "weight":
            hidden_size = q.shape[0]
            if q.shape != k.shape or q.shape != v.shape or q.shape[1] != hidden_size:
                raise ValueError(
                    f"{qkv_key} shape {tuple(tensor.shape)} does not describe three square Q/K/V projections."
                )
            # V2VBiGDNAttention represents Q and K as kernel-size-one Conv1d
            # layers, while V remains a Linear layer.
            state_dict[q_key] = q.unsqueeze(-1).contiguous()
            state_dict[k_key] = k.unsqueeze(-1).contiguous()
            state_dict[v_key] = v.contiguous()
            expected_shapes = ((hidden_size, hidden_size, 1), (hidden_size, hidden_size, 1), (hidden_size, hidden_size))
        else:
            state_dict[q_key] = q.contiguous()
            state_dict[k_key] = k.contiguous()
            state_dict[v_key] = v.contiguous()
            expected_shapes = (tuple(q.shape), tuple(k.shape), tuple(v.shape))

        actual_shapes = tuple(tuple(state_dict[key].shape) for key in (q_key, k_key, v_key))
        if actual_shapes != expected_shapes:
            raise AssertionError(
                f"Internal error: {qkv_key} conversion produced {actual_shapes}, expected {expected_shapes}."
            )

        del state_dict[qkv_key]
        converted.append(
            {
                "source": qkv_key,
                "destinations": (q_key, k_key, v_key),
                "source_shape": tuple(tensor.shape),
                "destination_shapes": actual_shapes,
            }
        )

    return converted


def _validate_split_attention(state_dict: Mapping[str, Any]) -> int:
    q_weight_keys = sorted(
        key for key in state_dict if key.endswith(".attn.q.weight") and isinstance(state_dict[key], torch.Tensor)
    )
    if not q_weight_keys:
        raise ValueError("Prepared checkpoint does not contain any split attention Q projection weights.")

    for q_key in q_weight_keys:
        prefix = q_key[: -len("q.weight")]
        k_key = f"{prefix}k.weight"
        v_key = f"{prefix}v.weight"
        missing = [key for key in (k_key, v_key) if key not in state_dict]
        if missing:
            raise KeyError(f"Incomplete split attention projection for {q_key}: missing {missing}.")

        q, k, v = state_dict[q_key], state_dict[k_key], state_dict[v_key]
        if not all(isinstance(tensor, torch.Tensor) for tensor in (q, k, v)):
            raise TypeError(f"Split attention values for {q_key} must all be tensors.")
        if q.shape != k.shape:
            raise ValueError(f"Q/K shape mismatch for {q_key}: {tuple(q.shape)} vs {tuple(k.shape)}.")
        if q.ndim == 3:
            if q.shape[-1] != 1 or tuple(q.shape[:-1]) != tuple(v.shape):
                raise ValueError(
                    f"GDN split shapes for {q_key} must be Q/K=[out,in,1], V=[out,in]; "
                    f"got Q={tuple(q.shape)}, K={tuple(k.shape)}, V={tuple(v.shape)}."
                )
        elif q.ndim == 2:
            if q.shape != v.shape:
                raise ValueError(
                    f"Softmax split Q/K/V shapes for {q_key} must match; "
                    f"got Q={tuple(q.shape)}, K={tuple(k.shape)}, V={tuple(v.shape)}."
                )
        else:
            raise ValueError(f"Unsupported Q projection shape for {q_key}: {tuple(q.shape)}.")

    remaining_qkv = [key for key in state_dict if QKV_PATTERN.match(key)]
    if remaining_qkv:
        raise AssertionError(f"Prepared checkpoint still contains fused QKV keys: {remaining_qkv}.")
    return len(q_weight_keys)


def prepare_state_dict(
    state_dict: Mapping[str, Any], *, expected_base_in_channels: int = 128, convert_qkv: bool = True
) -> tuple[MutableMapping[str, Any], dict[str, Any]]:
    """Return a converted copy of one model state dict and a conversion report."""

    prepared = _copy_mapping(state_dict)
    weight_key, source_shape, target_shape, duplicated = _duplicate_x_embedder(
        prepared, expected_base_in_channels=expected_base_in_channels
    )
    qkv_conversions = _convert_qkv_projections(prepared) if convert_qkv else []
    split_attention_layers = _validate_split_attention(prepared) if convert_qkv else 0
    report = {
        "x_embedder_key": weight_key,
        "x_embedder_source_shape": source_shape,
        "x_embedder_target_shape": target_shape,
        "x_embedder_duplicated": duplicated,
        "qkv_conversions": qkv_conversions,
        "split_attention_layers": split_attention_layers,
    }
    return prepared, report


def _state_dict_entries(checkpoint: Mapping[str, Any]) -> tuple[list[str], bool]:
    wrapped_entries = [
        key
        for key in ("state_dict", "state_dict_ema", "generator")
        if key in checkpoint and isinstance(checkpoint[key], Mapping)
    ]
    if wrapped_entries:
        if "state_dict" not in wrapped_entries and "generator" not in wrapped_entries:
            raise ValueError("Found state_dict_ema without a primary state_dict or generator entry.")
        return wrapped_entries, True

    if checkpoint and all(isinstance(value, torch.Tensor) for value in checkpoint.values()):
        return [], False
    raise ValueError(
        "Unsupported checkpoint layout. Expected a raw tensor state dict or a wrapper containing "
        "state_dict (optionally state_dict_ema) or generator."
    )


def prepare_checkpoint_object(
    checkpoint: Mapping[str, Any], *, expected_base_in_channels: int = 128, convert_qkv: bool = True
) -> tuple[MutableMapping[str, Any], dict[str, Any]]:
    """Convert a checkpoint object without mutating it or dropping metadata."""

    entries, wrapped = _state_dict_entries(checkpoint)
    reports: dict[str, Any] = {}
    if not wrapped:
        prepared, report = prepare_state_dict(
            checkpoint,
            expected_base_in_channels=expected_base_in_channels,
            convert_qkv=convert_qkv,
        )
        return prepared, {"raw_state_dict": report}

    prepared_checkpoint = _copy_mapping(checkpoint)
    for entry in entries:
        prepared_state_dict, report = prepare_state_dict(
            checkpoint[entry],
            expected_base_in_channels=expected_base_in_channels,
            convert_qkv=convert_qkv,
        )
        prepared_checkpoint[entry] = prepared_state_dict
        reports[entry] = report
    return prepared_checkpoint, reports


def _load_checkpoint(path: str) -> Mapping[str, Any]:
    try:
        return torch.load(path, map_location="cpu", mmap=True, weights_only=False)
    except TypeError:
        # ``mmap`` and ``weights_only`` are unavailable in older PyTorch
        # versions, but conversion remains functionally identical.
        return torch.load(path, map_location="cpu")
    except RuntimeError as exc:
        # Legacy torch.save files cannot be memory mapped. Avoid hiding other
        # deserialization failures behind an unrestricted retry.
        if "mmap can only be used with files saved with" not in str(exc):
            raise
        return torch.load(path, map_location="cpu", weights_only=False)


def prepare_checkpoint_file(
    input_path: str,
    output_path: str,
    *,
    expected_base_in_channels: int = 128,
    convert_qkv: bool = True,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Resolve, convert, validate, and atomically save one checkpoint file."""

    resolved_input = Path(resolve_hf_path(input_path)).expanduser().resolve()
    if not resolved_input.is_file():
        raise FileNotFoundError(f"Checkpoint input is not a file: {resolved_input}")

    output = Path(output_path).expanduser().resolve()
    if output == resolved_input:
        raise ValueError(
            "Input and output paths must differ; in-place checkpoint conversion is intentionally disabled."
        )
    if output.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {output}. Pass --overwrite to replace it.")
    output.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = _load_checkpoint(str(resolved_input))
    if not isinstance(checkpoint, Mapping):
        raise TypeError(f"Checkpoint root must be a mapping, got {type(checkpoint).__name__}.")
    prepared, reports = prepare_checkpoint_object(
        checkpoint,
        expected_base_in_channels=expected_base_in_channels,
        convert_qkv=convert_qkv,
    )

    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=f".{output.name}.", suffix=".tmp", dir=output.parent, delete=False
        ) as tmp:
            temp_path = Path(tmp.name)
        torch.save(prepared, temp_path)
        os.replace(temp_path, output)
    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()

    return {
        "input": str(resolved_input),
        "output": str(output),
        "state_dicts": reports,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Local checkpoint path or hf:// model-repository URI.")
    parser.add_argument("--output", required=True, help="Local path for the converted checkpoint.")
    parser.add_argument(
        "--expected-base-in-channels",
        type=int,
        default=128,
        help="Expected base patch-embedder input channels (default: 128 for LTX-2 latents).",
    )
    parser.add_argument(
        "--skip-qkv-conversion",
        action="store_true",
        help="Only duplicate x_embedder; keep fused attention QKV projections unchanged.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing output file.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = prepare_checkpoint_file(
        args.input,
        args.output,
        expected_base_in_channels=args.expected_base_in_channels,
        convert_qkv=not args.skip_qkv_conversion,
        overwrite=args.overwrite,
    )
    print(f"Prepared bidirectional V2V checkpoint: {report['input']} -> {report['output']}")
    for name, state_report in report["state_dicts"].items():
        action = "duplicated" if state_report["x_embedder_duplicated"] else "already prepared"
        print(
            f"  {name}: x_embedder {action} "
            f"{state_report['x_embedder_source_shape']} -> {state_report['x_embedder_target_shape']}; "
            f"split attention layers={state_report['split_attention_layers']}"
        )


if __name__ == "__main__":
    main()
