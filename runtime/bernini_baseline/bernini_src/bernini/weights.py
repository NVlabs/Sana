# Copyright (c) 2026 Bytedance Ltd. and/or its affiliate
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

"""Load Bernini Renderer transformer weights from a safetensors checkpoint.

Accepts a local directory, a ``*.safetensors.index.json`` file, or a Hugging
Face repo id. For each transformer a list of candidate key prefixes is tried
in order (``diff_dec.transformer.`` -> ``transformer.`` -> bare keys); keys may
carry a leading ``ema.`` prefix, and when both an EMA and a non-EMA copy of a
tensor exist the EMA copy is used.
"""

import json
import logging
import os

from safetensors import safe_open

logger = logging.getLogger("bernini.weights")

HIGH_NOISE_PREFIXES = ["diff_dec.transformer.", "transformer.", ""]
LOW_NOISE_PREFIXES = ["diff_dec.transformer_2.", "transformer_2.", ""]


def _resolve_dir(path: str) -> str:
    if os.path.isfile(path) and path.endswith("safetensors.index.json"):
        return os.path.dirname(path)
    if os.path.isdir(path):
        return path
    from huggingface_hub import snapshot_download  # treat as a HF repo id

    return snapshot_download(path, allow_patterns=["*.safetensors", "*.json"])


def _build_weight_map(ckpt_dir: str) -> dict:
    """Map every tensor key to the shard file that holds it."""
    index_files = sorted(f for f in os.listdir(ckpt_dir) if f.endswith("safetensors.index.json"))
    if index_files:
        with open(os.path.join(ckpt_dir, index_files[0])) as f:
            return json.load(f)["weight_map"]

    shard_files = sorted(f for f in os.listdir(ckpt_dir) if f.endswith(".safetensors"))
    if not shard_files:
        raise FileNotFoundError(f"no .safetensors weights found in {ckpt_dir}")
    weight_map = {}
    for fn in shard_files:
        with safe_open(os.path.join(ckpt_dir, fn), framework="pt", device="cpu") as f:
            for key in f.keys():
                weight_map[key] = fn
    return weight_map


def _select_keys(weight_map: dict, prefix: str) -> dict:
    """Map target tensor name -> source key for `prefix`, preferring EMA copies."""
    chosen = {}
    for key in weight_map:
        is_ema = key.startswith("ema.")
        base = key[len("ema.") :] if is_ema else key
        if not base.startswith(prefix):
            continue
        target = base[len(prefix) :]
        if target not in chosen or (is_ema and not chosen[target].startswith("ema.")):
            chosen[target] = key
    return chosen


def load_transformer_state_dict(ckpt_path: str, prefixes: list):
    """Return ``(state_dict, prefix_used)`` for one transformer."""
    ckpt_dir = _resolve_dir(ckpt_path)
    weight_map = _build_weight_map(ckpt_dir)

    for prefix in prefixes:
        chosen = _select_keys(weight_map, prefix)
        if not chosen:
            continue
        shards = {}
        for target, source in chosen.items():
            shards.setdefault(weight_map[source], []).append((source, target))
        state_dict = {}
        for shard_file, items in shards.items():
            with safe_open(os.path.join(ckpt_dir, shard_file), framework="pt", device="cpu") as f:
                for source, target in items:
                    state_dict[target] = f.get_tensor(source)
        return state_dict, prefix

    raise ValueError(f"no weights matching prefixes {prefixes} found in {ckpt_path}")


def load_weights(model, high_noise_ckpt: str, low_noise_ckpt: str):
    """Load the high-noise and low-noise transformer weights into a BerniniRendererModel."""
    high, prefix = load_transformer_state_dict(high_noise_ckpt, HIGH_NOISE_PREFIXES)
    miss, unexpected = model.diff_dec.transformer.load_state_dict(high, strict=False, assign=False)
    logger.info("high-noise: %d tensors via prefix '%s', missing=%d unexpected=%d",
                len(high), prefix, len(miss), len(unexpected))

    low, prefix = load_transformer_state_dict(low_noise_ckpt, LOW_NOISE_PREFIXES)
    miss, unexpected = model.diff_dec.transformer_2.load_state_dict(low, strict=False, assign=False)
    logger.info("low-noise: %d tensors via prefix '%s', missing=%d unexpected=%d",
                len(low), prefix, len(miss), len(unexpected))
