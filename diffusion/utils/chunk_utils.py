# Copyright 2024 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

"""Shared utilities for chunk-wise causal operations.

This module provides utilities for managing temporal chunking in causal video generation,
supporting multiple split strategies (uniform, first_frame, first_plus_one).
"""

from typing import Any, Dict, List, Optional, Tuple


def is_chunk_causal_request(
    chunk_size: Optional[int],
    T_effective: int,
    chunk_index: Optional[List[int]] = None,
) -> bool:
    """Decide whether a layer should run in chunk-causal (vs. fully bidirectional) mode.

    Chunk-causal mode applies when EITHER:
      1. ``chunk_size`` is set and strictly less than ``T_effective`` (the
         standard rule used by training and most inference paths), OR
      2. ``chunk_index`` is explicitly provided by the caller.

    Case (2) is required for the staircase cold-start at AR step 0
    phases 0 / 1, where ``T_effective`` (= ``K + G_eff``, with G_eff in
    {1, 2}) can be smaller than the model's pretrained ``chunk_size``
    (typically 3) but the caller still wants strict frame-causal cond
    boundaries via ``chunk_index = [0, 1]``.  Without this branch, the
    bidirectional fallback would silently leak gen-frame information
    into cond positions.

    The bidirectional fallback should be taken ONLY when both
    ``chunk_size`` is missing/non-restrictive AND ``chunk_index`` is
    not provided — i.e. the caller has not asked for any chunk
    structure at all.

    Args:
        chunk_size: Base chunk size from model config (typically 3 for
            Sana-WM); ``None`` if unset.
        T_effective: Total number of frames after CP all-gather (where
            applicable).  Use the local ``T`` for non-CP paths.
        chunk_index: Optional explicit chunk-start indices.  Anything
            non-``None`` is treated as the caller asking for chunk-
            causal semantics, regardless of ``chunk_size``.

    Returns:
        ``True`` if chunk-causal logic should run, ``False`` if the
        layer should fall back to fully bidirectional behavior.
    """
    if chunk_size is not None and chunk_size < T_effective:
        return True
    if chunk_index is not None:
        return True
    return False


def chunk_index_from_chunk_size(
    T: int,
    chunk_size: int,
    strategy: str = "uniform",
) -> List[int]:
    """Convert chunk_size to chunk_index list with a split strategy.

    Args:
        T: Number of latent frames.
        chunk_size: Base chunk size for the temporal dimension.
        strategy: Chunk split strategy. Supported values:
            - "uniform" (default): uniform chunks with optional remainder
              Example: T=21, chunk_size=4 → [0,4,8,12,16,20] → sizes [4,4,4,4,4,1]
            - "first_frame": first chunk is 1 frame, then uniform chunk_size
              Example: T=21, chunk_size=4 → [0,1,5,9,13,17] → sizes [1,4,4,4,4,4]
            - "first_plus_one": first chunk is chunk_size + 1, then uniform chunk_size
              Example: T=21, chunk_size=4 → [0,5,9,13,17] → sizes [5,4,4,4,4]

    Returns:
        List of chunk start indices (not including the final T).

    Raises:
        ValueError: If chunk_size or T are invalid, or strategy is unknown.
    """
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be > 0, got {chunk_size}.")
    if T <= 0:
        raise ValueError(f"T must be > 0, got {T}.")

    if strategy is None:
        strategy = "uniform"
    strategy = str(strategy).lower()

    if strategy in ("uniform", "default"):
        indices = list(range(0, T, chunk_size))
        # Absorb small remainder into last chunk to avoid degenerate chunks
        # (e.g., causal_conv1d crashes on length=1 sequences).
        if len(indices) > 1 and (T - indices[-1]) < chunk_size:
            indices.pop()
        return indices

    if strategy in ("first_frame", "first_frame_alone", "first_frame_only"):
        if T <= 1:
            return [0]
        indices = [0] + list(range(1, T, chunk_size))
        if len(indices) > 2 and (T - indices[-1]) < chunk_size:
            indices.pop()
        return indices

    if strategy in ("first_plus_one", "first_chunk_plus_one"):
        if T <= chunk_size + 1:
            return [0]
        indices = [0] + list(range(chunk_size + 1, T, chunk_size))
        # Absorb small remainder into last chunk to avoid degenerate chunks
        # (e.g., T_latent=41 with chunk_size=3 → last chunk would be 1 frame,
        # which crashes causal_conv1d). Merge it into the previous chunk instead.
        if len(indices) > 1 and (T - indices[-1]) < chunk_size:
            indices.pop()
        return indices

    raise ValueError(f"Unknown chunk_split_strategy '{strategy}'. Supported: uniform, first_frame, first_plus_one.")


def get_chunk_index_from_config(config: Any, num_frames: Optional[int] = None) -> Optional[List[int]]:
    """Resolve chunk_index from a config, supporting chunk_size and strategy.

    Priority:
      1) config.model.chunk_index (explicit list)
      2) config.model.chunk_size (compute with chunk_split_strategy)
      3) None (no chunking)

    Args:
        config: Config object or dict with a "model" field.
        num_frames: Number of latent frames. Required when using chunk_size.

    Returns:
        Chunk start indices, or None if chunking is disabled.

    Raises:
        ValueError: If chunk_size is set but num_frames is None.
    """
    model = getattr(config, "model", None)
    if model is None:
        return None

    def _get_model_attr(name: str, default: Any) -> Any:
        if hasattr(model, "get"):
            return model.get(name, default)
        if isinstance(model, dict):
            return model.get(name, default)
        return getattr(model, name, default)

    chunk_index = _get_model_attr("chunk_index", None)
    chunk_size = _get_model_attr("chunk_size", None)
    chunk_split_strategy = _get_model_attr("chunk_split_strategy", "uniform")

    if chunk_index is not None:
        if not isinstance(chunk_index, (list, tuple)):
            raise TypeError(f"chunk_index must be a list, got {type(chunk_index).__name__}")
        if len(chunk_index) == 0:
            raise ValueError("chunk_index cannot be empty. Provide at least one chunk boundary.")
        return list(chunk_index)
    if chunk_size is not None:
        if num_frames is None:
            raise ValueError(f"num_frames must be provided when using chunk_size={chunk_size}")
        return chunk_index_from_chunk_size(num_frames, chunk_size, strategy=chunk_split_strategy)
    return None


def compute_chunk_sizes(chunk_index: List[int], T: int) -> List[int]:
    """Compute actual chunk sizes from chunk_index.

    Args:
        chunk_index: List of chunk start indices (e.g., [0, 4, 8, 12]).
        T: Total number of frames.

    Returns:
        List of chunk sizes (e.g., [4, 4, 4, 1] if T=13).

    Example:
        >>> compute_chunk_sizes([0, 4, 8, 12], T=13)
        [4, 4, 4, 1]
        >>> compute_chunk_sizes([0, 1, 5, 9], T=13)
        [1, 4, 4, 4]
    """
    if not chunk_index:
        return []

    # Ensure chunk_index is clean
    chunk_index = [idx for idx in chunk_index if 0 <= idx < T]
    if not chunk_index:
        return []

    # Add T as the final boundary if not present
    if chunk_index[-1] != T:
        chunk_index = chunk_index + [T]

    # Compute sizes
    sizes = [chunk_index[i + 1] - chunk_index[i] for i in range(len(chunk_index) - 1)]
    return sizes


def size1_chunk_position_indices(chunk_index: List[int]) -> List[int]:
    """Return frame-time positions belonging to size-1 (singleton) chunks.

    A size-1 chunk has no intra-chunk lookahead, so the anti-causal
    branch (backward GDN scan and the per-chunk backward conv path)
    contributes nothing for these positions in a chunk-causal layer.
    This helper exposes those positions so downstream code can skip
    the reverse-direction compute (and zero-out the contribution).

    Args:
        chunk_index: Normalized chunk indices, including the trailing
            ``T`` boundary, e.g. ``[0, 1, 2, ..., K, K+G]`` for the
            ``cond_chunk_mode='frame_causal'`` layout.

    Returns:
        List of frame-time positions ``p`` for which ``[p, p+1)`` is a
        chunk of size 1.  Returns ``[]`` when no size-1 chunks exist
        (e.g. uniform ``chunk_size=3`` patterns).

    Examples:
        >>> size1_chunk_position_indices([0, 3, 6, 9])  # uniform size 3
        []
        >>> size1_chunk_position_indices([0, 1, 2, 3, 4, 7])  # frame_causal, K=4, G=3
        [0, 1, 2, 3]
    """
    return [s for s, e in zip(chunk_index[:-1], chunk_index[1:]) if e - s == 1]


def is_uniform_chunking(
    chunk_index: List[int],
    T: int,
    chunk_size: int,
) -> bool:
    """Check if chunk_index represents uniform chunking.

    Returns True if all chunks are equal to chunk_size except possibly the last
    chunk which may be smaller (the remainder). This is the pattern that allows
    safe vectorized padding with: pad_t = chunk_size - (T % chunk_size).

    Uniform patterns (return True):
        - [0,4,8,12,16,20] with T=21, chunk_size=4 → sizes [4,4,4,4,4,1] ✓
        - [0,4,8,12,16] with T=20, chunk_size=4 → sizes [4,4,4,4,4] ✓
        - [0,4,8] with T=10, chunk_size=4 → sizes [4,4,2] ✓

    Non-uniform patterns (return False):
        - [0,1,5,9,13,17] with T=21, chunk_size=4 → sizes [1,4,4,4,4,4] ✗
        - [0,5,9,13,17] with T=21, chunk_size=4 → sizes [5,4,4,4,4] ✗

    Args:
        chunk_index: List of chunk start indices.
        T: Total number of frames.
        chunk_size: Expected uniform chunk size.

    Returns:
        True if chunking is uniform, False otherwise.
    """
    if chunk_size <= 0:
        return False

    # Compute actual chunk sizes
    sizes = compute_chunk_sizes(chunk_index, T)

    if not sizes:
        return True  # Empty is trivially uniform

    # Check that all chunks except possibly the last are equal to chunk_size
    for i, size in enumerate(sizes):
        is_last = i == len(sizes) - 1
        if is_last:
            # Last chunk can be <= chunk_size (remainder)
            if size > chunk_size:
                return False
        else:
            # All other chunks must be exactly chunk_size
            if size != chunk_size:
                return False

    return True


def analyze_chunk_pattern(
    chunk_index: List[int],
    T: int,
    chunk_size: int,
) -> Tuple[str, Dict[str, Any]]:
    """Analyze chunk pattern and return vectorization strategy.

    Detects special patterns that allow hybrid vectorization:
    - uniform: All chunks equal except possibly last (vectorized baseline)
    - first_frame: [1, 4, 4, 4, ...] - first frame alone, then uniform tail
    - first_plus_one: [5, 4, 4, 4, ...] - first chunk+1, then uniform tail
    - arbitrary: Other patterns (no optimization available)

    Args:
        chunk_index: List of chunk start indices (e.g., [0, 4, 8, 12]).
        T: Total number of frames.
        chunk_size: Base chunk size for pattern detection.

    Returns:
        (pattern_type, metadata) where:
            pattern_type: "uniform", "first_frame", "first_plus_one", or "arbitrary"
            metadata: Dict with vectorization hints:
                - vectorizable: bool (True if optimization available)
                - first_chunk_size: int (size of first special chunk)
                - tail_start_index: int (where uniform tail begins in chunk_index)
                - tail_chunk_size: int (uniform size of tail chunks)
                - tail_is_uniform: bool (whether tail is vectorizable)

    Example:
        >>> analyze_chunk_pattern([0, 1, 5, 9, 13, 17], T=21, chunk_size=4)
        ("first_frame", {
            "vectorizable": True,
            "first_chunk_size": 1,
            "tail_start_index": 1,
            "tail_chunk_size": 4,
            "tail_is_uniform": True,
        })
    """
    sizes = compute_chunk_sizes(chunk_index, T)

    if not sizes:
        return "uniform", {"vectorizable": True}

    # Check uniform: all chunks equal to chunk_size except possibly last
    if is_uniform_chunking(chunk_index, T, chunk_size):
        return "uniform", {"vectorizable": True}

    # Check first_frame pattern: [1, 4, 4, 4, ...]
    if sizes[0] == 1:
        # Check if tail (sizes[1:]) is uniform
        tail_is_uniform = all(s == chunk_size for s in sizes[1:-1])
        # Allow last chunk to be <= chunk_size (remainder)
        if len(sizes) > 1:
            tail_is_uniform = tail_is_uniform and (sizes[-1] <= chunk_size)

        if tail_is_uniform:
            return "first_frame", {
                "vectorizable": True,
                "first_chunk_size": 1,
                "tail_start_index": 1,  # Skip first frame
                "tail_chunk_size": chunk_size,
                "tail_is_uniform": True,
            }

    # Check first_plus_one pattern: [chunk_size+1, chunk_size, chunk_size, ...]
    if sizes[0] == chunk_size + 1:
        # Check if tail (sizes[1:]) is uniform
        tail_is_uniform = all(s == chunk_size for s in sizes[1:-1])
        # Allow last chunk to be <= chunk_size (remainder)
        if len(sizes) > 1:
            tail_is_uniform = tail_is_uniform and (sizes[-1] <= chunk_size)

        if tail_is_uniform:
            return "first_plus_one", {
                "vectorizable": True,
                "first_chunk_size": chunk_size + 1,
                "tail_start_index": chunk_size + 1,  # Skip first chunk
                "tail_chunk_size": chunk_size,
                "tail_is_uniform": True,
            }

    # Arbitrary pattern - no vectorization available
    return "arbitrary", {"vectorizable": False}


def normalize_chunk_index(
    chunk_index: Optional[List[int]],
    T: int,
    chunk_size: Optional[int] = None,
    chunk_split_strategy: str = "uniform",
) -> Tuple[List[int], bool]:
    """Normalize chunk_index and detect if uniform.

    This function handles all the complex logic for:
    1. Converting chunk_size + strategy → chunk_index (if needed)
    2. Cleaning and validating chunk_index
    3. Detecting if the result is uniform (safe for vectorized padding)

    Args:
        chunk_index: Optional pre-computed chunk indices.
        T: Total number of frames.
        chunk_size: Chunk size (required if chunk_index is None or for uniformity check).
        chunk_split_strategy: Strategy to use if generating chunk_index from chunk_size.

    Returns:
        (normalized_chunk_index, is_uniform):
            - normalized_chunk_index: Clean list of chunk start indices
            - is_uniform: True if safe to use vectorized path with padding

    Raises:
        ValueError: If required parameters are missing or invalid.
    """
    # Case 1: chunk_index provided explicitly
    if chunk_index is not None:
        normalized_chunk_index = list(chunk_index)

        # Clean up: ensure starts with 0 and ends with T
        if not normalized_chunk_index or normalized_chunk_index[0] != 0:
            normalized_chunk_index = [0] + [idx for idx in normalized_chunk_index if idx > 0]
        normalized_chunk_index = [idx for idx in normalized_chunk_index if idx < T]
        if not normalized_chunk_index:
            normalized_chunk_index = [0]
        if normalized_chunk_index[-1] != T:
            normalized_chunk_index = normalized_chunk_index + [T]

        # Check if uniform (requires chunk_size for comparison)
        if chunk_size is None:
            # Can't verify uniformity without chunk_size, assume non-uniform (safe)
            is_uniform = False
        else:
            is_uniform = is_uniform_chunking(normalized_chunk_index, T, chunk_size)

        return normalized_chunk_index, is_uniform

    # Case 2: Generate chunk_index from chunk_size + strategy
    if chunk_size is None:
        raise ValueError("Either chunk_index or chunk_size must be provided.")

    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be > 0, got {chunk_size}.")

    # Normalize strategy
    strategy = "uniform" if chunk_split_strategy is None else str(chunk_split_strategy).lower()

    # Generate chunk_index
    chunk_index_gen = chunk_index_from_chunk_size(T, chunk_size, strategy=strategy)

    # Add T as final boundary
    if not chunk_index_gen:
        chunk_index_gen = [0]
    if chunk_index_gen[-1] != T:
        chunk_index_gen = chunk_index_gen + [T]

    # Check if uniform
    is_uniform = is_uniform_chunking(chunk_index_gen, T, chunk_size)

    return chunk_index_gen, is_uniform
