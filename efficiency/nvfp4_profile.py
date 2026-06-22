"""Pure profile policies for NVFP4 layer selection.

The runtime can only consume compact layer sets, but the algorithm boundary is
the profile-to-layer-set decision.  This module deliberately knows nothing
about Cosmos, LTX, or any model class names; it operates on integer layer ids
and scalar profile scores supplied by a caller or manifest.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from typing import Mapping


@dataclass(frozen=True)
class NVFP4ProfileSelection:
    profiled_layers: frozenset[int]
    dense_layers: frozenset[int]
    keep_count: int
    cutoff_score: float | None


def parse_int_ranges(value: str) -> frozenset[int]:
    """Parse comma-separated integer/range syntax such as ``0-1,30,31``."""

    values: set[int] = set()
    for raw_part in str(value or "").split(","):
        part = raw_part.strip()
        if not part or part.lower() in {"none", "off", "false"}:
            continue
        if "-" in part:
            start_text, end_text = part.split("-", 1)
            try:
                start = int(start_text)
                end = int(end_text)
            except ValueError:
                continue
            lo, hi = sorted((start, end))
            values.update(range(lo, hi + 1))
            continue
        try:
            values.add(int(part))
        except ValueError:
            continue
    return frozenset(index for index in values if index >= 0)


def format_int_ranges(values: set[int] | frozenset[int]) -> str:
    """Format a set of layer ids as stable compact comma/range syntax."""

    ordered = sorted(index for index in values if index >= 0)
    if not ordered:
        return ""

    ranges: list[str] = []
    start = prev = ordered[0]
    for value in ordered[1:]:
        if value == prev + 1:
            prev = value
            continue
        ranges.append(f"{start}" if start == prev else f"{start}-{prev}")
        start = prev = value
    ranges.append(f"{start}" if start == prev else f"{start}-{prev}")
    return ",".join(ranges)


def parse_layer_score_spec(value: str | Mapping[int, float]) -> dict[int, float]:
    """Parse layer scores from a compact manifest/env string.

    Accepted forms:
      - ``"0:0.1,1:0.2,2-5:1.0"``
      - ``"0=0.1,1=0.2"``
      - a mapping of layer id to scalar score
    """

    if isinstance(value, Mapping):
        parsed: dict[int, float] = {}
        for raw_layer, raw_score in value.items():
            try:
                layer = int(raw_layer)
                score = float(raw_score)
            except (TypeError, ValueError):
                continue
            if layer >= 0:
                parsed[layer] = score
        return parsed

    parsed: dict[int, float] = {}
    for raw_part in str(value or "").split(","):
        part = raw_part.strip()
        if not part:
            continue
        if ":" in part:
            layer_text, score_text = part.split(":", 1)
        elif "=" in part:
            layer_text, score_text = part.split("=", 1)
        else:
            continue
        try:
            score = float(score_text.strip())
        except ValueError:
            continue
        for layer in parse_int_ranges(layer_text):
            parsed[layer] = score
    return parsed


def select_profiled_nvfp4_layers(
    layer_scores: str | Mapping[int, float],
    *,
    keep_ratio: float = 0.0,
    keep_count: int = 0,
    min_score: float | None = None,
    guard_unselected: bool = True,
) -> NVFP4ProfileSelection:
    """Select layers for NVFP4 from profile scores.

    Higher score means "more worth quantizing" according to the profile source,
    typically latency contribution discounted by measured or estimated quality
    risk.  Non-selected profiled layers can be returned as dense guards so the
    runtime has an explicit BF16 fallback set.
    """

    scores = parse_layer_score_spec(layer_scores)
    if not scores:
        return NVFP4ProfileSelection(frozenset(), frozenset(), 0, None)

    if min_score is not None:
        selected = {layer for layer, score in scores.items() if score >= min_score}
    else:
        if keep_count > 0:
            count = min(len(scores), int(keep_count))
        elif keep_ratio > 0.0:
            count = max(1, min(len(scores), ceil(len(scores) * float(keep_ratio))))
        else:
            count = len(scores)
        ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
        selected = {layer for layer, _score in ranked[:count]}

    selected = {layer for layer in selected if layer >= 0}
    dense = set(scores) - selected if guard_unselected else set()
    cutoff = min((scores[layer] for layer in selected), default=None)
    return NVFP4ProfileSelection(
        profiled_layers=frozenset(selected),
        dense_layers=frozenset(dense),
        keep_count=len(selected),
        cutoff_score=cutoff,
    )
