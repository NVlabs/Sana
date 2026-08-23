"""Controlled 1080p extension for the pinned SGLang MiniMax-H3 runtime.

SGLang commit 12eadf86f12aec2e6f81a6e38b61b964a4c6b529 intentionally
admits only a 768-pixel short edge.  This overlay leaves that public path
unchanged and adds exactly one benchmark-only target: a semantic 1920x1080
request resolved onto MiniMax-H3's legal 32-pixel grid as 1920x1088.

The patched upstream symbols are:

* ``request_validation._validate_target`` (the public 768-only admission gate)
* ``resolved_plan._validate_base_short_edge`` (the second 768-only gate)
* ``resolved_plan.minimax_h3_resolve_spatial_shape`` (the 768p pixel budget)

No model, scheduler, attention, or denoising implementation is replaced.
"""

from __future__ import annotations

from collections.abc import Mapping
import math
from typing import Any, Callable


PINNED_SGLANG_COMMIT = "12eadf86f12aec2e6f81a6e38b61b964a4c6b529"
UPSTREAM_SHORT_EDGE = 768
UPSTREAM_MAX_PIXELS = 768 * 1344
EXTENDED_SHORT_EDGE = 1080
EXTENDED_MAX_PIXELS = 1080 * 1920
EXPECTED_INTERNAL_WIDTH = 1920
EXPECTED_INTERNAL_HEIGHT = 1088


def _nearest_multiple(value: float, multiple: int) -> int:
    return max(multiple, int(round(float(value) / multiple)) * multiple)


def resolve_1080_spatial_shape(
    *,
    width: int | float,
    height: int | float,
    multiple: int = 32,
    min_aspect_ratio: float = 1.0 / 4.0,
    max_aspect_ratio: float = 4.0,
) -> dict[str, Any]:
    """Mirror SGLang ``adapt_shape_v1`` with the 1080p pixel budget."""

    try:
        source_width = float(width)
        source_height = float(height)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "shape width and height must be positive finite numbers"
        ) from exc
    if (
        not math.isfinite(source_width)
        or not math.isfinite(source_height)
        or source_width <= 0.0
        or source_height <= 0.0
    ):
        raise ValueError("shape width and height must be positive finite numbers")

    ratio = source_width / source_height
    if not min_aspect_ratio <= ratio <= max_aspect_ratio:
        raise ValueError(
            "adapt_shape_v1 ratio must be within the inclusive range "
            f"1:4 to 4:1, got {source_width:g}:{source_height:g}"
        )

    if ratio >= 1.0:
        nominal_width = float(EXTENDED_SHORT_EDGE) * ratio
        nominal_height = float(EXTENDED_SHORT_EDGE)
    else:
        nominal_width = float(EXTENDED_SHORT_EDGE)
        nominal_height = float(EXTENDED_SHORT_EDGE) / ratio
    nominal_area = nominal_width * nominal_height
    if nominal_area > EXTENDED_MAX_PIXELS:
        size_mode = "area"
        scale = math.sqrt(float(EXTENDED_MAX_PIXELS) / nominal_area)
        nominal_width *= scale
        nominal_height *= scale
    else:
        size_mode = "short_edge"

    resolved_width = _nearest_multiple(nominal_width, multiple)
    resolved_height = _nearest_multiple(nominal_height, multiple)
    return {
        "geometry": "resolved_v2",
        "shape_policy_version": "adapt_shape_v1+1080p_overlay_v1",
        "base_short_edge": EXTENDED_SHORT_EDGE,
        "effective_short_edge": min(resolved_width, resolved_height),
        "size_mode": size_mode,
        "max_pixels": EXTENDED_MAX_PIXELS,
        "multiple": multiple,
        "rounding": "nearest",
        "width": resolved_width,
        "height": resolved_height,
    }


def install_1080p_overlay(short_edge: int) -> dict[str, Any]:
    """Install the process-local extension, failing closed on source drift.

    A 768 request remains entirely upstream.  A 1080 request gets the narrow
    admission/resolver override above.  Any other value is rejected.
    """

    if isinstance(short_edge, bool) or not isinstance(short_edge, int):
        raise ValueError("MiniMax-H3 short edge must be an integer")
    if short_edge == UPSTREAM_SHORT_EDGE:
        return {
            "installed": False,
            "requested_short_edge": short_edge,
            "reason": "upstream_768p_path",
        }
    if short_edge != EXTENDED_SHORT_EDGE:
        raise ValueError(
            "the controlled SGLang overlay admits only short_edge 768 or 1080, "
            f"got {short_edge}"
        )

    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3 import (  # noqa: E501
        prequeue,
        request_validation,
        resolved_plan,
    )

    marker = getattr(request_validation, "_h3_1080p_overlay", None)
    if marker is not None:
        if marker.get("requested_short_edge") != short_edge:
            raise RuntimeError(f"a different MiniMax-H3 overlay is active: {marker}")
        return dict(marker)

    required = {
        "request_validation._validate_target": getattr(
            request_validation, "_validate_target", None
        ),
        "resolved_plan._validate_base_short_edge": getattr(
            resolved_plan, "_validate_base_short_edge", None
        ),
        "resolved_plan.minimax_h3_resolve_spatial_shape": getattr(
            resolved_plan, "minimax_h3_resolve_spatial_shape", None
        ),
    }
    missing = [name for name, symbol in required.items() if not callable(symbol)]
    if missing:
        raise RuntimeError(f"pinned SGLang symbols are unavailable: {missing}")
    if int(getattr(resolved_plan, "MINIMAX_H3_BASE_SHORT_EDGE", -1)) != 768:
        raise RuntimeError("unexpected SGLang MiniMax-H3 base short edge")
    if int(getattr(resolved_plan, "MINIMAX_H3_MAX_PIXELS", -1)) != UPSTREAM_MAX_PIXELS:
        raise RuntimeError("unexpected SGLang MiniMax-H3 pixel budget")
    if int(getattr(resolved_plan, "MINIMAX_H3_CANVAS_MULTIPLE", -1)) != 32:
        raise RuntimeError("unexpected SGLang MiniMax-H3 canvas multiple")

    original_validate_target: Callable[..., dict[str, Any]] = required[
        "request_validation._validate_target"
    ]
    original_validate_short_edge: Callable[[Any], int] = required[
        "resolved_plan._validate_base_short_edge"
    ]
    original_resolve_shape: Callable[..., dict[str, Any]] = required[
        "resolved_plan.minimax_h3_resolve_spatial_shape"
    ]

    def validate_target_with_1080(target: Any, *, profile: Any) -> dict[str, Any]:
        if not isinstance(target, Mapping) or target.get("short_edge") != short_edge:
            return original_validate_target(target, profile=profile)
        # Reuse every upstream validation rule except its literal 768 check,
        # then restore the requested semantic edge in the canonical payload.
        legacy_target = dict(target)
        legacy_target["short_edge"] = UPSTREAM_SHORT_EDGE
        normalized = original_validate_target(legacy_target, profile=profile)
        normalized["short_edge"] = short_edge
        return normalized

    def validate_short_edge_with_1080(value: Any) -> int:
        if not isinstance(value, bool) and isinstance(value, int) and value == short_edge:
            return short_edge
        return original_validate_short_edge(value)

    def resolve_shape_with_1080(
        *,
        width: int | float,
        height: int | float,
        base_short_edge: int = UPSTREAM_SHORT_EDGE,
    ) -> dict[str, Any]:
        if base_short_edge != short_edge:
            return original_resolve_shape(
                width=width,
                height=height,
                base_short_edge=base_short_edge,
            )
        validate_short_edge_with_1080(base_short_edge)
        return resolve_1080_spatial_shape(
            width=width,
            height=height,
            multiple=int(resolved_plan.MINIMAX_H3_CANVAS_MULTIPLE),
            min_aspect_ratio=float(resolved_plan.MINIMAX_H3_MIN_ASPECT_RATIO),
            max_aspect_ratio=float(resolved_plan.MINIMAX_H3_MAX_ASPECT_RATIO),
        )

    request_validation._validate_target = validate_target_with_1080
    resolved_plan._validate_base_short_edge = validate_short_edge_with_1080
    resolved_plan.minimax_h3_resolve_spatial_shape = resolve_shape_with_1080
    # ``prequeue`` imports the resolver by value. T2VA does not take its
    # deferred branch, but updating the alias keeps this process internally
    # consistent and avoids a future silent 768p fallback.
    prequeue.minimax_h3_resolve_spatial_shape = resolve_shape_with_1080

    resolved = resolve_shape_with_1080(
        width=16,
        height=9,
        base_short_edge=short_edge,
    )
    if (
        int(resolved["width"]),
        int(resolved["height"]),
    ) != (EXPECTED_INTERNAL_WIDTH, EXPECTED_INTERNAL_HEIGHT):
        raise RuntimeError(f"1080p overlay resolved an unexpected canvas: {resolved}")

    marker = {
        "installed": True,
        "name": "sglang_minimax_h3_1080p_overlay_v1",
        "pinned_sglang_commit": PINNED_SGLANG_COMMIT,
        "requested_short_edge": short_edge,
        "internal_width": int(resolved["width"]),
        "internal_height": int(resolved["height"]),
        "max_pixels": EXTENDED_MAX_PIXELS,
        "canvas_multiple": 32,
        "upstream_768_path_preserved": True,
    }
    request_validation._h3_1080p_overlay = marker
    return dict(marker)


__all__ = [
    "EXPECTED_INTERNAL_HEIGHT",
    "EXPECTED_INTERNAL_WIDTH",
    "EXTENDED_SHORT_EDGE",
    "PINNED_SGLANG_COMMIT",
    "install_1080p_overlay",
    "resolve_1080_spatial_shape",
]
