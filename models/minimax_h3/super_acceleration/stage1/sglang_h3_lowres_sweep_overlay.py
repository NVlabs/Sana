"""Fixed low-resolution canvases for the pinned SGLang MiniMax-H3 runtime.

The upstream runtime admits ``short_edge=768`` only.  Student requests keep the
reviewed ``short_edge=512`` routing token, while this process-local overlay
resolves that token to one fixed landscape canvas selected before the process
starts.  The default 896x512 canvas remains equivalent to the original narrow
512p overlay.
"""

from __future__ import annotations

from collections.abc import Mapping
import math
from typing import Any, Callable


PINNED_SGLANG_COMMIT = "12eadf86f12aec2e6f81a6e38b61b964a4c6b529"
UPSTREAM_SHORT_EDGE = 768
UPSTREAM_MAX_PIXELS = 768 * 1344
STUDENT_REQUEST_SHORT_EDGE = 512
DEFAULT_WIDTH = 896
DEFAULT_HEIGHT = 512
MAX_WIDTH = DEFAULT_WIDTH
MAX_HEIGHT = DEFAULT_HEIGHT
MAX_PIXELS = DEFAULT_WIDTH * DEFAULT_HEIGHT
CANVAS_MULTIPLE = 32
H3_MIN_ASPECT_RATIO = 1.0 / 4.0
H3_MAX_ASPECT_RATIO = 4.0


def validate_lowres_canvas(
    width: int,
    height: int,
    *,
    multiple: int = CANVAS_MULTIPLE,
    min_aspect_ratio: float = H3_MIN_ASPECT_RATIO,
    max_aspect_ratio: float = H3_MAX_ASPECT_RATIO,
) -> dict[str, Any]:
    """Validate and describe one fixed low-resolution landscape canvas."""

    if isinstance(width, bool) or not isinstance(width, int):
        raise ValueError("MiniMax-H3 student width must be an integer")
    if isinstance(height, bool) or not isinstance(height, int):
        raise ValueError("MiniMax-H3 student height must be an integer")
    if width <= 0 or height <= 0:
        raise ValueError("MiniMax-H3 student width and height must be positive")
    if width % multiple != 0 or height % multiple != 0:
        raise ValueError(
            f"MiniMax-H3 student width and height must be multiples of {multiple}, "
            f"got {width}x{height}"
        )
    if width < height:
        raise ValueError(
            f"MiniMax-H3 low-resolution sweep accepts landscape canvases only, "
            f"got {width}x{height}"
        )
    if width > MAX_WIDTH or height > MAX_HEIGHT or width * height > MAX_PIXELS:
        raise ValueError(
            "MiniMax-H3 student canvas must be no larger than 896x512 "
            f"({MAX_PIXELS} pixels), got {width}x{height}"
        )
    ratio = float(width) / float(height)
    if not math.isfinite(ratio) or not min_aspect_ratio <= ratio <= max_aspect_ratio:
        raise ValueError(
            f"aspect ratio {ratio:g} is outside H3's "
            f"[{min_aspect_ratio:g}, {max_aspect_ratio:g}] range"
        )
    return {
        "width": width,
        "height": height,
        "pixels": width * height,
        "aspect_ratio": ratio,
        "multiple": multiple,
    }


def install_lowres_sweep_overlay(*, width: int, height: int) -> dict[str, Any]:
    """Resolve every student ``short_edge=512`` request to one fixed canvas."""

    requested_canvas = validate_lowres_canvas(width, height)

    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3 import (
        prequeue,
        request_validation,
        resolved_plan,
    )

    marker = getattr(request_validation, "_h3_lowres_sweep_overlay", None)
    if marker is not None:
        active = (int(marker.get("internal_width", -1)), int(marker.get("internal_height", -1)))
        if active != (width, height):
            raise RuntimeError(f"a different H3 low-resolution overlay is active: {marker}")
        return dict(marker)

    legacy_marker = getattr(request_validation, "_h3_512p_overlay", None)
    if legacy_marker is not None:
        active = (
            int(legacy_marker.get("internal_width", -1)),
            int(legacy_marker.get("internal_height", -1)),
        )
        if (width, height) != (DEFAULT_WIDTH, DEFAULT_HEIGHT) or active != (
            DEFAULT_WIDTH,
            DEFAULT_HEIGHT,
        ):
            raise RuntimeError(f"a conflicting H3 512p overlay is active: {legacy_marker}")
        request_validation._h3_lowres_sweep_overlay = legacy_marker
        return dict(legacy_marker)

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
        raise RuntimeError("unexpected upstream MiniMax-H3 base short edge")
    if int(getattr(resolved_plan, "MINIMAX_H3_MAX_PIXELS", -1)) != UPSTREAM_MAX_PIXELS:
        raise RuntimeError("unexpected upstream MiniMax-H3 pixel budget")
    if int(getattr(resolved_plan, "MINIMAX_H3_CANVAS_MULTIPLE", -1)) != CANVAS_MULTIPLE:
        raise RuntimeError("unexpected upstream MiniMax-H3 canvas multiple")
    upstream_min_ratio = float(
        getattr(resolved_plan, "MINIMAX_H3_MIN_ASPECT_RATIO", float("nan"))
    )
    upstream_max_ratio = float(
        getattr(resolved_plan, "MINIMAX_H3_MAX_ASPECT_RATIO", float("nan"))
    )
    if not math.isclose(
        upstream_min_ratio, H3_MIN_ASPECT_RATIO, rel_tol=0.0, abs_tol=1e-12
    ) or not math.isclose(
        upstream_max_ratio, H3_MAX_ASPECT_RATIO, rel_tol=0.0, abs_tol=1e-12
    ):
        raise RuntimeError(
            "unexpected upstream MiniMax-H3 aspect-ratio range: "
            f"[{upstream_min_ratio}, {upstream_max_ratio}]"
        )
    validate_lowres_canvas(
        width,
        height,
        multiple=int(resolved_plan.MINIMAX_H3_CANVAS_MULTIPLE),
        min_aspect_ratio=upstream_min_ratio,
        max_aspect_ratio=upstream_max_ratio,
    )

    original_validate_target: Callable[..., dict[str, Any]] = required[
        "request_validation._validate_target"
    ]
    original_validate_short_edge: Callable[[Any], int] = required[
        "resolved_plan._validate_base_short_edge"
    ]
    original_resolve_shape: Callable[..., dict[str, Any]] = required[
        "resolved_plan.minimax_h3_resolve_spatial_shape"
    ]

    def validate_target_with_lowres(target: Any, *, profile: Any) -> dict[str, Any]:
        if (
            not isinstance(target, Mapping)
            or target.get("short_edge") != STUDENT_REQUEST_SHORT_EDGE
        ):
            return original_validate_target(target, profile=profile)
        upstream_target = dict(target)
        upstream_target["short_edge"] = UPSTREAM_SHORT_EDGE
        normalized = original_validate_target(upstream_target, profile=profile)
        normalized["short_edge"] = STUDENT_REQUEST_SHORT_EDGE
        return normalized

    def validate_short_edge_with_lowres(value: Any) -> int:
        if (
            not isinstance(value, bool)
            and isinstance(value, int)
            and value == STUDENT_REQUEST_SHORT_EDGE
        ):
            return STUDENT_REQUEST_SHORT_EDGE
        return original_validate_short_edge(value)

    def resolve_shape_with_lowres(
        *,
        width: int | float,
        height: int | float,
        base_short_edge: int = UPSTREAM_SHORT_EDGE,
    ) -> dict[str, Any]:
        if base_short_edge != STUDENT_REQUEST_SHORT_EDGE:
            return original_resolve_shape(
                width=width,
                height=height,
                base_short_edge=base_short_edge,
            )
        validate_short_edge_with_lowres(base_short_edge)
        # Validate the upstream-provided source geometry before replacing it
        # with the process-fixed canvas.  This keeps H3's input-ratio guard.
        try:
            source_width = float(width)
            source_height = float(height)
        except (TypeError, ValueError) as exc:
            raise ValueError("shape width and height must be finite and positive") from exc
        if (
            not math.isfinite(source_width)
            or not math.isfinite(source_height)
            or source_width <= 0.0
            or source_height <= 0.0
        ):
            raise ValueError("shape width and height must be finite and positive")
        source_ratio = source_width / source_height
        if not upstream_min_ratio <= source_ratio <= upstream_max_ratio:
            raise ValueError(
                f"aspect ratio {source_ratio:g} is outside H3's "
                f"[{upstream_min_ratio:g}, {upstream_max_ratio:g}] range"
            )
        is_default = (width_fixed, height_fixed) == (DEFAULT_WIDTH, DEFAULT_HEIGHT)
        return {
            "geometry": "resolved_v2",
            "shape_policy_version": (
                "adapt_shape_v1+512p_overlay_v1"
                if is_default
                else "adapt_shape_v1+fixed_lowres_sweep_overlay_v1"
            ),
            "base_short_edge": STUDENT_REQUEST_SHORT_EDGE,
            "effective_short_edge": min(width_fixed, height_fixed),
            "size_mode": "short_edge" if is_default else "fixed_canvas",
            "max_pixels": MAX_PIXELS,
            "multiple": CANVAS_MULTIPLE,
            "rounding": "nearest" if is_default else "prevalidated_exact",
            "width": width_fixed,
            "height": height_fixed,
        }

    width_fixed = int(requested_canvas["width"])
    height_fixed = int(requested_canvas["height"])
    request_validation._validate_target = validate_target_with_lowres
    resolved_plan._validate_base_short_edge = validate_short_edge_with_lowres
    resolved_plan.minimax_h3_resolve_spatial_shape = resolve_shape_with_lowres
    prequeue.minimax_h3_resolve_spatial_shape = resolve_shape_with_lowres

    resolved = resolve_shape_with_lowres(
        width=width_fixed,
        height=height_fixed,
        base_short_edge=STUDENT_REQUEST_SHORT_EDGE,
    )
    if (int(resolved["width"]), int(resolved["height"])) != (
        width_fixed,
        height_fixed,
    ):
        raise RuntimeError(f"low-resolution overlay resolved an unexpected canvas: {resolved}")

    is_default = (width_fixed, height_fixed) == (DEFAULT_WIDTH, DEFAULT_HEIGHT)
    marker = {
        "installed": True,
        "name": (
            "sglang_minimax_h3_896x512_overlay_v1"
            if is_default
            else f"sglang_minimax_h3_{width_fixed}x{height_fixed}_lowres_overlay_v1"
        ),
        "pinned_sglang_commit": PINNED_SGLANG_COMMIT,
        "requested_short_edge": STUDENT_REQUEST_SHORT_EDGE,
        "internal_width": width_fixed,
        "internal_height": height_fixed,
        "max_pixels": MAX_PIXELS,
        "canvas_multiple": CANVAS_MULTIPLE,
        "aspect_ratio": float(requested_canvas["aspect_ratio"]),
        "fixed_process_canvas": True,
        "upstream_768_path_preserved": True,
    }
    request_validation._h3_lowres_sweep_overlay = marker
    if is_default:
        request_validation._h3_512p_overlay = marker
    return dict(marker)


__all__ = [
    "CANVAS_MULTIPLE",
    "DEFAULT_HEIGHT",
    "DEFAULT_WIDTH",
    "MAX_HEIGHT",
    "MAX_PIXELS",
    "MAX_WIDTH",
    "PINNED_SGLANG_COMMIT",
    "STUDENT_REQUEST_SHORT_EDGE",
    "install_lowres_sweep_overlay",
    "validate_lowres_canvas",
]
