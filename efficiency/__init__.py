# Copyright 2025 SGLang authors
#
# Model-agnostic inference-acceleration framework.
#
# A small typed/effect-checked composition layer for efficiency techniques:
#   * Schedule[T]  -- time-varying params (per step/stage), the time sub-DSL.
#   * Technique    -- a composable primitive with a phase + effect set (reads/
#                     writes) + Schedule[bool] enable; OFF == byte-identical.
#   * ModelSpec    -- a model's declaration of structural seams (capabilities).
#   * compose()    -- type-checks capabilities, rejects structural conflicts
#                     (effect system), orders by phase -> an executable Plan.
#   * registry     -- register_technique / register_model_spec (BlockAdapter idiom).
#
# Adapt a new model by writing ONE ModelSpec; reuse every technique. See
# techniques/ for the concrete techniques (token_prune is the first).

from __future__ import annotations

from efficiency.compose import (
    CompositionError,
    Plan,
    check_conflicts,
    compose,
)
from efficiency.registry import (
    build_technique,
    build_transform,
    get_model_spec,
    is_supported,
    register_model_spec,
    register_technique,
    register_transform,
    registered_models,
    registered_techniques,
    registered_transforms,
)
from efficiency.transform import (
    ModelTransform,
    TransformContext,
    TransformPhase,
)
from efficiency.schedule import (
    Schedule,
    as_schedule,
    at_steps,
    before,
    by_stage,
    const,
    predicate,
    parse_steps,
)
from efficiency.spec import ModelSpec
from efficiency.technique import (
    Capability,
    Phase,
    Seam,
    Technique,
    TechniqueContext,
)

# register built-in techniques, transforms, and model specs (import side-effects)
from efficiency import techniques  # noqa: E402,F401
from efficiency import transforms  # noqa: E402,F401
from efficiency import models  # noqa: E402,F401

__all__ = [
    "Schedule", "as_schedule", "at_steps", "before", "by_stage", "const",
    "predicate", "parse_steps",
    "Technique", "TechniqueContext", "Phase", "Seam", "Capability",
    "ModelTransform", "TransformContext", "TransformPhase",
    "ModelSpec",
    "compose", "Plan", "CompositionError", "check_conflicts",
    "register_technique", "register_transform", "register_model_spec",
    "build_technique", "build_transform",
    "get_model_spec", "is_supported",
    "registered_techniques", "registered_transforms", "registered_models",
]
