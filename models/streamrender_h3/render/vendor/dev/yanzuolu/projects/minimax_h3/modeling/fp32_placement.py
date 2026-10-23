# SPDX-License-Identifier: Apache-2.0
"""Keep MiniMax-H3's fp32 boundary modules computing in fp32 under FSDP.

The released checkpoint is 99.948% bf16 (33.106 B) and 0.052% fp32 (17.2 M).
The fp32 part is not scattered: it is six modules at the edges of the network --
the two patch projections, the two time-embedder linears, and the two output
heads -- and the model casts the ACTIVATION to match at each of them
(``_embed`` -> fp32 rows, ``time_embedder`` -> fp32 sinusoid, ``final_layer``
-> ``h.to(fp32)``). Weight and activation are fp32 together there and bf16
together everywhere else, which is why the released model needs no autocast.

FSDP breaks that pairing from one side only. ``MixedPrecisionPolicy`` is
constructed once in common/model/placement.py and handed to every
``fully_shard`` call, so ``param_dtype: bfloat16`` all-gathers those six
modules' weights as bf16 while the model keeps casting their inputs to fp32 --
and ``_Linear.forward`` is a bare ``F.linear``, which raises rather than
promoting. Going the other way (``param_dtype: null`` with fp32 masters) fails
symmetrically in the 50-block trunk.

This plugin shards those six modules FIRST, each with its own fp32 policy.
``_get_managed_modules`` stops its DFS at any submodule that already carries an
FSDP state, so the wrap units and the root that placement shards afterwards
never reclaim them, and each unit keeps the policy it was created with. The
result is what the config alone cannot express: fp32 master weights everywhere
(``weight_dtype: float32``), compute dtype following the source annotation.

Config::

    placement:
      weight_dtype: float32
      plugins:
        - module: dev.yanzuolu.projects.minimax_h3.modeling.fp32_placement
          class_name: MiniMaxH3FP32ModulePlacement
      fsdp:
        wrap_modules: [CausalMiniMaxH3DiTBlock]
        param_dtype: bfloat16      # everything this plugin did not claim
        reduce_dtype: float32
        cpu_offload: true

The plugin takes no config of its own: the module set comes from the model and
every FSDP setting is read back from ``placement.fsdp`` so the two cannot drift.
"""

from __future__ import annotations

import re
from typing import Any

import torch
from torch.distributed.fsdp import CPUOffloadPolicy, MixedPrecisionPolicy, fully_shard

from dev.yanzuolu.common.distributed import ops
from dev.yanzuolu.common.model.placement import PlacementPlugin, _get_hsdp_mesh
from dev.yanzuolu.projects.minimax_h3.modeling.transformer.model import MINIMAX_H3_FP32_PARAM_NAMES

# PEFT's wrapper levels between a declared module and its parameter: a LoRA
# layer's `base_layer`, and a `modules_to_save` wrapper's frozen
# `original_module` and per-adapter trainable copy.
_PEFT_WRAPPER_LEVEL = re.compile(r"\.(base_layer|original_module|modules_to_save\.[^.]+)\.")


class MiniMaxH3FP32ModulePlacement(PlacementPlugin):
    """Pre-shard the fp32 boundary modules so their compute stays fp32."""

    def before_placement(self, state: dict[str, Any]) -> dict[str, Any]:
        model = state["model"]
        fsdp = state["config"].get("placement", {}).get("fsdp", None)
        assert fsdp is not None, (
            "MiniMaxH3FP32ModulePlacement only means anything under placement.fsdp; "
            "without it nothing all-gathers and no dtype is overridden"
        )

        # Derived from the model's own fp32 declaration rather than listed here
        # (a second list would be free to drift), and resolved against the live
        # model because the declarations are relative to the DiT while the
        # configured class may add wrapper prefixes. PEFT also inserts wrapper
        # levels before the declared weight or bias. They are transparent for
        # matching, but the owner path comes from the live name: a LoRA layer's
        # frozen base layer receives this fp32 policy while its LoRA parameters
        # remain under the outer bf16 policy, and both copies a `modules_to_save`
        # wrapper holds receive it.
        owners: dict[str, set[str]] = {}
        for name, _ in model.named_parameters():
            declared_name = _PEFT_WRAPPER_LEVEL.sub(".", name)
            for declared in MINIMAX_H3_FP32_PARAM_NAMES:
                if declared_name == declared or declared_name.endswith("." + declared):
                    owners.setdefault(name.rsplit(".", 1)[0], set()).add(declared)
        found = {d for declared in owners.values() for d in declared}
        assert found == set(MINIMAX_H3_FP32_PARAM_NAMES), (
            "fp32 parameters the model declares but this model does not expose: "
            f"{sorted(set(MINIMAX_H3_FP32_PARAM_NAMES) - found)}"
        )
        paths = sorted(owners)

        # Mirror placement.py's mesh decision exactly. _get_hsdp_mesh is private
        # but importing it is the point: it caches by shape, and its own comment
        # requires every rank to build the same groups in the same order, so a
        # second mesh built here would not be the mesh the rest of the model uses.
        world_size = ops.get_world_size()
        shard_size = fsdp.get("shard_size") or min(ops.get_local_world_size(), world_size)
        kwargs: dict[str, Any] = {
            "mp_policy": MixedPrecisionPolicy(
                # The whole reason this plugin exists.
                param_dtype=torch.float32,
                reduce_dtype=getattr(torch, fsdp.get("reduce_dtype", "float32")),
                # Hardcoded, and NOT read from placement.fsdp: that knob is a
                # judgement about the bf16 trunk, where casting inputs DOWN
                # would collapse timesteps and rotary positions onto each other.
                # Here the cast would go UP and lose nothing -- which is exactly
                # why it must stay off. These six modules are handed fp32 rows
                # by the model on purpose; if that ever stops being true, the
                # mismatch should raise out of F.linear rather than be silently
                # upcast into an fp32 GEMM carrying bf16 precision. Passing it
                # explicitly is also required: torch's dataclass default is True.
                cast_forward_inputs=False,
            ),
            "reshard_after_forward": True,
        }
        if shard_size < world_size:
            kwargs["mesh"] = _get_hsdp_mesh((world_size // shard_size, shard_size))
        # Follows the main config rather than staying resident. 17.2 M params is
        # ~69 MB and would be cheap to keep on GPU, but a model whose parameters
        # straddle two devices makes the optimizer step and clip_grad_norm_'s
        # partial-norm reduction mixed-device for no real saving.
        if bool(fsdp.get("cpu_offload", False)):
            kwargs["offload_policy"] = CPUOffloadPolicy()

        for path in paths:
            module = model.get_submodule(path)
            # The policy above is a claim about every parameter in the module.
            # If the model ever adds a bf16 one here, the claim silently becomes
            # an upcast instead of a match.
            wrong = [
                f"{path}.{n}={p.dtype}"
                for n, p in module.named_parameters()
                if p.is_floating_point() and p.dtype != torch.float32
            ]
            assert not wrong, f"expected an all-fp32 module, got {wrong}"
            fully_shard(module, **kwargs)

        return state


__all__ = ["MiniMaxH3FP32ModulePlacement"]
