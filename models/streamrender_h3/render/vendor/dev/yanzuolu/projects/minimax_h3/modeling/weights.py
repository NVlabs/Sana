"""Weight loader for the released MiniMax-H3 DiT checkpoint.

Not vendored code -- this file is ours.

The released checkpoint stores every ``*.attn.qkv_proj.weight`` in the layout
MiniMax trained with: rows grouped **per head** as ``[q0, k0, v0, q1, k1, v1,
...]``. The model expects the layout SGLang runs with: three contiguous blocks
``[q_all, k_all, v_all]``. Both have identical shape and identical value
multisets -- only the row permutation differs -- so a state dict alone carries
no evidence of which layout it is in.

Upstream resolves this with a ``weight_loader`` closure installed on the qkv
parameter (``MiniMaxH3Attention._install_qkv_weight_loader``), which sglang's
own loader routes through. This repo does not: ``DefaultWeightLoader`` calls
``model.load_state_dict`` directly, which bypasses ``weight_loader`` entirely
and would load qkv in the wrong row order -- silently, since the shapes match.

Doing the permutation inside ``MiniMaxH3DiTModel.load_state_dict`` instead would
be wrong, because the permutation is not idempotent: training checkpoints are
saved in runtime layout, so resuming would permute an already-permuted tensor
and scramble it, again silently.

So the permutation belongs to the *path* that loads the released checkpoint, and
nowhere else. ``_remap_keys`` is the hook the repo already designates for that
(cf. ``TransformersWeightLoader``): it runs only for an explicit ``weight.path``
load, never for checkpointer/DCP resume. Which layout is being loaded is then a
property of the configured loader rather than something guessed at runtime.

The ``dit.`` prefix is the same kind of property. A model node is a
``MiniMaxH3X0Model`` holding the DiT at ``.dit``, so its state dict is ``dit.*``
while the released file -- a dump of the DiT alone -- carries no prefix. Adding
it here rather than delegating ``load_state_dict`` on the wrapper keeps one
namespace everywhere else, including DCP, which addresses state-dict keys as
paths into the module tree.

The released ``video_patch_proj.weight`` is also a property of the released
format: it reads only the noisy latent. A DiT configured with
``video_condition_channels`` has extra input columns for its clean condition
latent, and ``_remap_keys`` appends them as zeros, so the widened model starts
out computing exactly the released function and ignoring the condition.

This loader therefore reads exactly one thing: the RELEASED checkpoint.

  * Released weights -> here. Unprefixed keys in per-head qkv layout.
  * Our own checkpoints, resumed -> DCP, which never touches this file. It saves
    and loads through the module tree's own names, so it is symmetric by
    construction and needs no translation.
  * Our own checkpoints, re-loaded through ``weight.path`` to seed a new trial ->
    ``DefaultWeightLoader``. ``export_state`` writes ``dit.*`` with the LoRA
    already folded in and qkv already in runtime layout, so there is nothing left
    to translate; running them through this loader would prefix them twice and
    permute qkv a second time, and the permutation is not idempotent. Misrouting
    it is not silent -- the doubly-prefixed qkv keys match no attention module,
    which is what the assertion in ``_remap_keys`` refuses.

Usage -- in the trial YAML, on the model node that reads the released weights::

    weight:
      module: dev.yanzuolu.projects.minimax_h3.modeling.weights
      class_name: MiniMaxH3WeightLoader
      path: /root/models/MiniMaxAI/MiniMax-H3/FL2VA/transformer/model.safetensors.index.json
"""

import math
from typing import Any

import torch

from dev.yanzuolu.common.logging import get_logger
from dev.yanzuolu.common.model.weights import DefaultWeightLoader
from .transformer.model import (
    MiniMaxH3Attention,
    _reorder_grouped_qkv_to_qkv,
)

logger = get_logger()

_QKV_SUFFIX = ".qkv_proj.weight"
_VIDEO_PATCH_PROJ_WEIGHT = "video_patch_proj.weight"


class MiniMaxH3WeightLoader(DefaultWeightLoader):
    """Permute per-head-grouped qkv rows into ``[q_all, k_all, v_all]`` on load."""

    @staticmethod
    def _prefix(model: torch.nn.Module) -> str:
        """What the released file's keys must be prefixed with to reach the DiT.

        A model node is a ``MiniMaxH3X0Model`` holding the DiT at ``.dit``, so
        its state dict is ``dit.*`` while the file's keys carry no prefix. The
        wrapper deliberately does not paper over that by delegating
        ``load_state_dict``: doing so made ``state_dict`` and the module tree
        disagree, which broke DCP's checkpoint save. Reconciling it here instead
        keeps one namespace everywhere -- including the loader's own
        missing/unexpected report, which is then worth reading.
        """
        return "dit." if hasattr(model, "dit") else ""

    def _qkv_specs(self, model: torch.nn.Module) -> dict[str, dict[str, int]]:
        """Map every qkv weight key to the reorder parameters of its own module."""
        specs: dict[str, dict[str, int]] = {}
        for name, module in model.named_modules():
            if not isinstance(module, MiniMaxH3Attention):
                continue
            key = f"{name}{_QKV_SUFFIX}" if name else _QKV_SUFFIX.lstrip(".")
            specs[key] = {
                # Upstream passes arch.num_attention_heads, i.e. the pre-TP head
                # count; total_num_heads is that value.
                "num_query_groups": module.total_num_heads,
                # MiniMax-H3 is MHA: one query head per group.
                "heads_per_group": 1,
                "head_dim": module.head_dim,
            }
        return specs

    @staticmethod
    def _widen_video_patch_proj(
        state: dict[str, Any], prefix: str, value: torch.Tensor
    ) -> torch.Tensor:
        """Append zero input columns for the DiT's video condition channels."""
        model = state["model"]
        arch = model.get_submodule(prefix.rstrip(".")).arch
        if not arch.video_condition_channels:
            return value
        key = f"{prefix}{_VIDEO_PATCH_PROJ_WEIGHT}"
        target = model.get_parameter(key)
        extra = arch.video_condition_channels * math.prod(arch.patch_size)
        assert (
            value.shape[0] == target.shape[0]
            and value.shape[1] + extra == target.shape[1]
        ), (
            f"[{state['name']}] cannot widen '{key}' from {list(value.shape)} to "
            f"{list(target.shape)} by {extra} condition input columns"
        )
        logger.info(
            "[%s] widened %s from %d to %d input columns, zero for %d video "
            "condition channels",
            state["name"],
            key,
            value.shape[1],
            target.shape[1],
            arch.video_condition_channels,
        )
        return torch.cat((value, value.new_zeros(value.shape[0], extra)), dim=1)

    def _remap_keys(
        self,
        state: dict[str, Any],
        values: dict[str, Any],
        *,
        log_decision: bool,
    ) -> dict[str, Any]:
        specs = self._qkv_specs(state["model"])
        assert specs, (
            f"[{state['name']}] MiniMaxH3WeightLoader found no MiniMaxH3Attention "
            "submodules; it is configured on the wrong model node"
        )

        prefix = self._prefix(state["model"])
        remapped: dict[str, Any] = {}
        reordered = 0
        for raw_key, value in values.items():
            key = f"{prefix}{raw_key}"
            spec = specs.get(key)
            if spec is None:
                # A qkv key with no spec means the loader and the checkpoint
                # disagree about the key namespace. Passing it through would put
                # the run back exactly where it started -- correct shapes, wrong
                # rows, no error -- so refuse instead.
                assert not key.endswith(_QKV_SUFFIX), (
                    f"[{state['name']}] '{key}' is a qkv weight but no "
                    "MiniMaxH3Attention answers to that name; the loader resolved "
                    f"specs like '{next(iter(specs))}'"
                )
                if raw_key == _VIDEO_PATCH_PROJ_WEIGHT and isinstance(value, torch.Tensor):
                    value = self._widen_video_patch_proj(state, prefix, value)
                remapped[key] = value
                continue
            if not isinstance(value, torch.Tensor):
                # The sharded path calls this once on index["weight_map"], whose
                # values are filenames; only the per-shard call carries tensors.
                remapped[key] = value
                continue
            remapped[key] = _reorder_grouped_qkv_to_qkv(value, **spec)
            reordered += 1

        if reordered:
            logger.info(
                "[%s] reordered %d qkv weights from per-head [q,k,v] groups to "
                "[q_all, k_all, v_all]",
                state["name"],
                reordered,
            )
        elif log_decision:
            logger.info(
                "[%s] no qkv weights in this mapping (%d keys); nothing reordered",
                state["name"],
                len(values),
            )
        return remapped


# torch's deprecated ``nn.utils.weight_norm`` names its two parameters
# ``weight_g`` (magnitude) and ``weight_v`` (direction); the parametrization that
# replaced it stores the same two tensors as ``parametrizations.weight.original0``
# and ``...original1``. Verified equal element-wise, with identical forward
# output, on torch 2.11.
_WEIGHT_NORM_RENAMES = (
    (".weight_g", ".parametrizations.weight.original0"),
    (".weight_v", ".parametrizations.weight.original1"),
)


class MiniMaxH3AudioVAEWeightLoader(DefaultWeightLoader):
    """Rename legacy ``weight_norm`` keys to their parametrization equivalents.

    The released audio VAE checkpoint was written with the old ``weight_norm``
    API (172 ``weight_g`` + 172 ``weight_v`` keys). The vendored model uses
    ``nn.utils.parametrizations.weight_norm``, so those 344 keys would otherwise
    be reported missing and the convolutions would keep their meta placeholders.

    Unlike the DiT's qkv permutation this is a pure rename and therefore
    idempotent -- a state dict that has already been renamed contains no
    ``weight_g``/``weight_v`` keys and passes through untouched.
    """

    def _remap_keys(
        self,
        state: dict[str, Any],
        values: dict[str, Any],
        *,
        log_decision: bool,
    ) -> dict[str, Any]:
        remapped: dict[str, Any] = {}
        renamed = 0
        for key, value in values.items():
            for old, new in _WEIGHT_NORM_RENAMES:
                if key.endswith(old):
                    key = key[: -len(old)] + new
                    renamed += 1
                    break
            remapped[key] = value

        if renamed and log_decision:
            logger.info(
                "[%s] renamed %d legacy weight_norm keys to parametrizations",
                state["name"],
                renamed,
            )
        return remapped
