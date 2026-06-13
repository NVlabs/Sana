# Ported from Sol-LTX-Infer scripts/ltx23_official_kwl_ops.py @ 29d0d9e.
# Kept as a reference migration for the kwl_fusion loop; imports are guarded so
# this file smoke-imports in the autovideo worktree without the LTX runtime.
"""Kernel-wise lossless ops for the official LTX-2.3 HQ pipeline.

This module is intentionally independent of the SGLang model wrapper.  It
reuses the repo's Triton kernels, but installs them through the official
``ltx_core`` ``ModuleOps`` hook so the upstream two-stage HQ pipeline keeps the
same scheduler, LoRA strengths, guidance settings, and model loading flow.
"""

from __future__ import annotations

import importlib.util
import logging
import os
from pathlib import Path
from types import MethodType
from typing import TYPE_CHECKING, Any

try:  # pragma: no cover - exercised by smoke import in minimal environments
    import torch
except Exception as exc:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    _TORCH_IMPORT_ERROR = exc
else:
    _TORCH_IMPORT_ERROR = None

if TYPE_CHECKING:  # pragma: no cover
    from ltx_core.loader.module_ops import ModuleOps
    from ltx_core.model.transformer.attention import Attention
    from ltx_core.model.transformer.feed_forward import FeedForward
    from ltx_core.model.transformer.ops import PytorchAdaZeroFunction, PytorchPreAttention
    from ltx_core.model.transformer.rope import LTXRopeType, apply_rotary_emb
    from ltx_core.model.transformer.transformer import BasicAVTransformerBlock
else:
    ModuleOps = None
    Attention = None
    FeedForward = None
    PytorchAdaZeroFunction = None
    PytorchPreAttention = None
    LTXRopeType = None
    apply_rotary_emb = None
    BasicAVTransformerBlock = None

logger = logging.getLogger(__name__)


_REPO_ROOT = Path(__file__).resolve().parents[1]
_KERNEL_ROOT = _REPO_ROOT / "python" / "sglang" / "jit_kernel" / "diffusion" / "triton"
_KERNEL_CACHE: dict[str, Any] = {}
_WARNED: set[str] = set()


def _require_torch() -> Any:
    if torch is None:
        raise ImportError("KWL reference ops require torch to execute") from _TORCH_IMPORT_ERROR
    return torch


def _require_ltx_core() -> None:
    global ModuleOps, Attention, FeedForward, PytorchAdaZeroFunction
    global PytorchPreAttention, LTXRopeType, apply_rotary_emb, BasicAVTransformerBlock

    if ModuleOps is not None:
        return

    try:
        from ltx_core.loader.module_ops import ModuleOps as _ModuleOps
        from ltx_core.model.transformer.attention import Attention as _Attention
        from ltx_core.model.transformer.feed_forward import FeedForward as _FeedForward
        from ltx_core.model.transformer.ops import (
            PytorchAdaZeroFunction as _PytorchAdaZeroFunction,
            PytorchPreAttention as _PytorchPreAttention,
        )
        from ltx_core.model.transformer.rope import (
            LTXRopeType as _LTXRopeType,
            apply_rotary_emb as _apply_rotary_emb,
        )
        from ltx_core.model.transformer.transformer import (
            BasicAVTransformerBlock as _BasicAVTransformerBlock,
        )
    except Exception as exc:
        raise ImportError("KWL reference ops require ltx_core to install pipeline hooks") from exc

    ModuleOps = _ModuleOps
    Attention = _Attention
    FeedForward = _FeedForward
    PytorchAdaZeroFunction = _PytorchAdaZeroFunction
    PytorchPreAttention = _PytorchPreAttention
    LTXRopeType = _LTXRopeType
    apply_rotary_emb = _apply_rotary_emb
    BasicAVTransformerBlock = _BasicAVTransformerBlock


def _enabled(name: str, default: str = "1") -> bool:
    return os.environ.get(name, default) == "1"


def _warn_once(key: str, message: str, *args: object) -> None:
    if key in _WARNED:
        return
    _WARNED.add(key)
    logger.warning(message, *args)


def _load_kernel_module(name: str) -> Any:
    cached = _KERNEL_CACHE.get(name)
    if cached is not None:
        return cached

    path = _KERNEL_ROOT / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"_ltx23_official_kwl_{name}", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load kernel module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _KERNEL_CACHE[name] = module
    return module


def _can_use_bf16_3d(x: torch.Tensor) -> bool:
    torch_mod = _require_torch()
    return (
        x.is_cuda
        and x.dtype == torch_mod.bfloat16
        and x.ndim == 3
        and x.stride(-1) == 1
    )


class KWLAdaZeroFunction:
    """Fused RMSNorm + AdaLN scale/shift with eager fallback."""

    def __init__(self) -> None:
        _require_ltx_core()
        self._fallback = PytorchAdaZeroFunction()

    def __call__(
        self,
        x: torch.Tensor,
        eps: float,
        scale: torch.Tensor,
        shift: torch.Tensor,
    ) -> torch.Tensor:
        if (
            _enabled("LTX23_OFFICIAL_KWL_ADALN")
            and _can_use_bf16_3d(x)
            and scale.is_cuda
            and shift.is_cuda
            and scale.dtype == x.dtype
            and shift.dtype == x.dtype
            and scale.ndim == 3
            and shift.ndim == 3
            and scale.shape == shift.shape
            and scale.shape[0] == x.shape[0]
            and scale.shape[-1] == x.shape[-1]
            and scale.shape[1] in (1, x.shape[1])
            and scale.stride(-1) == 1
            and shift.stride(-1) == 1
        ):
            try:
                kernel = _load_kernel_module("ltx2_adaln").ltx2_rms_norm_modulate
                return kernel(x, scale, shift, eps)
            except Exception as exc:  # pragma: no cover - runtime safety path
                _warn_once("adaln", "Disabling official KWL AdaLN fast path after failure: %s", exc)
        return self._fallback(x, eps, scale, shift)


class KWLPreAttention:
    """Fused Q/K RMSNorm and optional SPLIT RoPE with eager fallback."""

    def __init__(self) -> None:
        _require_ltx_core()
        self._fallback = PytorchPreAttention()

    def __call__(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        attn_module: torch.nn.Module,
        mask: torch.Tensor | None,
        pe: tuple[torch.Tensor, torch.Tensor] | None,
        k_pe: tuple[torch.Tensor, torch.Tensor] | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del mask
        q_norm = attn_module.q_norm
        k_norm = attn_module.k_norm
        eps = float(getattr(q_norm, "eps", 1e-6) or 1e-6)

        if pe is not None:
            fused = self._try_qknorm_split_rope(q, k, q_norm, k_norm, eps, pe, k_pe, attn_module)
            if fused is not None:
                return fused

        fused_qk = self._try_qknorm_pair(q, k, q_norm, k_norm, eps)
        if fused_qk is None:
            return self._fallback(q, k, attn_module, None, pe, k_pe)

        q, k = fused_qk
        if pe is not None:
            q = apply_rotary_emb(q, pe, attn_module.rope_type)
            k = apply_rotary_emb(k, pe if k_pe is None else k_pe, attn_module.rope_type)
        return q, k

    def _try_qknorm_pair(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        q_norm: torch.nn.Module,
        k_norm: torch.nn.Module,
        eps: float,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        if (
            not _enabled("LTX23_OFFICIAL_KWL_QKNORM")
            or not _can_use_bf16_3d(q)
            or not _can_use_bf16_3d(k)
            or q.shape[-1] != k.shape[-1]
            or not isinstance(q_norm, torch.nn.RMSNorm)
            or not isinstance(k_norm, torch.nn.RMSNorm)
        ):
            return None

        qw = q_norm.weight
        kw = k_norm.weight
        hidden = q.shape[-1]
        if (
            qw is None
            or kw is None
            or qw.device != q.device
            or kw.device != k.device
            or qw.dtype != q.dtype
            or kw.dtype != k.dtype
            or qw.numel() != hidden
            or kw.numel() != hidden
        ):
            return None
        try:
            kernel = _load_kernel_module("ltx2_qknorm").ltx2_qknorm_pair_inplace
            kernel(q.view(-1, hidden), k.view(-1, hidden), qw, kw, eps)
            return q, k
        except Exception as exc:  # pragma: no cover - runtime safety path
            _warn_once("qknorm", "Disabling official KWL QKNorm fast path after failure: %s", exc)
            return None

    def _try_qknorm_split_rope(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        q_norm: torch.nn.Module,
        k_norm: torch.nn.Module,
        eps: float,
        pe: tuple[torch.Tensor, torch.Tensor],
        k_pe: tuple[torch.Tensor, torch.Tensor] | None,
        attn_module: torch.nn.Module,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        if (
            not _enabled("LTX23_OFFICIAL_KWL_QKNORM_ROPE")
            or getattr(attn_module, "rope_type", None) != LTXRopeType.SPLIT
            or not _can_use_bf16_3d(q)
            or not _can_use_bf16_3d(k)
            or not q.is_contiguous()
            or not k.is_contiguous()
            or q.shape[0] != k.shape[0]
            or q.shape[-1] != k.shape[-1]
            or not isinstance(q_norm, torch.nn.RMSNorm)
            or not isinstance(k_norm, torch.nn.RMSNorm)
        ):
            return None

        q_cos, q_sin = pe
        k_cos, k_sin = pe if k_pe is None else k_pe
        if (
            q_cos.ndim != 4
            or q_sin.shape != q_cos.shape
            or k_cos.ndim != 4
            or k_sin.shape != k_cos.shape
            or q_cos.dtype != q.dtype
            or q_sin.dtype != q.dtype
            or k_cos.dtype != q.dtype
            or k_sin.dtype != q.dtype
            or not q_cos.is_cuda
            or not q_sin.is_cuda
            or not k_cos.is_cuda
            or not k_sin.is_cuda
            or q_cos.shape[0] != q.shape[0]
            or k_cos.shape[0] != k.shape[0]
        ):
            return None

        qw = q_norm.weight
        kw = k_norm.weight
        hidden = q.shape[-1]
        if (
            qw is None
            or kw is None
            or qw.device != q.device
            or kw.device != k.device
            or qw.dtype != q.dtype
            or kw.dtype != k.dtype
            or qw.numel() != hidden
            or kw.numel() != hidden
        ):
            return None
        try:
            kernel = _load_kernel_module("ltx2_qknorm").ltx2_qknorm_split_rope_pair
            return kernel(q, k, qw, kw, q_cos, q_sin, k_cos, k_sin, eps)
        except Exception as exc:  # pragma: no cover - runtime safety path
            _warn_once("qknorm_rope", "Disabling official KWL QKNorm+RoPE fast path after failure: %s", exc)
            return None


def _kwl_feed_forward_forward(self: FeedForward, x: torch.Tensor) -> torch.Tensor:
    _require_torch()
    original = getattr(self, "_kwl_original_forward")
    if (
        _enabled("LTX23_OFFICIAL_KWL_FFN_PROJ_IN_GELU")
        and x.is_cuda
        and x.dtype in (torch.float16, torch.bfloat16)
        and x.ndim >= 2
        and x.stride(-1) == 1
        and len(self.net) >= 3
        and hasattr(self.net[0], "proj")
        and isinstance(self.net[2], torch.nn.Linear)
    ):
        proj_in = self.net[0].proj
        proj_out = self.net[2]
        weight = proj_in.weight
        bias = proj_in.bias
        if (
            bias is not None
            and weight.device == x.device
            and bias.device == x.device
            and weight.dtype == x.dtype
            and bias.dtype == x.dtype
            and weight.ndim == 2
            and bias.ndim == 1
            and weight.shape[1] == x.shape[-1]
            and weight.shape[0] == bias.shape[0]
            and weight.stride(-1) == 1
        ):
            try:
                x_2d = x.reshape(-1, x.shape[-1])
                hidden = torch.ops.aten._addmm_activation.default(
                    bias,
                    x_2d,
                    weight.t(),
                    beta=1,
                    alpha=1,
                    use_gelu=True,
                )
                hidden = hidden.reshape(*x.shape[:-1], weight.shape[0])
                return proj_out(hidden)
            except Exception as exc:  # pragma: no cover - runtime safety path
                _warn_once("ffn", "Disabling official KWL FFN proj_in+GELU fast path after failure: %s", exc)
    return original(x)


def _patch_feed_forward(module: FeedForward) -> None:
    if hasattr(module, "_kwl_original_forward"):
        return
    module._kwl_original_forward = module.forward
    module.forward = MethodType(_kwl_feed_forward_forward, module)


def _patch_block(module: BasicAVTransformerBlock) -> None:
    module.ada_zero_function = KWLAdaZeroFunction()


def _patch_attention(module: Attention, preattention: KWLPreAttention) -> None:
    module.preattention_function = preattention


def build_official_kwl_module_op() -> ModuleOps:
    """Return a ``ModuleOps`` installer for official LTX transformer KWL paths."""
    _require_torch()
    _require_ltx_core()

    def matcher(model: torch.nn.Module) -> bool:
        return any(isinstance(m, BasicAVTransformerBlock) for m in model.modules())

    def mutator(model: torch.nn.Module) -> torch.nn.Module:
        preattention = KWLPreAttention()
        blocks = attentions = ffs = 0
        for module in model.modules():
            if isinstance(module, BasicAVTransformerBlock):
                _patch_block(module)
                blocks += 1
            elif isinstance(module, Attention):
                _patch_attention(module, preattention)
                attentions += 1
            elif isinstance(module, FeedForward):
                _patch_feed_forward(module)
                ffs += 1
        logger.info(
            "Installed official LTX KWL ops: %d transformer blocks, %d attentions, %d FFNs",
            blocks,
            attentions,
            ffs,
        )
        return model

    return ModuleOps(name="ltx23_official_kwl_kernel_ops", matcher=matcher, mutator=mutator)


def _patch_ltx_pipelines_init() -> None:
    official_src = Path(os.environ.get("OFFICIAL_SRC", "outputs/LTX-2-official-main"))
    if not official_src.is_absolute():
        official_src = Path.cwd() / official_src
    init_file = official_src / "packages" / "ltx-pipelines" / "src" / "ltx_pipelines" / "__init__.py"
    if not init_file.exists():
        return
    text = init_file.read_text()
    if "from ltx_pipelines.a2vid_two_stage" not in text:
        return
    backup = init_file.with_suffix(init_file.suffix + ".official_bak")
    if not backup.exists():
        backup.write_text(text)
    init_file.write_text(
        '"""LTX-2 Pipelines package, local lightweight init for direct module execution."""\n\n__all__ = []\n'
    )


def install_official_kwl_pipeline() -> None:
    from ltx_pipelines.utils.blocks import DiffusionStage

    if getattr(DiffusionStage, "_official_kwl_installed", False):
        return

    original_init = DiffusionStage.__init__

    def patched_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        op = build_official_kwl_module_op()
        self._transformer_builder = self._transformer_builder.with_module_ops(
            (*self._transformer_builder.module_ops, op)
        )

    DiffusionStage.__init__ = patched_init
    DiffusionStage._official_kwl_installed = True


def main() -> None:
    _patch_ltx_pipelines_init()
    logging.basicConfig(level=logging.INFO)
    install_official_kwl_pipeline()
    from ltx_pipelines.ti2vid_two_stages_hq import main as official_main

    official_main()


if __name__ == "__main__":
    main()
