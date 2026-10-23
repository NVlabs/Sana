# SPDX-License-Identifier: Apache-2.0
"""The peft LoRA layer for MiniMax-H3's projections.

H3's projections are ``_Linear``, the shim standing in for sglang's
tensor-parallel layers. peft cannot dispatch on it -- ``Target module _Linear()
is not supported`` -- so ``adapter.custom_module_mapping`` names this class as
its handler.

Only one thing here is H3's: ``_Linear.forward`` returns sglang's
``(output, bias)`` pair, while peft's ``Linear.forward`` does
``result = self.base_layer(x)`` and then ``result.dtype``. Everything else --
creating lora_A/lora_B, initialization, scaling, dropout, merge/unmerge,
state-dict naming -- is peft's and is not reimplemented here. Dimensions need no
help either: peft already infers them from ``input_size``/``output_size``, which
is exactly what the shim exposes.
"""

from __future__ import annotations

from typing import Any

import torch
from peft.tuners.lora.layer import Linear as PeftLoraLinear


class MiniMaxH3LoraLinear(PeftLoraLinear):
    """peft's LoRA ``Linear`` against a base layer that returns a pair."""

    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any):
        """Transcribes ``peft.tuners.lora.layer.Linear.forward``.

        Kept line-for-line with upstream so the divergence is visibly just the
        unpacking; if a peft upgrade changes that method, this needs the same
        change, and ``test_minimax_h3_lora.py`` pins the arithmetic that would
        otherwise drift silently.
        """
        self._check_forward_args(x, *args, **kwargs)
        adapter_names = kwargs.pop("adapter_names", None)
        assert adapter_names is None, (
            "mixed-batch adapter dispatch is not supported for H3: peft's "
            "_mixed_batch_forward indexes into the base layer's output, which is "
            "a (tensor, bias) pair here"
        )

        if self.disable_adapters:
            if self.merged:
                self.unmerge()
            result, bias = self.base_layer(x, *args, **kwargs)
        elif self.merged:
            result, bias = self.base_layer(x, *args, **kwargs)
        else:
            result, bias = self.base_layer(x, *args, **kwargs)
            torch_result_dtype = result.dtype

            lora_A_keys = self.lora_A.keys()
            for active_adapter in self.active_adapters:
                if active_adapter not in lora_A_keys:
                    continue
                assert active_adapter not in self.lora_variant, (
                    f"LoRA variant {self.lora_variant[active_adapter]} is not "
                    "supported for H3; only vanilla LoRA is transcribed here"
                )

                lora_A = self.lora_A[active_adapter]
                lora_B = self.lora_B[active_adapter]
                dropout = self.lora_dropout[active_adapter]
                scaling = self.scaling[active_adapter]
                x = self._cast_input_dtype(x, lora_A.weight.dtype)
                result = result + lora_B(lora_A(dropout(x))) * scaling

            result = result.to(torch_result_dtype)

        # The shim's second element is always None -- upstream uses it to defer
        # the bias add -- but return what the base layer returned rather than
        # hard-coding that, so a base layer that does defer still composes.
        return result, bias


EntryClass = MiniMaxH3LoraLinear

__all__ = ["MiniMaxH3LoraLinear"]
