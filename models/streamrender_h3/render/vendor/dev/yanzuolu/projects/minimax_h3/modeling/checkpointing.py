"""Segment-wise gradient-checkpoint helper used by the wan2_1 model forwards.

Hosts ``maybe_checkpoint``, which optionally applies (chunked) gradient checkpointing
to a module or an iterable of modules.
"""
from typing import Callable, Iterable, Union

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint


def gradient_checkpointing(module: Union[Callable, nn.Module], *args, use_reentrant, enabled: bool, **kwargs):
    if enabled:
        return checkpoint(
            module,
            *args,
            use_reentrant=use_reentrant,
            **kwargs,
        )
    else:
        return module(*args, **kwargs)


def maybe_checkpoint(module, *args, enabled=True, gc_step=1, gc_start_idx=0, use_reentrant=False, **kwargs):
    if isinstance(module, Iterable):
        def create_custom_forward_sequential(modules, start, end):
            def custom_forward(*args, **kwargs):
                for idx in range(start, end):
                    args_ = modules[idx](*args, **kwargs)
                    if not isinstance(args_, tuple):
                        args_ = (args_,)
                    assert len(args_) == len(args), "All arguments must be returned from each module in the sequential checkpointing."
                    args = args_
                return args
            return custom_forward

        # if module.training is False, we should still enable gradient checkpointing if it is not within torch.no_grad
        if enabled and torch.is_grad_enabled():
            if gc_start_idx > 0:
                args = gradient_checkpointing(
                    create_custom_forward_sequential(module, 0, gc_start_idx),
                    *args,
                    use_reentrant=use_reentrant,
                    enabled=False,
                    **kwargs
                )

            num_chunks = (len(module) - gc_start_idx - 1) // gc_step + 1
            for chunk_idx in range(num_chunks):
                start_idx = chunk_idx * gc_step + gc_start_idx
                end_idx = min((chunk_idx + 1) * gc_step + gc_start_idx, len(module))
                args = gradient_checkpointing(
                    create_custom_forward_sequential(module, start_idx, end_idx),
                    *args,
                    use_reentrant=use_reentrant,
                    enabled=enabled,
                    **kwargs
                )
            if len(args) == 1:
                args = args[0]
            return args
        else:
            for sub_module in module:
                args_ = sub_module(*args, **kwargs)
                if not isinstance(args_, tuple):
                    args_ = (args_,)
                assert len(args_) == len(args), "All arguments must be returned from each module in the sequential checkpointing."
                args = args_
            if len(args) == 1:
                args = args[0]
            return args
    else:
        def create_custom_forward(module):
            def custom_forward(*args, **kwargs):
                return module(*args, **kwargs)
            return custom_forward

        # if module.training is False, we should still enable gradient checkpointing if it is not within torch.no_grad
        if enabled and torch.is_grad_enabled():
            return gradient_checkpointing(create_custom_forward(module), *args, use_reentrant=use_reentrant, enabled=enabled, **kwargs)
        else:
            return module(*args, **kwargs)
