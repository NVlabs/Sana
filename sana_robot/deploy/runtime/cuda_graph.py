# Copyright 2026 NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import gc

import torch
from torch.utils._pytree import tree_flatten, tree_unflatten

REPLAY_TOLERANCE_FLOOR = 1e-3
REPLAY_TOLERANCE_CEILING = 1e-1


def strided_copy(tensor: torch.Tensor) -> torch.Tensor:
    """Return a copy of a tensor with its own sizes and strides."""
    copy = torch.empty_strided(tensor.size(), tensor.stride(), dtype=tensor.dtype, device=tensor.device)
    return copy.copy_(tensor)


def leaf_signature(leaf) -> tuple:
    """Return what a captured graph depends on in one argument: a tensor's layout, or the value of anything else."""
    if isinstance(leaf, torch.Tensor):
        return ("tensor", tuple(leaf.shape), str(leaf.dtype), str(leaf.device), tuple(leaf.stride()))
    return ("value", repr(leaf))


def output_tensors(output) -> list:
    """Return the tensors of a graphed function's output, a tensor or a tuple of tensors."""
    return [output] if isinstance(output, torch.Tensor) else list(output)


class CudaGraphDriver:
    """Replays a function of tensors as a CUDA graph, one graph per layout of its arguments.

    The first calls with a layout run the function eagerly on the graph's own input buffers, on a side stream, so
    that every lazy initialization happens before the capture. The next call captures the graph; later calls copy
    the arguments into the buffers and replay it. A replay returns the graph's own output buffers, which the next
    replay overwrites. The function may only read state through its tensor arguments and through buffers that keep
    their addresses, and arguments that are not tensors must repeat.
    """

    def __init__(self, function, warmup_calls: int = 1, check_replay: bool = True) -> None:
        """Wrap a function.

        Args:
            function: The function to capture: tensors and constants in, a tensor or a tuple of tensors out.
            warmup_calls: Eager calls per layout before the capture.
            check_replay: Compare the first replay with eager runs on the same arguments.
        """
        self.function = function
        self.warmup_calls = max(1, int(warmup_calls))
        self.check_replay = check_replay
        self.entries = {}
        self.captures = 0
        self.replays = 0

    def __call__(self, *arguments):
        (leaves, spec) = tree_flatten(arguments)
        key = tuple(leaf_signature(leaf) for leaf in leaves)
        entry = self.entries.setdefault(key, {"calls": 0, "graph": None, "inputs": None, "output": None})
        entry["calls"] += 1
        if entry["graph"] is not None:
            self._copy_inputs(entry["inputs"], leaves)
            entry["graph"].replay()
            self.replays += 1
            return entry["output"]
        if entry["inputs"] is None:
            entry["inputs"] = [
                strided_copy(leaf.detach()) if isinstance(leaf, torch.Tensor) else leaf for leaf in leaves
            ]
        else:
            self._copy_inputs(entry["inputs"], leaves)

        def run():
            return self.function(*tree_unflatten(entry["inputs"], spec))

        if entry["calls"] <= self.warmup_calls:
            side_stream = torch.cuda.Stream()
            side_stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side_stream):
                output = run()
            torch.cuda.current_stream().wait_stream(side_stream)
            return output

        (reference, spread) = (None, 0.0)
        if self.check_replay:
            reference = [tensor.detach().clone() for tensor in output_tensors(run())]
            spread = self._largest_difference(output_tensors(run()), reference)
        else:
            run()
        torch.cuda.synchronize()
        torch.clear_autocast_cache()
        gc.collect()
        graph = torch.cuda.CUDAGraph()
        collecting = gc.isenabled()
        gc.disable()
        try:
            with torch.cuda.graph(graph, capture_error_mode="thread_local"):
                entry["output"] = run()
        finally:
            if collecting:
                gc.enable()
        torch.clear_autocast_cache()
        entry["graph"] = graph
        self.captures += 1
        graph.replay()
        self.replays += 1
        if self.check_replay:
            torch.cuda.synchronize()
            difference = self._largest_difference(output_tensors(entry["output"]), reference)
            if difference > min(max(4.0 * spread, REPLAY_TOLERANCE_FLOOR), REPLAY_TOLERANCE_CEILING):
                raise RuntimeError(
                    f"a CUDA graph replay differs from the eager run by {difference:.3e} "
                    f"(two eager runs differ by {spread:.3e})"
                )
        return entry["output"]

    @staticmethod
    def _copy_inputs(buffers, leaves) -> None:
        """Copy the tensor arguments of a call into the graph's input buffers."""
        for buffer, leaf in zip(buffers, leaves):
            if isinstance(buffer, torch.Tensor) and buffer.data_ptr() != leaf.data_ptr():
                buffer.copy_(leaf, non_blocking=True)

    @staticmethod
    def _largest_difference(outputs, reference) -> float:
        """Return the largest absolute difference between two lists of tensors."""
        differences = [
            float((output.float() - other.float()).abs().max())
            for (output, other) in zip(outputs, reference)
            if output.numel()
        ]
        return max(differences, default=0.0)
