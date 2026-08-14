#!/usr/bin/env python3

import sys
import types
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sol_attention import (
    STAGE2_TAUS,
    STAGE2_TRANSFORMER_LAYERS,
    Stage2SolAttention,
)


class FakeTensor:
    def __init__(self, shape):
        self.shape = shape

    def view(self, *shape):
        return FakeTensor(shape)

    def reshape(self, *shape):
        return FakeTensor(shape)

    def contiguous(self):
        return self


class FakeAll2AllAttention:
    """Contract double: communication wrapper around local-head attention."""

    def __init__(self, original_attention):
        self.original_attention = original_attention
        self.calls = 0

    def __call__(self, q, k, v, heads):
        self.calls += 1
        return self.original_attention(q, k, v, heads)


def make_blocks(dense, count=STAGE2_TRANSFORMER_LAYERS):
    cross = object()
    blocks = [
        SimpleNamespace(
            attn1=SimpleNamespace(
                attention_function=FakeAll2AllAttention(dense)
            ),
            attn2=SimpleNamespace(attention_function=cross),
        )
        for _ in range(count)
    ]
    return blocks, cross


class SolAttentionTest(unittest.TestCase):
    def setUp(self) -> None:
        self.dense_invocations = []
        self.sol_calls = []
        self.reset_calls = 0

        def sol(q, k, v, *, tau, thresh_type, kv_splits, dense_fn):
            self.sol_calls.append(
                {
                    "tau": tau,
                    "thresh_type": thresh_type,
                    "kv_splits": kv_splits,
                }
            )
            return dense_fn(q, k, v)

        def reset():
            self.reset_calls += 1

        backend = types.ModuleType("techniques.sparse_backends.sol_attn_backend")
        backend._run_sol_attn_bthd = sol
        backend.reset_sol_attn_state = reset
        backend.get_sol_attn_stats = lambda: {
            "kernel_calls": len(self.sol_calls)
        }
        self.modules = {
            "techniques": types.ModuleType("techniques"),
            "techniques.sparse_backends": types.ModuleType(
                "techniques.sparse_backends"
            ),
            "techniques.sparse_backends.sol_attn_backend": backend,
        }

        def dense(q, k, v, heads):
            self.dense_invocations.append(heads)
            return v

        self.dense = dense
        self.q = FakeTensor((1, 64, 128))
        self.k = FakeTensor((1, 64, 128))
        self.v = FakeTensor((1, 64, 128))

    def _run_three_steps(self, adapter, blocks):
        adapter.begin_denoise()
        for _ in STAGE2_TAUS:
            for block in blocks:
                block.attn1.attention_function(self.q, self.k, self.v, 1)
        return adapter.stats()

    def test_fixed_schedule_and_dense_policy(self) -> None:
        blocks, cross = make_blocks(self.dense)
        raw_transformer = SimpleNamespace(transformer_blocks=blocks)
        transformer = SimpleNamespace(
            velocity_model=SimpleNamespace(model=raw_transformer)
        )
        wrappers = [block.attn1.attention_function for block in blocks]
        layer0_dense = wrappers[0].original_attention

        with patch.dict(sys.modules, self.modules):
            adapter = Stage2SolAttention(transformer)
            self.assertTrue(
                all(
                    block.attn1.attention_function is wrapper
                    for block, wrapper in zip(blocks, wrappers)
                )
            )
            self.assertIs(wrappers[0].original_attention, layer0_dense)
            self.assertTrue(
                all(
                    wrapper.original_attention is not self.dense
                    for wrapper in wrappers[1:]
                )
            )
            stats = self._run_three_steps(adapter, blocks)

        expected_taus = [tau for tau in STAGE2_TAUS for _ in range(47)]
        self.assertEqual([call["tau"] for call in self.sol_calls], expected_taus)
        self.assertTrue(
            all(call["thresh_type"] == "diag" for call in self.sol_calls)
        )
        self.assertTrue(
            all(call["kv_splits"] == "auto" for call in self.sol_calls)
        )
        self.assertEqual(stats["completed_steps"], 3)
        self.assertEqual(stats["dense_calls"], 3)
        self.assertEqual(stats["sol_calls"], 141)
        self.assertEqual(stats["kernel"]["kernel_calls"], 141)
        self.assertEqual(stats["dense_layers"], [0])
        self.assertEqual(stats["sol_layers"], list(range(1, 48)))
        self.assertEqual(stats["cross_attention"], "dense")
        self.assertEqual(self.reset_calls, 1)
        self.assertTrue(all(wrapper.calls == 3 for wrapper in wrappers))
        self.assertTrue(all(block.attn2.attention_function is cross for block in blocks))

    def test_resolves_four_way_sequence_parallel_model(self) -> None:
        blocks, _ = make_blocks(self.dense)
        raw_transformer = SimpleNamespace(transformer_blocks=blocks)
        transformer = SimpleNamespace(
            velocity_model=SimpleNamespace(model=raw_transformer)
        )

        with patch.dict(sys.modules, self.modules):
            adapter = Stage2SolAttention(transformer)
            stats = self._run_three_steps(adapter, blocks)

        self.assertEqual(stats["sol_calls"], 141)

    def test_patches_underlying_blocks_wrapped_by_torch_compile(self) -> None:
        blocks, _ = make_blocks(self.dense)
        compiled_wrappers = [SimpleNamespace(_orig_mod=block) for block in blocks]
        transformer = SimpleNamespace(transformer_blocks=compiled_wrappers)
        fake_torch = types.ModuleType("torch")
        disabled = []

        def disable(function):
            disabled.append(function)
            return function

        fake_torch.compiler = SimpleNamespace(disable=disable)
        modules = dict(self.modules)
        modules["torch"] = fake_torch
        with patch.dict(sys.modules, modules):
            adapter = Stage2SolAttention(
                transformer,
                isolate_sol_from_compile=True,
            )
            stats = self._run_three_steps(adapter, blocks)

        self.assertEqual(len(disabled), 1)
        self.assertEqual(stats["sol_calls"], 141)
        self.assertEqual(stats["compile_boundary"], "eager_inner_sol_callable")

    def test_rejects_non_48_layer_transformer(self) -> None:
        blocks, _ = make_blocks(self.dense, count=47)
        transformer = SimpleNamespace(transformer_blocks=blocks)

        with self.assertRaisesRegex(ValueError, "exactly 48 transformer layers"):
            Stage2SolAttention(transformer)

    def test_rejects_sol_layer_before_layer_one(self) -> None:
        blocks, _ = make_blocks(self.dense)
        transformer = SimpleNamespace(transformer_blocks=blocks)

        with patch.dict(sys.modules, self.modules):
            adapter = Stage2SolAttention(transformer)
            adapter.begin_denoise()
            with self.assertRaisesRegex(RuntimeError, "before layer 1"):
                blocks[2].attn1.attention_function(
                    self.q, self.k, self.v, 1
                )

    def test_rejects_more_than_three_steps(self) -> None:
        blocks, _ = make_blocks(self.dense)
        transformer = SimpleNamespace(transformer_blocks=blocks)

        with patch.dict(sys.modules, self.modules):
            adapter = Stage2SolAttention(transformer)
            adapter.begin_denoise()
            for _ in STAGE2_TAUS:
                blocks[1].attn1.attention_function(
                    self.q, self.k, self.v, 1
                )
            with self.assertRaisesRegex(RuntimeError, "more than"):
                blocks[1].attn1.attention_function(
                    self.q, self.k, self.v, 1
                )

    def test_requires_all2all_wrapper_without_partially_patching(self) -> None:
        blocks, _ = make_blocks(self.dense)
        wrappers = [block.attn1.attention_function for block in blocks]
        blocks[17].attn1.attention_function = self.dense
        transformer = SimpleNamespace(transformer_blocks=blocks)

        with self.assertRaisesRegex(TypeError, "layer 17"):
            Stage2SolAttention(transformer)

        self.assertTrue(
            all(
                wrapper.original_attention is self.dense
                for wrapper in wrappers
            )
        )


if __name__ == "__main__":
    unittest.main()
