"""Bounded CPU metadata/ownership/factory tests; no Torch or GPU import."""
import ast
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "runtime/stage2_ops/single_gpu_all2all.py"
spec = importlib.util.spec_from_file_location("single_gpu_all2all", SOURCE)
entry = importlib.util.module_from_spec(spec)
spec.loader.exec_module(entry)


class Tensor:
    def __init__(self, *, shape=(1, 16128, 32, 128), dtype="bf16", device="cuda",
                 index=0, contiguous=True, requires_grad=False):
        self.shape, self.ndim, self.dtype = shape, len(shape), dtype
        self.device = SimpleNamespace(type=device, index=index)
        self.contiguous, self.requires_grad = contiguous, requires_grad

    def is_contiguous(self):
        return self.contiguous

    def numel(self):
        value = 1
        for dim in self.shape:
            value *= dim
        return value

    def clone(self):
        return Tensor(shape=self.shape, dtype=self.dtype)


def fixture(**updates):
    kwargs = dict(rank=0, world_size=1, seqlen=65536, hidden_dim=4096,
                  num_sms=48, tensor_dtype="bf16", torch_module=SimpleNamespace(bfloat16="bf16"))
    kwargs.update(updates)
    obj = entry.SingleGPUAll2All(**kwargs)
    obj.set_rank_tokens([16128])
    return obj


class SingleGPUAll2AllTests(unittest.TestCase):
    def test_exact_passthrough_and_owned_copy(self):
        obj = fixture()
        for dim in (64, 128):
            for method in (obj.send_recv_heads, obj.gather_heads):
                x = Tensor(shape=(1, 16128, 32, dim))
                before = dict(obj.__dict__)
                self.assertIs(method(x), x)
                owned = method(x, copy_out=True)
                self.assertIsNot(owned, x)
                self.assertEqual(owned.shape, x.shape)
                self.assertEqual(obj.__dict__, before)
        self.assertEqual(obj.buffer_size, 536870912)
        self.assertFalse(hasattr(obj, "runtime"))

    def test_refuses_unsupported_tensors(self):
        obj = fixture()
        for changes in ({"shape": (1, 16128, 4096)}, {"contiguous": False},
                        {"device": "cpu"}, {"index": 1}, {"dtype": "fp16"},
                        {"requires_grad": True}, {"shape": (2, 16128, 32, 128)},
                        {"shape": (1, 16127, 32, 128)}, {"shape": (1, 16128, 16, 128)},
                        {"shape": (1, 16128, 32, 32)}):
            with self.subTest(changes=changes), self.assertRaisesRegex(ValueError, "REFUSE"):
                obj.send_recv_heads(Tensor(**changes))
        with self.assertRaisesRegex(ValueError, "REFUSE"):
            obj.gather_heads(Tensor(), copy_out=1)

    def test_metadata_timeout_destroy_and_constructor_guards(self):
        for changes in ({"rank": 1}, {"world_size": 2}, {"hidden_dim": 2048},
                        {"tensor_dtype": "fp16"}, {"seqlen": 0}, {"num_sms": 0}):
            with self.subTest(changes=changes), self.assertRaisesRegex(ValueError, "REFUSE"):
                fixture(**changes)
        obj = fixture()
        obj.rank_tokens = None
        with self.assertRaisesRegex(RuntimeError, "REFUSE"):
            obj.send_recv_heads(Tensor())
        for tokens in ([], [0], [-1], [65537], [True], [1, 2]):
            with self.assertRaisesRegex(ValueError, "REFUSE"):
                obj.set_rank_tokens(tokens)
        obj.set_rank_tokens([16128])
        for timeout in (-1, float("nan"), float("inf")):
            with self.assertRaisesRegex(ValueError, "REFUSE"):
                obj.set_timeout_seconds(timeout)
        obj.set_timeout_seconds(600.0)
        self.assertIs(obj.send_recv_heads(Tensor()).dtype, obj.tensor_dtype)
        obj.destroy()
        obj.destroy()
        with self.assertRaisesRegex(RuntimeError, "REFUSE"):
            obj.gather_heads(Tensor())

    def test_context_restores_and_delegates_multi_gpu(self):
        calls = []
        def original(**kwargs):
            calls.append(kwargs)
            return "original"
        kernels = SimpleNamespace(All2All=original)
        torch = SimpleNamespace(bfloat16="bf16")
        kwargs = dict(rank=0, world_size=1, seqlen=65536, hidden_dim=4096,
                      num_sms=48, tensor_dtype="bf16", group="group", timeout_seconds=20)
        with entry.single_gpu_all2all_factory(torch_module=torch, kernels_module=kernels) as receipt:
            for _ in range(4):
                self.assertIsInstance(kernels.All2All(**kwargs), entry.SingleGPUAll2All)
            multi = dict(kwargs, rank=1, world_size=2, tensor_dtype="fp16")
            self.assertEqual(kernels.All2All(**multi), "original")
            self.assertEqual(calls, [multi])
        self.assertIs(kernels.All2All, original)
        self.assertEqual(receipt["single_gpu_instances"], 4)
        self.assertEqual(receipt["original_data_capacity_bytes_elided"], 2147483648)
        self.assertEqual(receipt["delegated_instances"], 1)
        self.assertTrue(receipt["factory_restored"])
        with self.assertRaisesRegex(RuntimeError, "test"):
            with entry.single_gpu_all2all_factory(torch_module=torch, kernels_module=kernels):
                raise RuntimeError("test")
        self.assertIs(kernels.All2All, original)



if __name__ == "__main__":
    unittest.main()
