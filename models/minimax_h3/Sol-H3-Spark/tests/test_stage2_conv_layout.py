"""Pure mock checks for the isolated native-code/global-binding layout seam."""
from contextlib import contextmanager
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

SOURCE = Path(__file__).resolve().parents[1] / "runtime/stage2_ops/conv_direct_nhwc.py"
spec = importlib.util.spec_from_file_location("direct_nhwc_test", SOURCE)
entry = importlib.util.module_from_spec(spec)
spec.loader.exec_module(entry)


class Tensor:
    shape = (1, 2, 3, 4, 5)

    def stride(self):
        return (120, 1, 40, 10, 2)

    def permute(self, *axes):
        assert axes == (0, 4, 1, 2, 3)
        return self

    def __getitem__(self, index):
        assert index == (slice(None), slice(None), slice(1, None), slice(None), slice(None))
        return self

    def is_contiguous(self, *, memory_format):
        assert memory_format == "NHWC"
        return True


@contextmanager
def fixture(pattern=None):
    events = []
    marker = object()
    def original(block, x, causal, prefer_channels_last_3d=False, *, default_marker=marker):
        events.append((causal, prefer_channels_last_3d, default_marker))
        x = rearrange(x, pattern or entry.NATIVE_PATTERN, p1=2, p2=2, p3=2)
        return x[:, :, 1:, :, :]
    def native_rearrange(x, pattern, **axes):
        events.append((pattern, axes))
        return x
    efficient = SimpleNamespace(__file__=original.__code__.co_filename,
        _upsample_forward_efficient=original, rearrange=native_rearrange)
    with patch.dict(original.__globals__, {"rearrange": native_rearrange}), \
         patch.dict(sys.modules, {"ltx_core.model.video_vae.memory_efficient_decode": efficient}):
        yield efficient, original, native_rearrange, events, marker


class DirectNHWCTests(unittest.TestCase):
    def test_private_globals_preserve_defaults_closure_native_slicing_and_restore(self):
        with fixture() as (module, original, rearrange, events, marker):
            with entry.installed_direct_upsample_nhwc(SimpleNamespace(channels_last_3d="NHWC")) as receipt:
                for _ in range(5):
                    result = module._upsample_forward_efficient(SimpleNamespace(residual=False), Tensor(), False, True)
                self.assertIsInstance(result, Tensor)
                self.assertIs(module.rearrange, rearrange)
                self.assertIs(original.__globals__["rearrange"], rearrange)
                self.assertEqual(events[:2], [(False, True, marker),
                    (entry.NHWC_PATTERN, {"p1": 2, "p2": 2, "p3": 2})])
                self.assertEqual((receipt["calls"], receipt["layout_changes"]), (5, 5))
                self.assertEqual(len(receipt["stride_examples"]), 4)
                self.assertTrue(receipt["all_outputs_channels_last_3d"])
            self.assertIs(module._upsample_forward_efficient, original)
            self.assertTrue(receipt["restored"])

    def test_unsupported_branches_refused_before_native_work(self):
        with fixture() as (module, original, _, events, _):
            with entry.installed_direct_upsample_nhwc(SimpleNamespace(channels_last_3d="NHWC")) as receipt:
                for residual, causal, prefer, shape in ((True, False, True, (1, 2, 3, 4, 5)),
                        (False, True, True, (1, 2, 3, 4, 5)), (False, False, False, (1, 2, 3, 4, 5)),
                        (False, False, True, (2, 2, 3, 4, 5))):
                    value = Tensor()
                    value.shape = shape
                    with self.assertRaisesRegex(RuntimeError, "requires B1"):
                        module._upsample_forward_efficient(SimpleNamespace(residual=residual), value, causal, prefer)
                self.assertEqual(events, [])
                self.assertEqual(receipt["calls"], 0)
            self.assertIs(module._upsample_forward_efficient, original)

    def test_pattern_failure_restores_original_binding(self):
        with fixture("unexpected") as (module, original, _, events, _):
            with self.assertRaisesRegex(RuntimeError, "unexpected native pixel-shuffle"):
                with entry.installed_direct_upsample_nhwc(SimpleNamespace(channels_last_3d="NHWC")) as receipt:
                    module._upsample_forward_efficient(SimpleNamespace(residual=False), Tensor(), False, True)
            self.assertIs(module._upsample_forward_efficient, original)
            self.assertTrue(receipt["restored"])
            self.assertEqual(len(events), 1)


if __name__ == "__main__":
    unittest.main()
