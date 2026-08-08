import os
from pathlib import Path
import sys
import unittest
from unittest import mock

import torch

RUNTIME_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RUNTIME_ROOT))

import adapter  # noqa: E402
import teacache  # noqa: E402


def _context(step: int, epoch: str = "request-0") -> adapter.StepContext:
    return adapter.StepContext(
        request_epoch=epoch,
        request_index=0,
        step_index=step,
        total_tokens=64,
        live_tokens=64,
        text_start=0,
        text_tokens=8,
        cu_seqlens_host=(0, 64),
        num_layers=50,
        timestep_max=float(50 - step),
    )


class MiniMaxH3TeaCacheTests(unittest.TestCase):
    def test_accumulates_modulated_input_l1_and_reuses_residual(self) -> None:
        controller = teacache._TeaCacheController(
            teacache.TeaCacheConfig(
                threshold=0.10,
                retain_steps=1,
                cooldown_steps=0,
                coefficients=(1.0, 0.0),
            )
        )
        with mock.patch.dict(
            os.environ, {"H3_TEACACHE_NUM_FORWARDS": "6"}
        ), mock.patch.object(teacache, "emit_event"):
            hidden, action = controller.before_blocks(
                torch.zeros(4), torch.ones(4), context=_context(0)
            )
            self.assertEqual(action, "compute")
            controller.after_blocks(hidden + 2.0)

            hidden, action = controller.before_blocks(
                torch.full((4,), 0.1),
                torch.full((4,), 1.01),
                context=_context(1),
            )
            self.assertEqual(action, "reuse")
            torch.testing.assert_close(hidden, torch.full((4,), 2.1))

            _, action = controller.before_blocks(
                torch.full((4,), 0.2),
                torch.full((4,), 1.05),
                context=_context(2),
            )
            self.assertEqual(action, "reuse")

            hidden, action = controller.before_blocks(
                torch.full((4,), 0.3),
                torch.full((4,), 1.11),
                context=_context(3),
            )
            self.assertEqual(action, "compute")
            controller.after_blocks(hidden + 3.0)

        self.assertEqual(controller.compute, 2)
        self.assertEqual(controller.reuse, 2)

    def test_disabled_path_is_identity(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=True):
            hidden = torch.randn(2, 3)
            output, action = teacache.before_blocks(
                hidden, torch.randn(2, 3), context=_context(0)
            )
        self.assertIs(output, hidden)
        self.assertIsNone(action)


if __name__ == "__main__":
    unittest.main()
