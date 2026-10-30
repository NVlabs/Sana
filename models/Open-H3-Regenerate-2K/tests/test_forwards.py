"""CPU checks: the two SoL-Engine presets and which attention calls they run sparse.

    python -m unittest discover -s tests -v
"""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from open_h3 import fast_dit, sol_engine_fwd  # noqa: E402


def plan(preset):
    sparse = sol_engine_fwd.SPARSE[preset]
    return fast_dit.SparsePlan(sparse["dense_steps"], sparse["dense_layers"], sparse["tau"], sparse["thresh_type"])


def sparse_steps(p, stage, steps=3, layer=10):
    out = []
    for step in range(steps):
        p.stage, p.step = stage, step
        out.append(p.sparse(layer))
    return out


class PresetTest(unittest.TestCase):
    def test_presets_differ_only_in_refine(self):
        a, b = sol_engine_fwd.SPARSE["sol_engine"], sol_engine_fwd.SPARSE["sol_engine_fast"]
        self.assertEqual(a.keys(), b.keys())
        for key in a:
            if isinstance(a[key], dict):
                self.assertEqual(a[key]["hyperflow"], b[key]["hyperflow"], key)
            else:
                self.assertEqual(a[key], b[key], key)

    def test_sol_engine_keeps_refine_step_0_dense(self):
        p = plan("sol_engine")
        self.assertEqual(sparse_steps(p, "refine"), [False, True, True])
        p.stage = "refine"
        self.assertEqual(p.kernel_kwargs()["tau"], 1.0)

    def test_sol_engine_fast_runs_every_refine_step_sparse(self):
        p = plan("sol_engine_fast")
        self.assertEqual(sparse_steps(p, "refine"), [True, True, True])
        p.stage = "refine"
        self.assertEqual(p.kernel_kwargs()["tau"], 1.75)

    def test_hyperflow_unchanged(self):
        for preset in sol_engine_fwd.SPARSE:
            p = plan(preset)
            self.assertEqual(sparse_steps(p, "hyperflow", steps=8), [False, False] + [True] * 6)
            p.stage, p.step = "hyperflow", 5
            self.assertFalse(p.sparse(0) or p.sparse(1))


if __name__ == "__main__":
    unittest.main()
