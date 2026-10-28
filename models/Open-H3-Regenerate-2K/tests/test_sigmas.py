"""CPU checks: the refine sigma grids.

    python -m unittest discover -s tests -v
"""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from open_h3.sigmas import SCHEDULES, shifted_grid  # noqa: E402


class SigmasTest(unittest.TestCase):
    def test_rule_reproduces_original_4_step_grid(self):
        expected = [0.909375, 0.841446, 0.732075, 0.526697, 0.0]
        for got, want in zip(shifted_grid(0.909375, 4), expected):
            self.assertAlmostEqual(got, want, places=5)

    def test_two_step_grid(self):
        grid = SCHEDULES[2]
        self.assertEqual(len(grid), 3)
        self.assertEqual(grid[0], 0.7)
        self.assertAlmostEqual(grid[1], 0.84 / 1.7, places=12)
        self.assertEqual(grid[-1], 0.0)

    def test_three_step_grid_unchanged(self):
        self.assertEqual(SCHEDULES[3], [0.909375, 6 / 7, 2 / 3, 0.0])

    def test_grids_decrease_to_zero(self):
        for grid in SCHEDULES.values():
            self.assertTrue(all(a > b for a, b in zip(grid, grid[1:])))
            self.assertEqual(grid[-1], 0.0)


if __name__ == "__main__":
    unittest.main()
