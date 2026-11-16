"""Regression checks for W1 candidate-helper argument forwarding.

Run with: python -B -m unittest discover -s code -p test_w1_candidates.py
"""
import unittest

import numpy as np

import wb_bary_algorithms as wb


class W1CandidateTests(unittest.TestCase):
    def setUp(self):
        self.measures = [
            wb.Measure([[0, 0], [2, 2]], [0.5, 0.5]),
            wb.Measure([[1, 1]], [1]),
        ]
        self.pooled = np.array([[0, 0], [1, 1], [2, 2]], dtype=float)
        self.grid = np.array([(x, y) for x in range(3) for y in range(3)], dtype=float)

    def test_full_dimension_enumeration_preserves_pooled_support(self):
        candidates, subsets = wb.build_W1_candidates_subspace_enumeration(self.measures, t=2)
        np.testing.assert_array_equal(candidates, self.pooled)
        self.assertEqual(subsets, [(0, 1)])

    def test_uncapped_default_enumerates_full_cartesian_product(self):
        for t in (0, 1):
            with self.subTest(t=t):
                candidates, _ = wb.build_W1_candidates_subspace_enumeration(self.measures, t=t)
                np.testing.assert_array_equal(candidates, self.grid)

    def test_sampling_defaults_forward_seed_and_no_cap(self):
        for t, expected in ((0, self.grid), (1, self.grid), (2, self.pooled)):
            with self.subTest(t=t):
                candidates, subset = wb.build_W1_candidates_subspace_sampling(self.measures, t=t, seed=7)
                np.testing.assert_array_equal(candidates, expected)
                self.assertEqual(len(subset), t)

    def test_optional_grid_cap_reaches_helper_reproducibly(self):
        # One complement coordinate is sampled for each of three selected values.
        for builder in (wb.build_W1_candidates_subspace_enumeration,
                        wb.build_W1_candidates_subspace_sampling):
            with self.subTest(builder=builder.__name__):
                candidates, _ = builder(self.measures, t=1, seed=7, max_grid=1)
                repeated, _ = builder(self.measures, t=1, seed=7, max_grid=1)
                np.testing.assert_array_equal(candidates, repeated)
                self.assertLess(len(candidates), len(self.grid))
                self.assertTrue(all(tuple(row) in set(map(tuple, self.grid)) for row in candidates))
        candidates = wb._S1_for_T(self.measures, (0,), max_grid=1, seed=7)
        self.assertEqual(len(candidates), 3)


if __name__ == "__main__":
    unittest.main()
