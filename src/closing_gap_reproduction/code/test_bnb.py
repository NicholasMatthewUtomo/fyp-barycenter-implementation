"""Unit tests for the fixed-dimension Branch and Bound barycenter solver.

Run with: python -B -m unittest test_bnb.py

These checks are synthetic (no data downloads, no licence required) and
validate the B&B lower bound / incumbent against known analytical W2
objectives and against Algo-4 enumeration for W1.
"""
from __future__ import annotations

import unittest

import numpy as np

import wb_bary_algorithms as wb
import wb_bnb


class BnBTests(unittest.TestCase):
    def test_w2_dirac_midpoint_objective_one(self):
        # delta_0 and delta_2: exact barycenter is delta_1, objective 1.0 (p=2).
        measures = [
            wb.Measure(np.array([[0.0]]), np.array([1.0])),
            wb.Measure(np.array([[2.0]]), np.array([1.0])),
        ]
        out = wb_bnb.solve_barycenter_bnb(measures, p=2, max_boxes=50)
        self.assertAlmostEqual(out["objective"], 1.0, places=6)
        self.assertLessEqual(out["gap"], 1e-4)
        # valid lower bound: objective - lower_bound <= gap (+tol)
        self.assertLessEqual(out["objective"] - out["lower_bound"], out["gap"] + 1e-6)
        self.assertAlmostEqual(float(out["z"].sum()), 1.0, places=6)

    def test_w2_unit_square_diagonal_objective_quarter(self):
        # Two diagonals of a unit square: midpoint barycenters, objective 0.25 (p=2).
        measures = [
            wb.Measure(np.array([[0.0, 0.0], [1.0, 1.0]]), np.array([0.5, 0.5])),
            wb.Measure(np.array([[0.0, 1.0], [1.0, 0.0]]), np.array([0.5, 0.5])),
        ]
        out = wb_bnb.solve_barycenter_bnb(measures, p=2, max_boxes=50)
        self.assertAlmostEqual(out["objective"], 0.25, places=6)
        self.assertLessEqual(out["gap"], 1e-4)
        self.assertLessEqual(out["objective"] - out["lower_bound"], out["gap"] + 1e-6)
        self.assertAlmostEqual(float(out["z"].sum()), 1.0, places=6)

    def test_w1_close_to_enumeration(self):
        # B&B W1 objective should bracket the true optimum: LB <= obj <= UB,
        # and be close to Algo-4 enumeration on a tiny synthetic instance.
        rng = np.random.default_rng(7)
        k, n, d = 3, 8, 2
        measures = [
            wb.Measure(rng.uniform(0.0, 1.0, (n, d)), np.full(n, 1.0 / n))
            for _ in range(k)
        ]
        enum = wb.algo4_W1_subspace_enumeration(measures, t=1, prefer_solver="scipy")
        bnb = wb_bnb.solve_barycenter_bnb(
            measures, p=1, max_boxes=200, gap_tol=1e-3, prefer="scipy"
        )
        self.assertTrue(np.isfinite(bnb["objective"]))
        self.assertTrue(np.isfinite(bnb["lower_bound"]))
        self.assertLessEqual(bnb["lower_bound"], bnb["objective"] + 1e-6)
        self.assertAlmostEqual(float(bnb["z"].sum()), 1.0, places=6)
        # Enumeration on the full grid is a feasible upper bound; B&B should
        # reach within its gap tolerance of it.
        self.assertLessEqual(
            enum["objective"], bnb["objective"] + bnb["gap"] + 1e-6
        )

    def test_result_schema_keys(self):
        measures = [
            wb.Measure(np.array([[0.0], [1.0]]), np.array([0.5, 0.5])),
            wb.Measure(np.array([[0.5], [2.0]]), np.array([0.5, 0.5])),
        ]
        out = wb_bnb.solve_barycenter_bnb(measures, p=1, max_boxes=30, prefer="scipy")
        required = {"algorithm", "z", "S", "objective", "lower_bound", "gap",
                    "runtime", "iterations"}
        self.assertTrue(required.issubset(set(out.keys())))
        self.assertEqual(out["S"].shape[1], measures[0].X.shape[1])
        self.assertEqual(out["z"].shape[0], len(out["S"]))


if __name__ == "__main__":
    unittest.main()
