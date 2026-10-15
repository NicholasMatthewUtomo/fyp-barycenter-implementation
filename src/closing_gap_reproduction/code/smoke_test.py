"""Small analytical W2 checks; no data downloads or Gurobi license needed.

Run: python smoke_test.py [--json PATH]
These checks validate numerical objectives and transport feasibility with SciPy.
They do not reproduce paper experiments or validate the Gurobi CG path.
"""

import argparse
import json
import platform
from pathlib import Path

import numpy as np
import scipy

from wb_bary_algorithms import (
    Measure,
    _stacked_costs,
    algo1_W2_subset_sampling,
    algo2_W2_subset_enumeration,
    solve_barycenter_lp_scipy,
)


def check_instance(name, measures, expected):
    records = []
    for t, target in expected.items():
        out = algo2_W2_subset_enumeration(measures, t=t, seed=0,
                                          prefer_solver="scipy")
        costs = _stacked_costs(out["S"], measures, p=2)
        z, plans, objective, _ = solve_barycenter_lp_scipy(costs, measures)
        assert np.isclose(objective, target, atol=1e-9), (name, t, objective, target)
        assert np.isclose(out["objective"], target, atol=1e-9)
        assert np.isclose(z.sum(), 1.0, atol=1e-9)
        assert np.min(z) >= -1e-9
        assert np.isclose(out["z"].sum(), 1.0, atol=1e-9)
        assert np.min(out["z"]) >= -1e-9
        residual = 0.0
        for mu, plan in zip(measures, plans):
            assert np.min(plan) >= -1e-9
            residual = max(residual,
                           float(np.max(np.abs(plan.sum(axis=1) - z))),
                           float(np.max(np.abs(plan.sum(axis=0) - mu.w))))
        assert residual < 1e-9, (name, t, residual)
        reconstructed = sum(float(np.sum(c * y)) for c, y in zip(costs, plans)) / len(measures)
        assert np.isclose(reconstructed, objective, atol=1e-9)
        records.append({"instance": name, "t": t, "objective": float(objective),
                        "expected_objective": target,
                        "candidate_count": len(z), "mass": float(z.sum()),
                        "max_marginal_residual": residual, "passed": True})
    # With t=k, sampling the measure subset includes every measure.
    sampled = algo1_W2_subset_sampling(measures, t=len(measures), seed=0,
                                        prefer_solver="scipy")
    assert np.isclose(sampled["objective"], expected[len(measures)], atol=1e-9)
    records.append({"instance": name, "algorithm": "subset_sampling_t_equals_k",
                    "objective": float(sampled["objective"]), "passed": True})
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", type=Path, help="Optional path for the validation report.")
    args = parser.parse_args()
    # delta_0 and delta_2: their midpoint is the exact barycenter, objective 1.
    # Restricting support to {0,2} (t=1 enumeration) gives objective 2.
    diracs = [Measure(np.array([[0.0]]), np.array([1.0])),
              Measure(np.array([[2.0]]), np.array([1.0]))]
    # The two diagonals of a unit square: every cross-measure transport edge
    # has squared length 1, so midpoint barycenters have objective 1/4.
    # Restricting to the four corners yields objective 1/2.
    square = [Measure(np.array([[0.0, 0.0], [1.0, 1.0]]), np.array([0.5, 0.5])),
              Measure(np.array([[0.0, 1.0], [1.0, 0.0]]), np.array([0.5, 0.5]))]
    checks = check_instance("two_diracs", diracs, {1: 2.0, 2: 1.0})
    checks += check_instance("unit_square_diagonals", square, {1: 0.5, 2: 0.25})
    report = {"status": "passed", "backend": "SciPy HiGHS",
              "python": platform.python_version(), "numpy": np.__version__,
              "scipy": scipy.__version__, "checks": checks,
              "limitations": ["Gurobi CG path was not exercised.",
                              "Paper-scale experiments were not run.",
                              "Original W1 pathways were not repaired or tested."]}
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
