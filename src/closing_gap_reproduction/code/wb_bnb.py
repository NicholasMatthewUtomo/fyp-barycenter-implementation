"""Fixed-dimension spatial Branch and Bound solver for discrete Wasserstein barycenters.

Replaces static grid enumeration with adaptive spatial partitioning of the
bounding hyper-rectangle. Each iteration:

  1) builds optimistic (lower-bound) cost matrices from the active boxes,
  2) solves the fixed-support barycenter LP  -> global lower bound,
  3) evaluates the exact cost at box midpoints  -> incumbent upper bound,
  4) prunes zero-mass boxes, branches on the largest mass-weighted box.

The optimistic cost from an input support point x to a box B is the minimum
L_p^p distance from x to any point inside B, which makes the LP relaxation a
mathematically valid global lower bound on the optimal barycenter objective.

Returns a dict matching the wb_bary_algorithms result schema:
    {'algorithm', 'z', 'S', 'objective', 'lower_bound', 'gap', 'runtime', 'iterations'}
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Union

import math
import time

import numpy as np

from wb_bary_algorithms import Measure, _stacked_costs, solve_barycenter_lp


@dataclass
class HyperBox:
    low: np.ndarray   # shape (d,)
    high: np.ndarray  # shape (d,)

    @property
    def midpoint(self) -> np.ndarray:
        return 0.5 * (self.low + self.high)

    @property
    def diameter(self) -> float:
        return float(np.max(self.high - self.low))


def optimistic_box_cost(boxes: List[HyperBox], X: np.ndarray, p: int = 1) -> np.ndarray:
    """Computes an (m, n) lower-bound cost matrix from m boxes to n points in X.

    C[l, j] = min_{s in B_l} || s - x_j ||_p^p, a valid underestimate of the
    true cost for every support point inside box B_l.
    """
    lows = np.array([b.low for b in boxes])[:, None, :]    # (m, 1, d)
    highs = np.array([b.high for b in boxes])[:, None, :]  # (m, 1, d)
    pts = X[None, :, :]                                    # (1, n, d)
    diff = np.maximum(0.0, np.maximum(lows - pts, pts - highs))
    if p == 1:
        return np.sum(diff, axis=-1)
    elif p == 2:
        return np.sum(diff ** 2, axis=-1)
    raise ValueError(f"Unsupported p={p}")


def _bounding_box(measures: List[Measure]) -> HyperBox:
    all_X = np.vstack([m.X for m in measures])
    return HyperBox(low=all_X.min(axis=0), high=all_X.max(axis=0))


def solve_barycenter_bnb(
    measures: List[Measure],
    p: int = 1,
    max_boxes: int = 500,
    gap_tol: float = 1e-4,
    time_limit_sec: float = 300.0,
    prefer: str = "mosek",
    params: Optional[Dict[str, Union[int, float]]] = None,
) -> dict:
    """
    Fixed-dimension spatial Branch-and-Bound solver for discrete Wasserstein
    barycenters.

    Args:
        measures: input measures P_i = (X_i, w_i), all in R^d.
        p: ground-metric exponent (1 for W1, 2 for W2).
        max_boxes: stop refining once this many boxes are active.
        gap_tol: stop when UB* - LB <= gap_tol.
        time_limit_sec: wall-clock time limit.
        prefer: LP solver passed to solve_barycenter_lp ("mosek", "gurobi", "scipy").
        params: optional solver parameter dict.

    Returns:
        dict with keys algorithm, z, S, objective, lower_bound, gap, runtime, iterations.
    """
    t0 = time.perf_counter()
    d = measures[0].X.shape[1]

    boxes: List[HyperBox] = [_bounding_box(measures)]

    UB_star = math.inf
    UB_star_S: Optional[np.ndarray] = None
    UB_star_z: Optional[np.ndarray] = None
    best_LB = -math.inf
    iterations = 0

    while len(boxes) < max_boxes:
        if time.perf_counter() - t0 > time_limit_sec:
            break
        iterations += 1

        # --- Lower bound: optimistic LP over the active boxes -------------
        C_list = [optimistic_box_cost(boxes, m.X, p=p) for m in measures]
        z, _y_list, lb_obj, _model = solve_barycenter_lp(
            C_list, measures, prefer=prefer, params=params
        )
        best_LB = float(lb_obj)

        active = z > 1e-9
        if not np.any(active):
            break

        # --- Upper bound: exact LP evaluated at active-box midpoints ------
        S_rep = np.stack([boxes[i].midpoint for i in range(len(boxes)) if active[i]])
        z_rep = z[active]
        z_rep = z_rep / z_rep.sum()
        C_exact_list = _stacked_costs(S_rep, measures, p=p)
        z2, _y2, ub_obj, _model2 = solve_barycenter_lp(
            C_exact_list, measures, prefer=prefer, params=params
        )
        if float(ub_obj) < UB_star:
            UB_star = float(ub_obj)
            UB_star_S = S_rep
            UB_star_z = z2

        gap = UB_star - best_LB if UB_star < math.inf else math.inf
        if gap <= gap_tol:
            break

        # --- Prune zero-mass boxes ---------------------------------------
        boxes = [b for i, b in enumerate(boxes) if active[i]]
        if not boxes:
            break
        masses = z[active]
        diam = np.array([b.diameter for b in boxes])

        # --- Branch: split the largest mass-weighted box ------------------
        idx = int(np.argmax(masses * diam))
        box = boxes[idx]
        r_star = int(np.argmax(box.high - box.low))
        mid = float(box.midpoint[r_star])

        left = HyperBox(low=box.low.copy(), high=box.high.copy())
        right = HyperBox(low=box.low.copy(), high=box.high.copy())
        left.high[r_star] = mid
        right.low[r_star] = mid
        boxes[idx] = left
        boxes.insert(idx + 1, right)

    runtime = float(time.perf_counter() - t0)
    gap = UB_star - best_LB if UB_star < math.inf else math.inf

    if UB_star_S is None:
        # No incumbent found: fall back to uniform weights over final boxes.
        S_out = np.stack([b.midpoint for b in boxes])
        z_out = np.full(len(boxes), 1.0 / len(boxes))
        obj_out = best_LB
    else:
        S_out = UB_star_S
        z_out = UB_star_z
        obj_out = UB_star

    return dict(
        algorithm=("BnB_W1_subspace" if p == 1 else "BnB_W2_subset"),
        z=z_out,
        S=S_out,
        objective=float(obj_out),
        lower_bound=float(best_LB),
        gap=float(gap),
        runtime=runtime,
        iterations=iterations,
    )
