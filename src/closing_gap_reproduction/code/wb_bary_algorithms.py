
# --- wb_bary_algorithms.py ---------------------------------------------------
"""
Implementations of four discrete Wasserstein barycenter algorithms (finite-support inputs).

Algorithms correspond to:
- Algo 1 (W2, subset sampling): choose random T ⊆ {1..k} with |T|=t; candidate S_T are centroids
  of one support point from each P_i, i ∈ T. Solve LP(S_T, p=2).
- Algo 2 (W2, subset enumeration): union S_t over all T of size t. Solve LP(S_t, 2).
- Algo 3 (W1, subspace sampling): choose random coordinate subset T ⊆ {1..d}, |T|=t; candidate
  S_1^(T) = A_T × ∏_{ℓ∉T} M_ℓ with A_T = {x_T : x∈U}, M_ℓ = {x_ℓ : x∈U}. Solve LP(S_1^(T), 1).
- Algo 4 (W1, subspace enumeration): union over all coordinate T of size t. Solve LP(S_1^(t), 1).

The fixed-support barycenter LP:
  min (1/k) ∑_i ⟨C_i, y_i⟩
  s.t.  y_i 1 = z,  y_i^T 1 = w_i,  z ≥ 0, y_i ≥ 0,  ∑ z = 1.

Where C_i[ℓ,j] = || s_ℓ - x_{ij} ||_p^p for p∈{1,2}.

Dependencies:
- numpy
- gurobipy (optional)
- scipy (optional)

Author: ChatGPT
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Sequence, Tuple, Optional, Dict, Union
import itertools, math, time
import numpy as np

# Optional solvers
try:
    import gurobipy as gp
    from gurobipy import GRB
    _HAS_GUROBI = True
except Exception:
    _HAS_GUROBI = False

try:
    import mosek
    _HAS_MOSEK = True
except Exception:
    _HAS_MOSEK = False

try:
    from scipy.optimize import linprog
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False


# ----------------------------- Data structure --------------------------------

@dataclass
class Measure:
    X: np.ndarray  # (n,d) support coordinates
    w: np.ndarray  # (n,) nonnegative weights summing to 1

    def __post_init__(self):
        self.X = np.asarray(self.X, dtype=np.float64)
        self.w = np.asarray(self.w, dtype=np.float64).reshape(-1)
        assert self.X.ndim == 2, "X must be (n,d)"
        assert self.w.ndim == 1 and self.w.shape[0] == self.X.shape[0], "w length must equal n"
        s = float(self.w.sum())
        if s <= 0: raise ValueError("weights must sum to positive")
        if not np.isclose(s, 1.0): self.w = self.w / s


# ----------------------------- Utilities -------------------------------------

def _dedup_rows(X: np.ndarray, decimals: int = 12) -> np.ndarray: ## remove duplicate rows
    Xr = np.round(np.asarray(X, dtype=np.float64), decimals)
    view = np.ascontiguousarray(Xr).view([('', Xr.dtype)] * Xr.shape[1])
    uniq = np.unique(view)
    return uniq.view(Xr.dtype).reshape(-1, Xr.shape[1])

def _pairwise_cost(X: np.ndarray, Y: np.ndarray, p: int) -> np.ndarray: ## pairwise cost between data in X and data in Y
    assert p in (1,2)
    X = np.asarray(X, float); Y = np.asarray(Y, float)
    if p == 2:
        XX = (X**2).sum(1)[:,None]
        YY = (Y**2).sum(1)[None,:]
        C = XX + YY - 2*X@Y.T
        C[C<0] = 0.0
        return C
    else:
        diffs = X[:,None,:] - Y[None,:,:]
        return np.abs(diffs).sum(axis=2)

def _stacked_costs(S: np.ndarray, measures: Sequence[Measure], p: int) -> List[np.ndarray]: ## pairwise cost between data in S and each measure
    return [_pairwise_cost(S, m.X, p=p) for m in measures]

def _union_support_points(measures: Sequence[Measure]) -> np.ndarray:
    X = np.vstack([m.X for m in measures]) if measures else np.empty((0,0))
    return _dedup_rows(X)

def _all_subsets_of_size(n: int, t: int):
    return itertools.combinations(range(n), t)

def _all_coordinate_subsets(d: int, t: int):
    return itertools.combinations(range(d), t)

def _nearest_indices(X: np.ndarray, c: np.ndarray, k: int) -> np.ndarray:
    # returns indices of k nearest points in X to center c (squared L2)
    dif = X - c[None,:]
    d2 = np.einsum('ij,ij->i', dif, dif)
    k = min(k, X.shape[0])
    idx = np.argpartition(d2, k-1)[:k]
    # sort those k by actual distance
    idx = idx[np.argsort(d2[idx])]
    return idx

def select_active_atoms(S_init: np.ndarray, z_init: np.ndarray, thr: float = 1e-12) -> np.ndarray:
    S_init = np.asarray(S_init, np.float64)
    z_init = np.asarray(z_init,  np.float64).reshape(-1)
    assert S_init.shape[0] == z_init.shape[0], "S_init and z_init length mismatch"
    return S_init[z_init > thr]



# ----------------------------- LP solvers ------------------------------------

# C_list: pairwise costs between 
def solve_barycenter_lp_gurobi(C_list: List[np.ndarray], measures: Sequence[Measure],
                               params: Optional[Dict[str,Union[int,float]]] = None):
    if not _HAS_GUROBI: raise ImportError("gurobipy not available")
    k = len(C_list); s = C_list[0].shape[0]; n_list = [Ci.shape[1] for Ci in C_list]
    for Ci, m in zip(C_list, measures):
        assert Ci.shape == (s, m.X.shape[0])

    model = gp.Model("bary_lp")
    if params:
        for kpar,vpar in params.items(): model.setParam(kpar, vpar)

    y_vars = [model.addMVar((s, n_i), lb=0.0, name=f"y{i}") for i,n_i in enumerate(n_list)]
    z = model.addMVar(s, lb=0.0, name="z")

    for i, Yi in enumerate(y_vars):
        model.addConstr(Yi.sum(axis=1) == z, name=f"match_z_{i}")
    for i, (Yi, meas) in enumerate(zip(y_vars, measures)):
        model.addConstr(Yi.sum(axis=0) == meas.w, name=f"match_w_{i}")
    model.addConstr(z.sum() == 1.0, name="norm")

    obj = gp.LinExpr()
    for i, (Ci, Yi) in enumerate(zip(C_list, y_vars)):
        obj += gp.quicksum(Ci[l,j]*Yi[l,j] for l in range(s) for j in range(n_list[i]))
    model.setObjective((1.0/k)*obj, GRB.MINIMIZE)
    model.optimize()
    if model.status != GRB.OPTIMAL:
        raise RuntimeError(f"Gurobi status {model.status}")
    z_val = z.X.copy(); y_val = [Yi.X.copy() for Yi in y_vars]
    return z_val, y_val, model.ObjVal, model

def solve_barycenter_lp_cg_persistent(
    C_list, measures, *,
    params=None,
    init_L=None,              # seed rows (indices in [0..s-1])
    add_batch=1,
    rc_tol=1e-9,
    time_limit_sec=100,
    max_iter=10**9,
    verbose=False
):
    k = len(C_list)
    s = C_list[0].shape[0]
    n_list = [Ci.shape[1] for Ci in C_list]
    for Ci, m in zip(C_list, measures):
        assert Ci.shape == (s, m.X.shape[0])

    start = time.perf_counter()
    def rem_time():
        if time_limit_sec is None: return None
        return max(0.0, time_limit_sec - (time.perf_counter() - start))

    # --- Build initial model on L0 once -------------------------------------
    if init_L:
        L = list(dict.fromkeys(int(l) for l in init_L if 0 <= l < s))
    else:
        L = sorted({int(np.argmin(Ci.mean(axis=1))) for Ci in C_list}) #pick the candidate from each measure with the minimum average distance to others

    m = gp.Model("bary_lp_cg_rmp_persist")
    if params:
        for p, v in params.items():
            m.setParam(p, v)

    # Per-solve remaining time
    rt = rem_time()
    if rt is not None:
        m.Params.TimeLimit = max(0.01, rt)

    # Variables kept in lists/dicts so we can append
    # z_vars[ell] corresponds to global row index L[ell]
    z_vars = []
    # Y_vars[i] is a list of row vectors (gurobi MVar of size n_i), one per ell in L
    Y_vars = [ [] for _ in range(k) ]

    # Constraints we need duals from:
    # row_match_constrs[i][ell]: sum_j Y_i[ell,j] == z[ell]
    row_match_constrs = [ [] for _ in range(k) ]
    # col_match_constrs[i][j]: sum_ell Y_i[ell,j] == w_i[j]
    col_match_constrs = [ [] for _ in range(k) ]
    # normalization
    norm_constr = None

    # Create column- (j) constraints once (they don’t change when we add rows)
    for i, meas in enumerate(measures):
        for j in range(n_list[i]):
            c = m.addConstr(gp.LinExpr() == float(meas.w[j]),
                            name=f"col_i{i}_j{j}")
            col_match_constrs[i].append(c)

    # Create normalization constraint once; RHS=1
    norm_constr = m.addConstr(gp.LinExpr() == 1.0, name="norm")

    # Objective accumulator (linear)
    obj = gp.LinExpr()

    # Helper: add one new global row index l to the model
    def add_atom(l):
        # create z_l
        z_l = m.addVar(lb=0.0, name=f"z[{l}]")
        z_vars.append(z_l)
        # z_l enters the normalization constraint with coeff +1
        m.chgCoeff(norm_constr, z_l, 1.0)

        # for each measure i, create Y_i[l, :]
        for i, Ci in enumerate(C_list):
            Yi_row = m.addMVar(n_list[i], lb=0.0, name=f"Y{i}[{l},:]")
            Y_vars[i].append(Yi_row)

            # row constraint: sum_j Y_i[l,j] == z_l
            c_row = m.addConstr(Yi_row.sum() - z_l == 0.0, name=f"row_i{i}_l{l}")
            row_match_constrs[i].append(c_row)

            # col constraints: each Y_i[l,j] contributes +1 to the column balance
            for j in range(n_list[i]):
                m.chgCoeff(col_match_constrs[i][j], Yi_row[j], 1.0)
                Yi_row[j].Obj=float(Ci[l,j]/k)
            # objective contributions: (1/k) * sum_j C_i[l,j] * Y_i[l,j]
            #obj.addTerms(Ci[l, :].tolist(), Yi_row.tolist())
        
    m.ModelSense = GRB.MINIMIZE
    m.setObjective(0.0)

    #m.Params.Method = 1  # dual simplex

    best = {"L": None, "zL": None, "YL": None, "obj": np.nan}

    def snapshot_incumbent():
        if m.SolCount > 0:
            best["L"] = list(L)
            best["zL"] = [v.X for v in z_vars]
            best["YL"] = [[row.X for row in rows] for rows in Y_vars]
            best["obj"] = m.ObjVal

    
    # Seed atoms
    for l in L:
        add_atom(l)

    # Set objective once
    #m.setObjective((1.0 / k) * obj, GRB.MINIMIZE)
    m.update()

    if verbose:
        print(f"[CG] persistent start |L|={len(L)} of s={s}")

    # --- Main CG loop (persistent) ------------------------------------------
    def extract_duals():
        # alpha_i: length-s array with zeros outside current L
        alpha = [np.zeros(s) for _ in range(k)]
        for i in range(k):
            for ell, l in enumerate(L):
                alpha[i][l] = row_match_constrs[i][ell].Pi
        # beta_i: length n_i
        beta = [np.array([c.Pi for c in col_match_constrs[i]]) for i in range(k)]
        gamma = norm_constr.Pi
        return alpha, beta, gamma

    # def price(alpha, beta, gamma, Lset):
    #     inL = np.zeros(s, dtype=bool); inL[list(Lset)] = True
    #     # m_i_min[l] = min_j( (1/k)*C_i[l,j] - beta_i[j] )
    #     m_i_min = []
    #     for i, Ci in enumerate(C_list):
    #         mat = (Ci / k) - beta[i][None, :]
    #         m_i_min.append(mat.min(axis=1))
    #     rc = gamma + sum(m_i_min) - sum(ai for ai in alpha)
    #     rc[np.where(inL)] = np.inf
    #     # pick up to add_batch best negative
    #     order = np.argsort(rc)[:add_batch]
    #     picks = [(int(l), float(rc[l])) for l in order if rc[l] < -rc_tol]
    #     best = np.min(rc[~np.isinf(rc)]) if np.any(~np.isinf(rc)) else np.nan
    #     return picks, best

    def price(beta, gamma, Lset):
        inL = np.zeros(s, dtype=bool); inL[list(Lset)] = True
    
        # m_i_min[l] = min_j( (1/k)*C_i[l,j] - beta_i[j] )
        m_i_min = []
        for i, Ci in enumerate(C_list):
            mat = (Ci / k) - beta[i][None, :]
            m_i_min.append(mat.min(axis=1))
    
        # Correct reduced cost: rc(l) = -gamma + sum_i m_i_min[i][l]
        rc_all = -gamma + sum(m_i_min)
    
        # never consider rows already in L
        rc_all[inL] = np.inf
    
        # pick up to add_batch best negative
        order = np.argsort(rc_all)[:add_batch]
        picks = [(int(l), float(rc_all[l])) for l in order if rc_all[l] < -rc_tol]
        best = np.min(rc_all[~np.isinf(rc_all)]) if np.any(~np.isinf(rc_all)) else np.nan
        return picks, best

    
    
    for it in range(1, max_iter + 1):
        # update remaining time for this optimize
        rt = rem_time()
        if rt is not None:
            if rt <= 0:
                if verbose: print("[CG] time limit reached (pre-opt).")
                break
            m.Params.TimeLimit = max(0.01, rt)

        m.optimize()

        if m.Status == GRB.TIME_LIMIT and m.SolCount == 0:
            if verbose: print("[CG] TimeLimit with no incumbent; returning last snapshot.")
            break
        
        if m.Status not in (GRB.OPTIMAL, GRB.TIME_LIMIT):
            raise RuntimeError(f"RMP status={m.Status}")

        snapshot_incumbent()

        # A time-limited RMP may have a feasible solution but no optimal duals.
        # Keep that solution; pricing requires an optimal RMP.
        if m.Status == GRB.TIME_LIMIT:
            break
        
        alpha, beta, gamma = extract_duals()
        #picks, best_rc = price(alpha, beta, gamma, set(L))
        picks, best_rc = price(beta, gamma, set(L))
        if verbose:
            print(f"[CG] it={it:03d} |L|={len(L)} best_rc={best_rc:.3e} add={len(picks)} "
                  f"obj={(m.ObjVal if m.SolCount>0 else np.nan):.6g}")

        if not picks:
            # optimal for full problem within tolerance
            break

        # Append new atoms (rows/variables/coeffs) in place
        for l, rc in picks:
            L.append(l)
            add_atom(l)
        m.update()

        # Optional: quick time check after pricing/adding
        rt = rem_time()
        if rt is not None and rt <= 0:
            if verbose: print("[CG] time limit reached (post-add).")
            break

    # One last optimize with remaining time (to polish)
    rt = rem_time()
    if rt is None or rt > 0:
        if rt is not None:
            m.Params.TimeLimit = rt
        m.optimize()
        if m.Status in (GRB.OPTIMAL, GRB.TIME_LIMIT) and m.SolCount > 0:
            snapshot_incumbent()
    # else:
    #     m.optimize()
    if best["L"] is None:
        raise RuntimeError("No feasible solution found before time limit.")

    # --- Recover full-size solution -----------------------------------------
    # z_full = np.zeros(s)
    # if m.SolCount > 0:
    #     for ell, l in enumerate(L):
    #         z_full[l] = z_vars[ell].X

    # Y_full = []
    # for i in range(k):
    #     Yi = np.zeros((s, n_list[i]))
    #     if m.SolCount > 0:
    #         for ell, l in enumerate(L):
    #             Yi[l, :] = Y_vars[i][ell].X
    #     Y_full.append(Yi)

    # Build full-sized arrays from snapshot
    z_full = np.zeros(s)
    for ell, l in enumerate(best["L"]):
        z_full[l] = best["zL"][ell]
    
    Y_full = []
    for i in range(k):
        Yi = np.zeros((s, n_list[i]))
        for ell, l in enumerate(best["L"]):
            Yi[l, :] = best["YL"][i][ell]
        Y_full.append(Yi)

    obj_val = best["obj"]

    # The returned objective must describe the same saved feasible solution.
    return z_full, Y_full, obj_val, m


def solve_barycenter_lp_cg(
    C_list: List[np.ndarray],
    measures: Sequence[Measure],
    params: Optional[Dict[str, Union[int, float]]] = None,
    init_L: Optional[Sequence[int]] = None,
    time_limit_sec: float=None,
    max_iter: int = 10000,
    add_batch: int = 1,
    rc_tol: float = 1e-9,
    verbose: bool = False,
) -> Tuple[np.ndarray, List[np.ndarray], float, gp.Model]:
    """
    Column-generation solver for the entropic-free (pure LP) Wasserstein barycenter.
    We generate columns/rows over the barycenter support index l in {0,...,s-1}.

    Args:
        C_list: list of cost matrices C_i of shape (s, n_i)
        measures: sequence of Measure objects with attributes .X (n_i x d) and .w (n_i,)
        params: optional Gurobi parameter dict
        init_L: optional iterable of initial row indices (barycenter atoms) to start with
        max_iter: maximum CG iterations
        add_batch: number of new rows (atoms) to add per iteration (>=1)
        rc_tol: reduced cost tolerance to decide optimality
        verbose: print CG progress

    Returns:
        z: (s,) barycenter weights (full-length with zeros outside the final L)
        y_vals: list [Y_i] where Y_i is shape (s, n_i) with zeros outside final L rows
        obj_val: final objective value
        model: final Gurobi model (RMP)
    """
    
    k = len(C_list)
    assert k == len(measures)
    s = C_list[0].shape[0]
    n_list = [Ci.shape[1] for Ci in C_list]
    for Ci, meas in zip(C_list, measures):
        assert Ci.shape == (s, meas.X.shape[0])

    start = time.perf_counter()
    
    def remaining_time():
        if time_limit_sec is None:
            return None
        return max(0.0, time_limit_sec - (time.perf_counter() - start))
        
    # --- helpers -------------------------------------------------------------
    def build_rmp(L):
        """Build a restricted master problem (RMP) with rows in L."""
        m = gp.Model("bary_lp_cg_rmp")
        if params:
            for p, v in params.items():
                m.setParam(p, v)

        rem = remaining_time()
        if rem is not None:
            # guard against tiny/zero values to avoid Gurobi raising errors
            m.Params.TimeLimit = max(0.01, rem)

        # Variables:
        # z_l for l in L
        z = m.addMVar(len(L), lb=0.0, name="z")

        # Y_i(l,j) for l in L, j in [n_i]
        Y_vars = []
        for i, n_i in enumerate(n_list):
            Y_i = m.addMVar((len(L), n_i), lb=0.0, name=f"Y{i}")
            Y_vars.append(Y_i)

        # Constraints:
        # (1) For each i and row index ell in L: sum_j Y_i[ell,j] == z[ell]
        row_match_constrs = []
        for i in range(k):
            constr = m.addConstrs(
                (Y_vars[i][ell, :].sum() == z[ell] for ell in range(len(L))),
                name=f"row_match_z_i{i}"
            )
            row_match_constrs.append(constr)

        # (2) For each i and each column j: sum_{ell in L} Y_i[ell,j] == w_i[j]
        col_match_constrs = []
        for i, meas in enumerate(measures):
            constr = m.addConstrs(
                (Y_vars[i][:, j].sum() == float(meas.w[j]) for j in range(n_list[i])),
                name=f"col_match_w_i{i}"
            )
            col_match_constrs.append(constr)

        # (3) Normalization: sum_{ell in L} z[ell] == 1
        norm = m.addConstr(z.sum() == 1.0, name="norm")

        # Objective:
        obj = gp.LinExpr()
        for i, Ci in enumerate(C_list):
            # Only rows in L contribute
            obj += gp.quicksum(Ci[L[ell], j] * Y_vars[i][ell, j]
                               for ell in range(len(L)) for j in range(n_list[i]))
        m.setObjective((1.0 / k) * obj, GRB.MINIMIZE)

        m.update()
        return m, z, Y_vars, row_match_constrs, col_match_constrs, norm

    def get_duals(model, L, row_constrs, col_constrs, norm_constr):
        """
        Extract duals (alpha_i, beta_i, gamma) from the RMP solution.

        For each i:
          - alpha_i[l] is the dual of (sum_j Y_i[l,j] == z[l]) for l in L.
            We'll return alpha_i as a length-s vector with zeros outside L.
          - beta_i[j] is the dual of (sum_l Y_i[l,j] == w_i[j]).
        gamma is the dual of (sum_l z[l] == 1).
        """
        # Gurobi: equality constraint duals are in .Pi
        # row_constrs[i][l] corresponds to alpha_i at L[l]
        alpha = [np.zeros(s, dtype=float) for _ in range(k)]
        for i in range(k):
            # row_constrs[i] is a tupledict; iterate in order of L
            for ell, l in enumerate(L):
                alpha[i][l] = row_constrs[i][ell].Pi

        beta = []
        for i in range(k):
            b_i = np.zeros(n_list[i], dtype=float)
            for j in range(n_list[i]):
                b_i[j] = col_constrs[i][j].Pi
            beta.append(b_i)

        gamma = norm_constr.Pi
        return alpha, beta, gamma

    def price(alpha, beta, gamma, L_set):
        """
        Pricing: compute reduced costs rho(l) for all l not in L_set and return best ones.
        rho(l) = gamma + sum_i min_j( (1/k)*C_i[l,j] - beta_i[j] ) - sum_i alpha_i[l].
        """
        in_L = np.zeros(s, dtype=bool)
        in_L[list(L_set)] = True

        # Precompute per i term: m_i[l] = min_j( (1/k)*C_i[l,j] - beta_i[j] )
        # Vectorized over l using mins along axis=1
        m_i_all = []
        for i, Ci in enumerate(C_list):
            m_i_all.append((Ci / k) - beta[i][None, :])
        # m_i_all[i] is shape (s, n_i); take min over columns j
        m_i_min = [Mi.min(axis=1) for Mi in m_i_all]  # list of (s,) arrays

        # reduced cost for all l
        rc = gamma + sum(m_i_min) - sum(ai for ai in alpha)  # broadcasting over l
        # mask out existing L
        rc_masked = np.where(~in_L, rc, np.inf)

        # pick add_batch best negatives
        cand_idx = np.argsort(rc_masked)[:add_batch]
        cand_rc = rc_masked[cand_idx]
        # filter by negativity tolerance
        pick = [(int(l), float(r)) for l, r in zip(cand_idx, cand_rc) if r < -rc_tol]
        return pick, rc

    # --- initialize L --------------------------------------------------------
    if init_L is not None and len(init_L) > 0:
        L = list(dict.fromkeys(int(l) for l in init_L if 0 <= l < s))  # dedup & sanitize
    else:
        # Heuristic: for each i, take the row minimizing average cost to that measure
        # Then union across i (often |L| starts small)
        L_set = set()
        for i, Ci in enumerate(C_list):
            l0 = int(np.argmin(Ci.mean(axis=1)))
            L_set.add(l0)
        if not L_set:
            L_set = {0}
        L = sorted(L_set)

    if verbose:
        print(f"[CG] start with |L|={len(L)} out of s={s}")

    # --- main CG loop --------------------------------------------------------
    for it in range(1, max_iter + 1):
        rem = remaining_time()
        if rem is not None and rem <= 0:
            if verbose: print("[CG] time limit reached before iteration build")
            break
        # Build/solve RMP
        model, zL, YL, row_constrs, col_constrs, norm_constr = build_rmp(L)
        model.optimize()

        if model.Status in (GRB.TIME_LIMIT,):
            if verbose: print("[CG] RMP hit Gurobi TimeLimit; exiting with incumbent.")
            break
        
        if model.status != GRB.OPTIMAL:
            raise RuntimeError(f"RMP infeasible or not optimal (status={model.status}).")

        # Extract duals
        alpha, beta, gamma = get_duals(model, L, row_constrs, col_constrs, norm_constr)

        # Pricing
        picks, rc_all = price(alpha, beta, gamma, set(L))
        if verbose:
            best_rc = np.min(rc_all[~np.isinf(rc_all)]) if np.any(np.isfinite(rc_all)) else np.nan
            print(f"[CG] iter {it}: |L|={len(L)}  best_rc={best_rc:.3e}  add={len(picks)}")

        rem = remaining_time()
        if rem is not None and rem <= 0:
            if verbose: print("[CG] time limit reached after pricing")
            break

        if not picks:
            # Optimal for full problem within tolerance
            break

        # Add candidates (by reduced cost) to L
        for l, r in picks:
            L.append(l)

    # --- recover full-size solutions ----------------------------------------
    # Re-optimize once more with final L to get primal vars and objective
    model, zL, YL, row_constrs, col_constrs, norm_constr = build_rmp(L)
    model.optimize()
    if model.status != GRB.OPTIMAL:
        raise RuntimeError(f"Final RMP not optimal (status={model.status}).")

    # Build full-length z (size s) and Y_i (s x n_i) with zeros outside L
    z_full = np.zeros(s, dtype=float)
    z_sol = zL.X
    for ell, l in enumerate(L):
        z_full[l] = z_sol[ell]

    Y_full = []
    for i in range(k):
        Yi = np.zeros((s, n_list[i]), dtype=float)
        YiL = YL[i].X
        for ell, l in enumerate(L):
            Yi[l, :] = YiL[ell, :]
        Y_full.append(Yi)

    return z_full, Y_full, model.ObjVal, model


def solve_barycenter_lp_scipy(C_list: List[np.ndarray], measures: Sequence[Measure]):
    if not _HAS_SCIPY: raise ImportError("scipy not available")
    k = len(C_list); s = C_list[0].shape[0]; n_list = [Ci.shape[1] for Ci in C_list]
    for Ci, m in zip(C_list, measures):
        assert Ci.shape == (s, m.X.shape[0])

    offsets = np.cumsum([0] + [s*n for n in n_list])
    z_off = offsets[-1]; n_vars = z_off + s
    c = np.zeros(n_vars)
    for i, Ci in enumerate(C_list): c[offsets[i]:offsets[i]+s*n_list[i]] = (Ci.reshape(-1))/k

    rows=[]; rhs=[]
    # sum_j y_{i, l, j} = z_l
    for i, n_i in enumerate(n_list):
        for l in range(s):
            row = np.zeros(n_vars); row[offsets[i]+l*n_i : offsets[i]+(l+1)*n_i] = 1.0; row[z_off+l] = -1.0
            rows.append(row); rhs.append(0.0)
    # sum_l y_{i, l, j} = w_{i,j}
    for i, (n_i, meas) in enumerate(zip(n_list, measures)):
        for j in range(n_i):
            row = np.zeros(n_vars)
            for l in range(s): row[offsets[i] + l*n_i + j] = 1.0
            rows.append(row); rhs.append(meas.w[j])
    # sum_l z_l = 1
    row = np.zeros(n_vars); row[z_off:z_off+s] = 1.0; rows.append(row); rhs.append(1.0)

    A_eq = np.vstack(rows); b_eq = np.array(rhs); bounds = [(0, None)]*n_vars
    res = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
    if not res.success: raise RuntimeError("linprog failed: "+res.message)
    x = res.x; z = x[z_off:z_off+s]; y_list=[]
    for i, n_i in enumerate(n_list):
        y_list.append(x[offsets[i]:offsets[i]+s*n_i].reshape(s, n_i))
    return z, y_list, res.fun, res

def solve_barycenter_lp_mosek(C_list: List[np.ndarray], measures: Sequence[Measure],
                              params: Optional[Dict[str,Union[int,float]]] = None):
    if not _HAS_MOSEK: raise ImportError("mosek not available")
    k = len(C_list); s = C_list[0].shape[0]; n_list = [Ci.shape[1] for Ci in C_list]
    for Ci, m in zip(C_list, measures):
        assert Ci.shape == (s, m.X.shape[0])

    offsets = np.cumsum([0] + [s*n for n in n_list])
    z_off = int(offsets[-1]); n_vars = z_off + s
    n_cons = k*s + sum(n_list) + 1
    with mosek.Env() as env:
        with env.Task(0, 0) as task:
            task.appendcons(n_cons)
            task.appendvars(n_vars)
            task.putvarboundlistconst(np.arange(n_vars, dtype=np.int32),
                                      mosek.boundkey.lo, 0.0, 0.0)
            task.putobjsense(mosek.objsense.minimize)
            task.putintparam(mosek.iparam.optimizer, mosek.optimizertype.dual_simplex)
            if params:
                if "Threads" in params:
                    task.putintparam(mosek.iparam.num_threads, int(params["Threads"]))
                if "TimeLimit" in params:
                    task.putdouparam(mosek.dparam.optimizer_max_time, float(params["TimeLimit"]))

            for i, Ci in enumerate(C_list):
                var_start = int(offsets[i])
                var_stop = var_start + s*n_list[i]
                task.putclist(np.arange(var_start, var_stop, dtype=np.int32),
                              np.asarray(Ci, dtype=np.float64).reshape(-1) / k)

            con_idx = 0
            for i, n_i in enumerate(n_list):
                for l in range(s):
                    var_start = int(offsets[i]) + l*n_i
                    row_vars = np.empty(n_i + 1, dtype=np.int32)
                    row_vars[:-1] = np.arange(var_start, var_start + n_i, dtype=np.int32)
                    row_vars[-1] = z_off + l
                    row_vals = np.ones(n_i + 1, dtype=np.float64)
                    row_vals[-1] = -1.0
                    task.putarow(con_idx, row_vars, row_vals)
                    task.putconbound(con_idx, mosek.boundkey.fx, 0.0, 0.0)
                    con_idx += 1

            for i, (n_i, meas) in enumerate(zip(n_list, measures)):
                support_offsets = int(offsets[i]) + np.arange(s, dtype=np.int32)*n_i
                for j, weight in enumerate(meas.w):
                    task.putarow(con_idx, support_offsets + j, np.ones(s, dtype=np.float64))
                    task.putconbound(con_idx, mosek.boundkey.fx, float(weight), float(weight))
                    con_idx += 1

            task.putarow(con_idx, np.arange(z_off, n_vars, dtype=np.int32),
                         np.ones(s, dtype=np.float64))
            task.putconbound(con_idx, mosek.boundkey.fx, 1.0, 1.0)
            task.optimize()
            status = task.getsolsta(mosek.soltype.bas)
            if status != mosek.solsta.optimal:
                raise RuntimeError(f"MOSEK status {status}")

            z = np.empty(s, dtype=np.float64)
            task.getxxslice(mosek.soltype.bas, z_off, n_vars, z)
            y_list = []
            for i, n_i in enumerate(n_list):
                y = np.empty((s, n_i), dtype=np.float64)
                start = int(offsets[i])
                task.getxxslice(mosek.soltype.bas, start, start + s*n_i, y.reshape(-1))
                y_list.append(y)
            objective = task.getprimalobj(mosek.soltype.bas)

    return z, y_list, objective, {"solver": "MOSEK", "status": str(status)}

def solve_barycenter_lp(C_list: List[np.ndarray], measures: Sequence[Measure],
                        prefer: str = "gurobi", params: Optional[Dict[str,Union[int,float]]] = None):
    if prefer == "gurobi":
        if _HAS_GUROBI: return solve_barycenter_lp_gurobi(C_list, measures, params)
        raise ImportError("gurobipy not available")
    if prefer == "mosek":
        if _HAS_MOSEK: return solve_barycenter_lp_mosek(C_list, measures, params)
        raise ImportError("mosek not available")
    if prefer == "scipy":
        if _HAS_SCIPY: return solve_barycenter_lp_scipy(C_list, measures)
        raise ImportError("scipy not available")
    raise ValueError(f"Unknown LP solver: {prefer}")


# -------------------------- W2 candidate sets --------------------------------

def _centroid_candidates_for_T(measures: Sequence[Measure], T: Sequence[int],
                               max_points: Optional[int] = None, seed: int = 0) -> np.ndarray:
    T = list(T); d = measures[0].X.shape[1]
    supports = [measures[i].X for i in T]; sizes = [Xi.shape[0] for Xi in supports]
    total = 1
    for s in sizes: total *= s
    cand = []
    rng = np.random.default_rng(seed)
    if max_points is not None and total > max_points:
        for _ in range(max_points):
            idxs = [rng.integers(0, s) for s in sizes]
            pts = [supports[j][idxs[j]] for j in range(len(T))]
            cand.append(np.mean(np.vstack(pts), axis=0))
    else:
        for idxs in itertools.product(*[range(s) for s in sizes]):
            pts = [supports[j][idxs[j]] for j in range(len(T))]
            cand.append(np.mean(np.vstack(pts), axis=0))
    return _dedup_rows(np.vstack(cand)) if cand else np.empty((0,d))

def build_W2_candidates_subset_sampling(measures: Sequence[Measure], t: int, seed: Optional[int] = None,
                                        max_points: Optional[int] = None) -> Tuple[np.ndarray, Tuple[int,...]]:
    k = len(measures); 
    if not (1 <= t <= k): raise ValueError("t must be in [1,k]")
    rng = np.random.default_rng(seed); T = tuple(sorted(rng.choice(k, size=t, replace=False)))
    S_T = _centroid_candidates_for_T(measures, T, max_points=max_points, seed=(seed or 0))
    return S_T, T

def build_W2_candidates_subset_enumeration(measures: Sequence[Measure], t: int,
                                           max_T: Optional[int] = None,
                                           max_points_per_T: Optional[int] = None,
                                           seed: Optional[int] = None) -> Tuple[np.ndarray, List[Tuple[int,...]]]:
    k = len(measures)
    all_T = list(_all_subsets_of_size(k, t))
    if max_T is not None and max_T < len(all_T):
        rng = np.random.default_rng(seed)
        all_T = list(rng.choice(all_T, size=max_T, replace=False))
    cand_list=[]; used_T=[]
    for T in all_T:
        S_T = _centroid_candidates_for_T(measures, T, max_points=max_points_per_T, seed=(seed or 0))
        if S_T.size:
            cand_list.append(S_T); used_T.append(T)
    d = measures[0].X.shape[1]
    S_union = _dedup_rows(np.vstack(cand_list)) if cand_list else np.empty((0,d))
    return S_union, used_T


def build_local_t2_around_t1(
    measures,
    S1_active: np.ndarray,                # active t=1 atoms from stage A (r x d)
    k_per_measure: int = 5,              # K nearest per measure
    max_pairs_per_atom: int = 200,        # cap per active atom over all (i,j)
    radius: Optional[float] = None,       # filter centroids near each active atom
    seed: Optional[int] = 0
) -> np.ndarray:
    """
    For each active barycenter atom 'c' and each pair of measures (i<j):
      take K nearest points from measure i and j to 'c', form a *limited* set
      of pairwise centroids, optionally filter by radius, and collect all.
    """
    rng = np.random.default_rng(seed)
    d = S1_active.shape[1]
    k = len(measures)
    if k < 2 or S1_active.size == 0:
        return np.empty((0, d))
    all_centroids = []

    # Precompute for speed: nothing fancy, just reuse nearest selections
    for c in S1_active:
        centroids_c = []
        for i in range(k):
            idx_i = _nearest_indices(measures[i].X, c, k_per_measure)
            Xi = measures[i].X[idx_i]
            for j in range(i+1, k):
                idx_j = _nearest_indices(measures[j].X, c, k_per_measure)
                Xj = measures[j].X[idx_j]
                # Cartesian but limited: sample up to M pairs
                M = min(max_pairs_per_atom, Xi.shape[0]*Xj.shape[0])
                if M <= Xi.shape[0]*Xj.shape[0]:
                    # sample without replacement in flattened grid indices
                    flatN = Xi.shape[0]*Xj.shape[0]
                    picks = rng.choice(flatN, size=M, replace=False)
                    ii = picks // Xj.shape[0]
                    jj = picks %  Xj.shape[0]
                else:
                    # shouldn't happen with the min(...) above, but keep robust
                    ii, jj = np.meshgrid(np.arange(Xi.shape[0]), np.arange(Xj.shape[0]), indexing="ij")
                    ii = ii.ravel(); jj = jj.ravel()
                Cij = 0.5*(Xi[ii] + Xj[jj])  # centroids (t=2)
                if radius is not None:
                    dif = Cij - c[None,:]
                    d2 = np.einsum('ij,ij->i', dif, dif)
                    Cij = Cij[d2 <= radius*radius]
                if Cij.size:
                    centroids_c.append(Cij)
        if centroids_c:
            all_centroids.append(np.vstack(centroids_c))

    return _dedup_rows(np.vstack(all_centroids)) if all_centroids else np.empty((0, d))

def build_local_t2_around_centers(
    measures: Sequence,              # your Measure list
    centers: np.ndarray,             # (r,d) active atoms to expand around
    k_per_measure: int = 5,         # nearest per measure
    max_pairs_per_center: int = 200, # cap per center over all (i,j)
    radius: Optional[float] = None,  # in the same coordinate scale; None = no filter
    seed: Optional[int] = 0
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if centers.size == 0 or len(measures) < 2:
        return np.empty((0, measures[0].X.shape[1]))
    d = measures[0].X.shape[1]
    k = len(measures)

    all_cand = []
    for c in centers:
        local_blocks = []
        # gather K-NN per measure
        neigh = [measures[i].X[_nearest_indices(measures[i].X, c, k_per_measure)] for i in range(k)]
        # form limited pairwise centroids around c
        for i in range(k):
            Xi = neigh[i]
            for j in range(i+1, k):
                Xj = neigh[j]
                total = Xi.shape[0] * Xj.shape[0]
                M = min(max_pairs_per_center, total)
                if M < total:
                    picks = rng.choice(total, size=M, replace=False)
                    ii = picks // Xj.shape[0]
                    jj = picks %  Xj.shape[0]
                else:
                    ii, jj = np.meshgrid(np.arange(Xi.shape[0]), np.arange(Xj.shape[0]), indexing="ij")
                    ii = ii.ravel(); jj = jj.ravel()
                cent = 0.5*(Xi[ii] + Xj[jj])
                if radius is not None:
                    d2 = np.sum((cent - c[None, :])**2, axis=1)
                    cent = cent[d2 <= radius*radius]
                if cent.size:
                    local_blocks.append(cent)
        if local_blocks:
            all_cand.append(np.vstack(local_blocks))

    return _dedup_rows(np.vstack(all_cand)) if all_cand else np.empty((0, d))


# -------------------------- W1 candidate sets --------------------------------

def _U_and_M(measures: Sequence[Measure]): #union of support
    U = _union_support_points(measures); d = U.shape[1] if U.size else measures[0].X.shape[1]
    M_list = [np.unique(U[:,l]) for l in range(d)]
    return U, M_list

def _cartesian_full(M_list: List[np.ndarray]) -> np.ndarray:
    """Full Cartesian product of 1D arrays in M_list -> (prod n_l, d)."""
    d = len(M_list)
    if d == 0:
        return np.zeros((1,0))
    mesh = np.array(np.meshgrid(*M_list, indexing="ij"))
    return mesh.reshape(d, -1).T  # (#comb, d)

def _cartesian_sample(M_list: List[np.ndarray], m: int, seed: Optional[int] = None) -> np.ndarray:
    """Uniformly sample m combinations from the Cartesian product without forming it fully."""
    rng = np.random.default_rng(seed)
    d = len(M_list)
    comb = np.empty((m, d), dtype=np.float64)
    for j in range(d):
        idx = rng.integers(0, len(M_list[j]), size=m)
        comb[:, j] = M_list[j][idx]
    return _dedup_rows(comb)

def _S1_for_T(measures: Sequence[Measure], T: Sequence[int]) -> np.ndarray:
    """S_1^(T) = A_T × ∏_{ℓ∉T} M_ℓ with A_T from pooled U."""
    U, M_list = _U_and_M(measures); d = len(M_list)
    T = tuple(sorted(T)); comp = tuple(i for i in range(d) if i not in T)

    # Build product over comp coordinates
    comp_arrays = [M_list[ℓ] for ℓ in comp]
    if len(comp_arrays) == 0:
        comp_vals = np.zeros((1,0))
    else:
        # Check total size to avoid blow-up
        total = 1
        for arr in comp_arrays: total *= len(arr)
        if (max_grid is not None) and (total > max_grid):
            comp_vals = _cartesian_sample(comp_arrays, m=max_grid, seed=seed)
        else:
            comp_vals = _cartesian_full(comp_arrays)

    # Now place A_T on T and comp_vals on comp
    if len(T) == 0:  # t=0: A_T is a single empty block
        if comp_vals.shape[1] == 0:
            return np.zeros((1, d))  # degenerate 0-dim case
        Y = np.zeros((comp_vals.shape[0], d), dtype=np.float64)
        for j, ℓ in enumerate(comp):
            Y[:, ℓ] = comp_vals[:, j]
        return _dedup_rows(Y)
    
    if U.size == 0:
        return np.empty((0,d))
    A_T = _dedup_rows(U[:, T])
    # # Build Cartesian product over complement coordinates
    # if comp:
    #     grids = [M_list[l] for l in comp]
    #     mesh = np.array(np.meshgrid(*grids, indexing="ij"))
    #     comp_vals = mesh.reshape(len(comp), -1).T  # (#comb, |comp|)
    # else:
    #     comp_vals = np.zeros((1,0))
    cand=[]
    for a in A_T:
        if comp_vals.shape[1] > 0:
            Y = np.zeros((comp_vals.shape[0], d), dtype=np.float64)
            Y[:, T] = a
            for j, ℓ in enumerate(comp):
                Y[:, ℓ] = comp_vals[:, j]
            cand.append(Y)
        else:
            Y = np.zeros((1, d), dtype=np.float64); Y[0, T] = a; cand.append(Y)
    return _dedup_rows(np.vstack(cand)) if cand else np.empty((0,d))
    # for a in A_T:
    #     if comp:
    #         Y = np.zeros((comp_vals.shape[0], d), dtype=np.float64)
    #         Y[:, T] = a
    #         for j, l in enumerate(comp): Y[:, l] = comp_vals[:, j]
    #         cand.append(Y)
    #     else:
    #         Y = np.zeros((1, d)); Y[0, T] = a; cand.append(Y)
    # return _dedup_rows(np.vstack(cand)) if cand else np.empty((0,d))

def build_W1_candidates_subspace_sampling(measures: Sequence[Measure], t: int,
                                          seed: Optional[int] = None,
                                          max_blocks: Optional[int] = None) -> Tuple[np.ndarray, Tuple[int,...]]:
    U, M_list = _U_and_M(measures); d = len(M_list)
    if not (0 <= t <= d): raise ValueError("t must be in [0,d]")
    rng = np.random.default_rng(seed); 
    if t == 0:
        T = tuple()
        S_T = _S1_for_T(measures, T, max_grid=max_grid, seed=seed)
        return S_T, T
    # t >= 1
    T = tuple(sorted(rng.choice(d, size=t, replace=False)))
    S_T_full = _S1_for_T(measures, T, max_grid=max_grid, seed=seed)
    if max_blocks is not None and S_T_full.shape[0] > max_blocks:
        idx = np.linspace(0, S_T_full.shape[0]-1, max_blocks, dtype=int)
        S_T_full = S_T_full[idx]
    return S_T_full, T

def build_W1_candidates_subspace_enumeration(measures: Sequence[Measure], t: int,
                                             max_T: Optional[int] = None, seed: Optional[int] = None,
                                             max_blocks_per_T: Optional[int] = None) -> Tuple[np.ndarray, List[Tuple[int,...]]]:
    U, M_list = _U_and_M(measures); d = len(M_list)
    if not (0 <= t <= d): raise ValueError("t must be in [0,d]")
    if t == 0:
        T = tuple()
        S_T = _S1_for_T(measures, T, max_grid=max_grid, seed=seed)
        if max_blocks_per_T is not None and S_T.shape[0] > max_blocks_per_T:
            idx = np.linspace(0, S_T.shape[0]-1, max_blocks_per_T, dtype=int)
            S_T = S_T[idx]
        return _dedup_rows(S_T), [T]

    all_T = list(_all_coordinate_subsets(d, t))
    if max_T is not None and max_T < len(all_T):
        rng = np.random.default_rng(seed)
        all_T = list(rng.choice(all_T, size=max_T, replace=False))
    cand_list=[]; used_T=[]
    for T in all_T:
        S_T = _S1_for_T(measures, T, max_grid=max_grid, seed=seed)
        if max_blocks_per_T is not None and S_T.shape[0] > max_blocks_per_T:
            idx = np.linspace(0, S_T.shape[0]-1, max_blocks_per_T, dtype=int)
            S_T = S_T[idx]
        if S_T.size: cand_list.append(S_T); used_T.append(T)
    S_union = _dedup_rows(np.vstack(cand_list)) if cand_list else np.empty((0,d))
    return S_union, used_T


# ------------------------------- Algorithms ----------------------------------

def algo1_W2_subset_sampling(measures: Sequence[Measure], t: int, prefer_solver: str = "gurobi",
                             params: Optional[Dict[str,Union[int,float]]] = None,
                             seed: Optional[int] = None,
                             max_points: Optional[int] = None):
    t0 = time.time()
    S_T, T = build_W2_candidates_subset_sampling(measures, t, seed=seed, max_points=max_points)
    C_list = _stacked_costs(S_T, measures, p=2)
    print("cost_length",len(C_list))
    print("cost_time",time.time()-t0)
    z, y_list, obj, model = solve_barycenter_lp(C_list, measures, prefer=prefer_solver, params=params)
    return dict(algorithm="Algo1_W2_subset_sampling", z=z, S=S_T, T=T, objective=obj,
                runtime=time.time()-t0, solver={"gurobi": "Gurobi", "mosek": "MOSEK", "scipy": "SciPy"}[prefer_solver])

def algo2_W2_subset_enumeration(measures: Sequence[Measure], t: int, prefer_solver: str = "gurobi",
                                params: Optional[Dict[str,Union[int,float]]] = None,
                                max_T: Optional[int] = None, max_points_per_T: Optional[int] = None,
                                seed: Optional[int] = None):
    t0 = time.time()
    S_t, used_T = build_W2_candidates_subset_enumeration(measures, t, max_T=max_T,
                                                         max_points_per_T=max_points_per_T, seed=seed)
    C_list = _stacked_costs(S_t, measures, p=2)
    print("cost_length",len(C_list))
    print("cost_time",time.time()-t0)
    z, y_list, obj, model = solve_barycenter_lp(C_list, measures, prefer=prefer_solver, params=params)
    return dict(algorithm="Algo2_W2_subset_enumeration", z=z, S=S_t, used_T=used_T, objective=obj,
                runtime=time.time()-t0, solver={"gurobi": "Gurobi", "mosek": "MOSEK", "scipy": "SciPy"}[prefer_solver])
    
def algo2_W2_subset_enumeration_cg(measures: Sequence[Measure], t: int, prefer_solver: str = "gurobi",
                                params: Optional[Dict[str,Union[int,float]]] = None,
                                max_T: Optional[int] = None, max_points_per_T: Optional[int] = None,
                                seed: Optional[int] = None):
    t0 = time.time()
    S_t, used_T = build_W2_candidates_subset_enumeration(measures, t, max_T=max_T,
                                                         max_points_per_T=max_points_per_T, seed=seed)
    C_list = _stacked_costs(S_t, measures, p=2)
    print("cost_length",len(C_list))
    print("cost_time",time.time()-t0)
    z, y_list, obj, model = solve_barycenter_lp_cg(C_list, measures, params=params)
    return dict(algorithm="Algo2_W2_subset_enumeration_cg", z=z, S=S_t, used_T=used_T, objective=obj,
                runtime=time.time()-t0, solver=("Gurobi" if _HAS_GUROBI else "SciPy"))

def algo2_W2_subset_enumeration_cg_persistent(measures: Sequence[Measure], t: int, prefer_solver: str = "gurobi",
                                params: Optional[Dict[str,Union[int,float]]] = None,
                                max_T: Optional[int] = None, max_points_per_T: Optional[int] = None,
                                seed: Optional[int] = None, time_limit_sec=100):
    t0 = time.time()
    S_t, used_T = build_W2_candidates_subset_enumeration(measures, t, max_T=max_T,
                                                         max_points_per_T=max_points_per_T, seed=seed)
    C_list = _stacked_costs(S_t, measures, p=2)
    print("cost_length",len(C_list))
    print("cost_time",time.time()-t0)
    z, y_list, obj, model = solve_barycenter_lp_cg_persistent(C_list, measures, params=params,time_limit_sec=time_limit_sec)
    return dict(algorithm="Algo2_W2_subset_enumeration_cg_persistent", z=z, S=S_t, used_T=used_T, objective=obj,
                runtime=time.time()-t0, solver=("Gurobi" if _HAS_GUROBI else "SciPy"))

def algo3_W1_subspace_sampling(measures: Sequence[Measure], t: int, prefer_solver: str = "gurobi",
                               params: Optional[Dict[str,Union[int,float]]] = None,
                               seed: Optional[int] = None, max_blocks: Optional[int] = None):
    t0 = time.time()
    S_T, T = build_W1_candidates_subspace_sampling(measures, t, seed=seed, max_blocks=max_blocks)
    C_list = _stacked_costs(S_T, measures, p=1)
    print("cost_length",len(C_list))
    print("cost_time",time.time()-t0)
    z, y_list, obj, model = solve_barycenter_lp(C_list, measures, prefer=prefer_solver, params=params)
    return dict(algorithm="Algo3_W1_subspace_sampling", z=z, S=S_T, T=T, objective=obj,
                runtime=time.time()-t0, solver={"gurobi": "Gurobi", "mosek": "MOSEK", "scipy": "SciPy"}[prefer_solver])

def algo4_W1_subspace_enumeration(measures: Sequence[Measure], t: int, prefer_solver: str = "gurobi",
                                  params: Optional[Dict[str,Union[int,float]]] = None,
                                  max_T: Optional[int] = None, seed: Optional[int] = None,
                                  max_blocks_per_T: Optional[int] = None):
    t0 = time.time()
    S_t, used_T = build_W1_candidates_subspace_enumeration(measures, t, max_T=max_T, seed=seed,
                                                           max_blocks_per_T=max_blocks_per_T)
    C_list = _stacked_costs(S_t, measures, p=1)
    print("cost_length",len(C_list))
    print("cost_time",time.time()-t0)
    z, y_list, obj, model = solve_barycenter_lp(C_list, measures, prefer=prefer_solver, params=params)
    return dict(algorithm="Algo4_W1_subspace_enumeration", z=z, S=S_t, used_T=used_T, objective=obj,
                runtime=time.time()-t0, solver={"gurobi": "Gurobi", "mosek": "MOSEK", "scipy": "SciPy"}[prefer_solver])

def two_stage_t1_then_local_t2(
    measures,
    *,
    # Stage A (t=1) knobs
    cg_params_t1: Optional[dict] = None,      # e.g., {"TimeLimit": 10, "OutputFlag": 0}
    t1_z_threshold: float = 1e-12,            # active-atom threshold
    # Stage B (local t=2) knobs
    k_per_measure: int = 20,
    max_pairs_per_atom: int = 200,
    radius: Optional[float] = None,
    t2_seed: Optional[int] = 0,
    # Final solve knobs
    cg_params_final: Optional[dict] = None,
    verbose: bool =True
):
    """
    Stage A: run CG on t=1 candidates (union of supports), stop early by time/iter,
             keep active atoms S1_active = S1[z>threshold].
    Stage B: build local t=2 candidates around S1_active and re-solve on S_mixed.
    """
    # --- Stage A: t=1 solve (union-of-support candidates) ---
    S1 = _union_support_points(measures)         # t=1 candidates
    C1 = _stacked_costs(S1, measures, 2)            # costs
    if verbose:
        print(f"[Stage A] t=1 candidates: {S1.shape[0]}")

    z1, Y1, obj1, _ = solve_barycenter_lp_cg_persistent(
        C1, measures, time_limit_sec=cg_params_t1, verbose=False
    )
    S1_active = S1[z1 > t1_z_threshold]
    if verbose:
        print(f"[Stage A] objective={obj1:.6g}, active atoms={S1_active.shape[0]}")

    # --- Stage B: local t=2 around active atoms ---
    S2_local = build_local_t2_around_t1(
        measures, S1_active,
        k_per_measure=k_per_measure,
        max_pairs_per_atom=max_pairs_per_atom,
        radius=radius,
        seed=t2_seed
    )
    if verbose:
        print(f"[Stage B] local t=2 candidates built: {S2_local.shape[0]}")

    # Mixed candidate set: keep all active t=1 + local t=2
    S_mixed = S1_active if S2_local.size == 0 else _dedup_rows(np.vstack([S1_active, S2_local]))
    if verbose:
        print(f"[Stage B] S_mixed size: {S_mixed.shape[0]}")

    # Final CG solve on mixed S
    Cmix = _stacked_costs(S_mixed, measures,2)
    z_mix, Y_mix, obj_mix, model = solve_barycenter_lp_cg_persistent(
        Cmix, measures, time_limit_sec=cg_params_final, verbose=False
    )
    if verbose:
        print(f"[Final] objective={obj_mix:.6g}")

    return dict(
        S_t1_all=S1, z_t1=z1, obj_t1=obj1,
        S1_active=S1_active,
        S2_local=S2_local,
        S_mixed=S_mixed, z=z_mix, Y_list=Y_mix, objective=obj_mix
    )


def refine_with_local_t2(
    measures: Sequence,
    S_init: np.ndarray,              # initial support (e.g., t=1 solution's support)
    z_init: np.ndarray,              # initial weights
    *,
    active_thr: float = 1e-12,
    k_per_measure: int = 15,
    max_pairs_per_center: int = 200,
    radius: Optional[float] = None,
    seed: Optional[int] = 0,
    # CG solver controls
    cg_params: Optional[dict] = None,
    rc_tol: float = 1e-9,
    add_batch: int = 10,
    time_limit_sec: Optional[float] = None,
    verbose: bool = True
):
    """
    1) pick active atoms from (S_init, z_init)
    2) build local t=2 candidates around them
    3) S_mixed = active ∪ local_t2 (dedup)
    4) run persistent CG on S_mixed, warm-starting with indices of active atoms
    """
    # 1) active atoms
    S_active = select_active_atoms(S_init, z_init, thr=active_thr)
    if verbose:
        print(f"[refine] active from init: {S_active.shape[0]} / {S_init.shape[0]}")

    # 2) local t=2
    S_local_t2 = build_local_t2_around_centers(
        measures, S_active,
        k_per_measure=k_per_measure,
        max_pairs_per_center=max_pairs_per_center,
        radius=radius,
        seed=seed
    )
    if verbose:
        print(f"[refine] local t=2 candidates: {S_local_t2.shape[0]}")

    # 3) union + dedup
    S_mixed = _dedup_rows(np.vstack([S_active, S_local_t2])) if S_local_t2.size else S_active
    if verbose:
        print(f"[refine] S_mixed size: {S_mixed.shape[0]}")

    # Map active atoms into indices of S_mixed to warm-start CG's L
    # (use rounding-based hashing to avoid floating-point mismatch)
    def _row_indexer(A: np.ndarray, B: np.ndarray, decimals: int = 12) -> List[int]:
        Ar = np.round(A, decimals); Br = np.round(B, decimals)
        # dict of row -> index for A
        keyA = {tuple(row): idx for idx, row in enumerate(Ar)}
        idxs = []
        for row in Br:
            idxs.append(keyA[tuple(row)])
        return idxs

    init_L = _row_indexer(S_mixed, S_active, decimals=12)

    # 4) costs and CG solve
    C_list = _stacked_costs(S_mixed, measures,2)
    # NOTE: assumes you have solve_barycenter_lp_cg_persistent in scope
    z, Y_list, obj, model = solve_barycenter_lp_cg_persistent(
        C_list, measures,
        params=(cg_params or {"OutputFlag": 0}),
        init_L=init_L,
        add_batch=add_batch,
        rc_tol=rc_tol,
        time_limit_sec=time_limit_sec,
        verbose=verbose
    )

    return dict(
        S_mixed=S_mixed, z=z, Y_list=Y_list, objective=obj,
        S_active=S_active, S_local_t2=S_local_t2, init_L=init_L
    )


def pairwise_cost_sqeuclid(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    X = np.asarray(X, np.float64); Y = np.asarray(Y, np.float64)
    XX = (X**2).sum(1)[:,None]; YY = (Y**2).sum(1)[None,:]
    C = XX + YY - 2.0*X@Y.T
    C[C<0] = 0.0
    return C

def exact_w2_squared_lp(mu: Measure, nu: Measure, params=None, return_plan=False):
    a, b = mu.w, nu.w
    C = pairwise_cost_sqeuclid(mu.X, nu.X)
    n, m = C.shape

    model = gp.Model("exact_w2_lp")
    # good defaults for repeated LPs
    model.Params.Method = 1         # dual simplex
    model.Params.Crossover = 1
    model.Params.NumericFocus = 2
    model.Params.Presolve = 1
    model.Params.OutputFlag = 0
    if params:
        for k,v in params.items():
            setattr(model.Params, k, v)

    P = model.addMVar((n,m), lb=0.0, name="P")
    model.addConstr(P.sum(axis=1) == a, name="row")
    model.addConstr(P.sum(axis=0) == b, name="col")
    P.Obj = C
    model.ModelSense = GRB.MINIMIZE
    model.optimize()

    if model.Status in (GRB.OPTIMAL, GRB.TIME_LIMIT) and model.SolCount > 0:
        cost = float(model.ObjVal)
        if return_plan:
            return cost, P.X.copy()
        return cost
    raise RuntimeError(f"Gurobi status {model.Status} (no feasible incumbent)")


def classify_images_exact_lp(
    image_paths: List[str],
    barycenters: List[Barycenter],
    preprocess_kwargs: Dict[str, Any],
    gurobi_params: Optional[Dict[str, Any]] = None
) -> List[Dict[str, Any]]:
    results = []
    for p in image_paths:
        mu = image_to_measure(p, **preprocess_kwargs)
        dists = []
        for b in barycenters:
            nu = b.measure  # wraps (S,z)
            # sanity check: same feature dimension
            if mu.X.shape[1] != nu.X.shape[1]:
                raise ValueError(f"Dim mismatch for {b.name}: "
                                 f"test {mu.X.shape[1]} vs bary {nu.X.shape[1]}")
            cost = exact_w2_squared_lp(mu, nu, params=gurobi_params)
            dists.append((b.name, cost))
        dists.sort(key=lambda t: t[1])
        results.append({"path": p, "pred": dists[0][0], "dists": dists})
    return results

# ------------------------------- Demo ----------------------------------------

def _demo_small():
    rng = np.random.default_rng(0)
    k, n, d = 3, 50, 2
    measures = [Measure(X=np.clip(rng.uniform(0,1,(n,d)) + 0.02*rng.standard_normal((n,d)),0,1),
                        w=np.ones(n)/n) for _ in range(k)]
    out1 = algo1_W2_subset_sampling(measures, t=2, prefer_solver="scipy", seed=0, max_points=800)
    out2 = algo2_W2_subset_enumeration(measures, t=2, prefer_solver="scipy", max_T=3, max_points_per_T=400, seed=0)
    out3 = algo3_W1_subspace_sampling(measures, t=1, prefer_solver="scipy", seed=0, max_blocks=50)
    out4 = algo4_W1_subspace_enumeration(measures, t=1, prefer_solver="scipy", max_T=2, seed=0, max_blocks_per_T=50)
    return {1: out1, 2: out2, 3: out3, 4: out4}

if __name__ == "__main__":
    res = _demo_small()
    for k,v in res.items():
        print(f"Algo{k}: solver={v['solver']}, |S|={v['S'].shape[0]}, obj={v['objective']:.6f}, time={v['runtime']:.3f}s")
