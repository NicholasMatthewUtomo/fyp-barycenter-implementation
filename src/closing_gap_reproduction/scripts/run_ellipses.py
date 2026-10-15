#!/usr/bin/env python3
"""Reproduce ellipse experiments using the recovered author's W2 implementations.

Examples (run from any working directory):
  python scripts/run_ellipses.py --inspect --mode all
  python scripts/run_ellipses.py --mode reference
  python scripts/run_ellipses.py --mode deterministic --t 1 --solver gurobi
  python scripts/run_ellipses.py --mode all --solver gurobi
  python scripts/run_ellipses.py --smoke --mode all --solver scipy

The reference mode renders an existing result; it never optimizes a barycenter.
The default ten randomized seeds (0..9) are a new reproducible protocol because
no ten-trial seed sequence was recovered from the original notebook.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import pickle
import platform
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))


def parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=("deterministic", "randomized", "all", "reference"), default="deterministic")
    p.add_argument("--t", type=int, choices=(1, 2), default=1,
                   help="Subset size for deterministic mode; all runs deterministic t=1 and t=2.")
    p.add_argument("--solver", choices=("gurobi", "scipy"), default="gurobi",
                   help="Explicit solver choice, without silent fallback. Original SciPy backend is dense.")
    p.add_argument("--seeds", type=int, nargs="+", default=list(range(10)),
                   help="Seeds for randomized t=1; default 0..9 is a newly documented protocol.")
    p.add_argument("--input", type=Path, default=ROOT / "data/ellipses/input_ellipse_data.pkl")
    p.add_argument("--reference-data", type=Path, default=ROOT / "data/ellipses/ours_exact.pkl")
    p.add_argument("--output", type=Path, default=ROOT / "results/ellipses")
    p.add_argument("--inspect", action="store_true", help="Read data and print experiment sizes; no solve or output writes.")
    p.add_argument("--smoke", action="store_true", help="Use first three marginals and four atoms each; not a paper reproduction.")
    p.add_argument("--threads", type=int, help="Optional Gurobi Threads setting; default leaves the solver default unchanged.")
    p.add_argument("--time-limit", type=float, help="Optional Gurobi TimeLimit; solver must still reach OPTIMAL to save results.")
    p.add_argument("--quiet", action="store_true", help="Suppress Gurobi optimizer log.")
    return p


def load_inputs(path):
    import numpy as np
    # These are trusted, bundled original research assets. Pickle is not an upload format.
    with path.open("rb") as f:
        raw = pickle.load(f)
    if not isinstance(raw, list) or not raw:
        raise ValueError("Input must contain a nonempty list of [x,y,mass] arrays.")
    arrays = []
    for i, item in enumerate(raw):
        a = np.asarray(item, dtype=float)
        if a.ndim != 2 or a.shape[1] != 3 or not len(a):
            raise ValueError(f"Marginal {i} is not a nonempty n-by-3 array.")
        if not np.isfinite(a).all() or (a[:, 2] < 0).any() or a[:, 2].sum() <= 0:
            raise ValueError(f"Marginal {i} has invalid coordinates or masses.")
        a = a[a[:, 2] > 0].copy()
        a[:, 2] /= a[:, 2].sum()
        arrays.append(a)
    return arrays


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def environment():
    result = {"python": platform.python_version(), "platform": platform.platform()}
    for package in ("numpy", "scipy", "gurobipy", "matplotlib"):
        try:
            result[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            result[package] = None
    return result


def sizes(arrays, t):
    """Raw centroid count upper bound; no Cartesian products are constructed."""
    from itertools import combinations
    ns = [len(a) for a in arrays]
    raw = sum(math.prod(ns[i] for i in T) for T in combinations(range(len(ns)), t))
    return {"t": t, "subsets": math.comb(len(ns), t), "raw_candidates_before_deduplication": raw,
            "lp_variables_upper_bound": raw * (sum(ns) + 1)}


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def pyplot():
    # Keep plot/font caches inside the requested output directory when writing figures.
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    return plt


def plot_points(ax, xy, weights, title, threshold=1e-8):
    import numpy as np
    mask = np.asarray(weights) > threshold
    points = np.asarray(xy)[mask]
    # Match the author's support scatter convention; NPZ always preserves masses.
    ax.scatter(points[:, 0], points[:, 1], s=10, alpha=0.7, c="black")
    ax.set(xlim=(0, 1), ylim=(0, 1), xticks=[], yticks=[], title=title)
    ax.set_aspect("equal")


def plot_inputs(arrays, output, smoke):
    plt = pyplot()
    cols = min(5, len(arrays)); rows = math.ceil(len(arrays) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows), squeeze=False, constrained_layout=True)
    for i, ax in enumerate(axes.ravel()):
        if i < len(arrays):
            a = arrays[i]
            plot_points(ax, a[:, :2], a[:, 2], f"Marginal {i + 1}")
        else:
            ax.axis("off")
    if smoke:
        fig.suptitle("Smoke input — reduced data, not the paper experiment")
    fig.savefig(output / "input_measures.png", dpi=200)
    plt.close(fig)


def save_result(name, result, metadata, output, title):
    import numpy as np
    S = np.asarray(result["S"], dtype=float)
    z = np.asarray(result["z"], dtype=float)
    if S.ndim != 2 or S.shape != (len(z), 2) or not np.isfinite(z).all():
        raise ValueError("Invalid barycenter result.")
    np.savez_compressed(output / f"{name}.npz", S=S, z=z)
    record = dict(metadata)
    record.update({"name": name, "candidate_atoms": len(z), "active_atoms_threshold_1e_8": int((z > 1e-8).sum()),
                   "total_mass": float(z.sum()), "minimum_mass": float(z.min()),
                   "arrays": f"{name}.npz", "figure": f"{name}.png"})
    for key in ("objective", "runtime"):
        if key in result:
            record[key] = float(result[key])
    if "T" in result:
        record["selected_marginals_zero_based"] = [int(x) for x in result["T"]]
    if "used_T" in result:
        record["subsets_zero_based"] = [[int(i) for i in T] for T in result["used_T"]]
    write_json(output / f"{name}.json", record)
    plt = pyplot()
    fig, ax = plt.subplots(figsize=(5, 5), constrained_layout=True)
    plot_points(ax, S, z, title)
    fig.savefig(output / f"{name}.png", dpi=200)
    plt.close(fig)
    return record


def render_reference(path, output, metadata):
    import numpy as np
    with path.open("rb") as f:
        a = np.asarray(pickle.load(f), dtype=float)
    if a.ndim != 2 or a.shape[1] != 3 or not np.isfinite(a).all() or (a[:, 2] < 0).any():
        raise ValueError("Invalid cached reference array.")
    xy = a[:, :2].copy()
    xy[:, 0] = 1.0 - xy[:, 0]  # Original notebook cell 6 and upstream cost evaluator convention.
    ref_metadata = dict(metadata, status="existing_reference_rendered", solver=None,
                        reference_file=path.name, reference_sha256=sha256(path),
                        coordinate_transform="x -> 1-x; y unchanged (align reference with input coordinates)",
                        reference_source="Altschuler & Boix-Adsera (2021), stored ours_exact.pkl",
                        objective_not_recomputed=True,
                        paper_reported_objective=0.02666, paper_reported_runtime_seconds=8865.03)
    # No objective/runtime field: the script has not recomputed either quantity.
    return save_result("reference_existing", {"S": xy, "z": a[:, 2]}, ref_metadata, output,
                       "Stored exact reference\nAltschuler & Boix-Adsera (2021)")


def ensure_solver(wb, solver, arrays, mode, t):
    available = wb._HAS_GUROBI if solver == "gurobi" else wb._HAS_SCIPY
    if not available:
        raise RuntimeError(f"Requested {solver} is unavailable; install it or explicitly choose another solver.")
    if solver != "scipy":
        return
    # The archived implementation allocates dense A_eq. A rigorous lower bound is
    # enough to reject the original ellipse experiment before any large allocation.
    ns = [len(a) for a in arrays]
    s_lower = min(ns) if mode == "randomized" else max(ns)
    nvars = s_lower * (sum(ns) + 1)
    nrows = len(ns) * s_lower + sum(ns) + 1
    lower_bytes = 8 * nrows * nvars
    if lower_bytes > 2_000_000_000:
        raise RuntimeError(
            f"The recovered SciPy backend builds dense A_eq, requiring at least {lower_bytes / 1e9:.1f} GB here. "
            "Use --solver gurobi for the paper data, or --smoke --solver scipy for a small verification. "
            "No optimization was started."
        )


def main(argv=None):
    args = parser().parse_args(argv)
    if len(set(args.seeds)) != len(args.seeds):
        raise ValueError("Use distinct randomized seeds.")
    if args.threads is not None and args.threads < 1:
        raise ValueError("--threads must be positive.")
    if args.time_limit is not None and args.time_limit <= 0:
        raise ValueError("--time-limit must be positive.")
    if args.solver == "scipy" and (args.threads is not None or args.time_limit is not None):
        raise ValueError("--threads and --time-limit apply only to Gurobi.")
    if args.smoke and args.mode == "reference":
        raise ValueError("--smoke cannot be combined with --mode reference.")
    arrays = load_inputs(args.input)
    if args.smoke:
        arrays = [a[:4].copy() for a in arrays[:3]]
        for a in arrays:
            a[:, 2] /= a[:, 2].sum()
    ts = [1, 2] if args.mode == "all" else ([args.t] if args.mode == "deterministic" else [1])
    if max(ts) > len(arrays):
        raise ValueError("Requested t exceeds number of marginals.")
    ns = [len(a) for a in arrays]
    inspection = {"mode": args.mode, "input": str(args.input.resolve()), "input_sha256": sha256(args.input),
                  "marginals": len(arrays), "dimension": 2, "atom_counts": ns,
                  "objective_convention": "mean_i W2_squared(barycenter, marginal_i)",
                  "experiments": [] if args.mode == "reference" else [sizes(arrays, t) for t in ts],
                  "solver_requested": args.solver, "smoke_reduced_data": args.smoke,
                  "randomized_seeds": args.seeds if args.mode in ("all", "randomized") else [],
                  "randomized_seed_provenance": "New protocol; published ten-trial sequence not recovered.",
                  "reference_available": args.reference_data.is_file(),
                  "reference_render_only": args.mode == "reference"}
    if args.inspect:
        print(json.dumps(inspection, indent=2))
        return 0
    if args.mode != "reference":
        import wb_bary_algorithms as wb
        ensure_solver(wb, args.solver, arrays, args.mode, args.t)
    if args.mode == "reference" and not args.reference_data.is_file():
        raise FileNotFoundError(f"Cached reference not found: {args.reference_data}")
    output = args.output.resolve()
    if args.smoke:
        output = output / "smoke"
    output.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(output / ".matplotlib"))
    metadata = {"created_utc": datetime.now(timezone.utc).isoformat(), "environment": environment(),
                "input_sha256": inspection["input_sha256"], "input_file": args.input.name,
                "objective_convention": inspection["objective_convention"],
                "smoke_reduced_data": args.smoke,
                "plot_convention": "Black support scatter for z > 1e-8; NPZ retains all masses."}
    write_json(output / "run_config.json", dict(inspection, **metadata))
    plot_inputs(arrays, output, args.smoke)
    records = []
    if args.mode == "reference" or (args.mode == "all" and not args.smoke and args.reference_data.is_file()):
        records.append(render_reference(args.reference_data, output, metadata))
    if args.mode == "reference":
        write_json(output / "summary.json", records)
        print(f"Rendered existing reference (no optimization): {output}")
        return 0
    measures = [wb.Measure(X=a[:, :2], w=a[:, 2]) for a in arrays]
    params = {}
    if args.threads is not None:
        params["Threads"] = args.threads
    if args.time_limit is not None:
        params["TimeLimit"] = args.time_limit
    if args.quiet:
        params["OutputFlag"] = 0
    metadata.update({"solver": args.solver, "solver_params": params,
                     "status": "new_optimization", "code_sha256": sha256(ROOT / "code/wb_bary_algorithms.py")})
    prefix = "SMOKE: " if args.smoke else ""
    if args.mode in ("deterministic", "all"):
        for t in ts:
            print(f"Running deterministic t={t}, solver={args.solver}", flush=True)
            result = wb.algo2_W2_subset_enumeration(measures, t=t, seed=0,
                                                    prefer_solver=args.solver, params=params or None)
            record = save_result(f"deterministic_t{t}", result,
                                 dict(metadata, paper_algorithm=4, implementation="algo2_W2_subset_enumeration", t=t),
                                 output, f"{prefix}Algorithm 4, t={t}\nobjective={result['objective']:.8f}")
            records.append(record)
            write_json(output / "summary.json", records)
    if args.mode in ("randomized", "all"):
        best_result = None; best_seed = None; elapsed = 0.0
        for seed in args.seeds:
            print(f"Running randomized t=1, seed={seed}, solver={args.solver}", flush=True)
            result = wb.algo1_W2_subset_sampling(measures, t=1, seed=seed,
                                                 prefer_solver=args.solver, params=params or None)
            elapsed += float(result["runtime"])
            record = save_result(f"randomized_t1_seed{seed}", result,
                                 dict(metadata, paper_algorithm=3, implementation="algo1_W2_subset_sampling", t=1, seed=seed,
                                      seed_provenance="New reproducible protocol; published ten-trial sequence not recovered."),
                                 output, f"{prefix}Algorithm 3, t=1, seed={seed}\nobjective={result['objective']:.8f}")
            records.append(record)
            if best_result is None or result["objective"] < best_result["objective"]:
                best_result, best_seed = result, seed
            write_json(output / "summary.json", records)
        records.append(save_result("randomized_t1_best", best_result,
                                   dict(metadata, paper_algorithm=3, t=1, best_seed=best_seed,
                                        all_seeds=args.seeds, total_trial_runtime_seconds=elapsed,
                                        runtime_field="Time of selected trial only; total_trial_runtime_seconds sums all trials.",
                                        seed_provenance="New reproducible protocol; published ten-trial sequence not recovered."),
                                   output, f"{prefix}Best of {len(args.seeds)} randomized t=1 runs\nobjective={best_result['objective']:.8f}"))
    write_json(output / "summary.json", records)
    print(f"Saved results: {output}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ImportError, OSError, ValueError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1)
