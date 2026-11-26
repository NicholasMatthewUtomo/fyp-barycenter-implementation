#!/usr/bin/env python3
"""Finance d=3 Wasserstein barycenter using Branch & Bound or enumeration.

Reads OHLCV CSV, computes d=3 features (return, normalized volume, volatility),
drops timestamps, and solves for the barycenter.
"""

from pathlib import Path
import argparse
import hashlib
import json
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

import numpy as np
import wb_bary_algorithms as wb
import wb_bnb
from finance_to_ot import finance_to_measures


def write_json(path, report):
    path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, default=ROOT / "data/finance/market_data.csv")
    p.add_argument("--output", type=Path,
                   help="Results directory; default includes k, p, method")
    p.add_argument("--k", type=int, default=10,
                   help="Number of windows / measures; default 10")
    p.add_argument("--window-size", type=int, default=20,
                   help="Rolling window for volume norm & volatility; default 20")
    p.add_argument("--points-per-window", type=int, default=None,
                   help="Bars per window; default splits series evenly into k windows")
    p.add_argument("--p", type=int, choices=(1, 2), default=2,
                   help="Ground-metric exponent; default 2 (W2)")
    p.add_argument("--method", choices=("bnb", "enum_t1", "enum_t2"), default="bnb",
                   help="Solver: bnb (Branch & Bound), enum_t1, enum_t2")
    p.add_argument("--bnb-max-boxes", type=int, default=500,
                   help="B&B: stop refining once this many boxes are active")
    p.add_argument("--bnb-gap-tol", type=float, default=1e-4,
                   help="B&B: stop when UB* - LB <= gap_tol")
    p.add_argument("--bnb-time-limit", type=float, default=300.0,
                   help="B&B: wall-clock time limit in seconds")
    args = p.parse_args()

    if args.output is None:
        args.output = ROOT / f"results/finance_k{args.k}_p{args.p}_{args.method}"
    if args.output.exists() and any(args.output.glob("finance_*")):
        p.error("Finance outputs already exist; choose another --output directory")

    measures, scaler, df = finance_to_measures(
        args.input,
        k=args.k,
        vol_window=args.window_size,
        n_per_window=args.points_per_window,
    )

    summary = dict(
        experiment="Finance",
        input=str(args.input),
        algorithm=("wb_bnb" if args.method == "bnb" else "wb_bary_algorithms"),
        method=args.method,
        p=args.p,
        k=args.k,
        n_windows=len(measures),
        points_per_window=[len(m.X) for m in measures],
        feature_dim=3,
        features=["return", "normalized_volume", "volatility"],
        scaler_mean=scaler.mean.tolist(),
        scaler_std=scaler.std.tolist(),
        rows=int(len(df)),
        classes=[],
    )
    summary["core_sha256"] = hashlib.sha256(
        Path(wb.__file__).read_bytes()
    ).hexdigest()

    args.output.mkdir(parents=True, exist_ok=True)

    # Save inputs
    X_all = np.stack([m.X for m in measures])
    w_all = np.stack([m.w for m in measures])
    np.savez_compressed(
        args.output / "finance_inputs.npz",
        X=X_all, w=w_all,
        scaler_mean=scaler.mean, scaler_std=scaler.std,
    )

    t0 = time.perf_counter()

    if args.method == "bnb":
        result = wb_bnb.solve_barycenter_bnb(
            measures,
            p=args.p,
            max_boxes=args.bnb_max_boxes,
            gap_tol=args.bnb_gap_tol,
            time_limit_sec=args.bnb_time_limit,
            prefer="mosek",
        )
        runtime = float(time.perf_counter() - t0)
        report = dict(
            method="bnb",
            p=args.p,
            k=args.k,
            n_windows=len(measures),
            status="completed",
            objective=float(result["objective"]),
            lower_bound=float(result["lower_bound"]),
            gap=float(result["gap"]),
            runtime_seconds=float(result["runtime"]),
            wall_seconds=runtime,
            iterations=int(result["iterations"]),
            support_size=int(len(result["z"])),
            active_support_size=int((result["z"] > 1e-12).sum()),
            mass_sum=float(result["z"].sum()),
        )
        S = result["S"]
        z = result["z"]
    else:
        # Enumeration benchmarks via wb_bary_algorithms
        if args.p == 2:
            result = wb.algo2_W2_subset_enumeration(measures, t=args.method[-1], prefer="mosek")
        else:
            result = wb.algo4_W1_subspace_enumeration(measures, t=args.method[-1], prefer="mosek")
        runtime = float(result.get("runtime", time.perf_counter() - t0))
        report = dict(
            method=args.method,
            p=args.p,
            k=args.k,
            n_windows=len(measures),
            status="completed",
            objective=float(result["objective"]),
            runtime_seconds=runtime,
            wall_seconds=runtime,
            support_size=int(len(result["z"])),
            active_support_size=int((result["z"] > 1e-12).sum()),
            mass_sum=float(result["z"].sum()),
        )
        S = result["S"]
        z = result["z"]

    summary["classes"].append(report)
    summary["status"] = "completed"
    summary["runtime_seconds"] = runtime
    write_json(args.output / "finance_summary.json", summary)

    np.savez_compressed(
        args.output / "finance_barycenter.npz",
        S=S, z=z,
        objective=float(report["objective"]),
        runtime_seconds=float(report["runtime_seconds"]),
        scaler_mean=scaler.mean, scaler_std=scaler.std,
    )
    np.savetxt(
        args.output / "finance_barycenter.csv",
        np.column_stack((S, z)),
        delimiter=",",
        header="feature_0_return,feature_1_norm_volume,feature_2_volatility,probability_weight",
        comments="",
    )

    # 3D scatter plot sized/colored by barycenter weight
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

        active = z > 1e-12
        fig = plt.figure(figsize=(9, 7))
        ax = fig.add_subplot(111, projection="3d")
        sc = ax.scatter(
            S[active, 0], S[active, 1], S[active, 2],
            s=20 + 120 * z[active] / z[active].max(),
            c=z[active], cmap="inferno", edgecolors="black", linewidths=0.2,
        )
        ax.set_xlabel("Return (z-scored)")
        ax.set_ylabel("Normalized Volume (z-scored)")
        ax.set_zlabel("Volatility (z-scored)")
        ax.set_title(
            f"Finance d=3 barycenter ({args.method}, p={args.p}, k={args.k})\n"
            f"Objective={report['objective']:.6g}, active atoms={int(active.sum())}"
        )
        fig.colorbar(sc, ax=ax, label="Probability mass")
        fig.savefig(args.output / "finance_barycenter_3d.png", dpi=180, bbox_inches="tight")
        plt.close(fig)
    except Exception as exc:  # pragma: no cover - plotting is best-effort
        print(f"Plot skipped: {type(exc).__name__}: {exc}", file=sys.stderr)

    print(json.dumps(summary, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
