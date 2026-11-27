#!/usr/bin/env python3
"""
Unified experiment runner for Wasserstein barycenter experiments.

Runs the following experiments and saves results to separate folders:

  1. ECG200 Algo4 W1 subspace enumeration, t=1, n=96, k=10
  2. ECG200 Algo4 W1 subspace enumeration, t=2, n=96, k=10
  3. ECG200 B&B W1 (fixed-dimension branch and bound), n=96, k=10
  4. Finance B&B d=3 (fixed-dimension branch and bound)

Designed to be run once via SLURM on Paratera (GPU node with more RAM).
Each experiment writes its own folder under --output-root.

Usage (from repo root):
    python scripts/run_all_experiments.py \
        --ecg200-data data/ecg200 \
        --finance-data data/finance/market_data.csv \
        --output-root results/unified_run \
        --solver mosek \
        --k 10 --points 96 \
        --finance-k 10
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

import wb_bary_algorithms as wb
import wb_bnb
from ecg200_to_ot import load_ecg200, ecg_support_indices, waveforms_to_measures
from finance_to_ot import finance_to_measures


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def write_json(path: Path, report: dict):
    path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def select_class_indices(labels, splits, label, k):
    """Select reproducible class inputs from both splits, without replacement."""
    selected = np.flatnonzero(labels == label)
    if k is None:
        return selected
    if k < 1 or k > len(selected):
        raise ValueError(f"Require 1 <= k <= {len(selected)} for label {label}; got {k}")
    by_split = [selected[splits[selected] == split] for split in ("train", "test")]
    interleaved = [indices[j] for j in range(max(map(len, by_split)))
                   for indices in by_split if j < len(indices)]
    return np.asarray(interleaved[:k], dtype=int)


# ---------------------------------------------------------------------------
# Experiment 1: ECG200 Algo4 W1 subspace enumeration, t=1
# ---------------------------------------------------------------------------

def run_ecg200_algo4(data_dir: Path, output_root: Path, t: int, k: int = 10,
                     points: int = 96, prefer_solver: str = "mosek",
                     max_blocks_per_T: int | None = None,
                     params: Optional[dict] = None) -> dict:
    """Run Algorithm 4 (W1 subspace enumeration) on ECG200 heartbeat data."""
    exp_dir = output_root / f"ecg200_algo4_t{t}_k{k}_n{points}"
    exp_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()

    waveforms, labels, splits, indices = load_ecg200(data_dir)
    sample_indices = ecg_support_indices(points)

    results = {}
    overall_status = "completed"

    for class_name, label in (("normal", 1), ("abnormal", -1)):
        class_t0 = time.perf_counter()
        selected = select_class_indices(labels, splits, label, k)
        amplitudes = waveforms[selected]
        measures = waveforms_to_measures(amplitudes, points=points)

        # Build candidates (to report sizes before solving)
        candidates, used_T = wb.build_W1_candidates_subspace_enumeration(
            measures, t=t, max_blocks_per_T=max_blocks_per_T,
        )
        candidate_count = len(candidates)
        lp_variables = candidate_count * (points * len(measures) + 1)
        lp_constraints = len(measures) * candidate_count + points * len(measures) + 1

        stem = f"ecg200_{class_name}_k{len(measures)}_n{points}_algo4_t{t}"

        # Save inputs
        np.savez_compressed(
            exp_dir / (stem + "_inputs.npz"),
            waveforms=amplitudes,
            X=np.stack([m.X for m in measures]),
            w=np.stack([m.w for m in measures]),
            labels=labels[selected],
            splits=splits[selected],
            split_indices=indices[selected],
            combined_indices=selected,
            retained_time_indices=sample_indices,
            sampled_waveforms=amplitudes[:, sample_indices],
        )

        try:
            result = wb.algo4_W1_subspace_enumeration(
                measures, t=t, prefer_solver=prefer_solver,
                max_blocks_per_T=max_blocks_per_T, params=params,
            )
            class_time = time.perf_counter() - class_t0

            results[class_name] = {
                "class_name": class_name,
                "label": int(label),
                "samples": len(measures),
                "k": len(measures),
                "t": t,
                "solver": prefer_solver,
                "candidate_count": candidate_count,
                "used_T": [list(T) for T in used_T],
                "lp_variables": lp_variables,
                "lp_constraints": lp_constraints,
                "objective": float(result["objective"]),
                "runtime_seconds": float(result["runtime"]),
                "wall_seconds": class_time,
                "support_size": int(len(result["z"])),
                "active_support_size": int((result["z"] > 1e-12).sum()),
                "mass_sum": float(result["z"].sum()),
                "status": "completed",
            }

            np.savez_compressed(
                exp_dir / (stem + ".npz"),
                S=result["S"], z=result["z"],
                objective=result["objective"],
                runtime_seconds=result["runtime"],
                combined_indices=selected,
                retained_time_indices=sample_indices,
            )
            np.savetxt(
                exp_dir / (stem + ".csv"),
                np.column_stack((result["S"], result["z"])),
                delimiter=",",
                header="normalized_time,ecg_amplitude,probability_weight",
                comments="",
            )

        except Exception as exc:
            overall_status = "failed"
            results[class_name] = {
                "class_name": class_name,
                "label": int(label),
                "samples": len(measures),
                "k": len(measures),
                "t": t,
                "solver": prefer_solver,
                "candidate_count": candidate_count,
                "lp_variables": lp_variables,
                "status": "failed",
                "error_type": type(exc).__name__,
                "error": str(exc),
            }

    total_time = time.perf_counter() - t0

    summary = {
        "experiment": "ECG200_Algo4_W1_subspace_enumeration",
        "t": t,
        "k": k,
        "points": points,
        "solver": prefer_solver,
        "data_dir": str(data_dir),
        "core_sha256": sha256_file(ROOT / "code" / "wb_bary_algorithms.py"),
        "status": overall_status,
        "total_runtime_seconds": total_time,
        "classes": results,
    }
    write_json(exp_dir / "summary.json", summary)
    return summary


# ---------------------------------------------------------------------------
# Experiment 3: ECG200 B&B W1 (fixed-dimension branch and bound)
# ---------------------------------------------------------------------------

def run_ecg200_bnb(data_dir: Path, output_root: Path, k: int = 10, points: int = 96,
                   p: int = 1, max_boxes: int = 500, gap_tol: float = 1e-4,
                   time_limit_sec: float = 300.0, prefer: str = "mosek",
                   params: Optional[dict] = None) -> dict:
    """Run fixed-dimension branch and bound on ECG200 heartbeat data (W1, p=1)."""
    exp_dir = output_root / f"ecg200_bnb_p{p}_k{k}_n{points}"
    exp_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()

    waveforms, labels, splits, indices = load_ecg200(data_dir)
    sample_indices = ecg_support_indices(points)

    results = {}
    overall_status = "completed"

    for class_name, label in (("normal", 1), ("abnormal", -1)):
        class_t0 = time.perf_counter()
        selected = select_class_indices(labels, splits, label, k)
        amplitudes = waveforms[selected]
        measures = waveforms_to_measures(amplitudes, points=points)

        stem = f"ecg200_{class_name}_k{len(measures)}_n{points}_bnb_p{p}"

        # Save inputs
        np.savez_compressed(
            exp_dir / (stem + "_inputs.npz"),
            waveforms=amplitudes,
            X=np.stack([m.X for m in measures]),
            w=np.stack([m.w for m in measures]),
            labels=labels[selected],
            splits=splits[selected],
            split_indices=indices[selected],
            combined_indices=selected,
            retained_time_indices=sample_indices,
            sampled_waveforms=amplitudes[:, sample_indices],
        )

        try:
            result = wb_bnb.solve_barycenter_bnb(
                measures, p=p, max_boxes=max_boxes, gap_tol=gap_tol,
                time_limit_sec=time_limit_sec, prefer=prefer, params=params,
            )
            class_time = time.perf_counter() - class_t0

            results[class_name] = {
                "class_name": class_name,
                "label": int(label),
                "samples": len(measures),
                "k": len(measures),
                "p": p,
                "solver": prefer,
                "algorithm": result["algorithm"],
                "objective": float(result["objective"]),
                "lower_bound": float(result["lower_bound"]),
                "gap": float(result["gap"]),
                "runtime_seconds": float(result["runtime"]),
                "wall_seconds": class_time,
                "iterations": int(result["iterations"]),
                "support_size": int(len(result["z"])),
                "active_support_size": int((result["z"] > 1e-12).sum()),
                "mass_sum": float(result["z"].sum()),
                "status": "completed",
            }

            np.savez_compressed(
                exp_dir / (stem + ".npz"),
                S=result["S"], z=result["z"],
                objective=result["objective"],
                lower_bound=result["lower_bound"],
                gap=result["gap"],
                runtime_seconds=result["runtime"],
                iterations=result["iterations"],
                combined_indices=selected,
                retained_time_indices=sample_indices,
            )
            np.savetxt(
                exp_dir / (stem + ".csv"),
                np.column_stack((result["S"], result["z"])),
                delimiter=",",
                header="normalized_time,ecg_amplitude,probability_weight",
                comments="",
            )

        except Exception as exc:
            overall_status = "failed"
            results[class_name] = {
                "class_name": class_name,
                "label": int(label),
                "samples": len(measures),
                "k": len(measures),
                "p": p,
                "solver": prefer,
                "status": "failed",
                "error_type": type(exc).__name__,
                "error": str(exc),
            }

    total_time = time.perf_counter() - t0

    summary = {
        "experiment": f"ECG200_BnB_W{p}",
        "k": k,
        "points": points,
        "p": p,
        "solver": prefer,
        "data_dir": str(data_dir),
        "core_sha256": sha256_file(ROOT / "code" / "wb_bnb.py"),
        "status": overall_status,
        "total_runtime_seconds": total_time,
        "classes": results,
    }
    write_json(exp_dir / "summary.json", summary)
    return summary


# ---------------------------------------------------------------------------
# Experiment 4: Finance B&B d=3 (fixed-dimension branch and bound)
# ---------------------------------------------------------------------------

def run_finance_bnb(finance_csv: Path, output_root: Path, k: int = 10,
                    vol_window: int = 20, n_per_window: int | None = None,
                    p: int = 2, max_boxes: int = 500, gap_tol: float = 1e-4,
                    time_limit_sec: float = 300.0, prefer: str = "mosek",
                    params: Optional[dict] = None) -> dict:
    """Run fixed-dimension branch and bound on finance d=3 OHLCV data."""
    exp_dir = output_root / f"finance_bnb_d3_p{p}_k{k}"
    exp_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()

    measures, scaler, df = finance_to_measures(
        finance_csv, k=k, vol_window=vol_window, n_per_window=n_per_window,
    )

    stem = f"finance_k{k}_p{p}_bnb"

    # Save inputs
    X_all = np.stack([m.X for m in measures])
    w_all = np.stack([m.w for m in measures])
    np.savez_compressed(
        exp_dir / (stem + "_inputs.npz"),
        X=X_all, w=w_all,
        scaler_mean=scaler.mean, scaler_std=scaler.std,
    )

    summary = {
        "experiment": f"Finance_BnB_d3_p{p}",
        "input": str(finance_csv),
        "algorithm": "wb_bnb",
        "method": "bnb",
        "p": p,
        "k": k,
        "n_windows": len(measures),
        "points_per_window": [len(m.X) for m in measures],
        "feature_dim": 3,
        "features": ["return", "normalized_volume", "volatility"],
        "scaler_mean": scaler.mean.tolist(),
        "scaler_std": scaler.std.tolist(),
        "rows": int(len(df)),
        "solver": prefer,
        "core_sha256": sha256_file(ROOT / "code" / "wb_bnb.py"),
        "classes": [],
        "status": "pending",
    }

    try:
        result = wb_bnb.solve_barycenter_bnb(
            measures, p=p, max_boxes=max_boxes, gap_tol=gap_tol,
            time_limit_sec=time_limit_sec, prefer=prefer, params=params,
        )
        runtime = time.perf_counter() - t0

        report = {
            "method": "bnb",
            "p": p,
            "k": k,
            "n_windows": len(measures),
            "status": "completed",
            "objective": float(result["objective"]),
            "lower_bound": float(result["lower_bound"]),
            "gap": float(result["gap"]),
            "runtime_seconds": float(result["runtime"]),
            "wall_seconds": runtime,
            "iterations": int(result["iterations"]),
            "support_size": int(len(result["z"])),
            "active_support_size": int((result["z"] > 1e-12).sum()),
            "mass_sum": float(result["z"].sum()),
        }
        summary["classes"].append(report)
        summary["status"] = "completed"
        summary["runtime_seconds"] = runtime
        write_json(exp_dir / "finance_summary.json", summary)

        np.savez_compressed(
            exp_dir / (stem + ".npz"),
            S=result["S"], z=result["z"],
            objective=float(report["objective"]),
            runtime_seconds=float(report["runtime_seconds"]),
            scaler_mean=scaler.mean, scaler_std=scaler.std,
        )
        np.savetxt(
            exp_dir / (stem + ".csv"),
            np.column_stack((result["S"], result["z"])),
            delimiter=",",
            header="feature_0_return,feature_1_norm_volume,feature_2_volatility,probability_weight",
            comments="",
        )

    except Exception as exc:
        summary["status"] = "failed"
        summary["error_type"] = type(exc).__name__
        summary["error"] = str(exc)
        write_json(exp_dir / "finance_summary.json", summary)
        raise

    return summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description="Unified Wasserstein barycenter experiment runner (ECG200 + Finance)."
    )
    p.add_argument("--ecg200-data", type=Path, default=ROOT / "data/ecg200",
                   help="ECG200 data directory (downloaded via aeon on first run)")
    p.add_argument("--finance-data", type=Path, default=ROOT / "data/finance/market_data.csv",
                   help="Finance OHLCV CSV path")
    p.add_argument("--output-root", type=Path, default=ROOT / "results/unified_run",
                   help="Root directory for all experiment output folders")
    p.add_argument("--solver", choices=("mosek", "gurobi", "scipy", "dpdlp"),
                   default="mosek",
                   help="LP solver for Algorithm 4 and B&B (default: mosek)")
    p.add_argument("--dpdlp-binary", type=str, default="dpdlp",
                   help="Path to D-PDLP executable (default: dpdlp)")
    p.add_argument("--dpdlp-work-dir", type=str, default=None,
                   help="Working directory for D-PDLP temp files (default: system temp)")
    p.add_argument("--k", type=int, default=10,
                   help="Heartbeats per class for ECG200 (default: 10)")
    p.add_argument("--finance-k", type=int, default=10,
                   help="Number of windows for finance (default: 10)")
    p.add_argument("--points", type=int, choices=range(2, 97), default=96, metavar="2..96",
                   help="Support points per heartbeat (default: 96)")
    p.add_argument("--finance-vol-window", type=int, default=20,
                   help="Rolling window for finance features (default: 20)")
    p.add_argument("--max-boxes", type=int, default=500,
                   help="B&B: max active boxes (default: 500)")
    p.add_argument("--gap-tol", type=float, default=1e-4,
                   help="B&B: gap tolerance (default: 1e-4)")
    p.add_argument("--time-limit", type=float, default=300.0,
                   help="B&B: time limit in seconds (default: 300)")
    p.add_argument("--max-blocks-per-T", type=int, default=None,
                   help="Algo4: cap candidate blocks per subset T (memory control)")
    p.add_argument("--run-ecg200-t1", action="store_true", default=True,
                   help="Run ECG200 Algo4 t=1 (default: on)")
    p.add_argument("--run-ecg200-t2", action="store_true", default=True,
                   help="Run ECG200 Algo4 t=2 (default: on)")
    p.add_argument("--run-ecg200-bnb", action="store_true", default=True,
                   help="Run ECG200 B&B W1 (default: on)")
    p.add_argument("--run-finance-bnb", action="store_true", default=True,
                   help="Run Finance B&B d=3 (default: on)")
    p.add_argument("--no-ecg200-t1", action="store_false", dest="run_ecg200_t1")
    p.add_argument("--no-ecg200-t2", action="store_false", dest="run_ecg200_t2")
    p.add_argument("--no-ecg200-bnb", action="store_false", dest="run_ecg200_bnb")
    p.add_argument("--no-finance-bnb", action="store_false", dest="run_finance_bnb")

    args = p.parse_args()

    args.output_root.mkdir(parents=True, exist_ok=True)

    # Build solver params (D-PDLP needs binary path and work dir)
    solver_params: Optional[dict] = None
    if args.solver == "dpdlp":
        solver_params = {
            "dpdlp_binary": args.dpdlp_binary,
            "dpdlp_work_dir": args.dpdlp_work_dir,
            "time_limit": args.time_limit,
        }

    all_results = {}
    overall_status = "completed"

    # Experiment 1: ECG200 Algo4 t=1
    if args.run_ecg200_t1:
        print("=" * 70)
        print("EXPERIMENT: ECG200 Algo4 W1 subspace enumeration, t=1")
        print("=" * 70, flush=True)
        try:
            r = run_ecg200_algo4(
                args.ecg200_data, args.output_root, t=1, k=args.k, points=args.points,
                prefer_solver=args.solver, max_blocks_per_T=args.max_blocks_per_T,
                params=solver_params,
            )
            all_results["ecg200_algo4_t1"] = r
            print(f"t=1 done: status={r['status']}, "
                  f"normal obj={r['classes'].get('normal', {}).get('objective', 'N/A')}, "
                  f"abnormal obj={r['classes'].get('abnormal', {}).get('objective', 'N/A')}", flush=True)
        except Exception as exc:
            overall_status = "failed"
            all_results["ecg200_algo4_t1"] = {
                "status": "failed", "error_type": type(exc).__name__, "error": str(exc)
            }
            print(f"t=1 FAILED: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)

    # Experiment 2: ECG200 Algo4 t=2
    if args.run_ecg200_t2:
        print("=" * 70)
        print("EXPERIMENT: ECG200 Algo4 W1 subspace enumeration, t=2")
        print("=" * 70, flush=True)
        try:
            r = run_ecg200_algo4(
                args.ecg200_data, args.output_root, t=2, k=args.k, points=args.points,
                prefer_solver=args.solver, max_blocks_per_T=args.max_blocks_per_T,
                params=solver_params,
            )
            all_results["ecg200_algo4_t2"] = r
            print(f"t=2 done: status={r['status']}, "
                  f"normal obj={r['classes'].get('normal', {}).get('objective', 'N/A')}, "
                  f"abnormal obj={r['classes'].get('abnormal', {}).get('objective', 'N/A')}", flush=True)
        except Exception as exc:
            overall_status = "failed"
            all_results["ecg200_algo4_t2"] = {
                "status": "failed", "error_type": type(exc).__name__, "error": str(exc)
            }
            print(f"t=2 FAILED: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)

    # Experiment 3: ECG200 B&B W1
    if args.run_ecg200_bnb:
        print("=" * 70)
        print("EXPERIMENT: ECG200 B&B W1 (fixed-dimension branch and bound)")
        print("=" * 70, flush=True)
        try:
            r = run_ecg200_bnb(
                args.ecg200_data, args.output_root, k=args.k, points=args.points,
                p=1, max_boxes=args.max_boxes, gap_tol=args.gap_tol,
                time_limit_sec=args.time_limit, prefer=args.solver,
                params=solver_params,
            )
            all_results["ecg200_bnb_p1"] = r
            print(f"B&B p=1 done: status={r['status']}, "
                  f"normal obj={r['classes'].get('normal', {}).get('objective', 'N/A')}, "
                  f"abnormal obj={r['classes'].get('abnormal', {}).get('objective', 'N/A')}", flush=True)
        except Exception as exc:
            overall_status = "failed"
            all_results["ecg200_bnb_p1"] = {
                "status": "failed", "error_type": type(exc).__name__, "error": str(exc)
            }
            print(f"B&B p=1 FAILED: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)

    # Experiment 4: Finance B&B d=3
    if args.run_finance_bnb:
        print("=" * 70)
        print("EXPERIMENT: Finance B&B d=3 (fixed-dimension branch and bound)")
        print("=" * 70, flush=True)
        try:
            r = run_finance_bnb(
                args.finance_data, args.output_root, k=args.finance_k,
                vol_window=args.finance_vol_window, p=2,
                max_boxes=args.max_boxes, gap_tol=args.gap_tol,
                time_limit_sec=args.time_limit, prefer=args.solver,
                params=solver_params,
            )
            all_results["finance_bnb_p2"] = r
            print(f"Finance B&B p=2 done: status={r['status']}, "
                  f"objective={r.get('classes', [{}])[0].get('objective', 'N/A')}", flush=True)
        except Exception as exc:
            overall_status = "failed"
            all_results["finance_bnb_p2"] = {
                "status": "failed", "error_type": type(exc).__name__, "error": str(exc)
            }
            print(f"Finance B&B FAILED: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)

    # Write overall summary
    overall = {
        "overall_status": overall_status,
        "experiments": all_results,
        "output_root": str(args.output_root),
        "solver": args.solver,
    }
    write_json(args.output_root / "unified_summary.json", overall)

    print("=" * 70)
    print("ALL EXPERIMENTS COMPLETE")
    print("=" * 70, flush=True)
    print(json.dumps(overall, indent=2), flush=True)

    return 0 if overall_status == "completed" else 1


if __name__ == "__main__":
    sys.exit(main())
