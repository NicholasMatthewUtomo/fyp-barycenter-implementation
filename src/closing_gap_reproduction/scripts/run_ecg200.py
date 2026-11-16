#!/usr/bin/env python3
"""ECG200 class barycenters using the unchanged Algorithm 4, t=2, and MOSEK.

Load all training and testing heartbeats; retain their amplitudes and use
normalized time and uniform atom weights. --points optionally reduces the
number of support points per heartbeat while retaining every heartbeat.
Run --inspect to report input and LP sizes without attempting optimization.
"""
from pathlib import Path
import argparse
import hashlib
import importlib.metadata
import json
import platform
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

import numpy as np
import wb_bary_algorithms as wb
from ecg200_to_ot import ecg_support_indices, load_ecg200, waveforms_to_measures


def write_json(path, report):
    path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def plot_waveforms(path, waveforms, class_name, result=None, sample_indices=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    count = 2 if result is not None else 1
    fig, axes = plt.subplots(1, count, figsize=(7 * count, 4), squeeze=False,
                             sharex=True, sharey=True, constrained_layout=True)
    times = np.linspace(0.0, 1.0, 96)
    for ax in axes[0]:
        ax.plot(times, waveforms.T, color="tab:blue", alpha=0.12, linewidth=0.7)
        ax.set(xlabel="Normalized time", ylabel="ECG amplitude", xlim=(0, 1))
    axes[0, 0].set_title(f"{class_name.title()}: all {len(waveforms)} input heartbeats")
    if sample_indices is not None and len(sample_indices) < 96:
        reduced = axes[0, 0].plot(
            times[sample_indices], waveforms[:, sample_indices].T,
            color="tab:orange", alpha=0.22, linewidth=0.7, marker=".", markersize=2
        )
        axes[0, 0].legend([axes[0, 0].lines[0], reduced[0]],
                          ["Original 96-point waveforms", f"Retained {len(sample_indices)}-point inputs"],
                          fontsize=8)
    if result is not None:
        S, z = result["S"], result["z"]
        active = z > 1e-12
        scatter = axes[0, 1].scatter(
            S[active, 0], S[active, 1], s=20 + 120 * z[active] / z[active].max(),
            c=z[active], cmap="inferno", edgecolors="black", linewidths=0.2,
            label="Barycenter atoms (size and color show mass)", zorder=3
        )
        point_label = f", {len(sample_indices)}-point inputs" if sample_indices is not None else ""
        axes[0, 1].set_title(f"Algorithm 4, t=2{point_label}\nObjective={result['objective']:.6g}")
        axes[0, 1].legend(fontsize=8, loc="lower right")
        fig.colorbar(scatter, ax=axes[0, 1], label="Probability mass")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def validate_result(result, measures):
    S = np.asarray(result["S"])
    z = np.asarray(result["z"])
    if S.ndim != 2 or S.shape[1] != 2 or z.shape != (len(S),) or not len(S):
        raise ValueError("Algorithm 4 returned inconsistent support/weight shapes")
    if not np.isfinite(S).all() or not np.isfinite(z).all():
        raise ValueError("Algorithm 4 returned nonfinite support or weights")
    if z.min() < -1e-8 or not np.isclose(z.sum(), 1.0, atol=1e-6, rtol=0):
        raise ValueError("Barycenter weights must be nonnegative and sum to one")
    if not np.isfinite(result["objective"]) or result["objective"] < -1e-8:
        raise ValueError("Invalid barycenter objective")
    # With d=t=2, the unchanged candidate generator should return the pooled
    # input support with its existing 12-decimal deduplication.
    if not np.array_equal(S, wb._union_support_points(measures)):
        raise ValueError("Unexpected t=2 candidate support")
    return dict(mass_sum=float(z.sum()), minimum_mass=float(z.min()),
                support_size=len(z), active_support_size=int((z > 1e-12).sum()))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, default=ROOT / "data/ecg200")
    p.add_argument("--output", type=Path, default=ROOT / "results/ecg200")
    p.add_argument("--points", type=int, choices=range(2, 97), default=96, metavar="2..96",
                   help="Support points per heartbeat; default 96 retains all original points")
    p.add_argument("--class", dest="class_name", choices=("both", "normal", "abnormal"),
                   default="both", help="Run one class independently, or both (default)")
    p.add_argument("--inspect", action="store_true", help="Load and validate inputs; no solve or result writes")
    args = p.parse_args()
    if not args.inspect:
        if args.output.exists() and not args.output.is_dir():
            p.error("--output must be a directory")
        if args.output.exists() and any(args.output.glob("ecg200_*")):
            p.error("ECG200 outputs already exist; choose another --output directory")

    waveforms, labels, splits, indices = load_ecg200(args.data)
    sample_indices = ecg_support_indices(args.points)
    versions = {"python": platform.python_version()}
    for package in ("numpy", "scipy", "matplotlib", "aeon", "mosek"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    summary = dict(
        experiment="ECG200", algorithm="algo4_W1_subspace_enumeration", t=2,
        solver="mosek", solver_parameters=None,
        algorithm_parameters=dict(max_T=None, seed=None, max_blocks_per_T=None),
        samples=len(waveforms), partitions={s: int((splits == s).sum()) for s in ("train", "test")},
        coordinates=["normalized time j/95", "unchanged ECG amplitude"],
        original_points_per_heartbeat=96, support_points_per_measure=args.points,
        retained_time_indices=sample_indices.tolist(), downsampled=args.points < 96,
        atom_weight=1 / args.points, marginal_weights="equal 1/k within each class",
        cost="existing W1 implementation: L1 distance in (time, amplitude)",
        core_sha256=hashlib.sha256(Path(wb.__file__).read_bytes()).hexdigest(),
        versions=versions, classes={}
    )
    if not args.inspect:
        args.output.mkdir(parents=True, exist_ok=True)

    failures = 0
    for class_name, label in (("normal", 1), ("abnormal", -1)):
        if args.class_name != "both" and class_name != args.class_name:
            continue
        selected = np.flatnonzero(labels == label)
        amplitudes = waveforms[selected]
        measures = waveforms_to_measures(amplitudes, points=args.points)
        # For this d=t=2 case, all input support points are the candidates.
        # This estimate does not replace the candidate generator used by Algo4.
        candidate_count = len(wb._union_support_points(measures))
        report = dict(
            class_name=class_name, label=label, samples=len(measures),
            partition_counts={s: int((splits[selected] == s).sum()) for s in ("train", "test")},
            support_points_per_measure=args.points, dimension=2, t=2, solver="mosek",
            candidate_count=candidate_count,
            lp_variables=candidate_count * (args.points * len(measures) + 1),
            lp_constraints=len(measures) * candidate_count + args.points * len(measures) + 1,
            cost_arrays_GiB=8 * candidate_count * args.points * len(measures) / 1024**3,
            status="inspected" if args.inspect else "pending",
            objective=None, runtime_seconds=None
        )
        summary["classes"][class_name] = report
        if args.inspect:
            continue
        stem = f"ecg200_{class_name}_k{len(measures)}_n{args.points}_algo4_t2_mosek"
        np.savez_compressed(
            args.output / (stem + "_inputs.npz"), waveforms=amplitudes,
            X=np.stack([m.X for m in measures]), w=np.stack([m.w for m in measures]),
            labels=labels[selected], splits=splits[selected], split_indices=indices[selected],
            combined_indices=selected, retained_time_indices=sample_indices,
            sampled_waveforms=amplitudes[:, sample_indices]
        )
        plot_waveforms(args.output / (stem + "_inputs.png"), amplitudes, class_name,
                       sample_indices=sample_indices)
        report["status"] = "running"
        summary["status"] = "running"
        write_json(args.output / (stem + ".json"), report)
        write_json(args.output / "ecg200_summary.json", summary)
        print(f"{class_name}: {len(measures)} heartbeats, {args.points} points each, "
              f"{candidate_count} candidates, {report['lp_variables']} LP variables", flush=True)
        started = time.perf_counter()
        try:
            # Keep all Algorithm 4 and MOSEK defaults. In particular, do not
            # cap candidates, patch helpers, select PDLP, or use another solver.
            result = wb.algo4_W1_subspace_enumeration(measures, t=2, prefer_solver="mosek")
            report["wall_seconds"] = time.perf_counter() - started
            report.update(validate_result(result, measures))
            report.update(status="completed", objective=float(result["objective"]),
                          runtime_seconds=float(result["runtime"]),
                          used_T=[list(T) for T in result["used_T"]])
            np.savez_compressed(
                args.output / (stem + ".npz"), S=result["S"], z=result["z"],
                objective=report["objective"], runtime_seconds=report["runtime_seconds"],
                wall_seconds=report["wall_seconds"], combined_indices=selected,
                retained_time_indices=sample_indices
            )
            np.savetxt(args.output / (stem + ".csv"),
                       np.column_stack((result["S"], result["z"])), delimiter=",",
                       header="normalized_time,ecg_amplitude,probability_weight", comments="")
            plot_waveforms(args.output / (stem + ".png"), amplitudes, class_name, result,
                           sample_indices=sample_indices)
        except Exception as exc:
            failures += 1
            report.update(status="failed", wall_seconds=time.perf_counter() - started,
                          error_type=type(exc).__name__, error=str(exc),
                          traceback=traceback.format_exc())
            print(f"{class_name}: {type(exc).__name__}: {exc}", file=sys.stderr)
        write_json(args.output / (stem + ".json"), report)
        # Checkpoint each class so its result survives a later failure.
        write_json(args.output / "ecg200_summary.json", summary)

    summary["status"] = "inspected" if args.inspect else ("failed" if failures else "completed")
    if not args.inspect:
        write_json(args.output / "ecg200_summary.json", summary)
    print(json.dumps(summary, indent=2, allow_nan=False))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
