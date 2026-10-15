#!/usr/bin/env python3
"""Sign-gesture experiment runner with explicit stage selection.

Helper functions copied from Sign_testing.ipynb, 1-based cell 2.
Training workflow: Sign_C.ipynb and high_precision_barycenters-master/Sign_D.ipynb,
cells 3-5 (t1), 8-9 (first hybrid stage), 11-12 (second hybrid stage),
13-14 (continued t1). A uses the equivalent cells in SignAlpha.ipynb.
Cached scores are historical evidence; numerical classification runs afresh only
with the classify command. No experiment runs when invoked without a command.
"""
from __future__ import annotations
import argparse
import csv
import json
import math
import os
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any

PACKAGE = Path(__file__).resolve().parents[1]


def load_runtime():
    """Keep --help and cached-table independent of scientific dependencies."""
    global np, cv2, Measure
    global algo2_W2_subset_enumeration_cg_persistent, refine_with_local_t2
    global build_W2_candidates_subset_enumeration, _stacked_costs
    global solve_barycenter_lp_cg_persistent
    global _HAS_GUROBI
    import numpy as np
    import cv2
    sys.path.insert(0, str(PACKAGE / "code"))
    from wb_bary_algorithms import (
        Measure, algo2_W2_subset_enumeration_cg_persistent,
        refine_with_local_t2, build_W2_candidates_subset_enumeration,
        _stacked_costs, solve_barycenter_lp_cg_persistent, _HAS_GUROBI,
    )

# Verbatim notebook helpers. The runner fixes mode='xy_rgb5d' and origin='xy'.
# Other preprocessing modes are retained for provenance but are not CLI options.
@dataclass
class Barycenter:
    name: str
    S: np.ndarray
    z: np.ndarray

    @property
    def measure(self) -> Measure:
        return Measure(self.S, self.z)

def image_to_measure(
    image_path: str,
    size: Tuple[int, int] = (64, 64),
    mode: str = "threshold",          # "threshold" | "grayscale" | "rgb_mass" | "xy_rgb5d" | "xy_lab5d"
    threshold: float = 0.5,           # used if mode == "threshold", in [0,1]
    keep_top: Optional[int] = None,   # if set, keep only the top-k mass pixels
    coordinate_origin: str = "xy",    # "ij" or "xy" (affects x/y order only)
    normalize_coords: bool=True,
) -> Measure:
    if not os.path.exists(image_path):
        raise FileNotFoundError(image_path)

    # Load color
    im_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if im_bgr is None:
        raise ValueError(f"Could not read image: {image_path}")

    # Resize (width, height) for cv2
    W, H = size
    im_bgr = cv2.resize(im_bgr, (W, H), interpolation=cv2.INTER_AREA).astype(np.float64) / 255.0
    im_rgb = im_bgr[:, :, ::-1]  # BGR->RGB

    # Build coordinates
    xs, ys = np.arange(W), np.arange(H)
    if coordinate_origin == "xy":
        grid = np.stack(np.meshgrid(xs, ys, indexing="xy"), axis=-1)  # (H,W,2) with (x,y)
        if normalize_coords:
            dx = max(W - 1, 1)
            dy = max(H - 1, 1)
            grid = grid.astype(np.float64)
            grid[..., 0] /= dx  # x in [0,1]
            grid[..., 1] /= dy  # y in [0,1]
    else:
        grid = np.stack(np.meshgrid(np.arange(H), np.arange(W), indexing="ij"), axis=-1)  # (i,j)
        if normalize_coords:
            di = max(W - 1, 1)
            di = max(H - 1, 1)
            grid = grid.astype(np.float64)
            grid[..., 0] /= di  # x in [0,1]
            grid[..., 1] /= dj  # y in [0,1]

    if mode in ("threshold", "grayscale", "rgb_mass"):
        # ---- 2D support; mass depends on mode ----
        gray = (0.2126*im_rgb[...,0] + 0.7152*im_rgb[...,1] + 0.0722*im_rgb[...,2])

        if mode == "threshold":
            mass = (gray >= threshold).astype(np.float64)
        elif mode == "grayscale":
            mass = gray
        else:  # "rgb_mass": weight by total RGB energy
            mass = im_rgb.sum(axis=2) / 3.0

        # Optional coreset: keep only top-k pixels by mass
        if keep_top is not None and keep_top < mass.size:
            flat = mass.ravel()
            kth = np.partition(flat, -keep_top)[-keep_top]
            mask = (flat >= kth).reshape(mass.shape)
            mass = mass * mask

        w = mass.ravel()
        nz = w > 0
        X = grid.reshape(-1, 2)[nz]
        w = w[nz]
        w = w / w.sum()
        return Measure(X, w)

    elif mode in ("xy_rgb5d", "xy_lab5d"):
        # ---- 5D support; mass is uniform over nonzero pixels ----
        if mode == "xy_lab5d":
            im_lab = cv2.cvtColor((im_rgb*255.0).astype(np.uint8), cv2.COLOR_RGB2LAB).astype(np.float64)
            color = im_lab / np.array([100.0, 255.0, 255.0])[None,None,:]  # roughly to [0,1]
        else:
            color = im_rgb  # already in [0,1]

        # Use brightness as a mask to drop pure background
        bright = (0.2126*im_rgb[...,0] + 0.7152*im_rgb[...,1] + 0.0722*im_rgb[...,2])
        mask = bright > 1e-6
        XY = grid.reshape(-1, 2)
        CLR = color.reshape(-1, 3)
        XYCLR = np.concatenate([XY, CLR], axis=1)
        XYCLR = XYCLR[mask.ravel()]

        n = XYCLR.shape[0]
        if n == 0:
            raise ValueError("No foreground pixels found.")
        w = np.full(n, 1.0/n)  # uniform over kept pixels
        return Measure(XYCLR, w)

    else:
        raise ValueError("mode must be one of: 'threshold', 'grayscale', 'rgb_mass', 'xy_rgb5d', 'xy_lab5d'")

def rasterize_barycenter(
    S: np.ndarray,
    z: np.ndarray,
    size: tuple = (64, 64),          # (W, H)
    coordinate_origin: str = "xy",   # "xy" or "ij"
    normalized_coords: bool = False, # NEW: if True, first two coords in [0,1]
    color_space=None, #: str | None = None,  # None | "rgb" | "lab"  (only for S.shape[1]==5)
    lab_denorm: tuple = (100.0, 255.0, 255.0),   # L*,a*,b* scales if using Lab
    method: str = "nearest",         # "nearest" or "bilinear"
    return_grayscale_if_no_color: bool = True,
):
    """
    Convert barycenter support/weights to an image.

    - If S is (N,2): returns (H,W) grayscale (mass map in [0,1]).
    - If S is (N,5): returns (H,W,3) RGB image; color_space="rgb" or "lab".
    - normalized_coords=True: treats coord dims as in [0,1]; denormalizes to pixel grid.
    - method: "nearest" (fast) or "bilinear" (smoother).
    """
    S = np.asarray(S, float); z = np.asarray(z, float).reshape(-1)
    assert S.shape[0] == z.shape[0], "S and z length mismatch"

    W, H = size
    # 1) extract coordinates
    if coordinate_origin == "xy":
        x = S[:, 0].copy()
        y = S[:, 1].copy()
        if normalized_coords:
            x *= max(W - 1, 1)
            y *= max(H - 1, 1)
    elif coordinate_origin == "ij":
        y = S[:, 0].copy()
        x = S[:, 1].copy()
        if normalized_coords:
            y *= max(H - 1, 1)
            x *= max(W - 1, 1)
    else:
        raise ValueError('coordinate_origin must be "xy" or "ij"')

    # Helper accumulators
    mass = np.zeros((H, W), dtype=np.float64)

    def _clip(xi, yi):
        return np.clip(xi, 0, W - 1), np.clip(yi, 0, H - 1)

    # 2D grayscale only?
    is_color = (S.shape[1] == 5) and (color_space is not None)
    if not is_color and (S.shape[1] != 2) and return_grayscale_if_no_color:
        # fall back to grayscale if 5D but no color_space specified
        is_color = False

    if not is_color:
        # ----- Grayscale mass -----
        if method == "nearest":
            xi = np.rint(x).astype(int); yi = np.rint(y).astype(int)
            xi, yi = _clip(xi, yi)
            np.add.at(mass, (yi, xi), z)
        elif method == "bilinear":
            x0 = np.floor(x).astype(int); y0 = np.floor(y).astype(int)
            x1 = np.clip(x0 + 1, 0, W - 1); y1 = np.clip(y0 + 1, 0, H - 1)
            wx = x - x0; wy = y - y0
            w00 = (1 - wx) * (1 - wy)
            w10 = wx * (1 - wy)
            w01 = (1 - wx) * wy
            w11 = wx * wy
            np.add.at(mass, (y0, x0), z * w00)
            np.add.at(mass, (y0, x1), z * w10)
            np.add.at(mass, (y1, x0), z * w01)
            np.add.at(mass, (y1, x1), z * w11)
        else:
            raise ValueError('method must be "nearest" or "bilinear"')

        if mass.max() > 0:
            mass /= mass.max()
        return mass  # (H,W) in [0,1]

    # ----- Color case (S is (N,5)) -----
    color_feats = S[:, 2:5].copy()

    if color_space.lower() == "lab":
        # de-normalize Lab if you normalized during measure creation
        Ls, as_, bs = lab_denorm
        color_feats[:, 0] *= Ls
        color_feats[:, 1] *= as_
        color_feats[:, 2] *= bs

    sumC = np.zeros((H, W, 3), dtype=np.float64)

    if method == "nearest":
        xi = np.rint(x).astype(int); yi = np.rint(y).astype(int)
        xi, yi = _clip(xi, yi)
        np.add.at(mass, (yi, xi), z)
        for c in range(3):
            np.add.at(sumC[..., c], (yi, xi), z * color_feats[:, c])
    elif method == "bilinear":
        x0 = np.floor(x).astype(int); y0 = np.floor(y).astype(int)
        x1 = np.clip(x0 + 1, 0, W - 1); y1 = np.clip(y0 + 1, 0, H - 1)
        wx = x - x0; wy = y - y0
        w00 = (1 - wx) * (1 - wy)
        w10 = wx * (1 - wy)
        w01 = (1 - wx) * wy
        w11 = wx * wy
        # mass
        np.add.at(mass, (y0, x0), z * w00)
        np.add.at(mass, (y0, x1), z * w10)
        np.add.at(mass, (y1, x0), z * w01)
        np.add.at(mass, (y1, x1), z * w11)
        # color
        for c in range(3):
            np.add.at(sumC[..., c], (y0, x0), z * color_feats[:, c] * w00)
            np.add.at(sumC[..., c], (y0, x1), z * color_feats[:, c] * w10)
            np.add.at(sumC[..., c], (y1, x0), z * color_feats[:, c] * w01)
            np.add.at(sumC[..., c], (y1, x1), z * color_feats[:, c] * w11)
    else:
        raise ValueError('method must be "nearest" or "bilinear"')

    eps = 1e-12
    avgC = sumC / (mass[..., None] + eps)

    if color_space.lower() == "lab":
        # Convert Lab -> RGB for display
        import cv2
        Lab = np.zeros_like(avgC, dtype=np.float32)
        Lab[..., 0] = np.clip(avgC[..., 0], 0, 100)
        Lab[..., 1] = np.clip(avgC[..., 1], 0, 255)
        Lab[..., 2] = np.clip(avgC[..., 2], 0, 255)
        Lab_u8 = np.zeros_like(Lab, dtype=np.uint8)
        Lab_u8[..., 0] = (Lab[..., 0] * 255.0 / 100.0).astype(np.uint8)
        Lab_u8[..., 1] = Lab[..., 1].astype(np.uint8)
        Lab_u8[..., 2] = Lab[..., 2].astype(np.uint8)
        bgr = cv2.cvtColor(Lab_u8, cv2.COLOR_Lab2BGR)
        rgb = bgr[..., ::-1]
        rgb[mass < eps] = 0
        return rgb.astype(np.uint8)

    elif color_space.lower() == "rgb":
        rgb = np.clip(avgC, 0.0, 1.0)
        rgb[mass < eps] = 0
        return (rgb * 255.0).astype(np.uint8)

    else:
        raise ValueError('color_space must be "rgb" or "lab"')

def pairwise_cost_sqeuclid(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    X = np.asarray(X, np.float64); Y = np.asarray(Y, np.float64)
    XX = (X**2).sum(1)[:,None]; YY = (Y**2).sum(1)[None,:]
    C = XX + YY - 2.0*X@Y.T
    C[C<0] = 0.0
    return C

def sinkhorn_distance2(mu: Measure, nu: Measure, reg: float=0.02,
                       max_iter: int=1000, tol: float=1e-9) -> float:
    """Return entropic OT cost (≈ W2^2 as reg→0) between discrete measures."""
    a, b = mu.w, nu.w
    C = pairwise_cost_sqeuclid(mu.X, nu.X)          # squared Euclidean
    K = np.exp(-C / reg)                            # Gibbs kernel
    # Stabilized scaling
    u = np.ones_like(a); v = np.ones_like(b)
    for _ in range(max_iter):
        u_prev = u
        K_v = K @ v + 1e-300
        u = a / K_v
        K_tu = K.T @ u + 1e-300
        v = b / K_tu
        if np.linalg.norm(u - u_prev, 1) < tol:
            break
    # Transport plan implicit: P = diag(u) K diag(v)
    P = (u[:,None] * K) * v[None,:]
    return float((P * C).sum())

def distance_to_barycenter(m: Measure, bary: Barycenter, reg: float=0.02) -> float:
    # Ensure same dimension
    assert m.X.shape[1] == bary.measure.X.shape[1], "Feature dimensions differ from barycenter!"
    return sinkhorn_distance2(m, bary.measure, reg=reg)

def classify_images_by_barycenters(
    image_paths: List[str],
    barycenters: List[Barycenter],
    preprocess_kwargs: Dict[str, Any],
    reg: float=0.02
) -> List[Dict[str, Any]]:
    results = []
    for p in image_paths:
        m = image_to_measure(p, **preprocess_kwargs)
        dists = [(b.name, distance_to_barycenter(m, b, reg=reg)) for b in barycenters]
        dists.sort(key=lambda x: x[1])
        pred = dists[0][0]
        results.append({"path": p, "pred": pred, "dists": dists})
    return results


def preprocess_options():
    return dict(size=(64, 64), mode="xy_rgb5d", coordinate_origin="xy",
                normalize_coords=True, keep_top=None)


def output_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def validate_sz(S, z):
    if S.ndim != 2 or S.shape[1] != 5 or z.ndim != 1 or len(S) != len(z):
        raise ValueError("Expected S shape (n,5) and z shape (n,).")
    if not np.isfinite(S).all() or not np.isfinite(z).all():
        raise ValueError("Nonfinite saved support or masses.")
    if z.min() < -1e-8 or not np.isclose(z.sum(), 1.0, atol=1e-6):
        raise ValueError("Saved masses must be nonnegative and sum to one.")


def load_initial(args):
    if not args.initial_support or not args.initial_weights:
        raise ValueError("Both --initial-support and --initial-weights are required.")
    S = np.load(args.initial_support, allow_pickle=False)
    z = np.load(args.initial_weights, allow_pickle=False)
    validate_sz(S, z)
    return S, z


def train(args):
    load_runtime()
    if not _HAS_GUROBI:
        raise RuntimeError("Sign training requires gurobipy and a configured license. "
                           "Install requirements-gurobi.txt before running a training stage.")
    folder = "SignAlphaSubset" if args.gesture == "A" else f"Sign_{args.gesture}"
    paths = sorted((args.data_dir / folder).glob("*.jpg"))
    if len(paths) != 3:
        raise ValueError(f"Expected exactly 3 training JPEGs in {args.data_dir / folder}; found {len(paths)}.")
    measures = [image_to_measure(str(p), **preprocess_options()) for p in paths]
    started = time.perf_counter()
    # The historical limit is passed to CG; upstream candidate/cost construction
    # and some model-building work can make total wall time exceed this limit.
    if args.command == "hybrid":
        S0, z0 = load_initial(args)
        result = refine_with_local_t2(
            measures, S_init=S0, z_init=z0, active_thr=1e-12,
            k_per_measure=5, max_pairs_per_center=200, radius=None, seed=0,
            cg_params={"OutputFlag":0, "Threads":4, "Method":1},
            rc_tol=1e-9, add_batch=1, time_limit_sec=args.time_limit,
            verbose=True)
        S, z = result["S_mixed"], result["z"]
    elif args.initial_support or args.initial_weights:
        # Source Sign_C cell 13: continue t1 from the active support of a saved run.
        S0, z0 = load_initial(args)
        S, _ = build_W2_candidates_subset_enumeration(measures, t=1)
        lookup = {tuple(row): i for i, row in enumerate(np.round(S, 12))}
        try:
            active = [lookup[tuple(row)] for row in np.round(S0[z0 > 1e-8], 12)]
        except KeyError as exc:
            raise ValueError("Initial support does not match current training data.") from exc
        costs = _stacked_costs(S, measures, p=2)
        z, _, objective, _ = solve_barycenter_lp_cg_persistent(
            costs, measures, init_L=active, time_limit_sec=args.time_limit)
        result = {"objective":objective}
    else:
        result = algo2_W2_subset_enumeration_cg_persistent(
            measures, t=1, seed=0, time_limit_sec=args.time_limit)
        S, z = result["S"], result["z"]
    validate_sz(S, z)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.save(args.output_dir / "S.npy", S)
    np.save(args.output_dir / "z.npy", z)
    image = rasterize_barycenter(S, z, size=(64,64), coordinate_origin="xy",
                                 normalized_coords=True, color_space="rgb")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.imsave(args.output_dir / "barycenter.png", image, dpi=300)
    info = {"command":args.command,"class":args.gesture,
            "training_files":[str(p) for p in paths],
            "preprocessing":preprocess_options(),
            "initial_support":str(args.initial_support) if args.initial_support else None,
            "initial_weights":str(args.initial_weights) if args.initial_weights else None,
            "cg_time_limit_sec":args.time_limit,
            "wall_runtime_sec":time.perf_counter()-started,
            "objective":float(result["objective"]),
            "candidate_atoms":len(S),"active_atoms_1e_8":int((z>1e-8).sum()),
            "seed":0,
            "note":"This is a new run. Time labels in archival folders are historical stage labels, not verified total wall time."}
    if args.command == "hybrid":
        info["hybrid_parameters"] = dict(active_thr=1e-12,k_per_measure=5,
            max_pairs_per_center=200,radius=None,rc_tol=1e-9,add_batch=1,
            cg_params={"OutputFlag":0,"Threads":4,"Method":1})
    output_json(args.output_dir / "run.json", info)
    print(json.dumps(info, indent=2))


def summarize(rows, classes):
    counts = Counter()
    total = Counter()
    confusion = {c:{p:0 for p in classes} for c in classes}
    for row in rows:
        label = row["label_from_filename"]
        total[label] += 1
        counts[label] += row["prediction"] == label
        confusion[label][row["prediction"]] += 1
    return {"classes":classes,"total":sum(total.values()),
            "correct":sum(counts.values()),
            "per_class":{c:{"correct":counts[c],"total":total[c]} for c in classes},
            "confusion":confusion}


def cached_table(args):
    cache = json.loads(args.scores.read_text())
    answer = {"kind":"reconstruction_from_historical_cached_scores",
              "source":str(args.scores),"selected_classes":args.classes,"runs":{}}
    for variant in args.variants:
        run = cache["runs"][variant]
        rows = []
        for row in run["rows"]:
            if row["label_from_filename"] not in args.classes:
                continue
            distances = [d for d in row["distances"] if d["class"] in args.classes]
            prediction = min(distances, key=lambda d:d["cost"])["class"]
            rows.append({**row, "prediction":prediction})
        answer["runs"][variant] = {"source_notebook":run["source_notebook"],
            "source_cell_1based":run["source_cell_1based"],
            "regularization":run["regularization"], **summarize(rows,args.classes)}
    if args.output:
        output_json(args.output, answer)
    print(json.dumps(answer, indent=2))


def classify(args):
    load_runtime()
    variant_names = {"t1_3600":(1,3600),"t1_7200":(1,7200),
                     "hybrid3600":(2,3600),"hybrid7200":(2,7200)}
    t, seconds = variant_names[args.variant]
    barycenters = []
    model_metadata = []
    for c in args.classes:
        if c in args.model_dirs:
            folder = args.model_dirs[c]
            support_path, weight_path = folder / "S.npy", folder / "z.npy"
        else:
            stem = f"sign_{c}_t={t}_time{seconds}"
            folder = args.artifacts_dir / stem
            support_path, weight_path = folder / f"{stem}_S.npy", folder / f"{stem}_z.npy"
        S = np.load(support_path, allow_pickle=False)
        z = np.load(weight_path, allow_pickle=False)
        validate_sz(S,z)
        # Exact zero-mass atoms contribute nothing to the measure. Removing
        # them avoids allocating large dense costs for unused candidates.
        nonzero = z != 0
        model_metadata.append({"class":c,"support_path":str(support_path),
            "weights_path":str(weight_path),"saved_atoms":len(z),
            "retained_atoms":int(nonzero.sum()),
            "removed_exact_zero_mass_atoms":int((~nonzero).sum())})
        barycenters.append(Barycenter(c,S[nonzero],z[nonzero]))
    paths = sorted(p for p in (args.data_dir / "Sign_test_64").glob("*.jpg")
                   if p.name.split("_",1)[0] in args.classes)
    if not paths:
        raise ValueError("No matching test JPEGs found.")
    if args.max_images is not None:
        paths = paths[:args.max_images]
    started = time.perf_counter()
    rows = []
    for path in paths:
        item = classify_images_by_barycenters([str(path)],barycenters,
                       preprocess_kwargs=preprocess_options(),reg=args.reg)[0]
        row = {"file":path.name,"label_from_filename":path.name.split("_",1)[0],
               "prediction":item["pred"],
               "distances":[{"class":c,"cost":float(d)} for c,d in item["dists"]]}
        rows.append(row)
        print(f"{path.name}: {item['pred']}",flush=True)
    result = {"kind":"fresh_notebook_sinkhorn_classification","variant":args.variant,
              "models":model_metadata,
              "support_reduction":"Only exact zero-mass atoms removed; represented measures unchanged.",
              "reg":args.reg,"max_iter":1000,"tolerance":1e-9,
              "wall_runtime_sec":time.perf_counter()-started,
              "summary":summarize(rows,args.classes),"rows":rows,
              "note":"Uses the original notebook Sinkhorn cost and stopping rule; this does not certify exact Wasserstein distances."}
    output_json(args.output_dir / "classification.json",result)
    with (args.output_dir / "predictions.csv").open("w",newline="") as file:
        writer = csv.DictWriter(file,fieldnames=["file","label","prediction"]+args.classes)
        writer.writeheader()
        for row in rows:
            writer.writerow({"file":row["file"],"label":row["label_from_filename"],
                "prediction":row["prediction"],**{d["class"]:d["cost"] for d in row["distances"]}})
    print(json.dumps(result["summary"],indent=2))


def positive(value):
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise argparse.ArgumentTypeError("Must be finite and positive.")
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command",required=True)
    for command in ["t1","hybrid"]:
        sub = commands.add_parser(command,help="Run one explicitly selected training stage.")
        sub.add_argument("--class",dest="gesture",choices=list("ABCD"),required=True)
        sub.add_argument("--data-dir",type=Path,default=PACKAGE/"data/sign")
        sub.add_argument("--output-dir",type=Path,required=True)
        sub.add_argument("--time-limit",type=positive,required=True,
                         help="Seconds passed to CG; candidate/cost construction is outside this budget.")
        sub.add_argument("--initial-support",type=Path,required=command=="hybrid")
        sub.add_argument("--initial-weights",type=Path,required=command=="hybrid")
        sub.set_defaults(func=train)
    sub = commands.add_parser("cached-table",help="Reconstruct Table 1 counts from cached scores; no optimization.")
    sub.add_argument("--scores",type=Path,default=PACKAGE/"data/sign/cached_classification_scores.json")
    sub.add_argument("--classes",nargs="+",choices=list("ABCD"),default=list("ACD"))
    sub.add_argument("--variants",nargs="+",choices=["t1_3600","t1_7200","hybrid3600","hybrid7200"],
                     default=["t1_7200","hybrid3600"])
    sub.add_argument("--output",type=Path)
    sub.set_defaults(func=cached_table)
    sub = commands.add_parser("classify",help="Recompute test distances using saved barycenters (can be slow).")
    sub.add_argument("--data-dir",type=Path,default=PACKAGE/"data/sign")
    sub.add_argument("--artifacts-dir",type=Path,default=PACKAGE/"artifacts/sign")
    sub.add_argument("--model-dir",action="append",default=[],metavar="CLASS=DIR",
                     help="Override a selected class with fresh training output containing S.npy and z.npy; repeat per class.")
    sub.add_argument("--output-dir",type=Path,required=True)
    sub.add_argument("--classes",nargs="+",choices=list("ABCD"),default=list("ACD"))
    sub.add_argument("--variant",choices=["t1_3600","t1_7200","hybrid3600","hybrid7200"],required=True)
    sub.add_argument("--reg",type=positive,default=0.01)
    sub.add_argument("--max-images",type=int,help="Small diagnostic subset; omit for all selected classes.")
    sub.set_defaults(func=classify)
    args = parser.parse_args()
    if hasattr(args,"classes") and len(args.classes)!=len(set(args.classes)):
        parser.error("--classes must not contain duplicates.")
    if getattr(args,"max_images",None) is not None and args.max_images < 1:
        parser.error("--max-images must be positive.")
    if args.command == "classify":
        args.model_dirs = {}
        for item in args.model_dir:
            if "=" not in item:
                parser.error("--model-dir must be CLASS=DIR")
            label, directory = item.split("=",1)
            if label not in args.classes or not directory:
                parser.error("--model-dir must name a selected class and a nonempty directory")
            if label in args.model_dirs:
                parser.error("--model-dir must not repeat a class")
            args.model_dirs[label] = Path(directory)
    if hasattr(args,"output_dir") and args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error("Choose an empty --output-dir to preserve prior results.")
    args.func(args)


if __name__ == "__main__":
    main()
