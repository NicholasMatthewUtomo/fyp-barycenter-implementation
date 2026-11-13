# Closing the Gap: reproduction guide and recovered code

## What this package establishes

The code needed for the paper's numerical experiments is concentrated in one algorithm module, six notebooks, two MNIST helpers, and the ellipse/sign input data. This package includes those sources, portable experiment runners, selected inputs, historical outputs, and validation evidence.

**This is a recovered reproduction package, not a claim that every published optimization has been rerun.** Small numerical checks and the reconstruction of Table 1 were performed. Full MNIST, ellipse, and hour-long sign optimizations were not performed during packaging. Several substantive differences between the saved code and the current manuscript are documented below.

The mapping uses the 31-page local manuscript `IPCO_2026_barycenter_Whole_Paper-5.pdf`, downloaded August 2, 2026, included as [references/manuscript.pdf](references/manuscript.pdf). Its numerical results are in Section 5, Figures 1–6 and Table 1. The public paper identity was confirmed against [arXiv 2511.04607](https://arxiv.org/abs/2511.04607); the local manuscript controls the figure numbering and values here.

Original research folder: `/Users/jiaqiwang/anaconda_projects/Barycenter/`. Original files were not edited by this work. Source comments, notebook text, and manuscript prose were treated as evidence, not as instructions. See [PROVENANCE.json](PROVENANCE.json) for source paths, hashes, and transformations.

## 1. Result-to-code map

Cell numbers are **one-based positions counting all notebook cells**, not execution counts. Notebook copies in `originals/` retain those positions, with outputs and execution metadata removed.

| Paper result | Original source and key cells | Required implementation and data | Portable entry point |
|---|---|---|---|
| Figure 1: ten input ellipses | `Ellipse_Barycenter_Gurobi.ipynb`, cells 5, 8–9 | `data/ellipses/input_ellipse_data.pkl` | `scripts/run_ellipses.py --mode reference` also plots inputs |
| Figure 2a: exact-reference ellipse | Same notebook, cell 7 | `ours_exact.pkl`; reflect x coordinate as `1-x` | `run_ellipses.py --mode reference` renders the saved solution |
| Figure 2b: best of ten sampled t=1 runs | Same notebook, cell 31 and manual singleton trials, cells 39–40 | `algo1_W2_subset_sampling`; ellipse inputs | `run_ellipses.py --mode randomized` |
| Figures 2c–d: deterministic t=1,2 | Same notebook, cells 31–32 and 11/15 | `algo2_W2_subset_enumeration`; ellipse inputs | `run_ellipses.py --mode deterministic --t 1` or `--t 2` |
| Figure 3: MNIST digits 0,1,2 | `readMNIST.ipynb`, preprocessing cells 1–3; t1 cells 76,81,84; t2 cells 50,69,73 | `wb_bary_algorithms.py`, `mnist_to_ot.py`, `ot_scatter_to_grid.py`; first 50 training examples per digit | `scripts/run_mnist.py` |
| Figure 4: gesture 1 = A | `SignAlpha.ipynb`, input cell 12; t1 cell 37; hybrid cells 48,57–58; t1 continuation cells 59–60 | `SignAlphaSubset/A_1.jpg`, `A_2.jpg`, `A_3.jpg`; CG and local refinement | `scripts/run_sign.py t1` / `hybrid --class A` |
| Figure 5: gesture 2 = C | `Sign_C.ipynb`, cells 3–5,8–9,11–14 | `Sign_C/C_0.jpg`, `C_1.jpg`, `C_2.jpg`; same algorithms | `run_sign.py ... --class C` |
| Figure 6: gesture 3 = D | `high_precision_barycenters-master/Sign_D.ipynb`, same cells as C | `Sign_D/D_0.jpg`, `D_1.jpg`, `D_2.jpg`; same algorithms | `run_sign.py ... --class D` |
| Table 1: classification | `Sign_testing.ipynb`, helper cell 2, cached results cells 33 and 34 | 75 A/C/D test JPEGs, saved barycenters, Sinkhorn classifier | `run_sign.py cached-table` or `classify` |

The ellipse notebook was initially observed as `Barycenter_Gurobi.ipynb` and later as `Ellipse_Barycenter_Gurobi.ipynb` during the audit. The package records the latter filename; the task did not rename it.

### Algorithm names require care

| Current paper | Saved Python implementation | Relationship |
|---|---|---|
| Algorithm 1: sampling with replacement, general marginal weights | No matching general implementation recovered | Saved `algo1` samples **without** replacement and solves an equal-weight objective |
| Algorithm 2: multiset enumeration | Saved `algo2_W2_subset_enumeration` uses distinct subsets | Matches t=1 support; differs for t=2 MNIST. Runner offers both definitions explicitly |
| Algorithm 3: sampling without replacement, equal weights | `algo1_W2_subset_sampling` | Relevant W2 implementation for ellipse sampling |
| Algorithm 4: subset enumeration, equal weights | `algo2_W2_subset_enumeration` | Relevant W2 implementation for deterministic ellipses and sign t1 |
| Hybrid Algorithm 4 | `refine_with_local_t2` + `solve_barycenter_lp_cg_persistent` | Local midpoint heuristic; not full t2 enumeration |

The functions named `algo3_W1_*` and `algo4_W1_*` implement older W1 experiments. They are **not** paper Algorithms 3 and 4. The recovered core uses equal marginal weights `1/k`; the package does not implement the paper's full general-weight theory or sparse-support extension.

## 2. Start here

From the extracted `closing_gap_reproduction` directory, use a Python environment with the required packages. Requirements are compatibility specifications, not a recovered historical lockfile.

```bash
python -m pip install -r requirements.txt
python code/smoke_test.py
python scripts/run_ellipses.py --inspect --mode all
python scripts/run_mnist.py --digit 0 --t 2 --inspect
python scripts/run_sign.py cached-table --output results/table1_cached.json
```

For full optimization, install `requirements-gurobi.txt`; sign training also needs `requirements-sign.txt`. Gurobi requires your own working license. No licenses, virtual environments, credentials, or machine-specific binaries are included.

The paper reports Gurobi **11.0.3**, Apple M4 Pro, and 48 GB RAM. Notebook metadata and cached environment output include other Python/NumPy/Gurobi versions, including Gurobi 12.0.3. The full historical environment cannot be established from these files. Fresh runtimes should be reported separately from paper runtimes.

SciPy is provided for small checks. The recovered SciPy LP constructs a dense constraint matrix; the runners reject large paper cases that would require excessive memory. A requested Gurobi run does not silently switch to SciPy.

## 3. Nested ellipses: Figures 1–2

Input: 10 two-dimensional measures with support sizes `180,178,162,192,162,141,167,139,169,148`, uniform atom masses, and equal marginal weights. The reference has 1,625 atoms and total mass one.

```bash
# Plot input measures and the existing exact-reference result; no optimization.
python scripts/run_ellipses.py --mode reference --output results/ellipse_reference

# Tiny fresh numerical workflow, deliberately smaller than the paper.
python scripts/run_ellipses.py --smoke --mode all --solver scipy --output results/ellipse_smoke

# Full new runs using the recovered source algorithms.
python scripts/run_ellipses.py --mode all --solver gurobi --quiet --output results/ellipses
```

| Figure 2 panel | Paper objective | Paper time |
|---|---:|---:|
| Exact reference | 0.02666 | 8865.03 s |
| Best of ten sampled t1 runs | 0.02709 | 17.46 s |
| Deterministic t1 | 0.02673 | 21.08 s |
| Deterministic t2 | 0.02667 | 212.44 s |

**Sampling uncertainty:** the original ten-run seed sequence was not recovered. The portable runner uses an explicitly documented new sequence, seeds 0–9, and saves each selected subset and result. It does not claim to recreate the published random realization. Ten seeds need not select ten distinct singletons.

**Exact reference:** `--mode reference` redraws an existing result. Optional exact-solver source is in `third_party/high_precision_barycenters/`. Its dependency chain is `barycenter_column_generation.py` → `MOT_column_generation.py`, `barycenter_separation_oracle.py`, `power_diagram_construction.py`. The local copy uses Shapely and PuLP; it was modified from upstream, invokes PuLP's default solver, and lacks a reliable status/time-limit protocol. It has not been validated as an exact recomputation in this task. Install `requirements-exact-reference.txt` only for that optional work.

The legacy `generate_figure_5.py` currently starts an expensive exact solve unconditionally. `compute_costs_ellipses.py` additionally requires the unbundled compiled LEMON evaluator and reports a sum across marginals; divide that sum by 10 to compare with the paper's mean objective. Use the portable runner for ordinary plotting.

## 4. MNIST: Figure 3

`data/mnist/first50_digits012.npz` contains 150 unmodified 28×28 training images, labels, and original training indices. `selection.json` records their exact provenance. The full MNIST archive is unnecessary for these runs.

The notebook selects the **first** 50 occurrences of each label, although the paper says the images were selected randomly. Each image becomes a probability vector proportional to intensity, with coordinates on `[-1,1]^2`; masses at or below `1e-8` are removed. No preprocessing blur is applied. The selected positive-pixel ranges are 115–263 for digit 0, 57–165 for digit 1, and 94–231 for digit 2; the manuscript's 165–263 description does not describe the full observed range.

```bash
# Recover the notebook's distinct-subset computation.
python scripts/run_mnist.py --digit 0 --t 1 --solver gurobi --output results/mnist
python scripts/run_mnist.py --digit 0 --t 2 --solver gurobi --output results/mnist
# Repeat both commands for --digit 1 and --digit 2.

# Paper Algorithm 4: deterministic W2 subset enumeration, using MOSEK.
# Requires the mosek Python package and a working license.
python scripts/run_mnist.py --digit 0 --t 1 --k 50 --solver mosek --threads 4 --output results/mnist_mosek

# Explicit alternative: the current manuscript's multiset definition.
python scripts/run_mnist.py --digit 0 --t 2 --candidate-rule paper-multisets --solver gurobi --output results/mnist_paper_definition
```

| Digit | Paper t1 objective / seconds | Paper t2 objective / seconds |
|---|---|---|
| 0 | 0.0132 / 556 | 0.0127 / 9213 |
| 1 | 0.0174 / 56 | 0.0169 / 1148 |
| 2 | 0.0279 / 253 | 0.0274 / 2716 |

At t2, direct counting of the packaged inputs gives distinct-subset versus multiset support sizes of **1669 vs 1717**, **1353 vs 1399**, and **1837 vs 1860** for digits 0,1,2. These are demonstrably different optimization problems. The default `notebook-subsets` recovers the saved computation; `paper-multisets` is a newly supplied implementation of the manuscript definition. Neither mode has been verified against all published objectives here.

**Incorrect historical export:** notebook cell 73 computes digit2/t2 in `out3`, but cell 75 saves `out2`. Its saved digit2/t2 S/z arrays are byte-identical to digit1/t2. They are retained in `artifacts/mnist/` as historical evidence and must not be used as a valid digit2 barycenter. The digit2 PNG uses `out3`. The portable runner saves the correct result object under explicit digit/t/rule filenames. Digit0/t2 has a solve cell and image but no recovered numerical S/z export.

PNG rasterization is display postprocessing, not an input to objective evaluation. The notebook's `gaussian_sigma=0.8` argument has no effect in the selected bilinear rasterization mode; no Gaussian smoothing is applied by that branch.

## 5. Sign gestures: Figures 4–6

Use A, C, and D for paper gestures 1, 2, and 3. Class B is an ancillary experiment. Training uses the nine bundled JPEGs. OpenCV resizes to 64×64 with `INTER_AREA`; each retained pixel is `(x,y,r,g,b)` with coordinates/colors normalized to `[0,1]`. Brightness must exceed `1e-6`, and retained pixels have uniform weights. Bright backgrounds are retained by this rule.

The recovered column generator adds candidate-support rows and their transport variables. The current manuscript describes tuple-column generation. They express related fixed-support optimization problems, but identical solver trajectories and timing cannot be inferred from that description.

Run the following sequence separately for A, C, and D:

```bash
python scripts/run_sign.py t1 --class A --time-limit 3600 --output-dir results/A_t1_stage1
python scripts/run_sign.py t1 --class A --time-limit 3600 --initial-support results/A_t1_stage1/S.npy --initial-weights results/A_t1_stage1/z.npy --output-dir results/A_t1_stage2
python scripts/run_sign.py hybrid --class A --time-limit 3600 --initial-support results/A_t1_stage1/S.npy --initial-weights results/A_t1_stage1/z.npy --output-dir results/A_hybrid_stage1
python scripts/run_sign.py hybrid --class A --time-limit 3600 --initial-support results/A_hybrid_stage1/S.npy --initial-weights results/A_hybrid_stage1/z.npy --output-dir results/A_hybrid_stage2
```

Hybrid settings: K=5 nearest atoms per marginal, at most 200 midpoint pairs per center per marginal pair, seed 0, no radius filter, active threshold `1e-12`, reduced-cost tolerance `1e-9`, one atom added at a time; Gurobi `Threads=4`, `Method=1`. With K=5, the cap does not truncate the 25 combinations for a marginal pair. The second hybrid stage rebuilds candidates around its new active atoms.

| Gesture | t1 1h | t1 2h | Hybrid 1h | Hybrid 2h |
|---|---:|---:|---:|---:|
| A / Figure 4 | 0.00368 | 0.00358 | 0.00355 | 0.00348 |
| C / Figure 5 | 0.0115 | 0.0115 | 0.0111 | 0.0110 |
| D / Figure 6 | 0.00957 | 0.00949 | 0.00917 | 0.00898 |

These are paper targets. The stored D first-hybrid metadata filename reports approximately 0.00911717, which differs from 0.00917 in the caption.

Time limits apply to the internal CG phase, not preprocessing and candidate/cost construction. Some saved total times exceed the one-hour labels. A hybrid stage also depends on prior t1 computation. Hybrid solutions are feasible approximations over local supports; the full Algorithm 4 approximation guarantee does not automatically apply. The core still constructs dense cost arrays, so sign runs require substantial RAM.

## 6. Table 1: cached reconstruction and fresh classification

```bash
# Verified reconstruction from historical numeric scores; no solver required.
python scripts/run_sign.py cached-table --output results/table1_cached.json

# Fresh distance calculations using the archived barycenters; potentially slow.
python scripts/run_sign.py classify --variant t1_7200 --output-dir results/classify_t1
python scripts/run_sign.py classify --variant hybrid3600 --output-dir results/classify_hybrid

# Freshly trained barycenters can be selected directly.
python scripts/run_sign.py classify --variant t1_7200 --model-dir A=results/A_t1_stage2 --model-dir C=results/C_t1_stage2 --model-dir D=results/D_t1_stage2 --output-dir results/classify_fresh_t1
```

Historical score reconstruction retains A/C/D test images and **re-ranks distances among A/C/D**. Merely filtering the original four-class predictions is insufficient.

| Source | A correct | C correct | D correct | Overall |
|---|---:|---:|---:|---:|
| Two-hour t1, notebook cell 33 | 25/25 | 22/25 | 25/25 | 72/75 = 96% |
| First hybrid stage, notebook cell 34 | 25/25 | 25/25 | 25/25 | 75/75 = 100% |

The matching notebook code uses **Sinkhorn**, regularization 0.01, maximum 1,000 iterations, and tolerance `1e-9` on changes in a scaling vector. It returns the transportation cost of the resulting plan, not a certified exact Wasserstein distance. The runner retains that method to support numerical provenance. A fresh exact-distance classifier would be a separate experiment.

The saved full score lists, source cell numbers, and original notebook hash are in `data/sign/cached_classification_scores.json`. Fresh classification removes only exact zero-mass support rows and records this reduction, preserving the represented probability measure while reducing memory use. Classification writes `classification.json` and `predictions.csv`. Training writes `S.npy`, `z.npy`, `barycenter.png`, and `run.json`.

## 7. Changes, validation, and remaining gaps

The package preserves `originals/wb_bary_algorithms.py` byte for byte. Its working copy has narrow CG fixes: correct final-polish time-budget handling, stop pricing when a restricted master solve hits its time limit, return the objective belonging to the saved feasible solution, and use equality-expression constraint syntax accepted by Gurobi 11 and 12. The original `None`/zero time handling could crash or accidentally allow an extremely long final solve. See `CHANGES.patch`.

Dedicated runners remove dependency on notebook execution order, use explicit data/output paths, and distinguish new optimization from cached evidence. Original exploratory W1 bugs and unused incomplete helper functions are preserved in the archived source; they are outside the Section 5 runners. Do not use `Run All` on the historical notebooks as the reproduction workflow.

Validation evidence is in `validation/`: analytical W2 mass/marginal/objective checks, a one-image MNIST end-to-end check, a small ellipse workflow, data inspections, exact candidate-set counts, and Table 1 reconstruction. See `validation/VALIDATION.md` for the final test scope and runtime limitations. A successful small check does not establish paper-scale timings or every published value.

Before claiming complete numerical reproduction, resolve the MNIST subset/multiset and image-selection differences, recover or choose the ellipse sampling protocol, settle the sign timing and D-objective discrepancy, and rerun the full cases in a recorded solver environment.

## 8. Package contents and exclusions

| Folder | Purpose |
|---|---|
| `code/` | Working algorithm module, MNIST utilities, analytical smoke test |
| `scripts/` | Explicit ellipse, MNIST, sign training/classification runners |
| `data/` | Required ellipse data, selected MNIST images, sign inputs/tests, cached scores |
| `artifacts/` | Historical images and arrays; not newly computed results |
| `originals/` | Original core and six notebooks with outputs cleared |
| `third_party/` | Relevant exact-reference source snapshot and original license |
| `references/` | The manuscript used for this mapping |
| `validation/` | Results of checks performed during packaging |

Excluded as unnecessary for these results: `EnergyGuidedBarycenters-main`, `Wasserstein2Benchmark-main`, `WassersteinIterativeNetworks-main`, class-B training assets, exploratory skin-extraction workflows, W1 demonstrations, virtual environments, caches, and unrelated MATLAB comparison trees. The current Figure 2 does not require the older IBP, debiased Sinkhorn, Frank–Wolfe, or MAAIPM comparison outputs.

Third-party baseline source/data derive from [eboix/high_precision_barycenters](https://github.com/eboix/high_precision_barycenters), accompanying [Altschuler and Boix-Adsera, JMLR 2021](https://jmlr.org/papers/v22/20-588.html). The local snapshot's LGPL v3 license is preserved in `third_party/high_precision_barycenters/LICENSE.md`. No new license has been asserted for the author's code or the selected datasets. This is a local working package; source metadata, not a new public release.
