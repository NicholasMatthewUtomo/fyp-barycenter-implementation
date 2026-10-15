# Validation performed

## Passed

- Six analytical W2 checks with SciPy/HiGHS: point masses at 0 and 2 give objectives 2 (t1) and 1 (t2); opposite unit-square diagonals give 0.5 and 0.25. Probability masses, transport marginals, nonnegativity, and independently reconstructed objectives were checked. See `core_scipy.json`.
- Six persistent-CG checks in **each** of Gurobi 11.0.3 and 12.0.3: analytic objectives; unlimited and five-second budgets; addition of a missing midpoint; saved-transport consistency; immediate expected failure at zero budget. See `core_gurobi.json`.
- MNIST runner end to end on one digit-1 training image with Gurobi 11.0.3: 96 active atoms, mass approximately one, objective approximately zero. This is a one-measure diagnostic, not Figure 3. See `mnist_one_image.json`.
- Ellipse runner on the first three marginals reduced to four atoms each: deterministic t1 objective 0.0318055556; t2 0.0227719907; sampled best of seeds 0 and 1 gives 0.0330092593. Probability and expected ordering checks passed. See `ellipse_smoke.json`.
- Cached exact ellipse was rendered with the required x reflection and visually inspected. No reference optimization was rerun. See `ellipse_reference_render.json`.
- Cached Table 1 scores were filtered and re-ranked over A/C/D: 72/75 for two-hour t1, 75/75 for first hybrid stage. See `table1_cached.json`.
- All nine sign training JPEGs converted into finite 5D normalized measures and rendered to 64×64 RGB. Archived A/C/D first-t1 rasterizations matched their saved RGB arrays exactly. All twelve archived S/z pairs had matching shapes and valid masses.
- MNIST subset-versus-multiset candidate counts were checked independently on the original integer grid and against the new candidate builder. See `mnist_candidate_sets.json`.
- Python parsing, CLI help/inspection, invalid argument rejection, default data paths, and archive hashes were checked. Archived numerical files known to be incorrect are documented in the guide, not treated as passing numerical evidence.

## Tested environments

| Role | Python | NumPy | SciPy | Gurobi |
|---|---|---|---|---|
| SciPy checks | 3.13.5 | 2.1.3 | 1.15.3 | None |
| Paper-version Gurobi checks | 3.8.3 | 1.24.4 | 1.10.1 | 11.0.3 |
| Gurobi API compatibility checks | 3.10.6 | 1.23.1 | 1.11.1 | 12.0.3 |

Sign image helpers were also checked in an existing Python 3.11 environment with NumPy 2.2.6 and OpenCV 4.12. That environment's SciPy 1.11.4 binary was incompatible with NumPy and could not run Gurobi's sparse-matrix operations. The successful Gurobi checks used the two compatible environments above. No environments or licenses were changed.

The requirements files were not installed into a newly created clean environment. Their version ranges are not a historical environment lock. Solver version compatibility on small instances does not establish identical solutions in degenerate LPs, identical timings, or identical full-data classification.

## Not performed

Full-data ellipse optimization, 50-image-per-digit MNIST optimization, hour-long sign training, fresh classification of all 75 test images, independent recomputation of the exact-reference baseline, and testing of the old W1 routines were not performed.

The positive finite-budget tests finished optimally before their limits. A positive time limit interrupting a large restricted master was not forced; that particular branch has static review plus the zero-budget regression check, not an exhaustive time-limit stress test.

## Repeat the small checks

From the package root:

```bash
python code/smoke_test.py --json results/core_scipy.json
python code/smoke_gurobi.py --json results/core_gurobi.json
python scripts/run_ellipses.py --smoke --mode all --seeds 0 1 --solver scipy --output results/ellipse_smoke
python scripts/run_sign.py cached-table --output results/table1_cached.json
```

Use an environment with the matching dependencies; `smoke_gurobi.py` additionally needs a usable license.
