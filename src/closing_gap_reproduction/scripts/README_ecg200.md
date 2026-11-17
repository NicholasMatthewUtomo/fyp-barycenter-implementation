# Run ECG200 manually

The runner defaults to **96 original points, k=10 heartbeats per class, t=1**,
using `algo4_W1_subspace_enumeration` and the existing MOSEK solver defaults.
It computes separate normal and abnormal barycenters. Each class uses the first
five training and first five testing examples, selected in alternating split
order. Original amplitudes are unchanged; time is j/95 and point weights are
1/96. This is a 20-heartbeat subset of the authentic 200-heartbeat ECG200 dataset.

On this computer, open Ubuntu with `wsl -d Ubuntu` in Windows Terminal, then run:

```bash
cd /mnt/c/Users/Albert/Downloads/fyp-barycenter-implementation/src/closing_gap_reproduction

# Inspect selection, candidate count, and LP size without solving or writing results.
/home/albert/venvs/barycenter/bin/python -B scripts/run_ecg200.py --points 96 --k 10 --t 1 --inspect

# Start both class solves yourself. Leave this terminal open.
/home/albert/venvs/barycenter/bin/python -u -B scripts/run_ecg200.py --points 96 --k 10 --t 1
```

Results default to `results/ecg200_k10_n96_t1`. For each class, the runner saves
selected inputs and indices, support coordinates, optimized weights, objective,
runtimes, input/comparison PNGs, CSV, NPZ, and JSON reports. An existing output
directory containing ECG200 results is protected from overwriting; use a fresh
`--output` directory for another attempt.

To run just one class, add `--class normal` or `--class abnormal`. To keep their
outputs separate, also use distinct directories with `--output`.

In this existing 2D W1 implementation, **t=1 creates the Cartesian product of
distinct time and amplitude coordinates**, rather than just pooling the input
points as t=2 does. The inspection reports the actual uncapped candidate count
and LP size. Inspection of the default selection gives:

| Class | Candidate points | LP variables |
|---|---:|---:|
| Normal | 92,160 | 88,565,760 |
| Abnormal | 91,968 | 88,381,248 |

These models are likely to exceed this computer's 8 GB RAM. Reducing k does not
guarantee a smaller model than the previous t=2 experiment, because t changes
the candidate set. Core algorithms and solver code are unchanged. No solver run
was started while preparing this configuration.

The legacy `run_ecg200_full.sh` explicitly retains its old all-samples, 96-point,
t=2 configuration; use the Python command above for the new k=10, t=1 run.
