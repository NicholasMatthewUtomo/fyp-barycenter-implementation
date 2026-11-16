# ECG200 with 12 points per heartbeat

This experiment retains all 200 ECG200 heartbeats from both partitions
(100 training and 100 testing): 133 normal and 67 abnormal.

Each waveform is approximated by retaining original time indices
`[0, 9, 17, 26, 35, 43, 52, 60, 69, 78, 86, 95]`. Coordinates are
`(original_index / 95, original_amplitude)`, with uniform probability mass
`1/12` per point. No interpolation or amplitude normalization is applied.
The original 96-point waveforms are also saved for comparison.

The experiment uses Algorithm 4 (`algo4_W1_subspace_enumeration`) at `t=2`
with the existing MOSEK implementation and default parameters. It uses the
existing L1 ground distance in time and amplitude. No candidate cap or PDLP
is used. The core algorithms and solver were not changed for this experiment.

The two classes run in separate processes, abnormal first and then normal.
The normal model has 2,548,812 variables and the abnormal model 647,220.

## Progress checkpoint

The abnormal class completed with objective `0.31649734069968016` and
algorithm runtime `1877.5463240146637` seconds (31.3 minutes). Its actual
barycenter NPZ, CSV, and comparison PNG are saved under `abnormal/` and were
verified; see `abnormal/verification.json`. Its observed RAM use was about
0.75 GiB. Raw solver weights are preserved, including numerical residuals
on the order of `1e-8` per atom.

The normal class is running in WSL. Its observed RAM use after model
construction was about 1.94 GiB. Its files will be saved automatically if
MOSEK returns an optimum. Check `normal/ecg200_summary.json` for the latest
status; `launch_status.json` is only a snapshot. Both classes' input data
were verified to cover all 200 original train/test samples exactly once.

## Output status and interpretation

Read each class's `ecg200_summary.json` or per-class JSON to check status.
`running` means no completed solution has been returned. `completed` means
MOSEK returned optimal status and the runner's support/mass checks passed.
`process_status.json` records the process exit code and wall time after exit.

- `*_inputs.npz`: original and retained waveforms, retained indices, input
  measures, labels, and partition provenance. These are input data.
- `*_inputs.png`: original waveforms overlaid with their 12-point inputs.
- Result `*.npz` (without `_inputs`): actual barycenter candidate coordinates
  `S`, weights `z`, objective, runtime, and sample provenance.
- Result `*.csv`: normalized time, amplitude, and probability mass per atom.
- Result `*.png` (without `_inputs`): inputs alongside the weighted barycenter
  distribution. Candidate atoms with mass above `1e-12` are displayed.
- `run.log`: process output. The existing solver does not emit an iteration
  log through this runner, so a quiet log does not establish completion.

The earlier deleted `results/ecg200` and `results/ecg200_retry` directories
are not used by this experiment.

To run from `src/closing_gap_reproduction` in a licensed MOSEK environment,
choose a fresh output directory:

```sh
python -B scripts/run_ecg200.py --points 12 --output results/ecg200_points12_next
```
