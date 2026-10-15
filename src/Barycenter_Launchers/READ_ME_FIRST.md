# Run the paper experiments on your Mac

## Start

**In Finder, double-click `Start Here.command`.** It opens Terminal with a menu. No installation or Conda activation is required.

You can also double-click an individual launcher:

| Launcher | What it runs |
|---|---|
| `Check Setup.command` | Small SciPy/Gurobi tests, nine image checks, and reconstruction of the cached classification table |
| `Run Ellipses.command` | Full ellipse comparison, one chosen t value, or saved-reference rendering |
| `Run MNIST.command` | Your selected digits and t values, using all 50 input images per digit |
| `Run Sign Training.command` | Four sequential optimization stages for each selected gesture |
| `Run Classification.command` | Fresh A/C/D test-image distance calculations using the **archived** barycenters |

Press Return to accept a menu default. In the main menu, **All recovered experiments** queues the ellipse comparison, six MNIST cases, twelve sign stages, and two fresh classifications using the newly trained sign barycenters. It can take longer than a day.

If the operating system associates `.command` files with an editor, right-click the file and choose **Open With → Terminal**.

## While a job runs

- Keep the Terminal window open, plug in the laptop, and leave the lid open.
- Jobs run one at a time. Solver and numerical-library thread limits are set to four. An additional launcher will decline to start while a queue is active.
- The launcher prevents automatic idle sleep while its queue runs. Closing the lid or manually sleeping the Mac can still suspend work.
- **Control-C stops the current job and the queue.** Completed outputs are retained. Interruptions do not create solver checkpoints; a new launch starts a new run.
- Some preparation/optimization phases do not print frequently. `status.json` records the active job and its process ID; the job log records output as it arrives.

## Find the results

Each launch creates a new folder inside `Barycenter_Launchers/results/`, for example:

```text
results/20260914_231500_123456_ellipse/
    status.json
    logs/
    ellipses/
```

The Terminal prints the exact folder. Completed optimization outputs include numerical arrays, an image, and a JSON record. `status.json` lists each job's command, start/finish time, exit status, and log path. Failed jobs stop the remaining queue so a failed warm start cannot silently propagate.

To classify **new** models, the main menu's all-experiments option wires them automatically. For a separately completed sign run, the underlying package also accepts `--model-dir A=... --model-dir C=... --model-dir D=...`; see the reproduction guide in the sibling `closing_gap_reproduction` folder.

## Experiment settings

**Ellipses:** MOSEK is the default LP solver because it handles the full sparse model without Gurobi's size-limited license restriction. Gurobi remains selectable. The full option runs deterministic t=1 and t=2, then ten sampled t=1 runs with seeds 0–9. The historical ten-seed sequence was not recovered. The exact reference is drawn from its saved arrays; the costly independent reference solver is not invoked. MOSEK selection applies only to ellipses; MNIST and sign training still use Gurobi.

**MNIST:** the default is the saved notebook's distinct-subset computation. You can explicitly choose the current manuscript's multiset definition. They differ at t=2, so outputs record the selected definition. The selected data are the notebook's first 50 training images per digit.

**Signs:** A, C, D are paper gestures 1, 2, 3. At the default 60 minutes per stage, each class runs:

1. Initial t=1.
2. Continued t=1 from stage 1.
3. First hybrid stage from stage 1.
4. Second hybrid stage from stage 3.

Three classes have twelve one-hour solver budgets, plus input preparation and model construction; stages may finish early. The minute budget applies to each internal solver stage, not total wall time. Shorter budgets are useful for exploratory runs and do not reproduce the paper's time settings. The final classifications in the all-experiments queue select the relevant actual result directories, even if you choose shorter stage budgets; the legacy variant strings remain labels.

**Classification:** fresh distances use the recovered Sinkhorn implementation and its original regularization, not a certified exact-distance solver. The quick setup test only reconstructs the table from cached scores.

## Your configured environment

All launchers use this interpreter:

```text
/opt/anaconda3/bin/python
```

MOSEK 11.2.6 is installed in this environment, and a licensed test LP solved successfully. The ellipse backend builds a sparse LP. These launchers do not install packages, change solver licenses, or alter Python environments.

Keep `Barycenter_Launchers` beside `closing_gap_reproduction`; the launcher finds the code and data in that sibling folder. Missing archival notebooks do not prevent running: all current runtime scripts, helpers, and data matched the delivered manifest when checked.

Only setup checks were run while creating these launchers. The full research experiments are ready for you to start.
