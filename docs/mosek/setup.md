# Wasserstein Barycenter Research Lab Setup & Checkpoint Guide

This repository contains the replication environment for your Final Year Project (FYP) on **Wasserstein Barycenter Experiments**. This document outlines the project structure, step-by-step setup troubleshooting, and configuration adjustments required to run large-scale computational optimization benchmarks inside Windows Subsystem for Linux (WSL).

---

## 📂 Repository Directory Layout

The repository is structured into two main peer directories:

```text
Barycenter_implementation/
├── Barycenter_Launchers/                 # Command-line configuration menus and workspace launchers
│   ├── results/                         # Datetime-stamped experiment outputs and log folders
│   ├── Check Setup.command              # Script to test core package availability
│   ├── check_images.py                  # Verification module for sign-language imagery arrays
│   ├── laptop_launcher.py               # Interactive CLI launcher matrix for all figures
│   ├── run_worker.py                    # Multi-threaded job scheduling worker
│   └── Start Here.command               # Main shell script to spin up the program interface
│
└── closing_gap_reproduction/            # Main scientific research implementation module
    ├── code/                            # Backend algorithmic implementations
    │   ├── smoke_test.py                # Initial SciPy environment verification check
    │   └── wb_bary_algorithms.py        # Core Wasserstein barycenter mathematical logic (43+ solver conditions)
    ├── scripts/                         # Individual simulation execution drivers
    │   ├── run_ellipses.py              # Figure 2 (Ellipse convergence) simulation driver
    │   ├── run_mnist.py                 # Figure 3 (MNIST digits) optimization runner
    │   └── run_sign.py                  # Figures 4–6 (Multi-stage Sign Gesture) matrix driver
    ├── data/                            # Input datasets (MNIST arrays, sign imagery vectors)
    ├── requirements.txt                 # Core dependencies (NumPy, SciPy, Matplotlib, Pillow)
    ├── requirements-gurobi.txt          # Gurobi pipeline module configuration
    └── requirements-sign.txt            # Sign experiment package definitions (OpenCV, etc.)
```

---

## 🛠️ Environment Configuration Checkpoint

### 📦 1. WSL Storage Allocation Strategy
* **Physical Location:** Your entire Ubuntu Linux OS environment is isolated within an extensible virtual disk container image file (**`ext4.vhdx`**).
* **Target Path:** `D:\Program Files\Ubuntu\ext4.vhdx`
* **Impact:** Every tool installed via `apt install` or package fetched via `pip3 install` grows *inside* this file on your **external D drive (Kingston)**, keeping your local C operating drive fully insulated from exhaustion during deep calculations.

### 🔑 2. The Solver Constraint: Switching from Gurobi to MOSEK
By default, the script tries to route linear program matrix calculations through `gurobipy`. However, installing it via standard pip channels provisions a **Restricted Size-Limited Trial License** capped strictly at 2,000 variables and 2,000 linear equations. Your experimental bounds exceed this.

Since you possess a full institutional **MOSEK** license, the system handles it natively without needing to change optimization logic. Use the following integration checklist:

#### Step A: Provide the License Token (`mosek.lic`)
MOSEK requires its authorization token inside your Linux home configuration path. Run this in your WSL terminal:
```bash
mkdir -p ~/mosek
cp /mnt/c/Users/Nicholas/mosek/mosek.lic ~/mosek/mosek.lic
```

#### Step B: Install the Linux Solver Package
```bash
pip3 install mosek opencv-python-headless --break-system-packages
```
*(Note: `--break-system-packages` is mandatory on Ubuntu 24.04+ profiles to allow global overrides inside a dedicated research sandboxed system).*

#### Step C: Update `laptop_launcher.py` for Permanent Redirection
Because lines **88** and **92** of your interaction file hardcode `--solver gurobi`, open `Barycenter_Launchers/laptop_launcher.py` in VS Code and swap those target definitions over to `'mosek'`:

```python
# Inside laptop_launcher.py -> build_jobs() function
if e != 'reference': args += ['--solver', 'mosek', '--threads', THREADS] # Line 88
# ...
['--digit', digit, '--t', t, '--solver', 'mosek', '--candidate-rule', opts['rule'], '--output', run/'mnist'] # Line 92
```

---

## 🚀 Execution Commands

To execute tests and verify solver handshakes, use proper Linux separation syntax:

```bash
# Move to your workspace launcher path
cd /mnt/c/Users/Nicholas/Documents/University/Academic/Research/fyp/Barycenter_implementation/Barycenter_Launchers

# Initiate the interactive experiment platform
bash "Start Here.command"
```

## ⚠️ Known Shell Patches Applied
1. **Sleep Mitigation Block:** Created an explicit mock utility trace shell layout via `echo -e '#!/bin/sh\nexit 0' | sudo tee /usr/bin/caffeinate > /dev/null` to intercept and gracefully satisfy rigid macOS sleep-prevention hook checks (`/usr/bin/caffeinate`) that would otherwise break validation pipelines inside Linux setups.
2. **Path Constraints:** Cleaned out Windows style backslash patterns (`\`) on line 3 of your execution command file, replacing them with a uniform native runtime interpreter hook `PYTHON = Path('/usr/bin/python3')`.
