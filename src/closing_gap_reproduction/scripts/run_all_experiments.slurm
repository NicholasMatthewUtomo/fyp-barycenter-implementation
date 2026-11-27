#!/bin/bash
# =====================================================================
# SLURM batch script for unified Wasserstein barycenter experiments
# Paratera GPU cluster — supports D-PDLP (GPU-accelerated PDLP)
#
# Submit with:  sbatch scripts/run_all_experiments.slurm
# =====================================================================
#SBATCH --job-name=barycenter_unified
#SBATCH --output=results/unified_run/slurm_%j.out
#SBATCH --error=results/unified_run/slurm_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --account=YOUR_ACCOUNT_NAME      # <-- REPLACE with your Paratera account
#SBATCH --qos=YOUR_QOS                   # <-- REPLACE (optional, remove if not needed)

# ----------------------------------------------------------------------
# 1. Load environment modules (adjust for Paratera)
# ----------------------------------------------------------------------
module load cuda/12.x              # D-PDLP requires CUDA
# module load anaconda3            # or module load miniconda3, python/3.11, etc.

# ----------------------------------------------------------------------
# 2. Activate Python environment
# ----------------------------------------------------------------------
source /opt/anaconda3/etc/profile.d/conda.sh   # adjust path as needed
conda activate barycenter

# ----------------------------------------------------------------------
# 3. Verify GPU is visible
# ----------------------------------------------------------------------
echo "=== GPU Info ==="
nvidia-smi || echo "nvidia-smi not found (check CUDA installation)"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-not set}"
echo ""

# ----------------------------------------------------------------------
# 4. Solver license environment variables (NOT needed for D-PDLP)
# ----------------------------------------------------------------------
# D-PDLP is open source — no license file required.
# If you also want to test MOSEK/Gurobi, uncomment:
# export MOSEKLM_LICENSE_FILE=$HOME/mosek.lic
# export GRB_LICENSE_FILE=$HOME/gurobi.lic

# ----------------------------------------------------------------------
# 5. Change to repo directory and run
# ----------------------------------------------------------------------
cd $SLURM_SUBMIT_DIR

echo "Host: $(hostname)"
echo "Python: $(which python)"
echo "Date: $(date)"
echo ""

# Choose one solver:
#   --solver mosek       (requires MOSEK license at $HOME/mosek.lic)
#   --solver gurobi      (requires Gurobi license)
#   --solver scipy       (no license, CPU, dense LPs — may OOM)
#   --solver dpdlp       (GPU-accelerated PDLP, no license)

python scripts/run_all_experiments.py \
    --ecg200-data data/ecg200 \
    --finance-data data/finance/market_data.csv \
    --output-root results/unified_run \
    --solver dpdlp \
    --dpdlp-binary /path/to/dpdlp/build/dpdlp \
    --dpdlp-work-dir $HOME/dpdlp_work \
    --k 10 --points 96 \
    --finance-k 10 \
    --max-boxes 500 \
    --gap-tol 1e-4 \
    --time-limit 300 \
    --max-blocks-per-T 5000

echo "EXIT CODE: $?"
echo "Finished: $(date)"
