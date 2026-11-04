#!/bin/bash
# J1630+4520 elliptical lens joint fit on the Loong cluster (galaxylens gl.fit).
# First elliptical-lens run: MGE stellar mass + gNFW + jampy, sigma_ap = 267 km/s.

#SBATCH --job-name=j1630_ellip
#SBATCH --output=j1630_ellip_%j.log
#SBATCH --time=12:00:00
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G

set -e

CONDA_ENV="$HOME/.conda/envs/autolens_jax"  # edit to your env
export PATH="$CONDA_ENV/bin:$PATH"
export PYTHONNOUSERSITE=1
export PIP_USER=false
export PYTHONUNBUFFERED=1

# Multi-core Nautilus iron rule (130326 lesson): with number_of_cores > 1 the
# per-worker BLAS threads must be pinned to 1 or the run silently crashes.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

cd "$HOME/galaxylens"  # edit to your repo path

export PYTHONPATH=$PWD/src
export GALAXYLENS_DATA_ROOT=/path/to/observations  # edit
export GALAXYLENS_OUTPUT_ROOT="$HOME/galaxylens/outputs"

PYTHON="$CONDA_ENV/bin/python"
echo "=== python = $PYTHON ==="
$PYTHON --version

$PYTHON -m galaxylens.api.runner --config configs/fit/j1630_ellip.yaml

echo "=== galaxylens J1630 fit done ==="
