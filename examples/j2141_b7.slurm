#!/bin/bash
# J2141 B7 joint fit on the Loong cluster (galaxylens gl.fit, stage 3).
# Environment/config copied from the B7 run that already worked (run_b7_cluster.slurm).

#SBATCH --job-name=j2141_b7_gl
#SBATCH --output=j2141_b7_gl_%j.log
#SBATCH --time=12:00:00
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

set -e

CONDA_ENV="$HOME/.conda/envs/autolens_jax"  # edit to your env
export PATH="$CONDA_ENV/bin:$PATH"
export PYTHONNOUSERSITE=1
export PIP_USER=false
export PYTHONUNBUFFERED=1

cd "$HOME/galaxylens"  # edit to your repo path

export PYTHONPATH=$PWD/src
export GALAXYLENS_DATA_ROOT=/path/to/observations  # edit
export GALAXYLENS_OUTPUT_ROOT="$HOME/galaxylens/outputs"

PYTHON="$CONDA_ENV/bin/python"
echo "=== python = $PYTHON ==="
$PYTHON --version

$PYTHON -m galaxylens.api.runner --config configs/fit/j2141_barnabe.yaml

echo "=== galaxylens J2141 fit done ==="
