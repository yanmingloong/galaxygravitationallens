#!/bin/bash
# =============================================================================
# run_selection_cluster.slurm — run mock_pipeline.selection on a SLURM cluster
#
# Example: full TNG disk-lens analog selection for one target on a cluster.
# The pipeline code is environment-agnostic — this script only supplies the
# SLURM resources, env vars, and the worker counts.
#
# Usage:
#   sbatch run_selection_cluster.slurm
#   sacct -j <jobid>          # check status
#
# Pitfalls (from the project's census runs):
#   - Non-interactive jobs do NOT source the login profile: export PATH
#     explicitly (avoid ~/bin/python hijacking).
#   - Set --cpus-per-task to (scan workers + l3 workers) or the threads
#     will thrash.
# =============================================================================
#SBATCH --job-name=galaxylens_sel
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=selection_%j.log

set -e

# --- environment ------------------------------------------------------------
# Python env with galaxylens installed (pip install -e .) — edit to your env
export PATH=$HOME/.conda/envs/autolens_jax/bin:$PATH

# Data / output roots (edit to your cluster paths)
export GALAXYLENS_DATA_ROOT=/Raid6/1b/teamlwp_b/ymdata/tng_census_catalog
export GALAXYLENS_OUTPUT_ROOT=/Raid6/1b/teamlwp_b/ymdata/tng_census_catalog/out

# TNG API key (downloads; keep out of git)
export TNG_API_KEY=...

# --- run --------------------------------------------------------------------
echo "=== galaxylens selection start $(date) ==="
python -m galaxylens.mock_pipeline.selection.run \
    --target j2141 \
    --data-root "$GALAXYLENS_DATA_ROOT" \
    --outdir "$GALAXYLENS_OUTPUT_ROOT" \
    --workers 8 \
    --l3-workers 8
echo "=== galaxylens selection done $(date) ==="
