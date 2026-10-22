"""Sampler construction: local (in-process Nautilus) vs slurm (submit script).

Core fitting code is scheduler-agnostic; this module only adapts *how the
sampler is launched*. ``backend: local`` runs ``af.Nautilus`` directly (laptop /
workstation / login node); ``backend: slurm`` renders an SBATCH script from a
template whose cluster-specific fields (partition, account, conda env path,
workdir, memory) come from the ``SearchSpec`` -- so switching slurm clusters is
a config edit, not a code change. Non-slurm schedulers (PBS/SGE/LSF) would be
new backends here.

The template follows the Loong-cluster script that already worked for B7
(``examples/j2141_b7.slurm``): it exports ``$CONDA_ENV/bin`` to PATH directly
(non-interactive ssh does not source the profile, so ``conda`` is unavailable),
sets ``PYTHONNOUSERSITE=1`` (a stale ``~/.local`` shadows conda deps), and calls
the full ``$CONDA_ENV/bin/python`` path (a ``~/bin/python`` hijacks ``python``).

References
----------
- Lange 2023, Nautilus (the sampler being wrapped)
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from .config_loader import SearchSpec

_SLURM_TEMPLATE = """#!/bin/bash
#SBATCH --job-name={name}
#SBATCH --output={name}_%j.log
#SBATCH --time={time_hours}:00:00
#SBATCH --partition={partition}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task={cpus_per_task}
{mem_line}{account_line}set -e

CONDA_ENV="{conda_env_path}"
export PATH="$CONDA_ENV/bin:$PATH"
export PYTHONNOUSERSITE=1
export PIP_USER=false
export PYTHONUNBUFFERED=1

cd {workdir}

export PYTHONPATH=$PWD/src
export GALAXYLENS_DATA_ROOT={data_root}
export GALAXYLENS_OUTPUT_ROOT={output_root}

PYTHON="$CONDA_ENV/bin/python"
echo "=== python = $PYTHON ==="
$PYTHON --version

$PYTHON -m galaxylens.api.runner --config {config_path}

echo "=== galaxylens {name} fit done ==="
"""


def _disable_autofit_corner_plot():
    """Disable autofit's built-in anesthetic corner plot.

    The anesthetic corner plot runs at the END of a nested-sampling fit
    (``NonLinearSearch.plot_results``) and depends on pandas/anesthetic. On the
    cluster with ``PYTHONNOUSERSITE=1`` the ``~/.local`` pandas is hidden, so
    this step hangs until the slurm time limit (observed: job 128011 sampled
    fine in ~1h20m then hung ~10h in "Updating results"). galaxylens draws its
    OWN weighted corner plot from ``samples.csv`` (``api.plotting.corner``), so
    autofit's is redundant -- disable it.
    """
    from autonerves import conf

    conf.instance["visualize"]["plots_search"]["nest"]["corner_anesthetic"] = False


def _local_search(spec: SearchSpec, output_root: Path):
    import autofit as af

    _disable_autofit_corner_plot()
    return af.Nautilus(
        path_prefix=str(output_root),
        name=spec.name,
        n_live=spec.n_live,
        iterations_per_update=5000,
        number_of_cores=spec.number_of_cores,
        use_jax_vmap=False,
        random_seed=42,
    )


def slurm_script(
    spec: SearchSpec,
    output_root: Path,
    config_path: str,
    data_root: str,
) -> str:
    """Render an SBATCH submission script (does NOT submit).

    ``config_path`` is the fit-recipe YAML the runner loads on the cluster;
    ``data_root`` becomes ``GALAXYLENS_DATA_ROOT``. Cluster-specific fields come
    from ``spec`` (partition, account, conda env path, workdir, memory).
    """
    if not spec.partition:
        raise ValueError("backend='slurm' requires search.partition")
    if not spec.conda_env_path:
        raise ValueError("backend='slurm' requires search.conda_env_path")
    if not spec.workdir:
        raise ValueError("backend='slurm' requires search.workdir (cluster project root)")
    mem_line = f"#SBATCH --mem={spec.mem}\n" if spec.mem else ""
    account_line = f"#SBATCH --account={spec.account}\n" if spec.account else ""
    return _SLURM_TEMPLATE.format(
        name=spec.name,
        partition=spec.partition,
        mem_line=mem_line,
        account_line=account_line,
        cpus_per_task=spec.cpus_per_task,
        time_hours=int(spec.time_hours),
        data_root=data_root,
        output_root=str(output_root),
        conda_env_path=spec.conda_env_path,
        workdir=spec.workdir,
        config_path=config_path,
    )


def submit_slurm(
    spec: SearchSpec,
    output_root: Path,
    config_path: str,
    data_root: str,
    submit: bool = True,
) -> dict:
    """Render + write + (optionally) submit an SBATCH job.

    Returns a dict with ``backend``, ``script_path`` and ``job_id`` (or ``None``
    when ``submit=False`` or ``sbatch`` is unavailable -- e.g. running from a
    non-slurm machine, in which case the script is still written and the caller
    can submit it by hand).
    """
    script = slurm_script(spec, output_root, config_path, data_root)
    output_root.mkdir(parents=True, exist_ok=True)
    script_path = output_root / f"{spec.name}.slurm"
    script_path.write_text(script, encoding="utf-8")

    result = {"backend": "slurm", "script_path": str(script_path), "job_id": None}
    if not submit:
        result["note"] = "script written; submit=False"
        return result
    try:
        proc = subprocess.run(
            ["sbatch", str(script_path)], capture_output=True, text=True, check=True
        )
        result["job_id"] = proc.stdout.strip().split()[-1]
        result["submit_stdout"] = proc.stdout.strip()
    except FileNotFoundError:
        result["note"] = "sbatch not found on this host; script written but not submitted"
    except subprocess.CalledProcessError as e:
        result["note"] = f"sbatch failed: {(e.stderr or '').strip()}"
    return result


def build_search(spec: SearchSpec, output_root: Path):
    """Build the local sampler. The slurm backend is handled by :func:`submit_slurm`."""
    if spec.backend == "local":
        return _local_search(spec, output_root)
    raise ValueError(
        f"build_search only builds the 'local' backend; backend={spec.backend!r} "
        "is submitted via facade.submit_slurm"
    )
