"""Weighted corner plot of a joint fit's posterior.

Reads autofit's ``samples.csv`` directly (weighted posterior), no dependence on
the autofit result object, so it runs on memory-constrained laptops (the B6
post-processing OOM lesson).

References
----------
- Barnabe et al. 2012, MNRAS 423, 1073 (the six shared mass parameters plotted)
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence, Tuple

import numpy as np
import pandas as pd

from ..result import _PARAM_COLUMNS, weighted_percentile
from ...lensfit.structures import StellarTemplate


def plot_corner(
    samples_csv,
    out_path,
    param_keys: Sequence[str] = ("ml_img", "v_vir", "c_minus2", "gamma", "q_h", "ratio"),
    param_labels: Sequence[str] = ("ml_img", "v_vir", "c_-2", "gamma", "q_h", "ratio"),
    title: str = "joint posterior (weighted)",
    dpi: int = 120,
) -> None:
    """Write a weighted corner plot of the six shared mass parameters.

    ``param_keys`` are the :data:`result._PARAM_COLUMNS` keys; the stellar M/L
    column is matched by pattern (the Gaussian index varies with the template),
    so this is robust to the stellar MGE's Gaussian count.
    """
    import re

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    df = pd.read_csv(samples_csv)
    df.columns = df.columns.str.strip()
    w = df["weight"].to_numpy(dtype=float)
    w = w / w.sum()  # non-in-place: to_numpy() may be read-only

    columns = []
    for key in param_keys:
        pattern = _PARAM_COLUMNS[key]
        matches = [c for c in df.columns if re.fullmatch(pattern, c)]
        if len(matches) != 1:
            raise KeyError(f"expected one column matching {pattern!r}, got {matches}")
        columns.append(matches[0])

    n = len(columns)
    fig, axes = plt.subplots(n, n, figsize=(3.2 * n, 3.2 * n))
    for i, (col_i, lab_i) in enumerate(zip(columns, param_labels)):
        xi = df[col_i].to_numpy(dtype=float)
        for j, (col_j, lab_j) in enumerate(zip(columns, param_labels)):
            ax = axes[i, j]
            if i == j:
                ax.hist(xi, bins=40, weights=w, histtype="step", color="k")
                lo, med, hi = weighted_percentile(xi, w, [16, 50, 84])
                ax.axvline(med, color="r", lw=1)
                ax.axvline(lo, color="r", lw=1, ls=":")
                ax.axvline(hi, color="r", lw=1, ls=":")
            elif j < i:
                xj = df[col_j].to_numpy(dtype=float)
                ax.hexbin(xj, xi, C=w, reduce_C_function=np.sum, gridsize=28, cmap="viridis")
            else:
                ax.axis("off")
            if i == n - 1:
                ax.set_xlabel(lab_j)
            else:
                ax.set_xticklabels([])
            if j == 0 and i != 0:
                ax.set_ylabel(lab_i)
            else:
                ax.set_yticklabels([])

    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
