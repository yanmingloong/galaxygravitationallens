"""Weighted corner plot with truth overlay (mock closure).

Like :mod:`galaxylens.api.plotting.corner` but for the mock-closure six
parameters ``(lg_ml, v_vir, gamma, c_-2, q_h, beta_z)`` and with the particle
truth overlaid (red dashed = weighted median, black = truth, red star = truth
position in the 2-D panels). Takes arrays directly, so it is independent of
the autofit result object and of the samples.csv column names.
"""
from __future__ import annotations

import os

import numpy as np

from ...api.result import weighted_percentile

LABELS = [r"$\lg\,M/L$", r"$v_{\rm vir}$", r"$\gamma$", r"$c_{-2}$",
          r"$q_h$", r"$\beta_z$"]


def plot_corner(theta, w, truth, out_path, labels=None, title="", dpi=120):
    """Weighted corner of the six mass parameters with truth overlay.

    ``theta`` is ``(N, 6)`` in ``(lg_ml, v_vir, gamma, c_minus2, q_h, beta_z)``
    order; ``w`` the posterior weights; ``truth`` a length-6 sequence with
    ``None`` where the truth is unknown (that panel draws no truth markers).
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    theta = np.asarray(theta, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)
    labels = labels or LABELS
    ndim = theta.shape[1]
    fig, axes = plt.subplots(ndim, ndim, figsize=(2.4 * ndim, 2.4 * ndim))
    for i in range(ndim):
        for j in range(ndim):
            ax = axes[i][j]
            if j > i:
                ax.axis("off")
            elif i == j:
                ax.hist(theta[:, i], bins=35, weights=w, color="steelblue",
                        histtype="step", density=True, lw=1.5)
                med = float(weighted_percentile(theta[:, i], w, 50))
                ax.axvline(med, color="crimson", lw=1, ls="--")
                if truth[i] is not None:
                    ax.axvline(truth[i], color="black", lw=1.2)
                ax.set_yticks([])
            else:
                ax.hist2d(theta[:, j], theta[:, i], bins=40, weights=w,
                          cmap="Blues")
                if truth[j] is not None and truth[i] is not None:
                    ax.plot(truth[j], truth[i], "r*", ms=11, mec="white",
                            mew=1)
            if i == ndim - 1:
                ax.set_xlabel(labels[j])
            else:
                ax.set_xticks([])
            if j == 0 and i != 0:
                ax.set_ylabel(labels[i])
            else:
                ax.set_yticks([])
    if title:
        fig.suptitle(title, fontsize=14)
        fig.tight_layout(rect=[0, 0, 1, 0.98])
    else:
        fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    plt.close()
    return out_path
