"""Result object: physical quantities read directly off a fit's posterior.

Wraps autofit's ``samples.csv`` (the weighted posterior) plus the stellar
template, exposing ``beta_z`` / ``c_minus2`` / ``gamma`` / ``v_vir`` / ``q_h`` /
``M*`` / ``f_dm`` as weighted medians with 16/84 percentiles -- the
"parameter -> physical quantity" conversions the B series worked out, so users
never dig into instance fields or re-derive the CALIB bridge or the weighted
percentiles by hand.

Reads ``samples.csv`` (a few MB) rather than the full autofit result object, so
it also works on memory-constrained laptops (the B6 post-processing OOM lesson).

References
----------
- Barnabe, Dutton, Marshall et al. 2012, MNRAS 423, 1073 (beta_z = 1 - ratio^2,
  M* = ml_img x CALIB x L2D_REL)
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from ..lensfit.structures import Geometry, StellarTemplate
from . import derived


def weighted_percentile(x, weights, q) -> np.ndarray:
    """Weighted percentile (``q`` in [0, 100]); matches autofit's weighted median."""
    x = np.asarray(x, dtype=float)
    weights = np.asarray(weights, dtype=float)
    order = np.argsort(x)
    xs = x[order]
    ws = np.cumsum(weights[order])
    ws /= ws[-1]
    return np.interp(np.asarray(q) / 100.0, ws, xs)


# autofit samples.csv column names. The stellar M/L column carries a Gaussian
# index that varies with the template's Gaussian count, so it is matched by
# pattern rather than hard-coded (B7 writes ``mge_13``, B4 ``mge_0``, ...).
_PARAM_COLUMNS = {
    "ml_img": r"galaxies\.lens\.mge_\d+\.mass_to_light_ratio",
    "v_vir": "galaxies.lens.dark.v_vir",
    "c_minus2": "galaxies.lens.dark.c_2",
    "gamma": "galaxies.lens.dark.inner_slope",
    "q_h": "galaxies.lens.dark.q_h",
    "ratio": "galaxies.lens.anisotropy.mass_to_light_ratio",
}


@dataclass
class Post:
    """One weighted posterior quantity: median + 16/84 percentiles."""

    median: float
    lower: float  # median - 16th percentile
    upper: float  # 84th percentile - median

    def __repr__(self) -> str:
        return f"{self.median:.4g} [-{self.lower:.3g} +{self.upper:.3g}]"


class Result:
    """Weighted posterior summary of a joint fit."""

    def __init__(self, samples_csv, template: StellarTemplate, geom: Geometry):
        self.samples_csv = Path(samples_csv)
        self.template = template
        self.geom = geom
        self.df = pd.read_csv(self.samples_csv)
        self.df.columns = self.df.columns.str.strip()
        w = self.df["weight"].to_numpy(dtype=float)
        self.weights = w / w.sum()  # non-in-place: to_numpy() may be read-only

    def _column(self, key: str) -> str:
        pattern = _PARAM_COLUMNS[key]
        matches = [c for c in self.df.columns if re.fullmatch(pattern, c)]
        if len(matches) != 1:
            raise KeyError(f"expected one column matching {pattern!r}, got {matches}")
        return matches[0]

    def _values(self, key: str) -> np.ndarray:
        return self.df[self._column(key)].to_numpy(dtype=float)

    def percentile(self, key: str, q=(16, 50, 84)) -> np.ndarray:
        return weighted_percentile(self._values(key), self.weights, q)

    def _post(self, key: str) -> Post:
        lo, med, hi = self.percentile(key, [16, 50, 84])
        return Post(float(med), float(med - lo), float(hi - med))

    # --- per-parameter posteriors ---
    @property
    def ml_img(self) -> Post:
        return self._post("ml_img")

    @property
    def v_vir(self) -> Post:
        return self._post("v_vir")

    @property
    def c_minus2(self) -> Post:
        return self._post("c_minus2")

    @property
    def gamma(self) -> Post:
        return self._post("gamma")

    @property
    def q_h(self) -> Post:
        return self._post("q_h")

    @property
    def ratio(self) -> Post:
        return self._post("ratio")

    # --- derived physical quantities (computed per sample, then weighted) ---
    @property
    def beta_z(self) -> np.ndarray:
        """Weighted beta_z = 1 - ratio^2 (median + 16/84)."""
        r = self._values("ratio")
        return weighted_percentile(1.0 - r ** 2, self.weights, [16, 50, 84])

    @property
    def c200(self) -> np.ndarray:
        """Weighted c_200 = (2 - gamma) * c_-2 (median + 16/84)."""
        gamma = self._values("gamma")
        c2 = self._values("c_minus2")
        return weighted_percentile((2.0 - gamma) * c2, self.weights, [16, 50, 84])

    @property
    def m_star(self) -> np.ndarray:
        """Weighted total stellar mass M* = ml_img x CALIB x L2D_REL (median + 16/84)."""
        ml = self._values("ml_img")
        return weighted_percentile(
            derived.m_star_total(ml, self.template), self.weights, [16, 50, 84]
        )

    def f_dm(self, ref_arcsec: float) -> np.ndarray:
        """Weighted f_dm within a circular aperture (median + 16/84).

        Loops over samples; each call runs two ``jam.mge.radial_mass`` integrals,
        so this is O(N_samples) -- the same cost as ``b6_fdm.py``.
        """
        ml = self._values("ml_img")
        vv = self._values("v_vir")
        gm = self._values("gamma")
        c2 = self._values("c_minus2")
        qh = self._values("q_h")
        fdm = np.array(
            [
                derived.f_dm_of(a, b, cc, d, e, ref_arcsec, self.template, self.geom)[0]
                for a, b, cc, d, e in zip(ml, vv, gm, c2, qh)
            ]
        )
        return weighted_percentile(fdm, self.weights, [16, 50, 84])
