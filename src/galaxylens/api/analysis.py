"""One-call analysis of a finished fit: derived quantities + prior-edge check + corner.

The "analyze" stage of the pipeline, but also callable on its own (manual mode).
Pulls the weighted posterior off ``samples.csv`` via :class:`~galaxylens.api.result.Result`,
prints the six mass parameters + derived quantities (M*, f_dm, theta_E, gamma_tot),
checks whether the posterior hits the prior edges, and writes the corner plot.

References
----------
- Barnabe et al. 2012, MNRAS 423, 1073 (M*, f_dm, beta_z)
- Tan et al. 2024 (Dinos) benchmark: theta_E, power-law slope gamma_tot
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from ..lensfit.structures import PriorSpec
from .result import Result
from .plotting.corner import plot_corner


def _prior_bounds(prior) -> tuple:
    """Return ``(lo, hi)`` for a prior field (tuple of two, or a PriorSpec)."""
    if isinstance(prior, PriorSpec):
        return prior.lower, prior.upper
    if isinstance(prior, (tuple, list)) and len(prior) == 2:
        return float(prior[0]), float(prior[1])
    raise TypeError(f"cannot read bounds from prior {prior!r}")


def einstein_radius_median(result: Result, ref_guess: float = 1.5) -> Dict[str, float]:
    """theta_E and gamma_tot at the weighted-median parameters (Dinos cross-check).

    theta_E solves M_2D(<R) = pi * Sigma_crit * R^2; gamma_tot is the local
    power-law slope 3 - d ln M_2D / d ln R evaluated at theta_E.
    """
    import jampy as jam
    from scipy.optimize import brentq

    from ..lensfit.models.dm.gnfw_virial import gnfw_halo_mge, halo_cosmo

    geom = result.geom
    tmpl = result.template
    ml = result.ml_img.median
    vv = result.v_vir.median
    gm = result.gamma.median
    c2 = result.c_minus2.median
    qh = result.q_h.median

    c = halo_cosmo(geom.zlens, geom.zsrc)
    sigma_crit_as2 = c.sigma_crit * c.kpc_per_arcsec**2
    surf_rel, sig, q = tmpl.mge
    surf_s = surf_rel * tmpl.calibration.calib * ml
    surf_dm_sph, sig_dm, _, _, _ = gnfw_halo_mge(vv, gm, c2, qh, c)
    si = np.sin(np.radians(geom.inclination_deg))
    co = np.cos(np.radians(geom.inclination_deg))
    q_obs = np.sqrt(qh**2 * si**2 + co**2)
    surf_dm = surf_dm_sph / q_obs

    surf_pot = np.concatenate([surf_s, surf_dm])
    sig_pot = np.concatenate([sig, sig_dm])
    q_pot = np.concatenate([q, np.full_like(surf_dm, q_obs)])

    def m2d(r):
        return jam.mge.cylindrical_mass(surf_pot, sig_pot, q_pot, r, distance=geom.da_mpc)

    f = lambda r: m2d(r) - np.pi * sigma_crit_as2 * r**2
    theta_e = brentq(f, 0.3, 3.0)

    eps = 0.05
    slope = 3.0 - np.log(m2d(theta_e * (1 + eps)) / m2d(theta_e * (1 - eps))) / np.log(
        (1 + eps) / (1 - eps)
    )
    return {"theta_E_arcsec": theta_e, "gamma_tot": slope}


def analyze(
    cfg,
    samples_csv: Optional[str] = None,
    out_dir: Optional[str] = None,
    corner: bool = True,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Analyze a finished fit: posterior summary + derived + edge check + corner.

    Args:
        cfg: the :class:`FitConfig` (supplies template / geometry / priors / name).
        samples_csv: path to ``samples.csv``; default = auto-locate under
            ``GALAXYLENS_OUTPUT_ROOT/<name>/...`` (same rule as ``gl.fit``).
        out_dir: corner output directory; default = the fit's own image dir.
        corner: whether to write the weighted corner plot.
        verbose: print the summary tables.

    Returns a dict with ``params`` (median + 16/84 per mass parameter),
    ``derived`` (beta_z / M* / f_dm / theta_E / gamma_tot), and ``edges``
    (per-parameter prior-edge status).
    """
    from .facade import _find_samples_csv, _output_root

    if samples_csv is None:
        samples_csv = str(_find_samples_csv(_output_root(cfg.search), cfg.search.name))

    result = Result(samples_csv, cfg.pipeline.stellar_template, cfg.pipeline.geometry)

    params = {
        k: {
            "median": float(getattr(result, k).median),
            "lo16": float(getattr(result, k).median - getattr(result, k).lower),
            "hi84": float(getattr(result, k).upper),
        }
        for k in ("ml_img", "v_vir", "c_minus2", "gamma", "q_h", "ratio")
    }

    fdm_ref = 2.2 * (cfg.pipeline.geometry.rd_arcsec or cfg.pipeline.geometry.re_arcsec)
    # f_dm at the median parameters (one call, fast). The O(N) posterior loop
    # (Result.f_dm) re-fits the halo MGE per sample and is minutes for thousands
    # of samples -- call it explicitly only when the full f_dm posterior is wanted.
    from .derived import f_dm_of

    fdm_pt, _, _ = f_dm_of(
        result.ml_img.median,
        result.v_vir.median,
        result.gamma.median,
        result.c_minus2.median,
        result.q_h.median,
        fdm_ref,
        result.template,
        result.geom,
    )
    derived: Dict[str, Any] = {
        "beta_z": [float(x) for x in result.beta_z],
        "M_star_Msun": [float(x) for x in result.m_star],
        f"f_dm({fdm_ref:.2f}\")": fdm_pt,
    }
    try:
        derived.update(einstein_radius_median(result))
    except Exception as e:  # noqa: BLE001 -- theta_E is a cross-check, not required
        derived["theta_E_error"] = str(e)

    # prior-edge check: is the 2.5/97.5% tail within 1% of the prior bound?
    edges = {}
    for key, prior in [
        ("ml_img", cfg.pipeline.priors.ml_img),
        ("v_vir", cfg.pipeline.priors.v_vir),
        ("gamma", cfg.pipeline.priors.gamma),
        ("q_h", cfg.pipeline.priors.q_h),
        ("ratio", cfg.pipeline.priors.ratio),
        ("c_minus2", cfg.pipeline.priors.c_2),
    ]:
        lo, hi = _prior_bounds(prior)
        q025, q975 = result.percentile(key, [2.5, 97.5])
        span = hi - lo
        edges[key] = {
            "prior": [lo, hi],
            "q025": float(q025),
            "q975": float(q975),
            "hits_lower": bool((q025 - lo) / span < 0.01),
            "hits_upper": bool((hi - q975) / span < 0.01),
        }

    if verbose:
        print("=== posterior (weighted median, 16/84) ===")
        for k, v in params.items():
            print(f"  {k:10s} = {v['median']:.4g}  [-{v['lo16']:.3g} +{v['hi84']:.3g}]")
        print("=== derived ===")
        for k, v in derived.items():
            if isinstance(v, list) and len(v) == 3:
                print(f"  {k:20s} = {v[1]:.4g}  [-{v[1]-v[0]:.3g} +{v[2]-v[1]:.3g}]")
            else:
                print(f"  {k:20s} = {v}")
        print("=== prior-edge check (2.5/97.5% within 1% of bound?) ===")
        for k, e in edges.items():
            flag = ""
            if e["hits_lower"]:
                flag += " hits_lower"
            if e["hits_upper"]:
                flag += " hits_upper"
            print(f"  {k:10s} {e['prior']}  ->  [{e['q025']:.3g}, {e['q975']:.3g}]{flag or '  OK'}")

    if corner:
        if out_dir is None:
            out_dir = str(
                Path(samples_csv).resolve().parents[2] / "image"
            )
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        corner_path = str(Path(out_dir) / "corner.png")
        plot_corner(
            samples_csv,
            corner_path,
            title=f"{cfg.search.name} joint posterior (weighted)",
        )

    return {"params": params, "derived": derived, "edges": edges}
