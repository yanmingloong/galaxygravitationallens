"""Mock-closure entry: fit samples -> f_dm -> truth bias.

The single entry point that ties a joint fit to the particle truth: build the
stellar + DM MGE from the fitted parameters (the same ``CALIB`` bridge and
gNFW parametrization as ``api.derived``), compute ``f_dm`` under the spherical
and cylindrical apertures (:mod:`.fdm`), and report the weighted-median bias
against :mod:`galaxylens.mock_pipeline.truth.fdm`.
"""
from __future__ import annotations

import numpy as np

from .fdm import spherical_f_dm, cylindrical_f_dm
from .bias import bias
from ...api.result import weighted_percentile


def sample_f_dm(lg_ml, v_vir, gamma, c_minus2, q_h, template, geom, apertures):
    """``f_dm`` (spherical + cylindrical) at each aperture for ONE sample.

    ``template`` is a :class:`~galaxylens.lensfit.structures.StellarTemplate`
    (relative MGE + calibration); ``apertures`` maps ``name -> R_as``.

    Calibration convention (the bridge ``surf_lum = surf_rel * calib * 10**lg_ml``
    is convention-agnostic — the CALLER must supply a consistent pair):

    * F814W / MF-mock: ``calib = 1`` and ``lg_ml = log10(ml_img * ml_to_dyn)``,
      i.e. the Sigma_crit bridge is baked into ``lg_ml``.
    * K'-band (B9 ``stellar_mge_kband``): ``calib = calibration.calib`` (the
      dynamics surface-density bridge, Msun/pc^2 per ml_img) and
      ``lg_ml = log10(ml_img)`` (pure M/L).

    Never mix them (``lg_ml`` with the bridge AND a non-unit ``calib`` would
    double-count the bridge). The imaging-only ``calibration_img`` is irrelevant
    here: f_dm uses the dynamics bridge ``calibration.calib``.
    """
    from galaxylens.lensfit.models.dm.gnfw_virial import (
        gnfw_halo_mge, halo_cosmo,
    )

    cosmo = halo_cosmo(geom.zlens, geom.zsrc)
    surf_rel, sig_s, q_s = template.mge
    calib = template.calibration.calib
    surf_lum = surf_rel * calib * (10.0 ** lg_ml)

    surf_dm_sph, sig_dm, _rs, _dc, _rv = gnfw_halo_mge(
        v_vir, gamma, c_minus2, q_h, cosmo)
    si = np.sin(np.radians(geom.inclination_deg))
    co = np.cos(np.radians(geom.inclination_deg))
    q_obs_h = np.sqrt(q_h ** 2 * si ** 2 + co ** 2)
    surf_dm = surf_dm_sph / q_obs_h

    out = {}
    for name, R_as in apertures.items():
        out[name] = {
            "f_dm_spherical": spherical_f_dm(
                surf_lum, sig_s, q_s, surf_dm, sig_dm, q_obs_h, R_as, geom),
            "f_dm_cyl_edgeon": cylindrical_f_dm(
                surf_lum, sig_s, q_s, surf_dm, sig_dm, q_obs_h, R_as, geom),
        }
    return out


def closure_f_dm(theta, w, template, geom, truth, apertures):
    """Weighted-median ``f_dm`` bias against truth, per aperture / convention.

    ``theta`` is ``(N, 6) = (lg_ml, v_vir, gamma, c_minus2, q_h, beta_z)``;
    ``w`` the posterior weights; ``truth`` the particle truth dict (the shape
    of ``mock_pipeline.truth.fdm.f_dm_apertures``); ``apertures`` maps
    ``name -> R_as``. Returns ``{name: {...}}`` with medians, 16/84 percentiles
    and the percent bias under both conventions.
    """
    theta = np.asarray(theta, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)
    n = theta.shape[0]

    # precompute the per-sample f_dm in vector-of-dicts form
    fs = {"spherical": {k: np.empty(n) for k in apertures},
          "cyl": {k: np.empty(n) for k in apertures}}
    for i in range(n):
        fd = sample_f_dm(theta[i, 0], theta[i, 1], theta[i, 2], theta[i, 3],
                         theta[i, 4], template, geom, apertures)
        for k in apertures:
            fs["spherical"][k][i] = fd[k]["f_dm_spherical"]
            fs["cyl"][k][i] = fd[k]["f_dm_cyl_edgeon"]

    out = {}
    for k in apertures:
        entry = {}
        for conv, key in (("spherical", "f_dm_spherical"),
                          ("cyl", "f_dm_cyl_edgeon")):
            vals = fs["spherical" if conv == "spherical" else "cyl"][k]
            lo, med, hi = weighted_percentile(vals, w, [16, 50, 84])
            entry[conv] = [float(lo), float(med), float(hi)]
            entry[conv + "_bias_pct"] = float(bias(med, truth[k][key]))
        out[k] = entry
    return out
