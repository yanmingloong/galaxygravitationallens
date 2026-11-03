"""Bias between fitted and truth quantities for mock closure.

The headline output of a mock-closure run: for each aperture and convention,
how far the fitted ``f_dm`` (or other quantity) is from the particle truth.
"""
from __future__ import annotations


def bias(fit, truth):
    """Relative bias in percent: ``(fit - truth) / truth * 100``."""
    return (fit - truth) / truth * 100.0


def f_dm_bias_table(fit, truth):
    """Per-aperture / per-convention ``f_dm`` bias.

    ``fit`` and ``truth`` are ``{aperture: {"f_dm_spherical", "f_dm_cyl_edgeon"}}``
    dicts (the truth shape of ``mock_pipeline.truth.fdm.f_dm_apertures``).
    Returns ``{aperture: {"spherical_pct", "cyl_edgeon_pct"}}``.
    """
    out = {}
    for name in fit:
        out[name] = {
            "spherical_pct": bias(
                fit[name]["f_dm_spherical"], truth[name]["f_dm_spherical"]
            ),
            "cyl_edgeon_pct": bias(
                fit[name]["f_dm_cyl_edgeon"], truth[name]["f_dm_cyl_edgeon"]
            ),
        }
    return out
