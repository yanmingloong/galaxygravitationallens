"""Unit tests for the PSF-stage fixes (2026-08-17).

Covers:
  1. psf_ee50 discrete half-light radius (narrow Gaussian < broad, correct order)
  2. filter_field_stars isolation filtering (KD-tree)
  3. filter_field_stars edge exclusion

Run: pytest tests/test_psf_selection.py
"""
import numpy as np
import pytest

pytest.importorskip("photutils", reason="photutils (PSF-stage optional dep) not installed")

from galaxylens.data_pipeline.imaging.stages.psf import psf_ee50, psf_ellipticity, filter_field_stars


def test_psf_ee50_ordering():
    """Narrow Gaussian EE50 < broad Gaussian EE50 (correct relative order, used
    for the selection threshold).

    Note: psf_ee50 is the discrete half-light radius (pixel-centre-distance
    ordering), slightly smaller than the continuous analytic value, but
    consistent across PSFs so relative comparisons and thresholds are
    unaffected.
    """
    yy, xx = np.mgrid[-15:16, -15:16]
    narrow = np.exp(-(xx**2 + yy**2) / (2 * (2.5 / 2.355)**2))  # FWHM=2.5px
    broad = np.exp(-(xx**2 + yy**2) / (2 * (5.0 / 2.355)**2))   # FWHM=5.0px
    narrow /= narrow.sum()
    broad /= broad.sum()
    ee_n, fw_n = psf_ee50(narrow)
    ee_b, fw_b = psf_ee50(broad)
    assert ee_n < ee_b, f"narrow EE50={ee_n:.2f} should be < broad {ee_b:.2f}"
    assert 0.5 < ee_n < 2.0, f"narrow EE50={ee_n:.2f} is anomalous"
    print(f"  [OK] psf_ee50: narrow={ee_n:.2f}px < broad={ee_b:.2f}px (order correct)")


def test_filter_isolation():
    """Two crowded stars (2px apart) should be rejected; one isolated star kept."""
    stars = [(100, 100, 50.0), (102, 100, 40.0), (200, 200, 30.0)]
    sel = filter_field_stars(stars, center=(300, 300),
                             exclusion_radius_pix=10, data_shape=(300, 300),
                             min_dist=5.0)
    assert len(sel) == 1 and sel[0][2] == 30.0, f"isolation filter wrong: {sel}"
    print("  [OK] isolation: 3 stars (2 crowded) -> 1 kept")


def test_filter_edge():
    """Edge source (y=5 < edge_margin=20) should be excluded."""
    sel = filter_field_stars([(150, 5, 99.0)], center=(300, 300),
                             exclusion_radius_pix=10, data_shape=(300, 300),
                             edge_margin=20.0)
    assert len(sel) == 0, f"edge exclusion wrong: {sel}"
    print("  [OK] edge exclusion: y=5 star rejected")


def test_ellipticity_circular():
    """Circular Gaussian ellipticity should be ~0."""
    yy, xx = np.mgrid[-10:11, -10:11]
    g = np.exp(-(xx**2 + yy**2) / (2 * 1.0**2))
    g /= g.sum()
    e = psf_ellipticity(g)
    assert e < 0.02, f"circular Gaussian ellipticity={e:.4f} should be ~0"
    print(f"  [OK] ellipticity: circular Gaussian e={e:.4f}")


if __name__ == "__main__":
    test_psf_ee50_ordering()
    test_filter_isolation()
    test_filter_edge()
    test_ellipticity_circular()
    print("\nAll 4 unit tests passed.")
