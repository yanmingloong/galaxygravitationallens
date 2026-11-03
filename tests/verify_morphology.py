"""Regression: image morphology (gl.identify_from_image).

Verifies the image-shape fallback distinguishes disk/elliptical from the FITS
alone, independent of the catalog. Calibration anchors (measured 2026-11-03):
    J2141 (edge-on S0 disk)  C = 1.58  -> disk
    J1630 (SLACS elliptical) C = 4.21  -> elliptical
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import galaxylens as gl
from galaxylens.sample_library.morphology import classify_morphology, measure_morphology

_BASE = "/mnt/d/HermesAgent/ResearchProjects/GalaxyGravLens/DataPipeline/archive/psf_comparisons"
J2141_F814 = f"{_BASE}/J2141-0001/v2_psf_epsf/j2141-0001_f814w_sci.fits"
J1630_F814 = f"{_BASE}/J1630+4520/v2_psf_epsf/j1630+4520_f814w_sci.fits"


def test_morphology_disk():
    m = measure_morphology(J2141_F814)
    assert m["C"] < 3.1, f"J2141 C={m['C']:.2f} 应判盘"
    lt, basis = classify_morphology(m)
    assert lt == "disk", basis
    print(f"[morph] J2141 C={m['C']:.2f} q={m['q']:.2f} -> disk ({basis})")


def test_morphology_elliptical():
    m = measure_morphology(J1630_F814)
    assert m["C"] > 3.4, f"J1630 C={m['C']:.2f} 应判椭圆"
    lt, basis = classify_morphology(m)
    assert lt == "elliptical", basis
    print(f"[morph] J1630 C={m['C']:.2f} q={m['q']:.2f} -> elliptical ({basis})")


def test_identify_from_image():
    p = gl.identify_from_image(J1630_F814, name="J1630+4520")
    assert p.lens_type == "elliptical"
    assert p.template_yaml == "configs/fit/j1630_ellip.yaml"

    p2 = gl.identify_from_image(J2141_F814, name="J2141-0001")
    assert p2.lens_type == "disk"
    assert p2.template_yaml == "configs/fit/j2141_barnabe.yaml"
    print("[morph] identify_from_image -> 正确 LensPlan（椭圆/盘各自模板）")


if __name__ == "__main__":
    test_morphology_disk()
    test_morphology_elliptical()
    test_identify_from_image()
    print("ALL verify_morphology checks passed")
