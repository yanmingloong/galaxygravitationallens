"""Regression: mock light-MGE front end (mock_pipeline.lensing.mge).

Covers the pure-function surface of the light-MGE component without needing a
real mock image or mgefit: geometry (two-pass 2nd moment), the SKIRT
parameters.xml pixel scale, and the PSF 2nd-moment sigma. The full
``fit_light_mge`` (mgefit fit) is validated against the MF 348489 MGE on real
data elsewhere; here we anchor the helpers it builds on.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from galaxylens.mock_pipeline.lensing.mge import (
    measure_geometry,
    read_pixel_scale_kpc,
    read_psf_sigma_px,
)


def _gaussian_img(nx=100, ny=80, xc=55.0, yc=40.0, sig_x=10.0, sig_y=5.0):
    yy, xx = np.mgrid[0:ny, 0:nx]
    return np.exp(-((xx - xc) ** 2 / (2 * sig_x ** 2) + (yy - yc) ** 2 / (2 * sig_y ** 2)))


def test_measure_geometry_round():
    # Round Gaussian -> centre recovered, eps ~ 0
    img = _gaussian_img(sig_x=10.0, sig_y=10.0)
    col_c, row_c, eps, theta = measure_geometry(img, rmax_px=30.0)
    assert abs(col_c - 55.0) < 0.5, col_c
    assert abs(row_c - 40.0) < 0.5, row_c
    assert eps < 0.05, eps
    print(f"[mge] round: centre=({col_c:.1f},{row_c:.1f}) eps={eps:.3f}")


def test_measure_geometry_elongated():
    # Elongated Gaussian (sig_x > sig_y) -> eps > 0, theta ~ 90 deg (long axis x)
    img = _gaussian_img(sig_x=15.0, sig_y=4.0)
    _, _, eps, theta = measure_geometry(img, rmax_px=40.0)
    assert eps > 0.3, eps
    # long axis along x -> theta near 0 or 180 (arctan2 of evec); accept either
    assert abs(abs(theta) - 90.0) < 20.0 or abs(abs(theta)) < 20.0, theta
    print(f"[mge] elongated: eps={eps:.3f} theta={theta:.1f} deg")


def test_read_pixel_scale_kpc(tmp_path=Path("/tmp")):
    xml = tmp_path / "test_parameters.xml"
    xml.write_text(
        '<?xml version="1.0"?>\n'
        '<parameters fieldOfViewX="40000 pc" fieldOfViewY="40000 pc" '
        'numPixelsX="200" numPixelsY="200"/>'
    )
    px = read_pixel_scale_kpc(str(xml))
    assert abs(px - 0.2) < 1e-9, px   # 40000 pc / 200 = 200 pc = 0.2 kpc
    print(f"[mge] pixel scale: {px:.3f} kpc/px")


def test_read_psf_sigma_px(tmp_path=Path("/tmp")):
    from astropy.io import fits

    psf_path = tmp_path / "test_psf.fits"
    ny = nx = 31
    yy, xx = np.mgrid[0:ny, 0:nx]
    cy = cx = (nx - 1) / 2.0
    sig = 2.0
    psf = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sig ** 2))
    fits.writeto(psf_path, psf, overwrite=True)

    s = read_psf_sigma_px(str(psf_path))
    assert abs(s - sig) < 0.05, s
    print(f"[mge] psf sigma: {s:.3f} px (input {sig})")


if __name__ == "__main__":
    test_measure_geometry_round()
    test_measure_geometry_elongated()
    test_read_pixel_scale_kpc()
    test_read_psf_sigma_px()
    print("ALL verify_light_mge checks passed")
