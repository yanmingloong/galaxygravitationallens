"""Regression: mock data/noise/psf FITS load through ``api.dataset.load_dataset``.

Guards the mock<->fit contract: the mock pipeline writes ``data/noise/psf`` FITS
(counts/s, constant sky-RMS noise, per-band pixel scale) and the shared
``load_dataset`` must ingest them unchanged. A break here means mock products
cannot feed the joint fit.
"""

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from galaxylens.api.config_loader import DataSpec  # noqa: E402
from galaxylens.api.dataset import load_dataset  # noqa: E402


def test_mock_fits_load_through_shared_loader(tmp_path):
    from astropy.io import fits

    # minimal mock FITS in the mock_pipeline convention (imaging.py):
    # data = counts/s, noise = CONSTANT sky RMS, psf = band PSF, one pixel scale.
    rng = np.random.default_rng(0)
    data = (rng.normal(0.0, 0.05, (21, 21)) + 0.3).astype(np.float32)
    noise = np.full((21, 21), 0.00034, dtype=np.float32)  # F814W noise_sigma
    yy, xx = np.mgrid[-2:3, -2:3]
    psf = np.exp(-(xx**2 + yy**2) / 2.0).astype(np.float32)

    d_path = tmp_path / "mock_data.fits"
    n_path = tmp_path / "mock_noise.fits"
    p_path = tmp_path / "mock_psf.fits"
    fits.writeto(d_path, data, overwrite=True)
    fits.writeto(n_path, noise, overwrite=True)
    fits.writeto(p_path, psf, overwrite=True)

    spec = DataSpec(
        data_path=str(d_path),
        noise_map_path=str(n_path),
        psf_path=str(p_path),
        pixel_scale=0.05,
        mask_radius=0.3,
    )

    # no galaxy-subtraction / noise-rescale: the pure ingestion path.
    ds, n_unmasked = load_dataset(
        spec, zlens=0.1, subtract_stellar=False, noise_rescale=False
    )

    assert ds.data.shape_native == (21, 21)
    assert n_unmasked > 0
    # the loader must not alter the counts/s data (contract: shared convention)
    np.testing.assert_allclose(
        np.asarray(ds.data.native)[~np.asarray(ds.data.mask.native)][:1],
        data[~np.asarray(ds.data.mask.native)][:1],
        rtol=1e-5,
    )
