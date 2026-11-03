"""Dataset loading: FITS -> masked Imaging (+ galaxy-subtract + noise rescale).

One entry point serves real and mock data alike: the mock pipeline writes
``data/noise/psf`` FITS in the same convention, so this loader is shared (the
mock<->fit contract). Orchestrates the B4/B7 chain: ``from_fits`` -> starmask +
circular mask -> oversample -> (galaxy-subtract) -> (noise rescale).

References
----------
- Barnabe, Dutton, Marshall et al. 2012, MNRAS 423, 1073 (galaxy-subtracted +
  reduced-chi2 rescale chain)
"""

from __future__ import annotations

import csv
import math
import tempfile
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from .config_loader import DataSpec

_SUBTRACT_OVERSAMPLE = 4  # Step-1 (galaxy-subtract fit) oversampling, B7 convention


def resolve_path(path: str) -> Path:
    """Resolve a data path: absolute, or relative to ``GALAXYLENS_DATA_ROOT``."""
    p = Path(path)
    if p.is_absolute():
        return p
    from ..config.paths import get_data_root

    return get_data_root() / p


def _from_fits(data: DataSpec):
    import autolens as al

    return al.Imaging.from_fits(
        data_path=resolve_path(data.data_path),
        noise_map_path=resolve_path(data.noise_map_path),
        psf_path=resolve_path(data.psf_path),
        pixel_scales=data.pixel_scale,
    )


def _apply_mask_oversample(dataset, data: DataSpec, noise_scale: float = 1.0, oversample: Optional[int] = None):
    import autolens as al
    from astropy.io import fits

    if noise_scale != 1.0:
        dataset.noise_map = dataset.noise_map * noise_scale

    if data.starmask_path is not None:
        sm_path = resolve_path(data.starmask_path)
        sm = fits.getdata(sm_path).astype(bool) if sm_path.exists() else \
            np.zeros(dataset.data.shape_native, dtype=bool)
    else:
        sm = np.zeros(dataset.data.shape_native, dtype=bool)

    mc = al.Mask2D.circular(
        shape_native=dataset.data.shape_native,
        pixel_scales=dataset.data.pixel_scales,
        radius=data.mask_radius,
        centre=(0.0, 0.0),
    )
    combined = al.Mask2D(mask=sm | np.array(mc), pixel_scales=data.pixel_scale)

    sub = data.oversample if oversample is None else oversample
    over = al.util.over_sample.over_sample_size_via_radial_bins_from(
        grid=dataset.grid, sub_size_list=[sub], radial_list=[float("inf")]
    )
    dataset = dataset.apply_over_sampling(over_sample_size_lp=over).apply_mask(mask=combined)
    return dataset, combined


def _load_mge_csv(path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load an MGE ``(L_k, sigma_arcsec, q)`` CSV (B2 ``mge_params.csv``).

    Expected columns: ``<name>, L_k, sigma_px, sigma_arcsec, q`` (header row
    skipped). ``L_k`` is each Gaussian's TOTAL flux (mgefit ``sol[0]``), not the
    peak surface density.
    """
    Ls, sigs, qs = [], [], []
    with open(path) as f:
        reader = csv.reader(f)
        next(reader)  # header
        for row in reader:
            _, L, _sp_px, sp_arc, q = row
            Ls.append(float(L))
            sigs.append(float(sp_arc))
            qs.append(float(q))
    return np.array(Ls), np.array(sigs), np.array(qs)


def load_dataset(
    data: DataSpec,
    zlens: float,
    subtract_stellar: bool = True,
    noise_rescale: bool = True,
    out_dir: Optional[Path] = None,
) -> Tuple:
    """Full chain: the masked ``al.Imaging`` ready for the joint fit.

    Returns ``(dataset, n_unmasked)`` where ``dataset`` is galaxy-subtracted
    (optional) and noise-rescaled (optional), masked and oversampled.

    Two subtraction recipes (selected by ``DataSpec.subtract_method``):

    - ``"sersic"`` (B4, default): fit the lens-galaxy light on a masked Step-1
      dataset (oversample=4, B7 convention), subtract it on the full native
      grid, reload, and rescale the noise by ``f = sqrt(reduced chi2)``.
    - ``"mge_agn"`` (B9): sky-RMS noise (``estimate_sky_noise``, bypassing the
      KAI weight-map pitfall) + a literature MGE convolved with the PSF plus a
      PSF-shaped AGN point source (``subtract_stellar_light_mge_agn``), all on
      the native grid; the noise is the sky RMS scaled by ``f = sqrt(red)``.
    """
    import autolens as al
    from astropy.io import fits

    from ..lensfit.lens.rescale import noise_rescale_factor, estimate_sky_noise
    from ..lensfit.lens.subtract import (
        subtract_stellar_light,
        subtract_stellar_light_sersic,
        subtract_stellar_light_mge_agn,
    )

    rescale_f = 1.0
    is_mge_agn = subtract_stellar and getattr(data, "subtract_method", "sersic") == "mge_agn"
    if is_mge_agn:
        dataset = None  # built below from sci/badmask (sky-RMS noise, no separate FITS)
    else:
        dataset = _from_fits(data)

    if is_mge_agn:
        # ---- B9 path: sky RMS + MGE + AGN point source, native grid ----
        if not (data.mge_params_path and data.badmask_path and data.psf_sigma_px):
            raise ValueError(
                "subtract_method='mge_agn' requires mge_params_path, "
                "badmask_path and psf_sigma_px in the data: block"
            )
        sci = fits.getdata(resolve_path(data.data_path)).astype(float)
        badmask = fits.getdata(resolve_path(data.badmask_path)).astype(np.uint8)

        sky_noise = estimate_sky_noise(
            sci, badmask, data.mask_radius, data.pixel_scale
        )
        noise = np.full_like(sci, sky_noise)
        Ls, sigs, qs = _load_mge_csv(resolve_path(data.mge_params_path))

        subtracted, red = subtract_stellar_light_mge_agn(
            sci, noise, badmask, Ls, sigs, qs,
            psf_sigma_px=data.psf_sigma_px,
            mask_radius_arcsec=data.mask_radius,
            pixel_scale=data.pixel_scale,
            position_angle_deg=data.position_angle_deg,
        )
        # red is already the reduced chi2, so f = sqrt(red) directly.
        rescale_f = math.sqrt(red) if noise_rescale else 1.0
        noise2 = noise * rescale_f

        if out_dir is not None:
            sub_path = Path(out_dir) / "galaxy_subtracted.fits"
            noi_path = Path(out_dir) / "noise_rescaled.fits"
            sub_path.parent.mkdir(parents=True, exist_ok=True)
        else:
            tmp = Path(tempfile.mkdtemp())
            sub_path = tmp / "galaxy_subtracted.fits"
            noi_path = tmp / "noise_rescaled.fits"
        fits.writeto(sub_path, subtracted.astype(np.float32), overwrite=True)
        fits.writeto(noi_path, noise2.astype(np.float32), overwrite=True)

        dataset = al.Imaging.from_fits(
            data_path=sub_path,
            noise_map_path=noi_path,
            psf_path=resolve_path(data.psf_path),
            pixel_scales=data.pixel_scale,
        )
        rescale_f = 1.0  # noise already carries the rescale factor

    elif subtract_stellar:
        # ---- B4 path: Sersic bulge + disk subtraction (al.Tracer) ----
        ds1, mask1 = _apply_mask_oversample(
            dataset, data, noise_scale=1.0, oversample=_SUBTRACT_OVERSAMPLE
        )
        n_unmasked = int(np.sum(~np.array(mask1)))

        if getattr(data, "subtract_method", "sersic") == "sersic_single":
            if not (
                data.sersic_re_arcsec
                and data.sersic_n
                and data.sersic_q
                and data.sersic_pa_deg is not None
            ):
                raise ValueError(
                    "subtract_method='sersic_single' requires "
                    "sersic_re_arcsec/n/q/pa_deg in the data block"
                )
            light, chi2 = subtract_stellar_light_sersic(
                ds1, zlens,
                re_arcsec=data.sersic_re_arcsec,
                n=data.sersic_n,
                q=data.sersic_q,
                pa_deg=data.sersic_pa_deg,
            )
        else:
            light, chi2 = subtract_stellar_light(ds1, zlens)

        full_grid = al.Grid2D.uniform(
            shape_native=ds1.data.shape_native, pixel_scales=ds1.data.pixel_scales
        )
        light_img = np.asarray(al.Tracer(galaxies=[light]).image_2d_from(grid=full_grid).native)
        subtracted = np.asarray(ds1.data.native) - light_img

        if out_dir is not None:
            sub_path = Path(out_dir) / "galaxy_subtracted.fits"
            sub_path.parent.mkdir(parents=True, exist_ok=True)
        else:
            sub_path = Path(tempfile.mkstemp(suffix="_galaxy_subtracted.fits")[1])
        fits.writeto(sub_path, subtracted.astype(np.float32), overwrite=True)

        dataset = al.Imaging.from_fits(
            data_path=sub_path,
            noise_map_path=resolve_path(data.noise_map_path),
            psf_path=resolve_path(data.psf_path),
            pixel_scales=data.pixel_scale,
        )
        rescale_f = noise_rescale_factor(chi2, n_unmasked) if noise_rescale else 1.0

    ds2, mask2 = _apply_mask_oversample(dataset, data, noise_scale=rescale_f)
    n_unmasked = int(np.sum(~np.array(mask2)))
    return ds2, n_unmasked
