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

    ``galaxy-subtract`` fits the lens-galaxy light on a masked Step-1 dataset
    (oversample=4, B7 convention), subtracts it on the full native grid, then
    reloads the subtracted image and rescales the noise by ``f = sqrt(Step-1
    reduced chi2)``.
    """
    import autolens as al
    from astropy.io import fits

    from ..lensfit.lens.rescale import noise_rescale_factor
    from ..lensfit.lens.subtract import subtract_stellar_light

    dataset = _from_fits(data)

    rescale_f = 1.0
    if subtract_stellar:
        ds1, mask1 = _apply_mask_oversample(
            dataset, data, noise_scale=1.0, oversample=_SUBTRACT_OVERSAMPLE
        )
        n_unmasked = int(np.sum(~np.array(mask1)))

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
