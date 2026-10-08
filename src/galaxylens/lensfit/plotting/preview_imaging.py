"""PyAutoLens-style imaging data preview (optional, requires autolens).
PyAutoLens-style imaging data preview (optional, requires autolens).

Plot the preprocessed sci/noise/psf FITS as a PyAutoLens triple preview
for quick data-quality checks.

This module depends on autolens, so it lives under lensfit/plotting/ rather
than data_pipeline/ (data_pipeline only depends on astropy/photutils, keeping
dependencies clean).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np


def plot_imaging_preview(
    sci_path: Path,
    noise_path: Path,
    psf_path: Path,
    pixel_scale: float,
    starmask: Optional[np.ndarray] = None,
    output_path: Optional[Path] = None,
    cutout_size_arcsec: float = 15.0,
) -> None:
    """Plot a sci/noise/psf triple preview with PyAutoLens.

    Args:
        sci_path: SCI FITS path
        noise_path: noise map FITS path
        psf_path: PSF FITS path
        pixel_scale: pixel scale (arcsec/pix)
        starmask: stellar mask array (optional, combined onto the circular mask)
        output_path: output PNG path (defaults to same directory and stem as sci_path)
        cutout_size_arcsec: cutout field-of-view diameter (arcsec)
    """
    import autolens as al

    # Construct the PyAutoLens Imaging dataset
    dataset = al.Imaging.from_fits(
        data_path=sci_path,
        noise_map_path=noise_path,
        psf_path=psf_path,
        pixel_scales=pixel_scale,
    )

    # Combine circular mask + stellar mask
    shape_native = starmask.shape if starmask is not None else dataset.shape_native
    circ_mask = al.Mask2D.circular(
        shape_native=shape_native,
        pixel_scales=pixel_scale,
        radius=cutout_size_arcsec / 2 * 0.98,
    )
    circ_native = np.array(circ_mask.native, dtype=bool)
    if starmask is not None:
        circ_native = circ_native | starmask
    combined_mask = al.Mask2D(mask=circ_native, pixel_scales=pixel_scale)
    dataset = dataset.apply_mask(mask=combined_mask)

    # Output path
    if output_path is None:
        output_path = sci_path.with_suffix("")
    output_dir = output_path.parent
    output_stem = output_path.name

    # Plot the subplot (sci / noise / psf / mask)
    from autoarray.plot.mat_plot.output import Output
    from autoarray.plot.mat_plot.two_d import MatPlot2D

    mat_plot = MatPlot2D(
        output=Output(path=str(output_dir), filename=output_stem, format="png")
    )
    plotter = al.ImagingPlotter(dataset=dataset, mat_plot_2d=mat_plot)
    plotter.subplot_dataset()
