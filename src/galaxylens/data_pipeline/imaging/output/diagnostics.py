"""
Diagnostic visualization
==========
Generate diagnostic plots of the processing results, including a 4-panel overview and a PSF profile.

Features:
  1. generate_diagnostic_plot() — 4-panel: SCI/Noise/SNR/StarMask
  2. generate_psf_profile()     — PSF profile plot

Output format: PNG (matplotlib Agg backend)

Example:
  >>> from galaxylens.data_pipeline.imaging.output.diagnostics import generate_diagnostic_plot
  >>> generate_diagnostic_plot(output_dir, lens_id, band, sci, noise, psf, starmask)
"""

from __future__ import annotations
from pathlib import Path
from typing import Optional

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ══════════════════════════════════════════════════════════════════════
# Main diagnostic plot
# ══════════════════════════════════════════════════════════════════════

def generate_diagnostic_plot(
    output_dir: Path,
    lens_id: str,
    band: str,
    sci: np.ndarray,
    noise: np.ndarray,
    psf: np.ndarray,
    starmask: np.ndarray,
    bg_rms: float = 0.0,
    psf_source: str = "",
    cr_mask: Optional[np.ndarray] = None,
    snr: Optional[np.ndarray] = None,
) -> Optional[Path]:
    """Generate a 4-panel or 5-panel diagnostic plot.

    Panel layout:
      1. SCI (grayscale, log-stretch like)
      2. Noise map (inferno colormap)
      3. SNR map (SCI/noise, with threshold overlay)
      4. Star mask (red overlay) or PSF (hot colormap)

    Args:
        output_dir: Output directory
        lens_id: Lens identifier
        band: Band name
        sci: Science image
        noise: Noise map
        psf: PSF array
        starmask: Star mask
        bg_rms: Background RMS (used for display range)
        psf_source: PSF source label
        cr_mask: Cosmic-ray mask (optional)
        snr: SNR map (optional; auto-computed)

    Returns:
        PNG file path, or None (on failure)
    """
    try:
        from .fits_writer import make_prefix
        prefix = make_prefix(lens_id, band)

        if bg_rms <= 0:
            bg_rms = np.std(sci[starmask == 0]) if starmask.sum() > 0 else np.std(sci)

        if snr is None:
            snr = np.where(noise > 0, np.abs(sci) / noise, 0)

        n_panels = 4 if cr_mask is None else 5
        fig, axes = plt.subplots(1, n_panels, figsize=(4 * n_panels, 4))

        # Panel 1: SCI
        ax = axes[0] if n_panels > 1 else axes
        im = ax.imshow(
            sci, origin="lower", cmap="gray",
            vmin=-bg_rms, vmax=bg_rms * 50,
        )
        ax.set_title(f"{lens_id} {band} — SCI", fontsize=10)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        # Panel 2: Noise
        ax = axes[1]
        im = ax.imshow(noise, origin="lower", cmap="inferno")
        ax.set_title(f"Noise σ (med={np.median(noise):.4f})", fontsize=10)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        # Panel 3: SNR
        ax = axes[2]
        snr_clip = np.clip(snr, 0, np.percentile(snr[snr > 0], 95) if snr.max() > 0 else 1)
        im = ax.imshow(snr_clip, origin="lower", cmap="viridis")
        ax.set_title(f"SNR (peak={snr.max():.0f})", fontsize=10)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        # Panel 4: Star Mask
        ax = axes[3]
        ax.imshow(sci, origin="lower", cmap="gray",
                  vmin=-bg_rms, vmax=bg_rms * 20)
        if starmask.sum() > 0:
            mask_overlay = np.ma.masked_where(~starmask, starmask.astype(float))
            ax.imshow(mask_overlay, origin="lower", cmap="Reds", alpha=0.6)
        ax.set_title(f"Star Mask ({starmask.sum()} px)", fontsize=10)

        # Panel 5: CR mask (optional)
        if cr_mask is not None and n_panels >= 5:
            ax = axes[4]
            ax.imshow(sci, origin="lower", cmap="gray",
                      vmin=-bg_rms, vmax=bg_rms * 20)
            if cr_mask.sum() > 0:
                cr_overlay = np.ma.masked_where(~cr_mask, cr_mask.astype(float))
                ax.imshow(cr_overlay, origin="lower", cmap="Blues", alpha=0.6)
            ax.set_title(f"CR ({cr_mask.sum()} px)", fontsize=10)

        title = f"{lens_id} {band} — Preprocessing Diagnostic"
        if psf_source:
            title += f" [PSF: {psf_source}]"
        fig.suptitle(title, fontsize=12, fontweight="bold")
        fig.tight_layout()

        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / f"{prefix}_diagnostic.png"
        fig.savefig(path, dpi=120, bbox_inches="tight")
        plt.close(fig)

        return path

    except Exception as e:
        print(f"  Diagnostic plot SKIP: {e}", flush=True)
        return None


def generate_psf_profile(
    output_dir: Path,
    lens_id: str,
    band: str,
    psf: np.ndarray,
    pixel_scale: float = 0.05,
    psf_source: str = "",
) -> Optional[Path]:
    """Generate the PSF profile plot (central row/column + 2D contours).

    Args:
        output_dir: Output directory
        lens_id: Lens identifier
        band: Band name
        psf: PSF array
        pixel_scale: Pixel scale (arcsec/pixel)
        psf_source: PSF source label

    Returns:
        PNG file path
    """
    try:
        from .fits_writer import make_prefix
        prefix = make_prefix(lens_id, band)

        fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

        # Profile
        ax = axes[0]
        cy, cx = psf.shape[0] // 2, psf.shape[1] // 2
        x_as = (np.arange(psf.shape[1]) - cx) * pixel_scale
        y_as = (np.arange(psf.shape[0]) - cy) * pixel_scale
        ax.plot(x_as, psf[cy, :], "b-", label="X profile", lw=1.5)
        ax.plot(y_as, psf[:, cx], "r--", label="Y profile", lw=1.5)
        ax.set_xlabel("Offset (arcsec)")
        ax.set_ylabel("Normalized intensity")
        ax.legend(fontsize=9)
        ax.set_title(f"PSF Profile ({psf_source})", fontsize=10)

        # 2D contour
        ax = axes[1]
        yy, xx = np.indices(psf.shape)
        x_as_2d = (xx - cx) * pixel_scale
        y_as_2d = (yy - cy) * pixel_scale
        levels = [0.5, 0.1, 0.01, 0.001]
        ax.contour(
            x_as_2d, y_as_2d, psf / psf.max(),
            levels=levels, colors="k", linewidths=0.8,
        )
        ax.set_xlabel("Δx (arcsec)")
        ax.set_ylabel("Δy (arcsec)")
        ax.set_aspect("equal")
        ax.set_title("PSF Contours (50/10/1/0.1%)", fontsize=10)

        fig.suptitle(f"{lens_id} {band} PSF", fontsize=12, fontweight="bold")
        fig.tight_layout()

        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / f"{prefix}_psf_profile.png"
        fig.savefig(path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        return path

    except Exception as e:
        print(f"  PSF profile SKIP: {e}", flush=True)
        return None
