"""Processing parameter defaults (global constants)."""
from __future__ import annotations

from typing import Dict

# Cropping and masking
CUTOUT_SIZE_ARCSEC: float = 15.0       # Cutout field of view (arcsec)
MASK_RADIUS_FACTOR: float = 3.0        # Star mask radius = N × PSF_FWHM

# PSF construction
PSF_STAMP_SIZE: int = 31               # PSF stamp size (odd number of pixels)
MIN_STAR_SNR: float = 8.0              # Star detection SNR threshold
STAR_SHARPLO: float = 0.7              # Lower bound for field star sharpness
N_STARS_PSF: int = 5                   # Number of stars used for the PSF

# PSF comparison/selection thresholds (EE50 normal ≈ 1.0-1.5px, bloating > 3px flagged as contamination; ellipticity > 0.15 flagged as malformed)
PSF_EE50_MIN_PX: float = 0.8
PSF_EE50_MAX_PX: float = 3.0
PSF_ELL_MAX: float = 0.15

# Instrument pixel scale (arcsec/pix)
INSTRUMENT_PIXEL_SCALE: Dict[str, float] = {
    "ACS/WFC":   0.050,
    "WFPC2/PC":  0.0455,
    "WFC3/IR":   0.128,
}
