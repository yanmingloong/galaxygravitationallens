#!/usr/bin/env python3
"""
Disk lens sample library — shared utility functions
=====================================
Name normalization, coordinate matching, and deduplication logic.
"""

import re
import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord, match_coordinates_sky
from astropy import units as u

from .config import DEDUP_COORD_RADIUS_ARCSEC


# ================================================================
# Name normalization
# ================================================================

def normalize_name(name: str) -> str:
    """
    Normalize a lens name into a unified comparison form.

    Rules:
    - Convert to lowercase
    - Remove all spaces and underscores
    - Remove the "SDSS " / "SDSS" prefix (for comparison)
    - Remove +/- signs (J2141-0001 → j21410001)

    Examples:
        "SDSS J0820+4847" → "j08204847"
        "SDSSJ0820+4847"  → "j08204847"
        "J0820+4847"      → "j08204847"
    """
    name = str(name).lower().strip()
    # Remove spaces and underscores
    name = name.replace(" ", "").replace("_", "")
    # Remove SDSS prefix
    name = re.sub(r'^sdssj?', 'j', name)
    # Ensure it starts with j
    if not name.startswith('j'):
        name = 'j' + name
    # Remove +/- signs
    name = name.replace('+', '').replace('-', '')
    return name


def names_match(name1: str, name2: str) -> bool:
    """Return whether two names refer to the same system."""
    return normalize_name(name1) == normalize_name(name2)


# ================================================================
# Coordinate matching
# ================================================================

def coord_match(ra1, dec1, ra2, dec2) -> float:
    """
    Compute the angular separation (arcsec) between two sets of coordinates.

    Returns:
        float: angular separation (arcsec); returns inf if any input is NaN
    """
    if any(pd.isna(v) for v in [ra1, dec1, ra2, dec2]):
        return float('inf')
    c1 = SkyCoord(ra=ra1 * u.deg, dec=dec1 * u.deg)
    c2 = SkyCoord(ra=ra2 * u.deg, dec=dec2 * u.deg)
    return c1.separation(c2).arcsec


def find_matches(catalog_a: pd.DataFrame, catalog_b: pd.DataFrame,
                 radius_arcsec: float = DEDUP_COORD_RADIUS_ARCSEC) -> list:
    """
    Perform bulk coordinate cross-matching using astropy.

    Args:
        catalog_a: source catalog (to match), must contain ra_deg, dec_deg
        catalog_b: target catalog (existing), must contain ra_deg, dec_deg
        radius_arcsec: matching radius

    Returns:
        list of (idx_a, idx_b, sep_arcsec) matched pairs
    """
    if len(catalog_a) == 0 or len(catalog_b) == 0:
        return []

    coords_a = SkyCoord(ra=catalog_a["ra_deg"].values * u.deg,
                        dec=catalog_a["dec_deg"].values * u.deg)
    coords_b = SkyCoord(ra=catalog_b["ra_deg"].values * u.deg,
                        dec=catalog_b["dec_deg"].values * u.deg)

    idx_b, sep2d, _ = match_coordinates_sky(coords_a, coords_b)

    matches = []
    for i, (j, sep) in enumerate(zip(idx_b, sep2d)):
        if sep.arcsec < radius_arcsec:
            matches.append((i, j, sep.arcsec))
    return matches


# ================================================================
# Deduplication arbitration
# ================================================================

SOURCE_PRIORITY = {
    "SWELLS": 0,   # highest — has B/D decomposition
    "SLACS": 1,    # paper-verified
    "CASTLES": 2,
    "lenscat": 3,
    "GLUE": 4,
}
DEFAULT_PRIORITY = 99


def resolve_duplicate(existing: pd.Series, new: pd.Series) -> pd.Series:
    """
    Deduplication arbitration: decide which row to keep when two rows refer to the same system.

    Rules:
    1. By source_catalog priority
    2. Same source: by grade (A > B > C)
    3. Merge cross_refs (cross-source reference names)

    Returns:
        pd.Series: the best merged row
    """
    existing_pri = SOURCE_PRIORITY.get(str(existing.get("source_catalog", "")), DEFAULT_PRIORITY)
    new_pri = SOURCE_PRIORITY.get(str(new.get("source_catalog", "")), DEFAULT_PRIORITY)

    if existing_pri < new_pri:
        keep, other = existing.copy(), new
    elif new_pri < existing_pri:
        keep, other = new.copy(), existing
    else:
        # Same source: compare grade
        from .config import GRADE_ORDER
        eg = GRADE_ORDER.get(str(existing.get("grade", "")), 99)
        ng = GRADE_ORDER.get(str(new.get("grade", "")), 99)
        if eg <= ng:
            keep, other = existing.copy(), new
        else:
            keep, other = new.copy(), existing

    # Merge cross_refs
    refs = set()
    for r in [existing.get("name", ""), new.get("name", ""),
              existing.get("cross_refs", ""), new.get("cross_refs", "")]:
        if pd.notna(r) and str(r).strip():
            refs.add(str(r).strip())
    keep["cross_refs"] = "|".join(sorted(refs))

    return keep


# ================================================================
# Helper functions
# ================================================================

def safe_float(val, default=np.nan):
    """Safely convert to float."""
    if pd.isna(val):
        return default
    try:
        return float(val)
    except (ValueError, TypeError):
        return default


def flag_if_present(val, flag_true: str, flag_false: str = "") -> str:
    """Return flag_true if a value is present, otherwise flag_false."""
    return flag_true if pd.notna(val) and str(val).strip() not in ("", "-", "nan") else flag_false
