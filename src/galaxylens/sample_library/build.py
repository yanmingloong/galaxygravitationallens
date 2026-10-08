#!/usr/bin/env python3
"""
Phase B — standardization & deduplication
=======================
Read the raw files produced by Phase A, map them to the unified schema, and deduplicate in three stages.
Output: 02_merged/master.csv
"""

import os
import sys
import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord, match_coordinates_sky
from astropy import units as u

# Add project scripts to path

from .config import (
    SCHEMA_COLUMNS, RAW_SOURCES, MASTER_PATH,
    GRADE_ORDER, DEDUP_COORD_RADIUS_ARCSEC,
)
from .utils import (
    normalize_name, coord_match, find_matches,
    resolve_duplicate, safe_float, flag_if_present,
)


# ================================================================
# Source Mappers
# ================================================================

def map_swells(data0_path: str, data1_path: str) -> pd.DataFrame:
    """SWELLS Excel → unified schema."""
    df0 = pd.read_excel(data0_path)
    df1 = pd.read_excel(data1_path)

    rows = []
    for i in range(len(df0)):
        row0 = df0.iloc[i]
        row = {col: np.nan for col in SCHEMA_COLUMNS}

        # ── Basic identifiers ──
        name = f"SDSS J{row0.iloc[0]}"
        row["name"] = name
        row["sample_id"] = f"SWELLS_{row0.iloc[0]}"
        row["ra_deg"] = safe_float(row0.iloc[1])
        row["dec_deg"] = safe_float(row0.iloc[2])

        # ── Redshift ──
        row["z_lens"] = safe_float(row0.iloc[3])
        row["z_source"] = safe_float(row0.iloc[4])
        row["z_lens_flag"] = "spec"
        row["z_source_flag"] = "spec"

        # ── Sigma ──
        row["sigma_km_s"] = safe_float(row0.iloc[6])
        row["sigma_err"] = safe_float(row0.iloc[7])
        row["sigma_flag"] = flag_if_present(row["sigma_km_s"], "sdss")

        # ── Grade ──
        row["grade"] = str(row0.iloc[5]).strip()

        # ── Axis ratio ──
        row["q_disk"] = safe_float(row0.iloc[10])

        # ── Hα ──
        row["Halpha_flux"] = safe_float(row0.iloc[8])

        # ── Metadata ──
        ref_map = {1: "2006ApJ...638..703B", 2: "2008ApJ...682..964B",
                   3: "2011MNRAS.411.2003A"}
        row["source_catalog"] = "SWELLS"
        row["source_ref"] = ref_map.get(int(row0.iloc[11]), "")
        row["lens_type_label"] = "S0"  # SWELLS lenses are all edge-on S0 galaxies
        row["n_images"] = 2
        row["imaging_available"] = "HST_ACS_WFPC2_NICMOS_WFC3"
        row["ifu_available"] = False
        row["IMF"] = "Chabrier"

        # ── Bulge/Disk decomposition (from data1) ──
        idx_bulge = 2 * i
        idx_disk = 2 * i + 1
        if idx_disk < len(df1):
            bulge = df1.iloc[idx_bulge]
            disk = df1.iloc[idx_disk]

            logM_b = safe_float(bulge.iloc[11])
            logM_d = safe_float(disk.iloc[11])
            row["logMstar_bulge"] = logM_b
            row["logMstar_disk"] = logM_d

            if not (np.isnan(logM_b) or np.isnan(logM_d)):
                row["B_D_ratio"] = round(10**logM_b / 10**logM_d, 3)
                row["logMstar_total"] = round(np.log10(10**logM_b + 10**logM_d), 3)

            row["Re_bulge_arcsec"] = safe_float(bulge.iloc[9])
            row["Re_disk_arcsec"] = safe_float(disk.iloc[9])
            row["q_bulge"] = safe_float(bulge.iloc[10])
            # q_disk already set from data0

        rows.append(row)

    df = pd.DataFrame(rows, columns=SCHEMA_COLUMNS)
    print(f"  SWELLS: mapped {len(df)} lenses")
    return df


def map_lenscat(catalog_path: str) -> pd.DataFrame:
    """lenscat.org CSV → unified schema."""
    raw = pd.read_csv(catalog_path)
    rows = []

    for _, r in raw.iterrows():
        row = {col: np.nan for col in SCHEMA_COLUMNS}

        name = str(r["name"])
        row["name"] = name
        row["sample_id"] = f"lenscat_{normalize_name(name)}"
        row["ra_deg"] = safe_float(r["RA [deg]"])
        row["dec_deg"] = safe_float(r["DEC [deg]"])
        row["z_lens"] = _parse_zlens(r.get("zlens", ""))
        row["z_lens_flag"] = _zlens_flag(r.get("zlens", ""))
        row["lens_type_label"] = str(r.get("type", "")).strip()
        row["grade"] = _map_lenscat_grade(str(r.get("grading", "")))
        row["source_catalog"] = "lenscat"
        row["source_ref"] = str(r.get("ref", ""))

        rows.append(row)

    df = pd.DataFrame(rows, columns=SCHEMA_COLUMNS)
    print(f"  lenscat: mapped {len(df)} lenses")
    return df


def map_glue(candidates_path: str) -> pd.DataFrame:
    """GLUE CSV → unified schema."""
    raw = pd.read_csv(candidates_path)
    rows = []

    for _, r in raw.iterrows():
        row = {col: np.nan for col in SCHEMA_COLUMNS}

        name = str(r["sys_name"])
        row["name"] = name
        row["sample_id"] = f"GLUE_{normalize_name(name)}"
        row["ra_deg"] = safe_float(r["ra"])
        row["dec_deg"] = safe_float(r["dec"])
        row["grade"] = str(r["grade"]).strip()
        row["mag_g"] = safe_float(r.get("mag_g"))
        row["mag_r"] = safe_float(r.get("mag_r"))
        row["mag_i"] = safe_float(r.get("mag_z"))

        # g-r color
        if pd.notna(row["mag_g"]) and pd.notna(row["mag_r"]):
            row["g_r_color"] = round(row["mag_g"] - row["mag_r"], 3)

        row["lens_type_label"] = str(r.get("type", "")).strip()

        # Redshift: spec-z takes priority over photoz
        z = safe_float(r.get("speczd_sdss"))
        z_flag = "spec"
        if np.isnan(z):
            z = safe_float(r.get("speczd_dr1"))
        if np.isnan(z):
            z = safe_float(r.get("photzd"))
            z_flag = "photoz"
        row["z_lens"] = z
        row["z_lens_flag"] = z_flag

        # Source redshift
        row["z_source"] = safe_float(r.get("speczs"))

        row["source_catalog"] = "GLUE"
        row["source_ref"] = "2025arXiv...Storfer"

        rows.append(row)

    df = pd.DataFrame(rows, columns=SCHEMA_COLUMNS)
    print(f"  GLUE: mapped {len(df)} lenses")
    return df


def map_castles(lenses_path: str) -> pd.DataFrame:
    """CASTLES CSV → unified schema."""
    raw = pd.read_csv(lenses_path)
    rows = []

    for _, r in raw.iterrows():
        row = {col: np.nan for col in SCHEMA_COLUMNS}

        name = str(r["lens_name"]).strip()
        row["name"] = name
        row["sample_id"] = f"CASTLES_{normalize_name(name)}"
        row["ra_deg"] = _parse_ra(r.get("ra", ""))
        row["dec_deg"] = _parse_dec(r.get("dec", ""))
        row["z_lens"] = safe_float(r.get("z_lens"))
        row["z_source"] = safe_float(r.get("z_source"))
        row["grade"] = str(r.get("grade", "")).strip()
        row["sigma_km_s"] = safe_float(r.get("sigma_kms"))
        row["sigma_flag"] = flag_if_present(row["sigma_km_s"], "literature")
        row["n_images"] = safe_float(r.get("n_images"))
        row["source_catalog"] = "CASTLES"
        row["source_ref"] = "cfa-castles"

        rows.append(row)

    df = pd.DataFrame(rows, columns=SCHEMA_COLUMNS)
    print(f"  CASTLES: mapped {len(df)} lenses")
    return df


def map_slacs(lenscat_slacs_path: str) -> pd.DataFrame:
    """SLACS (lenscat subset) → unified schema."""
    return map_lenscat(lenscat_slacs_path)


def map_cowls(catalogue_path: str) -> pd.DataFrame:
    """JWST COWLS → unified schema.
    
    COWLS columns: code, score, ra, dec, lens_cw_photo_z_med, lens_cw_stmass_med,
                   einstein_radius, F115W_magnification, ..., score
    """
    raw = pd.read_csv(catalogue_path, low_memory=False)
    rows = []

    for _, r in raw.iterrows():
        row = {col: np.nan for col in SCHEMA_COLUMNS}

        name = str(r.get("code", ""))
        row["name"] = name
        row["sample_id"] = f"COWLS_{normalize_name(name)}"
        row["ra_deg"] = safe_float(r.get("ra"))
        row["dec_deg"] = safe_float(r.get("dec"))

        # Redshift: photo-z from COSMOS-Web, spec-z if available
        z_photo = safe_float(r.get("lens_cw_photo_z_med"))
        z_spec = safe_float(r.get("lens_spec_z"))
        if not np.isnan(z_spec):
            row["z_lens"] = z_spec
            row["z_lens_flag"] = "spec"
        elif not np.isnan(z_photo) and z_photo > 0:
            row["z_lens"] = z_photo
            row["z_lens_flag"] = "photoz"
        else:
            row["z_lens_flag"] = "unknown"

        # Einstein radius (from PyAutoLens SIE fit)
        row["theta_E_arcsec"] = safe_float(r.get("einstein_radius"))
        if not np.isnan(row["theta_E_arcsec"]):
            row["theta_E_err"] = np.nan  # not provided

        # Stellar mass
        row["logMstar_total"] = safe_float(r.get("lens_cw_stmass_med"))
        if not np.isnan(row["logMstar_total"]) and row["logMstar_total"] < 0:
            row["logMstar_total"] = np.nan  # -99 sentinel

        # Grade from score
        score = str(r.get("score", "")).strip()
        if score == "M25":
            row["grade"] = "A"  # confirmed lens (COWLS II)
        elif score in ("S10", "S11", "S12", "P18"):
            row["grade"] = "A"
        elif score in ("S07", "S08", "S09"):
            row["grade"] = "B"
        elif score in ("S04", "S05", "S06"):
            row["grade"] = "C"
        else:
            row["grade"] = "X"

        row["source_catalog"] = "COWLS"
        row["source_ref"] = "Mercier+2025,Mahler+2025"
        row["imaging_available"] = "JWST_NIRCam_F115W_F150W_F277W_F444W"

        rows.append(row)

    df = pd.DataFrame(rows, columns=SCHEMA_COLUMNS)
    print(f"  COWLS: mapped {len(df)} lenses")
    return df


def map_manga(candidates_path: str) -> pd.DataFrame:
    """MaNGA SILO FITS → unified schema.
    
    Columns: BACKGROUND_ID, SDSS_TARGET_NAME, MANGA_ID, PLATE, IFU,
             NSA_ZF, MEAN_ZF_INNER, SIGMA_ZF_INNER, RA_IFU, DEC_IFU,
             MEAN_ZB, SIGMA_ZB, SOURCE_GRADE, SOURCE_COMMENT,
             NEAREST_RATIO, ER_FIREFLY, ER_JAM, DETECTION_COUNT
    """
    raw = pd.read_csv(candidates_path, low_memory=False)
    rows = []

    for _, r in raw.iterrows():
        row = {col: np.nan for col in SCHEMA_COLUMNS}

        # Name: prefer SDSS_TARGET_NAME, fallback to MANGA_ID
        name = str(r.get("SDSS_TARGET_NAME", ""))
        if not name or name == "nan":
            name = str(r.get("MANGA_ID", ""))
        row["name"] = name
        row["sample_id"] = f"MaNGA_{normalize_name(name)}"
        row["ra_deg"] = safe_float(r.get("RA_IFU"))
        row["dec_deg"] = safe_float(r.get("DEC_IFU"))

        # Redshift
        z_fore = safe_float(r.get("NSA_ZF"))
        z_back = safe_float(r.get("MEAN_ZB"))
        row["z_lens"] = z_fore
        row["z_lens_flag"] = "spec" if not np.isnan(z_fore) else "unknown"
        row["z_source"] = z_back

        # Velocity dispersion
        row["sigma_km_s"] = safe_float(r.get("SIGMA_ZF_INNER"))
        row["sigma_flag"] = flag_if_present(row["sigma_km_s"], "MaNGA_IFU")

        # Einstein radius (two methods available: FIREFLY stellar mass, JAM dynamical)
        er_firefly = safe_float(r.get("ER_FIREFLY"))
        er_jam = safe_float(r.get("ER_JAM"))
        if not np.isnan(er_jam):
            row["theta_E_arcsec"] = er_jam
        elif not np.isnan(er_firefly):
            row["theta_E_arcsec"] = er_firefly
        row["theta_E_err"] = np.nan

        # Grade
        grade_raw = str(r.get("SOURCE_GRADE", "")).strip()
        grade_map = {"Likely": "A", "Probable": "B", "Possible": "C", "False": "X"}
        row["grade"] = grade_map.get(grade_raw, "X")

        row["source_catalog"] = "MaNGA_SILO"
        row["source_ref"] = "Talbot+2022"
        row["imaging_available"] = "SDSS_imaging"
        row["ifu_available"] = True  # MaNGA IFU data available!

        # Detection count as quality indicator
        det = safe_float(r.get("DETECTION_COUNT"))
        if not np.isnan(det):
            row["notes"] = f"DETECTION_COUNT={int(det)}"

        rows.append(row)

    df = pd.DataFrame(rows, columns=SCHEMA_COLUMNS)
    print(f"  MaNGA: mapped {len(df)} lenses")
    return df


# ================================================================
# Helper functions
# ================================================================

def _parse_zlens(zval) -> float:
    """Parse the zlens field from lenscat (may contain multiple | -separated values)."""
    if pd.isna(zval) or str(zval).strip() in ("-", "nan", ""):
        return np.nan
    parts = str(zval).split("|")
    vals = []
    for p in parts:
        try:
            vals.append(float(p.strip()))
        except ValueError:
            pass
    return np.mean(vals) if vals else np.nan


def _zlens_flag(zval) -> str:
    """Determine the redshift source of the lenscat zlens value."""
    if pd.isna(zval) or str(zval).strip() in ("-", "nan", ""):
        return "unknown"
    parts = str(zval).split("|")
    vals = []
    for p in parts:
        try:
            vals.append(float(p.strip()))
        except ValueError:
            pass
    if len(vals) == 0:
        return "unknown"
    if len(vals) == 1:
        return "spec"
    # Multiple values: check consistency
    arr = np.array(vals)
    return "consistent" if arr.std() < 0.01 else "multi"


def _map_lenscat_grade(grading: str) -> str:
    """lenscat grading → A/B/C/X."""
    g = grading.lower().strip()
    if g in ("confident", "a"):
        return "A"
    if g in ("probable", "b"):
        return "B"
    if g in ("candidate", "c"):
        return "C"
    return "X"


def _parse_ra(ra_val) -> float:
    """Parse CASTLES right ascension (may be HH:MM:SS or degrees)."""
    if pd.isna(ra_val):
        return np.nan
    ra_str = str(ra_val).strip()
    # Try direct float first
    try:
        return float(ra_str)
    except ValueError:
        pass
    # Try HH:MM:SS
    try:
        from astropy.coordinates import Angle
        a = Angle(ra_str, unit='hourangle')
        return a.degree
    except Exception:
        return np.nan


def _parse_dec(dec_val) -> float:
    """Parse CASTLES declination."""
    if pd.isna(dec_val):
        return np.nan
    dec_str = str(dec_val).strip()
    try:
        return float(dec_str)
    except ValueError:
        pass
    try:
        from astropy.coordinates import Angle
        a = Angle(dec_str, unit='deg')
        return a.degree
    except Exception:
        return np.nan


# ================================================================
# Main flow: standardization + deduplication
# ================================================================

def build_master(verbose: bool = True) -> pd.DataFrame:
    """
    Read all available sources → map to the unified schema → three-stage deduplication → write master.csv.

    Returns:
        pd.DataFrame: the deduplicated unified catalog
    """
    all_entries = []

    # ── 1. Load each source ──
    sources = RAW_SOURCES

    if "SWELLS" in sources:
        try:
            df = map_swells(sources["SWELLS"]["files"]["data0"],
                           sources["SWELLS"]["files"]["data1"])
            all_entries.append(df)
        except Exception as e:
            print(f"  ✗ SWELLS failed: {e}")

    if "lenscat" in sources:
        try:
            df = map_lenscat(sources["lenscat"]["files"]["catalog"])
            all_entries.append(df)
        except Exception as e:
            print(f"  ✗ lenscat failed: {e}")

    if "GLUE" in sources:
        try:
            df = map_glue(sources["GLUE"]["files"]["candidates"])
            all_entries.append(df)
        except Exception as e:
            print(f"  ✗ GLUE failed: {e}")

    if "CASTLES" in sources:
        try:
            df = map_castles(sources["CASTLES"]["files"]["lenses"])
            all_entries.append(df)
        except Exception as e:
            print(f"  ✗ CASTLES failed: {e}")

    if "SLACS" in sources:
        try:
            df = map_slacs(sources["SLACS"]["files"]["lenscat_slacs"])
            all_entries.append(df)
        except Exception as e:
            print(f"  ✗ SLACS failed: {e}")

    if "COWLS" in sources:
        try:
            df = map_cowls(sources["COWLS"]["files"]["catalogue"])
            all_entries.append(df)
        except Exception as e:
            print(f"  ✗ COWLS failed: {e}")

    if "MaNGA" in sources:
        try:
            df = map_manga(sources["MaNGA"]["files"]["candidates"])
            all_entries.append(df)
        except Exception as e:
            print(f"  ✗ MaNGA failed: {e}")

    if not all_entries:
        print("ERROR: No sources loaded!")
        return pd.DataFrame(columns=SCHEMA_COLUMNS)

    master = pd.concat(all_entries, ignore_index=True)
    print(f"\n{'='*60}")
    print(f"Total entries (before dedup): {len(master)}")

    # ── 2. Deduplication ──
    master = deduplicate(master, verbose=verbose)
    print(f"Total systems (after dedup):  {len(master)}")

    # ── 3. Save ──
    os.makedirs(os.path.dirname(MASTER_PATH), exist_ok=True)
    master.to_csv(MASTER_PATH, index=False)
    print(f"\nSaved: {MASTER_PATH}")

    return master


def deduplicate(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Three-stage deduplication: coordinate matching → name validation → arbitration merge.

    Strategy: group by source priority, and use astropy batch matching to avoid O(N²).
    Higher-priority sources enter the deduped set first; later sources are batch-checked
    with match_coordinates_sky (coordinates only), then name validation is applied
    to the candidate matched pairs.

    Returns:
        pd.DataFrame: deduplicated catalog
    """
    if len(df) == 0:
        return df

    # Sort by source priority
    source_rank = {s: i for i, s in enumerate(
        ["SWELLS", "MaNGA_SILO", "COWLS", "SLACS", "CASTLES", "lenscat", "GLUE"]
    )}
    df["_src_rank"] = df["source_catalog"].map(source_rank).fillna(99)
    df = df.sort_values("_src_rank").reset_index(drop=True)

    # Process by group: higher-priority sources added in batches
    deduped_rows = []
    deduped_coords = None  # SkyCoord array
    duplicates_found = 0
    total_checked = 0

    for src_name, group in df.groupby("source_catalog", sort=False):
        # Filter invalid coordinates
        valid = group["ra_deg"].notna() & group["dec_deg"].notna()
        valid_group = group[valid].copy()
        invalid_group = group[~valid]

        # Add entries with invalid coordinates directly
        for _, row in invalid_group.iterrows():
            deduped_rows.append(row)

        if len(valid_group) == 0:
            continue

        # Coordinates of the new source
        new_coords = SkyCoord(
            ra=valid_group["ra_deg"].values * u.deg,
            dec=valid_group["dec_deg"].values * u.deg,
        )

        if deduped_coords is not None and len(deduped_coords) > 0:
            # Batch coordinate match: new source vs existing
            idx_deduped, sep2d, _ = match_coordinates_sky(
                new_coords, deduped_coords
            )

            for i_local, (j_dedup, sep) in enumerate(zip(idx_deduped, sep2d)):
                total_checked += 1
                if sep.arcsec < DEDUP_COORD_RADIUS_ARCSEC:
                    # Level 2: name validation
                    existing_name = str(deduped_rows[j_dedup]["name"])
                    new_name = str(valid_group.iloc[i_local]["name"])
                    if _names_compatible(existing_name, new_name):
                        # Level 3: arbitration merge
                        merged = resolve_duplicate(
                            deduped_rows[j_dedup],
                            valid_group.iloc[i_local],
                        )
                        deduped_rows[j_dedup] = merged
                        duplicates_found += 1
                        if verbose:
                            print(f"  DEDUP: {existing_name[:40]:40s} ≈ {new_name[:40]:40s} (sep={sep.arcsec:.2f}\")")
                    else:
                        # Coordinates close but names do not match → keep as new entry
                        deduped_rows.append(valid_group.iloc[i_local])
                else:
                    # No match → new entry
                    deduped_rows.append(valid_group.iloc[i_local])

            # Rebuild deduped_coords (new entries were added)
            deduped_coords = SkyCoord(
                ra=[r["ra_deg"] for r in deduped_rows if pd.notna(r["ra_deg"])],
                dec=[r["dec_deg"] for r in deduped_rows if pd.notna(r["dec_deg"])],
                unit=u.deg,
            )
        else:
            # First batch of sources: add all
            for _, row in valid_group.iterrows():
                deduped_rows.append(row)
            deduped_coords = SkyCoord(
                ra=[r["ra_deg"] for r in deduped_rows],
                dec=[r["dec_deg"] for r in deduped_rows],
                unit=u.deg,
            )

    result = pd.DataFrame(deduped_rows)
    result = result.drop(columns=["_src_rank"], errors="ignore")

    if verbose:
        print(f"  Dedup: {duplicates_found} duplicates merged out of {total_checked} checks")

    return result


def _names_compatible(name1: str, name2: str, threshold: float = 0.6) -> bool:
    """
    Compatibility check between two source names.

    Strategy:
    1. Exact match → True
    2. Match after normalization → True
    3. One contains the other → True
    4. Fuzzy match (allows missing prefix) → True if sufficiently similar
    """
    n1 = normalize_name(name1)
    n2 = normalize_name(name2)

    if n1 == n2:
        return True
    if n1 in n2 or n2 in n1:
        return True

    # Simple Jaccard similarity (character-bigram based)
    def bigrams(s):
        return {s[i:i+2] for i in range(len(s)-1)}
    b1, b2 = bigrams(n1), bigrams(n2)
    if not b1 or not b2:
        return False
    jaccard = len(b1 & b2) / len(b1 | b2)
    return jaccard >= threshold


# ================================================================
# CLI
# ================================================================

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Phase B: Build master catalog")
    parser.add_argument("--quiet", action="store_true", help="Suppress dedup details")
    args = parser.parse_args()

    print("Phase B — Build Master Catalog")
    print("=" * 60)
    master = build_master(verbose=not args.quiet)

    # Quick summary
    print(f"\nSource distribution:")
    print(master["source_catalog"].value_counts().to_string())
    print(f"\nGrade distribution:")
    print(master["grade"].value_counts().to_string())
