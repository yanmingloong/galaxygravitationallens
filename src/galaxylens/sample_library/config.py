#!/usr/bin/env python3
"""
Disk lens sample library — unified config
===================================
Paths, schema constants, tier rules, and sorting rules.
All scripts import from this file to avoid hardcoding.
"""

import os

from galaxylens.config.paths import get_sample_library_root

# ================================================================
# Project root directory
# ================================================================
PROJECT_ROOT = str(get_sample_library_root())

# Output directories for each stage
DIR_RAW = os.path.join(PROJECT_ROOT, "01_raw")
DIR_MERGED = os.path.join(PROJECT_ROOT, "02_merged")
DIR_IDENTIFIED = os.path.join(PROJECT_ROOT, "03_identified")
DIR_ENRICHED = os.path.join(PROJECT_ROOT, "04_enriched")
DIR_EXPORTS = os.path.join(PROJECT_ROOT, "05_exports")

# Main catalog paths
MASTER_PATH = os.path.join(DIR_MERGED, "master.csv")
MASTER_CLASSIFIED_PATH = os.path.join(DIR_IDENTIFIED, "master_classified.csv")

# ================================================================
# Unified schema (48 fields, grouped by category)
# ================================================================

# Fields to export (ordered by modeling priority)
SCHEMA_COLUMNS = [
    # ── Modeling core parameters ★★★ ──
    "sample_id",
    "name",
    "ra_deg",
    "dec_deg",
    "disk_tier",
    "grade",
    "z_lens",
    "z_lens_err",
    "z_source",
    "z_source_err",
    "sigma_km_s",
    "sigma_err",
    "theta_E_arcsec",
    "theta_E_err",
    # ── Structural parameters ★★☆ ──
    "logMstar_total",
    "logMstar_total_err",
    "B_D_ratio",
    "logMstar_bulge",
    "logMstar_disk",
    "n_sersic",
    "n_sersic_err",
    "Re_arcsec",
    "Re_bulge_arcsec",
    "Re_disk_arcsec",
    "q_bulge",
    "q_disk",
    "n_bulge",
    "PA_disk_deg",
    # ── Photometry and spectroscopy ★☆☆ ──
    "mag_r",
    "mag_g",
    "mag_i",
    "g_r_color",
    "Halpha_flux",
    "spec_class",
    "spec_sn",
    # ── Metadata ──
    "source_catalog",
    "source_ref",
    "cross_refs",
    "lens_type_label",
    "z_lens_flag",
    "z_source_flag",
    "sigma_flag",
    "n_images",
    "imaging_available",
    "ifu_available",
    "IMF",
    "tier_basis",
    "notes",
]

# Modeling core parameter columns (first 14)
MODELING_CORE_COLS = SCHEMA_COLUMNS[:14]

# ================================================================
# Data source config (raw file paths produced in Phase A)
# ================================================================

# Point to the new 01_raw/ directory
RAW_DIR = os.path.join(PROJECT_ROOT, "01_raw")

RAW_SOURCES = {
    "SWELLS": {
        "description": "SWELLS edge-on disk lens survey (Auger+2011)",
        "files": {
            "data0": os.path.join(RAW_DIR, "swells/SWELLSlensdata0.xlsx"),
            "data1": os.path.join(RAW_DIR, "swells/SWELLSlensdata1.xlsx"),
        },
        "fmt": "excel",
        "lens_count": 27,
    },
    "lenscat": {
        "description": "lenscat.org aggregated catalog (30+ surveys)",
        "files": {
            "catalog": os.path.join(RAW_DIR, "lenscat/lenscat_catalog.csv"),
        },
        "fmt": "csv",
        "lens_count": 32838,
    },
    "GLUE": {
        "description": "GLUE UNIONS/Euclid lens candidates (Storfer+2025)",
        "files": {
            "candidates": os.path.join(RAW_DIR, "glue/storfer_2025_candidates.csv"),
        },
        "fmt": "csv",
        "lens_count": 5555,
    },
    "CASTLES": {
        "description": "CfA-Arizona Space Telescope Lens Survey",
        "files": {
            "lenses": os.path.join(RAW_DIR, "castles/castles_lenses.csv"),
        },
        "fmt": "csv",
        "lens_count": 100,
    },
    "SLACS": {
        "description": "Sloan Lens ACS Survey",
        "files": {
            "lenscat_slacs": os.path.join(RAW_DIR, "slacs/lenscat_slacs.csv"),
        },
        "fmt": "csv",
        "lens_count": 100,
    },
    "COWLS": {
        "description": "COSMOS-Web Lens Survey — JWST NIRCam strong lenses (Mercier+2025)",
        "files": {
            "catalogue": os.path.join(RAW_DIR, "cowls/cowls_catalogue.csv"),
        },
        "fmt": "csv",
        "lens_count": 440,
    },
    "MaNGA": {
        "description": "MaNGA SILO — IFU-detected strong lens candidates (Talbot+2022)",
        "files": {
            "candidates": os.path.join(RAW_DIR, "manga/manga_silo_candidates.csv"),
        },
        "fmt": "csv",
        "lens_count": 457,
    },
}

# ================================================================
# Tier classification rules
# ================================================================

# Tier 0: confirmed disk galaxies (paper-supported + B/D decomposition)
TIER0_NAMES = [
    # SWELLS A-grade (known disk lenses from Auger+2011)
    "SDSS J0820+4847", "SDSS J0903+5116", "SDSS J0904+5112",
    "SDSS J0912+4845", "SDSS J0922+4508", "SDSS J0955+4640",
    "SDSS J1029+0420", "SDSS J1103+5322", "SDSS J1142+5055",
    "SDSS J1205+4910", "SDSS J1250+0523", "SDSS J1301+5117",
    "SDSS J1430+4105", "SDSS J2141-0001",
    # SWELLS B-grade
    "SDSS J1110+4638", "SDSS J1114+4727", "SDSS J1148+4950",
    "SDSS J1338+4447", "SDSS J1436+4654", "SDSS J1632+4117",
    "SDSS J2111-0114", "SDSS J2300+0022", "SDSS J2341+0000",
]

# Tier X keywords (exclusions: ellipticals, galaxy clusters, quasars)
TIER_X_LENS_TYPES = {"DEV", "group", "cluster", "QSO", "elliptical", "deV"}
TIER_X_SPEC_CLASSES = {"QSO", "STAR"}

# ================================================================
# Deduplication parameters
# ================================================================
DEDUP_COORD_RADIUS_ARCSEC = 2.0   # coordinate matching radius

# ================================================================
# Sorting rules
# ================================================================
SORT_ORDER = [
    ("disk_tier",   True),    # Tier 0 first
    ("grade",       True),    # A > B > C > X
    ("sigma_km_s",  False),   # prioritize those with σ (NULL last)
    ("theta_E_arcsec", False),
]

# Grade sort values
GRADE_ORDER = {"A": 0, "B": 1, "C": 2, "X": 3}

# ================================================================
# Enrichment parameter config
# ================================================================

# Simard+2011 VizieR catalog
SIMARD_VIZIER_ID = "J/ApJS/196/11/table3"

# Auger+2009 SLACS sigma catalog
AUGER2009_VIZIER_ID = "J/ApJ/705/1099"

# SDSS DR17 SkyServer
SDSS_SKYSERVER_URL = "https://skyserver.sdss.org/dr17/SkyServerWS/SearchTools/SqlSearch"

# ================================================================
# Validation
# ================================================================
for d in [DIR_RAW, DIR_MERGED, DIR_IDENTIFIED, DIR_ENRICHED, DIR_EXPORTS]:
    os.makedirs(d, exist_ok=True)

print(f"Config loaded. Project root: {PROJECT_ROOT}")
print(f"Schema columns: {len(SCHEMA_COLUMNS)}")
print(f"Tier0 systems: {len(TIER0_NAMES)}")
