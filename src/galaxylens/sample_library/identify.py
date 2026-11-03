"""Lens identification: catalog row -> lens type -> modeling plan -> recipe.

The identification layer (step 4): ``classify.py`` only does "disk / non-disk"
screening (ellipticals are pushed into Tier X); this module adds the forward
identification -- inferring "lens type (disk/elliptical) x lens constraint
(arc/quad)" from a catalog row, mapping it to a modeling-module combination +
recipe template. Output :class:`LensPlan`, then call :func:`write_recipe` to
emit a YAML for the existing ``gl.fit``, wiring "identify -> model -> fit".

Identification inputs (by priority):
    1. ``lens_type_label`` -- the survey's morphology label (DEV=elliptical,
       SER/EXP/S0=disk)
    2. ``n_sersic``        -- Sersic index assist (n<2 disk, n>=3 elliptical;
       mostly empty in the catalog)
    3. ``B_D_ratio``       -- disk if a bulge/disk decomposition exists
    4. ``n_images`` / known-quad list -- quads (point-level constraint; use the
       known-name list when n_images is missing)

Mapping for the three known samples (regression anchors):
    J2141-0001  disk  x imaging   -> j2141_barnabe.yaml (bulge+disk + jampy_kdc + arc)
    J1331+3628  disk  x positions -> j1331_trick.yaml   (point-level quad fit_positions)
    J1630+4520  ellip x imaging   -> j1630_ellip.yaml   (single Sersic + jampy + sigma_ap)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# Inline normalize_name / safe_float (do NOT import .utils: utils -> config
# triggers the GALAXYLENS_SAMPLE_LIB env check + directory creation; the
# identification layer only reads catalog CSVs and should stay light).
def _normalize_name(name: str) -> str:
    name = str(name).lower().strip()
    name = name.replace(" ", "").replace("_", "")
    name = re.sub(r"^sdssj?", "j", name)
    if not name.startswith("j"):
        name = "j" + name
    return name.replace("+", "").replace("-", "")


def _safe_float(val, default=np.nan):
    if pd.isna(val):
        return default
    try:
        return float(val)
    except (ValueError, TypeError):
        return default

# Elliptical / disk morphology labels (matching classify.py's TIER_X_LENS_TYPES
# and the disk tiers).
ELLIPTICAL_LABELS = {"DEV", "DEVAUCOULEURS", "ELLIPTICAL", "DEV2"}
DISK_LABELS = {"SER", "SERSIC", "EXP", "EXPONENTIAL", "DISKY", "S0"}

# Known-quad name list (n_images is mostly empty in the catalog; use the known
# list, same approach as TIER0_NAMES).
QUAD_NAMES = ["J1331+3628"]

# In-repo product CSVs (README declares this as the modeling_input product
# location). Read full (covers all tiers, incl. quads/special samples) + the
# elliptical modeling table (enrich_elliptical output, with DEV labels +
# sigma/thetaE/Re modeling params, so the identification layer can say
# "elliptical"). elliptical.csv (614-DEV exclusion list) and sample/tier01 are
# subsets of full, no need to re-read them.
_CATALOG_FILES = [
    "modeling_input_full.csv",
    "modeling_input_elliptical_modeling.csv",
    "modeling_input_elliptical.csv",
]


@dataclass
class LensPlan:
    """Lens identification result: type + modeling-module selection + recipe template.

    ``stellar_template`` / ``subtract_method`` / ``dynamics`` / ``source_model``
    are ``None`` when that combination does not need the module (e.g. a
    point-level quad does no stellar subtraction).
    """

    name: str
    lens_type: str                  # disk / elliptical
    constraint: str                 # imaging / positions / mein
    stellar_template: Optional[str]  # disk / ellip
    subtract_method: Optional[str]   # sersic / sersic_single
    dynamics: Optional[str]          # jampy_kdc / jampy
    source_model: Optional[str]      # sersic / pixelized
    template_yaml: str               # recipe template (relative to configs/fit/ or absolute)
    basis: str                       # identification basis (human-readable)

    def asdict(self) -> dict:
        return asdict(self)


def _is_nan(x) -> bool:
    return x is None or (isinstance(x, float) and np.isnan(x))


def identify_type(row: pd.Series) -> tuple[str, str]:
    """Infer the lens type ``(lens_type, basis)`` from a catalog row.

    ``lens_type`` is the morphology class: ``disk`` / ``elliptical`` (quads are
    a constraint, decided separately in :func:`identify_constraint`, not here).
    """
    lt = str(row.get("lens_type_label", "") or "").strip().upper()
    n_sersic = _safe_float(row.get("n_sersic"))
    b_d = _safe_float(row.get("B_D_ratio"))

    if lt in ELLIPTICAL_LABELS:
        return "elliptical", f"lens_type={lt}"

    if lt in DISK_LABELS:
        return "disk", f"lens_type={lt}"

    # Sersic n assist (n<2 disk, n>=3 elliptical)
    if not _is_nan(n_sersic):
        if n_sersic < 2.0:
            return "disk", f"n_sersic={n_sersic:.1f}_disk_like"
        if n_sersic >= 3.0:
            return "elliptical", f"n_sersic={n_sersic:.1f}_bulge_like"

    # Disk if a bulge/disk decomposition exists
    if not _is_nan(b_d):
        return "disk", f"B/D={b_d:.2f}"

    # Default disk (the sample library is a disk-screening product; type=None
    # cannot tell the morphology, so state it honestly)
    return "disk", "no_morphology_label_default_disk"


def identify_constraint(row: pd.Series, lens_type: str) -> tuple[str, str]:
    """Infer the lens constraint ``(constraint, basis)``: imaging / positions / mein.

    Quads (n_images>=4) go point-level ``positions``; extended arcs go
    ``imaging``. Fall back to the known-quad list when n_images is missing.
    """
    name = str(row.get("name", "") or "")
    nname = _normalize_name(name)

    if any(_normalize_name(qn) in nname for qn in QUAD_NAMES):
        return "positions", "known_quad"

    n_images = _safe_float(row.get("n_images"))
    if not _is_nan(n_images) and n_images >= 4:
        return "positions", f"n_images={int(n_images)}"

    return "imaging", "extended_arc"


def plan_modeling(name: str, lens_type: str, constraint: str, basis: str) -> LensPlan:
    """Type x constraint -> modeling-module combination + recipe template."""
    if constraint == "positions":
        # Point-level quad: no stellar subtraction/dynamics, image positions only
        # (the Trick+2016 J1331 route)
        return LensPlan(
            name, lens_type, "positions", None, None, None, None,
            "configs/fit/j1331_trick.yaml", f"point_level_quad:{basis}",
        )

    if lens_type == "elliptical":
        # Elliptical lens: single-Sersic subtraction + isotropic jampy + sigma_ap
        # (the J1630 route)
        return LensPlan(
            name, "elliptical", "imaging", "ellip", "sersic_single",
            "jampy", "sersic", "configs/fit/j1630_ellip.yaml",
            f"elliptical_sigma_ap_arc:{basis}",
        )

    # Disk lens: bulge+disk subtraction + jampy_kdc rotation + v_rms profile +
    # full arc (the Barnabe route)
    return LensPlan(
        name, "disk", "imaging", "disk", "sersic",
        "jampy_kdc", "sersic", "configs/fit/j2141_barnabe.yaml",
        f"disk_bulge_disk_arc:{basis}",
    )


def load_catalog(catalog_dir=None) -> pd.DataFrame:
    """Read the modeling_input product CSVs and merge/dedupe them.

    ``catalog_dir`` defaults to the in-repo ``sample_library/`` (in git, always
    available); alternatively pass the authoritative source dir
    ``GALAXYLENS_SAMPLE_LIB/05_exports``.
    """
    if catalog_dir is None:
        catalog_dir = Path(__file__).resolve().parents[3] / "sample_library"
    catalog_dir = Path(catalog_dir)

    frames = []
    for fn in _CATALOG_FILES:
        fp = catalog_dir / fn
        if fp.exists():
            frames.append(pd.read_csv(fp))

    if not frames:
        raise FileNotFoundError(
            f"no modeling_input CSVs under {catalog_dir}; expected "
            f"{_CATALOG_FILES}"
        )
    merged = pd.concat(frames, ignore_index=True)
    merged = merged.drop_duplicates(subset=["name"], keep="first")
    # Precompute the normalized name (vectorized once, reused by _lookup)
    merged["_norm"] = merged["name"].astype(str).map(_normalize_name)
    return merged


def _lookup(catalog: pd.DataFrame, name: str) -> Optional[pd.Series]:
    """Find a row by name (normalized substring match) in the catalog.

    Prefer the row with a morphology label (the DEV elliptical modeling table)
    over the dirty cross_refs rows with type=None in full.csv (otherwise an
    elliptical like J1630 falls to the disk default).
    """
    target = _normalize_name(name)
    if not target:
        return None
    hits = catalog[catalog["_norm"].str.contains(target, regex=False, na=False)]
    if len(hits) == 0:
        return None
    dev = hits[hits["lens_type_label"].astype(str).str.upper().isin(ELLIPTICAL_LABELS)]
    if len(dev) > 0:
        return dev.iloc[0]
    return hits.iloc[0]


def identify(name: str, catalog: Optional[pd.DataFrame] = None) -> LensPlan:
    """Identify a lens by name -> :class:`LensPlan` (no heavy deps, standalone)."""
    if catalog is None:
        catalog = load_catalog()

    row = _lookup(catalog, name)
    if row is None:
        raise LookupError(
            f"lens '{name}' not found in the sample catalog; "
            f"pass a catalog= DataFrame or set the right catalog_dir"
        )

    lens_type, type_basis = identify_type(row)
    constraint, constr_basis = identify_constraint(row, lens_type)
    basis = f"{type_basis}; {constr_basis}"
    return plan_modeling(name, lens_type, constraint, basis)


def write_recipe(plan: LensPlan, params: dict, out_path) -> Path:
    """Copy the ``plan.template_yaml`` and overwrite keys in ``params``, write YAML.

    ``params`` supports dot-nested keys (``"search.name"`` ->
    ``cfg["search"]["name"]``) and whole-dict values (``{"search": {"name": ...}}``).
    Complex params (data paths, geometry, priors) are still the caller's job to
    supply in full. Returns the written path.
    """
    import yaml

    out_path = Path(out_path)
    tpl_rel = Path(plan.template_yaml)
    if not tpl_rel.is_absolute():
        tpl_rel = Path(__file__).resolve().parents[3] / tpl_rel
    if not tpl_rel.exists():
        raise FileNotFoundError(f"recipe template not found: {tpl_rel}")

    with open(tpl_rel, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}

    for k, v in params.items():
        if "." in k:
            parts = k.split(".")
            d = cfg
            for p in parts[:-1]:
                d = d.setdefault(p, {})
            d[parts[-1]] = v
        else:
            cfg[k] = v

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(cfg, fh, allow_unicode=True, sort_keys=False)
    return out_path
