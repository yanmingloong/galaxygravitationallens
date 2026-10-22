# mock_pipeline usage documentation — mock data production and selection

*中文版见 [mock_pipeline_zh.md](mock_pipeline_zh.md).*

> Select disk-galaxy strong-lens analogs from IllustrisTNG and generate SKIRT mock images.
> This module produces **mock data with ground truth**, distinct from the observational sample library `sample_library` (real survey catalogs, no ground truth).

## 1. Module structure

```
mock_pipeline/
├── truth/        # ground-truth computation (shared by selection and validation)
│   ├── kinematics.py   # circularity → f_disk / B/D / disk axis / thickness
│   ├── measures.py     # half-light radius / projected ellipticity / intrinsic axis ratio / extinction
│   └── cosmology.py    # Planck 2015 distances / SIS θE / cosmic time / TNG unit conversion
├── selection/    # TNG disk-lens analog selection
│   ├── scan.py        # full scan (API pagination + detail fetching)
│   ├── filter.py      # L1 filtering (criteria as data)
│   ├── merger.py      # L2 merger history (mpb.hdf5 main-progenitor mass jumps)
│   ├── download.py    # particle download (3 fields / full fields)
│   ├── verify.py      # L3 verification (download + disk_kinematics)
│   ├── catalog.py     # L3 → catalog auto-generation
│   ├── config.py      # TargetConfig / SelectionCriteria (criteria)
│   └── run.py         # --target end-to-end orchestration
└── skirt/        # SKIRT radiative transfer → mock images
    ├── convert.py     # TNG particles → .dat (inertia-tensor disk-axis alignment)
    ├── config.py      # BandConfig/MockObsConfig (bands/instruments/geometry, as data)
    ├── render.py      # .ski generation + skirt run
    └── imaging.py     # fake HST + extinction map
```

## 2. End-to-end data flow

```
scan (full scan, mass__gt filter)
  → L1 filtering (field center + M* window + σ lower bound, criteria from TargetConfig)
  → L3 verification (download 3-field particles → disk_kinematics computes f_disk/B/D/L_hat)
  → catalog auto-generation (select disk-dominated subset, L_hat expanded into three columns)
  → skirt (download full-field particles → convert inertia-tensor alignment → .ski → SKIRT 1e9 → fake HST)
```

L2 (merger history) is not in the main chain; it is called independently per candidate via `merger.check_merger` (consistent with the census flow scan→L1→L3).

## 3. Quick start

```bash
# Environment variables
export TNG_API_KEY=...            # TNG API key (or write to ~/.tng_api_key, see guidance below)
export GALAXYLENS_DATA_ROOT=/path/to/tng_data   # data root (catalog/ + stars_l3/)
export SKIRT_BIN=/path/to/skirt   # SKIRT 9 binary (used by render)

# One command to search for J2141 analogs (built-in target)
python -m galaxylens.mock_pipeline.selection.run \
    --target j2141 --data-root $GALAXYLENS_DATA_ROOT --outdir ./out

# smoke test: limit detail count per snapshot
python -m galaxylens.mock_pipeline.selection.run \
    --target j2141 --max-detail 5 --l3-workers 1
```

> **Getting the TNG API key** (free, anyone can apply): register an account at tng-project.org → after logging in,
> get the key from the [API documentation](https://www.tng-project.org/data/docs/api/) → provide it to the script via one of two ways:
> `export TNG_API_KEY=your_key` or write it to `~/.tng_api_key` (single line). The repo does not bundle any key;
> when unconfigured, the script prints "TNG API key not found" and points to this guide.

**Switching target = adding one `TargetConfig`** (in `selection/config.py` or later YAML), with zero changes to the pipeline code:

```python
from galaxylens.mock_pipeline.selection.config import TargetConfig, SelectionCriteria
J1331 = TargetConfig(
    name="J1331+3628", z_lens=0.113, z_source=0.254, snap=88,
    incl="70 deg",
    criteria=SelectionCriteria(
        sims=["TNG100-1"], snapshots=[88, 91, 84, 95, 99],
        mass_min=68.0, mstar_range=(5e10, 3e11), sigma_min=140.0,
        f_disk_disk=0.30, f_disk_min=0.60,
    ),
)
```

## 4. Key physical definitions (must distinguish, do not mix up)

| Quantity | Definition | Location |
|----|------|------|
| **f_disk** | **mass fraction** with circularity κ≥0.7 (κ=jz/(R·vc), disk-particle criterion) | `truth/kinematics.py` |
| **disk axis L_hat** | disk-particle angular-momentum axis (iterated once), **not** total angular momentum, **not** inertia tensor | same as above |
| **B/D** | (1−f_disk)/f_disk, sentinel 999.0 when no disk | same as above |
| **projected ellipticity** | 1−√(λmin/λmax), F814W intensity-weighted second moment with **no threshold** | `truth/measures.py` |
| **intrinsic axis ratio** | σ_z/σ_x of aligned particles (3D disk thickness), **not** projected ellipticity | same as above |
| **disk-axis alignment (SKIRT)** | **inertia tensor** three axes (normal→+z, major axis→+x), does **not read** the old L_hat in catalog.csv | `skirt/convert.py` |
| **cosmology** | Planck 2015 (H0=67.74, Ωm=0.3089, ΩΛ=0.6911), not Planck18 | `truth/cosmology.py` |
| **kpc/″** | angular diameter distance D_A(z)/206265, varies with redshift (J2141 z=0.138 → 2.517) | same as above |

**Historical lesson** (easily confused points): σ(total) is the **full-particle dispersion** (including DM; cold thin disks land at 140–160), σ≥170 biases toward bulge-dominated disks; ellipticity once mixed up the "intrinsic axis ratio vs projected" two definitions, now unified to the projected covariance definition.

## 5. Selection criteria as data

Criteria live in the `SelectionCriteria` dataclass (**data, not code**), one per target:

| Field | Meaning | J2141 default |
|------|------|-----------|
| `mstar_range` | L1 stellar mass window (Msun) | [5e10, 3e11] (census definition, relaxed; the J2141-specific tight window [7e10,1.3e11] is a subset) |
| `sigma_min` | L1 σ(total) lower bound (km/s) | 140 (relaxed; 170 biases toward bulge-dominated disks) |
| `f_disk_min` | L3 disk-dominated subset threshold | 0.6 |
| `f_disk_disk` | L3 "has disk" threshold | 0.30 |
| `mass_min` | scan pre-filter (total mass, 1e10 Msun/h) | 68 (verified to not miss field-center targets) |
| `field_only` | L1 requires field center (primary_flag=1) | True |

## 6. API summary

```python
from galaxylens.mock_pipeline.truth import disk_kinematics, projected_ellipticity, theta_E_SIS, kpc_per_arcsec
from galaxylens.mock_pipeline.selection import run_selection, build_catalog, verify_one
from galaxylens.mock_pipeline.skirt import (
    tng_to_skirt, write_ski, run_skirt, make_fake_hst,
    BandConfig, MockObsConfig, J2141_BANDS, J2141_OBS,
)

# ground truth: particles → f_disk/B/D/L_hat/thickness/Re
res = disk_kinematics(pos_kpc, vel, mass, center)   # pos/vel/mass in physical units

# end-to-end selection
rows = run_selection("j2141", data_root, outdir)    # scan → L1 → L3 → catalog

# SKIRT: full-field particles → .dat → .ski → fake HST
tng_to_skirt(stars_h5, gas_h5, "348489", "TNG100-1", 88, outdir)
write_ski("348489", J2141_BANDS[0], outdir, obs=J2141_OBS, roll="0 deg", incl=J2141.incl)
run_skirt(f"{outdir}/348489_F814W.ski", outdir=outdir)
img = make_fake_hst(fits_path, px_mock, J2141_BANDS[0], psf)

# switch instrument (e.g. J1331 uses WFPC2 F814W, inclination 70°): create a new BandConfig, zero code changes
wfpc2 = BandConfig("F814W", "HST WFPC2 F814W", 0.10)
write_ski("J1331", wfpc2, outdir, incl="70 deg")
```

## 7. Regression tests

```bash
PYTHONPATH=src python -m pytest tests/test_truth.py tests/test_selection.py tests/test_skirt.py -v
```

22 tests, anchoring known values: 348489 f_disk=0.643, B/D=0.55, kpc_per_arcsec(0.138)=2.517, θE(σ=200)=0.89″.
