# galaxylens — Gravitational Lens Modeling Library

**Strong gravitational lens modeling for disk and elliptical galaxies.**

Core scientific goal: jointly model the strong gravitational lensing and
stellar dynamics of disk and elliptical galaxies to constrain their total mass
distribution, dark-matter halo, and stellar mass-to-light ratio, and thereby
constrain dark-matter models; and to validate the modeling method and the
predictions of cosmological hydrodynamical simulations (IllustrisTNG) against
space-telescope (HST) observations through mock observations. Elliptical
galaxies serve as the comparison sample.

The library provides a joint lensing + dynamics fitting pipeline. Gravitational
lens models (a mass model plus a source model, ray-traced and fitted against
the imaging data) are constructed with PyAutoLens, while stellar dynamical
models are built with jampy (Jeans Anisotropic MGE), which solves the
axisymmetric Jeans equations for a multi-Gaussian-expansion stellar
distribution. The two are fitted jointly to constrain the stellar
mass-to-light ratio, the dark-matter halo (gNFW / NFW) and the velocity
anisotropy β_z, yielding the dark-matter fraction within the Einstein radius,
f_DM. The methodological framework follows Barnabè et al. (2012, SWELLS IV);
the implementation uses the independent software stack above.

> Originating from the undergraduate innovation project
> **"Disk-galaxy strong gravitational lensing research" (盘星系引力透镜研究)**.

> **Status**: pre-alpha.

中文文档见 [README_zh.md](README_zh.md).

## Architecture

```
galaxygravitationallens/
├── src/galaxylens/              # Python package (src layout)
│   ├── config/                  #   YAML → dataclass config loading (shared)
│   ├── data_pipeline/           #   Observational data production
│   │   ├── imaging/             #     HST lens images (FLC/FLT preprocessing)
│   │   ├── spectra/             #     spectra (SDSS / KCWI / DEIMOS)
│   │   └── other_obs/           #     other observations (NIRC2 K'-band etc.)
│   ├── mock_pipeline/           #   Simulated data production + selection
│   │   ├── selection/           #     TNG disk-lens analog selection
│   │   ├── skirt/               #     SKIRT radiative transfer → mock images
│   │   └── truth/               #     ground-truth extraction (f_disk etc.)
│   ├── lensfit/                 #   Lens fitting engine (pluggable components + config assembly)
│   │   ├── models/              #     pluggable model components (dm / stellar / source)
│   │   ├── dynamics/            #     jampy (primary) / AGAMA / JAX Jeans (cross-check)
│   │   ├── fit/                 #     JointAnalysis + pluggable lens constraints
│   │   ├── pipeline/            #     config assembly (zero physics)
│   │   ├── data/                #     read processed → imaging + PSF objects
│   │   ├── mge/                 #     MGE fitting + deprojection
│   │   ├── lens/                #     lens model builder (disk / elliptical)
│   │   ├── plotting/            #     unified plotting (paper style)
│   │   └── validation/          #     ground-truth comparison (mock validation)
│   ├── api/                     #   High-level API: gl.fit / gl.analyze / gl.identify / gl.mock / gl.load_config / gl.load_dataset
│   └── sample_library/          #   Observational sample library access
├── configs/                     # per-lens YAML (disk/ + slacs/)
├── sample_library/              # observational sample data
├── examples/                    # cluster run templates (slurm)
├── notebooks/                   # tutorials
├── tests/                       # unit tests
└── docs/                        # documentation
```

## Capabilities

| Module | Description |
|---|---|
| `data_pipeline/imaging` | HST FLC/FLT preprocessing -> sci/noise/psf/starmask (SExtractor + TinyTim) |
| `config` (YAML + dataclass) | YAML -> dataclass config loading (16 lens YAMLs under configs/) |
| `sample_library` | observational sample library access (5 modeling_input CSVs) |
| `data_pipeline/spectra` | spectra (SDSS / KCWI / DEIMOS) — planned |
| `data_pipeline/other_obs` | other observations (NIRC2 K'-band) — planned |
| `lensfit` | pluggable fitting engine: gnfw_virial / nfw / mge_const_ml / mge_radial_ml / single_sersic / pixelized components; jampy / jampy_kdc dynamics; LensingConstraint imaging / mein / scale-free; joint fit + Trick+2016 pointwise fit |
| `api` | high-level API: gl.fit / gl.fit_positions / gl.fit_mein / gl.analyze / gl.mock / gl.load_config / gl.load_dataset |
| `mock_pipeline/truth` | ground-truth extraction: f_disk / f_dm (spherical + cylindrical) / beta_z / gNFW halo fit |
| `mock_pipeline/lensing` | true-mass lens galaxy + Einstein radius + arc ray-tracing + dataset merge |
| `mock_pipeline/selection` | TNG disk-lens analog selection (J2141 + J1331 built-in targets) |
| `mock_pipeline/skirt` | SKIRT radiative transfer -> mock images (inertia-tensor alignment + .ski + fake HST + multi-band dataset) |

## Design principles

- **Split by capability, not by paper or galaxy type.** B/D series (disk) and
  e series (elliptical) all call the same `lensfit` engine; the difference lives
  in lens model builders, dynamics engine choice, and configuration.
- **Data production vs consumption are separated.** `data_pipeline` /
  `mock_pipeline` produce data; `lensfit` consumes it. Mock fitting = `lensfit`
  + ground-truth comparison (`validation/`).
- **Observational vs simulated data are different in nature** (no ground truth
  vs ground truth), so they are separated at the production layer but share the
  fitting engine.

## Installation

### Python package

```bash
pip install -e .            # core (numpy/scipy/astropy/pyyaml/pandas)
pip install -e ".[lens]"    # + PyAutoLens fitting stack
pip install -e ".[dynamics]" # + AGAMA / jampy / mgefit
```

### External tools (not pip-installable)

| Tool | Purpose | Env var |
|---|---|---|
| PyAutoLens | lens imaging / ray-tracing / fitting | pip, `[lens]` extra |
| jampy + mgefit | Jeans anisotropic dynamics | pip, `[dynamics]` extra |
| Nautilus | nested sampling | pip, via PyAutoLens |
| SKIRT | Monte Carlo radiative transfer (mock images) | `SKIRT_BIN` |
| SExtractor | source detection / photometry | `SEXTRACTOR_BIN` |
| TinyTim | HST PSF generation | `TINYTIM_BIN`, `TINYTIM_GRID_BASE` |

### Environment variables

| Variable | Points to | Needed by |
|---|---|---|
| `GALAXYLENS_DATA_ROOT` | observational data root (raw/ + processed_v2/) | data_pipeline, lensfit |
| `GALAXYLENS_SAMPLE_LIB` | sample-library root (01_raw ~ 05_exports) | sample_library |
| `GALAXYLENS_OUTPUT_ROOT` | results root (kept out of the repo) | facade / fits |
| `GALAXYLENS_CONFIG_DIR` | configs/ dir (optional; defaults to repo configs/) | config loader |
| `TNG_API_KEY` | IllustrisTNG download API key | mock_pipeline selection |
| `SKIRT_BIN` | SKIRT executable | mock_pipeline skirt |
| `SEXTRACTOR_BIN` | SExtractor binary | data_pipeline imaging |
| `TINYTIM_BIN` / `TINYTIM_GRID_BASE` / `TINYTIM_BREATHING_CSV` | TinyTim paths | data_pipeline psf |

### Data

The repository contains no scientific data (`*.fits` / `*.hdf5` / `*.npz` are
gitignored). You must provide:

- **Observational data**: HST FITS from the MAST archive (or equivalent).
- **Simulation data**: IllustrisTNG particle HDF5 (requires a TNG API key).
- **Regression-test data** (optional): set `GALAXYLENS_MF_ROOT` /
  `GALAXYLENS_TEST_*` to run the data-dependent tests; they skip otherwise.

## Quick start

```bash
# point to the directory containing raw/ and processed_v2/
export GALAXYLENS_DATA_ROOT=/path/to/observations

# preprocess one lens
python -m galaxylens.data_pipeline.imaging.preprocess_master \
    --lens J2141-0001 --version-tag v1
```

See [docs/data_pipeline.md](docs/data_pipeline.md) for the full guide.

### Fit (lens + dynamics)

```python
import galaxylens as gl

cfg = gl.load_config("configs/fit/j2141_barnabe.yaml")
result = gl.fit(cfg)                    # assemble -> Nautilus -> post-process
print(result.beta_z, result.m_star, result.f_dm(2.2 * cfg.pipeline.geometry.rd_arcsec))

gl.analyze(cfg)                         # one-call post-fit summary + corner plot
```

See [docs/usage_guide.md](docs/usage_guide.md) Step 3 for the full fitting flow.

### Identify (lens type -> modeling plan)

```python
plan = gl.identify("J2141-0001")     # LensPlan: disk / imaging / jampy_kdc
plan = gl.identify_from_image("j1630_f814w_sci.fits")  # image fallback: C+q -> disk/elliptical
print(plan.lens_type, plan.constraint, plan.template_yaml)
```

`gl.identify` reads the sample catalog, infers lens type (disk/elliptical) and
constraint (arc/quad), and maps to modeling modules + a recipe template.
`gl.identify_from_image` is the image-shape fallback — light concentration +
axis ratio measured directly from a FITS when the catalog has no morphology —
the first link of the "identify -> model -> fit" chain.

### Orchestrate (auto + manual)

`gl.Pipeline` / `gl.Stage` wire the pluggable components into an ordered run
that supports full-auto, partial (`run(only=...)`) and range (`run(start=...,
stop=...)`) modes, while every stage stays individually callable. `gl.mock`
is the pre-built mock-closure pipeline (select -> truth -> render -> arc ->
fit -> closure); see [docs/usage_guide.md](docs/usage_guide.md) Step 4.

### Tutorials (runnable notebooks)

Three `# %%` notebooks under `notebooks/` walk the main chains end-to-end —
see [docs/README.md](docs/README.md#tutorials-runnable--notebooks-under-notebooks):

- `identify_demo.py` — identify -> recipe (fully offline)
- `fit_demo.py` — recipe -> fit -> analyse (synthetic-FITS smoke, offline)
- `mock_pipeline_demo.py` — TNG selection -> SKIRT -> fake HST

## Noise methods (data_pipeline/imaging)

`preprocess_master.py --noise-method` generates three noise versions per lens
system, all sharing the same science / PSF / starmask but differing only in the
noise map:

| Method | Noise map | Notes |
|---|---|---|
| `whtonly`  | σ = 1/√WHT | Pure WHT, physical baseline |
| `rescaled` | pure WHT × global factor | Preprocessing approximation (scale = σ_bg / median(noise_raw_bg)); stage-1 preferred |
| `bugnoise` | √(1/WHT + SCI) | Signal-dependent absorption term; backup |

For disk galaxies, pure WHT noise is too small — model systematics (≳16 e⁻/s)
dwarf the measurement noise (≈0.35 e⁻/s), blowing χ²/px up to ≈2167.
Barnabè+2012 (SWELLS IV) rescales globally to reduced χ² ≈ 1; `rescaled`
approximates that single factor at the preprocessing stage, while the fitting
stage (B4) applies the exact two-pass Barnabè rescaling (`RESCALE=True`,
f = √reduced χ²). `bugnoise` is a signal-dependent alternative, kept as a
backup.

## Cross-platform support

galaxylens is pure-Python (numpy/scipy/astropy/pyyaml/pandas + optional
PyAutoLens / jampy / mgefit / nautilus) and runs on Windows, WSL, Linux and
macOS (CPU). Three honest boundaries:

1. **PyAutoLens is Linux/macOS-first.** The `autolens`/`autofit` stack is
   primarily tested on Linux and macOS; on Windows it is "usable but not
   first-class support". galaxylens inherits this support level for the `[lens]`
   extra. Core (numpy/scipy/astropy) and the pure-python lensfit modules run
   identically on all three platforms.
2. **BLAS/LAPACK numeric non-determinism ~1%.** MGE decomposition (mgefit)
   converges to slightly different Gaussian sets across BLAS backends (B6/B7:
   CALIB 1427 vs 1157), but conserved products (CALIB x L_2D_REL) agree to ~1%.
   Cross-platform comparisons should anchor on the conserved quantity, not a
   single decomposition output.
3. **slurm cluster scheduling is Linux-only; GPU/JAX acceleration is
   Linux+NVIDIA-only.** The `backend: slurm` search backend submits SBATCH jobs
   on a Linux cluster; the `backend: local` backend runs everywhere. The CI
   matrix (`.github/workflows/ci.yml`) runs core + pure-python on Windows and
   Linux, and the full fitting stack on Linux only.

## References

Methods and models this library builds upon.

### Methods and models

| Reference | DOI | Used for |
|---|---|---|
| Barnabè, M.; Dutton, A.A.; Marshall, P.J.; et al. 2012, MNRAS, 423, 1073 | [10.1111/j.1365-2966.2012.20934.x](https://doi.org/10.1111/j.1365-2966.2012.20934.x) | joint lens + dynamics fitting method (galaxy-subtracted) |
| Treu, T.; Dutton, A.A.; Auger, M.W.; et al. 2011, MNRAS, 417, 1601 | [10.1111/j.1365-2966.2011.19378.x](https://doi.org/10.1111/j.1365-2966.2011.19378.x) | SWELLS survey (disk-lens sample) |
| Trick, W.H.; van de Ven, G.; Dutton, A.A. 2016, MNRAS, 463, 3151 | [10.1093/mnras/stw2142](https://doi.org/10.1093/mnras/stw2142) | quadruply-imaged disk lens J1331+3628 |
| Cappellari, M. 2002, MNRAS, 333, 400 | [10.1046/j.1365-8711.2002.05412.x](https://doi.org/10.1046/j.1365-8711.2002.05412.x) | multi-Gaussian expansion (MGE) |
| Cappellari, M. 2020, MNRAS, 494, 4819 | [10.1093/mnras/staa1128](https://doi.org/10.1093/mnras/staa1128) | jampy Jeans anisotropic dynamics |
| Wyithe, J.S.B.; Turner, E.L.; Spergel, D.N. 2001, ApJ, 555, 504 | [10.1086/321437](https://doi.org/10.1086/321437) | gNFW density profile |
| Navarro, J.F.; Frenk, C.S.; White, S.D.M. 1997, ApJ, 490, 493 | [10.1086/304888](https://doi.org/10.1086/304888) | NFW density profile |
| Dutton, A.A.; Macciò, A.V. 2014, MNRAS, 441, 3359 | [10.1093/mnras/stu742](https://doi.org/10.1093/mnras/stu742) | concentration–mass prior |
| Wright, C.O.; Brainerd, T.G. 2000, ApJ, 534, 34 | [10.1086/308744](https://doi.org/10.1086/308744) | analytic projected NFW mass |
| Evans, N.W.; Witt, H.J. 2003, MNRAS, 345, 1351 | [10.1046/j.1365-2966.2003.07057.x](https://doi.org/10.1046/j.1365-2966.2003.07057.x) | scale-free lens potential |
| Kochanek, C.S. 1991, ApJ, 373, 354 | [10.1086/170057](https://doi.org/10.1086/170057) | source-plane χ² with magnification weighting |
| Vasiliev, E. 2019, MNRAS, 482, 1525 | [10.1093/mnras/sty2672](https://doi.org/10.1093/mnras/sty2672) | AGAMA dynamics (cross-check) |
| Nightingale, J.W.; Dye, S.; Massey, R.J. 2018, MNRAS, 478, 4738 | [10.1093/mnras/sty1264](https://doi.org/10.1093/mnras/sty1264) | PyAutoLens lens imaging |
| Stetson, P.B. 1987, PASP, 99, 191 | [10.1086/131977](https://doi.org/10.1086/131977) | DAOStarFinder point-source detection |

### Mock-image pipeline

| Reference | DOI | Used for |
|---|---|---|
| Camps, P.; Baes, M. 2020, A&C, 31, 100381 | [10.1016/j.ascom.2020.100381](https://doi.org/10.1016/j.ascom.2020.100381) | SKIRT radiative transfer |
| Jones, A.P.; Köhler, M.; Ysard, N.; et al. 2017, A&A, 602, A46 | [10.1051/0004-6361/201630225](https://doi.org/10.1051/0004-6361/201630225) | THEMIS dust model |
| Bruzual, G.; Charlot, S. 2003, MNRAS, 344, 1000 | [10.1046/j.1365-8711.2003.06897.x](https://doi.org/10.1046/j.1365-8711.2003.06897.x) | stellar population synthesis (BC03) |
| Chabrier, G. 2003, PASP, 115, 763 | [10.1086/376392](https://doi.org/10.1086/376392) | Chabrier IMF |
| Nelson, D.; Pillepich, A.; Springel, V.; et al. 2019, MNRAS, 490, 3234 | [10.1093/mnras/stz2306](https://doi.org/10.1093/mnras/stz2306) | IllustrisTNG simulations |
| The Astropy Collaboration; Price-Whelan, A.M.; et al. 2022, ApJ, 935, 167 | [10.3847/1538-4357/ac7c74](https://doi.org/10.3847/1538-4357/ac7c74) | Astropy software |

## AI usage statement

This repository was organized with assistance from Hermes Agent. Large language
model: deepseek-v4.1-flash.

## Contributing

Issues (feature requests, bug reports) and pull requests are welcome on the
hosting platform (Gitee / GitHub). See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

MIT License — see [LICENSE](LICENSE).
