# galaxylens — Gravitational Lens Modeling Library

**Strong gravitational lens modeling for disk and elliptical galaxies.**

Core scientific goal: use galaxy-formation hydrodynamical simulations to study
strong lensing by disk galaxies, compare with space-telescope (HST)
observations, test simulation predictions, and constrain dark matter models and
the nature of dark matter particles. Elliptical galaxies serve as the
comparison sample.

> **Status**: pre-alpha. Repository skeleton established; module code is being
> migrated from the internal research project incrementally.

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
│   ├── lensfit/                 #   Lens fitting engine (observations + mocks)
│   │   ├── data/                #     read processed → imaging + PSF objects
│   │   ├── mge/                 #     MGE fitting + deprojection
│   │   ├── lens/                #     lens model builder (disk / elliptical)
│   │   ├── dynamics/            #     jampy (primary) / AGAMA / JAX Jeans (cross-check)
│   │   ├── fit/                 #     lens-only / dynamics-only / joint likelihood
│   │   ├── plotting/            #     unified plotting (paper style)
│   │   └── validation/          #     ground-truth comparison (mock validation)
│   └── sample_library/          #   Observational sample library access
├── configs/                     # per-lens YAML (disk/ + slacs/)
├── sample_library/              # observational sample data
├── examples/                    # cluster run templates (slurm)
├── notebooks/                   # tutorials
├── tests/                       # unit tests
└── docs/                        # documentation
```

## Migration status

Code is being migrated from the internal research project incrementally.

| Module | Status |
|---|---|
| `data_pipeline/imaging` | ✅ migrated (21 files; psf split into subpackage; path + SExtractor + TinyTim env-var decoupling) |
| `config` (YAML + dataclass) | ✅ migrated (paths/schema/loader/defaults; 16 lens YAMLs under configs/) |
| `sample_library` | ✅ migrated (build pipeline + access; 5 modeling_input CSVs) |
| `data_pipeline/spectra` | ⏳ pending |
| `data_pipeline/other_obs` | ⏳ pending |
| `lensfit` (data/mge/lens/dynamics/fit/plotting/validation) | ⏳ pending |
| `mock_pipeline/truth` (kinematics/measures/cosmology) | ✅ migrated (pure functions + regression tests; 348489 f_disk=0.643 reproduced) |
| `mock_pipeline/selection` (scan/filter/merger/download/verify/catalog) | ✅ migrated (per-target config; selection criteria are data) |
| `mock_pipeline/skirt` (convert/render/imaging) | ✅ migrated (inertia-tensor alignment + .ski generation + fake HST) |

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

```bash
pip install -e .            # core (numpy/scipy/astropy/pyyaml/pandas)
pip install -e ".[lens]"    # + PyAutoLens fitting stack
pip install -e ".[dynamics]" # + AGAMA / jampy / mgefit
```

## Quick start

```bash
# point to the directory containing raw/ and processed_v2/
export GALAXYLENS_DATA_ROOT=/path/to/DataPipeline

# preprocess one lens
python -m galaxylens.data_pipeline.imaging.preprocess_master \
    --lens J2141-0001 --version-tag v1
```

See [docs/data_pipeline.md](docs/data_pipeline.md) for the full guide.

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

## License

MIT
