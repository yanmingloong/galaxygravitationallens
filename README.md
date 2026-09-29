# galaxylens

Strong gravitational lens modeling for **disk** and **elliptical** galaxies.

The project's core scientific goal: use galaxy-formation hydrodynamical simulations to study strong lensing by disk galaxies, compare with space-telescope observations, test simulation predictions, and constrain dark matter models and the nature of dark matter particles. Elliptical galaxies serve as the comparison sample.

> **Status**: pre-alpha. Repository skeleton established; module code is being migrated from the internal research project incrementally.

## Architecture

```
galaxygravitationallens/
├── src/galaxylens/              # Python package (src layout)
│   ├── config/                  #   YAML -> dataclass configuration loading (shared)
│   ├── data_pipeline/           #   Observational data production
│   │   ├── imaging/             #     HST lens imaging (FLC/FLT preprocessing)
│   │   ├── spectra/             #     spectra (SDSS / KCWI / DEIMOS)
│   │   └── other_obs/           #     other observations (e.g. NIRC2 K'-band)
│   ├── mock_pipeline/           #   Simulated data production + selection
│   │   ├── selection/           #     TNG disk-lens analog screening
│   │   ├── skirt/               #     SKIRT radiative transfer -> mock images
│   │   └── truth/               #     ground-truth extraction (f_disk, etc.)
│   ├── lensfit/                 #   Lens fitting engine (shared by obs + mock)
│   │   ├── data/                #     read processed data -> imaging + PSF objects
│   │   ├── mge/                 #     MGE fit + deprojection
│   │   ├── lens/                #     lens model builders (disk / etg series)
│   │   ├── dynamics/            #     JAX Jeans / AGAMA / jampy (unified API)
│   │   ├── fit/                 #     lens-only / dynamics-only / joint likelihood
│   │   ├── plotting/            #     unified publication-style plotting
│   │   └── validation/          #     ground-truth comparison (mock validation)
│   └── sample_library/          #   observational sample access
├── configs/                     # per-lens YAML configs (disk/ + slacs/)
├── sample_library/              # observational sample catalog data
├── notebooks/                   # tutorials
├── tests/                       # unit tests
└── docs/                        # documentation
```

## Migration status

Code is being migrated from the internal research project incrementally.

| Module | Status |
|---|---|
| `data_pipeline/imaging` | ✅ migrated (21 files; psf split into sub-package; paths + SExtractor + TinyTim env-decoupled) |
| `data_pipeline/spectra` | ⏳ pending |
| `data_pipeline/other_obs` | ⏳ pending |
| `lensfit` (data/psf/mge/lens/dynamics/fit/plotting/validation) | ⏳ pending |
| `mock_pipeline` (selection/skirt/truth) | ⏳ pending |
| `sample_library` | ✅ migrated (build pipeline + access; 5 modeling_input CSVs) |
| `config` (YAML + dataclass) | ✅ migrated (paths/schema/loader/defaults; 16 lens YAMLs in configs/) |

## Design principles

- **Code is split by capability, not by paper or galaxy type.** B/D series (disk) and e series (elliptical) all call the same `lensfit` engine; the difference lives in lens model builders, dynamics engine choice, and configuration.
- **Data production vs consumption are separated.** `data_pipeline` / `mock_pipeline` produce data; `lensfit` consumes it. Mock fitting = `lensfit` + ground-truth comparison (`validation/`).
- **Observational vs simulated data are different in nature** (no ground truth vs ground truth), so they are separated at the production layer but share the fitting engine.

## Installation

```bash
pip install -e .            # core (numpy/scipy/astropy)
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

## License

MIT
