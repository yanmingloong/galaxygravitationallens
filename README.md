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
│   ├── api/                     #   Facade layer: gl.fit / gl.analyze / gl.identify / gl.mock / gl.load_config / gl.load_dataset
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
| `lensfit` (models/dynamics/fit/pipeline/data/mge/lens/plotting/validation) | ✅ migrated (pluggable model components + config assembly + JointAnalysis; gnfw_virial/nfw/mge_const_ml/mge_radial_ml/single_sersic/pixelized components; jampy/jampy_kdc engines; LensingConstraint imaging/mein/scale-free; B7 joint fit + D-series pointwise fit reproduced; B9 AGN point-source subtraction + sky-RMS noise + pixelized source) |
| `api` (config_loader/dataset/search/result/derived/plotting/facade + pipeline/analysis) | ✅ migrated (gl.fit / gl.fit_positions / gl.fit_mein / gl.analyze / gl.mock / gl.load_config / gl.load_dataset) |
| `mock_pipeline/truth` (particles/fdm/anisotropy/halo + kinematics/measures/cosmology) | ✅ migrated (pure functions + regression tests; 348489 f_disk=0.643 reproduced; f_dm spherical/cylindrical three apertures + beta_z + gNFW halo fit added 2026-10-27) |
| `mock_pipeline/lensing` (arc + dataset) | ✅ migrated (true-mass lens galaxy + Einstein radius + arc ray-tracing + dataset merge, 2026-10-27) |
| `mock_pipeline/selection` (scan/filter/merger/download/verify/catalog) | ✅ migrated (per-target config; J2141 + J1331 built-in targets) |
| `mock_pipeline/skirt` (convert/render/imaging) | ✅ migrated (inertia-tensor alignment + .ski generation + fake HST + single-scale multi-band dataset 2026-10-20) |

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

## License

MIT
