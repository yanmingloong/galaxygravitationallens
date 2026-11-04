---
name: galaxylens-pipeline
description: "Use when operating the galaxylens lens-analysis pipeline."
version: 0.1.0
author: Yanming Long
license: MIT
metadata:
  hermes:
    tags: [galaxylens, gravitational-lensing, fitting, operations]
    related_skills: [galaxygravitationallens-migration]
---

# galaxylens Pipeline Skill (Operations)

## When to Use

Load this skill when the user asks to run or operate the galaxylens package —
run a fit, analyze a result, identify a lens, produce mock images, submit a
cluster job, or read/diagnose fit output. Do NOT load it for extending the
package (adding models/engines) — that is galaxygravitationallens-migration.

## Purpose

Operate the `galaxylens` package for strong-gravitational-lens + stellar-dynamics
analysis of disk and elliptical galaxies. This skill covers **running** the
pipeline (fit / identify / mock / analyze) and diagnosing its output. Extending
the package (adding models/engines, migrating research scripts) belongs to the
`galaxygravitationallens-migration` skill, not this one.

## Non-Negotiable Rules

- **`import galaxylens` must stay light.** The heavy deps (autolens/autofit/mgefit/jampy)
  are imported lazily inside each function. Do not add top-level imports of them.
- **Results never live in the repo.** All fit/mock outputs go under
  `GALAXYLENS_OUTPUT_ROOT` (or `search.output_root`); never write FITS/samples
  into the repository tree.
- **Read the weighted posterior, never an unweighted one.** `samples.csv` carries
  a `weight` column (Nautilus posterior weights). Median/percentiles must be
  weight-weighted; unweighted `np.percentile` is biased low.
- **`log Z` is not comparable across different priors or `search.name`s.** A
  higher `log Z` with looser priors can be physically wrong (see Pitfalls).
- **Any change that alters the likelihood (not just model/search) must bump
  `search.name`**, or Nautilus resumes the old samples under the same hash.
- **Never run multi-core Nautilus without `OMP_NUM_THREADS=1`** (plus OPENBLAS/
  MKL/VECLIB/NUMEXPR pinned to 1) — BLAS threads thrash and crash silently.

## Codebase map

```
src/galaxylens/
├── config/          # env-var path resolution (GALAXYLENS_DATA_ROOT etc.)
├── data_pipeline/   # observational data production (HST imaging; spectra/other_obs pending)
├── mock_pipeline/   # simulated data production + selection (TNG -> SKIRT -> mock FITS)
│   ├── selection/   # TNG disk-lens analog selection (built-in targets J2141, J1331)
│   ├── skirt/       # SKIRT radiative transfer -> mock images
│   └── truth/       # ground-truth extraction (f_disk / f_dm / beta_z / gNFW halo)
├── lensfit/         # fitting engine (pluggable models/dynamics/constraints + assemble)
├── sample_library/  # observational sample access + identify + morphology
└── api/             # high-level facade (fit/analyze/identify/load_config/load_dataset)
```

### Facade API (`import galaxylens as gl`)

| Entry | Signature | What it does |
|---|---|---|
| `gl.load_config` | `(path) -> FitConfig` | Load a fit-recipe YAML (geometry/kinematics/components/priors/stellar template/data/search) |
| `gl.load_dataset` | `(cfg, out_dir=None) -> (dataset, n_unmasked)` | Load masked (+ optional galaxy-subtracted + rescaled) `al.Imaging` |
| `gl.fit` | `(cfg, config_path=None) -> Result` | Full joint fit: dataset -> assemble -> Nautilus -> `Result` |
| `gl.fit_positions` | `(theta_obs, err_arcsec, zlens=0.113, zsrc=0.254, ...) -> ScaleFreeFit` | Point-level quad fit (scale-free potential, Trick+2016 J1331) |
| `gl.fit_mein` | `(geometry, kinematics, lum_mge, ml_array, rein_arcsec, mein_msun, ...) -> dict` | Point-level joint fit (NFW + radial-M/L + KDC jampy, scipy MAP) |
| `gl.analyze` | `(cfg, samples_csv=None, out_dir=None, corner=True, verbose=True) -> dict` | Post-fit summary + derived + prior-edge check + corner |
| `gl.identify` | `(name, catalog=None) -> LensPlan` | Lens type + constraint + modeling module + template_yaml |
| `gl.identify_from_image` | `(fits_path, name=None, center=None, rmax_pix=None) -> LensPlan` | Image morphology (C + q) fallback classification |
| `gl.mock` | `build_mock_pipeline(...) -> Pipeline` | Mock-closure pipeline (select -> truth -> render -> arc -> fit -> closure) |
| `gl.Pipeline` / `gl.Stage` | orchestration | Ordered stages, `run(only=...)` / `run(start=, stop=)` partial modes |

## Environment requirements

| Variable / tool | Needed by | Notes |
|---|---|---|
| `GALAXYLENS_DATA_ROOT` | data_pipeline, lensfit (data loading) | observational data root (`raw/` + `processed_v2/`) |
| `GALAXYLENS_SAMPLE_LIB` | sample_library, enrich_elliptical | sample-library root (`01_raw` ~ `05_exports`) |
| `GALAXYLENS_OUTPUT_ROOT` | facade / fits | results root — REQUIRED unless `search.output_root` set |
| `GALAXYLENS_CONFIG_DIR` | config loader | optional; defaults to repo `configs/` |
| `TNG_API_KEY` | mock selection downloads | env var, or fallback `~/.tng_api_key` file (single line) |
| `SKIRT_BIN` | mock skirt render | SKIRT executable |
| `SEXTRACTOR_BIN` / `TINYTIM_*` | data_pipeline imaging/psf | external tools |

- **Local full-stack env**: `autolens_jax` (numpy/scipy/astropy/h5py/autolens/autofit/jampy/mgefit/nautilus all present). Run tests with
  `PYTHONPATH=src /home/longym/anaconda3/envs/autolens_jax/bin/python -m pytest tests/`.
- **SKIRT** (mock images) is a compiled binary, not pip — set `SKIRT_BIN`.

## Standard workflows

### Observational analysis (identify -> recipe -> fit -> analyze)

```python
import galaxylens as gl

# 1. identify the lens -> modeling plan + recipe template
plan = gl.identify("J2141-0001")           # disk + arc -> j2141_barnabe.yaml
# plan = gl.identify_from_image("j1630_f814w_sci.fits")  # image fallback

# 2. load the recipe, run the joint fit
cfg = gl.load_config("configs/fit/j2141_barnabe.yaml")
result = gl.fit(cfg)                       # assemble -> Nautilus -> post-process
print(result.beta_z, result.m_star, result.f_dm(2.2 * cfg.pipeline.geometry.rd_arcsec))

# 3. one-call post-fit summary + corner
gl.analyze(cfg)                            # -> {"params", "derived", "edges"}
```

### Point-level quad (J1331, no extended arc)

```python
from galaxylens.api.facade import fit_positions, fit_mein
sf = fit_positions(theta_obs, err_arcsec)  # scale-free, 4 image positions
res = fit_mein(geometry, kinematics, lum_mge, ml_array,
               rein_arcsec=..., mein_msun=...)  # NFW + radial-M/L + KDC, scipy MAP
```

### Mock closure (TNG -> SKIRT -> fit -> truth comparison)

```python
import galaxylens as gl
pipe = gl.mock(target="j2141", sid=348489)   # select -> truth -> render -> arc -> fit -> closure
pipe.run(only=["truth", "render", "arc"])    # heavy stages need TNG data + SKIRT
```

### Cluster (backend: slurm)

Set `search.backend: slurm` + `search.name` in the recipe, then
`gl.fit(cfg, config_path="configs/fit/xxx.yaml")` renders + writes + submits the
SBATCH script (returns `script_path` / `job_id`). The cluster runner
(`galaxylens.api.runner`) loads the recipe and runs the fit. Template env block
lives in `examples/*.slurm`.

## Reading results (weighted posterior)

- `samples.csv` columns: parameters (`galaxies.lens.dark.v_vir`,
  `galaxies.lens.dark.inner_slope`, `galaxies.lens.mge_<i>.mass_to_light_ratio`,
  `galaxies.lens.anisotropy.mass_to_light_ratio`, ...) + `weight`.
- Weighted percentile (never `np.percentile` unweighted):
  `idx = np.argsort(x); cdf = np.cumsum(w[idx]); cdf /= cdf[-1]; med = np.interp(0.5, cdf, x[idx])`.
- The M/L column name carries a Gaussian index (`mge_13`, `mge_0`, ...) that
  changes with the stellar template; match it with
  `re.search(r"galaxies\.lens\.mge_\d+\.mass_to_light_ratio", col)`.
- `beta_z = 1 - ratio**2`; `M* = ml_img * (CALIB * L2D_REL)` (the product
  `CALIB*L2D_REL` is the BLAS-stable anchor, not the individual factors).
- Median in log space for M*: `10 ** wmedian(log10(ml_img * bridge))`, not
  `wmedian(10 ** ...)`.

## Pitfalls (operations)

- **Unweighted percentile is biased low.** Always weight by the `weight` column
  (see above). B7: unweighted `beta_z=0.70` vs correctly weighted `0.65`.
- **Prior-edge (pinned) check = quantile distance, not weight share.** A
  parameter is pinned when its 2.5%/97.5% weighted quantile touches the prior
  bound (< 1% of the prior width). "Weight share near the bound" is a wrong
  test. Pinned posteriors -> median/error bars are truncated (biased); loosen
  the prior and re-run.
- **`log Z` is not comparable across priors or `search.name`s.** A looser prior
  that slides to the "high-evidence but physically wrong" end of a degeneracy
  valley can have a higher `log Z` (J1630 v3 `log Z=-4.08` vs v2 `-20.61`, but
  v3 is physically wrong). Judge physics by cross-checks (e.g. Einstein radius
  vs literature), not `log Z`.
- **Single-aperture `sigma_ap` cannot break the mass-anisotropy degeneracy**
  for slow-rotator ellipticals: "high M/L + isotropic + concentrated halo" and
  "low M/L + radial beta_z + diffuse halo" both reproduce one `sigma_ap`. Keep
  a tight isotropy prior, or add a radial v_rms profile (like B7's 11 points).
- **Bump `search.name` whenever the likelihood changes** (not just model/search):
  the output dir hash depends on model+search only, so a changed likelihood
  resumes the OLD samples silently.
- **`scancel` once sampling is Finished.** samples.csv is already on disk;
  autofit then spends hours computing latent derived quantities (Einstein radius
  per sample, ~145 s/100 samples) you don't need. Pull `samples.csv` and stop.
- **`scp -r cluster:.../<hash>/files local:.../<hash>/` flattens `files/` into
  `<hash>/`.** After pull, `find <hash> -name samples.csv`; if flattened,
  `mkdir -p <hash>/files && mv`.
- **SLURM array `--array=0-1` with one candidate -> index 1 runs empty (~1 s
  COMPLETED).** Check `sacct` Elapsed≈00:00:01 + MaxRSS≈588K means "did not fit".
- **`gl.fit` on `backend: local` requires `GALAXYLENS_OUTPUT_ROOT`** (or
  `search.output_root`); it raises otherwise.
- **Heavy deps are lazy-imported.** Pyright "Import could not be resolved" for
  autolens/autofit/jampy/mgefit is expected, not an error.
- **Cross-engine environments**: mgefit -> `jampy_env`; autolens/autofit ->
  `autolens_jax`; photutils -> `astro_kai`. One script mixing imports fails on
  the missing package.

## Anchor values (validate against)

- **J2141 B7 joint fit** (weighted median, `samples_summary.json`): ml_img=13.5759,
  v_vir=212.267, c_-2=5.095, gamma=0.7715, q_h=0.6938, ratio=0.5908 ->
  beta_z=0.651, M*=1.436e11 Msun, V_circ(2.2Rd)~271.8 km/s.
- **J1630+4520 elliptical** (v2, tight isotropy prior): log Z=-20.61, N_eff=2443,
  theta_E=1.746", gamma_tot=1.87, f_DM(<theta_E)=0.789, M*=2.05e11 Msun,
  beta_z=+0.311, bulge_i0=232.0 counts/s/px.
- **348489 mock truth**: M*=8.90e10, v_vir=218 km/s, c_-2=5.55, gamma=1.53,
  q_h=0.736, f_dm 5 kpc=0.4433 / 10 kpc=0.5984; theta_E(arc)=0.609".
- **Cosmology**: Planck 2015 (H0=70, Om0=0.3 for `halo_cosmo`); H0_GYR=0.06928
  (NOT 0.06774, a source-script typo).
