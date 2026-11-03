# Usage guide: finding disk galaxies through to lens fitting

*中文版见 [usage_guide_zh.md](usage_guide_zh.md).*

This guide walks the full chain implemented in `galaxylens`, from assembling a disk-galaxy lens sample to the modelling fit. The repository splits into four sub-packages by *capability* (not by galaxy type or paper):

```
sample_library   ->  data_pipeline   ->  mock_pipeline   ->  lensfit
(find disks)         (download+reduce)  (simulate+truth)    (fit models)
```

- `sample_library` — aggregate + dedup + classify lens candidates from multiple survey catalogues.
- `data_pipeline` — download (external) and reduce HST imaging: cutout, noise, PSF, starmask, inpaint.
- `mock_pipeline` — simulate disk-lens systems (TNG + SKIRT) for ground-truth comparison.
- `lensfit` — fitting engine: pluggable model components + config assembly, exposed through the `api` facade (`gl.fit` / `gl.fit_positions` / `gl.fit_mein`).

## Environment variables

The packages are decoupled from any hard-coded data root via environment variables resolved in `galaxylens.config.paths`:

| Variable | What it points to | Used by |
|---|---|---|
| `GALAXYLENS_DATA_ROOT` | reduced/processed data root | `data_pipeline` |
| `GALAXYLENS_SAMPLE_LIB` | sample-library root (01_raw .. 05_exports) | `sample_library` |
| `GALAXYLENS_CONFIG_DIR` | per-lens YAML configs | `config` |

Set them in your shell profile before running anything.

---

## Step 1 — find disk galaxies: `sample_library`

Aggregates lens catalogues from several surveys, deduplicates them, classifies each by type (disk / elliptical / ...) and enriches with physical parameters.

```bash
# 1a. download survey catalogues (cowls / glue / lenscat / manga / vizier / simard)
python -m galaxylens.sample_library.download.cowls
python -m galaxylens.sample_library.download.lenscat
# ... each writes to 01_raw/.

# 1b. run the B -> C -> D -> E pipeline
python -m galaxylens.sample_library.pipeline              # no network
python -m galaxylens.sample_library.pipeline --network    # enable SDSS/VizieR queries
python -m galaxylens.sample_library.pipeline --from phase_c   # resume from a checkpoint

# 1c. add your own candidates
python -m galaxylens.sample_library.pipeline --add mine.csv --source paper
```

Phases: **B** standardize + three-level dedup -> **C** Tier classification -> **D** enrich (physical params) -> **E** export modelling subsets. Outputs land in `05_exports/` as `modeling_input` CSVs (full / sample / elliptical / tier0 / tier01). The disk-lens selection uses the disk-dominance criteria worked out for the B/D series.

## Step 2 — download + reduce: `data_pipeline`

Downloading the raw HST frames from MAST is *outside* the package (astroquery or manual); the pipeline's `file_discovery` locates the already-downloaded FLC files under the raw root.

```bash
# single lens
python -m galaxylens.data_pipeline.imaging.preprocess_master \
    --lens J2141-0001 --version-tag v2_flc_rescaled_tinytim_inpaint

# whole sample family (SWELLS -> disk/, SLACS -> ellip/)
python -m galaxylens.data_pipeline.imaging.preprocess_master \
    --sample SWELLS --version-tag v2_flc_whtonly

# single band / dry run
python -m galaxylens.data_pipeline.imaging.preprocess_master \
    --lens J1630+4520 --bands F814W --dry-run
```

The 11-stage flow: `file_discovery -> loaders -> cutout -> centroid -> background -> cosmic_ray -> noise -> psf -> starmask -> alignment -> output`. Noise is generated in three flavours (`whtonly` / `rescaled` / `bugnoise`) via `--noise-method`. Outputs (sci / noise / psf / starmask FITS + diagnostics + quality) land under the `GALAXYLENS_DATA_ROOT` version tag.

## Step 3 — fit: `gl.fit` facade + pointwise entry points

The fitting engine is pluggable model components + a config-driven assembler,
exposed through the `galaxylens.api` facade. Two fitting modes:

```python
import galaxylens as gl

# 3a. pixel-level joint fit (B-series arcs): one recipe, one call
cfg = gl.load_config("configs/fit/j2141_barnabe.yaml")
result = gl.fit(cfg)                          # assemble -> Nautilus -> post-process

print(result.beta_z)                          # weighted beta_z = 1 - ratio^2
print(result.c_minus2, result.gamma, result.v_vir, result.q_h)
print(result.c200)                            # c_200 = (2 - gamma) * c_-2
print(result.m_star)                          # M* = ml_img x CALIB x L_2D_REL
print(result.f_dm(2.2 * cfg.pipeline.geometry.rd_arcsec))

# one-call post-fit analysis: posterior summary + derived + prior-edge check + corner
gl.analyze(cfg)

# 3b. point-level quad-image fit (D-series J1331): scale-free potential
sf = gl.fit_positions(theta_obs, err_arcsec=0.05, zlens=0.113, zsrc=0.254)
print(sf.rein_arcsec, sf.mein_1e10, sf.chi2)

# 3c. point-level joint fit (D-series J1331): dynamics + Einstein mass (scipy MAP)
res = gl.fit_mein(geometry, kinematics, lum_mge, ml_array,
                  rein_arcsec=0.91, mein_msun=7.7e10)
print(res["v200"], res["c200"], res["beta_z"], res["chi2_dyn"], res["chi2_lens"])
```

`gl.fit` runs the Barnabe-style disk-lens recipe end-to-end (galaxy-subtract,
noise rescale, gNFW + MGE x M/L + Sersic source, jampy dynamics) and samples it
with Nautilus. `gl.fit_positions` / `gl.fit_mein` are the pointwise
(coordinate-level) paths for quadruple-image systems (Trick+2016 J1331), where
the lensing information is the image positions / a single Einstein mass rather
than a pixel-level arc. See `lensfit/README.md` for the full pitfall list and
`docs/refinement_plan.md` for the pluggable architecture.

## Step 4 — mock cross-check: `mock_pipeline`

For the same disk-lens systems, generate TNG/SKIRT mock observations and feed them back through `lensfit` to close the loop against known truth. Observed samples have no ground truth; mocks do. This is how `lensfit` generality is validated. See `docs/mock_pipeline.md`.

The mock-closure chain is pre-wired as a `Pipeline` (`gl.mock`); each stage is
also callable on its own:

```python
import galaxylens as gl

p = gl.mock("j2141", data_root="/data/tng", outdir="mock_out",
            star_h5="...stars.hdf5", dm_h5="...dm.hdf5", gas_h5="...gas.hdf5",
            zlens=0.138, zsrc=1.0, inclination=78.5, mock_cfg_path="configs/fit/j2141_mock.yaml")

p.run()                              # full auto: select -> truth -> render -> arc -> fit -> closure
p.run(only=["truth", "closure"])     # partial: re-run just these two stages
```

The heavy stages (`truth` / `render` / `arc` / `fit` / `closure`) need TNG
particle HDF5 + the SKIRT binary and fail with a clear message when their
`ctx` inputs are missing. `gl.Pipeline` / `gl.Stage` (in `api.pipeline`) are
the generic orchestration layer behind this; any stage can be run by hand by
calling the underlying module directly.

## End-to-end in one line

find (sample_library) -> download (astroquery/MAST) -> reduce (data_pipeline/imaging) -> fit (gl.fit) -> analyze (gl.analyze); mock loop (gl.mock).
