# Changelog

*中文版见 [CHANGELOG_zh.md](CHANGELOG_zh.md).*

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **Componentized `lensfit` core layer + facade `api` layer** (2026-10-21, refinement stages 1-4): rebuilt the fitting engine as pluggable model components + a config-driven assembler. `lensfit.structures` (Geometry/KinData/Calibration/PriorSet/PipelineConfig); `lensfit.models` (MassComponent/SourceModel/DynamicsEngine ABCs; `dm.gnfw_virial` Wyithe-2001 gNFW in v_vir/c_-2/gamma/q_h + `gnfw_virial_profile`, `stellar.mge_const_ml`, `source.single_sersic`); `lensfit.dynamics.jampy` (Cappellari 2020, anisotropy as explicit `ratio = sqrt(1-beta_z)`); `lensfit.fit.joint` (JointAnalysis, log L = -0.5*(chi2_img/nu_img + chi2_dyn/nu_dyn)); `lensfit.pipeline.assemble` (component registry, zero physics). New facade `galaxylens.api` (config_loader / dataset / search(Nautilus) / result / derived / plotting.corner / facade); `result.py` reads beta_z / c_-2 / gamma / v_vir / q_h / M* / f_dm as weighted medians directly off samples.csv. First recipe `configs/fit/j2141_barnabe.yaml`; `gl.fit()` reproduces B7 on the Loong cluster (job 128011: log Z = -8.04, all parameters within +/-2.5% of B7). Updated `lensfit.lens.models` from `gNFWMCRLudlow` (B4) to `GNFWVirialVelocity`. Regressions: verify_lensfit_joint_b7 / verify_api / verify_lensfit_lens / verify_fit_smoke.
- **Second component set + pointwise (coordinate-level) fitting** (2026-10-21, stage 5): `lensfit.models.dm.nfw` (spherical v200+c200), `lensfit.models.stellar.mge_radial_ml` (radial M/L, luminous tracer != mass), `lensfit.dynamics.jampy_kdc` (kappa counter-rotating core + direct beta_z + PSF convolution), `lensfit.lens.scale_free` (quad-image pointwise re-projection). Abstracted `lensfit.fit.constraints.LensingConstraint` (ImagingConstraint pixel-level / MeinConstraint single-value Einstein mass / scale-free quad-image) so the pipeline supports both pixel-level (B-series arcs) and coordinate-level (D-series quads) fitting; added `gl.fit_positions` / `gl.fit_mein`. Regressions: verify_lensfit_d1 / verify_lensfit_d5 / verify_facade_point.
- **Cross-platform + slurm backend + CI** (2026-10-21/22, stage 6): `api.search` slurm backend (`slurm_script` + `submit_slurm`, template matches the verified `examples/j2141_b7.slurm`), `SearchSpec.workdir/mem`, `api.runner` CLI (forced `backend=local`), `.github/workflows/ci.yml` (Windows+Linux core/pure-python, Linux-only full stack), portable test paths, README cross-platform declarations. Disabled autofit's built-in anesthetic corner plot on headless runs (`_disable_autofit_corner_plot`) -- the repo's own weighted corner reads samples.csv with the Agg backend.

- `mock_pipeline` J1331 support + KDC/roll/single-scale (2026-10-20): `truth.major_axis_angle` (eigen-decomposition roll, removes the +/-90 deg ambiguity of the 0.5*arctan2 formula), `truth.counter_rotation_angle` (core-vs-outer angular-momentum KDC test, ~180 deg hard criterion; a mild reversed rotation in a hot bulge is NOT a KDC), `skirt.single_scale_normalize` + `skirt.make_mock_dataset` (single photometric scale preserving the SED colour ratio for multi-band joint fitting + data/noise/psf fitting-ready FITS; noise map generated AFTER crop/pad so zero-padding never contaminates the border), `lensfit.psf.moffat_mge_psf` (four-Gaussian Moffat MGE PSF, Trick+2016 Table 2), and `selection.config.J1331` (second built-in target: snap90, M* in [1e11, 3e11], sigma >= 190, incl 70 deg, F435W/F814W).
- Data-fied the skirt observation config (2026-10-17): new `BandConfig`/`MockObsConfig` (band+instrument name+pixel scale+noise+peak / geometry); `write_ski` now takes a `BandConfig`+`MockObsConfig` (no more hard-coded BANDS dict — same filter on a different instrument, e.g. WFPC2 F814W, is expressible); `make_fake_hst` takes a `BandConfig` (dropped F814W-specific PX_HST/SIGMA/J_PEAK). `TargetConfig` gains an `obs` field. `snap_to_a` now raises on unknown snapshots instead of silently returning a=1.0.
- Renewed `notebooks/mock_pipeline_demo.py` into a full tutorial (8 sections; offline sections fully runnable, heavy sections gated on TNG key / particle data / SKIRT binary).
- Made `read_api_key` silent on import; the "key not found" hint now prints on-demand in the download/request paths.
- Added `incl` field to `TargetConfig` + `read_api_key()` (2026-10-17): disk inclination moved into target config (J2141=78.5 deg, J1331=70 deg); `skirt.render.write_ski` reads it from target.incl. TNG API key now read from env `TNG_API_KEY` or the `~/.tng_api_key` file (no key hard-coded), with a one-time hint pointing to docs/mock_pipeline.md; `get()`/`download_cutout()` fail fast without a key.
- `mock_pipeline.truth` module (2026-10-14): ground-truth computation — kinematics (circularity → f_disk/B_D/disk axis/thickness), measures (half-mass radius / projected-ellipticity covariance convention / intrinsic axis ratios / extinction), cosmology (Planck 2015 distances + SIS theta_E + cosmic time + TNG unit conversions), with regression tests (348489 f_disk=0.643, B/D=0.55 reproduced); fixed tng_to_skirt H0_GYR typo (0.06774→0.06928).
- `mock_pipeline.selection` module (2026-10-14): TNG disk-lens analog selection — scan / filter (L1 criteria as data) / merger (L2 mpb.hdf5 main-branch mass-jump) / download (3-field & all-field merged) / verify (L3 disk_kinematics) / catalog (L3→catalog auto-assembly) / run (--target end-to-end). Selection criteria live in a SelectionCriteria dataclass; switching target needs no code change.

- `mock_pipeline.skirt` module (2026-10-14): SKIRT radiative transfer — convert (TNG particles → .dat, inertia-tensor alignment, no legacy L_hat) / render (.ski generation + skirt run, SKIRT_BIN env var) / imaging (fake HST + extinction map). Fixed write_dust ``a``-parameter bug and vectorized cosmic_time.

## [0.1.0] - 2026-09-24

### Added

- Repository skeleton: package layout, module directories, pyproject, README, LICENSE, CHANGELOG, .gitignore.
- Architecture decision: single monorepo, src layout, split by capability (not paper/galaxy type), data-production vs data-consumption separation, mock fitting reuses `lensfit` + ground-truth validation.
- `data_pipeline.imaging` module migrated (21 files; psf split; paths/SExtractor/TinyTim env-decoupled).
- `config` module (YAML + dataclass; 16 lens YAMLs).
- `sample_library` module migrated (build pipeline + 5 product CSVs).
