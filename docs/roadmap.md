# Roadmap / Extension Items

Tracked planned extensions and analysis-flow gaps, so they are not lost. Items
are grouped into "planned" and "flow mismatch / needs fixing".

## Mock fitting (mock closure)

### Disk mock (MF series) — complete
Truth (particles/fdm/anisotropy/halo), validation (closure etc.), arc synthesis
(`mock_pipeline/lensing`), and dynamics `beta_z`-direct are all in place; 6/6
candidates closed (3 improve / 3 overshoot under M/L + c-M priors).

### Next disk-mock iteration (B9) — deferred
Re-run the MF mock with the B9 improvements. Fork to resolve first:
- **pixelized source** (`source/pixelized.py`) — directly applicable; drops the
  source-mass degeneracy (MF source intensity 0.27 vs truth 88).
- **K'-band subtraction** — needs a K'-band mock image (SKIRT band not yet
  generated; MF only has F814W / F450W proxy).

### Elliptical mock (new series) — not started
- `truth/aperture.py`: particle -> aperture velocity dispersion `sigma_ap`
  (ADDED 2026-11-02). Spec decisions (default in brackets): mass-weighted [yes];
  dispersion sigma = sqrt(<v_los^2> - <v_los>^2), not v_rms [yes]; circular
  aperture, note jampy's square `pixsize=2*R_ap` approximation [yes]; no PSF
  convolution in v1 [yes]. Cross-term sign vs jampy to be checked when the
  elliptical mock runs.
- `selection`: no elliptical target registered (`BUILTIN_TARGETS` = j2141/j1331
  disk only).
- **J1630 analog**: SLACS slow rotator, sigma_ap = 267 km/s (Re aperture),
  theta_E = 1.791" / gamma = 1.924 (Dinos reference). Match on morphology
  (Sersic n~4, low f_disk / slow rotator), sigma_e ~ 250-270 km/s (M(Re) ~ 1.3e11),
  M* ~ 1e11 Msun.

## Analysis-flow gaps / mismatches

1. **mock output -> api/dataset convention** — VERIFIED (2026-11-02,
   `tests/test_mock_api_contract.py`): mock data/noise/psf FITS load through
   `api.dataset.load_dataset` unchanged. Elliptical path reuses the same
   morphology-agnostic loader.
2. **closure calibration bridge** — DOCUMENTED (2026-11-02): `sample_f_dm`
   docstring now spells out the two conventions (F814W calib=1 + bridge in
   lg_ml; K'-band calib=dyn-bridge + lg_ml=log10 M/L) and forbids mixing.
3. **elliptical recipe not end-to-end verified** — DEFERRED (needs a full
   Nautilus run on the cluster once the elliptical mock exists).
4. **multiprocessing pickle** — FIXED (2026-11-02, `tests/test_joint_pickle.py`):
   `_JointAnalysis` now carries a stable module path + namespace registration;
   picklable on Linux (fork). Spawn platforms still single-core.

## Data pipeline pending

- `data_pipeline/spectra`: DEIMOS reduction + pPXF extraction not yet in place
  (reduction + pPXF scripts and E-MILES templates are user-provided).
- `data_pipeline/other_obs`: empty.
