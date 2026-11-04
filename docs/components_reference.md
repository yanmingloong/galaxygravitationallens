# Component and parameter reference

*中文版见 [components_reference_zh.md](components_reference_zh.md).*

This is the authoritative index of every pluggable model component and every
configurable parameter in the `lensfit` fitting engine. Use it as a lookup when
writing a new fit recipe (a `configs/fit/*.yaml` file). The engine is a
**component library + config assembly**, not a hard-coded pipeline: one YAML
combines a dark-matter model, a stellar-mass model, a source model, a dynamics
engine and a lensing constraint into a full joint fit.

Only the J2141 (disk, extended arc) and J1331 (disk, quadruple image) recipes
are end-to-end validated so far; other targets need their data + a new YAML (no
code change).

## 1. Pluggable components (the `_COMPONENT_REGISTRY`)

| Part | Component (`components.<key>`) | Free parameters | Validated by |
|---|---|---|---|
| Dark matter (`dark`) | `gnfw_virial` | v_vir, c_-2, gamma, q_h | B series (J2141) |
| | `nfw` | v200, c200 | D series (J1331) |
| Stellar mass (`stellar`) | `mge_const_ml` | 1 (constant M/L) | B series |
| | `mge_radial_ml` | per-Gaussian radial M/L | D series (J1331) |
| Source (`source`) | `single_sersic` | 7 (I / Re / n / ell / centre) | B7 |
| | `pixelized` | 1 reg coefficient + mesh pixels | B8/B9 |
| Dynamics (`dynamics`) | `jampy` | ratio or beta_z | B series |
| | `jampy_kdc` | counter-rotating core + PSF | D series (J1331) |
| Lensing constraint (`lens_constraint`) | `imaging` (pixel-level arc) | — | B series |
| | `mein` (single Einstein mass) | — | D5 |
| | quadruple-image positions (via `gl.fit_positions`) | — | D1 |

Adding a component = one class implementing the `MassComponent` /
`SourceModel` / `DynamicsEngine` interface + one registry entry in
`pipeline/assemble.py`. No existing component needs to change.

## 2. Stellar-template calibration (`stellar_template`)

Two flavours, selected by `type`:

| `type` | Builder | Inputs | Used by |
|---|---|---|---|
| (default, F814W) | `stellar_mge_peak` | n_bulge, re_bulge, q_bulge, rd_disk, q_disk, disk_over_bulge, psf_sigma_arcsec, bulge_i0 | B7 / F814W disk |
| `ellip` (elliptical) | `stellar_mge_ellip` | n, re_arcsec, q, psf_sigma_arcsec, bulge_i0 | J1630 elliptical |
| `mge_csv` (K'-band) | `stellar_mge_kband` | mge_params_path, pixel_scale, mstar_ref, ml_ref | B9 / K'-band |

The K'-band builder is two-sided: `StellarTemplate.calibration` is the dynamics
surface-density bridge (`surf_s = peak * calib * ml_img` in Msun/pc^2) and
`calibration_img` is the Sigma_crit-normalised imaging bridge
(`g.intensity = peak * calib_img`). The F814W builder shares one `calibration`
for both sides.

## 3. Galaxy subtraction and noise (`data.subtract_method`)

| Method | Recipe | Noise |
|---|---|---|
| `sersic` (default) | free Sersic bulge + disk subtraction via `al.Tracer` (B4) | read from `noise_map_path` |
| `mge_agn` | literature MGE + PSF-shaped AGN point source, joint least-squares (B9); needs `mge_params_path`/`badmask_path`/`psf_sigma_px`/`position_angle_deg` | `estimate_sky_noise` (sky RMS, bypasses the KAI drizzle weight-map pitfall); no separate noise FITS |

## 4. Configurable parameters (YAML blocks)

### geometry
`zlens`, `zsrc`, `inclination_deg`, `re_arcsec`, `rd_arcsec`

### kinematics
`v_rms` (list of `[x_arcsec, vrms, err]`), `vcirc`, `vcirc_err`; elliptical-lens
single aperture dispersion: `aperture_radius`, `aperture_vrms`,
`aperture_vrms_err`, `kin_psf_sigma` (a single sigma_ap; `v_rms` may be omitted).

### priors
- `ml_img`: LogUniform bounds `[low, high]` (default)
- `ml_img_spec`: Gaussian M/L prior `{kind, mean, sigma, lower, upper}` (B9
  K'-band N(0.8, 0.3)); overrides `ml_img`
- `v_vir`, `gamma`, `q_h`, `ratio`: Uniform bounds `[low, high]`
- `c_2`: c-M relation prior `{kind, mean, sigma, lower, upper}`
- `v200`, `c200`, `beta_z`: D-series (spherical NFW / tangential anisotropy)

### source_priors (Sersic source)
`centre_sigma`, `ell_sigma`, `ell_bound`, `intensity_bounds`,
`effective_radius_bounds`, `sersic_index_bounds`

### source_pixelization (pixelized source)
`mesh_shape` `[nx, ny]`, `reg` (`constant`), `coeff_bounds` `[low, high]`

### stellar_template
See section 2.

### data
`data_path`, `noise_map_path`, `psf_path`, `starmask_path`, `pixel_scale`,
`mask_radius`, `oversample`, `subtract_method`, `mge_params_path`,
`badmask_path`, `psf_sigma_px`, `position_angle_deg`

### search
`backend` (`local` | `slurm`), `name`, `n_live`, `number_of_cores`,
`output_root`; slurm-only: `partition`, `account`, `conda_env_path`, `workdir`,
`cpus_per_task`, `mem`, `time_hours`

### switches
`noise_rescale`, `subtract_stellar`

## 5. Honest boundary: what is and is not yet validated

- End-to-end validated recipes: J2141 (Barnabe 2012, `configs/fit/j2141_barnabe.yaml`
  and `j2141_b9_kband_pixelized.yaml`), J1331 (Trick 2016,
  `configs/fit/j1331_trick.yaml`), and J1630+4520 (SLACS elliptical,
  `configs/fit/j1630_ellip.yaml`, v2 as authoritative).
- The components are target-agnostic: a new disk lens needs its FITS + v_rms +
  MGE and a new YAML (no code change). An elliptical lens reuses `jampy` +
  `gnfw_virial` + `mge_const_ml` + `imaging` (jampy is natively an elliptical-
  galaxy method); the single-aperture velocity-dispersion entry point
  (`KinData.aperture_*` + `JampyEngine` sigma_ap) was validated on J1630, but
  a single aperture dispersion cannot break the mass-anisotropy degeneracy, so
  the elliptical beta_z / M* are prior-limited.

## 6. Orchestration layer (`api.pipeline` + `api.analysis` + `mock_pipeline.orchestrate`)

A thin auto-mode layer over the pluggable components; the components stay
individually callable (manual mode).

- `Stage` — one named component call, with optional `depends` (other stage names).
- `Pipeline` — ordered stages sharing a `ctx` dict; `run(start/stop/only)` for
  full-auto / partial / range runs.
- `gl.analyze(cfg)` — one-call post-fit summary: six mass parameters, derived
  quantities (beta_z / M* / f_dm at median / theta_E / gamma_tot), a prior-edge
  check, and the corner plot. `f_dm` is evaluated at the median parameters (the
  `Result.f_dm` per-sample loop is O(N) and slow).
- `gl.identify(name)` — lens identification: reads the sample catalog, infers
  lens type (disk/elliptical) + constraint (arc/quad), returns a `LensPlan`
  (type + modeling modules + `template_yaml`). `write_recipe(plan, params)`
  emits a runnable YAML (dot-nested key overrides).
- `gl.mock(target, ...)` — the pre-built mock-closure pipeline select -> pick ->
  truth -> render -> make_arc -> mock_fit -> closure. The heavy stages need TNG
  particle HDF5 + the SKIRT binary. truth / render / arc are validated
  end-to-end on real TNG particles + SKIRT v9.0 (subhalo 348489: truth
  reproduces the MF ground truth exactly, arc M* self-check 1.000); fit +
  closure still need a real Nautilus run. The arc front end
  `mock_pipeline.lensing.mge.fit_light_mge` fits the light MGE from the mock
  image (mgefit); a runnable mock recipe is `configs/fit/j2141_mock.yaml`.
