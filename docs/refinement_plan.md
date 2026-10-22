# galaxylens Refinement & Release Plan (Blueprint)

> Finalized 2026-10-20. This document is the landing blueprint for upgrading
> galaxylens from "a B-series script port" into "pluggable components + config
> assembly + facade API". Development proceeds stage by stage according to this
> document, each stage paired with regression tests.

*中文版见 [refinement_plan_zh.md](refinement_plan_zh.md).*

## 1. Goal & positioning

Turn galaxylens into an autolens-like package: after `pip install galaxylens`,
users call `import galaxylens as gl` to run fits, output results and plot.
Core requirements:

1. **Pluggable components**: dark-matter models (gNFW/NFW/SIE...), stellar-mass
   models (MGE constant M/L / radial M/L / multi-component...), dynamics engines
   (jampy/AGAMA/JAX Jeans...), and source models (single Sersic / multi Sersic)
   are each independent and freely combinable.
2. **Config-driven**: changing galaxy or method = changing YAML, not code.
3. **Facade API**: `gl.fit(cfg)` runs everything in one line; `result.beta_z`
   reads physical quantities directly; `result.plot_*()` plots.
4. **Cross-platform**: Windows / WSL / Linux (CPU).

**Two iron rules**:

- **Design the interface once (pluggable), fill components on demand (each
  component backed by a real fit)** — do not build the "all models x all
  engines" matrix up front, to avoid over-abstraction; the D-series refinement
  validates the second component.
- **The core layer is the engine, the facade layer is the handle** — all
  physics lives in the components, the assembler holds zero physics.

## 2. Current refinement status

lensfit currently has 17 .py files (early B1-B4 conventions):

| Status | Module | Notes |
|---|---|---|
| Migrated | `psf/stars.py` | B1 field-star Moffat PSF |
| Migrated | `mge/fit.py` | B2 MGE fit (rbounds anti-spurious-core pitfall fixed) |
| Migrated | `mge/deproject.py` | B3 deprojection |
| Migrated | `mge/io.py` | MGE I/O |
| Migrated | `lens/subtract.py` | B4 Step1 galaxy-subtract |
| Migrated | `lens/rescale.py` | B4 noise rescale (f = sqrt(reduced chi2)) |
| **Outdated** | `lens/models.py` | uses gNFWMCRLudlow (early B4); needs update to gNFWVirialVelocity (B6/B7) |
| Migrated | `plotting/preview_imaging.py` | imaging preview |
| **Empty** | `dynamics/` | B5 Jeans dynamics not ported at all |
| **Empty** | `fit/` | B6/B7 joint fit not ported at all |
| **Empty** | `validation/` | regression + degeneracy scan |

**To update to the final convention**: `lens/models.py`'s gNFWMCRLudlow ->
gNFWVirialVelocity (v_vir parameterization + free c_-2 + q_h ellipticity +
c-M prior).

## 2.5 Mock pipeline refinement (upstream image generation, added 2026-10-20)

`mock_pipeline` (`truth/` `selection/` `skirt/`) is **upstream**: it generates
truth-bearing mock FITS that feed `api/dataset` -> lensfit fit closure. It is
orthogonal to the lensfit refinement, but one **interface contract** must be
fixed together:

**Interface contract (mock <-> fit)**: mock output = `data/noise/psf` three
FITS, units counts/s, pixel scale identical to the real data (e.g. J1331
0.0455"/px), so that `api/dataset.py` loads mock and real data with the **same
code**. The noise map is a constant background RMS (median=SIGMA, std~0); do not
zero-fill the border when cropping/padding.

**This round's 4 refinement items (2026-10-20 J1331 work frozen)**:

| # | Function / capability | Target module | Anchor |
|---|---|---|---|
| 1 | `major_axis_angle()`: eigen-decomposition disk long axis, removes the +/-90 deg ambiguity | `mock_pipeline/truth/measures.py` | 342537 = -21.32 deg |
| 2 | `counter_rotation_angle()`: core-vs-outer angular-momentum KDC test (hard ~180 deg criterion) | `mock_pipeline/truth/kinematics.py` | 342537 = 115.7 deg |
| 3 | Single-scale normalization (colour-ratio conserving) + data/noise/psf separation | `mock_pipeline/skirt/imaging.py` | F814W anchor 0.317 |
| 4 | 4-Gaussian Moffat MGE PSF (paper Table 2 normpsf/sigmapsf) | `lensfit/psf/` (**shared**: mock convolution + fit deconvolution both use it) | delta=0.038/0.085/0.169/0.487" |

**Register J1331 as the second target**: `selection/config.py` adds
`J1331 = TargetConfig(...)` (snap90, M* in [1e11, 3e11], sigma >= 190, incl=70 deg,
obs=J1331_OBS, merger criterion **inverted** to select "has merger"), registered
into `BUILTIN_TARGETS` — validating the "change target, zero code change"
interface generality (J2141 is the first, J1331 the second).

**Known debt (verified)**: `selection/merger.py` is **already synced** to the
mpb.hdf5 main-branch mass jump (major>+30%/minor>+10%, with branch-switch
artifact detection), no fix needed. The earlier "merger.py not synced" note in
the migration skill was stale and corrected with this round.

**D-series (J1331) physical dimension gap**: roadmap stage 5 only listed the
J1331 components "`nfw` + `mge_radial_ml`", missing J1331's signature physics
that distinguishes it from J2141 — **counter-rotating core KDC + tangential
anisotropy beta_z=-0.4**. The mock-side KDC detection (item 2) and the fit-side
beta_z sign are two sides of the same thing; the refinement should map them
(KDC~180 deg <-> tangential beta_z<0), otherwise the D-series fit components
would "reproduce J1331's mass profile yet miss its dynamics signature".

## 3. Two-layer architecture

```
galaxylens/
├── mock_pipeline/           # upstream: mock image generation (truth/selection/skirt, feeds api/dataset)
├── lensfit/                 # core layer: physics engine (no user interaction)
│   ├── models/              #   pluggable model components
│   │   ├── dm/              #     dark matter: gnfw_virial, gnfw_ludlow, nfw...
│   │   ├── stellar/         #     stellar mass: mge_const_ml, mge_radial_ml...
│   │   └── source/          #     source: single_sersic, multi_sersic...
│   ├── dynamics/            #   dynamics engines: jampy, agama, jax_jeans...
│   ├── fit/                 #   assembler: JointAnalysis
│   ├── pipeline/            #   config assembly: assemble (read YAML -> combine)
│   ├── mge/ psf/ lens/      #   existing (B1-B4)
│   └── validation/          #   regression + degeneracy scan
├── api/                     # facade layer: user-facing API
│   ├── dataset.py           #   load_dataset (FITS -> Imaging + mask + PSF + rescale)
│   ├── search.py            #   Nautilus wrapper (local / slurm dual backend)
│   ├── result.py            #   Result object (reads physical quantities directly, no instance digging)
│   ├── derived.py           #   derived: f_dm, M_2D(<R_E), M*
│   ├── config_loader.py     #   YAML -> PipelineConfig
│   └── plotting/            #   corner (weighted), fit, residual, dynamics comparison
└── __init__.py              #   facade entry: gl.fit / gl.load_config / gl.load_dataset
```

- **Core layer**: components + engines + assembly + Analysis, pure physics,
  unit-testable, no "how the user calls it".
- **Facade layer**: wraps the core layer into a fool-proof API.
- **result.py + derived.py are the value-add over autolens**: autolens makes
  users dig `instance.galaxies.lens.dark.v_vir`, compute beta_z=1-ratio^2 and
  f_dm by hand; galaxylens gives `beta_z` / `m_star` / `f_dm()` directly, freezing
  the "parameter -> physical quantity" conversions the B series worked out
  (CALIB bridge, weighted percentile, f_dm inverse calculation).

## 4. Core layer interface design

### 4.1 Basic data structures (the common language of all interfaces)

```python
@dataclass(frozen=True)
class Geometry:                     # fixed observables, not free parameters
    zlens: float
    zsrc: float
    inclination_deg: float          # i=78.5(J2141) / 70(J1331)
    da_mpc: float
    pc_per_arcsec: float
    re_arcsec: float | None = None  # for f_dm/M_2D inverse aperture
    rd_arcsec: float | None = None

@dataclass(frozen=True)
class KinData:                      # kinematic observables
    xbin: np.ndarray
    ybin: np.ndarray
    vrms: np.ndarray
    vrms_err: np.ndarray
    vcirc: float | None = None      # optional Halpha V_circ(2.2Rd)
    vcirc_err: float | None = None

MGE = tuple[np.ndarray, np.ndarray, np.ndarray]   # (surf, sigma, q_obs)

@dataclass(frozen=True)
class Calibration:                  # freezes B6's CALIB pitfall: ml_dyn = ml_img * calib
    calib: float
```

### 4.2 MassComponent (DM + stellar unified)

```python
class MassComponent(ABC):
    name: ClassVar[str]
    def attach(self, lens: af.Model, priors: PriorSet, geom: Geometry) -> None: ...
    def dyn_mge(self, instance, geom: Geometry, calib: Calibration) -> MGE: ...
    @property
    def n_free_params(self) -> int: ...
```

- DM and stellar are unified because both are deflection in imaging and MGE
  surface density in dynamics; the assembler just "iterates components to build
  surf_pot", changing model = changing component, zero assembler change.
- `dyn_mge` reads from instance (not model), because post-fit parameter values
  live in instance, and dynamics recomputes MGE at every likelihood call.

### 4.3 SourceModel / DynamicsEngine

```python
class SourceModel(ABC):             # source is light only, not mass
    name: ClassVar[str]
    def attach(self, source: af.Model, priors: PriorSet) -> None: ...
    @property
    def n_free_params(self) -> int: ...

class DynamicsEngine(ABC):
    name: ClassVar[str]
    def chi2(self, model_lum: MGE, model_pot: MGE,
             ratio: float, geom: Geometry, kin: KinData) -> tuple[float, dict]: ...
```

- `ratio = sqrt(1 - beta_z)` is passed as an explicit parameter to the engine,
  **no longer using B6's zero-mass Gaussian container hack**.

### 4.4 Priors and config (the assembly recipe)

```python
@dataclass
class PriorSpec:                    # c-M prior parameterization: None=Uniform, else GaussianPrior
    kind: str = "gaussian"
    mean: float = 5.9               # D&M centre
    sigma: float = 1.49             # 0.11 dex linearized
    lower: float = 3.0
    upper: float = 20.0

@dataclass
class PriorSet:                     # all components read priors from here when attaching
    ml_img: tuple[float, float]
    v_vir: tuple[float, float]
    gamma: tuple[float, float]
    q_h: tuple[float, float]
    ratio: tuple[float, float]
    c_2: PriorSpec | None = None

@dataclass
class PipelineConfig:               # one YAML = one reproduction recipe
    geometry: Geometry
    kinematics: KinData
    dark: str                       # "gnfw_virial" | "gnfw_ludlow" | "nfw" | ...
    stellar: str                    # "mge_const_ml" | "mge_radial_ml" | ...
    source: str                     # "single_sersic" | ...
    dynamics: str                   # "jampy" | "agama" | ...
    priors: PriorSet
    noise_rescale: bool = True
    subtract_stellar: bool = True
```

### 4.5 Assembler + Joint Analysis

```python
def assemble(cfg: PipelineConfig) -> tuple[af.Collection, JointAnalysis]:
    """read config -> instantiate components -> assemble Collection + JointAnalysis. Zero physics."""
    ...

class JointAnalysis(al.AnalysisImaging):
    def log_likelihood_function(self, instance):
        chi2_img = self.fit_from(instance).chi_squared
        # iterate components to build model_pot, call engine.chi2
        ...
        nu_img = self.n_unmasked - sum(c.n_free_params for c in ...)   # imaging dof
        nu_dyn = self.kinematics.n_points - sum(c.n_free_params for c in self.components)
        return -0.5 * (chi2_img / nu_img + chi2_dyn / nu_dyn)
```

`JointAnalysis` is fully generic: it does not know whether DM is gNFW or NFW,
whether stellar is single M/L or radial M/L — it just "iterates to build MGE ->
call engine". All B-series vs D-series differences collapse into a few YAML
lines of PipelineConfig.

### 4.6 Key design decisions (why)

1. **DM and stellar unified as MassComponent**: the assembler does not
   distinguish them; changing model = changing component, zero change.
2. **ratio made explicit**: it belongs only to dynamics, as an independent
   engine.chi2 parameter (value still read from the instance's anisotropy
   carrier).
3. **Calibration as its own object**: B6's CALIB pitfall (imaging/dynamics M/L
   differ) forces every stellar component to explicitly bridge in dyn_mge.
4. **c-M prior is PriorSpec config, not component hard-coding**: the same
   gnfw_virial switches between B6 (Uniform) and B7 (D&M prior) via config.
5. **Assembler zero physics**: all physics (zeta integral, peak formula, beta_z
   conversion) lives in components; assemble only "reads config -> attach ->
   assemble Collection".

## 5. Facade modules

| Module | Responsibility | autolens equivalent |
|---|---|---|
| `api/dataset.py` | FITS -> Imaging + circular/stellar mask + PSF + rescale in one go | `al.Imaging.from_fits()` |
| `api/search.py` | Nautilus wrapper (n_live/n_eff/path/local vs slurm dual backend) | `af.Nautilus()` |
| `api/result.py` | Result object: reads beta_z/c/gamma/M*/f_dm directly | `result.instance` (but needs field digging) |
| `api/derived.py` | f_dm(2.2Rd), M_2D(<R_E), M* inverse calc (B6 b6_fdm.py frozen) | none (autolens doesn't provide) |
| `api/plotting/` | corner (weighted percentile), fit, residual, dynamics v_rms comparison | `result.plot_...` |
| `api/config_loader.py` | YAML -> PipelineConfig (reuses config/ YAML+dataclass) | none |
| top-level `__init__.py` | `gl.fit / gl.load_config / gl.load_dataset` | `import autolens as al` |
| (optional) CLI | `galaxylens fit config.yaml` | autolens CLI |

## 6. Usage flow (a new galaxy)

Two segments: **data preparation (one-time)** + **config fit (YAML + gl.fit)**.

Data preparation (gl helps part of it): imaging FITS (data_pipeline) ->
`gl.load_dataset()`; stellar MGE -> `gl.fit_mge()` (rbounds pitfall frozen);
geometry into YAML; kinematic digitization (gl cannot help, manual).

```python
import galaxylens as gl

cfg = gl.load_config("configs/disk/j1331_trick.yaml")
result = gl.fit(cfg)                        # assemble -> sample -> post-process -> write out

print(result.beta_z)                        # reads physical quantities directly
print(result.c200, result.m_star, result.f_dm(2.2 * cfg.rd))
result.plot_corner("j1331_corner.png")
result.plot_fit("j1331_fit.png")
result.plot_dynamics("j1331_vrms.png")
```

**Key point — priors must go into YAML**: J1331 is tangential (beta_z=-0.4,
ratio~1.18), J2141 is radial (beta_z=+0.65, ratio~0.59), the same ratio
parameter has opposite physical ranges in the two galaxies. Whatever varies by
galaxy/method (geometry, prior ranges, c-M centre, data paths) goes into YAML;
whatever is invariant (zeta integral, peak formula, rbounds, weighted
percentile) is frozen in components.

## 7. Cross-platform strategy

**Basis**: the dependency chain is all pure Python (numpy/scipy/astropy/pandas/
matplotlib + PyAutoLens + jampy + mgefit + nautilus), CPU, runs on all three
platforms. Paths are already decoupled via 7 environment variables
(GALAXYLENS_DATA_ROOT etc.).

**Three honest boundaries (written into README)**:

1. PyAutoLens is officially mainly tested on Linux/macOS; Windows is "usable but
   not first-class support" (galaxylens inherits this support level).
2. BLAS/LAPACK numeric non-determinism ~1% (B6 CALIB 1427 vs 1157); cross-platform
   comparisons anchor on the same BLAS.
3. slurm cluster scheduling is Linux-only; GPU/JAX acceleration is Linux+NVIDIA-only.

**Landing**: GitHub Actions CI matrix (Windows + Linux dual import + single
likelihood + MGE fit smoke); README declaration section; search.py local/slurm
dual backend (YAML `backend: local` switch).

## 8. Landing roadmap (staged, each stage paired with regression tests)

**Stage 0 (upstream, mock pipeline)**: refine 4 items (`major_axis_angle` /
`counter_rotation_angle` / single-scale normalization + data/noise/psf
separation / 4-Gaussian Moffat MGE PSF) + register J1331 as the second
TargetConfig + sync merger.py to mpb.hdf5. Done, the mock side can emit
fitting-ready dual-band mocks end-to-end for the fit-closure (MF series) after
stage 3.

1. **Core layer first component combination**: `gnfw_virial` + `mge_const_ml` +
   `jampy` + `assemble` + `JointAnalysis` — regression-assert likelihood
   reproduction with B7's real numbers (CALIB=1157, beta_z=0.65, M*=1.437e11,
   V_circ=290).
2. **Facade layer**: dataset / search / result / derived / plotting /
   config_loader / top-level entry.
3. **First complete YAML**: J2141 Barnabe reproduction, `gl.fit()` full chain +
   regression.
4. **Update outdated modules**: `lens/models.py` gNFWMCRLudlow ->
   gNFWVirialVelocity; fill `dynamics/`, `fit/`.
5. **Second component (during D-series refinement)**: `nfw` + `mge_radial_ml`,
   validating interface generality.
   ✅ **Done (2026-10-21)**: refined `nfw` (spherical v200+c200) + `mge_radial_ml`
   (radial M/L, `lum_mge` luminous tracer != `dyn_mge` mass) + `jampy_kdc`
   (kappa counter-rotating core + direct beta_z + PSF) + `MeinConstraint`
   (single-value Einstein mass, fixed D5's original M_star unit bug) +
   `scale_free.py` (quad-image pointwise re-projection, D1). Also abstracted
   `LensingConstraint` (ImagingConstraint pixel-level / MeinConstraint single
   value / scale-free quad-image), making the JointAnalysis imaging term
   pluggable, so the pipeline supports both pixel-level (B-series arcs) and
   coordinate-level (D-series quads) fitting. Facade gains `gl.fit_positions` /
   `gl.fit_mein`. Regression: tests/verify_lensfit_d1.py + verify_lensfit_d5.py
   + verify_facade_point.py.
6. **Cross-platform**: CI matrix + README declaration + search dual backend.
   ✅ **Done (2026-10-21)**: search.py fully landed the slurm backend
   (`slurm_script` render + `submit_slurm` submit, template aligned with the
   verified `examples/j2141_b7.slurm`); facade `gl.fit` wired to `backend=slurm`
   (render+write+sbatch), runner forced `backend=local` (anti re-submit);
   SearchSpec gained `workdir`/`mem`; `.github/workflows/ci.yml`
   (Windows+Linux core+pure-Python, Linux-only full stack); README/README_zh
   cross-platform three boundaries; tests' hard-coded WSL paths made portable
   `Path(__file__).parents[1]`.

## 9. Validation & release conventions

- **Regression tests**: each refined module pairs with an assertion script using
  B-series real numbers to verify the wrapper reproduces the original script
  output; pass = done.
- **Citation annotation**: each module docstring annotates references
  (Cappellari 2002 MGE / Barnabe 2012 galaxy-subtract / Wyithe 2001 gNFW /
  Ludlow 2016 c-M / Dutton & Maccio 2014 / Cappellari 2020 jampy / Vasiliev 2019 AGAMA).
- **English code**: continue the existing 100% English (docstring + comments +
  string literals).
- **Release**: pyproject.toml dependency declaration + MIT + public at paper
  completion + Zenodo DOI.
