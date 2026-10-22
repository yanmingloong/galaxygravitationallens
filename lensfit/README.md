# lensfit — multi-method lens fitting

*中文版见 [README_zh.md](README_zh.md).*

`lensfit` is a **method collection** for fitting galaxy-scale gravitational
lenses of any morphology (disks, ellipticals, ...). It assembles third-party
engines — PyAutoLens, `mgefit`, `jampy`, `agama` — into a reusable, parameterized
chain, and codifies the pitfalls learned while reproducing the literature
(the B / D / R / e series).

It is **not** a re-implementation of those engines: it contributes the usage
conventions, the pitfall guards, and the parameterization on top of them.

## Design principles

1. **Multi-method**: stellar-mass models (MGE today, multi-component later) and
   dynamics engines (`jampy` today, `agama` later) are interchangeable behind
   the same interfaces. Nothing is hard-wired to one paper or galaxy type.
2. **Wrappers, not re-implementations**: every method that is someone else's
   algorithm is a thin, documented wrapper.
3. **Attribution is mandatory**: every module docstring names the paper and/or
   code it borrows from. See the reference table below.

## Submodules

| Submodule | Purpose | Engines | Status |
|---|---|---|---|
| `psf` | empirical PSF from field stars | photutils | field-star detection done; Moffat fit TODO |
| `mge` | MGE fit / I/O / deprojection | mgefit | done (B2/B3) |
| `lens` | lens model builders + galaxy-subtracted recipes | autolens/autofit | done (B4) |
| `dynamics` | Jeans / dynamics engines | jampy (agama later) | TODO (B5) |
| `fit` | samplers / search wrappers | Nautilus | TODO |
| `validation` | mock-fit ground-truth cross-checks | — | TODO |
| `plotting` | data/model preview | autolens | `preview_imaging` |

## References

Every module carries these in its docstring. Canonical sources:

- Cappellari 2002, MNRAS 333, 400 — MGE decomposition & deprojection (`mge`)
- Cappellari 2020, MNRAS 494, 4819 — `jampy` (`dynamics`, future)
- Barnabè, Dutton, Marshall et al. 2012, MNRAS 423, 1073 — SWELLS IV disk-lens
  galaxy-subtracted modelling (`lens`)
- Wyithe, Turner & Spergel 2001, ApJ 555, 504 — gNFW (`lens.models`)
- Ludlow et al. 2016, MNRAS 460, 1214 — c–M relation (`lens.models`)
- Vasiliev 2019, MNRAS 482, 1525 — AGAMA (`dynamics`, future)
- Stetson 1987, PASP 99, 191 — DAOStarFinder (`psf`)

## Usage

```python
from pathlib import Path
from galaxylens.lensfit import mge, lens, psf

# --- MGE: fit a Sersic profile (PSF-limited sigma_min) ---
import numpy as np
sigma_min = 0.0637          # PSF sigma in arcsec
r = np.geomspace(0.05, 6 * 6.55, 200)
prof = np.exp(-2.095 * ((r / 6.55) ** 0.826 - 1))   # Sersic n=1.21, Re=0.26"
L, sigma = mge.fit_mge_1d(r, prof, ngauss=6, psf_sigma=sigma_min)

# --- MGE: load + deproject (bulge/disk labels on each component) ---
mge_list = mge.load_mge(Path("mge_params.csv"))
q_intr = mge.deproject(np.array([g.q_obs for g in mge_list]), incl_deg=78.0)

# --- lens: build the Barnabe disk-lens model (gNFW + MGE x M/L + source) ---
model = lens.build_lens_model(zlens=0.138, zsrc=0.7127, mge=mge_list)
print(model.prior_count)   # 12

# --- lens: galaxy-subtracted Step-1 (scipy) + rescale ---
light_galaxy, chi2 = lens.subtract_stellar_light(dataset, zlens=0.138)
f = lens.noise_rescale_factor(chi2, n_unmasked=31389)
```

## Key pitfalls codified here

- **MGE sigma_min >= PSF sigma** (`mge.fit`): without `rbounds`, `fit_1d`
  over-decomposes a Sersic flat core into a `sigma < PSF` spike that injects an
  unphysically dense core downstream. See `mge.fit.fit_mge_1d`.
- **Subtract light, keep mass** (`lens.models`): galaxy-subtraction removes the
  stellar LIGHT, but the stellar MASS still acts as a lens. Dropping it leaves
  the halo holding only ~14% of the Einstein mass.
- **Rescale the noise** (`lens.rescale`): the subtraction residual must be
  absorbed by scaling the noise map by `sqrt(reduced chi2)`, or the arc fit's
  chi2/px ~ 30 collapses the posterior and the sampler grinds for days.
- **Field-star peak** (`psf.stars`): locate the galaxy centre AFTER masking bad
  pixels (a cosmic ray can outshine the bulge); do not smooth (drags the peak).

## TODO

- `psf`: Moffat/PSF fitting on the field-star candidates (B1 second step).
- `dynamics`: `jampy` wrapper + hand-written cylindrical JAM cross-check (B5).
- `lens`: multi-component stellar-mass model (Sersic bulge+disk) as an
  alternative to MGE.
