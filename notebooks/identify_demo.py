# %% [markdown]
# # Lens Identification Tutorial
#
# Identify a lens's type (disk/elliptical) and constraint (arc/quad) from its
# name or a FITS image, map it to a modeling plan, and emit a runnable recipe.
#
# This is the "identify -> model -> fit" automation chain's first link, feeding
# the `gl.fit` recipes. Entirely offline: it reads the in-repo `sample_library`
# catalog CSVs and the recipe templates under `configs/fit/` — seconds, runs out
# of the box. No heavy deps (autolens/mgefit) are loaded.
#
# Identification inputs (by priority): the survey morphology label
# (`lens_type_label`: DEV=elliptical, SER/EXP/S0=disk), Sersic index `n_sersic`,
# `B_D_ratio`, and the known-quad name list.

# %%
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))  # repo root/src

import galaxylens as gl

# %% [markdown]
# ## 1. Identify by name (catalog-driven)
#
# Three anchor lenses span the three modeling branches: disk+arc, disk+quad,
# elliptical+arc.

# %%
for name in ["J2141-0001", "J1331+3628", "J1630+4520"]:
    plan = gl.identify(name)
    print(f"{name}")
    print(f"  lens_type     = {plan.lens_type}")
    print(f"  constraint    = {plan.constraint}")
    print(f"  template_yaml = {plan.template_yaml}")
    print(f"  stellar={plan.stellar_template}  subtract={plan.subtract_method}  "
          f"dynamics={plan.dynamics}  source={plan.source_model}")
    print(f"  basis         = {plan.basis}")
    print()

# %% [markdown]
# ## 2. Emit a runnable recipe (write_recipe)
#
# `write_recipe` copies the plan's template YAML and overrides the keys you
# pass (dot-nested keys like `"search.name"` reach into nested blocks).

# %%
from galaxylens.sample_library.identify import write_recipe

plan = gl.identify("J1630+4520")
out = write_recipe(
    plan,
    {"search.name": "j1630_identify_demo", "search.n_live": 50},
    Path("/tmp/j1630_identify_demo.yaml"),
)
print(f"written: {out}")

cfg = gl.load_config(str(out))
print(f"load OK: search.name={cfg.search.name}  zlens={cfg.pipeline.geometry.zlens}")
print("(hand this FitConfig to gl.fit to run)")

# %% [markdown]
# ## 3. Identify from an image (catalog-independent fallback)
#
# When catalog morphology fields are missing/unreliable, measure the light
# concentration C = 5*log10(r80/r20) + axis ratio q directly from the science
# FITS (C < 3.1 disk, C > 3.4 elliptical). Calibrated: J2141 disk C=1.58,
# J1630 elliptical C=4.21. Requires a FITS; skipped if none is present.

# %%
import os

root = os.environ.get("GALAXYLENS_DATA_ROOT", "")
fits_path = f"{root}/processed_v2/disk/J2141-0001/v1/j2141-0001_f814w_sci.fits"
if os.path.exists(fits_path):
    plan = gl.identify_from_image(fits_path, name="J2141-0001")
    print(f"image -> lens_type={plan.lens_type}  constraint={plan.constraint}")
    print(f"         basis={plan.basis}")
else:
    print(f"[skip] no FITS at {fits_path}")
    print("       set GALAXYLENS_DATA_ROOT to a data root with processed_v2/ to try this.")
