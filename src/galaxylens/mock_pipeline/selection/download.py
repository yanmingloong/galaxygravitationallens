"""Particle cutout download (stars and gas).

Downloads HDF5 cutouts from the TNG API. Stars may be requested with the
3-field subset ``Coordinates,Velocities,Masses`` (enough for L3 circularity)
or ``all`` fields (needed for SKIRT radiative transfer, which also requires
GFM_Metallicity / GFM_StellarFormationTime / StellarHsml). Gas is always
``all`` (Density / GFM_Metallicity / InternalEnergy / Masses for dust).

This merges the project's ``dl_field_stars.py`` (3-field) and
``dl_skirt_particles.py`` (all-field) into one parameterized helper.
"""

import os
import time

import requests

from .scan import BASE, HEADERS

STARS_3FIELD = "Coordinates,Velocities,Masses"


def download_cutout(sim, snap, sid, ptype, fields, outdir, timeout=300):
    """Download one cutout (``ptype`` in {"stars","gas"}) into ``outdir``.

    Returns True on success (or if a valid file already exists), False on
    failure after retries. Files are written with a ``.part`` suffix then
    atomically renamed to avoid leaving corrupt HDF5 on interruption.
    """
    out = os.path.join(outdir, f"{sim}_snap{snap}_id{sid}_{ptype}.hdf5")
    if os.path.exists(out) and os.path.getsize(out) > 100_000:
        return True
    url = (f"{BASE}{sim}/snapshots/{snap}/subhalos/{sid}/"
           f"cutout.hdf5?{ptype}={fields}")
    for attempt in range(3):
        try:
            r = requests.get(url, headers=HEADERS, timeout=timeout)
            if r.status_code == 200 and len(r.content) > 100_000:
                tmp = out + ".part"
                with open(tmp, "wb") as f:
                    f.write(r.content)
                os.replace(tmp, out)
                return True
            print(f"  {sid} {ptype}: HTTP {r.status_code} (attempt {attempt + 1})",
                  flush=True)
        except Exception as e:
            print(f"  {sid} {ptype}: {type(e).__name__} (attempt {attempt + 1})",
                  flush=True)
        time.sleep(5 * (attempt + 1))
    return False


def download_stars(sim, snap, ids, outdir, fields=STARS_3FIELD):
    """Download star cutouts (3-field by default; ``fields="all"`` for SKIRT)."""
    os.makedirs(outdir, exist_ok=True)
    ok = 0
    for sid in ids:
        if download_cutout(sim, snap, sid, "stars", fields, outdir):
            ok += 1
        time.sleep(0.5)
    return ok


def download_gas(sim, snap, ids, outdir):
    """Download gas cutouts (all fields, for SKIRT dust)."""
    os.makedirs(outdir, exist_ok=True)
    ok = 0
    for sid in ids:
        if download_cutout(sim, snap, sid, "gas", "all", outdir):
            ok += 1
        time.sleep(1.0)
    return ok
