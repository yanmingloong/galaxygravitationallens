"""TNG snapshot catalog scan (full and parallel).

Downloads the subhalo catalog above a total-mass threshold via the
TNG REST API and writes a CSV/JSON with the L1-relevant fields. Merges the
single-threaded ``tng_full_scan.py`` and the threaded ``tng_scan_parallel.py``
into one entry point: ``workers=1`` is sequential, ``workers>1`` uses a
thread pool. Resume is supported (existing subhalo ids are skipped).

The API key is read from the ``TNG_API_KEY`` environment variable (never
hard-coded). Lengths are converted comoving kpc/h -> physical kpc with the
scale factor, fixing the earlier missing-``a`` bug.

Output columns: sim, snap, subhalo_id, log_mstar, mstar_msun, sigma_km_s,
vmax_km_s, re_star_kpc, sfr_msun_yr, mass_log_total, primary_flag, grnr,
group_mass_log.
"""

import csv
import json
import math
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

from ..truth.cosmology import H, snap_to_a

API_KEY = os.environ.get("TNG_API_KEY", "")
HEADERS = {"api-key": API_KEY}
BASE = "https://www.tng-project.org/api/"
RATE_LIMIT = 0.25  # seconds/request

COLS = ["sim", "snap", "subhalo_id", "log_mstar", "mstar_msun", "sigma_km_s",
        "vmax_km_s", "re_star_kpc", "sfr_msun_yr", "mass_log_total",
        "primary_flag", "grnr", "group_mass_log"]


def get(url, timeout=45, retries=4):
    """GET with api-key header, retry/backoff, rate limiting."""
    for i in range(retries + 1):
        try:
            r = requests.get(url, headers=HEADERS, timeout=timeout)
            if r.status_code == 200:
                time.sleep(RATE_LIMIT)
                return r
        except Exception:
            pass
        time.sleep(3 * (i + 1))
    r = requests.Response()
    r.status_code = -1
    return r


def paginate_list(sim, snap, mass_min):
    """Page through ``/subhalos/?mass__gt=`` and return {id, mass_log} list."""
    url = f"{BASE}{sim}/snapshots/{snap}/subhalos/?mass__gt={mass_min}&limit=500"
    hits = []
    page = 0
    while url:
        r = get(url)
        if r.status_code != 200:
            print(f"  [paginate] HTTP {r.status_code}", flush=True)
            break
        d = r.json()
        for res in d.get("results", []):
            hits.append({"id": res["id"], "mass_log": res.get("mass_log_msun")})
        url = d.get("next")
        page += 1
        if page % 10 == 0:
            print(f"  [paginate] page {page}: {len(hits)} total", flush=True)
    print(f"  [paginate] {len(hits)} subhalos (mass>{mass_min})", flush=True)
    return hits


def group_mass(sim, snap, grnr, cache):
    """FoF group mass (Msun) for a group id, cached (thread-safe)."""
    with _gm_lock:
        if grnr in cache:
            return cache[grnr]
    r = get(f"{BASE}{sim}/snapshots/{snap}/halos/{grnr}/info.json")
    gm = 0.0
    if r.status_code == 200:
        gm = r.json().get("GroupMass", 0.0) * 1e10 / H
    with _gm_lock:
        cache[grnr] = gm
    return gm


def fetch_detail(sim, snap, sid):
    """Fetch one subhalo detail and return a catalog row (or None)."""
    r = get(f"{BASE}{sim}/snapshots/{snap}/subhalos/{sid}/")
    if r.status_code != 200:
        return None
    d = r.json()
    a = snap_to_a(snap)
    grnr = d.get("grnr")
    gm = group_mass(sim, snap, grnr, _gm_cache) if grnr is not None else None
    return {
        "sim": sim, "snap": snap, "subhalo_id": d.get("id"),
        "log_mstar": round(math.log10(d["mass_stars"] / H * 1e10), 2)
        if d.get("mass_stars") else None,
        "mstar_msun": round(d.get("mass_stars", 0) / H * 1e10, 1),
        "sigma_km_s": d.get("veldisp"), "vmax_km_s": d.get("vmax"),
        "re_star_kpc": round(d.get("halfmassrad_stars", 0) * H * a, 2),
        "sfr_msun_yr": d.get("sfr"), "mass_log_total": d.get("mass_log_msun"),
        "primary_flag": d.get("primary_flag"), "grnr": grnr,
        "group_mass_log": round(math.log10(gm), 2) if gm and gm > 0 else None,
    }


_gm_cache = {}
_gm_lock = threading.Lock()


def scan_snapshot(sim, snap, mass_min, outdir, workers=1,
                  max_detail=None, resume=True):
    """Scan one snapshot and write ``{sim}_snap{snap}_full.{json,csv}``.

    Returns ``(json_path, csv_path)``. ``workers=1`` is sequential;
    ``workers>1`` uses a thread pool with a shared group-mass cache.
    """
    os.makedirs(outdir, exist_ok=True)
    key = sim.replace("-", "_").lower()
    out_json = os.path.join(outdir, f"{key}_snap{snap}_full.json")

    hits = paginate_list(sim, snap, mass_min)
    if not hits:
        return out_json, None

    rows = []
    scanned = set()
    if resume and os.path.exists(out_json):
        try:
            with open(out_json) as f:
                rows = list(json.load(f))
            scanned = {r.get("subhalo_id") for r in rows}
            print(f"  [resume] {len(rows)} already scanned", flush=True)
        except Exception:
            rows = []

    todo = [h["id"] for h in hits if h["id"] not in scanned]
    if max_detail is not None:
        todo = todo[:max_detail]
    if not todo:
        print("  nothing to scan", flush=True)
    else:
        print(f"  [detail] {len(todo)} subhalos to fetch", flush=True)
        if workers <= 1:
            for i, sid in enumerate(todo):
                row = fetch_detail(sim, snap, sid)
                if row is not None:
                    rows.append(row)
                if (i + 1) % 50 == 0:
                    _write_json(out_json, rows)
        else:
            done = 0
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futs = {pool.submit(fetch_detail, sim, snap, sid): sid
                        for sid in todo}
                for fut in as_completed(futs):
                    row = fut.result()
                    done += 1
                    if row is not None:
                        rows.append(row)
                    if done % 500 == 0:
                        _write_json(out_json, rows)
        _write_json(out_json, rows)

    out_csv = out_json.replace(".json", ".csv")
    _write_csv(out_csv, rows)
    print(f"  done: {len(rows)} subhalos -> {out_json} / {out_csv}", flush=True)
    return out_json, out_csv


def _write_json(path, rows):
    with open(path, "w") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)


def _write_csv(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
