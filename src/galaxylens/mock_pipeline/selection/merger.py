"""L2 merger-history check (simple.json method).

Determines whether a candidate had a recent major merger, using the SubLink
``simple.json`` merger list: a satellite/main stellar-mass ratio > 0.1 since
``recent_snap`` (z<0.5). This is the simple.json method, NOT the mpb.hdf5
main-branch mass-jump method (mpb SubfindID has a known bug and is not used).
"""

import gzip
import json
import time

from ..truth.cosmology import SNAP_Z
from .scan import BASE, get


def z_of(snap):
    """Linear-interpolated redshift for an arbitrary TNG snapshot."""
    snaps = sorted(SNAP_Z)
    if snap <= snaps[0]:
        return SNAP_Z[snaps[0]]
    if snap >= snaps[-1]:
        return SNAP_Z[snaps[-1]]
    for s1, s2 in zip(snaps, snaps[1:]):
        if s1 <= snap <= s2:
            z1, z2 = SNAP_Z[s1], SNAP_Z[s2]
            return z1 + (z2 - z1) * (snap - s1) / (s2 - s1)
    return 0.0


def _mass_stars(sim, snap, sid):
    r = get(f"{BASE}{sim}/snapshots/{snap}/subhalos/{sid}/")
    return r.json().get("mass_stars") if r.status_code == 200 else None


def check_merger(sim, snap, sid, recent_snap=68, major_ratio=0.1):
    """Return a merger-history verdict dict for one subhalo.

    ``recent_snap`` is the snapshot floor for "recent" (68 -> z<0.5).
    ``major_ratio`` is the satellite/main stellar-mass ratio threshold.
    """
    r = get(f"{BASE}{sim}/snapshots/{snap}/subhalos/{sid}/sublink/simple.json")
    if r.status_code != 200:
        return {"subhalo": sid, "error": "simple.json failed"}

    raw = r.content
    if raw[:2] == b"\x1f\x8b":
        raw = gzip.decompress(raw)
    tree = json.loads(raw.decode())

    main = {int(sn): int(i) for sn, i in tree["Main"]}
    mergers = [(int(sn), int(i)) for sn, i in tree["Mergers"]]
    recent = [m for m in mergers if m[0] >= recent_snap]

    major = []
    for m_snap, sat_id in recent:
        sat_m = _mass_stars(sim, m_snap, sat_id)
        main_id = main.get(m_snap)
        main_m = _mass_stars(sim, m_snap, main_id) if main_id is not None else None
        if sat_m and main_m and main_m > 0:
            ratio = sat_m / main_m
            if ratio > major_ratio:
                major.append({"snap": m_snap, "z": round(z_of(m_snap), 2),
                              "ratio": round(ratio, 3)})
        time.sleep(0.3)

    verdict = "RECENT_MAJOR_MERGER" if major else "QUIET"
    return {"subhalo": sid, "n_recent": len(recent),
            "major": major, "verdict": verdict}
