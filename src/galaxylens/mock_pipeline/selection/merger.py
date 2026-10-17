"""L2 merger-history check (mpb.hdf5 main-branch mass-jump method).

Determines whether a candidate had a recent major merger by tracking the
main-progenitor branch stellar mass in the SubLink ``mpb.hdf5`` merger tree.
A recent (z<0.5) main-branch stellar-mass jump > +30% is a major merger
(> +10% minor); a >30% drop followed by a >30% rise is flagged as a
branch-switch artifact (SubfindID re-mapping, not a real merger).

This replaces the earlier simple.json satellite/main-ratio method, which
counted every dwarf accretion (field galaxies have 85-385 recent mergers,
all minor) and so mis-flagged "quiet" systems. The mpb SubfindID
branch-switch issue is handled by the artifact detection below, not by
avoiding mpb.

mpb.hdf5 fields used: SnapNum, SubhaloMassType[:, 4] (stellar mass, 1e10 Msun/h).
"""

import io
import os
import time

import h5py
import numpy as np

from ..truth.cosmology import H, SNAP_Z
from .scan import BASE, get

MAJOR_JUMP = 0.30    # main-branch stellar-mass jump for a major merger (~1:3)
MINOR_JUMP = 0.10    # ~1:10
BRANCH_DROP = -0.30  # preceding drop marking a branch-switch artifact


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


def _read_mpb(f):
    """Read SnapNum and main-branch stellar mass (Msun) from an open mpb.hdf5."""
    snapnums = f["SnapNum"][:].astype(int)
    mstar_raw = f["SubhaloMassType"][:, 4].astype(float)  # 1e10 Msun/h
    return snapnums, mstar_raw / H * 1e10


def _load_mpb(sim, snap, sid, cache_dir=None):
    """Download (or read cached) mpb.hdf5 -> (snapnums, mstar_msun, err)."""
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        local = os.path.join(cache_dir, f"snap{snap}_id{sid}_mpb.hdf5")
        if not os.path.exists(local):
            r = get(f"{BASE}{sim}/snapshots/{snap}/subhalos/{sid}/sublink/mpb.hdf5")
            if r.status_code != 200:
                return None, None, "mpb.hdf5 download failed"
            with open(local, "wb") as fh:
                fh.write(r.content)
            time.sleep(0.3)
        with h5py.File(local, "r") as f:
            return _read_mpb(f) + (None,)
    r = get(f"{BASE}{sim}/snapshots/{snap}/subhalos/{sid}/sublink/mpb.hdf5")
    if r.status_code != 200:
        return None, None, "mpb.hdf5 download failed"
    with h5py.File(io.BytesIO(r.content), "r") as f:
        return _read_mpb(f) + (None,)


def _analyze(snapnums, mstar, recent_snap):
    """Recent main-branch mass jumps -> (majors, minors, artifacts, max_jump)."""
    idx = np.argsort(snapnums)
    sn = snapnums[idx]
    m = mstar[idx]
    sel = sn >= recent_snap
    sn, m = sn[sel], m[sel]
    if len(m) < 2:
        return [], [], [], 0.0
    rel = np.zeros(len(m))
    rel[1:] = (m[1:] - m[:-1]) / np.maximum(m[:-1], 1e-6)
    majors, minors, artifacts = [], [], []
    for i in range(1, len(m)):
        j = rel[i]
        if j > MAJOR_JUMP:
            if rel[i - 1] < BRANCH_DROP:
                artifacts.append((int(sn[i - 1]), int(sn[i]),
                                  float(rel[i - 1]), float(j)))
            else:
                majors.append((int(sn[i]), float(j)))
        elif j > MINOR_JUMP:
            minors.append((int(sn[i]), float(j)))
    return majors, minors, artifacts, float(rel.max()) if len(rel) else 0.0


def check_merger(sim, snap, sid, recent_snap=68, cache_dir=None):
    """Return a merger-history verdict dict for one subhalo.

    ``recent_snap`` is the snapshot floor for "recent" (68 -> z<0.5).
    ``cache_dir`` (optional) caches the downloaded mpb.hdf5 locally.

    Verdicts: ``QUIET``, ``RECENT_MAJOR_MERGER``, ``BRANCH_ARTIFACT``
    (verify by hand), or an ``error`` key on download/parse failure.
    """
    snapnums, mstar, err = _load_mpb(sim, snap, sid, cache_dir)
    if err:
        return {"subhalo": sid, "error": err}

    majors, minors, artifacts, max_jump = _analyze(snapnums, mstar, recent_snap)

    if artifacts:
        verdict = "BRANCH_ARTIFACT"
        detail = "; ".join(
            f"snap{s1}->{s2} drop{abs(r1) * 100:.0f}% rise{r2 * 100:.0f}%"
            for s1, s2, r1, r2 in artifacts)
    elif majors:
        verdict = "RECENT_MAJOR_MERGER"
        detail = "; ".join(f"snap{s} +{j * 100:.0f}%" for s, j in majors)
    else:
        verdict = "QUIET"
        detail = f"max jump +{max_jump * 100:.0f}%" + (" (minor)" if minors else "")

    return {"subhalo": sid, "verdict": verdict, "detail": detail,
            "max_jump": round(max_jump, 3), "majors": majors,
            "minors": minors, "artifacts": artifacts}
