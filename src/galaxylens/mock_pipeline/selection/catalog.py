"""L3 result -> mock catalog assembly (automatic).

Reads L3 verification results (one row per candidate carrying f_disk / B/D /
L_hat / Re / thickness) and writes the ``catalog.csv`` consumed by the SKIRT
pipeline, expanding the L_hat unit-vector list into L_hat_x/y/z columns and
keeping only the disk-dominant subset (``f_disk >= f_disk_min``).

This replaces the legacy manual/archived ``rescan_disk_galaxies.py`` and
removes the old-convention L_hat that lived in the hand-maintained catalog —
the L_hat here comes straight from ``truth.kinematics.disk_kinematics``
(new convention, disk-particle angular-momentum axis).
"""

import csv

from ..truth.cosmology import snap_to_z

CATALOG_COLS = ["label", "simulation", "snap", "z", "subhalo_id", "f_disk",
                "B_over_D", "n_particles", "disk_thickness_med_kpc",
                "disk_Re_kpc", "re_kpc", "mstar_msun",
                "L_hat_x", "L_hat_y", "L_hat_z", "axis_angle_deg",
                "classification"]


def classify(f_disk, f_disk_disk=0.30, f_disk_min=0.60):
    """Morphology label from f_disk (DISK / WEAK / BULGE)."""
    if f_disk > f_disk_min:
        return "DISK"
    if f_disk > f_disk_disk:
        return "WEAK"
    return "BULGE"


def build_catalog(l3_results, out_csv, f_disk_min=0.60, f_disk_disk=0.30):
    """Write ``catalog.csv`` from L3 result rows.

    Each row must carry: sim, snap, subhalo_id, f_disk, B_over_D, L_hat
    (3-list), disk_thickness_med_kpc, disk_Re_kpc, re_kpc, mstar_msun,
    axis_angle_deg, n_particles. Rows with an ``error`` or missing f_disk
    are skipped; the output is sorted by f_disk descending.
    """
    rows = []
    for r in l3_results:
        if "error" in r or r.get("f_disk") is None:
            continue
        lhat = r.get("L_hat") or [0.0, 0.0, 0.0]
        rows.append({
            "label": str(r["subhalo_id"]),
            "simulation": r.get("sim", ""),
            "snap": r["snap"],
            "z": snap_to_z(r["snap"]),
            "subhalo_id": r["subhalo_id"],
            "f_disk": r["f_disk"],
            "B_over_D": r["B_over_D"],
            "n_particles": r.get("n_particles", 0),
            "disk_thickness_med_kpc": r["disk_thickness_med_kpc"],
            "disk_Re_kpc": r["disk_Re_kpc"],
            "re_kpc": r["re_kpc"],
            "mstar_msun": r["mstar_msun"],
            "L_hat_x": lhat[0],
            "L_hat_y": lhat[1],
            "L_hat_z": lhat[2],
            "axis_angle_deg": r["axis_angle_deg"],
            "classification": classify(r["f_disk"], f_disk_disk, f_disk_min),
        })
    rows.sort(key=lambda x: -x["f_disk"])
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CATALOG_COLS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    return rows
