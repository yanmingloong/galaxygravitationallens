"""Regression tests for mock_pipeline.selection (against user-provided data).

No network access: the L1 filter and L3 verify tests run against a user-provided
catalog CSV and 348489 particle HDF5 (via GALAXYLENS_TEST_CATALOG /
GALAXYLENS_TEST_STARS_348489); skipped when those variables are unset.
"""

import csv
import os

import pytest

from galaxylens.mock_pipeline.selection import (
    J2141, J1331, get_target, filter_l1, pass_l1, verify_one, build_catalog,
    classify,
)

CATALOG_CSV = os.environ.get("GALAXYLENS_TEST_CATALOG", "")
STARS_348489 = os.environ.get("GALAXYLENS_TEST_STARS_348489", "")


def _load_catalog():
    with open(CATALOG_CSV) as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["mstar_msun"] = float(r["mstar_msun"])
        r["sigma_km_s"] = float(r["sigma_km_s"])
        r["primary_flag"] = int(r["primary_flag"])
        r["group_mass_log"] = (float(r["group_mass_log"])
                               if r["group_mass_log"] else None)
        r["subhalo_id"] = int(r["subhalo_id"])
    return rows


def test_filter_l1_j2141():
    if not os.path.exists(CATALOG_CSV):
        pytest.skip("catalog CSV not available")
    rows = _load_catalog()
    cands = filter_l1(rows, J2141.criteria)
    assert len(cands) > 0
    ids = {c["subhalo_id"] for c in cands}
    # 348489 (f_disk=0.643, sigma=160, M*=8.9e10, field) must pass L1
    assert 348489 in ids


def test_verify_one_348489():
    if not os.path.exists(STARS_348489):
        pytest.skip("348489 particle HDF5 not available")
    res = verify_one("TNG100-1", 88, 348489,
                     os.path.dirname(STARS_348489))
    assert "error" not in res
    assert res["f_disk"] == pytest.approx(0.643, abs=0.01)
    assert res["B_over_D"] == pytest.approx(0.55, abs=0.05)
    assert res["subhalo_id"] == 348489


def test_classify():
    assert classify(0.8) == "DISK"
    assert classify(0.45) == "WEAK"
    assert classify(0.1) == "BULGE"


def test_build_catalog(tmp_path):
    l3 = [
        {"sim": "TNG100-1", "snap": 88, "subhalo_id": 348489,
         "f_disk": 0.643, "B_over_D": 0.55, "L_hat": [0.1, 0.2, 0.9],
         "disk_thickness_med_kpc": 2.3, "disk_Re_kpc": 11.9, "re_kpc": 8.0,
         "mstar_msun": 8.9e10, "axis_angle_deg": 1.9, "n_particles": 100000},
        {"sim": "TNG100-1", "snap": 88, "subhalo_id": 343259,
         "f_disk": 0.561, "B_over_D": 0.78, "L_hat": [0.3, 0.0, 0.95],
         "disk_thickness_med_kpc": 2.0, "disk_Re_kpc": 6.3, "re_kpc": 5.0,
         "mstar_msun": 9.74e10, "axis_angle_deg": 3.0, "n_particles": 90000},
        {"sim": "TNG100-1", "snap": 88, "subhalo_id": 1,
         "f_disk": 0.1, "B_over_D": 999.0, "L_hat": [0.0, 0.0, 1.0],
         "disk_thickness_med_kpc": 5.0, "disk_Re_kpc": 3.0, "re_kpc": 2.0,
         "mstar_msun": 1e10, "axis_angle_deg": 30.0, "n_particles": 50000,
         "error": "download failed"},
    ]
    out = tmp_path / "catalog.csv"
    rows = build_catalog(l3, str(out), f_disk_min=0.6, f_disk_disk=0.3)
    # 348489 (0.643) and 343259 (0.561) pass; subhalo 1 has error -> skipped
    assert len(rows) == 2
    # sorted by f_disk descending
    assert rows[0]["subhalo_id"] == 348489
    assert rows[1]["subhalo_id"] == 343259
    # L_hat list expanded to three columns
    assert rows[0]["L_hat_x"] == 0.1
    assert rows[0]["L_hat_z"] == 0.9
    # classification follows f_disk
    assert rows[0]["classification"] == "DISK"
    assert rows[1]["classification"] == "WEAK"  # 0.561 < 0.6
    # header matches the SKIRT-consumable schema
    with open(out) as f:
        header = next(csv.reader(f))
    assert header[:4] == ["label", "simulation", "snap", "z"]


def test_get_target_j1331():
    """J1331 registered as the second built-in target (config-driven)."""
    j = get_target("j1331")
    assert j.name == "J1331+3628"
    assert j.z_lens == pytest.approx(0.113)
    assert j.z_source == pytest.approx(0.254)
    assert j.snap == 90
    assert j.incl == "70 deg"
    assert j.criteria.sigma_min == pytest.approx(190.0)
    assert j.criteria.mstar_range == (1e11, 3e11)
    assert len(j.obs.bands) == 2
    assert j.obs.bands[0].skirt_name == "HST ACS WFC F435W"
    assert j.obs.bands[1].peak == pytest.approx(0.3166)


def test_get_target_unknown_raises():
    import pytest as _pytest
    with _pytest.raises(KeyError):
        get_target("nonexistent")
