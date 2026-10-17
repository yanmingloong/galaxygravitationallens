"""Regression tests for mock_pipeline.skirt (against 348489 all-field data)."""

import os

import numpy as np
import pytest

from galaxylens.mock_pipeline.skirt import (
    inertia_rotation, tng_to_skirt, write_ski,
    BandConfig, MockObsConfig, J2141_BANDS, J2141_OBS,
)

RAW_STARS_348489 = ("/mnt/d/HermesAgent/ResearchProjects/cosmology_simulations/"
                    "IllustrisTNG/mock_observations/01_data/raw_stars/"
                    "TNG100-1_snap88_id348489_stars.hdf5")
RAW_GAS_348489 = ("/mnt/d/HermesAgent/ResearchProjects/cosmology_simulations/"
                  "IllustrisTNG/mock_observations/01_data/raw_gas/"
                  "TNG100-1_snap88_id348489_gas.hdf5")


def test_inertia_rotation_thin_disk():
    """A flattened disk -> smallest eigenvector along z, flat ~ small."""
    rng = np.random.default_rng(0)
    n = 20000
    x = rng.normal(0, 10, n)
    y = rng.normal(0, 10, n)
    z = rng.normal(0, 1, n)
    pos = np.column_stack([x, y, z])
    mass = np.ones(n)
    R, evals = inertia_rotation(pos, mass)
    flat = evals[0] / evals[2]
    # variance ratio sigma_z^2 / sigma_x^2 = 1/100 for a 10:1 thin disk
    assert flat < 0.05
    # smallest eigenvector (disk normal) maps to +z row of R
    assert abs(R[2, 2]) == pytest.approx(1.0, abs=0.05)


def test_write_ski(tmp_path):
    dust_path, nodust_path = write_ski("S5", J2141_BANDS[0], str(tmp_path),
                                       roll="-14 deg")
    for p in (dust_path, nodust_path):
        assert os.path.exists(p)
        txt = open(p).read()
        assert "S5_stars.dat" in txt
        assert "HST ACS WFC F814W" in txt
        assert "78.5 deg" in txt
        assert 'numPixelsX="200"' in txt
        assert 'distance="10 Mpc"' in txt
        assert 'azimuth="90 deg"' in txt
    assert "ParticleMedium" in open(dust_path).read()
    assert "NoMedium" in open(nodust_path).read()


def test_write_ski_wfpc2_band(tmp_path):
    """A different instrument for the same filter must be expressible."""
    wfpc2_f814w = BandConfig("F814W", "HST WFPC2 F814W", 0.10)
    dust_path, _ = write_ski("J1331", wfpc2_f814w, str(tmp_path))
    txt = open(dust_path).read()
    assert "HST WFPC2 F814W" in txt
    assert "HST ACS WFC F814W" not in txt


def test_obs_config_defaults():
    assert len(J2141_OBS.bands) == 2
    assert J2141_OBS.bands[0].skirt_name == "HST ACS WFC F814W"
    assert J2141_OBS.bands[1].skirt_name == "HST WFC3 IR F160W"
    assert J2141_OBS.bands[1].pixel_scale == pytest.approx(0.13)
    assert J2141_OBS.num_pixels == 200


def test_tng_to_skirt_348489(tmp_path):
    if not (os.path.exists(RAW_STARS_348489) and os.path.exists(RAW_GAS_348489)):
        pytest.skip("348489 all-field stars/gas HDF5 not available")
    res = tng_to_skirt(RAW_STARS_348489, RAW_GAS_348489, "348489",
                       "TNG100-1", 88, str(tmp_path))
    assert res["n_stars"] > 0
    assert res["n_dust"] > 0
    # 348489 is a thin disk: intrinsic flattening lambda3/lambda1 ~ 0.1
    assert res["flat"] < 0.2

    # stars .dat: 7 columns (x y z h M_init Z age)
    stars_dat = tmp_path / "348489_stars.dat"
    assert stars_dat.exists()
    first = open(stars_dat).readline()
    assert first.startswith("# column 1: x (pc)")
    line0 = None
    for line in open(stars_dat):
        if not line.startswith("#"):
            line0 = line.split()
            break
    assert len(line0) == 7

    # dust .dat: 5 columns (x y z h M_dust)
    dust_dat = tmp_path / "348489_dust.dat"
    assert dust_dat.exists()
    for line in open(dust_dat):
        if not line.startswith("#"):
            assert len(line.split()) == 5
            break
