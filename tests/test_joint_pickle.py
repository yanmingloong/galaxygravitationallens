"""Regression: JointAnalysis is picklable for multiprocessing.

The joint analysis class is built inside ``joint_analysis`` (dynamic base +
lazy autolens/autofit import); it must carry a stable module path so
``number_of_cores > 1`` can pickle it. Guards the fix that sets
``__module__``/``__qualname__`` and registers the class in the module namespace.
"""

import pickle
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from galaxylens.lensfit.fit.joint import joint_analysis  # noqa: E402
from galaxylens.lensfit.fit.constraints import MeinConstraint  # noqa: E402
from galaxylens.lensfit.models.dm.nfw import NFW  # noqa: E402
from galaxylens.lensfit.models.stellar.mge_radial_ml import MgeRadialMl  # noqa: E402
from galaxylens.lensfit.dynamics.jampy_kdc import JampyKdcEngine  # noqa: E402
from galaxylens.lensfit.structures import Geometry, KinData, Calibration  # noqa: E402


def test_joint_analysis_picklable():
    geometry = Geometry(
        zlens=0.113, zsrc=0.254, inclination_deg=70.0,
        da_mpc=414.0, pc_per_arcsec=414.0 * 1e6 / 206265.0,
    )
    r9 = np.array([0.032, 0.601, 0.663, 1.230, 1.294, 1.863, 1.924, 2.808, 2.870])
    kinematics = KinData(
        xbin=r9, ybin=np.zeros(9),
        vrms=np.full(9, 200.0), vrms_err=np.full(9, 5.0),
    )
    sigma = np.array([0.051, 0.178, 0.503, 1.180, 3.891])
    qobs = np.array([1.00, 0.76, 0.58, 0.56, 0.57])
    surf_lum = np.array([20768.0, 3161.2, 1588.2, 502.25, 136.51])
    ml = np.array([4.7, 2.2, 2.2, 4.7, 4.7])

    stellar = MgeRadialMl((surf_lum, sigma, qobs), ml)
    dm = NFW()
    engine = JampyKdcEngine(
        kappa_prime=0.76, r0_kappa_arcsec=2.2, seeing_fwhm_arcsec=1.1,
    )
    mein = MeinConstraint(0.907, 7.72e10, 0.10)
    calib = Calibration(1.0)

    analysis = joint_analysis(
        mein, [stellar, dm], engine, geometry, kinematics, calib, n_source=0,
    )

    # pickle round-trip (the operation multiprocessing performs)
    data = pickle.dumps(analysis)
    analysis2 = pickle.loads(data)

    assert isinstance(analysis2, type(analysis))
    # the reconstructed analysis still exposes its attributes
    assert analysis2.n_source == 0
    assert analysis2.n_mass_params == analysis.n_mass_params
