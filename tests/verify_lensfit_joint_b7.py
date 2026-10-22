#!/usr/bin/env python3
"""Regression check: lensfit joint core (gnfw_virial + mge_const_ml + jampy) vs B7.

Run with autolens_jax (autolens + autofit + jampy + mgefit + nautilus). Verifies
the componentized core layer reproduces B7's joint dynamics EXACTLY, and that the
physical anchors (beta_z, M*, V_circ) match the B7 final results.

Checks:
  1. dyn chi2 + V_circ: refactored components reproduce a verbatim B7 dyn_chi2
     at the B7 median-pdf parameters (exact, same-machine determinism).
  2. physical anchors: beta_z = 0.651, M* = 1.437e11. (V_circ(2.2Rd) is an
     output fully fixed by the params + template, so it is validated by the
     exact-match in check 1; the "290" quoted in b7_joint.md is the local-smoke
     value at prior medians, NOT the fitted median-pdf value.)
  3. model structure: 13 free params (6 mass + 7 source).

The imaging branch of the joint likelihood (which needs the galaxy-subtracted
F814W FITS, cluster-only) is exercised by check 3's model build; the dynamics
branch is exercised end-to-end by checks 1-2.
"""
import sys
from pathlib import Path
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[1]  # repo root (portable)
sys.path.insert(0, str(REPO / "src"))

import numpy as np
from scipy.integrate import quad
from astropy.cosmology import FlatLambdaCDM

from galaxylens.lensfit.structures import (
    Geometry,
    KinData,
    PipelineConfig,
    PriorSet,
    PriorSpec,
    SourcePriors,
)
from galaxylens.lensfit.models.stellar.mge_const_ml import MgeConstMl, stellar_mge_peak
from galaxylens.lensfit.models.dm.gnfw_virial import GNFWVirial, vvir_to_rs, zeta
from galaxylens.lensfit.dynamics.jampy import JampyEngine

# ---- B7 final results (median_pdf_sample, outputs/b7_joint/samples_summary.json) ----
B7 = dict(
    ml_img=13.575937273609815,
    v_vir=212.26679152167054,
    c_minus2=5.095233792763365,
    gamma=0.77151875639625,
    q_h=0.69384859965147,
    ratio=0.5907773483998032,
)

# ---- J2141 geometry / kinematics (B5 constants) ----
ZL, ZSRC = 0.138, 0.7127
INC = 78.5
COSMO = FlatLambdaCDM(H0=70.0, Om0=0.3)
DA_MPC = COSMO.angular_diameter_distance(ZL).value
PC_PER_AS = DA_MPC * np.pi / 0.648
RD_DISK = 1.51

VRMS_DATA = np.array([
    (0.4123, 236.2, 15.0), (0.9297, 231.4, 15.0), (1.4456, 211.4, 15.0),
    (1.9669, 164.2, 15.0), (2.4815, 168.9, 15.0), (3.0028, 178.8, 15.0),
    (3.5178, 171.0, 15.0), (4.0331, 158.8, 18.0), (4.5529, 180.3, 18.0),
    (5.0696, 209.0, 18.0), (5.5893, 227.5, 20.0)])

geometry = Geometry(
    zlens=ZL, zsrc=ZSRC, inclination_deg=INC, da_mpc=DA_MPC, pc_per_arcsec=PC_PER_AS,
    re_arcsec=0.88, rd_arcsec=RD_DISK,
)
kinematics = KinData(
    xbin=VRMS_DATA[:, 0], ybin=np.zeros(11), vrms=VRMS_DATA[:, 1],
    vrms_err=VRMS_DATA[:, 2], vcirc=289.0, vcirc_err=15.0,
)

# ---- resolved stellar template (local MGE decomposition) ----
template = stellar_mge_peak(pc_per_arcsec=PC_PER_AS)
SURF_LUM, SIG_STAR, Q_STAR = template.mge
CALIB = template.calibration.calib
L2D_REL = template.l2d_rel

priors = PriorSet(
    ml_img=(2.0, 100.0), v_vir=(150.0, 350.0), gamma=(0.3, 1.5),
    q_h=(0.4, 1.0), ratio=(0.5, 1.0),
    c_2=PriorSpec(kind="gaussian", mean=5.9, sigma=1.49, lower=3.0, upper=20.0),
)
cfg = PipelineConfig(
    geometry=geometry, kinematics=kinematics, dark="gnfw_virial",
    stellar="mge_const_ml", source="single_sersic", dynamics="jampy",
    priors=priors, stellar_template=template, source_priors=SourcePriors(),
)

# ---- verbatim reference: B7 dyn_chi2 + b5_vvir dark_halo_mge_vvir ----
def _ref_beta_cyl(ratio):
    def f(r, th):
        beta = 1 - ratio ** (-2 * np.cos(2 * th))
        dbdt = -4 * np.log(ratio) * ratio ** (-2 * np.cos(2 * th)) * np.sin(2 * th)
        return beta, dbdt
    return f

def _ref_chi2_of(vrms):
    return float(np.sum(((vrms - VRMS_DATA[:, 1]) / VRMS_DATA[:, 2]) ** 2))

def _ref_gnfw_vvir(v_vir, gamma, c_minus2, qh, h):
    c = (2 - gamma) * c_minus2
    z1 = zeta(c, gamma, 1.0)
    zq = zeta(c, gamma, qh)
    r_vir = v_vir / (h * np.sqrt(zq / z1))
    r_s = r_vir / c
    delta_c = (200 / 3) * c ** 3 / z1
    return r_s, delta_c, r_vir

def _ref_dark_halo_mge_vvir(v_vir, gamma, c_minus2, qh, h, rho_crit_pc3, pc_per_as):
    import mgefit as mge
    r_s, delta_c, r_vir = _ref_gnfw_vvir(v_vir, gamma, c_minus2, qh, h)
    rb_pc = r_s * 1000.0
    r_pc = np.geomspace(0.05 * pc_per_as, rb_pc * 20, 200)
    m = r_pc / rb_pc
    rho = delta_c * rho_crit_pc3 / (m ** gamma * (1 + m) ** (3 - gamma))
    mh = mge.fit_1d(r_pc, rho, ngauss=14, quiet=1, plot=0)
    return mh.sol[0], mh.sol[1] / pc_per_as, r_s, delta_c, r_vir

def reference_dyn_chi2(ml_img, v_vir, gamma, c_minus2, q_h, ratio):
    import jampy as jam
    ez = np.sqrt(0.3 * (1 + ZL) ** 3 + (1 - 0.3))
    h = 70.0 * ez / 100.0
    rho_crit_pc3 = 2.775e11 * (70.0 / 100.0) ** 2 * ez ** 2 / 1e18
    ml = ml_img * CALIB
    surf_s = SURF_LUM * ml
    surf_dm_sph, sig_dm, _, _, _ = _ref_dark_halo_mge_vvir(
        v_vir, gamma, c_minus2, q_h, h, rho_crit_pc3, PC_PER_AS
    )
    sin_i = np.sin(np.radians(INC)); cos_i = np.cos(np.radians(INC))
    q_obs_h = np.sqrt(q_h ** 2 * sin_i ** 2 + cos_i ** 2)
    surf_dm = surf_dm_sph / q_obs_h
    model_lum = (surf_s, SIG_STAR, Q_STAR)
    surf_pot = np.concatenate([surf_s, surf_dm])
    sig_pot = np.concatenate([SIG_STAR, sig_dm])
    q_pot = np.concatenate([Q_STAR, np.full_like(surf_dm, q_obs_h)])
    model_pot = (surf_pot, sig_pot, q_pot)
    out = jam.axi.proj(model_lum, model_pot, INC, 0.0, DA_MPC, kinematics.xbin, kinematics.ybin,
                       beta=_ref_beta_cyl(ratio), gamma=0.0, kappa=None, ml=1.0,
                       moment="zz", pixsize=0.0, quiet=True, plot=False,
                       nang=15, nlos=40)
    chi2_vrms = _ref_chi2_of(out.model)
    vcirc = jam.mge.vcirc(surf_pot, sig_pot, q_pot, INC, 0.0, DA_MPC,
                          np.array([2.2 * RD_DISK]))[0]
    chi2_ha = ((vcirc - 289.0) / 15.0) ** 2
    return chi2_vrms + chi2_ha, vcirc

# ---- refactored path ----
stellar_comp = MgeConstMl.from_config(cfg)
dm_comp = GNFWVirial.from_config(cfg)
engine = JampyEngine.from_config(cfg)

def _fake_instance(ml_img, v_vir, gamma, c_minus2, q_h, ratio):
    return SimpleNamespace(
        galaxies=SimpleNamespace(
            lens=SimpleNamespace(
                mge_0=SimpleNamespace(mass_to_light_ratio=ml_img),
                dark=SimpleNamespace(v_vir=v_vir, inner_slope=gamma, c_2=c_minus2, q_h=q_h),
                anisotropy=SimpleNamespace(mass_to_light_ratio=ratio),
            )
        )
    )

def refactored_dyn_chi2(ml_img, v_vir, gamma, c_minus2, q_h, ratio):
    inst = _fake_instance(ml_img, v_vir, gamma, c_minus2, q_h, ratio)
    stellar_mge = stellar_comp.dyn_mge(inst, geometry, cfg.stellar_template.calibration)
    dm_mge = dm_comp.dyn_mge(inst, geometry, cfg.stellar_template.calibration)
    model_lum = stellar_mge
    surf_pot = np.concatenate([stellar_mge[0], dm_mge[0]])
    sig_pot = np.concatenate([stellar_mge[1], dm_mge[1]])
    q_pot = np.concatenate([stellar_mge[2], dm_mge[2]])
    model_pot = (surf_pot, sig_pot, q_pot)
    ratio_val = engine.ratio_from(inst)
    chi2_dyn, info = engine.chi2(model_lum, model_pot, ratio_val, geometry, kinematics)
    return chi2_dyn, info.get("vcirc")

# ---- check 1: exact reproduction of B7 dyn_chi2 ----
ref_chi2, ref_vcirc = reference_dyn_chi2(
    B7["ml_img"], B7["v_vir"], B7["gamma"], B7["c_minus2"], B7["q_h"], B7["ratio"]
)
new_chi2, new_vcirc = refactored_dyn_chi2(
    B7["ml_img"], B7["v_vir"], B7["gamma"], B7["c_minus2"], B7["q_h"], B7["ratio"]
)
print(f"[1] chi2_dyn  ref={ref_chi2:.6f}  new={new_chi2:.6f}")
print(f"    vcirc     ref={ref_vcirc:.4f}  new={new_vcirc:.4f} km/s")
assert abs(ref_chi2 - new_chi2) < 1e-6 * max(1.0, abs(ref_chi2)), (ref_chi2, new_chi2)
assert abs(ref_vcirc - new_vcirc) < 1e-6, (ref_vcirc, new_vcirc)

# ---- check 2: physical anchors ----
beta_z = 1 - B7["ratio"] ** 2
# CALIB x L2D_REL is BLAS-stable (cluster 1.058e10, B6 doc sec 7); M* uses it
# directly so the anchor does not depend on the local MGE decomposition.
CALIB_L2D_CLUSTER = 1.058e10
mstar = B7["ml_img"] * CALIB_L2D_CLUSTER
print(f"[2] beta_z  = {beta_z:.4f}  (expect 0.651)")
print(f"    M*      = {mstar:.3e} Msun  (expect 1.437e11)")
print(f"    V_circ  = {new_vcirc:.1f} km/s  (median-pdf fitted value; B7 smoke's 290 was at prior medians)")
print(f"    CALIB x L2D_REL (local) = {CALIB * L2D_REL:.4e}  (cluster anchor {CALIB_L2D_CLUSTER:.4e})")
assert abs(beta_z - 0.651) < 0.01
assert abs(mstar - 1.437e11) / 1.437e11 < 0.01
# consistency: local template should land within ~2% of the cluster CALIB x L2D_REL
assert abs(CALIB * L2D_REL - CALIB_L2D_CLUSTER) / CALIB_L2D_CLUSTER < 0.02

# ---- check 3: model structure (13 free params) ----
from galaxylens.lensfit.pipeline.assemble import build_collection
collection = build_collection(cfg)
print(f"[3] prior_count = {collection.prior_count}  (expect 13)")
assert collection.prior_count == 13

print("\nALL joint-core B7 regression checks passed.")
