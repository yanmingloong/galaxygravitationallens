"""lensfit: a multi-method pipeline for fitting galaxy-scale gravitational lenses.

This subpackage is a *method collection*, not a single-model pipeline. It
assembles third-party engines (PyAutoLens, mgefit, jampy, AGAMA, ...) into a
reusable, parameterized chain for modelling lenses of any morphology (disks,
ellipticals, ...), and codifies the pitfalls learned while reproducing the
literature (the B / D / R / e series).

Design principles
-----------------
- **Multi-method**: stellar-mass models (MGE today, multi-component later) and
  dynamics engines (jampy today, AGAMA later) are interchangeable behind the
  same interfaces. Nothing here is hard-wired to a single paper or galaxy type.
- **Wrappers, not re-implementations**: MGE decomposition uses Cappellari's
  ``mgefit``, JAM dynamics uses ``jampy``, lensing uses PyAutoLens, AGAMA uses
  the ``agama`` package. lensfit contributes the *usage conventions*, the
  *pitfall guards*, and the *parameterization* on top of those engines.
- **Attribution is mandatory**: every module that borrows a method or formula
  MUST cite the paper and/or code it comes from in its docstring (see each
  submodule). Do not silently re-use an algorithm without naming its source.

Submodules
----------
- ``psf``      : empirical PSF measurement from field stars (DAOStarFinder)
- ``mge``      : Multi-Gaussian Expansion fit, I/O, and deprojection
- ``lens``     : lens model builders, galaxy-subtracted recipes, scale-free quad fit
- ``models``   : pluggable mass/source components (gNFW, NFW, MGE x M/L, Sersic)
- ``dynamics`` : Jeans/dynamics engines (jampy, jampy_kdc with counter-rotating core)
- ``fit``      : joint imaging + dynamics analysis (``joint_analysis`` + ``LensingConstraint``)
- ``pipeline`` : config -> model assembly (``assemble`` / ``build_collection``)
- ``structures``: shared data structures (Geometry, KinData, PipelineConfig, ...)
- ``validation``: mock-fit and ground-truth cross-checks
- ``plotting`` : data/model preview helpers

References
----------
The canonical method sources used across this package:
- Cappellari 2002, MNRAS 333, 400 (MGE decomposition and deprojection)
- Cappellari 2020, MNRAS 494, 4819 (jampy)
- Barnabe, Dutton, Marshall et al. 2012, MNRAS 423, 1073 (SWELLS IV, disk-lens
  galaxy-subtracted modelling)
- Wyithe, Turner & Spergel 2001, ApJ 555, 504 (gNFW)
- Navarro, Frenk & White 1997, ApJ 490, 493 (NFW)
- Evans & Witt 2003, MNRAS 345, 1351 (scale-free potential, quad image positions)
- Trick, van de Ven & Dutton 2016, MNRAS 463, 3151 / arXiv:1609.05477 (J1331 quad)
- Ludlow et al. 2016, MNRAS 460, 1214 (concentration-mass relation)
- Vasiliev 2019, MNRAS 482, 1525 (AGAMA)
"""

from __future__ import annotations
