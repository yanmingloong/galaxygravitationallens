"""Selection target configuration for TNG disk-lens analog searches.

Each observational lens target (J2141, J1331, ...) carries its own selection
criteria. Criteria are *data, not code*: switching target must never require
editing the pipeline. A YAML loader can populate the same dataclass later;
for now the default J2141 target is built in.

Selection thresholds (project-convention, see the disk_lens_analogs reports):

- L1 (API): field center + stellar mass window + sigma(total) floor. The
  sigma floor is 140 km/s (relaxed) — a 170 km/s floor biases toward
  bulge-dominated systems because ``veldisp`` is the all-particle dispersion
  (DM-dominated) and cold thin disks fall at 140-160.
- L2 (merger): no recent major merger (mpb.hdf5 main-branch stellar-mass
  jump > +30%) since snapshot ``recent_merger_snap`` (z<0.5).
- L3 (particle): ``f_disk`` from circularity; ``f_disk_min`` marks the
  disk-dominant subset, ``f_disk_disk`` marks "has a disk".
"""

from dataclasses import dataclass, field

from ..skirt.config import MockObsConfig, J2141_OBS, J1331_OBS


@dataclass
class SelectionCriteria:
    """L1/L2/L3 selection thresholds for one lens target."""

    sims: list                     # simulation boxes to scan
    snapshots: list                # snapshots in the survey window
    mass_min: float                # scan pre-filter (total mass, 1e10 Msun/h)
    mstar_range: tuple             # L1 stellar-mass window (Msun)
    sigma_min: float               # L1 sigma(total) floor (km/s)
    field_only: bool = True        # L1: require field center (primary_flag=1)
    f_disk_disk: float = 0.30      # L3 "has a disk" threshold
    f_disk_min: float = 0.60       # L3 disk-dominant subset threshold
    recent_merger_snap: int = 68   # L2 merger history floor (z<0.5)


@dataclass
class TargetConfig:
    """One observational lens target for analog selection."""

    name: str
    z_lens: float
    z_source: float
    snap: int                      # snapshot matching z_lens
    criteria: SelectionCriteria
    incl: str = "78.5 deg"         # disk inclination for the SKIRT mock (J2141)
    obs: MockObsConfig = field(default_factory=lambda: J2141_OBS)  # observation setup (bands/geometry)


# ---------------------------------------------------------------------------
# Built-in default target: SDSS J2141-0001
# ---------------------------------------------------------------------------
J2141 = TargetConfig(
    name="J2141-0001",
    z_lens=0.138,
    z_source=0.7127,
    snap=88,
    criteria=SelectionCriteria(
        sims=["TNG100-1"],
        snapshots=[88, 91, 84, 95, 99],
        mass_min=68.0,
        # Census convention (relaxed). The J2141-specific tight window
        # [7e10, 1.3e11] is a subset; the census window avoids pre-biasing
        # toward the target before the L3 disk verification.
        mstar_range=(5e10, 3e11),
        sigma_min=140.0,
        field_only=True,
        f_disk_disk=0.30,
        f_disk_min=0.60,
        recent_merger_snap=68,
    ),
)

# ---------------------------------------------------------------------------
# Built-in target: SDSS J1331+3628 (SWELLS disk lens, Trick+2016, bulge-dominant)
#
# J1331 differs from J2141 in three ways that live OUTSIDE this dataclass:
# - sigma_min=190 (not 140): the L1 sigma floor is the calibrated bulge
#   sigma0~200, NOT the relaxed 140 (see disk_lens_analogs/J1331/README.md).
# - L2 merger judgment is REVERSED: J1331 keeps "recent merger" systems
#   (counter-rotating core = merger relic), unlike J2141's "quiet" criterion.
#   This is a standalone merger.check_merger call, not in the run chain.
# - L3 criterion is counter-rotation (counter_rotation_angle ~ 180 deg KDC),
#   NOT f_disk-dominant; J1331 is bulge-dominant (B/D ~ 2.7).
# ---------------------------------------------------------------------------
J1331 = TargetConfig(
    name="J1331+3628",
    z_lens=0.113,
    z_source=0.254,
    snap=90,
    criteria=SelectionCriteria(
        sims=["TNG100-1"],
        snapshots=[90],
        mass_min=68.0,
        mstar_range=(1e11, 3e11),
        sigma_min=190.0,
        field_only=True,
        f_disk_disk=0.30,
        f_disk_min=0.60,
        recent_merger_snap=68,
    ),
    incl="70 deg",
    obs=J1331_OBS,
)

BUILTIN_TARGETS = {"j2141": J2141, "j1331": J1331}


def get_target(name: str) -> TargetConfig:
    """Return a built-in target by key (e.g. ``"j2141"``)."""
    key = name.lower()
    if key not in BUILTIN_TARGETS:
        raise KeyError(
            f"unknown target {name!r}; available: {sorted(BUILTIN_TARGETS)}")
    return BUILTIN_TARGETS[key]
