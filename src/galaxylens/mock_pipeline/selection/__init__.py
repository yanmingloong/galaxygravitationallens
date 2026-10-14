"""TNG disk-lens analog selection (scan -> L1 -> L2 -> L3 -> catalog).

Selection criteria are data (:class:`~.config.TargetConfig`), not code; the
L3 disk criterion and ground-truth quantities come from the shared
``truth`` package, so selection and mock validation never diverge.
"""

from .catalog import build_catalog, classify
from .config import (
    SelectionCriteria, TargetConfig, J2141, get_target, BUILTIN_TARGETS,
)
from .download import (
    download_stars, download_gas, download_cutout, STARS_3FIELD,
)
from .filter import filter_l1, pass_l1, is_field
from .merger import check_merger, z_of
from .scan import scan_snapshot
from .verify import verify_one, load_stars

__all__ = [
    "build_catalog", "classify",
    "SelectionCriteria", "TargetConfig", "J2141", "get_target",
    "BUILTIN_TARGETS",
    "download_stars", "download_gas", "download_cutout", "STARS_3FIELD",
    "filter_l1", "pass_l1", "is_field",
    "check_merger", "z_of",
    "scan_snapshot",
    "verify_one", "load_stars",
]
