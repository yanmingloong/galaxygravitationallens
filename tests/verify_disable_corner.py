#!/usr/bin/env python3
"""Verify _disable_autofit_corner_plot sets the config and survives."""
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import autofit as af  # noqa: E402  (registers the autofit config)
from autonerves import conf  # noqa: E402

print("before:", conf.instance["visualize"]["plots_search"]["nest"]["corner_anesthetic"])

from galaxylens.api.search import _disable_autofit_corner_plot  # noqa: E402

_disable_autofit_corner_plot()
val = conf.instance["visualize"]["plots_search"]["nest"]["corner_anesthetic"]
print("after :", val)
assert val is False, val

# a fresh af.Nautilus sees the same singleton
print("af.Nautilus importable:", af.Nautilus is not None)
print("OK: autofit corner plot disabled")
