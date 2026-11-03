"""Strong-lens mock production: arc ray-tracing + dataset merge.

Complementary to :mod:`galaxylens.mock_pipeline.skirt` (which renders the
lens GALAXY): this package produces the strong-lens ARC (a background source
deflected by the true-mass model) and merges it into the final mock dataset.
"""
from .arc import build_lens_galaxy, einstein_radius, make_arc
from .dataset import scale_arc, merge_arc

__all__ = [
    "build_lens_galaxy", "einstein_radius", "make_arc",
    "scale_arc", "merge_arc",
]
