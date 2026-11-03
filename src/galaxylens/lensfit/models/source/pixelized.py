"""Pixelized (inversion) source model for the pluggable source layer.

The B8/B9 source recipe: instead of a parametric Sersic profile, the source is
reconstructed on a regular pixel mesh via a PyAutoLens ``Pixelization`` --
``mesh`` (RectangularUniform) + ``regularization`` (Constant). Each mesh pixel
is a latent inversion parameter solved by the linear inversion inside the
likelihood; the ONLY explicit (sampler-visible) source parameter is the
regularization coefficient, which trades data fit against pixel-to-pixel
smoothness.

Degrees-of-freedom subtlety (the reason ``n_source`` != ``n_free_params`` here):
the sampler sees ``n_free_params = 1`` (the coefficient), so a B9 fit has
``6 mass + 1 regularization = 7`` explicit parameters. But the imaging
degrees of freedom subtract the MESH pixel count (``n_source = nx*ny = 225``
for a 15x15 mesh), because those pixels are free parameters of the source
reconstruction in the ``nu_img = n_unmasked - n_source`` normalisation. The
parametric ``SingleSersic`` keeps ``n_source == n_free_params == 7``.

References
----------
- Nightingale, Dye & Massey 2018, MNRAS 478, 4738 (PyAutoLens pixelized
  source reconstruction / inversion)
- Barnabe, Dutton, Marshall et al. 2012, MNRAS 423, 1073 (source galaxy of the
  B-series; the B8/B9 pixelized-source upgrade)
"""

from __future__ import annotations

from typing import ClassVar, Tuple

from ..base import SourceModel
from ...structures import PriorSet


class PixelizedSource(SourceModel):
    """A pixelized (inversion) source on a regular mesh (B8/B9 recipe).

    Parameters
    ----------
    mesh_shape : (nx, ny) tuple
        Shape of the ``RectangularUniform`` reconstruction mesh. B8/B9 used
        ``(15, 15)`` (a 20x20 mesh blew past the 64 GB cluster memory limit --
        see the B9 OOM lesson).
    reg : str
        Regularization flavour. Only ``"constant"`` (``al.reg.Constant``) is
        implemented and B9-validated; other flavours are a documented extension
        point.
    coeff_bounds : (low, high) tuple
        LogUniform bounds of the regularization coefficient. B8/B9 used
        ``(1e-6, 1e6)``.
    """

    name: ClassVar[str] = "pixelized"

    def __init__(
        self,
        mesh_shape: Tuple[int, int] = (15, 15),
        reg: str = "constant",
        coeff_bounds: Tuple[float, float] = (1e-6, 1e6),
    ):
        self.mesh_shape = (int(mesh_shape[0]), int(mesh_shape[1]))
        self.reg = reg
        self.coeff_bounds = (float(coeff_bounds[0]), float(coeff_bounds[1]))

    @classmethod
    def from_config(cls, cfg):
        return cls(
            mesh_shape=getattr(cfg, "source_mesh_shape", (15, 15)),
            reg=getattr(cfg, "source_reg", "constant"),
            coeff_bounds=getattr(cfg, "source_reg_coeff_bounds", (1e-6, 1e6)),
        )

    def attach(self, source, priors: PriorSet) -> None:
        import autolens as al
        import autofit as af

        source.pixelization = af.Model(al.Pixelization)
        source.pixelization.mesh = al.mesh.RectangularUniform(shape=self.mesh_shape)
        if self.reg == "constant":
            source.pixelization.regularization = af.Model(al.reg.Constant)
        else:
            raise ValueError(
                f"unsupported regularization {self.reg!r}; only 'constant' "
                "is B9-validated"
            )
        source.pixelization.regularization.coefficient = af.LogUniformPrior(
            lower_limit=self.coeff_bounds[0], upper_limit=self.coeff_bounds[1]
        )

    @property
    def n_free_params(self) -> int:
        # Only the regularization coefficient is an explicit sampler parameter;
        # the mesh pixels are latent inversion parameters.
        return 1

    @property
    def n_source(self) -> int:
        # The imaging degrees of freedom subtract the MESH pixel count (the
        # latent source parameters), not the explicit-parameter count.
        return int(self.mesh_shape[0] * self.mesh_shape[1])
