"""SKIRT simulation-file (.ski) generation and execution.

Generates the SKIRT9 ``MonteCarloSimulation`` XML for arbitrary bands
(each a :class:`~.config.BandConfig`) in a dust (``ExtinctionOnly`` +
``ParticleMedium``) and a no-dust (``NoMedium``) version, then runs the
``skirt`` binary. Inclination/azimuth/roll come from the target + the
:class:`~.config.MockObsConfig`; roll compensates each galaxy's residual tilt.

The ``skirt`` binary is taken from ``SKIRT_BIN`` (env var) or ``skirt`` on
PATH.
"""

import os
import subprocess

from .config import J2141_OBS

# ExtinctionOnly + dust medium
TEMPLATE_DUST = """<?xml version="1.0" encoding="UTF-8"?>
<skirt-simulation-hierarchy type="MonteCarloSimulation" format="9" producer="SMILE" time="2026-09-08T08:00:00">
    <MonteCarloSimulation simulationMode="ExtinctionOnly" numPackets="{npackets}">
        <units type="Units"><ExtragalacticUnits/></units>
        <random type="Random"><Random/></random>
        <cosmology type="Cosmology"><LocalUniverseCosmology/></cosmology>
        <sourceSystem type="SourceSystem">
            <SourceSystem minWavelength="0.6 micron" maxWavelength="2.0 micron">
                <sources type="Source">
                    <ParticleSource filename="{label}_stars.dat">
                        <sedFamily type="SEDFamily">
                            <BruzualCharlotSEDFamily imf="Chabrier" resolution="Low"/>
                        </sedFamily>
                    </ParticleSource>
                </sources>
            </SourceSystem>
        </sourceSystem>
        <mediumSystem type="MediumSystem">
            <MediumSystem>
                <media type="Medium">
                    <ParticleMedium filename="{label}_dust.dat">
                        <materialMix type="MaterialMix">
                            <ThemisDustMix/>
                        </materialMix>
                    </ParticleMedium>
                </media>
                <grid type="SpatialGrid">
                    <PolicyTreeSpatialGrid minX="-50 kpc" maxX="50 kpc" minY="-50 kpc" maxY="50 kpc" minZ="-50 kpc" maxZ="50 kpc"/>
                </grid>
            </MediumSystem>
        </mediumSystem>
        <instrumentSystem type="InstrumentSystem">
            <InstrumentSystem>
                <defaultWavelengthGrid type="WavelengthGrid">
                    <ConfigurableBandWavelengthGrid>
                        <bands type="Band"><BroadBand bandName="{bandname}"/></bands>
                    </ConfigurableBandWavelengthGrid>
                </defaultWavelengthGrid>
                <instruments type="Instrument">
                    <FrameInstrument instrumentName="{band}" distance="{distance}" inclination="{incl}" azimuth="{azimuth}" roll="{roll}"
                                     fieldOfViewX="{fov}" fieldOfViewY="{fov}" numPixelsX="{num_pixels}" numPixelsY="{num_pixels}"/>
                </instruments>
            </InstrumentSystem>
        </instrumentSystem>
    </MonteCarloSimulation>
</skirt-simulation-hierarchy>
"""

# NoMedium (no dust, extinction-diagnostic baseline)
TEMPLATE_NODUST = """<?xml version="1.0" encoding="UTF-8"?>
<skirt-simulation-hierarchy type="MonteCarloSimulation" format="9" producer="SMILE" time="2026-09-08T08:00:00">
    <MonteCarloSimulation simulationMode="NoMedium" numPackets="{npackets}">
        <units type="Units"><ExtragalacticUnits/></units>
        <random type="Random"><Random/></random>
        <cosmology type="Cosmology"><LocalUniverseCosmology/></cosmology>
        <sourceSystem type="SourceSystem">
            <SourceSystem minWavelength="0.6 micron" maxWavelength="2.0 micron">
                <sources type="Source">
                    <ParticleSource filename="{label}_stars.dat">
                        <sedFamily type="SEDFamily">
                            <BruzualCharlotSEDFamily imf="Chabrier" resolution="Low"/>
                        </sedFamily>
                    </ParticleSource>
                </sources>
            </SourceSystem>
        </sourceSystem>
        <instrumentSystem type="InstrumentSystem">
            <InstrumentSystem>
                <defaultWavelengthGrid type="WavelengthGrid">
                    <ConfigurableBandWavelengthGrid>
                        <bands type="Band"><BroadBand bandName="{bandname}"/></bands>
                    </ConfigurableBandWavelengthGrid>
                </defaultWavelengthGrid>
                <instruments type="Instrument">
                    <FrameInstrument instrumentName="{band}" distance="{distance}" inclination="{incl}" azimuth="{azimuth}" roll="{roll}"
                                     fieldOfViewX="{fov}" fieldOfViewY="{fov}" numPixelsX="{num_pixels}" numPixelsY="{num_pixels}"/>
                </instruments>
            </InstrumentSystem>
        </instrumentSystem>
    </MonteCarloSimulation>
</skirt-simulation-hierarchy>
"""


def write_ski(label, band_cfg, outdir, obs=J2141_OBS, incl="78.5 deg",
              roll="0 deg", npackets="1e9"):
    """Write the dust and no-dust .ski files for one label/band.

    ``band_cfg`` is a :class:`~.config.BandConfig`; ``obs`` is the
    :class:`~.config.MockObsConfig` (fov / num_pixels / azimuth /
    distance). ``incl`` is the disk inclination; pass ``target.incl``
    from the target's
    :class:`~galaxylens.mock_pipeline.selection.config.TargetConfig`
    (default 78.5 deg = J2141). Returns ``(dust_path, nodust_path)``.
    """
    os.makedirs(outdir, exist_ok=True)
    dust_path = os.path.join(outdir, f"{label}_{band_cfg.label}.ski")
    nodust_path = os.path.join(outdir, f"{label}_{band_cfg.label}_nodust.ski")
    with open(dust_path, "w") as f:
        f.write(TEMPLATE_DUST.format(
            label=label, band=band_cfg.label, bandname=band_cfg.skirt_name,
            incl=incl, fov=obs.fov, roll=roll, npackets=npackets,
            num_pixels=obs.num_pixels, azimuth=obs.azimuth, distance=obs.distance))
    with open(nodust_path, "w") as f:
        f.write(TEMPLATE_NODUST.format(
            label=label, band=band_cfg.label, bandname=band_cfg.skirt_name,
            incl=incl, fov=obs.fov, roll=roll, npackets=npackets,
            num_pixels=obs.num_pixels, azimuth=obs.azimuth, distance=obs.distance))
    return dust_path, nodust_path


def run_skirt(ski_file, skirt_bin=None, outdir=None, timeout=None):
    """Run the ``skirt`` binary on one .ski file.

    ``skirt_bin`` defaults to ``SKIRT_BIN`` env var, then ``skirt`` on PATH.
    Runs in ``outdir`` (default: the .ski file's directory) so SKIRT writes
    its output next to the input. Returns the CompletedProcess.
    """
    bin_ = skirt_bin or os.environ.get("SKIRT_BIN", "skirt")
    cwd = outdir or os.path.dirname(os.path.abspath(ski_file))
    return subprocess.run([bin_, os.path.abspath(ski_file)], cwd=cwd,
                          check=True, timeout=timeout)
