"""SKIRT simulation-file (.ski) generation and execution.

Generates the SKIRT9 ``MonteCarloSimulation`` XML for F814W/F160W in a dust
(``ExtinctionOnly`` + ``ParticleMedium``) and a no-dust (``NoMedium``)
version, then runs the ``skirt`` binary. Inclination is 78.5 deg (edge-on,
matching J2141), azimuth 90 deg (long axis horizontal after the three-axis
alignment), and ``roll`` compensates each galaxy's residual tilt.

The ``skirt`` binary is taken from ``SKIRT_BIN`` (env var) or ``skirt`` on
PATH.
"""

import os
import subprocess

BANDS = {
    "F814W": "HST ACS WFC F814W",
    "F160W": "HST WFC3 IR F160W",
}

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
                    <FrameInstrument instrumentName="{band}" distance="10 Mpc" inclination="{incl}" azimuth="90 deg" roll="{roll}"
                                     fieldOfViewX="{fov}" fieldOfViewY="{fov}" numPixelsX="200" numPixelsY="200"/>
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
                    <FrameInstrument instrumentName="{band}" distance="10 Mpc" inclination="{incl}" azimuth="90 deg" roll="{roll}"
                                     fieldOfViewX="{fov}" fieldOfViewY="{fov}" numPixelsX="200" numPixelsY="200"/>
                </instruments>
            </InstrumentSystem>
        </instrumentSystem>
    </MonteCarloSimulation>
</skirt-simulation-hierarchy>
"""


def write_ski(label, band, outdir, incl="78.5 deg", fov="40 kpc", roll="0 deg",
              npackets="1e9"):
    """Write the dust and no-dust .ski files for one label/band.

    Returns ``(dust_path, nodust_path)``.
    """
    os.makedirs(outdir, exist_ok=True)
    bandname = BANDS[band]
    dust_path = os.path.join(outdir, f"{label}_{band}.ski")
    nodust_path = os.path.join(outdir, f"{label}_{band}_nodust.ski")
    with open(dust_path, "w") as f:
        f.write(TEMPLATE_DUST.format(label=label, band=band, bandname=bandname,
                                     incl=incl, fov=fov, roll=roll,
                                     npackets=npackets))
    with open(nodust_path, "w") as f:
        f.write(TEMPLATE_NODUST.format(label=label, band=band, bandname=bandname,
                                       incl=incl, fov=fov, roll=roll,
                                       npackets=npackets))
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
