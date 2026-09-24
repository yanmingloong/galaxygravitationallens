# Observational Sample Library

This directory holds the observational lens sample catalog data (real lenses
from SWELLS / SLACS, with photometric and spectroscopic parameters).

**Data source**: the authoritative catalog currently lives in the internal
research project at `GalaxyGravLens/SampleLibrary/`. It will be migrated here
as part of the P1 module migration — small metadata CSVs are tracked in git;
large raw tables and FITS are excluded via `.gitignore`.

Structure (planned):

```
sample_library/
├── disk/        # 11 SWELLS disk lenses (flagship: J2141-0001, J1331+3628)
├── slacs/       # 5 SLACS elliptical lenses (flagship: J1630)
└── README.md
```
