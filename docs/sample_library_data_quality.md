# Sample Library Data Quality Notes

*中文版见 [sample_library_data_quality_zh.md](sample_library_data_quality_zh.md).*

> Recorded 2026-10-14 from a sample-library data audit of the J2141 analog project. These pitfalls once let erroneous values such as θE=1.10″ and Re=9.16 kpc propagate into the modeling inputs; take care to avoid them.

## 1. Hard-coded parameter dicts must cite their literature source

`enrich.py`'s `swell_s_thetas` dict was once hand-filled with θE, and 4 of its SWELLS lens values were wrong (no literature check; J2141 was filled in as 1.10, but the real value is 0.88).

- Any parameter dict "hand-filled from the literature" **must state its source in a comment**: author, year, Table number, column name.
- When a dict mixes lenses from different surveys (SWELLS vs SLACS), **comment them by group**, not all under one comment.
- When fixing or adding a value, **add date + source + old value** (see the NOTE comments in `enrich.py` for the format).

## 2. Field units/semantics must be preserved across Phase handoffs

`Re_bulge_arcsec`/`Re_disk_arcsec` were once **kpc values mislabeled as arcsec** — Phase B/C copied the SWELLS source Excel's `Re/kpc` directly into the arcsec field without the kpc→arcsec conversion.

- Each field's **unit must be declared explicitly in the schema**, and unit conversion must be done when handing off between Phases — no "same-name direct copy".
- Raw source field names are not trustworthy: the SWELLS Excel's `Re/kpc` actually stores **Rd (scale length)** for the disk, not Re (half-light radius R50). **Field semantics must be checked against the original paper**, not just the field name.

## 3. kpc→arcsec conversion must use each lens's own redshift

Different lenses have different redshifts, hence different kpc/″ factors (Planck 2015):

| z_lens | kpc/″ |
|--------|-------|
| 0.104 | 1.972 |
| 0.131 | 2.408 |
| 0.138 | 2.517 |
| 0.158 | 2.818 |

**Do not apply one fixed 2.517 to all lenses.**

## 4. Copy-sync rule

The modeling_input CSVs under `DiskLenses/SampleLibrary/05_exports/` have copies in `galaxygravitationallens/sample_library/` (which go into git). **When the source file changes, the copy must be synced**, otherwise the repo data silently falls behind (this happened once: an early θE patch only changed the DiskLenses source, and the copy stayed at the old value until it was caught weeks later).

## 5. Data fixes must be logged

Any data fix (value change, unit change, filling a gap) must be recorded in the "Data fix log" section of `sample_library/README.md`: what changed, old value → new value, source, date. Otherwise nobody later knows why a value is what it is.

## 6. Known leftover items (documented, not yet fixed)

- 19 SLACS lenses' θE not yet checked against the SLACS literature (Auger+2009).
- `Re_disk` field semantics (Rd vs Re) vs the bulge Re definition still to be verified — only unit conversion has been done against the raw source.
- J1037+3517 is also ellipsis in SWELLS III Table 1, so there is no authoritative θE value.
