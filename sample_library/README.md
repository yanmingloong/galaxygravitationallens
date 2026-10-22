# Observational Sample Library

*中文版见 [README_zh.md](README_zh.md).*

Observational lens sample library for disk + elliptical galaxies. Data are **real survey lens catalogs (no ground truth)**, distinct from the mock screening library (`mock_pipeline` selection stage, which has ground truth).

## Structure

- **Code**: `src/galaxylens/sample_library/` (build pipeline + access)
- **Data**: this directory (product modeling_input CSVs)

## Data (products)

5 modeling input CSVs, exported by Phase E of the build pipeline.

| File | Content | Size |
|---|---|---|
| modeling_input_full.csv | full sample (tier 0/1/2/3/X) | 2.6M |
| modeling_input_sample.csv | modeling sample | 79K |
| modeling_input_elliptical.csv | elliptical lenses | 99K |
| modeling_input_tier0.csv | Tier 0 (confirmed disk) | 1.4K |
| modeling_input_tier01.csv | Tier 0+1 | 14K |

**Authoritative source**: the research-project SampleLibrary directory pointed to by `GALAXYLENS_SAMPLE_LIB` (full `01_raw`~`05_exports` + raw survey downloads). This directory holds only products; raw downloads (505M) and regenerable intermediates stay local (not in git).

## Rebuild

```bash
# point to the sample library root (containing 01_raw ~ 05_exports)
export GALAXYLENS_SAMPLE_LIB=/path/to/SampleLibrary

# full rebuild B→C→D→E (no network)
python -m galaxylens.sample_library.pipeline

# resume from a phase
python -m galaxylens.sample_library.pipeline --from phase_d

# with network queries (Phase D online supplement)
python -m galaxylens.sample_library.pipeline --network

# re-download one survey (Phase A)
python -m galaxylens.sample_library.download.lenscat
```

## Build pipeline (5 phases)

| Phase | Module | Output |
|---|---|---|
| A download | `download/` (6 surveys) | 01_raw |
| B normalize | `build.py` (48-field Schema + 3-level dedup) | 02_merged/master.csv |
| C classify | `classify.py` (Tier 0/1/2/3/X) | 03_identified |
| D enrich | `enrich.py` (SDSS/Simard/Auger) | 04_enriched/tier012_vN.csv |
| E export | `export.py` | 05_exports |

**Add a new survey**: write a `download/xxx.py` fetcher + add an entry in `config.py`'s `RAW_SOURCES`; Phase B merges and deduplicates automatically.
