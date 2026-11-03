# galaxylens documentation index

*中文版见 [README_zh.md](README_zh.md).*

| Doc | Content |
|---|---|
| [data_pipeline.md](data_pipeline.md) | observational data pipeline guide + tutorial |
| [sample_library_data_quality.md](sample_library_data_quality.md) | sample library data-quality rules (units / semantics / copy sync / correction log) |
| [mock_pipeline.md](mock_pipeline.md) | mock data production + selection (TNG search → SKIRT mock images) |
| [usage_guide.md](usage_guide.md) | end-to-end: find disk galaxies → download → reduce → analyse & fit |
| [components_reference.md](components_reference.md) | fitting engine: pluggable components + configurable parameters index |
| [refinement_plan.md](refinement_plan.md) | pluggable-architecture blueprint (components + facade + roadmap) |
| [roadmap.md](roadmap.md) | pending extraction + analysis-flow gaps / extension items |
| [../lensfit/README.md](../lensfit/README.md) | lensfit engine guide (PSF/MGE/lens, reference table + pitfall list) |
| [README.md](../README.md) | project overview + architecture |

## Environment quick reference

```bash
# data root (required)
export GALAXYLENS_DATA_ROOT=/path/to/DataPipeline

# results output root (keep out of the repo)
export GALAXYLENS_OUTPUT_ROOT=/path/to/results

# SExtractor (optional)
export SEXTRACTOR_BIN=/path/to/sex

# sample library root (sample_library module)
export GALAXYLENS_SAMPLE_LIB=/path/to/SampleLibrary

# TNG API key (for mock_pipeline.selection/skirt downloads)
export TNG_API_KEY=...

# SKIRT 9 binary (for mock_pipeline.skirt.render)
export SKIRT_BIN=/path/to/skirt
```
