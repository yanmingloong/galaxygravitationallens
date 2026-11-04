# Contributing

*中文版见 [CONTRIBUTING_zh.md](CONTRIBUTING_zh.md).*

Conventions for maintaining the `galaxygravitationallens` repository.

## Language policy

- **Code comments (docstrings + inline comments): English only.** This keeps the
  source clean and internationally readable.
- **Documents (.md): bilingual, as two separate files.**
  - `README.md` — English (default entry point, auto-rendered by GitHub).
  - `README_zh.md` — Chinese.
  - Each file links to the other in its header.

## Contributing

Issues (bug reports, feature requests) and pull requests are welcome on the
hosting platform (Gitee / GitHub).

1. Fork the repository on the hosting platform.
2. Create a feature branch (`git checkout -b feature-xxx`), make your change,
   commit, and push to your fork.
3. Open a pull request against `main`; describe what and why.
4. A maintainer reviews and merges (or requests changes).

## Repository layout

- `src/galaxylens/` — package (config / data_pipeline / mock_pipeline / lensfit / sample_library).
- `configs/` — per-lens YAML configs.
- `sample_library/` — curated sample data (git-tracked subset).
- `examples/` — cluster/run templates (slurm), not package code.
- `tests/`, `notebooks/`, `docs/`.

## Repository organization

Conventions that keep this a **code library**, not a data/results store.

### Output location

- **Results and intermediate data never live inside the repo** — they are
  large, non-diffable, and environment-specific.
- Locate them via environment variables, never hard-coded paths:
  - `GALAXYLENS_DATA_ROOT` — observational data root (data_pipeline).
  - `GALAXYLENS_OUTPUT_ROOT` — results root (keep out of the repo).
  - `GALAXYLENS_SAMPLE_LIB` — sample-library root (01_raw~05_exports).
  - `TNG_API_KEY`, `SKIRT_BIN`, `SEXTRACTOR_BIN`, `TINYTIM_*` — external tools/secrets.
- Notebooks and scripts read outputs from these variables (see
  `notebooks/mock_pipeline_demo.py`).

### Documentation location

- Keep usage docs **centralized under `docs/` with an index**
  (`docs/README.md`) — one index to find everything, not scattered module
  READMEs.
- Module specifics belong in code docstrings; only module-specific *data
  inventories* (e.g. `sample_library/README.md`'s git-tracked CSV list) may
  live in the module directory.

### Local vs cluster

- **One code version, no local/cluster forks.** Core logic is
  environment-agnostic.
- Environment differences are expressed via function arguments (e.g.
  `workers`) and external scheduler templates under `examples/`
  (`run_selection_cluster.slurm`).
- Never add per-cluster branching inside the package.
