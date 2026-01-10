# coint4 (coint2)

This repository contains a cointegration pairs trading framework and related tooling.

## Repo layout

- `coint4/`: main application workspace (full src, tests, configs, UI, Docker files)
- `src/`: legacy/core package code (kept for reference)
- `scripts/`: helper scripts for optimization, validation, and utilities
- `configs/`: run and search-space configs
- `docs/`: architecture, data, and testing docs
- `tests/`: root-level tests (legacy)
- `data/`: local datasets (ignored)
- `outputs/`, `results/`, `logs/`: generated artifacts (ignored)

Notes:
- There are two Poetry projects: `pyproject.toml` at the repo root (legacy) and `coint4/pyproject.toml` (full app). Use the one that matches your workflow.
- The primary CLI entrypoint is `coint2`.

## Data and outputs

Large datasets and generated outputs are intentionally excluded from Git. Use external storage or Git LFS for any large files you must version.
See `docs/data_storage.md` for details.

## Quickstart

See `docs/quickstart.md` and `docs/testing_guide.md`.

