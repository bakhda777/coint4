# coint4 (coint2)

This repository contains a cointegration pairs trading framework and related tooling.

## Repo layout

- `coint4/`: main application workspace (full src, tests, configs, UI, Docker files)
- `coint4/data_downloaded/`: canonical dataset location (ignored, large files)
- `legacy/`: archived root-level code/tests/configs/scripts and legacy tooling
- `docs/`: architecture, data, and testing docs (paths assume `cd coint4`)
- `data/`: local datasets (ignored)
- `outputs/`, `results/`, `artifacts/` (including `artifacts/live/logs/`): generated artifacts (ignored)

Notes:
- The active Poetry project lives in `coint4/pyproject.toml`. Legacy Poetry and requirements files are in `legacy/`.
- The primary CLI entrypoint is `coint2`.

## Data and outputs

Large datasets and generated outputs are intentionally excluded from Git. Use external storage or Git LFS for any large files you must version.
See `docs/data_storage.md` for details.

## Quickstart

See `docs/quickstart.md` and `docs/testing_guide.md` (paths assume `cd coint4`).
For production runs, use `docs/production_checklist.md`. Overlays and configs:
- `coint4/configs/data_quality_strict.yaml` (strict QA)
- `coint4/configs/data_window_clean.yaml` (clean window + symbol exclusions overlay)
- `coint4/configs/main_2024_wfa_balanced.yaml` (balanced WFA)
Note: WFA supports optional fixed universe via `--pairs-file` and custom output via `--results-dir` (see `docs/quickstart.md`).

Optimization plan: `docs/optimization_plan_20260116.md` (supersedes `docs/optimization_plan_20260114.md`, legacy/archived).
Selection grid runs (filters): `docs/optimization_runs_20260115.md` with configs in `coint4/configs/selection_grid_20260115/` and strict p-value grid in `coint4/configs/selection_grid_20260115_strictpv/` (parallel run notes in docs).
SSD top-N sweep (dynamic selection): `docs/optimization_runs_20260115.md` with configs in `coint4/configs/ssd_topn_sweep_20260115/` and агрегатором `coint4/artifacts/wfa/aggregate/20260115_ssd_topn_sweep/` (конфиги обновлены под `n_jobs: -1`, запуск по одному).
SSD top-N sweep (subset 4 values): `docs/optimization_runs_20260115.md` with configs in `coint4/configs/ssd_topn_sweep_20260115_4vals/` and агрегатором `coint4/artifacts/wfa/aggregate/20260115_ssd_topn_sweep_4vals/`.
SSD top-N sweep (subset 3 values, 30k/40k/50k): `docs/optimization_runs_20260115.md` with configs in `coint4/configs/ssd_topn_sweep_20260115_3vals/` and агрегатором `coint4/artifacts/wfa/aggregate/20260115_ssd_topn_sweep_3vals/`.
Sharpe target runs (strict signals): `docs/optimization_runs_20260115.md` with configs in `coint4/configs/sharpe_target_20260115/` and агрегатором `coint4/artifacts/wfa/aggregate/20260115_sharpe_target/`.
Quality universe (data-driven exclusions): `docs/optimization_runs_20260115.md` with configs in `coint4/configs/quality_runs_20260115/` (including corr0.45 alignment runs, signal_strict, tradeability/hl0p05-45, z0p9/z1p0 exit0p1 variants, and denylist `coint4/configs/quality_runs_20260115/denylist_symbols_20260115.yaml`), universe artifacts in `coint4/artifacts/universe/quality_universe_20260115/`, `coint4/artifacts/universe/quality_universe_20260115_250k/`, and `coint4/artifacts/universe/quality_universe_20260115_200k/`, WFA aggregates in `coint4/artifacts/wfa/aggregate/20260115_quality_universe_500k/`, `coint4/artifacts/wfa/aggregate/20260115_quality_universe/`, and `coint4/artifacts/wfa/aggregate/20260115_quality_universe_200k/`.
Rollup index for WFA runs: `coint4/artifacts/wfa/aggregate/rollup/` (built by `coint4/scripts/optimization/build_run_index.py`).
SSD refine / signal / risk sweeps (2026-01-16): `docs/optimization_runs_20260116.md` with configs in `coint4/configs/ssd_topn_refine_20260116/`, `coint4/configs/signal_sweep_20260116/`, `coint4/configs/signal_grid_20260116/`, `coint4/configs/risk_sweep_20260116/` and агрегаторами `coint4/artifacts/wfa/aggregate/20260116_ssd_topn_refine/`, `coint4/artifacts/wfa/aggregate/20260116_signal_sweep/`, `coint4/artifacts/wfa/aggregate/20260116_signal_grid/`, `coint4/artifacts/wfa/aggregate/20260116_risk_sweep/`.
Piogoga grid (leader filters, zscore sweep): `docs/optimization_runs_20260116.md` with configs in `coint4/configs/piogoga_grid_20260116/` and агрегатором `coint4/artifacts/wfa/aggregate/20260116_piogoga_grid/`.
Leader validation (post-analysis, SSD leader): `docs/optimization_runs_20260116.md` with configs in `coint4/configs/leader_validation_20260116/` and агрегатором `coint4/artifacts/wfa/aggregate/20260116_leader_validation/`.
WFA queue watcher (CPU heartbeat, idle detection): `coint4/scripts/optimization/watch_wfa_queue.sh`.
WFA pair filtering: parallelized via `backtest.n_jobs` (processes when memory-map is enabled) to maximize CPU usage.
Optimization state file (headless continuation): `docs/optimization_state.md`.
Headless codex prompt template: `coint4/scripts/optimization/on_done_codex_prompt.txt` (ключевая фраза: "Прогон завершён, продолжай выполнение плана" + инструкции headless/запись причины сбоя в `docs/optimization_state.md`).
