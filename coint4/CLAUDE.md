# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Coint2 is a cointegration pairs trading framework for cryptocurrencies. It provides a complete pipeline for:
- Pair selection using cointegration analysis
- Backtesting with numba-optimized engines
- Hyperparameter optimization using Optuna
- Walk-forward validation
- Live trading capabilities

Remote run helper (Serverspace): `scripts/remote/run_server_job.sh` powers on the server, waits for SSH, pulls latest code, runs a command, syncs artifacts, and powers off. It uses `SERVSPACE_API_KEY` + `SERVER_ID`/`SERVER_NAME` + `SERVER_IP` env vars.

## Key Commands

### Testing
```bash
# Quick smoke tests (< 30 seconds)
bash scripts/test.sh

# Full test suite (default skips slow/serial via pytest.ini)
./.venv/bin/pytest -q

# Run specific test categories
./.venv/bin/pytest -m smoke               # Critical tests
./.venv/bin/pytest -m "not slow"          # All except slow tests
./.venv/bin/pytest tests/unit -v          # Unit tests only
./.venv/bin/pytest tests/integration      # Integration tests
./.venv/bin/pytest -n auto -m "not serial" # Parallel tests (if xdist installed)

# Run single test
./.venv/bin/pytest tests/unit/core/test_math_utils.py::test_calculate_ssd_when_using_blocks_then_matches_brute_force -xvs
```

### Optimization & Backtesting
```bash
# Run hyperparameter optimization
PYTHONPATH=src ./.venv/bin/python scripts/core/optimize.py \
  --mode balanced \
  --n-trials 100 \
  --config configs/main_2024.yaml \
  --search-space configs/search_spaces/fast.yaml

# Signals-only optimization (dynamic selection, Q4 2023 baseline)
PYTHONPATH=src ./.venv/bin/python scripts/core/optimize.py \
  --mode balanced \
  --n-trials 12 \
  --config configs/main_2024_optimize_dynamic.yaml \
  --search-space configs/search_spaces/optimize_signals.yaml \
  --study-name opt_signals_dynamic_coarse_20260114_v2

# Selection grid (filters, Q4 2023, 3 steps)
./run_wfa_fullcpu.sh \
  configs/selection_grid_20260115/selgrid_20260115_exit0p06_pv0p30_kpss0p03_h0p60_c0p35_hl0p001-100.yaml \
  artifacts/wfa/runs/20260115_selgrid/selgrid_20260115_exit0p06_pv0p30_kpss0p03_h0p60_c0p35_hl0p001-100

# Selection grid (strict p-value, Q4 2023, 3 steps)
./run_wfa_fullcpu.sh \
  configs/selection_grid_20260115_strictpv/selgrid_20260115_strictpv_exit0p06_pv0p05_kpss0p05_h0p55_c0p45_hl0p01-60.yaml \
  artifacts/wfa/runs/20260115_selgrid_strictpv/selgrid_20260115_strictpv_exit0p06_pv0p05_kpss0p05_h0p55_c0p45_hl0p01-60

# Selection grid (strict p-value, n_jobs=-1, sequential)
cat artifacts/wfa/aggregate/20260115_selgrid_strictpv/strictpv_configs.txt | \
  xargs -P 1 -I {} bash -lc 'cfg="$1"; run_id=$(basename "$cfg" .yaml); ./run_wfa_fullcpu.sh "$cfg" "artifacts/wfa/runs/20260115_selgrid_strictpv/$run_id"' _ {}

# SSD top-N sweep (dynamic selection, full CPU, sequential runs)
cat configs/ssd_topn_sweep_20260115/ssd_topn_configs.txt | \
  xargs -P 1 -I {} bash -lc 'cfg="$1"; run_id=$(basename "$cfg" .yaml); ./run_wfa_fullcpu.sh "$cfg" "artifacts/wfa/runs/20260115_ssd_topn_sweep/$run_id" > "artifacts/wfa/runs/20260115_ssd_topn_sweep/$run_id/run.log" 2>&1' _ {}

# Resume queued runs (run_queue.csv)
PYTHONPATH=src ./.venv/bin/python scripts/optimization/run_wfa_queue.py \
  --queue artifacts/wfa/aggregate/20260115_ssd_topn_sweep/run_queue.csv

# Resume queued runs with CPU heartbeat (recommended)
bash scripts/optimization/watch_wfa_queue.sh \
  --queue artifacts/wfa/aggregate/20260115_ssd_topn_sweep/run_queue.csv \
  --parallel "$(nproc)"

# Resume queued runs with headless Codex on completion
bash scripts/optimization/watch_wfa_queue.sh \
  --queue artifacts/wfa/aggregate/20260115_ssd_topn_sweep/run_queue.csv \
  --parallel "$(nproc)" \
  --on-done-prompt-file scripts/optimization/on_done_codex_prompt.txt \
  --on-done-log artifacts/wfa/aggregate/20260115_ssd_topn_sweep/codex_on_done.log

watch_wfa_queue.sh по умолчанию ставит `--parallel` равным числу CPU (можно переопределить).
run_wfa_queue.py по умолчанию ставит `--parallel` равным числу CPU (override через флаг).
run_wfa_fullcpu.sh фиксирует `NUMBA_NUM_THREADS=1`, чтобы не раздувать потоки при параллельных прогонах (override через env `NUMBA_NUM_THREADS`).

Шаблон prompt должен включать ключевую фразу "Прогон завершён, продолжай выполнение плана" и инструкции: headless‑режим + запись причины сбоя в `docs/optimization_state.md`.

# Strict PV grid (queue)
PYTHONPATH=src ./.venv/bin/python scripts/optimization/run_wfa_queue.py \
  --queue artifacts/wfa/aggregate/20260115_selgrid_strictpv/run_queue.csv

# Selection grid (queue)
PYTHONPATH=src ./.venv/bin/python scripts/optimization/run_wfa_queue.py \
  --queue artifacts/wfa/aggregate/20260115_selgrid/run_queue.csv

# Build rollup index of WFA runs
PYTHONPATH=src ./.venv/bin/python scripts/optimization/build_run_index.py \
  --output-dir artifacts/wfa/aggregate/rollup

# SSD refine / signal / risk sweeps (2026-01-16)
PYTHONPATH=src ./.venv/bin/python scripts/optimization/run_wfa_queue.py \
  --queue artifacts/wfa/aggregate/20260116_ssd_topn_refine/run_queue.csv

PYTHONPATH=src ./.venv/bin/python scripts/optimization/run_wfa_queue.py \
  --queue artifacts/wfa/aggregate/20260116_signal_sweep/run_queue.csv

PYTHONPATH=src ./.venv/bin/python scripts/optimization/run_wfa_queue.py \
  --queue artifacts/wfa/aggregate/20260116_signal_grid/run_queue.csv

PYTHONPATH=src ./.venv/bin/python scripts/optimization/run_wfa_queue.py \
  --queue artifacts/wfa/aggregate/20260116_risk_sweep/run_queue.csv

# Piogoga grid (leader filters, zscore sweep, 2026-01-16)
PYTHONPATH=src ./.venv/bin/python scripts/optimization/run_wfa_queue.py \
  --queue artifacts/wfa/aggregate/20260116_piogoga_grid/run_queue.csv

# Leader holdout (next run, rollup leader, 2024-05-01 → 2024-12-31)
./.venv/bin/coint2 walk-forward \
  --config configs/best_config__leader_holdout_ssd25000__20260116_211943.yaml

# Sharpe target (strict signals, Q4 2023, 3 steps)
./run_wfa_fullcpu.sh \
  configs/sharpe_target_20260115/sharpe_target_strict_z1p2.yaml \
  artifacts/wfa/runs/20260115_sharpe_target_strict_z1p2

./run_wfa_fullcpu.sh \
  configs/sharpe_target_20260115/sharpe_target_strict_z1p4.yaml \
  artifacts/wfa/runs/20260115_sharpe_target_strict_z1p4

# Optimization modes:
# - fast: Quick optimization with fewer trials (max 50)
# - balanced: Standard optimization (default)
# - strict: Conservative parameters, tighter filters
# - relaxed: Looser parameters, more pairs
# - parallel: Parallel optimization with multiple workers
# - iterative: Iterative refinement of parameters
# - large: Large-scale optimization with many trials

# Quick test mode (set QUICK_TEST=true for faster development)
QUICK_TEST=true PYTHONPATH=src ./.venv/bin/python scripts/core/optimize.py \
  --mode fast \
  --n-trials 5 \
  --config configs/main_2024.yaml \
  --search-space configs/search_spaces/fast.yaml
```

### Pair Universe Selection
```bash
# Scan and select pairs
python scripts/universe/select_pairs.py \
  --data-root ./data_downloaded \
  --timeframe 15T \
  --period-start 2024-01-01 \
  --period-end 2024-01-31 \
  --criteria-config configs/criteria_relaxed.yaml \
  --out-dir artifacts/universe/

# Build quality universe (liquidity/history screen)
./.venv/bin/python scripts/universe/build_quality_universe.py \
  --data-root data_downloaded \
  --period-start 2023-01-01 \
  --period-end 2023-09-30 \
  --bar-minutes 15 \
  --min-history-days 180 \
  --min-coverage-ratio 0.9 \
  --min-avg-daily-turnover-usd 250000 \
  --max-days-since-last 14 \
  --out-dir artifacts/universe/quality_universe_20260115_250k

Последние quality universe артефакты:
- `artifacts/universe/quality_universe_20260115/`
- `artifacts/wfa/aggregate/20260115_quality_universe_500k/`
- `artifacts/universe/quality_universe_20260115_250k/`
- `artifacts/wfa/aggregate/20260115_quality_universe/` (corr0.45 alignment)
- `artifacts/universe/quality_universe_20260115_200k/`
- `artifacts/wfa/aggregate/20260115_quality_universe_200k/`
- `configs/quality_runs_20260115/denylist_symbols_20260115.yaml`
- `configs/quality_runs_20260115/quality_200k_z0p8_corr0p45_tradeable.yaml`
- `configs/quality_runs_20260115/quality_200k_z0p8_corr0p45_signal_strict.yaml`
- `configs/quality_runs_20260115/quality_200k_z0p8_corr0p5_hl0p05-45.yaml`
- `configs/quality_runs_20260115/quality_200k_z0p9_exit0p1.yaml`
- `configs/quality_runs_20260115/quality_200k_z1p0_exit0p1.yaml`

# Merge pairs from multiple selections
python scripts/universe/merge_pairs.py

# Scan data availability
python scripts/universe/scan_data.py
```

### Live Trading (Legacy)
```bash
# Note: Live trading scripts are archived in archive/old_scripts/live/
# Run live trading (paper mode by default)
python archive/old_scripts/live/run.py --config configs/main_2024.yaml --mode paper

# Extract live snapshot
python archive/old_scripts/live/extract_snapshot.py
```

### Analysis & Monitoring (Active + Legacy)
```bash
# Active scripts
PYTHONPATH=src ./.venv/bin/python scripts/run_preflight.py
PYTHONPATH=src ./.venv/bin/python scripts/monitor_drift.py
PYTHONPATH=src ./.venv/bin/python scripts/extract_live_snapshot.py

# Legacy scripts (archived)
./archive/old_scripts/ops/dashboard.sh
python archive/old_scripts/analysis/optuna_report.py --study-path outputs/studies/latest.db
python archive/old_scripts/ops/monitor_drift.py
```

## Architecture

### Core Components (`src/coint2/`)

**Data Pipeline:**
- `core/data_loader.py`: Handles parquet data loading with caching (monthly `data_downloaded/` + legacy `symbol=...` layouts)
- `core/normalization_improvements.py`: Rolling z-score and other normalization methods
- `pipeline/pair_scanner.py`: Cointegration testing and pair selection
- `pipeline/filters.py`: Advanced filtering (Hurst exponent, liquidity, etc.)

**Backtesting Engines:**
- `engine/base_engine.py`: Reference implementation
- `engine/optimized_backtest_engine.py`: Numba-optimized for speed
- Both engines must maintain parity (≥90% signal agreement)

**Optimization (`src/optimiser/`):**
- `fast_objective.py`: Optuna objective function for walk-forward optimization
- `run_optimization.py`: CLI entry point for optimization
- `metric_utils.py`: Sharpe ratio and other performance metrics

**Configuration:**
- `utils/config.py`: Pydantic models for all configuration
- Configs in `configs/` use YAML format
- Main config: `configs/main_2024.yaml`

### Critical Design Patterns

1. **Numba Optimization**: Performance-critical code uses `@njit` decoration. When modifying numba functions, ensure type stability and avoid Python objects.

2. **Walk-Forward Validation**: The system uses rolling windows for training/testing to avoid lookahead bias:
   - Train on N days → Test on M days → Roll forward
   - Implemented in `pipeline/walk_forward_orchestrator.py`

3. **Cost Modeling**: Trading costs are aggregated to avoid double-counting:
   - `commission_pct`: Total commission (includes maker/taker fees)
   - `slippage_pct`: Total slippage (includes spread and execution slippage)
   - `enable_realistic_costs: false` по умолчанию; детализированные параметры используются только при явном включении

4. **Pair Selection Pipeline**:
   ```
   Load Data → Test Cointegration → Apply Filters → Rank by SSD → Select Top N
   ```

5. **Position Sizing**: Volatility-based dynamic sizing with risk limits per position

## Testing Philosophy

- **Test Coverage Required**: Any code change must include corresponding tests
- **Test Naming**: `test_<what>_when_<condition>_then_<result>`
- **Test Markers**: Use appropriate pytest markers (smoke, slow, serial, etc.)
- **Determinism**: Use fixed seeds for reproducibility

## Configuration Management

- **Main Config**: `configs/main_2024.yaml`
- **Strict Data QA Overlay**: `configs/data_quality_strict.yaml` (ffill-only, stricter thresholds)
- **Clean Data Window**: `configs/data_window_clean.yaml` (clean window + symbol exclusions)
- **Balanced WFA Config**: `configs/main_2024_wfa_balanced.yaml`
- **SSD top-N sweep**: `configs/ssd_topn_sweep_20260115/` (см. `docs/optimization_runs_20260115.md`)
- **SSD top-N sweep (subset 4 values)**: `configs/ssd_topn_sweep_20260115_4vals/` (см. `docs/optimization_runs_20260115.md`)
- **SSD top-N sweep (subset 3 values, 30k/40k/50k)**: `configs/ssd_topn_sweep_20260115_3vals/` (см. `docs/optimization_runs_20260115.md`)
- **Leader validation (post-analysis)**: `configs/leader_validation_20260116/` (см. `docs/optimization_runs_20260116.md`)
- **Leader holdout (next run)**: `configs/best_config__leader_holdout_ssd25000__20260116_211943.yaml` (см. `docs/optimization_runs_20260116.md`)
- **Search Spaces**: Define parameter ranges for optimization in `configs/search_spaces/`
- **Criteria**: Pair selection criteria in `configs/criteria_*.yaml`
- **Environment Variables**: `DATA_ROOT`, `COINT_LOG_EVERY`, `QUICK_TEST`

`data_filters.clean_window` и `data_filters.exclude_symbols` применяются в загрузчике данных и клипают окно/символы для WFA и backtest.
`walk_forward.max_steps` ограничивает число шагов WFA (по умолчанию 5, больше — только по согласованию).
`walk_forward.pairs_file` позволяет зафиксировать universe для WFA (например, top-200 из scan); CLI поддерживает `--pairs-file` и `--results-dir`.

Operational checklist: `docs/production_checklist.md`.

## Performance Considerations

- **Vectorization**: Use numpy/pandas operations, avoid Python loops in hot paths
- **Caching**: Data loading is cached via joblib Memory
- **Parallel Processing**: Use `joblib.Parallel` for pair processing
- **Memory Management**: Large datasets use memory-mapped arrays when possible
- **Pair Filtering**: WFA filters scale with `backtest.n_jobs` (default backend: threads; `COINT_FILTER_BACKEND=processes` uses spawn for OpenMP safety)

## Common Development Tasks

### Adding a New Filter
1. Add filter logic to `src/coint2/pipeline/filters.py`
2. Add configuration to `PairSelectionConfig` in `src/coint2/utils/config.py`
3. Add tests in `tests/unit/pipeline/`
4. Update configs in `configs/`

### Modifying Backtest Logic
1. Update both `base_engine.py` and `optimized_backtest_engine.py`
2. Run parity tests to ensure consistency
3. Update tests in `tests/integration/backtest/`

### Adding New Metrics
1. Implement in `src/optimiser/metric_utils.py`
2. Add to objective function if needed
3. Add unit tests in `tests/unit/optimiser/`

## Debugging Tips

- Set `QUICK_TEST=true` to use smaller datasets
- Use `--verbose` flag for detailed logging
- Check `outputs/studies/` for Optuna databases
- Use `scripts/run_preflight.py` and `scripts/perf_audit.py` for sanity checks
- Use `scripts/ui_preflight.py` for Streamlit dependencies/configs
- Enable `COINT_LOG_EVERY=100000` for progress logging in large operations

## Poetry CLI Commands

The project provides several CLI commands via Poetry:
```bash
# Main CLI (scan/backtest/walk-forward)
coint2

# Optimization CLI
coint2-optimize              # Run Optuna optimization

# Live trading CLI
coint2-live                  # Live trading interface

# Health check CLI
coint2-check-health          # Check system health

# Universe building CLI
coint2-build-universe        # Build pair universe
```

## Important Files

- `configs/main_2024.yaml`: Primary configuration file
- `src/optimiser/constants.py`: System-wide constants
- `pytest.ini`: Test configuration and markers
- `pyproject.toml`: Project dependencies and metadata
- `scripts/run_pipeline.sh`: End-to-end scan/backtest/walk-forward helper

## CI/CD Integration

- Smoke tests run on every commit
- Full test suite runs on PR
- Quality gates defined in `configs/ci_gates.yaml`

## Project Structure Notes

Many components have been archived during refactoring:
- Live trading: `archive/old_scripts/live/`
- Analysis tools: `archive/old_scripts/analysis/`
- Operations tools: `archive/old_scripts/ops/`
- CLI modules: `archive/old_modules/cli/`
- Test scripts: `archive/old_scripts/`

Active components remain in:
- Core optimization: `scripts/core/optimize.py`, `src/optimiser/run_optimization.py`
- Universe selection: `scripts/universe/`
- Pipeline helpers: `scripts/run_pipeline.sh`, `scripts/run_preflight.py`, `scripts/perf_audit.py`
- Main source: `src/coint2/` and `src/optimiser/`

Heartbeat watcher: поле `run_cpu` в `watch_wfa_queue.sh` теперь относится к воркеру `coint2 walk-forward`; отдельно пишутся `watcher_cpu`, `worker_pid`, `worker_cpu`.
