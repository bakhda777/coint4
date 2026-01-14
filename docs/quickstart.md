# Quickstart

This guide walks you through preparing data, configuring the project and running the pipeline.
All paths assume you run commands from `coint4/` (start with `cd coint4`).

## 1. Prepare data

Price data should be stored as partitioned parquet files inside the directory specified by `data_dir` (by default `data_downloaded`). The current dataset uses a monthly consolidated layout (one file per month with all symbols inside):

```
data_downloaded/
  year=2024/
    month=01/
      data_part_01.parquet
```

Each file contains multiple symbols; filter by the `symbol` column.
If `data_optimized/` exists next to `data_downloaded/`, the loader will prefer it automatically (same layout).
Legacy per-symbol/day layouts are still supported — see `docs/data_structure.md` (archived section) for details.
At minimum the parquet files must contain `timestamp`, `symbol`, and `close` (full OHLCV is supported too).

## 2. Configure

Edit `configs/main_2024.yaml` to match your dataset location and desired backtest parameters. Important options:

```yaml
data_dir: "data_downloaded"
results_dir: "results"
data_filters:
  clean_window:
    start_date: "2022-03-01"
    end_date: "2025-06-30"
  exclude_symbols: []
pair_selection:
  lookback_days: 90
  coint_pvalue_threshold: 0.05
backtest:
  timeframe: "1d"
  rolling_window: 30
  zscore_threshold: 1.5
  fill_limit_pct: 0.2  # доля подряд идущих пропусков, заполняемых только ffill
  enable_realistic_costs: false
max_shards: null
```

Для строгих проверок качества данных используйте overlay:

```bash
./.venv/bin/coint2 backtest \
  --config configs/main_2024.yaml \
  --config-delta configs/data_quality_strict.yaml \
  --pairs-file bench/pairs_universe.yaml \
  --period-start 2023-06-01 \
  --period-end 2023-08-31 \
  --out-dir outputs/fixed_run
```

Рекомендуемое «чистое» окно (исключает проблемные месяцы): `2022-03-01` → `2025-06-30`.
Оно закреплено в `data_filters.clean_window` и дублируется в `configs/data_window_clean.yaml`.
Для WFA `walk_forward.start_date` должен учитывать training‑период (при 60 днях — `2022-04-30`).
Для fixed backtest задавайте даты через `--period-start/--period-end` внутри clean window.

## 3. Run the project

```bash
# Scan pairs
./.venv/bin/coint2 scan \
  --config configs/criteria_relaxed.yaml \
  --base-config configs/main_2024.yaml \
  --output-dir bench

# Backtest scanned universe
./.venv/bin/coint2 backtest \
  --config configs/main_2024.yaml \
  --pairs-file bench/pairs_universe.yaml \
  --period-start 2023-06-01 \
  --period-end 2023-08-31 \
  --out-dir outputs/fixed_run

# Walk-forward analysis
./.venv/bin/coint2 walk-forward \
  --config configs/main_2024.yaml
```

`walk_forward.max_steps` ограничивает количество шагов WFA (по умолчанию 5 в `configs/main_2024.yaml` и `configs/main_2024_wfa_balanced.yaml`).

Для более быстрых проверок WFA используйте сбалансированный конфиг:
```bash
./.venv/bin/coint2 walk-forward \
  --config configs/main_2024_wfa_balanced.yaml
```

You can also run the full sequence with `bash scripts/run_pipeline.sh`.
Fixed backtest outputs go to `--out-dir`, and walk-forward outputs go to `results_dir`.

## Fast iteration (smoke)

Для быстрой наладки используйте smoke-конфиг, короткие окна и ограниченный список символов:

```bash
./.venv/bin/coint2 scan \
  --config configs/criteria_relaxed.yaml \
  --base-config configs/main_2024_smoke.yaml \
  --symbols-file configs/symbols_fast20.yaml \
  --train-days 30 \
  --valid-days 10 \
  --end-date 2025-06-30 \
  --top-n 50 \
  --output-dir bench/fast_iter

./.venv/bin/coint2 backtest \
  --config configs/main_2024_smoke.yaml \
  --pairs-file bench/fast_iter/pairs_universe.yaml \
  --period-start 2023-06-01 \
  --period-end 2023-06-30 \
  --max-bars 3000 \
  --out-dir outputs/fixed_run_fast_iter

./.venv/bin/coint2 walk-forward \
  --config configs/main_2024_smoke.yaml
```

Для расширенной быстрой итерации (50 символов, top-N=100):
```bash
./.venv/bin/coint2 scan \
  --config configs/criteria_relaxed.yaml \
  --base-config configs/main_2024_smoke.yaml \
  --symbols-file configs/symbols_fast50.yaml \
  --train-days 45 \
  --valid-days 15 \
  --end-date 2025-06-30 \
  --top-n 100 \
  --output-dir bench/fast_iter_top100
```
