# Quickstart

This guide walks you through preparing data, configuring the project and running the pipeline.
All paths assume you run commands from `coint4/` (start with `cd coint4`).

## 1. Prepare data

Price data should be stored as partitioned parquet files inside the directory specified by `data_dir` (by default `data_downloaded`). Each symbol lives in its own partition:

```
data_downloaded/
  BTCUSDT/
    year=2024/
      month=01/
        day=01/
          data.parquet
```

If `data_optimized/` exists next to `data_downloaded/`, the loader will prefer it automatically.
At minimum the parquet files must contain `timestamp`, `symbol`, and `close` (full OHLCV is supported too).

## 2. Configure

Edit `configs/main_2024.yaml` to match your dataset location and desired backtest parameters. Important options:

```yaml
data_dir: "data_downloaded"
results_dir: "results"
pair_selection:
  lookback_days: 90
  coint_pvalue_threshold: 0.05
backtest:
  timeframe: "1d"
  rolling_window: 30
  zscore_threshold: 1.5
  fill_limit_pct: 0.2  # доля подряд идущих пропусков, заполняемых при ffill/bfill
max_shards: null
```

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

You can also run the full sequence with `bash scripts/run_pipeline.sh`.
Fixed backtest outputs go to `--out-dir`, and walk-forward outputs go to `results_dir`.
