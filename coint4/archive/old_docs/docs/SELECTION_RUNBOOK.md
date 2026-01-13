> NOTE: Archived/legacy document. It may describe historical behavior and can be out of date. See `docs/` for current usage.

# Universe Selection Runbook

## Quick Start Commands

### 0. Preparation
```bash
# Create necessary directories
mkdir -p artifacts/universe artifacts/optuna benchmarks configs/windows
```

### 1. Scan Data Inventory
```bash
# Scan available data to determine date ranges and symbols
python scripts/scan_data_inventory.py \
  --data-root ./data_downloaded \
  --timeframe 15T

# Output: artifacts/universe/DATA_SCAN.json
#         artifacts/universe/DATA_SCAN.md
```

### 2. Generate Window Configurations
```bash
# Generate time windows with strict and relaxed criteria
python scripts/generate_universe_windows.py \
  --data-root ./data_downloaded \
  --timeframe 15T \
  --train-days 120 \
  --valid-days 30 \
  --num-windows 4

# Output: configs/windows/universe_win*_*.yaml
#         configs/windows/index.json
```

### 3. Run Universe Selection for Each Window
```bash
# Sequential execution (recommended for debugging)
for cfg in configs/windows/*.yaml; do
  label=$(basename "$cfg" .yaml)
  mkdir -p "artifacts/universe/$label"
  echo "🔄 Processing $label..."
  python scripts/data/build_universe.py --config "$cfg" \
    2>&1 | tee "artifacts/universe/$label/run.log"
done

# Parallel execution (for faster processing)
# Use GNU parallel if available:
ls configs/windows/*.yaml | parallel -j 4 '
  label=$(basename {} .yaml)
  mkdir -p artifacts/universe/$label
  python scripts/data/build_universe.py --config {} \
    2>&1 | tee artifacts/universe/$label/run.log
'
```

### 4. Aggregate Stable Pairs
```bash
# Aggregate pairs that pass in multiple windows
python scripts/aggregate_universe.py \
  --inputs "artifacts/universe/*/universe_metrics.csv" \
  --out-yaml benchmarks/pairs_universe.yaml \
  --out-csv artifacts/universe/AGGREGATED_PAIRS.csv \
  --min-windows 2 \
  --top-n 20 \
  --diversify-by-base true \
  --max-per-base 5

# Output: benchmarks/pairs_universe.yaml
#         artifacts/universe/AGGREGATED_PAIRS.csv
#         artifacts/universe/AGGREGATED_REPORT.md
```

### 5. Validate Results
```bash
# Validate the final pairs YAML
python scripts/validate_yaml.py benchmarks/pairs_universe.yaml
```

### 6. Run Optuna Optimization (Optional)

#### Quick Canary Test (25 trials)
```bash
python scripts/run_optuna_real.py \
  --study-name canary_real \
  --n-trials 25 \
  --data-root ./data_downloaded \
  --period-start 2024-01-01 \
  --period-end 2024-03-31 \
  --timeframe 15T \
  --pairs-file benchmarks/pairs_universe.yaml \
  --k-folds 3
```

#### Full Portfolio Optimization (150 trials)
```bash
python scripts/run_optuna_real.py \
  --study-name portfolio_real \
  --n-trials 150 \
  --data-root ./data_downloaded \
  --period-start 2024-01-01 \
  --period-end 2024-06-30 \
  --timeframe 15T \
  --pairs-file benchmarks/pairs_universe.yaml \
  --k-folds 5 \
  --resume
```

## Troubleshooting

### If no pairs pass selection:

1. **Relax criteria** - Edit configs/windows/*.yaml:
   - Increase `coint_pvalue_max` to 0.20
   - Decrease `min_cross` to 5
   - Expand `hurst_min/max` to 0.05-0.85
   - Increase `beta_drift_max` to 0.30

2. **Reduce aggregation threshold**:
   ```bash
   # Allow pairs that pass in just 1 window
   python scripts/aggregate_universe.py \
     --inputs "artifacts/universe/*/universe_metrics.csv" \
     --out-yaml benchmarks/pairs_universe.yaml \
     --min-windows 1 \
     --top-n 30
   ```

3. **Check data availability**:
   ```bash
   # Verify symbols in the period
   python -c "
   import pandas as pd
   df = pd.read_parquet('data_downloaded/year=2024/month=01/data_part_01.parquet', columns=['symbol'])
   print(f'Symbols: {df.symbol.nunique()}')
   print(df.symbol.value_counts().head(20))
   "
   ```

### If YAML validation fails:

1. Check for EOF markers:
   ```bash
   grep -n "EOF" benchmarks/pairs_universe.yaml
   ```

2. Clean and re-run aggregation:
   ```bash
   rm benchmarks/pairs_universe.yaml
   python scripts/aggregate_universe.py --min-windows 1
   ```

## Parameters Reference

### Window Generation
- `--num-windows`: Number of time windows (default: 4)
- `--train-days`: Training period length (default: 120)
- `--valid-days`: Validation period length (default: 30)

### Aggregation
- `--min-windows`: Minimum windows a pair must pass (default: 2)
- `--top-n`: Maximum pairs to select (default: 20)
- `--diversify-by-base`: Enable base asset diversification (default: true)
- `--max-per-base`: Max pairs per base asset (default: 5)

### Criteria Profiles

**Strict** (higher quality, fewer pairs):
- `coint_pvalue_max`: 0.05
- `hl_min/max`: 10-150
- `hurst_min/max`: 0.2-0.5
- `min_cross`: 15
- `beta_drift_max`: 0.10

**Relaxed** (more pairs, lower quality):
- `coint_pvalue_max`: 0.15
- `hl_min/max`: 5-300
- `hurst_min/max`: 0.1-0.7
- `min_cross`: 8
- `beta_drift_max`: 0.25

## Output Structure
```
artifacts/universe/
├── DATA_SCAN.json               # Data inventory
├── DATA_SCAN.md                 # Data report
├── win1_strict/                 # Window results
│   ├── pairs.yaml
│   ├── universe_metrics.csv
│   └── REPORT.md
├── win1_relaxed/
├── ...
├── AGGREGATED_PAIRS.csv         # Final aggregated metrics
└── AGGREGATED_REPORT.md         # Final report

benchmarks/
└── pairs_universe.yaml          # Final pairs for Optuna

configs/windows/
├── index.json                   # Window configurations index
├── universe_win1_strict.yaml
├── universe_win1_relaxed.yaml
└── ...
```

---
Generated: 2025-08-11