# Universe Selection Report

## Summary
- **Period**: 2024-01-01 to 2024-03-31
- **Total pairs tested**: 56280
- **Pairs passed criteria**: 2415
- **Pairs selected**: 200
- **Selection rate**: 4.3%

## Criteria Used
```yaml
beta_drift_max: 0.2
coint_pvalue_max: 0.1
hl_max: 300
hl_min: 5
min_cross: 8

```

## Distributions

### P-value Distribution
- Min: 0.0000
- Median: 0.0740
- Max: 1.0000
- Passed (<0.1): 30841

### Half-life Distribution
- Min: 1.0
- Median: 311.1
- Max: inf
- In range (5-300): 26752

## Top 20 Selected Pairs

| Pair | Score | P-value | Half-life | Crossings | Beta Drift |
|------|-------|---------|-----------|-----------|------------|
| ETHUSDC/ETHUSDT | 2.988 | 0.0000 | 24.5 | 615 | 0.002 |
| TUSDUSDT/WEMIXUSDT | 2.979 | 0.0017 | 33.7 | 134 | 0.001 |
| BTCUSDC/BTCUSDT | 2.972 | 0.0000 | 24.1 | 662 | 0.006 |
| FIDAUSDT/PSGUSDT | 2.941 | 0.0000 | 49.3 | 261 | 0.012 |
| KASTAUSDT/TUSDUSDT | 2.886 | 0.0027 | 47.0 | 246 | 0.017 |
| TUSDUSDT/UNIUSDT | 2.877 | 0.0065 | 41.8 | 224 | 0.012 |
| AFCUSDT/FIDAUSDT | 2.872 | 0.0001 | 47.4 | 192 | 0.026 |
| LTCBTC/METHETH | 2.867 | 0.0002 | 29.8 | 148 | 0.026 |
| USDTBRZ/XLMBTC | 2.860 | 0.0000 | 21.7 | 418 | 0.028 |
| BATUSDT/CBKUSDT | 2.854 | 0.0000 | 30.7 | 381 | 0.029 |
| PSTAKEUSDT/TELUSDT | 2.841 | 0.0000 | 42.3 | 275 | 0.032 |
| TUSDUSDT/XLMUSDC | 2.785 | 0.0005 | 29.4 | 237 | 0.042 |
| TUSDUSDT/XLMUSDT | 2.781 | 0.0005 | 29.2 | 263 | 0.043 |
| HBARUSDT/PYUSDUSDT | 2.781 | 0.0000 | 41.4 | 215 | 0.044 |
| AFCUSDT/UNIUSDT | 2.773 | 0.0000 | 25.5 | 355 | 0.045 |
| DMAILUSDT/SUNUSDT | 2.759 | 0.0000 | 35.3 | 140 | 0.048 |
| CBKUSDT/OMGUSDT | 2.752 | 0.0000 | 37.7 | 356 | 0.050 |
| CBKUSDT/MANAUSDC | 2.750 | 0.0006 | 49.4 | 291 | 0.049 |
| APEUSDT/KUBUSDT | 2.744 | 0.0000 | 44.4 | 383 | 0.051 |
| APEUSDC/KUBUSDT | 2.740 | 0.0000 | 42.6 | 421 | 0.052 |


---
Generated: 2025-08-15T21:59:39.397342+00:00


## Rejection Breakdown

**Tested**: 56280 pairs  
**Passed**: 2415 pairs

### Top Rejection Reasons:
- **pvalue**: 25439 pairs (45.2%)
- **beta_drift**: 22038 pairs (39.2%)
- **half_life**: 6388 pairs (11.4%)
