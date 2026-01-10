# Universe Selection Report

## Summary
- **Period**: 2024-01-01 to 2024-03-31
- **Total pairs tested**: 12090
- **Pairs passed criteria**: 644
- **Pairs selected**: 40
- **Selection rate**: 5.3%

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
- Median: 0.0571
- Max: 1.0000
- Passed (<0.1): 7066

### Half-life Distribution
- Min: 1.0
- Median: 293.2
- Max: inf
- In range (5-300): 6098

## Top 20 Selected Pairs

| Pair | Score | P-value | Half-life | Crossings | Beta Drift |
|------|-------|---------|-----------|-----------|------------|
| ETHUSDC/ETHUSDT | 2.988 | 0.0000 | 24.5 | 615 | 0.002 |
| BTCUSDC/BTCUSDT | 2.972 | 0.0000 | 24.1 | 662 | 0.006 |
| AFCUSDT/FIDAUSDT | 2.872 | 0.0001 | 47.4 | 192 | 0.026 |
| BATUSDT/CBKUSDT | 2.854 | 0.0000 | 30.7 | 381 | 0.029 |
| APEUSDT/KUBUSDT | 2.744 | 0.0000 | 44.4 | 383 | 0.051 |
| APEUSDC/KUBUSDT | 2.740 | 0.0000 | 42.6 | 421 | 0.052 |
| DMAILUSDT/DUELUSDT | 2.701 | 0.0000 | 47.5 | 121 | 0.060 |
| DMAILUSDT/FLOWUSDT | 2.672 | 0.0000 | 39.6 | 154 | 0.066 |
| AKIUSDT/FIREUSDT | 2.636 | 0.0010 | 45.8 | 195 | 0.071 |
| ETHDAI/KUBUSDT | 2.603 | 0.0005 | 45.6 | 205 | 0.079 |
| KCALUSDT/KDAUSDT | 2.489 | 0.0000 | 76.4 | 326 | 0.002 |
| AXSUSDT/CBKUSDT | 2.486 | 0.0001 | 45.4 | 390 | 0.103 |
| AFCUSDT/JUVUSDT | 2.483 | 0.0000 | 21.4 | 213 | 0.103 |
| ETHUSDT/FORTUSDT | 2.464 | 0.0001 | 97.8 | 248 | 0.007 |
| ETHUSDC/FORTUSDT | 2.462 | 0.0001 | 98.6 | 244 | 0.007 |
| AFGUSDT/FETUSDT | 2.456 | 0.0000 | 56.4 | 163 | 0.009 |
| CRVUSDT/KDAUSDT | 2.453 | 0.0000 | 78.4 | 333 | 0.009 |
| ACSUSDT/FLOWUSDT | 2.453 | 0.0000 | 78.1 | 216 | 0.009 |
| BTTUSDT/JUVUSDT | 2.439 | 0.0002 | 47.1 | 244 | 0.112 |
| CRVUSDT/HFTUSDC | 2.436 | 0.0000 | 83.0 | 295 | 0.013 |


---
Generated: 2025-08-16T11:47:24.468355+00:00


## Rejection Breakdown

**Tested**: 12090 pairs  
**Passed**: 644 pairs

### Top Rejection Reasons:
- **pvalue**: 5024 pairs (41.6%)
- **beta_drift**: 4981 pairs (41.2%)
- **half_life**: 1441 pairs (11.9%)
