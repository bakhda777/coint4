# Universe Selection Report

## Summary
- **Period**: 2024-01-01 to 2024-01-15
- **Total pairs tested**: 42778
- **Pairs passed criteria**: 13591
- **Pairs selected**: 100
- **Selection rate**: 31.8%

## Criteria Used
```yaml
beta_drift_max: 1.0
coint_pvalue_max: 0.15
hl_max: 500
hl_min: 5
hurst_max: 0.999
hurst_min: 0.1
min_cross: 5

```

## Distributions

### P-value Distribution
- Min: 0.0000
- Median: 0.1072
- Max: 0.9991
- Passed (<0.15): 24238

### Half-life Distribution
- Min: 1.0
- Median: 333.0
- Max: inf
- In range (5-500): 28685

### Hurst Exponent Distribution
- Min: 0.523
- Median: 0.999
- Max: 1.000
- In range (0.1-0.999): 29508

## Top 20 Selected Pairs

| Pair | Score | P-value | Half-life | Hurst | Crossings | Beta Drift |
|------|-------|---------|-----------|-------|-----------|------------|
| CRVUSDT/MVLUSDT | 2.973 | 0.0000 | 30.6 | 0.988 | 525 | 0.005 |
| LUNAUSDT/WBTCBTC | 2.960 | 0.0001 | 45.1 | 0.986 | 125 | 0.008 |
| MVLUSDT/XEMUSDT | 2.943 | 0.0000 | 29.3 | 0.988 | 538 | 0.011 |
| BELUSDT/CITYUSDT | 2.940 | 0.0001 | 38.1 | 0.986 | 264 | 0.012 |
| ALGOBTC/FIDAUSDT | 2.917 | 0.0000 | 49.2 | 0.993 | 277 | 0.017 |
| AVAUSDT/SOLEUR | 2.880 | 0.0000 | 49.1 | 0.993 | 278 | 0.024 |
| COTUSDT/OASUSDT | 2.876 | 0.0000 | 25.2 | 0.986 | 238 | 0.025 |
| IZIUSDT/RUNEUSDT | 2.847 | 0.0000 | 31.7 | 0.988 | 374 | 0.031 |
| GALFTUSDT/SEIUSDT | 2.847 | 0.0007 | 34.7 | 0.990 | 150 | 0.029 |
| GALFTUSDT/SOLBTC | 2.834 | 0.0102 | 45.8 | 0.992 | 125 | 0.013 |
| JUVUSDT/SUSHIUSDT | 2.821 | 0.0000 | 49.2 | 0.993 | 247 | 0.036 |
| SWEATUSDT/XETAUSDT | 2.802 | 0.0000 | 35.8 | 0.990 | 319 | 0.040 |
| JUVUSDT/LINKUSDC | 2.786 | 0.0000 | 36.5 | 0.991 | 241 | 0.043 |
| IZIUSDT/SHRAPUSDT | 2.783 | 0.0000 | 21.5 | 0.984 | 258 | 0.043 |
| PSGUSDT/RPKUSDT | 2.776 | 0.0000 | 23.9 | 0.983 | 203 | 0.045 |
| JUVUSDT/LINKUSDT | 2.766 | 0.0000 | 36.5 | 0.991 | 237 | 0.047 |
| AGLAUSDT/BTCUSDT | 2.739 | 0.0000 | 31.3 | 0.989 | 359 | 0.052 |
| GLMRUSDT/PSGUSDT | 2.730 | 0.0006 | 35.8 | 0.988 | 223 | 0.053 |
| IZIUSDT/SIDUSUSDT | 2.728 | 0.0000 | 29.2 | 0.988 | 278 | 0.054 |
| CITYUSDT/NFTUSDT | 2.724 | 0.0000 | 22.2 | 0.984 | 345 | 0.055 |


---
Generated: 2025-08-14T14:07:36.587432+00:00


## Rejection Breakdown

**Tested**: 42778 pairs  
**Passed**: 13591 pairs

### Top Rejection Reasons:
- **pvalue**: 18540 pairs (43.3%)
- **beta_drift**: 7815 pairs (18.3%)
- **hurst**: 1893 pairs (4.4%)
- **half_life**: 938 pairs (2.2%)
- **crossings**: 1 pairs (0.0%)
