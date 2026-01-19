# Universe Selection Report

## Summary
- **Period**: 2023-07-01 to 2024-04-30
- **Total pairs tested**: 5356
- **Pairs passed criteria**: 168
- **Pairs selected**: 110
- **Selection rate**: 3.1%

## Criteria Used
```yaml
beta_drift_max: 0.15
coint_pvalue_max: 0.05
hl_max: 200
hl_min: 5
hurst_max: 0.6
hurst_min: 0.2
min_cross: 10

```

## Distributions

### P-value Distribution
- Min: 0.0000
- Median: 0.2361
- Max: 0.9908
- Passed (<0.05): 1363

### Half-life Distribution
- Min: 1.0
- Median: 493.7
- Max: inf
- In range (5-200): 957

## Top 20 Selected Pairs

| Pair | Score | P-value | Half-life | Crossings | Beta Drift |
|------|-------|---------|-----------|-----------|------------|
| BTCUSDC/BTCUSDT | 2.988 | 0.0000 | 23.2 | 693 | 0.002 |
| ETHUSDC/ETHUSDT | 2.979 | 0.0000 | 21.3 | 741 | 0.004 |
| BATUSDT/CBKUSDT | 2.888 | 0.0003 | 47.6 | 365 | 0.022 |
| AGLAUSDT/CSPRUSDT | 2.785 | 0.0001 | 48.6 | 194 | 0.043 |
| AFCUSDT/BICOUSDT | 2.612 | 0.0000 | 36.2 | 250 | 0.078 |
| CITYUSDT/COQUSDT | 2.492 | 0.0005 | 69.4 | 228 | 0.001 |
| ACSUSDT/AGLDUSDT | 2.490 | 0.0000 | 98.8 | 178 | 0.002 |
| AFCUSDT/CITYUSDT | 2.483 | 0.0008 | 41.2 | 380 | 0.102 |
| 1INCHUSDT/ACSUSDT | 2.444 | 0.0000 | 68.3 | 376 | 0.011 |
| APEXUSDT/AVAUSDT | 2.440 | 0.0000 | 58.7 | 357 | 0.012 |
| APEXUSDC/AVAUSDT | 2.419 | 0.0000 | 53.6 | 348 | 0.016 |
| AFCUSDT/C98USDT | 2.418 | 0.0052 | 85.9 | 228 | 0.006 |
| 1INCHUSDT/AXSUSDT | 2.417 | 0.0000 | 94.2 | 223 | 0.017 |
| AEGUSDT/DOMEUSDT | 2.391 | 0.0028 | 69.4 | 115 | 0.016 |
| AAVEUSDT/ERTHAUSDT | 2.375 | 0.0000 | 90.7 | 241 | 0.025 |
| ACSUSDT/ATOMUSDT | 2.353 | 0.0002 | 95.1 | 163 | 0.029 |
| AGLAUSDT/APPUSDT | 2.341 | 0.0106 | 98.7 | 218 | 0.011 |
| ARKMUSDT/ETHDAI | 2.296 | 0.0002 | 82.5 | 158 | 0.041 |
| CRVUSDT/DOTUSDT | 2.248 | 0.0000 | 93.3 | 295 | 0.050 |
| CRVUSDT/DOTUSDC | 2.230 | 0.0000 | 91.7 | 289 | 0.054 |


---
Generated: 2026-01-19T14:20:24.285223+00:00


## Rejection Breakdown

**Tested**: 5356 pairs  
**Passed**: 168 pairs

### Top Rejection Reasons:
- **pvalue**: 3993 pairs (74.6%)
- **beta_drift**: 624 pairs (11.7%)
- **half_life**: 571 pairs (10.7%)
