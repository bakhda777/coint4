# Universe Selection Report

## Summary
- **Period**: 2023-07-01 to 2024-04-30
- **Total pairs tested**: 13203
- **Pairs passed criteria**: 515
- **Pairs selected**: 250
- **Selection rate**: 3.9%

## Criteria Used
```yaml
beta_drift_max: 0.15
coint_pvalue_max: 0.05
hl_max: 200
hl_min: 5
min_cross: 10

```

## Distributions

### P-value Distribution
- Min: 0.0000
- Median: 0.2101
- Max: 0.9908
- Passed (<0.05): 3574

### Half-life Distribution
- Min: 1.0
- Median: 474.8
- Max: inf
- In range (5-200): 2456

## Top 20 Selected Pairs

| Pair | Score | P-value | Half-life | Crossings | Beta Drift |
|------|-------|---------|-----------|-----------|------------|
| BTCUSDC/BTCUSDT | 2.988 | 0.0000 | 23.2 | 693 | 0.002 |
| GALFTUSDT/HVHUSDT | 2.980 | 0.0000 | 36.2 | 249 | 0.004 |
| ETHUSDC/ETHUSDT | 2.979 | 0.0000 | 21.3 | 741 | 0.004 |
| GALFTUSDT/GRTUSDT | 2.951 | 0.0000 | 28.9 | 205 | 0.010 |
| BATUSDT/CBKUSDT | 2.888 | 0.0003 | 47.6 | 365 | 0.022 |
| AGLAUSDT/CSPRUSDT | 2.785 | 0.0001 | 48.6 | 194 | 0.043 |
| CITYUSDT/KASTAUSDT | 2.785 | 0.0000 | 47.9 | 347 | 0.043 |
| DMAILUSDT/GALFTUSDT | 2.685 | 0.0000 | 47.5 | 319 | 0.063 |
| AFCUSDT/BICOUSDT | 2.612 | 0.0000 | 36.2 | 250 | 0.078 |
| GALFTUSDT/IZIUSDT | 2.541 | 0.0001 | 47.1 | 337 | 0.092 |
| ETCUSDT/JUVUSDT | 2.541 | 0.0000 | 45.1 | 318 | 0.092 |
| DOTUSDT/KDAUSDT | 2.507 | 0.0000 | 48.4 | 396 | 0.099 |
| DYDXUSDT/KCALUSDT | 2.493 | 0.0001 | 89.3 | 356 | 0.001 |
| CITYUSDT/COQUSDT | 2.492 | 0.0005 | 69.4 | 228 | 0.001 |
| DOTUSDC/KDAUSDT | 2.491 | 0.0000 | 48.0 | 404 | 0.102 |
| ACSUSDT/AGLDUSDT | 2.490 | 0.0000 | 98.8 | 178 | 0.002 |
| AFCUSDT/CITYUSDT | 2.483 | 0.0008 | 41.2 | 380 | 0.102 |
| CTCUSDT/GALFTUSDT | 2.463 | 0.0011 | 50.7 | 172 | 0.005 |
| 1INCHUSDT/ACSUSDT | 2.444 | 0.0000 | 68.3 | 376 | 0.011 |
| APEXUSDT/AVAUSDT | 2.440 | 0.0000 | 58.7 | 357 | 0.012 |


---
Generated: 2026-01-19T15:54:58.046412+00:00


## Rejection Breakdown

**Tested**: 13203 pairs  
**Passed**: 515 pairs

### Top Rejection Reasons:
- **pvalue**: 9629 pairs (72.9%)
- **beta_drift**: 1607 pairs (12.2%)
- **half_life**: 1452 pairs (11.0%)
