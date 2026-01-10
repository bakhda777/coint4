# Universe Selection Report

## Summary
- **Period**: 2024-04-01 to 2024-05-15
- **Total pairs tested**: 5886
- **Pairs passed criteria**: 424
- **Pairs selected**: 30
- **Selection rate**: 7.2%

## Criteria Used
```yaml
beta_drift_max: 0.2
coint_pvalue_max: 0.1
hl_max: 300
hl_min: 5
hurst_max: 0.65
hurst_min: 0.15
min_cross: 8

```

## Distributions

### P-value Distribution
- Min: 0.0000
- Median: 0.1205
- Max: 0.9318
- Passed (<0.1): 2731

### Half-life Distribution
- Min: 1.0
- Median: 406.9
- Max: 1000.0
- In range (5-300): 2272

## Top 20 Selected Pairs

| Pair | Score | P-value | Half-life | Crossings | Beta Drift |
|------|-------|---------|-----------|-----------|------------|
| ETHUSDC/ETHUSDT | 2.997 | 0.0000 | 20.6 | 763 | 0.001 |
| BTCUSDC/BTCUSDT | 2.990 | 0.0000 | 23.9 | 699 | 0.002 |
| AAVEUSDT/ACSUSDT | 2.864 | 0.0000 | 44.7 | 387 | 0.027 |
| AVAXUSDT/BRAWLUSDT | 2.764 | 0.0000 | 44.2 | 120 | 0.047 |
| AVAXUSDC/BRAWLUSDT | 2.710 | 0.0000 | 44.1 | 122 | 0.058 |
| APRSUSDT/APTUSDC | 2.502 | 0.0002 | 47.5 | 113 | 0.099 |
| ALGOUSDT/ERTHAUSDT | 2.495 | 0.0000 | 99.8 | 214 | 0.001 |
| DEGENUSDT/DGBUSDT | 2.490 | 0.0004 | 72.8 | 166 | 0.001 |
| ARTYUSDT/ENSUSDT | 2.478 | 0.0000 | 72.2 | 258 | 0.004 |
| ARTYUSDT/EGLDUSDT | 2.478 | 0.0000 | 54.3 | 332 | 0.004 |
| APRSUSDT/APTUSDT | 2.475 | 0.0000 | 47.7 | 113 | 0.105 |
| AEGUSDT/ERTHAUSDT | 2.463 | 0.0000 | 80.0 | 177 | 0.007 |
| ENJUSDT/ETHDAI | 2.453 | 0.0004 | 57.7 | 247 | 0.009 |
| APEXUSDC/CAKEUSDT | 2.435 | 0.0011 | 95.3 | 205 | 0.011 |
| AVAUSDT/ERTHAUSDT | 2.429 | 0.0000 | 81.1 | 272 | 0.014 |
| BOMEUSDT/CGPTUSDT | 2.356 | 0.0013 | 77.1 | 102 | 0.026 |
| AAVEUSDT/BRAWLUSDT | 2.325 | 0.0000 | 62.0 | 188 | 0.035 |
| AEGUSDT/DOMEUSDT | 2.305 | 0.0080 | 70.6 | 139 | 0.023 |
| APRSUSDT/CELOUSDT | 2.244 | 0.0003 | 63.2 | 177 | 0.051 |
| BTCBRZ/BTCUSDT | 2.240 | 0.0006 | 28.9 | 275 | 0.151 |


---
Generated: 2025-08-24T13:14:25.217143+00:00


## Rejection Breakdown

**Tested**: 5886 pairs  
**Passed**: 424 pairs

### Top Rejection Reasons:
- **pvalue**: 3155 pairs (53.6%)
- **beta_drift**: 1618 pairs (27.5%)
- **half_life**: 689 pairs (11.7%)
