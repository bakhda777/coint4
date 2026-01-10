# Universe Selection Report

## Summary
- **Period**: 2024-01-01 to 2024-02-15
- **Total pairs tested**: 378
- **Pairs passed criteria**: 49
- **Pairs selected**: 10
- **Selection rate**: 13.0%

## Criteria Used
```yaml
beta_drift_max: 0.3
coint_pvalue_max: 0.2
hl_max: 500
hl_min: 3
hurst_max: 0.7
hurst_min: 0.1
min_cross: 5

```

## Distributions

### P-value Distribution
- Min: 0.0000
- Median: 0.0460
- Max: 0.8784
- Passed (<0.2): 284

### Half-life Distribution
- Min: 1.0
- Median: 282.1
- Max: 1000.0
- In range (3-500): 290

## Top 20 Selected Pairs

| Pair | Score | P-value | Half-life | Crossings | Beta Drift |
|------|-------|---------|-----------|-----------|------------|
| AAVEUSDT/AGLAUSDT | 2.341 | 0.0002 | 70.4 | 337 | 0.031 |
| 5IREUSDT/ARTYUSDT | 2.314 | 0.0001 | 81.4 | 191 | 0.037 |
| APEXUSDC/APEXUSDT | 1.939 | 0.0000 | 3.9 | 2384 | 0.012 |
| ACHUSDT/AFCUSDT | 1.915 | 0.0000 | 57.9 | 360 | 0.117 |
| AGLAUSDT/APTUSDT | 1.849 | 0.0163 | 68.6 | 340 | 0.098 |
| 5IREUSDT/ADAUSDC | 1.844 | 0.0073 | 155.2 | 151 | 0.017 |
| AGLAUSDT/APTUSDC | 1.808 | 0.0163 | 68.5 | 340 | 0.106 |
| 5IREUSDT/ADAUSDT | 1.775 | 0.0072 | 155.0 | 147 | 0.031 |
| APTUSDC/ARKMUSDT | 1.755 | 0.0155 | 215.3 | 192 | 0.018 |
| AGLAUSDT/AGLDUSDT | 1.681 | 0.0169 | 68.8 | 326 | 0.130 |


---
Generated: 2025-08-24T10:39:44.420761+00:00


## Rejection Breakdown

**Tested**: 378 pairs  
**Passed**: 49 pairs

### Top Rejection Reasons:
- **beta_drift**: 226 pairs (59.8%)
- **pvalue**: 94 pairs (24.9%)
- **half_life**: 9 pairs (2.4%)
