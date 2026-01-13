> NOTE: Archived/legacy document. It may describe historical behavior and can be out of date. See `docs/` for current usage.

# Universe Selection Report

## Summary
- **Period**: 2024-01-01 to 2024-03-31
- **Total pairs tested**: 276
- **Pairs passed criteria**: 21
- **Pairs selected**: 21
- **Selection rate**: 7.6%

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
- Median: 0.1616
- Max: 0.9237
- Passed (<0.1): 122

### Half-life Distribution
- Min: 1.0
- Median: 441.9
- Max: 1000.0
- In range (5-300): 76

## Top 20 Selected Pairs

| Pair | Score | P-value | Half-life | Crossings | Beta Drift |
|------|-------|---------|-----------|-----------|------------|
| ETHUSDC/ETHUSDT | 2.988 | 0.0000 | 24.5 | 615 | 0.002 |
| BTCUSDC/BTCUSDT | 2.972 | 0.0000 | 24.1 | 662 | 0.006 |
| ETHEUR/ETHUSDT | 2.415 | 0.0325 | 26.2 | 349 | 0.052 |
| ETHEUR/ETHUSDC | 2.342 | 0.0388 | 27.3 | 387 | 0.054 |
| SOLEUR/SOLUSDC | 2.272 | 0.0133 | 13.9 | 749 | 0.019 |
| SOLEUR/SOLUSDT | 2.203 | 0.0176 | 13.0 | 675 | 0.024 |
| BTCEUR/BTCUSDT | 2.186 | 0.0532 | 24.1 | 513 | 0.056 |
| BTCEUR/BTCUSDC | 2.093 | 0.0653 | 25.6 | 565 | 0.051 |
| LTCEUR/LTCUSDT | 2.035 | 0.0381 | 18.1 | 467 | 0.017 |
| LTCEUR/LTCUSDC | 1.991 | 0.0431 | 18.5 | 577 | 0.016 |
| ADAEUR/ADAUSDT | 1.936 | 0.0063 | 7.7 | 709 | 0.000 |
| ADAEUR/ADAUSDC | 1.908 | 0.0086 | 7.8 | 777 | 0.001 |
| ADAEUR/ETHUSDC | 1.896 | 0.0032 | 172.0 | 157 | 0.014 |
| XRPEUR/XRPUSDT | 1.885 | 0.0276 | 17.2 | 585 | 0.068 |
| XRPEUR/XRPUSDC | 1.880 | 0.0275 | 17.5 | 633 | 0.069 |
| ADAEUR/ETHUSDT | 1.832 | 0.0034 | 174.0 | 165 | 0.027 |
| ADAUSDT/ETHUSDC | 1.443 | 0.0049 | 229.7 | 187 | 0.102 |
| ADAUSDC/ETHUSDC | 1.420 | 0.0053 | 232.6 | 193 | 0.105 |
| ADAUSDT/ETHUSDT | 1.368 | 0.0051 | 232.7 | 193 | 0.116 |
| ADAUSDC/ETHUSDT | 1.345 | 0.0056 | 235.6 | 199 | 0.120 |


---
Generated: 2025-08-17T10:02:31.841805+00:00


## Rejection Breakdown

**Tested**: 276 pairs  
**Passed**: 21 pairs

### Top Rejection Reasons:
- **pvalue**: 154 pairs (55.8%)
- **beta_drift**: 53 pairs (19.2%)
- **half_life**: 48 pairs (17.4%)
