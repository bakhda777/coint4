# Candidate: relaxed8_nokpss_u250 (2026-01-19)

## Summary
- Candidate: relaxed8_nokpss_u250 (fixed universe, KPSS disabled)
- Pairs file: `coint4/artifacts/universe/20260119_relaxed8_strict_preholdout_v2/pairs_universe.yaml`
- Holdout period: 2024-05-01 -> 2024-12-31 (train=90d, w2m1)

## Metrics
Holdout (u250):
- Sharpe: 4.2025
- PnL: 421.60
- Max DD: -72.64
- Trades: 6572
- Pairs traded: 168
- Costs: 174.35
- Win rate: 0.6225

Stress holdout (u250):
- Sharpe: 2.8931
- PnL: 289.30
- Max DD: -75.32
- Trades: 6572
- Pairs traded: 168
- Costs: 309.95
- Win rate: 0.5828

## Concentration (by pair PnL)
Holdout:
- Top 1 share: 0.1562
- Top 5 share: 0.4340
- Top 10 share: 0.6381

Stress holdout:
- Top 1 share: 0.2270
- Top 5 share: 0.6127
- Top 10 share: 0.8871

## Decision
- Passes Sharpe > 1 and pairs >= 100 in both holdout and stress.
- Concentration rises under stress (top 10 ~88.7% of PnL). Consider adding concentration limits or risk caps before paper/live.
