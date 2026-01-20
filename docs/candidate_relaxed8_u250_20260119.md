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

## Top20 + turnover (max_pairs=20)
Holdout (z1.05 / exit 0.08 / hold 120 / cd 120):
- Sharpe: 3.8534
- PnL: 128.71
- Max DD: -21.51
- Trades: 630
- Pairs traded: 53
- Costs: 18.04

Stress (z1.05 / exit 0.08 / hold 120 / cd 120):
- Sharpe: 3.4392
- PnL: 114.68
- Max DD: -23.52
- Trades: 630
- Pairs traded: 53
- Costs: 32.07

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
- Baseline u250 passes Sharpe > 1 and pairs >= 100 in both holdout and stress.
- Concentration rises under stress (top 10 ~88.7% of PnL). Consider adding concentration limits or risk caps before paper/live.
- Recommendation: switch to top20 + turnover (z1.05/0.08/120/120) as the main candidate; Sharpe > 3 with trades down to 630 (~4.17/day), pairs 53.
