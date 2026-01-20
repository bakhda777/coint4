# Candidate: relaxed8_nokpss_u250 (2026-01-22)

## Summary
- Candidate: relaxed8_nokpss_u250 (fixed universe, KPSS disabled)
- Pairs file: `coint4/artifacts/universe/20260119_relaxed8_strict_preholdout_v2/pairs_universe.yaml`
- Holdout period: 2024-05-01 -> 2024-12-31 (train=90d, w2m1)
- Обновление: churnfix grid после фикса min_spread_move_sigma.

## Metrics (churnfix_v3 winner)
Holdout (z0.95 / exit 0.06 / hold 180 / cd 180 / ms 0.1):
- Sharpe: 6.9996
- PnL: 1069.34
- Trades: 20236
- Pairs traded: 168
- Costs: 521.98

Stress holdout (z0.95 / exit 0.06 / hold 180 / cd 180 / ms 0.1):
- Sharpe: 5.8412
- PnL: 892.62
- Trades: 20236
- Pairs traded: 168
- Costs: 927.96

## Tradeoff notes
- z0.90/exit0.08 дает выше Sharpe, но больше turnover.
- maxpos10 ухудшает Sharpe/PnL.
- Turnover все еще высокий (20k+), нужен контроль концентрации по парам.

## Decision
- Новый основной кандидат: z0.95 / exit 0.06 / hold 180 / cd 180 / ms 0.1.
- Следующий шаг: проверить концентрацию PnL по парам и при необходимости ограничить top-k.
