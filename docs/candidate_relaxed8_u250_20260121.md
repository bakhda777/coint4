# Candidate: relaxed8_nokpss_u250 (2026-01-21)

## Summary
- Candidate: relaxed8_nokpss_u250 (fixed universe, KPSS disabled)
- Pairs file: `coint4/artifacts/universe/20260119_relaxed8_strict_preholdout_v2/pairs_universe.yaml`
- Holdout period: 2024-05-01 -> 2024-12-31 (train=90d, w2m1)
- Обновление: micro-grid показал улучшение для `z0.95/exit0.06/hold120/cd120`.

## Metrics (u250 micro-grid winner)
Holdout (z0.95 / exit 0.06 / hold 120 / cd 120):
- Sharpe: 4.5432
- PnL: 450.51
- Max DD: -60.36
- Trades: 3936
- Pairs traded: 168
- Costs: 107.86

Stress holdout (z0.95 / exit 0.06 / hold 120 / cd 120):
- Sharpe: 3.7267
- PnL: 369.19
- Max DD: -64.81
- Trades: 3936
- Pairs traded: 168
- Costs: 191.75

## Tradeoff notes
- Exit 0.10 почти идентичен по метрикам (Sharpe 4.54/3.73).
- max_pairs=50 снижает turnover (2269 trades, 120 pairs) при Sharpe 4.27/3.59.
- Ограничения hold/cooldown 90/180 не изменили метрики (скорее всего, не активируются на этих данных).

## Decision
- Новый основной кандидат: z0.95 / exit 0.06 / hold 120 / cd 120.
- Следующий шаг: проверить концентрацию PnL по парам и подтвердить устойчивость на paper/live.
