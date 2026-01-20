# Candidate: relaxed8_nokpss_u250 (2026-01-22)

## Summary
- Candidate: relaxed8_nokpss_u250_top50 (fixed universe, KPSS disabled)
- Pairs file: `coint4/artifacts/universe/20260119_relaxed8_strict_preholdout_v2/pairs_universe.yaml`
- Holdout period: 2024-05-01 -> 2024-12-31 (train=90d, w2m1)
- Обновление: churnfix top‑k после фикса min_spread_move_sigma.

## Metrics (churnfix_topk winner)
Holdout (top50 / z0.95 / exit 0.06 / hold 180 / cd 180 / ms 0.1):
- Sharpe: 7.6269
- PnL: 1082.62
- Trades: 11823
- Pairs traded: 120
- Costs: 337.64

Stress holdout (top50 / z0.95 / exit 0.06 / hold 180 / cd 180 / ms 0.1):
- Sharpe: 6.4348
- PnL: 914.21
- Trades: 11823
- Pairs traded: 120
- Costs: 600.24

## Tradeoff notes
- top20 даёт ещё выше Sharpe, но снижает PnL относительно top50.
- Turnover снижен до 11.8k при сопоставимом PnL и улучшенном Sharpe.
- Концентрация PnL (top10/top20): holdout ~59%/82%, stress ~68%/94%; отрицательных пар 39/43 из 120.

## Decision
- Новый основной кандидат: top50 / z0.95 / exit 0.06 / hold 180 / cd 180 / ms 0.1.
- Следующий шаг: проверить устойчивость на альтернативных периодах или ограничить top20, если требуется максимизация Sharpe ценой PnL.
