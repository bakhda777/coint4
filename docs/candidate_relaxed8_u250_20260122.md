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

## Sanity checks
- Sharpe пересчитан из `equity_curve.csv`: 15m ≈ 7.63, daily ≈ 7.85 (holdout), 15m ≈ 6.43, daily ≈ 6.69 (stress).
- `total_costs` > 0 (holdout 337.64, stress 600.24), издержки учитываются.

## Tradeoff notes
- top20 даёт ещё выше Sharpe, но снижает PnL относительно top50.
- Turnover снижен до 11.8k при сопоставимом PnL и улучшенном Sharpe.
- Концентрация PnL (gross, из `trade_statistics.csv`): top10/top20 ~59%/82% (holdout), ~68%/94% (stress); отрицательных пар 39/43 из 120.
- Концентрация по базовым активам (gross, PnL делится 50/50 на base): топ‑5 holdout BTC~12.3%, ETH~9.9%, JUV~8.7%, KUB~6.0%, KASTA~5.8%; stress BTC~14.0%, ETH~10.4%, JUV~9.7%, KASTA~6.7%, KUB~6.6%.

## Decision
- Новый основной кандидат: top50 / z0.95 / exit 0.06 / hold 180 / cd 180 / ms 0.1.
- Следующий шаг: проверить устойчивость на альтернативных периодах или ограничить top20, если требуется максимизация Sharpe ценой PnL.
