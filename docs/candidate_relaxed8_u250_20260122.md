# Candidate: relaxed8_nokpss_u250 (2026-01-22)

## Summary
- Candidate: relaxed8_nokpss_u250_top50_z1p00 (fixed universe, KPSS disabled)
- Pairs file: `coint4/artifacts/universe/20260119_relaxed8_strict_preholdout_v2/pairs_universe.yaml`
- Holdout period: 2024-05-01 -> 2024-12-31 (train=90d, w2m1)
- Обновление: churnfix top‑k + sensitivity (z/exit/hold/cd).

## Canonical configs
- Primary: `coint4/configs/candidate_20260123_relaxed8_nokpss_u250_top50/candidate_relaxed8_nokpss_20260123_u250_top50_z1p00_exit0p06_hold180_cd180_ms0p2.yaml`
- PnL alt: `coint4/configs/candidate_20260123_relaxed8_nokpss_u250_top50/candidate_relaxed8_nokpss_20260123_u250_top50_z0p95_exit0p08_hold180_cd180_ms0p1.yaml`

## Metrics (top50 churngrid winner)
Holdout (top50 / z1.00 / exit 0.06 / hold 180 / cd 180 / ms 0.2):
- Sharpe: 9.09
- PnL: 1135.26
- Trades: 11384
- Pairs traded: 120
- Costs: 326.39

Stress holdout (top50 / z1.00 / exit 0.06 / hold 180 / cd 180 / ms 0.2):
- Sharpe: 7.73
- PnL: 966.13
- Trades: 11384
- Pairs traded: 120
- Costs: 580.25

## Sanity checks
- Sharpe пересчитан из `equity_curve.csv`: значения совпадают с `strategy_metrics.csv` (проверено на run_index).
- `total_costs` > 0 (holdout 326.77, stress 580.92), издержки учитываются.

## Tradeoff notes
- ms0p2 даёт небольшое улучшение Sharpe/PNL при схожем turnover vs ms0p1.
- z0.95/exit0.08 даёт максимум PnL, но ниже Sharpe; hold240 ухудшает метрики.
- Концентрация PnL (gross, из `trade_statistics.csv`): top10/top20 ~59%/81% (holdout), ~67%/91% (stress); отрицательных пар 38/41 из 120.
- Концентрация по базовым активам (gross, PnL делится 50/50 на base): топ‑5 holdout BTC~11.9%, ETH~10.0%, JUV~7.6%, KASTA~5.1%, KUB~5.1%; stress BTC~13.4%, ETH~10.4%, JUV~8.4%, KASTA~5.8%, CITY~5.6%.
- OOS: 2023H2 лучше у z0.95/exit0.08, 2025H1 лучше у z1.00/exit0.06.

## Decision
- Новый основной кандидат: top50 / z1.00 / exit 0.06 / hold 180 / cd 180 / ms 0.2.
- Альтернативный holdout подтверждён: z1.00/exit0.06 даёт Sharpe 8.65/7.48 и PnL 1049/907 (holdout/stress); z0.95/exit0.08 чуть слабее.
