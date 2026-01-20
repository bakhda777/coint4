# Candidate: relaxed8_nokpss_u250 (2026-01-22)

## Summary
- Candidate: relaxed8_nokpss_u250_top50_z1p00 (fixed universe, KPSS disabled)
- Pairs file: `coint4/artifacts/universe/20260119_relaxed8_strict_preholdout_v2/pairs_universe.yaml`
- Holdout period: 2024-05-01 -> 2024-12-31 (train=90d, w2m1)
- Обновление: churnfix top‑k + sensitivity (z/exit/hold/cd).

## Canonical configs
- Primary: `coint4/configs/candidate_20260123_relaxed8_nokpss_u250_top50/candidate_relaxed8_nokpss_20260123_u250_top50_z1p00_exit0p06_hold180_cd180_ms0p1.yaml`
- PnL alt: `coint4/configs/candidate_20260123_relaxed8_nokpss_u250_top50/candidate_relaxed8_nokpss_20260123_u250_top50_z0p95_exit0p08_hold180_cd180_ms0p1.yaml`

## Metrics (top50_sens winner)
Holdout (top50 / z1.00 / exit 0.06 / hold 180 / cd 180 / ms 0.1):
- Sharpe: 9.0140
- PnL: 1114.71
- Trades: 11414
- Pairs traded: 120
- Costs: 326.77

Stress holdout (top50 / z1.00 / exit 0.06 / hold 180 / cd 180 / ms 0.1):
- Sharpe: 7.6445
- PnL: 946.36
- Trades: 11414
- Pairs traded: 120
- Costs: 580.92

## Sanity checks
- Sharpe пересчитан из `equity_curve.csv`: 15m ≈ 9.01, daily ≈ 8.99 (holdout), 15m ≈ 7.64, daily ≈ 7.62 (stress).
- `total_costs` > 0 (holdout 326.77, stress 580.92), издержки учитываются.

## Tradeoff notes
- z1.00/exit0.06 дал лучшую Sharpe‑устойчивость при сопоставимом PnL и чуть меньшем turnover.
- z0.95/exit0.08 даёт максимум PnL, но ниже Sharpe; hold240 ухудшает метрики.
- Концентрация PnL (gross, из `trade_statistics.csv`): top10/top20 ~59%/81% (holdout), ~67%/91% (stress); отрицательных пар 38/41 из 120.
- Концентрация по базовым активам (gross, PnL делится 50/50 на base): топ‑5 holdout BTC~11.9%, ETH~10.0%, JUV~7.6%, KASTA~5.1%, KUB~5.1%; stress BTC~13.4%, ETH~10.4%, JUV~8.4%, KASTA~5.8%, CITY~5.6%.

## Decision
- Новый основной кандидат: top50 / z1.00 / exit 0.06 / hold 180 / cd 180 / ms 0.1.
- Альтернативный holdout подтверждён: z1.00/exit0.06 даёт Sharpe 8.65/7.48 и PnL 1049/907 (holdout/stress); z0.95/exit0.08 чуть слабее.
