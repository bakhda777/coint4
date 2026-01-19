# План оптимизации параметров (2026-01-18)

Цель: добиться стабильного `sharpe_ratio_abs > 1.0` в OOS по WFA.

## Критерии отбора
- Медиана Sharpe по 5 шагам WFA >= 1.0; минимум по шагам >= 0.6.
- Минимумы стабильности: `total_trades >= 500`, `total_pairs_traded >= 100`.
- Издержки учитываются: комиссии/слиппедж (и стресс-режимы), funding моделируем повышением cost факторов, если в движке нет прямой поддержки.
- Sharpe считаем годовым и согласуем с таймфреймом: для intraday‑серий используем `periods_per_year = annualizing_factor * (24*60/bar_minutes)` (например, 15m → `365*96`).
- Holdout не используется для подбора параметров.

## План (12 шагов, может меняться)
1. Санити чек метрик: `total_costs` не нулевые при наличии сделок + Sharpe корректно annualized (15m → `365*96`). (done)
2. Зафиксировать baseline-конфиг и прогнать 5-step WFA. (done)
3. Sweep сигналов на снижение turnover: entry/exit + min_hold + cooldown (малый grid). (done)
4. Sweep качества пар: corr thresholds (0.65/0.70/0.75) + один строгий пресет фильтров. (done)
5. Sweep рисков: risk_per_position, max_active_positions, max_margin_usage, max_kelly_fraction. (done)
6. Отобрать shortlist (3-5 конфигов) по Sharpe/DD/стабильности и достаточной статистике. (done)
7. Повторная 5-step WFA валидация для shortlist. (done)
8. Holdout фиксированный для top-1/2 + стресс-издержки. (done; holdout негативный)
9. Диагностика holdout: шаги, концентрация, overlap пар, фильтры. (done)
10. Ввести фильтр стабильности пар (pair_stability_window_steps/min_steps) + ужесточить pvalue/half-life/hurst + пересмотр universe.
    - Если `total_pairs_traded < 100`, ослабить стабильность (window=2/min=1) и/или увеличить `ssd_top_n`.
    - Выполнено: stability_relaxed2_20260119 (corr0.55, pv0.06, ssd30000, max_hurst0.6, liquidity 300k, min_volume 250k, backtest min_correlation=0.55) → total_pairs_traded 52.
11. Новый shortlist (3-5 конфигов) → 5-step WFA + holdout (max_steps=5).
12. Финализация 1-2 кандидатов и paper/live проверка.
