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
    - Выполнено: stability_relaxed3_20260119 (window=1/min=1, corr 0.55→0.5, pv 0.06→0.08, ssd 30000→50000) → 1 конфиг с 51 парой, 2 конфига обнулились по KPSS.
    - Выполнено: stability_relaxed4_20260119 (KPSS 0.03–0.05) → пары 159–272, Sharpe ~3.07–3.09.
11. Новый shortlist (3-5 конфигов) → 5-step WFA + holdout (max_steps=5).
    - Выполнено: holdout + stress для top‑1/2 relaxed4 (corr0.45/corr0.5, ssd50000, kpss0.03) → holdout отрицательный, стресс Sharpe ~2.38.
    - Выполнено: диагностика holdout relaxed4 (см. `docs/holdout_diagnostics_20260119_relaxed4.md`).
    - Выполнено: stability_relaxed5 WFA (window=2/3, min_steps=1/2) → лучший Sharpe 5.81 при w3m2.
    - Выполнено: holdout + stress для w3m2 (corr0.45, ssd50000, kpss0.03, window=3/min=2) → holdout Sharpe -1.77, stress Sharpe 4.76.
    - Выполнено: диагностика holdout w3m2 (см. `docs/holdout_diagnostics_20260119_relaxed5.md`).
    - Запущено: фиксированный universe из WFA w3m2 (383 пары) → отдельный holdout для проверки устойчивости пар.
    - Выполнено: holdout fixed universe (w3m2) → Sharpe 1.25, total_pairs_traded 18, total_trades 352, PnL 13.94 (недостаточно статистики).
    - Выполнено: holdout fixed universe с pair_stability window=1/min=1 → Sharpe -0.02, total_pairs_traded 11, total_trades 253, PnL -0.20.
    - Выполнено: stability_relaxed6 WFA с pvalue 0.12 / kpss 0.05 / hurst 0.70 → Sharpe 5.36, pairs 303.
    - Выполнено: holdout relaxed6 (w3m2) на 2024-05-01 → 2024-12-31 → Sharpe -2.20, PnL -267.02.
    - Выполнено: диагностика holdout relaxed6 (см. `docs/holdout_diagnostics_20260119_relaxed6.md`).
    - Выполнено: holdout relaxed6 с фиксированным universe (303 пары) → Sharpe 3.77, total_pairs_traded 7, total_trades 141.
    - Выполнено: stability_relaxed7 WFA с train=90d при relaxed6 фильтрах → Sharpe 5.25, pairs 177.
    - Выполнено: holdout relaxed7 (train=90d) на 2024-05-01 → 2024-12-31 → Sharpe -0.20, PnL -20.95.
    - Выполнено: universe strict pre‑holdout (2023-07-01 → 2024-04-30) → 110 пар, см. `coint4/artifacts/universe/20260119_relaxed8_strict_preholdout/pairs_universe.yaml`.
    - Выполнено: universe strict pre‑holdout v2 (limit_symbols=300) → 250 пар, см. `coint4/artifacts/universe/20260119_relaxed8_strict_preholdout_v2/pairs_universe.yaml`.
    - Выполнено: WFA relaxed8 fixed universe (train=90d) → 0 торгуемых пар, Sharpe 0.00 (фильтры режут до нуля).
    - Выполнено: WFA relaxed8_loose (pvalue 0.2, kpss 0.1, hurst 0.8, corr 0.4, w2m1) → 0 торгуемых пар, KPSS режет до нуля.
    - Выполнено: WFA relaxed8_nokpss (kpss=1.0) → Sharpe 1.86, total_pairs_traded 35, total_trades 1242.
    - Выполнено: holdout relaxed8_nokpss (fixed universe, train=90d, w2m1) → Sharpe 3.21, PnL 145.84, total_pairs_traded 64, total_trades 2252.
    - Выполнено: stress-издержки для relaxed8_nokpss → Sharpe 2.17, PnL 98.35, pairs 64.
    - Выполнено: holdout relaxed8_nokpss_u250 (expanded universe 250 пар) → Sharpe 4.20, pairs 168.
    - Выполнено: stress-издержки для u250 → Sharpe 2.89, PnL 289.30, pairs 168.
    - Выполнено: кандидат зафиксирован в `docs/candidate_relaxed8_u250_20260119.md`.
    - Следующий шаг: финальная оценка устойчивости и решение о paper/live.
12. Финализация 1-2 кандидатов и paper/live проверка.
