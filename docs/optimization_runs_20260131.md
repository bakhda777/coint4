# Журнал прогонов оптимизации (2026-01-31)

Назначение: extended OOS (2023-05-01 -> 2024-04-30) turnover-grid для снижения издержек (max_pairs 10/15, ms 0.25/0.30, hold/cd 240).

## Критерии отбора (актуальные)
- Гейтинг: `total_costs>0`, `total_trades>=500`, `total_pairs_traded>=50`, `sharpe_ratio_abs>0`, `total_pnl>0`.
- Stress: `cost_ratio <= 0.5`.
- WFA-стабильность: медиана Sharpe по 5 шагам >= 1.0, минимум по шагам >= 0.6.

## Очередь: realcost_oos20230501_20240430_turnover
- Очередь: `coint4/artifacts/wfa/aggregate/20260131_realcost_oos20230501_20240430_turnover/run_queue.csv` (8 прогонов).
- Конфиги:
  - `coint4/configs/holdout_20260131_relaxed8_nokpss_u250_churnfix_turnover_oos20230501_20240430/*.yaml` (4 шт.)
  - `coint4/configs/stress_20260131_relaxed8_nokpss_u250_churnfix_turnover_oos20230501_20240430/*.yaml` (4 шт.)
- Статус: `completed`.

## Результаты (holdout + stress)
| config | kind | sharpe | pnl | costs | trades | pairs | max_dd | cost_ratio |
|---|---|---|---|---|---|---|---|---|
| top10 ms0.25 | holdout | -0.69 | -71.4 | 110.6 | 2264 | 32 | -265.5 | -1.55 |
| top10 ms0.25 | stress | -1.54 | -157.4 | 196.5 | 2264 | 32 | -299.6 | -1.25 |
| top10 ms0.30 | holdout | -0.61 | -63.2 | 110.6 | 2263 | 32 | -264.5 | -1.75 |
| top10 ms0.30 | stress | -1.45 | -149.2 | 196.6 | 2263 | 32 | -298.7 | -1.32 |
| top15 ms0.25 | holdout | 1.18 | 151.4 | 154.9 | 3264 | 47 | -206.3 | 1.02 |
| top15 ms0.25 | stress | 0.25 | 31.0 | 275.4 | 3264 | 47 | -243.9 | 8.89 |
| top15 ms0.30 | holdout | 1.31 | 168.8 | 154.9 | 3263 | 47 | -203.6 | 0.92 |
| top15 ms0.30 | stress | 0.39 | 48.3 | 275.4 | 3263 | 47 | -241.2 | 5.70 |

## Выводы
- Top10 провалился: отрицательный Sharpe и PnL, пары 32 (<50).
- Top15 улучшает Sharpe в holdout, но не проходит гейт по парам (47 < 50) и стресс по издержкам (cost_ratio >> 0.5).
- Усиление hold/cooldown + ms не решило проблему extended OOS: стрессовые издержки доминируют PnL.

## Рекомендации
- Зафиксировать stop-condition по extended OOS: дальнейшая оптимизация через turnover-grid не дает приемлемого stress cost_ratio.
- Рассмотреть смену режима: либо более жесткая фильтрация пар/ликвидности, либо переход к paper/forward тесту базового конфига с оговорками по рискам.

## Budget $1000: очередь tlow extended OOS
- Обзор: `docs/budget1000_overview_20260131.md`.
- Очередь: `coint4/artifacts/wfa/aggregate/20260131_budget1000_tlow_extended_oos20230501_20240430/run_queue.csv` (4 прогона).
- Конфиги: `coint4/configs/budget_20260131_1000_tlow_extended_oos20230501_20240430/*.yaml` (holdout+stress, risk 0.0175/0.015).
- Статус: `completed`.

### Результаты (holdout + stress)
| config | kind | sharpe | pnl | costs | trades | pairs | max_dd | cost_ratio |
|---|---|---|---|---|---|---|---|---|
| risk0.0175 | holdout | 1.70 | 307.8 | 79.6 | 2348 | 36 | -322.3 | 0.26 |
| risk0.0175 | stress | 1.36 | 232.3 | 138.3 | 2348 | 36 | -327.4 | 0.60 |
| risk0.015 | holdout | -0.54 | -80.4 | 28.5 | 961 | 36 | -274.2 | — |
| risk0.015 | stress | -0.72 | -100.6 | 50.4 | 961 | 36 | -276.2 | — |

### Выводы (budget $1000)
- `risk0.0175` даёт положительный PnL, но stress cost_ratio 0.60 (> 0.5) и DD ~ -32% — слишком агрессивно для $1000.
- `risk0.015` уходит в минус; в обоих случаях пар 36 (<50) → гейт не пройден.

## Queue: budget1000_tlow_extended_refine (completed)
- Очередь: `coint4/artifacts/wfa/aggregate/20260131_budget1000_tlow_extended_refine/run_queue.csv` (8 прогонов).
- Конфиги: `coint4/configs/budget_20260131_1000_tlow_extended_refine/*.yaml`.
- Изменения: z=1.25/1.30, min_spread_move_sigma=0.30/0.35, hold/cd=300.

### Результаты (holdout + stress)
| config | kind | sharpe | pnl | costs | trades | pairs | max_dd | cost_ratio |
|---|---|---|---|---|---|---|---|---|
| z1.25/ms0.30 | holdout | 2.16 | 298.1 | 73.2 | 2071 | 36 | -172.9 | 0.25 |
| z1.25/ms0.30 | stress | 1.76 | 236.2 | 129.4 | 2071 | 36 | -174.4 | 0.55 |
| z1.25/ms0.35 | holdout | 2.17 | 299.3 | 73.0 | 2067 | 36 | -172.9 | 0.24 |
| z1.25/ms0.35 | stress | 1.77 | 237.6 | 129.1 | 2067 | 36 | -174.4 | 0.54 |
| z1.30/ms0.30 | holdout | 2.14 | 294.9 | 70.8 | 2008 | 36 | -168.9 | 0.24 |
| z1.30/ms0.30 | stress | 1.75 | 235.5 | 125.1 | 2008 | 36 | -177.2 | 0.53 |
| z1.30/ms0.35 | holdout | 2.04 | 280.1 | 70.6 | 2004 | 36 | -174.1 | 0.25 |
| z1.30/ms0.35 | stress | 1.65 | 220.8 | 124.7 | 2004 | 36 | -182.9 | 0.56 |

### Выводы (refine)
- DD заметно снизился (≈ -17…-18% против -32% в базовом tlow), но stress cost_ratio всё ещё 0.53–0.56 (>0.5).
- Количество пар не выросло (36), гейт по парам всё ещё не пройден.
- Концентрация PnL остаётся высокой: по стрессу z1.30/ms0.30 top10_share ~2.24, top20_share ~2.93 (есть значимые отрицательные пары).
- Фильтрация пар в основном режется по low_correlation и pvalue; дальше идут hurst_too_high и beta_out_of_range (см. `results/filter_reasons_20260131_0903*.csv`).

## Queue: budget1000_tlow_extended_refine2 (completed)
- Очередь: `coint4/artifacts/wfa/aggregate/20260131_budget1000_tlow_extended_refine2/run_queue.csv` (8 прогонов).
- Конфиги: `coint4/configs/budget_20260131_1000_tlow_extended_refine2/*.yaml`.
- Изменения: z=1.35/1.40, min_spread_move_sigma=0.35/0.40, hold/cd=360.

### Результаты (holdout + stress)
| config | kind | sharpe | pnl | costs | trades | pairs | max_dd | cost_ratio |
|---|---|---|---|---|---|---|---|---|
| z1.35/ms0.35 | holdout | 0.17 | 1.8 | 60.5 | 1805 | 36 | -184.4 | 33.68 |
| z1.35/ms0.35 | stress | -0.18 | -43.5 | 105.6 | 1805 | 36 | -195.4 | — |
| z1.35/ms0.40 | holdout | 0.11 | -6.2 | 60.3 | 1801 | 36 | -181.9 | — |
| z1.35/ms0.40 | stress | -0.25 | -51.0 | 105.3 | 1801 | 36 | -192.8 | — |
| z1.40/ms0.35 | holdout | 0.81 | 102.0 | 58.1 | 1739 | 36 | -175.4 | 0.57 |
| z1.40/ms0.35 | stress | 0.53 | 54.8 | 101.6 | 1739 | 36 | -201.3 | 1.85 |
| z1.40/ms0.40 | holdout | 0.81 | 101.7 | 58.0 | 1733 | 36 | -173.5 | 0.57 |
| z1.40/ms0.40 | stress | 0.53 | 54.6 | 101.4 | 1733 | 36 | -194.6 | 1.86 |

### Выводы (refine2)
- Ужесточение z/ms ухудшило PnL и Sharpe; часть конфигов уходит в минус.
- Stress cost_ratio вырос (до 1.85–1.86), гейт по издержкам провален сильнее.
- Пар всё ещё 36 → проблема отбора/фильтров, а не только turnover.

## Queue: budget1000_tlow_extended_tradeability_basecap3 (completed)
- Очередь: `coint4/artifacts/wfa/aggregate/20260131_budget1000_tlow_extended_tradeability_basecap3/run_queue.csv` (8 прогонов).
- Конфиги: `coint4/configs/budget_20260131_1000_tlow_extended_tradeability_basecap3/*.yaml`.
- Пары: `artifacts/universe/20260122_relaxed8_strict_preholdout_v2_basecap3/pairs_universe.yaml` (102 пары, max_per_base=3).
- Изменения:
  - tradeability‑фильтры (liquidity_usd_daily / max_bid_ask_pct / max_avg_funding_pct, min_volume_usd_24h, min_days_live, max_funding_rate_abs, max_tick_size_pct) — уровни tradeM/tradeS;
  - corr/pvalue grid: 0.40/0.20 и 0.35/0.25.

### Результаты (holdout + stress)
| config | kind | sharpe | pnl | max_dd | costs | cost_ratio | trades | pairs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| tradeM/corr0.35/pv0.25 | holdout | -2.05 | -249.5 | -330.3 | 32.8 | — | 982 | 27 |
| tradeM/corr0.35/pv0.25 | stress | -2.27 | -271.3 | -340.0 | 57.9 | — | 982 | 27 |
| tradeM/corr0.40/pv0.20 | holdout | -1.89 | -215.9 | -296.0 | 29.1 | — | 879 | 26 |
| tradeM/corr0.40/pv0.20 | stress | -2.09 | -235.2 | -299.5 | 51.5 | — | 879 | 26 |
| tradeS/corr0.35/pv0.25 | holdout | -2.05 | -249.5 | -330.3 | 32.8 | — | 982 | 27 |
| tradeS/corr0.35/pv0.25 | stress | -2.27 | -271.3 | -340.0 | 57.9 | — | 982 | 27 |
| tradeS/corr0.40/pv0.20 | holdout | -1.89 | -215.9 | -296.0 | 29.1 | — | 879 | 26 |
| tradeS/corr0.40/pv0.20 | stress | -2.09 | -235.2 | -299.5 | 51.5 | — | 879 | 26 |

### Выводы (tradeability + basecap3)
- Все варианты ушли в минус: Sharpe < 0, PnL отрицательный, пары 26–27.
- tradeM и tradeS дали идентичные метрики → ужесточение tradeability не сработало на выбранном universe (порог не стал лимитирующим).
- basecap3 снизил число пар и ухудшил результат; ветку закрываем.

## Сводка фильтрации (2026-01-31)
- Основные причины отказов: `low_correlation` (46.7%) и `pvalue` (27.9%); вместе ≈74.6% всех отказов.
- Детали и таблица: `docs/filter_reasons_20260131.md`.
