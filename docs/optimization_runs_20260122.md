# Журнал прогонов оптимизации (2026-01-22)

Назначение: churn-control micro-grid после включения cooldown/min_hold/min_spread_move, проверка компромисса Sharpe/PnL/turnover.

## Обновления (2026-01-22)

### Queue: relaxed8_nokpss_u250_churnfix (holdout + stress)
- Очередь: `coint4/artifacts/wfa/aggregate/20260122_relaxed8_nokpss_u250_churnfix/run_queue.csv`.
- Цель: проверить churn-контроль (min_hold, cooldown, min_spread_move_sigma, entry/exit, max_active_positions).
- Конфиги:
  - `coint4/configs/holdout_20260122_relaxed8_nokpss_u250_churnfix/*.yaml` (8 шт.)
  - `coint4/configs/stress_20260122_relaxed8_nokpss_u250_churnfix/*.yaml` (8 шт.)
- Статус: `completed` (16 прогонов).

#### Результаты (holdout + stress)
| config | hold_sharpe | hold_pnl | hold_trades | hold_pairs | stress_sharpe | stress_pnl | stress_trades | stress_pairs |
|---|---|---|---|---|---|---|---|---|
| z0p90/exit0p08/hold120/cd120/ms0p1 | 0.00 | 0.00 | 0 | 168 | 0.00 | 0.00 | 0 | 168 |
| z0p95/exit0p06/hold120/cd120/ms0p1 | 0.00 | 0.00 | 0 | 168 | 0.00 | 0.00 | 0 | 168 |
| z0p95/exit0p06/hold120/cd120/ms0p1/maxpos10 | 0.00 | 0.00 | 0 | 168 | 0.00 | 0.00 | 0 | 168 |
| z0p95/exit0p06/hold120/cd120/ms0p2 | 0.00 | 0.00 | 0 | 168 | 0.00 | 0.00 | 0 | 168 |
| z0p95/exit0p06/hold180/cd180/ms0p1 | 0.00 | 0.00 | 0 | 168 | 0.00 | 0.00 | 0 | 168 |
| z0p95/exit0p06/hold60/cd60/ms0p1 | 0.00 | 0.00 | 0 | 168 | 0.00 | 0.00 | 0 | 168 |
| z0p95/exit0p10/hold120/cd120/ms0p1 | 0.00 | 0.00 | 0 | 168 | 0.00 | 0.00 | 0 | 168 |
| z1p00/exit0p06/hold120/cd120/ms0p1 | 0.00 | 0.00 | 0 | 168 | 0.00 | 0.00 | 0 | 168 |

Выводы:
- Все варианты дали 0 сделок; причина — слишком консервативные адаптивные пороги в Numba (мультипликатор упирался в max).
- Исправление: адаптивные пороги переведены на base-volatility + `max_var_multiplier` (см. `coint4/src/coint2/core/numba_kernels.py`).
- Повторный прогон queued: `coint4/artifacts/wfa/aggregate/20260122_relaxed8_nokpss_u250_churnfix_v2/run_queue.csv`.

### Queue: relaxed8_nokpss_u250_churnfix_v2 (holdout + stress)
- Очередь: `coint4/artifacts/wfa/aggregate/20260122_relaxed8_nokpss_u250_churnfix_v2/run_queue.csv`.
- Статус: `completed` (16 прогонов).

#### Результаты (holdout + stress)
| config | hold_sharpe | hold_pnl | hold_trades | hold_pairs | stress_sharpe | stress_pnl | stress_trades | stress_pairs |
|---|---|---|---|---|---|---|---|---|
| z0p90/exit0p08/hold120/cd120/ms0p1 | 0.00 | 0.00 | 0 | 168 | 0.00 | 0.00 | 0 | 168 |
| z0p95/exit0p06/hold120/cd120/ms0p1 | 0.00 | 0.00 | 0 | 168 | 0.00 | 0.00 | 0 | 168 |
| z0p95/exit0p06/hold120/cd120/ms0p1/maxpos10 | 0.00 | 0.00 | 0 | 168 | 0.00 | 0.00 | 0 | 168 |
| z0p95/exit0p06/hold120/cd120/ms0p2 | 0.00 | 0.00 | 0 | 168 | 0.00 | 0.00 | 0 | 168 |
| z0p95/exit0p06/hold180/cd180/ms0p1 | 0.00 | 0.00 | 0 | 168 | 0.00 | 0.00 | 0 | 168 |
| z0p95/exit0p06/hold60/cd60/ms0p1 | 0.00 | 0.00 | 0 | 168 | 0.00 | 0.00 | 0 | 168 |
| z0p95/exit0p10/hold120/cd120/ms0p1 | 0.00 | 0.00 | 0 | 168 | 0.00 | 0.00 | 0 | 168 |
| z1p00/exit0p06/hold120/cd120/ms0p1 | 0.00 | 0.00 | 0 | 168 | 0.00 | 0.00 | 0 | 168 |

Выводы:
- После фикса адаптивных порогов сделки всё ещё отсутствуют (holdout/stress = 0).
- Вероятно, пороги остаются слишком высокими относительно z-score (нужно проверить базовую/текущую волатильность и фактические z-распределения).

### Queue: relaxed8_nokpss_u250_churnfix_sanity (no-adapt, diagnostic)
- Очередь: `coint4/artifacts/wfa/aggregate/20260122_relaxed8_nokpss_u250_churnfix_sanity/run_queue.csv`.
- Цель: sanity-проверка сделок при отключённых adaptive_thresholds/market_regime/structural_breaks и снижении entry до 0.75.
- Конфиги: 2 варианта (z0.95 и z0.75) × holdout/stress.
- Статус: `completed` (4 прогона).

#### Результаты (holdout + stress)
| config | hold_sharpe | hold_pnl | hold_trades | hold_pairs | stress_sharpe | stress_pnl | stress_trades | stress_pairs |
|---|---|---|---|---|---|---|---|---|
| z0p75/exit0p06/hold120/cd120/ms0p1/noadapt | 0.00 | 0.00 | 0 | 168 | 0.00 | 0.00 | 0 | 168 |
| z0p95/exit0p06/hold120/cd120/ms0p1/noadapt | 0.00 | 0.00 | 0 | 168 | 0.00 | 0.00 | 0 | 168 |

Выводы:
- Даже при no-adapt и entry 0.75 сделки отсутствуют; см. diag-логи в run.log (z_score превышает порог, но позиций нет).

### Queue: relaxed8_nokpss_u250_churnfix_sanity_v2 (current-bar signals)
- Очередь: `coint4/artifacts/wfa/aggregate/20260122_relaxed8_nokpss_u250_churnfix_sanity_v2/run_queue.csv`.
- Цель: проверить сделки после выравнивания Numba-сигналов по текущему бару.
- Статус: `completed` (4 прогона).

#### Результаты (holdout + stress)
| config | hold_sharpe | hold_pnl | hold_trades | hold_pairs | stress_sharpe | stress_pnl | stress_trades | stress_pairs |
|---|---|---|---|---|---|---|---|---|
| z0p75/exit0p06/hold120/cd120/ms0p1/noadapt | 0.00 | 0.00 | 0 | 168 | 0.00 | 0.00 | 0 | 168 |
| z0p95/exit0p06/hold120/cd120/ms0p1/noadapt | 0.00 | 0.00 | 0 | 168 | 0.00 | 0.00 | 0 | 168 |

Выводы:
- Сделки всё ещё отсутствуют; диагностический `z_score` превышает порог, но позиции не открываются.
- Гипотеза смещается к рассинхрону между z-score (выходные метрики) и торговой логикой Numba (min_volatility clamp и порог std).
- План: привести Numba к базовой логике по std (guard 1e-6, без clamp в z-score) и прогнать sanity v3.

### Queue: relaxed8_nokpss_u250_churnfix_sanity_v3 (sigma guard alignment)
- Очередь: `coint4/artifacts/wfa/aggregate/20260122_relaxed8_nokpss_u250_churnfix_sanity_v3/run_queue.csv`.
- Цель: sanity-проверка после выравнивания расчёта z-score в Numba (std guard + без clamp) и передачи beta/mu/sigma.
- Статус: `completed` (4 прогона).

#### Результаты (holdout + stress)
| config | hold_sharpe | hold_pnl | hold_trades | hold_pairs | stress_sharpe | stress_pnl | stress_trades | stress_pairs |
|---|---|---|---|---|---|---|---|---|
| z0p75/exit0p06/hold120/cd120/ms0p1/noadapt | 0.00 | 0.00 | 0 | 168 | 0.00 | 0.00 | 0 | 168 |
| z0p95/exit0p06/hold120/cd120/ms0p1/noadapt | 0.00 | 0.00 | 0 | 168 | 0.00 | 0.00 | 0 | 168 |

Выводы:
- Диагностика локально показала, что min_spread_move_sigma блокирует все входы из-за NaN в last_flat_spread при fastmath: diff считался 0, и can_enter всегда становился False.
- Исправление: заменить NaN sentinel на флаг last_flat_valid (без арифметики с NaN).

### Queue: relaxed8_nokpss_u250_churnfix_sanity_v4 (min_spread_move fix)
- Очередь: `coint4/artifacts/wfa/aggregate/20260122_relaxed8_nokpss_u250_churnfix_sanity_v4/run_queue.csv`.
- Цель: sanity-проверка после фикса min_spread_move_sigma (last_flat_valid).
- Статус: `completed` (4 прогона).

#### Результаты (holdout + stress)
| config | hold_sharpe | hold_pnl | hold_trades | hold_pairs | stress_sharpe | stress_pnl | stress_trades | stress_pairs |
|---|---|---|---|---|---|---|---|---|
| z0p75/exit0p06/hold120/cd120/ms0p1/noadapt | -0.88 | -138.77 | 27164 | 168 | -1.83 | -286.48 | 27164 | 168 |
| z0p95/exit0p06/hold120/cd120/ms0p1/noadapt | 0.53 | 78.85 | 26517 | 168 | -0.45 | -72.02 | 26517 | 168 |

Выводы:
- Сделки восстановились, но turnover слишком высокий (26k+), Sharpe в стресс‑издержках отрицательный.
- Нужен повтор churnfix‑grid с рабочей логикой min_spread_move_sigma и адаптивными порогами.

### Queue: relaxed8_nokpss_u250_churnfix_v3 (min_spread_move fix)
- Очередь: `coint4/artifacts/wfa/aggregate/20260122_relaxed8_nokpss_u250_churnfix_v3/run_queue.csv`.
- Цель: повтор churnfix‑grid (holdout + stress) после фикса min_spread_move_sigma.
- Статус: `completed` (16 прогонов).

#### Результаты (holdout + stress)
| config | hold_sharpe | hold_pnl | hold_trades | hold_pairs | stress_sharpe | stress_pnl | stress_trades | stress_pairs |
|---|---|---|---|---|---|---|---|---|
| z0p90/exit0p08/hold120/cd120/ms0p1 | 7.98 | 1071.62 | 25338 | 168 | 6.36 | 855.72 | 25338 | 168 |
| z0p95/exit0p10/hold120/cd120/ms0p1 | 7.21 | 1012.62 | 25460 | 168 | 5.53 | 777.97 | 25460 | 168 |
| z0p95/exit0p06/hold180/cd180/ms0p1 | 7.00 | 1069.34 | 20236 | 168 | 5.84 | 892.62 | 20236 | 168 |
| z1p00/exit0p06/hold120/cd120/ms0p1 | 6.70 | 932.60 | 22503 | 168 | 5.26 | 731.30 | 22503 | 168 |
| z0p95/exit0p06/hold60/cd60/ms0p1 | 6.61 | 914.52 | 27869 | 168 | 4.99 | 691.80 | 27869 | 168 |
| z0p95/exit0p06/hold120/cd120/ms0p2 | 6.48 | 900.79 | 23229 | 168 | 5.05 | 702.77 | 23229 | 168 |
| z0p95/exit0p06/hold120/cd120/ms0p1 | 6.45 | 906.62 | 23317 | 168 | 5.04 | 707.73 | 23317 | 168 |
| z0p95/exit0p06/hold120/cd120/ms0p1/maxpos10 | 5.52 | 583.36 | 23317 | 168 | 4.21 | 445.32 | 23317 | 168 |

Выводы:
- Лучший баланс Sharpe/PnL/turnover: z0p95/exit0p06/hold180/cd180/ms0p1 (меньше сделок при сопоставимом PnL).
- Top Sharpe у z0p90/exit0p08, но выше turnover; maxpos10 ухудшает метрики.

### Queue: relaxed8_nokpss_u250_churnfix_topk (pair cap sweep)
- Очередь: `coint4/artifacts/wfa/aggregate/20260122_relaxed8_nokpss_u250_churnfix_topk/run_queue.csv`.
- Цель: проверить концентрацию прибыли на топ‑20/50 пар при текущем лучшем профиле.
- Конфиги: top20/top50 × holdout/stress (4 прогона).
- Статус: `completed` (4 прогона).

#### Результаты (holdout + stress)
| config | hold_sharpe | hold_pnl | hold_trades | hold_pairs | stress_sharpe | stress_pnl | stress_trades | stress_pairs |
|---|---|---|---|---|---|---|---|---|
| top20/z0p95/exit0p06/hold180/cd180/ms0p1 | 7.74 | 897.92 | 4841 | 53 | 6.84 | 793.03 | 4841 | 53 |
| top50/z0p95/exit0p06/hold180/cd180/ms0p1 | 7.63 | 1082.62 | 11823 | 120 | 6.43 | 914.21 | 11823 | 120 |

Выводы:
- top20 даёт максимальный Sharpe и резкое снижение сделок/пар, но заметно ниже PnL.
- top50 сохраняет высокий PnL при ~40% меньшем turnover vs full‑universe и Sharpe выше базового churnfix‑кандидата.
- Концентрация PnL (gross): top10/top20 ~59%/82% (holdout), ~68%/94% (stress); отрицательных пар 39/43 из 120.
- Базовые активы (gross, PnL 50/50 на base): топ‑5 holdout BTC~12%, ETH~10%, JUV~9%, KUB~6%, KASTA~6%; stress BTC~14%, ETH~10%, JUV~10%, KASTA~7%, KUB~7%.

### Queue: relaxed8_nokpss_u250_churnfix_msgrid (min_spread_move/hold grid)
- Очередь: `coint4/artifacts/wfa/aggregate/20260122_relaxed8_nokpss_u250_churnfix_msgrid/run_queue.csv`.
- Цель: проверить min_spread_move_sigma 0.2/0.3 и увеличение min_hold/cooldown до 180/240.
- Конфиги: 4 варианта × holdout/stress (8 прогонов).
- Статус: `completed` (8 прогонов).

#### Результаты (holdout + stress)
| config | hold_sharpe | hold_pnl | hold_trades | hold_pairs | stress_sharpe | stress_pnl | stress_trades | stress_pairs |
|---|---|---|---|---|---|---|---|---|
| z0p95/exit0p06/hold180/cd180/ms0p2 | 6.97 | 1051.30 | 20175 | 168 | 5.79 | 874.10 | 20175 | 168 |
| z0p95/exit0p06/hold180/cd180/ms0p3 | 6.95 | 1025.94 | 20109 | 168 | 5.75 | 848.88 | 20109 | 168 |
| z0p95/exit0p06/hold240/cd240/ms0p2 | 4.95 | 799.66 | 18048 | 168 | 3.92 | 631.44 | 18048 | 168 |
| z0p95/exit0p06/hold240/cd240/ms0p3 | 4.98 | 801.50 | 17996 | 168 | 3.95 | 634.61 | 17996 | 168 |

Выводы:
- ms0p2/ms0p3 при hold180 почти не улучшают Sharpe/PNL vs ms0p1 и оставляют turnover высоким.
- hold240 снижает сделки, но ухудшает Sharpe/PNL — не лучший кандидат.

### Queue: relaxed8_nokpss_u250_churnfix_alt (alt holdout 2022-09-01 → 2023-04-30)
- Очередь: `coint4/artifacts/wfa/aggregate/20260122_relaxed8_nokpss_u250_churnfix_alt/run_queue.csv`.
- Цель: проверить top20/top50/full на альтернативном периоде (WFA ≤ 5 шагов).
- Конфиги: top20/top50/full × holdout/stress (6 прогонов).
- Статус: `completed` (6 прогонов).

#### Результаты (holdout + stress)
| config | hold_sharpe | hold_pnl | hold_trades | hold_pairs | stress_sharpe | stress_pnl | stress_trades | stress_pairs |
|---|---|---|---|---|---|---|---|---|
| full/z0p95/exit0p06/hold180/cd180/ms0p1 | 7.96 | 941.40 | 3836 | 47 | 6.72 | 793.57 | 3836 | 47 |
| top20/z0p95/exit0p06/hold180/cd180/ms0p1 | 7.41 | 861.47 | 3699 | 46 | 6.17 | 716.39 | 3699 | 46 |
| top50/z0p95/exit0p06/hold180/cd180/ms0p1 | 7.96 | 941.40 | 3836 | 47 | 6.72 | 793.57 | 3836 | 47 |

Выводы:
- На альтернативном периоде торгуется всего ~46–47 пар, поэтому top50 и full дают идентичные метрики.
- top20 слегка снижает Sharpe/PnL, но слабо влияет на turnover.

### Queue: relaxed8_nokpss_u250_churnfix_alt_top50_sens (alt holdout sensitivity)
- Очередь: `coint4/artifacts/wfa/aggregate/20260123_relaxed8_nokpss_u250_churnfix_alt_top50_sens/run_queue.csv`.
- Цель: подтвердить два лучших top50 варианта (z1.00/exit0.06 и z0.95/exit0.08) на альтернативном периоде.
- Конфиги: 2 варианта × holdout/stress (4 прогона).
- Статус: `completed` (4 прогона).

#### Результаты (holdout + stress)
| config | hold_sharpe | hold_pnl | hold_trades | hold_pairs | stress_sharpe | stress_pnl | stress_trades | stress_pairs |
|---|---|---|---|---|---|---|---|---|
| top50/z0p95/exit0p08/hold180/cd180/ms0p1 | 8.60 | 1038.45 | 3977 | 47 | 7.33 | 884.34 | 3977 | 47 |
| top50/z1p00/exit0p06/hold180/cd180/ms0p1 | 8.65 | 1049.23 | 3666 | 47 | 7.48 | 906.77 | 3666 | 47 |

Выводы:
- На альтернативном периоде z1.00/exit0.06 сохраняет преимущество по Sharpe и чуть лучше PnL при меньшем числе сделок.

### Queue: relaxed8_nokpss_u250_churnfix_oos20230501_20231231_top50 (OOS 2023H2)
- Очередь: `coint4/artifacts/wfa/aggregate/20260123_relaxed8_nokpss_u250_churnfix_oos20230501_20231231_top50/run_queue.csv`.
- Цель: независимый OOS‑период 2023-05-01 → 2023-12-31 (WFA 5 шагов).
- Конфиги: top50 × (z1.00/exit0.06, z0.95/exit0.08) × holdout/stress (4 прогона).
- Статус: `completed` (4 прогона).

#### Результаты (holdout + stress)
| config | hold_sharpe | hold_pnl | hold_trades | hold_pairs | stress_sharpe | stress_pnl | stress_trades | stress_pairs |
|---|---|---|---|---|---|---|---|---|
| top50/z0p95/exit0p08/hold180/cd180/ms0p1 | 4.24 | 809.72 | 7443 | 69 | 2.94 | 561.27 | 7443 | 69 |
| top50/z1p00/exit0p06/hold180/cd180/ms0p1 | 2.63 | 556.48 | 6878 | 69 | 1.55 | 326.34 | 6878 | 69 |

Выводы:
- В OOS 2023H2 лучше себя показывает PnL‑вариант z0.95/exit0.08 (выше Sharpe и PnL).
- Стабильность по шагам (holdout, daily_pnl срезы): z0.95 min/median Sharpe = -1.85/6.12; z1.00 = -1.72/5.60.
- Концентрация (gross PnL, holdout): top10/top20 = 47%/67% (z1.00) и 45%/63% (z0.95); отрицательных пар 54/145 и 50/145. Top базы: z1.00 → GODS/KCAL/KDA/CHZ/JST, z0.95 → GODS/KCAL/KDA/FTT/ENJ.

### Queue: relaxed8_nokpss_u250_churnfix_oos20230501_20231231_top50_ms0p2 (OOS 2023H2, ms0p2)
- Очередь: `coint4/artifacts/wfa/aggregate/20260123_relaxed8_nokpss_u250_churnfix_oos20230501_20231231_top50_ms0p2/run_queue.csv`.
- Цель: перепроверить OOS 2023H2 для ms0p2 кандидата (holdout+stress).
- Конфиги: z1.00/exit0.06/ms0p2 × holdout/stress (2 прогона).
- Статус: `completed` (2 прогона).

#### Результаты (holdout + stress)
| config | hold_sharpe | hold_pnl | hold_trades | hold_pairs | stress_sharpe | stress_pnl | stress_trades | stress_pairs |
|---|---|---|---|---|---|---|---|---|
| top50/z1p00/exit0p06/ms0p2 | 2.83 | 597.57 | 6871 | 69 | 1.74 | 367.13 | 6871 | 69 |
Выводы:
- Стабильность по шагам (holdout, daily_pnl срезы): min/median Sharpe = -1.56/5.91.
- Концентрация (gross PnL, holdout): top10/top20 = 47%/67%; отрицательных пар 53/145 (почти как ms0p1).

### Queue: relaxed8_nokpss_u250_churnfix_oos20230501_20231231_top30_top40 (OOS 2023H2, top30/top40)
- Очередь: `coint4/artifacts/wfa/aggregate/20260124_relaxed8_nokpss_u250_churnfix_oos20230501_20231231_top30_top40/run_queue.csv`.
- Цель: проверить компромисс turnover vs PnL между top20 и top50 для z1.00/exit0.06/ms0p1.
- Конфиги: top30/top40 × holdout/stress (4 прогона).
- Статус: `completed` (4 прогона).

#### Результаты (holdout + stress)
| config | hold_sharpe | hold_pnl | hold_trades | hold_pairs | stress_sharpe | stress_pnl | stress_trades | stress_pairs |
|---|---|---|---|---|---|---|---|---|
| top30/z1p00/exit0p06/hold180/cd180/ms0p1 | 2.74 | 575.45 | 6391 | 68 | 1.66 | 348.20 | 6391 | 68 |
| top40/z1p00/exit0p06/hold180/cd180/ms0p1 | 2.63 | 556.48 | 6878 | 69 | 1.55 | 326.34 | 6878 | 69 |
Выводы:
- top30 доминирует top40 по Sharpe/PNL при меньшем turnover; топ‑50 всё ещё лучше по Sharpe/PNL.

### Queue: relaxed8_nokpss_u250_churnfix_oos20250101_20250630_top50 (OOS 2025H1)
- Очередь: `coint4/artifacts/wfa/aggregate/20260123_relaxed8_nokpss_u250_churnfix_oos20250101_20250630_top50/run_queue.csv`.
- Цель: независимый OOS‑период 2025-01-01 → 2025-06-30 (WFA 3 шага).
- Конфиги: top50 × (z1.00/exit0.06, z0.95/exit0.08) × holdout/stress (4 прогона).
- Статус: `completed` (4 прогона).

#### Результаты (holdout + stress)
| config | hold_sharpe | hold_pnl | hold_trades | hold_pairs | stress_sharpe | stress_pnl | stress_trades | stress_pairs |
|---|---|---|---|---|---|---|---|---|
| top50/z0p95/exit0p08/hold180/cd180/ms0p1 | 2.83 | 238.30 | 7218 | 79 | 1.22 | 101.26 | 7218 | 79 |
| top50/z1p00/exit0p06/hold180/cd180/ms0p1 | 3.83 | 399.53 | 6715 | 79 | 2.61 | 271.10 | 6715 | 79 |

Выводы:
- В OOS 2025H1 лучше выглядит z1.00/exit0.06 (Sharpe/PNL выше при меньшем turnover).
- Стабильность по шагам (holdout, daily_pnl срезы): z0.95 min/median Sharpe = -5.10/0.09; z1.00 = -2.61/-2.21.
- Концентрация (gross PnL, holdout): top10/top20 = 50%/68% (z1.00) и 44%/64% (z0.95); отрицательных пар 69/141. Top базы: z1.00 → ADA/ETH/CBK/FIL/CHZ, z0.95 → ADA/ETH/CBK/GTAI/CHZ.

### Queue: relaxed8_nokpss_u250_churnfix_oos20250101_20250630_top50_ms0p2 (OOS 2025H1, ms0p2)
- Очередь: `coint4/artifacts/wfa/aggregate/20260123_relaxed8_nokpss_u250_churnfix_oos20250101_20250630_top50_ms0p2/run_queue.csv`.
- Цель: перепроверить OOS 2025H1 для ms0p2 кандидата (holdout+stress).
- Конфиги: z1.00/exit0.06/ms0p2 × holdout/stress (2 прогона).
- Статус: `completed` (2 прогона).

#### Результаты (holdout + stress)
| config | hold_sharpe | hold_pnl | hold_trades | hold_pairs | stress_sharpe | stress_pnl | stress_trades | stress_pairs |
|---|---|---|---|---|---|---|---|---|
| top50/z1p00/exit0p06/ms0p2 | 2.66 | 278.02 | 6694 | 79 | 1.44 | 149.29 | 6694 | 79 |
Выводы:
- Стабильность по шагам (holdout, daily_pnl срезы): min/median Sharpe = -3.67/-3.24.
- Концентрация (gross PnL, holdout): top10/top20 = 50%/68%; отрицательных пар 68/141 (на уровне ms0p1).

### Queue: relaxed8_nokpss_u250_churnfix_oos20250101_20250630_top30_top40 (OOS 2025H1, top30/top40)
- Очередь: `coint4/artifacts/wfa/aggregate/20260124_relaxed8_nokpss_u250_churnfix_oos20250101_20250630_top30_top40/run_queue.csv`.
- Цель: проверить компромисс turnover vs PnL между top20 и top50 для z1.00/exit0.06/ms0p1.
- Конфиги: top30/top40 × holdout/stress (4 прогона).
- Статус: `completed` (4 прогона).

#### Результаты (holdout + stress)
| config | hold_sharpe | hold_pnl | hold_trades | hold_pairs | stress_sharpe | stress_pnl | stress_trades | stress_pairs |
|---|---|---|---|---|---|---|---|---|
| top30/z1p00/exit0p06/hold180/cd180/ms0p1 | 2.49 | 249.86 | 4416 | 55 | 1.37 | 136.25 | 4416 | 55 |
| top40/z1p00/exit0p06/hold180/cd180/ms0p1 | 2.15 | 179.01 | 5781 | 72 | 0.65 | 53.23 | 5781 | 72 |
Выводы:
- top30 снова лучше top40, но заметно уступает топ‑50 по Sharpe/PNL при снижении turnover ~34%.

### Queue: relaxed8_nokpss_u250_churnfix_top50_sens (entry/exit/hold/cd sensitivity)
- Очередь: `coint4/artifacts/wfa/aggregate/20260122_relaxed8_nokpss_u250_churnfix_top50_sens/run_queue.csv`.
- Цель: проверить устойчивость вокруг базовых параметров top50.
- Конфиги: 6 вариантов × holdout/stress (12 прогонов), включая hold120/hold240 (cooldown_hours целое).
- Статус: `completed` (12 прогонов).

#### Результаты (holdout + stress)
| config | hold_sharpe | hold_pnl | hold_trades | hold_pairs | stress_sharpe | stress_pnl | stress_trades | stress_pairs |
|---|---|---|---|---|---|---|---|---|
| top50/z0p90/exit0p06/hold180/cd180/ms0p1 | 7.90 | 1106.56 | 12191 | 120 | 6.70 | 940.26 | 12191 | 120 |
| top50/z0p95/exit0p04/hold180/cd180/ms0p1 | 6.79 | 899.31 | 11055 | 120 | 5.65 | 748.45 | 11055 | 120 |
| top50/z0p95/exit0p06/hold120/cd120/ms0p1 | 7.49 | 911.10 | 13576 | 120 | 5.95 | 724.32 | 13576 | 120 |
| top50/z0p95/exit0p06/hold240/cd240/ms0p1 | 5.02 | 715.34 | 10550 | 120 | 3.92 | 558.22 | 10550 | 120 |
| top50/z0p95/exit0p08/hold180/cd180/ms0p1 | 8.02 | 1179.79 | 12367 | 120 | 6.77 | 997.37 | 12367 | 120 |
| top50/z1p00/exit0p06/hold180/cd180/ms0p1 | 9.01 | 1114.71 | 11414 | 120 | 7.64 | 946.36 | 11414 | 120 |

Выводы:
- Лучшая Sharpe‑устойчивость у z1.00/exit0.06 (Sharpe 9.01/7.64) при сильном PnL и умеренном turnover.
- z0.95/exit0.08 даёт максимальный PnL, но немного ниже Sharpe.
- hold240 ухудшает метрики; hold120 увеличивает сделки без улучшения Sharpe.

### Queue: relaxed8_nokpss_u250_churnfix_top50_churngrid (min_spread_move_sigma)
- Очередь: `coint4/artifacts/wfa/aggregate/20260123_relaxed8_nokpss_u250_churnfix_top50_churngrid/run_queue.csv`.
- Цель: проверить снижение churn через min_spread_move_sigma для базового кандидата z1.00/exit0.06.
- Конфиги: ms0.15/ms0.2 × holdout/stress (4 прогона).
- Статус: `completed` (4 прогона).

#### Результаты (holdout + stress)
| config | hold_sharpe | hold_pnl | hold_trades | hold_pairs | stress_sharpe | stress_pnl | stress_trades | stress_pairs |
|---|---|---|---|---|---|---|---|---|
| top50/ms0p15 | 8.95 | 1107.52 | 11400 | 120 | 7.58 | 939.51 | 11400 | 120 |
| top50/ms0p2 | 9.09 | 1135.26 | 11384 | 120 | 7.73 | 966.13 | 11384 | 120 |

Выводы:
- ms0p2 даёт небольшое улучшение Sharpe/PNL при схожем turnover; кандидат можно сдвинуть на ms0p2.

### Queue: relaxed8_nokpss_u250_churnfix_top50_basecap3 (base cap)
- Очередь: `coint4/artifacts/wfa/aggregate/20260122_relaxed8_nokpss_u250_churnfix_top50_basecap3/run_queue.csv`.
- Цель: ограничить концентрацию по базовым активам (max_per_base=3 в pairs_universe).
- Конфиги: top50 × holdout/stress (2 прогона).
- Статус: `completed` (2 прогона).

#### Результаты (holdout + stress)
| config | hold_sharpe | hold_pnl | hold_trades | hold_pairs | stress_sharpe | stress_pnl | stress_trades | stress_pairs |
|---|---|---|---|---|---|---|---|---|
| top50/basecap3/z0p95/exit0p06/hold180/cd180/ms0p1 | 4.87 | 673.66 | 8459 | 71 | 3.80 | 525.93 | 8459 | 71 |

Выводы:
- basecap3 слишком жёсткий: Sharpe/PNL проседают, торговых пар меньше.

### Fixed backtest smoke (cap1000, pairs_smoke, 2024-01)
- Команда (из `coint4/`): `./.venv/bin/coint2 backtest --config configs/budget_20260122_1000/holdout_relaxed8_nokpss_20260123_top50_z1p00_exit0p06_hold180_cd180_ms0p2_cap1000.yaml --pairs-file bench/pairs_smoke.yaml --period-start 2024-01-01 --period-end 2024-01-31 --out-dir outputs/backtest_smoke_cap1000_top50_202401`
- Цель: быстрый sanity-check пайплайна на загруженных данных (3 пары, 1 месяц).
- Результаты: sharpe 0.1072; total_pnl 876.60; max_drawdown -5456.44; trades 363; win_rate 0.4738; avg_bars_held 17.79.
- Артефакты: `outputs/backtest_smoke_cap1000_top50_202401/` (metrics.yaml, trades.csv, equity.csv).

### Лучшие прогоны (rollup 2026-01-22, realcost) + адаптация под $1000
- Источник: `coint4/artifacts/wfa/aggregate/rollup/run_index.csv` (refresh).
- Отбор: status=completed, metrics_present, total_costs>0, trades>=200, pairs>=20.

| run_id | sharpe | pnl | dd_abs | trades | pairs | costs | run_dir |
|---|---|---|---|---|---|---|---|
| holdout_relaxed8_nokpss_20260123_top50_ms0p2 | 9.09 | 1135.26 | -82.22 | 11384 | 120 | 326.39 | `coint4/artifacts/wfa/runs/20260120_realcost_shortlist/holdout_relaxed8_nokpss_20260123_pv0p2_hurst0p8_hl0p02_60_corr0p4_ssd50000_kpss1p0_w2m1_t90_fixed_u250_top50_z1p00_exit0p06_hold180_cd180_ms0p2` |
| holdout_relaxed8_nokpss_20260125_top30_ms0p2 | 8.79 | 1151.40 | -60.67 | 6865 | 75 | 189.54 | `coint4/artifacts/wfa/runs/20260125_realcost_churngrid/holdout_relaxed8_nokpss_20260125_pv0p2_hurst0p8_hl0p02_60_corr0p4_ssd50000_kpss1p0_w2m1_t90_fixed_u250_top30_z1p00_exit0p06_hold180_cd180_ms0p2` |

Фильтрация/стабильность (по run.log):
- Pair stability filter: 115 → 87 (history=1), 113 → 93 (history=2), 76 → 66 (history=3), 77 → 52 (history=4); шаг 0 — insufficient history.
- В обоих run.log: предупреждения о нулевом gap между train/test (0 дней).

Бюджет $1000:
- Добавлены конфиги: `coint4/configs/budget_20260122_1000/holdout_relaxed8_nokpss_20260123_top50_z1p00_exit0p06_hold180_cd180_ms0p2_cap1000.yaml`, `coint4/configs/budget_20260122_1000/stress_relaxed8_nokpss_20260123_top50_z1p00_exit0p06_hold180_cd180_ms0p2_cap1000.yaml`, `coint4/configs/budget_20260122_1000/holdout_relaxed8_nokpss_20260125_top30_z1p00_exit0p06_hold180_cd180_ms0p2_cap1000.yaml`, `coint4/configs/budget_20260122_1000/stress_relaxed8_nokpss_20260125_top30_z1p00_exit0p06_hold180_cd180_ms0p2_cap1000.yaml`.
- Изменения: initial_capital=1000, min_notional_per_trade=10, max_notional_per_trade=250, pair_stop_loss_usd=7.5 (масштаб 0.1).
- При линейном масштабировании ожидается PnL ~$113–115 и DD ~$6–8, но для $1000 часть дорогих пар может не пройти `_check_capital_sufficiency` (min_position_size=0.01), поэтому фактические метрики могут отличаться.
- Очередь для WFA на удаленном сервере: `coint4/artifacts/wfa/aggregate/20260122_budget1000_top50_top30/run_queue.csv` (status=completed, 4 прогона).

#### Результаты (holdout + stress, cap1000)
| config | hold_sharpe | hold_pnl | hold_dd | hold_trades | hold_pairs | stress_sharpe | stress_pnl | stress_dd | stress_trades | stress_pairs |
|---|---|---|---|---|---|---|---|---|---|---|
| top50/ms0p2 | 8.63 | 1135.26 | -82.22 | 11384 | 120 | 7.39 | 966.13 | -82.91 | 11384 | 120 |
| top30/ms0p2 | 8.76 | 1151.40 | -60.67 | 6865 | 75 | 7.73 | 1015.75 | -63.39 | 6865 | 75 |

Выводы:
- Метрики PnL/издержек/трейдов совпали с $10k версиями (см. rollup), что указывает на отсутствие масштабирования капитала/позиционирования в этих прогонах.
- По профилю рисков лучше выглядит top30 (меньше DD/turnover и издержек) при сопоставимом Sharpe в stress, но выбор предварительный до фикса масштабирования.

#### Фикс масштабирования капитала (Numba) + sanity-check
- Исправление: в Numba-сигнале позиции масштабируются через `capital_at_risk` и учитывают `min/max_notional_per_trade` (см. `coint4/src/coint2/core/numba_kernels.py`, `coint4/src/coint2/engine/numba_backtest_engine_full.py`).
- Быстрый локальный чек (BTC/ETH, 2024-01):
  - С отключенным max_notional: pnl_1k=-181.53, pnl_10k=-1815.31 (ratio=10.00).
  - С бюджетным max_notional=250: pnl_1k=-45.38, pnl_10k=-45.38 (cap биндинг ожидаем).
- Для повторного WFA под фикс: новая очередь `coint4/artifacts/wfa/aggregate/20260122_budget1000_top50_top30_scaled/run_queue.csv` (status=completed).

#### Результаты (holdout + stress, cap1000, scaled)
| config | hold_sharpe | hold_pnl | hold_dd | hold_trades | hold_pairs | stress_sharpe | stress_pnl | stress_dd | stress_trades | stress_pairs |
|---|---|---|---|---|---|---|---|---|---|---|
| top50/ms0p2 | 4.11 | 4724.37 | -760.20 | 11384 | 120 | 3.81 | 4002.68 | -695.20 | 11384 | 120 |
| top30/ms0p2 | 3.15 | 2764.43 | -1107.34 | 6865 | 75 | 2.91 | 2347.79 | -1018.56 | 6865 | 75 |

Выводы:
- Метрики теперь масштабируются с капиталом; значения PnL выросли из-за учета notional per-trade.
- Абсолютные PnL существенно выше initial_capital из-за суммирования по всем парам; требуется отдельная проверка реалистичной агрегированной экспозиции.

### Queue: budget1000_top50_top30_scaled_caps_v2 (entry notional diagnostics)
- Очередь: `coint4/artifacts/wfa/aggregate/20260122_budget1000_top50_top30_scaled_caps_v2/run_queue.csv`.
- Цель: пересчитать scaled WFA с корректными `entry_notional_*` (fallback на pair_data при нулевых y/x).
- Статус: `completed` (4 прогона).
- Примечание: в `coint4/artifacts/wfa/runs/20260122_budget1000_top50_top30_scaled_caps/` `entry_notional_*` были нулями (диагностика не сработала).

#### Entry notional (holdout + stress)
| config | split | entry_count | cap_hits | below_min | notional_avg | notional_p50 | notional_min | notional_max |
|---|---|---|---|---|---|---|---|---|
| top50/ms0p2 | holdout | 11384 | 0 | 0 | 27.18 | 15.00 | 12.31 | 50.97 |
| top30/ms0p2 | holdout | 6865 | 0 | 0 | 24.68 | 15.00 | 12.38 | 45.72 |
| top50/ms0p2 | stress | 11384 | 0 | 0 | 25.02 | 15.00 | 11.73 | 45.47 |
| top30/ms0p2 | stress | 6865 | 0 | 0 | 23.10 | 15.00 | 11.97 | 41.64 |

Выводы:
- `cap_hits=0` и `below_min=0`: фактические notional лежат в диапазоне ~12–51, лимиты min/max не срабатывают.
- Средний notional 23–27 при медиане 15; max_notional=250 не является ограничением.
- Высокие PnL объясняются суммированием по большому числу сделок/пар; стоит проверить реальную агрегированную экспозицию (gross) и лимиты на число позиций.

#### Оценка gross exposure (на основе max_active_positions)
Оценка = `entry_notional_* * max_active_positions / initial_capital` (капитал=1000, max_active_positions=15).

| config | split | notional_avg | notional_p50 | notional_max | gross_avg | gross_p50 | gross_max | max_notional_limit |
|---|---|---|---|---|---|---|---|---|
| top50/ms0p2 | holdout | 27.18 | 15.00 | 50.97 | 0.41 | 0.23 | 0.76 | 250 |
| top30/ms0p2 | holdout | 24.68 | 15.00 | 45.72 | 0.37 | 0.23 | 0.69 | 250 |
| top50/ms0p2 | stress | 25.02 | 15.00 | 45.47 | 0.38 | 0.23 | 0.68 | 250 |
| top30/ms0p2 | stress | 23.10 | 15.00 | 41.64 | 0.35 | 0.23 | 0.62 | 250 |

Выводы:
- Средняя оценка gross exposure 35–41% при p50 ~22.5%; верхняя оценка до ~76% (по max_notional_obs).
- Даже по верхней оценке лимит max_notional=250 не задействован, поэтому имеет смысл проверить tighter cap.

#### Распределение entry_notional_avg по парам (holdout)
Top50:
- Квантили (avg_notional): p10=12.31, p25=12.67, p50=15.00, p75=43.00, p90=50.97, p95=50.97, p99=50.97.
- Топ-5 пар по notional_avg: ADAEUR-ADAUSDC, CELOUSDT-CGPTUSDT, ARTYUSDT-KUBUSDT, CBKUSDT-ENJUSDT, FITFIUSDT-JUVUSDT (все ~50.97).

Top30:
- Квантили (avg_notional): p10=12.38, p25=13.04, p50=15.00, p75=36.27, p90=45.72, p95=45.72, p99=45.72.
- Топ-5 пар по notional_avg: BTCDAI-BTCEUR, FTTUSDT-KCSUSDT, HOOKUSDT-KAVAUSDT, INJUSDT-KSMUSDT, ADAUSDC-KUBUSDT (все ~45.72).

Выводы:
- Распределение дискретное: заметные кластеры около ~15 и верхнего значения (45–51), что указывает на влияние правил sizing и min/max границ.

### Queue: budget1000_capsweep_maxnot25 (completed)
- Очередь: `coint4/artifacts/wfa/aggregate/20260123_budget1000_capsweep_maxnot25/run_queue.csv`.
- Цель: проверить tighter cap по `max_notional_per_trade=25` при текущей сетке top50/top30.
- Конфиги:
  - `coint4/configs/budget_20260123_1000_capsweep/holdout_relaxed8_nokpss_20260123_top50_z1p00_exit0p06_hold180_cd180_ms0p2_cap1000_maxnot25.yaml`
  - `coint4/configs/budget_20260123_1000_capsweep/stress_relaxed8_nokpss_20260123_top50_z1p00_exit0p06_hold180_cd180_ms0p2_cap1000_maxnot25.yaml`
  - `coint4/configs/budget_20260123_1000_capsweep/holdout_relaxed8_nokpss_20260125_top30_z1p00_exit0p06_hold180_cd180_ms0p2_cap1000_maxnot25.yaml`
  - `coint4/configs/budget_20260123_1000_capsweep/stress_relaxed8_nokpss_20260125_top30_z1p00_exit0p06_hold180_cd180_ms0p2_cap1000_maxnot25.yaml`
- Статус: `completed` (4 прогона).

#### Результаты (holdout + stress, cap1000, max_notional=25)
| config | hold_sharpe | hold_pnl | hold_dd | hold_trades | hold_pairs | hold_costs | stress_sharpe | stress_pnl | stress_dd | stress_trades | stress_pairs | stress_costs |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| top50/ms0p2 | 3.70 | 3316.63 | -688.69 | 11384 | 120 | 393.27 | 3.48 | 2967.65 | -695.20 | 11384 | 120 | 687.70 |
| top30/ms0p2 | 3.05 | 2244.03 | -725.06 | 6865 | 75 | 238.61 | 2.85 | 1983.11 | -733.60 | 6865 | 75 | 418.49 |

#### Entry notional (holdout + stress, max_notional=25)
| config | split | entry_count | cap_hits | cap_hit_pct | notional_avg | notional_p50 | notional_min | notional_max |
|---|---|---|---|---|---|---|---|---|
| top50/ms0p2 | holdout | 11384 | 4705 | 41.33% | 18.13 | 15.00 | 12.31 | 25.00 |
| top30/ms0p2 | holdout | 6865 | 2810 | 40.93% | 18.18 | 15.00 | 12.38 | 25.00 |
| top50/ms0p2 | stress | 11384 | 4705 | 41.33% | 17.84 | 15.00 | 11.73 | 25.00 |
| top30/ms0p2 | stress | 6865 | 2810 | 40.93% | 17.94 | 15.00 | 11.97 | 25.00 |

#### Оценка gross exposure (max_notional=25)
Оценка = `entry_notional_* * max_active_positions / initial_capital` (капитал=1000, max_active_positions=15).

| config | split | gross_avg | gross_p50 | gross_max |
|---|---|---|---|---|
| top50/ms0p2 | holdout | 0.27 | 0.23 | 0.38 |
| top30/ms0p2 | holdout | 0.27 | 0.23 | 0.38 |
| top50/ms0p2 | stress | 0.27 | 0.23 | 0.38 |
| top30/ms0p2 | stress | 0.27 | 0.23 | 0.38 |

Выводы:
- Cap=25 стал активным (cap_hit_pct ~41%); notional_avg снизился до ~18 (vs ~23–27 на max_notional=250).
- PnL и Sharpe снизились: top50 holdout 3316 vs 4724 (baseline), Sharpe 3.70 vs 4.11.
- По gross exposure средний уровень опустился к ~27% (max ~38%), что делает экспозицию более реалистичной, но ухудшает доходность.

### Queue: budget1000_capsweep_maxnot50 (planned)
- Очередь: `coint4/artifacts/wfa/aggregate/20260123_budget1000_capsweep_maxnot50/run_queue.csv`.
- Цель: промежуточный cap (max_notional=50) для компромисса между доходностью и экспозицией.
- Конфиги:
  - `coint4/configs/budget_20260123_1000_capsweep_maxnot50/holdout_relaxed8_nokpss_20260123_top50_z1p00_exit0p06_hold180_cd180_ms0p2_cap1000_maxnot50.yaml`
  - `coint4/configs/budget_20260123_1000_capsweep_maxnot50/stress_relaxed8_nokpss_20260123_top50_z1p00_exit0p06_hold180_cd180_ms0p2_cap1000_maxnot50.yaml`
  - `coint4/configs/budget_20260123_1000_capsweep_maxnot50/holdout_relaxed8_nokpss_20260125_top30_z1p00_exit0p06_hold180_cd180_ms0p2_cap1000_maxnot50.yaml`
  - `coint4/configs/budget_20260123_1000_capsweep_maxnot50/stress_relaxed8_nokpss_20260125_top30_z1p00_exit0p06_hold180_cd180_ms0p2_cap1000_maxnot50.yaml`

### Queue: budget1000_capsweep_maxnot100 (planned)
- Очередь: `coint4/artifacts/wfa/aggregate/20260123_budget1000_capsweep_maxnot100/run_queue.csv`.
- Цель: более мягкий cap (max_notional=100) для сравнения с baseline 250 и cap25.
- Конфиги:
  - `coint4/configs/budget_20260123_1000_capsweep_maxnot100/holdout_relaxed8_nokpss_20260123_top50_z1p00_exit0p06_hold180_cd180_ms0p2_cap1000_maxnot100.yaml`
  - `coint4/configs/budget_20260123_1000_capsweep_maxnot100/stress_relaxed8_nokpss_20260123_top50_z1p00_exit0p06_hold180_cd180_ms0p2_cap1000_maxnot100.yaml`
  - `coint4/configs/budget_20260123_1000_capsweep_maxnot100/holdout_relaxed8_nokpss_20260125_top30_z1p00_exit0p06_hold180_cd180_ms0p2_cap1000_maxnot100.yaml`
  - `coint4/configs/budget_20260123_1000_capsweep_maxnot100/stress_relaxed8_nokpss_20260125_top30_z1p00_exit0p06_hold180_cd180_ms0p2_cap1000_maxnot100.yaml`
