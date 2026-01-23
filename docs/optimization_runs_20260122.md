# Журнал прогонов оптимизации (2026-01-22)

Назначение: churn-control micro-grid после включения cooldown/min_hold/min_spread_move, проверка компромисса Sharpe/PnL/turnover.

Примечание по метрикам:
- `sharpe_ratio_abs` считается по `equity_curve` (pct_change), `sharpe_ratio_on_returns` — по `PnL / capital_per_pair` (см. `coint4/src/coint2/core/performance.py`, `coint4/src/coint2/pipeline/walk_forward_orchestrator.py`).

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

### Queue: budget1000_capsweep_maxnot50 (completed)
- Очередь: `coint4/artifacts/wfa/aggregate/20260123_budget1000_capsweep_maxnot50/run_queue.csv`.
- Цель: промежуточный cap (max_notional=50) для компромисса между доходностью и экспозицией.
- Конфиги:
  - `coint4/configs/budget_20260123_1000_capsweep_maxnot50/holdout_relaxed8_nokpss_20260123_top50_z1p00_exit0p06_hold180_cd180_ms0p2_cap1000_maxnot50.yaml`
  - `coint4/configs/budget_20260123_1000_capsweep_maxnot50/stress_relaxed8_nokpss_20260123_top50_z1p00_exit0p06_hold180_cd180_ms0p2_cap1000_maxnot50.yaml`
  - `coint4/configs/budget_20260123_1000_capsweep_maxnot50/holdout_relaxed8_nokpss_20260125_top30_z1p00_exit0p06_hold180_cd180_ms0p2_cap1000_maxnot50.yaml`
  - `coint4/configs/budget_20260123_1000_capsweep_maxnot50/stress_relaxed8_nokpss_20260125_top30_z1p00_exit0p06_hold180_cd180_ms0p2_cap1000_maxnot50.yaml`
- Статус: `completed` (4 прогона).

#### Результаты (holdout + stress, cap1000, max_notional=50)
| config | hold_sharpe | hold_pnl | hold_dd | hold_trades | hold_pairs | hold_costs | stress_sharpe | stress_pnl | stress_dd | stress_trades | stress_pairs | stress_costs |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| top50/ms0p2 | 4.10 | 4680.03 | -745.71 | 11384 | 120 | 586.86 | 3.81 | 4002.68 | -695.20 | 11384 | 120 | 967.06 |
| top30/ms0p2 | 3.15 | 2764.43 | -1107.34 | 6865 | 75 | 324.41 | 2.91 | 2347.79 | -1018.56 | 6865 | 75 | 539.73 |

#### Entry notional (holdout + stress, max_notional=50)
| config | split | entry_count | cap_hits | cap_hit_pct | notional_avg | notional_p50 | notional_min | notional_max |
|---|---|---|---|---|---|---|---|---|
| top50/ms0p2 | holdout | 11384 | 2300 | 20.20% | 26.99 | 15.00 | 12.31 | 50.00 |
| top30/ms0p2 | holdout | 6865 | 0 | 0.00% | 24.68 | 15.00 | 12.38 | 45.72 |
| top50/ms0p2 | stress | 11384 | 0 | 0.00% | 25.02 | 15.00 | 11.73 | 45.47 |
| top30/ms0p2 | stress | 6865 | 0 | 0.00% | 23.10 | 15.00 | 11.97 | 41.64 |

#### Оценка gross exposure (max_notional=50)
Оценка = `entry_notional_* * max_active_positions / initial_capital` (капитал=1000, max_active_positions=15).

| config | split | gross_avg | gross_p50 | gross_max |
|---|---|---|---|---|
| top50/ms0p2 | holdout | 0.40 | 0.23 | 0.75 |
| top30/ms0p2 | holdout | 0.37 | 0.23 | 0.69 |
| top50/ms0p2 | stress | 0.38 | 0.23 | 0.68 |
| top30/ms0p2 | stress | 0.35 | 0.23 | 0.62 |

Выводы:
- Cap=50 частично активен только в top50 holdout (cap_hit_pct ~20%); в остальных сплитах cap не срабатывает.
- Метрики близки к baseline (max_notional=250); для top30 изменений нет.

### Queue: budget1000_capsweep_maxnot100 (completed)
- Очередь: `coint4/artifacts/wfa/aggregate/20260123_budget1000_capsweep_maxnot100/run_queue.csv`.
- Цель: более мягкий cap (max_notional=100) для сравнения с baseline 250 и cap25.
- Конфиги:
  - `coint4/configs/budget_20260123_1000_capsweep_maxnot100/holdout_relaxed8_nokpss_20260123_top50_z1p00_exit0p06_hold180_cd180_ms0p2_cap1000_maxnot100.yaml`
  - `coint4/configs/budget_20260123_1000_capsweep_maxnot100/stress_relaxed8_nokpss_20260123_top50_z1p00_exit0p06_hold180_cd180_ms0p2_cap1000_maxnot100.yaml`
  - `coint4/configs/budget_20260123_1000_capsweep_maxnot100/holdout_relaxed8_nokpss_20260125_top30_z1p00_exit0p06_hold180_cd180_ms0p2_cap1000_maxnot100.yaml`
  - `coint4/configs/budget_20260123_1000_capsweep_maxnot100/stress_relaxed8_nokpss_20260125_top30_z1p00_exit0p06_hold180_cd180_ms0p2_cap1000_maxnot100.yaml`
- Статус: `completed` (4 прогона).

#### Результаты (holdout + stress, cap1000, max_notional=100)
| config | hold_sharpe | hold_pnl | hold_dd | hold_trades | hold_pairs | hold_costs | stress_sharpe | stress_pnl | stress_dd | stress_trades | stress_pairs | stress_costs |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| top50/ms0p2 | 4.11 | 4724.37 | -760.20 | 11384 | 120 | 591.31 | 3.81 | 4002.68 | -695.20 | 11384 | 120 | 967.06 |
| top30/ms0p2 | 3.15 | 2764.43 | -1107.34 | 6865 | 75 | 324.41 | 2.91 | 2347.79 | -1018.56 | 6865 | 75 | 539.73 |

#### Entry notional (holdout + stress, max_notional=100)
| config | split | entry_count | cap_hits | cap_hit_pct | notional_avg | notional_p50 | notional_min | notional_max |
|---|---|---|---|---|---|---|---|---|
| top50/ms0p2 | holdout | 11384 | 0 | 0.00% | 27.18 | 15.00 | 12.31 | 50.97 |
| top30/ms0p2 | holdout | 6865 | 0 | 0.00% | 24.68 | 15.00 | 12.38 | 45.72 |
| top50/ms0p2 | stress | 11384 | 0 | 0.00% | 25.02 | 15.00 | 11.73 | 45.47 |
| top30/ms0p2 | stress | 6865 | 0 | 0.00% | 23.10 | 15.00 | 11.97 | 41.64 |

#### Оценка gross exposure (max_notional=100)
Оценка = `entry_notional_* * max_active_positions / initial_capital` (капитал=1000, max_active_positions=15).

| config | split | gross_avg | gross_p50 | gross_max |
|---|---|---|---|---|
| top50/ms0p2 | holdout | 0.41 | 0.23 | 0.76 |
| top30/ms0p2 | holdout | 0.37 | 0.23 | 0.69 |
| top50/ms0p2 | stress | 0.38 | 0.23 | 0.68 |
| top30/ms0p2 | stress | 0.35 | 0.23 | 0.62 |

Выводы:
- Cap=100 не активен (cap_hits=0); метрики совпадают с baseline max_notional=250.

### Queue: budget1000_capsweep_maxnot25_risk0p75 (completed)
- Очередь: `coint4/artifacts/wfa/aggregate/20260123_budget1000_capsweep_maxnot25_risk0p75/run_queue.csv`.
- Цель: проверить снижение `risk_per_position_pct` до 0.0075 при cap=25.
- Конфиги:
  - `coint4/configs/budget_20260123_1000_capsweep_maxnot25_risk0p75/holdout_relaxed8_nokpss_20260123_top50_z1p00_exit0p06_hold180_cd180_ms0p2_cap1000_maxnot25_risk0p75.yaml`
  - `coint4/configs/budget_20260123_1000_capsweep_maxnot25_risk0p75/stress_relaxed8_nokpss_20260123_top50_z1p00_exit0p06_hold180_cd180_ms0p2_cap1000_maxnot25_risk0p75.yaml`
  - `coint4/configs/budget_20260123_1000_capsweep_maxnot25_risk0p75/holdout_relaxed8_nokpss_20260125_top30_z1p00_exit0p06_hold180_cd180_ms0p2_cap1000_maxnot25_risk0p75.yaml`
  - `coint4/configs/budget_20260123_1000_capsweep_maxnot25_risk0p75/stress_relaxed8_nokpss_20260125_top30_z1p00_exit0p06_hold180_cd180_ms0p2_cap1000_maxnot25_risk0p75.yaml`
- Статус: `completed` (4 прогона).

#### Результаты (holdout + stress, cap1000, risk=0.0075)
| config | hold_sharpe | hold_pnl | hold_dd | hold_trades | hold_pairs | hold_costs | stress_sharpe | stress_pnl | stress_dd | stress_trades | stress_pairs | stress_costs |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| top50/ms0p2 | 0.00 | 0.00 | 0.00 | 0 | 120 | 0.00 | 0.00 | 0.00 | 0.00 | 0 | 120 | 0.00 |
| top30/ms0p2 | 0.00 | 0.00 | 0.00 | 0 | 75 | 0.00 | 0.00 | 0.00 | 0.00 | 0 | 75 | 0.00 |

Выводы:
- При `risk_per_position_pct=0.0075` trade_notional=7.5 < min_notional=10, поэтому входы блокируются (0 сделок).
- Для проверки этой ветки нужно либо снизить `min_notional_per_trade`, либо поднять риск (например, 0.01).

### Queue: budget1000_capsweep_maxnot25_risk_minnot_grid (completed)
- Очередь: `coint4/artifacts/wfa/aggregate/20260123_budget1000_capsweep_maxnot25_risk_minnot_grid/run_queue.csv`.
- Цель: micro-grid для `risk_per_position_pct` (0.01/0.0125/0.02) при cap=25 + проверка `min_notional_per_trade` (7.5/5) при `risk_per_position_pct=0.0075`.
- Конфиги:
  - `coint4/configs/budget_20260123_1000_capsweep_maxnot25_riskgrid/holdout_relaxed8_nokpss_20260123_top50_z1p00_exit0p06_hold180_cd180_ms0p2_cap1000_maxnot25_risk0p01.yaml`
  - `coint4/configs/budget_20260123_1000_capsweep_maxnot25_riskgrid/stress_relaxed8_nokpss_20260123_top50_z1p00_exit0p06_hold180_cd180_ms0p2_cap1000_maxnot25_risk0p01.yaml`
  - `coint4/configs/budget_20260123_1000_capsweep_maxnot25_riskgrid/holdout_relaxed8_nokpss_20260123_top50_z1p00_exit0p06_hold180_cd180_ms0p2_cap1000_maxnot25_risk0p0125.yaml`
  - `coint4/configs/budget_20260123_1000_capsweep_maxnot25_riskgrid/stress_relaxed8_nokpss_20260123_top50_z1p00_exit0p06_hold180_cd180_ms0p2_cap1000_maxnot25_risk0p0125.yaml`
  - `coint4/configs/budget_20260123_1000_capsweep_maxnot25_riskgrid/holdout_relaxed8_nokpss_20260123_top50_z1p00_exit0p06_hold180_cd180_ms0p2_cap1000_maxnot25_risk0p02.yaml`
  - `coint4/configs/budget_20260123_1000_capsweep_maxnot25_riskgrid/stress_relaxed8_nokpss_20260123_top50_z1p00_exit0p06_hold180_cd180_ms0p2_cap1000_maxnot25_risk0p02.yaml`
  - `coint4/configs/budget_20260123_1000_capsweep_maxnot25_minnotgrid/holdout_relaxed8_nokpss_20260123_top50_z1p00_exit0p06_hold180_cd180_ms0p2_cap1000_maxnot25_risk0p75_minnot7p5.yaml`
  - `coint4/configs/budget_20260123_1000_capsweep_maxnot25_minnotgrid/stress_relaxed8_nokpss_20260123_top50_z1p00_exit0p06_hold180_cd180_ms0p2_cap1000_maxnot25_risk0p75_minnot7p5.yaml`
  - `coint4/configs/budget_20260123_1000_capsweep_maxnot25_minnotgrid/holdout_relaxed8_nokpss_20260123_top50_z1p00_exit0p06_hold180_cd180_ms0p2_cap1000_maxnot25_risk0p75_minnot5.yaml`
  - `coint4/configs/budget_20260123_1000_capsweep_maxnot25_minnotgrid/stress_relaxed8_nokpss_20260123_top50_z1p00_exit0p06_hold180_cd180_ms0p2_cap1000_maxnot25_risk0p75_minnot5.yaml`
- Статус: `completed` (10 прогонов).

#### Результаты (holdout + stress, cap1000, max_notional=25)
| config | hold_sharpe | hold_pnl | hold_dd | hold_trades | hold_pairs | hold_costs | stress_sharpe | stress_pnl | stress_dd | stress_trades | stress_pairs | stress_costs |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| risk0p01 | -0.55 | -119.63 | -459.13 | 2139 | 120 | 40.33 | -0.69 | -142.93 | -463.47 | 2139 | 120 | 71.69 |
| risk0p0125 | 3.93 | 3059.40 | -573.91 | 11384 | 120 | 368.16 | 3.70 | 2767.31 | -579.33 | 11384 | 120 | 646.47 |
| risk0p02 | 3.36 | 3763.63 | -918.25 | 11384 | 120 | 439.74 | 3.15 | 3283.57 | -926.93 | 11384 | 120 | 761.84 |
| risk0p75/minnot7p5 | -0.59 | -89.72 | -344.35 | 2139 | 120 | 30.24 | -0.73 | -107.20 | -347.60 | 2139 | 120 | 53.77 |
| risk0p75/minnot5 | 4.28 | 1976.41 | -344.35 | 11384 | 120 | 232.98 | 4.01 | 1771.05 | -353.57 | 11384 | 120 | 398.42 |

#### Entry notional (holdout + stress)
| config | split | entry_count | cap_hits | below_min | notional_avg | notional_p50 | notional_min | notional_max |
|---|---|---|---|---|---|---|---|---|
| risk0p01 | holdout | 2139 | 0 | 2139 | 10.00 | 0.00 | 0.00 | 10.00 |
| risk0p01 | stress | 2139 | 0 | 2139 | 10.00 | 0.00 | 0.00 | 10.00 |
| risk0p0125 | holdout | 11384 | 4705 | 0 | 16.97 | 12.50 | 10.63 | 25.00 |
| risk0p0125 | stress | 11384 | 4705 | 0 | 16.77 | 12.50 | 10.23 | 25.00 |
| risk0p02 | holdout | 11384 | 4705 | 0 | 20.28 | 20.00 | 15.21 | 25.00 |
| risk0p02 | stress | 11384 | 4705 | 0 | 19.77 | 20.00 | 14.20 | 25.00 |
| risk0p75/minnot7p5 | holdout | 2139 | 0 | 2139 | 7.50 | 0.00 | 0.00 | 7.50 |
| risk0p75/minnot7p5 | stress | 2139 | 0 | 2139 | 7.50 | 0.00 | 0.00 | 7.50 |
| risk0p75/minnot5 | holdout | 11384 | 0 | 0 | 10.73 | 7.50 | 6.83 | 16.63 |
| risk0p75/minnot5 | stress | 11384 | 0 | 0 | 10.32 | 7.50 | 6.68 | 15.68 |

Выводы:
- `risk0p01`: много `entry_notional_below_min` и резкое падение Sharpe/PnL; текущий min_notional блокирует значимую часть входов.
- `risk0p0125`: лучший компромисс по Sharpe и DD при умеренном снижении PnL относительно базового risk0p015 (cap25).
- `risk0p02`: выше PnL, но глубже просадка и ниже Sharpe — более агрессивный профиль.
- `risk0p75/minnot7p5`: даже при min_notional=7.5 входы остаются заблокированными.
- `risk0p75/minnot5`: сделки восстановились, Sharpe самый высокий, но PnL заметно ниже — вариант для более консервативной цели.

### Queue: budget1000_capsweep_maxnot50_posgrid (completed)
- Очередь: `coint4/artifacts/wfa/aggregate/20260123_budget1000_capsweep_maxnot50_posgrid/run_queue.csv`.
- Цель: проверить влияние `max_active_positions` (10/12 vs базовые 15) при cap=50 и top50.
- Конфиги:
  - `coint4/configs/budget_20260123_1000_capsweep_maxnot50_posgrid/holdout_relaxed8_nokpss_20260123_top50_z1p00_exit0p06_hold180_cd180_ms0p2_cap1000_maxnot50_maxpos10.yaml`
  - `coint4/configs/budget_20260123_1000_capsweep_maxnot50_posgrid/stress_relaxed8_nokpss_20260123_top50_z1p00_exit0p06_hold180_cd180_ms0p2_cap1000_maxnot50_maxpos10.yaml`
  - `coint4/configs/budget_20260123_1000_capsweep_maxnot50_posgrid/holdout_relaxed8_nokpss_20260123_top50_z1p00_exit0p06_hold180_cd180_ms0p2_cap1000_maxnot50_maxpos12.yaml`
  - `coint4/configs/budget_20260123_1000_capsweep_maxnot50_posgrid/stress_relaxed8_nokpss_20260123_top50_z1p00_exit0p06_hold180_cd180_ms0p2_cap1000_maxnot50_maxpos12.yaml`
- Статус: `completed` (4 прогона).

#### Результаты (holdout + stress, cap1000, max_notional=50)
| config | hold_sharpe | hold_pnl | hold_dd | hold_trades | hold_pairs | hold_costs | stress_sharpe | stress_pnl | stress_dd | stress_trades | stress_pairs | stress_costs |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| top50/maxpos10 | -1.23 | -395.22 | -689.44 | 4326 | 120 | 101.36 | -1.49 | -357.78 | -618.61 | 2139 | 120 | 107.53 |
| top50/maxpos12 | -1.74 | -414.28 | -679.73 | 2139 | 120 | 60.49 | -1.88 | -443.54 | -696.24 | 2139 | 120 | 107.53 |

#### Entry notional (holdout + stress)
| config | split | entry_count | below_min | notional_avg | notional_p50 | notional_min | notional_max |
|---|---|---|---|---|---|---|---|
| top50/maxpos10 | holdout | 4326 | 2187 | 12.48 | 0.00 | 0.00 | 15.00 |
| top50/maxpos12 | holdout | 2139 | 0 | 15.00 | 0.00 | 0.00 | 15.00 |
| top50/maxpos10 | stress | 2139 | 0 | 15.00 | 0.00 | 0.00 | 15.00 |
| top50/maxpos12 | stress | 2139 | 0 | 15.00 | 0.00 | 0.00 | 15.00 |

Выводы:
- Снижение max_active_positions до 10/12 приводит к отрицательному PnL и Sharpe.
- В maxpos10 holdout большое число `entry_notional_below_min`; p50=0 указывает на нулевые значения в диагностике (нужно проверить корректность entry_notional при низкой активности).

### Queue: budget1000_oos20230501_20231231_top50_maxnot25 (completed)
- Очередь: `coint4/artifacts/wfa/aggregate/20260123_budget1000_oos20230501_20231231_top50_maxnot25/run_queue.csv`.
- Цель: OOS 2023H2 для top50 cap25.
- Конфиги:
  - `coint4/configs/budget_20260123_1000_capsweep_oos20230501_20231231_top50_maxnot25/holdout_relaxed8_nokpss_20260123_oos20230501_20231231_top50_z1p00_exit0p06_hold180_cd180_ms0p2_cap1000_maxnot25.yaml`
  - `coint4/configs/budget_20260123_1000_capsweep_oos20230501_20231231_top50_maxnot25/stress_relaxed8_nokpss_20260123_oos20230501_20231231_top50_z1p00_exit0p06_hold180_cd180_ms0p2_cap1000_maxnot25.yaml`
- Статус: `completed` (2 прогона).

#### Результаты (holdout + stress, OOS 2023H2, cap1000, max_notional=25)
| config | hold_sharpe | hold_pnl | hold_dd | hold_trades | hold_pairs | hold_costs | stress_sharpe | stress_pnl | stress_dd | stress_trades | stress_pairs | stress_costs |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| top50/ms0p2 | 1.55 | 471.06 | -646.70 | 6871 | 69 | 233.44 | 1.17 | 291.74 | -643.45 | 6871 | 69 | 395.18 |

#### Entry notional (OOS 2023H2)
| split | entry_count | cap_hits | below_min | notional_avg | notional_p50 | notional_min | notional_max |
|---|---|---|---|---|---|---|---|
| holdout | 6871 | 0 | 0 | 18.38 | 17.86 | 15.00 | 24.73 |
| stress | 6871 | 0 | 0 | 17.51 | 16.00 | 15.00 | 22.92 |

Выводы:
- OOS 2023H2 остаётся прибыльным; Sharpe 1.17–1.55 при умеренном DD.
- Cap=25 не ограничивает (max_notional < 25).

### Queue: budget1000_oos20250101_20250630_top50_maxnot25 (completed)
- Очередь: `coint4/artifacts/wfa/aggregate/20260123_budget1000_oos20250101_20250630_top50_maxnot25/run_queue.csv`.
- Цель: OOS 2025H1 (3 шага) для top50 cap25.
- Конфиги:
  - `coint4/configs/budget_20260123_1000_capsweep_oos20250101_20250630_top50_maxnot25/holdout_relaxed8_nokpss_20260123_oos20250101_20250630_top50_z1p00_exit0p06_hold180_cd180_ms0p2_cap1000_maxnot25.yaml`
  - `coint4/configs/budget_20260123_1000_capsweep_oos20250101_20250630_top50_maxnot25/stress_relaxed8_nokpss_20260123_oos20250101_20250630_top50_z1p00_exit0p06_hold180_cd180_ms0p2_cap1000_maxnot25.yaml`
- Статус: `completed` (2 прогона).

#### Результаты (holdout + stress, OOS 2025H1, cap1000, max_notional=25)
| config | hold_sharpe | hold_pnl | hold_dd | hold_trades | hold_pairs | hold_costs | stress_sharpe | stress_pnl | stress_dd | stress_trades | stress_pairs | stress_costs |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| top50/ms0p2 | 1.96 | 443.56 | -716.08 | 6694 | 79 | 232.29 | 1.59 | 318.28 | -738.34 | 6694 | 79 | 400.82 |

#### Entry notional (OOS 2025H1)
| split | entry_count | cap_hits | below_min | notional_avg | notional_p50 | notional_min | notional_max |
|---|---|---|---|---|---|---|---|
| holdout | 6694 | 0 | 0 | 18.74 | 19.15 | 15.00 | 22.85 |
| stress | 6694 | 0 | 0 | 18.19 | 18.02 | 15.00 | 22.33 |

Выводы:
- OOS 2025H1 показывает Sharpe 1.59–1.96 и положительный PnL.
- Cap=25 не ограничивает (max_notional < 25); экспозиция определяется risk_per_position.

### Queue: budget1000_oos_risk_minnot_grid (completed)
- Очередь: `coint4/artifacts/wfa/aggregate/20260123_budget1000_oos_risk_minnot_grid/run_queue.csv`.
- Цель: OOS 2023H2 и 2025H1 для кандидатов `risk0p0125` и `risk0p75/minnot5` при cap25 (top50).
- Конфиги (2023H2):
  - `coint4/configs/budget_20260123_1000_capsweep_oos20230501_20231231_top50_maxnot25_riskgrid/holdout_relaxed8_nokpss_20260123_oos20230501_20231231_top50_z1p00_exit0p06_hold180_cd180_ms0p2_cap1000_maxnot25_risk0p0125.yaml`
  - `coint4/configs/budget_20260123_1000_capsweep_oos20230501_20231231_top50_maxnot25_riskgrid/stress_relaxed8_nokpss_20260123_oos20230501_20231231_top50_z1p00_exit0p06_hold180_cd180_ms0p2_cap1000_maxnot25_risk0p0125.yaml`
  - `coint4/configs/budget_20260123_1000_capsweep_oos20230501_20231231_top50_maxnot25_riskgrid/holdout_relaxed8_nokpss_20260123_oos20230501_20231231_top50_z1p00_exit0p06_hold180_cd180_ms0p2_cap1000_maxnot25_risk0p75_minnot5.yaml`
  - `coint4/configs/budget_20260123_1000_capsweep_oos20230501_20231231_top50_maxnot25_riskgrid/stress_relaxed8_nokpss_20260123_oos20230501_20231231_top50_z1p00_exit0p06_hold180_cd180_ms0p2_cap1000_maxnot25_risk0p75_minnot5.yaml`
- Конфиги (2025H1):
  - `coint4/configs/budget_20260123_1000_capsweep_oos20250101_20250630_top50_maxnot25_riskgrid/holdout_relaxed8_nokpss_20260123_oos20250101_20250630_top50_z1p00_exit0p06_hold180_cd180_ms0p2_cap1000_maxnot25_risk0p0125.yaml`
  - `coint4/configs/budget_20260123_1000_capsweep_oos20250101_20250630_top50_maxnot25_riskgrid/stress_relaxed8_nokpss_20260123_oos20250101_20250630_top50_z1p00_exit0p06_hold180_cd180_ms0p2_cap1000_maxnot25_risk0p0125.yaml`
  - `coint4/configs/budget_20260123_1000_capsweep_oos20250101_20250630_top50_maxnot25_riskgrid/holdout_relaxed8_nokpss_20260123_oos20250101_20250630_top50_z1p00_exit0p06_hold180_cd180_ms0p2_cap1000_maxnot25_risk0p75_minnot5.yaml`
  - `coint4/configs/budget_20260123_1000_capsweep_oos20250101_20250630_top50_maxnot25_riskgrid/stress_relaxed8_nokpss_20260123_oos20250101_20250630_top50_z1p00_exit0p06_hold180_cd180_ms0p2_cap1000_maxnot25_risk0p75_minnot5.yaml`
- Статус: `completed` (8 прогонов).

#### Результаты (holdout + stress, OOS 2023H2)
| config | hold_sharpe | hold_pnl | hold_dd | hold_trades | hold_pairs | hold_costs | stress_sharpe | stress_pnl | stress_dd | stress_trades | stress_pairs | stress_costs |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| risk0p0125 | 1.58 | 421.31 | -514.38 | 6871 | 69 | 189.73 | 1.21 | 276.26 | -510.21 | 6871 | 69 | 323.81 |
| risk0p75/minnot5 | 1.66 | 281.69 | -279.31 | 6871 | 69 | 107.17 | 1.28 | 201.39 | -280.47 | 6871 | 69 | 185.89 |

#### Entry notional (OOS 2023H2)
| config | split | entry_count | cap_hits | below_min | notional_avg | notional_p50 | notional_min | notional_max |
|---|---|---|---|---|---|---|---|---|
| risk0p0125 | holdout | 6871 | 0 | 0 | 14.94 | 14.85 | 12.50 | 19.33 |
| risk0p0125 | stress | 6871 | 0 | 0 | 14.35 | 13.57 | 12.50 | 18.13 |
| risk0p75/minnot5 | holdout | 6871 | 0 | 0 | 8.44 | 8.60 | 7.50 | 9.99 |
| risk0p75/minnot5 | stress | 6871 | 0 | 0 | 8.24 | 8.15 | 7.50 | 9.60 |

#### Результаты (holdout + stress, OOS 2025H1)
| config | hold_sharpe | hold_pnl | hold_dd | hold_trades | hold_pairs | hold_costs | stress_sharpe | stress_pnl | stress_dd | stress_trades | stress_pairs | stress_costs |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| risk0p0125 | 1.99 | 377.64 | -572.14 | 6694 | 79 | 187.81 | 1.61 | 276.69 | -593.00 | 6694 | 79 | 325.59 |
| risk0p75/minnot5 | 2.07 | 235.28 | -311.68 | 6694 | 79 | 105.30 | 1.66 | 179.00 | -326.69 | 6694 | 79 | 184.31 |

#### Entry notional (OOS 2025H1)
| config | split | entry_count | cap_hits | below_min | notional_avg | notional_p50 | notional_min | notional_max |
|---|---|---|---|---|---|---|---|---|
| risk0p0125 | holdout | 6694 | 0 | 0 | 15.15 | 15.53 | 12.50 | 17.95 |
| risk0p0125 | stress | 6694 | 0 | 0 | 14.77 | 14.76 | 12.50 | 17.59 |
| risk0p75/minnot5 | holdout | 6694 | 0 | 0 | 8.49 | 8.70 | 7.50 | 9.46 |
| risk0p75/minnot5 | stress | 6694 | 0 | 0 | 8.36 | 8.43 | 7.50 | 9.33 |

Выводы:
- OOS 2023H2/2025H1: `risk0p0125` даёт выше PnL, но глубже DD.
- `risk0p75/minnot5` даёт меньший PnL, зато более высокий Sharpe и заметно меньшую просадку.
- В обоих периодах cap=25 не активен (cap_hits=0), разница определяется риск‑параметрами и min_notional.

#### Концентрация (gross PnL, holdout)
| period | config | top10_share | top20_share | neg_pairs | total_pairs |
|---|---|---|---|---|---|
| 2023H2 | risk0p0125 | 64% | 87% | 28 | 69 |
| 2023H2 | risk0p75/minnot5 | 64% | 86% | 28 | 69 |
| 2025H1 | risk0p0125 | 66% | 88% | 31 | 79 |
| 2025H1 | risk0p75/minnot5 | 65% | 87% | 30 | 79 |
| 2024H1 | risk0p0125 | 85% | 98% | 24 | 53 |
| 2024H1 | risk0p75/minnot5 | 85% | 98% | 24 | 53 |

### Queue: budget1000_oos20240101_20240630_top50_maxnot25_risk0p75_minnot5 (completed)
- Очередь: `coint4/artifacts/wfa/aggregate/20260123_budget1000_oos20240101_20240630_top50_maxnot25_risk0p75_minnot5/run_queue.csv`.
- Цель: дополнительный OOS 2024H1 (2024-01-01 → 2024-06-30, max_steps=3) для `risk0p75/minnot5`.
- Конфиги:
  - `coint4/configs/budget_20260123_1000_capsweep_oos20240101_20240630_top50_maxnot25_risk0p75_minnot5/holdout_relaxed8_nokpss_20260123_oos20240101_20240630_top50_z1p00_exit0p06_hold180_cd180_ms0p2_cap1000_maxnot25_risk0p75_minnot5.yaml`
  - `coint4/configs/budget_20260123_1000_capsweep_oos20240101_20240630_top50_maxnot25_risk0p75_minnot5/stress_relaxed8_nokpss_20260123_oos20240101_20240630_top50_z1p00_exit0p06_hold180_cd180_ms0p2_cap1000_maxnot25_risk0p75_minnot5.yaml`
- Статус: `completed` (2 прогона).

#### Результаты (holdout + stress, OOS 2024H1)
| config | hold_sharpe | hold_pnl | hold_dd | hold_trades | hold_pairs | hold_costs | stress_sharpe | stress_pnl | stress_dd | stress_trades | stress_pairs | stress_costs |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| risk0p75/minnot5 | 0.09 | -35.02 | -225.56 | 5009 | 53 | 71.54 | -0.22 | -78.66 | -226.98 | 5009 | 53 | 125.41 |

#### Entry notional (OOS 2024H1)
| config | split | entry_count | cap_hits | below_min | notional_avg | notional_p50 | notional_min | notional_max |
|---|---|---|---|---|---|---|---|---|
| risk0p75/minnot5 | holdout | 5009 | 0 | 0 | 7.61 | 7.50 | 7.48 | 7.91 |
| risk0p75/minnot5 | stress | 5009 | 0 | 0 | 7.51 | 7.50 | 7.36 | 7.67 |

Выводы:
- OOS 2024H1 для `risk0p75/minnot5` около нуля/негативный по Sharpe и PnL.
- Концентрация очень высокая (top10/top20 = 85%/98%), период выглядит нестабильным.

### Queue: budget1000_oos20240101_20240630_top50_maxnot25_risk0p0125 (completed)
- Очередь: `coint4/artifacts/wfa/aggregate/20260123_budget1000_oos20240101_20240630_top50_maxnot25_risk0p0125/run_queue.csv`.
- Цель: дополнительный OOS 2024H1 (2024-01-01 → 2024-06-30, max_steps=3) для `risk0p0125`.
- Конфиги:
  - `coint4/configs/budget_20260123_1000_capsweep_oos20240101_20240630_top50_maxnot25_risk0p0125/holdout_relaxed8_nokpss_20260123_oos20240101_20240630_top50_z1p00_exit0p06_hold180_cd180_ms0p2_cap1000_maxnot25_risk0p0125.yaml`
  - `coint4/configs/budget_20260123_1000_capsweep_oos20240101_20240630_top50_maxnot25_risk0p0125/stress_relaxed8_nokpss_20260123_oos20240101_20240630_top50_z1p00_exit0p06_hold180_cd180_ms0p2_cap1000_maxnot25_risk0p0125.yaml`
- Статус: `completed` (2 прогона).

#### Результаты (holdout + stress, OOS 2024H1)
| config | hold_sharpe | hold_pnl | hold_dd | hold_trades | hold_pairs | hold_costs | stress_sharpe | stress_pnl | stress_dd | stress_trades | stress_pairs | stress_costs |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| risk0p0125 | 0.33 | -63.57 | -375.93 | 5009 | 53 | 120.41 | 0.03 | -134.30 | -378.30 | 5009 | 53 | 209.09 |

#### Entry notional (OOS 2024H1)
| config | split | entry_count | cap_hits | below_min | notional_avg | notional_p50 | notional_min | notional_max |
|---|---|---|---|---|---|---|---|---|
| risk0p0125 | holdout | 5009 | 0 | 0 | 12.81 | 12.50 | 12.46 | 13.63 |
| risk0p0125 | stress | 5009 | 0 | 0 | 12.51 | 12.50 | 12.11 | 12.96 |

Выводы:
- OOS 2024H1 для `risk0p0125` негативный по PnL и почти нулевой по Sharpe; профиль нестабилен.
- Концентрация в 2024H1 такая же высокая (top10/top20 = 85%/98%).

### Queue: budget1000_oos20240101_20240630_top50_maxnot25_risk0p0125_maxpairsgrid (completed)
- Очередь: `coint4/artifacts/wfa/aggregate/20260123_budget1000_oos20240101_20240630_top50_maxnot25_risk0p0125_maxpairsgrid/run_queue.csv`.
- Цель: проверить снижение концентрации через `max_pairs=30/40` для `risk0p0125` на OOS 2024H1.
- Конфиги:
  - `coint4/configs/budget_20260123_1000_capsweep_oos20240101_20240630_top50_maxnot25_risk0p0125_maxpairsgrid/holdout_relaxed8_nokpss_20260123_oos20240101_20240630_top50_z1p00_exit0p06_hold180_cd180_ms0p2_cap1000_maxnot25_risk0p0125_maxpairs30.yaml`
  - `coint4/configs/budget_20260123_1000_capsweep_oos20240101_20240630_top50_maxnot25_risk0p0125_maxpairsgrid/stress_relaxed8_nokpss_20260123_oos20240101_20240630_top50_z1p00_exit0p06_hold180_cd180_ms0p2_cap1000_maxnot25_risk0p0125_maxpairs30.yaml`
  - `coint4/configs/budget_20260123_1000_capsweep_oos20240101_20240630_top50_maxnot25_risk0p0125_maxpairsgrid/holdout_relaxed8_nokpss_20260123_oos20240101_20240630_top50_z1p00_exit0p06_hold180_cd180_ms0p2_cap1000_maxnot25_risk0p0125_maxpairs40.yaml`
  - `coint4/configs/budget_20260123_1000_capsweep_oos20240101_20240630_top50_maxnot25_risk0p0125_maxpairsgrid/stress_relaxed8_nokpss_20260123_oos20240101_20240630_top50_z1p00_exit0p06_hold180_cd180_ms0p2_cap1000_maxnot25_risk0p0125_maxpairs40.yaml`
- Статус: `completed` (4 прогона).

#### Результаты (holdout + stress, OOS 2024H1)
| config | hold_sharpe | hold_pnl | hold_dd | hold_trades | hold_pairs | hold_costs | stress_sharpe | stress_pnl | stress_dd | stress_trades | stress_pairs | stress_costs |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| maxpairs30 | 0.83 | 85.48 | -342.09 | 4354 | 49 | 118.62 | 0.44 | 11.81 | -350.89 | 4354 | 49 | 206.32 |
| maxpairs40 | 0.31 | -59.12 | -367.04 | 4846 | 53 | 116.94 | 0.00 | -129.06 | -368.96 | 4846 | 53 | 203.06 |

#### Entry notional (OOS 2024H1)
| config | split | entry_count | cap_hits | below_min | notional_avg | notional_p50 | notional_min | notional_max |
|---|---|---|---|---|---|---|---|---|
| maxpairs30 | holdout | 4354 | 0 | 0 | 14.51 | 15.22 | 12.50 | 15.79 |
| maxpairs30 | stress | 4354 | 0 | 0 | 14.20 | 14.93 | 12.50 | 15.15 |
| maxpairs40 | holdout | 4846 | 0 | 0 | 12.86 | 12.52 | 12.50 | 13.69 |
| maxpairs40 | stress | 4846 | 0 | 0 | 12.56 | 12.50 | 12.18 | 13.04 |

#### Концентрация (gross PnL, holdout)
| config | top10_share | top20_share | neg_pairs | total_pairs |
|---|---|---|---|---|
| maxpairs30 | 88% | 99% | 24 | 49 |
| maxpairs40 | 85% | 97% | 24 | 53 |

Выводы:
- `max_pairs=30` улучшает PnL на 2024H1, но концентрация остаётся очень высокой.
- `max_pairs=40` не улучшает PnL/Sharpe относительно базового `risk0p0125` и концентрацию не снижает.

### Queue: budget1000_oos_notional20_grid (completed)
- Очередь: `coint4/artifacts/wfa/aggregate/20260123_budget1000_oos_notional20_grid/run_queue.csv`.
- Цель: проверить "notional ~20 на сторону" (cap1000, max_notional=40) при `risk_per_position_pct=0.03/0.04` и `min_notional_per_trade=30/40` в OOS 2024H1 и 2025H1.
- Конфиги (2024H1):
  - `coint4/configs/budget_20260123_1000_capsweep_oos20240101_20240630_top50_maxnot40_notionalgrid/holdout_relaxed8_nokpss_20260123_oos20240101_20240630_top50_z1p00_exit0p06_hold180_cd180_ms0p2_cap1000_maxnot40_risk0p03_minnot30_maxnot40.yaml`
  - `coint4/configs/budget_20260123_1000_capsweep_oos20240101_20240630_top50_maxnot40_notionalgrid/stress_relaxed8_nokpss_20260123_oos20240101_20240630_top50_z1p00_exit0p06_hold180_cd180_ms0p2_cap1000_maxnot40_risk0p03_minnot30_maxnot40.yaml`
  - `coint4/configs/budget_20260123_1000_capsweep_oos20240101_20240630_top50_maxnot40_notionalgrid/holdout_relaxed8_nokpss_20260123_oos20240101_20240630_top50_z1p00_exit0p06_hold180_cd180_ms0p2_cap1000_maxnot40_risk0p04_minnot40_maxnot40.yaml`
  - `coint4/configs/budget_20260123_1000_capsweep_oos20240101_20240630_top50_maxnot40_notionalgrid/stress_relaxed8_nokpss_20260123_oos20240101_20240630_top50_z1p00_exit0p06_hold180_cd180_ms0p2_cap1000_maxnot40_risk0p04_minnot40_maxnot40.yaml`
- Конфиги (2025H1):
  - `coint4/configs/budget_20260123_1000_capsweep_oos20250101_20250630_top50_maxnot40_notionalgrid/holdout_relaxed8_nokpss_20260123_oos20250101_20250630_top50_z1p00_exit0p06_hold180_cd180_ms0p2_cap1000_maxnot40_risk0p03_minnot30_maxnot40.yaml`
  - `coint4/configs/budget_20260123_1000_capsweep_oos20250101_20250630_top50_maxnot40_notionalgrid/stress_relaxed8_nokpss_20260123_oos20250101_20250630_top50_z1p00_exit0p06_hold180_cd180_ms0p2_cap1000_maxnot40_risk0p03_minnot30_maxnot40.yaml`
  - `coint4/configs/budget_20260123_1000_capsweep_oos20250101_20250630_top50_maxnot40_notionalgrid/holdout_relaxed8_nokpss_20260123_oos20250101_20250630_top50_z1p00_exit0p06_hold180_cd180_ms0p2_cap1000_maxnot40_risk0p04_minnot40_maxnot40.yaml`
  - `coint4/configs/budget_20260123_1000_capsweep_oos20250101_20250630_top50_maxnot40_notionalgrid/stress_relaxed8_nokpss_20260123_oos20250101_20250630_top50_z1p00_exit0p06_hold180_cd180_ms0p2_cap1000_maxnot40_risk0p04_minnot40_maxnot40.yaml`
- Статус: `completed` (8 прогонов).

#### Результаты (holdout + stress, OOS 2024H1)
| config | hold_sharpe | hold_pnl | hold_dd | hold_trades | hold_pairs | hold_costs | stress_sharpe | stress_pnl | stress_dd | stress_trades | stress_pairs | stress_costs |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| risk0p03/minnot30 | 2.53 | -8.42 | -902.23 | 2056 | 53 | 115.22 | 2.48 | -74.99 | -907.93 | 2056 | 53 | 204.84 |
| risk0p04/minnot40 | -1.90 | -11.22 | -1202.98 | 2056 | 53 | 153.63 | -1.17 | -99.98 | -1210.57 | 2056 | 53 | 273.12 |

#### Entry notional (OOS 2024H1)
| config | split | entry_count | cap_hits | below_min | notional_avg | notional_p50 | notional_min | notional_max |
|---|---|---|---|---|---|---|---|---|
| risk0p03/minnot30 | holdout | 2056 | 0 | 2056 | 30.00 | 30.00 | 30.00 | 30.00 |
| risk0p03/minnot30 | stress | 2056 | 0 | 2056 | 30.00 | 30.00 | 30.00 | 30.00 |
| risk0p04/minnot40 | holdout | 2056 | 0 | 2056 | 40.00 | 40.00 | 40.00 | 40.00 |
| risk0p04/minnot40 | stress | 2056 | 0 | 2056 | 40.00 | 40.00 | 40.00 | 40.00 |

#### Результаты (holdout + stress, OOS 2025H1)
| config | hold_sharpe | hold_pnl | hold_dd | hold_trades | hold_pairs | hold_costs | stress_sharpe | stress_pnl | stress_dd | stress_trades | stress_pairs | stress_costs |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| risk0p03/minnot30 | 2.24 | 963.35 | -1381.59 | 6694 | 79 | 451.82 | 1.89 | 722.42 | -1477.87 | 6694 | 79 | 803.23 |
| risk0p04/minnot40 | 2.65 | 1311.89 | -1385.04 | 6694 | 79 | 497.07 | 2.32 | 1047.62 | -1481.57 | 6694 | 79 | 883.69 |

#### Entry notional (OOS 2025H1)
| config | split | entry_count | cap_hits | below_min | notional_avg | notional_p50 | notional_min | notional_max |
|---|---|---|---|---|---|---|---|---|
| risk0p03/minnot30 | holdout | 6694 | 4314 | 2380 | 36.44 | 40.00 | 30.00 | 40.00 |
| risk0p03/minnot30 | stress | 6694 | 4314 | 2380 | 36.44 | 40.00 | 30.00 | 40.00 |
| risk0p04/minnot40 | holdout | 6694 | 4314 | 2380 | 40.06 | 40.10 | 40.00 | 40.10 |
| risk0p04/minnot40 | stress | 6694 | 4314 | 2380 | 40.06 | 40.10 | 40.00 | 40.10 |

#### Концентрация (gross PnL, holdout)
| period | config | top10_share | top20_share | neg_pairs | total_pairs |
|---|---|---|---|---|---|
| 2024H1 | risk0p03/minnot30 | 83% | 98% | 16 | 53 |
| 2024H1 | risk0p04/minnot40 | 83% | 98% | 16 | 53 |
| 2025H1 | risk0p03/minnot30 | 65% | 88% | 30 | 79 |
| 2025H1 | risk0p04/minnot40 | 62% | 85% | 30 | 79 |

Выводы:
- OOS 2024H1 остаётся слабым: PnL около нуля/минус при высокой концентрации (83%/98%).
- OOS 2025H1 даёт высокий PnL, но просадка ~-1380 превышает капитал 1000 (рисковый профиль слишком агрессивный).
- `risk0p04/minnot40` в 2025H1 повышает PnL, но увеличивает издержки и не снижает DD.
- Исправлена агрегация entry_notional (исключаем строки с `entry_notional_count=0`), значения для 2024H1 пересчитаны по `trade_statistics.csv`.

### Queue: budget1000_oos20240101_20240630_top50_maxnot40_lowrisk_maxpairsgrid (completed)
- Очередь: `coint4/artifacts/wfa/aggregate/20260123_budget1000_oos20240101_20240630_top50_maxnot40_lowrisk_maxpairsgrid/run_queue.csv`.
- Цель: lower‑risk grid под капитал 1000 (risk 0.015/0.02, min_notional 20/30, max_pairs 20/30) в OOS 2024H1.
- Конфиги: `coint4/configs/budget_20260123_1000_capsweep_oos20240101_20240630_top50_maxnot40_lowrisk_maxpairsgrid/*.yaml`.
- Статус: `completed` (16 прогонов).

#### Результаты (holdout + stress, OOS 2024H1)
| config | hold_sharpe | hold_pnl | hold_dd | hold_trades | hold_pairs | hold_costs | stress_sharpe | stress_pnl | stress_dd | stress_trades | stress_pairs | stress_costs |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| risk0p015/minnot20/maxpairs20 | 0.00 | 0.00 | 0.00 | 0 | 31 | 0.00 | 0.00 | 0.00 | 0.00 | 0 | 31 | 0.00 |
| risk0p015/minnot20/maxpairs30 | 0.00 | 0.00 | 0.00 | 0 | 49 | 0.00 | 0.00 | 0.00 | 0.00 | 0 | 49 | 0.00 |
| risk0p015/minnot30/maxpairs20 | 0.00 | 0.00 | 0.00 | 0 | 31 | 0.00 | 0.00 | 0.00 | 0.00 | 0 | 31 | 0.00 |
| risk0p015/minnot30/maxpairs30 | 0.00 | 0.00 | 0.00 | 0 | 49 | 0.00 | 0.00 | 0.00 | 0.00 | 0 | 49 | 0.00 |
| risk0p02/minnot20/maxpairs20 | 1.55 | 299.61 | -512.24 | 3050 | 31 | 155.77 | 1.22 | 194.43 | -498.63 | 3050 | 31 | 269.95 |
| risk0p02/minnot20/maxpairs30 | 0.96 | 107.27 | -619.25 | 4354 | 49 | 206.04 | 0.58 | -12.61 | -621.54 | 4354 | 49 | 354.12 |
| risk0p02/minnot30/maxpairs20 | 0.00 | 0.00 | 0.00 | 0 | 31 | 0.00 | 0.00 | 0.00 | 0.00 | 0 | 31 | 0.00 |
| risk0p02/minnot30/maxpairs30 | 0.00 | 0.00 | 0.00 | 0 | 49 | 0.00 | 0.00 | 0.00 | 0.00 | 0 | 49 | 0.00 |

#### Entry notional (OOS 2024H1)
| config | split | entry_count | cap_hits | below_min | notional_avg | notional_p50 | notional_min | notional_max |
|---|---|---|---|---|---|---|---|---|
| risk0p015/minnot20/maxpairs20 | holdout | 0 | 0 | 0 | 0.00 | 0.00 | 0.00 | 0.00 |
| risk0p015/minnot20/maxpairs20 | stress | 0 | 0 | 0 | 0.00 | 0.00 | 0.00 | 0.00 |
| risk0p015/minnot20/maxpairs30 | holdout | 0 | 0 | 0 | 0.00 | 0.00 | 0.00 | 0.00 |
| risk0p015/minnot20/maxpairs30 | stress | 0 | 0 | 0 | 0.00 | 0.00 | 0.00 | 0.00 |
| risk0p015/minnot30/maxpairs20 | holdout | 0 | 0 | 0 | 0.00 | 0.00 | 0.00 | 0.00 |
| risk0p015/minnot30/maxpairs20 | stress | 0 | 0 | 0 | 0.00 | 0.00 | 0.00 | 0.00 |
| risk0p015/minnot30/maxpairs30 | holdout | 0 | 0 | 0 | 0.00 | 0.00 | 0.00 | 0.00 |
| risk0p015/minnot30/maxpairs30 | stress | 0 | 0 | 0 | 0.00 | 0.00 | 0.00 | 0.00 |
| risk0p02/minnot20/maxpairs20 | holdout | 3050 | 0 | 1018 | 27.15 | 30.20 | 20.00 | 31.26 |
| risk0p02/minnot20/maxpairs20 | stress | 3050 | 0 | 1018 | 26.47 | 29.62 | 20.00 | 29.81 |
| risk0p02/minnot20/maxpairs30 | holdout | 4354 | 0 | 1439 | 25.20 | 26.95 | 20.00 | 28.59 |
| risk0p02/minnot20/maxpairs30 | stress | 4354 | 0 | 1439 | 24.36 | 26.21 | 20.00 | 26.83 |
| risk0p02/minnot30/maxpairs20 | holdout | 0 | 0 | 0 | 0.00 | 0.00 | 0.00 | 0.00 |
| risk0p02/minnot30/maxpairs20 | stress | 0 | 0 | 0 | 0.00 | 0.00 | 0.00 | 0.00 |
| risk0p02/minnot30/maxpairs30 | holdout | 0 | 0 | 0 | 0.00 | 0.00 | 0.00 | 0.00 |
| risk0p02/minnot30/maxpairs30 | stress | 0 | 0 | 0 | 0.00 | 0.00 | 0.00 | 0.00 |

#### Концентрация (gross PnL, holdout)
| config | top10_share | top20_share | neg_pairs | total_pairs |
|---|---|---|---|---|
| risk0p015/minnot20/maxpairs20 | 0% | 0% | 0 | 31 |
| risk0p015/minnot20/maxpairs30 | 0% | 0% | 0 | 49 |
| risk0p015/minnot30/maxpairs20 | 0% | 0% | 0 | 31 |
| risk0p015/minnot30/maxpairs30 | 0% | 0% | 0 | 49 |
| risk0p02/minnot20/maxpairs20 | 98% | 100% | 17 | 31 |
| risk0p02/minnot20/maxpairs30 | 88% | 99% | 25 | 49 |
| risk0p02/minnot30/maxpairs20 | 0% | 0% | 0 | 31 |
| risk0p02/minnot30/maxpairs30 | 0% | 0% | 0 | 49 |

Выводы:
- `risk0p015` и `minnot30` дали 0 сделок — слишком консервативно при капитале 1000.
- Жизнеспособные варианты только `risk0p02/minnot20`, при этом `maxpairs20` заметно лучше по PnL/Sharpe и издержкам.
- Концентрация остаётся высокой (top10/top20 до 98%/100%).

### Queue: budget1000_oos20250101_20250630_top50_maxnot40_lowrisk_maxpairsgrid (completed, shortlist)
- Очередь: `coint4/artifacts/wfa/aggregate/20260123_budget1000_oos20250101_20250630_top50_maxnot40_lowrisk_maxpairsgrid/run_queue.csv`.
- Цель: проверить shortlisted конфиги `risk0p02/minnot20` в OOS 2025H1.
- Конфиги: `coint4/configs/budget_20260123_1000_capsweep_oos20250101_20250630_top50_maxnot40_lowrisk_maxpairsgrid/*.yaml` (12 skipped, 4 completed).
- Статус: `completed` (4 прогона).

#### Результаты (holdout + stress, OOS 2025H1)
| config | hold_sharpe | hold_pnl | hold_dd | hold_trades | hold_pairs | hold_costs | stress_sharpe | stress_pnl | stress_dd | stress_trades | stress_pairs | stress_costs |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| risk0p02/minnot20/maxpairs20 | 1.95 | 400.64 | -960.70 | 2960 | 37 | 163.04 | 1.57 | 287.67 | -992.42 | 2960 | 37 | 282.96 |
| risk0p02/minnot20/maxpairs30 | 0.92 | 108.87 | -1254.38 | 4403 | 55 | 228.73 | 0.43 | -21.46 | -1282.02 | 4403 | 55 | 393.52 |

#### Entry notional (OOS 2025H1)
| config | split | entry_count | cap_hits | below_min | notional_avg | notional_p50 | notional_min | notional_max |
|---|---|---|---|---|---|---|---|---|
| risk0p02/minnot20/maxpairs20 | holdout | 2960 | 0 | 981 | 29.55 | 29.27 | 20.00 | 39.12 |
| risk0p02/minnot20/maxpairs20 | stress | 2960 | 0 | 981 | 28.85 | 27.73 | 20.00 | 38.54 |
| risk0p02/minnot20/maxpairs30 | holdout | 4403 | 0 | 1450 | 27.99 | 24.53 | 20.00 | 38.96 |
| risk0p02/minnot20/maxpairs30 | stress | 4403 | 0 | 1450 | 27.08 | 22.59 | 20.00 | 38.19 |

#### Концентрация (gross PnL, holdout)
| config | top10_share | top20_share | neg_pairs | total_pairs |
|---|---|---|---|---|
| risk0p02/minnot20/maxpairs20 | 95% | 100% | 21 | 37 |
| risk0p02/minnot20/maxpairs30 | 84% | 96% | 27 | 55 |

Выводы:
- `maxpairs20` стабильно лучше по Sharpe/PnL и контролю издержек; DD около -960 (на грани капитала 1000).
- `maxpairs30` ухудшает Sharpe и даёт отрицательный stress PnL.

### Queue: budget1000_oos20240101_20240630_top50_maxnot40_riskfine_maxpairs20 (completed)
- Очередь: `coint4/artifacts/wfa/aggregate/20260123_budget1000_oos20240101_20240630_top50_maxnot40_riskfine_maxpairs20/run_queue.csv`.
- Цель: уточнить риск ниже 0.02 при min_notional=20 и max_pairs=20 (risk=0.0175/0.018/0.019) в OOS 2024H1.
- Конфиги: `coint4/configs/budget_20260123_1000_capsweep_oos20240101_20240630_top50_maxnot40_riskfine_maxpairs20/*.yaml`.
- Статус: `completed` (6 прогонов).

#### Результаты (holdout + stress, OOS 2024H1)
| config | hold_sharpe | hold_pnl | hold_dd | hold_trades | hold_pairs | hold_costs | stress_sharpe | stress_pnl | stress_dd | stress_trades | stress_pairs | stress_costs |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| risk0p0175/minnot20/maxpairs20 | 0.00 | 0.00 | 0.00 | 0 | 31 | 0.00 | 0.00 | 0.00 | 0.00 | 0 | 31 | 0.00 |
| risk0p018/minnot20/maxpairs20 | 0.00 | 0.00 | 0.00 | 0 | 31 | 0.00 | 0.00 | 0.00 | 0.00 | 0 | 31 | 0.00 |
| risk0p019/minnot20/maxpairs20 | 0.00 | 0.00 | 0.00 | 0 | 31 | 0.00 | 0.00 | 0.00 | 0.00 | 0 | 31 | 0.00 |

#### Entry notional (OOS 2024H1)
| config | split | entry_count | cap_hits | below_min | notional_avg | notional_p50 | notional_min | notional_max |
|---|---|---|---|---|---|---|---|---|
| risk0p0175/minnot20/maxpairs20 | holdout | 0 | 0 | 0 | 0.00 | 0.00 | 0.00 | 0.00 |
| risk0p0175/minnot20/maxpairs20 | stress | 0 | 0 | 0 | 0.00 | 0.00 | 0.00 | 0.00 |
| risk0p018/minnot20/maxpairs20 | holdout | 0 | 0 | 0 | 0.00 | 0.00 | 0.00 | 0.00 |
| risk0p018/minnot20/maxpairs20 | stress | 0 | 0 | 0 | 0.00 | 0.00 | 0.00 | 0.00 |
| risk0p019/minnot20/maxpairs20 | holdout | 0 | 0 | 0 | 0.00 | 0.00 | 0.00 | 0.00 |
| risk0p019/minnot20/maxpairs20 | stress | 0 | 0 | 0 | 0.00 | 0.00 | 0.00 | 0.00 |

#### Концентрация (gross PnL, holdout)
| config | top10_share | top20_share | neg_pairs | total_pairs |
|---|---|---|---|---|
| risk0p0175/minnot20/maxpairs20 | 0% | 0% | 0 | 31 |
| risk0p018/minnot20/maxpairs20 | 0% | 0% | 0 | 31 |
| risk0p019/minnot20/maxpairs20 | 0% | 0% | 0 | 31 |

Выводы:
- При risk < 0.02 и min_notional=20 сделки не открываются (0 трейдов).
- Для 2025H1 эти конфиги помечены как `skipped` и не запускались.

### Queue: budget1000_oos20250101_20250630_top50_maxnot40_riskfine_maxpairs20 (skipped)
- Очередь: `coint4/artifacts/wfa/aggregate/20260123_budget1000_oos20250101_20250630_top50_maxnot40_riskfine_maxpairs20/run_queue.csv`.
- Причина: 0 сделок на OOS 2024H1, запуск нецелесообразен.
