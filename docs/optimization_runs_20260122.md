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
