# Обзор результатов для капитала $1000 (2026-01-31)

Цель: быстро показать, что реально работает на $1000, где искать артефакты и что делать дальше.

## Ключевые выводы
- Extended OOS `2023-05-01 → 2024-04-30` в режиме `base` провалился по stress-издержкам (cost_ratio ~1.4–1.5).
- Turnover-grid на extended OOS (top10/top15, hold/cd 240) не спасает: мало пар, stress cost_ratio >> 0.5.
- Режим `tlow` (`z=1.20/exit=0.08/hold=240/cd=240/ms=0.25`) стабилен на OOS 2023H2/2024H1 (cost_ratio ~0.04–0.14, DD ~7–19%), **но** на extended OOS 2023-05 → 2024-04 всё ещё слаб: пары 36, stress cost_ratio 0.60, DD ~-32%.
- Refine (z=1.25/1.30, ms=0.30/0.35, hold/cd=300) снизил DD до ~-17…-18%, но stress cost_ratio всё ещё 0.53–0.56 и пар 36.
- Refine2 (z=1.35/1.40, ms=0.35/0.40, hold/cd=360) ухудшил PnL/Sharpe; stress cost_ratio вырос до 1.85–1.86 при тех же 36 парах.
- Tradeability+basecap3 (corr 0.35/0.40, pv 0.25/0.20, пары basecap3=102) дал отрицательные Sharpe/PnL и всего 26–27 пар; ветка закрыта.
- Результаты `20260122_budget1000_top50_top30` считаем историческими/некорректными (до фикса масштабирования капитала) — не использовать для решения.

## Где смотреть (навигатор)
- Итоговый индекс метрик: `coint4/artifacts/wfa/aggregate/rollup/run_index.csv`.
- Очереди прогонов: `coint4/artifacts/wfa/aggregate/<run_group>/run_queue.csv`.
- Артефакты прогона: `coint4/artifacts/wfa/runs/<run_id>/`:
  - `strategy_metrics.csv` (итоговые метрики),
  - `trade_statistics.csv` (концентрация PnL),
  - `filter_reasons_*.csv` (фильтрация пар),
  - `run.log` (warnings/аномалии).
- Обновление индекса (из `coint4/`):
  - `PYTHONPATH=src ./.venv/bin/python scripts/optimization/build_run_index.py --output-dir artifacts/wfa/aggregate/rollup`

## Extended OOS (2023-05-01 → 2024-04-30, base)
| run_id | sharpe | pnl | max_dd | costs | cost_ratio | trades | pairs |
|---|---:|---:|---:|---:|---:|---:|---:|
| holdout_relaxed8_nokpss_20260130_oos20230501_20240430_pv0p2_hurst0p8_hl0p02_60_corr0p4_ssd50000_kpss1p0_w2m1_t90_fixed_u250_top20_z1p00_exit0p06_hold180_cd180_ms0p2 | 2.44 | 452.61 | -206.85 | 227.30 | 0.50 | 4770 | 55 |
| holdout_relaxed8_nokpss_20260130_oos20230501_20240430_pv0p2_hurst0p8_hl0p02_60_corr0p4_ssd50000_kpss1p0_w2m1_t90_fixed_u250_top30_z1p00_exit0p06_hold180_cd180_ms0p2 | 2.95 | 620.02 | -288.08 | 310.76 | 0.50 | 6384 | 68 |
| stress_relaxed8_nokpss_20260130_oos20230501_20240430_pv0p2_hurst0p8_hl0p02_60_corr0p4_ssd50000_kpss1p0_w2m1_t90_fixed_u250_top20_z1p00_exit0p06_hold180_cd180_ms0p2 | 1.50 | 276.18 | -252.56 | 404.08 | 1.46 | 4770 | 55 |
| stress_relaxed8_nokpss_20260130_oos20230501_20240430_pv0p2_hurst0p8_hl0p02_60_corr0p4_ssd50000_kpss1p0_w2m1_t90_fixed_u250_top30_z1p00_exit0p06_hold180_cd180_ms0p2 | 1.87 | 392.57 | -314.14 | 552.46 | 1.41 | 6384 | 68 |

Итог: stress cost_ratio > 1.0, значит edge «съеден» издержками.

## Turnover-grid (extended OOS, top10/top15, hold/cd 240)
Результаты `20260131_realcost_oos20230501_20240430_turnover`:
- top10: отрицательный Sharpe и PnL, пар 32.
- top15: holdout Sharpe ~1.2–1.3, но stress cost_ratio 5.7–8.9 и пар 47 (<50).

## Extended OOS (2023-05-01 → 2024-04-30, tlow, cap1000)
| run_id | sharpe | pnl | max_dd | costs | cost_ratio | trades | pairs |
|---|---:|---:|---:|---:|---:|---:|---:|
| holdout_relaxed8_nokpss_20260131_oos20230501_20240430_top250_z1p20_exit0p08_hold240_cd240_ms0p25_cap1000_maxnot20_risk0p0175_minnot15_maxpairs12 | 1.70 | 307.81 | -322.29 | 79.55 | 0.26 | 2348 | 36 |
| stress_relaxed8_nokpss_20260131_oos20230501_20240430_top250_z1p20_exit0p08_hold240_cd240_ms0p25_cap1000_maxnot20_risk0p0175_minnot15_maxpairs12 | 1.36 | 232.26 | -327.39 | 138.35 | 0.60 | 2348 | 36 |
| holdout_relaxed8_nokpss_20260131_oos20230501_20240430_top250_z1p20_exit0p08_hold240_cd240_ms0p25_cap1000_maxnot20_risk0p015_minnot15_maxpairs12 | -0.54 | -80.35 | -274.22 | 28.52 | — | 961 | 36 |
| stress_relaxed8_nokpss_20260131_oos20230501_20240430_top250_z1p20_exit0p08_hold240_cd240_ms0p25_cap1000_maxnot20_risk0p015_minnot15_maxpairs12 | -0.72 | -100.56 | -276.21 | 50.45 | — | 961 | 36 |

Итог: даже в `tlow` extended OOS не проходит гейт (pairs < 50, stress cost_ratio > 0.5, DD ~ -32%).

## Extended OOS (2023-05-01 → 2024-04-30, tlow refine)
| config | kind | sharpe | pnl | max_dd | costs | cost_ratio | trades | pairs |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| z1.25/ms0.30 | holdout | 2.16 | 298.1 | -172.9 | 73.2 | 0.25 | 2071 | 36 |
| z1.25/ms0.30 | stress | 1.76 | 236.2 | -174.4 | 129.4 | 0.55 | 2071 | 36 |
| z1.25/ms0.35 | holdout | 2.17 | 299.3 | -172.9 | 73.0 | 0.24 | 2067 | 36 |
| z1.25/ms0.35 | stress | 1.77 | 237.6 | -174.4 | 129.1 | 0.54 | 2067 | 36 |
| z1.30/ms0.30 | holdout | 2.14 | 294.9 | -168.9 | 70.8 | 0.24 | 2008 | 36 |
| z1.30/ms0.30 | stress | 1.75 | 235.5 | -177.2 | 125.1 | 0.53 | 2008 | 36 |
| z1.30/ms0.35 | holdout | 2.04 | 280.1 | -174.1 | 70.6 | 0.25 | 2004 | 36 |
| z1.30/ms0.35 | stress | 1.65 | 220.8 | -182.9 | 124.7 | 0.56 | 2004 | 36 |

Итог: DD улучшен, но stress cost_ratio всё ещё выше 0.5, пары не растут. Концентрация высокая (top10/top20 > 2 по стрессу).

## Extended OOS (2023-05-01 → 2024-04-30, tlow refine2)
| config | kind | sharpe | pnl | max_dd | costs | cost_ratio | trades | pairs |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| z1.35/ms0.35 | holdout | 0.17 | 1.8 | -184.4 | 60.5 | 33.68 | 1805 | 36 |
| z1.35/ms0.35 | stress | -0.18 | -43.5 | -195.4 | 105.6 | — | 1805 | 36 |
| z1.35/ms0.40 | holdout | 0.11 | -6.2 | -181.9 | 60.3 | — | 1801 | 36 |
| z1.35/ms0.40 | stress | -0.25 | -51.0 | -192.8 | 105.3 | — | 1801 | 36 |
| z1.40/ms0.35 | holdout | 0.81 | 102.0 | -175.4 | 58.1 | 0.57 | 1739 | 36 |
| z1.40/ms0.35 | stress | 0.53 | 54.8 | -201.3 | 101.6 | 1.85 | 1739 | 36 |
| z1.40/ms0.40 | holdout | 0.81 | 101.7 | -173.5 | 58.0 | 0.57 | 1733 | 36 |
| z1.40/ms0.40 | stress | 0.53 | 54.6 | -194.6 | 101.4 | 1.86 | 1733 | 36 |

Итог: ужесточение z/ms ухудшило PnL/Sharpe и увеличило stress cost_ratio; гейт по парам не пройден.

## Extended OOS (2023-05-01 → 2024-04-30, tlow tradeability + basecap3)
| config | kind | sharpe | pnl | max_dd | costs | cost_ratio | trades | pairs |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| tradeM/corr0.35/pv0.25 | holdout | -2.05 | -249.5 | -330.3 | 32.8 | — | 982 | 27 |
| tradeM/corr0.35/pv0.25 | stress | -2.27 | -271.3 | -340.0 | 57.9 | — | 982 | 27 |
| tradeM/corr0.40/pv0.20 | holdout | -1.89 | -215.9 | -296.0 | 29.1 | — | 879 | 26 |
| tradeM/corr0.40/pv0.20 | stress | -2.09 | -235.2 | -299.5 | 51.5 | — | 879 | 26 |

Примечание: tradeS (более строгие tradeability‑пороги) дали идентичные метрики, поэтому не дублируются.

Итог: все варианты в минус, пар 26–27 → basecap3 + tradeability не работают на extended OOS.

## Лучшие $1000 кандидаты (tlow) — OOS 2024H1
| run_id | sharpe | pnl | max_dd | costs | cost_ratio | trades | pairs |
|---|---:|---:|---:|---:|---:|---:|---:|
| holdout_relaxed8_nokpss_20260124_oos20240101_20240630_top50_z1p20_exit0p08_hold240_cd240_ms0p25_cap1000_maxnot20_risk0p0175_minnot15_maxpairs12 | 6.04 | 799.69 | -177.30 | 55.89 | 0.07 | 1542 | 19 |
| holdout_relaxed8_nokpss_20260124_oos20240101_20240630_top50_z1p20_exit0p08_hold240_cd240_ms0p25_cap1000_maxnot20_risk0p015_minnot15_maxpairs12 | 5.74 | 742.78 | -177.30 | 53.47 | 0.07 | 1542 | 19 |
| stress_relaxed8_nokpss_20260124_oos20240101_20240630_top50_z1p20_exit0p08_hold240_cd240_ms0p25_cap1000_maxnot20_risk0p0175_minnot15_maxpairs12 | 5.73 | 756.22 | -189.06 | 99.36 | 0.13 | 1542 | 19 |
| stress_relaxed8_nokpss_20260124_oos20240101_20240630_top50_z1p20_exit0p08_hold240_cd240_ms0p25_cap1000_maxnot20_risk0p015_minnot15_maxpairs12 | 5.44 | 700.75 | -188.72 | 94.98 | 0.14 | 1542 | 19 |

## Подтверждение на OOS 2023H2 (tlow)
| run_id | sharpe | pnl | max_dd | costs | cost_ratio | trades | pairs |
|---|---:|---:|---:|---:|---:|---:|---:|
| holdout_relaxed8_nokpss_20260124_oos20230701_20231231_top50_z1p20_exit0p08_hold240_cd240_ms0p25_cap1000_maxnot30_risk0p015_minnot15_maxpairs12 | 4.25 | 1136.15 | -70.69 | 42.91 | 0.04 | 1348 | 24 |
| holdout_relaxed8_nokpss_20260124_oos20230701_20231231_top50_z1p20_exit0p08_hold240_cd240_ms0p25_cap1000_maxnot30_risk0p0175_minnot15_maxpairs12 | 4.20 | 1370.99 | -82.48 | 51.07 | 0.04 | 1348 | 24 |
| stress_relaxed8_nokpss_20260124_oos20230701_20231231_top50_z1p20_exit0p08_hold240_cd240_ms0p25_cap1000_maxnot30_risk0p015_minnot15_maxpairs12 | 4.13 | 1086.60 | -74.70 | 75.55 | 0.07 | 1348 | 24 |
| stress_relaxed8_nokpss_20260124_oos20230701_20231231_top50_z1p20_exit0p08_hold240_cd240_ms0p25_cap1000_maxnot30_risk0p0175_minnot15_maxpairs12 | 4.08 | 1308.83 | -87.15 | 89.78 | 0.07 | 1348 | 24 |

## План действий (выполнено)
1) Очередь `tlow` extended OOS для $1000 выполнена.
2) Refine‑очередь (z=1.25/1.30, ms=0.30/0.35, hold/cd=300) выполнена.
3) Refine2‑очередь (z=1.35/1.40, ms=0.35/0.40, hold/cd=360) выполнена.
4) Tradeability+basecap3‑очередь выполнена (corr/pvalue grid, пары basecap3).
5) Rollup индекс обновлён (`run_index.csv`), журнал прогонов обновлён (`docs/optimization_runs_20260131.md`).

## Примечания
- В `20260122_budget1000_top50_top30` метрики совпадали с $10k (до фикса масштабирования). Эти результаты считаем нерелевантными.
- Для $1000 ключевая цель — удержать `cost_ratio <= 0.5` и `max_dd <= -250` (25% капитала), иначе прирост PnL не компенсирует риск.
