# Журнал прогонов оптимизации (2026-02-13)

Цель: продолжить Max-Sharpe оптимизацию для капитала `$1000` на extended OOS `2023-05-01 -> 2024-04-30` вокруг текущего лидера `ts1p5` (holdout/stress Sharpe `4.424/4.119`).

## Extra sweep: signal sprint19 (hold/cooldown sweep under `ts1p5`, 10 прогонов)
- Очередь: `coint4/artifacts/wfa/aggregate/20260213_budget1000_sharpe_signal_sprint19/run_queue.csv`
- Конфиги: `coint4/configs/budget_20260213_1000_sharpe_signal_sprint19/*.yaml`
- Размер: 10 прогонов (`5` вариантов × `holdout/stress`)
- Статус: `10/10 completed`
- Валидация: `Sharpe consistency OK (10 run(s))`

### Матрица параметров (hc*)
Фиксируем лидера `ts1p5` и меняем только:
- `backtest.min_position_hold_minutes`
- `backtest.anti_churn_cooldown_minutes`

| variant | hold_minutes | cooldown_minutes |
|---|---:|---:|
| hc60 | 60 | 60 |
| hc180 | 180 | 180 |
| hc300 | 300 | 300 |
| hc600 | 600 | 600 |
| hc900 | 900 | 900 |

### Результаты (10 прогонов)
| variant | kind | sharpe | pnl | max_dd | cost_ratio | trades | pairs |
|---|---|---:|---:|---:|---:|---:|---:|
| hc60 | holdout | -0.168 | -1216.47 | -1914.57 | — | 3313 | 58 |
| hc60 | stress | -0.434 | -1258.56 | -1880.71 | — | 3313 | 58 |
| hc180 | holdout | 3.466 | 1763.72 | -549.91 | 0.14 | 5677 | 58 |
| hc180 | stress | 3.143 | 1497.59 | -550.52 | 0.28 | 5677 | 58 |
| hc300 | holdout | 4.424 | 2230.75 | -466.47 | 0.09 | 4646 | 58 |
| hc300 | stress | 4.119 | 1978.76 | -457.92 | 0.17 | 4646 | 58 |
| hc600 | holdout | 0.872 | 155.88 | -462.23 | 0.66 | 3301 | 58 |
| hc600 | stress | 0.593 | 76.99 | -459.14 | 2.32 | 3301 | 58 |
| hc900 | holdout | 1.294 | 248.25 | -418.90 | 0.35 | 2575 | 58 |
| hc900 | stress | 1.039 | 182.56 | -430.10 | 0.83 | 2575 | 58 |

### Итог по sprint19
- Лучший robust по `min(Sharpe_holdout, Sharpe_stress)` остаётся baseline `hc300` (то есть `hold300/cd300`), совпадает с текущим лидером `ts1p5`.
- `hc60` ломает стратегию (отрицательный Sharpe и очень глубокий DD), а `hc600/900` режут turnover, но Sharpe и PnL падают, cost_ratio ухудшается.

## Extra sweep: signal sprint20 (min_spread_move_sigma sweep under `ts1p5`, 10 прогонов)
- Очередь: `coint4/artifacts/wfa/aggregate/20260213_budget1000_sharpe_signal_sprint20/run_queue.csv`
- Конфиги: `coint4/configs/budget_20260213_1000_sharpe_signal_sprint20/*.yaml`
- Размер: 10 прогонов (`5` вариантов × `holdout/stress`)
- Статус: `10/10 completed`
- Валидация: `Sharpe consistency OK (10 run(s))`

### Матрица параметров (ms*)
Фиксируем лидера `ts1p5` и меняем только `backtest.min_spread_move_sigma`.

Примечание: `min_spread_move_sigma` должен быть строго `> 0`, поэтому вместо `0.0` используем `0.1`.

| variant | min_spread_move_sigma |
|---|---:|
| ms0p1 | 0.1 |
| ms0p2 | 0.2 |
| ms0p4 | 0.4 |
| ms0p6 | 0.6 |
| ms0p8 | 0.8 |

### Результаты (10 прогонов)
| variant | kind | sharpe | pnl | max_dd | cost_ratio | trades | pairs |
|---|---|---:|---:|---:|---:|---:|---:|
| ms0p1 | holdout | 4.572 | 2463.52 | -536.99 | 0.08 | 4659 | 58 |
| ms0p1 | stress | 4.277 | 2196.52 | -525.86 | 0.16 | 4659 | 58 |
| ms0p2 | holdout | 4.424 | 2230.75 | -466.47 | 0.09 | 4646 | 58 |
| ms0p2 | stress | 4.119 | 1978.76 | -457.92 | 0.17 | 4646 | 58 |
| ms0p4 | holdout | 4.190 | 2064.46 | -483.21 | 0.09 | 4629 | 58 |
| ms0p4 | stress | 3.888 | 1823.16 | -488.91 | 0.18 | 4629 | 58 |
| ms0p6 | holdout | 3.899 | 1605.32 | -449.38 | 0.11 | 4573 | 58 |
| ms0p6 | stress | 3.562 | 1394.77 | -448.81 | 0.21 | 4573 | 58 |
| ms0p8 | holdout | 4.184 | 1956.38 | -486.64 | 0.10 | 4531 | 58 |
| ms0p8 | stress | 3.875 | 1726.73 | -494.39 | 0.19 | 4531 | 58 |

### Итог по sprint20
- Новый лучший robust по `min(Sharpe_holdout, Sharpe_stress)` — `ms0p1` (min_spread_move_sigma=0.1): Sharpe `4.572/4.277` (лучше baseline `ms0p2` = `4.424/4.119`).

## Extra sweep: signal sprint21 (corr/pvalue sweep under `ms0p1`, 10 прогонов)
- Очередь: `coint4/artifacts/wfa/aggregate/20260213_budget1000_sharpe_signal_sprint21/run_queue.csv`
- Конфиги: `coint4/configs/budget_20260213_1000_sharpe_signal_sprint21/*.yaml`
- Размер: 10 прогонов (`5` вариантов × `holdout/stress`)
- Статус: `10/10 completed`
- Валидация: `Sharpe consistency OK (10 run(s))`

### Матрица параметров (c*_pv*)
Фиксируем лидера `ms0p1` и меняем только:
- `backtest.min_correlation_threshold`
- `pair_selection.min_correlation`
- `pair_selection.coint_pvalue_threshold`

| variant | min_correlation | coint_pvalue_threshold |
|---|---:|---:|
| c0p28_pv0p45 | 0.28 | 0.45 |
| c0p30_pv0p40 | 0.30 | 0.40 |
| c0p34_pv0p35 | 0.34 | 0.35 |
| c0p40_pv0p30 | 0.40 | 0.30 |
| c0p50_pv0p25 | 0.50 | 0.25 |

### Результаты (10 прогонов)
| variant | kind | sharpe | pnl | max_dd | cost_ratio | trades | pairs |
|---|---|---:|---:|---:|---:|---:|---:|
| c0p28_pv0p45 | holdout | 3.746 | 1396.33 | -427.66 | 0.12 | 4677 | 55 |
| c0p28_pv0p45 | stress | 3.383 | 1195.10 | -439.00 | 0.23 | 4677 | 55 |
| c0p30_pv0p40 | holdout | 3.749 | 1632.64 | -427.66 | 0.11 | 4655 | 58 |
| c0p30_pv0p40 | stress | 3.425 | 1414.86 | -439.00 | 0.21 | 4655 | 58 |
| c0p34_pv0p35 | holdout | 4.572 | 2463.52 | -536.99 | 0.08 | 4659 | 58 |
| c0p34_pv0p35 | stress | 4.277 | 2196.52 | -525.86 | 0.16 | 4659 | 58 |
| c0p40_pv0p30 | holdout | 3.927 | 1907.67 | -506.71 | 0.11 | 4735 | 59 |
| c0p40_pv0p30 | stress | 3.625 | 1673.73 | -494.84 | 0.20 | 4735 | 59 |
| c0p50_pv0p25 | holdout | 3.777 | 1787.70 | -505.88 | 0.10 | 4773 | 63 |
| c0p50_pv0p25 | stress | 3.468 | 1557.50 | -502.06 | 0.20 | 4773 | 63 |

### Итог по sprint21
- Ни один вариант `corr/pvalue` не улучшил `ms0p1` baseline `c0p34_pv0p35`; loosen/tighten снижает robust Sharpe.

## Extra sweep: signal sprint22 (time_stop_multiplier sweep under `ms0p1`, 10 прогонов)
- Очередь: `coint4/artifacts/wfa/aggregate/20260213_budget1000_sharpe_signal_sprint22/run_queue.csv`
- Конфиги: `coint4/configs/budget_20260213_1000_sharpe_signal_sprint22/*.yaml`
- Размер: 10 прогонов (`5` вариантов × `holdout/stress`)
- Статус: `10/10 completed`
- Валидация: `Sharpe consistency OK (10 run(s))`

### Матрица параметров (ts*)
Фиксируем лидера `ms0p1` и меняем только `backtest.time_stop_multiplier`.

| variant | time_stop_multiplier |
|---|---:|
| tsOff | — |
| ts1p0 | 1.0 |
| ts1p5 | 1.5 |
| ts2p0 | 2.0 |
| ts3p0 | 3.0 |

### Результаты (10 прогонов)
| variant | kind | sharpe | pnl | max_dd | cost_ratio | trades | pairs |
|---|---|---:|---:|---:|---:|---:|---:|
| tsOff | holdout | 4.502 | 2404.55 | -529.30 | 0.08 | 4523 | 58 |
| tsOff | stress | 4.214 | 2147.83 | -518.24 | 0.16 | 4523 | 58 |
| ts1p0 | holdout | 4.565 | 2458.15 | -547.24 | 0.08 | 4687 | 58 |
| ts1p0 | stress | 4.269 | 2190.74 | -535.45 | 0.16 | 4687 | 58 |
| ts1p5 | holdout | 4.572 | 2463.52 | -536.99 | 0.08 | 4659 | 58 |
| ts1p5 | stress | 4.277 | 2196.52 | -525.86 | 0.16 | 4659 | 58 |
| ts2p0 | holdout | 4.529 | 2419.70 | -543.60 | 0.08 | 4624 | 58 |
| ts2p0 | stress | 4.236 | 2156.84 | -532.19 | 0.16 | 4624 | 58 |
| ts3p0 | holdout | 4.503 | 2403.74 | -537.30 | 0.08 | 4585 | 58 |
| ts3p0 | stress | 4.214 | 2144.82 | -525.66 | 0.16 | 4585 | 58 |

### Итог по sprint22
- `ts1p5` остаётся локальным максимумом и под `ms0p1`; новый лидер не найден.
