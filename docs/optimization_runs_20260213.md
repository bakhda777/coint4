# Журнал прогонов оптимизации (2026-02-13)

Цель: продолжить Max-Sharpe оптимизацию для капитала `$1000` на extended OOS `2023-05-01 -> 2024-04-30` вокруг текущего лидера `ms0p1` (holdout/stress Sharpe `4.572/4.277`).

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

## Extra sweep: signal sprint23 (pair_stop_loss_zscore sweep under `ms0p1+ts1p5`, 10 прогонов)
- Очередь: `coint4/artifacts/wfa/aggregate/20260213_budget1000_sharpe_signal_sprint23/run_queue.csv`
- Конфиги: `coint4/configs/budget_20260213_1000_sharpe_signal_sprint23/*.yaml`
- Размер: 10 прогонов (`5` вариантов × `holdout/stress`)
- Статус: `10/10 completed`
- Валидация: `Sharpe consistency OK (10 run(s))`

### Матрица параметров (slz*)
Фиксируем лидера `ms0p1+ts1p5` и меняем только `backtest.pair_stop_loss_zscore`.

Примечание: `pair_stop_loss_zscore` должен быть строго `> 0`, отключить через `0.0` нельзя.

| variant | pair_stop_loss_zscore |
|---|---:|
| slz2p0 | 2.0 |
| slz2p5 | 2.5 |
| slz3p0 | 3.0 |
| slz3p5 | 3.5 |
| slz4p0 | 4.0 |

### Результаты (10 прогонов)
| variant | kind | sharpe | pnl | max_dd | cost_ratio | trades | pairs |
|---|---|---:|---:|---:|---:|---:|---:|
| slz2p0 | holdout | 2.032 | 375.70 | -312.73 | 0.50 | 5844 | 58 |
| slz2p0 | stress | 1.340 | 220.01 | -323.71 | 1.44 | 5844 | 58 |
| slz2p5 | holdout | 2.948 | 916.25 | -384.97 | 0.21 | 5149 | 58 |
| slz2p5 | stress | 2.531 | 736.51 | -388.44 | 0.44 | 5149 | 58 |
| slz3p0 | holdout | 4.572 | 2463.52 | -536.99 | 0.08 | 4659 | 58 |
| slz3p0 | stress | 4.277 | 2196.52 | -525.86 | 0.16 | 4659 | 58 |
| slz3p5 | holdout | 2.547 | 1001.95 | -631.18 | 0.15 | 4352 | 58 |
| slz3p5 | stress | 2.277 | 840.24 | -616.24 | 0.30 | 4352 | 58 |
| slz4p0 | holdout | 1.669 | 564.83 | -698.37 | 0.21 | 4126 | 58 |
| slz4p0 | stress | 1.431 | 435.47 | -730.71 | 0.47 | 4126 | 58 |

### Итог по sprint23
- Лидер остаётся `slz3p0` (stop_loss_z=3.0): Sharpe `4.572/4.277`.
- Более агрессивный stop (`2.0–2.5`) резко увеличивает churn и издержки (stress `cost_ratio` до `1.44`) и ломает Sharpe.
- Более мягкий stop (`3.5–4.0`) ухудшает Sharpe и раздувает DD (holdout max_dd до `-698…-730`).

## Extra sweep: signal sprint24 (protections toggles sweep under `ms0p1+ts1p5+slz3p0`, 10 прогонов)
- Очередь: `coint4/artifacts/wfa/aggregate/20260213_budget1000_sharpe_signal_sprint24/run_queue.csv`
- Конфиги: `coint4/configs/budget_20260213_1000_sharpe_signal_sprint24/*.yaml`
- Размер: 10 прогонов (`5` вариантов × `holdout/stress`)
- Статус: `10/10 completed`
- Валидация: `Sharpe consistency OK (10 run(s))`

### Матрица параметров
Фиксируем текущего лидера `ms0p1+ts1p5+slz3p0` и проверяем влияние основных защитных механизмов:
- `backtest.adaptive_thresholds`
- `backtest.market_regime_detection`
- `backtest.structural_break_protection`

| variant | adaptive_thresholds | market_regime_detection | structural_break_protection |
|---|---:|---:|---:|
| base | true | true | true |
| noadapt | false | true | true |
| noregime | true | false | true |
| nostruct | true | true | false |
| alloff | false | false | false |

### Результаты (10 прогонов)
| variant | kind | sharpe | pnl | max_dd | cost_ratio | trades | pairs |
|---|---|---:|---:|---:|---:|---:|---:|
| base | holdout | 4.572 | 2463.52 | -536.99 | 0.08 | 4659 | 58 |
| base | stress | 4.277 | 2196.52 | -525.86 | 0.16 | 4659 | 58 |
| noadapt | holdout | 4.499 | 2393.68 | -529.88 | 0.08 | 4665 | 58 |
| noadapt | stress | 4.203 | 2130.60 | -518.85 | 0.16 | 4665 | 58 |
| noregime | holdout | 1.179 | 303.39 | -831.60 | 0.40 | 5446 | 58 |
| noregime | stress | 0.905 | 147.28 | -825.06 | 1.40 | 5446 | 58 |
| nostruct | holdout | 3.541 | 1515.46 | -459.81 | 0.11 | 5267 | 58 |
| nostruct | stress | 3.188 | 1280.46 | -440.27 | 0.22 | 5267 | 58 |
| alloff | holdout | 1.459 | 601.85 | -1166.22 | 0.27 | 5597 | 58 |
| alloff | stress | 1.292 | 416.67 | -1180.86 | 0.66 | 5597 | 58 |

### Итог по sprint24
- Лидер не изменился: `base` (все защиты включены) остаётся лучшим по robust-метрике.
- `market_regime_detection=false` резко ухудшает Sharpe и раздувает DD; эту защиту выключать нельзя.
- `structural_break_protection=false` заметно ухудшает Sharpe; защита полезна, но кандидат на параметризацию (чтобы тюнить интенсивность, а не только on/off).

## Extra sweep: signal sprint25 (rolling_window sweep under `ms0p1+ts1p5+slz3p0`, 10 прогонов)
- Очередь: `coint4/artifacts/wfa/aggregate/20260213_budget1000_sharpe_signal_sprint25/run_queue.csv`
- Конфиги: `coint4/configs/budget_20260213_1000_sharpe_signal_sprint25/*.yaml`
- Размер: 10 прогонов (`5` вариантов × `holdout/stress`)
- Статус: `10/10 completed`
- Валидация: `Sharpe consistency OK (10 run(s))`

### Матрица параметров (rw*)
Фиксируем текущего лидера `ms0p1+ts1p5+slz3p0` и меняем только `backtest.rolling_window`.

| variant | rolling_window |
|---|---:|
| rw48 | 48 |
| rw96 | 96 |
| rw144 | 144 |
| rw192 | 192 |
| rw288 | 288 |

### Результаты (10 прогонов)
| variant | kind | sharpe | pnl | max_dd | cost_ratio | trades | pairs |
|---|---|---:|---:|---:|---:|---:|---:|
| rw48 | holdout | 2.341 | 1190.89 | -634.14 | 0.17 | 5846 | 58 |
| rw48 | stress | 2.071 | 958.48 | -598.20 | 0.36 | 5846 | 58 |
| rw96 | holdout | 4.572 | 2463.52 | -536.99 | 0.08 | 4659 | 58 |
| rw96 | stress | 4.277 | 2196.52 | -525.86 | 0.16 | 4659 | 58 |
| rw144 | holdout | 0.603 | 50.90 | -453.29 | 2.20 | 3971 | 58 |
| rw144 | stress | 0.398 | -38.24 | -437.92 | -5.04 | 3971 | 58 |
| rw192 | holdout | 1.183 | 309.39 | -582.47 | 0.32 | 3532 | 58 |
| rw192 | stress | 0.970 | 212.88 | -581.62 | 0.79 | 3532 | 58 |
| rw288 | holdout | 0.922 | 161.41 | -599.74 | 0.47 | 2887 | 58 |
| rw288 | stress | 0.646 | 87.23 | -608.51 | 1.51 | 2887 | 58 |

### Итог по sprint25
- `rolling_window=96` остаётся явным максимумом Sharpe; остальные окна резко ухудшают Sharpe и/или уводят PnL в ноль/минус (особенно `rw144`).

## Extra sweep: signal sprint26 (z-entry sweep under `ms0p1+ts1p5+slz3p0`, 10 прогонов)
- Очередь: `coint4/artifacts/wfa/aggregate/20260213_budget1000_sharpe_signal_sprint26/run_queue.csv`
- Конфиги: `coint4/configs/budget_20260213_1000_sharpe_signal_sprint26/*.yaml`
- Размер: 10 прогонов (`5` вариантов × `holdout/stress`)
- Статус: `10/10 completed`
- Валидация: `Sharpe consistency OK (10 run(s))`

### Матрица параметров (z*)
Фиксируем текущего лидера `ms0p1+ts1p5+slz3p0` и меняем только `backtest.zscore_entry_threshold` (синхронно с `backtest.zscore_threshold` для консистентности).

| variant | zscore_entry_threshold |
|---|---:|
| z0p90 | 0.90 |
| z1p00 | 1.00 |
| z1p15 | 1.15 |
| z1p30 | 1.30 |
| z1p45 | 1.45 |

### Результаты (10 прогонов)
| variant | kind | sharpe | pnl | max_dd | cost_ratio | trades | pairs |
|---|---|---:|---:|---:|---:|---:|---:|
| z0p90 | holdout | 3.671 | 2021.14 | -525.13 | 0.09 | 5379 | 58 |
| z0p90 | stress | 3.360 | 1734.41 | -522.71 | 0.18 | 5379 | 58 |
| z1p00 | holdout | 3.505 | 1995.60 | -586.21 | 0.09 | 5103 | 58 |
| z1p00 | stress | 3.224 | 1724.38 | -559.22 | 0.18 | 5103 | 58 |
| z1p15 | holdout | 4.572 | 2463.52 | -536.99 | 0.08 | 4659 | 58 |
| z1p15 | stress | 4.277 | 2196.52 | -525.86 | 0.16 | 4659 | 58 |
| z1p30 | holdout | 3.965 | 1102.78 | -365.68 | 0.13 | 4147 | 58 |
| z1p30 | stress | 3.554 | 944.21 | -366.28 | 0.25 | 4147 | 58 |
| z1p45 | holdout | 3.552 | 884.72 | -273.98 | 0.14 | 3647 | 58 |
| z1p45 | stress | 3.163 | 755.30 | -280.35 | 0.27 | 3647 | 58 |

### Итог по sprint26
- Лидер по robust Sharpe остаётся `z1p15` (текущий baseline); как занижение `z` (churn), так и завышение (резкое падение PnL/Sharpe) ухудшают метрики.

## Extra sweep: signal sprint27 (structural-break intensity sweep under `ms0p1+ts1p5+slz3p0`, 10 прогонов)
- Очередь: `coint4/artifacts/wfa/aggregate/20260213_budget1000_sharpe_signal_sprint27/run_queue.csv`
- Конфиги: `coint4/configs/budget_20260213_1000_sharpe_signal_sprint27/*.yaml`
- Размер: 10 прогонов (`5` вариантов × `holdout/stress`)
- Статус: `10/10 completed`
- Валидация: `Sharpe consistency OK (10 run(s))`

### Матрица параметров (sb*)
Фиксируем текущего лидера `ms0p1+ts1p5+slz3p0` и меняем только параметры structural-break защиты (новые поля Numba):
- `backtest.structural_break_min_correlation`
- `backtest.structural_break_entry_multiplier`
- `backtest.structural_break_exit_multiplier`

| variant | min_correlation | entry_multiplier | exit_multiplier |
|---|---:|---:|---:|
| sb_base | 0.30 | 1.5 | 1.2 |
| sb_mc0p2 | 0.20 | 1.5 | 1.2 |
| sb_mc0p4 | 0.40 | 1.5 | 1.2 |
| sb_mulLo | 0.30 | 1.3 | 1.1 |
| sb_mulHi | 0.30 | 2.0 | 1.5 |

### Результаты (10 прогонов)
| variant | kind | sharpe | pnl | max_dd | cost_ratio | trades | pairs |
|---|---|---:|---:|---:|---:|---:|---:|
| sb_base | holdout | 4.572 | 2463.52 | -536.99 | 0.08 | 4659 | 58 |
| sb_base | stress | 4.277 | 2196.52 | -525.86 | 0.16 | 4659 | 58 |
| sb_mc0p2 | holdout | 4.172 | 1941.10 | -541.44 | 0.09 | 4807 | 58 |
| sb_mc0p2 | stress | 3.840 | 1695.03 | -524.08 | 0.18 | 4807 | 58 |
| sb_mc0p4 | holdout | 4.171 | 2030.99 | -548.13 | 0.09 | 4525 | 58 |
| sb_mc0p4 | stress | 3.873 | 1796.70 | -534.93 | 0.17 | 4525 | 58 |
| sb_mulLo | holdout | 3.949 | 2014.33 | -498.66 | 0.09 | 4922 | 58 |
| sb_mulLo | stress | 3.650 | 1760.95 | -480.59 | 0.17 | 4922 | 58 |
| sb_mulHi | holdout | 4.000 | 1571.70 | -446.27 | 0.09 | 4017 | 58 |
| sb_mulHi | stress | 3.689 | 1384.88 | -447.89 | 0.18 | 4017 | 58 |

### Итог по sprint27
- Никакой вариант не улучшил baseline `sb_base` (параметры по умолчанию, эквивалентны прежним константам Numba): лидер остаётся `ms0p1+ts1p5+slz3p0`.
- Сдвиг `structural_break_min_correlation` в обе стороны (`0.20`/`0.40`) ухудшает robust Sharpe (stress до `~3.84–3.87`).
- Ослабление/усиление мультипликаторов (`sb_mulLo/sb_mulHi`) тоже снижает robust Sharpe; `sb_mulHi` уменьшает DD, но сильнее режет PnL и Sharpe.

## Extra sweep: signal sprint28 (market-regime clamp sweep under `ms0p1+ts1p5+slz3p0`, 10 прогонов)
- Очередь: `coint4/artifacts/wfa/aggregate/20260213_budget1000_sharpe_signal_sprint28/run_queue.csv`
- Конфиги: `coint4/configs/budget_20260213_1000_sharpe_signal_sprint28/*.yaml`
- Размер: 10 прогонов (`5` вариантов × `holdout/stress`)
- Статус: `10/10 completed`
- Валидация: `Sharpe consistency OK (10 run(s))`

### Матрица параметров (rg*)
Фиксируем текущего лидера `ms0p1+ts1p5+slz3p0` и меняем только clamp диапазон для Numba market-regime factor:
- `backtest.market_regime_factor_min`
- `backtest.market_regime_factor_max`

| variant | factor_min | factor_max |
|---|---:|---:|
| rg0p5to1p5 | 0.5 | 1.5 |
| rg1p0to1p5 | 1.0 | 1.5 |
| rg0p5to2p0 | 0.5 | 2.0 |
| rg0p5to3p0 | 0.5 | 3.0 |
| rg0p8to1p2 | 0.8 | 1.2 |

### Результаты (10 прогонов)
| variant | kind | sharpe | pnl | max_dd | cost_ratio | trades | pairs |
|---|---|---:|---:|---:|---:|---:|---:|
| rg0p5to1p5 | holdout | 4.572 | 2463.52 | -536.99 | 0.08 | 4659 | 58 |
| rg0p5to1p5 | stress | 4.277 | 2196.52 | -525.86 | 0.16 | 4659 | 58 |
| rg1p0to1p5 | holdout | 4.572 | 2463.52 | -536.99 | 0.08 | 4659 | 58 |
| rg1p0to1p5 | stress | 4.277 | 2196.52 | -525.86 | 0.16 | 4659 | 58 |
| rg0p5to2p0 | holdout | 3.713 | 1050.72 | -292.77 | 0.11 | 3699 | 58 |
| rg0p5to2p0 | stress | 3.354 | 909.77 | -295.12 | 0.22 | 3699 | 58 |
| rg0p5to3p0 | holdout | 3.832 | 1096.33 | -292.77 | 0.11 | 3689 | 58 |
| rg0p5to3p0 | stress | 3.473 | 953.14 | -295.12 | 0.21 | 3689 | 58 |
| rg0p8to1p2 | holdout | 2.654 | 1148.31 | -504.01 | 0.14 | 5219 | 58 |
| rg0p8to1p2 | stress | 2.349 | 942.19 | -501.96 | 0.29 | 5219 | 58 |

### Итог по sprint28
- Лидер не изменился: базовый clamp `rg0p5to1p5` остаётся лучшим по robust Sharpe.
- `rg1p0to1p5` даёт идентичные метрики baseline → в текущей реализации `regime_factor` практически не опускается ниже `1.0`, нижний clamp не лимитирует.
- Расширение верхнего clamp (`2.0–3.0`) снижает robust Sharpe (хотя DD становится меньше и trades ниже), narrow clamp (`0.8–1.2`) ломает Sharpe через рост churn/издержек.

## Extra sweep: signal sprint29 (max_pairs sweep under `ms0p1+ts1p5+slz3p0`, 10 прогонов)
- Очередь: `coint4/artifacts/wfa/aggregate/20260213_budget1000_sharpe_signal_sprint29/run_queue.csv`
- Конфиги: `coint4/configs/budget_20260213_1000_sharpe_signal_sprint29/*.yaml`
- Размер: 10 прогонов (`5` вариантов × `holdout/stress`)
- Статус: `10/10 completed`
- Валидация: `Sharpe consistency OK (10 run(s))`

### Матрица параметров (mp*)
Фиксируем текущего лидера `ms0p1+ts1p5+slz3p0` и меняем только `pair_selection.max_pairs`.

| variant | max_pairs |
|---|---:|
| mp12 | 12 |
| mp24 | 24 |
| mp36 | 36 |
| mp48 | 48 |
| mp60 | 60 |

### Результаты (10 прогонов)
| variant | kind | sharpe | pnl | max_dd | cost_ratio | trades | pairs |
|---|---|---:|---:|---:|---:|---:|---:|
| mp12 | holdout | 3.209 | 668.07 | -293.43 | 0.12 | 2502 | 27 |
| mp12 | stress | 2.905 | 586.43 | -300.67 | 0.24 | 2502 | 27 |
| mp24 | holdout | 4.572 | 2463.52 | -536.99 | 0.08 | 4659 | 58 |
| mp24 | stress | 4.277 | 2196.52 | -525.86 | 0.16 | 4659 | 58 |
| mp36 | holdout | 3.899 | 2594.83 | -620.87 | 0.11 | 6986 | 81 |
| mp36 | stress | 3.557 | 2199.49 | -591.42 | 0.22 | 6986 | 81 |
| mp48 | holdout | 3.277 | 2099.90 | -573.67 | 0.14 | 7654 | 83 |
| mp48 | stress | 2.946 | 1729.31 | -544.59 | 0.28 | 7654 | 83 |
| mp60 | holdout | 3.277 | 2099.90 | -573.67 | 0.14 | 7654 | 83 |
| mp60 | stress | 2.946 | 1729.31 | -544.59 | 0.28 | 7654 | 83 |

### Итог по sprint29
- Лидер не изменился: `max_pairs=24` остаётся лучшим по robust Sharpe.
- `max_pairs=12` режет диверсификацию и ухудшает Sharpe.
- `max_pairs>=36` увеличивает turnover/издержки и ухудшает Sharpe; `max_pairs=48` и `60` дают идентичные метрики (сaturation по доступным парам/сигналам).

## Extra sweep: signal sprint30 (training_period_days sweep under `ms0p1+ts1p5+slz3p0`, 10 прогонов)
- Очередь: `coint4/artifacts/wfa/aggregate/20260213_budget1000_sharpe_signal_sprint30/run_queue.csv`
- Конфиги: `coint4/configs/budget_20260213_1000_sharpe_signal_sprint30/*.yaml`
- Размер: 10 прогонов (`5` вариантов × `holdout/stress`)
- Статус: `10/10 completed`
- Валидация: `Sharpe consistency OK (10 run(s))`

### Матрица параметров (tr*)
Фиксируем текущего лидера `ms0p1+ts1p5+slz3p0` и меняем только `walk_forward.training_period_days`.

| variant | training_period_days |
|---|---:|
| tr60 | 60 |
| tr90 | 90 |
| tr120 | 120 |
| tr180 | 180 |
| tr240 | 240 |

### Результаты (10 прогонов)
| variant | kind | sharpe | pnl | max_dd | cost_ratio | trades | pairs |
|---|---|---:|---:|---:|---:|---:|---:|
| tr60 | holdout | -0.778 | -310.88 | -792.49 | — | 4708 | 62 |
| tr60 | stress | -1.117 | -384.53 | -822.47 | — | 4708 | 62 |
| tr90 | holdout | 4.572 | 2463.52 | -536.99 | 0.08 | 4659 | 58 |
| tr90 | stress | 4.277 | 2196.52 | -525.86 | 0.16 | 4659 | 58 |
| tr120 | holdout | 2.475 | 921.49 | -653.33 | 0.16 | 4567 | 63 |
| tr120 | stress | 2.176 | 753.91 | -651.20 | 0.33 | 4567 | 63 |
| tr180 | holdout | 2.421 | 890.07 | -459.06 | 0.13 | 4460 | 54 |
| tr180 | stress | 2.126 | 726.91 | -458.41 | 0.28 | 4460 | 54 |
| tr240 | holdout | 3.740 | 1817.58 | -469.10 | 0.09 | 4436 | 54 |
| tr240 | stress | 3.459 | 1594.82 | -467.88 | 0.17 | 4436 | 54 |

### Итог по sprint30
- Лидер не изменился: baseline `tr90` (training=90d) остаётся лучшим по robust Sharpe.
- `tr60` (training=60d) ломает стратегию: отрицательный Sharpe и PnL при заметно большем DD.
- Увеличение training-window (`120–240d`) снижает robust Sharpe и повышает stress cost_ratio; `tr240` частично восстанавливает Sharpe, но остаётся хуже `tr90`.
