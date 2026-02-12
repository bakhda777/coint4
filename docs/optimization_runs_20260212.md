# Журнал прогонов оптимизации (2026-02-12)

Цель: проверить гипотезу восстановления Sharpe для капитала `$1000` на extended OOS `2023-05-01 -> 2024-04-30` через:
- смягчение фильтров (`min_correlation`, `coint_pvalue_threshold`) для роста числа торгуемых пар;
- ослабление нелинейности sizing (`min_notional_per_trade`, `max_notional_per_trade`);
- умеренный рост `max_pairs`/`max_active_positions` при сохранении `tlow` логики.

## Валидация перед запуском
- Проверка консистентности Sharpe для последних `$1000` очередей: `Sharpe consistency OK (28 run(s))`.
- Проверка артефактов для очередей `20260131_budget1000_*`: обязательные файлы (`strategy_metrics.csv`, `trade_statistics.csv`, `equity_curve.csv`, `daily_pnl.csv`) присутствуют.
- В глобальном rollup есть 3 старых `completed` без метрик (группа `20260119_relaxed8_nokpss_u250_turnover_stress`), на текущую серию это не влияет.

## Новая очередь
- Очередь: `coint4/artifacts/wfa/aggregate/20260212_budget1000_tlow_extended_sharpe_recover10/run_queue.csv`
- Конфиги: `coint4/configs/budget_20260212_1000_tlow_extended_sharpe_recover10/*.yaml`
- Размер: 10 прогонов (`5` вариантов × `holdout/stress`)

### Матрица параметров (варианты r1-r5)
| variant | z | ms | corr | pvalue | max_pairs | max_active_positions | risk | min_notional | max_notional |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| r1 | 1.25 | 0.35 | 0.38 | 0.25 | 15 | 15 | 0.0175 | 10 | 25 |
| r2 | 1.25 | 0.35 | 0.36 | 0.25 | 18 | 15 | 0.0175 | 10 | 30 |
| r3 | 1.30 | 0.35 | 0.38 | 0.25 | 15 | 15 | 0.0175 | 10 | 25 |
| r4 | 1.25 | 0.30 | 0.36 | 0.30 | 20 | 16 | 0.0150 | 8 | 30 |
| r5 | 1.20 | 0.30 | 0.40 | 0.22 | 15 | 15 | 0.0175 | 10 | 40 |

## Запуск на удалённом сервере
- Целевой хост: `85.198.90.128` (только для тяжёлых WFA).
- Команда очереди:
  - `COINT_FILTER_BACKEND=threads PYTHONPATH=src ./.venv/bin/python scripts/optimization/run_wfa_queue.py --queue artifacts/wfa/aggregate/20260212_budget1000_tlow_extended_sharpe_recover10/run_queue.csv --parallel 8`
- Локальная dry-run валидация очереди: `10` run(s), все `planned`.
- Фактический статус после запуска: `completed` для всех `10/10` прогонов.

## Критерии отбора после завершения
- Primary: максимизировать `min(Sharpe_holdout, Sharpe_stress)`.
- Гейты: `total_trades >= 500`, `total_pairs_traded >= 50`, `max_drawdown_abs >= -250`.
- Cost gate: `stress cost_ratio <= 0.5` (при положительном `total_pnl`).

## Результаты (10 прогонов)
| variant | kind | sharpe | pnl | max_dd | cost_ratio | trades | pairs |
|---|---|---:|---:|---:|---:|---:|---:|
| r1 | holdout | 0.255 | 7.86 | -282.69 | 11.26 | 2563 | 44 |
| r1 | stress | -0.121 | -56.26 | -289.28 | — | 2563 | 44 |
| r2 | holdout | 0.501 | 52.75 | -339.20 | 1.99 | 3001 | 52 |
| r2 | stress | 0.117 | -24.62 | -344.77 | — | 3001 | 52 |
| r3 | holdout | 0.455 | 43.21 | -293.13 | 1.92 | 2481 | 44 |
| r3 | stress | 0.081 | -20.47 | -311.10 | — | 2481 | 44 |
| r4 | holdout | 1.871 | 343.31 | -343.73 | 0.27 | 3271 | 53 |
| r4 | stress | 1.482 | 254.89 | -355.54 | 0.62 | 3271 | 53 |
| r5 | holdout | 1.258 | 199.49 | -326.89 | 0.44 | 2644 | 48 |
| r5 | stress | 0.878 | 123.38 | -335.93 | 1.24 | 2644 | 48 |

## Итог
- Лучший по `min(Sharpe_holdout, Sharpe_stress)`: `r4` (`1.482`), также это единственный вариант с устойчиво положительным stress PnL.
- Плюс: удалось поднять число пар до `53` (в прошлой серии было `36`), что подтверждает гипотезу про bottleneck в фильтрах/caps.
- Минус: ни один вариант не прошёл риск-гейт по просадке (`max_dd >= -250`), а `r4` не прошёл cost-gate (`stress cost_ratio 0.62 > 0.5`).
- Вывод: Sharpe улучшили, но профиль риска для `$1000` в extended OOS остаётся слишком агрессивным; дальше нужен отдельный DD/cost downshift вокруг `r4`.

## Extra sweep: min_notional без ограничений (12 прогонов)
- Запрос: «максимальный Sharpe любой ценой, стартовый капитал `$1000`».
- Очередь: `coint4/artifacts/wfa/aggregate/20260212_budget1000_sharpe_unbounded_minnot/run_queue.csv`
- Конфиги: `coint4/configs/budget_20260212_1000_sharpe_unbounded_minnot/*.yaml`
- Размер: 12 прогонов (`6` вариантов `u1-u6` × `holdout/stress`)
- Статус: `12/12 completed`

### Матрица параметров (u1-u6)
| variant | z | ms | corr | pvalue | max_pairs | max_active_positions | risk | min_notional | max_notional |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| u1 | 1.25 | 0.30 | 0.36 | 0.30 | 20 | 16 | 0.0150 | 0.5 | 1000 |
| u2 | 1.25 | 0.30 | 0.36 | 0.30 | 20 | 16 | 0.0150 | 1.0 | 1000 |
| u3 | 1.25 | 0.30 | 0.36 | 0.30 | 20 | 16 | 0.0150 | 2.0 | 1000 |
| u4 | 1.25 | 0.30 | 0.34 | 0.35 | 24 | 18 | 0.0125 | 1.0 | 1000 |
| u5 | 1.20 | 0.25 | 0.34 | 0.35 | 24 | 18 | 0.0200 | 1.0 | 500 |
| u6 | 1.20 | 0.25 | 0.34 | 0.35 | 24 | 18 | 0.0150 | 5.0 | 1000 |

### Результаты (12 прогонов)
| variant | kind | sharpe | pnl | max_dd | cost_ratio | trades | pairs |
|---|---|---:|---:|---:|---:|---:|---:|
| u1 | holdout | 1.870 | 343.31 | -343.73 | 0.27 | 3271 | 53 |
| u1 | stress | 1.482 | 254.89 | -355.54 | 0.62 | 3271 | 53 |
| u2 | holdout | 1.870 | 343.31 | -343.73 | 0.27 | 3271 | 53 |
| u2 | stress | 1.482 | 254.89 | -355.54 | 0.62 | 3271 | 53 |
| u3 | holdout | 1.870 | 343.31 | -343.73 | 0.27 | 3271 | 53 |
| u3 | stress | 1.482 | 254.89 | -355.54 | 0.62 | 3271 | 53 |
| u4 | holdout | 1.461 | 247.49 | -288.18 | 0.39 | 3895 | 58 |
| u4 | stress | 1.076 | 165.67 | -289.07 | 1.01 | 3895 | 58 |
| u5 | holdout | 2.771 | 1122.42 | -467.99 | 0.16 | 4047 | 58 |
| u5 | stress | 2.421 | 912.62 | -470.57 | 0.34 | 4047 | 58 |
| u6 | holdout | 2.775 | 806.80 | -343.12 | 0.16 | 4047 | 58 |
| u6 | stress | 2.425 | 668.69 | -346.67 | 0.34 | 4047 | 58 |

### Выводы по min_notional
- В зоне `u1-u3` изменение `min_notional=0.5/1/2` дало идентичные метрики: в этих параметрах `min_notional` не является лимитирующим.
- Новый максимум Sharpe получен в `u6`:
  - holdout Sharpe `2.775`
  - stress Sharpe `2.425`
  - `min(holdout, stress) = 2.425` (лучший результат на `$1000` в текущей extended OOS ветке).
- `u5` и `u6` резко лучше `u1-u4` из-за комбинированного сдвига режима (`z=1.20`, `ms=0.25`, более мягкие corr/pvalue, больше пар), а не только за счёт `min_notional`.
- Если цель только максимум Sharpe (без ограничений DD), текущий лидер: `u6`.

## Extra sweep: signal sprint around `u6` (10 прогонов)
- Запрос: ещё поднять Sharpe при фиксированном стартовом капитале `$1000` и `min_notional=5` (без обрезаний по мелким сделкам).
- Очередь: `coint4/artifacts/wfa/aggregate/20260212_budget1000_sharpe_signal_sprint1/run_queue.csv`
- Конфиги: `coint4/configs/budget_20260212_1000_sharpe_signal_sprint1/*.yaml`
- Размер: 10 прогонов (`5` вариантов `v1-v5` × `holdout/stress`)
- Запуск: `85.198.90.128`, статус `10/10 completed`

### Валидация записи результатов
- `Sharpe consistency OK (10 run(s))` для всей очереди `signal_sprint1`.
- Обязательные артефакты (`strategy_metrics.csv`, `trade_statistics.csv`, `equity_curve.csv`, `daily_pnl.csv`, `run.log`, `run.commands.log`) присутствуют в `10/10` run directories.
- В `run.log` нет `Traceback`/`ERROR`/`Exception`.

### Матрица параметров (v1-v5)
| variant | z | exit | ms | corr | pvalue | max_pairs | max_active_positions | risk | min_notional | max_notional |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| v1 | 1.15 | 0.08 | 0.20 | 0.34 | 0.35 | 24 | 18 | 0.0150 | 5 | 1000 |
| v2 | 1.15 | 0.08 | 0.25 | 0.34 | 0.35 | 24 | 18 | 0.0150 | 5 | 1000 |
| v3 | 1.20 | 0.08 | 0.20 | 0.34 | 0.35 | 24 | 18 | 0.0150 | 5 | 1000 |
| v4 | 1.20 | 0.10 | 0.25 | 0.32 | 0.40 | 32 | 18 | 0.0150 | 5 | 1000 |
| v5 | 1.10 | 0.10 | 0.25 | 0.32 | 0.40 | 32 | 18 | 0.0150 | 5 | 1000 |

### Результаты (10 прогонов)
| variant | kind | sharpe | pnl | max_dd | cost_ratio | trades | pairs |
|---|---|---:|---:|---:|---:|---:|---:|
| v1 | holdout | 3.338 | 1150.58 | -327.39 | 0.13 | 4204 | 58 |
| v1 | stress | 3.007 | 986.38 | -313.70 | 0.26 | 4204 | 58 |
| v2 | holdout | 3.222 | 1055.74 | -310.93 | 0.14 | 4199 | 58 |
| v2 | stress | 2.880 | 897.68 | -302.34 | 0.28 | 4199 | 58 |
| v3 | holdout | 2.905 | 860.20 | -344.06 | 0.15 | 4049 | 58 |
| v3 | stress | 2.555 | 718.63 | -347.41 | 0.31 | 4049 | 58 |
| v4 | holdout | 1.585 | 415.59 | -467.90 | 0.44 | 5509 | 71 |
| v4 | stress | 1.173 | 263.84 | -476.99 | 1.17 | 5509 | 71 |
| v5 | holdout | 0.776 | 130.81 | -612.52 | 1.29 | 5903 | 71 |
| v5 | stress | 0.380 | -6.09 | -607.87 | — | 5903 | 71 |

### Итог по sprint1
- Новый лидер по `min(Sharpe_holdout, Sharpe_stress)`: `v1` c `3.007` (`3.338/3.007` holdout/stress).
- Прирост к прошлому лидеру `u6` (`2.425` stress Sharpe): `+0.582` по robust-метрике `min_sharpe`.
- Режим `v1-v3` (58 пар, ~4.0-4.2k сделок) дал лучший Sharpe и PnL; расширение до 71 пар (`v4-v5`) ухудшило Sharpe и резко подняло DD/cost_ratio.
- Если цель действительно «максимальный Sharpe любой ценой» при `$1000`, текущий лучший конфиг: `v1`.

## Extra sweep: signal sprint2 (local search around `v1`, 10 прогонов)
- Очередь: `coint4/artifacts/wfa/aggregate/20260212_budget1000_sharpe_signal_sprint2/run_queue.csv`
- Конфиги: `coint4/configs/budget_20260212_1000_sharpe_signal_sprint2/*.yaml`
- Размер: 10 прогонов (`5` вариантов `s1-s5` × `holdout/stress`)
- Статус: `10/10 completed`
- Валидация: `Sharpe consistency OK (10 run(s))`
- Note: в части прогонов были `ERROR/Traceback` при инициализации memory-mapped кэша `.cache/consolidated_prices.parquet` (коррупция из-за параллельных записей); результаты при этом корректно записались (полный набор артефактов), а в коде добавлен lock + atomic replace + range-keyed cache filename.

### Матрица параметров (s1-s5)
Все параметры, кроме signal-части, фиксированы как в `v1` (universe/filters/sizing/limits).

| variant | z | exit | ms | hold | cd |
|---|---:|---:|---:|---:|---:|
| s1 | 1.10 | 0.08 | 0.18 | 300 | 300 |
| s2 | 1.10 | 0.07 | 0.16 | 300 | 300 |
| s3 | 1.12 | 0.07 | 0.16 | 300 | 300 |
| s4 | 1.14 | 0.07 | 0.18 | 300 | 300 |
| s5 | 1.10 | 0.08 | 0.16 | 240 | 240 |

### Результаты (10 прогонов)
| variant | kind | sharpe | pnl | max_dd | cost_ratio | trades | pairs |
|---|---|---:|---:|---:|---:|---:|---:|
| s1 | holdout | 2.166 | 619.08 | -470.06 | 0.21 | 4351 | 58 |
| s1 | stress | 1.815 | 481.59 | -488.43 | 0.47 | 4351 | 58 |
| s2 | holdout | 1.101 | 205.99 | -499.48 | 0.55 | 4310 | 58 |
| s2 | stress | 0.691 | 99.07 | -510.35 | 1.95 | 4310 | 58 |
| s3 | holdout | 1.283 | 253.40 | -475.60 | 0.45 | 4257 | 58 |
| s3 | stress | 0.874 | 144.61 | -483.32 | 1.34 | 4257 | 58 |
| s4 | holdout | 1.757 | 396.14 | -422.76 | 0.30 | 4200 | 58 |
| s4 | stress | 1.365 | 279.83 | -431.69 | 0.73 | 4200 | 58 |
| s5 | holdout | 2.467 | 656.73 | -464.57 | 0.20 | 4464 | 58 |
| s5 | stress | 2.067 | 514.32 | -463.73 | 0.44 | 4464 | 58 |

### Итог по sprint2
- Ни один вариант `s1-s5` не улучшил `v1`; лучший robust `min_sharpe` у `s5` = `2.067` (хуже `v1` = `3.007`).
- Понижение `z` и `min_spread_move_sigma` относительно `v1` в этом extended OOS ухудшает Sharpe и увеличивает DD → локальный оптимум в зоне `z≈1.15`, `ms≈0.20`, `hold/cd=300`.

## Extra sweep: signal sprint3 (z fine sweep, 10 прогонов)
- Очередь: `coint4/artifacts/wfa/aggregate/20260212_budget1000_sharpe_signal_sprint3/run_queue.csv`
- Конфиги: `coint4/configs/budget_20260212_1000_sharpe_signal_sprint3/*.yaml`
- Размер: 10 прогонов (`5` вариантов `zf1-zf5` × `holdout/stress`)
- Статус: `10/10 completed`
- Валидация: `Sharpe consistency OK (10 run(s))`

### Матрица параметров (zf1-zf5)
Фиксируем `exit=0.08`, `ms=0.20`, `hold/cd=300` как в `v1`, меняем только `z`.

| variant | z |
|---|---:|
| zf1 | 1.12 |
| zf2 | 1.13 |
| zf3 | 1.14 |
| zf4 | 1.15 |
| zf5 | 1.16 |

### Результаты (10 прогонов)
| variant | kind | sharpe | pnl | max_dd | cost_ratio | trades | pairs |
|---|---|---:|---:|---:|---:|---:|---:|
| zf1 | holdout | 2.680 | 822.07 | -365.47 | 0.17 | 4291 | 58 |
| zf1 | stress | 2.332 | 674.33 | -386.57 | 0.35 | 4291 | 58 |
| zf2 | holdout | 2.959 | 937.11 | -325.13 | 0.15 | 4270 | 58 |
| zf2 | stress | 2.611 | 783.47 | -347.15 | 0.32 | 4270 | 58 |
| zf3 | holdout | 3.055 | 1012.78 | -318.19 | 0.14 | 4238 | 58 |
| zf3 | stress | 2.720 | 855.57 | -340.73 | 0.29 | 4238 | 58 |
| zf4 | holdout | 3.338 | 1150.58 | -327.39 | 0.13 | 4204 | 58 |
| zf4 | stress | 3.007 | 986.38 | -313.70 | 0.26 | 4204 | 58 |
| zf5 | holdout | 2.878 | 923.03 | -380.25 | 0.15 | 4178 | 58 |
| zf5 | stress | 2.545 | 773.27 | -380.73 | 0.32 | 4178 | 58 |

### Итог по sprint3
- `zf4` (z=1.15) снова лучший по `min(Sharpe_holdout, Sharpe_stress)` и совпадает с `v1` один-в-один → пик по `z` для этого режима найден.
- Дальнейший рост Sharpe требует следующего измерения: `exit`, `min_spread_move_sigma`, `hold/cd`, sizing/risk, или изменения selection/filters.

## Extra sweep: signal sprint4 (exit sweep, 10 прогонов)
- Очередь: `coint4/artifacts/wfa/aggregate/20260212_budget1000_sharpe_signal_sprint4/run_queue.csv`
- Конфиги: `coint4/configs/budget_20260212_1000_sharpe_signal_sprint4/*.yaml`
- Размер: 10 прогонов (`5` вариантов `ex1-ex5` × `holdout/stress`)
- Статус: `10/10 completed`
- Валидация: `Sharpe consistency OK (10 run(s))`

### Матрица параметров (ex1-ex5)
Фиксируем `z=1.15`, `ms=0.20`, `hold/cd=300` как в `v1`, меняем только `exit`.

| variant | exit |
|---|---:|
| ex1 | 0.06 |
| ex2 | 0.07 |
| ex3 | 0.08 |
| ex4 | 0.09 |
| ex5 | 0.10 |

### Результаты (10 прогонов)
| variant | kind | sharpe | pnl | max_dd | cost_ratio | trades | pairs |
|---|---|---:|---:|---:|---:|---:|---:|
| ex1 | holdout | 3.057 | 856.26 | -287.33 | 0.16 | 4109 | 58 |
| ex1 | stress | 2.687 | 714.35 | -285.84 | 0.33 | 4109 | 58 |
| ex2 | holdout | 2.355 | 581.74 | -328.52 | 0.22 | 4165 | 58 |
| ex2 | stress | 1.967 | 455.20 | -337.43 | 0.48 | 4165 | 58 |
| ex3 | holdout | 3.338 | 1150.58 | -327.39 | 0.13 | 4204 | 58 |
| ex3 | stress | 3.007 | 986.38 | -313.70 | 0.26 | 4204 | 58 |
| ex4 | holdout | 2.815 | 819.99 | -315.07 | 0.17 | 4227 | 58 |
| ex4 | stress | 2.456 | 676.56 | -321.05 | 0.36 | 4227 | 58 |
| ex5 | holdout | 2.923 | 864.31 | -290.78 | 0.17 | 4260 | 58 |
| ex5 | stress | 2.562 | 716.90 | -292.24 | 0.35 | 4260 | 58 |

### Итог по sprint4
- `ex3` (exit=0.08) снова лучший по `min(Sharpe_holdout, Sharpe_stress)` и совпадает с `v1`.
- По `exit` локальный максимум найден на `0.08` в этом режиме.

## Extra sweep: signal sprint5 (ms sweep, 10 прогонов)
- Очередь: `coint4/artifacts/wfa/aggregate/20260212_budget1000_sharpe_signal_sprint5/run_queue.csv`
- Конфиги: `coint4/configs/budget_20260212_1000_sharpe_signal_sprint5/*.yaml`
- Размер: 10 прогонов (`5` вариантов `ms1-ms5` × `holdout/stress`)
- Статус: `10/10 completed`
- Валидация: `Sharpe consistency OK (10 run(s))`

### Матрица параметров (ms1-ms5)
Фиксируем `z=1.15`, `exit=0.08`, `hold/cd=300` как в `v1`, меняем только `min_spread_move_sigma`.

| variant | ms |
|---|---:|
| ms1 | 0.16 |
| ms2 | 0.18 |
| ms3 | 0.20 |
| ms4 | 0.22 |
| ms5 | 0.24 |

### Результаты (10 прогонов)
| variant | kind | sharpe | pnl | max_dd | cost_ratio | trades | pairs |
|---|---|---:|---:|---:|---:|---:|---:|
| ms1 | holdout | 3.097 | 1035.17 | -368.39 | 0.14 | 4206 | 58 |
| ms1 | stress | 2.764 | 877.37 | -376.48 | 0.28 | 4206 | 58 |
| ms2 | holdout | 3.103 | 1037.72 | -369.16 | 0.14 | 4205 | 58 |
| ms2 | stress | 2.770 | 879.72 | -374.65 | 0.28 | 4205 | 58 |
| ms3 | holdout | 3.338 | 1150.58 | -327.39 | 0.13 | 4204 | 58 |
| ms3 | stress | 3.007 | 986.38 | -313.70 | 0.26 | 4204 | 58 |
| ms4 | holdout | 3.134 | 1018.38 | -306.22 | 0.15 | 4199 | 58 |
| ms4 | stress | 2.793 | 862.61 | -307.33 | 0.29 | 4199 | 58 |
| ms5 | holdout | 3.259 | 1071.91 | -313.37 | 0.14 | 4198 | 58 |
| ms5 | stress | 2.917 | 912.80 | -302.29 | 0.28 | 4198 | 58 |

### Итог по sprint5
- `ms3` (ms=0.20) снова лучший по `min(Sharpe_holdout, Sharpe_stress)` и совпадает с `v1`.
- По `min_spread_move_sigma` локальный максимум найден на `0.20` в этом режиме.

## Extra sweep: signal sprint6 (max_pairs sweep, 10 прогонов)
- Очередь: `coint4/artifacts/wfa/aggregate/20260212_budget1000_sharpe_signal_sprint6/run_queue.csv`
- Конфиги: `coint4/configs/budget_20260212_1000_sharpe_signal_sprint6/*.yaml`
- Размер: 10 прогонов (`5` вариантов `mp1-mp5` × `holdout/stress`)
- Статус: `10/10 completed`
- Валидация: `Sharpe consistency OK (10 run(s))`

### Матрица параметров (mp1-mp5)
Фиксируем signal-параметры `v1` (`z=1.15`, `exit=0.08`, `ms=0.20`, `hold/cd=300`) и фильтры, меняем только `pair_selection.max_pairs`.

| variant | max_pairs |
|---|---:|
| mp1 | 12 |
| mp2 | 16 |
| mp3 | 20 |
| mp4 | 24 |
| mp5 | 28 |

### Результаты (10 прогонов)
| variant | kind | sharpe | pnl | max_dd | cost_ratio | trades | pairs |
|---|---|---:|---:|---:|---:|---:|---:|
| mp1 | holdout | 2.924 | 524.63 | -172.40 | 0.14 | 2256 | 27 |
| mp1 | stress | 2.622 | 456.86 | -180.01 | 0.27 | 2256 | 27 |
| mp2 | holdout | 2.649 | 526.25 | -217.65 | 0.17 | 2922 | 39 |
| mp2 | stress | 2.302 | 439.38 | -236.54 | 0.36 | 2922 | 39 |
| mp3 | holdout | 1.807 | 362.61 | -304.73 | 0.29 | 3536 | 52 |
| mp3 | stress | 1.432 | 266.70 | -326.74 | 0.67 | 3536 | 52 |
| mp4 | holdout | 3.338 | 1150.58 | -327.39 | 0.13 | 4204 | 58 |
| mp4 | stress | 3.007 | 986.38 | -313.70 | 0.26 | 4204 | 58 |
| mp5 | holdout | 2.695 | 920.49 | -379.02 | 0.19 | 4935 | 64 |
| mp5 | stress | 2.335 | 746.60 | -402.78 | 0.40 | 4935 | 64 |

### Итог по sprint6
- Лучший по `min(Sharpe_holdout, Sharpe_stress)` — `mp4` (max_pairs=24), совпадает с `v1`.
- Уменьшение `max_pairs` снижает Sharpe, но резко уменьшает DD (например `mp1` DD ~-18% против ~-31% у `v1`) — это путь для “quality/risk” режима, но не для max-Sharpe.
