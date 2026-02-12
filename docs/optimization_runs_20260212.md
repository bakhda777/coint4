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

## Extra sweep: signal sprint7 (stop_loss_zscore sweep, 10 прогонов)
- Очередь: `coint4/artifacts/wfa/aggregate/20260212_budget1000_sharpe_signal_sprint7/run_queue.csv`
- Конфиги: `coint4/configs/budget_20260212_1000_sharpe_signal_sprint7/*.yaml`
- Размер: 10 прогонов (`5` вариантов `slz2p0-slz4p0` × `holdout/stress`)
- Статус: `10/10 completed`
- Валидация: `Sharpe consistency OK (10 run(s))`

### Матрица параметров (slz2p0-slz4p0)
Фиксируем параметры `v1`, меняем только `backtest.pair_stop_loss_zscore`.

| variant | stop_loss_z |
|---|---:|
| slz2p0 | 2.0 |
| slz2p5 | 2.5 |
| slz3p0 | 3.0 |
| slz3p5 | 3.5 |
| slz4p0 | 4.0 |

### Результаты (10 прогонов)
| variant | kind | sharpe | pnl | max_dd | cost_ratio | trades | pairs |
|---|---|---:|---:|---:|---:|---:|---:|
| slz2p0 | holdout | 1.632 | 257.18 | -276.90 | 0.62 | 5193 | 58 |
| slz2p0 | stress | 0.944 | 127.44 | -290.61 | 2.14 | 5193 | 58 |
| slz2p5 | holdout | 1.330 | 244.60 | -241.94 | 0.55 | 4638 | 58 |
| slz2p5 | stress | 0.843 | 129.30 | -254.82 | 1.78 | 4638 | 58 |
| slz3p0 | holdout | 3.338 | 1150.58 | -327.39 | 0.13 | 4204 | 58 |
| slz3p0 | stress | 3.007 | 986.38 | -313.70 | 0.26 | 4204 | 58 |
| slz3p5 | holdout | 1.172 | 250.76 | -589.47 | 0.47 | 3953 | 58 |
| slz3p5 | stress | 0.855 | 149.72 | -625.86 | 1.36 | 3953 | 58 |
| slz4p0 | holdout | 0.726 | 112.28 | -811.17 | 0.89 | 3776 | 58 |
| slz4p0 | stress | 0.463 | 21.32 | -836.05 | 8.06 | 3776 | 58 |

### Итог по sprint7
- Лучший по `min(Sharpe_holdout, Sharpe_stress)` — `slz3p0` (stop_loss_z=3.0), совпадает с `v1`.
- Слишком агрессивный stop-loss (`2.0-2.5`) резко повышает churn/издержки и обнуляет edge; слишком мягкий (`3.5-4.0`) раздувает DD и тоже рушит Sharpe.

## Extra sweep: signal sprint8 (max_active_positions sweep, 10 прогонов)
- Очередь: `coint4/artifacts/wfa/aggregate/20260212_budget1000_sharpe_signal_sprint8/run_queue.csv`
- Конфиги: `coint4/configs/budget_20260212_1000_sharpe_signal_sprint8/*.yaml`
- Размер: 10 прогонов (`5` вариантов `ap6-ap24` × `holdout/stress`)
- Статус: `10/10 completed`
- Валидация: `Sharpe consistency OK (10 run(s))`

### Матрица параметров (ap6-ap24)
Фиксируем параметры `v1`, меняем только `portfolio.max_active_positions`.

| variant | max_active_positions |
|---|---:|
| ap6 | 6 |
| ap10 | 10 |
| ap14 | 14 |
| ap18 | 18 |
| ap24 | 24 |

### Результаты (10 прогонов)
| variant | kind | sharpe | pnl | max_dd | cost_ratio | trades | pairs |
|---|---|---:|---:|---:|---:|---:|---:|
| ap6 | holdout | 2.679 | 697.98 | -235.75 | 0.20 | 4204 | 58 |
| ap6 | stress | 2.411 | 605.28 | -249.73 | 0.40 | 4204 | 58 |
| ap10 | holdout | 3.219 | 1060.25 | -317.72 | 0.14 | 4204 | 58 |
| ap10 | stress | 2.907 | 914.15 | -316.52 | 0.28 | 4204 | 58 |
| ap14 | holdout | 3.294 | 1128.06 | -323.97 | 0.13 | 4204 | 58 |
| ap14 | stress | 2.964 | 966.27 | -309.02 | 0.27 | 4204 | 58 |
| ap18 | holdout | 3.338 | 1150.58 | -327.39 | 0.13 | 4204 | 58 |
| ap18 | stress | 3.007 | 986.38 | -313.70 | 0.26 | 4204 | 58 |
| ap24 | holdout | 3.334 | 1148.42 | -327.07 | 0.13 | 4204 | 58 |
| ap24 | stress | 3.002 | 984.30 | -314.69 | 0.26 | 4204 | 58 |

### Итог по sprint8
- Лучший по `min(Sharpe_holdout, Sharpe_stress)` — `ap18` (max_active_positions=18), совпадает с `v1`.
- `ap24` почти идентичен, но чуть хуже по robust-метрике (и не даёт прироста Sharpe).
- Снижение лимита позиций уменьшает DD, но Sharpe падает (особенно `ap6`).

## Extra sweep: signal sprint9 (max_var_multiplier sweep, 10 прогонов)
- Очередь: `coint4/artifacts/wfa/aggregate/20260212_budget1000_sharpe_signal_sprint9/run_queue.csv`
- Конфиги: `coint4/configs/budget_20260212_1000_sharpe_signal_sprint9/*.yaml`
- Размер: 10 прогонов (`5` вариантов `vm1/vm2/vm3/vm4/vm6` × `holdout/stress`)
- Статус: `10/10 completed`
- Валидация: `Sharpe consistency OK (10 run(s))`

### Матрица параметров (vm1-vm6)
Фиксируем параметры `v1`, меняем только `backtest.max_var_multiplier` (cap для адаптивных порогов). `max_var_multiplier` должен быть `> 1`.

| variant | max_var_multiplier |
|---|---:|
| vm1 | 1.10 |
| vm2 | 2.00 |
| vm3 | 3.00 |
| vm4 | 4.00 |
| vm6 | 6.00 |

### Результаты (10 прогонов)
| variant | kind | sharpe | pnl | max_dd | cost_ratio | trades | pairs |
|---|---|---:|---:|---:|---:|---:|---:|
| vm1 | holdout | 3.650 | 1362.97 | -369.32 | 0.12 | 4478 | 58 |
| vm1 | stress | 3.306 | 1173.87 | -370.45 | 0.24 | 4478 | 58 |
| vm2 | holdout | 3.378 | 1171.92 | -329.96 | 0.13 | 4212 | 58 |
| vm2 | stress | 3.046 | 1006.14 | -313.17 | 0.26 | 4212 | 58 |
| vm3 | holdout | 3.338 | 1150.58 | -327.39 | 0.13 | 4204 | 58 |
| vm3 | stress | 3.007 | 986.38 | -313.70 | 0.26 | 4204 | 58 |
| vm4 | holdout | 3.341 | 1152.21 | -327.64 | 0.13 | 4204 | 58 |
| vm4 | stress | 3.010 | 987.92 | -312.97 | 0.26 | 4204 | 58 |
| vm6 | holdout | 3.345 | 1154.56 | -328.00 | 0.13 | 4204 | 58 |
| vm6 | stress | 3.014 | 990.13 | -312.25 | 0.26 | 4204 | 58 |

### Итог по sprint9
- Sharpe заметно растёт при уменьшении `max_var_multiplier` (меньше “задавливаем” входы в высокую волатильность).
- Новый лидер по `min(Sharpe_holdout, Sharpe_stress)` — `vm1` (`3.650/3.306`), выше `v1` (`3.338/3.007`).

## Extra sweep: signal sprint10 (max_var_multiplier fine sweep, 10 прогонов)
- Очередь: `coint4/artifacts/wfa/aggregate/20260212_budget1000_sharpe_signal_sprint10/run_queue.csv`
- Конфиги: `coint4/configs/budget_20260212_1000_sharpe_signal_sprint10/*.yaml`
- Размер: 10 прогонов (`5` вариантов `vmf101-vmf120` × `holdout/stress`)
- Статус: `10/10 completed`
- Валидация: `Sharpe consistency OK (10 run(s))`

### Матрица параметров (vmf101-vmf120)
Фиксируем параметры `v1`, меняем только `backtest.max_var_multiplier`.

| variant | max_var_multiplier |
|---|---:|
| vmf101 | 1.01 |
| vmf105 | 1.05 |
| vmf110 | 1.10 |
| vmf115 | 1.15 |
| vmf120 | 1.20 |

### Результаты (10 прогонов)
| variant | kind | sharpe | pnl | max_dd | cost_ratio | trades | pairs |
|---|---|---:|---:|---:|---:|---:|---:|
| vmf101 | holdout | 4.348 | 2153.76 | -424.18 | 0.09 | 4602 | 58 |
| vmf101 | stress | 4.043 | 1908.57 | -418.11 | 0.17 | 4602 | 58 |
| vmf105 | holdout | 4.055 | 1899.82 | -449.96 | 0.10 | 4557 | 58 |
| vmf105 | stress | 3.752 | 1674.80 | -442.22 | 0.19 | 4557 | 58 |
| vmf110 | holdout | 3.650 | 1362.97 | -369.32 | 0.12 | 4478 | 58 |
| vmf110 | stress | 3.306 | 1173.87 | -370.45 | 0.24 | 4478 | 58 |
| vmf115 | holdout | 3.893 | 1429.29 | -341.98 | 0.11 | 4442 | 58 |
| vmf115 | stress | 3.538 | 1237.71 | -365.46 | 0.23 | 4442 | 58 |
| vmf120 | holdout | 3.620 | 1284.60 | -342.32 | 0.12 | 4409 | 58 |
| vmf120 | stress | 3.268 | 1103.44 | -365.42 | 0.25 | 4409 | 58 |

### Итог по sprint10
- Лучший по `min(Sharpe_holdout, Sharpe_stress)` — `vmf101` (max_var_multiplier=1.01).
- Это новый глобальный лидер для `$1000` в extended OOS `2023-05-01 → 2024-04-30`: Sharpe `4.348/4.043`.

## Extra sweep: signal sprint11 (adaptive/regime/struct toggles, 10 прогонов)
- Очередь: `coint4/artifacts/wfa/aggregate/20260212_budget1000_sharpe_signal_sprint11/run_queue.csv`
- Конфиги: `coint4/configs/budget_20260212_1000_sharpe_signal_sprint11/*.yaml`
- Размер: 10 прогонов (`5` вариантов `at*` × `holdout/stress`)
- Статус: `10/10 completed`
- Валидация: `Sharpe consistency OK (10 run(s))`

### Матрица параметров (at*)
Проверяем, не “ломают” ли результат отключения `adaptive_thresholds`, `market_regime_detection`, `structural_break_protection`.

| variant | adaptive_thresholds | max_var_multiplier | market_regime_detection | structural_break_protection |
|---|---|---:|---|---|
| at0 | false | (n/a) | true | true |
| at0rg0 | false | (n/a) | false | true |
| at0sb0 | false | (n/a) | true | false |
| at0rg0sb0 | false | (n/a) | false | false |
| at1vm101 | true | 1.01 | true | true |

### Результаты (10 прогонов)
| variant | kind | sharpe | pnl | max_dd | cost_ratio | trades | pairs |
|---|---|---:|---:|---:|---:|---:|---:|
| at0 | holdout | 4.285 | 2108.25 | -467.05 | 0.09 | 4614 | 58 |
| at0 | stress | 3.981 | 1865.27 | -458.08 | 0.17 | 4614 | 58 |
| at0rg0 | holdout | 1.005 | 203.03 | -827.07 | 0.58 | 5395 | 58 |
| at0rg0 | stress | 0.734 | 58.21 | -823.04 | 3.42 | 5395 | 58 |
| at0rg0sb0 | holdout | 1.236 | 361.17 | -1167.57 | 0.43 | 5553 | 58 |
| at0rg0sb0 | stress | 1.070 | 202.07 | -1187.21 | 1.29 | 5553 | 58 |
| at0sb0 | holdout | 3.088 | 1219.54 | -420.08 | 0.13 | 5220 | 58 |
| at0sb0 | stress | 2.734 | 1008.32 | -401.43 | 0.26 | 5220 | 58 |
| at1vm101 | holdout | 4.348 | 2153.76 | -424.18 | 0.09 | 4602 | 58 |
| at1vm101 | stress | 4.043 | 1908.57 | -418.11 | 0.17 | 4602 | 58 |

### Итог по sprint11
- `market_regime_detection` выключать нельзя: Sharpe и DD разваливаются.
- `structural_break_protection` тоже помогает (при выключении Sharpe заметно падает).
- `adaptive_thresholds=false` близко к лидеру, но хуже по robust-метрике → оставляем `adaptive_thresholds=true` и `max_var_multiplier=1.01` как в `vmf101`.

## Extra sweep: signal sprint12 (z sweep under max_var_multiplier=1.01, 10 прогонов)
- Очередь: `coint4/artifacts/wfa/aggregate/20260212_budget1000_sharpe_signal_sprint12/run_queue.csv`
- Конфиги: `coint4/configs/budget_20260212_1000_sharpe_signal_sprint12/*.yaml`
- Размер: 10 прогонов (`5` вариантов `z1p10-z1p30` × `holdout/stress`)
- Статус: `10/10 completed`
- Валидация: `Sharpe consistency OK (10 run(s))`

### Матрица параметров (z1p10-z1p30)
Фиксируем `max_var_multiplier=1.01` (как в `vmf101`), меняем только `z`.

| variant | z |
|---|---:|
| z1p10 | 1.10 |
| z1p15 | 1.15 |
| z1p20 | 1.20 |
| z1p25 | 1.25 |
| z1p30 | 1.30 |

### Результаты (10 прогонов)
| variant | kind | sharpe | pnl | max_dd | cost_ratio | trades | pairs |
|---|---|---:|---:|---:|---:|---:|---:|
| z1p10 | holdout | 4.133 | 2070.76 | -508.90 | 0.09 | 4761 | 58 |
| z1p10 | stress | 3.826 | 1820.11 | -494.30 | 0.17 | 4761 | 58 |
| z1p15 | holdout | 4.348 | 2153.76 | -424.18 | 0.09 | 4602 | 58 |
| z1p15 | stress | 4.043 | 1908.57 | -418.11 | 0.17 | 4602 | 58 |
| z1p20 | holdout | 3.841 | 1714.90 | -409.58 | 0.10 | 4434 | 58 |
| z1p20 | stress | 3.539 | 1505.39 | -406.32 | 0.20 | 4434 | 58 |
| z1p25 | holdout | 2.655 | 692.96 | -443.52 | 0.20 | 4246 | 58 |
| z1p25 | stress | 2.258 | 555.15 | -442.97 | 0.42 | 4246 | 58 |
| z1p30 | holdout | 4.055 | 1161.65 | -364.34 | 0.12 | 4105 | 58 |
| z1p30 | stress | 3.655 | 1000.64 | -364.98 | 0.24 | 4105 | 58 |

### Итог по sprint12
- Лучший robust снова на `z=1.15` (`z1p15`), то есть оптимум `z` не сдвинулся после фикса `max_var_multiplier`.
- Текущий лидер: `vmf101` (`z=1.15`, `max_var_multiplier=1.01`).

## Extra sweep: signal sprint13 (exit sweep under vmf101, 10 прогонов)
- Очередь: `coint4/artifacts/wfa/aggregate/20260212_budget1000_sharpe_signal_sprint13/run_queue.csv`
- Конфиги: `coint4/configs/budget_20260212_1000_sharpe_signal_sprint13/*.yaml`
- Размер: 10 прогонов (`5` вариантов `ex04-ex12` × `holdout/stress`)
- Статус: `10/10 completed`
- Валидация: `Sharpe consistency OK (10 run(s))`

### Матрица параметров (ex04-ex12)
Фиксируем параметры лидера `vmf101` (`z=1.15`, `ms=0.20`, `max_var_multiplier=1.01`), меняем только `zscore_exit`.

| variant | zscore_exit |
|---|---:|
| ex04 | 0.04 |
| ex06 | 0.06 |
| ex08 | 0.08 |
| ex10 | 0.10 |
| ex12 | 0.12 |

### Результаты (10 прогонов)
| variant | kind | sharpe | pnl | max_dd | cost_ratio | trades | pairs |
|---|---|---:|---:|---:|---:|---:|---:|
| ex04 | holdout | 3.301 | 1207.93 | -302.88 | 0.13 | 4304 | 58 |
| ex04 | stress | 2.984 | 1036.96 | -308.72 | 0.26 | 4304 | 58 |
| ex06 | holdout | 4.112 | 1694.67 | -372.57 | 0.10 | 4482 | 58 |
| ex06 | stress | 3.786 | 1486.55 | -369.98 | 0.20 | 4482 | 58 |
| ex08 | holdout | 4.348 | 2153.76 | -424.18 | 0.09 | 4602 | 58 |
| ex08 | stress | 4.043 | 1908.57 | -418.11 | 0.17 | 4602 | 58 |
| ex10 | holdout | 3.665 | 1536.11 | -396.32 | 0.11 | 4673 | 58 |
| ex10 | stress | 3.344 | 1329.21 | -387.65 | 0.22 | 4673 | 58 |
| ex12 | holdout | 3.504 | 1428.31 | -405.03 | 0.12 | 4747 | 58 |
| ex12 | stress | 3.177 | 1226.00 | -395.80 | 0.25 | 4747 | 58 |

### Итог по sprint13
- Лучший robust снова на `exit=0.08` (`ex08`) и совпадает с лидером `vmf101`.
- Отклонение `exit` в обе стороны ухудшает Sharpe и/или DD для `$1000`.

## Extra sweep: signal sprint14 (stop_loss_zscore sweep under vmf101, 10 прогонов)
- Очередь: `coint4/artifacts/wfa/aggregate/20260212_budget1000_sharpe_signal_sprint14/run_queue.csv`
- Конфиги: `coint4/configs/budget_20260212_1000_sharpe_signal_sprint14/*.yaml`
- Размер: 10 прогонов (`5` вариантов `slz2p5-slz3p5` × `holdout/stress`)
- Статус: `10/10 completed`
- Валидация: `Sharpe consistency OK (10 run(s))`

### Матрица параметров (slz2p5-slz3p5)
Фиксируем параметры лидера `vmf101` (`z=1.15`, `exit=0.08`, `ms=0.20`, `max_var_multiplier=1.01`), меняем только `pair_stop_loss_zscore`.

| variant | pair_stop_loss_zscore |
|---|---:|
| slz2p5 | 2.50 |
| slz2p75 | 2.75 |
| slz3p0 | 3.00 |
| slz3p25 | 3.25 |
| slz3p5 | 3.50 |

### Результаты (10 прогонов)
| variant | kind | sharpe | pnl | max_dd | cost_ratio | trades | pairs |
|---|---|---:|---:|---:|---:|---:|---:|
| slz2p5 | holdout | 2.573 | 734.19 | -345.24 | 0.24 | 5100 | 58 |
| slz2p5 | stress | 2.148 | 569.81 | -349.24 | 0.52 | 5100 | 58 |
| slz2p75 | holdout | 3.506 | 1348.49 | -360.89 | 0.13 | 4825 | 58 |
| slz2p75 | stress | 3.157 | 1148.34 | -355.78 | 0.27 | 4825 | 58 |
| slz3p0 | holdout | 4.348 | 2153.76 | -424.18 | 0.09 | 4602 | 58 |
| slz3p0 | stress | 4.043 | 1908.57 | -418.11 | 0.17 | 4602 | 58 |
| slz3p25 | holdout | 1.784 | 581.54 | -513.04 | 0.24 | 4433 | 58 |
| slz3p25 | stress | 1.511 | 445.61 | -500.86 | 0.53 | 4433 | 58 |
| slz3p5 | holdout | 2.472 | 938.79 | -593.32 | 0.15 | 4302 | 58 |
| slz3p5 | stress | 2.199 | 782.65 | -580.37 | 0.32 | 4302 | 58 |

### Итог по sprint14
- Лучший robust снова на `pair_stop_loss_zscore=3.0` (`slz3p0`) и совпадает с лидером `vmf101`.
- Отклонение stop-loss в обе стороны ухудшает Sharpe (особенно >3.0 раздувает DD).
