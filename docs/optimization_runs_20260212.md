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
