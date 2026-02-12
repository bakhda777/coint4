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

## Критерии отбора после завершения
- Primary: максимизировать `min(Sharpe_holdout, Sharpe_stress)`.
- Гейты: `total_trades >= 500`, `total_pairs_traded >= 50`, `max_drawdown_abs >= -250`.
- Cost gate: `stress cost_ratio <= 0.5` (при положительном `total_pnl`).
