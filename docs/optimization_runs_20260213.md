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

