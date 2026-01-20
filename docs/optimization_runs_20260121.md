# Журнал прогонов оптимизации (2026-01-21)

Назначение: micro-grid вокруг базового u250 кандидата + max_pairs sweep, проверка Sharpe annualization.

## Обновления (2026-01-21)

### Sharpe annualization sanity
- WFA метрики считают Sharpe через `periods_per_year = annualizing_factor * (24*60/bar_minutes)`.
- Base engine приведён к динамическому periods_per_year по медианному шагу индекса (см. `coint4/src/coint2/engine/base_engine.py`).

### Queue: relaxed8_nokpss_u250_search (micro-grid + max_pairs)
- Очередь: `coint4/artifacts/wfa/aggregate/20260121_relaxed8_nokpss_u250_search/run_queue.csv`.
- Цель: найти локальные улучшения вокруг базового z0.95/0.08/120/120 и проверить компромисс Sharpe/PnL/turnover через `max_pairs`.
- Параллельность: `1` (каждый прогон использует `backtest.n_jobs: -1`).
- Конфиги:
  - `coint4/configs/holdout_20260121_relaxed8_nokpss_u250_search/*.yaml` (9 шт.)
  - `coint4/configs/stress_20260121_relaxed8_nokpss_u250_search/*.yaml` (9 шт.)
- Грид параметров:
  - entry: `0.90`, `0.95`, `1.00`
  - exit: `0.06`, `0.08`, `0.10`
  - hold/cooldown: `90`, `120`, `180`
  - max_pairs: `50/100/150` (baseline z0.95/0.08/120/120)
- Статус: `planned` (18 прогонов: 9 holdout + 9 stress).

### Очереди статусов
- `coint4/artifacts/wfa/aggregate/20260119_relaxed8_nokpss_u250_turnover_stress/run_queue.csv` переведена в `completed` (соответствующие результаты уже в `coint4/artifacts/wfa/runs/20260120_relaxed8_nokpss_u250_turnover_full`).
