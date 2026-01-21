# Журнал прогонов оптимизации (2026-01-29)

Назначение: risk‑sweep для топ‑30 (z1.00/exit0.06/hold180/cd180/ms0.2) — влияние `risk_per_position_pct` и `max_active_positions`.

## Очередь: realcost_riskgrid_top30
- Очередь: `coint4/artifacts/wfa/aggregate/20260129_realcost_riskgrid_top30/run_queue.csv` (8 прогонов).
- Конфиги:
  - `coint4/configs/holdout_20260129_relaxed8_nokpss_u250_churnfix_riskgrid/*.yaml` (4 шт.)
  - `coint4/configs/stress_20260129_relaxed8_nokpss_u250_churnfix_riskgrid/*.yaml` (4 шт.)
- Статус: `completed`.

### Результаты (holdout + stress)
| config | kind | sharpe | pnl | costs | trades | pairs | max_dd | cost_ratio |
|---|---|---|---|---|---|---|---|---|
| r0.015/maxpos15 | holdout | 8.79 | 1151.4 | 189.5 | 6865 | 75 | -60.7 | 0.16 |
| r0.015/maxpos10 | holdout | 8.26 | 988.2 | 189.5 | 6865 | 75 | -55.6 | 0.19 |
| r0.01/maxpos15 | holdout | 8.79 | 1151.4 | 189.5 | 6865 | 75 | -60.7 | 0.16 |
| r0.01/maxpos10 | holdout | 8.26 | 988.2 | 189.5 | 6865 | 75 | -55.6 | 0.19 |
| r0.015/maxpos15 | stress | 7.75 | 1015.8 | 337.0 | 6865 | 75 | -63.4 | 0.33 |
| r0.015/maxpos10 | stress | 7.40 | 884.5 | 337.0 | 6865 | 75 | -57.7 | 0.38 |
| r0.01/maxpos15 | stress | 7.75 | 1015.8 | 337.0 | 6865 | 75 | -63.4 | 0.33 |
| r0.01/maxpos10 | stress | 7.40 | 884.5 | 337.0 | 6865 | 75 | -57.7 | 0.38 |

## Выводы
- `risk_per_position_pct` не влияет на метрики (вероятно, упирается в `max_position_size_pct`).
- Снижение `max_active_positions` до 10 уменьшает PnL/Sharpe и немного снижает DD.
- Лидер остаётся r0.015/maxpos15 (базовый вариант).

## Итоговый shortlist
- Основной: top30 z1.00/exit0.06/hold180/cd180/ms0.2 (стандартный stress проходит, alt‑OOS подтверждён).
- Стресс‑fallback: top20 z1.00/exit0.06/hold180/cd180/ms0.2 (проходит stress+ `cost_ratio <= 0.5`).
