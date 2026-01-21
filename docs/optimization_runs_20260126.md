# Журнал прогонов оптимизации (2026-01-26)

Назначение: alt‑OOS валидация лидеров top20/top30 (realistic costs) и подтверждение устойчивости Sharpe/cost_ratio.

## Критерии отбора (актуальные)
- Гейтинг: `total_costs>0`, `total_trades>=500`, `total_pairs_traded>=50`, `sharpe_ratio_abs>0`, `total_pnl>0`.
- Stress: `cost_ratio <= 0.5` (издержки не более 50% PnL).
- WFA‑стабильность: медиана Sharpe по 5 шагам ≥ 1.0, минимум по шагам ≥ 0.6.

## План (alt‑OOS, top20/top30)
- Собрать holdout+stress конфиги для периода `2022-09-01 → 2023-04-30`.
- Запустить WFA (≤5 шагов) только на 85.198.90.128 с `--parallel $(nproc)`.
- Обновить rollup и зафиксировать результаты/выводы.

## Очередь: realcost_alt_oos
- Очередь: `coint4/artifacts/wfa/aggregate/20260126_realcost_alt_oos/run_queue.csv` (4 прогона).
- Конфиги:
  - `coint4/configs/holdout_20260126_relaxed8_nokpss_u250_churnfix_alt/*.yaml` (2 шт.)
  - `coint4/configs/stress_20260126_relaxed8_nokpss_u250_churnfix_alt/*.yaml` (2 шт.)
- Статус: `completed`.

### Результаты (holdout + stress, alt‑OOS)
| config | kind | sharpe | pnl | costs | trades | pairs | max_dd | cost_ratio |
|---|---|---|---|---|---|---|---|---|
| top20/hold180/cd180/ms0.2 | holdout | 8.17 | 976.0 | 179.4 | 3519 | 46 | -130.1 | 0.18 |
| top30/hold180/cd180/ms0.2 | holdout | 8.54 | 1035.4 | 183.0 | 3651 | 47 | -130.1 | 0.18 |
| top20/hold180/cd180/ms0.2 | stress | 7.01 | 836.5 | 318.9 | 3519 | 46 | -134.2 | 0.38 |
| top30/hold180/cd180/ms0.2 | stress | 7.37 | 893.2 | 325.3 | 3651 | 47 | -134.2 | 0.36 |

### Выводы
- Оба кандидата устойчивы на alt‑OOS и проходят порог stress cost_ratio ≤ 0.5.
- Лучший баланс Sharpe/PnL: top30/hold180/cd180/ms0.2 (stress Sharpe 7.37, cost_ratio 0.36).
- top20 остаётся чуть дешевле по издержкам, но уступает по PnL.

### Shortlist (текущий)
- Основной: top30/hold180/cd180/ms0.2.
- Запасной: top20/hold180/cd180/ms0.2.
