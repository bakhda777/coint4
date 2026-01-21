# Журнал прогонов оптимизации (2026-01-31)

Назначение: extended OOS (2023-05-01 -> 2024-04-30) turnover-grid для снижения издержек (max_pairs 10/15, ms 0.25/0.30, hold/cd 240).

## Критерии отбора (актуальные)
- Гейтинг: `total_costs>0`, `total_trades>=500`, `total_pairs_traded>=50`, `sharpe_ratio_abs>0`, `total_pnl>0`.
- Stress: `cost_ratio <= 0.5`.
- WFA-стабильность: медиана Sharpe по 5 шагам >= 1.0, минимум по шагам >= 0.6.

## Очередь: realcost_oos20230501_20240430_turnover
- Очередь: `coint4/artifacts/wfa/aggregate/20260131_realcost_oos20230501_20240430_turnover/run_queue.csv` (8 прогонов).
- Конфиги:
  - `coint4/configs/holdout_20260131_relaxed8_nokpss_u250_churnfix_turnover_oos20230501_20240430/*.yaml` (4 шт.)
  - `coint4/configs/stress_20260131_relaxed8_nokpss_u250_churnfix_turnover_oos20230501_20240430/*.yaml` (4 шт.)
- Статус: `completed`.

## Результаты (holdout + stress)
| config | kind | sharpe | pnl | costs | trades | pairs | max_dd | cost_ratio |
|---|---|---|---|---|---|---|---|---|
| top10 ms0.25 | holdout | -0.69 | -71.4 | 110.6 | 2264 | 32 | -265.5 | -1.55 |
| top10 ms0.25 | stress | -1.54 | -157.4 | 196.5 | 2264 | 32 | -299.6 | -1.25 |
| top10 ms0.30 | holdout | -0.61 | -63.2 | 110.6 | 2263 | 32 | -264.5 | -1.75 |
| top10 ms0.30 | stress | -1.45 | -149.2 | 196.6 | 2263 | 32 | -298.7 | -1.32 |
| top15 ms0.25 | holdout | 1.18 | 151.4 | 154.9 | 3264 | 47 | -206.3 | 1.02 |
| top15 ms0.25 | stress | 0.25 | 31.0 | 275.4 | 3264 | 47 | -243.9 | 8.89 |
| top15 ms0.30 | holdout | 1.31 | 168.8 | 154.9 | 3263 | 47 | -203.6 | 0.92 |
| top15 ms0.30 | stress | 0.39 | 48.3 | 275.4 | 3263 | 47 | -241.2 | 5.70 |

## Выводы
- Top10 провалился: отрицательный Sharpe и PnL, пары 32 (<50).
- Top15 улучшает Sharpe в holdout, но не проходит гейт по парам (47 < 50) и стресс по издержкам (cost_ratio >> 0.5).
- Усиление hold/cooldown + ms не решило проблему extended OOS: стрессовые издержки доминируют PnL.

## Рекомендации
- Зафиксировать stop-condition по extended OOS: дальнейшая оптимизация через turnover-grid не дает приемлемого stress cost_ratio.
- Рассмотреть смену режима: либо более жесткая фильтрация пар/ликвидности, либо переход к paper/forward тесту базового конфига с оговорками по рискам.
