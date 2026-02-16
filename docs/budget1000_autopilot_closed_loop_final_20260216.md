# Budget $1000 autopilot closed-loop: финальный winner

Generated at (UTC): 2026-02-16T08:06:56Z

## Контекст цикла

- Controller group: `20260216_budget1000_cl_autopilot`
- Конфиг: `coint4/configs/autopilot/budget1000_closed_loop_20260216.yaml`
- State: `coint4/artifacts/wfa/aggregate/20260216_budget1000_cl_autopilot/state.json`
- Политика отбора (DD-first): `min_windows=3`, `max_dd_pct=0.25`, `dd_target_pct=0.15`, `dd_penalty=8.0`

## Финальный winner

- run_group: `20260216_budget1000_cl_r02_risk`
- variant_id: `prod_final_budget1000_risk0p019_slusd6p5_slusd4p5_vm1p0035_risk0p015_slusd2p5_risk0p011_risk0p009`
- sample_config_path: `coint4/configs/budget1000_autopilot/20260216_budget1000_cl_r02_risk/holdout_prod_final_budget1000_risk0p019_slusd6p5_slusd4p5_vm1p0035_risk0p015_slusd2p5_risk0p011_oos20220601_20230430_risk0p009.yaml`
- score: `2.8563555378997316`
- worst_robust_sharpe: `3.269437519079036`
- worst_dd_pct: `0.20163524764741306`

Выбор зафиксирован по DD-first правилам контроллера: кандидат проходит `max_dd_pct<=0.25` и имеет лучший итоговый worst-window профиль среди зафиксированных best-снимков раундов (`r01=2.7431`, `r02=3.2694`, `r03=2.1832` по worst robust Sharpe в `state.history`).

## Stop condition

- `stop_reason`: `no_improvement_streak_reached: streak=1, rounds=1, min_improvement=0.02`
- Техническое объяснение остановки:
  - В `round_03` `best_before_round == best_after_round`, `delta_score=0.0`.
  - По search-конфигу `no_improvement_rounds=1`, `min_improvement=0.02`, поэтому цикл остановлен автоматически.
  - Локальный refine в `r03` проверил соседние значения `portfolio.risk_per_position_pct = [0.007, 0.009, 0.011]` и не дал улучшения относительно winner из `r02`.

## Сравнение с baseline цикла

Базой closed-loop принят стартовый `base_config` из `budget1000_closed_loop_20260216.yaml` (тот же DD-first winner из follow-up, зафиксированный в `docs/budget1000_autopilot_followup_final_20260216.md`).

| Metric | Baseline | Final winner | Delta (winner - baseline) |
|---|---:|---:|---:|
| score | `1.6467851707901258` | `2.8563555378997316` | `+1.2095703671096059` |
| worst-window robust Sharpe | `2.2885744023753896` | `3.269437519079036` | `+0.9808631167036466` |
| worst-window DD pct | `0.23022365394815797` | `0.20163524764741306` | `-0.028588406300744912` |

Итог closed-loop: относительно baseline цикла улучшены и worst-window robust Sharpe, и worst-window DD pct (DD ниже на `2.8588` п.п.), поэтому кандидат `..._risk0p009` зафиксирован финальным winner.
