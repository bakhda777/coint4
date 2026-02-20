# Optimization runs — 2026-02-19 (baseline + final фиксация winner для live)

Контекст: baseline для PRD07 US-LOOP-002 + итоговая фиксация результата после US-LOOP-003 fail-closed stop.

## Rebuild rollup (including `runs_clean`)

Команда (из `coint4/`):

`PYTHONPATH=src ./.venv/bin/python scripts/optimization/build_run_index.py --output-dir artifacts/wfa/aggregate/rollup`

Результат:
- `Run index entries: 1800`.
- Проверка покрытия `runs_clean` в `run_index.csv`: `20` строк, `20/20 metrics_present=true`, `20/20 status=completed`.

## Baseline snapshot: top robust candidates

Команда (из `coint4/`):

`PYTHONPATH=src ./.venv/bin/python scripts/optimization/rank_multiwindow_robust_runs.py --top 10 --max-dd-pct 0.14 --min-windows 3 --min-trades 200 --min-pairs 20 --contains budget1000 --include-noncompleted`

Результат (top-10):

| rank | worst_robust_sh | avg_robust_sh | worst_dd_pct | avg_dd_pct | windows | run_group | variant_id |
|---:|---:|---:|---:|---:|---:|---|---|
| 1 | 3.530 | 4.388 | 0.132 | 0.089 | 3 | 20260213_budget1000_dd_sprint08_stoplossusd_micro | prod_final_budget1000_risk0p006_slusd1p91 |
| 2 | 3.530 | 4.388 | 0.132 | 0.089 | 3 | 20260214_budget1000_dd_sprint09_hurst_slusd1p91 | prod_final_budget1000_risk0p006_slusd1p91_max_hurst_exponent0p8 |
| 3 | 3.470 | 4.357 | 0.128 | 0.088 | 3 | 20260213_budget1000_dd_sprint08_stoplossusd_micro | prod_final_budget1000_risk0p006_slusd1p88 |
| 4 | 3.448 | 4.306 | 0.132 | 0.089 | 3 | 20260213_budget1000_dd_sprint06_stoplossusd_fine | pruned168_slusd1p85_risk0p006 |
| 5 | 3.448 | 4.306 | 0.132 | 0.089 | 3 | 20260213_budget1000_dd_sprint08_stoplossusd_micro | prod_final_budget1000_risk0p006_slusd1p85 |
| 6 | 3.323 | 4.254 | 0.137 | 0.087 | 3 | 20260213_budget1000_dd_sprint05_stoplossusd_refine | pruned168_slusd1p75_risk0p006 |
| 7 | 3.323 | 4.254 | 0.137 | 0.087 | 3 | 20260213_budget1000_dd_sprint06_stoplossusd_fine | pruned168_slusd1p75_risk0p006 |
| 8 | 3.299 | 4.165 | 0.138 | 0.091 | 3 | 20260213_budget1000_dd_sprint06_stoplossusd_fine | pruned168_slusd1p8_risk0p006 |
| 9 | 3.179 | 4.200 | 0.137 | 0.086 | 3 | 20260213_budget1000_dd_sprint06_stoplossusd_fine | pruned168_slusd1p65_risk0p006 |
| 10 | 2.864 | 4.115 | 0.108 | 0.073 | 3 | 20260213_budget1000_dd_sprint05_stoplossusd_refine | pruned168_slusd1p5_risk0p006 |

Baseline freeze для следующего loop:
- Baseline score (worst robust Sharpe): `3.530254`.
- Baseline worst_dd_pct: `0.132205`.
- Baseline run_group: `20260213_budget1000_dd_sprint08_stoplossusd_micro`.
- Baseline run_id (worst-window holdout): `holdout_prod_final_budget1000_oos20240501_20250630_risk0p006_slusd1p91`.
- Baseline paired stress run_id (worst-window): `stress_prod_final_budget1000_oos20240501_20250630_risk0p006_slusd1p91`.
- Baseline variant_id: `prod_final_budget1000_risk0p006_slusd1p91`.
- Gate check: `worst_dd_pct=0.132205 <= 0.14`, `windows=3`, `min_trades>=200`, `min_pairs>=20`.

## US-LOOP-003: исполненные run_group раунды

Срез по `run_index.csv` и очередям `BL11`:

| run_group | queue status | run_index rows | completed | metrics_present=true | комментарий |
|---|---|---:|---:|---:|---|
| `20260219_budget1000_bl11_r08_queuefix01_grid10x3` | `60 completed` | 60 | 60 | 0 | Артефакты есть, но метрики в rollup не появились, кандидат не прошёл в robust ranking |
| `20260219_budget1000_bl11_r09_pairgate02_micro24` | `24 planned` | 24 | 0 | 0 | Новый remote batch не выполнен из-за infra-блокера (sandbox outbound/DNS + SSH) |

## US-LOOP-003 stop (fail-closed) и финальный winner

LLM stop:
- `decision_id`: `us-loop-003-stop-20260220T0149Z-infra-block`
- `next_action`: `stop`
- `stop_reason`: `INFRA_BLOCKED_SANDBOX_NETWORK: codex backend and serverspace api unreachable; powered runner repeats RC4`

Финальный winner для live:
- `run_group`: `20260213_budget1000_dd_sprint08_stoplossusd_micro`
- `run_id` (worst-window holdout): `holdout_prod_final_budget1000_oos20240501_20250630_risk0p006_slusd1p91`
- paired stress run_id: `stress_prod_final_budget1000_oos20240501_20250630_risk0p006_slusd1p91`
- `variant_id`: `prod_final_budget1000_risk0p006_slusd1p91`
- `score` (`worst_robust_sharpe`): `3.530254`
- `worst_dd_pct`: `0.132205`
- `sample_config`: `configs/budget_20260213_1000_dd_sprint08_stoplossusd_micro/holdout_prod_final_budget1000_oos20220601_20230430_risk0p006_slusd1p91.yaml`

Почему выигрывает:
- Это top-1 по каноническому objective (`worst_robust_sharpe`) при полном прохождении gate (`max_dd_pct<=0.14`, `min_windows=3`, `min_trades>=200`, `min_pairs>=20`).
- У `20260214_budget1000_dd_sprint09_hurst_slusd1p91` tie по robust-метрике, но для continuity закреплён baseline sprint08.

## Экспорт winner в prod-candidate для live

- Создан экспортный файл: `coint4/configs/prod_final_budget1000_bestparams_20260219.yaml`.
- Источник экспорта: `docs/best_params_latest.yaml` (winner snapshot, без секретов).

## Guardrail по артефактам

- Для commit проверять, что тяжёлые артефакты не staged: `coint4/artifacts/wfa/runs/**` остаются вне индекса.
