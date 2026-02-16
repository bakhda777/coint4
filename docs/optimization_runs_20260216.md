# Optimization runs — cycle `20260216_clean_top10` (written 2026-02-16)

Контекст: baseline batch clean-cycle TOP-10 (`20260216_clean_top10`) был выполнен на VPS и затем `sync_back` на эту машину. Ниже — локальная пост-обработка (без тяжёлых прогонов).

## Baseline post-processing (локально, после sync_back)

Пути:
- Очередь: `coint4/artifacts/wfa/aggregate/clean_cycle_top10/20260216_clean_top10/baseline_run_queue.csv`
- Baseline results: `coint4/artifacts/wfa/runs_clean/20260216_clean_top10/baseline_top10`

Шаги:
- Canonical metrics пересчитаны из `equity_curve.csv` в `canonical_metrics.json` для baseline results_dir (10/10 OK):
  - `PYTHONPATH=src ./.venv/bin/python scripts/optimization/recompute_canonical_metrics.py --bar-minutes 15 --run-dir ...`
  - Примечание: в baseline `equity_curve.csv` содержит только 1 точку (нет timestamp deltas), поэтому bar timeframe задан явно.
- Статусы очереди baseline: уже `10/10 completed` (sync не требовался).
- Baseline "заморожен":
  - создан sentinel `coint4/artifacts/wfa/runs_clean/20260216_clean_top10/baseline_top10/BASELINE_FROZEN.txt`
  - входной manifest для freeze: `coint4/scripts/optimization/clean_cycle_top10/inputs/20260216_clean_top10/baseline_manifest.json`
  - `verify_baseline_frozen.py`: OK; WARN только про дополнительные `walk_forward.*` ключи в baseline-конфигах (`enabled/min_training_samples/pairs_file`), не влияющие на fingerprint `FIXED_WINDOWS`.
- Построен baseline-only rollup (10 строк):
  - `PYTHONPATH=src ./.venv/bin/python scripts/optimization/clean_cycle_top10/build_clean_rollup.py --baseline-manifest artifacts/wfa/aggregate/clean_cycle_top10/20260216_clean_top10/baseline_run_queue.csv --allow-baseline-only ...`
  - outputs: `coint4/artifacts/wfa/aggregate/clean_cycle_top10/20260216_clean_top10/rollup_clean_cycle_top10.(csv|md)`
- Raw vs canonical diff: `missing_raw=0`, `missing_canonical=0`, `over_threshold=0` (selected=10, `compare_metrics.py`).

Наблюдение (важно): все baseline метрики получились нулевыми (0 сделок; `strategy_metrics.csv` = 0 и `equity_curve.csv` состоит из стартовой точки). Перед sweeps нужно убедиться, что baseline batch на VPS действительно отработал корректно (а не завершился ранним no-op).

## Budget1000 autopilot (VPS WFA -> postprocess -> selection)

Команда (из repo root):
- `cd coint4 && PYTHONPATH=src ./.venv/bin/python scripts/optimization/autopilot_budget1000.py --config configs/autopilot/budget1000.yaml --reset`

Итог:
- Отчёт: `docs/budget1000_autopilot_final_20260216.md` (завершение по `max_rounds=3`).
- Best candidate (из отчёта): `risk_per_position_pct=0.015`, `pair_stop_loss_usd=4.5`, `max_var_multiplier=1.0035`.
- Очереди: `coint4/artifacts/wfa/aggregate/20260215_budget1000_ap_r{01,02,03}_{risk,slusd,vm}/run_queue.csv` (9 очередей, по 30 runs каждая).
- Каждый remote job запускался через `run_server_job.sh` с `STOP_AFTER=1` (VPS выключался после выполнения).

## Budget1000 autopilot follow-up (APF-03: post-sync_back rollup + DD-first фиксация)

Команда постпроцесса (из `coint4/`):
- `PYTHONPATH=src ./.venv/bin/python scripts/optimization/postprocess_queue.py --queue artifacts/wfa/aggregate/20260216_budget1000_ap2_r01_risk/run_queue.csv --bar-minutes 15 --overwrite-canonical --build-rollup --print-rank-multiwindow --rank-contains 20260216_budget1000_ap2`

Факт после sync_back:
- Rollup пересобран (`run_index.csv`, entries=1890).
- Очередь `20260216_budget1000_ap2_r01_risk` осталась в состоянии `planned=30` (`metrics_present=False` для всех строк) — completed-run'ов нет.
- Прямое ранжирование по `20260216_budget1000_ap2` вернуло `No variants matched`.

DD-first фиксация best-кандидата для продолжения follow-up (fallback из завершённых `budget1000_ap_r*`):
- run_group: `20260215_budget1000_ap_r03_slusd`
- variant_id: `prod_final_budget1000_risk0p019_slusd6p5_slusd4p5_vm1p0035_risk0p015_slusd2p5`
- score: `1.646785`
- worst_robust_sharpe: `2.288574`
- worst_dd_pct: `0.230224` (DD gate `<=0.25` проходит)
- sample_config_path: `coint4/configs/budget1000_autopilot/20260215_budget1000_ap_r03_slusd/holdout_prod_final_budget1000_oos20220601_20230430_risk0p019_oos20220601_20230430_slusd6p5_oos20220601_20230430_slusd4p5_oos20220601_20230430_vm1p0035_oos20220601_20230430_risk0p015_oos20220601_20230430_slusd2p5.yaml`

Новый итоговый файл follow-up:
- `docs/budget1000_autopilot_followup_final_20260216.md`

## Сравнение циклов: `20260215_budget1000_ap_autopilot` vs `20260216_budget1000_ap2_autopilot`

| Метрика | Цикл `20260215_budget1000_ap_autopilot` (`max_rounds=3`) | Follow-up `20260216_budget1000_ap2_autopilot` |
| --- | --- | --- |
| Итоговый отчёт | `docs/budget1000_autopilot_final_20260216.md` | `docs/budget1000_autopilot_followup_final_20260216.md` |
| Best source | `20260215_budget1000_ap_r03_risk` | fallback `20260215_budget1000_ap_r03_slusd` |
| Score | `2.357188` | `1.646785` |
| Worst-window robust Sharpe | `3.310223` | `2.288574` |
| Worst-window DD pct | `0.340607` (DD gate `<=0.25` FAIL) | `0.230224` (DD gate `<=0.25` PASS) |
| Состояние follow-up run_group `20260216_budget1000_ap2_r01_risk` | n/a | `planned=30`, `completed=0`, `metrics_present=False` |

Итог сравнения:
- DD-first follow-up улучшил worst DD на `11.04` п.п. относительно max_rounds-цикла (`0.340607 -> 0.230224`) и дал прохождение DD-gate `<=0.25`.
- Цена улучшения DD: `score` ниже на `0.710403`, worst-window robust Sharpe ниже на `1.021648`.

## Budget1000 closed-loop autopilot (resume до done=true)

Команда (из `coint4/`):
- `PYTHONPATH=src ./.venv/bin/python scripts/optimization/autopilot_budget1000.py --config configs/autopilot/budget1000_closed_loop_20260216.yaml --resume`

Что выполнено:
- Heavy шаги шли только через `scripts/remote/run_server_job.sh` на `85.198.90.128`.
- Выполнены очереди: `20260216_budget1000_cl_r01_risk`, `20260216_budget1000_cl_r02_risk`, `20260216_budget1000_cl_r03_risk`.
- После каждого remote job VPS выключался автоматически (`STOP_AFTER=1`, API shutdown).
- Локальный postprocess/rollup выполнен автопилотом после каждого раунда; финальный `run_index` пересобран до `entries=1932`.

Итог контроллера:
- State: `coint4/artifacts/wfa/aggregate/20260216_budget1000_cl_autopilot/state.json`
- `done=true`
- `stop_reason=no_improvement_streak_reached: streak=1, rounds=1, min_improvement=0.02`
- Финальный отчёт: `docs/budget1000_autopilot_final_20260216.md`

Лучший кандидат closed-loop:
- run_group: `20260216_budget1000_cl_r02_risk`
- variant_id: `prod_final_budget1000_risk0p019_slusd6p5_slusd4p5_vm1p0035_risk0p015_slusd2p5_risk0p011_risk0p009`
- score: `2.856356`
- worst_robust_sharpe: `3.269438`
- worst_dd_pct: `0.201635`
- sample_config_path: `coint4/configs/budget1000_autopilot/20260216_budget1000_cl_r02_risk/holdout_prod_final_budget1000_risk0p019_slusd6p5_slusd4p5_vm1p0035_risk0p015_slusd2p5_risk0p011_oos20220601_20230430_risk0p009.yaml`
