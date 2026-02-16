# Optimization runs — cycle `20260216_clean_top10` (written 2026-02-15)

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
