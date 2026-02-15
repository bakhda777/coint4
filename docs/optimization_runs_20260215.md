# Журнал прогонов оптимизации (2026-02-15)

Важно: сегодня тяжёлые прогоны не запускались (на этом сервере их не запускаем). Обновление касается только подготовки и фиксации clean-цикла TOP-10.

## Clean Cycle TOP-10 (prep): цель и источники истины

- Мотивация: убрать двусмысленность "старой партии" (`coint4/artifacts/wfa/runs/**`) и сравнивать baseline/sweeps только в clean-контуре по `canonical_metrics.json`, пересчитанным из `equity_curve.csv`.
- Док процесса/guardrails: `docs/clean_cycle_top10.md`.
- Константы цикла: `coint4/scripts/optimization/clean_cycle_top10/definitions.py` (`CYCLE_NAME=20260215_clean_top10`).
- FIXED_WINDOWS (инвариант clean-цикла, одинаковый для baseline и всех sweeps):
  - `walk_forward.start_date=2024-05-01`
  - `walk_forward.end_date=2024-12-31`
  - `training_period_days=90`, `testing_period_days=30`, `step_size_days=30`, `max_steps=5`
  - `gap_minutes=15`, `refit_frequency=weekly`
- Seed (legacy rollup input): `coint4/artifacts/wfa/aggregate/rollup/run_index.csv` (sha256=`67b90c09be7d247b2d38702ce3900a0ae08d8d94dc7a0255613160bf7dde492b`).

## Seed TOP-10 (зафиксировано здесь, т.к. baseline_manifest gitignored)

`select_top10.py` выбирает TOP-N *runs* из `run_index.csv` (без дедупликации по `config_sha256`), поэтому `baseline_manifest.json` может содержать дубли по конфигам. Текущий seed: 10 строк, 5 уникальных конфигов.

Таблица уникальных seed-конфигов (по `config_sha256`):

| n | best_rank | rank_sharpe | max_dd | pnl | config_sha256 | config_file |
|---:|---:|---:|---:|---:|---|---|
| 2 | 1 | 9.091 | -82.2 | 1135.3 | `dcd2f22850467386cdf7ba866463bf95a86c065e110de58d9335218368b257a2` | `holdout_relaxed8_nokpss_20260123_pv0p2_hurst0p8_hl0p02_60_corr0p4_ssd50000_kpss1p0_w2m1_t90_fixed_u250_top50_z1p00_exit0p06_hold180_cd180_ms0p2.yaml` |
| 2 | 3 | 9.014 | -82.1 | 1114.7 | `b0a8417e40a9da1bf6e7201a8e93d375d61081fceafaab6e93eb774cb579d646` | `holdout_relaxed8_nokpss_20260122_pv0p2_hurst0p8_hl0p02_60_corr0p4_ssd50000_kpss1p0_w2m1_t90_fixed_u250_top50_z1p00_exit0p06_hold180_cd180_ms0p1.yaml` |
| 2 | 5 | 8.950 | -80.3 | 1107.5 | `72cdec7b27c100722e0df13aa89346ce508a3b7841e689be3b852ed38cae77bf` | `holdout_relaxed8_nokpss_20260123_pv0p2_hurst0p8_hl0p02_60_corr0p4_ssd50000_kpss1p0_w2m1_t90_fixed_u250_top50_z1p00_exit0p06_hold180_cd180_ms0p15.yaml` |
| 3 | 7 | 8.792 | -60.7 | 1151.4 | `e2c2a21c0f95432d0ed0bb6ff9140882fd9610a518a76acb8a2f8a2d04e51925` | `holdout_relaxed8_nokpss_20260125_pv0p2_hurst0p8_hl0p02_60_corr0p4_ssd50000_kpss1p0_w2m1_t90_fixed_u250_top30_z1p00_exit0p06_hold180_cd180_ms0p2.yaml` |
| 1 | 10 | 8.792 | -60.7 | 1151.4 | `b3ea7cdaab7071ba50f5e260ab85069a53dc7c230f43594af4708d7f2e9957c7` | `holdout_relaxed8_nokpss_20260129_pv0p2_hurst0p8_hl0p02_60_corr0p4_ssd50000_kpss1p0_w2m1_t90_fixed_u250_top30_z1p00_exit0p06_hold180_cd180_ms0p2_r0p01_maxpos15.yaml` |

## Следующие шаги (для выполнения на VPS, не на этом сервере)

1. Локально (из `coint4/`): snapshot/select/validate/init + baseline configs + baseline queue (см. `docs/clean_cycle_top10.md`).
2. На VPS `85.198.90.128`: прогон baseline queue через `coint4/scripts/remote/run_server_job.sh`, затем `recompute_canonical_metrics.py` и `build_clean_rollup.py` для `rollup_clean_cycle_top10.*`.
3. Далее: sweeps от победителя baseline (внутри clean-цикла, с теми же `FIXED_WINDOWS`).
