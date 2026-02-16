# Budget $1000 autopilot follow-up: итог post-sync_back

Generated at (UTC): 2026-02-16T07:36:00Z

## Контекст

- Follow-up конфиг: `coint4/configs/autopilot/budget1000_followup_20260216.yaml`
- Follow-up очередь: `coint4/artifacts/wfa/aggregate/20260216_budget1000_ap2_r01_risk/run_queue.csv`
- Controller state: `coint4/artifacts/wfa/aggregate/20260216_budget1000_ap2_autopilot/state.json`
- Политика отбора (DD-first): `min_windows=3`, `max_dd_pct=0.25`, `dd_target_pct=0.15`, `dd_penalty=8.0`

## Постпроцесс после sync_back

Выполнено из `coint4/`:

```bash
PYTHONPATH=src ./.venv/bin/python scripts/optimization/postprocess_queue.py \
  --queue artifacts/wfa/aggregate/20260216_budget1000_ap2_r01_risk/run_queue.csv \
  --bar-minutes 15 \
  --overwrite-canonical \
  --build-rollup \
  --print-rank-multiwindow \
  --rank-contains 20260216_budget1000_ap2
```

Результат:
- Rollup пересобран: `coint4/artifacts/wfa/aggregate/rollup/run_index.csv` (entries=`1890`).
- Для `20260216_budget1000_ap2_r01_risk`: `planned=30`, `metrics_present=False` для всех 30.
- Ранжирование по `20260216_budget1000_ap2` не дало кандидатов (`No variants matched`).

## Best-кандидат (DD-first фиксация для follow-up)

Внутри текущего follow-up run_group (`20260216_budget1000_ap2_*`) best-кандидат пока отсутствует: completed-метрик нет.

Для продолжения цикла зафиксирован безопасный fallback-кандидат из последней завершённой autopilot-семьи (`budget1000_ap_r*`) под теми же DD-first ограничениями:

- run_group: `20260215_budget1000_ap_r03_slusd`
- variant_id: `prod_final_budget1000_risk0p019_slusd6p5_slusd4p5_vm1p0035_risk0p015_slusd2p5`
- windows: `3`
- score: `1.6467851707901258`
- worst-window robust Sharpe: `2.2885744023753896`
- worst-window DD pct: `0.23022365394815797`
- sample_config_path: `coint4/configs/budget1000_autopilot/20260215_budget1000_ap_r03_slusd/holdout_prod_final_budget1000_oos20220601_20230430_risk0p019_oos20220601_20230430_slusd6p5_oos20220601_20230430_slusd4p5_oos20220601_20230430_vm1p0035_oos20220601_20230430_risk0p015_oos20220601_20230430_slusd2p5.yaml`

Гейт `max_dd_pct=0.25` для fallback-кандидата проходит (`0.2302 <= 0.25`).

## Следующий шаг

- Выполнить heavy для `20260216_budget1000_ap2_r01_risk` на VPS (`run_server_job.sh`, `STOP_AFTER=1`), затем снова запустить `postprocess_queue.py` + DD-first ранжирование уже по `20260216_budget1000_ap2`.
