# Optimization runs — 2026-02-22 (FS-008 workflow + DoD alignment)

Контекст: в этой итерации новых heavy-прогонов не запускалось; цель записи — зафиксировать единый рабочий цикл fullspan v1, обязательный postprocess и решения по tail-risk на базе последнего confirm/fullspan блока.

## Execution record (reference block)

- run_group (shortlist/confirm): `20260220_confirm_top10_bl11`
- queue_path (shortlist/confirm): `coint4/artifacts/wfa/aggregate/20260220_confirm_top10_bl11/run_queue.csv`
- run_group (fullspan replay): `20260220_top3_fullspan_wfa`
- queue_path (fullspan replay): `coint4/artifacts/wfa/aggregate/20260220_top3_fullspan_wfa/run_queue.csv`
- host: `85.198.90.128` (через `coint4/scripts/remote/run_server_job.sh`, `RUN_HOST=coint`)
- ALLOW_HEAVY_RUN: `n/a` (запуск через remote helper)
- result: `success` (`60/60 completed` confirm; `6/6 completed` fullspan)

## Before -> After (fullspan case)

Before (short OOS confirm ranking, top-3 from `20260220_confirm_top10_bl11`):

| rank | variant (short) | worst_robust_sh | worst_dd_pct | verdict in short OOS |
|---:|---|---:|---:|---|
| 1 | `..._slusd1p81` | 4.505 | 0.085 | shortlist PASS |
| 2 | `..._max_pairs24p0` | 4.366 | 0.112 | shortlist PASS |
| 3 | `..._pv0p365` | 4.311 | 0.090 | shortlist PASS |

After (fullspan replay `20260220_top3_fullspan_wfa`, policy `fullspan_v1`, strict command из postprocess-блока):

- Result: `No variants matched fullspan policy v1 (missing_tail=0, worst_step_gate_failed=1)`.

Diagnostic snapshot (для объяснения причин reject по каждому кандидату; значения `worst_step_pnl` получены в introspection-прогоне с ослабленным tail gate):

| rank | variant (short) | holdout_sh | stress_sh | robust_pnl | worst_dd_pct | worst_step_pnl | verdict in fullspan |
|---:|---|---:|---:|---:|---:|---:|---|
| 1 | `..._slusd1p81` | 1.463 | 1.088 | 1500.25 | 0.393 | -209.15 | REJECT (`worst_step_pnl < -200`) |
| 2 | `..._pv0p365` | 1.367 | -0.342 | -210.92 | 0.456 | -190.05 | REJECT (`worst_robust_pnl < 0`) |
| 3 | `..._max_pairs24p0` | 1.478 | -0.346 | -215.28 | 0.423 | -255.48 | REJECT (`worst_robust_pnl < 0`, `worst_step_pnl < -200`) |

## Tail-risk decisions (фиксировано)

- `selection_policy=fullspan_v1`
- `selection_mode=shortlist_only` (strict fullspan-v1 verdict: no-pass).
- Policy decision (2026-02-22): для final promote сохраняется строгий hard-gate `tail_worst_gate_pct=0.20`; режим `0.21` разрешён только как research/diagnostic.
- Baseline lock (2026-02-22): live winner остаётся baseline из `20260213_budget1000_dd_sprint08_stoplossusd_micro` до отдельного strict-pass по `fullspan_v1`.
- `promotion_verdict=reject` для `..._slusd1p81` (`worst_step_pnl=-209.15 < -200` hard gate).
- `promotion_verdict=reject` для `..._pv0p365` и `..._max_pairs24p0` (economic gate fail: `worst_robust_pnl < 0`).
- `promotion_verdict=shortlist_only` для short OOS top-10 без отдельного fullspan-pass.
- Временная настройка controller-профиля: `coint4/configs/autopilot/budget1000_batch_loop_bridge11_20260217.yaml` -> `selection.min_dsr: null` (до согласованной нормализации DSR-шкалы).
- Условия возврата DSR-gate: канонизировать шкалу DSR + добавить регрессионный тест ранжирования + подтвердить на отдельном confirm/fullspan-блоке без ложных reject.
- Прямой запрет сохраняется: winner нельзя выбирать только по `avg_robust_sharpe`/short OOS.

## VPS verify/confirm/fetch (2026-02-22 UTC)

- Remote helper: `coint4/scripts/remote/run_server_job.sh`, host `85.198.90.128`, `STOP_AFTER=1`.
- Smoke check: `RUN_HOST=coint`, после завершения SSH недоступен (shutdown подтверждён).
- Verify #1 (`SYNC_UP=0`, `make lint && make test`): `lint` passed, `test` failed (20 fail; серверная копия кода отставала от локальной ревизии).
- Verify #2 (`SYNC_UP=1`, `SYNC_UP_MODE=code`): устранён drift по tracked-файлам; полный `make test` всё ещё не зелёный в текущем рабочем дереве (31 import errors на этапе collection, ключевой индикатор — отсутствовавший ранее `coint2.pipeline.cost_model` был добавлен в tracked и доставлен на VPS).
- Verify #3 (targeted, `tests/utils/test_run_index.py` + `tests/scripts/test_build_run_index_auto_sync.py` + `tests/scripts/test_run_wfa_queue_postprocess.py`): `3 passed`, `4 failed`; падения в `run_wfa_queue` postprocess CLI и в auto-sync expected status (`planned` vs `completed`).
- Verify #4 (после фикса контрактов postprocess/auto-sync, `SYNC_UP=1`): `7 passed, 0 failed` для `tests/scripts/test_build_run_index_auto_sync.py`, `tests/scripts/test_run_wfa_queue_postprocess.py`, `tests/utils/test_run_index.py`.
- Confirm replay (VPS postprocess + ranking):
  - `sync_queue_status.py`: обе очереди `20260220_confirm_top10_bl11` и `20260220_top3_fullspan_wfa` без изменений.
  - `build_run_index.py`: `Run index entries: 7927` (VPS snapshot).
  - strict fullspan command: `No variants matched fullspan policy v1 (missing_tail=0, worst_step_gate_failed=1)`.
  - diagnostic `tail_worst_gate_pct=0.21`: проходит только `..._slusd1p81`.
  - diagnostic all-variants (`tail_worst_gate_pct=1.00`, `min_pnl=-1000`): подтверждены причины reject по top-3 (`worst_step_pnl`/`worst_robust_pnl`).
- Fetch выполнен в изолированную директорию: `.remote_fetch_20260222/` (без перезаписи рабочего `docs/artifacts`).

## Automation (single command cycle)

- Добавлен оркестратор канонического цикла: `coint4/scripts/optimization/run_fullspan_decision_cycle.py`.
- Что делает скрипт по шагам: `sync_queue_status -> build_run_index (--no-auto-sync-status) -> strict rank (promote profile) -> diagnostic rank (research profile)`.
- Пример запуска:
  - `cd coint4 && PYTHONPATH=src ./.venv/bin/python scripts/optimization/run_fullspan_decision_cycle.py --queue artifacts/wfa/aggregate/20260220_confirm_top10_bl11/run_queue.csv --queue artifacts/wfa/aggregate/20260220_top3_fullspan_wfa/run_queue.csv --contains 20260220_top3_fullspan_wfa`
- Результат запуска (локально, 2026-02-22): `run_index entries=8077`, strict profile -> `No variants matched`, research profile (`tail_worst_gate_pct=0.21`) -> PASS только `..._slusd1p81`.
- Повторный канонический прогон (локально, 2026-02-22T17:42:16Z) подтвердил тот же verdict:
  - `strict_profile_rc=1` (`No variants matched fullspan policy v1`),
  - `research_profile_rc=0` (PASS только `..._slusd1p81`),
  - `run_index entries=8077` после `build_run_index --no-auto-sync-status`.
- Source of truth для этого отчёта: локальный `coint4/artifacts/wfa/aggregate/rollup/run_index.csv` (`8077` строк). VPS snapshot `7927` считается отдельным временным срезом (изолированный fetch), не переопределяющим локальный канонический rollup.

## Contracts fixed in code

- `scripts/optimization/run_wfa_queue.py`:
  - восстановлены флаги `--postprocess true|false`, `--rollup-output-dir`, `--rollup-queue-dir`, `--rollup-runs-dir`;
  - при `postprocess=true` выполняются status sync, `config_snapshot.yaml`, `git_commit.txt`, `canonical_metrics.json`, rebuild rollup.
- `scripts/optimization/build_run_index.py`:
  - добавлен авто-sync статусов очереди по наличию `strategy_metrics.csv` (по умолчанию включён),
  - добавлен флаг `--no-auto-sync-status` для fail-safe режима без записи в queue.

## Canonical postprocess (mandatory per block)

Выполнять после каждого блока прогонов в одинаковой последовательности:

1. `cd coint4 && PYTHONPATH=src ./.venv/bin/python scripts/optimization/sync_queue_status.py --queue artifacts/wfa/aggregate/<run_group>/run_queue.csv`
2. `cd coint4 && PYTHONPATH=src ./.venv/bin/python scripts/optimization/build_run_index.py --output-dir artifacts/wfa/aggregate/rollup --no-auto-sync-status`
3. `cd coint4 && PYTHONPATH=src ./.venv/bin/python scripts/optimization/rank_multiwindow_robust_runs.py --run-index artifacts/wfa/aggregate/rollup/run_index.csv --contains <run_group_or_tag> --fullspan-policy-v1 --min-windows 1 --min-trades 200 --min-pairs 20 --max-dd-pct 0.50 --min-pnl 0 --initial-capital 1000 --tail-quantile 0.20 --tail-q-soft-loss-pct 0.03 --tail-worst-soft-loss-pct 0.10 --tail-q-penalty 2.0 --tail-worst-penalty 1.0 --tail-worst-gate-pct 0.20`

Postprocess status for reference block (`20260220_*`): `yes` (синхронизация статусов, пересборка rollup, ranker-верификация выполнены; strict fullspan verdict rechecked on 2026-02-22).

## Next actions

1. Любой новый кандидат из short OOS сначала фиксировать как `shortlist_only`.
2. Перед promote выполнять fullspan replay и ранжирование только через `--fullspan-policy-v1`.
3. Развести policy-пороги для research shortlist и final promote в live (сейчас это единый strict gate).
4. Пересмотреть DSR-gate (`min_dsr`) отдельно от PSR: текущая шкала DSR в rollup не является вероятностью `[0..1]`.
5. В каждом новом блоке дневника сохранять `run_group`, `queue_path`, `host`, `ALLOW_HEAVY_RUN`, `result`, `postprocess`.

## Planned block: 20260222_tailguard_r01 (worst_step mitigation)

- run_group: `20260222_tailguard_r01`
- queue_path: `coint4/artifacts/wfa/aggregate/20260222_tailguard_r01/run_queue.csv`
- size: `10` run'ов (`5` вариантов × holdout/stress), статус очереди: `planned`
- configs: `coint4/configs/budget1000_autopilot/20260222_tailguard_r01/*.yaml`

Гипотеза:

- Снизить `worst_step_pnl` (сейчас -209.15 у лучшего кандидата) ниже hard-gate (`>= -200`) за счёт более консервативного tail-guard:
  - tighter `pair_stop_loss_usd` (`1.70`);
  - lower `risk_per_position_pct` (`0.0055`);
  - lower `max_var_multiplier` (`1.0055`);
  - combo (`slusd1p70 + risk0p0055`).

Безопасный remote-run шаблон (heavy только на VPS):

- `cd coint4`
- `SYNC_UP=1 UPDATE_CODE=0 STOP_AFTER=1 SYNC_BACK=1 bash scripts/remote/run_server_job.sh bash -lc 'echo RUN_HOST=$(hostname); bash scripts/optimization/watch_wfa_queue.sh --queue artifacts/wfa/aggregate/20260222_tailguard_r01/run_queue.csv --parallel 10'`
- Проверка shutdown после завершения: `ssh root@85.198.90.128 "echo ok"` должен перестать отвечать.

## Implementation update: NO-EXEC + guardrails + runner (2026-02-22)

Кодовые изменения:

- Добавлен общий guardrail-модуль: `coint4/src/coint2/ops/heavy_guardrails.py`.
- Guardrails подключены в heavy entrypoints:
  - `coint4/scripts/optimization/run_wfa_queue.py`
  - `coint4/scripts/optimization/watch_wfa_queue.sh`
  - `coint4/run_wfa_fullcpu.sh`
  - `coint4/scripts/core/optimize.py`
  - `coint4/src/optimiser/run_optimization.py` (CLI)
  - `coint4/scripts/optimization/web_optimizer.py`
- Добавлен канонический ручной runner: `coint4/scripts/batch/run_heavy_queue.sh`.
- Добавлена политика: `docs/operating_mode_no_exec.md` (`OPERATING MODE: CODE-ONLY / NO-EXEC`) + ссылка в `docs/optimization_state.md`.

Проверки:

- `make lint` -> `All checks passed!`
- `make test` -> `359 passed, 7 skipped, 429 deselected`
- Новые/обновлённые тесты:
  - `coint4/tests/scripts/test_heavy_guardrails.py`
  - `coint4/tests/scripts/test_run_heavy_queue.py`
  - `coint4/tests/scripts/test_run_wfa_queue_postprocess.py` (добавлен негативный сценарий блокировки без `ALLOW_HEAVY_RUN`)

### Tailguard run attempts (`20260222_tailguard_r01`)

Фактические попытки запуска через `run_heavy_queue.sh`:

1. `runner=watch`, `SYNC_UP=1`:
   - fail: на VPS отсутствовал `coint2.ops.heavy_guardrails` (файл был untracked локально и не попал при `git ls-files` sync).
2. После добавления нового модуля в tracked set, `runner=watch`:
   - fail-fast по контракту watcher: `walk_forward.max_steps` в очереди равен `null` (fullspan), а `watch_wfa_queue.sh` разрешает только `<=5`.
3. `runner=queue` (отдельный fullspan pipeline):
   - infra failure: `SSH not ready after 15 minutes` в `run_server_job.sh`; затем VPS отправлен в shutdown через API.

Состояние очереди после попыток:

- `coint4/artifacts/wfa/aggregate/20260222_tailguard_r01/run_queue.csv` -> `planned=10`, `completed=0`.
- Проверка после завершения попытки: `ssh root@85.198.90.128 "echo ok"` -> timeout (VPS недоступен, не оставлен включённым).

Операционный verdict:

- `result=infra_blocked_ssh_unavailable`
- `selection_mode=shortlist_only` (без изменений baseline/live winner до успешного heavy + postprocess + strict fullspan pass).

## FS-009 execution update: 20260222_tailguard_r02 (final)

Execution record:

- run_group: `20260222_tailguard_r02`
- queue_path: `coint4/artifacts/wfa/aggregate/20260222_tailguard_r02/run_queue.csv`
- host: `85.198.90.128` (через `coint4/scripts/remote/run_server_job.sh`)
- ALLOW_HEAVY_RUN: `1` (remote command)
- result: `success` (`32/32 completed`)
- VPS shutdown: `yes` (после завершения `ssh root@85.198.90.128` -> timeout)

Postprocess:

1. `cd coint4 && PYTHONPATH=src ./.venv/bin/python scripts/optimization/sync_queue_status.py --queue artifacts/wfa/aggregate/20260222_tailguard_r02/run_queue.csv`
   - result: `no changes (metrics_present=32, missing=0, skipped=0)`
2. `cd coint4 && PYTHONPATH=src ./.venv/bin/python scripts/optimization/build_run_index.py --output-dir artifacts/wfa/aggregate/rollup --no-auto-sync-status`
   - result: `Run index entries: 8119`
3. strict rank:
   - `cd coint4 && PYTHONPATH=src ./.venv/bin/python scripts/optimization/rank_multiwindow_robust_runs.py --run-index artifacts/wfa/aggregate/rollup/run_index.csv --contains 20260222_tailguard_r02 --fullspan-policy-v1 --min-windows 1 --tail-worst-gate-pct 0.20 --top 20`
4. diagnostic rank:
   - та же команда, но `--tail-worst-gate-pct 0.21`

Strict/diagnostic result (совпадает):

- top-1: `tailguard_risk0p0055_risk0p0055_slusd1p81_max_var_multiplier1p0065_mp21_corr0p335_pv0p39`
- `score=1.043`, `worst_robust_sh=1.142`, `worst_dd_pct=0.291`, `worst_pnl=1147.93`, `worst_step_pnl=-198.90`

Comparison vs `20260222_tailguard_r01`:

- У лидера те же ключевые метрики (`score=1.043`, `worst_robust_sh=1.142`, `worst_dd_pct=0.291`, `worst_pnl=1147.93`, `worst_step_pnl=-198.90`).
- Вывод: `r02` не дал улучшения относительно `r01`; цикл закрыт как completed-without-improvement.

Decision:

- `selection_policy=fullspan_v1`
- `selection_mode=strict_fullspan_pass`
- `promotion_verdict=deferred_for_optimization`
- next step: запуск `r03` с новым search-space (выход из локального tailguard-плато).

## FS-009 execution update: 20260222_tailguard_r03 (finalized 2026-02-23)

Execution record:

- run_group: `20260222_tailguard_r03`
- queue_path: `coint4/artifacts/wfa/aggregate/20260222_tailguard_r03/run_queue.csv`
- host: `85.198.90.128` (через `coint4/scripts/remote/run_server_job.sh`)
- ALLOW_HEAVY_RUN: `1` (remote command)
- result: `success` (`48/48 completed`)
- VPS shutdown: `yes` (после завершения `ssh root@85.198.90.128` -> timeout)

Postprocess:

1. `cd coint4 && PYTHONPATH=src ./.venv/bin/python scripts/optimization/sync_queue_status.py --queue artifacts/wfa/aggregate/20260222_tailguard_r03/run_queue.csv`
   - result: `no changes (metrics_present=48, missing=0, skipped=0)`
2. `cd coint4 && PYTHONPATH=src ./.venv/bin/python scripts/optimization/build_run_index.py --output-dir artifacts/wfa/aggregate/rollup --no-auto-sync-status`
   - result: `Run index entries: 8167`
3. strict rank:
   - `cd coint4 && PYTHONPATH=src ./.venv/bin/python scripts/optimization/rank_multiwindow_robust_runs.py --run-index artifacts/wfa/aggregate/rollup/run_index.csv --contains 20260222_tailguard_r03 --fullspan-policy-v1 --min-windows 1 --min-trades 200 --min-pairs 20 --max-dd-pct 0.50 --min-pnl 0 --initial-capital 1000 --tail-quantile 0.20 --tail-q-soft-loss-pct 0.03 --tail-worst-soft-loss-pct 0.10 --tail-q-penalty 2.0 --tail-worst-penalty 1.0 --tail-worst-gate-pct 0.20 --top 500`
4. diagnostic rank:
   - та же команда, но `--tail-worst-gate-pct 0.21`

Strict/diagnostic result (совпадает):

- `strict_pass_count=2`, `diagnostic_pass_count=2`
- top-1: `tailguard_risk0p0055_risk0p0055_slusd1p81_max_var_multiplier1p0065_mp21_corr0p335_pv0p39_risk0p0055_slusd1p81_max_var_multiplier1p0065_mp21_corr0p335_pv0p39_z1p15_exit0p08_dstop0p02_maxpos18`
- `score=1.043`, `worst_robust_sh=1.142`, `worst_dd_pct=0.291`, `worst_pnl=1147.93`, `worst_step_pnl=-198.90`

Comparison vs `r02` and `r01` top-1:

| run_group | score | worst_robust_sh | worst_dd_pct | worst_pnl | worst_step_pnl | verdict |
|---|---:|---:|---:|---:|---:|---|
| `20260222_tailguard_r01` | 1.043 | 1.142 | 0.291 | 1147.93 | -198.90 | baseline plateau |
| `20260222_tailguard_r02` | 1.043 | 1.142 | 0.291 | 1147.93 | -198.90 | no improvement |
| `20260222_tailguard_r03` | 1.043 | 1.142 | 0.291 | 1147.93 | -198.90 | no improvement |

Decision:

- `selection_policy=fullspan_v1`
- `selection_mode=strict_fullspan_pass`
- `promotion_verdict=deferred_for_optimization`
- cycle verdict: `completed-without-improvement` (gap до цели `Sharpe=3.0` остаётся `+1.858`)
- next step: запуск `r04` с новым search-space, не повторяющим оси `r03`.
