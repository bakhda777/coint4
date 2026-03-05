# Optimization state

Last updated: 2026-03-05 14:48 America/New_York

Snapshot bundle:
- `process_slo_state.json`: `2026-03-05T19:47:32Z`
- `candidate.csv`: `2026-03-05T19:45:30Z`
- `fullspan_decision_state.json`: `2026-03-05T19:47:32Z`
- `run_index.md`: `Generated at: 2026-03-05 19:43:00Z`

## Winner contract
Критерий winner неизменен: strict fullspan pass + минимум 2 независимых confirm + успешный confirm replay; до этого только `FAIL_CLOSED` (см. `docs/fullspan_selection_policy.md`).

## Подтверждённые факты этого раунда
Источник: `coint4/artifacts/wfa/aggregate/.autonomous/process_slo_state.json`
- Funnel: `generated=13869`, `executable=8315`, `completed=12614`, `strict_pass=0`, `confirm_ready=0`, `promote_eligible=0`.
- Queue: `pending=360`, `running=10`, `stalled=308`, `local_runner_count=2`, `remote_runner_count=0`, `remote_reachable=true`.
- Alert: `STALLED_RATIO_HIGH` (`stalled_ratio=0.856 >= 0.600`).

Источник: `coint4/artifacts/wfa/aggregate/.autonomous/candidate.csv`
- Active candidate: `artifacts/wfa/aggregate/autonomous_seed_20260305_194423/run_queue.csv`.
- Статус: `planned=10`, `promotion_potential=POSSIBLE`, `gate_status=OPEN`, `strict_gate_status=FULLSPAN_PREFILTER_PASSED`.

Источник: `coint4/artifacts/wfa/aggregate/.autonomous/fullspan_decision_state.json`
- Verdicts: `ANALYZE=4`, `REJECT=2`.
- BL11 `r19/r20`: `REJECT` с `rejection_reason=no_progress_streak`, `cutover_permission=FAIL_CLOSED`.
- `autonomous_seed_20260305_175223`: `contract_reason=METRICS_MISSING`, `cutover_permission=FAIL_CLOSED`.

Источник: `coint4/artifacts/wfa/aggregate/.autonomous/run_20260305_194327_autonomous_seed_20260305_154313.log` и `run_20260305_194629_autonomous_seed_20260305_194423.log`
- Предыдущий запуск упал на sync: `powered: FAIL reason=REMOTE_SYNC_FAILED` и `scp: Connection closed`.
- Текущий retry дошёл до remote execution: `scp_ok .../autonomous_seed_20260305_194423/run_queue.csv`, `powered: remote run start`.

Источник: VPS canonical probe (`2026-03-05T19:56:22Z`, `STATE_DIR=/opt/coint4/coint4/artifacts/wfa/aggregate/.autonomous`)
- `STRICT_PASS_COUNT=0`, `PROMOTE_ELIGIBLE_COUNT=0`.
- `FAIL_CLOSED_POLICY_DEFAULT_PRESENT=1`.
- На VPS отсутствуют `fullspan_decision_state.json`, `decision_notes.jsonl`, `driver.log` (поэтому event/log `FAIL_CLOSED_COUNT=0` в этом probe).

Источник: `coint4/artifacts/wfa/aggregate/rollup/run_index.csv`
- Размер индекса: `13803` строк данных (`13804` с заголовком).

Источник: `coint4/artifacts/wfa/aggregate/.autonomous/capacity_controller_state.json`
- Политика anti-idle усилена: `search_parallel_min=8`, `search_parallel_max=24` (при headroom/backlog).

## Что фиксировано как progress
- `autonomous_seed_20260305_175223` переведена в terminal-state (`completed=10/10`), что дало сдвиг funnel (`completed +10`, `pending -10`).
- После транспортного сбоя sync для нового candidate проходит (`scp_ok`), запуск очереди возобновлён.

## Что остаётся blocker до winner
- Нет strict-evidence: `strict_pass=0`.
- Нет confirm-evidence: `confirm_count=0`/`confirm_ready=0`.
- Высокий stalled-pressure (`308/360`) удерживает throughput около нуля.
- Активные/недавние очереди продолжают идти через `METRICS_MISSING`.
- `remote_runner_count=0` и history сетевых сбоев увеличивают риск очередного no-progress.

## Exact next action
По завершении `run_20260305_194629_autonomous_seed_20260305_194423.log` сразу выполнить mini-cycle (`sync_queue_status -> build_run_index -> run_fullspan_decision_cycle.py`) для `autonomous_seed_20260305_194423`; при повторном `METRICS_MISSING` автоматически зафиксировать root-cause и переключиться на следующий seed.
