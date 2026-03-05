# Plan: strict fullspan winner closure

Обновлено: 2026-03-05 16:29 America/New_York (snapshot: 2026-03-05T16:29:49Z)

## Winner definition (без двусмысленности)
Winner достигается только при одновременном выполнении:
- `strict_pass >= 1` по контракту `fullspan_v1`.
- `confirm_count >= 2` в независимых `run_group`.
- Confirm replay без деградации strict-критериев.
- До этого всегда `FAIL_CLOSED` (без promote).

## Что подтверждённо исправлено в этом раунде
- Очередь `autonomous_seed_20260305_175223` доведена до terminal-state: `completed=10/10` (было `planned=10`).
- Funnel сдвинулся: `completed 12604 -> 12614`, `pending 360 -> 350`.
- После `REMOTE_SYNC_FAILED` на `autonomous_seed_20260305_154313` новый запуск `autonomous_seed_20260305_194423` прошёл этап sync (`scp_ok`) и дошёл до `remote run start`.
- Контур вернулся в reachable-state: `remote_reachable=true`, `local_runner_count=2`.
- Каноничный VPS probe по `.autonomous` выполнен: `STRICT_PASS_COUNT=0`, `PROMOTE_ELIGIBLE_COUNT=0`, `FAIL_CLOSED_POLICY_DEFAULT_PRESENT=1`.
- Anti-idle ёмкость усилена: `search_parallel_min/max` подняты до `8/24`, автосидинг переведён на `threshold=48`, `num_variants=24`.

## Что всё ещё блокирует winner
- `strict_pass=0`, `confirm_ready=0`, `promote_eligible=0`.
- Высокий stalled-pressure: `308/360` (`stalled_ratio=0.856`).
- Fullspan решения остаются fail-closed: `ANALYZE=4`, `REJECT=2`; у очередей BL11 `r19/r20` `rejection_reason=no_progress_streak`.
- Для обработанных очередей сохраняется `contract_reason=METRICS_MISSING` (нет нового strict-evidence).
- `remote_runner_count=0` (дистанционный worker сейчас не считается активным), поэтому риск повторного no-progress остаётся.
- На VPS пока не материализованы `fullspan_decision_state.json` / `decision_notes.jsonl` / `driver.log`, поэтому event/log `FAIL_CLOSED_COUNT` там остаётся `0`.

## Exact next action
Дождаться завершения `run_20260305_194629_autonomous_seed_20260305_194423.log` и немедленно выполнить mini-cycle (`sync_queue_status -> build_run_index -> run_fullspan_decision_cycle.py`) для `autonomous_seed_20260305_194423`; если снова `METRICS_MISSING`, зафиксировать root-cause и автоматом перейти к следующему seed без ручного bypass.

## Parallel implementation batch (merged)
Выполнено параллельно через 3 под-агента и слито в основной workspace.

### Subtask A: Gate surrogate + director overlay
- Изменены файлы:
  - `coint4/scripts/optimization/gate_surrogate_agent.py` (новый)
  - `coint4/scripts/optimization/search_director_agent.py`
- Что сделано:
  - Добавлен расчёт `decision={allow|refine|reject}`, `risk_score`, `reason`, `evidence` по очередям.
  - `search_director_agent.py` теперь читает `gate_surrogate_state.json` и публикует:
    - `hard_fail_risk_policy`
    - `lineage_priority`
    - `repair_mode`
  - При отсутствии валидного surrogate-state действует neutral fail-closed overlay.

### Subtask B: Seeder policy integration
- Изменён файл:
  - `coint4/scripts/optimization/autonomous_queue_seeder.py`
- Что сделано:
  - Подключено чтение `gate_surrogate_state.json`.
  - `reject` -> исключение reject-lineage из `contains` (с fallback fail-closed).
  - `refine` -> принудительный repair-mode + консервативные knobs/caps.
  - Добавлены/прокинуты planner-аргументы:
    - `--gate-surrogate-state-path`
    - `--repair-mode/--no-repair-mode`
    - `--repair-max-neighbors`
    - `--exclude-knob` (repeatable)

### Subtask C: Planner repair + driver pre-dispatch gate
- Изменены файлы:
  - `coint4/scripts/optimization/evolve_next_batch.py`
  - `coint4/scripts/optimization/autonomous_wfa_driver.sh`
- Что сделано:
  - В planner добавлен `validation_neighbor` repair:
    - чтение deterministic quarantine (`CONFIG_VALIDATION_ERROR`),
    - поиск соседей в `coordinate_sweep_v1`,
    - фильтрация по changed_keys/dedupe/`exclude_knob`,
    - валидация через `AppConfig`, fallback на исходный вариант.
  - В драйвер добавлен pre-dispatch surrogate gate:
    - `SURROGATE_REJECT` -> skip + orphan + decision-note,
    - `SURROGATE_REFINE` -> skip heavy dispatch + auto-seed,
    - `SURROGATE_ALLOW` -> обычный dispatch.

### Merge verification
- Команды:
  - `bash -n coint4/scripts/optimization/autonomous_wfa_driver.sh`
  - `./coint4/.venv/bin/python -m py_compile coint4/scripts/optimization/{gate_surrogate_agent.py,search_director_agent.py,autonomous_queue_seeder.py,evolve_next_batch.py}`
  - `./coint4/.venv/bin/ruff check coint4/scripts/optimization/{gate_surrogate_agent.py,search_director_agent.py,autonomous_queue_seeder.py,evolve_next_batch.py}`
  - `./coint4/.venv/bin/pytest -q coint4/tests/scripts/test_autonomous_wfa_driver_runtime_policy.py coint4/tests/scripts/test_anti_idle_capacity_controller.py`
  - `./coint4/.venv/bin/pytest -q coint4/tests/scripts/test_autonomous_wfa_driver_selector_contract.py coint4/tests/scripts/test_strict_fullspan_regression_guard.py`
- Результат:
  - syntax/lint: OK
  - tests: `27 passed`

### Residual risks
- Пороговые эвристики surrogate (`reject/refine`) требуют калибровки по мере накопления новых run_index паттернов.
- `lineage_priority` зависит от naming-конвенции `evo_<hash>`.
- End-to-end прогон на живой очереди с активным `gate_surrogate_state.json` после merge ещё должен подтверждаться наблюдением в driver logs (`SURROGATE_*` события).

## Execution Update (2026-03-05, calibration + explicit lineage + runtime proof)
- Реализован `surrogate_calibrator_agent.py`: пишет `surrogate_calibration_state.json`, соблюдает `min_sample_size=100`, hysteresis `0.05` и apply-interval guard `86400s`.
- `gate_surrogate_agent.py` теперь:
  - читает calibration-state fail-safe;
  - применяет только валидные `applied_*` thresholds;
  - строит lineage priority по explicit `lineage_uid`, а legacy `evo_*` держит как fallback evidence.
- Добавлены systemd units:
  - `autonomous-gate-surrogate-agent.service/.timer`
  - `autonomous-surrogate-calibrator-agent.service/.timer`
- `install_autonomous_wfa_supervisor.sh` обновлён: новые таймеры устанавливаются, включаются и рестартуются вместе с остальным supervisor-контуром.
- Explicit lineage протянут в generation/consumers:
  - `evolve_next_batch.py` пишет `lineage_uid` и `metadata_json` в `run_queue.csv`;
  - `fullspan_lineage.py` предпочитает explicit lineage/metadata;
  - `deterministic_error_blacklist_agent.py` теперь блокирует explicit lineage, сохраняя alias `blocked_evo_uids` для совместимости;
  - `confirm_dispatch_agent.sh` сохраняет/читает `lineage_uid` из shortlist и регистрирует его в lineage registry.
- Runtime proof усилен:
  - `probe_autonomous_markers.py` теперь видит freshness gate/directive, `SURROGATE_REJECT/REFINE/ALLOW`, runtime counters и различает `evidence_present/no_eligible_case/broken_branch/stale_or_inconclusive`;
  - `autonomous_10m_report.sh` выводит surrogate gate/evidence/branch.
- Driver дополнен `surrogate_allow_count` + `SURROGATE_ALLOW` decision-note при allow-path с reason.
- Supervisor переустановлен и рестартован с новыми timers; текущий probe после рестарта показывает:
  - `GATE_SURROGATE_MODE=active`
  - `DIRECTIVE_GATE_SURROGATE_MODE=active`
  - `SURROGATE_BRANCH_STATUS=stale_or_inconclusive`
  - то есть state/directive живы, но свежего runtime-hit по `reject/refine` ещё не произошло в наблюдаемом окне.
