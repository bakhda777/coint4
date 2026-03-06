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

## Execution Update (2026-03-06, VPS anti-idle / ready-buffer / cold-tail)
- Реализован `ready buffer` для hot selector:
  - driver теперь пишет `candidate_pool.csv` и `ready_queue_buffer.json`;
  - `candidate_parse_empty` сначала пытается взять queue из ready-buffer и только потом делает auto-seed;
  - в active remote-run path добавлен `overlap dispatch` через `maybe_dispatch_overlap_from_buffer()` при tail-фазе текущей queue.
- Реализован `TTL cold fail index`:
  - `candidate_gated_reject` теперь добавляет queue в `.autonomous/cold_fail_index.json`;
  - `find_candidate()` исключает активные cold-fail entries из hot scan до истечения TTL или изменения queue mtime.
- Реализован driver-side `SURROGATE_IDLE_OVERRIDE`:
  - если fresh queue получила `SURROGATE_REFINE reason=queue_pending_backlog`, но `remote_active_queue_jobs=0` или `idle_with_executable_pending=true`, driver переопределяет решение в `allow` с reason `cold_start_idle_slot`.
- Реализован surrogate-side idle-safe override:
  - `gate_surrogate_agent.py` читает `.autonomous/process_slo_state.json`;
  - evidence/state теперь содержат `idle_slot_available`, `idle_slot_source`, `cold_start_idle_slot_override`.
- Реализован seeder coordination по ready-buffer:
  - `autonomous_queue_seeder.py` читает `ready_queue_buffer.json`;
  - seed может триггериться не только по global backlog, но и по `ready_buffer_below_refill_threshold`.
- Runtime observability усилен:
  - `process_slo_guard_agent.py`, `probe_autonomous_markers.py`, `autonomous_10m_report.sh` теперь выводят:
    - `ready_buffer_depth`
    - `cold_fail_active_count`
    - `remote_active_queue_jobs`
    - `surrogate_idle_override_count`
    - `overlap_dispatch_count`

### Verification
- `bash -n`:
  - `coint4/scripts/optimization/autonomous_wfa_driver.sh`
  - `coint4/scripts/dev/autonomous_10m_report.sh`
- `py_compile`:
  - `gate_surrogate_agent.py`
  - `autonomous_queue_seeder.py`
  - `probe_autonomous_markers.py`
  - `process_slo_guard_agent.py`
- `ruff`:
  - `All checks passed!`
- `pytest`:
  - `test_gate_surrogate_agent.py`
  - `test_autonomous_queue_seeder_ready_buffer.py`
  - `test_autonomous_wfa_driver_runtime_policy.py`
  - `test_autonomous_wfa_driver_selector_contract.py`
  - итог: `30 passed, 1 deselected`

### Runtime spot-check
- `probe_autonomous_markers.py` после обновления:
  - `SURROGATE_BRANCH_STATUS=evidence_present`
  - `READY_BUFFER_DEPTH=0`
  - `COLD_FAIL_ACTIVE_COUNT=0`
  - `REMOTE_ACTIVE_QUEUE_JOBS=0`
  - `SURROGATE_IDLE_OVERRIDE_COUNT=0`
  - `OVERLAP_DISPATCH_COUNT=0`
- `autonomous_10m_report.sh` печатает:
  - `Runtime observability: ready_buffer_depth=0 cold_fail_active_count=0 remote_active_queue_jobs=0 surrogate_idle_override_count=0 overlap_dispatch_count=0`

### Residual runtime risks
- Нужно реальное наблюдение в живом цикле, что:
  - ready-buffer перестал давать `candidate_parse_empty` как основной bottleneck;
  - `SURROGATE_IDLE_OVERRIDE` срабатывает на fresh queue при idle VPS;
  - cold-tail действительно снижает повторный hot-scan уже proven `HARD_FAIL` очередей;
  - overlap dispatch держит VPS занятым без конфликтов top-level queue jobs.

## Execution Update (2026-03-06, quality-first funnel)
- Доведён `invalid proposal firewall` в planner до проверяемого рабочего контура:
  - `evolve_next_batch.py` теперь гарантированно отсеивает invalid proposals до materialization queue/config;
  - `invalid_proposal_index.json` используется как persistent quarantine/fingerprint state;
  - покрыто регрессиями в `test_evolve_next_batch_invalid_proposals.py`.
- Добавлен `yield governor`:
  - новый `coint4/scripts/optimization/yield_governor_agent.py`;
  - строит fail-safe state по recent queue/run_index/fullspan evidence;
  - публикует `preferred_contains`, `cooldown_contains`, `winner_proximate`, `lane_weights`, `policy_overrides`.
- `search_director_agent.py` теперь:
  - пишет `yield_governor_state.json`;
  - публикует `winner_proximate` lanes и `lane_weights` в `search_director_directive.json`;
  - materialize/backfill `cold_fail_index.json` из текущего `fullspan_decision_state.json`.
- `autonomous_queue_seeder.py` теперь:
  - читает `yield_governor_state.json`;
  - подмешивает `winner_proximate`/`preferred_contains` в `contains`;
  - применяет fail-safe `policy_overrides` от governor;
  - пишет snapshot governor-состояния в `queue_seeder.state.json`.
- Runtime split и early-abort подтверждены regression-пакетом:
  - queue-level `EARLY_ABORT_ZERO_ACTIVITY` уже присутствует в driver;
  - `process_slo_guard_agent.py`/`probe_autonomous_markers.py`/`autonomous_10m_report.sh` уже ведут `remote_child_process_count`, `remote_queue_job_count`, `cpu_busy_without_queue_job`.

### Verification (quality-first funnel batch)
- `bash -n`:
  - `coint4/scripts/optimization/autonomous_wfa_driver.sh`
  - `coint4/scripts/dev/autonomous_10m_report.sh`
- `py_compile`:
  - `evolve_next_batch.py`
  - `search_director_agent.py`
  - `yield_governor_agent.py`
  - `autonomous_queue_seeder.py`
  - `process_slo_guard_agent.py`
  - `probe_autonomous_markers.py`
- `ruff`:
  - `All checks passed!`
- `pytest`:
  - `test_evolve_next_batch_invalid_proposals.py`
  - `test_search_director_agent.py`
  - `test_yield_governor_agent.py`
  - `test_autonomous_queue_seeder_ready_buffer.py`
  - `test_autonomous_wfa_driver_runtime_policy.py`
  - `test_anti_idle_capacity_controller.py`
  - `test_autonomous_wfa_driver_selector_contract.py`
  - итог: `33 passed, 1 deselected`

### Runtime spot-check
- После запуска `search_director_agent.py` materialized state присутствует:
  - `yield_governor_state.json` создан;
  - `search_director_directive.json` содержит `winner_proximate` и `lane_weights`;
  - `cold_fail_index.json` создан (сейчас `entries=0`, то есть backfill-path рабочий, но в текущем ledger нет активных reject-entry для cold-tail).
- Текущий governor предпочитает high-yield lineage:
  - `20260216_budget1000_bl4_r05_slusd`
  - `20260216_budget1000_cl_r03_vm`
  - несколько `20260226_evo_smoke_i13x/...` lineage как boost-path.

### Remaining risks
- `cold_fail_index.json` теперь materialized, но пока пустой в живом snapshot; нужно дождаться следующего reject/backfill, чтобы подтвердить non-zero path end-to-end.
- Winner-proximate routing сейчас использует fail-safe aggregation по доступным полям (`run_group`, `lineage_uid`, queue metadata); при появлении richer lineage/operator fields governor можно сделать точнее без изменения контракта.
- Один legacy mismatch, замеченный в отдельном под-агенте: `test_patch_ast_topup_fills_target_variants` исторически ожидает holdout-only поведение и не отражает текущий strict holdout+stress pairing. В этот merge он не входил.
