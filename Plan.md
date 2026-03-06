# Plan: strict fullspan winner closure

Обновлено: 2026-03-06 05:18 America/New_York (snapshot: 2026-03-06T10:18:00Z)

## Execution Update (2026-03-06, review regressions: watcher-only + replay-fastlane)
- Закрыты оба интеграционных регресса из review без изменения winner-контракта.
- `watcher-only` окно на VPS теперь считается ownership-aware активной работой:
  - `remote_runtime_probe.py` публикует `watch_queue_paths`, `remote_queue_job_count`, `remote_active_queue_jobs`;
  - `remote_work_active=true`, если слот всё ещё удерживается watcher-процессом, даже когда child-jobs временно отсутствуют;
  - `cpu_busy_without_queue_job=true` теперь возможно только если queue-owner действительно отсутствует.
- Consumers переведены на queue ownership, а не на raw top-level процесс:
  - `vps_capacity_controller_agent.py`
  - `process_slo_guard_agent.py`
  - `probe_autonomous_markers.py`
  - `autonomous_10m_report.sh`
- В 10m-report добавлен явный режим `REMOTE_QUEUE_WATCHER_ONLY`, чтобы watcher-owned слот больше не выглядел как `idle`.
- `autonomous_queue_seeder.py` нормализует `replay_fastlane` как канонический источник confirm backlog:
  - loader читает `replay_fastlane.contains` / `replay_ready_count`;
  - backfill-ит legacy `confirm_replay` и `confirm_replay_contains` для совместимости;
  - lane selection берёт hints в порядке:
    1. `directive.replay_fastlane.contains`
    2. `yield_governor.replay_fastlane.contains`
    3. legacy `confirm_replay.contains`
    4. legacy `confirm_replay_contains`
    5. `winner_fallback` только если replay fastlane реально пуст.
- В seeder snapshot добавлен `confirm_replay_source`, чтобы было видно, откуда пришли replay hints.

### Verification
- `python3 -m py_compile` по изменённым Python-файлам: OK
- `bash -n coint4/scripts/optimization/autonomous_wfa_driver.sh coint4/scripts/dev/autonomous_10m_report.sh`: OK
- `coint4/.venv/bin/ruff check` по затронутым Python-файлам/тестам: `All checks passed!`
- `coint4/.venv/bin/pytest -q coint4/tests/scripts/test_remote_runtime_probe.py coint4/tests/scripts/test_anti_idle_capacity_controller.py coint4/tests/scripts/test_autonomous_queue_seeder_ready_buffer.py coint4/tests/scripts/test_autonomous_wfa_driver_runtime_policy.py`
  - итог: `39 passed, 1 deselected`

## Execution Update (2026-03-06, canonical remote runtime snapshot)
- Закрыт источник ложного `idle` на VPS: введён канонический state-файл `.autonomous/remote_runtime_state.json`.
- Новый probe `coint4/scripts/optimization/remote_runtime_probe.py` за один SSH-срез собирает:
  - `reachable`
  - `load1`
  - `top_level_queue_jobs`
  - `remote_child_process_count`
  - `remote_runner_count` (legacy alias)
  - `remote_work_active`
  - `cpu_busy_without_queue_job`
- `autonomous_wfa_driver.sh` теперь обновляет runtime metrics из этого snapshot и пишет в `fullspan_decision_state.json`:
  - `remote_active_queue_jobs`
  - `remote_queue_job_count`
  - `top_level_queue_jobs`
  - `remote_child_process_count`
  - `remote_work_active`
  - `cpu_busy_without_queue_job`
  - `remote_runtime_snapshot_age_sec`
- `vps_capacity_controller_agent.py` переведён на этот же источник истины:
  - пишет `capacity_controller_state.json` с `top_level_queue_jobs`, `remote_active_queue_jobs`, `remote_queue_job_count`, `remote_child_process_count`, `remote_work_active`, `cpu_busy_without_queue_job`, `remote_snapshot_age_sec`;
  - anti-idle больше не раскрывает search policy, если на VPS уже есть child-process activity без top-level queue job.
- `process_slo_guard_agent.py` теперь использует приоритет источников:
  1. `remote_runtime_state.json`
  2. `capacity_controller_state.json`
  3. `fullspan_decision_state.runtime_metrics`
- `idle_with_executable_pending=true` теперь возможно только если snapshot свежий и `remote_work_active=false`; stale snapshot + высокий `load1`/child-process activity больше не объявляется idle.
- `autonomous_10m_report.sh` теперь показывает:
  - `remote_mode={REMOTE_QUEUE_ACTIVE|REMOTE_HEAVY_ACTIVE_CHILDREN|REMOTE_WORK_ACTIVE_UNKNOWN|REMOTE_IDLE}`
  - `remote_work_active`
  - `top_level_queue_jobs`
  - `remote_child_process_count`
  - `remote_snapshot_age_sec`

### Verification
- `python3 -m py_compile` по изменённым Python-файлам: OK
- `bash -n` по driver/report: OK
- `ruff check` по изменённым Python-файлам: `All checks passed!`
- `pytest -q coint4/tests/scripts/test_anti_idle_capacity_controller.py coint4/tests/scripts/test_autonomous_wfa_driver_runtime_policy.py coint4/tests/scripts/test_remote_runtime_probe.py`
  - покрывает:
    - child-only remote activity
    - SSH failure fallback
    - preference of fresh canonical snapshot over stale runtime metrics
    - stale snapshot + high remote load => `busy`, not `idle`

## Execution Update (2026-03-06, reconcile->ready-buffer dispatch fix)
- Найдена и закрыта конкретная orchestration-дыра после завершения active queue:
  - при `candidate_reconcile` и `pending<=0` driver сначала ресеlect-ил из общего hot scan (`find_candidate`), а `ready_buffer` использовал только как fallback на parse-empty;
  - это позволяло старому `HARD_FAIL` хвосту вытеснить свежую `SURROGATE_ALLOW` queue из `ready_buffer`, даже если новый batch уже был materialized и готов к dispatch.
- В `autonomous_wfa_driver.sh` при `pending<=0` после reconcile порядок теперь такой:
  1. очистить `candidate.csv`;
  2. `ready_buffer_refresh`;
  3. `ready_buffer_emit_candidate`;
  4. только если buffer пуст, идти в `find_candidate`.
- Добавлена регрессия:
  - `test_candidate_reselect_after_reconcile_prefers_ready_buffer_before_hot_scan`
  - гарантирует, что completed queue передаёт слот ready-buffer candidate, а не уходит обратно в stale reject-scan.

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

## Execution Update (2026-03-06, winner-proximate parent binding)
- Закрыт скрытый fallback `winner_proximate -> cold_start`:
  - причина была в том, что `autonomous_queue_seeder.py` передавал несколько lineage/run_group токенов в `evolve_next_batch.py` как повторяющиеся `--contains`, а planner трактовал их как `AND` по одной строке `run_index`.
  - из-за этого `build_variant_diagnostics()` часто возвращал пусто, хотя каждый токен по отдельности имел completed holdout history.
- В `coint4/src/coint2/ops/evolution_targeting.py` добавлен `contains_mode={all|any}`:
  - legacy `contains` остаётся `all`;
  - explicit winner-proximate resolution использует `any`.
- В `coint4/scripts/optimization/evolve_next_batch.py` добавлен explicit путь parent resolution:
  - новый repeatable CLI аргумент `--winner-proximate-token`;
  - новый слой `ParentResolution`, который сначала пытается выбрать parent по `winner_proximate` (`OR`-semantics), и только потом уходит в generic/base fallback;
  - в `search_space.md` и decision payload теперь фиксируются:
    - `winner_proximate_requested`
    - `winner_proximate_resolved`
    - `preferred_parent_source`
    - `winner_proximate_fallback_reason`
- В `coint4/scripts/optimization/autonomous_queue_seeder.py` focus split теперь явный:
  - generic `contains` = один anchor token, а не неразрешимая конъюнкция всех lineage сразу;
  - winner-proximate lineage передаются отдельно через `--winner-proximate-token`;
  - `queue_seeder.state.json` теперь сохраняет `parent_resolution` из decision payload для наблюдаемой деградации.
- Регрессии добавлены:
  - `test_evolve_next_batch_prefers_winner_parent_when_contains_disjoint_tokens`
  - `test_evolve_next_batch_records_winner_fallback_reason_when_unresolved`
  - `test_derive_planner_focus_separates_winner_tokens_from_generic_anchor`
- Сопутствующий legacy test обновлён под текущий strict fullspan контракт:
  - `test_patch_ast_topup_fills_target_variants` теперь ожидает paired `holdout+stress`, а не holdout-only.

### Verification
- `./coint4/.venv/bin/python -m py_compile ...` по изменённым файлам: OK
- `./coint4/.venv/bin/ruff check ...`: `All checks passed!`
- `./coint4/.venv/bin/pytest -q coint4/tests/scripts/test_evolve_next_batch.py coint4/tests/scripts/test_evolve_next_batch_patch_ast.py coint4/tests/scripts/test_evolve_next_batch_stress_naming.py coint4/tests/scripts/test_search_director_agent.py coint4/tests/scripts/test_autonomous_queue_seeder_ready_buffer.py`
  - итог: `21 passed, 1 deselected`
  - новый `coint4/scripts/optimization/yield_governor_agent.py`;
  - строит fail-safe state по recent queue/run_index/fullspan evidence;
  - публикует `preferred_contains`, `cooldown_contains`, `winner_proximate`, `lane_weights`, `policy_overrides`.
- `search_director_agent.py` теперь:
  - пишет `yield_governor_state.json`;
  - публикует `winner_proximate` lanes и `lane_weights` в `search_director_directive.json`;
  - materialize/backfill `cold_fail_index.json` из текущего `fullspan_decision_state.json`.
- `autonomous_queue_seeder.py` теперь:

## Execution Update (2026-03-06, exploit-first scheduler / policy-hash / hot-standby)
- В policy-layer введён exploit-first режим:
  - `yield_governor_agent.py` и `search_director_agent.py` теперь публикуют `lane_weights=65/20/15` (`winner_proximate` / `confirm_replay` / `broad_search`);
  - добавлены `replay_fastlane`, `policy_hash`, `planner_policy_inputs` и совместимые alias-поля `policy-hash`, `planner-policy-inputs`;
  - director materialize-ит hash уже после surrogate overlay, чтобы stale-buffer invalidation шёл по финальной policy, а не по промежуточной.
- В seeder/planner доведён provenance + diversification:
  - `autonomous_queue_seeder.py` теперь ведёт `selected_lane`, `lane_streak`, `token_rotation`, `parent_rotation_offset`, `confirm_replay_hints`, `policy_hash`;
  - на успешный seed пишется sidecar `queue_policy.json` рядом с queue и декорируется `metadata_json` в `run_queue.csv` (`planner_policy_hash`, `buffer_policy_version`, `queue_policy_path`);
  - `evolve_next_batch.py` принимает и записывает новые provenance/diversification args:
    - `--planner-policy-hash`
    - `--planner-hash`
    - `--seed-lane`
    - `--seed-lane-index`
    - `--parent-diversity-depth`
    - `--parent-rotation-offset`
    - `--confirm-replay-hint`
  - parent resolution теперь реально использует `parent_diversity_depth + rotation_offset`, а не всегда первый preferred parent.
- В driver/runtime замкнут exploit-first observability:
  - `autonomous_wfa_driver.sh` по умолчанию включает `DRIVER_CONFIRM_FASTLANE_ENABLE=1`;
  - hot-standby policy активна (`VPS_HOT_STANDBY_ENABLE=1`, `VPS_HOT_STANDBY_TTL_SEC=2700`);
  - runtime-observability слой считает и публикует:
    - `vps_duty_cycle_30m`
    - `ready_buffer_policy_mismatch_count`
    - `winner_parent_duplication_rate`
    - `fastlane_replay_pending`
    - `metrics_missing_abort_count_30m`
    - `winner_proximate_dispatch_count_30m`
    - `hot_standby_active`
  - `winner_proximate_dispatch` и `metrics_missing_abort` теперь пишутся в runtime observability state как события окна `30m`.
- В SLO/report слой прокинуты новые метрики:
  - `process_slo_guard_agent.py` включает их в `process_slo_state.json` и в guard-log;
  - `autonomous_10m_report.sh` выводит runtime-строку с duty/duplication/mismatch/replay/abort/winner-dispatch.

### Verification
- `python3 -m py_compile` по policy/seeder/planner/process_slo: OK
- `bash -n coint4/scripts/optimization/autonomous_wfa_driver.sh coint4/scripts/dev/autonomous_10m_report.sh`: OK
- `cd coint4 && ./.venv/bin/ruff check ...`: `All checks passed!`
- `cd coint4 && ./.venv/bin/python -m pytest ...`
  - итог: `48 passed, 1 deselected`

### Expected operational effect
- stale ready-buffer entries теперь инвалидируются по `policy_hash`, а не живут до случайного выгорания;
- winner-proximate search перестаёт застревать на одном parent за счёт lane rotation + parent diversity offset;
- confirm/fullspan replay fastlane перестаёт быть выключенным по умолчанию;
- VPS не должен выключаться между близкими batch-ами, пока есть ready-buffer/replay pressure или недавняя dispatch-активность.
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

## Execution Update (2026-03-06, anti-idle start softpass)

### Root cause
- Последний idle-window был не из-за отсутствия очередей: ready-buffer был заполнен, `capacity_controller_state.json` показывал `remote.reachable=true`, но `autonomous_wfa_driver.sh` жёстко блокировал старт на повторном `ensure_vps_ready` и писал `start_blocked reason=vps_unreachable`.

### Changes
- В `autonomous_wfa_driver.sh` добавлен узкий anti-idle fallback:
  - новые policy knobs: `VPS_RECENT_READY_GRACE_SEC`, `VPS_CAPACITY_STATE_MAX_AGE_SEC`;
  - новые helper-функции: `capacity_controller_remote_reachable_recent()`, `vps_recently_recovered()`, `start_queue_softpass_reason()`;
  - если `ensure_vps_ready` вернул false, но recovery был совсем недавно или есть свежий capacity-state с `remote.reachable=true`, driver пишет `START_VPS_SOFTPASS` и продолжает dispatch вместо простоя.
- Regression coverage добавлен в `tests/scripts/test_autonomous_wfa_driver_runtime_policy.py`.

### Verification
- `bash -n coint4/scripts/optimization/autonomous_wfa_driver.sh`: OK
- `cd coint4 && ./.venv/bin/python -m pytest tests/scripts/test_autonomous_wfa_driver_runtime_policy.py -q`
  - итог: `15 passed`

### Operational confirmation
- После окна проблемы автономный driver запустил:
  - `artifacts/wfa/aggregate/autonomous_seed_20260306_085124/run_queue.csv`
- На VPS подтверждён живой heavy-run:
  - `run_wfa_queue.py --queue artifacts/wfa/aggregate/autonomous_seed_20260306_085124/run_queue.csv --parallel 28`
  - `load average` вышел далеко из idle-zone, а в `ps` видны множественные `coint2 walk-forward` worker-процессы.
