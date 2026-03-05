# Plan: Финализация фиксов review P1/P2 и подготовка к интеграции

Обновлено: 2026-03-05 (America/New_York)

## Summary
Цель: довести пакет исправлений по review (P1/P2) до состояния «готово к интеграции» без смешивания с нерелевантными изменениями в грязном рабочем дереве.

Текущий статус:
- Логика P1/P2 в коде реализована.
- Регрессионные тесты по целевому набору проходят.
- Осталось аккуратно зафиксировать scope и документацию.

## Execution Status (2026-03-05)
- [x] P1 реализован: квота цикла confirm-fastlane учитывается только после успешного dispatch.
- [x] P2 реализован: stress naming совместим с shortlist/filter (`{variant_tag}_stress.yaml` + stress-исключения в shortlist).
- [x] Регрессионные тесты по P1/P2 добавлены и проходят.
- [x] Документация синхронизирована: `Plan.md` + `docs/optimization_state.md` отражают статус `READY_FOR_INTEGRATION`.
- [x] Обязательный pre-merge VPS smoke выполнен через `scripts/remote/run_server_job.sh`, сервер авто-остановлен после прогона.
- [ ] Интеграционный коммит не делался в рамках этого шага (рабочее дерево содержит много нерелевантных изменений).

Выполненная валидация:
- `cd coint4 && PYTHONPATH=src ./.venv/bin/pytest -q tests/scripts/test_confirm_dispatch_agent.py tests/scripts/test_evolve_next_batch.py tests/scripts/test_evolve_next_batch_stress_naming.py tests/scripts/test_autonomous_wfa_driver_selector_contract.py tests/scripts/test_anti_idle_capacity_controller.py tests/scripts/test_strict_fullspan_regression_guard.py tests/scripts/test_fullspan_lineage.py`
  - результат: `24 passed`.
- `cd coint4 && ./.venv/bin/ruff check scripts/optimization/evolve_next_batch.py tests/scripts/test_confirm_dispatch_agent.py tests/scripts/test_evolve_next_batch_stress_naming.py tests/scripts/test_autonomous_wfa_driver_selector_contract.py`
  - результат: `All checks passed!`.

## VPS Smoke Evidence (2026-03-05)
- Команда:
- `cd /home/claudeuser/coint4/coint4 && SKIP_POWER=1 STOP_AFTER=1 STOP_VIA_SSH=1 UPDATE_CODE=0 SYNC_UP=1 SYNC_UP_MODE=tracked SYNC_BACK=0 SERVER_IP=85.198.90.128 bash scripts/remote/run_server_job.sh bash -lc 'cd /opt/coint4/coint4 && REPO_ROOT=/opt/coint4/coint4 python3 /tmp/vps_smoke_confirm.py'`
- Ключевой лог:
- `SMOKE_CONFIRM_CONTRACT_PASS`
- `{"quota":{"confirm_fastlane_trigger_empty_shortlist":1,"confirm_fastlane_trigger_attempt":2,"confirm_fastlane_trigger_count":1,"confirm_fastlane_cycle_dispatch_count":1},"stress_filter":{"shortlist_rows":1,"fallback_rows":1,"kept_config_path":"configs/target.yaml"}}`
- `[server] stopping via SSH (shutdown -h now)`
- Пост-условие инфраструктуры:
- Первичная авто-остановка была запрошена helper-скриптом; после проверки доступности VPS выполнен явный `shutdown -h now`, повторная проверка дала `ssh timeout`.
- Вывод:
- P1 подтверждён поведенчески: неуспешная первая попытка не съедает квоту цикла (`dispatch_count=1` при `attempt=2`).
- P2 подтверждён поведенчески: stress-строки вычищаются из shortlist/fallback, остаётся только holdout `configs/target.yaml`.

## Execution Update (2026-03-05, parallel workers)
Статус: IMPLEMENTED.

### Worker A — Confirm-dispatch quota (P1)
- Список изменённых файлов:
- `coint4/scripts/optimization/confirm_dispatch_agent.sh` (правка уже присутствует в текущем дереве)
- `coint4/tests/scripts/test_confirm_dispatch_agent.py` (регрессия активна)
- Команды, которые запускал:
- `cd /home/claudeuser/coint4/coint4 && PYTHONPATH=src ./.venv/bin/pytest -q tests/scripts/test_confirm_dispatch_agent.py`
- Результат (успех/ошибка + кусок лога):
- Успех.
- `collected 1 item` -> `test_confirm_dispatch_agent.py . [100%]` -> `1 passed in 13.47s`
- Что осталось/риски:
- Интеграционная проверка confirm-потока под реальными квотами на VPS остаётся опциональной до merge.

### Worker B — Stress naming compatibility (P2)
- Список изменённых файлов:
- `coint4/scripts/optimization/evolve_next_batch.py` (правка уже присутствует в текущем дереве)
- `coint4/tests/scripts/test_evolve_next_batch_stress_naming.py` (регрессия активна)
- Команды, которые запускал:
- `cd /home/claudeuser/coint4/coint4 && PYTHONPATH=src ./.venv/bin/pytest -q tests/scripts/test_evolve_next_batch_stress_naming.py`
- `cd /home/claudeuser/coint4/coint4 && ./.venv/bin/ruff check scripts/optimization/evolve_next_batch.py tests/scripts/test_evolve_next_batch_stress_naming.py tests/scripts/test_confirm_dispatch_agent.py`
- Результат (успех/ошибка + кусок лога):
- Успех.
- `test_evolve_next_batch_stress_naming.py . [100%]` -> `1 passed in 13.17s`; `All checks passed!`
- Что осталось/риски:
- Риск только регресса при будущих изменениях нейминга/парсинга, текущий тест это покрывает.

### Worker C — Driver/Docs contract alignment
- Список изменённых файлов:
- `coint4/scripts/optimization/autonomous_wfa_driver.sh` (контракт `_stress.yaml` подтверждён)
- `docs/optimization_state.md` (P1/P2: статус DONE и критерии приёмки)
- `Plan.md` (этот execution update)
- Команды, которые запускал:
- `cd /home/claudeuser/coint4 && rg -n "_stress.yaml|confirm_fastlane_cycle_dispatch_count|DONE" coint4/scripts/optimization/autonomous_wfa_driver.sh coint4/scripts/optimization/confirm_dispatch_agent.sh coint4/scripts/optimization/evolve_next_batch.py docs/optimization_state.md Plan.md`
- Результат (успех/ошибка + кусок лога):
- Успех.
- Найдены ожидаемые маркеры в коде и доках: `_stress.yaml`, `confirm_fastlane_cycle_dispatch_count`, `Status: DONE`.
- Что осталось/риски:
- Большой pre-existing dirty worktree: перед коммитом нужен строгий selective add по scope P1/P2.

## Execution Update (2026-03-05, parallel subtasks round 2)
Статус: IMPLEMENTED.

### Worker A — VPS smoke (mandatory pre-merge)
- Список изменённых файлов:
- Нет (read-only smoke через remote helper).
- Команды, которые запускал:
- `cd /home/claudeuser/coint4/coint4 && STOP_AFTER=1 UPDATE_CODE=0 SYNC_BACK=0 SYNC_UP=0 bash scripts/remote/run_server_job.sh bash -lc '...python3 smoke...'`
- Результат (успех/ошибка + кусок лога):
- Попытка 1: ошибка regex в smoke-check (`re.error: missing ), unterminated subpattern`), сервер остановлен через SSH.
- Попытка 2: ошибка экранирования токена в assertion (`driver stress filter token missing`), сервер остановлен через API.
- Попытка 3: успех.
- Лог: `[server] SSH ready` -> `RUN_HOST=coint` -> `VPS_SMOKE_OK P1/P2 contract markers verified` -> `[server] stopping ... (API)`.
- Что осталось/риски:
- Это smoke на контрактных маркерах кода; полноценный end-to-end replay confirm-очереди остаётся отдельным усилением перед merge/cutover.

### Worker B — Selective integration scope audit
- Список изменённых файлов:
- Нет (read-only аудит).
- Команды, которые запускал:
- `git status --short <targeted files>`
- `rg -n "confirm_fastlane_cycle_dispatch_count|if dispatched|_stress.yaml|startswith\\('stress_'\\)|run_id.startswith\\('stress_'\\)" ...`
- Результат (успех/ошибка + кусок лога):
- Успех: scope релевантен P1/P2 + docs/tests.
- Кусок лога: `TARGET_SCOPE_STATUS: M ...confirm_dispatch_agent.sh ...evolve_next_batch.py ...autonomous_wfa_driver.sh ... docs/optimization_state.md ?? Plan.md ?? tests/...`
- Что осталось/риски:
- Перед коммитом нужен строгий selective add только по целевому scope.

### Worker C — Local validation gate
- Список изменённых файлов:
- Нет (валидация).
- Команды, которые запускал:
- `PYTHONPATH=src ./.venv/bin/pytest -q tests/scripts/test_confirm_dispatch_agent.py tests/scripts/test_evolve_next_batch.py tests/scripts/test_evolve_next_batch_stress_naming.py tests/scripts/test_autonomous_wfa_driver_selector_contract.py tests/scripts/test_anti_idle_capacity_controller.py tests/scripts/test_strict_fullspan_regression_guard.py tests/scripts/test_fullspan_lineage.py`
- `./.venv/bin/ruff check scripts/optimization/evolve_next_batch.py tests/scripts/test_confirm_dispatch_agent.py tests/scripts/test_evolve_next_batch_stress_naming.py tests/scripts/test_autonomous_wfa_driver_selector_contract.py`
- Результат (успех/ошибка + кусок лога):
- Успех: `24 passed in 8.68s`, `All checks passed!`.
- Что осталось/риски:
- Локальный gate зелёный, но интеграционный commit-пакет всё ещё не сформирован.

Готовые команды для безопасной упаковки selective commit scope (без commit):
- `git add Plan.md docs/optimization_state.md coint4/scripts/optimization/confirm_dispatch_agent.sh coint4/scripts/optimization/evolve_next_batch.py coint4/scripts/optimization/autonomous_wfa_driver.sh coint4/tests/scripts/test_confirm_dispatch_agent.py coint4/tests/scripts/test_evolve_next_batch_stress_naming.py coint4/tests/scripts/test_autonomous_wfa_driver_selector_contract.py`
- `git diff --cached -- coint4/scripts/optimization/confirm_dispatch_agent.sh coint4/scripts/optimization/evolve_next_batch.py coint4/scripts/optimization/autonomous_wfa_driver.sh coint4/tests/scripts/test_confirm_dispatch_agent.py coint4/tests/scripts/test_evolve_next_batch_stress_naming.py coint4/tests/scripts/test_autonomous_wfa_driver_selector_contract.py docs/optimization_state.md Plan.md`

## Implementation Changes
1. Изолировать scope fixed-пакета:
- `coint4/scripts/optimization/confirm_dispatch_agent.sh` (P1 quota after successful dispatch).
- `coint4/scripts/optimization/evolve_next_batch.py` (P2 stress naming compatibility).
- `coint4/scripts/optimization/autonomous_wfa_driver.sh` (stress shortlist filter parity).
- Регрессионные тесты: `coint4/tests/scripts/test_confirm_dispatch_agent.py`, `coint4/tests/scripts/test_evolve_next_batch_stress_naming.py`, `coint4/tests/scripts/test_autonomous_wfa_driver_selector_contract.py`.

2. Зафиксировать консистентность docs:
- `Plan.md` отражает только текущую цель интеграции P1/P2.
- `docs/optimization_state.md` содержит краткую запись о фактическом статусе P1/P2 (DONE + что валидация пройдена).

3. Подготовить интеграционный набор (без посторонних изменений):
- В коммиты включать только релевантные файлы fixed-пакета и docs.
- Не затрагивать прочие historical/unrelated изменения в репозитории.

## Test Plan
Обязательный минимум перед интеграцией:
- `cd coint4 && PYTHONPATH=src ./.venv/bin/pytest -q tests/scripts/test_confirm_dispatch_agent.py tests/scripts/test_evolve_next_batch.py tests/scripts/test_evolve_next_batch_stress_naming.py tests/scripts/test_autonomous_wfa_driver_selector_contract.py tests/scripts/test_anti_idle_capacity_controller.py tests/scripts/test_strict_fullspan_regression_guard.py tests/scripts/test_fullspan_lineage.py`
- `cd coint4 && ./.venv/bin/ruff check scripts/optimization/evolve_next_batch.py tests/scripts/test_confirm_dispatch_agent.py tests/scripts/test_evolve_next_batch_stress_naming.py tests/scripts/test_autonomous_wfa_driver_selector_contract.py`

Критерии приёмки:
- Все проверки зелёные.
- Нерелевантные файлы не включены в интеграционный пакет.
- P1/P2 формулировки в документации соответствуют фактической реализации.

## Assumptions
- Репозиторий изначально в грязном состоянии; это не блокер при selective integration.
- Обязательный VPS smoke перед merge выполнен; финальный блокер только аккуратная упаковка selective commit без нерелевантных файлов.
