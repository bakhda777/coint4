## 2026-03-05 — Процессный слой fullspan-контракта

Что внедрено:
- Добавлен автоматический `process_slo_guard_agent.py` (воронка/KPI/WIP/SLA + события).
- Добавлены systemd units `autonomous-process-slo-guard-agent.{service,timer}`.
- Инсталлятор supervisor обновлён: новый таймер ставится/включается/перезапускается вместе с остальными.
- `autonomous_10m_report.sh` расширен: теперь выводит процессную воронку, KPI и process-alerts.
- Документирован процессный контракт в `docs/optimization_process_contract.md`.

Проверки:
- `ruff`, `py_compile`, `bash -n` пройдены.
- user systemd timers активированы после переустановки supervisor.

Ожидаемый эффект:
- Снижение процессной двусмысленности (единая воронка и KPI).
- Быстрое обнаружение процессных узких мест (SLA/WIP), а не только технических ошибок runner.
- Более предсказуемое движение к `PROMOTE_ELIGIBLE` по strict fullspan-контракту.

## 2026-03-05 — VPS E2E smoke (remote helper, lightweight)

Цель:
- Проверить E2E контур `coint4/scripts/remote/run_server_job.sh` на `85.198.90.128` без heavy-run.
- Снять evidence по observability/fail-closed маркерам (`strict_pass`, `promote_eligible`, `FAIL_CLOSED`) если они присутствуют на VPS.

Команды (из `/home/claudeuser/coint4`):
- `ssh -i ~/.ssh/id_ed25519 -o BatchMode=yes -o StrictHostKeyChecking=no -o ConnectTimeout=5 root@85.198.90.128 'echo ok'`
- `SKIP_POWER=1 STOP_AFTER=1 STOP_VIA_SSH=1 UPDATE_CODE=0 SYNC_UP=0 SYNC_BACK=0 SERVER_IP=85.198.90.128 SERVER_USER=root bash coint4/scripts/remote/run_server_job.sh bash -lc '...read-only markers probe...'`
- `SKIP_POWER=0 STOP_AFTER=1 STOP_VIA_SSH=1 UPDATE_CODE=0 SYNC_UP=0 SYNC_BACK=0 SERVER_IP=85.198.90.128 SERVER_USER=root bash coint4/scripts/remote/run_server_job.sh bash -lc '...find /opt/coint4 .../.autonomous + markers probe...'`
- `SKIP_POWER=0 STOP_AFTER=1 STOP_VIA_SSH=1 UPDATE_CODE=0 SYNC_UP=0 SYNC_BACK=0 SERVER_IP=85.198.90.128 SERVER_USER=root bash coint4/scripts/remote/run_server_job.sh bash -lc '...find /opt /root /home driver.log|decision_notes|fullspan_decision_state...'`

Фактический вывод (excerpt):
- Smoke #1 (`SKIP_POWER=1`):
  - `RUN_HOST=coint`
  - `RUN_UTC=2026-03-05T19:26:32Z`
  - `DRIVER_STATE_MISSING=1`
  - `driver.log missing`
  - `decision_notes missing`
  - `fullspan_decision_state_missing=1`
  - `[server] stopping via SSH (shutdown -h now)`
- Smoke #2 (`SKIP_POWER=0`, fallback power-on):
  - `RUN_HOST=coint`
  - `RUN_UTC=2026-03-05T19:33:05Z`
  - `PWD=/opt/coint4/coint4`
  - `AUTONOMOUS_STATE_DIR_NOT_FOUND=1`
  - `[server] stopping l45s1251192 (API)`
- Smoke #3 (`SKIP_POWER=0`, расширенный поиск путей):
  - `RUN_HOST=coint`
  - `RUN_UTC=2026-03-05T19:34:04Z`
  - `SEARCH_STATE_FILES`
  - `STATE_FILES_NOT_FOUND=1`
  - `[server] stopping l45s1251192 (API)`

Итог smoke:
- E2E remote helper подтверждён: SSH-ready, удалённая команда выполняется, shutdown отрабатывает (`STOP_AFTER=1`).
- На VPS не обнаружены автономные state/log файлы, поэтому маркеры `strict_pass/promote_eligible/FAIL_CLOSED` на стороне VPS в этом smoke не извлечены (не доступны по путям `/.autonomous` и при расширенном поиске).

## 2026-03-05 — Subtask B: lightweight VPS E2E smoke (owner-only doc update)

Цель:
- Прогнать лёгкий E2E smoke через `coint4/scripts/remote/run_server_job.sh` и снять реальные evidence-строки.
- Проверить наличие маркеров `strict_pass` / `promote_eligible` / `fail-closed` (`FAIL_CLOSED`) в доступных autonomous/rollup источниках.

Команда (из `/home/claudeuser/coint4`, с авто-shutdown):
- `UPDATE_CODE=0 SYNC_BACK=0 STOP_AFTER=1 coint4/scripts/remote/run_server_job.sh bash -lc 'set -euo pipefail; echo RUN_HOST=$(hostname); echo RUN_DATE_UTC=$(date -u +%Y-%m-%dT%H:%M:%SZ); cd /opt/coint4/coint4 || exit 3; for p in artifacts/wfa/autonomous/driver.log artifacts/wfa/autonomous/decision_notes.jsonl artifacts/wfa/aggregate/rollup/run_index.md artifacts/wfa/aggregate/rollup/run_index.json; do [[ -f "$p" ]] && echo FOUND:$p || echo MISSING:$p; done; ... marker_scan(strict_pass|promote_eligible|fail-closed|FAIL_CLOSED) ...'`

Фактический вывод (ключевой excerpt):
- `[server] starting l45s1251192`
- `[server] SSH ready`
- `RUN_HOST=coint`
- `RUN_DATE_UTC=2026-03-05T19:43:38Z`
- `MISSING:artifacts/wfa/autonomous/driver.log`
- `MISSING:artifacts/wfa/autonomous/decision_notes.jsonl`
- `FOUND:artifacts/wfa/aggregate/rollup/run_index.md`
- `FOUND:artifacts/wfa/aggregate/rollup/run_index.json`
- `SCAN:artifacts/wfa/aggregate/rollup/run_index.md` (совпадений по marker regex нет)
- `SCAN:artifacts/wfa/aggregate/rollup/run_index.json` (совпадений по marker regex нет)
- `[server] stopping l45s1251192 (API)`
- `EXIT_CODE=0`

Статус маркеров:
- `strict_pass`: не найден (проверено в `artifacts/wfa/aggregate/rollup/run_index.md` и `artifacts/wfa/aggregate/rollup/run_index.json`).
- `promote_eligible`: не найден (те же пути).
- `fail-closed` / `FAIL_CLOSED`: не найден (те же пути).
- Автономные журналы недоступны в ожидаемых путях: `artifacts/wfa/autonomous/driver.log`, `artifacts/wfa/autonomous/decision_notes.jsonl` отсутствуют на VPS в текущем smoke.

## 2026-03-05 — VPS canonical probe (fixed path `.autonomous`)

Цель:
- Убрать двусмысленность пути и снять маркеры из канонического state-dir: `artifacts/wfa/aggregate/.autonomous`.

Команда (из `coint4/`, lightweight + auto-shutdown):
- `SKIP_POWER=0 STOP_AFTER=1 UPDATE_CODE=0 SYNC_UP=0 SYNC_BACK=0 SERVER_IP=85.198.90.128 SERVER_USER=root bash scripts/remote/run_server_job.sh bash -lc 'cd /opt/coint4/coint4; python3 scripts/optimization/process_slo_guard_agent.py --root /opt/coint4/coint4 >/dev/null 2>&1 || true; python3 - <<\"PY\" ... PY'`

Фактический вывод (excerpt):
- `RUN_UTC=2026-03-05T19:56:22Z`
- `STATE_DIR=/opt/coint4/coint4/artifacts/wfa/aggregate/.autonomous`
- `EXISTS:.../process_slo_state.json=1`
- `EXISTS:.../fullspan_decision_state.json=0`
- `EXISTS:.../decision_notes.jsonl=0`
- `EXISTS:.../driver.log=0`
- `STRICT_PASS_COUNT=0`
- `PROMOTE_ELIGIBLE_COUNT=0`
- `FAIL_CLOSED_COUNT=0`
- `FAIL_CLOSED_POLICY_DEFAULT_PRESENT=1`
- `[server] stopping l45s1251192 (API)`

Итог:
- Каноничный probe подтверждён: strict-pass/ promote отсутствуют на VPS (`0/0`).
- Fail-closed режим подтверждён как policy-default (`FAIL_CLOSED_POLICY_DEFAULT_PRESENT=1`) при отсутствии promote.
- Явный инфра-факт: на VPS не материализованы `fullspan_decision_state.json`, `decision_notes.jsonl`, `driver.log`; поэтому `FAIL_CLOSED_COUNT` по event/log-источникам остаётся `0` и требует отдельной материализации driver/fullspan state.

## 2026-03-05 — Anti-idle throughput tuning (VPS load)

Что изменено:
- `vps_capacity_controller_agent.py`: anti-idle defaults усилены до `search_parallel_min=8`, `search_parallel_max=24`, hard max `24`.
- `autonomous_wfa_driver.sh`: автосидинг переведён на отдельные параметры `AUTO_SEED_PENDING_THRESHOLD=48`, `AUTO_SEED_NUM_VARIANTS=24` (вместо привязки к `MIN_PLANNED_BACKLOG=12`).
- Обновлён тест `test_anti_idle_capacity_controller.py` под новый контракт.

Проверки:
- `pytest tests/scripts/test_anti_idle_capacity_controller.py` -> `1 passed`.
- `pytest tests/scripts/test_autonomous_wfa_driver_runtime_policy.py tests/scripts/test_autonomous_wfa_driver_selector_contract.py tests/scripts/test_anti_idle_capacity_controller.py` -> `12 passed`.
- `pytest` целевой регрессионный пакет -> `32 passed`.

Срез нагрузки VPS (NPROC=2, SSH sample):
- `2026-03-05T20:09:19Z` -> loadavg `1.31 1.25 0.62`
- `2026-03-05T20:09:29Z` -> loadavg `1.26 1.24 0.62`
- `2026-03-05T20:09:39Z` -> loadavg `1.22 1.23 0.63`

Итог:
- После тюнинга фактическая средняя загрузка по `loadavg(1m)` держится выше порога 50% для 2 vCPU (`>1.0`), при сохранении strict fullspan fail-closed логики.
