#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

LOCK_FILE="${REPO_ROOT}/.ai_autopilot/autopilot_loop.lock"
exec 9>"${LOCK_FILE}"
if ! flock -n 9; then
  echo "[autopilot][ERROR] another autopilot_loop is already running (lock: ${LOCK_FILE})" >&2
  exit 2
fi

mkdir -p "${REPO_ROOT}/logs"
RALPH_LOG_PATH="${REPO_ROOT}/logs/ralph_headless.log"

log() {
  printf '[autopilot] %s\n' "$*"
}

die() {
  printf '[autopilot][ERROR] %s\n' "$*" >&2
  exit 1
}

require_cmd() {
  command -v "$1" >/dev/null 2>&1 || die "command not found: $1"
}

cleanup_ralph_lock_and_sessions() {
  python3 - <<'PY'
from pathlib import Path

lock = Path(".ralph-tui/ralph.lock")
try:
    if lock.exists():
        lock.unlink()
except Exception as e:
    print(f"[autopilot] WARN: failed to delete {lock}: {e}")

for p in sorted(Path(".ralph-tui").glob("session*.json")):
    try:
        p.unlink()
    except Exception as e:
        print(f"[autopilot] WARN: failed to delete {p}: {e}")
PY
}

cleanup_stale_ralph_lock_and_sessions() {
  python3 - <<'PY'
from pathlib import Path
import time

lock = Path(".ralph-tui/ralph.lock")
if not lock.exists():
    raise SystemExit(0)

try:
    age_sec = time.time() - lock.stat().st_mtime
except Exception:
    age_sec = 0

if age_sec <= 30 * 60:
    raise SystemExit(0)

try:
    lock.unlink()
    print(f"[autopilot] stale ralph.lock removed (age_sec={int(age_sec)})")
except Exception as e:
    print(f"[autopilot] WARN: failed to delete {lock}: {e}")

for p in sorted(Path(".ralph-tui").glob("session*.json")):
    try:
        p.unlink()
        print(f"[autopilot] removed stale session file: {p}")
    except Exception as e:
        print(f"[autopilot] WARN: failed to delete {p}: {e}")
PY
}

read_ralph_status() {
  local status_json
  set +e
  status_json="$(ralph-tui status --json 2>/dev/null)"
  RALPH_STATUS_RC=$?
  set -e

  local parsed
  parsed="$(python3 - "${status_json}" <<'PY'
import json, sys

def _out(status: str = "", progress: str = "", pid: str = "", locked: str = "0") -> None:
    print(f"{status}\t{progress}\t{pid}\t{locked}")

try:
    text = (sys.argv[1] if len(sys.argv) > 1 else "") or ""
    text = text.strip()
    data = {}
    if text:
        try:
            data = json.loads(text)
        except Exception:
            data = {}

    status = str(data.get("status") or "")
    session = data.get("session") or {}
    progress = session.get("progress") or {}
    percent = progress.get("percent")
    completed = progress.get("completed")
    progress_val = ""
    if isinstance(percent, (int, float)):
        progress_val = str(int(percent))
    elif isinstance(completed, (int, float)):
        progress_val = str(int(completed))

    lock = data.get("lock") or {}
    pid = lock.get("pid")
    pid_val = str(pid) if isinstance(pid, int) else ""
    locked_val = "1" if lock.get("isLocked") else "0"

    _out(status, progress_val, pid_val, locked_val)
except Exception:
    _out()
PY
)"

  IFS=$'\t' read -r RALPH_STATUS RALPH_PROGRESS RALPH_LOCK_PID RALPH_LOCKED <<< "${parsed}"
}

pick_pending_remote_queue() {
  python3 - <<'PY'
import csv
import re
from pathlib import Path

repo_root = Path.cwd().resolve()
agg_dir = repo_root / "coint4" / "artifacts" / "wfa" / "aggregate"

pattern = re.compile(r"^(?P<date>\\d{8})_s(?P<sprint>\\d+)_")
pending = []

if agg_dir.exists():
    for queue_path in agg_dir.glob("*/run_queue.csv"):
        group = queue_path.parent.name
        if "tailguard" not in group:
            continue
        m = pattern.match(group)
        if not m:
            continue
        try:
            date_key = int(m.group("date"))
            sprint_key = int(m.group("sprint"))
        except Exception:
            continue
        try:
            with queue_path.open(newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                rows = list(reader)
        except Exception:
            continue
        pending_cnt = 0
        for row in rows:
            status = str(row.get("status") or "").strip().lower()
            if status in ("planned", "stalled"):
                pending_cnt += 1
        if pending_cnt <= 0:
            continue
        rel = queue_path.relative_to(repo_root)
        pending.append(((date_key, sprint_key, group), pending_cnt, str(rel)))

if not pending:
    print("")
else:
    # Prefer newest sprint queues (avoid spending time on stale backlog).
    pending.sort(key=lambda x: (x[0], x[1], x[2]), reverse=True)
    print(pending[0][2])
PY
}

run_pending_remote_queue() {
  if [[ "${AUTOPILOT_REMOTE:-1}" != "1" ]]; then
    return 0
  fi

  local queue_path
  queue_path="$(pick_pending_remote_queue || true)"
  queue_path="$(printf '%s' "${queue_path}" | tr -d '\r' | tr -d '\n')"
  if [[ -z "${queue_path}" ]]; then
    return 0
  fi
  if [[ ! -f "${queue_path}" ]]; then
    log "remote-run: pending queue not found: ${queue_path}"
    return 0
  fi

  local queue_rel
  queue_rel="${queue_path#coint4/}"
  if [[ "${queue_rel}" == "${queue_path}" ]]; then
    # Unexpected path format; assume it's already relative to app root.
    queue_rel="${queue_path}"
  fi

  local parallel
  parallel="${AUTOPILOT_REMOTE_PARALLEL:-10}"

  log "remote-run: starting queue=${queue_path} (parallel=${parallel})"
  local rc=0
  set +e
  (
    cd "${REPO_ROOT}/coint4"
    SYNC_UP=1 UPDATE_CODE=1 STOP_AFTER=1 SYNC_BACK=1 \
      bash scripts/remote/run_server_job.sh \
      bash -lc "ALLOW_HEAVY_RUN=1 bash scripts/optimization/watch_wfa_queue.sh --queue ${queue_rel} --parallel ${parallel}"
  )
  rc=$?
  set -e
  if [[ "$rc" -ne 0 ]]; then
    log "remote-run: FAILED rc=${rc} queue=${queue_path}"
    return 0
  fi
  log "remote-run: finished queue=${queue_path}"
}

run_ralph_with_watchdog() {
  local epic_id="$1"
  local poll_seconds=60
  local stall_seconds=$((20 * 60))

  cleanup_stale_ralph_lock_and_sessions

  : >> "${RALPH_LOG_PATH}"
  local tail_pid=""
  # By default, do NOT tail the ralph log from inside the loop. It easily causes duplicated lines
  # when multiple loops/tails exist (or when stdout is redirected). If you want follow-mode:
  #   AUTOPILOT_FOLLOW_LOG=1 bash tools/autopilot_loop.sh
  if [[ "${AUTOPILOT_FOLLOW_LOG:-0}" == "1" && -t 2 ]]; then
    tail -n 0 -F "${RALPH_LOG_PATH}" >&2 &
    tail_pid=$!
    trap 'if [[ -n "${tail_pid}" ]]; then kill -TERM "${tail_pid}" 2>/dev/null || true; wait "${tail_pid}" 2>/dev/null || true; fi' RETURN
  fi

  log "starting ralph-tui headless (log: ${RALPH_LOG_PATH})"
  ralph-tui run --headless --no-setup --serial --tracker beads --epic "${epic_id}" </dev/null >> "${RALPH_LOG_PATH}" 2>&1 &
  local run_pid=$!
  log "ralph-tui run pid=${run_pid}"

  local last_progress=""
  local last_change_ts
  last_change_ts="$(date +%s)"
  local seen_session=0

  while true; do
    read_ralph_status
    if [[ -n "${RALPH_STATUS}" && "${RALPH_STATUS}" != "no-session" ]]; then
      seen_session=1
    fi

    if [[ -n "${RALPH_PROGRESS}" && "${RALPH_PROGRESS}" != "${last_progress}" ]]; then
      last_progress="${RALPH_PROGRESS}"
      last_change_ts="$(date +%s)"
      log "progress=${last_progress} (status=${RALPH_STATUS})"
    fi

    if [[ "${RALPH_STATUS}" == "completed" || "${RALPH_STATUS_RC}" -eq 0 ]]; then
      log "ralph completed (status=${RALPH_STATUS}, rc=${RALPH_STATUS_RC})"
      break
    fi

    if [[ "${RALPH_STATUS}" == "failed" || ( "${RALPH_STATUS}" == "no-session" && "${seen_session}" -eq 1 ) || ( "${RALPH_STATUS_RC}" -eq 2 && "${seen_session}" -eq 1 ) ]]; then
      log "ralph not running (status=${RALPH_STATUS}, rc=${RALPH_STATUS_RC})"
      break
    fi

    if [[ -n "${run_pid}" ]] && ! kill -0 "${run_pid}" 2>/dev/null; then
      wait "${run_pid}" || true
      if [[ "${RALPH_STATUS}" == "running" || "${RALPH_STATUS}" == "paused" ]]; then
        log "run pid exited but status=${RALPH_STATUS}; continuing to monitor"
        run_pid=""
        last_change_ts="$(date +%s)"
      else
        log "ralph run process exited"
        break
      fi
    fi

    local now_ts
    now_ts="$(date +%s)"
    if (( now_ts - last_change_ts >= stall_seconds )); then
      log "watchdog: no progress change for 20m → restart with --force"

      if [[ -n "${run_pid}" ]] && kill -0 "${run_pid}" 2>/dev/null; then
        kill -TERM "${run_pid}" 2>/dev/null || true
        sleep 10
        kill -KILL "${run_pid}" 2>/dev/null || true
        wait "${run_pid}" || true
      elif [[ "${RALPH_LOCKED}" == "1" && -n "${RALPH_LOCK_PID}" ]]; then
        kill -TERM "${RALPH_LOCK_PID}" 2>/dev/null || true
        sleep 10
        kill -KILL "${RALPH_LOCK_PID}" 2>/dev/null || true
      fi

      cleanup_ralph_lock_and_sessions

      ralph-tui run --force --headless --no-setup --serial --tracker beads --epic "${epic_id}" </dev/null >> "${RALPH_LOG_PATH}" 2>&1 &
      run_pid=$!
      log "ralph restarted pid=${run_pid}"

      last_progress=""
      last_change_ts="$(date +%s)"
      seen_session=0
    fi

    sleep "${poll_seconds}"
  done

  if [[ -n "${run_pid}" ]] && kill -0 "${run_pid}" 2>/dev/null; then
    wait "${run_pid}" || true
  fi
}

require_cmd bd
require_cmd ralph-tui
require_cmd python3

STATE_PATH="${REPO_ROOT}/.ai_autopilot/state.json"
GOAL_PATH="${REPO_ROOT}/.ai_autopilot/goal.md"

[[ -f "${STATE_PATH}" ]] || die "missing state.json: ${STATE_PATH}"
[[ -f "${GOAL_PATH}" ]] || die "missing goal.md: ${GOAL_PATH}"

if [[ ! -d "${REPO_ROOT}/.beads" ]]; then
  log "no .beads/ found → bd init"
  bd init --silent
fi

EPIC_TITLE="Sharpe>3 autopilot"

find_epic_id() {
  bd list --json --type epic --limit 0 --all | python3 -c '
import json, sys
title = "Sharpe>3 autopilot".strip().lower()
items = json.loads(sys.stdin.read() or "[]")
found = ""
if isinstance(items, list):
  for row in items:
    if isinstance(row, dict) and str(row.get("title","")).strip().lower() == title:
      found = str(row.get("id","")).strip()
      break
print(found)
'
}

EPIC_ID="$(find_epic_id || true)"
if [[ -z "${EPIC_ID}" ]]; then
  log "creating epic: ${EPIC_TITLE}"
  epic_desc="$(cat <<'EOF'
Замкнутый autopilot-цикл вокруг ralph-tui + beads:
- worker (ralph) выполняет задачи спринта,
- manager (tools/sprint_manager.py) делает Retro/Planning и создаёт следующий спринт.

Цель: устойчивый Sharpe>3 при контроле DD/сделок/хвостов. Тяжёлые прогоны — только remote (85.198.90.128).
EOF
)"
  EPIC_ID="$(bd create --type epic --title "${EPIC_TITLE}" --description "${epic_desc}" --labels "ralph,autopilot" --priority 1 --silent)"
  bd sync || true
else
  log "found epic: ${EPIC_ID}"
fi

log "epic_id=${EPIC_ID}"

read_state_field() {
  local field="$1"
  python3 - <<PY "${STATE_PATH}" "${field}"
import json, sys
from pathlib import Path

path = Path(sys.argv[1])
field = sys.argv[2]
data = json.loads(path.read_text(encoding="utf-8"))
val = data.get(field)
if isinstance(val, bool):
    print("true" if val else "false")
elif val is None:
    print("")
else:
    print(val)
PY
}

count_open_tasks_under_epic() {
  bd list --json --parent "${EPIC_ID}" --limit 0 | python3 -c '
import json, sys
items = json.loads(sys.stdin.read() or "[]")
open_cnt = 0
if isinstance(items, list):
  for row in items:
    if isinstance(row, dict) and str(row.get("status","")) in ("open","in_progress"):
      open_cnt += 1
print(open_cnt)
'
}

create_initial_sprint_tasks() {
  local sprint="$1"
  local label="sprint-${sprint}"
  local labels="ralph,autopilot,${label}"

  # Idempotency: if sprint tasks already exist, do nothing.
  local existing
  existing="$(bd list --json --parent "${EPIC_ID}" --label "autopilot,${label}" --limit 0 --all | python3 -c 'import json,sys; items=json.loads(sys.stdin.read() or "[]"); print(len(items) if isinstance(items,list) else 0)')"
  if [[ "${existing}" != "0" ]]; then
    log "sprint ${sprint} tasks already exist (${existing}); skipping creation"
    return 0
  fi

  log "creating sprint ${sprint} tasks (labels: ${labels})"

  local desc1 desc2 desc3 desc4 desc5 desc6

  desc1="$(cat <<'EOF'
Цель: снять «моментальный срез» текущих лучших результатов из rollup run_index.*

Ожидаемый эффект: в репозитории появляется отчёт-таблица baseline для дальнейших сравнений.

Проверка:
- [ ] Создан файл reports/sprint_1_baseline.md с top-10 строками (Sharpe, |DD|, trades, config_path, run_group)
- [ ] В конце файла — краткий вывод «что выглядит подозрительно/что проверить дальше»

Ограничение: не запускать тяжёлые прогоны; только чтение существующих артефактов.
EOF
)"

  desc2="$(cat <<'EOF'
Цель: формализовать “качество” кандидата (gates) и критерии остановки.

Ожидаемый эффект: в docs/optimization_state.md есть короткий, однозначный раздел с фильтрами:
- минимальные trades,
- ограничение |DD|,
- tail-loss ограничения,
- требования к подтверждениям (несколько run_group/holdout).

Проверка:
- [ ] Обновлён docs/optimization_state.md: добавлен раздел “Gates / Stop condition”
- [ ] Раздел не противоречит AGENTS.md (remote runs, artifacts paths)
EOF
)"

  desc3="$(cat <<'EOF'
Цель: обновить дневник прогонов текущим контекстом и гипотезами.

Ожидаемый эффект: в docs/optimization_runs_YYYYMMDD.md добавлена запись:
- что считаем baseline,
- какие 2–3 гипотезы самые дешёвые/информативные,
- какой следующий remote run_group планируем и почему.

Проверка:
- [ ] Создан/обновлён файл docs/optimization_runs_$(date -u +%Y%m%d).md с новой записью
EOF
)"

  desc4="$(cat <<'EOF'
Цель: убедиться, что локально корректно собираются статусы/rollup без “вечного planned”.

Ожидаемый эффект: если есть ручные прогоны/несостыковки статусов — они синхронизированы.

Проверка (best-effort):
- [ ] Для 1–2 актуальных очередей (run_queue.csv) прогнан scripts/optimization/sync_queue_status.py (если применимо)
- [ ] Пересобран rollup индекс build_run_index.py (если не слишком долго)
- [ ] В docs/optimization_runs_YYYYMMDD.md кратко записано, что обновили

Ограничение: не трогать тяжёлые runs/ артефакты.
EOF
)"

  desc5="$(cat <<'EOF'
Цель: подготовить следующий remote-run в виде очереди (без запуска на этом сервере).

Ожидаемый эффект: новая очередь coint4/artifacts/wfa/aggregate/<group>/run_queue.csv с max_steps<=5.

Проверка:
- [ ] Создан новый датированный run_group и run_queue.csv (6–20 задач)
- [ ] Явно задан walk_forward.max_steps (<=5)
- [ ] В описании очереди/дневнике указано, что запускать через coint4/scripts/remote/run_server_job.sh
EOF
)"

  desc6="$(cat <<'EOF'
Цель: Sprint Retro + Plan Next Sprint.

Ожидаемый эффект: .ralph-tui/progress.md содержит краткую ретро-заметку и план следующего remote-блока.

Проверка:
- [ ] В .ralph-tui/progress.md добавлена секция по спринту (что сделали/выводы/что дальше)
- [ ] Git commit руками не создавался (autoCommit=true)
EOF
)"

  bd create --type task --title "S${sprint}: Baseline snapshot (top-10 rollup)" --description "${desc1}" --labels "${labels}" --priority 1 --parent "${EPIC_ID}" --silent >/dev/null
  bd create --type task --title "S${sprint}: Gates + stop condition в docs" --description "${desc2}" --labels "${labels}" --priority 1 --parent "${EPIC_ID}" --silent >/dev/null
  bd create --type task --title "S${sprint}: Обновить дневник прогонов (hypotheses)" --description "${desc3}" --labels "${labels}" --priority 2 --parent "${EPIC_ID}" --silent >/dev/null
  bd create --type task --title "S${sprint}: Sync queue status + rollup (best-effort)" --description "${desc4}" --labels "${labels}" --priority 2 --parent "${EPIC_ID}" --silent >/dev/null
  bd create --type task --title "S${sprint}: Подготовить remote run_queue.csv (max_steps<=5)" --description "${desc5}" --labels "${labels}" --priority 3 --parent "${EPIC_ID}" --silent >/dev/null
  bd create --type task --title "S${sprint}: Sprint Retro + Plan Next Sprint" --description "${desc6}" --labels "${labels}" --priority 4 --parent "${EPIC_ID}" --silent >/dev/null

  bd sync || true
  log "sprint ${sprint} tasks created"
}

log "starting loop..."

while true; do
  done_flag="$(read_state_field done || true)"
  if [[ "${done_flag}" == "true" ]]; then
    reason="$(read_state_field done_reason || true)"
    log "done=true → stopping (${reason})"
    exit 0
  fi

  # If there is any pending sprint remote queue, execute it first so postprocess tasks
  # (sync statuses / rollup / ranking) see fresh results.
  run_pending_remote_queue

  sprint="$(read_state_field sprint || true)"
  if [[ -z "${sprint}" ]]; then
    sprint="1"
  fi

  open_cnt="$(count_open_tasks_under_epic)"
  if [[ "${open_cnt}" -eq 0 ]]; then
    log "no open tasks under epic → creating sprint ${sprint}"
    create_initial_sprint_tasks "${sprint}"
    open_cnt="$(count_open_tasks_under_epic)"
  fi

  log "running ralph-tui headless (open tasks: ${open_cnt})"
  run_ralph_with_watchdog "${EPIC_ID}"

  log "ralph finished; running sprint_manager (review+retro+planning)"
  python3 tools/sprint_manager.py --epic-id "${EPIC_ID}" --goal "${GOAL_PATH}" --state "${STATE_PATH}"

  log "cycle complete; continuing..."
done
