#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
QUEUE_ROOT="$ROOT_DIR/artifacts/wfa/aggregate"
STATE_DIR="$QUEUE_ROOT/.autonomous"
STATE_FILE="$STATE_DIR/driver_state.txt"
DECISION_NOTES_FILE="$STATE_DIR/decision_notes.jsonl"
FULLSPAN_DECISION_STATE_FILE="$STATE_DIR/fullspan_decision_state.json"
FULLSPAN_CYCLE_STATE_FILE="$STATE_DIR/mini_cycle_state.txt"
LOG_FILE="$STATE_DIR/driver.log"
CANDIDATE_FILE="$STATE_DIR/candidate.csv"
HEARTBEAT_FILE="$STATE_DIR/heartbeat_state.json"
ORPHAN_FILE="$STATE_DIR/orphan_queues.csv"
REPORT_STATE_FILE="$STATE_DIR/10m_human_report_state.json"
SERVER_IP="${SERVER_IP:-85.198.90.128}"
SERVER_USER="${SERVER_USER:-root}"

qline="$(tr -d '\r\n' <"$STATE_FILE" 2>/dev/null || true)"
current_queue=""
if [[ -n "$qline" ]]; then
  current_queue="${qline##*queue=}"
  current_queue="${current_queue%% *}"
fi

candidate_top=""
if [[ -f "$CANDIDATE_FILE" ]]; then
  candidate_top="$(tail -n 1 "$CANDIDATE_FILE" 2>/dev/null || true)"
fi

heartbeat_for_current="none"

last_action='none'
if [[ -f "$LOG_FILE" ]]; then
  last_action="$(awk '/action=/{ if (match($0, /action=([^ ]+)/, a)) { print a[1]; } }' "$LOG_FILE" | tail -n 1 || true)"
fi
if [[ -z "$last_action" ]]; then
  last_action='none'
fi

last_fullspan_action="none"
if [[ -f "$DECISION_NOTES_FILE" ]]; then
  last_fullspan_action="$(python3 - "$DECISION_NOTES_FILE" -c "import json,sys; from pathlib import Path; p=Path(sys.argv[1]); out="none";
for line in reversed(p.read_text(encoding=\'utf-8\').splitlines()):
    rec=json.loads(line)
    if rec.get(\'action\')==\'FULLSPAN_CYCLE\':
        out=f\"{rec.get(\'queue\', \'\')} {rec.get(\'next_step\', \'\')}\"; break
print(out)" )"
fi

if [[ -n "$current_queue" && -f "$HEARTBEAT_FILE" ]]; then
  heartbeat_for_current="$(python3 - "$current_queue" "$HEARTBEAT_FILE" <<'PY'
import json
import sys
from pathlib import Path

q = Path(sys.argv[1]).as_posix()
f = Path(sys.argv[2])
if not f.exists():
    raise SystemExit(0)

try:
    data = json.loads(f.read_text(encoding='utf-8'))
except Exception:
    raise SystemExit(0)

item = data.get(q)
if not item:
    item = data.get(('artifacts/' + q.lstrip('/')))
if not item:
    raise SystemExit(0)

print('pending={pending} completed={completed} running={running} stalled={stalled} rate_per_min={rate:.3f} eta_min={eta} stale_sec={stale:.1f} done={done}'.format(
    pending=item.get('pending'),
    completed=item.get('completed'),
    running=item.get('running'),
    stalled=item.get('stalled'),
    rate=float(item.get('rate_per_min', 0.0)),
    eta=item.get('eta_min'),
    stale=float(item.get('stale_sec', 0.0)),
    done=int(bool(item.get('done'))),
))
PY
)"
fi

state_verdict="none"
state_pass_count="0"
state_run_group_count="0"
state_confirm_count="0"
state_strict_gate_status=""
state_strict_gate_reason=""
if [[ -n "$current_queue" && -f "$FULLSPAN_DECISION_STATE_FILE" ]]; then
  state_fields="$(python3 - "$FULLSPAN_DECISION_STATE_FILE" "$current_queue" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
queue = Path(sys.argv[2]).as_posix() if len(sys.argv) > 2 else sys.argv[2]
if not path.exists():
    print('none 0 0 0')
    raise SystemExit(0)
try:
    data = json.loads(path.read_text(encoding='utf-8'))
    state = data.get('queues', {}).get(queue, {})
except Exception:
    print('none 0 0 0')
    raise SystemExit(0)
if not state:
    print('none 0 0 0')
    raise SystemExit(0)
print(f"{state.get('promotion_verdict', 'none')} {state.get('strict_pass_count', 0)} {state.get('strict_run_group_count', 0)} {state.get('confirm_count', 0)} {state.get('strict_gate_status', '')} {state.get('strict_gate_reason', '')}")
PY
  )"
  read -r state_verdict state_pass_count state_run_group_count state_confirm_count state_strict_gate_status state_strict_gate_reason <<< "$state_fields"
fi

fullspan_state_pass_metric="0"
fullspan_state_reject_metric="0"
fullspan_state_confirm_pending_metric="0"
fullspan_state_promo_metric="0"
fullspan_state_cycle_since_last_pass="0"
fullspan_state_last_strict_pass_epoch="0"
if [[ -f "$FULLSPAN_DECISION_STATE_FILE" ]]; then
  fullspan_metrics="$(python3 - "$FULLSPAN_DECISION_STATE_FILE" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
if not path.exists():
    print('0 0 0 0 0 0')
    raise SystemExit(0)
try:
    data = json.loads(path.read_text(encoding='utf-8'))
    m = data.get('runtime_metrics', {}) if isinstance(data, dict) else {}
except Exception:
    print('0 0 0 0 0 0')
    raise SystemExit(0)
print(f"{m.get('strict_fullspan_pass_count', 0)} {m.get('strict_fullspan_reject_count', 0)} {m.get('confirm_pending_count', 0)} {m.get('promotion_eligible_count', 0)} {m.get('cycles_since_last_strict_pass', 0)} {m.get('last_strict_pass_epoch', 0)}")
PY
  )"
  read -r fullspan_state_pass_metric fullspan_state_reject_metric fullspan_state_confirm_pending_metric fullspan_state_promo_metric fullspan_state_cycle_since_last_pass fullspan_state_last_strict_pass_epoch <<< "$fullspan_metrics"
fi
  
orphans="0"
if [[ -f "$ORPHAN_FILE" ]]; then
  orphans="$(python3 - "$ORPHAN_FILE" <<'PY'
import csv
import sys
from pathlib import Path

p = Path(sys.argv[1])
try:
    with p.open(newline='', encoding='utf-8') as f:
        print(sum(1 for _ in csv.DictReader(f)))
except Exception:
    print(0)
PY
)"
fi

vps_processes="$(ssh -o BatchMode=yes -o StrictHostKeyChecking=no -o ConnectTimeout=6 "$SERVER_USER@$SERVER_IP" 'ps aux | egrep -c "watch_wfa_queue|run_wfa_queue|run_wfa_fullcpu|python.*walk_forward" || true' 2>/dev/null || echo 0)"

active_queues="$(python3 - <<'PY'
import csv
from collections import Counter
from pathlib import Path
root = Path('/home/claudeuser/coint4/coint4/artifacts/wfa/aggregate')
out = []
for p in sorted(root.rglob('run_queue.csv')):
    try:
        rows = list(csv.DictReader(p.open(newline='')))
    except Exception:
        continue
    st = Counter((r.get('status') or '').strip().lower() for r in rows)
    pending = st.get('planned', 0) + st.get('running', 0) + st.get('stalled', 0) + st.get('failed', 0) + st.get('error', 0)
    if pending > 0:
        out.append((p.as_posix(), st.get('planned', 0), st.get('running', 0), st.get('stalled', 0), st.get('completed', 0), st.get('failed', 0) + st.get('error', 0)))
out = sorted(out, key=lambda x: -(x[1] + x[2] + x[3] + x[5]))
if not out:
    print('none')
else:
    for i, item in enumerate(out[:6], 1):
        pth, planned, running, stalled, completed, failed = item
        print(f"{i}) {pth} P={planned} R={running} S={stalled} F={failed} C={completed}")
PY
)"

human_payload="$(python3 - "$ROOT_DIR" "$current_queue" "$REPORT_STATE_FILE" "$state_verdict" "$vps_processes" "$qline" <<'PY'
import csv
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
queue = (sys.argv[2] or '').strip()
state_path = Path(sys.argv[3])
state_verdict = (sys.argv[4] or '').strip()
try:
    vps_processes = int(float(sys.argv[5]))
except Exception:
    vps_processes = 0
qline = (sys.argv[6] or '').strip()

def queue_counts(queue_rel: str):
    if not queue_rel:
        return {"pending": 0, "completed": 0, "stalled": 0, "total": 0}
    p = Path(queue_rel)
    if not p.is_absolute():
        p = root / p
    if not p.exists():
        return {"pending": 0, "completed": 0, "stalled": 0, "total": 0}
    try:
        rows = list(csv.DictReader(p.open(newline='', encoding='utf-8')))
    except Exception:
        return {"pending": 0, "completed": 0, "stalled": 0, "total": 0}
    planned = running = stalled = failed = completed = 0
    for r in rows:
        s = (r.get('status') or '').strip().lower()
        if s == 'planned': planned += 1
        elif s == 'running': running += 1
        elif s == 'stalled': stalled += 1
        elif s in {'failed','error'}: failed += 1
        elif s == 'completed': completed += 1
    return {
        "pending": planned + running + stalled + failed,
        "completed": completed,
        "stalled": stalled,
        "total": len(rows),
    }

cur = queue_counts(queue)
prev = {"pending": cur["pending"], "completed": cur["completed"], "stalled": cur["stalled"]}
if state_path.exists():
    try:
        prev_raw = json.loads(state_path.read_text(encoding='utf-8'))
        if isinstance(prev_raw, dict):
            prev = {
                "pending": int(prev_raw.get("pending", cur["pending"]) or 0),
                "completed": int(prev_raw.get("completed", cur["completed"]) or 0),
                "stalled": int(prev_raw.get("stalled", cur["stalled"]) or 0),
            }
    except Exception:
        pass

state_path.parent.mkdir(parents=True, exist_ok=True)
state_path.write_text(json.dumps({
    "pending": cur["pending"],
    "completed": cur["completed"],
    "stalled": cur["stalled"],
}, ensure_ascii=False, indent=2), encoding='utf-8')

delta_completed = cur["completed"] - prev["completed"]
delta_stalled = cur["stalled"] - prev["stalled"]

goal = "достигнута" if state_verdict == "PROMOTE_ELIGIBLE" else "не достигнута"

blockers = []
if cur["pending"] > 0 and vps_processes <= 0:
    blockers.append("нет активных процессов на VPS при pending>0")
if state_verdict == "REJECT":
    blockers.append("текущая очередь в fail-closed/REJECT")
if cur["stalled"] > 0:
    blockers.append(f"stalled={cur['stalled']}")
if not blockers:
    if cur["pending"] > 0:
        blockers.append("идёт обработка очереди")
    else:
        blockers.append("pending=0, ждём новых кандидатов")

next_action = "продолжаю автономный search по fullspan-контракту"
if cur["pending"] > 0 and vps_processes <= 0:
    next_action = "автоперезапуск/repair через watchdog до запуска runner на VPS"
elif state_verdict == "PROMOTE_ELIGIBLE":
    next_action = "готовлю confirm/cutover шаг по контракту"

need_user = "да" if (state_verdict == "REJECT" and cur["pending"] <= 0 and "idle" in qline.lower()) else "нет"

print(json.dumps({
    "goal": goal,
    "blockers": blockers[:2],
    "delta_completed": delta_completed,
    "delta_stalled": delta_stalled,
    "next_action": next_action,
    "need_user": need_user,
    "queue": queue or "none",
    "pending": cur["pending"],
    "completed": cur["completed"],
    "stalled": cur["stalled"],
}, ensure_ascii=False))
PY
)"

goal_line="$(python3 -c 'import json,sys; d=json.loads(sys.argv[1]); print(d.get("goal","не достигнута"))' "$human_payload")"
blocker_line="$(python3 -c 'import json,sys; d=json.loads(sys.argv[1]); b=d.get("blockers",[]); print("; ".join(b) if b else "нет")' "$human_payload")"
delta_completed="$(python3 -c 'import json,sys; d=json.loads(sys.argv[1]); print(int(d.get("delta_completed",0)))' "$human_payload")"
delta_stalled="$(python3 -c 'import json,sys; d=json.loads(sys.argv[1]); print(int(d.get("delta_stalled",0)))' "$human_payload")"
next_action_line="$(python3 -c 'import json,sys; d=json.loads(sys.argv[1]); print(d.get("next_action","продолжаю автономный цикл"))' "$human_payload")"
need_user_line="$(python3 -c 'import json,sys; d=json.loads(sys.argv[1]); print(d.get("need_user","нет"))' "$human_payload")"
queue_line="$(python3 -c 'import json,sys; d=json.loads(sys.argv[1]); print(d.get("queue","none"))' "$human_payload")"

printf '📌 10m human report\n'
printf 'Цель: %s\n' "$goal_line"
printf 'Что мешает: %s\n' "$blocker_line"
printf 'Прогресс за 10 минут: %+d completed / %+d stalled\n' "$delta_completed" "$delta_stalled"
printf 'Текущая очередь: %s\n' "$queue_line"
printf 'Что делаю дальше: %s\n' "$next_action_line"
printf 'Нужен ли ты: %s\n' "$need_user_line"
