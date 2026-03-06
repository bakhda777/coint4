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
PROCESS_SLO_STATE_FILE="$STATE_DIR/process_slo_state.json"
CAPACITY_STATE_FILE="$STATE_DIR/capacity_controller_state.json"
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

active_queues="$(python3 - "$QUEUE_ROOT" <<'PY'
import csv
from collections import Counter
from pathlib import Path
import sys

root = Path(sys.argv[1])
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

leaders_line="$(python3 - "$FULLSPAN_DECISION_STATE_FILE" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
if not path.exists():
    print('нет strict-лидеров в fullspan state')
    raise SystemExit(0)

try:
    data = json.loads(path.read_text(encoding='utf-8'))
except Exception:
    print('нет strict-лидеров в fullspan state')
    raise SystemExit(0)

queues = data.get('queues', {}) if isinstance(data, dict) else {}
rows = []
for queue_rel, entry in queues.items():
    if not isinstance(entry, dict):
        continue
    try:
        sp = int(float(entry.get('strict_pass_count', 0) or 0))
        rg = int(float(entry.get('strict_run_group_count', 0) or 0))
        cc = int(float(entry.get('confirm_count', 0) or 0))
        score = float(entry.get('top_score', 0) or 0)
    except Exception:
        sp, rg, cc, score = 0, 0, 0, 0.0
    if sp <= 0:
        continue
    rows.append((
        score,
        sp,
        rg,
        cc,
        str(entry.get('promotion_verdict', 'ANALYZE') or 'ANALYZE'),
        str(entry.get('top_run_group', '') or ''),
        str(entry.get('top_variant', '') or ''),
        str(queue_rel),
    ))

if not rows:
    print('нет strict-лидеров в fullspan state')
    raise SystemExit(0)

rows.sort(key=lambda x: (x[0], x[1], x[2], x[3]), reverse=True)
out = []
for score, sp, rg, cc, verdict, top_rg, top_var, queue_rel in rows[:3]:
    label = top_var or queue_rel
    rg_label = top_rg or queue_rel
    out.append(f"{rg_label}/{label} score={score:.3f} passes={sp} run_groups={rg} confirm={cc} verdict={verdict}")
print(' | '.join(out))
PY
)"

process_funnel_line="нет данных process_slo_guard"
process_kpi_line="нет данных process_slo_guard"
process_runtime_line="нет данных process_slo_guard"
process_alerts_line="нет"
if [[ -f "$PROCESS_SLO_STATE_FILE" ]]; then
  process_payload="$(python3 - "$PROCESS_SLO_STATE_FILE" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
try:
    data = json.loads(path.read_text(encoding='utf-8'))
except Exception:
    print('нет данных process_slo_guard')
    print('нет данных process_slo_guard')
    print('нет данных process_slo_guard')
    print('нет')
    raise SystemExit(0)

funnel = data.get('funnel', {}) if isinstance(data, dict) else {}
kpi = data.get('kpi', {}) if isinstance(data, dict) else {}
alerts = data.get('alerts', []) if isinstance(data, dict) else []
queue = data.get('queue', {}) if isinstance(data, dict) else {}

def to_int(v):
    try:
        return int(float(v or 0))
    except Exception:
        return 0

def to_float(v):
    try:
        return float(v or 0.0)
    except Exception:
        return 0.0

lead = kpi.get('lead_time_to_promote_min')
lead_text = '-' if lead is None else f"{to_float(lead):.2f}m"

print(
    "generated={g} -> executable={e} -> completed={c} -> strict_pass={sp} -> confirm_ready={cr} -> promote={p}".format(
        g=to_int(funnel.get('generated')),
        e=to_int(funnel.get('executable')),
        c=to_int(funnel.get('completed')),
        sp=to_int(funnel.get('strict_pass')),
        cr=to_int(funnel.get('confirm_ready')),
        p=to_int(funnel.get('promote_eligible')),
    )
)
print(
    "throughput={t:.2f}/h strict_pass_rate={spr:.4f} confirm_conv={ccr:.4f} promote_conv={pcr:.4f} lead_to_promote={lead}".format(
        t=to_float(kpi.get('throughput_completed_per_hour')),
        spr=to_float(kpi.get('strict_pass_rate')),
        ccr=to_float(kpi.get('confirm_conversion_rate')),
        pcr=to_float(kpi.get('promote_conversion_rate')),
        lead=lead_text,
    )
)
local_runners = to_int(queue.get('local_runner_count'))
executable_pending = to_int(queue.get('executable_pending'))
pending = to_int(queue.get('pending'))
idle_flag = 1 if pending > 0 and executable_pending > 0 and local_runners <= 0 else 0
print(
    "runtime pending={p} executable_pending={ep} local_runners={lr} idle_with_executable_pending={idle} fastlane_replay_pending={fastlane} hot_standby_active={standby} duty30m={duty:.2f} ready_buffer_policy_mismatch_count={mismatch} winner_parent_duplication_rate={dup:.2f} metrics_missing_abort_count_30m={mm_abort} winner_proximate_dispatch_count_30m={winner_dispatch}".format(
        p=pending,
        ep=executable_pending,
        lr=local_runners,
        idle=idle_flag,
        fastlane=to_int((data.get('runtime', {}) or {}).get('fastlane_replay_pending')),
        standby=int(bool((data.get('runtime', {}) or {}).get('hot_standby_active'))),
        duty=to_float((data.get('runtime', {}) or {}).get('vps_duty_cycle_30m')),
        mismatch=to_int((data.get('runtime', {}) or {}).get('ready_buffer_policy_mismatch_count')),
        dup=to_float((data.get('runtime', {}) or {}).get('winner_parent_duplication_rate')),
        mm_abort=to_int((data.get('runtime', {}) or {}).get('metrics_missing_abort_count_30m')),
        winner_dispatch=to_int((data.get('runtime', {}) or {}).get('winner_proximate_dispatch_count_30m')),
    )
)
if not isinstance(alerts, list) or not alerts:
    print('нет')
else:
    head = []
    for item in alerts[:3]:
        if not isinstance(item, dict):
            continue
        code = str(item.get('code') or 'UNKNOWN')
        msg = str(item.get('message') or '').strip()
        head.append(f"{code}: {msg}" if msg else code)
    print(' | '.join(head) if head else 'нет')
PY
)"
  process_funnel_line="$(printf '%s\n' "$process_payload" | sed -n '1p')"
  process_kpi_line="$(printf '%s\n' "$process_payload" | sed -n '2p')"
  process_runtime_line="$(printf '%s\n' "$process_payload" | sed -n '3p')"
  process_alerts_line="$(printf '%s\n' "$process_payload" | sed -n '4p')"
fi

capacity_line="нет данных capacity_controller"
if [[ -f "$CAPACITY_STATE_FILE" ]]; then
  capacity_line="$(python3 - "$CAPACITY_STATE_FILE" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
try:
    data = json.loads(path.read_text(encoding='utf-8'))
except Exception:
    print('нет данных capacity_controller')
    raise SystemExit(0)

policy = data.get('policy', {}) if isinstance(data, dict) else {}
remote = data.get('remote', {}) if isinstance(data, dict) else {}
print(
    "search[min,max]=[{smin},{smax}] confirm[min,max]=[{cmin},{cmax}] dispatches={d} lane_active={la} remote_load1={load} remote_runners={rr}".format(
        smin=int(float(policy.get('search_parallel_min', 0) or 0)),
        smax=int(float(policy.get('search_parallel_max', 0) or 0)),
        cmin=int(float(policy.get('confirm_parallel_min', 0) or 0)),
        cmax=int(float(policy.get('confirm_parallel_max', 0) or 0)),
        d=int(float(policy.get('confirm_dispatches_per_cycle', 0) or 0)),
        la=int(float(policy.get('confirm_lane_max_active', 0) or 0)),
        load=remote.get('load1', 'n/a'),
        rr=remote.get('runner_count', 'n/a'),
    )
)
PY
)"
fi

surrogate_gate_line="нет данных surrogate runtime"
surrogate_evidence_line="нет данных surrogate runtime"
surrogate_branch_line="нет данных surrogate runtime"
runtime_observability_line="нет данных runtime observability"
probe_json="$(python3 "$ROOT_DIR/scripts/optimization/probe_autonomous_markers.py" --root "$ROOT_DIR" --ensure-process-slo never --format json 2>/dev/null || true)"
if [[ -n "$probe_json" ]]; then
  surrogate_payload="$(python3 - "$probe_json" <<'PY'
import json
import sys

try:
    data = json.loads(sys.argv[1])
except Exception:
    print("нет данных surrogate runtime")
    print("нет данных surrogate runtime")
    print("нет данных surrogate runtime")
    print("нет данных runtime observability")
    raise SystemExit(0)

markers = data.get("markers", {}) if isinstance(data, dict) else {}
sur = markers.get("surrogate_runtime", {}) if isinstance(markers, dict) else {}
gate = sur.get("gate_surrogate", {}) if isinstance(sur, dict) else {}
directive = sur.get("directive_overlay", {}) if isinstance(sur, dict) else {}
combined = (sur.get("evidence", {}) or {}).get("combined", {}) if isinstance(sur, dict) else {}
branch = sur.get("branch_health", {}) if isinstance(sur, dict) else {}
runtime_obs = markers.get("runtime_observability", {}) if isinstance(markers, dict) else {}

def to_int(v):
    try:
        return int(float(v or 0))
    except Exception:
        return 0

gate_counts = gate.get("decision_counts", {}) if isinstance(gate.get("decision_counts"), dict) else {}
gate_fresh = gate.get("freshness", {}) if isinstance(gate.get("freshness"), dict) else {}
directive_fresh = directive.get("freshness", {}) if isinstance(directive.get("freshness"), dict) else {}

print(
    "mode={mode} fresh={fresh} age={age}s allow/refine/reject={allow}/{refine}/{reject} directive_mode={dmode} directive_fresh={dfresh}".format(
        mode=str(gate.get("mode") or "-"),
        fresh=int(bool(gate_fresh.get("fresh"))),
        age=to_int(gate_fresh.get("age_sec")),
        allow=to_int(gate_counts.get("allow")),
        refine=to_int(gate_counts.get("refine")),
        reject=to_int(gate_counts.get("reject")),
        dmode=str(directive.get("gate_surrogate_mode") or directive.get("mode") or "-"),
        dfresh=int(bool(directive_fresh.get("fresh"))),
    )
)
print(
    "evidence reject/refine/allow={r}/{f}/{a}".format(
        r=to_int((combined.get("SURROGATE_REJECT") or {}).get("count") if isinstance(combined.get("SURROGATE_REJECT"), dict) else 0),
        f=to_int((combined.get("SURROGATE_REFINE") or {}).get("count") if isinstance(combined.get("SURROGATE_REFINE"), dict) else 0),
        a=to_int((combined.get("SURROGATE_ALLOW") or {}).get("count") if isinstance(combined.get("SURROGATE_ALLOW"), dict) else 0),
    )
)
print(
    "branch status={status} broken={broken} eligible_refine_reject={eligible} observed={observed} reason={reason}".format(
        status=str(branch.get("status") or "-"),
        broken=int(bool(branch.get("broken_branch"))),
        eligible=to_int(branch.get("eligible_refine_reject_count")),
        observed=to_int(branch.get("observed_refine_reject_evidence")),
        reason=str(branch.get("reason") or "-"),
    )
)

def metric_value(name):
    item = runtime_obs.get(name, {}) if isinstance(runtime_obs, dict) else {}
    if isinstance(item, dict):
        return to_int(item.get("value"))
    return to_int(item)

print(
    "ready_buffer_depth={ready} cold_fail_active_count={cold} remote_child_process_count={child} remote_active_queue_jobs={remote} cpu_busy_without_queue_job={cpu_busy} surrogate_idle_override_count={idle} overlap_dispatch_count={overlap}".format(
        ready=metric_value("ready_buffer_depth"),
        cold=metric_value("cold_fail_active_count"),
        child=metric_value("remote_child_process_count"),
        remote=metric_value("remote_active_queue_jobs"),
        cpu_busy=metric_value("cpu_busy_without_queue_job"),
        idle=metric_value("surrogate_idle_override_count"),
        overlap=metric_value("overlap_dispatch_count"),
    )
)
PY
)"
  surrogate_gate_line="$(printf '%s\n' "$surrogate_payload" | sed -n '1p')"
  surrogate_evidence_line="$(printf '%s\n' "$surrogate_payload" | sed -n '2p')"
  surrogate_branch_line="$(printf '%s\n' "$surrogate_payload" | sed -n '3p')"
  runtime_observability_line="$(printf '%s\n' "$surrogate_payload" | sed -n '4p')"
fi

printf '📌 10m human report\n'
printf 'Цель: %s\n' "$goal_line"
printf 'Что мешает: %s\n' "$blocker_line"
printf 'Прогресс за 10 минут: %+d completed / %+d stalled\n' "$delta_completed" "$delta_stalled"
printf 'Текущая очередь: %s\n' "$queue_line"
printf 'Лидеры сейчас: %s\n' "$leaders_line"
printf 'Процесс (воронка): %s\n' "$process_funnel_line"
printf 'Процесс (KPI): %s\n' "$process_kpi_line"
printf 'Процесс (runtime): %s\n' "$process_runtime_line"
printf 'Процесс (alerts): %s\n' "$process_alerts_line"
printf 'Capacity policy: %s\n' "$capacity_line"
printf 'Runtime observability: %s\n' "$runtime_observability_line"
printf 'Surrogate gate: %s\n' "$surrogate_gate_line"
printf 'Surrogate evidence: %s\n' "$surrogate_evidence_line"
printf 'Surrogate branch: %s\n' "$surrogate_branch_line"
printf 'Что делаю дальше: %s\n' "$next_action_line"
printf 'Нужен ли ты: %s\n' "$need_user_line"
