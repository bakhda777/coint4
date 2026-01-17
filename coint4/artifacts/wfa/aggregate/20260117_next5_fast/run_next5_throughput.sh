#!/usr/bin/env bash
set -euo pipefail

QUEUE_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "${QUEUE_DIR}/../../../.." && pwd)"
QUEUE_PATH="${QUEUE_DIR}/run_queue_next5_fast.csv"
RUN_LOG="${QUEUE_DIR}/run_queue.log"
WATCH_LOG="${QUEUE_DIR}/run_queue.watch.log"
PID_FILE="${QUEUE_DIR}/run_queue.pid"
HEARTBEAT=30

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export COINT_WFA_NO_MEMORY_MAP=1
export COINT_FILTER_BACKEND=processes

if [[ ! -f "$QUEUE_PATH" ]]; then
  echo "Queue not found: $QUEUE_PATH" >&2
  exit 1
fi

if ! pgrep -f "coint2 walk-forward" >/dev/null 2>&1; then
  python3 - <<'PY' "$QUEUE_PATH"
import csv
import sys
from pathlib import Path

queue_path = Path(sys.argv[1])
rows = []
changed = 0

with queue_path.open(newline="") as handle:
    reader = csv.DictReader(handle)
    fieldnames = reader.fieldnames
    for row in reader:
        if row.get("status") == "running":
            row["status"] = "stalled"
            changed += 1
        rows.append(row)

if changed:
    with queue_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Reset {changed} running -> stalled in {queue_path}")
PY
fi

python3 - <<'PY' "$QUEUE_PATH" "$ROOT_DIR"
import csv
import re
import sys
from pathlib import Path

queue_path = Path(sys.argv[1])
root_dir = Path(sys.argv[2])

bad = []
missing = []
with queue_path.open(newline="") as handle:
    reader = csv.DictReader(handle)
    for row in reader:
        config_path = row.get("config_path")
        if not config_path:
            continue
        path = (root_dir / config_path).resolve()
        if not path.exists():
            missing.append(config_path)
            continue
        max_steps = None
        with path.open() as cfg:
            for line in cfg:
                match = re.match(r"\s*max_steps:\s*(\d+)", line)
                if match:
                    max_steps = int(match.group(1))
                    break
        if max_steps is None:
            continue
        if max_steps > 5:
            bad.append((config_path, max_steps))

if missing:
    print("WARN: configs missing:", ", ".join(missing))
if bad:
    print("ERROR: max_steps > 5:", ", ".join(f"{p}={v}" for p, v in bad))
    sys.exit(2)
PY

timestamp() {
  date -u +"%Y-%m-%dT%H:%M:%SZ"
}

log_watch() {
  echo "$(timestamp) $*" | tee -a "$WATCH_LOG"
}

log_watch "START queue=$QUEUE_PATH parallel=5 heartbeat=${HEARTBEAT}s"

PYTHONPATH=src "$ROOT_DIR/.venv/bin/python" "$ROOT_DIR/scripts/optimization/run_wfa_queue.py" \
  --queue "$QUEUE_PATH" \
  --statuses planned,stalled \
  --parallel 5 \
  >"$RUN_LOG" 2>&1 &

RUN_PID=$!
echo "$RUN_PID" > "$PID_FILE"

while kill -0 "$RUN_PID" 2>/dev/null; do
  ps_out="$(ps -eo pid,pcpu,cmd --no-headers)"
  cpu_total="$(echo "$ps_out" | awk '/coint2 walk-forward/ {sum+=$2} END {printf "%.1f", sum+0}')"
  proc_count="$(echo "$ps_out" | awk '/coint2 walk-forward/ {count++} END {print count+0}')"
  log_watch "HEARTBEAT run_pid=$RUN_PID coint2_procs=$proc_count cpu_total=${cpu_total}%"
  sleep "$HEARTBEAT"
 done

RC=0
wait "$RUN_PID" || RC=$?
log_watch "DONE rc=$RC"

python3 - <<'PY' "$QUEUE_PATH"
import csv
import sys
from pathlib import Path

queue_path = Path(sys.argv[1])
rows = []
changed = 0
with queue_path.open(newline="") as handle:
    reader = csv.DictReader(handle)
    fieldnames = reader.fieldnames
    for row in reader:
        if row.get("status") == "running":
            row["status"] = "stalled"
            changed += 1
        rows.append(row)

if changed:
    with queue_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Updated {changed} running -> stalled in {queue_path}")
PY
