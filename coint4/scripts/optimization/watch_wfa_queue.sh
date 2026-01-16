#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: watch_wfa_queue.sh --queue <path> [options]

Options:
  --queue <path>           Path to run_queue.csv (relative to repo root or absolute).
  --parallel <n>           Number of concurrent runs (default: 1).
  --heartbeat <sec>        Heartbeat interval in seconds (default: 30).
  --idle-minutes <min>     Minutes without workers before marking idle (default: 5).
  --cpu-threshold <pct>    CPU threshold (sum %) for informational logging (default: 50).
  --on-done-cmd <cmd>      Command to run after queue completes (optional).
  --on-done-prompt-file <path> Prompt file for codex exec (optional).
  --on-done-log <path>     Log file for on-done output (optional).
  --codex-bin <path>       Codex binary (default: env CODEX_BIN or "codex").
  --no-reset-running       Do not reset stale 'running' statuses on startup.
  --help                   Show this help.
EOF
}

QUEUE_PATH=""
PARALLEL=1
HEARTBEAT=30
IDLE_MINUTES=5
CPU_THRESHOLD=50
ON_DONE_CMD=""
ON_DONE_PROMPT_FILE=""
ON_DONE_LOG=""
CODEX_BIN="${CODEX_BIN:-codex}"
RESET_RUNNING=1

while [[ $# -gt 0 ]]; do
  case "$1" in
    --queue)
      QUEUE_PATH="$2"
      shift 2
      ;;
    --parallel)
      PARALLEL="$2"
      shift 2
      ;;
    --heartbeat)
      HEARTBEAT="$2"
      shift 2
      ;;
    --idle-minutes)
      IDLE_MINUTES="$2"
      shift 2
      ;;
    --cpu-threshold)
      CPU_THRESHOLD="$2"
      shift 2
      ;;
    --on-done-cmd)
      ON_DONE_CMD="$2"
      shift 2
      ;;
    --on-done-prompt-file)
      ON_DONE_PROMPT_FILE="$2"
      shift 2
      ;;
    --on-done-log)
      ON_DONE_LOG="$2"
      shift 2
      ;;
    --codex-bin)
      CODEX_BIN="$2"
      shift 2
      ;;
    --no-reset-running)
      RESET_RUNNING=0
      shift 1
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "Unknown аргумент: $1"
      usage
      exit 1
      ;;
  esac
done

if [[ -z "$QUEUE_PATH" ]]; then
  echo "Нужно указать --queue"
  usage
  exit 1
fi

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
if [[ "$QUEUE_PATH" != /* ]]; then
  QUEUE_PATH="$ROOT_DIR/$QUEUE_PATH"
fi
if [[ -n "$ON_DONE_PROMPT_FILE" && "$ON_DONE_PROMPT_FILE" != /* ]]; then
  ON_DONE_PROMPT_FILE="$ROOT_DIR/$ON_DONE_PROMPT_FILE"
fi
if [[ -n "$ON_DONE_LOG" && "$ON_DONE_LOG" != /* ]]; then
  ON_DONE_LOG="$ROOT_DIR/$ON_DONE_LOG"
fi

if [[ ! -f "$QUEUE_PATH" ]]; then
  echo "run_queue.csv не найден: $QUEUE_PATH"
  exit 1
fi

QUEUE_DIR="$(cd "$(dirname "$QUEUE_PATH")" && pwd)"
RUN_LOG="$QUEUE_DIR/run_queue.log"
WATCH_LOG="$QUEUE_DIR/run_queue.watch.log"
PID_FILE="$QUEUE_DIR/run_queue.pid"

timestamp() {
  date -u +"%Y-%m-%dT%H:%M:%SZ"
}

log_watch() {
  echo "$(timestamp) $*" | tee -a "$WATCH_LOG"
}

resolve_results_dir_path() {
  local results_dir="$1"
  if [[ -z "$results_dir" ]]; then
    echo ""
    return
  fi
  if [[ "$results_dir" == /* ]]; then
    echo "$results_dir"
  else
    echo "$ROOT_DIR/$results_dir"
  fi
}

list_running_results_dirs() {
  python3 - <<'PY' "$QUEUE_PATH"
import csv
import sys
from pathlib import Path

queue_path = Path(sys.argv[1])
if not queue_path.exists():
    sys.exit(0)

with queue_path.open(newline="") as handle:
    reader = csv.DictReader(handle)
    for row in reader:
        if row.get("status") == "running":
            results_dir = row.get("results_dir") or ""
            results_dir = results_dir.strip()
            if results_dir:
                print(results_dir)
PY
}

select_freshest_pid() {
  local best_pid=""
  local best_etime=""
  local pid=""
  for pid in "$@"; do
    local etime=""
    etime="$(ps -p "$pid" -o etimes= 2>/dev/null | awk '{print $1+0}' || true)"
    if [[ -z "$etime" ]]; then
      continue
    fi
    if [[ -z "$best_etime" || "$etime" -lt "$best_etime" ]]; then
      best_etime="$etime"
      best_pid="$pid"
    fi
  done
  echo "$best_pid"
}

check_max_steps() {
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
}

reset_running_statuses() {
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
}

if [[ "$RESET_RUNNING" == "1" ]]; then
  if ! pgrep -f "coint2 walk-forward" >/dev/null 2>&1; then
    reset_running_statuses
  fi
fi

check_max_steps

log_watch "START queue=$QUEUE_PATH parallel=$PARALLEL heartbeat=${HEARTBEAT}s idle_minutes=${IDLE_MINUTES} cpu_threshold=${CPU_THRESHOLD}"

PYTHONPATH=src "$ROOT_DIR/.venv/bin/python" "$ROOT_DIR/scripts/optimization/run_wfa_queue.py" \
  --queue "$QUEUE_PATH" \
  --statuses planned,stalled \
  --parallel "$PARALLEL" \
  >"$RUN_LOG" 2>&1 &

RUN_PID=$!
echo "$RUN_PID" > "$PID_FILE"

idle_seconds=0

while kill -0 "$RUN_PID" 2>/dev/null; do
  ps_out="$(ps -eo pid,pcpu,etimes,cmd --no-headers)"
  cpu_total="$(echo "$ps_out" | awk '/coint2 walk-forward/ {sum+=$2} END {printf "%.1f", sum+0}')"
  proc_count="$(echo "$ps_out" | awk '/coint2 walk-forward/ {count++} END {print count+0}')"
  watcher_cpu="$(ps -p "$$" -o pcpu= 2>/dev/null | awk '{print $1+0}' || true)"

  running_dirs=()
  while IFS= read -r dir; do
    [[ -n "$dir" ]] && running_dirs+=("$dir")
  done < <(list_running_results_dirs)

  worker_pid_candidates=()
  for dir in "${running_dirs[@]}"; do
    results_dir_path="$(resolve_results_dir_path "$dir")"
    pid_file="$results_dir_path/worker.pid"
    if [[ -f "$pid_file" ]]; then
      pid="$(tr -d '[:space:]' < "$pid_file")"
      if [[ -n "$pid" ]]; then
        cmd="$(ps -p "$pid" -o cmd= 2>/dev/null || true)"
        if [[ -n "$cmd" && "$cmd" == *"coint2 walk-forward"* && "$cmd" == *"$dir"* ]]; then
          worker_pid_candidates+=("$pid")
        fi
      fi
    fi
  done

  worker_pid=""
  if [[ ${#worker_pid_candidates[@]} -gt 0 ]]; then
    worker_pid="$(select_freshest_pid "${worker_pid_candidates[@]}")"
  elif [[ ${#running_dirs[@]} -gt 0 ]]; then
    freshest_etime=""
    while read -r pid pcpu etimes cmd; do
      [[ "$cmd" != *"coint2 walk-forward"* ]] && continue
      for dir in "${running_dirs[@]}"; do
        [[ "$cmd" == *"$dir"* ]] || continue
        if [[ -z "$freshest_etime" || "$etimes" -lt "$freshest_etime" ]]; then
          freshest_etime="$etimes"
          worker_pid="$pid"
        fi
        break
      done
    done <<< "$ps_out"
  fi

  worker_cpu="0"
  if [[ -n "$worker_pid" ]]; then
    worker_cpu="$(ps -p "$worker_pid" -o pcpu= 2>/dev/null | awk '{print $1+0}' || true)"
  fi

  if [[ -z "$worker_pid" ]]; then
    idle_seconds=$((idle_seconds + HEARTBEAT))
  else
    idle_seconds=0
  fi

  log_watch "HEARTBEAT run_pid=$RUN_PID run_cpu=${worker_cpu}% watcher_cpu=${watcher_cpu}% worker_pid=${worker_pid:-none} worker_cpu=${worker_cpu}% coint2_procs=$proc_count cpu_total=${cpu_total}% idle_seconds=$idle_seconds"

  if [[ $idle_seconds -ge $((IDLE_MINUTES * 60)) ]]; then
    log_watch "IDLE no_workers for ${idle_seconds}s (cpu_threshold=${CPU_THRESHOLD}% informational)"
  fi

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

run_cmd_with_log() {
  local cmd="$1"
  local log_path="$2"
  local rc=0

  set +e
  if [[ -n "$log_path" ]]; then
    bash -lc "$cmd" |& tee -a "$log_path"
    rc=${PIPESTATUS[0]}
  else
    bash -lc "$cmd"
    rc=$?
  fi
  set -e
  return "$rc"
}

run_codex_prompt() {
  local prompt_file="$1"
  local log_path="$2"
  local prompt=""
  local rc=0

  if [[ ! -f "$prompt_file" ]]; then
    log_watch "ON_DONE_PROMPT missing: $prompt_file"
    return 1
  fi
  if ! command -v "$CODEX_BIN" >/dev/null 2>&1; then
    log_watch "ON_DONE_PROMPT codex not found: $CODEX_BIN"
    return 1
  fi

  prompt="$(cat "$prompt_file")"
  prompt="${prompt//RUN_DIR/$QUEUE_DIR}"
  log_watch "ON_DONE_PROMPT start: $prompt_file"

  set +e
  if [[ -n "$log_path" ]]; then
    "$CODEX_BIN" exec --full-auto "$prompt" |& tee -a "$log_path"
    rc=${PIPESTATUS[0]}
  else
    "$CODEX_BIN" exec --full-auto "$prompt"
    rc=$?
  fi
  set -e

  if [[ "$rc" -ne 0 ]]; then
    log_watch "ON_DONE_PROMPT failed rc=$rc"
  fi
  return "$rc"
}

if [[ -n "$ON_DONE_PROMPT_FILE" ]]; then
  if [[ -z "$ON_DONE_LOG" ]]; then
    ON_DONE_LOG="$QUEUE_DIR/codex_on_done.log"
  fi
  run_codex_prompt "$ON_DONE_PROMPT_FILE" "$ON_DONE_LOG" || true
fi

if [[ -n "$ON_DONE_CMD" ]]; then
  log_watch "ON_DONE_CMD start: $ON_DONE_CMD"
  if ! run_cmd_with_log "$ON_DONE_CMD" "$ON_DONE_LOG"; then
    log_watch "ON_DONE_CMD failed"
  fi
fi
