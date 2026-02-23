#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: run_heavy_queue.sh --queue <path> [options]

Options:
  --queue <path>     Queue CSV path (relative to coint4/ or absolute).
  --parallel <n>     Parallel workers passed to runner (default: 10).
  --runner <name>    Runner type: watch|queue (default: watch).
  --dry-run          Print validated plan and exit without remote execution.
  --help             Show this help.
EOF
}

QUEUE_PATH=""
PARALLEL="${PARALLEL:-10}"
RUNNER="watch"
DRY_RUN=0

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
    --runner)
      RUNNER="$2"
      shift 2
      ;;
    --dry-run)
      DRY_RUN=1
      shift 1
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ -z "$QUEUE_PATH" ]]; then
  echo "Missing --queue" >&2
  usage
  exit 1
fi

APP_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
if [[ "$QUEUE_PATH" != /* ]]; then
  QUEUE_PATH="$APP_ROOT/$QUEUE_PATH"
fi
if [[ ! -f "$QUEUE_PATH" ]]; then
  echo "Queue file not found: $QUEUE_PATH" >&2
  exit 1
fi

case "$RUNNER" in
  watch|queue) ;;
  *)
    echo "Unsupported --runner: $RUNNER (expected watch|queue)" >&2
    exit 1
    ;;
esac

if [[ "$PARALLEL" =~ [^0-9] || "$PARALLEL" -lt 1 ]]; then
  echo "--parallel must be a positive integer" >&2
  exit 1
fi

if [[ "$QUEUE_PATH" != "$APP_ROOT/"* ]]; then
  echo "Queue must be inside app root: $APP_ROOT" >&2
  exit 1
fi
QUEUE_REL="${QUEUE_PATH#"$APP_ROOT/"}"
RUN_GROUP="$(basename "$(dirname "$QUEUE_PATH")")"
SERVER_IP="${SERVER_IP:-85.198.90.128}"

"$APP_ROOT/.venv/bin/python" - <<'PY' "$QUEUE_PATH" "$APP_ROOT"
import csv
import sys
from pathlib import Path

queue_path = Path(sys.argv[1])
app_root = Path(sys.argv[2])

with queue_path.open(newline="", encoding="utf-8") as handle:
    reader = csv.DictReader(handle)
    rows = list(reader)
    fields = set(reader.fieldnames or [])

required = {"config_path", "results_dir", "status"}
missing_fields = sorted(required - fields)
if missing_fields:
    print(f"Queue is missing required columns: {', '.join(missing_fields)}", file=sys.stderr)
    raise SystemExit(2)

if not rows:
    print("Queue has no entries.", file=sys.stderr)
    raise SystemExit(2)

missing_configs = []
for row in rows:
    raw = str(row.get("config_path") or "").strip()
    if not raw:
        missing_configs.append("<empty-config-path>")
        continue
    cfg = Path(raw) if Path(raw).is_absolute() else app_root / raw
    if not cfg.exists():
        missing_configs.append(raw)

if missing_configs:
    unique = sorted(set(missing_configs))
    print(f"Missing config files referenced by queue: {', '.join(unique)}", file=sys.stderr)
    raise SystemExit(2)
PY

HEAVY_ALLOW_ENV="${HEAVY_ALLOW_ENV:-ALLOW_HEAVY_RUN}"
HEAVY_HOST_ALLOWLIST="${HEAVY_HOSTNAME_ALLOWLIST:-85.198.90.128,coint}"
HEAVY_MIN_RAM_GB="${HEAVY_MIN_RAM_GB:-28}"
HEAVY_MIN_CPU="${HEAVY_MIN_CPU:-8}"

PYTHONPATH="$APP_ROOT/src" "$APP_ROOT/.venv/bin/python" -m coint2.ops.heavy_guardrails \
  --entrypoint "scripts/batch/run_heavy_queue.sh" \
  --allow-env "$HEAVY_ALLOW_ENV" \
  --allowlist "$HEAVY_HOST_ALLOWLIST" \
  --min-ram-gb "$HEAVY_MIN_RAM_GB" \
  --min-cpu "$HEAVY_MIN_CPU"

REMOTE_ARGS=()
if [[ "$RUNNER" == "watch" ]]; then
  REMOTE_ARGS=(bash -lc "ALLOW_HEAVY_RUN=1 bash scripts/optimization/watch_wfa_queue.sh --queue $QUEUE_REL --parallel $PARALLEL")
else
  REMOTE_ARGS=(bash -lc "ALLOW_HEAVY_RUN=1 PYTHONPATH=src ./.venv/bin/python scripts/optimization/run_wfa_queue.py --queue $QUEUE_REL --statuses planned,stalled --parallel $PARALLEL")
fi

printf -v REMOTE_CMD_PREVIEW '%q ' "${REMOTE_ARGS[@]}"
echo "[heavy-runner] plan:"
echo "  run_group=$RUN_GROUP"
echo "  queue_path=$QUEUE_REL"
echo "  host=$SERVER_IP"
echo "  runner=$RUNNER"
echo "  remote_cmd=${REMOTE_CMD_PREVIEW}"
echo "  postprocess=manual-required"

if [[ "$DRY_RUN" == "1" ]]; then
  echo "[heavy-runner] dry-run: no remote execution"
  exit 0
fi

(
  cd "$APP_ROOT"
  SYNC_UP="${SYNC_UP:-1}" \
  UPDATE_CODE="${UPDATE_CODE:-0}" \
  STOP_AFTER="${STOP_AFTER:-1}" \
  SYNC_BACK="${SYNC_BACK:-1}" \
  bash scripts/remote/run_server_job.sh "${REMOTE_ARGS[@]}"
)
