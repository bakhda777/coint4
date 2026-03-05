#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF' >&2
Usage:
  probe_autonomous_markers.sh [--json] [--root <path|auto>] [--ensure-process-slo <auto|always|never>] [--ensure-timeout-sec <int>]

Runs lightweight autonomous marker probe on VPS via run_server_job.sh.
Defaults are read-only except optional process_slo_state materialization handled by probe script.

Environment (optional):
  SERVER_IP, SERVER_USER, SERVER_WORK_DIR, STOP_AFTER, SKIP_POWER, STOP_VIA_SSH
  UPDATE_CODE (default forced to 0), SYNC_UP (default 0), SYNC_BACK (default 0)
EOF
}

OUTPUT_FORMAT="text"
ROOT_ARG="auto"
ENSURE_MODE="auto"
ENSURE_TIMEOUT_SEC="30"
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --json)
      OUTPUT_FORMAT="json"
      shift
      ;;
    --root)
      ROOT_ARG="${2:-}"
      if [[ -z "$ROOT_ARG" ]]; then
        echo "Missing value for --root" >&2
        exit 2
      fi
      shift 2
      ;;
    --ensure-process-slo)
      ENSURE_MODE="${2:-}"
      if [[ -z "$ENSURE_MODE" ]]; then
        echo "Missing value for --ensure-process-slo" >&2
        exit 2
      fi
      case "$ENSURE_MODE" in
        auto|always|never) ;;
        *)
          echo "Invalid --ensure-process-slo: $ENSURE_MODE (expected auto|always|never)" >&2
          exit 2
          ;;
      esac
      shift 2
      ;;
    --ensure-timeout-sec)
      ENSURE_TIMEOUT_SEC="${2:-}"
      if [[ ! "$ENSURE_TIMEOUT_SEC" =~ ^[0-9]+$ ]]; then
        echo "Invalid --ensure-timeout-sec: $ENSURE_TIMEOUT_SEC" >&2
        exit 2
      fi
      shift 2
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    --)
      shift
      EXTRA_ARGS+=("$@")
      break
      ;;
    *)
      EXTRA_ARGS+=("$1")
      shift
      ;;
  esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_SERVER_JOB="$SCRIPT_DIR/run_server_job.sh"

if [[ ! -x "$RUN_SERVER_JOB" ]]; then
  echo "run_server_job.sh not found or not executable: $RUN_SERVER_JOB" >&2
  exit 1
fi

UPDATE_CODE="${UPDATE_CODE:-0}" \
SYNC_UP="${SYNC_UP:-0}" \
SYNC_BACK="${SYNC_BACK:-0}" \
"$RUN_SERVER_JOB" \
  python3 scripts/optimization/probe_autonomous_markers.py \
    --format "$OUTPUT_FORMAT" \
    --root "$ROOT_ARG" \
    --ensure-process-slo "$ENSURE_MODE" \
    --ensure-timeout-sec "$ENSURE_TIMEOUT_SEC" \
    "${EXTRA_ARGS[@]}"
