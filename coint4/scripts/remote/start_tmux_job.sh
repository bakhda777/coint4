#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF' >&2
Usage:
  start_tmux_job.sh --session <name> --cmd <command>

Starts (idempotently) a detached tmux session on the remote VPS via run_server_job.sh.

Contract:
  - If the tmux session already exists, the run exits 0 and prints:
      session_exists=<name>
      tmux ls
      tmux list-windows -t <name>
    with NO side effects (no git pull/rsync/sync_back/stop/pipeline restart).
  - If the session does not exist, it is created detached and prints:
      session_started=<name>
      tmux ls
      tmux list-windows -t <name>
    then exits 0.

Notes:
  - Remote execution is always done via `bash -lc` for consistent env loading.
  - The CMD is passed as an argument (not interpolated into remote strings) to avoid quoting issues.
EOF
}

SESSION=""
CMD=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --session)
      SESSION="${2:-}"
      shift 2
      ;;
    --cmd)
      CMD="${2:-}"
      shift 2
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "Unknown arg: $1" >&2
      usage
      exit 2
      ;;
  esac
done

if [[ -z "$SESSION" ]]; then
  echo "Missing --session" >&2
  usage
  exit 2
fi
if [[ -z "$CMD" ]]; then
  echo "Missing --cmd" >&2
  usage
  exit 2
fi

# Keep session names simple and deterministic (also helps preflight substring match).
if [[ ! "$SESSION" =~ ^[A-Za-z0-9_.-]+$ ]]; then
  echo "Invalid session name (allowed: [A-Za-z0-9_.-]): $SESSION" >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_SERVER_JOB="${SCRIPT_DIR}/run_server_job.sh"

REMOTE_WORK_DIR="${SERVER_WORK_DIR:-/opt/coint4/coint4}"

# Preflight is a single remote call used by run_server_job.sh to exit early (no sync/update/stop)
# if the tmux session already exists.
export REMOTE_PREFLIGHT_CMD="cd ${REMOTE_WORK_DIR} && bash scripts/remote/start_tmux_session.sh --check --session ${SESSION}"
export PREFLIGHT_MATCH="session_exists=${SESSION}"

# Remote command string is executed by `bash -lc` on the VPS. We pass SESSION/CMD as positional args
# to avoid interpolating them into the remote string (robust against spaces/quotes).
# NOTE: Keep this in single quotes so $1/$2 are expanded only on the remote side.
REMOTE_CMD='cd "'"${REMOTE_WORK_DIR}"'" && bash scripts/remote/start_tmux_session.sh --session "$1" --cmd "$2"'

# Fixed flags for tmux session start:
# - STOP_AFTER=0: do not auto-shutdown VPS on success
# - SYNC_BACK=0: do not pull artifacts back (session start is non-blocking)
# - SYNC_UP=1: ensure remote has the current tracked repo state (guarded by preflight on session_exists)
STOP_AFTER=0 SYNC_BACK=0 SYNC_UP=1 UPDATE_CODE=0 bash "${RUN_SERVER_JOB}" \
  bash -lc "${REMOTE_CMD}" \
  _ "${SESSION}" "${CMD}"
