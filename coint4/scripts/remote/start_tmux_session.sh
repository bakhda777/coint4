#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF' >&2
Usage:
  start_tmux_session.sh --session <name> [--cmd <command>] [--check]

Behavior contract:
  - If session exists: print "session_exists=<name>", then `tmux ls` and
    `tmux list-windows -t <name>`, exit 0. No side effects.
  - If session does not exist:
      - with --check: exit 1 (no side effects)
      - without --check: create detached session running `bash -lc <command>`,
        print "session_started=<name>", then `tmux ls` and `tmux list-windows`,
        exit 0.

Notes:
  - Command is executed inside tmux as: bash -lc '<cmd>'.
  - The caller should pass <command> as a single argument to avoid quoting issues.
EOF
}

SESSION=""
CMD=""
CHECK_ONLY=0

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
    --check)
      CHECK_ONLY=1
      shift 1
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

# Keep session names simple and deterministic (also helps preflight substring match).
if [[ ! "$SESSION" =~ ^[A-Za-z0-9_.-]+$ ]]; then
  echo "Invalid session name (allowed: [A-Za-z0-9_.-]): $SESSION" >&2
  exit 2
fi

if ! command -v tmux >/dev/null 2>&1; then
  echo "tmux not found on remote host" >&2
  exit 2
fi

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "session_exists=${SESSION}"
  tmux ls || true
  tmux list-windows -t "$SESSION" || true
  exit 0
fi

if [[ "$CHECK_ONLY" == "1" ]]; then
  exit 1
fi

if [[ -z "$CMD" ]]; then
  echo "Missing --cmd (required when session is absent)" >&2
  usage
  exit 2
fi

shell_quote_single() {
  local s="$1"
  # Wrap a string in single quotes and escape embedded single quotes as: '\''.
  # Example: a'b -> 'a'\''b'
  s=${s//\'/\'\\\'\'}
  printf "'%s'" "$s"
}

# Build a POSIX-sh compatible command string for tmux: `bash -lc '<CMD>'`.
cmd_quoted="$(shell_quote_single "$CMD")"
tmux new-session -d -s "$SESSION" "bash -lc ${cmd_quoted}"

echo "session_started=${SESSION}"
tmux ls || true
tmux list-windows -t "$SESSION" || true
exit 0
