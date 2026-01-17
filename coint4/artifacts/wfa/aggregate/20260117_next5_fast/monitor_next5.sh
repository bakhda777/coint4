#!/usr/bin/env bash
set -euo pipefail

QUEUE_DIR="$(cd "$(dirname "$0")" && pwd)"
RUNS_DIR="${QUEUE_DIR%/aggregate/*}/runs/20260117_next5_fast"

cat <<EOF
Очередь/логи:
- watch: $QUEUE_DIR/run_queue.watch.log
- run:   $QUEUE_DIR/run_queue.log
- runs:  $RUNS_DIR/<run_name>/run.log

CPU мониторинг (основной):
ps -eo pid,pcpu,etimes,cmd --no-headers | rg "coint2 walk-forward"

Подсказка:
ls -1 "$RUNS_DIR"
EOF

tail -n 200 -f "$QUEUE_DIR/run_queue.watch.log"
