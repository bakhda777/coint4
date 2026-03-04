#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: recover_stalled_queue.sh --queue <path> [--parallel N]

Recover only stalled entries in a run_queue with fast triage.
It:
  1) prints queue summary + top error signatures,
  2) reruns only stalled items,
  3) prints final summary.
EOF
}

QUEUE=""
PARALLEL=4
while [[ $# -gt 0 ]]; do
  case "$1" in
    --queue)
      QUEUE="$2"
      shift 2
      ;;
    --parallel)
      PARALLEL="$2"
      shift 2
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "Unknown arg: $1"
      usage
      exit 1
      ;;
  esac
done

if [[ -z "$QUEUE" ]]; then
  echo "Need --queue"
  exit 1
fi

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
if [[ "$QUEUE" != /* ]]; then
  QUEUE="$ROOT_DIR/$QUEUE"
fi

if [[ ! -f "$QUEUE" ]]; then
  echo "Queue not found: $QUEUE"
  exit 1
fi

python3 - <<'PY' "$QUEUE" "$ROOT_DIR"
import csv
import sys
from collections import Counter
from pathlib import Path

q = Path(sys.argv[1])
root_dir = Path(sys.argv[2])
rows = list(csv.DictReader(q.open(newline='')))
st = Counter((r.get('status') or '').strip().lower() for r in rows)

print(f"SUMMARY total={len(rows)} planned={st.get('planned',0)} running={st.get('running',0)} completed={st.get('completed',0)} stalled={st.get('stalled',0)} failed={st.get('failed',0)+st.get('error',0)}")

sig_counter = Counter()
for r in rows:
    status = (r.get('status') or '').strip().lower()
    if status != 'stalled':
        continue
    for key in ('run_id', 'config_path', 'results_dir'):
        if r.get(key):
            print('STALLED', status, r.get(key))
            break

# quick error signatures from existing run logs
for r in rows:
    if (r.get('status') or '').strip().lower() != 'stalled':
        continue
    rd = r.get('results_dir')
    if not rd:
        continue
    p = (root_dir / rd).resolve()
    for fname in ('run.log', 'run.log.txt', 'watch_wfa.log'):
        fp = p / fname
        if fp.exists():
            with fp.open(errors='ignore') as f:
                for line in f:
                    if 'equity_series is empty' in line:
                        sig_counter['equity_series is empty'] += 1
                    if 'RuntimeError:' in line:
                        sig_counter[line.strip().split('RuntimeError:', 1)[1].strip()] += 1
            break

if sig_counter:
    print('TOP_SIGNATURES')
    for k, v in sig_counter.most_common(5):
        print(f'{v}\t{k}')
PY

echo "==> rerun stalled only (parallel=$PARALLEL)"
ALLOW_HEAVY_RUN=1 "$ROOT_DIR/.venv/bin/python" "$ROOT_DIR/scripts/optimization/run_wfa_queue.py" \
  --queue "$QUEUE" \
  --statuses stalled \
  --parallel "$PARALLEL"

python3 - <<'PY' "$QUEUE"
import csv
from collections import Counter
from pathlib import Path

q = Path(sys.argv[1])
rows = list(csv.DictReader(q.open(newline='')))
st = Counter((r.get('status') or '').strip().lower() for r in rows)
print(f"SUMMARY total={len(rows)} planned={st.get('planned',0)} running={st.get('running',0)} completed={st.get('completed',0)} stalled={st.get('stalled',0)} failed={st.get('failed',0)+st.get('error',0)}")
PY
