#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 2 ]; then
  echo "Usage: $0 <config_path> <results_dir>"
  exit 1
fi

CONFIG_PATH="$1"
RESULTS_DIR="$2"
NO_MEMORY_MAP="${COINT_WFA_NO_MEMORY_MAP:-0}"
ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"

HEAVY_ALLOW_ENV="${HEAVY_ALLOW_ENV:-ALLOW_HEAVY_RUN}"
HEAVY_HOST_ALLOWLIST="${HEAVY_HOSTNAME_ALLOWLIST:-85.198.90.128,coint}"
HEAVY_MIN_RAM_GB="${HEAVY_MIN_RAM_GB:-28}"
HEAVY_MIN_CPU="${HEAVY_MIN_CPU:-8}"

PYTHONPATH="$ROOT_DIR/src" "$ROOT_DIR/.venv/bin/python" -m coint2.ops.heavy_guardrails \
  --entrypoint "run_wfa_fullcpu.sh" \
  --allow-env "$HEAVY_ALLOW_ENV" \
  --allowlist "$HEAVY_HOST_ALLOWLIST" \
  --min-ram-gb "$HEAVY_MIN_RAM_GB" \
  --min-cpu "$HEAVY_MIN_CPU"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
if [[ -z "${NUMBA_NUM_THREADS:-}" ]]; then
  export NUMBA_NUM_THREADS=1
fi

export TMPDIR="${TMPDIR:-/tmp}"
export JOBLIB_TEMP_FOLDER="${JOBLIB_TEMP_FOLDER:-$TMPDIR/joblib}"
export JOBLIB_MULTIPROCESSING=1

mkdir -p "$JOBLIB_TEMP_FOLDER"
mkdir -p "$RESULTS_DIR"

CMD_LOG="${RESULTS_DIR}/run.commands.log"
if [[ -f "$CMD_LOG" ]]; then
  STAMP="$(date -u +%Y%m%d_%H%M%S)"
  mv "$CMD_LOG" "${CMD_LOG}.stalled_${STAMP}"
fi
exec 3>>"$CMD_LOG"
export BASH_XTRACEFD=3
export PS4='+ $(date -u +%Y-%m-%dT%H:%M:%SZ) [run_wfa_fullcpu] '
set -x

echo "[run_wfa_fullcpu] start: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo -n "[run_wfa_fullcpu] os.cpu_count: "
"$ROOT_DIR/.venv/bin/python" - <<'PY'
import os
print(os.cpu_count())
PY

WORKER_PID_FILE="$RESULTS_DIR/worker.pid"

CLI_ARGS=(--config "$CONFIG_PATH" --results-dir "$RESULTS_DIR")
if [[ "$NO_MEMORY_MAP" == "1" ]]; then
  CLI_ARGS+=(--no-memory-map)
fi

"$ROOT_DIR/.venv/bin/coint2" walk-forward "${CLI_ARGS[@]}" &
WORKER_PID=$!
echo "$WORKER_PID" > "$WORKER_PID_FILE"
echo "[run_wfa_fullcpu] worker_pid=$WORKER_PID"

set +e
wait "$WORKER_PID"
RC=$?
set -e

echo "[run_wfa_fullcpu] end: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
exit "$RC"
