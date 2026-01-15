#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 2 ]; then
  echo "Usage: $0 <config_path> <results_dir>"
  exit 1
fi

CONFIG_PATH="$1"
RESULTS_DIR="$2"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

export TMPDIR="${TMPDIR:-/tmp}"
export JOBLIB_TEMP_FOLDER="${JOBLIB_TEMP_FOLDER:-$TMPDIR/joblib}"
export JOBLIB_MULTIPROCESSING=1

mkdir -p "$JOBLIB_TEMP_FOLDER"
mkdir -p "$RESULTS_DIR"

echo "[run_wfa_fullcpu] start: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo -n "[run_wfa_fullcpu] os.cpu_count: "
./.venv/bin/python - <<'PY'
import os
print(os.cpu_count())
PY

./.venv/bin/coint2 walk-forward --config "$CONFIG_PATH" --results-dir "$RESULTS_DIR"

echo "[run_wfa_fullcpu] end: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
