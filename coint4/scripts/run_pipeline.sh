#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

LOG_DIR="${LOG_DIR:-$ROOT_DIR/artifacts/command_logs}"
mkdir -p "$LOG_DIR"
RUN_STAMP="$(date -u +%Y%m%d_%H%M%S)"
CMD_LOG="${LOG_DIR}/run_pipeline_${RUN_STAMP}.log"
exec 3>>"$CMD_LOG"
export BASH_XTRACEFD=3
export PS4='+ $(date -u +%Y-%m-%dT%H:%M:%SZ) [run_pipeline] '
set -x

PYTHON_BIN="${PYTHON_BIN:-$ROOT_DIR/.venv/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="python"
fi

export PYTHONPATH="${PYTHONPATH:-$ROOT_DIR/src}"

BASE_CONFIG="${BASE_CONFIG:-configs/main_2024.yaml}"
CRITERIA_CONFIG="${CRITERIA_CONFIG:-configs/criteria_relaxed.yaml}"
OUT_DIR="${OUT_DIR:-bench}"
PAIRS_FILE="${PAIRS_FILE:-${OUT_DIR}/pairs_universe.yaml}"
BACKTEST_OUT="${BACKTEST_OUT:-outputs/fixed_run}"
SYMBOLS="${SYMBOLS:-ALL}"

COMMON_ARGS=()
if [[ -n "${DATA_ROOT:-}" ]]; then
  COMMON_ARGS+=(--data-root "$DATA_ROOT")
fi

SCAN_ARGS=("${COMMON_ARGS[@]}" --config "$CRITERIA_CONFIG" --base-config "$BASE_CONFIG" --output-dir "$OUT_DIR" --symbols "$SYMBOLS")
if [[ -n "${END_DATE:-}" ]]; then
  SCAN_ARGS+=(--end-date "$END_DATE")
fi

"$PYTHON_BIN" -m coint2.cli scan "${SCAN_ARGS[@]}"

BACKTEST_ARGS=("${COMMON_ARGS[@]}" --config "$BASE_CONFIG" --pairs-file "$PAIRS_FILE" --out-dir "$BACKTEST_OUT")
if [[ -n "${START_DATE:-}" ]]; then
  BACKTEST_ARGS+=(--period-start "$START_DATE")
fi
if [[ -n "${END_DATE:-}" ]]; then
  BACKTEST_ARGS+=(--period-end "$END_DATE")
fi

"$PYTHON_BIN" -m coint2.cli backtest "${BACKTEST_ARGS[@]}"

WF_ARGS=("${COMMON_ARGS[@]}" --config "$BASE_CONFIG")
"$PYTHON_BIN" -m coint2.cli walk-forward "${WF_ARGS[@]}"
