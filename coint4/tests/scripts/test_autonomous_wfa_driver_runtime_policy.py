from __future__ import annotations

import re
import subprocess
import textwrap
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "optimization" / "autonomous_wfa_driver.sh"


def _source() -> str:
    return SCRIPT_PATH.read_text(encoding="utf-8")


def _extract_shell_function(source: str, function_name: str) -> str:
    pattern = rf"^{re.escape(function_name)}\(\)\s*\{{\n(?P<body>.*?)^}}"
    match = re.search(pattern, source, flags=re.DOTALL | re.MULTILINE)
    assert match is not None, f"shell function {function_name} not found"
    return f"{function_name}() {{\n{match.group('body')}}}"


def _extract_reselect_after_reconcile_block(source: str) -> str:
    anchor = 'pending_before_reconcile="$pending"'
    anchor_idx = source.find(anchor)
    assert anchor_idx != -1, "pending_before_reconcile anchor not found"
    scoped = source[anchor_idx:]
    pattern = (
        r'if \(\( pending <= 0 \)\); then\s*\n'
        r'(?P<body>.*?log "candidate_reselect_after_reconcile queue=\$queue_rel pending=\$pending"\n\s*fi)'
    )
    match = re.search(pattern, scoped, flags=re.DOTALL)
    assert match is not None, "pending<=0 reconcile reselect block not found"
    return 'if (( pending <= 0 )); then\n' + match.group("body")


def _run_reselect_block(
    tmp_path: Path,
    block: str,
    *,
    fallback_success: bool,
    ready_buffer_success: bool = False,
    pending_before_reconcile: int = 0,
    completed: int = 0,
) -> subprocess.CompletedProcess[str]:
    candidate_file = tmp_path / "candidate.csv"
    header = (
        "queue,planned,running,stalled,failed,completed,total,urgency,mtime,promotion_potential,"
        "gate_status,gate_reason,pre_rank_score,strict_gate_status,strict_gate_reason,"
        "effective_planned_count,stalled_share,queue_yield_score,recent_yield"
    )
    if fallback_success:
        fallback_impl = (
            '  cat > "$CANDIDATE_FILE" <<\'CSV\'\n'
            f"{header}\n"
            "artifacts/wfa/aggregate/demo/run_queue.csv,1,0,0,0,0,1,1.000,123,"
            "POSSIBLE,OPEN,fallback_pending,0.000,FULLSPAN_PREFILTER_UNKNOWN,,1,0.000000,4.000,0.000\n"
            "CSV\n"
            "  return 0\n"
        )
    else:
        fallback_impl = "  return 1\n"

    script = f"""#!/usr/bin/env bash
set -euo pipefail
CANDIDATE_FILE="{candidate_file}"
ORPHAN_FILE="{tmp_path / 'orphan.csv'}"
FULLSPAN_DECISION_STATE_FILE="{tmp_path / 'fullspan_state.json'}"
LAST_REJECTED_QUEUE=""
ROOT_DIR="{tmp_path}"
adaptive_idle_sleep=0
pending=0
pending_before_reconcile={pending_before_reconcile}
planned=0
running=0
stalled=0
failed=0
completed={completed}
total=0
queue="{tmp_path / 'artifacts/wfa/aggregate/current/run_queue.csv'}"
queue_rel="artifacts/wfa/aggregate/current/run_queue.csv"
recent_yield=0
cleanup_orphans() {{ :; }}
find_candidate() {{
  cat > "$CANDIDATE_FILE" <<'CSV'
{header}
CSV
}}
ready_buffer_refresh() {{ :; }}
ready_buffer_emit_candidate() {{
{"  cat > \"$CANDIDATE_FILE\" <<'CSV'\n" + header + "\nartifacts/wfa/aggregate/ready/run_queue.csv,2,0,0,0,0,2,1.000,456,POSSIBLE,OPEN,ready_buffer,9.000,FULLSPAN_PREFILTER_PASSED,,2,0.000000,9.000,0.000\nCSV\n  return 0\n" if ready_buffer_success else "  return 1\n"}
}}
fallback_calls=0
fallback_pending_candidate() {{
  fallback_calls=$((fallback_calls + 1))
{fallback_impl}}}
log() {{ echo "LOG:$*"; }}
log_state() {{ echo "STATE:$*"; }}
maybe_trigger_auto_seed() {{ echo "SEED:$1"; return 0; }}
batch_session_maybe_stop() {{ echo "STOP:$1"; }}
fullspan_state_metric_set() {{ :; }}
log_decision_note() {{ echo "NOTE:$*"; }}
cycle_calls=0
run_fullspan_cycle() {{
  cycle_calls=$((cycle_calls + 1))
  echo "CYCLE:$1|$2|$3"
}}
iter=0
while true; do
  iter=$((iter + 1))
  if (( iter > 1 )); then
    echo "LOOP_EXIT"
    break
  fi
{textwrap.indent(block, "  ")}
  echo "BLOCK_DONE"
  break
done
echo "FALLBACK_CALLS:$fallback_calls"
echo "CYCLE_CALLS:$cycle_calls"
"""
    return subprocess.run(["bash", "-lc", script], cwd=tmp_path, capture_output=True, text=True)


def _assert_contains_all(source: str, snippets: list[str]) -> None:
    missing = [snippet for snippet in snippets if snippet not in source]
    assert not missing, f"missing required snippets: {missing}"


def _assert_contains_any(source: str, snippets: list[str], *, label: str) -> None:
    assert any(snippet in source for snippet in snippets), f"missing {label}: expected one of {snippets}"


def test_candidate_reselect_after_reconcile_header_only_parse_falls_back_to_idle(tmp_path: Path) -> None:
    block = _extract_reselect_after_reconcile_block(_source())
    proc = _run_reselect_block(tmp_path, block, fallback_success=False)
    assert proc.returncode == 0, f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    assert "LOG:candidate_parse_empty_after_reconcile" in proc.stdout
    assert "STATE:idle now=none completed=all" in proc.stdout
    assert "STOP:candidate_parse_empty_after_reconcile" in proc.stdout
    assert "FALLBACK_CALLS:1" in proc.stdout


def test_candidate_reselect_after_reconcile_header_only_parse_can_recover_via_fallback(tmp_path: Path) -> None:
    block = _extract_reselect_after_reconcile_block(_source())
    proc = _run_reselect_block(tmp_path, block, fallback_success=True)
    assert proc.returncode == 0, f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    assert "LOG:candidate_parse_fallback reason=parse_empty_after_reconcile" in proc.stdout
    assert "LOG:candidate_reselect_after_reconcile queue=artifacts/wfa/aggregate/demo/run_queue.csv pending=1" in proc.stdout
    assert "BLOCK_DONE" in proc.stdout
    assert "LOG:candidate_parse_empty_after_reconcile" not in proc.stdout
    assert "FALLBACK_CALLS:1" in proc.stdout


def test_candidate_reselect_after_reconcile_prefers_ready_buffer_before_hot_scan(tmp_path: Path) -> None:
    block = _extract_reselect_after_reconcile_block(_source())
    proc = _run_reselect_block(tmp_path, block, fallback_success=False, ready_buffer_success=True)
    assert proc.returncode == 0, f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    assert "LOG:ready_buffer_hit reason=reconcile_pending_zero" in proc.stdout
    assert "NOTE:global READY_BUFFER_HIT reason=reconcile_pending_zero reuse_ready_buffer_candidate" in proc.stdout
    assert "LOG:candidate_reselect_after_reconcile queue=artifacts/wfa/aggregate/ready/run_queue.csv pending=2" in proc.stdout
    assert "LOG:candidate_parse_empty_after_reconcile" not in proc.stdout
    assert "FALLBACK_CALLS:0" in proc.stdout


def test_candidate_reselect_after_reconcile_runs_cycle_once_on_pending_transition(tmp_path: Path) -> None:
    block = _extract_reselect_after_reconcile_block(_source())
    proc = _run_reselect_block(
        tmp_path,
        block,
        fallback_success=True,
        pending_before_reconcile=2,
        completed=3,
    )
    assert proc.returncode == 0, f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    assert "CYCLE:artifacts/wfa/aggregate/current/run_queue.csv|" in proc.stdout
    assert "CYCLE_CALLS:1" in proc.stdout
    assert "LOG:candidate_reselect_after_reconcile queue=artifacts/wfa/aggregate/demo/run_queue.csv pending=1" in proc.stdout


def test_queue_hygiene_noop_skip_contract() -> None:
    src = _source()
    _assert_contains_all(
        src,
        [
            'FORCE_SYNC_BEFORE_START',
            'sync_queue_status "$queue_rel"',
        ],
    )
    _assert_contains_any(
        src,
        [
            'no_op_queue_skips',
            'queue_hygiene_noop_skip',
            'candidate_noop_skip',
        ],
        label='queue hygiene no-op skip metric/hook',
    )


def test_low_backlog_seed_trigger_contract() -> None:
    src = _source()
    _assert_contains_all(
        src,
        [
            'MIN_PLANNED_BACKLOG',
            'AUTO_SEED_COOLDOWN_SEC',
            'AUTO_SEED_PENDING_THRESHOLD',
            'AUTO_SEED_NUM_VARIANTS',
            'AUTO_SEED_NUM_VARIANTS_FLOOR',
        ],
    )
    _assert_contains_any(
        src,
        [
            'autonomous_queue_seeder.py',
            'scripts/optimization/autonomous_queue_seeder.py',
        ],
        label='auto-seed invocation',
    )
    _assert_contains_any(
        src,
        [
            'seed_trigger_reason',
            'auto_seed_trigger',
        ],
        label='seed trigger reason metric/hook',
    )


def test_candidate_parse_empty_force_seed_contract() -> None:
    src = _source()
    _assert_contains_all(
        src,
        [
            'candidate_parse_empty',
            'candidate_parse_empty_after_reconcile',
            'remote_count="$(remote_runner_count)"',
            'force_seed=1',
            'effective_seed_pending_threshold',
            'force=$force_seed',
            'ready_depth="$(ready_buffer_depth)"',
            'READY_BUFFER_REFILL_THRESHOLD',
        ],
    )


def test_surrogate_refresh_recomputes_when_queue_file_is_newer_contract() -> None:
    src = _source()
    _assert_contains_all(
        src,
        [
            'queue_abs="${2:-$ROOT_DIR/$queue_rel}"',
            'queue_mtime > state_mtime',
            'surrogate_gate_state_needs_refresh "$queue_rel" "$ROOT_DIR/$queue_rel"',
        ],
    )


def test_no_progress_phase_switch_respects_live_powered_queue_contract() -> None:
    src = _source()
    _assert_contains_all(
        src,
        [
            'local_powered_queue_running()',
            'no_progress_phase_switch_deferred',
            'reason=local_powered_runner_active',
            'remote_queue_activity_source "$queue_rel"',
        ],
    )
    _assert_contains_any(
        src,
        [
            'remote_queue_running "$queue_rel" || local_powered_queue_running "$queue_rel"',
        ],
        label="live queue runner defer hook",
    )


def test_no_progress_phase_switch_grants_fresh_queue_grace_contract() -> None:
    src = _source()
    _assert_contains_all(
        src,
        [
            'NO_PROGRESS_BREAKER_FRESH_QUEUE_GRACE_SEC',
            'queue_age_sec=$((current_epoch - mtime))',
            'reason=fresh_queue_grace',
            'pending == planned',
        ],
    )


def test_no_progress_breaker_contract() -> None:
    src = _source()
    _assert_contains_all(
        src,
        [
            'NO_PROGRESS_BREAKER_WINDOW_SEC',
            'NO_PROGRESS_BREAKER_STREAK',
        ],
    )
    _assert_contains_any(
        src,
        [
            'no_progress_breaker',
            'phase_switch',
            'NO_PROGRESS_BREAKER',
        ],
        label='no-progress breaker hook',
    )


def test_runtime_metric_fields_contract() -> None:
    src = _source()
    _assert_contains_all(
        src,
        [
            'recent_yield',
            'queue_yield_score',
            'effective_planned_count',
            'stalled_share',
            'ready_buffer_depth',
            'cold_fail_active_count',
            'remote_child_process_count',
            'remote_queue_job_count',
            'remote_active_queue_jobs',
            'cpu_busy_without_queue_job',
            'surrogate_idle_override_count',
            'overlap_dispatch_count',
        ],
    )
    _assert_contains_any(
        src,
        [
            'no_op_queue_skips',
            'seed_trigger_reason',
        ],
        label='runtime policy metrics',
    )


def test_ready_buffer_and_cold_fail_contract() -> None:
    src = _source()
    _assert_contains_all(
        src,
        [
            'READY_BUFFER_POOL_FILE',
            'READY_BUFFER_STATE_FILE',
            'COLD_FAIL_STATE_FILE',
            'READY_BUFFER_MAX_AGE_SEC',
            'ready_buffer_policy_hash()',
            'ready_buffer_refresh()',
            'ready_buffer_emit_candidate()',
            'policy_hash',
            'queue_file_mtime',
            'ready_buffer_policy_mismatch_count',
            'cold_fail_state_add()',
            'HARD_FAIL_COLD_TTL_SEC',
        ],
    )


def test_overlap_dispatch_and_idle_override_contract() -> None:
    src = _source()
    _assert_contains_all(
        src,
        [
            'maybe_dispatch_overlap_from_buffer()',
            'READY_BUFFER_OVERLAP_TAIL_PENDING',
            'READY_BUFFER_MAX_ACTIVE_REMOTE_QUEUES',
            'remote_active_queue_jobs()',
            'remote_child_process_count()',
            'remote_cpu_busy_without_queue_job()',
            'remote_postprocess_active()',
            'remote_build_index_active()',
            'SURROGATE_IDLE_OVERRIDE',
            'cold_start_idle_slot',
        ],
    )


def test_hot_standby_and_replay_fastlane_contract() -> None:
    src = _source()
    _assert_contains_all(
        src,
        [
            'VPS_HOT_STANDBY_ENABLE',
            'VPS_HOT_STANDBY_GRACE_SEC',
            'REPLAY_FASTLANE_SCAN_LIMIT',
            'is_hot_standby_enabled()',
            'hot_standby_needed()',
            'maybe_prepare_hot_standby()',
            'dispatch_replay_fastlane_hooks()',
            'fastlane_replay_pending',
            'winner_parent_duplication_rate',
            'vps_duty_cycle_30m',
            'metrics_missing_abort_count_30m',
            'winner_proximate_dispatch_count_30m',
        ],
    )


def test_vps_start_softpass_contract() -> None:
    src = _source()
    _assert_contains_all(
        src,
        [
            'VPS_RECENT_READY_GRACE_SEC',
            'VPS_CAPACITY_STATE_MAX_AGE_SEC',
            'capacity_controller_remote_reachable_recent()',
            'vps_recently_recovered()',
            'start_queue_softpass_reason()',
            'START_VPS_SOFTPASS',
            'start_softpass queue=$queue_rel',
            'vps_start_softpass_count',
            'continue_dispatch',
        ],
    )


def test_remote_runtime_snapshot_contract() -> None:
    src = _source()
    _assert_contains_all(
        src,
        [
            'REMOTE_RUNTIME_STATE_FILE',
            'REMOTE_RUNTIME_SNAPSHOT_MAX_AGE_SEC',
            'remote_runtime_state_value()',
            'remote_runtime_state_age_sec()',
            'remote_runtime_snapshot_is_fresh()',
            'refresh_remote_runtime_snapshot()',
            'remote_runtime_queue_snapshot()',
            'remote_queue_activity_source()',
            'active_remote_queue_rel',
            'remote_queue_sync_age_sec',
            'postprocess_active',
            'build_index_active',
            'scripts/optimization/remote_runtime_probe.py',
            'top_level_queue_jobs',
            'remote_work_active',
            'remote_runtime_snapshot_age_sec',
        ],
    )


def test_start_queue_requires_confirmed_start_contract() -> None:
    src = _source()
    _assert_contains_all(
        src,
        [
            'START_QUEUE_SSH_READY_TIMEOUT_SEC',
            'START_QUEUE_SYNC_TIMEOUT_SEC',
            'START_QUEUE_STARTUP_BUDGET_SEC',
            'START_QUEUE_CONFIRM_TIMEOUT_SEC',
            'START_QUEUE_CONFIRM_POLL_SEC',
            'record_infra_fail_closed()',
            'queue_start_confirmation_status()',
            'wait_for_queue_start_confirmation()',
            'set_infra_gate_state "starting"',
            'log_state "starting queue=$queue_rel',
            'log_state "running queue=$queue_rel',
            'start_fail_closed queue=$queue_rel',
            'startup_failure_code',
            'INFRA_FAIL_CLOSED',
            'startup_state',
            'SSH_WAIT_TIMEOUT',
            'MANIFEST_MISMATCH',
            'STARTUP_TIMEOUT',
        ],
    )
    _assert_contains_any(
        src,
        [
            'powered: remote run start queue=',
            'QUEUE_STATUS_PROGRESS',
        ],
        label='startup confirmation marker',
    )


def test_auto_seed_hard_block_and_coverage_policy_contract() -> None:
    src = _source()
    _assert_contains_all(
        src,
        [
            'yield_governor_hard_block_snapshot()',
            'process_slo_remote_coverage_snapshot()',
            'queue_coverage_policy_snapshot()',
            'AUTO_SEED_HARD_BLOCK',
            'coverage_fail_closed queue=$queue_rel',
            'COVERAGE_FAIL_CLOSED',
            'auto_seed_hard_block',
            'coverage_verified',
            'infra_gate_status',
            'infra_recovery_mode_remote_unreachable_no_coverage_ready',
            'startup_failure_code',
        ],
    )
    _assert_contains_any(
        src,
        [
            'set_infra_gate_state "$gate_block_status"',
            'set_infra_gate_state "hard_block"',
            'set_infra_gate_state "fail_closed"',
        ],
        label='auto-seed hard block infra gate update',
    )


def test_wait_for_queue_start_confirmation_fails_closed_on_waiting_for_ssh_timeout(tmp_path: Path) -> None:
    fn = _extract_shell_function(_source(), "wait_for_queue_start_confirmation")
    script = f"""#!/usr/bin/env bash
set -euo pipefail
START_QUEUE_CONFIRM_TIMEOUT_SEC=10
START_QUEUE_CONFIRM_POLL_SEC=0.2
START_QUEUE_SSH_READY_TIMEOUT_SEC=1
START_QUEUE_SYNC_TIMEOUT_SEC=5
START_QUEUE_STARTUP_BUDGET_SEC=5
{fn}
queue_start_confirmation_status() {{
  printf 'pending\\tWAITING_FOR_SSH\\twaiting_for_ssh\\n'
}}
local_powered_queue_running() {{
  return 0
}}
if out="$(wait_for_queue_start_confirmation queue_rel startup.log "$START_QUEUE_STARTUP_BUDGET_SEC" "$START_QUEUE_CONFIRM_POLL_SEC" "$START_QUEUE_SSH_READY_TIMEOUT_SEC" "$START_QUEUE_SYNC_TIMEOUT_SEC" "$START_QUEUE_STARTUP_BUDGET_SEC")"; then
  rc=0
else
  rc=$?
fi
printf 'RC:%s\\n' "$rc"
printf 'OUT:%s\\n' "$out"
"""
    proc = subprocess.run(["bash", "-lc", script], cwd=tmp_path, capture_output=True, text=True)
    assert proc.returncode == 0, f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    assert "RC:1" in proc.stdout
    assert "OUT:SSH_WAIT_TIMEOUT" in proc.stdout
    assert "waiting_for_ssh_timeout" in proc.stdout


def test_maybe_trigger_auto_seed_respects_infra_fail_closed_runtime_block(tmp_path: Path) -> None:
    fn = _extract_shell_function(_source(), "maybe_trigger_auto_seed")
    script = f"""#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="{tmp_path}"
LOG_FILE="{tmp_path / 'driver.log'}"
AUTO_SEED_PENDING_THRESHOLD=96
READY_BUFFER_REFILL_THRESHOLD=2
AUTO_SEED_COOLDOWN_SEC=0
AUTO_SEED_NUM_VARIANTS=64
AUTO_SEED_NUM_VARIANTS_FLOOR=24
{fn}
global_backlog_snapshot() {{
  echo "0 0 0"
}}
ready_buffer_depth() {{
  echo "0"
}}
yield_governor_hard_block_snapshot() {{
  printf '0\\t\\t0\\t0\\n'
}}
remote_active_queue_jobs() {{
  echo "0"
}}
fullspan_state_metric_get() {{
  case "$1" in
    last_seed_trigger_epoch) echo "0" ;;
    auto_seed_hard_block) echo "0" ;;
    auto_seed_block_reason) echo "" ;;
    vps_infra_fail_closed) echo "0" ;;
    infra_gate_status) echo "fail_closed" ;;
    startup_failure_code) echo "SSH_WAIT_TIMEOUT" ;;
    *) echo "0" ;;
  esac
}}
fullspan_state_metric_set() {{
  printf 'SET:%s=%s\\n' "$1" "${{2:-}}"
}}
fullspan_state_metric_inc() {{
  printf 'INC:%s=%s\\n' "$1" "${{2:-}}"
}}
set_infra_gate_state() {{
  printf 'INFRA:%s|%s|%s|%s|%s\\n' "${{1:-}}" "${{2:-}}" "${{3:-}}" "${{4:-}}" "${{5:-}}"
}}
log() {{
  printf 'LOG:%s\\n' "$*"
}}
log_decision_note() {{
  printf 'NOTE:%s|%s|%s|%s\\n' "${{1:-}}" "${{2:-}}" "${{3:-}}" "${{4:-}}"
}}
maybe_trigger_auto_seed candidate_empty
"""
    proc = subprocess.run(["bash", "-lc", script], cwd=tmp_path, capture_output=True, text=True)
    assert proc.returncode == 0, f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    assert "INFRA:fail_closed|" in proc.stdout
    assert "infra_fail_closed:SSH_WAIT_TIMEOUT" in proc.stdout
    assert "NOTE:global|AUTO_SEED_HARD_BLOCK|" in proc.stdout
    assert "AUTO_SEED_TRIGGER" not in proc.stdout


def test_maybe_trigger_auto_seed_respects_remote_recovery_mode_block(tmp_path: Path) -> None:
    fn = _extract_shell_function(_source(), "maybe_trigger_auto_seed")
    script = f"""#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="{tmp_path}"
LOG_FILE="{tmp_path / 'driver.log'}"
AUTO_SEED_PENDING_THRESHOLD=96
READY_BUFFER_REFILL_THRESHOLD=2
AUTO_SEED_COOLDOWN_SEC=0
AUTO_SEED_NUM_VARIANTS=64
AUTO_SEED_NUM_VARIANTS_FLOOR=24
{fn}
global_backlog_snapshot() {{
  echo "0 0 0"
}}
ready_buffer_depth() {{
  echo "0"
}}
yield_governor_hard_block_snapshot() {{
  printf '0\\t\\t0\\t0\\n'
}}
process_slo_remote_coverage_snapshot() {{
  printf '0\\t0\\n'
}}
remote_active_queue_jobs() {{
  echo "0"
}}
fullspan_state_metric_get() {{
  case "$1" in
    last_seed_trigger_epoch) echo "0" ;;
    auto_seed_hard_block) echo "0" ;;
    auto_seed_block_reason) echo "" ;;
    vps_infra_fail_closed) echo "0" ;;
    infra_gate_status) echo "" ;;
    startup_failure_code) echo "" ;;
    *) echo "0" ;;
  esac
}}
fullspan_state_metric_set() {{
  printf 'SET:%s=%s\\n' "$1" "${{2:-}}"
}}
fullspan_state_metric_inc() {{
  printf 'INC:%s=%s\\n' "$1" "${{2:-}}"
}}
set_infra_gate_state() {{
  printf 'INFRA:%s|%s|%s|%s|%s\\n' "${{1:-}}" "${{2:-}}" "${{3:-}}" "${{4:-}}" "${{5:-}}"
}}
log() {{
  printf 'LOG:%s\\n' "$*"
}}
log_decision_note() {{
  printf 'NOTE:%s|%s|%s|%s\\n' "${{1:-}}" "${{2:-}}" "${{3:-}}" "${{4:-}}"
}}
maybe_trigger_auto_seed candidate_empty_after_reconcile
"""
    proc = subprocess.run(["bash", "-lc", script], cwd=tmp_path, capture_output=True, text=True)
    assert proc.returncode == 0, f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    assert "INFRA:hard_block|infra_recovery_mode_remote_unreachable_no_coverage_ready|" in proc.stdout
    assert "NOTE:global|AUTO_SEED_HARD_BLOCK|" in proc.stdout
    assert "AUTO_SEED_TRIGGER" not in proc.stdout


def test_early_abort_zero_activity_and_confirm_guard_contract() -> None:
    src = _source()
    _assert_contains_all(
        src,
        [
            'EARLY_ABORT_MIN_COMPLETED',
            'EARLY_ABORT_ZERO_ACTIVITY_SHARE',
            'EARLY_ABORT_ZERO_ACTIVITY_MIN',
            'EARLY_ABORT_ZERO_ACTIVITY',
            'EARLY_ABORT_LOW_INFORMATION_',
            '"confirm_fastlane_" in queue_rel',
            'zero_activity_fraction',
            'runtime_observability_record_event "metrics_missing_abort"',
        ],
    )
