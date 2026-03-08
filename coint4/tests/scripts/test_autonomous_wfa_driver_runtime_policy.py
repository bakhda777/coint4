from __future__ import annotations

import json
import re
import subprocess
import textwrap
import time
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "optimization" / "autonomous_wfa_driver.sh"


def _source() -> str:
    return SCRIPT_PATH.read_text(encoding="utf-8")


def _extract_shell_function(source: str, function_name: str) -> str:
    lines = source.splitlines()
    start_idx = None
    for idx, line in enumerate(lines):
        if re.match(rf"^{re.escape(function_name)}\(\)\s*\{{\s*$", line):
            start_idx = idx
            break
    assert start_idx is not None, f"shell function {function_name} not found"

    body_lines: list[str] = []
    heredoc_delim: str | None = None
    for line in lines[start_idx + 1 :]:
        if heredoc_delim is not None:
            body_lines.append(line)
            if line == heredoc_delim:
                heredoc_delim = None
            continue
        if line == "}":
            return f"{function_name}() {{\n" + "\n".join(body_lines) + "\n}"
        body_lines.append(line)
        stripped = line.strip()
        match = re.search(r"<<-?'?([A-Za-z_][A-Za-z0-9_]*)'?$", stripped)
        if match is not None:
            heredoc_delim = match.group(1)

    raise AssertionError(f"shell function {function_name} has no closing brace")


def _extract_shell_function_between(source: str, function_name: str, next_function_name: str) -> str:
    start_marker = f"{function_name}() {{"
    end_marker = f"\n{next_function_name}() {{"
    start = source.find(start_marker)
    assert start != -1, f"shell function {function_name} not found"
    end = source.find(end_marker, start)
    assert end != -1, f"next shell function {next_function_name} not found after {function_name}"
    return source[start:end].rstrip() + "\n"


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
    selector_guard: bool = False,
    empty_expected: bool = False,
    pending_before_reconcile: int = 0,
    completed: int = 0,
    simple_mode: bool = False,
    completion_followup_handled: bool = False,
) -> subprocess.CompletedProcess[str]:
    count_fn = _extract_shell_function(_source(), "csv_data_row_count")
    has_rows_fn = _extract_shell_function(_source(), "candidate_file_has_rows")
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
{count_fn}
{has_rows_fn}
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
selector_empty_candidate_pool_guard() {{
{"  echo \"GUARD:$1\"\n  return 0\n" if selector_guard else "  return 1\n"}
}}
handle_empty_candidate_state() {{
{"  echo \"HANDLE_EMPTY_EXPECTED:$1\"\n  echo \"STOP:candidate_pool_empty_expected\"\n  return 0\n" if empty_expected else ("  echo \"GUARD:$1\"\n  return 2\n" if selector_guard else "  return 1\n")}
}}
handle_empty_candidate_or_exit() {{
  rc=0
  handle_empty_candidate_state "$1" || rc=$?
  if (( rc == 0 )); then
    return 0
  fi
  if (( rc == 2 )); then
    echo "FAIL_FAST:$1"
    exit 75
  fi
  return 1
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
maybe_process_completion_followup() {{
{"  echo \"FOLLOWUP:$1\"\n  return 0\n" if completion_followup_handled else "  return 1\n"}
}}
simple_control_plane_enabled() {{
{"  return 0\n" if simple_mode else "  return 1\n"}
}}
cycle_calls=0
run_fullspan_cycle() {{
  cycle_calls=$((cycle_calls + 1))
  echo "CYCLE:$1|$2|$3"
}}
completion_followup_enqueue_ready_queue() {{
  cycle_calls=$((cycle_calls + 1))
  echo "FOLLOWUP_ENQUEUE:$1|$2"
}}
maybe_kick_completion_followup_worker() {{
  echo "FOLLOWUP_WORKER:$1"
  return 0
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


def test_normalize_fullspan_queue_name_uses_parent_dir_for_default_queue_name(tmp_path: Path) -> None:
    fn = _extract_shell_function(_source(), "normalize_fullspan_queue_name")
    queue_path = tmp_path / "artifacts" / "wfa" / "aggregate" / "demo_group" / "run_queue.csv"
    queue_path.parent.mkdir(parents=True)
    queue_path.write_text("run_name,config_path,status\n", encoding="utf-8")

    script = f"""#!/usr/bin/env bash
set -euo pipefail
{fn}
normalize_fullspan_queue_name "{queue_path}" "run_queue.csv"
normalize_fullspan_queue_name "{queue_path}" ""
normalize_fullspan_queue_name "{queue_path}" "explicit_queue"
"""
    proc = subprocess.run(["bash", "-lc", script], cwd=tmp_path, capture_output=True, text=True)
    assert proc.returncode == 0, f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    assert proc.stdout.splitlines() == ["demo_group", "demo_group", "explicit_queue"]


def test_completion_followup_queue_enqueue_skips_duplicate_for_active_queue(tmp_path: Path) -> None:
    fn = _extract_shell_function(_source(), "completion_followup_queue_enqueue")
    queue_file = tmp_path / "completion_followup_queue.jsonl"
    worker_state_file = tmp_path / "completion_followup_worker_state.json"
    queue_lock_file = tmp_path / "completion_followup_queue.lock"
    queue_rel = "artifacts/wfa/aggregate/demo/run_queue.csv"
    worker_state_file.write_text(
        json.dumps(
            {
                "status": "active",
                "queue_rel": queue_rel,
                "pid": 123,
                "result": "processing",
                "backlog": 0,
                "ts": "2026-03-07T20:00:00Z",
            }
        ),
        encoding="utf-8",
    )

    script = f"""#!/usr/bin/env bash
set -euo pipefail
COMPLETION_FOLLOWUP_QUEUE_FILE="{queue_file}"
COMPLETION_FOLLOWUP_WORKER_STATE_FILE="{worker_state_file}"
COMPLETION_FOLLOWUP_QUEUE_LOCK_FILE="{queue_lock_file}"
{fn}
completion_followup_queue_enqueue "{queue_rel}" "track"
"""
    proc = subprocess.run(["bash", "-lc", script], cwd=tmp_path, capture_output=True, text=True)
    assert proc.returncode == 0, f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    assert proc.stdout.strip() == "0\t0"
    assert queue_file.read_text(encoding="utf-8") == ""


def test_completion_followup_queue_enqueue_allows_worker_requeue_for_active_queue(tmp_path: Path) -> None:
    fn = _extract_shell_function(_source(), "completion_followup_queue_enqueue")
    queue_file = tmp_path / "completion_followup_queue.jsonl"
    worker_state_file = tmp_path / "completion_followup_worker_state.json"
    queue_lock_file = tmp_path / "completion_followup_queue.lock"
    queue_rel = "artifacts/wfa/aggregate/demo/run_queue.csv"
    worker_state_file.write_text(
        json.dumps(
            {
                "status": "active",
                "queue_rel": queue_rel,
                "pid": 123,
                "result": "processing",
                "backlog": 0,
                "ts": "2026-03-07T20:00:00Z",
            }
        ),
        encoding="utf-8",
    )

    script = f"""#!/usr/bin/env bash
set -euo pipefail
COMPLETION_FOLLOWUP_QUEUE_FILE="{queue_file}"
COMPLETION_FOLLOWUP_WORKER_STATE_FILE="{worker_state_file}"
COMPLETION_FOLLOWUP_QUEUE_LOCK_FILE="{queue_lock_file}"
{fn}
completion_followup_queue_enqueue "{queue_rel}" "worker_requeue_pending"
cat "{queue_file}"
"""
    proc = subprocess.run(["bash", "-lc", script], cwd=tmp_path, capture_output=True, text=True)
    assert proc.returncode == 0, f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    lines = [line for line in proc.stdout.splitlines() if line.strip()]
    assert lines[0] == "1\t1"
    payload = json.loads(lines[1])
    assert payload["queue_rel"] == queue_rel


def test_completion_followup_queue_dequeue_preserves_empty_queue_rel_field(tmp_path: Path) -> None:
    fn = _extract_shell_function(_source(), "completion_followup_queue_dequeue")
    queue_file = tmp_path / "completion_followup_queue.jsonl"
    queue_lock_file = tmp_path / "completion_followup_queue.lock"
    queue_file.write_text("", encoding="utf-8")

    script = f"""#!/usr/bin/env bash
set -euo pipefail
COMPLETION_FOLLOWUP_QUEUE_FILE="{queue_file}"
COMPLETION_FOLLOWUP_QUEUE_LOCK_FILE="{queue_lock_file}"
{fn}
completion_followup_queue_dequeue
"""
    proc = subprocess.run(["bash", "-lc", script], cwd=tmp_path, capture_output=True, text=True)
    assert proc.returncode == 0, f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    assert proc.stdout.strip() == "|0"


def test_completion_followup_recover_stale_worker_does_not_parse_empty_queue_as_zero(tmp_path: Path) -> None:
    fn = _extract_shell_function(_source(), "completion_followup_recover_stale_worker")
    state_file = tmp_path / "completion_followup_worker_state.json"
    state_file.write_text(
        json.dumps(
            {
                "version": 1,
                "status": "idle",
                "queue_rel": "",
                "pid": 0,
                "result": "idle",
                "backlog": 0,
                "ts": "2026-03-08T00:00:00Z",
            }
        ),
        encoding="utf-8",
    )

    script = f"""#!/usr/bin/env bash
set -euo pipefail
COMPLETION_FOLLOWUP_WORKER_STATE_FILE="{state_file}"
COMPLETION_FOLLOWUP_WORKER_PID_FILE="{tmp_path / 'completion_followup_worker.pid'}"
{fn}
if completion_followup_recover_stale_worker; then
  echo RECOVERED
else
  echo SKIPPED
fi
"""
    proc = subprocess.run(["bash", "-lc", script], cwd=tmp_path, capture_output=True, text=True)
    assert proc.returncode == 0, f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    assert proc.stdout.strip() == "SKIPPED"


def test_fallback_pending_candidate_skips_active_cold_fail_queue(tmp_path: Path) -> None:
    fn = _extract_shell_function(_source(), "fallback_pending_candidate")
    queue_root = tmp_path / "artifacts" / "wfa" / "aggregate"
    toxic_queue = queue_root / "toxic" / "run_queue.csv"
    healthy_queue = queue_root / "healthy" / "run_queue.csv"
    toxic_queue.parent.mkdir(parents=True, exist_ok=True)
    healthy_queue.parent.mkdir(parents=True, exist_ok=True)
    config_dir = tmp_path / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / "toxic.yaml").write_text("demo: 1\n", encoding="utf-8")
    (config_dir / "healthy.yaml").write_text("demo: 1\n", encoding="utf-8")
    toxic_queue.write_text(
        "config_path,results_dir,status\nconfigs/toxic.yaml,artifacts/wfa/runs/toxic/run_01,stalled\n",
        encoding="utf-8",
    )
    healthy_queue.write_text(
        "config_path,results_dir,status\nconfigs/healthy.yaml,artifacts/wfa/runs/healthy/run_01,planned\n",
        encoding="utf-8",
    )
    now_epoch = int(time.time())
    cold_fail_path = tmp_path / "cold_fail_index.json"
    cold_fail_path.write_text(
        json.dumps(
            {
                "entries": [
                    {
                        "queue": "artifacts/wfa/aggregate/toxic/run_queue.csv",
                        "inserted_ts": now_epoch,
                        "until_ts": now_epoch + 3600,
                        "policy_version": "fullspan_v1",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    candidate_file = tmp_path / "candidate.csv"

    script = f"""#!/usr/bin/env bash
set -euo pipefail
QUEUE_ROOT="{queue_root}"
CANDIDATE_FILE="{candidate_file}"
COLD_FAIL_STATE_FILE="{cold_fail_path}"
{fn}
fallback_pending_candidate "" "" ""
tail -n 1 "$CANDIDATE_FILE"
"""
    proc = subprocess.run(["bash", "-lc", script], cwd=tmp_path, capture_output=True, text=True)
    assert proc.returncode == 0, f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    assert "artifacts/wfa/aggregate/healthy/run_queue.csv" in proc.stdout
    assert "artifacts/wfa/aggregate/toxic/run_queue.csv" not in proc.stdout


def test_candidate_reselect_after_reconcile_header_only_parse_falls_back_to_idle(tmp_path: Path) -> None:
    block = _extract_reselect_after_reconcile_block(_source())
    proc = _run_reselect_block(tmp_path, block, fallback_success=False)
    assert proc.returncode == 0, f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    assert "LOG:candidate_empty_after_reconcile" in proc.stdout
    assert "STATE:idle now=none completed=all" in proc.stdout
    assert "STOP:candidate_empty_after_reconcile" in proc.stdout
    assert "FALLBACK_CALLS:1" in proc.stdout


def test_candidate_reselect_after_reconcile_header_only_parse_can_recover_via_fallback(tmp_path: Path) -> None:
    block = _extract_reselect_after_reconcile_block(_source())
    proc = _run_reselect_block(tmp_path, block, fallback_success=True)
    assert proc.returncode == 0, f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    assert "LOG:candidate_reselect_after_reconcile queue=artifacts/wfa/aggregate/demo/run_queue.csv pending=1" in proc.stdout
    assert "BLOCK_DONE" in proc.stdout
    assert "LOG:candidate_empty_after_reconcile" not in proc.stdout
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
    assert "FOLLOWUP_ENQUEUE:artifacts/wfa/aggregate/current/run_queue.csv|candidate_reconcile" in proc.stdout
    assert "CYCLE_CALLS:1" in proc.stdout
    assert "LOG:candidate_reselect_after_reconcile queue=artifacts/wfa/aggregate/demo/run_queue.csv pending=1" in proc.stdout


def test_candidate_reselect_after_reconcile_selector_guard_skips_auto_seed(tmp_path: Path) -> None:
    block = _extract_reselect_after_reconcile_block(_source())
    proc = _run_reselect_block(tmp_path, block, fallback_success=False, selector_guard=True, simple_mode=True)
    assert proc.returncode == 75, f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    assert "LOG:candidate_empty_after_reconcile" in proc.stdout
    assert "GUARD:candidate_empty_after_reconcile" in proc.stdout
    assert "SEED:candidate_empty_after_reconcile" not in proc.stdout
    assert "FAIL_FAST:candidate_empty_after_reconcile" in proc.stdout


def test_candidate_reselect_after_reconcile_runs_completion_followup_before_idle(tmp_path: Path) -> None:
    block = _extract_reselect_after_reconcile_block(_source())
    proc = _run_reselect_block(
        tmp_path,
        block,
        fallback_success=False,
        completion_followup_handled=True,
    )
    assert proc.returncode == 0, f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    assert "LOG:candidate_empty_after_reconcile" in proc.stdout
    assert "FOLLOWUP:candidate_empty_after_reconcile" in proc.stdout
    assert "SEED:candidate_empty_after_reconcile" in proc.stdout
    assert "STOP:candidate_empty_after_reconcile" in proc.stdout


def test_candidate_empty_contract_triggers_auto_seed_in_simple_control_plane() -> None:
    src = _source()
    pattern = re.compile(
        r'if handle_empty_candidate_or_exit "candidate_empty"; then\s*'
        r'continue\s*'
        r'fi\s*'
        r'maybe_trigger_auto_seed "candidate_empty" \|\| true',
        re.DOTALL,
    )
    assert pattern.search(src), "candidate_empty path must trigger auto-seed before idle sleep in simple mode"


def test_handle_empty_candidate_state_triggers_auto_seed_before_idle_sleep(tmp_path: Path) -> None:
    handle_empty_fn = _extract_shell_function(_source(), "handle_empty_candidate_state")
    script = f"""#!/usr/bin/env bash
set -euo pipefail
adaptive_idle_sleep=1
{handle_empty_fn}
candidate_pool_status_snapshot() {{
  printf 'empty_expected\\t0\\t0\\t0\\t7\\t140\\n'
}}
fullspan_state_metric_set() {{ :; }}
log() {{ echo "LOG:$*"; }}
log_decision_note() {{ echo "NOTE:$*"; }}
maybe_trigger_auto_seed() {{ echo "SEED:$1"; return 0; }}
batch_session_maybe_stop() {{ echo "STOP:$1"; }}
selector_empty_candidate_pool_guard() {{ return 1; }}
sleep() {{ echo "SLEEP:$1"; }}
handle_empty_candidate_state candidate_empty
echo "ADAPTIVE_IDLE:$adaptive_idle_sleep"
"""
    proc = subprocess.run(["bash", "-lc", script], cwd=tmp_path, capture_output=True, text=True)
    assert proc.returncode == 0, f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    assert "LOG:candidate_pool_empty_expected reason=candidate_empty" in proc.stdout
    assert "SEED:candidate_pool_empty_degraded" in proc.stdout
    assert "STOP:candidate_pool_empty_expected" in proc.stdout
    assert "SLEEP:2" in proc.stdout
    assert "ADAPTIVE_IDLE:2" in proc.stdout


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


def test_no_progress_phase_switch_requires_current_session_dispatch_attempt_contract() -> None:
    src = _source()
    _assert_contains_all(
        src,
        [
            'queue_has_current_session_dispatch_attempt "$queue_rel"',
            'reason=no_dispatch_attempt',
            'record_dispatch_attempt()',
            'update_dispatch_attempt_result()',
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
            'candidate_pool_status',
            'candidate_pool_dispatchable_pending',
            'candidate_pool_executable_pending',
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
            'ready_buffer_snapshot_hash()',
            'ready_buffer_refresh()',
            'ready_buffer_emit_candidate()',
            'policy_hash',
            'snapshot_hash',
            'planner_policy_hash',
            'queue_file_mtime',
            'ready_buffer_policy_mismatch_count',
            'cold_fail_state_add()',
            'HARD_FAIL_COLD_TTL_SEC',
        ],
    )


def test_start_queue_uses_runtime_first_sync_policy_contract() -> None:
    src = _source()
    _assert_contains_all(
        src,
        [
            'local sync_code_policy="runtime-first"',
            '--sync-code-policy "$sync_code_policy"',
            'sync_policy=$sync_code_policy',
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
            'driver_idle_with_dispatchable_pending()',
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
            'powered: remote handoff confirmed queue=',
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
            'process_slo_controlled_recovery_snapshot()',
            'queue_coverage_policy_snapshot()',
            'AUTO_SEED_HARD_BLOCK',
            'CONTROLLED_RECOVERY_TRIGGER',
            'CONTROLLED_RECOVERY_SUCCESS',
            'CONTROLLED_RECOVERY_EXHAUSTED',
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


def test_wait_for_queue_start_confirmation_accepts_fast_completed_queue(tmp_path: Path) -> None:
    fn = _extract_shell_function(_source(), "wait_for_queue_start_confirmation")
    script = f"""#!/usr/bin/env bash
set -euo pipefail
START_QUEUE_CONFIRM_TIMEOUT_SEC=10
START_QUEUE_CONFIRM_POLL_SEC=0.1
START_QUEUE_SSH_READY_TIMEOUT_SEC=1
START_QUEUE_SYNC_TIMEOUT_SEC=5
START_QUEUE_STARTUP_BUDGET_SEC=5
{fn}
queue_start_confirmation_status() {{
  printf 'pending\\tWAITING_FOR_REMOTE_HANDOFF\\twaiting_for_remote_handoff\\n'
}}
local_powered_queue_running() {{
  return 1
}}
sync_queue_status() {{
  :
}}
queue_hygiene_snapshot() {{
  echo "0 0 0 0 0 0 0 0 8"
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
    assert "RC:0" in proc.stdout
    assert "OUT:REMOTE_HANDOFF_COMPLETED_FASTPATH" in proc.stdout
    assert "local_queue_completed_before_confirmation" in proc.stdout


def test_queue_start_confirmation_status_does_not_confirm_stalled_queue_without_remote_handoff(tmp_path: Path) -> None:
    fn = _extract_shell_function(_source(), "queue_start_confirmation_status")
    queue_path = tmp_path / "queue.csv"
    queue_path.write_text(
        "config_path,results_dir,status\n"
        "configs/demo.yaml,artifacts/wfa/runs/demo/run_01,stalled\n",
        encoding="utf-8",
    )
    qlog = tmp_path / "startup.log"
    qlog.write_text("powered: remote handoff start queue=queue.csv statuses=stalled\n", encoding="utf-8")

    script = f"""#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="{tmp_path}"
START_QUEUE_SSH_READY_TIMEOUT_SEC=60
START_QUEUE_SYNC_TIMEOUT_SEC=60
START_QUEUE_STARTUP_BUDGET_SEC=300
{fn}
queue_start_confirmation_status "queue.csv" "{qlog}" "$(date +%s)" "$START_QUEUE_SSH_READY_TIMEOUT_SEC" "$START_QUEUE_SYNC_TIMEOUT_SEC" "$START_QUEUE_STARTUP_BUDGET_SEC"
"""
    proc = subprocess.run(["bash", "-lc", script], cwd=tmp_path, capture_output=True, text=True)
    assert proc.returncode == 0, f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    assert proc.stdout.strip() == "pending\tWAITING_FOR_REMOTE_HANDOFF\twaiting_for_remote_handoff"


def test_queue_start_confirmation_status_confirms_all_completed_fastpath(tmp_path: Path) -> None:
    fn = _extract_shell_function(_source(), "queue_start_confirmation_status")
    queue_path = tmp_path / "queue.csv"
    queue_path.write_text(
        "config_path,results_dir,status\n"
        "configs/demo.yaml,artifacts/wfa/runs/demo/run_01,completed\n",
        encoding="utf-8",
    )
    qlog = tmp_path / "startup.log"
    qlog.write_text(
        "powered: auto_statuses counts={'completed': 8} chosen_statuses=<none> rationale=ALL_COMPLETED\n"
        "powered: queue-run skipped chosen_statuses=<none> rationale=ALL_COMPLETED\n",
        encoding="utf-8",
    )

    script = f"""#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="{tmp_path}"
START_QUEUE_SSH_READY_TIMEOUT_SEC=60
START_QUEUE_SYNC_TIMEOUT_SEC=60
START_QUEUE_STARTUP_BUDGET_SEC=300
{fn}
queue_start_confirmation_status "queue.csv" "{qlog}" "$(date +%s)" "$START_QUEUE_SSH_READY_TIMEOUT_SEC" "$START_QUEUE_SYNC_TIMEOUT_SEC" "$START_QUEUE_STARTUP_BUDGET_SEC"
"""
    proc = subprocess.run(["bash", "-lc", script], cwd=tmp_path, capture_output=True, text=True)
    assert proc.returncode == 0, f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    assert proc.stdout.strip() == "confirmed\tREMOTE_HANDOFF_COMPLETED_FASTPATH\tpowered_queue_already_completed"


def test_maybe_trigger_auto_seed_respects_infra_fail_closed_runtime_block(tmp_path: Path) -> None:
    heal_fn = _extract_shell_function(_source(), "reconcile_seed_next_retry_epoch")
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
REMOTE_RUNTIME_SNAPSHOT_MAX_AGE_SEC=90
{heal_fn}
{fn}
global_backlog_snapshot() {{
  echo "0 0 0 0"
}}
candidate_pool_status_snapshot() {{
  printf 'ready\\t1\\t0\\t0\\t0\\t0\\n'
}}
ready_buffer_depth() {{
  echo "0"
}}
yield_governor_hard_block_snapshot() {{
  printf '0\\t\\t0\\t0\\n'
}}
process_slo_remote_coverage_snapshot() {{
  printf '1\\t0\\n'
}}
process_slo_controlled_recovery_snapshot() {{
  printf '0\\t\\t0\\t0\\n'
}}
ensure_remote_runtime_snapshot() {{
  return 0
}}
remote_runtime_snapshot_is_fresh() {{
  echo "1"
}}
remote_runtime_state_value() {{
  echo "1"
}}
remote_active_queue_jobs() {{
  echo "0"
}}
remote_cpu_busy_without_queue_job() {{
  echo "0"
}}
remote_work_active() {{
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


def test_maybe_clear_sync_retryable_infra_fail_closed_clears_on_idle_reachable_vps(tmp_path: Path) -> None:
    retryable_fn = _extract_shell_function(_source(), "startup_failure_code_is_sync_retryable")
    clear_fn = _extract_shell_function(_source(), "maybe_clear_sync_retryable_infra_fail_closed")
    script = f"""#!/usr/bin/env bash
set -euo pipefail
REMOTE_RUNTIME_SNAPSHOT_MAX_AGE_SEC=90
{retryable_fn}
{clear_fn}
global_dispatchable_pending_count() {{
  echo "7"
}}
process_slo_remote_coverage_snapshot() {{
  printf '1\\t0\\n'
}}
refresh_remote_runtime_metrics() {{
  :
}}
ensure_remote_runtime_snapshot() {{
  return 0
}}
remote_runtime_snapshot_is_fresh() {{
  echo "1"
}}
remote_runtime_state_value() {{
  echo "1"
}}
remote_active_queue_jobs() {{
  echo "0"
}}
remote_postprocess_active() {{
  echo "0"
}}
remote_build_index_active() {{
  echo "0"
}}
remote_cpu_busy_without_queue_job() {{
  echo "0"
}}
remote_work_active() {{
  echo "0"
}}
fullspan_state_metric_get() {{
  case "$1" in
    startup_failure_code) echo "MANIFEST_MISMATCH" ;;
    infra_gate_status) echo "fail_closed" ;;
    vps_infra_fail_closed) echo "1" ;;
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
cleanup_orphans() {{
  echo "CLEANUP"
}}
log() {{
  printf 'LOG:%s\\n' "$*"
}}
log_decision_note() {{
  printf 'NOTE:%s|%s|%s|%s\\n' "${{1:-}}" "${{2:-}}" "${{3:-}}" "${{4:-}}"
}}
if maybe_clear_sync_retryable_infra_fail_closed loop_head; then
  echo "RC:0"
else
  rc=$?
  echo "RC:$rc"
fi
"""
    proc = subprocess.run(["bash", "-lc", script], cwd=tmp_path, capture_output=True, text=True)
    assert proc.returncode == 0, f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    assert "RC:0" in proc.stdout
    assert "SET:vps_infra_fail_closed=0" in proc.stdout
    assert "SET:auto_seed_hard_block=0" in proc.stdout
    assert "SET:auto_seed_block_reason=" in proc.stdout
    assert "INC:retryable_infra_gate_clear_count=1" in proc.stdout
    assert "INFRA:ok|||0|" in proc.stdout
    assert "CLEANUP" in proc.stdout
    assert "LOG:retryable_infra_gate_cleared reason=loop_head startup_failure_code=MANIFEST_MISMATCH" in proc.stdout
    assert "NOTE:global|INFRA_FAIL_CLOSED_CLEARED|" in proc.stdout


def test_maybe_clear_sync_retryable_infra_fail_closed_clears_remote_handoff_failure_without_queue_jobs(
    tmp_path: Path,
) -> None:
    retryable_fn = _extract_shell_function(_source(), "startup_failure_code_is_sync_retryable")
    clear_fn = _extract_shell_function(_source(), "maybe_clear_sync_retryable_infra_fail_closed")
    script = f"""#!/usr/bin/env bash
set -euo pipefail
REMOTE_RUNTIME_SNAPSHOT_MAX_AGE_SEC=90
{retryable_fn}
{clear_fn}
global_dispatchable_pending_count() {{
  echo "5"
}}
process_slo_remote_coverage_snapshot() {{
  printf '1\\t0\\n'
}}
refresh_remote_runtime_metrics() {{
  :
}}
ensure_remote_runtime_snapshot() {{
  return 0
}}
remote_runtime_snapshot_is_fresh() {{
  echo "1"
}}
remote_runtime_state_value() {{
  echo "1"
}}
remote_active_queue_jobs() {{
  echo "0"
}}
remote_postprocess_active() {{
  echo "0"
}}
remote_build_index_active() {{
  echo "0"
}}
remote_cpu_busy_without_queue_job() {{
  echo "1"
}}
remote_work_active() {{
  echo "1"
}}
fullspan_state_metric_get() {{
  case "$1" in
    startup_failure_code) echo "REMOTE_HANDOFF_FAILED" ;;
    infra_gate_status) echo "fail_closed" ;;
    vps_infra_fail_closed) echo "1" ;;
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
cleanup_orphans() {{
  echo "CLEANUP"
}}
log() {{
  printf 'LOG:%s\\n' "$*"
}}
log_decision_note() {{
  printf 'NOTE:%s|%s|%s|%s\\n' "${{1:-}}" "${{2:-}}" "${{3:-}}" "${{4:-}}"
}}
if maybe_clear_sync_retryable_infra_fail_closed loop_head; then
  echo "RC:0"
else
  rc=$?
  echo "RC:$rc"
fi
"""
    proc = subprocess.run(["bash", "-lc", script], cwd=tmp_path, capture_output=True, text=True)
    assert proc.returncode == 0, f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    assert "RC:0" in proc.stdout
    assert "SET:vps_infra_fail_closed=0" in proc.stdout
    assert "SET:auto_seed_hard_block=0" in proc.stdout
    assert "INFRA:ok|||0|" in proc.stdout
    assert "LOG:retryable_infra_gate_cleared reason=loop_head startup_failure_code=REMOTE_HANDOFF_FAILED" in proc.stdout


def test_maybe_clear_fast_completed_start_process_exit_fail_closed_clears_stale_block(tmp_path: Path) -> None:
    clear_fn = _extract_shell_function(_source(), "maybe_clear_fast_completed_start_process_exit_fail_closed")
    state_path = tmp_path / "fullspan_decision_state.json"
    state_path.write_text(
        json.dumps(
            {
                "queues": {
                    "artifacts/wfa/aggregate/demo/run_queue.csv": {
                        "startup_state": "fail_closed",
                        "startup_failure_code": "START_PROCESS_EXIT",
                        "startup_failure_reason": "local_powered_runner_exited_before_confirmation",
                        "dispatch_attempt_result": "failed",
                    }
                }
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    script = f"""#!/usr/bin/env bash
set -euo pipefail
FULLSPAN_DECISION_STATE_FILE="{state_path}"
{clear_fn}
refresh_remote_runtime_metrics() {{
  :
}}
ensure_remote_runtime_snapshot() {{
  return 0
}}
remote_active_queue_jobs() {{
  echo "0"
}}
remote_postprocess_active() {{
  echo "0"
}}
remote_build_index_active() {{
  echo "0"
}}
remote_work_active() {{
  echo "0"
}}
queue_hygiene_snapshot() {{
  echo "0 0 0 0 0 0 0 0 8"
}}
fullspan_state_metric_get() {{
  case "$1" in
    startup_failure_code) echo "START_PROCESS_EXIT" ;;
    infra_gate_status) echo "fail_closed" ;;
    vps_infra_fail_closed) echo "0" ;;
    auto_seed_blocked) echo "1" ;;
    auto_seed_block_reason) echo "local_powered_runner_exited_before_confirmation" ;;
    *) echo "0" ;;
  esac
}}
fullspan_state_metric_set() {{
  printf 'SET:%s=%s\\n' "$1" "${{2:-}}"
}}
fullspan_state_metric_inc() {{
  printf 'INC:%s=%s\\n' "$1" "${{2:-}}"
}}
fullspan_state_queue_set() {{
  printf 'QUEUESET:%s|%s\\n' "$1" "${{*:2}}"
}}
clear_orphan() {{
  printf 'CLEAR_ORPHAN:%s\\n' "$1"
}}
completion_followup_enqueue_ready_queue() {{
  printf 'FOLLOWUP:%s|%s\\n' "$1" "$2"
}}
set_infra_gate_state() {{
  printf 'INFRA:%s|%s|%s|%s|%s\\n' "${{1:-}}" "${{2:-}}" "${{3:-}}" "${{4:-}}" "${{5:-}}"
}}
cleanup_orphans() {{
  echo "CLEANUP"
}}
log() {{
  printf 'LOG:%s\\n' "$*"
}}
log_decision_note() {{
  printf 'NOTE:%s|%s|%s|%s\\n' "${{1:-}}" "${{2:-}}" "${{3:-}}" "${{4:-}}"
}}
if maybe_clear_fast_completed_start_process_exit_fail_closed loop_head; then
  echo "RC:0"
else
  rc=$?
  echo "RC:$rc"
fi
"""
    proc = subprocess.run(["bash", "-lc", script], cwd=tmp_path, capture_output=True, text=True)
    assert proc.returncode == 0, f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    assert "RC:0" in proc.stdout
    assert "QUEUESET:artifacts/wfa/aggregate/demo/run_queue.csv|" in proc.stdout
    assert "dispatch_attempt_result completed_fastpath" in proc.stdout
    assert "CLEAR_ORPHAN:artifacts/wfa/aggregate/demo/run_queue.csv" in proc.stdout
    assert "SET:vps_infra_fail_closed=0" in proc.stdout
    assert "SET:auto_seed_hard_block=0" in proc.stdout
    assert "INFRA:ok|||0|" in proc.stdout
    assert "LOG:retryable_infra_gate_cleared reason=loop_head startup_failure_code=START_PROCESS_EXIT healed_fast_completed_queues=1" in proc.stdout


def test_clear_retryable_remote_handoff_queue_blocks_removes_stale_queue_blocks(tmp_path: Path) -> None:
    fn = _extract_shell_function(_source(), "clear_retryable_remote_handoff_queue_blocks")
    orphan_path = tmp_path / "orphan_queues.csv"
    orphan_path.write_text(
        "queue,until_ts,reason\n"
        "artifacts/wfa/aggregate/demo_a/run_queue.csv,9999999999,infra_fail_closed_REMOTE_HANDOFF_FAILED\n"
        "artifacts/wfa/aggregate/demo_b/run_queue.csv,9999999999,queue_start_failed_remote_handoff\n"
        "artifacts/wfa/aggregate/demo_c/run_queue.csv,9999999999,dispatch_block_fail_closed\n"
        "artifacts/wfa/aggregate/demo_d/run_queue.csv,9999999999,infra_fail_closed_MANIFEST_MISMATCH\n",
        encoding="utf-8",
    )
    state_path = tmp_path / "fullspan_decision_state.json"
    state_path.write_text(
        json.dumps(
            {
                "queues": {
                    "artifacts/wfa/aggregate/demo_a/run_queue.csv": {
                        "promotion_verdict": "ANALYZE",
                        "startup_state": "fail_closed",
                        "startup_failure_code": "REMOTE_HANDOFF_FAILED",
                        "startup_failure_reason": "powered_fail",
                        "dispatch_attempt_result": "failed",
                        "dispatch_attempt_session_epoch": 123,
                    },
                    "artifacts/wfa/aggregate/demo_b/run_queue.csv": {
                        "promotion_verdict": "ANALYZE",
                        "startup_state": "failed",
                        "startup_failure_code": "REMOTE_HANDOFF_FAILED",
                        "startup_failure_reason": "powered_fail",
                        "dispatch_attempt_result": "remote_handoff_failed",
                        "dispatch_attempt_session_epoch": 456,
                    },
                    "artifacts/wfa/aggregate/demo_c/run_queue.csv": {
                        "promotion_verdict": "ANALYZE",
                        "startup_state": "fail_closed",
                        "startup_failure_code": "REMOTE_HANDOFF_FAILED",
                        "startup_failure_reason": "powered_fail",
                        "dispatch_attempt_result": "failed",
                        "dispatch_attempt_session_epoch": 789,
                    },
                    "artifacts/wfa/aggregate/demo_d/run_queue.csv": {
                        "promotion_verdict": "REJECT",
                        "strict_gate_status": "FULLSPAN_PREFILTER_REJECT",
                        "startup_state": "fail_closed",
                        "startup_failure_code": "REMOTE_HANDOFF_FAILED",
                        "dispatch_attempt_result": "failed",
                        "dispatch_attempt_session_epoch": 999,
                    },
                }
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    script = f"""#!/usr/bin/env bash
set -euo pipefail
ORPHAN_FILE="{orphan_path}"
FULLSPAN_DECISION_STATE_FILE="{state_path}"
{fn}
clear_retryable_remote_handoff_queue_blocks
"""
    proc = subprocess.run(["bash", "-lc", script], cwd=tmp_path, capture_output=True, text=True)
    assert proc.returncode == 0, f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    assert proc.stdout.strip() == "3"

    state = json.loads(state_path.read_text(encoding="utf-8"))
    demo_a = state["queues"]["artifacts/wfa/aggregate/demo_a/run_queue.csv"]
    assert demo_a["startup_state"] == ""
    assert demo_a["startup_failure_code"] == ""
    assert demo_a["startup_failure_reason"] == ""
    assert demo_a["dispatch_attempt_result"] == ""
    assert demo_a["dispatch_attempt_session_epoch"] == 0
    demo_d = state["queues"]["artifacts/wfa/aggregate/demo_d/run_queue.csv"]
    assert demo_d["startup_failure_code"] == "REMOTE_HANDOFF_FAILED"

    orphan_rows = orphan_path.read_text(encoding="utf-8")
    assert "demo_a" not in orphan_rows
    assert "demo_b" not in orphan_rows
    assert "demo_c" not in orphan_rows
    assert "demo_d" in orphan_rows


def test_maybe_trigger_auto_seed_respects_remote_recovery_mode_block(tmp_path: Path) -> None:
    heal_fn = _extract_shell_function(_source(), "reconcile_seed_next_retry_epoch")
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
REMOTE_RUNTIME_SNAPSHOT_MAX_AGE_SEC=90
{heal_fn}
{fn}
global_backlog_snapshot() {{
  echo "0 0 0 0"
}}
candidate_pool_status_snapshot() {{
  printf 'ready\\t1\\t0\\t0\\t0\\t0\\n'
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
process_slo_controlled_recovery_snapshot() {{
  printf '0\\t\\t0\\t0\\n'
}}
ensure_remote_runtime_snapshot() {{
  return 0
}}
remote_runtime_snapshot_is_fresh() {{
  echo "1"
}}
remote_runtime_state_value() {{
  echo "0"
}}
remote_active_queue_jobs() {{
  echo "0"
}}
remote_cpu_busy_without_queue_job() {{
  echo "0"
}}
remote_work_active() {{
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


def test_maybe_trigger_auto_seed_logs_remote_state_mismatch_without_hard_block(tmp_path: Path) -> None:
    heal_fn = _extract_shell_function(_source(), "reconcile_seed_next_retry_epoch")
    fn = _extract_shell_function(_source(), "maybe_trigger_auto_seed")
    script = f"""#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="{tmp_path}"
LOG_FILE="{tmp_path / 'driver.log'}"
AUTO_SEED_PENDING_THRESHOLD=96
READY_BUFFER_REFILL_THRESHOLD=2
AUTO_SEED_COOLDOWN_SEC=999999
AUTO_SEED_NUM_VARIANTS=64
AUTO_SEED_NUM_VARIANTS_FLOOR=24
REMOTE_RUNTIME_SNAPSHOT_MAX_AGE_SEC=90
{heal_fn}
{fn}
simple_control_plane_enabled() {{
  return 1
}}
reconcile_selector_backlog_before_seed() {{
  return 0
}}
maybe_clear_sync_retryable_infra_fail_closed() {{
  return 0
}}
global_backlog_snapshot() {{
  echo "0 0 0 0"
}}
candidate_pool_status_snapshot() {{
  printf 'ready\\t1\\t0\\t0\\t0\\t0\\n'
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
process_slo_controlled_recovery_snapshot() {{
  printf '0\\t\\t0\\t0\\n'
}}
ensure_remote_runtime_snapshot() {{
  return 0
}}
remote_runtime_snapshot_is_fresh() {{
  echo "1"
}}
remote_runtime_state_value() {{
  echo "1"
}}
remote_active_queue_jobs() {{
  echo "0"
}}
remote_cpu_busy_without_queue_job() {{
  echo "0"
}}
remote_work_active() {{
  echo "0"
}}
fullspan_state_metric_get() {{
  case "$1" in
    last_seed_trigger_epoch) echo "9999999999" ;;
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
    assert "LOG:REMOTE_STATE_MISMATCH reason=candidate_empty_after_reconcile process_remote_reachable=0 remote_runtime_reachable=1 coverage_ready=0" in proc.stdout
    assert "NOTE:global|REMOTE_STATE_MISMATCH|" in proc.stdout
    assert "infra_recovery_mode_remote_unreachable_no_coverage_ready" not in proc.stdout
    assert "AUTO_SEED_HARD_BLOCK" not in proc.stdout


def test_maybe_trigger_auto_seed_suppresses_remote_state_mismatch_when_empty_expected_idle(tmp_path: Path) -> None:
    heal_fn = _extract_shell_function(_source(), "reconcile_seed_next_retry_epoch")
    fn = _extract_shell_function(_source(), "maybe_trigger_auto_seed")
    script = f"""#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="{tmp_path}"
LOG_FILE="{tmp_path / 'driver.log'}"
AUTO_SEED_PENDING_THRESHOLD=96
READY_BUFFER_REFILL_THRESHOLD=2
AUTO_SEED_COOLDOWN_SEC=999999
AUTO_SEED_NUM_VARIANTS=64
AUTO_SEED_NUM_VARIANTS_FLOOR=24
REMOTE_RUNTIME_SNAPSHOT_MAX_AGE_SEC=90
{heal_fn}
{fn}
simple_control_plane_enabled() {{
  return 1
}}
reconcile_selector_backlog_before_seed() {{
  return 0
}}
maybe_clear_sync_retryable_infra_fail_closed() {{
  return 0
}}
global_backlog_snapshot() {{
  echo "0 0 1 5"
}}
candidate_pool_status_snapshot() {{
  printf 'empty_expected\\t0\\t0\\t0\\t1\\t5\\n'
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
process_slo_controlled_recovery_snapshot() {{
  printf '0\\t\\t0\\t0\\n'
}}
ensure_remote_runtime_snapshot() {{
  return 0
}}
remote_runtime_snapshot_is_fresh() {{
  echo "1"
}}
remote_runtime_state_value() {{
  case "$1" in
    reachable) echo "1" ;;
    remote_work_active) echo "0" ;;
    cpu_busy_without_queue_job) echo "0" ;;
    *) echo "0" ;;
  esac
}}
remote_active_queue_jobs() {{
  echo "0"
}}
remote_cpu_busy_without_queue_job() {{
  echo "0"
}}
remote_work_active() {{
  echo "0"
}}
fullspan_state_metric_get() {{
  case "$1" in
    last_seed_trigger_epoch) echo "9999999999" ;;
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
    assert "SET:remote_state_mismatch=1" in proc.stdout
    assert "SET:remote_state_mismatch_idle_suppressed=1" in proc.stdout
    assert "LOG:REMOTE_STATE_MISMATCH" not in proc.stdout
    assert "NOTE:global|REMOTE_STATE_MISMATCH|" not in proc.stdout
    assert "AUTO_SEED_HARD_BLOCK" not in proc.stdout


def test_maybe_trigger_auto_seed_allows_controlled_recovery_for_zero_coverage_hard_block(tmp_path: Path) -> None:
    heal_fn = _extract_shell_function(_source(), "reconcile_seed_next_retry_epoch")
    fn = _extract_shell_function(_source(), "maybe_trigger_auto_seed")
    seeder_python = tmp_path / ".venv" / "bin" / "python"
    seeder_python.parent.mkdir(parents=True, exist_ok=True)
    seeder_args = tmp_path / "seeder_args.txt"
    seeder_state = tmp_path / "artifacts" / "wfa" / "aggregate" / ".autonomous" / "queue_seeder.state.json"
    seeder_state.parent.mkdir(parents=True, exist_ok=True)
    seeder_python.write_text(
        "#!/usr/bin/env bash\n"
        "printf '%s\\n' \"$@\" > \"" + str(seeder_args) + "\"\n"
        "mkdir -p \"" + str(seeder_state.parent) + "\"\n"
        "cat > \"" + str(seeder_state) + "\" <<'JSON'\n"
        '{"status":"seeded","status_detail":"queued","reason":"","run_group":"autonomous_seed_demo","queue_path":"artifacts/wfa/aggregate/autonomous_seed_demo/run_queue.csv","queue_rows_generated":8,"next_retry_epoch":0}\n'
        "JSON\n"
        "exit 0\n",
        encoding="utf-8",
    )
    seeder_python.chmod(0o755)

    script = f"""#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="{tmp_path}"
LOG_FILE="{tmp_path / 'driver.log'}"
AUTO_SEED_PENDING_THRESHOLD=96
READY_BUFFER_REFILL_THRESHOLD=2
AUTO_SEED_COOLDOWN_SEC=0
AUTO_SEED_NUM_VARIANTS=64
AUTO_SEED_NUM_VARIANTS_FLOOR=24
REMOTE_RUNTIME_SNAPSHOT_MAX_AGE_SEC=90
{heal_fn}
{fn}
simple_control_plane_enabled() {{
  return 1
}}
global_backlog_snapshot() {{
  echo "0 0 0 0"
}}
candidate_pool_status_snapshot() {{
  printf 'empty_expected\\t0\\t0\\t0\\t0\\t0\\n'
}}
ready_buffer_depth() {{
  echo "0"
}}
yield_governor_hard_block_snapshot() {{
  printf '1\\tzero_coverage_seed_streak\\t0\\t8\\n'
}}
process_slo_remote_coverage_snapshot() {{
  printf '1\\t0\\n'
}}
process_slo_controlled_recovery_snapshot() {{
  printf '1\\tzero_coverage_seed_streak_with_positive_lineage\\t2\\t8\\n'
}}
reconcile_selector_backlog_before_seed() {{
  return 0
}}
maybe_clear_sync_retryable_infra_fail_closed() {{
  return 0
}}
ensure_remote_runtime_snapshot() {{
  return 0
}}
remote_runtime_snapshot_is_fresh() {{
  echo "1"
}}
remote_runtime_state_value() {{
  echo "1"
}}
remote_active_queue_jobs() {{
  echo "0"
}}
remote_cpu_busy_without_queue_job() {{
  echo "0"
}}
remote_work_active() {{
  echo "0"
}}
fullspan_state_metric_get() {{
  case "$1" in
    last_seed_trigger_epoch) echo "0" ;;
    auto_seed_hard_block) echo "0" ;;
    auto_seed_block_reason) echo "" ;;
    controlled_recovery_exhausted) echo "0" ;;
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
    assert "LOG:CONTROLLED_RECOVERY_TRIGGER reason=candidate_empty_after_reconcile" in proc.stdout
    assert "NOTE:global|CONTROLLED_RECOVERY_TRIGGER|" in proc.stdout
    assert "LOG:CONTROLLED_RECOVERY_SUCCESS reason=candidate_empty_after_reconcile" in proc.stdout
    assert "NOTE:global|CONTROLLED_RECOVERY_SUCCESS|" in proc.stdout
    assert "NOTE:global|AUTO_SEED_TRIGGER|" in proc.stdout
    assert "AUTO_SEED_HARD_BLOCK" not in proc.stdout
    seeder_args_text = seeder_args.read_text(encoding="utf-8")
    assert "--num-variants" in seeder_args_text
    assert "\n8\n" in seeder_args_text
    assert "--num-variants-floor" in seeder_args_text


def test_maybe_trigger_auto_seed_requires_empty_expected_for_controlled_recovery(tmp_path: Path) -> None:
    heal_fn = _extract_shell_function(_source(), "reconcile_seed_next_retry_epoch")
    fn = _extract_shell_function(_source(), "maybe_trigger_auto_seed")
    seeder_python = tmp_path / ".venv" / "bin" / "python"
    seeder_python.parent.mkdir(parents=True, exist_ok=True)
    seeder_args = tmp_path / "seeder_args.txt"
    seeder_python.write_text(
        "#!/usr/bin/env bash\n"
        "printf '%s\\n' \"$@\" > \"" + str(seeder_args) + "\"\n"
        "exit 0\n",
        encoding="utf-8",
    )
    seeder_python.chmod(0o755)

    script = f"""#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="{tmp_path}"
LOG_FILE="{tmp_path / 'driver.log'}"
AUTO_SEED_PENDING_THRESHOLD=96
READY_BUFFER_REFILL_THRESHOLD=2
AUTO_SEED_COOLDOWN_SEC=0
AUTO_SEED_NUM_VARIANTS=64
AUTO_SEED_NUM_VARIANTS_FLOOR=24
REMOTE_RUNTIME_SNAPSHOT_MAX_AGE_SEC=90
{heal_fn}
{fn}
global_backlog_snapshot() {{
  echo "0 0 0 0"
}}
candidate_pool_status_snapshot() {{
  printf 'empty_error\\t0\\t0\\t0\\t0\\t0\\n'
}}
ready_buffer_depth() {{
  echo "0"
}}
yield_governor_hard_block_snapshot() {{
  printf '1\\tzero_coverage_seed_streak\\t0\\t8\\n'
}}
process_slo_remote_coverage_snapshot() {{
  printf '1\\t0\\n'
}}
process_slo_controlled_recovery_snapshot() {{
  printf '1\\tzero_coverage_seed_streak_with_positive_lineage\\t2\\t8\\n'
}}
reconcile_selector_backlog_before_seed() {{
  return 0
}}
ensure_remote_runtime_snapshot() {{
  return 0
}}
remote_runtime_snapshot_is_fresh() {{
  echo "1"
}}
remote_runtime_state_value() {{
  echo "1"
}}
remote_active_queue_jobs() {{
  echo "0"
}}
remote_cpu_busy_without_queue_job() {{
  echo "0"
}}
remote_work_active() {{
  echo "0"
}}
fullspan_state_metric_get() {{
  case "$1" in
    last_seed_trigger_epoch) echo "0" ;;
    auto_seed_hard_block) echo "0" ;;
    auto_seed_block_reason) echo "" ;;
    controlled_recovery_exhausted) echo "0" ;;
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
    assert "CONTROLLED_RECOVERY_TRIGGER" not in proc.stdout
    assert "AUTO_SEED_HARD_BLOCK" in proc.stdout
    assert not seeder_args.exists()


def test_maybe_trigger_auto_seed_logs_controlled_recovery_exhausted_under_zero_coverage_hard_block(tmp_path: Path) -> None:
    heal_fn = _extract_shell_function(_source(), "reconcile_seed_next_retry_epoch")
    fn = _extract_shell_function(_source(), "maybe_trigger_auto_seed")
    seeder_python = tmp_path / ".venv" / "bin" / "python"
    seeder_python.parent.mkdir(parents=True, exist_ok=True)
    seeder_state = tmp_path / "artifacts" / "wfa" / "aggregate" / ".autonomous" / "queue_seeder.state.json"
    seeder_state.parent.mkdir(parents=True, exist_ok=True)
    seeder_python.write_text(
        "#!/usr/bin/env bash\n"
        "mkdir -p \"" + str(seeder_state.parent) + "\"\n"
        "cat > \"" + str(seeder_state) + "\" <<'JSON'\n"
        '{"status":"skipped","status_detail":"hard_block","reason":"hard_block:zero_coverage_seed_streak","run_group":"","queue_path":"","queue_rows_generated":0,"next_retry_epoch":1773000000}\n'
        "JSON\n"
        "exit 0\n",
        encoding="utf-8",
    )
    seeder_python.chmod(0o755)
    script = f"""#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="{tmp_path}"
LOG_FILE="{tmp_path / 'driver.log'}"
AUTO_SEED_PENDING_THRESHOLD=96
READY_BUFFER_REFILL_THRESHOLD=2
AUTO_SEED_COOLDOWN_SEC=0
AUTO_SEED_NUM_VARIANTS=64
AUTO_SEED_NUM_VARIANTS_FLOOR=24
REMOTE_RUNTIME_SNAPSHOT_MAX_AGE_SEC=90
{heal_fn}
{fn}
simple_control_plane_enabled() {{
  return 1
}}
global_backlog_snapshot() {{
  echo "0 0 0 0"
}}
candidate_pool_status_snapshot() {{
  printf 'empty_expected\\t0\\t0\\t0\\t0\\t0\\n'
}}
ready_buffer_depth() {{
  echo "0"
}}
yield_governor_hard_block_snapshot() {{
  printf '1\\tzero_coverage_seed_streak\\t0\\t8\\n'
}}
process_slo_remote_coverage_snapshot() {{
  printf '1\\t0\\n'
}}
process_slo_controlled_recovery_snapshot() {{
  printf '1\\tzero_coverage_seed_streak_with_positive_lineage\\t0\\t8\\n'
}}
reconcile_selector_backlog_before_seed() {{
  return 0
}}
maybe_clear_sync_retryable_infra_fail_closed() {{
  return 0
}}
ensure_remote_runtime_snapshot() {{
  return 0
}}
remote_runtime_snapshot_is_fresh() {{
  echo "1"
}}
remote_runtime_state_value() {{
  echo "1"
}}
remote_active_queue_jobs() {{
  echo "0"
}}
remote_cpu_busy_without_queue_job() {{
  echo "0"
}}
remote_work_active() {{
  echo "0"
}}
fullspan_state_metric_get() {{
  case "$1" in
    last_seed_trigger_epoch) echo "0" ;;
    auto_seed_hard_block) echo "0" ;;
    auto_seed_block_reason) echo "" ;;
    controlled_recovery_exhausted) echo "0" ;;
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
    assert "LOG:CONTROLLED_RECOVERY_EXHAUSTED reason=candidate_empty_after_reconcile" in proc.stdout
    assert "NOTE:global|CONTROLLED_RECOVERY_EXHAUSTED|" in proc.stdout
    assert "delegate_zero_coverage_rearm_to_seeder" in proc.stdout
    assert "NOTE:global|AUTO_SEED_SKIPPED|" in proc.stdout
    assert "NOTE:global|AUTO_SEED_REARM_SCHEDULED|" in proc.stdout
    assert "AUTO_SEED_TRIGGER" not in proc.stdout


def test_maybe_trigger_auto_seed_self_heals_stale_zero_coverage_retry_epoch(tmp_path: Path) -> None:
    heal_fn = _extract_shell_function(_source(), "reconcile_seed_next_retry_epoch")
    fn = _extract_shell_function(_source(), "maybe_trigger_auto_seed")
    seeder_python = tmp_path / ".venv" / "bin" / "python"
    seeder_python.parent.mkdir(parents=True, exist_ok=True)
    seeder_args = tmp_path / "seeder_args.txt"
    state_dir = tmp_path / "artifacts" / "wfa" / "aggregate" / ".autonomous"
    state_dir.mkdir(parents=True, exist_ok=True)
    seeder_state = state_dir / "queue_seeder.state.json"
    yield_state = state_dir / "yield_governor_state.json"
    directive_state = state_dir / "search_director_directive.json"
    fullspan_state = state_dir / "fullspan_decision_state.json"
    fullspan_state.write_text(
        json.dumps({"runtime_metrics": {"seed_last_skip_reason": "hard_block:zero_coverage_seed_streak"}}),
        encoding="utf-8",
    )
    seeder_state.write_text(
        json.dumps(
            {
                "status": "skipped",
                "status_detail": "hard_block",
                "reason": "hard_block:zero_coverage_seed_streak",
                "next_retry_epoch": 1773000000,
                "controlled_broad_rearm_after_epoch": 1,
            }
        ),
        encoding="utf-8",
    )
    yield_state.write_text(
        json.dumps(
            {
                "hard_block_active": True,
                "hard_block_reason": "zero_coverage_seed_streak",
                "controlled_broad_rearm_after_epoch": 1,
                "search_quality": {
                    "controlled_recovery_attempts_remaining": 0,
                    "controlled_broad_rearm_after_epoch": 1,
                },
            }
        ),
        encoding="utf-8",
    )
    directive_state.write_text(
        json.dumps(
            {
                "search_quality": {
                    "controlled_recovery_attempts_remaining": 0,
                    "controlled_broad_rearm_after_epoch": 1,
                },
                "controlled_broad_recovery": {
                    "enabled": True,
                    "reason": "controlled_broad_rearm_after_exhausted_recovery",
                    "rearm_after_epoch": 1,
                },
            }
        ),
        encoding="utf-8",
    )
    seeder_python.write_text(
        "#!/usr/bin/env bash\n"
        "printf '%s\\n' \"$@\" > \"" + str(seeder_args) + "\"\n"
        "cat > \"" + str(seeder_state) + "\" <<'JSON'\n"
        '{"status":"seeded","status_detail":"queued","reason":"","run_group":"autonomous_seed_demo","queue_path":"artifacts/wfa/aggregate/autonomous_seed_demo/run_queue.csv","queue_rows_generated":8,"next_retry_epoch":0}\n'
        "JSON\n"
        "exit 0\n",
        encoding="utf-8",
    )
    seeder_python.chmod(0o755)

    script = f"""#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="{tmp_path}"
LOG_FILE="{tmp_path / 'driver.log'}"
FULLSPAN_DECISION_STATE_FILE="{fullspan_state}"
QUEUE_SEEDER_STATE_FILE="{seeder_state}"
YIELD_GOVERNOR_STATE_FILE="{yield_state}"
SEARCH_DIRECTOR_DIRECTIVE_FILE="{directive_state}"
AUTO_SEED_PENDING_THRESHOLD=96
READY_BUFFER_REFILL_THRESHOLD=2
AUTO_SEED_COOLDOWN_SEC=0
AUTO_SEED_NUM_VARIANTS=64
AUTO_SEED_NUM_VARIANTS_FLOOR=24
REMOTE_RUNTIME_SNAPSHOT_MAX_AGE_SEC=90
{heal_fn}
{fn}
simple_control_plane_enabled() {{
  return 1
}}
global_backlog_snapshot() {{
  echo "0 0 0 0"
}}
candidate_pool_status_snapshot() {{
  printf 'empty_expected\\t0\\t0\\t0\\t0\\t0\\n'
}}
ready_buffer_depth() {{
  echo "0"
}}
yield_governor_hard_block_snapshot() {{
  printf '1\\tzero_coverage_seed_streak\\t0\\t8\\n'
}}
process_slo_remote_coverage_snapshot() {{
  printf '1\\t0\\n'
}}
process_slo_controlled_recovery_snapshot() {{
  printf '1\\tzero_coverage_seed_streak_with_positive_lineage\\t0\\t8\\n'
}}
reconcile_selector_backlog_before_seed() {{
  return 0
}}
maybe_clear_sync_retryable_infra_fail_closed() {{
  return 0
}}
ensure_remote_runtime_snapshot() {{
  return 0
}}
remote_runtime_snapshot_is_fresh() {{
  echo "1"
}}
remote_runtime_state_value() {{
  echo "1"
}}
remote_active_queue_jobs() {{
  echo "0"
}}
remote_cpu_busy_without_queue_job() {{
  echo "0"
}}
remote_work_active() {{
  echo "0"
}}
fullspan_state_metric_get() {{
  case "$1" in
    last_seed_trigger_epoch) echo "0" ;;
    seed_next_retry_epoch) echo "1773000000" ;;
    auto_seed_hard_block) echo "0" ;;
    auto_seed_block_reason) echo "" ;;
    controlled_recovery_exhausted) echo "0" ;;
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
    assert proc.returncode == 0, f"stdout:\\n{proc.stdout}\\nstderr:\\n{proc.stderr}"
    assert "SET:seed_next_retry_epoch=0" in proc.stdout
    assert "NOTE:global|AUTO_SEED_TRIGGER|" in proc.stdout
    assert seeder_args.exists()


def test_process_slo_controlled_recovery_snapshot_reads_search_quality_fields(tmp_path: Path) -> None:
    fn = _extract_shell_function(_source(), "process_slo_controlled_recovery_snapshot")
    state_path = tmp_path / "process_slo_state.json"
    state_path.write_text(
        json.dumps(
            {
                "search_quality": {
                    "controlled_recovery_active": True,
                    "controlled_recovery_reason": "zero_coverage_seed_streak_with_positive_lineage",
                    "controlled_recovery_attempts_remaining": 2,
                    "controlled_recovery_variants_cap": 8,
                }
            }
        ),
        encoding="utf-8",
    )
    script = f"""#!/usr/bin/env bash
set -euo pipefail
PROCESS_SLO_STATE_FILE="{state_path}"
{fn}
process_slo_controlled_recovery_snapshot
"""
    proc = subprocess.run(["bash", "-lc", script], cwd=tmp_path, capture_output=True, text=True)
    assert proc.returncode == 0, f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    assert proc.stdout.strip() == "1\tzero_coverage_seed_streak_with_positive_lineage\t2\t8"


def test_consume_controlled_recovery_attempt_decrements_budget_only_on_dispatch(tmp_path: Path) -> None:
    snapshot_fn = _extract_shell_function(_source(), "queue_recovery_policy_snapshot")
    consume_fn = _extract_shell_function(_source(), "consume_controlled_recovery_attempt")
    queue_path = tmp_path / "artifacts" / "wfa" / "aggregate" / "autonomous_seed_demo" / "run_queue.csv"
    queue_path.parent.mkdir(parents=True, exist_ok=True)
    queue_path.write_text(
        "run_name,config_path,status\n"
        "valid,configs/demo.yaml,planned\n",
        encoding="utf-8",
    )
    (queue_path.parent / "queue_policy.json").write_text(
        json.dumps(
            {
                "recovery_mode": "controlled",
                "recovery_reason": "zero_coverage_seed_streak",
                "recovery_lineage_anchor": "strict_rg",
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    yield_state_path = tmp_path / "yield_governor_state.json"
    yield_state_path.write_text(
        json.dumps(
            {
                "controlled_recovery_active": True,
                "controlled_recovery_reason": "zero_coverage_seed_streak_with_positive_lineage",
                "controlled_recovery_attempts_remaining": 2,
                "controlled_recovery_variants_cap": 8,
                "search_quality": {
                    "controlled_recovery_active": True,
                    "controlled_recovery_reason": "zero_coverage_seed_streak_with_positive_lineage",
                    "controlled_recovery_attempts_remaining": 2,
                    "controlled_recovery_variants_cap": 8,
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    script = f"""#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="{tmp_path}"
YIELD_GOVERNOR_STATE_FILE="{yield_state_path}"
{snapshot_fn}
{consume_fn}
fullspan_state_get() {{
  case "$2" in
    controlled_recovery_dispatch_consumed) echo "0" ;;
    *) echo "0" ;;
  esac
}}
fullspan_state_queue_set() {{
  printf 'QSET:%s|%s|%s\\n' "$1" "$2" "$3"
}}
fullspan_state_metric_set() {{
  printf 'METRIC:%s=%s\\n' "$1" "$2"
}}
log() {{
  printf 'LOG:%s\\n' "$*"
}}
log_decision_note() {{
  printf 'NOTE:%s|%s|%s|%s\\n' "$1" "$2" "$3" "$4"
}}
consume_controlled_recovery_attempt "artifacts/wfa/aggregate/autonomous_seed_demo/run_queue.csv"
"""
    proc = subprocess.run(["bash", "-lc", script], cwd=tmp_path, capture_output=True, text=True)
    assert proc.returncode == 0, f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    payload = json.loads(yield_state_path.read_text(encoding="utf-8"))
    assert payload["controlled_recovery_attempts_remaining"] == 1
    assert payload["search_quality"]["controlled_recovery_attempts_remaining"] == 1
    assert "QSET:artifacts/wfa/aggregate/autonomous_seed_demo/run_queue.csv|controlled_recovery_dispatch_consumed|1" in proc.stdout
    assert "METRIC:controlled_recovery_attempts_remaining=1" in proc.stdout
    assert "LOG:controlled_recovery_attempt_consumed queue=artifacts/wfa/aggregate/autonomous_seed_demo/run_queue.csv anchor=strict_rg attempts_before=2 attempts_after=1" in proc.stdout


def test_process_slo_remote_coverage_snapshot_prefers_queue_remote_reachable_with_fallback(tmp_path: Path) -> None:
    fn = _extract_shell_function(_source(), "process_slo_remote_coverage_snapshot")
    state_dir = tmp_path / "state"
    state_dir.mkdir(parents=True, exist_ok=True)
    state_path = state_dir / "process_slo_state.json"

    def _run_snapshot(payload: dict[str, object]) -> str:
        state_path.write_text(json.dumps(payload), encoding="utf-8")
        script = f"""#!/usr/bin/env bash
set -euo pipefail
PROCESS_SLO_STATE_FILE="{state_path}"
{fn}
process_slo_remote_coverage_snapshot
"""
        proc = subprocess.run(["bash", "-lc", script], cwd=tmp_path, capture_output=True, text=True)
        assert proc.returncode == 0, f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
        return proc.stdout.strip()

    assert _run_snapshot({"queue": {"remote_reachable": True}, "kpi": {"coverage_verified_ready_count": 2}}) == "1\t2"
    assert _run_snapshot({"remote_reachable": False, "coverage_verified_ready_count": 1}) == "0\t1"


def test_surrogate_gate_decision_bypasses_sidecar_in_simple_control_plane(tmp_path: Path) -> None:
    simple_fn = _extract_shell_function(_source(), "simple_control_plane_enabled")
    fn = _extract_shell_function(_source(), "surrogate_gate_decision")
    surrogate_path = tmp_path / "gate_surrogate_state.json"
    surrogate_path.write_text(json.dumps({"queues": {"demo": {"decision": "reject", "reason": "legacy"}}}), encoding="utf-8")
    script = f"""#!/usr/bin/env bash
set -euo pipefail
AUTONOMOUS_SIMPLE_CONTROL_PLANE=1
ROOT_DIR="{tmp_path}"
GATE_SURROGATE_STATE_FILE="{surrogate_path}"
{simple_fn}
{fn}
surrogate_gate_decision "artifacts/wfa/aggregate/demo/run_queue.csv"
"""
    proc = subprocess.run(["bash", "-lc", script], cwd=tmp_path, capture_output=True, text=True)
    assert proc.returncode == 0, f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    assert proc.stdout.strip() == "allow"


def test_yield_governor_hard_block_snapshot_is_open_in_simple_control_plane(tmp_path: Path) -> None:
    simple_fn = _extract_shell_function(_source(), "simple_control_plane_enabled")
    fn = _extract_shell_function(_source(), "yield_governor_hard_block_snapshot")
    yield_state = tmp_path / "yield_governor_state.json"
    yield_state.write_text(
        json.dumps({"hard_block_active": True, "hard_block_reason": "legacy_block", "hard_block_until_epoch": 9_999_999_999}),
        encoding="utf-8",
    )
    script = f"""#!/usr/bin/env bash
set -euo pipefail
AUTONOMOUS_SIMPLE_CONTROL_PLANE=1
YIELD_GOVERNOR_STATE_FILE="{yield_state}"
{simple_fn}
{fn}
yield_governor_hard_block_snapshot 1777000000
"""
    proc = subprocess.run(["bash", "-lc", script], cwd=tmp_path, capture_output=True, text=True)
    assert proc.returncode == 0, f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    assert proc.stdout.strip() == "0\t\t0\t0"


def test_simple_repairable_queue_count_detects_non_blocked_stalled_backlog(tmp_path: Path) -> None:
    fn = _extract_shell_function(_source(), "simple_repairable_queue_count")
    contract_src = (
        Path(__file__).resolve().parents[2] / "scripts" / "optimization" / "_queue_status_contract.py"
    ).read_text(encoding="utf-8")
    root = tmp_path / "app"
    queue_dir = root / "artifacts" / "wfa" / "aggregate" / "demo"
    queue_dir.mkdir(parents=True, exist_ok=True)
    (root / "configs").mkdir(parents=True, exist_ok=True)
    (root / "configs" / "demo.yaml").write_text("alpha: 1\n", encoding="utf-8")
    (root / "scripts" / "optimization").mkdir(parents=True, exist_ok=True)
    (root / "scripts" / "optimization" / "_queue_status_contract.py").write_text(contract_src, encoding="utf-8")
    (queue_dir / "run_queue.csv").write_text(
        "config_path,results_dir,status\n"
        "configs/demo.yaml,artifacts/wfa/runs/demo/run_01,stalled\n",
        encoding="utf-8",
    )
    state_dir = root / "artifacts" / "wfa" / "aggregate" / ".autonomous"
    state_dir.mkdir(parents=True, exist_ok=True)
    (state_dir / "fullspan_decision_state.json").write_text(json.dumps({"queues": {}}), encoding="utf-8")
    (state_dir / "orphan_queues.csv").write_text("queue,until_ts,reason\n", encoding="utf-8")
    script = f"""#!/usr/bin/env bash
set -euo pipefail
QUEUE_ROOT="{root / 'artifacts' / 'wfa' / 'aggregate'}"
ROOT_DIR="{root}"
FULLSPAN_DECISION_STATE_FILE="{state_dir / 'fullspan_decision_state.json'}"
ORPHAN_FILE="{state_dir / 'orphan_queues.csv'}"
{fn}
simple_repairable_queue_count
"""
    proc = subprocess.run(["bash", "-lc", script], cwd=root, capture_output=True, text=True)
    assert proc.returncode == 0, f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    assert proc.stdout.strip() == "1"


def test_selector_empty_candidate_pool_guard_signals_when_pool_empty_with_healthy_vps(tmp_path: Path) -> None:
    row_count_fn = _extract_shell_function(_source(), "csv_data_row_count")
    count_fn = _extract_shell_function(_source(), "candidate_pool_ready_count")
    status_fn = _extract_shell_function(_source(), "candidate_pool_status_snapshot")
    guard_fn = _extract_shell_function(_source(), "selector_empty_candidate_pool_guard")
    pool_path = tmp_path / "candidate_pool.csv"
    pool_path.write_text(
        "queue,planned,running,stalled,failed,completed,total,urgency,mtime,promotion_potential,gate_status,"
        "gate_reason,pre_rank_score,strict_gate_status,strict_gate_reason,effective_planned_count,"
        "stalled_share,queue_yield_score,recent_yield,dispatchable_pending,executable_pending\n",
        encoding="utf-8",
    )

    script = f"""#!/usr/bin/env bash
set -euo pipefail
READY_BUFFER_POOL_FILE="{pool_path}"
FULLSPAN_DECISION_STATE_FILE="{tmp_path / 'fullspan_state.json'}"
RUNTIME_OBSERVABILITY_STATE_FILE="{tmp_path / 'runtime_observability.json'}"
SELECTOR_EMPTY_POOL_GUARD_COOLDOWN_SEC=300
{row_count_fn}
{count_fn}
{status_fn}
{guard_fn}
global_backlog_snapshot() {{
  echo "1 3 0 3"
}}
process_slo_remote_coverage_snapshot() {{
  printf '1\\t0\\n'
}}
capacity_controller_remote_reachable_recent() {{
  echo "1"
}}
remote_runner_count() {{
  echo "0"
}}
fullspan_state_metric_get() {{
  case "$1" in
    selector_empty_candidate_pool_last_epoch) echo "0" ;;
    selector_empty_candidate_pool_last_reason) echo "" ;;
    vps_infra_fail_closed) echo "0" ;;
    infra_gate_status) echo "" ;;
    *) echo "0" ;;
  esac
}}
fullspan_state_metric_set() {{
  printf 'SET:%s=%s\\n' "$1" "${{2:-}}"
}}
fullspan_state_metric_inc() {{
  printf 'INC:%s=%s\\n' "$1" "${{2:-}}"
}}
runtime_observability_record_event() {{
  printf 'EVENT:%s|%s\\n' "${{1:-}}" "${{2:-}}"
}}
log() {{
  printf 'LOG:%s\\n' "$*"
}}
log_decision_note() {{
  printf 'NOTE:%s|%s|%s|%s\\n' "${{1:-}}" "${{2:-}}" "${{3:-}}" "${{4:-}}"
}}
if selector_empty_candidate_pool_guard candidate_parse_empty; then
  echo "RC:0"
else
  echo "RC:1"
fi
"""
    proc = subprocess.run(["bash", "-lc", script], cwd=tmp_path, capture_output=True, text=True)
    assert proc.returncode == 0, f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    assert "RC:0" in proc.stdout
    assert "SET:candidate_pool_status=empty_error" in proc.stdout
    assert "SET:selector_error=empty_candidate_pool_with_dispatchable_pending" in proc.stdout
    assert "INC:selector_empty_candidate_pool_count=1" in proc.stdout
    assert "EVENT:selector_empty_candidate_pool|" in proc.stdout
    assert "NOTE:global|SELECTOR_EMPTY_CANDIDATE_POOL|" in proc.stdout
    assert "LOG:selector_empty_candidate_pool reason=candidate_parse_empty dispatchable_pending=3 planned_dispatchable=1 executable_pending=3" in proc.stdout


def test_candidate_pool_status_snapshot_marks_empty_expected_when_dispatchable_is_zero(tmp_path: Path) -> None:
    row_count_fn = _extract_shell_function(_source(), "csv_data_row_count")
    count_fn = _extract_shell_function(_source(), "candidate_pool_ready_count")
    status_fn = _extract_shell_function(_source(), "candidate_pool_status_snapshot")
    pool_path = tmp_path / "candidate_pool.csv"
    pool_path.write_text(
        "queue,planned,running,stalled,failed,completed,total,urgency,mtime,promotion_potential,gate_status,"
        "gate_reason,pre_rank_score,strict_gate_status,strict_gate_reason,effective_planned_count,"
        "stalled_share,queue_yield_score,recent_yield,dispatchable_pending,executable_pending\n",
        encoding="utf-8",
    )

    script = f"""#!/usr/bin/env bash
set -euo pipefail
READY_BUFFER_POOL_FILE="{pool_path}"
{row_count_fn}
{count_fn}
{status_fn}
global_backlog_snapshot() {{
  echo "0 0 1 5"
}}
candidate_pool_status_snapshot
"""
    proc = subprocess.run(["bash", "-lc", script], cwd=tmp_path, capture_output=True, text=True)
    assert proc.returncode == 0, f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    assert proc.stdout.strip() == "empty_expected\t0\t0\t0\t1\t5"


def test_handle_empty_candidate_state_empty_expected_stops_batch_session(tmp_path: Path) -> None:
    row_count_fn = _extract_shell_function(_source(), "csv_data_row_count")
    count_fn = _extract_shell_function(_source(), "candidate_pool_ready_count")
    status_fn = _extract_shell_function(_source(), "candidate_pool_status_snapshot")
    handle_fn = _extract_shell_function(_source(), "handle_empty_candidate_state")
    pool_path = tmp_path / "candidate_pool.csv"
    pool_path.write_text(
        "queue,planned,running,stalled,failed,completed,total,urgency,mtime,promotion_potential,gate_status,"
        "gate_reason,pre_rank_score,strict_gate_status,strict_gate_reason,effective_planned_count,"
        "stalled_share,queue_yield_score,recent_yield,dispatchable_pending,executable_pending\n",
        encoding="utf-8",
    )

    script = f"""#!/usr/bin/env bash
set -euo pipefail
READY_BUFFER_POOL_FILE="{pool_path}"
adaptive_idle_sleep=30
{row_count_fn}
{count_fn}
{status_fn}
{handle_fn}
global_backlog_snapshot() {{
  echo "0 0 1 5"
}}
fullspan_state_metric_set() {{
  printf 'SET:%s=%s\\n' "$1" "${{2:-}}"
}}
log() {{
  printf 'LOG:%s\\n' "$*"
}}
log_decision_note() {{
  printf 'NOTE:%s|%s|%s|%s\\n' "${{1:-}}" "${{2:-}}" "${{3:-}}" "${{4:-}}"
}}
batch_session_maybe_stop() {{
  printf 'STOP:%s\\n' "$1"
}}
selector_empty_candidate_pool_guard() {{
  printf 'GUARD:%s\\n' "$1"
  return 1
}}
sleep() {{ :; }}
if handle_empty_candidate_state candidate_parse_empty; then
  echo "RC:0"
else
  echo "RC:1"
fi
printf 'ADAPTIVE:%s\\n' "$adaptive_idle_sleep"
"""
    proc = subprocess.run(["bash", "-lc", script], cwd=tmp_path, capture_output=True, text=True)
    assert proc.returncode == 0, f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    assert "RC:0" in proc.stdout
    assert "SET:candidate_pool_status=empty_expected" in proc.stdout
    assert "LOG:candidate_pool_empty_expected reason=candidate_parse_empty dispatchable_pending=0 executable_pending=5 planned_dispatchable=0 no_dispatchable_queues=1" in proc.stdout
    assert "NOTE:global|CANDIDATE_POOL_EMPTY_EXPECTED|" in proc.stdout
    assert "STOP:candidate_pool_empty_expected" in proc.stdout
    assert "GUARD:" not in proc.stdout
    assert "ADAPTIVE:60" in proc.stdout


def test_handle_empty_candidate_state_degrades_empty_error_when_only_cold_failed_backlog_remains(tmp_path: Path) -> None:
    row_count_fn = _extract_shell_function(_source(), "csv_data_row_count")
    count_fn = _extract_shell_function(_source(), "candidate_pool_ready_count")
    status_fn = _extract_shell_function(_source(), "candidate_pool_status_snapshot")
    handle_fn = _extract_shell_function(_source(), "handle_empty_candidate_state")
    pool_path = tmp_path / "candidate_pool.csv"
    pool_path.write_text(
        "queue,planned,running,stalled,failed,completed,total,urgency,mtime,promotion_potential,gate_status,"
        "gate_reason,pre_rank_score,strict_gate_status,strict_gate_reason,effective_planned_count,"
        "stalled_share,queue_yield_score,recent_yield,dispatchable_pending,executable_pending\n",
        encoding="utf-8",
    )

    script = f"""#!/usr/bin/env bash
set -euo pipefail
READY_BUFFER_POOL_FILE="{pool_path}"
adaptive_idle_sleep=30
{row_count_fn}
{count_fn}
{status_fn}
{handle_fn}
global_backlog_snapshot() {{
  echo "0 24 2 106"
}}
cold_fail_active_count() {{
  echo "3"
}}
maybe_trigger_auto_seed() {{
  printf 'SEED:%s\\n' "$1"
}}
fullspan_state_metric_set() {{
  printf 'SET:%s=%s\\n' "$1" "${{2:-}}"
}}
log() {{
  printf 'LOG:%s\\n' "$*"
}}
log_decision_note() {{
  printf 'NOTE:%s|%s|%s|%s\\n' "${{1:-}}" "${{2:-}}" "${{3:-}}" "${{4:-}}"
}}
batch_session_maybe_stop() {{
  printf 'STOP:%s\\n' "$1"
}}
selector_empty_candidate_pool_guard() {{
  printf 'GUARD:%s\\n' "$1"
  return 0
}}
sleep() {{ :; }}
if handle_empty_candidate_state candidate_empty; then
  echo "RC:0"
else
  echo "RC:1"
fi
printf 'ADAPTIVE:%s\\n' "$adaptive_idle_sleep"
"""
    proc = subprocess.run(["bash", "-lc", script], cwd=tmp_path, capture_output=True, text=True)
    assert proc.returncode == 0, f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    assert "RC:0" in proc.stdout
    assert "SET:candidate_pool_status=empty_error" in proc.stdout
    assert "SET:candidate_pool_status=empty_expected_degraded" in proc.stdout
    assert "LOG:candidate_pool_empty_degraded reason=candidate_empty dispatchable_pending=24 executable_pending=106 planned_dispatchable=0 no_dispatchable_queues=2 cold_fail_active_count=3" in proc.stdout
    assert "NOTE:global|CANDIDATE_POOL_EMPTY_DEGRADED|" in proc.stdout
    assert "SEED:candidate_empty" in proc.stdout
    assert "STOP:candidate_pool_empty_degraded" in proc.stdout
    assert "GUARD:" not in proc.stdout
    assert "ADAPTIVE:60" in proc.stdout


def test_handle_empty_candidate_state_simple_selector_guard_returns_fail_fast(tmp_path: Path) -> None:
    row_count_fn = _extract_shell_function(_source(), "csv_data_row_count")
    count_fn = _extract_shell_function(_source(), "candidate_pool_ready_count")
    status_fn = _extract_shell_function(_source(), "candidate_pool_status_snapshot")
    simple_fn = _extract_shell_function(_source(), "simple_control_plane_enabled")
    handle_fn = _extract_shell_function(_source(), "handle_empty_candidate_state")
    pool_path = tmp_path / "candidate_pool.csv"
    pool_path.write_text(
        "queue,planned,running,stalled,failed,completed,total,urgency,mtime,promotion_potential,gate_status,"
        "gate_reason,pre_rank_score,strict_gate_status,strict_gate_reason,effective_planned_count,"
        "stalled_share,queue_yield_score,recent_yield,dispatchable_pending,executable_pending\n",
        encoding="utf-8",
    )

    script = f"""#!/usr/bin/env bash
set -euo pipefail
AUTONOMOUS_SIMPLE_CONTROL_PLANE=1
READY_BUFFER_POOL_FILE="{pool_path}"
adaptive_idle_sleep=30
{row_count_fn}
{count_fn}
{status_fn}
{simple_fn}
{handle_fn}
global_backlog_snapshot() {{
  echo "1 3 0 3"
}}
fullspan_state_metric_set() {{
  printf 'SET:%s=%s\\n' "$1" "${{2:-}}"
}}
log() {{
  printf 'LOG:%s\\n' "$*"
}}
log_decision_note() {{
  printf 'NOTE:%s|%s|%s|%s\\n' "${{1:-}}" "${{2:-}}" "${{3:-}}" "${{4:-}}"
}}
batch_session_maybe_stop() {{
  printf 'STOP:%s\\n' "$1"
}}
selector_empty_candidate_pool_guard() {{
  printf 'GUARD:%s\\n' "$1"
  return 0
}}
sleep() {{ :; }}
handle_empty_candidate_state candidate_parse_empty || rc=$?
printf 'RC:%s\\n' "${{rc:-0}}"
printf 'ADAPTIVE:%s\\n' "$adaptive_idle_sleep"
"""
    proc = subprocess.run(["bash", "-lc", script], cwd=tmp_path, capture_output=True, text=True)
    assert proc.returncode == 0, f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    assert "GUARD:candidate_parse_empty" in proc.stdout
    assert "RC:2" in proc.stdout
    assert "STOP:" not in proc.stdout
    assert "ADAPTIVE:30" in proc.stdout


def test_batch_session_maybe_stop_uses_dispatchable_pending_for_auto_stop(tmp_path: Path) -> None:
    fn = _extract_shell_function(_source(), "batch_session_maybe_stop")
    script = f"""#!/usr/bin/env bash
set -euo pipefail
VPS_BATCH_SESSION_MAX_JOBS=3
VPS_BATCH_SESSION_IDLE_GRACE_SEC=60
VPS_BATCH_SESSION_MAX_SECONDS=3600
VPS_HOT_STANDBY_GRACE_SEC=120
batch_session_active=1
batch_session_start_epoch=$(( $(date +%s) - 300 ))
batch_session_last_dispatch_epoch=$(( $(date +%s) - 300 ))
batch_session_runs_started=1
completion_followup_pending=0
completion_followup_queue_rel=""
{fn}
batch_session_enabled() {{
  return 0
}}
remote_active_queue_jobs() {{
  echo "0"
}}
remote_cpu_busy_without_queue_job() {{
  echo "0"
}}
global_pending_count() {{
  echo "5"
}}
global_backlog_snapshot() {{
  echo "0 0 1 5"
}}
ready_buffer_depth() {{
  echo "0"
}}
confirm_fastlane_pending_count() {{
  echo "0"
}}
is_hot_standby_enabled() {{
  return 1
}}
batch_session_stop() {{
  printf 'STOP:%s\\n' "$1"
}}
batch_session_maybe_stop idle_probe
"""
    proc = subprocess.run(["bash", "-lc", script], cwd=tmp_path, capture_output=True, text=True)
    assert proc.returncode == 0, f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    assert "STOP:dispatchable_backlog_empty:idle_probe" in proc.stdout


def test_batch_session_maybe_stop_defers_when_started_queue_still_has_pending_work(tmp_path: Path) -> None:
    fn = _extract_shell_function(_source(), "batch_session_maybe_stop")
    queue_path = tmp_path / "artifacts" / "wfa" / "aggregate" / "demo" / "run_queue.csv"
    queue_path.parent.mkdir(parents=True)
    queue_path.write_text("run_name,config_path,status\nrun1,configs/a.yaml,running\n", encoding="utf-8")
    script = f"""#!/usr/bin/env bash
set -euo pipefail
VPS_BATCH_SESSION_MAX_JOBS=3
VPS_BATCH_SESSION_IDLE_GRACE_SEC=60
VPS_BATCH_SESSION_MAX_SECONDS=3600
VPS_HOT_STANDBY_GRACE_SEC=120
batch_session_active=1
batch_session_start_epoch=$(( $(date +%s) - 300 ))
batch_session_last_dispatch_epoch=$(( $(date +%s) - 300 ))
batch_session_runs_started=1
completion_followup_pending=1
completion_followup_queue_rel="artifacts/wfa/aggregate/demo/run_queue.csv"
{fn}
batch_session_enabled() {{
  return 0
}}
ROOT_DIR="{tmp_path}"
sync_queue_status() {{
  :
}}
queue_hygiene_snapshot() {{
  echo "1 0 1 0 0 0 0 0"
}}
log() {{
  printf 'LOG:%s\\n' "$*"
}}
batch_session_stop() {{
  printf 'STOP:%s\\n' "$1"
}}
batch_session_maybe_stop idle_probe
"""
    proc = subprocess.run(["bash", "-lc", script], cwd=tmp_path, capture_output=True, text=True)
    assert proc.returncode == 0, f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    assert "LOG:batch_session_stop_deferred reason=idle_probe completion_followup_queue=artifacts/wfa/aggregate/demo/run_queue.csv pending=1" in proc.stdout
    assert "STOP:" not in proc.stdout


def test_batch_session_maybe_stop_does_not_wait_for_local_completion_followup_worker(tmp_path: Path) -> None:
    fn = _extract_shell_function(_source(), "batch_session_maybe_stop")
    queue_path = tmp_path / "artifacts" / "wfa" / "aggregate" / "demo" / "run_queue.csv"
    queue_path.parent.mkdir(parents=True)
    queue_path.write_text("run_name,config_path,status\nrun1,configs/a.yaml,completed\n", encoding="utf-8")
    script = f"""#!/usr/bin/env bash
set -euo pipefail
VPS_BATCH_SESSION_MAX_JOBS=3
VPS_BATCH_SESSION_IDLE_GRACE_SEC=60
VPS_BATCH_SESSION_MAX_SECONDS=3600
VPS_HOT_STANDBY_GRACE_SEC=120
batch_session_active=1
batch_session_start_epoch=$(( $(date +%s) - 300 ))
batch_session_last_dispatch_epoch=$(( $(date +%s) - 300 ))
batch_session_runs_started=1
completion_followup_pending=1
completion_followup_queue_rel="artifacts/wfa/aggregate/demo/run_queue.csv"
{fn}
batch_session_enabled() {{
  return 0
}}
ROOT_DIR="{tmp_path}"
sync_queue_status() {{
  :
}}
queue_hygiene_snapshot() {{
  echo "0 0 0 0 0 0 0 8"
}}
log() {{
  printf 'LOG:%s\\n' "$*"
}}
remote_active_queue_jobs() {{
  echo "0"
}}
remote_cpu_busy_without_queue_job() {{
  echo "0"
}}
global_pending_count() {{
  echo "0"
}}
global_backlog_snapshot() {{
  echo "0 0 1 0"
}}
ready_buffer_depth() {{
  echo "0"
}}
confirm_fastlane_pending_count() {{
  echo "0"
}}
is_hot_standby_enabled() {{
  return 1
}}
batch_session_stop() {{
  printf 'STOP:%s\\n' "$1"
}}
batch_session_maybe_stop idle_probe
"""
    proc = subprocess.run(["bash", "-lc", script], cwd=tmp_path, capture_output=True, text=True)
    assert proc.returncode == 0, f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    assert "LOG:batch_session_stop_deferred" not in proc.stdout
    assert "STOP:dispatchable_backlog_empty:idle_probe" in proc.stdout


def test_driver_runtime_contract_includes_runtime_state_file_and_reconcile_hook() -> None:
    src = _source()
    _assert_contains_all(
        src,
        [
            'DRIVER_RUNTIME_STATE_FILE',
            'driver_runtime_guard()',
            'write_driver_runtime_state "active"',
            'driver_runtime_reconcile_derived_state()',
            'DRIVER_RUNTIME_STALE',
            'reason=driver_runtime_stale',
        ],
    )


def test_driver_selector_contract_uses_cold_fail_state_in_simple_mode() -> None:
    src = _source()
    assert 'cold_fail_state = load_cold_fail_state(cold_fail_state_path)' in src
    assert 'cold_fail_state = {} if simple_control_plane else load_cold_fail_state(cold_fail_state_path)' not in src


def test_simple_control_plane_uses_direct_stalled_dispatch_contract() -> None:
    src = _source()
    _assert_contains_all(
        src,
        [
            'refresh_control_plane_guard_state || true',
            'process_slo_guard_agent.py',
            'repair_stalled_bypass queue=$queue_rel reason=simple_control_plane_direct_dispatch',
            'local run_statuses="auto"',
            'run_statuses="stalled"',
            '--statuses "$run_statuses"',
            'DRIVER_SIMPLE_POST_START_SLEEP_SEC',
        ],
    )


def test_completion_followup_contract_tracks_started_queue_and_short_poll_path() -> None:
    src = _source()
    _assert_contains_all(
        src,
        [
            "completion_followup_track()",
            "maybe_process_completion_followup()",
            "completion_followup_queue_enqueue()",
            "completion_followup_worker_main()",
            "maybe_kick_completion_followup_worker()",
            'COMPLETION_FOLLOWUP_QUEUE_FILE=',
            'COMPLETION_FOLLOWUP_WORKER_STATE_FILE=',
            'completion_followup_track "$queue_rel"',
            'maybe_process_completion_followup "candidate_empty_after_reconcile" || true',
            'maybe_kick_completion_followup_worker "candidate_empty_after_reconcile" || true',
            'completion_followup_enqueue_ready_queue "$queue_rel" "progress_milestone"',
            'completion_followup_enqueue_ready_queue "$queue_rel" "confirm_fastlane_watch"',
            'fullspan_rollup_sync_deferred queue=$queue_rel reason=remote_runner_active_sync',
            'completion_followup_worker_started trigger=$trigger_reason',
            'completion_followup_enqueued queue=$queue_rel',
            'if [[ "${completion_followup_pending:-0}" == "1" && "${completion_followup_queue_rel:-}" == "$queue_rel" ]]; then',
            "sleep 1",
        ],
    )


def test_watchdog_is_liveness_only_contract() -> None:
    watchdog_path = Path(__file__).resolve().parents[2] / "scripts" / "optimization" / "autonomous_wfa_watchdog.sh"
    source = watchdog_path.read_text(encoding="utf-8")
    assert "WATCHDOG_LIVENESS_ONLY queue=$queue_rel action=skip_powered_repair" in source
    assert "ALLOW_HEAVY_RUN=1" not in source
    assert "WATCHDOG_TRIGGER_REPAIR queue=" not in source


def test_candidate_file_has_rows_treats_header_only_csv_as_empty(tmp_path: Path) -> None:
    count_fn = _extract_shell_function(_source(), "csv_data_row_count")
    has_rows_fn = _extract_shell_function(_source(), "candidate_file_has_rows")
    candidate_path = tmp_path / "candidate.csv"
    candidate_path.write_text("queue,planned,running\n", encoding="utf-8")

    script = f"""#!/usr/bin/env bash
set -euo pipefail
CANDIDATE_FILE="{candidate_path}"
{count_fn}
{has_rows_fn}
if candidate_file_has_rows; then
  echo "HEADER_ONLY:rows"
else
  echo "HEADER_ONLY:empty"
fi
cat > "$CANDIDATE_FILE" <<'CSV'
queue,planned,running
artifacts/wfa/aggregate/demo/run_queue.csv,1,0
CSV
if candidate_file_has_rows; then
  echo "WITH_DATA:rows"
else
  echo "WITH_DATA:empty"
fi
"""
    proc = subprocess.run(["bash", "-lc", script], cwd=tmp_path, capture_output=True, text=True)
    assert proc.returncode == 0, f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    assert "HEADER_ONLY:empty" in proc.stdout
    assert "WITH_DATA:rows" in proc.stdout


def test_driver_runtime_guard_detects_script_hash_drift(tmp_path: Path) -> None:
    sha_fn = _extract_shell_function(_source(), "driver_script_sha256")
    guard_fn = _extract_shell_function(_source(), "driver_runtime_guard")
    script_path = tmp_path / "autonomous_wfa_driver.sh"
    script_path.write_text("#!/usr/bin/env bash\necho first\n", encoding="utf-8")
    runtime_state_path = tmp_path / "driver_runtime_state.json"
    runtime_sha = subprocess.run(
        ["bash", "-lc", f"python3 - <<'PY'\nimport hashlib\nfrom pathlib import Path\nprint(hashlib.sha256(Path({json.dumps(str(script_path))}).read_bytes()).hexdigest())\nPY"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    script_path.write_text("#!/usr/bin/env bash\necho second\n", encoding="utf-8")

    script = f"""#!/usr/bin/env bash
set -euo pipefail
DRIVER_RUNTIME_STATE_FILE="{runtime_state_path}"
DRIVER_SCRIPT_PATH="{script_path}"
DRIVER_SCRIPT_SHA256="{runtime_sha}"
DRIVER_STARTED_AT="2026-03-06T12:00:00Z"
STATE_FILE="{tmp_path / 'driver_state.txt'}"
LOG_FILE="{tmp_path / 'driver.log'}"
{sha_fn}
{guard_fn}
write_driver_runtime_state() {{
  printf '%s\n' '{{"status": "stale"}}' > "$DRIVER_RUNTIME_STATE_FILE"
}}
fullspan_state_metric_set() {{
  printf 'SET:%s=%s\\n' "$1" "${{2:-}}"
}}
log() {{
  printf 'LOG:%s\\n' "$*"
}}
log_decision_note() {{
  printf 'NOTE:%s|%s|%s|%s\\n' "${{1:-}}" "${{2:-}}" "${{3:-}}" "${{4:-}}"
}}
log_state() {{
  printf 'STATE:%s\\n' "$*"
}}
if driver_runtime_guard; then
  echo "RC:0"
else
  echo "RC:1"
fi
"""
    proc = subprocess.run(["bash", "-lc", script], cwd=tmp_path, capture_output=True, text=True)
    assert proc.returncode == 0, f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    runtime_payload = json.loads(runtime_state_path.read_text(encoding="utf-8"))
    assert "RC:1" in proc.stdout
    assert "SET:driver_runtime_stale=1" in proc.stdout
    assert "LOG:DRIVER_RUNTIME_STALE pid=" in proc.stdout
    assert "NOTE:global|DRIVER_RUNTIME_STALE|" in proc.stdout
    assert "STATE:idle now=none reason=driver_runtime_stale" in proc.stdout
    assert runtime_payload["status"] == "stale"


def test_refresh_control_plane_guard_state_keeps_process_slo_current_in_simple_mode(tmp_path: Path) -> None:
    simple_fn = _extract_shell_function(_source(), "simple_control_plane_enabled")
    fn = _extract_shell_function(_source(), "refresh_control_plane_guard_state")
    calls_path = tmp_path / "calls.log"
    spy_path = tmp_path / "spy-python.sh"
    spy_path.write_text(
        "#!/usr/bin/env bash\n"
        "printf '%s\\n' \"$*\" >> \"$CALLS_PATH\"\n",
        encoding="utf-8",
    )
    spy_path.chmod(0o755)

    script = f"""#!/usr/bin/env bash
set -euo pipefail
AUTONOMOUS_SIMPLE_CONTROL_PLANE=1
ROOT_DIR="{tmp_path}"
CALLS_PATH="{calls_path}"
export CALLS_PATH
{simple_fn}
{fn}
control_plane_python_bin() {{
  echo "{spy_path}"
}}
if refresh_control_plane_guard_state; then
  echo "RC:0"
else
  echo "RC:$?"
fi
cat "{calls_path}"
"""
    proc = subprocess.run(["bash", "-lc", script], cwd=tmp_path, capture_output=True, text=True)
    assert proc.returncode == 0, f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    assert "RC:0" in proc.stdout
    assert "scripts/optimization/process_slo_guard_agent.py --root" in proc.stdout
    assert "vps_capacity_controller_agent.py" not in proc.stdout


def test_repair_stalled_queue_records_running_state_for_serial_repair_dispatch(tmp_path: Path) -> None:
    fn = _extract_shell_function(_source(), "repair_stalled_queue")
    queue_path = tmp_path / "artifacts" / "wfa" / "aggregate" / "demo" / "run_queue.csv"
    queue_path.parent.mkdir(parents=True, exist_ok=True)
    queue_path.write_text(
        "config_path,results_dir,status\n"
        "configs/demo.yaml,artifacts/wfa/runs/demo/run_01,stalled\n",
        encoding="utf-8",
    )
    recover_script = tmp_path / "scripts" / "optimization" / "recover_stalled_queue.sh"
    recover_script.parent.mkdir(parents=True, exist_ok=True)
    recover_script.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
    recover_script.chmod(0o755)

    script = f"""#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="{tmp_path}"
LOG_FILE="{tmp_path / 'driver.log'}"
FULLSPAN_POLICY_NAME="fullspan_v1"
PROMOTION_SELECTION_MODE="fullspan"
SERVER_IP="85.198.90.128"
SERVER_USER="root"
DETERMINISTIC_QUARANTINE_FILE="{tmp_path / 'deterministic_quarantine.json'}"
SAME_REASON_REPAIR_CAP=3
hb_stale_sec=901
{fn}
choose_parallel() {{
  echo "4"
}}
repair_reason_streak_record() {{
  echo "1"
}}
fullspan_state_get() {{
  echo "0"
}}
fullspan_state_set() {{
  :
}}
fullspan_state_metric_inc() {{
  :
}}
mark_orphan() {{
  printf 'ORPHAN:%s\\n' "$*"
}}
log_decision_note() {{
  printf 'NOTE:%s\\n' "$*"
}}
choose_max_retries() {{
  echo "2"
}}
record_dispatch_attempt() {{
  printf 'DISPATCH:%s\\n' "$*"
}}
update_dispatch_attempt_result() {{
  printf 'RESULT:%s\\n' "$*"
}}
set_infra_gate_state() {{
  printf 'INFRA:%s|%s|%s|%s|%s\\n' "${{1:-}}" "${{2:-}}" "${{3:-}}" "${{4:-}}" "${{5:-}}"
}}
log_state() {{
  printf 'STATE:%s\\n' "$*"
}}
refresh_control_plane_guard_state() {{
  printf 'GUARD_REFRESH\\n'
}}
ensure_vps_ready() {{
  printf 'VPS:%s\\n' "$1"
  return 0
}}
clear_orphan() {{
  printf 'CLEAR:%s\\n' "$1"
}}
log() {{
  printf 'LOG:%s\\n' "$*"
}}
if repair_stalled_queue "artifacts/wfa/aggregate/demo/run_queue.csv" 0 0 1 1 stalled_queue_no_progress; then
  echo "RC:0"
else
  echo "RC:$?"
fi
"""
    proc = subprocess.run(["bash", "-lc", script], cwd=tmp_path, capture_output=True, text=True)
    assert proc.returncode == 0, f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    assert "DISPATCH:artifacts/wfa/aggregate/demo/run_queue.csv repair_preflight" in proc.stdout
    assert "INFRA:starting|repair_startup_confirmation_pending||0|" in proc.stdout
    assert "STATE:starting queue=artifacts/wfa/aggregate/demo/run_queue.csv reason=stalled_queue_repair" in proc.stdout
    assert "RESULT:artifacts/wfa/aggregate/demo/run_queue.csv started" in proc.stdout
    assert "CLEAR:artifacts/wfa/aggregate/demo/run_queue.csv" in proc.stdout
    assert "STATE:running queue=artifacts/wfa/aggregate/demo/run_queue.csv reason=stalled_queue_repair" in proc.stdout
    assert "RC:0" in proc.stdout


def test_start_queue_remote_handoff_failure_stays_queue_scoped_contract() -> None:
    src = _source()
    match = re.search(
        r'if \[\[ "\$\{startup_code:-\}" == "REMOTE_HANDOFF_FAILED" \]\]; then(?P<body>.*?)\n\s*return 1\n\s*fi',
        src,
        flags=re.DOTALL,
    )
    assert match is not None, "REMOTE_HANDOFF_FAILED branch not found in start_queue"
    body = match.group("body")
    assert 'fullspan_state_metric_set "vps_infra_fail_closed" 0' in body
    assert 'fullspan_state_metric_set "auto_seed_hard_block" 0' in body
    assert 'fullspan_state_metric_set "auto_seed_block_reason" ""' in body
    assert '"dispatch_attempt_result" "remote_handoff_failed"' in body
    assert '"dispatch_attempt_session_epoch" "0"' in body
    assert 'set_infra_gate_state "ok" "" "" "0" ""' in body
    assert 'reason=queue_start_softfail' in body
    assert 'mark_orphan' not in body
    assert 'record_infra_fail_closed' not in body


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
    _assert_contains_any(
        src,
        [
            'trigger_confirm_fastlane "$queue_rel"',
            'CONFIRM_FASTLANE_TRIGGER',
        ],
        label="confirm fastlane hook",
    )


def test_ensure_vps_ready_propagates_active_ssh_force_cycle_contract() -> None:
    src = _source()
    assert 'VPS_RECOVER_TIMEOUT_SEC="${VPS_RECOVER_TIMEOUT_SEC:-600}"' in src
    fn = _extract_shell_function(src, "ensure_vps_ready")
    assert 'SSH_READY_TIMEOUT_SEC="$timeout_sec"' in fn
    assert 'SSH_ACTIVE_FORCE_CYCLE_AFTER_SEC="$VPS_ACTIVE_SSH_FORCE_CYCLE_AFTER_SEC"' in fn
    assert 'SSH_ACTIVE_FORCE_CYCLE_MAX_ATTEMPTS="$VPS_ACTIVE_SSH_FORCE_CYCLE_MAX_ATTEMPTS"' in fn
    assert 'SSH_FORCE_CYCLE_SHUTDOWN_WAIT_SEC="$VPS_FORCE_CYCLE_SHUTDOWN_WAIT_SEC"' in fn


def test_queue_start_confirmation_status_surfaces_remote_handoff_failure_reason(tmp_path: Path) -> None:
    fn = _extract_shell_function(_source(), "queue_start_confirmation_status")
    queue_path = tmp_path / "queue.csv"
    queue_path.write_text(
        "config_path,results_dir,status\n"
        "configs/demo.yaml,artifacts/wfa/runs/demo/run_01,stalled\n",
        encoding="utf-8",
    )
    qlog = tmp_path / "startup.log"
    qlog.write_text(
        "powered: remote_handoff outcome=REMOTE_HANDOFF_FAILED reason=no_remote_process_or_queue_activity\n"
        "powered: FAIL reason=REMOTE_HANDOFF_FAILED\n",
        encoding="utf-8",
    )

    script = f"""#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="{tmp_path}"
START_QUEUE_SSH_READY_TIMEOUT_SEC=60
START_QUEUE_SYNC_TIMEOUT_SEC=60
START_QUEUE_STARTUP_BUDGET_SEC=300
{fn}
queue_start_confirmation_status "queue.csv" "{qlog}" "$(date +%s)" "$START_QUEUE_SSH_READY_TIMEOUT_SEC" "$START_QUEUE_SYNC_TIMEOUT_SEC" "$START_QUEUE_STARTUP_BUDGET_SEC"
"""
    proc = subprocess.run(["bash", "-lc", script], cwd=tmp_path, capture_output=True, text=True)
    assert proc.returncode == 0, f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    assert proc.stdout.strip() == "fail\tREMOTE_HANDOFF_FAILED\tno_remote_process_or_queue_activity"


def test_handoff_retry_preflight_escalates_then_quarantines_and_clears(tmp_path: Path) -> None:
    src = _source()
    snapshot_fn = _extract_shell_function_between(src, "handoff_retry_queue_snapshot", "handoff_retry_cleanup_state")
    clear_fn = _extract_shell_function_between(src, "handoff_retry_clear_queue", "handoff_retry_record_failure")
    record_fn = _extract_shell_function_between(src, "handoff_retry_record_failure", "handoff_retry_preflight_action")
    preflight_fn = _extract_shell_function_between(src, "handoff_retry_preflight_action", "handoff_retry_mark_quarantine")
    state_path = tmp_path / "handoff_retry_state.json"
    queue_rel = "artifacts/wfa/aggregate/demo/run_queue.csv"

    script = f"""#!/usr/bin/env bash
set -euo pipefail
HANDOFF_RETRY_STATE_FILE="{state_path}"
HANDOFF_RETRY_WINDOW_SEC=1800
HANDOFF_RETRY_CAP=3
HANDOFF_SYNC_ESCALATION_ATTEMPT=2
{snapshot_fn}
{record_fn}
{preflight_fn}
{clear_fn}
printf 'P0:%s\\n' "$(handoff_retry_preflight_action "{queue_rel}")"
printf 'R1:%s\\n' "$(handoff_retry_record_failure "{queue_rel}" REMOTE_HANDOFF_FAILED no_remote_process_or_queue_activity runtime-first)"
printf 'P1:%s\\n' "$(handoff_retry_preflight_action "{queue_rel}")"
printf 'R2:%s\\n' "$(handoff_retry_record_failure "{queue_rel}" REMOTE_HANDOFF_FAILED no_remote_process_or_queue_activity strict)"
printf 'P2:%s\\n' "$(handoff_retry_preflight_action "{queue_rel}")"
printf 'CLEAR:%s\\n' "$(handoff_retry_clear_queue "{queue_rel}")"
printf 'P3:%s\\n' "$(handoff_retry_preflight_action "{queue_rel}")"
"""
    proc = subprocess.run(["bash", "-lc", script], cwd=tmp_path, capture_output=True, text=True)
    assert proc.returncode == 0, f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    assert "P0:dispatch\truntime-first\t0\t0" in proc.stdout
    assert "R1:1\t0\t0" in proc.stdout
    assert "P1:dispatch\tstrict\t1\t1\tREMOTE_HANDOFF_FAILED\tno_remote_process_or_queue_activity" in proc.stdout
    assert "R2:2\t1\t0" in proc.stdout
    assert "P2:quarantine\thold\t2\t1\tREMOTE_HANDOFF_FAILED\tno_remote_process_or_queue_activity" in proc.stdout
    assert "CLEAR:1" in proc.stdout
    assert "P3:dispatch\truntime-first\t0\t0" in proc.stdout
    assert not state_path.exists()


def test_handoff_retry_quarantine_adds_long_cold_fail(tmp_path: Path) -> None:
    fn = _extract_shell_function(_source(), "handoff_retry_mark_quarantine")
    state_path = tmp_path / "handoff_retry_state.json"
    queue_rel = "artifacts/wfa/aggregate/demo/run_queue.csv"

    script = f"""#!/usr/bin/env bash
set -euo pipefail
HANDOFF_RETRY_STATE_FILE="{state_path}"
HANDOFF_QUARANTINE_SEC=1800
{fn}
mark_orphan_with_ttl() {{
  :
}}
fullspan_state_queue_set() {{
  :
}}
log_decision_note() {{
  :
}}
log() {{
  :
}}
cold_fail_state_add() {{
  printf 'COLD:%s|%s\\n' "$1" "$2"
}}
handoff_retry_mark_quarantine "{queue_rel}" REMOTE_HANDOFF_FAILED no_remote_process_or_queue_activity
"""
    proc = subprocess.run(["bash", "-lc", script], cwd=tmp_path, capture_output=True, text=True)
    assert proc.returncode == 0, f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    assert "COLD:artifacts/wfa/aggregate/demo/run_queue.csv|handoff_retry_quarantine_remote_handoff_failed" in proc.stdout


def test_handoff_retry_expired_quarantine_resets_on_snapshot_and_cleanup(tmp_path: Path) -> None:
    snapshot_fn = _extract_shell_function(_source(), "handoff_retry_queue_snapshot")
    preflight_fn = _extract_shell_function(_source(), "handoff_retry_preflight_action")
    cleanup_fn = _extract_shell_function(_source(), "handoff_retry_cleanup_state")
    state_path = tmp_path / "handoff_retry_state.json"
    queue_rel = "artifacts/wfa/aggregate/demo/run_queue.csv"
    now_epoch = int(time.time())
    state_path.write_text(
        json.dumps(
            {
                "entries": {
                    queue_rel: {
                        "fail_count": 2,
                        "last_failure_code": "REMOTE_HANDOFF_FAILED",
                        "last_failure_reason": "no_remote_process_or_queue_activity",
                        "last_attempt_epoch": now_epoch - 30,
                        "sync_escalated": True,
                        "cooldown_until_epoch": now_epoch - 5,
                    }
                },
                "updated_at": now_epoch - 30,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    script = f"""#!/usr/bin/env bash
set -euo pipefail
HANDOFF_RETRY_STATE_FILE="{state_path}"
HANDOFF_RETRY_WINDOW_SEC=1800
HANDOFF_RETRY_CAP=3
HANDOFF_SYNC_ESCALATION_ATTEMPT=2
{snapshot_fn}
{preflight_fn}
{cleanup_fn}
printf 'P:%s\\n' "$(handoff_retry_preflight_action "{queue_rel}")"
printf 'C:%s\\n' "$(handoff_retry_cleanup_state)"
"""
    proc = subprocess.run(["bash", "-lc", script], cwd=tmp_path, capture_output=True, text=True)
    assert proc.returncode == 0, f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    assert "P:dispatch\truntime-first\t0\t0" in proc.stdout
    assert "C:1" in proc.stdout
    assert not state_path.exists()


def test_handoff_retry_clear_contract_on_progress_and_idle() -> None:
    src = _source()
    assert src.count('handoff_retry_clear_queue "$queue_rel" >/dev/null 2>&1 || true') >= 3
    _assert_contains_all(
        src,
        [
            'clear_orphan "$queue_rel"',
            'log "progress_seen queue=$queue_rel prev=$prev_pending curr=$pending"',
            'if [[ "$reason" == "no_pending" ]]; then',
            'log "no_pending queue=$queue_rel action=WAIT',
        ],
    )


def test_confirm_fastlane_contract_runs_even_in_simple_control_plane() -> None:
    src = _source()
    dispatch_fn = _extract_shell_function(src, "dispatch_replay_fastlane_hooks")
    assert "simple_control_plane_enabled" not in dispatch_fn
    _assert_contains_all(
        src,
        [
            'dispatch_replay_fastlane_hooks || true',
            'if [[ "$DRIVER_CONFIRM_FASTLANE_ENABLE" == "1" || "$DRIVER_CONFIRM_FASTLANE_ENABLE" == "true" ]]; then',
            'trigger_confirm_fastlane "$queue_rel"',
        ],
    )
    assert 'dispatch_replay_fastlane_hooks || true\n  if ! simple_control_plane_enabled; then' in src
    assert 'if ! simple_control_plane_enabled && [[ "$DRIVER_CONFIRM_FASTLANE_ENABLE"' not in src


def test_winner_hold_target_returns_cutover_ready_promote_eligible_queue(tmp_path: Path) -> None:
    fn = _extract_shell_function(_source(), "winner_hold_target")
    state_path = tmp_path / "fullspan_decision_state.json"
    state_path.write_text(
        json.dumps(
            {
                "queues": {
                    "artifacts/wfa/aggregate/demo_b/run_queue.csv": {
                        "promotion_verdict": "PROMOTE_ELIGIBLE",
                        "cutover_permission": "ALLOW_PROMOTE",
                        "cutover_ready": False,
                        "top_run_group": "rg_b",
                        "top_variant": "variant_b",
                    },
                    "artifacts/wfa/aggregate/demo_a/run_queue.csv": {
                        "promotion_verdict": "PROMOTE_ELIGIBLE",
                        "cutover_permission": "ALLOW_PROMOTE",
                        "cutover_ready": True,
                        "top_run_group": "rg_a",
                        "top_variant": "variant_a",
                    },
                }
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    script = f"""#!/usr/bin/env bash
set -euo pipefail
FULLSPAN_DECISION_STATE_FILE="{state_path}"
{fn}
winner_hold_target
"""
    proc = subprocess.run(["bash", "-lc", script], cwd=tmp_path, capture_output=True, text=True)
    assert proc.returncode == 0, f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    assert proc.stdout.strip() == "artifacts/wfa/aggregate/demo_a/run_queue.csv\trg_a\tvariant_a"


def test_winner_hold_loop_contract() -> None:
    src = _source()
    _assert_contains_all(
        src,
        [
            'WINNER_HOLD_POLL_SEC',
            'winner_hold_target()',
            'winner_hold_active',
            'winner_hold_queue',
            'winner_hold_top_run_group',
            'winner_hold_top_variant',
            'log "winner_hold queue=$winner_queue_rel top_run_group=$winner_top_run_group top_variant=$winner_top_variant"',
            'log_decision_note "$winner_queue_rel" "WINNER_HOLD" "top_run_group=$winner_top_run_group top_variant=$winner_top_variant" "stop_new_dispatch_and_hold_winner"',
            'batch_session_maybe_stop "winner_hold"',
            'log_state "winner_hold queue=$winner_queue_rel top_run_group=$winner_top_run_group top_variant=$winner_top_variant"',
            'sleep "$WINNER_HOLD_POLL_SEC"',
        ],
    )
