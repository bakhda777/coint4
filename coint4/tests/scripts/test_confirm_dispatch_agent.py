from __future__ import annotations

import csv
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "optimization" / "confirm_dispatch_agent.sh"


def _extract_embedded_python() -> str:
    source = SCRIPT_PATH.read_text(encoding="utf-8")
    match = re.search(r"<<'PY'\n(?P<body>.*?)\nPY", source, flags=re.DOTALL)
    assert match is not None, "embedded python not found in confirm_dispatch_agent.sh"
    return match.group("body")


def test_confirm_dispatch_agent_defers_to_driver_contract() -> None:
    source = SCRIPT_PATH.read_text(encoding="utf-8")
    assert 'CONFIRM_DISPATCH_ALLOW_WITH_DRIVER="${CONFIRM_DISPATCH_ALLOW_WITH_DRIVER:-0}"' in source
    assert "driver_loop_active()" in source
    assert "autonomous-wfa-driver.service" in source
    assert "confirm_dispatch_agent_deferred reason=driver_active" in source


def _run_embedded_python(code: str, argv: list[str], cwd: Path, env: dict[str, str]) -> None:
    proc = subprocess.run([sys.executable, "-c", code, *argv], cwd=cwd, capture_output=True, text=True, env=env)
    assert proc.returncode == 0, f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"


def _write_queue(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["config_path", "results_dir", "status"])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_repo_stubs(root: Path) -> None:
    python_shim = root / ".venv" / "bin" / "python"
    python_shim.parent.mkdir(parents=True, exist_ok=True)
    python_shim.write_text(f"#!/usr/bin/env bash\nexec {sys.executable} \"$@\"\n", encoding="utf-8")
    python_shim.chmod(0o755)

    scripts_dir = root / "scripts" / "optimization"
    scripts_dir.mkdir(parents=True, exist_ok=True)

    (scripts_dir / "build_confirm_queue.py").write_text(
        """#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--shortlist-queue", required=True)
parser.add_argument("--queue-dir", required=True)
args, _ = parser.parse_known_args()

shortlist = Path(args.shortlist_queue)
rows = []
if shortlist.exists():
    with shortlist.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

queue_dir = Path(args.queue_dir)
queue_dir.mkdir(parents=True, exist_ok=True)
with (queue_dir / "run_queue.csv").open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=["config_path", "results_dir", "status"])
    writer.writeheader()
    if rows:
        writer.writerow(
            {
                "config_path": str(rows[0].get("config_path") or ""),
                "results_dir": str(rows[0].get("results_dir") or ""),
                "status": "planned",
            }
        )
""",
        encoding="utf-8",
    )

    (scripts_dir / "run_wfa_queue_powered.py").write_text(
        """#!/usr/bin/env python3
import os
import sys
from pathlib import Path

marker = os.environ.get("DISPATCH_MARKER", "")
if marker:
    m = Path(marker)
    m.parent.mkdir(parents=True, exist_ok=True)
    with m.open("a", encoding="utf-8") as handle:
        handle.write("dispatch " + " ".join(sys.argv[1:]) + "\\n")
""",
        encoding="utf-8",
    )

    (scripts_dir / "fullspan_lineage.py").write_text(
        "#!/usr/bin/env python3\n",
        encoding="utf-8",
    )


def _write_ssh_stub(bin_dir: Path) -> None:
    bin_dir.mkdir(parents=True, exist_ok=True)
    ssh_stub = bin_dir / "ssh"
    ssh_stub.write_text(
        "#!/usr/bin/env bash\n"
        "echo 0\n",
        encoding="utf-8",
    )
    ssh_stub.chmod(0o755)


def test_failed_first_candidate_does_not_consume_cycle_quota(tmp_path: Path) -> None:
    root = tmp_path / "app"
    state_dir = root / "artifacts" / "wfa" / "aggregate" / ".autonomous"
    state_dir.mkdir(parents=True, exist_ok=True)
    _write_repo_stubs(root)
    _write_ssh_stub(tmp_path / "bin")

    queue_first_rel = "artifacts/wfa/aggregate/group_first/run_queue.csv"
    queue_second_rel = "artifacts/wfa/aggregate/group_second/run_queue.csv"

    _write_queue(
        root / queue_first_rel,
        [
            {
                "config_path": "configs/alpha_stress.yaml",
                "results_dir": "artifacts/wfa/runs/group_first/stress_run_01",
                "status": "planned",
            }
        ],
    )
    _write_queue(
        root / queue_second_rel,
        [
            {
                "config_path": "configs/beta_main.yaml",
                "results_dir": "artifacts/wfa/runs/group_second/run_01",
                "status": "planned",
            }
        ],
    )

    state_path = state_dir / "fullspan_decision_state.json"
    state_payload = {
        "queues": {
            queue_first_rel: {
                "strict_pass_count": 1,
                "strict_run_group_count": 2,
                "confirm_count": 0,
                "promotion_verdict": "PROMOTE_PENDING_CONFIRM",
                "top_run_group": "group_first",
                "top_variant": "alpha",
                "top_score": "10.0",
                "last_update": "2026-03-04T00:00:00Z",
            },
            queue_second_rel: {
                "strict_pass_count": 1,
                "strict_run_group_count": 2,
                "confirm_count": 0,
                "promotion_verdict": "PROMOTE_PENDING_CONFIRM",
                "top_run_group": "group_second",
                "top_variant": "beta",
                "top_score": "11.0",
                "last_update": "2026-03-04T00:00:01Z",
            },
        }
    }
    state_path.write_text(json.dumps(state_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    notes_path = state_dir / "decision_notes.jsonl"
    log_path = state_dir / "confirm_dispatch_agent.log"
    registry_path = state_dir / "confirm_lineage_registry.json"
    capacity_state_path = state_dir / "capacity_controller_state.json"
    dispatch_marker = tmp_path / "dispatch.marker"

    code = _extract_embedded_python()
    argv = [
        str(root),
        str(state_path),
        str(notes_path),
        str(state_dir),
        str(log_path),
        str(registry_path),
        "2",
        "2",
        "1",
        "1",
        "1",
        "1",
        "0",
        "85.198.90.128",
        "root",
        "false",
        "1",
        "10",
        "10",
        str(capacity_state_path),
        "0",
        "1",
    ]
    env = os.environ.copy()
    env["PATH"] = f"{tmp_path / 'bin'}:{env.get('PATH', '')}"
    env["DISPATCH_MARKER"] = str(dispatch_marker)

    _run_embedded_python(code, argv, cwd=root, env=env)

    deadline = time.time() + 2.0
    while time.time() < deadline and not dispatch_marker.exists():
        time.sleep(0.05)

    updated_state = json.loads(state_path.read_text(encoding="utf-8"))
    metrics = updated_state["runtime_metrics"]
    assert int(metrics["confirm_fastlane_trigger_empty_shortlist"]) == 1
    assert int(metrics["confirm_fastlane_trigger_attempt"]) == 2
    assert int(metrics["confirm_fastlane_trigger_count"]) == 1
    assert int(metrics["confirm_fastlane_cycle_dispatch_count"]) == 1

    first_entry = updated_state["queues"][queue_first_rel]
    second_entry = updated_state["queues"][queue_second_rel]
    assert str(first_entry.get("confirm_fastlane_queue_rel", "")).strip() == ""
    assert str(second_entry.get("confirm_fastlane_queue_rel", "")).strip().startswith(
        "artifacts/wfa/aggregate/confirm_fastlane_group_second_"
    )
    assert dispatch_marker.exists(), "successful dispatch marker must be written for the second candidate"
