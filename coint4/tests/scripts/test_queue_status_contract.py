from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path


def _load_module(script_name: str, tmp_path: Path):
    script_path = Path(__file__).resolve().parents[2] / "scripts" / "optimization" / script_name
    spec = importlib.util.spec_from_file_location(f"{script_name}_{tmp_path.name}", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_process_slo_and_seeder_share_stalled_contract(tmp_path: Path) -> None:
    process_module = _load_module("process_slo_guard_agent.py", tmp_path)
    seeder_module = _load_module("autonomous_queue_seeder.py", tmp_path)

    app_root = tmp_path / "app"
    aggregate_dir = app_root / "artifacts" / "wfa" / "aggregate" / "demo"
    aggregate_dir.mkdir(parents=True)
    (app_root / "configs").mkdir(parents=True)
    (app_root / "configs/ok.yaml").write_text("alpha: 1\n", encoding="utf-8")
    (aggregate_dir / "run_queue.csv").write_text(
        "run_name,config_path,status\n"
        "planned_ok,configs/ok.yaml,planned\n"
        "stalled_ok,configs/ok.yaml,stalled\n"
        "running_missing,configs/missing.yaml,running\n"
        "failed_ok,configs/ok.yaml,failed\n",
        encoding="utf-8",
    )

    stats = process_module.queue_stats(
        aggregate_root=app_root / "artifacts" / "wfa" / "aggregate",
        app_root=app_root,
    )
    queue_stats = next(iter(stats["per_queue"].values()))
    _, dispatchable_pending, executable_pending, runnable_queue_count, _, _ = seeder_module._summarize_queues(
        app_root / "artifacts" / "wfa" / "aggregate"
    )

    assert queue_stats["stalled"] == 1
    assert queue_stats["executable_pending"] == 4
    assert dispatchable_pending == 3
    assert executable_pending == 4
    assert runnable_queue_count == 1


def test_seeder_summary_excludes_queue_level_blocked_pending_rows(tmp_path: Path) -> None:
    seeder_module = _load_module("autonomous_queue_seeder.py", tmp_path)

    app_root = tmp_path / "app"
    aggregate_root = app_root / "artifacts" / "wfa" / "aggregate"
    state_dir = aggregate_root / ".autonomous"
    state_dir.mkdir(parents=True, exist_ok=True)
    (app_root / "configs").mkdir(parents=True)
    (app_root / "configs/ok.yaml").write_text("alpha: 1\n", encoding="utf-8")

    blocked_queue = aggregate_root / "blocked_q" / "run_queue.csv"
    blocked_queue.parent.mkdir(parents=True)
    blocked_queue.write_text(
        "run_name,config_path,status\n"
        "blocked,configs/ok.yaml,planned\n",
        encoding="utf-8",
    )
    open_queue = aggregate_root / "open_q" / "run_queue.csv"
    open_queue.parent.mkdir(parents=True)
    open_queue.write_text(
        "run_name,config_path,status\n"
        "open,configs/ok.yaml,stalled\n"
        "missing_cfg,configs/missing.yaml,planned\n",
        encoding="utf-8",
    )
    (state_dir / "fullspan_decision_state.json").write_text(
        json.dumps(
            {
                "queues": {
                    "artifacts/wfa/aggregate/blocked_q/run_queue.csv": {
                        "promotion_verdict": "ANALYZE",
                        "cutover_permission": "FAIL_CLOSED",
                    }
                }
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    total_pending, dispatchable_pending, executable_pending, runnable_queue_count, _scanned, _paths = (
        seeder_module._summarize_queues(aggregate_root)
    )

    assert total_pending == 3
    assert dispatchable_pending == 1
    assert executable_pending == 2
    assert runnable_queue_count == 2


def test_hygiene_converts_legacy_header_only_seed_queue_to_blocked_orphan(tmp_path: Path) -> None:
    module = _load_module("autonomous_queue_seeder.py", tmp_path)

    app_root = tmp_path / "app"
    aggregate_dir = app_root / "artifacts" / "wfa" / "aggregate"
    queue_dir = aggregate_dir / "autonomous_seed_legacy"
    queue_dir.mkdir(parents=True)
    queue_path = queue_dir / "run_queue.csv"
    queue_path.write_text("run_name,config_path,status\n", encoding="utf-8")
    orphan_path = aggregate_dir / ".autonomous" / "orphan_queues.csv"

    result = module._hygiene_seed_queues(
        aggregate_dir=aggregate_dir,
        app_root=app_root,
        run_group_prefix="autonomous_seed",
        orphan_path=orphan_path,
    )

    assert result["orphaned"] == 1
    rows = list(csv.DictReader(queue_path.open(newline="", encoding="utf-8")))
    assert len(rows) == 1
    assert rows[0]["status"] == "blocked"
    assert rows[0]["note"] == "queue_pruned_empty"

    orphan_rows = list(csv.DictReader(orphan_path.open(newline="", encoding="utf-8")))
    assert orphan_rows[0]["queue"] == "artifacts/wfa/aggregate/autonomous_seed_legacy/run_queue.csv"
    assert orphan_rows[0]["reason"] == "queue_pruned_empty_legacy"


def test_queue_policy_defaults_fill_missing_gate_fields(tmp_path: Path) -> None:
    module = _load_module("autonomous_queue_seeder.py", tmp_path)

    app_root = tmp_path / "app"
    queue_dir = app_root / "artifacts" / "wfa" / "aggregate" / "autonomous_seed_demo"
    queue_dir.mkdir(parents=True)
    queue_path = queue_dir / "run_queue.csv"
    queue_path.write_text(
        "run_name,config_path,status\n"
        "demo,configs/example.yaml,planned\n",
        encoding="utf-8",
    )
    sidecar_path = module._write_queue_policy_sidecar(
        queue_path=queue_path,
        app_root=app_root,
        planner_policy_hash="policy",
        selected_lane="winner_proximate",
        selected_lane_index=0,
        token_rotation=0,
        parent_rotation_offset=0,
        parent_diversity_depth=0,
        confirm_replay_hints=[],
        decision_payload={},
    )
    module._decorate_queue_metadata(
        queue_path=queue_path,
        planner_policy_hash="policy",
        queue_policy_path=sidecar_path,
        app_root=app_root,
    )

    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    assert sidecar["gate_status"] == "UNKNOWN"
    assert sidecar["pre_rank_score"] == 0.0

    rows = list(csv.DictReader(queue_path.open(newline="", encoding="utf-8")))
    meta = json.loads(rows[0]["metadata_json"])
    assert meta["gate_status"] == "UNKNOWN"
    assert meta["strict_gate_status"] == "UNKNOWN"


def test_queue_dispatch_block_reason_distinguishes_reject_fail_closed_and_orphan(tmp_path: Path) -> None:
    module = _load_module("_queue_status_contract.py", tmp_path)

    now_epoch = 1_777_000_000.0

    assert (
        module.queue_dispatch_block_reason(
            queue_rel="artifacts/wfa/aggregate/demo/run_queue.csv",
            fullspan_entry={
                "promotion_verdict": "REJECT",
                "cutover_permission": "FAIL_CLOSED",
                "contract_hard_pass": False,
            },
            now_epoch=now_epoch,
        )
        == "FULLSPAN_REJECT"
    )

    assert (
        module.queue_dispatch_block_reason(
            queue_rel="artifacts/wfa/aggregate/demo/run_queue.csv",
            fullspan_entry={
                "promotion_verdict": "ANALYZE",
                "cutover_permission": "FAIL_CLOSED",
                "contract_hard_pass": False,
            },
            now_epoch=now_epoch,
        )
        == "FAIL_CLOSED"
    )

    assert (
        module.queue_dispatch_block_reason(
            queue_rel="artifacts/wfa/aggregate/demo/run_queue.csv",
            orphan_entry={"until_ts": now_epoch + 600, "reason": "no_progress_streak_6"},
            now_epoch=now_epoch,
        )
        == "ORPHAN_COOLDOWN"
    )

    assert (
        module.queue_dispatch_block_reason(
            queue_rel="artifacts/wfa/aggregate/demo/run_queue.csv",
            orphan_entry={"until_ts": now_epoch + 600, "reason": "coverage_fail_closed"},
            now_epoch=now_epoch,
        )
        == "FAIL_CLOSED"
    )
