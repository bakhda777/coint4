from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path


def _load_script(tmp_name: str):
    script_path = Path(__file__).resolve().parents[2] / "scripts/optimization/evolve_next_batch.py"
    spec = importlib.util.spec_from_file_location(f"evolve_next_batch_test_{tmp_name}", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_run_index(path: Path, *, cfg_path: Path, run_group: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "run_id",
        "run_group",
        "config_path",
        "results_dir",
        "status",
        "metrics_present",
        "sharpe_ratio_abs",
        "max_drawdown_on_equity",
        "total_trades",
        "total_pairs_traded",
        "wf_zero_pair_steps_pct",
        "tail_loss_worst_pair_share",
        "tail_loss_worst_period_share",
    ]
    base_id = f"{run_group}_seed_oos20220101_20221231"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(
            {
                "run_id": f"holdout_{base_id}",
                "run_group": run_group,
                "config_path": str(cfg_path),
                "results_dir": f"artifacts/wfa/runs/{run_group}/holdout_{base_id}",
                "status": "completed",
                "metrics_present": "true",
                "sharpe_ratio_abs": "1.4",
                "max_drawdown_on_equity": "-0.08",
                "total_trades": "260",
                "total_pairs_traded": "32",
                "wf_zero_pair_steps_pct": "0.01",
                "tail_loss_worst_pair_share": "0.20",
                "tail_loss_worst_period_share": "0.25",
            }
        )


def _write_base_config(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "walk_forward:",
                "  start_date: '2022-01-01'",
                "  end_date: '2022-12-31'",
                "portfolio:",
                "  risk_per_position_pct: 0.01",
                "  max_active_positions: 16",
                "backtest:",
                "  zscore_entry_threshold: 1.2",
                "  zscore_exit: 0.15",
                "  rolling_window: 96",
                "  max_var_multiplier: 1.05",
                "pair_selection:",
                "  max_pairs: 24",
                "  min_correlation: 0.4",
                "  coint_pvalue_threshold: 0.2",
                "filter_params:",
                "  max_hurst_exponent: 0.8",
                "  min_mean_crossings: 2",
                "  max_half_life_days: 60",
                "",
            ]
        ),
        encoding="utf-8",
    )


def test_evolve_next_batch_dry_run(tmp_path: Path) -> None:
    module = _load_script(tmp_path.name)
    base_cfg = tmp_path / "base.yaml"
    _write_base_config(base_cfg)
    run_index = tmp_path / "run_index.csv"
    _write_run_index(run_index, cfg_path=base_cfg, run_group="rgdry")

    rc = module.main(
        [
            "--base-config",
            str(base_cfg),
            "--controller-group",
            "ctrl_dry",
            "--run-group",
            "rgdry_next",
            "--run-index",
            str(run_index),
            "--contains",
            "rgdry",
            "--num-variants",
            "4",
            "--dedupe-distance",
            "0.0",
            "--min-windows",
            "1",
            "--window",
            "2022-01-01,2022-12-31",
            "--dry-run",
        ]
    )
    assert rc == 0


def test_evolve_next_batch_writes_queue_and_decision(tmp_path: Path) -> None:
    module = _load_script(f"{tmp_path.name}_write")
    base_cfg = tmp_path / "base.yaml"
    _write_base_config(base_cfg)
    run_index = tmp_path / "run_index.csv"
    _write_run_index(run_index, cfg_path=base_cfg, run_group="rgwrite")

    queue_root = tmp_path / "queue"
    configs_root = tmp_path / "configs"
    runs_root = tmp_path / "runs"
    state_path = tmp_path / "state" / "evolution_state.json"
    decision_dir = tmp_path / "decisions"
    rc = module.main(
        [
            "--base-config",
            str(base_cfg),
            "--controller-group",
            "ctrl_write",
            "--run-group",
            "rgwrite_next",
            "--run-index",
            str(run_index),
            "--contains",
            "rgwrite",
            "--num-variants",
            "2",
            "--dedupe-distance",
            "0.0",
            "--min-windows",
            "1",
            "--window",
            "2022-01-01,2022-12-31",
            "--queue-dir",
            str(queue_root),
            "--configs-dir",
            str(configs_root),
            "--runs-dir",
            str(runs_root),
            "--state-path",
            str(state_path),
            "--decision-dir",
            str(decision_dir),
            "--policy-scale",
            "macro",
        ]
    )
    assert rc == 0

    queue_path = queue_root / "rgwrite_next" / "run_queue.csv"
    assert queue_path.exists()
    with queue_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows
    assert all(str(row.get("status") or "").strip() == "planned" for row in rows)
    assert all(str(row.get("lineage_uid") or "").strip().startswith("lnuid_") for row in rows)
    assert all('"lineage_uid"' in str(row.get("metadata_json") or "") for row in rows)

    decisions = sorted(decision_dir.glob("*.json"))
    assert decisions
    payload = json.loads(decisions[-1].read_text(encoding="utf-8"))
    assert payload["run_group"] == "rgwrite_next"
    assert payload["llm_policy"]["used"] is False
    assert payload["lineage"]["uid_field"] == "lineage_uid"
    assert payload["lineage"]["unique_uids"] >= 1
    assert payload["operators"][0]["kind"] in {"crossover_uniform_v1", "coordinate_sweep_v1"}
    assert state_path.exists()


def test_invalid_firewall_blocks_known_invalid_before_materialization(tmp_path: Path) -> None:
    module = _load_script(f"{tmp_path.name}_invalid")
    base_cfg = tmp_path / "base.yaml"
    _write_base_config(base_cfg)
    parent_cfg = module.load_effective_yaml_config(base_cfg)
    knob_space = module._load_knob_space(None)
    invalid_cfg = dict(parent_cfg)
    invalid_cfg.setdefault("backtest", {})
    invalid_cfg["backtest"] = dict(invalid_cfg["backtest"])
    invalid_cfg["backtest"]["max_var_multiplier"] = 0.95
    invalid_genome = module.genome_from_config(invalid_cfg, knob_space=knob_space)
    proposal = module.CandidateProposal(
        candidate_id="cand_invalid",
        operator_id="op_test_invalid",
        parents=("parent::base",),
        genome=invalid_genome,
        changed_keys=("backtest.max_var_multiplier",),
        nearest_id="",
        nearest_distance=1.0,
        notes="invalid proposal",
        patch_ir=None,
    )
    invalid_state = tmp_path / "invalid_proposal_index.json"
    accepted, summary, state = module.filter_invalid_proposals_before_materialization(
        proposals=[proposal],
        app_root=tmp_path,
        invalid_index_path=invalid_state,
        parent_cfg=parent_cfg,
        windows=[("2022-01-01", "2022-12-31")],
        include_stress=True,
        ir_mode="knob",
        persist_state=True,
    )
    assert accepted == []
    assert summary["skipped_invalid"] == 1
    assert summary["codes"]["MAX_VAR_MULTIPLIER_INVALID"] == 1
    assert invalid_state.exists()
    persisted = json.loads(invalid_state.read_text(encoding="utf-8"))
    assert persisted["entries"][0]["code"] == "MAX_VAR_MULTIPLIER_INVALID"
    assert state["entries"][0]["fingerprint"].startswith("invalid_")


def test_invalid_firewall_quarantines_repeat_fingerprint(tmp_path: Path) -> None:
    module = _load_script(f"{tmp_path.name}_repeat")
    base_cfg = tmp_path / "base.yaml"
    _write_base_config(base_cfg)
    parent_cfg = module.load_effective_yaml_config(base_cfg)
    knob_space = module._load_knob_space(None)
    invalid_cfg = dict(parent_cfg)
    invalid_cfg.setdefault("backtest", {})
    invalid_cfg["backtest"] = dict(invalid_cfg["backtest"])
    invalid_cfg["backtest"]["max_var_multiplier"] = 0.95
    invalid_genome = module.genome_from_config(invalid_cfg, knob_space=knob_space)
    proposal = module.CandidateProposal(
        candidate_id="cand_repeat",
        operator_id="op_test_repeat",
        parents=("parent::base",),
        genome=invalid_genome,
        changed_keys=("backtest.max_var_multiplier",),
        nearest_id="",
        nearest_distance=1.0,
        notes="repeat invalid proposal",
        patch_ir=None,
    )
    invalid_state = tmp_path / "invalid_proposal_index.json"
    module.filter_invalid_proposals_before_materialization(
        proposals=[proposal],
        app_root=tmp_path,
        invalid_index_path=invalid_state,
        parent_cfg=parent_cfg,
        windows=[("2022-01-01", "2022-12-31")],
        include_stress=True,
        ir_mode="knob",
        persist_state=True,
    )
    accepted, summary, state = module.filter_invalid_proposals_before_materialization(
        proposals=[proposal],
        app_root=tmp_path,
        invalid_index_path=invalid_state,
        parent_cfg=parent_cfg,
        windows=[("2022-01-01", "2022-12-31")],
        include_stress=True,
        ir_mode="knob",
        persist_state=True,
    )
    assert accepted == []
    assert summary["skipped_quarantined"] == 1
    assert state["entries"][0]["occurrences"] >= 2


def test_invalid_firewall_allows_valid_candidate(tmp_path: Path) -> None:
    module = _load_script(f"{tmp_path.name}_valid")
    base_cfg = tmp_path / "base.yaml"
    _write_base_config(base_cfg)
    parent_cfg = module.load_effective_yaml_config(base_cfg)
    knob_space = module._load_knob_space(None)
    valid_cfg = dict(parent_cfg)
    valid_cfg.setdefault("backtest", {})
    valid_cfg["backtest"] = dict(valid_cfg["backtest"])
    valid_cfg["backtest"]["max_var_multiplier"] = 1.10
    valid_genome = module.genome_from_config(valid_cfg, knob_space=knob_space)
    proposal = module.CandidateProposal(
        candidate_id="cand_valid",
        operator_id="op_test_valid",
        parents=("parent::base",),
        genome=valid_genome,
        changed_keys=("backtest.max_var_multiplier",),
        nearest_id="",
        nearest_distance=1.0,
        notes="valid proposal",
        patch_ir=None,
    )
    invalid_state = tmp_path / "invalid_proposal_index.json"
    accepted, summary, _state = module.filter_invalid_proposals_before_materialization(
        proposals=[proposal],
        app_root=tmp_path,
        invalid_index_path=invalid_state,
        parent_cfg=parent_cfg,
        windows=[("2022-01-01", "2022-12-31")],
        include_stress=True,
        ir_mode="knob",
        persist_state=True,
    )
    assert len(accepted) == 1
    assert accepted[0].candidate_id == "cand_valid"
    assert summary["accepted"] == 1
    assert summary["skipped_invalid"] == 0
