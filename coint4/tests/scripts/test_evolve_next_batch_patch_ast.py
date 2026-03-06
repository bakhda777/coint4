from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path


def _load_script(tmp_name: str):
    script_path = Path(__file__).resolve().parents[2] / "scripts/optimization/evolve_next_batch.py"
    spec = importlib.util.spec_from_file_location(f"evolve_next_batch_patchast_test_{tmp_name}", script_path)
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


def test_evolve_next_batch_patch_ast_dry_run(tmp_path: Path) -> None:
    module = _load_script(tmp_path.name)
    base_cfg = tmp_path / "base.yaml"
    base_cfg.write_text(
        "\n".join(
            [
                "walk_forward:",
                "  start_date: '2022-01-01'",
                "  end_date: '2022-12-31'",
                "portfolio:",
                "  risk_per_position_pct: 0.01",
                "backtest:",
                "  zscore_entry_threshold: 1.2",
                "pair_selection:",
                "  max_pairs: 24",
                "  min_correlation: 0.4",
                "",
            ]
        ),
        encoding="utf-8",
    )
    run_index = tmp_path / "run_index.csv"
    _write_run_index(run_index, cfg_path=base_cfg, run_group="rgpatch")

    knob_space = tmp_path / "knobs.json"
    knob_space.write_text(
        json.dumps(
            [
                {"key": "portfolio.risk_per_position_pct", "type": "float", "min": 0.003, "max": 0.03, "step": 0.001},
                {"key": "backtest.zscore_entry_threshold", "type": "float", "min": 0.6, "max": 3.0, "step": 0.05},
                {"key": "pair_selection.max_pairs", "type": "int", "min": 8, "max": 96, "step": 2},
                {"key": "pair_selection.min_correlation", "type": "float", "min": 0.1, "max": 0.9, "step": 0.02},
            ],
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    rc = module.main(
        [
            "--base-config",
            str(base_cfg),
            "--controller-group",
            "ctrl_patch",
            "--run-group",
            "rgpatch_next",
            "--run-index",
            str(run_index),
            "--contains",
            "rgpatch",
            "--knob-space",
            str(knob_space),
            "--ir-mode",
            "patch_ast",
            "--num-variants",
            "2",
            "--dedupe-distance",
            "0.0",
            "--ast-max-redundancy-similarity",
            "1.0",
            "--ast-max-complexity-score",
            "999.0",
            "--patch-max-attempts",
            "50",
            "--min-windows",
            "1",
            "--window",
            "2022-01-01,2022-12-31",
            "--dry-run",
        ]
    )
    assert rc == 0


def test_patch_ast_topup_fills_target_variants(tmp_path: Path, monkeypatch) -> None:
    module = _load_script(f"{tmp_path.name}_topup")
    base_cfg = tmp_path / "base.yaml"
    base_cfg.write_text(
        "\n".join(
            [
                "walk_forward:",
                "  start_date: '2022-01-01'",
                "  end_date: '2022-12-31'",
                "portfolio:",
                "  risk_per_position_pct: 0.01",
                "backtest:",
                "  zscore_entry_threshold: 1.2",
                "pair_selection:",
                "  max_pairs: 24",
                "  min_correlation: 0.4",
                "",
            ]
        ),
        encoding="utf-8",
    )
    run_index = tmp_path / "run_index.csv"
    _write_run_index(run_index, cfg_path=base_cfg, run_group="rgpatch_topup")

    queue_root = tmp_path / "queue"
    configs_root = tmp_path / "configs"
    runs_root = tmp_path / "runs"
    state_path = tmp_path / "state" / "evolution_state.json"
    decision_dir = tmp_path / "decisions"

    def _proposal(candidate_id: str, risk_value: float):
        return module.CandidateProposal(
            candidate_id=candidate_id,
            operator_id="op_targeted_primary",
            parents=("parent::seed",),
            genome={"portfolio.risk_per_position_pct": float(risk_value)},
            changed_keys=("portfolio.risk_per_position_pct",),
            nearest_id="",
            nearest_distance=1.0,
            notes="test proposal",
            patch_ir=None,
        )

    def _fake_patch_ast(**_: object):
        return [_proposal("evo_patch_only", 0.011)]

    def _fake_generate_proposals(**kwargs: object):
        count = int(kwargs.get("num_variants", 0))
        return [_proposal(f"evo_topup_{idx:03d}", 0.012 + idx * 0.001) for idx in range(1, count + 1)]

    monkeypatch.setattr(module, "_generate_patch_ast_proposals", _fake_patch_ast)
    monkeypatch.setattr(module, "_generate_proposals", _fake_generate_proposals)

    rc = module.main(
        [
            "--base-config",
            str(base_cfg),
            "--controller-group",
            "ctrl_patch_topup",
            "--run-group",
            "rgpatch_topup_next",
            "--run-index",
            str(run_index),
            "--contains",
            "rgpatch_topup",
            "--planner-policy-hash",
            "policy_patch_hash",
            "--planner-hash",
            "planner_patch_hash",
            "--seed-lane",
            "confirm_replay",
            "--seed-lane-index",
            "3",
            "--parent-diversity-depth",
            "3",
            "--parent-rotation-offset",
            "2",
            "--confirm-replay-hint",
            "rgpatch_topup",
            "--ir-mode",
            "patch_ast",
            "--num-variants",
            "4",
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
        ]
    )
    assert rc == 0

    queue_path = queue_root / "rgpatch_topup_next" / "run_queue.csv"
    assert queue_path.exists()
    with queue_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 8  # 4 variants * paired holdout+stress for one OOS window
    row_meta = json.loads(str(rows[0]["metadata_json"]))
    assert row_meta["planner_policy_hash"] == "policy_patch_hash"
    assert row_meta["planner_hash"] == "planner_patch_hash"
    assert row_meta["seed_lane"] == "confirm_replay"
    assert row_meta["seed_lane_index"] == 3
    assert row_meta["parent_diversity_depth"] == 3
    assert row_meta["parent_rotation_offset"] == 2
    assert row_meta["confirm_replay_hints"] == ["rgpatch_topup"]

    decisions = sorted(decision_dir.glob("*.json"))
    assert decisions
    payload = json.loads(decisions[-1].read_text(encoding="utf-8"))
    assert len(payload.get("proposals") or []) == 4
    assert payload["planner_hashes"]["policy_hash"] == "policy_patch_hash"
    assert payload["planner_hashes"]["planner_hash"] == "planner_patch_hash"
    assert payload["lane_selection"]["seed_lane"] == "confirm_replay"
    assert payload["lane_selection"]["seed_lane_index"] == 3
    assert payload["lane_selection"]["confirm_replay_hints"] == ["rgpatch_topup"]
    assert payload["fastlane_materialization"]["prepared"] is True


def test_segment_mutation_candidate_rewrites_only_one_root_segment(tmp_path: Path) -> None:
    module = _load_script(f"{tmp_path.name}_mutseg")
    knob_space = module.knob_specs_from_dicts(
        [
            {"key": "portfolio.risk_per_position_pct", "type": "float", "min": 0.003, "max": 0.03, "step": 0.001},
            {"key": "pair_selection.max_pairs", "type": "int", "min": 8, "max": 96, "step": 2},
        ]
    )
    parent_entry = {
        "candidate_id": "cand_prev",
        "factors": [
            {
                "factor_id": "F-A01",
                "category": "risk",
                "target_key": "portfolio.risk_per_position_pct",
                "op": "set",
                "value": 0.01,
                "rationale": "seed risk",
            },
            {
                "factor_id": "F-A02",
                "category": "selection",
                "target_key": "pair_selection.max_pairs",
                "op": "set",
                "value": 24,
                "rationale": "seed selection",
            },
        ],
    }
    hypothesis, factors, parent_id = module._segment_mutation_candidate(
        parent_entry=parent_entry,
        generation=3,
        variant_index=2,
        failure_mode="dd",
        allowed_keys=["portfolio.risk_per_position_pct", "pair_selection.max_pairs"],
        knob_space=knob_space,
        rng=module.np.random.default_rng(2026),
        max_factors=3,
    )
    assert parent_id == "cand_prev"
    assert str(hypothesis.get("hypothesis_id", "")).startswith("HYP-MUTSEG_")
    assert factors
    assert any("segment_mutation[" in str(item.get("rationale") or "") for item in factors)
    assert any("frozen prefix segment[" in str(item.get("rationale") or "") for item in factors)
    assert {
        str(item.get("target_key") or "")
        for item in factors
    } <= {"portfolio.risk_per_position_pct", "pair_selection.max_pairs"}


def test_load_patch_zoo_skips_invalid_patch_ir(tmp_path: Path) -> None:
    module = _load_script(f"{tmp_path.name}_zoo")
    decisions_dir = tmp_path / "decisions"
    decisions_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "proposals": [
            {
                "candidate_id": "bad_patch_ir",
                "patch_ir": {
                    "ir_version": "config_patch_ast.v1",
                    "source": "deterministic",
                    "parents": ["p0"],
                    "hypothesis": {"thesis": "Короткая."},
                    "factors": [
                        {
                            "target_key": "risk.daily_stop_pct",
                            "op": "set",
                            "value": 0.02,
                            "rationale": "ok rationale",
                        }
                    ],
                    "materialized_patch": {"risk": {"daily_stop_pct": 0.02}},
                    "gates": {},
                    "semantic_gate": {"ok": True, "source": "deterministic", "reasons": []},
                },
            },
            {
                "candidate_id": "good_patch_ir",
                "patch_ir": {
                    "ir_version": "config_patch_ast.v1",
                    "source": "deterministic",
                    "parents": ["p1"],
                    "hypothesis": {"thesis": "Достаточно длинная гипотеза про механику стратегии."},
                    "factors": [
                        {
                            "target_key": "risk.daily_stop_pct",
                            "op": "set",
                            "value": 0.02,
                            "rationale": "ok rationale",
                        }
                    ],
                    "materialized_patch": {"risk": {"daily_stop_pct": 0.02}},
                    "gates": {
                        "complexity": {
                            "score": 10.0,
                            "symbolic_length": 5,
                            "parameter_count": 1,
                            "feature_count": 1,
                        },
                        "redundancy": {"nearest_id": None, "nearest_similarity": None, "nearest_common_subtree": None},
                        "limits": {
                            "max_complexity_score": 60.0,
                            "max_redundancy_similarity": 0.85,
                            "alpha_sl": 1.0,
                            "alpha_pc": 1.0,
                            "alpha_feat": 1.0,
                        },
                    },
                    "semantic_gate": {"ok": True, "source": "deterministic", "reasons": []},
                },
            },
        ]
    }
    (decisions_dir / "decision.json").write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    entries = module._load_patch_zoo_from_decisions(decisions_dir, max_items=10)
    assert len(entries) == 1
    assert entries[0]["candidate_id"] == "good_patch_ir"
