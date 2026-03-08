from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "optimization" / "evolve_next_batch.py"


def _load_script(tmp_name: str):
    spec = importlib.util.spec_from_file_location(f"evolve_next_batch_invalid_{tmp_name}", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


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
                "  max_var_multiplier: 1.05",
                "  zscore_entry_threshold: 1.2",
                "  zscore_exit: 0.15",
                "  rolling_window: 96",
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


def _build_candidate(module, *, parent_cfg: dict[str, object], candidate_id: str, overrides: dict[str, object]):
    genome = module.genome_from_config(parent_cfg, knob_space=module._load_knob_space(None))
    genome.update(overrides)
    return module.CandidateProposal(
        candidate_id=candidate_id,
        operator_id="op_test",
        parents=("parent::base",),
        genome=genome,
        changed_keys=tuple(overrides.keys()),
        nearest_id="parent::base",
        nearest_distance=0.25,
        notes="test candidate",
        patch_ir=None,
    )


def test_main_skips_invalid_proposal_before_materialization(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_script(f"{tmp_path.name}_main")
    base_cfg = tmp_path / "base.yaml"
    _write_base_config(base_cfg)
    parent_cfg = module.load_effective_yaml_config(base_cfg)
    invalid_candidate = _build_candidate(
        module,
        parent_cfg=parent_cfg,
        candidate_id="invalid_mvm",
        overrides={"backtest.max_var_multiplier": 0.995},
    )
    monkeypatch.setattr(module, "_generate_proposals", lambda **_kwargs: [invalid_candidate])

    invalid_state_path = tmp_path / ".autonomous" / "invalid_proposal_index.json"
    with pytest.raises(SystemExit, match="no valid queue entries were generated after preflight firewall"):
        module.main(
            [
                "--base-config",
                str(base_cfg),
                "--controller-group",
                "ctrl_invalid",
                "--run-group",
                "rg_invalid",
                "--run-index",
                str(tmp_path / "missing_run_index.csv"),
                "--num-variants",
                "1",
                "--dedupe-distance",
                "0.0",
                "--min-windows",
                "1",
                "--window",
                "2022-01-01,2022-12-31",
                "--queue-dir",
                str(tmp_path / "queue"),
                "--configs-dir",
                str(tmp_path / "configs"),
                "--runs-dir",
                str(tmp_path / "runs"),
                "--state-path",
                str(tmp_path / "state" / "evolution_state.json"),
                "--decision-dir",
                str(tmp_path / "decisions"),
                "--invalid-proposal-state-path",
                str(invalid_state_path),
            ]
        )

    assert not (tmp_path / "queue" / "rg_invalid" / "run_queue.csv").exists()
    assert not any((tmp_path / "configs").rglob("*.yaml"))
    state = module._load_invalid_proposal_index(invalid_state_path)
    assert len(state["entries"]) == 1
    assert state["entries"][0]["code"] == "MAX_VAR_MULTIPLIER_INVALID"


def test_neutral_max_var_multiplier_passes_preflight_firewall(tmp_path: Path) -> None:
    module = _load_script(f"{tmp_path.name}_neutral")
    base_cfg = tmp_path / "base.yaml"
    _write_base_config(base_cfg)
    parent_cfg = module.load_effective_yaml_config(base_cfg)
    neutral_candidate = _build_candidate(
        module,
        parent_cfg=parent_cfg,
        candidate_id="neutral_mvm",
        overrides={"backtest.max_var_multiplier": 1.0},
    )
    invalid_state_path = tmp_path / ".autonomous" / "invalid_proposal_index.json"

    accepted, summary, state = module.filter_invalid_proposals_before_materialization(
        proposals=[neutral_candidate],
        app_root=tmp_path,
        invalid_index_path=invalid_state_path,
        parent_cfg=parent_cfg,
        windows=[("2022-01-01", "2022-12-31")],
        include_stress=True,
        ir_mode="knob",
        persist_state=True,
    )

    assert len(accepted) == 1
    assert accepted[0].candidate_id == "neutral_mvm"
    assert summary["skipped_invalid"] == 0
    assert summary["codes"] == {}
    assert state["entries"] == []
    assert not invalid_state_path.exists()


def test_repeated_invalid_fingerprint_is_quarantined(tmp_path: Path) -> None:
    module = _load_script(f"{tmp_path.name}_repeat")
    base_cfg = tmp_path / "base.yaml"
    _write_base_config(base_cfg)
    parent_cfg = module.load_effective_yaml_config(base_cfg)
    invalid_candidate = _build_candidate(
        module,
        parent_cfg=parent_cfg,
        candidate_id="invalid_repeat",
        overrides={"backtest.max_var_multiplier": 0.95},
    )
    invalid_state_path = tmp_path / ".autonomous" / "invalid_proposal_index.json"

    accepted_first, summary_first, _state_first = module.filter_invalid_proposals_before_materialization(
        proposals=[invalid_candidate],
        app_root=tmp_path,
        invalid_index_path=invalid_state_path,
        parent_cfg=parent_cfg,
        windows=[("2022-01-01", "2022-12-31")],
        include_stress=True,
        ir_mode="knob",
        persist_state=True,
    )
    accepted_second, summary_second, state_second = module.filter_invalid_proposals_before_materialization(
        proposals=[invalid_candidate],
        app_root=tmp_path,
        invalid_index_path=invalid_state_path,
        parent_cfg=parent_cfg,
        windows=[("2022-01-01", "2022-12-31")],
        include_stress=True,
        ir_mode="knob",
        persist_state=True,
    )

    assert accepted_first == []
    assert summary_first["skipped_invalid"] == 1
    assert accepted_second == []
    assert summary_second["skipped_quarantined"] == 1
    assert state_second["entries"][0]["occurrences"] == 2
    assert state_second["entries"][0]["code"] == "MAX_VAR_MULTIPLIER_INVALID"


def test_valid_proposal_passes_firewall(tmp_path: Path) -> None:
    module = _load_script(f"{tmp_path.name}_valid")
    base_cfg = tmp_path / "base.yaml"
    _write_base_config(base_cfg)
    parent_cfg = module.load_effective_yaml_config(base_cfg)
    valid_candidate = _build_candidate(
        module,
        parent_cfg=parent_cfg,
        candidate_id="valid_candidate",
        overrides={"portfolio.max_active_positions": 18},
    )
    invalid_state_path = tmp_path / ".autonomous" / "invalid_proposal_index.json"

    accepted, summary, state = module.filter_invalid_proposals_before_materialization(
        proposals=[valid_candidate],
        app_root=tmp_path,
        invalid_index_path=invalid_state_path,
        parent_cfg=parent_cfg,
        windows=[("2022-01-01", "2022-12-31")],
        include_stress=True,
        ir_mode="knob",
        persist_state=True,
    )

    assert len(accepted) == 1
    assert accepted[0].candidate_id == "valid_candidate"
    assert summary["skipped_invalid"] == 0
    assert summary["skipped_quarantined"] == 0
    assert summary["codes"] == {}
    assert state["entries"] == []
    assert not invalid_state_path.exists()
