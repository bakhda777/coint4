from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPTS_PAPERS = REPO_ROOT / "scripts" / "papers"
if str(SCRIPTS_PAPERS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_PAPERS))

from make_cards import validate_card_payload  # noqa: E402


def test_card_schema_validator() -> None:
    schema_path = REPO_ROOT / "papers" / "cards" / "card.schema.json"
    schema = json.loads(schema_path.read_text(encoding="utf-8"))

    valid_card = {
        "schema_version": "card-v1.1.0",
        "text_sha256": "a" * 64,
        "paper_id": "abcd1234efgh",
        "title": "A Practical Study of Crypto Pair Trading",
        "year": 2024,
        "summary": "Paper compares cointegration variants with costs, OOS checks, and drawdown controls.",
        "method_family": ["cointegration", "ou_model"],
        "key_findings": ["Cost-aware variant was more stable OOS."],
        "pitfalls": ["Ignoring slippage leads to optimistic Sharpe."],
        "actionability_score": 84,
        "actionable_experiments": [
            {
                "idea": "Run funding-aware entry filter with turnover cap",
                "expected_effect": "Sharpe↑ and turnover↓ in volatile regimes",
                "impact": "high",
                "effort": "medium",
                "why": "Funding spikes degrade spread mean reversion in perp pairs.",
                "wfa_checks": ["OOS Sharpe", "Max Drawdown", "Turnover"]
            }
        ],
        "evidence": ["Sharpe improved from 0.8 to 1.1 in OOS window"]
    }

    ok, errors = validate_card_payload(valid_card, schema)
    assert ok, errors

    invalid_card = dict(valid_card)
    invalid_card["summary"] = "too short"
    invalid_card["actionable_experiments"] = [
        {
            "idea": "x",
            "expected_effect": "too short",
            "impact": "huge",
            "effort": "hard",
            "why": "n/a",
            "wfa_checks": []
        }
    ]

    ok_invalid, errors_invalid = validate_card_payload(invalid_card, schema)
    assert not ok_invalid
    assert errors_invalid
