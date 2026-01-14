from __future__ import annotations

import yaml

from coint2.utils.pairs_loader import load_pairs, load_pair_tuples


def test_load_pairs_from_pairs_key(tmp_path):
    payload = {
        "pairs": [
            {"symbol1": "AAAUSDT", "symbol2": "BBBUSDT"},
            {"pair": "CCCUSDT/DDDUSDT"},
        ]
    }
    path = tmp_path / "pairs.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    pairs = load_pairs(path)

    assert len(pairs) == 2
    assert pairs[0]["symbol1"] == "AAAUSDT"
    assert pairs[0]["symbol2"] == "BBBUSDT"
    assert pairs[1]["symbol1"] == "CCCUSDT"
    assert pairs[1]["symbol2"] == "DDDUSDT"


def test_load_pairs_from_list_root(tmp_path):
    payload = ["AAAUSDT/BBBUSDT", {"symbol1": "CCCUSDT", "symbol2": "DDDUSDT"}]
    path = tmp_path / "pairs.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    tuples = load_pair_tuples(path)

    assert tuples == [("AAAUSDT", "BBBUSDT"), ("CCCUSDT", "DDDUSDT")]
