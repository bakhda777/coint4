from coint2.pipeline.pair_scanner import calculate_pair_score, evaluate_pair


def test_evaluate_pair_when_pvalue_high_then_fails() -> None:
    metrics = {
        "pvalue": 0.2,
        "half_life": 20,
        "crossings": 20,
        "beta_drift": 0.01,
    }
    config = {
        "criteria": {
            "coint_pvalue_max": 0.05,
            "hl_min": 5,
            "hl_max": 200,
            "min_cross": 10,
            "beta_drift_max": 0.15,
        }
    }

    verdict, reason = evaluate_pair(metrics, config)

    assert verdict == "FAIL"
    assert reason == "pvalue"


def test_calculate_pair_score_prefers_lower_pvalue() -> None:
    metrics_low = {
        "pvalue": 0.01,
        "half_life": 30,
        "crossings": 20,
        "beta_drift": 0.05,
    }
    metrics_high = {
        "pvalue": 0.10,
        "half_life": 30,
        "crossings": 20,
        "beta_drift": 0.05,
    }

    score_low = calculate_pair_score(metrics_low, {})
    score_high = calculate_pair_score(metrics_high, {})

    assert score_low > score_high
