#!/usr/bin/env python3
"""Minimal uncertainty analysis runner for pipeline integration tests."""

from __future__ import annotations

from pathlib import Path
import argparse

import pandas as pd


def run_uncertainty_analysis(output_dir: str, quick: bool = False) -> Path:
    """Create placeholder uncertainty artifacts for downstream consumers."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    confidence = pd.DataFrame(
        {
            "pair": ["PORTFOLIO"],
            "metric": ["sharpe"],
            "p05": [0.4 if quick else 0.5],
            "p50": [0.7],
            "p95": [1.0],
            "observed": [0.75],
        }
    )
    confidence.to_csv(output_path / "confidence.csv", index=False)

    report = (
        "# Confidence Intervals Report\n"
        "## Portfolio Confidence Intervals\n"
        "| Metric | P05 | P50 | P95 |\n"
        "|--------|-----|-----|-----|\n"
        f"| SHARPE | {confidence.loc[0, 'p05']:.3f} | {confidence.loc[0, 'p50']:.3f} | {confidence.loc[0, 'p95']:.3f} |\n"
    )
    (output_path / "CONFIDENCE_REPORT.md").write_text(report)

    return output_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Run uncertainty analysis")
    parser.add_argument("--output-dir", default="artifacts/uncertainty")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    run_uncertainty_analysis(args.output_dir, quick=args.quick)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
