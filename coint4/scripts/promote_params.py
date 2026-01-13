#!/usr/bin/env python3
"""Parameter promotion utilities."""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Tuple

import yaml


class ParameterPromoter:
    """Promote best parameters into locked production config."""

    def __init__(self, base_dir: str = "."):
        self.base_dir = Path(base_dir)
        self.wfa_path = self.base_dir / "artifacts" / "wfa" / "performance_summary.json"
        self.optuna_path = self.base_dir / "artifacts" / "optuna" / "best_params.json"
        self.locked_dir = self.base_dir / "artifacts" / "production" / "locked"
        self.catalog_path = self.base_dir / "artifacts" / "production" / "PARAMS_CATALOG.json"

    def _load_wfa_metrics(self) -> Dict:
        with open(self.wfa_path, "r") as f:
            return json.load(f)

    def _load_optuna_params(self) -> Dict:
        with open(self.optuna_path, "r") as f:
            return json.load(f)

    def validate_params(self) -> Dict:
        """Validate that metrics pass promotion thresholds."""
        metrics = self._load_wfa_metrics()
        sharpe = metrics.get("sharpe", 0)
        max_dd = metrics.get("max_dd", 1)
        trades = metrics.get("trades", 0)
        psr = metrics.get("psr", 0)

        if sharpe < 1.0:
            return {"passed": False, "reason": "Sharpe too low", "sharpe": sharpe, "max_dd": max_dd}
        if trades < 10:
            return {"passed": False, "reason": "Trade count too low", "sharpe": sharpe, "max_dd": max_dd}
        if psr < 1.5:
            return {"passed": False, "reason": "PSR too low", "sharpe": sharpe, "max_dd": max_dd}
        if max_dd > 0.3:
            return {"passed": False, "reason": "Drawdown too high", "sharpe": sharpe, "max_dd": max_dd}

        return {
            "passed": True,
            "sharpe": sharpe,
            "max_dd": max_dd,
            "trades": trades,
            "psr": psr,
        }

    def _locked_path(self, pair: str, timeframe: str, week: str) -> Path:
        return self.locked_dir / f"params_{pair}_{timeframe}_{week}.yaml"

    def _load_catalog(self) -> Dict:
        if self.catalog_path.exists():
            with open(self.catalog_path, "r") as f:
                return json.load(f)
        return {"promotions": []}

    def _save_catalog(self, catalog: Dict) -> None:
        self.catalog_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.catalog_path, "w") as f:
            json.dump(catalog, f, indent=2)

    def promote(self, pair: str, timeframe: str, week: str | None = None) -> bool:
        """Promote best params into locked config."""
        week = week or datetime.now(timezone.utc).strftime("%YW%W")
        locked_path = self._locked_path(pair, timeframe, week)

        try:
            validation = self.validate_params()
            if not validation.get("passed"):
                return False

            params = self._load_optuna_params()
            metadata = {
                "pair": pair,
                "timeframe": timeframe,
                "week": week,
                "promoted_at": datetime.now(timezone.utc).isoformat(),
                "sharpe": validation["sharpe"],
                "max_dd": validation["max_dd"],
                "trades": validation.get("trades", 0),
                "psr": validation.get("psr", 0),
            }
            payload = {"parameters": params, "metadata": metadata}

            self.locked_dir.mkdir(parents=True, exist_ok=True)
            temp_path = locked_path.with_suffix(".yaml.tmp")

            with open(temp_path, "w") as f:
                yaml.safe_dump(payload, f)
            temp_path.replace(locked_path)

            catalog = self._load_catalog()
            catalog.setdefault("promotions", []).append(
                {
                    "pair": pair,
                    "timeframe": timeframe,
                    "week": week,
                    "sharpe": validation["sharpe"],
                    "max_dd": validation["max_dd"],
                    "timestamp": metadata["promoted_at"],
                }
            )
            self._save_catalog(catalog)
            return True
        except Exception:
            # Ensure no partial locked file remains
            if locked_path.exists():
                try:
                    locked_path.unlink()
                except Exception:
                    pass
            return False


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Promote best parameters")
    parser.add_argument("--pair", required=True)
    parser.add_argument("--timeframe", required=True)
    parser.add_argument("--week", default=None)
    parser.add_argument("--base-dir", default=".")
    args = parser.parse_args()

    promoter = ParameterPromoter(base_dir=args.base_dir)
    success = promoter.promote(args.pair, args.timeframe, args.week)
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
