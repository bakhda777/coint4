#!/usr/bin/env python3
"""Parameter rollback utilities."""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import yaml


class ParameterRollback:
    """Rollback locked parameter versions."""

    def __init__(self, base_dir: str = "."):
        self.base_dir = Path(base_dir)
        self.locked_dir = self.base_dir / "artifacts" / "production" / "locked"
        self.catalog_path = self.base_dir / "artifacts" / "production" / "PARAMS_CATALOG.json"

    def _current_link(self, pair: str, timeframe: str) -> Path:
        return self.locked_dir / f"params_{pair}_{timeframe}_current.yaml"

    def _week_path(self, pair: str, timeframe: str, week: str) -> Path:
        return self.locked_dir / f"params_{pair}_{timeframe}_{week}.yaml"

    def _load_catalog(self) -> Dict:
        if self.catalog_path.exists():
            with open(self.catalog_path, "r") as f:
                return json.load(f)
        return {"promotions": [], "rollbacks": []}

    def _save_catalog(self, catalog: Dict) -> None:
        self.catalog_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.catalog_path, "w") as f:
            json.dump(catalog, f, indent=2)

    def rollback_to_week(self, pair: str, timeframe: str, week: str) -> bool:
        """Rollback to a specific week."""
        target = self._week_path(pair, timeframe, week)
        if not target.exists():
            return False

        self.locked_dir.mkdir(parents=True, exist_ok=True)
        current = self._current_link(pair, timeframe)
        original_target = current.resolve() if current.exists() else None

        try:
            if current.exists() or current.is_symlink():
                current.unlink()
            current.symlink_to(target)

            catalog = self._load_catalog()
            catalog.setdefault("rollbacks", []).append(
                {
                    "pair": pair,
                    "timeframe": timeframe,
                    "from_week": original_target.stem.split("_")[-1] if original_target else None,
                    "to_week": week,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
            )
            self._save_catalog(catalog)
            return True
        except Exception:
            # Restore original link if possible
            if original_target is not None and not current.exists():
                try:
                    current.symlink_to(original_target)
                except Exception:
                    pass
            return False

    def find_best_week(self, pair: str, timeframe: str) -> Optional[str]:
        """Find week with best Sharpe ratio."""
        if not self.locked_dir.exists():
            return None

        best_week = None
        best_sharpe = float("-inf")
        for file_path in self.locked_dir.glob(f"params_{pair}_{timeframe}_*.yaml"):
            if file_path.name.endswith("_current.yaml"):
                continue
            with open(file_path, "r") as f:
                data = yaml.safe_load(f)
            sharpe = data.get("metadata", {}).get("sharpe", float("-inf"))
            if sharpe > best_sharpe:
                best_sharpe = sharpe
                best_week = data.get("metadata", {}).get("week")
        return best_week

    def rollback_to_best(self, pair: str, timeframe: str) -> bool:
        """Rollback to best performing week."""
        best_week = self.find_best_week(pair, timeframe)
        if not best_week:
            return False
        return self.rollback_to_week(pair, timeframe, best_week)

    def get_rollback_history(self, pair: str, timeframe: str) -> List[Dict]:
        """Get rollback history for pair/timeframe."""
        catalog = self._load_catalog()
        return [
            entry
            for entry in catalog.get("rollbacks", [])
            if entry.get("pair") == pair and entry.get("timeframe") == timeframe
        ]

    def compare_weeks(self, pair: str, timeframe: str, week1: str, week2: str) -> Dict:
        """Compare metrics for two weeks."""
        path1 = self._week_path(pair, timeframe, week1)
        path2 = self._week_path(pair, timeframe, week2)
        with open(path1, "r") as f:
            data1 = yaml.safe_load(f)
        with open(path2, "r") as f:
            data2 = yaml.safe_load(f)

        sharpe1 = data1.get("metadata", {}).get("sharpe", 0)
        sharpe2 = data2.get("metadata", {}).get("sharpe", 0)
        dd1 = data1.get("metadata", {}).get("max_dd", 0)
        dd2 = data2.get("metadata", {}).get("max_dd", 0)

        better = week1 if sharpe1 >= sharpe2 else week2
        return {
            "week1": week1,
            "week2": week2,
            "sharpe_diff": round(sharpe1 - sharpe2, 2),
            "dd_diff": round(dd1 - dd2, 2),
            "better": better,
        }


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Rollback parameters")
    parser.add_argument("--pair", required=True)
    parser.add_argument("--timeframe", required=True)
    parser.add_argument("--week", default=None)
    parser.add_argument("--base-dir", default=".")
    parser.add_argument("--best", action="store_true")
    args = parser.parse_args()

    rollback = ParameterRollback(base_dir=args.base_dir)
    if args.best:
        success = rollback.rollback_to_best(args.pair, args.timeframe)
    else:
        if not args.week:
            raise SystemExit("week is required unless --best is set")
        success = rollback.rollback_to_week(args.pair, args.timeframe, args.week)
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
