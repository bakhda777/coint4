#!/usr/bin/env python3
"""
Quick validation for parquet data dumps.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd

try:
    import yaml  # type: ignore
    _YAML_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - optional dependency
    yaml = None
    _YAML_IMPORT_ERROR = str(exc)


EXPECTED_COLUMNS = {"timestamp", "symbol", "open", "high", "low", "close", "volume", "ts_ms"}


def _load_yaml(path: Optional[str]) -> dict:
    if not path:
        return {}
    if yaml is None:
        raise RuntimeError(f"PyYAML не установлен: {_YAML_IMPORT_ERROR}")
    cfg_path = Path(path)
    with cfg_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _iter_symbol_dirs(data_root: Path, symbols: Optional[list[str]]) -> list[Path]:
    if symbols:
        result = []
        for sym in symbols:
            sym_path = data_root / sym
            if sym_path.exists():
                result.append(sym_path)
        return result
    return [p for p in sorted(data_root.iterdir()) if p.is_dir() and not p.name.startswith(".")]


def _iter_day_dirs(symbol_dir: Path) -> Iterable[Path]:
    for year_dir in sorted(symbol_dir.glob("year=*")):
        for month_dir in sorted(year_dir.glob("month=*")):
            for day_dir in sorted(month_dir.glob("day=*")):
                yield day_dir


def _read_day(day_dir: Path, mode: str) -> Optional[pd.DataFrame]:
    if mode == "clean":
        data_file = day_dir / "data.parquet"
        if not data_file.exists():
            return None
        return pd.read_parquet(data_file)
    if not day_dir.exists():
        return None
    return pd.read_parquet(day_dir)


def _calculate_gaps(timestamps: pd.Series, bar_minutes: int) -> tuple[int, int]:
    if timestamps.empty:
        return 0, 0
    sorted_ts = pd.to_datetime(timestamps.drop_duplicates().sort_values(), unit="ms")
    deltas = sorted_ts.diff().dropna()
    expected_delta = pd.Timedelta(minutes=bar_minutes)
    gap_bars = (deltas / expected_delta) - 1
    gap_bars = gap_bars[gap_bars > 0]
    if gap_bars.empty:
        return 0, 0
    gap_bars_int = gap_bars.astype(int)
    return int(gap_bars_int.sum()), int(gap_bars_int.max())


def _expected_bars_per_day(bar_minutes: int) -> int:
    return int(24 * 60 / bar_minutes)


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate parquet data dumps")
    parser.add_argument("--data-root", default="data_downloaded", help="Data root directory")
    parser.add_argument(
        "--mode",
        choices=["raw", "clean"],
        default="raw",
        help="Data layout: raw (many files per day) or clean (data.parquet per day)",
    )
    parser.add_argument("--symbols", help="Comma-separated symbols to validate")
    parser.add_argument("--max-days", type=int, default=3, help="Max days per symbol to inspect")
    parser.add_argument("--config", help="Optional config to derive thresholds")
    parser.add_argument("--bar-minutes", type=int, help="Override bar interval (minutes)")
    parser.add_argument("--report", help="Optional path to save CSV report")
    args = parser.parse_args()

    cfg = _load_yaml(args.config) if args.config else {}
    data_cfg = cfg.get("data_processing", {}) if isinstance(cfg, dict) else {}
    backtest_cfg = cfg.get("backtest", {}) if isinstance(cfg, dict) else {}
    pair_cfg = cfg.get("pair_selection", {}) if isinstance(cfg, dict) else {}

    bar_minutes = args.bar_minutes or pair_cfg.get("bar_minutes") or 15
    min_history_ratio = data_cfg.get("min_history_ratio", 0.8)
    fill_limit_pct = backtest_cfg.get("fill_limit_pct", 0.1)

    expected_bars = _expected_bars_per_day(bar_minutes)
    fill_limit = min(5, max(1, int(expected_bars * float(fill_limit_pct))))

    data_root = Path(args.data_root)
    if not data_root.exists():
        print(f"Data root не найден: {data_root}")
        return 1

    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()] if args.symbols else None
    symbol_dirs = _iter_symbol_dirs(data_root, symbols)
    if not symbol_dirs:
        print(f"Нет символов для проверки в {data_root}")
        return 1

    rows = []
    errors = 0

    for symbol_dir in symbol_dirs:
        day_count = 0
        for day_dir in _iter_day_dirs(symbol_dir):
            if day_count >= args.max_days:
                break
            df = _read_day(day_dir, args.mode)
            if df is None or df.empty:
                rows.append(
                    {
                        "symbol": symbol_dir.name,
                        "day": day_dir.name,
                        "status": "missing",
                        "note": "нет данных",
                    }
                )
                errors += 1
                day_count += 1
                continue

            missing_cols = EXPECTED_COLUMNS - set(df.columns)
            if missing_cols:
                rows.append(
                    {
                        "symbol": symbol_dir.name,
                        "day": day_dir.name,
                        "status": "invalid",
                        "note": f"нет колонок: {', '.join(sorted(missing_cols))}",
                    }
                )
                errors += 1
                day_count += 1
                continue

            unique_ts = df["timestamp"].nunique()
            duplicates = int(df["timestamp"].duplicated().sum())
            missing_bars, max_gap_bars = _calculate_gaps(df["timestamp"], bar_minutes)
            missing_ratio = 0.0
            if expected_bars:
                missing_ratio = max(0.0, (expected_bars - unique_ts) / expected_bars)

            status = "ok"
            notes = []
            if duplicates and args.mode == "clean":
                status = "warn"
                notes.append(f"дубликаты timestamp: {duplicates}")
            if missing_ratio > (1 - min_history_ratio):
                status = "warn"
                notes.append(f"мало данных: {missing_ratio:.1%} пропусков")
            if max_gap_bars > fill_limit:
                status = "warn"
                notes.append(f"gap {max_gap_bars} баров > лимита {fill_limit}")
            if df["symbol"].nunique() > 1:
                status = "warn"
                notes.append("несколько символов в одном дне")

            rows.append(
                {
                    "symbol": symbol_dir.name,
                    "day": day_dir.name,
                    "status": status,
                    "rows": len(df),
                    "unique_ts": unique_ts,
                    "duplicates": duplicates,
                    "missing_ratio": round(missing_ratio, 4),
                    "missing_bars": missing_bars,
                    "max_gap_bars": max_gap_bars,
                    "note": "; ".join(notes),
                }
            )
            day_count += 1

    report_df = pd.DataFrame(rows)
    if not report_df.empty:
        print(report_df.to_string(index=False))
    else:
        print("Нет данных для отчета.")

    if args.report:
        report_path = Path(args.report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_df.to_csv(report_path, index=False)
        print(f"Отчет сохранен: {report_path}")

    warn_count = int((report_df["status"] == "warn").sum()) if not report_df.empty else 0
    print(f"Итог: ошибок={errors}, предупреждений={warn_count}")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
