#!/usr/bin/env python3
"""
Streamlit UI preflight checks.
"""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path
from typing import Iterable

try:
    import yaml  # type: ignore
    _YAML_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - optional dependency
    yaml = None
    _YAML_IMPORT_ERROR = str(exc)


def _check_modules(modules: Iterable[str]) -> list[str]:
    errors = []
    for module in modules:
        try:
            importlib.import_module(module)
        except Exception as exc:  # pragma: no cover - defensive
            errors.append(f"Модуль '{module}' не импортируется: {exc}")
    return errors


def _load_yaml(path: Path, label: str) -> tuple[dict, list[str]]:
    if yaml is None:
        return {}, [f"PyYAML не установлен: {_YAML_IMPORT_ERROR}"]
    if not path.exists():
        return {}, [f"{label} не найден: {path}"]
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
        if not isinstance(data, dict):
            return {}, [f"{label} должен быть словарем: {path}"]
        return data, []
    except Exception as exc:  # pragma: no cover - defensive
        return {}, [f"Не удалось прочитать {label} ({path}): {exc}"]


def _ensure_dirs(paths: Iterable[Path]) -> list[str]:
    errors = []
    for path in paths:
        try:
            path.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            errors.append(f"Не удалось создать директорию {path}: {exc}")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description="UI preflight checks")
    parser.add_argument("--config", default="configs/main_2024.yaml", help="Path to base config")
    parser.add_argument(
        "--search-space",
        default="configs/search_spaces/web_ui.yaml",
        help="Path to UI search space config",
    )
    parser.add_argument(
        "--data-root",
        default=None,
        help="Optional data root to validate (e.g. data_downloaded)",
    )
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="Report missing items as warnings instead of errors",
    )
    args = parser.parse_args()

    errors: list[str] = []
    warnings: list[str] = []

    module_errors = _check_modules(["streamlit", "optuna", "plotly", "pandas", "numpy"])
    errors.extend(module_errors)

    cfg_path = Path(args.config)
    config, cfg_errors = _load_yaml(cfg_path, "Базовый конфиг")
    errors.extend(cfg_errors)

    search_space_path = Path(args.search_space)
    search_space, search_errors = _load_yaml(search_space_path, "Search space")
    errors.extend(search_errors)

    required_config_keys = {"data_dir", "results_dir", "portfolio", "pair_selection", "backtest", "walk_forward"}
    missing_cfg = required_config_keys - set(config.keys())
    if missing_cfg:
        errors.append(f"В конфиге отсутствуют ключи: {', '.join(sorted(missing_cfg))}")

    required_space_keys = {"signals", "risk_management", "portfolio", "filters", "costs"}
    missing_space = required_space_keys - set(search_space.keys())
    if missing_space:
        errors.append(f"В search space отсутствуют секции: {', '.join(sorted(missing_space))}")

    errors.extend(
        _ensure_dirs(
            [
                Path("outputs/studies"),
                Path("outputs/optimization"),
                Path("outputs/validation"),
            ]
        )
    )

    if args.data_root:
        data_root = Path(args.data_root)
        if not data_root.exists():
            errors.append(f"Data root не найден: {data_root}")
        else:
            symbols = [p for p in data_root.iterdir() if p.is_dir() and not p.name.startswith(".")]
            if not symbols:
                warnings.append(f"В data root нет каталогов с символами: {data_root}")

    if errors and args.allow_missing:
        warnings.extend(errors)
        errors = []

    if errors:
        print("UI preflight завершен с ошибками:")
        for item in errors:
            print(f" - {item}")
        if warnings:
            print("Предупреждения:")
            for item in warnings:
                print(f" - {item}")
        return 1

    print("UI preflight завершен успешно.")
    if warnings:
        print("Предупреждения:")
        for item in warnings:
            print(f" - {item}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
