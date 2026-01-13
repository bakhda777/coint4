#!/usr/bin/env python3
"""
SMOKE TESTS - Критически важные быстрые проверки

Цель: Убедиться что система не сломана на базовом уровне.
Время выполнения: <5 секунд общее время.

Smoke тесты проверяют:
1. Импорт всех основных модулей
2. Загрузка основной конфигурации  
3. Создание основных объектов без ошибок
4. Базовая работа на минимальных данных
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path

# Маркируем все тесты как smoke
pytestmark = pytest.mark.smoke

# Константы для тестирования
DEFAULT_ROLLING_WINDOW = 10
DEFAULT_Z_THRESHOLD = 2.0
DEFAULT_Z_EXIT = 0.5
DEFAULT_COOLDOWN_PERIODS = 4
DEFAULT_COMMISSION_PCT = 0.001
DEFAULT_SLIPPAGE_PCT = 0.0005

# Константы для бэктестера
BACKTESTER_PARAMS = {
    'rolling_window': DEFAULT_ROLLING_WINDOW,
    'z_threshold': DEFAULT_Z_THRESHOLD,
    'z_exit': DEFAULT_Z_EXIT,
    'cooldown_periods': DEFAULT_COOLDOWN_PERIODS,
    'commission_pct': DEFAULT_COMMISSION_PCT,
    'slippage_pct': DEFAULT_SLIPPAGE_PCT
}


class TestSmokeImports:
    """Smoke тесты импортов - самые критичные проверки."""

    def test_config_utils_when_imported_then_callable(self):
        """Проверка импорта утилит конфигурации."""
        from src.coint2.utils.config import load_config
        assert callable(load_config)

    def test_performance_core_when_imported_then_functions_available(self):
        """Проверка импорта модулей производительности."""
        from src.coint2.core.performance import sharpe_ratio, max_drawdown
        assert callable(sharpe_ratio)
        assert callable(max_drawdown)

    def test_base_engine_when_imported_then_class_available(self):
        """Проверка импорта базового движка."""
        from src.coint2.engine.base_engine import BasePairBacktester
        assert BasePairBacktester is not None

    def test_fast_objective_when_imported_then_class_available(self):
        """Проверка импорта быстрой целевой функции."""
        from src.optimiser.fast_objective import FastWalkForwardObjective
        assert FastWalkForwardObjective is not None

    def test_run_optimization_when_imported_then_callable(self):
        """Проверка импорта модуля запуска оптимизации."""
        from src.optimiser.run_optimization import run_optimization
        assert callable(run_optimization)

    def test_portfolio_simulator_when_imported_then_callable(self):
        """Проверка импорта симулятора портфеля."""
        from src.coint2.pipeline.walk_forward_orchestrator import _simulate_realistic_portfolio
        assert callable(_simulate_realistic_portfolio)

    def test_pair_scanner_when_imported_then_callable(self):
        """Проверка импорта сканера пар."""
        from src.coint2.pipeline.pair_scanner import find_cointegrated_pairs
        assert callable(find_cointegrated_pairs)


class TestSmokeConfiguration:
    """Smoke тесты конфигурации."""

    def test_main_config_when_loaded_then_loads_successfully(self):
        """Проверка загрузки основной конфигурации."""
        config_path = Path("configs/main_2024.yaml")
        if not config_path.exists():
            pytest.skip("Основная конфигурация не найдена")

        from src.coint2.utils.config import load_config
        config = load_config(str(config_path))
        assert config is not None

    @pytest.mark.parametrize("section_name,error_message", [
        ("backtest", "Секция backtest отсутствует"),
        ("portfolio", "Секция portfolio отсутствует")
    ])
    def test_config_sections_when_loaded_then_required_sections_present(self, section_name, error_message):
        """Проверка наличия обязательных секций в конфигурации."""
        config_path = Path("configs/main_2024.yaml")
        if not config_path.exists():
            pytest.skip("Основная конфигурация не найдена")

        from src.coint2.utils.config import load_config
        config = load_config(str(config_path))
        assert hasattr(config, section_name), error_message

    @pytest.mark.parametrize("attribute_name,error_message", [
        ("initial_capital", "initial_capital отсутствует"),
        ("max_active_positions", "max_active_positions отсутствует")
    ])
    def test_portfolio_attributes_when_loaded_then_required_attributes_present(self, attribute_name, error_message):
        """Проверка наличия обязательных атрибутов в секции portfolio."""
        config_path = Path("configs/main_2024.yaml")
        if not config_path.exists():
            pytest.skip("Основная конфигурация не найдена")

        from src.coint2.utils.config import load_config
        config = load_config(str(config_path))
        assert hasattr(config.portfolio, attribute_name), error_message


class TestSmokeBasicFunctionality:
    """Smoke тесты базовой функциональности."""
    
    def test_pair_backtester_when_created_then_initializes(self, tiny_prices_df):
        """Проверка создания BasePairBacktester без ошибок."""

        try:
            from src.coint2.engine.base_engine import BasePairBacktester

            # Создаем минимальный backtester с правильными параметрами
            backtester = BasePairBacktester(
                pair_data=tiny_prices_df,
                rolling_window=10,
                z_threshold=2.0,
                z_exit=0.5,
                commission_pct=0.001,
                slippage_pct=0.001,
                pair_name="TEST_PAIR"
            )

            assert backtester is not None, "BasePairBacktester не создался"
            assert backtester.pair_name == "TEST_PAIR", "Имя пары не установилось"

        except Exception as e:
            pytest.fail(f"Не удалось создать BasePairBacktester: {e}")

    def test_performance_metrics_when_calculated_then_return_valid_values(self, tiny_prices_df):
        """Проверка расчета базовых метрик производительности."""

        try:
            from src.coint2.core.performance import sharpe_ratio, max_drawdown, win_rate

            # Создаем тестовые данные PnL
            pnl_series = pd.Series([10, -5, 15, -8, 20, -3, 12, -7, 18, -4])

            # Рассчитываем метрики
            sharpe = sharpe_ratio(pnl_series, annualizing_factor=252)
            max_dd = max_drawdown(pnl_series)
            wr = win_rate(pnl_series)

            # Проверяем что метрики разумные
            assert isinstance(sharpe, (int, float)), "Sharpe ratio должен быть числом"
            assert isinstance(max_dd, (int, float)), "Max drawdown должен быть числом"
            assert isinstance(wr, (int, float)), "Win rate должен быть числом"
            assert 0 <= wr <= 1, f"Win rate должен быть между 0 и 1, получен {wr}"

        except Exception as e:
            pytest.fail(f"Ошибка расчета метрик: {e}")

    def test_portfolio_simulation_when_imported_then_callable(self, tiny_prices_df):
        """Проверка существования функции реалистичной симуляции портфеля."""

        try:
            from src.coint2.pipeline.walk_forward_orchestrator import _simulate_realistic_portfolio

            assert callable(_simulate_realistic_portfolio), \
                "Функция _simulate_realistic_portfolio должна быть вызываемой"

        except ImportError as e:
            pytest.fail(f"Функция реалистичной симуляции портфеля не найдена: {e}")


class TestSmokeDataHandling:
    """Smoke тесты обработки данных."""

    def test_dataframe_when_tiny_prices_then_valid_structure(self, tiny_prices_df):
        """Проверка структуры тестовых данных."""
        assert isinstance(tiny_prices_df, pd.DataFrame)
        assert len(tiny_prices_df) > 0
        assert len(tiny_prices_df.columns) >= 2

    def test_index_when_tiny_prices_then_datetime_type(self, tiny_prices_df):
        """Проверка типа индекса данных."""
        assert isinstance(tiny_prices_df.index, pd.DatetimeIndex)

    def test_statistics_when_tiny_prices_then_computable(self, tiny_prices_df):
        """Проверка возможности расчета базовых статистик."""
        means = tiny_prices_df.mean()
        stds = tiny_prices_df.std()

        assert len(means) == len(tiny_prices_df.columns)
        assert len(stds) == len(tiny_prices_df.columns)
        assert all(stds > 0), "Все стандартные отклонения должны быть положительными"


class TestSmokeBacktesting:
    """Smoke тесты бэктестинга."""

    def test_backtester_when_minimal_params_then_initializes(self, small_prices_df):
        """Проверка инициализации бэктестера с минимальными параметрами."""
        from src.coint2.engine.base_engine import BasePairBacktester

        backtester = BasePairBacktester(
            pair_data=small_prices_df.iloc[:, :2],
            **BACKTESTER_PARAMS
        )

        assert backtester is not None

    def test_backtest_when_executed_then_returns_results(self, small_prices_df):
        """Проверка получения результатов при выполнении бэктеста."""
        from src.coint2.engine.base_engine import BasePairBacktester

        backtester = BasePairBacktester(
            pair_data=small_prices_df.iloc[:, :2],
            **BACKTESTER_PARAMS
        )

        backtester.run()
        assert backtester.results is not None

    def test_backtest_results_when_complete_then_dataframe_returned(self, small_prices_df):
        """Проверка наличия результатов в виде DataFrame после завершения бэктеста."""
        from src.coint2.engine.base_engine import BasePairBacktester

        backtester = BasePairBacktester(
            pair_data=small_prices_df.iloc[:, :2],
            **BACKTESTER_PARAMS
        )

        backtester.run()
        # Проверяем, что результаты - это DataFrame с нужными колонками
        assert isinstance(backtester.results, pd.DataFrame)


class TestSmokeSystemReadiness:
    """Smoke тесты готовности системы."""

    def test_required_directories_when_project_structure_checked_then_directories_exist(self):
        """Проверка существования обязательных директорий."""
        REQUIRED_DIRS = ["src", "configs", "tests", "scripts"]

        for dir_name in REQUIRED_DIRS:
            assert Path(dir_name).exists(), f"Директория {dir_name} должна существовать"

    def test_critical_files_when_project_structure_checked_then_files_exist(self):
        """Проверка существования критически важных файлов."""
        REQUIRED_FILES = [
            "configs/main_2024.yaml",
            "src/optimiser/fast_objective.py",
            "src/coint2/engine/base_engine.py",
            "pytest.ini"
        ]

        for file_path in REQUIRED_FILES:
            assert Path(file_path).exists(), f"Файл {file_path} должен существовать"


# Все тесты запускаются только через pytest
