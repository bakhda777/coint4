import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

# Ensure the src directory is on the Python path for tests
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# Ограничиваем потоки для Numba/BLAS при параллельном запуске тестов
os.environ.setdefault("NUMBA_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")


@pytest.fixture(autouse=True)
def ensure_determinism():
    """Автоматически фиксирует seed для всех тестов."""
    # Используем современный подход с default_rng
    rng = np.random.default_rng(42)
    # Устанавливаем глобальный seed для совместимости с legacy кодом
    np.random.seed(42)
    yield
    # Cleanup если нужен


@pytest.fixture(scope="session")
def rng():
    """Генератор случайных чисел с фиксированным seed."""
    return np.random.default_rng(42)


def generate_test_data(rng, n=100, n_symbols=3):
    """Генерирует синтетические данные для тестов.

    Параметры
    ----------
    rng : np.random.Generator
        Генератор случайных чисел с фиксированным seed.
    n : int, optional
        Количество временных точек (по умолчанию 100).
    n_symbols : int, optional
        Количество символов (по умолчанию 3).

    Возвращает
    -------
    pd.DataFrame
        DataFrame с синтетическими ценами.
    """
    dates = pd.date_range("2024-01-01", periods=n, freq="15min")
    data = {}
    symbols = [f"SYM{i}" for i in range(n_symbols)]

    for i, symbol in enumerate(symbols):
        base = rng.standard_normal(n).cumsum()
        price = 100 * np.exp(base * 0.01 * (1 + i * 0.1))  # Разные волатильности
        data[symbol] = price

    return pd.DataFrame(data, index=dates)


@pytest.fixture(scope="session")
def tiny_prices_df(rng):
    """Минимальные данные для smoke тестов (20 строк, 3 символа)."""
    return generate_test_data(rng, n=20, n_symbols=3)


@pytest.fixture(scope="session")
def small_prices_df(rng):
    """Малые данные для быстрых тестов (100 строк, 5 символов)."""
    return generate_test_data(rng, n=100, n_symbols=5)


@pytest.fixture(scope="session")
def medium_prices_df(rng):
    """Средние данные для интеграционных тестов (500 строк, 10 символов)."""
    return generate_test_data(rng, n=500, n_symbols=10)


@pytest.fixture(scope="session")
def large_prices_df(rng):
    """Большие данные для полных тестов (2000 строк, 20 символов)."""
    return generate_test_data(rng, n=2000, n_symbols=20)


@pytest.fixture(scope="session")
def mock_config():
    """Легковесная конфигурация для юнит‑тестов."""
    from unittest.mock import Mock

    config = Mock()
    config.data = Mock()
    config.data.symbols = ['AAPL', 'MSFT', 'GOOGL']
    config.backtest = Mock()
    config.backtest.initial_capital = 100000
    config.backtest.commission_pct = 0.001
    config.backtest.slippage_pct = 0.001
    config.portfolio = Mock()
    config.portfolio.max_active_positions = 3
    config.portfolio.risk_per_position_pct = 0.02
    config.walk_forward = Mock()
    config.walk_forward.testing_period_days = 7

    return config


@pytest.fixture
@pytest.mark.serial
def fast_study(tmp_path):
    """Быстрая Optuna study для тестов."""
    try:
        import optuna
        storage = f"sqlite:///{tmp_path/'study.db'}"  # локально и изолированно
        return optuna.create_study(
            storage=storage,
            load_if_exists=True,
            sampler=optuna.samplers.RandomSampler(seed=0),
            direction="maximize",
        )
    except ImportError:
        pytest.skip("Optuna not available")


@pytest.fixture(scope="session", autouse=True)
def _setup_global_cache():
    """Инициализация и очистка глобального кэша."""
    try:
        from coint2.core.global_rolling_cache import (
            initialize_global_rolling_cache,
            cleanup_global_rolling_cache,
        )
        # Минимальная конфигурация для тестов
        test_config = {
            'rolling_window': 30,
            'volatility_lookback': 96,
            'correlation_window': 720,
            'hurst_window': 720,
            'variance_ratio_window': 480,
        }
        initialize_global_rolling_cache(test_config)
        yield
        cleanup_global_rolling_cache()
    except (ImportError, Exception):
        # Если модуль не найден или ошибка инициализации, просто пропускаем
        yield


@pytest.fixture
def minimal_config():
    """Минимальная конфигурация для быстрых тестов."""
    return {
        "signals": {
            "zscore_threshold": 2.0,
            "zscore_exit": 0.5,
            "rolling_window": 20,
        },
        "portfolio": {
            "max_active_positions": 5,
            "risk_per_position_pct": 0.01,
            "max_position_size_pct": 0.1,
        },
        "costs": {
            "commission_pct": 0.001,
            "slippage_pct": 0.0005,
        },
        "normalization": {
            "normalization_method": "minmax",
            "min_history_ratio": 0.6,
        },
    }


def pytest_collection_modifyitems(config, items):
    """Автоматически маркирует тесты на основе имен и содержимого."""
    for item in items:
        # Автоматически маркируем smoke тесты (ПРИОРИТЕТ!)
        if (
            'smoke' in item.nodeid.lower()
            or 'TestSmoke' in item.nodeid
            or (item.cls and 'Smoke' in item.cls.__name__)
        ):
            item.add_marker(pytest.mark.smoke)

        # Автоматически маркируем медленные тесты
        if any(
            keyword in item.nodeid.lower()
            for keyword in [
                'integration',
                'performance',
                'full_',
                'comprehensive',
                'large_',
                'complete_',
                'optimization',
                'walk_forward',
                'global_cache',
            ]
        ):
            item.add_marker(pytest.mark.slow)

        # Маркируем serial тесты (нельзя параллелить)
        if any(
            keyword in item.nodeid.lower()
            for keyword in [
                'cache',
                'sqlite',
                'threading',
                'parallel',
                'concurrent',
                'global_',
                'thread_safe',
                'serial',
            ]
        ):
            item.add_marker(pytest.mark.serial)

        # Integration тесты (с реальными данными)
        if any(
            keyword in item.nodeid.lower()
            for keyword in ['integration', 'comprehensive', 'full_', 'complete_']
        ):
            item.add_marker(pytest.mark.integration)

        # Performance тесты (бенчмарки)
        if any(
            keyword in item.nodeid.lower()
            for keyword in ['performance', 'benchmark', 'speed', 'timing']
        ):
            item.add_marker(pytest.mark.performance)

        # Unit тесты (изолированные с моками)
        if any(
            keyword in item.nodeid.lower()
            for keyword in ['mock', 'unit', 'test_unit']
        ) or 'mock' in str(item.function):
            item.add_marker(pytest.mark.unit)

        # Critical fixes тесты (ВАЖНО для lookahead bias!)
        if any(
            keyword in item.nodeid.lower()
            for keyword in ['critical', 'fix', 'task', 'audit', 'lookahead', 'bias']
        ):
            item.add_marker(pytest.mark.critical_fixes)

        # Deprecated тесты
        if any(
            keyword in item.nodeid.lower()
            for keyword in ['deprecated', 'old_', 'legacy', 'obsolete']
        ):
            item.add_marker(pytest.mark.deprecated)

        # Fast тесты (по умолчанию, если не медленные и не интеграционные)
        existing_markers = {marker.name for marker in item.iter_markers()}
        if not any(
            marker in existing_markers
            for marker in ['slow', 'integration', 'performance', 'serial']
        ):
            item.add_marker(pytest.mark.fast)


def pytest_sessionfinish(session, exitstatus):
    """Hook для вывода статуса завершения pytest сессии."""
    print(f"PYTEST_SESSION_FINISHED_EXIT_STATUS={exitstatus}")
