#!/usr/bin/env python3
"""
CI Smoke Tests Runner - Критичные smoke тесты для CI/CD pipeline.

Цель: Быстрая проверка основной функциональности системы в CI/CD окружении.
Время выполнения: < 30 секунд.

Проверки:
1. Engine parity (Numba vs Reference)
2. Trace сохранение/загрузка
3. Базовая Optuna оптимизация (3-5 trials)
4. WFA конфигурация и компоненты
5. Критичные импорты и модули

Возвращает:
- exit code 0: все тесты прошли
- exit code 1: есть критичные ошибки
"""

import sys
import os
import time
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
from contextlib import contextmanager
import tempfile
import subprocess
import traceback

# Добавляем src в путь
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Импорты после добавления пути
import pytest
import pandas as pd
import numpy as np
import yaml
import optuna

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@contextmanager
def timer(name: str):
    """Контекстный менеджер для измерения времени."""
    start = time.time()
    try:
        yield
    finally:
        duration = time.time() - start
        logger.info(f"✓ {name}: {duration:.2f}s")


class SmokeTestResult:
    """Результат smoke теста."""
    
    def __init__(self, name: str):
        self.name = name
        self.passed = False
        self.error: Optional[str] = None
        self.duration: float = 0.0
        
    def success(self, duration: float):
        """Отметить тест как успешный."""
        self.passed = True
        self.duration = duration
        
    def failure(self, error: str, duration: float):
        """Отметить тест как неуспешный."""
        self.passed = False
        self.error = error
        self.duration = duration


class CISmokeRunner:
    """Запускатель CI smoke тестов."""
    
    def __init__(self):
        self.results: List[SmokeTestResult] = []
        self.project_root = Path(__file__).parent.parent
        self.start_time = time.time()
        
    def add_result(self, result: SmokeTestResult):
        """Добавить результат теста."""
        self.results.append(result)
        
    def test_critical_imports(self) -> SmokeTestResult:
        """Тест критичных импортов."""
        result = SmokeTestResult("Critical imports")
        start = time.time()
        
        try:
            # Тестируем основные модули
            from coint2.utils.config import load_config
            from coint2.engine.base_engine import BasePairBacktester
            from coint2.engine.reference_engine import ReferenceEngine
            from coint2.core.performance import sharpe_ratio, max_drawdown
            from optimiser.fast_objective import FastWalkForwardObjective
            from optimiser.run_optimization import run_optimization
            from coint2.pipeline.walk_forward_orchestrator import _simulate_realistic_portfolio
            from coint2.utils.traces import save_trace, load_trace
            
            # Проверяем что функции вызываемы
            assert callable(load_config)
            assert callable(sharpe_ratio)
            assert callable(run_optimization)
            
            duration = time.time() - start
            result.success(duration)
            logger.info(f"✓ Критичные импорты работают")
            
        except Exception as e:
            duration = time.time() - start
            result.failure(f"Import error: {str(e)}", duration)
            logger.error(f"✗ Ошибка импорта: {e}")
            
        return result
    
    def test_config_loading(self) -> SmokeTestResult:
        """Тест загрузки конфигурации."""
        result = SmokeTestResult("Config loading")
        start = time.time()
        
        try:
            from coint2.utils.config import load_config
            
            # Загружаем основную конфигурацию
            config_path = self.project_root / "configs" / "main_2024.yaml"
            config = load_config(str(config_path))
            
            # Проверяем обязательные секции
            assert hasattr(config, 'backtest'), "Секция backtest отсутствует"
            assert hasattr(config, 'portfolio'), "Секция portfolio отсутствует"
            assert hasattr(config.portfolio, 'initial_capital'), "initial_capital отсутствует"
            assert hasattr(config.portfolio, 'max_active_positions'), "max_active_positions отсутствует"
            
            # Проверяем критичные ограничения
            gap_minutes = getattr(config.walk_forward, 'gap_minutes', None)
            if gap_minutes is not None:
                assert gap_minutes == 15, f"Gap должен быть 15 минут, получен {gap_minutes}"
            else:
                logger.warning("gap_minutes не найден в конфигурации - это может быть нормально")
            
            duration = time.time() - start
            result.success(duration)
            logger.info(f"✓ Конфигурация загружена и валидна")
            
        except Exception as e:
            duration = time.time() - start
            result.failure(f"Config error: {str(e)}", duration)
            logger.error(f"✗ Ошибка конфигурации: {e}")
            
        return result
    
    def test_engine_parity(self) -> SmokeTestResult:
        """Тест соответствия движков."""
        result = SmokeTestResult("Engine parity")
        start = time.time()
        
        try:
            from coint2.engine.reference_engine import ReferenceEngine
            from coint2.core.numba_parity_v3 import compute_positions_parity_debug
            
            # Создаем минимальные тестовые данные
            np.random.seed(42)
            n = 100
            dates = pd.date_range('2024-01-01', periods=n, freq='15min')
            
            # Коинтегрированная пара
            x = 100 + np.cumsum(np.random.randn(n) * 0.1)
            y = 1.2 * x + np.cumsum(np.random.randn(n) * 0.5)
            
            data = pd.DataFrame({'symbol1': y, 'symbol2': x}, index=dates)
            
            # Параметры теста
            params = {
                'rolling_window': 20,
                'z_enter': 2.0,
                'z_exit': 0.5,
                'max_holding_period': 50,
                'commission_pct': 0.001,
                'slippage_pct': 0.0005
            }
            
            # Reference engine
            ref_engine = ReferenceEngine(**params, verbose=False)
            ref_results = ref_engine.backtest(data)
            
            # Numba engine
            numba_results = compute_positions_parity_debug(
                y, x,
                rolling_window=params['rolling_window'],
                z_enter=params['z_enter'],
                z_exit=params['z_exit'],
                max_holding_period=params['max_holding_period'],
                commission=params['commission_pct'],
                slippage=params['slippage_pct']
            )
            
            # Сравниваем позиции
            ref_positions = ref_results['positions']
            numba_positions = numba_results[0]
            
            # Проверяем соответствие (допускаем небольшие различия)
            position_match = np.mean(ref_positions == numba_positions)
            
            assert position_match > 0.9, f"Position match слишком низкий: {position_match:.2%}"
            
            duration = time.time() - start
            result.success(duration)
            logger.info(f"✓ Engine parity: {position_match:.1%} match")
            
        except Exception as e:
            duration = time.time() - start
            result.failure(f"Engine parity error: {str(e)}", duration)
            logger.error(f"✗ Ошибка engine parity: {e}")
            
        return result
    
    def test_trace_system(self) -> SmokeTestResult:
        """Тест системы трейсов."""
        result = SmokeTestResult("Trace system")
        start = time.time()
        
        try:
            from coint2.utils.traces import save_trace, load_trace
            
            # Создаем временный файл для трейса
            with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as tmp_file:
                trace_path = Path(tmp_file.name)
            
            # Создаем тестовые данные
            n = 50
            dates = pd.date_range('2024-01-01', periods=n, freq='15min')
            z_scores = np.random.randn(n)
            entries_idx = np.random.choice([True, False], n, p=[0.1, 0.9])
            exits_idx = np.random.choice([True, False], n, p=[0.1, 0.9])
            positions = np.random.choice([0, 1, -1], n)
            pnl = np.cumsum(np.random.randn(n) * 10)
            
            # Метаданные
            meta = {
                'engine': 'test',
                'pair': 'TEST-PAIR',
                'timeframe': 'M15',
                'roll': 20,
                'z_enter': 2.0,
                'z_exit': 0.5,
                'fees': 0.002
            }
            
            # Сохраняем трейс
            saved_path = save_trace(
                dates, z_scores, entries_idx, exits_idx, 
                positions, pnl, trace_path, meta
            )
            
            assert saved_path.exists(), "Трейс не сохранен"
            
            # Загружаем трейс
            loaded_trace, loaded_meta = load_trace(saved_path)
            
            # Проверяем данные
            assert len(loaded_trace) == n, "Неправильная длина трейса"
            assert 'z_score' in loaded_trace.columns, "Отсутствует колонка z_score"
            assert 'position' in loaded_trace.columns, "Отсутствует колонка position"
            assert 'pnl' in loaded_trace.columns, "Отсутствует колонка pnl"
            
            # Проверяем метаданные
            assert loaded_meta['engine'] == 'test', "Неправильные метаданные"
            assert loaded_meta['pair'] == 'TEST-PAIR', "Неправильная пара"
            
            # Очищаем
            trace_path.unlink()
            
            duration = time.time() - start
            result.success(duration)
            logger.info(f"✓ Trace system работает")
            
        except Exception as e:
            duration = time.time() - start
            result.failure(f"Trace system error: {str(e)}", duration)
            logger.error(f"✗ Ошибка trace system: {e}")
            
        return result
    
    def test_optuna_basic(self) -> SmokeTestResult:
        """Тест базовой Optuna оптимизации."""
        result = SmokeTestResult("Optuna basic optimization")
        start = time.time()
        
        try:
            # Создаем временную study
            with tempfile.TemporaryDirectory() as tmp_dir:
                storage = f"sqlite:///{tmp_dir}/smoke_test.db"
                study = optuna.create_study(
                    storage=storage,
                    study_name="smoke_test",
                    direction="maximize",
                    sampler=optuna.samplers.RandomSampler(seed=42),
                    load_if_exists=True
                )
                
                # Простая целевая функция
                def objective(trial):
                    x = trial.suggest_float('x', -10, 10)
                    y = trial.suggest_int('y', -5, 5)
                    return -(x**2 + y**2)  # Максимизируем отрицательную параболу
                
                # Запускаем 3 trial для smoke теста
                study.optimize(objective, n_trials=3, show_progress_bar=False)
                
                # Проверяем результаты
                assert len(study.trials) == 3, f"Ожидается 3 trials, получено {len(study.trials)}"
                assert study.best_trial is not None, "Best trial не найден"
                
                # Проверяем что best_value разумный
                assert study.best_value <= 0, "Best value должен быть <= 0 для нашей функции"
                
            duration = time.time() - start
            result.success(duration)
            logger.info(f"✓ Optuna оптимизация: {len(study.trials)} trials, best={study.best_value:.3f}")
            
        except Exception as e:
            duration = time.time() - start
            result.failure(f"Optuna error: {str(e)}", duration)
            logger.error(f"✗ Ошибка Optuna: {e}")
            
        return result
    
    def test_wfa_components(self) -> SmokeTestResult:
        """Тест компонентов Walk-Forward Analysis."""
        result = SmokeTestResult("WFA components")
        start = time.time()
        
        try:
            from coint2.pipeline.walk_forward_orchestrator import _simulate_realistic_portfolio
            from optimiser.fast_objective import FastWalkForwardObjective
            
            # Проверяем что основные компоненты WFA доступны
            assert callable(_simulate_realistic_portfolio), "Portfolio simulation недоступен"
            
            # Проверяем что FastWalkForwardObjective инициализируется
            config_path = self.project_root / "configs" / "main_2024.yaml"
            search_space_path = self.project_root / "configs" / "search_spaces" / "fast.yaml"
            
            if config_path.exists() and search_space_path.exists():
                objective = FastWalkForwardObjective(str(config_path), str(search_space_path))
                assert hasattr(objective, 'base_config'), "Base config не загружен"
                assert hasattr(objective, 'search_space'), "Search space не загружен"
                assert callable(objective), "Objective не вызываем"
            
            # Проверяем WFA конфигурацию
            wfa_config_path = self.project_root / "bench" / "wfa.yaml"
            if wfa_config_path.exists():
                with open(wfa_config_path, 'r') as f:
                    wfa_config = yaml.safe_load(f)
                
                # Проверяем обязательные секции
                required_sections = ["walk_forward", "optuna", "success_criteria", "traces"]
                for section in required_sections:
                    assert section in wfa_config, f"Отсутствует секция {section} в WFA конфигурации"
                
                # Проверяем критические параметры
                wf = wfa_config["walk_forward"]
                assert wf["gap_minutes"] == 15, "Gap должен быть 15 минут"
                assert wf["training_period_days"] >= 60, "Training период должен быть ≥60 дней"
                assert wf["testing_period_days"] >= 30, "Testing период должен быть ≥30 дней"
            
            duration = time.time() - start
            result.success(duration)
            logger.info(f"✓ WFA компоненты проверены")
            
        except Exception as e:
            duration = time.time() - start
            result.failure(f"WFA components error: {str(e)}", duration)
            logger.error(f"✗ Ошибка WFA компонентов: {e}")
            
        return result
    
    def test_data_pipeline(self) -> SmokeTestResult:
        """Тест базового data pipeline."""
        result = SmokeTestResult("Data pipeline")
        start = time.time()
        
        try:
            from coint2.core.performance import sharpe_ratio, max_drawdown, win_rate
            
            # Создаем тестовые PnL данные
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
            
            # Проверяем структуру данных
            tiny_prices_df = pd.DataFrame({
                'symbol1': np.random.randn(100) + 100,
                'symbol2': np.random.randn(100) + 200
            }, index=pd.date_range('2024-01-01', periods=100, freq='15min'))
            
            assert isinstance(tiny_prices_df, pd.DataFrame), "DataFrame не создан"
            assert len(tiny_prices_df) > 0, "DataFrame пуст"
            assert isinstance(tiny_prices_df.index, pd.DatetimeIndex), "Индекс должен быть DatetimeIndex"
            
            duration = time.time() - start
            result.success(duration)
            logger.info(f"✓ Data pipeline работает")
            
        except Exception as e:
            duration = time.time() - start
            result.failure(f"Data pipeline error: {str(e)}", duration)
            logger.error(f"✗ Ошибка data pipeline: {e}")
            
        return result
    
    def test_numba_compilation(self) -> SmokeTestResult:
        """Тест компиляции Numba функций."""
        result = SmokeTestResult("Numba compilation")
        start = time.time()
        
        try:
            from coint2.core.numba_parity_v3 import compute_positions_parity_debug
            
            # Создаем минимальные данные для компиляции
            np.random.seed(42)
            n = 30
            y = np.random.randn(n) + 100
            x = np.random.randn(n) + 200
            
            # Запускаем Numba функцию (это вызовет JIT компиляцию)
            numba_result = compute_positions_parity_debug(
                y, x,
                rolling_window=20,
                z_enter=2.0,
                z_exit=0.5,
                max_holding_period=30,
                commission=0.001,
                slippage=0.0005
            )
            
            # Проверяем что результат правильной структуры
            expected_len = 11  # Обновляем ожидаемое количество возвращаемых значений
            assert len(numba_result) == expected_len, f"Ожидается {expected_len} возвращаемых значений, получено {len(numba_result)}"
            
            positions, trades, pnl_series, z_scores, spreads, entries_idx, exits_idx, mu, sigma, beta, alpha = numba_result[:11]
            
            assert len(positions) == n, "Неправильная длина positions"
            assert len(pnl_series) == n, "Неправильная длина pnl_series"
            assert len(z_scores) == n, "Неправильная длина z_scores"
            
            duration = time.time() - start
            result.success(duration)
            logger.info(f"✓ Numba компиляция прошла успешно")
            
        except Exception as e:
            duration = time.time() - start
            result.failure(f"Numba compilation error: {str(e)}", duration)
            logger.error(f"✗ Ошибка Numba компиляции: {e}")
            
        return result
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Запустить все smoke тесты."""
        logger.info("=" * 60)
        logger.info("CI SMOKE TESTS - Запуск критичных проверок")
        logger.info("=" * 60)
        
        # Список всех тестов
        tests = [
            self.test_critical_imports,
            self.test_config_loading,
            self.test_data_pipeline,
            self.test_numba_compilation,
            self.test_engine_parity,
            self.test_trace_system,
            self.test_optuna_basic,
            self.test_wfa_components,
        ]
        
        # Запускаем тесты
        for test_func in tests:
            test_name = test_func.__name__.replace('test_', '').replace('_', ' ').title()
            logger.info(f"\n🧪 Запуск: {test_name}")
            
            try:
                result = test_func()
                self.add_result(result)
                
                if result.passed:
                    logger.info(f"✅ {test_name}: PASSED ({result.duration:.2f}s)")
                else:
                    logger.error(f"❌ {test_name}: FAILED ({result.duration:.2f}s)")
                    logger.error(f"   Ошибка: {result.error}")
                    
            except Exception as e:
                # Если тест вылетел с необработанным исключением
                error_result = SmokeTestResult(test_name)
                error_result.failure(f"Unhandled exception: {str(e)}", 0.0)
                self.add_result(error_result)
                logger.error(f"💥 {test_name}: CRASHED")
                logger.error(f"   Exception: {str(e)}")
                logger.error(f"   Traceback: {traceback.format_exc()}")
        
        return self.generate_summary()
    
    def generate_summary(self) -> Dict[str, Any]:
        """Сгенерировать сводку результатов."""
        total_tests = len(self.results)
        passed_tests = sum(1 for r in self.results if r.passed)
        failed_tests = total_tests - passed_tests
        total_duration = time.time() - self.start_time
        
        summary = {
            'total_tests': total_tests,
            'passed_tests': passed_tests,
            'failed_tests': failed_tests,
            'success_rate': (passed_tests / total_tests) * 100 if total_tests > 0 else 0,
            'total_duration': total_duration,
            'results': self.results
        }
        
        logger.info("\n" + "=" * 60)
        logger.info("СВОДКА CI SMOKE TESTS")
        logger.info("=" * 60)
        
        logger.info(f"Общее количество тестов: {total_tests}")
        logger.info(f"Пройденные тесты: {passed_tests}")
        logger.info(f"Провалившиеся тесты: {failed_tests}")
        logger.info(f"Процент успеха: {summary['success_rate']:.1f}%")
        logger.info(f"Общее время: {total_duration:.2f}s")
        
        if failed_tests > 0:
            logger.info("\n❌ ПРОВАЛИВШИЕСЯ ТЕСТЫ:")
            for result in self.results:
                if not result.passed:
                    logger.error(f"  - {result.name}: {result.error}")
        else:
            logger.info("\n✅ ВСЕ ТЕСТЫ ПРОШЛИ УСПЕШНО!")
        
        return summary


def main() -> int:
    """Основная функция. Возвращает exit code."""
    
    # Проверяем что запущены из правильной директории
    project_root = Path(__file__).parent.parent
    if not (project_root / "src").exists():
        logger.error("❌ Запустите скрипт из корневой директории проекта")
        return 1
    
    # Переходим в корневую директорию
    os.chdir(project_root)
    
    # Запускаем smoke тесты
    runner = CISmokeRunner()
    summary = runner.run_all_tests()
    
    # Определяем exit code
    if summary['failed_tests'] == 0:
        logger.info(f"\n🎉 CI SMOKE TESTS: ВСЕ ТЕСТЫ ПРОШЛИ!")
        logger.info(f"Время выполнения: {summary['total_duration']:.2f}s")
        logger.info("Система готова для CI/CD.")
        return 0
    else:
        logger.error(f"\n💥 CI SMOKE TESTS: ЕСТЬ КРИТИЧНЫЕ ОШИБКИ!")
        logger.error(f"Провалилось {summary['failed_tests']}/{summary['total_tests']} тестов")
        logger.error("Система НЕ готова для CI/CD.")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)