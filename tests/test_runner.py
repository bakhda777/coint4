"""Модуль для запуска всех тестов и генерации отчетов.

Этот модуль объединяет все созданные тесты:
- Тесты робастности (test_backtest_robustness.py)
- Синтетические тесты (test_synthetic_scenarios.py)
- Генерацию детализированного отчета
"""

import sys
import os
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime
import unittest
import warnings

# Добавляем путь к проекту
sys.path.append(str(Path(__file__).parent.parent))

from src.coint2.utils.config import load_config
from src.coint2.engine.base_engine import BasePairBacktester
from src.coint2.reporting.detailed_report import DetailedReportGenerator
try:
    from tests.test_backtest_robustness import BacktestRobustnessTests
    from tests.test_synthetic_scenarios import SyntheticScenariosTests
except ImportError as e:
    print(f"Warning: Could not import test classes: {e}")
    BacktestRobustnessTests = None
    SyntheticScenariosTests = None


class ComprehensiveTestRunner:
    """Комплексный запуск всех тестов и генерация отчетов."""
    
    def __init__(self, config_path: str = "config.yaml"):
        """Инициализация тест-раннера.
        
        Args:
            config_path: Путь к файлу конфигурации
        """
        self.config = load_config(config_path)
        self.logger = self._setup_logging()
        
        # Подавляем предупреждения для чистого вывода
        warnings.filterwarnings('ignore')
    
    def _setup_logging(self) -> logging.Logger:
        """Настройка логирования."""
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def generate_synthetic_data(self, 
                              n_samples: int = 10000,
                              correlation: float = 0.8,
                              noise_ratio: float = 0.1) -> pd.DataFrame:
        """Генерирует синтетические данные для тестирования.
        
        Args:
            n_samples: Количество наблюдений
            correlation: Корреляция между активами
            noise_ratio: Уровень шума
            
        Returns:
            DataFrame с синтетическими данными
        """
        np.random.seed(self.config.random_seed)
        
        # Генерируем коинтегрированную пару
        dates = pd.date_range('2020-01-01', periods=n_samples, freq='15T')
        
        # Базовый процесс (случайное блуждание)
        base_process = np.cumsum(np.random.normal(0, 0.01, n_samples))
        
        # Первый актив
        s1 = 100 * np.exp(base_process + np.random.normal(0, 0.005, n_samples))
        
        # Второй актив (коинтегрированный с первым)
        cointegration_error = np.random.normal(0, noise_ratio, n_samples)
        s2 = 100 * np.exp(base_process * correlation + cointegration_error)
        
        # Добавляем объемы
        volume1 = np.random.lognormal(10, 1, n_samples)
        volume2 = np.random.lognormal(10, 1, n_samples)
        
        return pd.DataFrame({
            'timestamp': dates,
            's1': s1,
            's2': s2,
            'volume_s1': volume1,
            'volume_s2': volume2
        }).set_index('timestamp')
    
    def run_robustness_tests(self, pair_data: pd.DataFrame) -> Dict[str, Any]:
        """Запускает тесты робастности.
        
        Args:
            pair_data: Данные для тестирования
            
        Returns:
            Результаты тестов
        """
        self.logger.info("Запуск тестов робастности...")
        
        if BacktestRobustnessTests is None:
            self.logger.warning("Пропущено: класс BacktestRobustnessTests не доступен")
            return {'error': 'BacktestRobustnessTests class not available'}
        
        try:
             # Создаем тестовый класс
             test_class = BacktestRobustnessTests()
             
             # Заменяем данные на наши
             test_class.pair_data = pair_data
             
             # Запускаем все тесты через метод run_all_tests
             results = test_class.run_all_tests(pair_data)
             
             # Преобразуем результаты в нужный формат
             formatted_results = {}
             for result in results:
                 test_name = result.get('test_name', 'unknown')
                 status = result.get('status', 'UNKNOWN')
                 error = result.get('error', None)
                 formatted_results[test_name] = {
                     'status': status,
                     'error': error,
                     'details': result
                 }
                 
                 if status == 'PASSED':
                     self.logger.info(f"✓ {test_name} - ПРОШЕЛ")
                 else:
                     self.logger.error(f"✗ {test_name} - ПРОВАЛЕН: {error}")
             
             return formatted_results
            
        except Exception as e:
            self.logger.error(f"Ошибка при выполнении тестов робастности: {e}")
            return {'error': str(e)}
    
    def run_synthetic_tests(self, pair_data: pd.DataFrame) -> Dict[str, Any]:
        """Запускает синтетические тесты.
        
        Args:
            pair_data: Данные для тестирования
            
        Returns:
            Результаты тестов
        """
        self.logger.info("Запуск синтетических тестов...")
        
        if SyntheticScenariosTests is None:
            self.logger.warning("Пропущено: класс SyntheticScenariosTests не доступен")
            return {'error': 'SyntheticScenariosTests class not available'}
        
        try:
             # Создаем тестовый класс
             test_class = SyntheticScenariosTests()
             
             # Запускаем все тесты через метод run_all_tests
             results = test_class.run_all_tests()
             
             # Преобразуем результаты в нужный формат
             formatted_results = {}
             for result in results:
                 test_name = result.get('test_name', 'unknown')
                 status = result.get('status', 'UNKNOWN')
                 error = result.get('error', None)
                 formatted_results[test_name] = {
                     'status': status,
                     'error': error,
                     'details': result
                 }
                 
                 if status == 'PASSED':
                     self.logger.info(f"✓ {test_name} - ПРОШЕЛ")
                 else:
                     self.logger.error(f"✗ {test_name} - ПРОВАЛЕН: {error}")
             
             return formatted_results
            
        except Exception as e:
            self.logger.error(f"Ошибка при выполнении синтетических тестов: {e}")
            return {'error': str(e)}
    
    def run_baseline_backtest(self, pair_data: pd.DataFrame) -> Dict[str, Any]:
        """Запускает базовый бэктест для генерации отчета.
        
        Args:
            pair_data: Данные для бэктеста
            
        Returns:
            Результаты бэктеста
        """
        self.logger.info("Запуск базового бэктеста...")
        
        try:
            # Создаем параметры из конфигурации
            params = {
                'rolling_window': self.config.walk_forward.rolling_window,
                'z_threshold': self.config.signals.z_entry,
                'z_exit': self.config.signals.z_exit,
                'stop_loss_multiplier': self.config.risk_management.stop_loss_multiplier,
                'capital_at_risk': self.config.risk_management.capital_at_risk,
                'commission_pct': self.config.costs.commission_pct,
                'slippage_pct': self.config.costs.slippage_pct,
                'bid_ask_spread_pct_s1': self.config.costs.bid_ask_spread_pct_s1,
                'bid_ask_spread_pct_s2': self.config.costs.bid_ask_spread_pct_s2,
                'half_life': 10.0,  # Будет рассчитано автоматически
                'time_stop_multiplier': self.config.risk_management.time_stop_multiplier,
                'cooldown_periods': self.config.risk_management.cooldown_periods
            }
            
            # Запускаем бэктест
            engine = BasePairBacktester(pair_data=pair_data, **params)
            engine.run()
            
            # Получаем результаты
            results = {
                'results_df': engine.results,
                'performance_metrics': engine.get_performance_metrics(),
                'engine': engine
            }
            
            self.logger.info("Базовый бэктест завершен успешно")
            return results
            
        except Exception as e:
            self.logger.error(f"Ошибка в базовом бэктесте: {e}")
            return {'error': str(e)}
    
    def generate_comprehensive_report(self, 
                                    pair_data: pd.DataFrame,
                                    backtest_results: Dict[str, Any],
                                    robustness_results: Dict[str, Any],
                                    synthetic_results: Dict[str, Any],
                                    output_dir: str = "reports") -> str:
        """Генерирует комплексный отчет.
        
        Args:
            pair_data: Исходные данные
            backtest_results: Результаты бэктеста
            robustness_results: Результаты тестов робастности
            synthetic_results: Результаты синтетических тестов
            output_dir: Директория для сохранения
            
        Returns:
            Путь к сгенерированному отчету
        """
        self.logger.info("Генерация комплексного отчета...")
        
        try:
            # Создаем генератор отчета
            report_generator = DetailedReportGenerator(self.config)
            
            # Добавляем результаты тестов к результатам бэктеста
            enhanced_results = backtest_results.copy()
            enhanced_results['robustness_tests'] = robustness_results
            enhanced_results['synthetic_tests'] = synthetic_results
            
            # Генерируем отчет
            report_path = report_generator.generate_full_report(
                backtest_results=enhanced_results,
                pair_data=pair_data,
                output_dir=output_dir
            )
            
            self.logger.info(f"Комплексный отчет сгенерирован: {report_path}")
            return report_path
            
        except Exception as e:
            self.logger.error(f"Ошибка при генерации отчета: {e}")
            raise
    
    def run_comprehensive_tests(self, data, output_dir='test_reports'):
        """Запуск всех тестов и генерация отчетов."""
        print("\n=== Запуск комплексного тестирования ===")
        
        # Создаем директорию для отчетов
        os.makedirs(output_dir, exist_ok=True)
        
        all_results = {}
        
        # 1. Тесты робастности бэктеста
        print("\n1. Тесты робастности бэктеста...")
        try:
            robustness_results = self.run_robustness_tests(data)
            all_results['robustness'] = robustness_results
            if 'error' not in robustness_results:
                print(f"   Выполнено {len(robustness_results)} тестов робастности")
            else:
                print(f"   Ошибка при выполнении тестов робастности: {robustness_results['error']}")
        except Exception as e:
            print(f"   Ошибка при выполнении тестов робастности: {e}")
            all_results['robustness'] = {'error': str(e)}
        
        # 2. Синтетические сценарии
        print("\n2. Синтетические сценарии...")
        try:
            synthetic_results = self.run_synthetic_tests(data)
            all_results['synthetic'] = synthetic_results
            if 'error' not in synthetic_results:
                print(f"   Выполнено {len(synthetic_results)} синтетических тестов")
            else:
                print(f"   Ошибка при выполнении синтетических тестов: {synthetic_results['error']}")
        except Exception as e:
            print(f"   Ошибка при выполнении синтетических тестов: {e}")
            all_results['synthetic'] = {'error': str(e)}
        
        # 3. Базовый бэктест
        print("\n3. Базовый бэктест...")
        try:
            backtest_results = self.run_baseline_backtest(data)
            all_results['backtest'] = backtest_results
            if 'error' not in backtest_results:
                metrics = backtest_results.get('performance_metrics', {})
                print(f"   Sharpe ratio: {metrics.get('sharpe_ratio', 'N/A'):.3f}")
                print(f"   Total return: {metrics.get('total_return', 'N/A'):.2%}")
            else:
                print(f"   Ошибка в базовом бэктесте: {backtest_results['error']}")
        except Exception as e:
            print(f"   Ошибка в базовом бэктесте: {e}")
            all_results['backtest'] = {'error': str(e)}
        
        # 4. Генерация отчетов
        print("\n4. Генерация отчетов...")
        self._generate_test_reports(all_results, output_dir)
        
        print(f"\n=== Тестирование завершено. Отчеты сохранены в {output_dir} ===")
        return all_results
    
    def _generate_test_reports(self, all_results, output_dir):
        """Генерирует отчеты по результатам тестов."""
        try:
            # Создаем сводный отчет
            report_path = os.path.join(output_dir, 'test_summary.txt')
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write("=== СВОДНЫЙ ОТЧЕТ ПО ТЕСТИРОВАНИЮ ===\n\n")
                f.write(f"Дата: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                
                # Тесты робастности
                if 'robustness' in all_results:
                    f.write("ТЕСТЫ РОБАСТНОСТИ:\n")
                    robustness = all_results['robustness']
                    if 'error' in robustness:
                        f.write(f"  Ошибка: {robustness['error']}\n")
                    else:
                        for test_name, result in robustness.items():
                            status = result.get('status', 'UNKNOWN')
                            f.write(f"  {test_name}: {status}\n")
                            if result.get('error'):
                                f.write(f"    Ошибка: {result['error']}\n")
                    f.write("\n")
                
                # Синтетические тесты
                if 'synthetic' in all_results:
                    f.write("СИНТЕТИЧЕСКИЕ ТЕСТЫ:\n")
                    synthetic = all_results['synthetic']
                    if 'error' in synthetic:
                        f.write(f"  Ошибка: {synthetic['error']}\n")
                    else:
                        for test_name, result in synthetic.items():
                            status = result.get('status', 'UNKNOWN')
                            f.write(f"  {test_name}: {status}\n")
                            if result.get('error'):
                                f.write(f"    Ошибка: {result['error']}\n")
                    f.write("\n")
                
                # Базовый бэктест
                if 'backtest' in all_results:
                    f.write("БАЗОВЫЙ БЭКТЕСТ:\n")
                    backtest = all_results['backtest']
                    if 'error' in backtest:
                        f.write(f"  Ошибка: {backtest['error']}\n")
                    else:
                        metrics = backtest.get('performance_metrics', {})
                        f.write(f"  Sharpe ratio: {metrics.get('sharpe_ratio', 'N/A')}\n")
                        f.write(f"  Total return: {metrics.get('total_return', 'N/A')}\n")
                        f.write(f"  Max drawdown: {metrics.get('max_drawdown', 'N/A')}\n")
                        f.write(f"  Number of trades: {metrics.get('num_trades', 'N/A')}\n")
                    f.write("\n")
            
            print(f"   Сводный отчет сохранен: {report_path}")
            
        except Exception as e:
            print(f"   Ошибка при генерации отчетов: {e}")
    
    def run_full_test_suite(self, 
                           pair_data: Optional[pd.DataFrame] = None,
                           output_dir: str = "reports") -> Dict[str, Any]:
        """Запускает полный набор тестов и генерирует отчет.
        
        Args:
            pair_data: Данные для тестирования (если None, генерируются синтетические)
            output_dir: Директория для сохранения отчетов
            
        Returns:
            Сводные результаты всех тестов
        """
        self.logger.info("=" * 60)
        self.logger.info("ЗАПУСК ПОЛНОГО НАБОРА ТЕСТОВ")
        self.logger.info("=" * 60)
        
        start_time = datetime.now()
        
        # Генерируем или используем предоставленные данные
        if pair_data is None:
            self.logger.info("Генерация синтетических данных...")
            pair_data = self.generate_synthetic_data()
        
        # Запускаем все тесты
        results = {
            'start_time': start_time,
            'config_snapshot': self.config.__dict__,
            'data_info': {
                'shape': pair_data.shape,
                'date_range': [str(pair_data.index.min()), str(pair_data.index.max())],
                'missing_values': pair_data.isnull().sum().to_dict()
            }
        }
        
        # 1. Тесты робастности
        try:
            results['robustness_tests'] = self.run_robustness_tests(pair_data)
        except Exception as e:
            self.logger.error(f"Ошибка в тестах робастности: {e}")
            results['robustness_tests'] = {'error': str(e)}
        
        # 2. Синтетические тесты
        try:
            results['synthetic_tests'] = self.run_synthetic_tests(pair_data)
        except Exception as e:
            self.logger.error(f"Ошибка в синтетических тестах: {e}")
            results['synthetic_tests'] = {'error': str(e)}
        
        # 3. Базовый бэктест
        try:
            results['backtest_results'] = self.run_baseline_backtest(pair_data)
        except Exception as e:
            self.logger.error(f"Ошибка в базовом бэктесте: {e}")
            results['backtest_results'] = {'error': str(e)}
        
        # 4. Генерация отчета
        try:
            if 'error' not in results['backtest_results']:
                report_path = self.generate_comprehensive_report(
                    pair_data=pair_data,
                    backtest_results=results['backtest_results'],
                    robustness_results=results['robustness_tests'],
                    synthetic_results=results['synthetic_tests'],
                    output_dir=output_dir
                )
                results['report_path'] = report_path
            else:
                self.logger.warning("Пропуск генерации отчета из-за ошибок в бэктесте")
        except Exception as e:
            self.logger.error(f"Ошибка при генерации отчета: {e}")
            results['report_error'] = str(e)
        
        # Финализация
        end_time = datetime.now()
        results['end_time'] = end_time
        results['duration'] = str(end_time - start_time)
        
        # Сводная статистика
        self._print_summary(results)
        
        return results
    
    def _print_summary(self, results: Dict[str, Any]) -> None:
        """Выводит сводку результатов."""
        self.logger.info("\n" + "=" * 60)
        self.logger.info("СВОДКА РЕЗУЛЬТАТОВ")
        self.logger.info("=" * 60)
        
        # Статистика тестов робастности
        if 'robustness_tests' in results and 'error' not in results['robustness_tests']:
            robustness = results['robustness_tests']
            passed = sum(1 for r in robustness.values() if r['status'] == 'PASSED')
            total = len(robustness)
            self.logger.info(f"Тесты робастности: {passed}/{total} прошли")
        
        # Статистика синтетических тестов
        if 'synthetic_tests' in results and 'error' not in results['synthetic_tests']:
            synthetic = results['synthetic_tests']
            passed = sum(1 for r in synthetic.values() if r['status'] == 'PASSED')
            total = len(synthetic)
            self.logger.info(f"Синтетические тесты: {passed}/{total} прошли")
        
        # Результаты бэктеста
        if 'backtest_results' in results and 'error' not in results['backtest_results']:
            metrics = results['backtest_results'].get('performance_metrics', {})
            self.logger.info(f"Базовый бэктест: Sharpe = {metrics.get('sharpe_ratio', 'N/A'):.3f}")
            self.logger.info(f"Базовый бэктест: Доходность = {metrics.get('total_return', 'N/A'):.2%}")
        
        # Отчет
        if 'report_path' in results:
            self.logger.info(f"Отчет сгенерирован: {results['report_path']}")
        
        self.logger.info(f"Общее время выполнения: {results['duration']}")
        self.logger.info("=" * 60)


def main():
    """Основная функция для запуска тестов."""
    # Создаем тест-раннер
    runner = ComprehensiveTestRunner()
    
    # Запускаем полный набор тестов
    results = runner.run_full_test_suite()
    
    # Возвращаем код выхода на основе результатов
    if 'error' in results.get('backtest_results', {}):
        return 1
    
    # Проверяем критические тесты
    critical_tests = [
        'test_no_future_reference',
        'test_fee_sensitivity', 
        'test_signal_shift_sanity_check'
    ]
    
    robustness = results.get('robustness_tests', {})
    for test_name in critical_tests:
        if test_name in robustness and robustness[test_name]['status'] == 'FAILED':
            print(f"КРИТИЧЕСКИЙ ТЕСТ ПРОВАЛЕН: {test_name}")
            return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())