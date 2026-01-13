#!/usr/bin/env python3
"""
Скрипт для автоматической разметки медленных тестов маркерами @pytest.mark.slow
на основе результатов профилирования.
"""

import re
from pathlib import Path

# Список медленных тестов (время > 5 секунд) из профилирования
SLOW_TESTS = [
    # Очень медленные (>100s)
    ("tests/test_optimization_fixes.py", "test_fast_objective_with_simple_params"),
    
    # Медленные backtest тесты (>10s)
    ("tests/engine/test_01_backtest_engine_core.py", "test_ols_cache_memory_limit_with_15min_data"),
    ("tests/test_backtest_correctness_with_blas.py", "test_memory_usage_with_blas_optimization"),
    ("tests/test_11_robustness_checks.py", "test_fee_sensitivity"),
    ("tests/test_critical_fixes.py", "test_logging_improvement"),
    ("tests/test_11_robustness_checks.py", "test_date_permutation_destroys_performance"),
    ("tests/test_global_cache_integration.py", "test_cache_performance_vs_traditional_approach"),
    ("tests/engine/test_market_regime_optimization.py", "test_performance_improvement_integration"),
    ("tests/test_global_cache_integration.py", "test_concurrent_cache_access_simulation"),
    ("tests/test_11_robustness_checks.py", "test_cost_breakdown_consistency"),
    ("tests/test_11_robustness_checks.py", "test_signal_shift_sanity_check"),
    ("tests/test_11_robustness_checks.py", "test_no_future_reference"),
    ("tests/test_11_robustness_checks.py", "test_time_shift_degrades_performance"),
    ("tests/test_backtest_correctness_with_blas.py", "test_backtest_results_identical_with_blas_optimization"),
    ("tests/test_backtest_correctness_with_blas.py", "test_concurrent_backtest_execution"),
    
    # Walk-forward тесты
    ("tests/test_walk_forward_integration.py", "test_memory_usage_optimization"),
    ("tests/test_walk_forward_integration.py", "test_walk_forward_with_optimizations"),
    ("tests/test_walk_forward_integration.py", "test_error_handling_and_robustness"),
    ("tests/test_walk_forward_integration.py", "test_performance_comparison"),
    ("tests/test_walk_forward_integration.py", "test_results_consistency_across_runs"),
    ("tests/pipeline/test_walk_forward.py", "test_walk_forward"),
    ("tests/test_multiple_walk_forward_steps.py", "*"),  # все тесты в файле
    ("tests/test_walk_forward_debug.py", "*"),
    ("tests/test_walk_forward_enhancements.py", "*"),
    
    # Optuna тесты (кроме concurrency - он уже serial)
    ("tests/test_optuna_integration.py", "*"),
    ("tests/test_optuna_final_comprehensive_check.py", "*"),
    ("tests/test_optuna_final_critical_fixes.py", "*"),
    ("tests/test_optuna_fixes_validation.py", "*"),
    ("tests/test_optuna_lookahead_bias_fix.py", "*"),
    ("tests/test_optuna_objective_fix.py", "*"),
    ("tests/test_optuna_optimization_fix.py", "*"),
    ("tests/test_optuna_pruner.py", "*"),
    ("tests/test_optuna_audit_fixes.py", "*"),
    ("tests/test_optuna_critical_fixes_2.py", "*"),
]

# Тесты для маркера serial (глобальный кэш и SQLite)
SERIAL_TESTS = [
    ("tests/test_optuna_sqlite_concurrency.py", "*"),  # уже сделано
    ("tests/test_global_cache_integration.py", "*"),
]

def add_marker_to_test(file_path: Path, test_name: str, marker: str):
    """Добавляет маркер к конкретному тесту."""
    content = file_path.read_text()
    
    if test_name == "*":
        # Добавляем маркер ко всем тестам в файле
        pattern = r'(    def test_\w+\(self.*?\):)'
        replacement = f'    @pytest.mark.{marker}\n\\1'
    else:
        # Добавляем маркер к конкретному тесту
        pattern = f'(    def {test_name}\\(self.*?\\):)'
        replacement = f'    @pytest.mark.{marker}\n\\1'
    
    new_content = re.sub(pattern, replacement, content, flags=re.MULTILINE)
    
    # Проверяем, что изменения были внесены
    if new_content != content:
        # Убеждаемся, что импорт pytest есть
        if 'import pytest' not in new_content:
            new_content = new_content.replace('import pytest', 'import pytest', 1)
            if 'import pytest' not in new_content:
                # Добавляем импорт после первого импорта
                new_content = re.sub(r'(import \w+)', '\\1\nimport pytest', new_content, count=1)
        
        file_path.write_text(new_content)
        print(f"✅ Добавлен маркер @pytest.mark.{marker} в {file_path}")
        return True
    else:
        print(f"⚠️  Не удалось найти тест {test_name} в {file_path}")
        return False

def main():
    """Основная функция для разметки тестов."""
    project_root = Path(__file__).parent.parent
    
    print("🏷️  Разметка медленных тестов...")
    
    # Размечаем медленные тесты
    for file_path_str, test_name in SLOW_TESTS:
        file_path = project_root / file_path_str
        if file_path.exists():
            add_marker_to_test(file_path, test_name, "slow")
        else:
            print(f"❌ Файл не найден: {file_path}")
    
    # Размечаем serial тесты
    for file_path_str, test_name in SERIAL_TESTS:
        file_path = project_root / file_path_str
        if file_path.exists():
            add_marker_to_test(file_path, test_name, "serial")
        else:
            print(f"❌ Файл не найден: {file_path}")
    
    print("✅ Разметка завершена!")

if __name__ == "__main__":
    main()
