#!/usr/bin/env python3
"""
Скрипт для разметки дымовых тестов - самых быстрых и критически важных тестов.
"""

import re
from pathlib import Path

# Дымовые тесты - самые быстрые и критически важные
SMOKE_TESTS = [
    # Базовые тесты движка (быстрые)
    ("tests/engine/test_01_backtest_engine_core.py", "test_backtester_outputs"),
    ("tests/engine/test_01_backtest_engine_core.py", "test_cost_validation"),

    # Утилиты и конфигурация
    ("tests/utils/test_42_time_utilities.py", "test_ensure_datetime_index_sorts_and_drops_tz"),
    ("tests/utils/test_42_time_utilities.py", "test_infer_frequency_irregular"),

    # Базовые синтетические сценарии
    ("tests/test_22_synthetic_scenarios.py", "test_single_asset_constant"),
    ("tests/test_22_synthetic_scenarios.py", "test_cointegration_breakdown"),
    ("tests/test_22_synthetic_scenarios.py", "test_extreme_price_movements"),

    # Быстрые тесты конфигурации
    ("tests/core/test_file_glob.py", "test_rglob_finds_all_files"),

    # Быстрые тесты исправлений
    ("tests/test_lookahead_bias_fix.py", "test_basic_lookahead_prevention"),

    # Быстрые тесты оптимизации
    ("tests/test_03_backtest_engine_optimized.py", "test_inheritance_from_pair_backtester"),

    # Быстрые тесты торговых логов
    ("tests/engine/test_trades_log_fix.py", "test_trades_log_initialization"),

    # Быстрые тесты размера позиций
    ("tests/engine/test_volatility_based_sizing.py", "test_volatility_multiplier_calculation"),

    # Быстрые тесты интеграции
    ("tests/test_10_integration_pipeline.py", "test_backtest_initialization"),
]

def add_smoke_marker_to_test(file_path: Path, test_name: str):
    """Добавляет маркер smoke к конкретному тесту."""
    if not file_path.exists():
        print(f"❌ Файл не найден: {file_path}")
        return False
        
    content = file_path.read_text()
    
    # Ищем функцию теста
    pattern = f'(    def {test_name}\\(.*?\\):)'
    match = re.search(pattern, content, re.MULTILINE)
    
    if not match:
        # Попробуем найти без self параметра
        pattern = f'(def {test_name}\\(.*?\\):)'
        match = re.search(pattern, content, re.MULTILINE)
    
    if match:
        # Проверяем, нет ли уже маркера smoke
        if '@pytest.mark.smoke' in content:
            lines_before = content[:match.start()].split('\n')
            for line in reversed(lines_before[-5:]):  # проверяем 5 строк перед функцией
                if '@pytest.mark.smoke' in line and test_name in content[match.start():match.start()+200]:
                    print(f"⚠️  Маркер smoke уже есть для {test_name} в {file_path}")
                    return False
        
        # Добавляем маркер
        if match.group(1).startswith('    '):
            # Метод класса
            replacement = f'    @pytest.mark.smoke\n{match.group(1)}'
        else:
            # Обычная функция
            replacement = f'@pytest.mark.smoke\n{match.group(1)}'
        
        new_content = content.replace(match.group(1), replacement)
        
        # Убеждаемся, что импорт pytest есть
        if 'import pytest' not in new_content:
            # Добавляем импорт после первого импорта
            new_content = re.sub(r'(import \w+)', '\\1\nimport pytest', new_content, count=1)
        
        file_path.write_text(new_content)
        print(f"✅ Добавлен маркер @pytest.mark.smoke для {test_name} в {file_path}")
        return True
    else:
        print(f"⚠️  Не удалось найти тест {test_name} в {file_path}")
        return False

def main():
    """Основная функция для разметки дымовых тестов."""
    project_root = Path(__file__).parent.parent
    
    print("💨 Разметка дымовых тестов...")
    
    success_count = 0
    for file_path_str, test_name in SMOKE_TESTS:
        file_path = project_root / file_path_str
        if add_smoke_marker_to_test(file_path, test_name):
            success_count += 1
    
    print(f"✅ Размечено дымовых тестов: {success_count}/{len(SMOKE_TESTS)}")

if __name__ == "__main__":
    main()
