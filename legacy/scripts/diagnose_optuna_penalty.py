#!/usr/bin/env python3
"""
Диагностический скрипт для выявления причин штрафа (-1000) в Optuna оптимизации.
"""

import sys
from pathlib import Path

# Добавляем корневую директорию проекта в PYTHONPATH
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import logging
import yaml
from unittest.mock import Mock

from src.optimiser.objective import WalkForwardObjective
from src.optimiser.metric_utils import normalize_params, validate_params
from src.coint2.utils.config import load_config

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_single_trial():
    """Тестирует один trial с известными параметрами."""
    
    print("🔍 ДИАГНОСТИКА OPTUNA PENALTY")
    print("=" * 50)
    
    # Пути к конфигурациям
    base_config_path = "configs/main_2024.yaml"
    search_space_path = "configs/search_space.yaml"
    
    try:
        # 1. Проверяем существование файлов
        print(f"📁 Проверка файлов конфигурации...")
        if not Path(base_config_path).exists():
            print(f"❌ Базовая конфигурация не найдена: {base_config_path}")
            return False
        if not Path(search_space_path).exists():
            print(f"❌ Пространство поиска не найдено: {search_space_path}")
            return False
        print(f"✅ Файлы конфигурации найдены")
        
        # 2. Загружаем пространство поиска
        print(f"📊 Загрузка пространства поиска...")
        with open(search_space_path, 'r') as f:
            search_space = yaml.safe_load(f)
        print(f"✅ Пространство поиска загружено: {list(search_space.keys())}")
        
        # 3. Создаем objective
        print(f"🎯 Создание objective функции...")
        objective = WalkForwardObjective(
            base_config_path=base_config_path,
            search_space_path=search_space_path
        )
        print(f"✅ Objective функция создана")
        
        # 4. Создаем мок trial с "рабочими" параметрами
        print(f"🧪 Создание тестового trial...")
        mock_trial = Mock()
        mock_trial.number = 999  # Тестовый номер
        
        # Используем средние значения из search_space
        mock_trial.suggest_float = Mock(side_effect=lambda name, low, high: (low + high) / 2)
        mock_trial.suggest_int = Mock(side_effect=lambda name, low, high, step=1: low + step)
        mock_trial.set_user_attr = Mock()
        
        print(f"✅ Мок trial создан")
        
        # 5. Запускаем один trial
        print(f"🚀 Запуск тестового trial...")
        print(f"   Это может занять несколько минут...")
        
        score = objective(mock_trial)
        
        print(f"📊 РЕЗУЛЬТАТ TRIAL:")
        print(f"   Score: {score}")
        
        if score == -1000.0:
            print(f"❌ ПОЛУЧЕН ШТРАФ! Проверьте логи выше для деталей.")
            
            # Проверяем user attributes для дополнительной информации
            if mock_trial.set_user_attr.called:
                print(f"📝 User attributes:")
                for call in mock_trial.set_user_attr.call_args_list:
                    key, value = call[0]
                    if key == "error_message":
                        print(f"   Ошибка: {value}")
                    elif key == "traceback":
                        print(f"   Traceback: {value[:200]}...")
            
            return False
        else:
            print(f"✅ ПОЛУЧЕН ВАЛИДНЫЙ SCORE: {score}")
            return True
            
    except Exception as e:
        print(f"❌ КРИТИЧЕСКАЯ ОШИБКА: {e}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        return False

def test_parameter_validation():
    """Тестирует валидацию параметров отдельно."""
    
    print("\n🔧 ТЕСТ ВАЛИДАЦИИ ПАРАМЕТРОВ")
    print("=" * 50)
    
    # Тестовые параметры
    test_params = {
        'zscore_threshold': 1.5,
        'zscore_exit': 0.0,
        'stop_loss_multiplier': 3.0,
        'time_stop_multiplier': 2.0,
        'risk_per_position_pct': 0.02,
        'max_position_size_pct': 0.05,
        'max_active_positions': 10,
        'trial_number': 999
    }
    
    print(f"📊 Тестовые параметры: {test_params}")
    
    try:
        # Нормализация
        normalized = normalize_params(test_params)
        print(f"✅ Нормализация прошла успешно: {normalized}")
        
        # Валидация
        validated = validate_params(normalized)
        print(f"✅ Валидация прошла успешно: {validated}")
        
        return True
        
    except Exception as e:
        print(f"❌ Ошибка валидации: {e}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        return False

def test_config_loading():
    """Тестирует загрузку и применение конфигурации."""
    
    print("\n⚙️ ТЕСТ ЗАГРУЗКИ КОНФИГУРАЦИИ")
    print("=" * 50)
    
    try:
        # Загружаем базовую конфигурацию
        cfg = load_config("configs/main_2024.yaml")
        print(f"✅ Базовая конфигурация загружена")
        
        # Проверяем ключевые поля
        print(f"📊 Ключевые параметры:")
        print(f"   zscore_threshold: {cfg.backtest.zscore_threshold}")
        print(f"   zscore_exit: {cfg.backtest.zscore_exit}")
        print(f"   risk_per_position_pct: {cfg.portfolio.risk_per_position_pct}")
        print(f"   max_active_positions: {cfg.portfolio.max_active_positions}")
        
        return True
        
    except Exception as e:
        print(f"❌ Ошибка загрузки конфигурации: {e}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        return False

if __name__ == "__main__":
    print("🚀 ЗАПУСК ДИАГНОСТИКИ OPTUNA PENALTY")
    print("=" * 60)
    
    # Запускаем все тесты
    tests = [
        ("Валидация параметров", test_parameter_validation),
        ("Загрузка конфигурации", test_config_loading),
        ("Один trial", test_single_trial),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n🧪 {test_name.upper()}")
        print("-" * 40)
        success = test_func()
        results.append((test_name, success))
    
    # Итоговый отчет
    print(f"\n📊 ИТОГОВЫЙ ОТЧЕТ")
    print("=" * 60)
    for test_name, success in results:
        status = "✅ ПРОШЕЛ" if success else "❌ ПРОВАЛЕН"
        print(f"{test_name}: {status}")
    
    all_passed = all(success for _, success in results)
    if all_passed:
        print(f"\n🎉 ВСЕ ТЕСТЫ ПРОШЛИ УСПЕШНО!")
    else:
        print(f"\n⚠️ НЕКОТОРЫЕ ТЕСТЫ ПРОВАЛЕНЫ. ПРОВЕРЬТЕ ЛОГИ ВЫШЕ.")
        sys.exit(1)
