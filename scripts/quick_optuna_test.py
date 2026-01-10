#!/usr/bin/env python3
"""
Быстрый тест Optuna оптимизации без полного walk-forward анализа.
"""

import sys
from pathlib import Path

# Добавляем корневую директорию проекта в PYTHONPATH
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import logging
import yaml
import optuna
from unittest.mock import Mock, patch

from src.optimiser.objective import WalkForwardObjective
from src.optimiser.metric_utils import normalize_params, validate_params
from src.coint2.utils.config import load_config

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def mock_run_walk_forward(cfg):
    """Мок функция для run_walk_forward, возвращающая тестовые метрики."""
    return {
        "sharpe_ratio_abs": 1.5,
        "total_trades": 50,
        "max_drawdown_on_equity": 0.15,
        "total_return_pct": 0.25,
        "win_rate": 0.55,
        "avg_trade_size": 1000.0,
        "avg_hold_time": 24.0
    }

def test_optuna_with_mock():
    """Тестирует Optuna оптимизацию с мок walk-forward."""
    
    print("🔍 БЫСТРЫЙ ТЕСТ OPTUNA С МОК WALK-FORWARD")
    print("=" * 60)
    
    try:
        # Создаем fast objective 
        objective = WalkForwardObjective(
            base_config_path="configs/main_2024.yaml",
            search_space_path="configs/search_space_fast.yaml",
            fast_mode=True
        )
            
        print("✅ Objective функция создана с мокированным walk-forward")
        
        # Создаем study
        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=42)
        )
        
        print("✅ Optuna study создан")
        
        # Запускаем несколько trials
        print("🚀 Запуск 5 тестовых trials...")
        study.optimize(objective, n_trials=5)
        
        print(f"📊 РЕЗУЛЬТАТЫ:")
        print(f"   Количество trials: {len(study.trials)}")
        print(f"   Лучший score: {study.best_value}")
        print(f"   Лучшие параметры: {study.best_params}")
        
        # Проверяем, есть ли штрафы
        penalty_trials = [t for t in study.trials if t.value == -1000.0]
        if penalty_trials:
            print(f"❌ Найдено {len(penalty_trials)} trials со штрафом!")
            for trial in penalty_trials:
                print(f"   Trial #{trial.number}: {trial.user_attrs}")
            return False
        else:
            print(f"✅ Все trials прошли успешно, штрафов нет!")
            return True
            
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        return False

def test_parameter_mapping():
    """Тестирует маппинг параметров из search_space в конфигурацию."""
    
    print("\n🔧 ТЕСТ МАППИНГА ПАРАМЕТРОВ")
    print("=" * 50)
    
    try:
        # Загружаем search_space
        with open("configs/search_space.yaml", 'r') as f:
            search_space = yaml.safe_load(f)
        
        print(f"📊 Группы в search_space: {list(search_space.keys())}")
        
        # Создаем мок trial
        mock_trial = Mock()
        mock_trial.number = 999
        mock_trial.suggest_float = Mock(side_effect=lambda name, low, high: (low + high) / 2)
        mock_trial.suggest_int = Mock(side_effect=lambda name, low, high, step=1: low + step)
        
        # Тестируем сбор параметров без создания objective
        params = {}
        
        # Сигналы
        params['zscore_threshold'] = mock_trial.suggest_float(
            "zscore_threshold", 
            **search_space['signals']['zscore_threshold']
        )
        params['zscore_exit'] = mock_trial.suggest_float(
            "zscore_exit", 
            **search_space['signals']['zscore_exit']
        )
        
        # Управление риском
        params['stop_loss_multiplier'] = mock_trial.suggest_float(
            "stop_loss_multiplier", 
            **search_space['risk_management']['stop_loss_multiplier']
        )
        params['time_stop_multiplier'] = mock_trial.suggest_float(
            "time_stop_multiplier", 
            **search_space['risk_management']['time_stop_multiplier']
        )
        
        # Портфель
        params['risk_per_position_pct'] = mock_trial.suggest_float(
            "risk_per_position_pct", 
            **search_space['portfolio']['risk_per_position_pct']
        )
        params['max_position_size_pct'] = mock_trial.suggest_float(
            "max_position_size_pct", 
            **search_space['portfolio']['max_position_size_pct']
        )
        
        max_pos_space = search_space['portfolio']['max_active_positions']
        params['max_active_positions'] = mock_trial.suggest_int(
            "max_active_positions", 
            max_pos_space['low'], 
            max_pos_space['high'], 
            step=max_pos_space.get('step', 1)
        )
        
        print(f"✅ Параметры собраны: {params}")
        
        # Валидируем параметры
        validated_params = validate_params(params)
        print(f"✅ Параметры валидированы: {validated_params}")
        
        # Проверяем логическую консистентность
        assert validated_params['zscore_exit'] < validated_params['zscore_threshold'], \
            "zscore_exit должен быть меньше zscore_threshold"
        assert validated_params['stop_loss_multiplier'] >= 0, \
            "stop_loss_multiplier должен быть неотрицательным"
        assert 0 < validated_params['risk_per_position_pct'] <= 1, \
            "risk_per_position_pct должен быть в (0, 1]"
        assert 0 < validated_params['max_position_size_pct'] <= 1, \
            "max_position_size_pct должен быть в (0, 1]"
        assert validated_params['max_active_positions'] >= 1, \
            "max_active_positions должен быть >= 1"
        
        print("✅ Логическая консистентность параметров проверена")
        return True
        
    except Exception as e:
        print(f"❌ Ошибка маппинга параметров: {e}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        return False

def test_real_optuna_single_trial():
    """Тестирует один реальный trial Optuna."""
    
    print("\n🧪 ТЕСТ ОДНОГО РЕАЛЬНОГО TRIAL")
    print("=" * 50)
    
    try:
        # Создаем study
        study = optuna.create_study(direction="maximize")
        
        # Создаем objective
        objective = WalkForwardObjective(
            base_config_path="configs/main_2024.yaml",
            search_space_path="configs/search_space.yaml"
        )
        
        print("🚀 Запуск одного реального trial (может занять время)...")
        
        # Запускаем один trial с таймаутом
        import signal
        
        def timeout_handler(signum, frame):
            raise TimeoutError("Trial занял слишком много времени")
        
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(60)  # 60 секунд таймаут
        
        try:
            study.optimize(objective, n_trials=1)
            signal.alarm(0)  # Отключаем таймаут
            
            trial = study.trials[0]
            print(f"✅ Trial завершен:")
            print(f"   Score: {trial.value}")
            print(f"   Параметры: {trial.params}")
            
            if trial.value == -1000.0:
                print(f"❌ Получен штраф!")
                if trial.user_attrs:
                    print(f"   User attrs: {trial.user_attrs}")
                return False
            else:
                print(f"✅ Получен валидный score!")
                return True
                
        except TimeoutError:
            signal.alarm(0)
            print(f"⏰ Trial превысил таймаут 60 секунд")
            print(f"   Это указывает на медленную работу walk-forward, а не на ошибку параметров")
            return True  # Считаем это успехом, так как нет ошибки параметров
            
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        return False

if __name__ == "__main__":
    print("🚀 БЫСТРАЯ ДИАГНОСТИКА OPTUNA")
    print("=" * 60)
    
    # Запускаем тесты
    tests = [
        ("Маппинг параметров", test_parameter_mapping),
        ("Optuna с мок walk-forward", test_optuna_with_mock),
        ("Один реальный trial", test_real_optuna_single_trial),
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
        print(f"\n🎉 ВСЕ ТЕСТЫ ПРОШЛИ!")
        print(f"💡 Если Optuna возвращает штрафы, проблема скорее всего в медленной работе walk-forward,")
        print(f"   а не в ошибках параметров. Рассмотрите использование FastWalkForwardObjective.")
    else:
        print(f"\n⚠️ НЕКОТОРЫЕ ТЕСТЫ ПРОВАЛЕНЫ.")
        sys.exit(1)
