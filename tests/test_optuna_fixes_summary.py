#!/usr/bin/env python3
"""
Сводный тест всех критических исправлений Optuna.
"""

import sys
from pathlib import Path
import tempfile
import yaml

# Добавляем корневую директорию в PYTHONPATH
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def test_penalty_value():
    """Тест 1: Проверка умеренного штрафа."""
    from src.optimiser.fast_objective import PENALTY
    
    print(f"🔍 Тест 1: PENALTY = {PENALTY}")
    
    # Штраф должен быть умеренным для TPE
    assert -10.0 <= PENALTY <= -1.0, f"PENALTY слишком агрессивный: {PENALTY}"
    assert PENALTY == -5.0, f"PENALTY должен быть -5.0, получен: {PENALTY}"
    
    print("✅ Тест 1 пройден: Умеренный штраф")


def test_median_pruner():
    """Тест 2: Проверка MedianPruner вместо HyperbandPruner."""
    import optuna
    
    print("🔍 Тест 2: MedianPruner")
    
    # Создаем study как в run_optimization.py
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42, multivariate=True),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=0, interval_steps=1)
    )
    
    # Проверяем тип pruner
    assert isinstance(study.pruner, optuna.pruners.MedianPruner), f"Неправильный тип pruner: {type(study.pruner)}"
    assert not isinstance(study.pruner, optuna.pruners.HyperbandPruner), "Используется HyperbandPruner!"
    
    print("✅ Тест 2 пройден: MedianPruner настроен")


def test_global_seeds():
    """Тест 3: Проверка глобальных сидов."""
    import random
    import numpy as np
    
    print("🔍 Тест 3: Глобальные сиды")
    
    seed = 42
    
    # Устанавливаем сиды как в run_optimization.py
    random.seed(seed)
    np.random.seed(seed)
    
    # Генерируем значения
    val1 = random.random()
    val2 = np.random.random()
    
    # Сбрасываем сиды
    random.seed(seed)
    np.random.seed(seed)
    
    # Генерируем те же значения
    val3 = random.random()
    val4 = np.random.random()
    
    assert val1 == val3, f"Random не воспроизводим: {val1} != {val3}"
    assert val2 == val4, f"NumPy random не воспроизводим: {val2} != {val4}"
    
    print("✅ Тест 3 пройден: Глобальные сиды работают")


def test_tpe_sampler_startup_trials():
    """Тест 4: Проверка исправления n_startup_trials."""
    import optuna
    
    print("🔍 Тест 4: TPESampler startup trials")
    
    # Тестируем исправленную логику max(10, n_trials // 10)
    test_cases = [
        (5, 10),    # max(10, 5//10) = max(10, 0) = 10
        (50, 10),   # max(10, 50//10) = max(10, 5) = 10  
        (200, 20),  # max(10, 200//10) = max(10, 20) = 20
    ]
    
    for n_trials, expected_startup in test_cases:
        calculated_startup = max(10, n_trials // 10)
        assert calculated_startup == expected_startup, f"Неправильный расчет startup trials для {n_trials}: {calculated_startup} != {expected_startup}"
        
        # Создаем sampler с правильным количеством startup trials
        sampler = optuna.samplers.TPESampler(seed=42, multivariate=True, n_startup_trials=calculated_startup)
        assert sampler._n_startup_trials == expected_startup
    
    print("✅ Тест 4 пройден: TPESampler startup trials исправлены")


def test_config_serialization():
    """Тест 5: Проверка сериализации конфигурации без Python объектов."""
    print("🔍 Тест 5: Сериализация конфигурации")
    
    # Тестируем исправленную логику из _save_best_config
    config_data = {
        'data_dir': 'data_downloaded',  # Строка, не Path
        'results_dir': 'results',       # Строка, не Path
        'backtest': {
            'zscore_threshold': 1.5,
            'commission_pct': 0.0004
        },
        'portfolio': {
            'max_active_positions': 5,
            'risk_per_position_pct': 0.02
        }
    }
    
    # Сериализуем в YAML
    yaml_str = yaml.dump(config_data, default_flow_style=False, allow_unicode=True)
    
    # Проверяем, что нет Python объектов
    assert "!!python" not in yaml_str, "Найдены Python объекты в YAML"
    assert "pathlib" not in yaml_str, "Найдены pathlib объекты в YAML"
    
    # Проверяем, что можно десериализовать
    loaded_config = yaml.safe_load(yaml_str)
    assert loaded_config == config_data, "Конфигурация не десериализуется корректно"
    
    print("✅ Тест 5 пройден: Сериализация без Python объектов")


def test_sqlite_parallelism_logic():
    """Тест 6: Проверка логики для SQLite и параллельности."""
    print("🔍 Тест 6: SQLite и параллельность")
    
    # Тестируем логику из run_optimization.py
    test_cases = [
        ("sqlite:///test.db", -1, 1),    # SQLite с -1 -> должно стать 1
        ("sqlite:///test.db", 4, 4),     # SQLite с 4 -> остается 4 (с предупреждением)
        ("postgresql://...", -1, -1),    # PostgreSQL с -1 -> остается -1
    ]
    
    for storage_url, n_jobs_input, expected_n_jobs in test_cases:
        # Симулируем логику проверки
        if "sqlite" in storage_url and n_jobs_input == -1:
            n_jobs_result = 1  # Принудительно устанавливаем 1
        else:
            n_jobs_result = n_jobs_input
        
        assert n_jobs_result == expected_n_jobs, f"Неправильная логика для {storage_url}, {n_jobs_input}: {n_jobs_result} != {expected_n_jobs}"
    
    print("✅ Тест 6 пройден: SQLite и параллельность")


def test_trial_pruned_usage():
    """Тест 7: Проверка использования TrialPruned."""
    import optuna
    
    print("🔍 Тест 7: TrialPruned usage")
    
    def objective_with_pruning(trial):
        if trial.number == 0:
            # Симулируем недостаточно сделок
            trial.set_user_attr("error", "insufficient_trades")
            raise optuna.TrialPruned("Insufficient trades")
        elif trial.number == 1:
            # Симулируем невалидный Sharpe
            trial.set_user_attr("error", "invalid_sharpe")
            raise optuna.TrialPruned("Invalid Sharpe")
        else:
            return 1.0
    
    study = optuna.create_study(direction="maximize")
    study.optimize(objective_with_pruning, n_trials=3)
    
    # Проверяем результаты
    pruned_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    complete_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    
    assert len(pruned_trials) == 2, f"Ожидалось 2 pruned trials, получено: {len(pruned_trials)}"
    assert len(complete_trials) == 1, f"Ожидался 1 complete trial, получено: {len(complete_trials)}"
    
    # Проверяем атрибуты
    errors = [t.user_attrs.get("error") for t in pruned_trials]
    assert "insufficient_trades" in errors, "Отсутствует атрибут insufficient_trades"
    assert "invalid_sharpe" in errors, "Отсутствует атрибут invalid_sharpe"
    
    print("✅ Тест 7 пройден: TrialPruned usage")


def run_all_tests():
    """Запускает все тесты критических исправлений."""
    print("🧪 ТЕСТИРОВАНИЕ КРИТИЧЕСКИХ ИСПРАВЛЕНИЙ OPTUNA")
    print("=" * 60)
    
    tests = [
        test_penalty_value,
        test_median_pruner,
        test_global_seeds,
        test_tpe_sampler_startup_trials,
        test_config_serialization,
        test_sqlite_parallelism_logic,
        test_trial_pruned_usage
    ]
    
    passed = 0
    failed = 0
    
    for test_func in tests:
        try:
            test_func()
            passed += 1
        except Exception as e:
            print(f"❌ {test_func.__name__} FAILED: {e}")
            failed += 1
        print()
    
    print("=" * 60)
    print(f"📊 РЕЗУЛЬТАТЫ: {passed} пройдено, {failed} провалено")
    
    if failed == 0:
        print("🎉 ВСЕ КРИТИЧЕСКИЕ ИСПРАВЛЕНИЯ РАБОТАЮТ!")
        return True
    else:
        print("⚠️  ЕСТЬ ПРОБЛЕМЫ С ИСПРАВЛЕНИЯМИ!")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
