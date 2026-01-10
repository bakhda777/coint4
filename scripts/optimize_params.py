#!/usr/bin/env python3
"""Пример скрипта для быстрого запуска оптимизации параметров."""

import sys
import time
from pathlib import Path

# Добавляем src в путь для импорта
sys.path.append(str(Path(__file__).parent.parent / "src"))

import optuna
from optimiser import FastWalkForwardObjective
import logging

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def main():
    """Основная функция для запуска оптимизации."""

    # Пути к конфигам
    base_config = "configs/main_2024.yaml"
    search_space = "configs/search_space_fast.yaml"  # Используем быстрое пространство поиска

    print("🚀 Запускаем быструю оптимизацию параметров...")
    print(f"📁 Базовый конфиг: {base_config}")
    print(f"🔍 Пространство поиска: {search_space}")
    print("⏱️  Количество trials: 20 (для быстрого тестирования)")
    print()

    try:
        # Создаем быструю целевую функцию
        print("⚙️  Инициализация быстрой целевой функции...")
        objective = FastWalkForwardObjective(
            base_config_path=base_config,
            search_space_path=search_space
        )

        # Создаем study с уникальным именем
        from datetime import datetime
        study_name = f"quick_optimization_{datetime.now():%Y%m%d_%H%M%S}"
        storage_path = f"sqlite:///outputs/studies/{study_name}.db"

        # Создаем директорию для studies
        Path("outputs/studies").mkdir(parents=True, exist_ok=True)

        study = optuna.create_study(
            study_name=study_name,
            storage=storage_path,
            direction='maximize',
            load_if_exists=True,
            pruner=optuna.pruners.MedianPruner(
                n_startup_trials=5,
                n_warmup_steps=5,
                interval_steps=1
            )
        )

        # Добавляем базовую точку для TPE
        base_params = {
            "zscore_threshold": 0.9,
            "hysteresis": 0.4,
            "rolling_window": 20,
            "risk_per_position_pct": 0.015,
            "max_position_size_pct": 0.10,
            "stop_loss_multiplier": 3.0,
            "time_stop_multiplier": 5.0,
            "cooldown_hours": 2,
            "commission_pct": 0.0004,
            "slippage_pct": 0.0005,
            "normalization_method": "minmax",
            "min_history_ratio": 0.6
        }
        study.enqueue_trial(base_params)

        print(f"✅ Optuna study создан: {study_name}")
        print("⏱️  Запуск оптимизации...")
        print()

        # Запускаем оптимизацию
        start_time = time.time()

        study.optimize(
            objective,
            n_trials=20,
            timeout=1800,  # 30 минут максимум
            n_jobs=1  # Один процесс для стабильности
        )

        optimization_time = time.time() - start_time

        # Выводим результаты
        print("\n" + "="*60)
        print("🎯 РЕЗУЛЬТАТЫ БЫСТРОЙ ОПТИМИЗАЦИИ")
        print("="*60)

        completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]

        if completed_trials:
            print(f"📈 Лучший Sharpe Ratio: {study.best_value:.4f}")
            print(f"🔢 Количество завершенных trials: {len(completed_trials)}")
            print(f"⏰ Время оптимизации: {optimization_time:.1f} сек")
            print("\n🏆 Лучшие параметры:")

            for param, value in study.best_params.items():
                if isinstance(value, float):
                    print(f"   {param}: {value:.4f}")
                else:
                    print(f"   {param}: {value}")

            # Сохраняем лучший конфиг
            output_config = "configs/optimized_quick_test.yaml"

            # Загружаем базовый конфиг и обновляем его лучшими параметрами
            import yaml
            import sys
            from pathlib import Path
            sys.path.append(str(Path(__file__).parent.parent / "src"))
            from coint2.utils.config import load_config

            config_obj = load_config(base_config)
            config = config_obj.model_dump()

            # Обновляем конфиг лучшими параметрами
            for param, value in study.best_params.items():
                # Простое обновление - можно улучшить для вложенных параметров
                config[param] = value

            with open(output_config, 'w') as f:
                yaml.dump(config, f, default_flow_style=False, allow_unicode=True)

            print(f"\n💾 Лучший конфиг сохранен: {output_config}")

            if len(completed_trials) >= 10:
                print("\n✅ Быстрая оптимизация завершена успешно!")
                print("\n🎯 Следующие шаги:")
                print("1. Запустите полную оптимизацию:")
                print("   python src/optimiser/run_optimization.py --n-trials 100")
                print("2. Проанализируйте результаты в Optuna Dashboard:")
                print(f"   optuna-dashboard {storage_path}")
            else:
                print("⚠️  Мало успешных trials. Возможно, нужно:")
                print("   • Ослабить фильтры в configs/main_2024.yaml")
                print("   • Проверить качество предотобранных пар")
                print("   • Увеличить количество trials")
        else:
            print("❌ Ни один trial не завершился успешно")
            print("🔍 Проверьте логи для диагностики проблем")

    except Exception as e:
        print(f"❌ Ошибка при запуске оптимизации: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True

if __name__ == "__main__":
    main()