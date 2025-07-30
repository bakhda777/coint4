#!/usr/bin/env python3
"""
Скрипт для быстрой оптимизации параметров на предварительно отобранных парах.
"""

import sys
from pathlib import Path
import optuna
import time

# Добавляем src в путь для импорта
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.optimiser.fast_objective import FastWalkForwardObjective

def main():
    """Основная функция для запуска быстрой оптимизации."""
    
    print("🚀 Запуск БЫСТРОЙ оптимизации параметров...")
    print("⚡ Использует предварительно отобранные пары для ускорения")
    
    # Проверяем наличие предотобранных пар
    pairs_file = Path("outputs/preselected_pairs.csv")
    if not pairs_file.exists():
        print("❌ Файл outputs/preselected_pairs.csv не найден!")
        print("🔧 Сначала запустите: poetry run python scripts/preselect_pairs.py")
        return
    
    # Пути к конфигам
    base_config = "configs/main_2024.yaml"
    search_space = "configs/search_space.yaml"
    
    print(f"📁 Базовый конфиг: {base_config}")
    print(f"🔍 Пространство поиска: {search_space}")
    
    try:
        # Создаем быструю целевую функцию
        print("⚙️  Инициализация быстрой целевой функции...")
        objective = FastWalkForwardObjective(
            base_config_path=base_config,
            search_space_path=search_space
        )
        
        # Создаем study с уникальным именем
        from datetime import datetime
        study_name = f"fast_optimization_{datetime.now():%Y%m%d_%H%M%S}"
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
                n_warmup_steps=10,
                interval_steps=5
            )
        )
        
        print(f"✅ Optuna study создан: {study_name}")
        print("⏱️  Запуск оптимизации...")
        print("📊 Количество trials: 200 (полная быстрая оптимизация)")
        print("🔄 Параллельность: 1 процесс (для стабильности)")
        print()

        # Запускаем оптимизацию
        start_time = time.time()

        study.optimize(
            objective,
            n_trials=200,  # Увеличиваем количество trials
            timeout=3600,  # 1 час максимум
            n_jobs=1  # Один процесс для стабильности
        )
        
        optimization_time = time.time() - start_time
        
        # Выводим результаты
        print("\n" + "="*60)
        print("🎯 РЕЗУЛЬТАТЫ БЫСТРОЙ ОПТИМИЗАЦИИ")
        print("="*60)
        
        completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        
        if len(completed_trials) > 0:
            print(f"📈 Лучший результат: {study.best_value:.4f}")
            print(f"🔢 Завершенных trials: {len(completed_trials)}")
            print(f"⏰ Время оптимизации: {optimization_time:.1f} сек ({optimization_time/60:.1f} мин)")
            print(f"⚡ Скорость: {len(completed_trials)/optimization_time*60:.1f} trials/мин")
            
            print("\n🏆 Лучшие параметры:")
            for param, value in study.best_params.items():
                if isinstance(value, float):
                    print(f"   {param}: {value:.4f}")
                else:
                    print(f"   {param}: {value}")
            
            # Сохраняем лучший конфиг
            output_config = "configs/optimized_fast.yaml"
            save_best_config(study, base_config, output_config)
            print(f"\n💾 Лучший конфиг сохранен: {output_config}")
            
            # Статистика trials
            print(f"\n📊 Статистика trials:")
            pruned_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
            failed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.FAIL]
            
            print(f"   ✅ Завершенных: {len(completed_trials)}")
            print(f"   ✂️  Прерванных: {len(pruned_trials)}")
            print(f"   ❌ Провалившихся: {len(failed_trials)}")
            
            if len(completed_trials) >= 10:
                print("\n✅ Быстрая оптимизация завершена успешно!")
                print("\n🎯 Следующие шаги:")
                print("1. Проанализируйте результаты:")
                print("   poetry run python src/optimiser/analyze_results.py")
                print("2. Запустите полную валидацию с лучшими параметрами:")
                print(f"   poetry run python -c \"from coint2.pipeline.walk_forward_orchestrator import run_walk_forward; from coint2.utils.config import load_config; cfg = load_config('{output_config}'); run_walk_forward(cfg)\"")
            else:
                print("⚠️  Мало успешных trials. Возможно, нужно:")
                print("   • Ослабить фильтры в configs/main_2024.yaml")
                print("   • Проверить качество предотобранных пар")
                print("   • Увеличить количество trials")
        else:
            print("❌ Ни один trial не завершился успешно")
            print("🔧 Рекомендации:")
            print("   • Проверьте предотобранные пары: outputs/preselected_pairs.csv")
            print("   • Ослабьте торговые параметры в configs/main_2024.yaml")
            print("   • Проверьте логи на ошибки")
        
        print("\n" + "="*60)
        
    except Exception as e:
        print(f"❌ Ошибка при оптимизации: {e}")
        import traceback
        traceback.print_exc()

def save_best_config(study, base_config_path, output_path):
    """Сохраняет конфигурацию с лучшими параметрами."""
    from coint2.utils.config import load_config
    import yaml
    
    # Загружаем базовую конфигурацию
    cfg = load_config(base_config_path)
    
    # Применяем лучшие параметры
    best_params = study.best_params
    
    if 'z_entry' in best_params:
        cfg.backtest.zscore_threshold = best_params['z_entry']
        cfg.backtest.zscore_entry_threshold = best_params['z_entry']
    
    if 'z_exit' in best_params:
        cfg.backtest.zscore_exit = best_params['z_exit']
    
    if 'sl_mult' in best_params:
        cfg.backtest.stop_loss_multiplier = best_params['sl_mult']
    
    if 'time_stop_mult' in best_params:
        cfg.backtest.time_stop_multiplier = best_params['time_stop_mult']
    
    if 'risk_per_pos' in best_params:
        cfg.portfolio.risk_per_position_pct = best_params['risk_per_pos']
    
    # ИСПРАВЛЕНО: Единое имя параметра
    if 'max_position_size_pct' in best_params:
        cfg.portfolio.max_position_size_pct = best_params['max_position_size_pct']
    elif 'max_pos_size' in best_params:  # Обратная совместимость
        cfg.portfolio.max_position_size_pct = best_params['max_pos_size']
    
    if 'max_active_pos' in best_params:
        cfg.portfolio.max_active_positions = int(best_params['max_active_pos'])
    
    # Сохраняем в YAML (используем model_dump для Pydantic v2)
    with open(output_path, 'w') as f:
        if hasattr(cfg, 'model_dump'):
            yaml.dump(cfg.model_dump(), f, default_flow_style=False, allow_unicode=True)
        else:
            yaml.dump(cfg.dict(), f, default_flow_style=False, allow_unicode=True)

if __name__ == "__main__":
    main()
