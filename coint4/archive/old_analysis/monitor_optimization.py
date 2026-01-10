#!/usr/bin/env python3
"""
Мониторинг прогресса оптимизации Optuna в реальном времени.
"""

import sys
import time
import optuna
import argparse
from pathlib import Path
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def monitor_optimization(db_path: str, study_name: str, refresh_interval: int = 5):
    """
    Мониторинг прогресса оптимизации.
    
    Args:
        db_path: Путь к базе данных Optuna
        study_name: Имя исследования
        refresh_interval: Интервал обновления в секундах
    """
    if not Path(db_path).exists():
        logger.error(f"База данных не найдена: {db_path}")
        return
    
    storage = f"sqlite:///{db_path}"
    
    print("=" * 60)
    print(f"📊 МОНИТОРИНГ ОПТИМИЗАЦИИ: {study_name}")
    print("=" * 60)
    print(f"База данных: {db_path}")
    print(f"Обновление каждые {refresh_interval} секунд")
    print("Нажмите Ctrl+C для выхода")
    print("=" * 60)
    
    last_n_trials = 0
    
    try:
        while True:
            try:
                study = optuna.load_study(study_name=study_name, storage=storage)
                
                n_trials = len(study.trials)
                completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
                failed = [t for t in study.trials if t.state == optuna.trial.TrialState.FAIL]
                pruned = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
                running = [t for t in study.trials if t.state == optuna.trial.TrialState.RUNNING]
                
                # Очистка экрана для обновления
                print("\033[2J\033[H")  # ANSI escape codes для очистки экрана
                
                print("=" * 60)
                print(f"📊 МОНИТОРИНГ: {study_name}")
                print(f"⏰ Время: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                print("=" * 60)
                
                print(f"\n📈 ПРОГРЕСС:")
                print(f"  Всего trials: {n_trials}")
                print(f"  ✅ Завершено: {len(completed)}")
                print(f"  ⚡ Выполняется: {len(running)}")
                print(f"  ✂️ Отсечено: {len(pruned)}")
                print(f"  ❌ Ошибки: {len(failed)}")
                
                if n_trials > last_n_trials:
                    print(f"  🆕 Новых с прошлой проверки: {n_trials - last_n_trials}")
                    last_n_trials = n_trials
                
                if completed:
                    values = [t.value for t in completed if t.value is not None]
                    if values:
                        print(f"\n📊 СТАТИСТИКА:")
                        print(f"  Лучший Sharpe: {max(values):.4f}")
                        print(f"  Худший Sharpe: {min(values):.4f}")
                        print(f"  Средний Sharpe: {sum(values)/len(values):.4f}")
                        
                        positive = [v for v in values if v > 0]
                        if positive:
                            print(f"  Положительных: {len(positive)} ({len(positive)/len(values)*100:.1f}%)")
                        
                        if study.best_trial:
                            print(f"\n🏆 ЛУЧШИЙ РЕЗУЛЬТАТ:")
                            print(f"  Trial #{study.best_trial.number}")
                            print(f"  Sharpe: {study.best_value:.4f}")
                            
                            if hasattr(study.best_trial, 'user_attrs') and 'metrics' in study.best_trial.user_attrs:
                                metrics = study.best_trial.user_attrs['metrics']
                                print(f"  Trades: {metrics.get('total_trades', 'N/A')}")
                                print(f"  Win Rate: {metrics.get('win_rate', 0):.1%}")
                                print(f"  Max DD: {metrics.get('max_drawdown', 0):.2%}")
                            
                            print(f"\n📊 ПАРАМЕТРЫ:")
                            for key, value in study.best_params.items():
                                if isinstance(value, float):
                                    print(f"  {key}: {value:.3f}")
                                else:
                                    print(f"  {key}: {value}")
                
                # Последние 5 trials
                if study.trials:
                    print(f"\n📜 ПОСЛЕДНИЕ TRIALS:")
                    for trial in study.trials[-5:]:
                        status = "✅" if trial.state == optuna.trial.TrialState.COMPLETE else \
                                "✂️" if trial.state == optuna.trial.TrialState.PRUNED else \
                                "⚡" if trial.state == optuna.trial.TrialState.RUNNING else "❌"
                        value_str = f"{trial.value:.4f}" if trial.value is not None else "N/A"
                        print(f"  {status} Trial #{trial.number}: {value_str}")
                
                print(f"\n🔄 Обновление через {refresh_interval} секунд...")
                
            except Exception as e:
                print(f"⚠️ Ошибка при чтении study: {e}")
            
            time.sleep(refresh_interval)
            
    except KeyboardInterrupt:
        print("\n👋 Мониторинг остановлен")


def main():
    """Главная функция."""
    parser = argparse.ArgumentParser(description="Мониторинг оптимизации Optuna")
    parser.add_argument("--db", type=str, default="outputs/studies/fixed_normalization.db",
                      help="Путь к базе данных")
    parser.add_argument("--study", type=str, default="fixed_normalization",
                      help="Имя исследования")
    parser.add_argument("--interval", type=int, default=5,
                      help="Интервал обновления в секундах")
    
    args = parser.parse_args()
    
    monitor_optimization(args.db, args.study, args.interval)


if __name__ == "__main__":
    main()