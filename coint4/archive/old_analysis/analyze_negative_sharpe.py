#!/usr/bin/env python3
"""
Анализ причин отрицательных Sharpe ratio в оптимизациях.
"""

import sys
from pathlib import Path
import optuna
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent / "src"))


def analyze_negative_sharpe():
    """Анализирует, почему получаются отрицательные Sharpe."""
    
    print("="*60)
    print("🔍 АНАЛИЗ ОТРИЦАТЕЛЬНЫХ SHARPE RATIO")
    print("="*60)
    
    # Загружаем результаты последней оптимизации с отрицательными Sharpe
    db_path = "outputs/studies/ultra_optimization_numba_100_trials.db"
    
    if not Path(db_path).exists():
        print(f"❌ База данных не найдена: {db_path}")
        return
    
    storage = f"sqlite:///{db_path}"
    
    try:
        # Загружаем исследование
        study = optuna.load_study(
            study_name="ultra_optimization_numba_100_trials",
            storage=storage
        )
        
        print(f"📊 Загружено trials: {len(study.trials)}")
        
        # Анализируем завершенные trials
        completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        
        if not completed:
            print("❌ Нет завершенных trials")
            return
        
        # Извлекаем метрики
        results = []
        for trial in completed:
            if trial.value is not None:
                metrics = trial.user_attrs.get('metrics', {})
                results.append({
                    'trial': trial.number,
                    'value': trial.value,
                    'sharpe': metrics.get('sharpe', trial.value),
                    'trades': metrics.get('total_trades', 0),
                    'drawdown': metrics.get('max_drawdown', 0),
                    'win_rate': metrics.get('win_rate', 0),
                    'zscore_threshold': trial.params.get('zscore_threshold', 0),
                    'zscore_exit': trial.params.get('zscore_exit', 0),
                    'rolling_window': trial.params.get('rolling_window', 0),
                    'normalization': trial.params.get('normalization_method', 'unknown')
                })
        
        df = pd.DataFrame(results)
        
        print(f"\n📊 СТАТИСТИКА SHARPE RATIO:")
        print(f"  Минимум: {df['sharpe'].min():.2f}")
        print(f"  Максимум: {df['sharpe'].max():.2f}")
        print(f"  Среднее: {df['sharpe'].mean():.2f}")
        print(f"  Медиана: {df['sharpe'].median():.2f}")
        
        # Анализ по количеству сделок
        print(f"\n📊 КОЛИЧЕСТВО СДЕЛОК:")
        print(f"  Минимум: {df['trades'].min()}")
        print(f"  Максимум: {df['trades'].max()}")
        print(f"  Среднее: {df['trades'].mean():.1f}")
        
        # Корреляция между параметрами и Sharpe
        print(f"\n📊 КОРРЕЛЯЦИЯ С SHARPE:")
        numeric_cols = ['trades', 'drawdown', 'win_rate', 'zscore_threshold', 
                       'zscore_exit', 'rolling_window']
        for col in numeric_cols:
            if col in df.columns:
                corr = df['sharpe'].corr(df[col])
                print(f"  {col}: {corr:.3f}")
        
        # Топ-5 лучших и худших
        print(f"\n📊 ТОП-5 ЛУЧШИХ:")
        top5 = df.nlargest(5, 'sharpe')
        for _, row in top5.iterrows():
            print(f"  Trial {row['trial']}: Sharpe={row['sharpe']:.2f}, "
                  f"Trades={row['trades']}, Z-in={row['zscore_threshold']:.2f}, "
                  f"Z-out={row['zscore_exit']:.2f}")
        
        print(f"\n📊 ТОП-5 ХУДШИХ:")
        bottom5 = df.nsmallest(5, 'sharpe')
        for _, row in bottom5.iterrows():
            print(f"  Trial {row['trial']}: Sharpe={row['sharpe']:.2f}, "
                  f"Trades={row['trades']}, Z-in={row['zscore_threshold']:.2f}, "
                  f"Z-out={row['zscore_exit']:.2f}")
        
        # Анализ нормализации
        print(f"\n📊 МЕТОДЫ НОРМАЛИЗАЦИИ:")
        norm_stats = df.groupby('normalization')['sharpe'].agg(['mean', 'count'])
        print(norm_stats)
        
        # ВЫВОДЫ
        print("\n" + "="*60)
        print("💡 ВЫВОДЫ:")
        print("="*60)
        
        all_negative = (df['sharpe'] < 0).all()
        if all_negative:
            print("❌ ВСЕ Sharpe отрицательные!")
            
            # Анализ причин
            avg_trades = df['trades'].mean()
            if avg_trades < 50:
                print("  → Слишком мало сделок (< 50)")
            
            avg_win_rate = df['win_rate'].mean()
            if avg_win_rate < 0.4:
                print(f"  → Низкий win rate ({avg_win_rate:.1%})")
            
            if df['normalization'].value_counts().get('minmax', 0) > 0:
                print("  → Используется minmax нормализация (lookahead bias!)")
            
            print("\n📝 РЕКОМЕНДАЦИИ:")
            print("  1. Использовать ТОЛЬКО rolling_zscore нормализацию")
            print("  2. Снизить пороги входа (zscore_threshold < 1.5)")
            print("  3. Увеличить гистерезис (zscore_exit > 0)")
            print("  4. Ослабить фильтры для большего числа пар")
            print("  5. Проверить качество данных")
        
    except Exception as e:
        print(f"❌ Ошибка анализа: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    analyze_negative_sharpe()