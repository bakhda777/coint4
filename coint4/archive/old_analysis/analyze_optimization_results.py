#!/usr/bin/env python3
"""
Анализ результатов оптимизации из Optuna баз данных.
Сравнение результатов с ослабленными и усиленными фильтрами.
"""

import optuna
import pandas as pd
from pathlib import Path
import sqlite3
from datetime import datetime

def analyze_study(db_path: str, study_name: str = None) -> dict:
    """
    Анализирует результаты одного исследования.
    
    Args:
        db_path: Путь к SQLite базе данных
        study_name: Имя исследования (если None, берется из базы)
        
    Returns:
        Словарь с метриками
    """
    if not Path(db_path).exists():
        print(f"❌ База данных не найдена: {db_path}")
        return None
    
    # Получаем имя исследования из базы если не указано
    if not study_name:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT study_name FROM studies LIMIT 1")
        result = cursor.fetchone()
        conn.close()
        if result:
            study_name = result[0]
        else:
            print(f"❌ Не удалось получить имя исследования из {db_path}")
            return None
    
    storage = f"sqlite:///{db_path}"
    
    try:
        study = optuna.load_study(study_name=study_name, storage=storage)
    except Exception as e:
        print(f"❌ Ошибка загрузки исследования: {e}")
        return None
    
    completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    
    if not completed_trials:
        print(f"❌ Нет завершенных trials в {study_name}")
        return None
    
    # Извлекаем метрики
    sharpe_values = []
    drawdown_values = []
    win_rates = []
    trade_counts = []
    
    for trial in completed_trials:
        if trial.value is not None:
            metrics = trial.user_attrs.get('metrics', {})
            if metrics:
                sharpe = metrics.get('sharpe')
                if sharpe is not None:
                    sharpe_values.append(sharpe)
                dd = metrics.get('max_drawdown')
                if dd is not None:
                    drawdown_values.append(dd)
                wr = metrics.get('win_rate')
                if wr is not None:
                    win_rates.append(wr)
                trades = metrics.get('total_trades')
                if trades is not None:
                    trade_counts.append(trades)
    
    if not sharpe_values:
        print(f"⚠️ Нет Sharpe ratio в {study_name}")
        return None
    
    results = {
        'study_name': study_name,
        'db_path': db_path,
        'total_trials': len(study.trials),
        'completed_trials': len(completed_trials),
        'best_value': study.best_value if hasattr(study, 'best_trial') else None,
        'best_sharpe': max(sharpe_values) if sharpe_values else None,
        'avg_sharpe': sum(sharpe_values) / len(sharpe_values) if sharpe_values else None,
        'positive_sharpe_count': len([s for s in sharpe_values if s > 0]),
        'sharpe_gt_1_count': len([s for s in sharpe_values if s > 1]),
        'avg_drawdown': sum(drawdown_values) / len(drawdown_values) if drawdown_values else None,
        'avg_win_rate': sum(win_rates) / len(win_rates) if win_rates else None,
        'avg_trades': sum(trade_counts) / len(trade_counts) if trade_counts else None,
        'best_params': study.best_params if hasattr(study, 'best_params') else None
    }
    
    return results

def compare_studies():
    """Сравнивает результаты разных оптимизаций."""
    
    print("="*60)
    print("📊 АНАЛИЗ РЕЗУЛЬТАТОВ ОПТИМИЗАЦИИ")
    print("="*60)
    
    # Анализируем основные базы
    studies_to_analyze = [
        ("outputs/studies/pairs_strategy_v1.db", "pairs_strategy_v1"),
        ("outputs/studies/strict_optimization.db", "strict_optimization"),
        ("outputs/studies/full_optimization_2025.db", None),
        ("outputs/studies/ultra_optimization_numba_100_trials.db", None),
    ]
    
    results_list = []
    
    for db_path, study_name in studies_to_analyze:
        if Path(db_path).exists():
            print(f"\n📂 Анализ: {db_path}")
            results = analyze_study(db_path, study_name)
            if results:
                results_list.append(results)
                
                print(f"  📈 Лучший Sharpe: {results['best_sharpe']:.3f}" if results['best_sharpe'] else "  ❌ Нет Sharpe")
                print(f"  📊 Средний Sharpe: {results['avg_sharpe']:.3f}" if results['avg_sharpe'] else "")
                print(f"  ✅ Sharpe > 0: {results['positive_sharpe_count']}/{results['completed_trials']}")
                print(f"  🎯 Sharpe > 1: {results['sharpe_gt_1_count']}/{results['completed_trials']}")
                print(f"  📉 Средний Drawdown: {results['avg_drawdown']:.2%}" if results['avg_drawdown'] else "")
                print(f"  🎲 Средний Win Rate: {results['avg_win_rate']:.2%}" if results['avg_win_rate'] else "")
                print(f"  📊 Среднее число сделок: {results['avg_trades']:.0f}" if results['avg_trades'] else "")
    
    if not results_list:
        print("\n❌ Не удалось проанализировать ни одно исследование")
        return
    
    # Сравнительная таблица
    print("\n" + "="*60)
    print("📊 СРАВНИТЕЛЬНАЯ ТАБЛИЦА")
    print("="*60)
    
    df = pd.DataFrame(results_list)
    
    # Выбираем ключевые метрики для сравнения
    comparison_cols = [
        'study_name', 
        'completed_trials',
        'best_sharpe',
        'avg_sharpe',
        'sharpe_gt_1_count',
        'avg_drawdown',
        'avg_win_rate',
        'avg_trades'
    ]
    
    # Фильтруем только существующие колонки
    comparison_cols = [col for col in comparison_cols if col in df.columns]
    
    if comparison_cols:
        comparison_df = df[comparison_cols].copy()
        
        # Форматирование
        if 'best_sharpe' in comparison_df.columns:
            comparison_df['best_sharpe'] = comparison_df['best_sharpe'].apply(lambda x: f"{x:.3f}" if pd.notna(x) else "N/A")
        if 'avg_sharpe' in comparison_df.columns:
            comparison_df['avg_sharpe'] = comparison_df['avg_sharpe'].apply(lambda x: f"{x:.3f}" if pd.notna(x) else "N/A")
        if 'avg_drawdown' in comparison_df.columns:
            comparison_df['avg_drawdown'] = comparison_df['avg_drawdown'].apply(lambda x: f"{x:.2%}" if pd.notna(x) else "N/A")
        if 'avg_win_rate' in comparison_df.columns:
            comparison_df['avg_win_rate'] = comparison_df['avg_win_rate'].apply(lambda x: f"{x:.2%}" if pd.notna(x) else "N/A")
        if 'avg_trades' in comparison_df.columns:
            comparison_df['avg_trades'] = comparison_df['avg_trades'].apply(lambda x: f"{x:.0f}" if pd.notna(x) else "N/A")
        
        print(comparison_df.to_string(index=False))
    
    # Выводы
    print("\n" + "="*60)
    print("📝 ВЫВОДЫ")
    print("="*60)
    
    # Находим лучшее исследование по Sharpe
    best_study_idx = df['best_sharpe'].idxmax() if 'best_sharpe' in df.columns and not df['best_sharpe'].isna().all() else None
    
    if best_study_idx is not None:
        best_study = df.loc[best_study_idx]
        print(f"🏆 Лучшее исследование: {best_study['study_name']}")
        print(f"   Sharpe: {best_study['best_sharpe']:.3f}")
        
        if best_study['best_sharpe'] >= 1.0:
            print("   ✅ ЦЕЛЬ ДОСТИГНУТА: Sharpe >= 1.0")
        else:
            print(f"   ⚠️ До цели (Sharpe >= 1.0) осталось: {1.0 - best_study['best_sharpe']:.3f}")
        
        if best_study['best_params']:
            print("\n📊 Лучшие параметры:")
            for param, value in best_study['best_params'].items():
                print(f"   {param}: {value}")
    else:
        print("❌ Не удалось определить лучшее исследование")

if __name__ == "__main__":
    compare_studies()