#!/usr/bin/env python
"""
Анализ всех существующих баз Optuna для извлечения лучших параметров.
"""

import sys
sys.path.insert(0, "src")
import warnings
warnings.filterwarnings("ignore")

import optuna
import sqlite3
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime

def analyze_study_db(db_path):
    """Анализирует одну базу данных Optuna"""
    
    results = {
        'db_name': db_path.name,
        'size_mb': db_path.stat().st_size / 1024 / 1024,
        'modified': datetime.fromtimestamp(db_path.stat().st_mtime)
    }
    
    try:
        # Подключаемся к базе
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        
        # Получаем список исследований
        cursor.execute("SELECT study_id, study_name FROM studies")
        studies = cursor.fetchall()
        
        if not studies:
            results['status'] = 'empty'
            conn.close()
            return results
        
        # Анализируем первое исследование
        study_id = studies[0][0]
        study_name = studies[0][1]
        
        results['study_name'] = study_name
        
        # Получаем trials
        cursor.execute("""
            SELECT trial_id, state, value, params, user_attrs
            FROM trials 
            WHERE study_id = ?
        """, (study_id,))
        
        trials = cursor.fetchall()
        results['total_trials'] = len(trials)
        
        # Фильтруем завершенные trials
        completed = [t for t in trials if t[1] == 'COMPLETE']
        results['completed_trials'] = len(completed)
        
        if completed:
            # Извлекаем значения
            values = []
            for trial in completed:
                try:
                    # Пробуем разные способы получения value
                    if trial[2] is not None:
                        values.append(float(trial[2]))
                except:
                    pass
            
            if values:
                results['best_sharpe'] = max(values)
                results['mean_sharpe'] = np.mean(values)
                results['positive_sharpe'] = sum(1 for v in values if v > 0)
                results['above_1_sharpe'] = sum(1 for v in values if v > 1.0)
                
                # Лучшие параметры
                best_idx = values.index(max(values))
                best_trial = completed[best_idx]
                
                # Пытаемся распарсить параметры
                try:
                    import json
                    params = json.loads(best_trial[3]) if best_trial[3] else {}
                    results['best_params'] = params
                except:
                    results['best_params'] = {}
        
        conn.close()
        results['status'] = 'success'
        
    except Exception as e:
        results['status'] = 'error'
        results['error'] = str(e)
    
    return results

def main():
    """Главная функция анализа"""
    
    print("=" * 80)
    print("📊 АНАЛИЗ ВСЕХ БАЗ OPTUNA")
    print("=" * 80)
    
    # Находим все базы
    db_files = list(Path("outputs/studies").glob("*.db"))
    
    if not db_files:
        print("❌ Не найдено баз данных Optuna")
        return
    
    print(f"Найдено баз данных: {len(db_files)}\n")
    
    # Анализируем каждую базу
    all_results = []
    
    for db_path in sorted(db_files, key=lambda x: x.stat().st_size, reverse=True):
        print(f"📁 Анализ {db_path.name}...")
        result = analyze_study_db(db_path)
        all_results.append(result)
        
        if result['status'] == 'success' and result.get('best_sharpe'):
            print(f"   ✅ Trials: {result['completed_trials']}, "
                  f"Best Sharpe: {result['best_sharpe']:.3f}")
        elif result['status'] == 'empty':
            print(f"   ⚠️ Пустая база")
        else:
            print(f"   ❌ Ошибка: {result.get('error', 'Unknown')}")
    
    # Создаем сводку
    print("\n" + "=" * 80)
    print("📊 СВОДКА РЕЗУЛЬТАТОВ")
    print("=" * 80)
    
    # Фильтруем успешные результаты
    successful = [r for r in all_results if r['status'] == 'success' and r.get('best_sharpe')]
    
    if not successful:
        print("❌ Нет успешных результатов для анализа")
        return
    
    # Сортируем по best_sharpe
    successful.sort(key=lambda x: x.get('best_sharpe', 0), reverse=True)
    
    print("\n🏆 ТОП-5 РЕЗУЛЬТАТОВ:")
    print("-" * 60)
    
    for i, result in enumerate(successful[:5], 1):
        print(f"\n{i}. {result['db_name']}")
        print(f"   Best Sharpe: {result['best_sharpe']:.3f}")
        print(f"   Mean Sharpe: {result.get('mean_sharpe', 0):.3f}")
        print(f"   Completed trials: {result['completed_trials']}")
        print(f"   Positive Sharpe: {result.get('positive_sharpe', 0)}")
        print(f"   Sharpe > 1.0: {result.get('above_1_sharpe', 0)}")
        
        if result.get('best_params'):
            print("   Best params:")
            for key, value in list(result['best_params'].items())[:5]:
                if isinstance(value, float):
                    print(f"     {key}: {value:.3f}")
                else:
                    print(f"     {key}: {value}")
    
    # Общая статистика
    print("\n" + "=" * 80)
    print("📈 ОБЩАЯ СТАТИСТИКА")
    print("=" * 80)
    
    total_trials = sum(r.get('completed_trials', 0) for r in successful)
    all_best_sharpes = [r['best_sharpe'] for r in successful if r.get('best_sharpe')]
    
    print(f"Всего завершенных trials: {total_trials}")
    print(f"Баз с результатами: {len(successful)}")
    
    if all_best_sharpes:
        print(f"Лучший Sharpe overall: {max(all_best_sharpes):.3f}")
        print(f"Средний best Sharpe: {np.mean(all_best_sharpes):.3f}")
        
        above_1 = sum(1 for s in all_best_sharpes if s > 1.0)
        above_07 = sum(1 for s in all_best_sharpes if s > 0.7)
        above_05 = sum(1 for s in all_best_sharpes if s > 0.5)
        
        print(f"\nРаспределение лучших Sharpe:")
        print(f"  > 1.0: {above_1} баз ({above_1/len(all_best_sharpes)*100:.0f}%)")
        print(f"  > 0.7: {above_07} баз ({above_07/len(all_best_sharpes)*100:.0f}%)")
        print(f"  > 0.5: {above_05} баз ({above_05/len(all_best_sharpes)*100:.0f}%)")
    
    # Выводы и рекомендации
    print("\n" + "=" * 80)
    print("💡 ВЫВОДЫ И РЕКОМЕНДАЦИИ")
    print("=" * 80)
    
    max_sharpe = max(all_best_sharpes) if all_best_sharpes else 0
    
    if max_sharpe > 1.0:
        print("✅ Найдены параметры с Sharpe > 1.0!")
        print("Рекомендация: Провести валидацию на out-of-sample данных")
    elif max_sharpe > 0.7:
        print("⚠️ Максимальный Sharpe = {:.3f} (близко к цели)".format(max_sharpe))
        print("Рекомендации:")
        print("  1. Увеличить количество trials до 1000+")
        print("  2. Расширить search space")
        print("  3. Оптимизировать фильтры пар")
    else:
        print("❌ Все Sharpe < 0.7")
        print("Рекомендации:")
        print("  1. Пересмотреть базовую стратегию")
        print("  2. Снизить издержки")
        print("  3. Улучшить качество данных")
    
    # Сохраняем результаты
    output_file = Path("results/optuna_analysis_summary.csv")
    output_file.parent.mkdir(exist_ok=True)
    
    df = pd.DataFrame(successful)
    df.to_csv(output_file, index=False)
    print(f"\n💾 Результаты сохранены: {output_file}")

if __name__ == "__main__":
    main()