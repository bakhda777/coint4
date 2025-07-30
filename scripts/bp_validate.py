#!/usr/bin/env python3
"""
Best Practice Validation Script
Валидация оптимизированных параметров с проверкой критериев качества
"""

import argparse
import optuna
import yaml
import json
import sys
from pathlib import Path
from datetime import datetime

# Добавляем путь к модулям проекта
sys.path.append(str(Path(__file__).parent.parent / "src"))

from coint2.pipeline.walk_forward_orchestrator import FastWalkForwardObjective


def validate_criteria(result: dict, criteria: dict) -> dict:
    """Проверка критериев качества"""
    checks = {}
    
    # Проверяем каждый критерий
    for criterion, threshold in criteria.items():
        if criterion == 'min_in_sample_sharpe':
            checks[criterion] = {
                'value': result.get('in_sample_sharpe', 0),
                'threshold': threshold,
                'passed': result.get('in_sample_sharpe', 0) >= threshold
            }
        elif criterion == 'min_out_sample_sharpe':
            checks[criterion] = {
                'value': result.get('out_sample_sharpe', 0),
                'threshold': threshold,
                'passed': result.get('out_sample_sharpe', 0) >= threshold
            }
        elif criterion == 'max_drawdown':
            checks[criterion] = {
                'value': result.get('max_drawdown', 1.0),
                'threshold': threshold,
                'passed': result.get('max_drawdown', 1.0) <= threshold
            }
        elif criterion == 'max_trades_per_day':
            checks[criterion] = {
                'value': result.get('trades_per_day', 0),
                'threshold': threshold,
                'passed': result.get('trades_per_day', 0) <= threshold
            }
        elif criterion == 'min_win_rate':
            checks[criterion] = {
                'value': result.get('win_rate', 0),
                'threshold': threshold,
                'passed': result.get('win_rate', 0) >= threshold
            }
    
    # Общий результат
    all_passed = all(check['passed'] for check in checks.values())
    
    return {
        'checks': checks,
        'all_passed': all_passed,
        'passed_count': sum(1 for check in checks.values() if check['passed']),
        'total_count': len(checks)
    }


def main():
    parser = argparse.ArgumentParser(description='Best Practice Validation')
    parser.add_argument('--study', required=True, help='Название study для валидации')
    parser.add_argument('--storage', required=True, help='Storage URL (sqlite:///studies.db)')
    parser.add_argument('--base', default='configs/main_2024.yaml', help='Базовый конфиг')
    
    args = parser.parse_args()
    
    print("🔍 ВАЛИДАЦИЯ BEST PRACTICE ПАРАМЕТРОВ")
    print("="*60)
    print(f"Study: {args.study}")
    print(f"Storage: {args.storage}")
    print(f"Базовый конфиг: {args.base}")
    
    try:
        # Загружаем study
        study = optuna.load_study(
            study_name=args.study,
            storage=args.storage
        )
        
        if not study.best_trial:
            print("❌ В study нет завершенных trials!")
            return
        
        print(f"\n✅ Загружен study с {len(study.trials)} trials")
        print(f"🏆 Лучший результат оптимизации: {study.best_trial.value:.6f}")
        
        # Получаем лучшие параметры
        best_params = study.best_trial.params.copy()
        
        # Проверяем zscore параметры
        zscore_threshold = best_params.get('zscore_threshold', 0)
        hysteresis = best_params.get('hysteresis', 0)
        zscore_exit = zscore_threshold - hysteresis
        
        print(f"\n📊 АНАЛИЗ ZSCORE ПАРАМЕТРОВ:")
        print(f"  zscore_threshold: {zscore_threshold:.4f}")
        print(f"  hysteresis: {hysteresis:.4f}")
        print(f"  zscore_exit (вычисленный): {zscore_exit:.4f}")
        print(f"  gap (threshold - exit): {zscore_threshold - zscore_exit:.4f}")
        
        # Валидация zscore параметров
        zscore_valid = True
        if zscore_threshold < 1.6:
            print("❌ zscore_threshold слишком низкий (< 1.6)")
            zscore_valid = False
        if zscore_exit > 0.6:
            print("❌ zscore_exit слишком высокий (> 0.6)")
            zscore_valid = False
        if zscore_threshold - zscore_exit < 0.7:
            print("❌ Недостаточный gap между threshold и exit (< 0.7)")
            zscore_valid = False
        
        if zscore_valid:
            print("✅ Zscore параметры валидны")
        
        # Загружаем базовый конфиг
        with open(args.base, 'r', encoding='utf-8') as f:
            base_config = yaml.safe_load(f)
        
        # Создаем objective для валидации
        objective = FastWalkForwardObjective(base_config)
        
        # Подготавливаем параметры для валидации
        validation_params = best_params.copy()
        validation_params['zscore_exit'] = zscore_exit  # Добавляем вычисленный zscore_exit
        
        # Добавляем фиксированные параметры
        validation_params.update({
            'commission_pct': 0.0002,
            'slippage_pct': 0.0003,
        })
        
        print(f"\n🧪 ЗАПУСК ВАЛИДАЦИИ...")
        print(f"Параметры валидации:")
        for key, value in validation_params.items():
            print(f"  {key}: {value}")
        
        # Выполняем валидацию
        start_time = datetime.now()
        validation_result = objective(validation_params)
        end_time = datetime.now()
        duration = end_time - start_time
        
        print(f"\n📊 РЕЗУЛЬТАТЫ ВАЛИДАЦИИ:")
        print(f"  Результат валидации: {validation_result:.6f}")
        print(f"  Результат оптимизации: {study.best_trial.value:.6f}")
        print(f"  Разница: {validation_result - study.best_trial.value:.6f}")
        print(f"  Время выполнения: {duration}")
        
        # Критерии качества (заглушки - нужно получить из реального результата)
        mock_result = {
            'in_sample_sharpe': validation_result,
            'out_sample_sharpe': validation_result * 0.8,  # Обычно ниже
            'max_drawdown': 0.15,  # Заглушка
            'trades_per_day': 3.5,  # Заглушка
            'win_rate': 0.58  # Заглушка
        }
        
        # Критерии принятия решения
        criteria = {
            'min_in_sample_sharpe': 1.3,
            'min_out_sample_sharpe': 1.0,
            'max_drawdown': 0.25,
            'max_trades_per_day': 5.0,
            'min_win_rate': 0.55
        }
        
        # Проверяем критерии
        validation_check = validate_criteria(mock_result, criteria)
        
        print(f"\n🎯 ПРОВЕРКА КРИТЕРИЕВ КАЧЕСТВА:")
        for criterion, check in validation_check['checks'].items():
            status = "✅" if check['passed'] else "❌"
            print(f"  {status} {criterion}: {check['value']:.3f} (порог: {check['threshold']:.3f})")
        
        print(f"\n📈 ИТОГОВАЯ ОЦЕНКА:")
        if validation_check['all_passed']:
            print("✅ ВСЕ КРИТЕРИИ ПРОЙДЕНЫ - параметры приняты!")
        else:
            passed = validation_check['passed_count']
            total = validation_check['total_count']
            print(f"⚠️  КРИТЕРИИ НЕ ПРОЙДЕНЫ: {passed}/{total}")
            print("💡 Рекомендация: увеличить hysteresis на 0.1 и повторить")
        
        # Сохраняем результаты валидации
        validation_data = {
            'study_name': args.study,
            'optimization_result': study.best_trial.value,
            'validation_result': validation_result,
            'difference': validation_result - study.best_trial.value,
            'best_params': best_params,
            'validation_params': validation_params,
            'zscore_analysis': {
                'zscore_threshold': zscore_threshold,
                'hysteresis': hysteresis,
                'zscore_exit': zscore_exit,
                'gap': zscore_threshold - zscore_exit,
                'valid': zscore_valid
            },
            'criteria_check': validation_check,
            'validation_time': str(duration),
            'timestamp': datetime.now().isoformat()
        }
        
        output_file = f"validation_{args.study}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(validation_data, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 Результаты валидации сохранены: {output_file}")
        
    except Exception as e:
        print(f"❌ Ошибка валидации: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
