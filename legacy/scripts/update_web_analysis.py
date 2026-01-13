#!/usr/bin/env python3
"""
Скрипт для обновления данных веб-анализа из результатов оптимизации.
Читает результаты из Optuna БД и обновляет JavaScript файлы.
"""

import sys
from pathlib import Path
import json
import yaml
import optuna
import pandas as pd
from datetime import datetime

# Добавляем корневую директорию проекта в PYTHONPATH
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def load_optuna_results(study_name: str, storage_path: str):
    """Загружает результаты из Optuna БД."""
    try:
        storage_url = f"sqlite:///{storage_path}"
        study = optuna.load_study(study_name=study_name, storage=storage_url)
        
        if not study.trials:
            print(f"❌ Нет trials в study '{study_name}'")
            return None
        
        best_trial = study.best_trial
        best_params = study.best_params
        best_value = study.best_value
        
        print(f"✅ Загружены результаты Optuna:")
        print(f"   Study: {study_name}")
        print(f"   Trials: {len(study.trials)}")
        print(f"   Best value: {best_value:.4f}")
        
        return {
            'study_name': study_name,
            'best_trial': best_trial,
            'best_params': best_params,
            'best_value': best_value,
            'total_trials': len(study.trials),
            'completed_trials': len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
        }
        
    except Exception as e:
        print(f"❌ Ошибка загрузки Optuna результатов: {e}")
        return None

def load_validation_results(results_dir: str = "results"):
    """Загружает результаты валидации из CSV файлов."""
    try:
        results_path = Path(results_dir)
        
        # Ищем последние результаты
        metrics_files = list(results_path.glob("strategy_metrics.csv"))
        if not metrics_files:
            print(f"❌ Нет файлов strategy_metrics.csv в {results_dir}")
            return None
        
        # Берем самый новый файл
        latest_metrics = max(metrics_files, key=lambda x: x.stat().st_mtime)
        
        metrics_df = pd.read_csv(latest_metrics)
        if metrics_df.empty:
            print(f"❌ Пустой файл метрик: {latest_metrics}")
            return None
        
        metrics = metrics_df.iloc[0].to_dict()
        
        print(f"✅ Загружены результаты валидации:")
        print(f"   Файл: {latest_metrics}")
        print(f"   Sharpe (abs): {metrics.get('sharpe_ratio_abs', 'N/A')}")
        print(f"   Trades: {metrics.get('total_trades', 'N/A')}")
        print(f"   Pairs: {metrics.get('total_pairs_traded', 'N/A')}")
        print(f"   PnL: {metrics.get('total_pnl', 'N/A')}")

        return metrics
        
    except Exception as e:
        print(f"❌ Ошибка загрузки результатов валидации: {e}")
        return None

def update_analysis_js(optuna_results, validation_results):
    """Обновляет analysis.js с новыми данными."""

    analysis_js_path = Path("src/web_analysis/analysis.js")

    if not analysis_js_path.exists():
        print(f"❌ Файл не найден: {analysis_js_path}")
        return False

    try:
        # ВАЖНО: У нас нет отдельных данных для оптимизации и валидации
        # Optuna дает только лучший Sharpe, но не количество сделок
        # CSV содержит результаты валидации (walk-forward)

        # Формируем данные оптимизации (только то, что знаем из Optuna)
        optimization_data = {
            'sharpe_ratio': round(optuna_results['best_value'], 4),
            'trades_count': "N/A",  # Optuna не сохраняет количество сделок
            'pairs_count': "N/A",   # Optuna не сохраняет количество пар
            'period': f"Optuna optimization ({optuna_results['completed_trials']} trials)",
            'parameters': {
                'zscore_entry_threshold': round(optuna_results['best_params'].get('zscore_threshold', 1.0), 4),
                'zscore_exit': round(optuna_results['best_params'].get('zscore_exit', 0.0), 4),
                'max_active_positions': int(optuna_results['best_params'].get('max_active_positions', 10)),
                'risk_per_position_pct': round(optuna_results['best_params'].get('risk_per_position_pct', 0.02), 4),
                'max_position_size_pct': round(optuna_results['best_params'].get('max_position_size_pct', 0.05), 4),
                'stop_loss_multiplier': round(optuna_results['best_params'].get('stop_loss_multiplier', 3.0), 3),
                'time_stop_multiplier': round(optuna_results['best_params'].get('time_stop_multiplier', 2.0), 3)
            }
        }

        # Формируем данные валидации (из walk-forward результатов)
        validation_data = {
            'sharpe_ratio': round(validation_results.get('sharpe_ratio_abs', 0.0), 4) if validation_results else 0.0,
            'trades_count': int(validation_results.get('total_trades', 0)) if validation_results else 0,
            'pnl_usd': round(validation_results.get('total_pnl', 0.0), 2) if validation_results else 0.0,
            'period': f"Walk-forward validation ({int(validation_results.get('total_days', 0))} days)" if validation_results else "No validation data"
        }

        # Универсальная замена с регулярными выражениями
        import re

        with open(analysis_js_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Заменяем sharpe_ratio в optimizationResults
        content = re.sub(
            r'"sharpe_ratio": [0-9.-]+,',
            f'"sharpe_ratio": {optimization_data["sharpe_ratio"]},',
            content
        )

        # Заменяем trades_count в optimizationResults
        trades_count_str = f'"{optimization_data["trades_count"]}"' if optimization_data["trades_count"] == "N/A" else str(optimization_data["trades_count"])
        content = re.sub(
            r'"trades_count": (?:"N/A"|\d+),',
            f'"trades_count": {trades_count_str},',
            content
        )

        # Заменяем pairs_count в optimizationResults
        pairs_count_str = f'"{optimization_data["pairs_count"]}"' if optimization_data["pairs_count"] == "N/A" else str(optimization_data["pairs_count"])
        content = re.sub(
            r'"pairs_count": (?:"N/A"|\d+),',
            f'"pairs_count": {pairs_count_str},',
            content
        )

        # Заменяем period в optimizationResults
        content = re.sub(
            r'"period": "[^"]*",',
            f'"period": "{optimization_data["period"]}",',
            content
        )

        # Заменяем весь блок параметров целиком для правильного форматирования
        parameters_block = f'''            "parameters": {{
                "zscore_entry_threshold": {optimization_data["parameters"]["zscore_entry_threshold"]},
                "zscore_exit": {optimization_data["parameters"]["zscore_exit"]},
                "max_active_positions": {optimization_data["parameters"]["max_active_positions"]},
                "risk_per_position_pct": {optimization_data["parameters"]["risk_per_position_pct"]},
                "max_position_size_pct": {optimization_data["parameters"]["max_position_size_pct"]},
                "stop_loss_multiplier": {optimization_data["parameters"]["stop_loss_multiplier"]},
                "time_stop_multiplier": {optimization_data["parameters"]["time_stop_multiplier"]}
            }}'''

        # Заменяем весь блок параметров
        content = re.sub(
            r'"parameters": \{[^}]*\}',
            parameters_block,
            content,
            flags=re.DOTALL
        )

        # Заменяем данные валидации с регулярными выражениями
        content = re.sub(
            r'sharpe_ratio: [0-9.-]+,',
            f'sharpe_ratio: {validation_data["sharpe_ratio"]},',
            content
        )
        content = re.sub(
            r'trades_count: \d+,',
            f'trades_count: {validation_data["trades_count"]},',
            content
        )
        content = re.sub(
            r'pnl_usd: [0-9.-]+,',
            f'pnl_usd: {validation_data["pnl_usd"]},',
            content
        )
        content = re.sub(
            r"period: '[^']*',",
            f"period: '{validation_data['period']}',",
            content
        )
        
        print(f"✅ Заменены данные оптимизации")
        print(f"✅ Заменены данные валидации")
        
        # Сохраняем обновленный файл
        with open(analysis_js_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print(f"✅ Обновлен файл: {analysis_js_path}")
        return True
        
    except Exception as e:
        print(f"❌ Ошибка обновления analysis.js: {e}")
        return False

def main():
    """Главная функция."""
    
    print("🔄 ОБНОВЛЕНИЕ ДАННЫХ ВЕБ-АНАЛИЗА")
    print("=" * 50)
    
    # Параметры по умолчанию
    study_name = "pairs_strategy_v1"
    storage_path = "outputs/studies/pairs_strategy_v1.db"
    results_dir = "results"
    
    # Проверяем аргументы командной строки
    if len(sys.argv) > 1:
        study_name = sys.argv[1]
    if len(sys.argv) > 2:
        storage_path = sys.argv[2]
    if len(sys.argv) > 3:
        results_dir = sys.argv[3]
    
    print(f"📊 Параметры:")
    print(f"   Study: {study_name}")
    print(f"   Storage: {storage_path}")
    print(f"   Results dir: {results_dir}")
    
    # Загружаем данные
    optuna_results = load_optuna_results(study_name, storage_path)
    if not optuna_results:
        print("❌ Не удалось загрузить результаты Optuna")
        return False
    
    validation_results = load_validation_results(results_dir)
    # Валидация не обязательна
    
    # Обновляем JavaScript файлы
    success = update_analysis_js(optuna_results, validation_results)
    
    if success:
        print(f"\n✅ ВЕБ-АНАЛИЗ ОБНОВЛЕН!")
        print(f"🌐 Откройте: file://{project_root}/src/web_analysis/index.html")
        print(f"🔄 Обновите страницу в браузере (Ctrl+F5)")
    else:
        print(f"\n❌ Ошибка обновления веб-анализа")
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
