#!/usr/bin/env python3
"""
Диагностика проблемы с нулевым количеством сделок в валидации.
Проверка 5 основных гипотез.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import yaml

# Добавляем src в путь для импорта
sys.path.append(str(Path(__file__).parent.parent / "src"))

from coint2.utils.config import load_config

def load_config_for_analysis():
    """Загрузка конфигурации для анализа."""
    try:
        # Пробуем загрузить разные конфигурации
        configs_to_try = [
            'configs/main_2024.yaml',
            'configs/optimized_manual.yaml',
            'configs/analysis_results.yaml'
        ]
        
        for config_path in configs_to_try:
            try:
                cfg = load_config(config_path)
                print(f"✅ Загружена конфигурация: {config_path}")
                return cfg, config_path
            except Exception as e:
                print(f"❌ Ошибка загрузки {config_path}: {e}")
                continue
        
        print("❌ Не удалось загрузить ни одну конфигурацию")
        return None, None
        
    except Exception as e:
        print(f"❌ Критическая ошибка загрузки конфигурации: {e}")
        return None, None

def hypothesis_1_wrong_threshold_usage(cfg, config_path):
    """Гипотеза 1: Используется z_threshold вместо zscore_entry_threshold."""
    print("\n🔍 ГИПОТЕЗА 1: Неправильное использование порога входа")
    
    # Проверяем значения порогов в конфигурации
    try:
        z_threshold = getattr(cfg.backtest, 'zscore_threshold', None)
        zscore_entry_threshold = getattr(cfg.backtest, 'zscore_entry_threshold', None)
        
        print(f"📊 zscore_threshold: {z_threshold}")
        print(f"📊 zscore_entry_threshold: {zscore_entry_threshold}")
        
        # Проверяем, есть ли различия
        if z_threshold is not None and zscore_entry_threshold is not None:
            if abs(z_threshold - zscore_entry_threshold) > 1e-6:
                print("⚠️  ПРОБЛЕМА НАЙДЕНА: Разные значения порогов!")
                print(f"   Разница: {abs(z_threshold - zscore_entry_threshold):.6f}")
                print("💡 В коде base_engine.py используется self.z_threshold вместо self.zscore_entry_threshold")
                print("💡 Это означает, что фактически используется zscore_threshold, а не zscore_entry_threshold")
                return True
            else:
                print("✅ Пороги совпадают")
        elif zscore_entry_threshold is not None and z_threshold is None:
            print("⚠️  ВОЗМОЖНАЯ ПРОБЛЕМА: zscore_entry_threshold задан, но zscore_threshold отсутствует")
            print("💡 Код может использовать значение по умолчанию вместо настроенного")
            return True
        elif z_threshold is not None and zscore_entry_threshold is None:
            print("✅ Используется только zscore_threshold")
        else:
            print("❌ Оба порога отсутствуют в конфигурации")
            return True
            
    except Exception as e:
        print(f"❌ Ошибка анализа порогов: {e}")
        return True
    
    return False

def hypothesis_2_too_strict_parameters(cfg, config_path):
    """Гипотеза 2: Слишком строгие параметры оптимизации."""
    print("\n🔍 ГИПОТЕЗА 2: Слишком строгие параметры")
    
    try:
        # Анализируем параметры входа
        z_entry = getattr(cfg.backtest, 'zscore_entry_threshold', 
                         getattr(cfg.backtest, 'zscore_threshold', 1.5))
        z_exit = getattr(cfg.backtest, 'zscore_exit', 0.0)
        
        print(f"📊 Z-score вход: {z_entry}")
        print(f"📊 Z-score выход: {z_exit}")
        
        # Проверяем строгость параметров
        problems = []
        
        if z_entry > 2.0:
            problems.append(f"Очень высокий порог входа: {z_entry} (рекомендуется < 2.0)")
        elif z_entry > 1.8:
            problems.append(f"Высокий порог входа: {z_entry} (может быть слишком строгим)")
        
        if abs(z_exit) > 0.5:
            problems.append(f"Высокий порог выхода: {z_exit} (рекомендуется ближе к 0)")
        
        # Проверяем другие ограничивающие параметры
        max_positions = getattr(cfg.portfolio, 'max_active_positions', 15)
        if max_positions < 5:
            problems.append(f"Мало активных позиций: {max_positions} (может ограничивать торговлю)")
        
        # Проверяем стоп-лоссы
        stop_loss_mult = getattr(cfg.backtest, 'stop_loss_multiplier', 3.0)
        if stop_loss_mult < 2.0:
            problems.append(f"Слишком жесткий стоп-лосс: {stop_loss_mult} (может закрывать позиции преждевременно)")
        
        # Проверяем временные ограничения
        cooldown_hours = getattr(cfg.backtest, 'cooldown_hours', 0)
        if cooldown_hours > 12:
            problems.append(f"Долгий кулдаун: {cooldown_hours} часов (может пропускать сигналы)")
        
        if problems:
            print("⚠️  ПРОБЛЕМЫ НАЙДЕНЫ:")
            for i, problem in enumerate(problems, 1):
                print(f"   {i}. {problem}")
            
            # Предлагаем рекомендации
            print("💡 РЕКОМЕНДАЦИИ:")
            if z_entry > 1.8:
                print(f"   - Снизить zscore_entry_threshold до 1.5-1.8")
            if abs(z_exit) > 0.3:
                print(f"   - Установить zscore_exit ближе к 0 (например, 0.0 или ±0.2)")
            if cooldown_hours > 6:
                print(f"   - Снизить cooldown_hours до 2-6 часов")
            
            return True
        else:
            print("✅ Параметры в разумных пределах")
            return False
            
    except Exception as e:
        print(f"❌ Ошибка анализа параметров: {e}")
        return True

def hypothesis_3_data_quality_issues(cfg, config_path):
    """Гипотеза 3: Проблемы с качеством данных."""
    print("\n🔍 ГИПОТЕЗА 3: Проблемы с качеством данных")
    
    try:
        # Проверяем настройки данных
        data_dir = getattr(cfg, 'data_dir', 'data_downloaded')
        data_path = Path(data_dir)
        
        print(f"📊 Директория данных: {data_path}")
        print(f"📊 Существует: {data_path.exists()}")
        
        if not data_path.exists():
            print("⚠️  ПРОБЛЕМА НАЙДЕНА: Директория данных не существует!")
            return True
        
        # Проверяем наличие файлов данных
        parquet_files = list(data_path.rglob("*.parquet"))
        print(f"📊 Найдено parquet файлов: {len(parquet_files)}")
        
        if len(parquet_files) == 0:
            print("⚠️  ПРОБЛЕМА НАЙДЕНА: Нет файлов данных!")
            return True
        
        # Проверяем настройки обработки данных
        if hasattr(cfg, 'data_processing'):
            norm_method = getattr(cfg.data_processing, 'normalization_method', 'none')
            fill_method = getattr(cfg.data_processing, 'fill_method', 'none')
            min_history = getattr(cfg.data_processing, 'min_history_ratio', 0.8)
            
            print(f"📊 Метод нормализации: {norm_method}")
            print(f"📊 Метод заполнения: {fill_method}")
            print(f"📊 Мин. история: {min_history}")
            
            if min_history > 0.95:
                print("⚠️  ВОЗМОЖНАЯ ПРОБЛЕМА: Очень строгие требования к истории данных")
                return True
        
        # Проверяем временные рамки
        if hasattr(cfg, 'walk_forward'):
            start_date = cfg.walk_forward.start_date
            end_date = cfg.walk_forward.end_date
            
            print(f"📊 Период тестирования: {start_date} - {end_date}")
            
            # Проверяем разумность периода
            start_ts = pd.Timestamp(start_date)
            end_ts = pd.Timestamp(end_date)
            period_days = (end_ts - start_ts).days
            
            print(f"📊 Длительность периода: {period_days} дней")
            
            if period_days < 7:
                print("⚠️  ВОЗМОЖНАЯ ПРОБЛЕМА: Очень короткий период тестирования")
                return True
            
            if period_days > 365:
                print("⚠️  ВОЗМОЖНАЯ ПРОБЛЕМА: Очень длинный период может содержать структурные сдвиги")
        
        print("✅ Настройки данных выглядят корректно")
        return False
        
    except Exception as e:
        print(f"❌ Ошибка анализа данных: {e}")
        return True

def hypothesis_4_filtering_logic_errors(cfg, config_path):
    """Гипотеза 4: Ошибки в логике фильтрации пар."""
    print("\n🔍 ГИПОТЕЗА 4: Ошибки в логике фильтрации")
    
    try:
        # Проверяем параметры фильтрации пар
        if hasattr(cfg, 'pair_selection'):
            ps = cfg.pair_selection
            
            # Основные параметры фильтрации
            min_volume = getattr(ps, 'min_volume_usd_24h', 0)
            max_bid_ask = getattr(ps, 'max_bid_ask_pct', 1.0)
            min_correlation = getattr(cfg.backtest, 'min_correlation_threshold', 0.0)
            coint_pvalue = getattr(ps, 'coint_pvalue_threshold', 0.05)
            
            print(f"📊 Мин. объем: ${min_volume:,.0f}")
            print(f"📊 Макс. спред: {max_bid_ask:.1%}")
            print(f"📊 Мин. корреляция: {min_correlation:.2f}")
            print(f"📊 P-value коинтеграции: {coint_pvalue:.3f}")
            
            problems = []
            
            # Проверяем слишком строгие фильтры
            if min_volume > 10_000_000:  # 10M USD
                problems.append(f"Очень высокие требования к объему: ${min_volume:,.0f}")
            
            if max_bid_ask < 0.001:  # 0.1%
                problems.append(f"Очень низкий допустимый спред: {max_bid_ask:.1%}")
            
            if min_correlation > 0.8:
                problems.append(f"Очень высокие требования к корреляции: {min_correlation:.2f}")
            
            if coint_pvalue < 0.01:
                problems.append(f"Очень строгий p-value коинтеграции: {coint_pvalue:.3f}")
            
            # Проверяем параметры half-life
            min_half_life = getattr(ps, 'min_half_life_days', 0)
            max_half_life = getattr(ps, 'max_half_life_days', 365)
            
            print(f"📊 Half-life диапазон: {min_half_life:.1f} - {max_half_life:.1f} дней")
            
            if min_half_life > 5:
                problems.append(f"Высокий мин. half-life: {min_half_life} дней")
            
            if max_half_life < 7:
                problems.append(f"Низкий макс. half-life: {max_half_life} дней")
            
            # Проверяем количество кандидатов
            ssd_top_n = getattr(ps, 'ssd_top_n', 1000)
            pvalue_top_n = getattr(ps, 'pvalue_top_n', 100)
            
            print(f"📊 SSD топ-N: {ssd_top_n}")
            print(f"📊 P-value топ-N: {pvalue_top_n}")
            
            if ssd_top_n < 100:
                problems.append(f"Мало кандидатов SSD: {ssd_top_n}")
            
            if pvalue_top_n < 20:
                problems.append(f"Мало кандидатов p-value: {pvalue_top_n}")
            
            if problems:
                print("⚠️  ПРОБЛЕМЫ НАЙДЕНЫ:")
                for i, problem in enumerate(problems, 1):
                    print(f"   {i}. {problem}")
                
                print("💡 РЕКОМЕНДАЦИИ:")
                print("   - Ослабить фильтры для получения большего количества пар")
                print("   - Увеличить ssd_top_n и pvalue_top_n")
                print("   - Снизить требования к объему и корреляции")
                
                return True
            else:
                print("✅ Фильтрация выглядит разумной")
        else:
            print("❌ Секция pair_selection отсутствует в конфигурации")
            return True
        
        return False
        
    except Exception as e:
        print(f"❌ Ошибка анализа фильтрации: {e}")
        return True

def hypothesis_5_walk_forward_issues(cfg, config_path):
    """Гипотеза 5: Проблемы с настройками walk-forward тестирования."""
    print("\n🔍 ГИПОТЕЗА 5: Проблемы с walk-forward тестированием")
    
    try:
        if hasattr(cfg, 'walk_forward'):
            wf = cfg.walk_forward
            
            enabled = getattr(wf, 'enabled', False)
            training_days = getattr(wf, 'training_period_days', 30)
            testing_days = getattr(wf, 'testing_period_days', 7)
            step_size = getattr(wf, 'step_size_days', 7)
            min_samples = getattr(wf, 'min_training_samples', 1000)
            
            print(f"📊 Walk-forward включен: {enabled}")
            print(f"📊 Период обучения: {training_days} дней")
            print(f"📊 Период тестирования: {testing_days} дней")
            print(f"📊 Шаг сдвига: {step_size} дней")
            print(f"📊 Мин. образцов: {min_samples}")
            
            problems = []
            
            if not enabled:
                print("ℹ️  Walk-forward отключен - используется простой бэктест")
                return False
            
            # Проверяем разумность параметров
            if training_days < 7:
                problems.append(f"Очень короткий период обучения: {training_days} дней")
            
            if testing_days < 1:
                problems.append(f"Очень короткий период тестирования: {testing_days} дней")
            
            if step_size > testing_days:
                problems.append(f"Шаг сдвига больше периода тестирования: {step_size} > {testing_days}")
            
            if min_samples > training_days * 96:  # 96 = 24*4 для 15-минутных данных
                problems.append(f"Слишком много требуемых образцов: {min_samples}")
            
            # Проверяем соотношение периодов
            if training_days < testing_days * 3:
                problems.append(f"Период обучения слишком короткий относительно тестирования")
            
            if problems:
                print("⚠️  ПРОБЛЕМЫ НАЙДЕНЫ:")
                for i, problem in enumerate(problems, 1):
                    print(f"   {i}. {problem}")
                
                print("💡 РЕКОМЕНДАЦИИ:")
                print("   - Увеличить training_period_days до 30-60 дней")
                print("   - Установить step_size_days <= testing_period_days")
                print("   - Снизить min_training_samples если нужно")
                
                return True
            else:
                print("✅ Настройки walk-forward выглядят корректно")
        else:
            print("ℹ️  Walk-forward не настроен")
        
        return False
        
    except Exception as e:
        print(f"❌ Ошибка анализа walk-forward: {e}")
        return True

def analyze_code_issue():
    """Анализ найденной проблемы в коде."""
    print("\n🔧 АНАЛИЗ ПРОБЛЕМЫ В КОДЕ")
    
    base_engine_path = Path("src/coint2/engine/base_engine.py")
    
    if not base_engine_path.exists():
        print("❌ Файл base_engine.py не найден")
        return
    
    print(f"📁 Анализируем файл: {base_engine_path}")
    
    try:
        with open(base_engine_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Ищем проблемную строку
        lines = content.split('\n')
        problem_found = False
        
        for i, line in enumerate(lines, 1):
            # Ищем использование z_threshold в условиях входа
            if 'z_curr > self.z_threshold' in line or 'z_curr < -self.z_threshold' in line:
                print(f"🔍 Строка {i}: {line.strip()}")
                
                # Проверяем, есть ли инициализация z_threshold
                z_threshold_init = False
                zscore_entry_threshold_init = False
                
                for j, init_line in enumerate(lines, 1):
                    if 'self.z_threshold =' in init_line:
                        z_threshold_init = True
                        print(f"📍 Строка {j}: {init_line.strip()}")
                    if 'self.zscore_entry_threshold =' in init_line:
                        zscore_entry_threshold_init = True
                        print(f"📍 Строка {j}: {init_line.strip()}")
                
                if zscore_entry_threshold_init and not z_threshold_init:
                    print("⚠️  КРИТИЧЕСКАЯ ПРОБЛЕМА НАЙДЕНА!")
                    print("   В коде используется self.z_threshold, но инициализируется self.zscore_entry_threshold")
                    print("   Это означает, что self.z_threshold может быть None или иметь неправильное значение")
                    problem_found = True
                elif zscore_entry_threshold_init and z_threshold_init:
                    print("ℹ️  Оба порога инициализированы, но используется z_threshold")
                    print("   Проверьте, что z_threshold = zscore_entry_threshold")
                
                break
        
        if problem_found:
            print("\n💡 РЕШЕНИЕ:")
            print("   1. Заменить 'self.z_threshold' на 'self.zscore_entry_threshold' в условиях входа")
            print("   2. Или убедиться, что self.z_threshold = self.zscore_entry_threshold при инициализации")
            print("   3. Проверить все места использования порогов в коде")
        
    except Exception as e:
        print(f"❌ Ошибка анализа кода: {e}")

def main():
    """Основная функция диагностики."""
    print("🔍 ДИАГНОСТИКА ПРОБЛЕМЫ С НУЛЕВЫМИ СДЕЛКАМИ")
    print("=" * 50)
    
    # Загружаем конфигурацию
    cfg, config_path = load_config_for_analysis()
    if cfg is None:
        return
    
    # Проверяем все гипотезы
    problems_found = []
    
    if hypothesis_1_wrong_threshold_usage(cfg, config_path):
        problems_found.append("Неправильное использование порога входа")
    
    if hypothesis_2_too_strict_parameters(cfg, config_path):
        problems_found.append("Слишком строгие параметры")
    
    if hypothesis_3_data_quality_issues(cfg, config_path):
        problems_found.append("Проблемы с качеством данных")
    
    if hypothesis_4_filtering_logic_errors(cfg, config_path):
        problems_found.append("Ошибки в логике фильтрации")
    
    if hypothesis_5_walk_forward_issues(cfg, config_path):
        problems_found.append("Проблемы с walk-forward тестированием")
    
    # Анализируем код
    analyze_code_issue()
    
    # Выводим итоги
    print("\n" + "=" * 50)
    print("📋 ИТОГИ ДИАГНОСТИКИ")
    
    if problems_found:
        print(f"⚠️  Найдено проблем: {len(problems_found)}")
        for i, problem in enumerate(problems_found, 1):
            print(f"   {i}. {problem}")
    else:
        print("✅ Явных проблем в конфигурации не найдено")
    
    print("\n🔧 ОСНОВНЫЕ РЕКОМЕНДАЦИИ:")
    print("1. ⭐ КРИТИЧНО: Исправить использование zscore_entry_threshold в base_engine.py")
    print("2. Проверить и ослабить слишком строгие параметры фильтрации")
    print("3. Убедиться в наличии и качестве данных")
    print("4. Добавить детальное логирование сигналов для отладки")
    print("5. Протестировать на более простых параметрах")
    
    print("\n📝 СЛЕДУЮЩИЕ ШАГИ:")
    print("1. Исправить код в base_engine.py (строка ~1874)")
    print("2. Запустить валидацию повторно")
    print("3. Если проблема остается - проверить данные и логи")
    print("4. Рассмотреть упрощение параметров для получения сделок")

if __name__ == "__main__":
    main()