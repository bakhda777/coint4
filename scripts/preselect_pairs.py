#!/usr/bin/env python3
"""
Скрипт для предварительного отбора пар.
ТОЧНО ПОВТОРЯЕТ логику первого шага walk-forward анализа из оригинальной системы.
"""

import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
import time
import argparse

# Добавляем src в путь для импорта
sys.path.append(str(Path(__file__).parent.parent / "src"))

from coint2.utils.config import load_config
from coint2.core.data_loader import DataHandler, load_master_dataset
from coint2.core.normalization_improvements import preprocess_and_normalize_data
from coint2.core import math_utils
from coint2.pipeline.filters import filter_pairs_by_coint_and_half_life
from coint2.utils.logger import get_logger

def preselect_and_save_pairs():
    """
    ТОЧНО ПОВТОРЯЕТ логику отбора пар из первого шага walk-forward анализа.
    """
    print("🚀 Запуск предварительного отбора пар...")
    print("📊 ТОЧНО ПОВТОРЯЕМ логику первого walk-forward шага")

    # Загружаем конфигурацию (обычную или relaxed)
    use_relaxed = os.getenv("USE_RELAXED_CONFIG", "false").lower() == "true"
    config_path = "configs/relaxed_config.yaml" if use_relaxed else "configs/main_2024.yaml"

    print(f"📁 Используем конфигурацию: {config_path}")
    if use_relaxed:
        print("⚡ RELAXED режим: ослабленные фильтры для быстрой оптимизации")

    cfg = load_config(config_path)
    logger = get_logger("preselect_pairs")

    # ТОЧНО как в оригинальной системе: определяем первый walk-forward шаг
    start_date = pd.to_datetime(cfg.walk_forward.start_date)
    bar_minutes = getattr(cfg.pair_selection, "bar_minutes", None) or 15
    bar_delta = pd.Timedelta(minutes=bar_minutes)

    # Первый шаг walk-forward
    current_test_start = start_date
    training_start = current_test_start - pd.Timedelta(days=cfg.walk_forward.training_period_days)
    training_end = current_test_start - bar_delta
    testing_start = current_test_start
    testing_end = testing_start + pd.Timedelta(days=cfg.walk_forward.testing_period_days)

    print(f"🗓️  ПЕРВЫЙ WALK-FORWARD ШАГ:")
    print(f"   Тренировка: {training_start.date()} -> {training_end.date()}")
    print(f"   Тестирование: {testing_start.date()} -> {testing_end.date()}")

    # Загружаем данные ТОЧНО как в оригинальной системе
    handler = DataHandler(cfg)
    print("📈 Загрузка данных...")

    try:
        # Загружаем данные за весь период (тренировка + тестирование)
        full_range_start = training_start
        full_range_end = testing_end

        raw_data = load_master_dataset(
            data_path=cfg.data_dir,
            start_date=full_range_start,
            end_date=full_range_end
        )
        
        if raw_data.empty:
            print("❌ Не удалось загрузить данные для отбора пар.")
            return

        print(f"📊 Загружено {raw_data.shape[0]} записей для {len(raw_data['symbol'].unique())} символов")

        # ТОЧНО как в оригинальной системе: преобразуем в pivot table
        step_df = raw_data.pivot_table(index="timestamp", columns="symbol", values="close")
        print(f"📊 Pivot table: {step_df.shape}")

        # Извлекаем ТОЛЬКО тренировочный период для отбора пар
        training_slice = step_df.loc[training_start:training_end]
        print(f"📊 Тренировочный срез: {training_slice.shape}")

        if training_slice.empty or len(training_slice.columns) < 2:
            print("❌ Недостаточно данных для обучения")
            return

        # ТОЧНО как в оригинальной системе: собираем параметры нормализации
        training_normalization_params = {}
        if not training_slice.empty:
            for col in training_slice.columns:
                first_valid_idx = training_slice[col].first_valid_index()
                if first_valid_idx is not None:
                    training_normalization_params[col] = training_slice.loc[first_valid_idx, col]

        # ТОЧНО как в оригинальной системе: нормализация
        print("🔄 Нормализация данных (как в оригинальной системе)...")

        # Используем улучшенный метод нормализации
        norm_method = getattr(cfg.data_processing, 'normalization_method', 'minmax')
        fill_method = getattr(cfg.data_processing, 'fill_method', 'ffill')
        min_history_ratio = getattr(cfg.data_processing, 'min_history_ratio', 0.8)
        handle_constant = getattr(cfg.data_processing, 'handle_constant', True)

        print(f"  Применяем метод нормализации: {norm_method}")
        normalized_training, norm_stats = preprocess_and_normalize_data(
            training_slice,
            min_history_ratio=min_history_ratio,
            fill_method=fill_method,
            norm_method=norm_method,
            handle_constant=handle_constant
        )
        
        # Выводим статистику нормализации
        print("  Статистика нормализации:")
        print(f"    Исходные символы: {norm_stats['initial_symbols']}")
        print(f"    Недостаточная история: {norm_stats['low_history_ratio']}")
        print(f"    Постоянная цена: {norm_stats['constant_price']}")
        print(f"    NaN после нормализации: {norm_stats['nan_after_norm']}")
        print(f"    Итоговые символы: {norm_stats['final_symbols']} ({norm_stats['final_symbols']/norm_stats['initial_symbols']*100:.1f}%)")

        if len(normalized_training.columns) < 2:
            print("❌ После нормализации осталось менее 2 символов")
            return

        print(f"✅ Нормализовано {len(normalized_training.columns)} символов")

        # ТОЧНО как в оригинальной системе: SSD computation
        print("🔍 Расчет SSD для всех пар (без ограничения)...")
        # Сначала считаем SSD для всех пар
        ssd = math_utils.calculate_ssd(normalized_training, top_k=None)
        print(f"  SSD результат (все пары): {len(ssd)} пар")

        # Затем берем только top-N пар для дальнейшей фильтрации
        ssd_top_n = cfg.pair_selection.ssd_top_n
        if len(ssd) > ssd_top_n:
            print(f"  Ограничиваем до top-{ssd_top_n} пар для дальнейшей обработки")
            ssd = ssd.sort_values().head(ssd_top_n)

        ssd_pairs = [(s1, s2) for s1, s2 in ssd.index]
        print(f"📈 Найдено {len(ssd_pairs)} кандидатов по SSD")

        # ТОЧНО как в оригинальной системе: Filter pairs
        print("🔬 Фильтрация пар по коинтеграции и другим критериям...")

        filtered_pairs = filter_pairs_by_coint_and_half_life(
            ssd_pairs,
            training_slice,  # Используем НЕнормализованные данные для фильтрации!
            pvalue_threshold=cfg.pair_selection.coint_pvalue_threshold,
            min_beta=cfg.filter_params.min_beta,
            max_beta=cfg.filter_params.max_beta,
            min_half_life=cfg.filter_params.min_half_life_days,
            max_half_life=cfg.filter_params.max_half_life_days,
            min_mean_crossings=cfg.filter_params.min_mean_crossings,
            max_hurst_exponent=cfg.filter_params.max_hurst_exponent,
            save_filter_reasons=cfg.pair_selection.save_filter_reasons,
            kpss_pvalue_threshold=cfg.pair_selection.kpss_pvalue_threshold,
        )

        print(f"  Фильтрация: {len(ssd_pairs)} → {len(filtered_pairs)} пар")

        if not filtered_pairs:
            print("❌ Не найдено ни одной пары после фильтрации.")
            return

        print(f"✅ Найдено {len(filtered_pairs)} качественных пар")

        # ТОЧНО как в оригинальной системе: сортируем пары по качеству
        quality_sorted_pairs = sorted(filtered_pairs, key=lambda x: abs(x[4]), reverse=True)  # x[4] = std
        active_pairs = quality_sorted_pairs  # Берем ВСЕ отфильтрованные пары

        print("  Топ-3 пары по волатильности спреда:")
        for i, (s1, s2, beta, mean, std, metrics) in enumerate(active_pairs[:3], 1):
            print(f"    {i}. {s1}-{s2}: beta={beta:.4f}, std={std:.4f}")

        # Сохраняем в DataFrame в том же формате, что и оригинальная система
        pairs_list = []
        for s1, s2, beta, mean, std, metrics in active_pairs:
            pairs_list.append({
                's1': s1,
                's2': s2,
                'beta': beta,
                'mean': mean,
                'std': std,
                'half_life': metrics.get('half_life', 0.0),
                'pvalue': metrics.get('pvalue', 0.05),
                'hurst': metrics.get('hurst', 0.5),
                'mean_crossings': metrics.get('mean_crossings', 0)
            })
        
        df_pairs = pd.DataFrame(pairs_list)
        
        # Создаем директорию outputs
        Path("outputs").mkdir(exist_ok=True)
        
        # Сохраняем в CSV
        output_path = "outputs/preselected_pairs.csv"
        df_pairs.to_csv(output_path, index=False)
        
        print(f"💾 Отобранные пары сохранены в: {output_path}")
        print(f"📊 Статистика отобранных пар:")
        print(f"   • Всего пар: {len(df_pairs)}")
        print(f"   • Средний half-life: {df_pairs['half_life'].mean():.2f} дней")
        print(f"   • Средний p-value: {df_pairs['pvalue'].mean():.4f}")
        print(f"   • Средний Hurst: {df_pairs['hurst'].mean():.3f}")
        
        # Сохраняем также параметры нормализации из тренировочного периода
        norm_params_path = "outputs/training_normalization_params.csv"
        if training_normalization_params:
            pd.Series(training_normalization_params).to_csv(norm_params_path)
            print(f"💾 Параметры нормализации сохранены в: {norm_params_path}")

        # Сохраняем также полные данные для быстрой оптимизации
        full_data_path = "outputs/full_step_data.csv"
        step_df.to_csv(full_data_path)
        print(f"💾 Полные данные шага сохранены в: {full_data_path}")

        print("\n✅ Предварительный отбор пар завершен успешно!")
        print("📊 ТОЧНО ПОВТОРЕНА логика первого walk-forward шага")
        print("🚀 Теперь можно запускать быструю оптимизацию:")
        print("   poetry run python scripts/fast_optimize.py")
        
    except Exception as e:
        print(f"❌ Ошибка при отборе пар: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Предварительный отбор пар для оптимизации")
    parser.add_argument("--relaxed", action="store_true",
                       help="Использовать relaxed конфигурацию с ослабленными фильтрами")
    args = parser.parse_args()

    # Устанавливаем переменную окружения для использования в функции
    if args.relaxed:
        os.environ["USE_RELAXED_CONFIG"] = "true"

    preselect_and_save_pairs()
