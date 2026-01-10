"""
Быстрая целевая функция для Optuna.
Использует предварительно отобранные пары для ускорения оптимизации.
"""

import optuna
import pandas as pd
import numpy as np
from pathlib import Path
import yaml
import logging
import os
import math
import threading

from coint2.utils.config import load_config
from coint2.core.data_loader import DataHandler, load_master_dataset
from coint2.engine.numba_engine import NumbaPairBacktester
from coint2.engine.optimized_backtest_engine import OptimizedPairBacktester
from coint2.engine.numba_backtest_engine_full import FullNumbaPairBacktester
# УСКОРЕНИЕ: Используем ПОЛНОСТЬЮ Numba-оптимизированный движок для максимального ускорения
PairBacktester = FullNumbaPairBacktester
from coint2.core.portfolio import Portfolio
from coint2.core.math_utils import calculate_ssd
from coint2.pipeline.filters import filter_pairs_by_coint_and_half_life
from coint2.core.normalization_improvements import preprocess_and_normalize_data, compute_normalization_params, apply_normalization_with_params
from coint2.utils.logging_utils import get_logger
from src.optimiser.metric_utils import extract_sharpe, normalize_params, validate_params

# УСКОРЕНИЕ: Импорты для глобального кэша rolling-статистик
from coint2.core.global_rolling_cache import initialize_global_rolling_cache, cleanup_global_rolling_cache
from coint2.core.memory_optimization import initialize_global_price_data, determine_required_windows

# Импортируем константы из единого источника
from .constants import PENALTY, PENALTY_SOFT, PENALTY_HARD, MIN_TRADES_THRESHOLD, MAX_DRAWDOWN_SOFT_THRESHOLD, MAX_DRAWDOWN_HARD_THRESHOLD, \
    WIN_RATE_BONUS_THRESHOLD, WIN_RATE_PENALTY_THRESHOLD, DD_PENALTY_SOFT_MULTIPLIER, DD_PENALTY_HARD_MULTIPLIER, \
    WIN_RATE_BONUS_MULTIPLIER, WIN_RATE_PENALTY_MULTIPLIER, INTERMEDIATE_REPORT_INTERVAL

# Настройка логгера для оптимизации
logger = logging.getLogger(__name__)


def convert_hours_to_periods(hours: float, bar_minutes: int) -> int:
    """
    Convert hours to number of periods based on bar timeframe.
    ИСПРАВЛЕНО: Используем ceil для правильного округления вверх.
    """
    if hours <= 0:
        return 0
    return int(math.ceil(hours * 60 / bar_minutes))


def convert_hours_to_periods(hours: float, bar_minutes: int) -> int:
    """
    Convert hours to number of periods based on bar timeframe.
    ИСПРАВЛЕНО: Используем ceil для правильного округления вверх.
    """
    import math
    if hours <= 0:
        return 0
    return int(math.ceil(hours * 60 / bar_minutes))

class FastWalkForwardObjective:
    """
    Быстрая целевая функция для оптимизации торговых параметров
    на предварительно отобранных парах.
    """
    
    def __init__(self, base_config_path: str, search_space_path: str):
        self.base_config = load_config(base_config_path)

        # Загружаем пространство поиска
        with open(search_space_path, 'r') as f:
            self.search_space = yaml.safe_load(f)

        # УСКОРЕНИЕ: Кэш для отбора пар по периодам тренировки
        self.pair_selection_cache = {}

        # ПОТОКОБЕЗОПАСНОСТЬ: Блокировка для кэша отбора пар
        self._cache_lock = threading.Lock()

        # ОПТИМИЗАЦИЯ: Кэш для данных между trials
        self.data_cache = {}
        self.data_cache_lock = threading.Lock()
        self.max_cache_size = 100  # Ограничиваем размер кэша

        # УСКОРЕНИЕ: Инициализация глобального кэша rolling-статистик
        self.global_cache_initialized = self._initialize_global_rolling_cache()
        if self.global_cache_initialized:
            print("✅ Глобальный кэш успешно инициализирован в FastWalkForwardObjective")
        else:
            print("❌ Глобальный кэш НЕ инициализирован в FastWalkForwardObjective")
    
    def convert_hours_to_periods(self, hours: float, bar_minutes: int) -> int:
        """Convert hours to number of periods based on bar timeframe."""
        return convert_hours_to_periods(hours, bar_minutes)
    
    def _validate_params(self, params):
        """Валидирует параметры используя функцию validate_params из metric_utils."""
        return validate_params(params)

        # КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ: Убираем зависимость от предварительно отобранных пар
        # Теперь пары отбираются динамически для каждого walk-forward шага

        logger.info("✅ Инициализация FastWalkForwardObjective с динамическим отбором пар и глобальным кэшированием")
        
        # КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ: Новое предупреждение о правильном walk-forward анализе
        logger.info(
            "🔄 ИСПРАВЛЕН LOOKAHEAD BIAS: Пары теперь отбираются динамически "
            "для каждого walk-forward шага используя только тренировочные данные этого шага. "
            "Это обеспечивает корректный walk-forward анализ без lookahead bias."
        )

        # ИСПРАВЛЕНО: Жесткая проверка фильтров в fast-режиме
        if 'filters' in self.search_space:
            raise ValueError(
                "В fast-режиме параметры 'filters' в search_space не применяются. "
                "Пары уже предотобраны из outputs/preselected_pairs.csv. "
                "Используйте search_space_fast.yaml или перенесите отбор пар в objective."
            )

        # ИСПРАВЛЕНО: Удален старый код проверки filter_params - заменен на жесткую проверку выше

        # Данные будут загружаться динамически для каждого шага как в оригинальном бэктесте

    # УДАЛЕНО: _validate_pairs_temporal_boundaries больше не нужен
    # при динамическом отборе пар для каждого шага

    # УДАЛЕНО: _run_pair_selection больше не нужен
    # при динамическом отборе пар для каждого шага

    def _initialize_global_rolling_cache(self):
        """Инициализирует глобальный кэш rolling-статистик для ускорения оптимизации."""
        try:
            print("🚀 Инициализация глобального кэша rolling-статистик...")

            # ИСПРАВЛЕНО: Определяем полный диапазон для всей оптимизации
            start_date = pd.to_datetime(self.base_config.walk_forward.start_date) - pd.Timedelta(days=self.base_config.walk_forward.training_period_days)
            end_date = pd.to_datetime(self.base_config.walk_forward.end_date)

            print(f"📅 Загрузка данных для кэша: {start_date.date()} -> {end_date.date()}")

            # Загружаем мастер-датасет напрямую
            all_raw_data = load_master_dataset(self.base_config.data_dir, start_date, end_date)
            if all_raw_data.empty:
                print("❌ Не удалось загрузить данные для кэша. Кэш не будет создан.")
                return False

            # Пивотирование данных в широкий формат
            all_data = all_raw_data.pivot_table(index="timestamp", columns="symbol", values="close")
            # Простое заполнение пропусков для полноты кэша
            all_data = all_data.ffill().bfill()

            print(f"📊 Загружено данных: {all_data.shape[0]} временных точек, {all_data.shape[1]} символов")

            # Инициализируем глобальные данные о ценах
            from src.coint2.core.memory_optimization import initialize_global_price_data_from_dataframe
            print("🔄 Инициализация глобальных данных о ценах...")
            success = initialize_global_price_data_from_dataframe(all_data)
            if not success:
                print("❌ Не удалось инициализировать глобальные данные о ценах")
                return False
            print("✅ Глобальные данные о ценах инициализированы")

            # ИСПРАВЛЕНО: Определяем все возможные rolling_window из полной конфигурации
            from src.coint2.core.memory_optimization import determine_required_windows
            print("🔄 Определение требуемых rolling windows...")

            # Передаем полную конфигурацию вместо только search_space
            full_config = self.base_config.model_dump() if hasattr(self.base_config, 'model_dump') else self.base_config.__dict__
            required_windows = determine_required_windows(full_config)

            # Также добавляем rolling_window из search_space если есть
            if 'signals' in self.search_space and 'rolling_window' in self.search_space['signals']:
                rolling_window_values = self.search_space['signals']['rolling_window']
                if isinstance(rolling_window_values, list):
                    required_windows.update(rolling_window_values)
                elif isinstance(rolling_window_values, dict):
                    if 'choices' in rolling_window_values:
                        required_windows.update(rolling_window_values['choices'])
                    elif 'low' in rolling_window_values and 'high' in rolling_window_values:
                        # Генерируем все возможные значения из диапазона
                        low = rolling_window_values['low']
                        high = rolling_window_values['high']
                        step = rolling_window_values.get('step', 1)
                        range_values = list(range(low, high + 1, step))
                        required_windows.update(range_values)
                        print(f"📊 Добавлены rolling windows из диапазона {low}-{high} (step={step}): {range_values}")
            elif 'rolling_window' in self.search_space:
                # Старый формат для совместимости
                rolling_window_values = self.search_space['rolling_window']
                if isinstance(rolling_window_values, list):
                    required_windows.update(rolling_window_values)
                elif isinstance(rolling_window_values, dict) and 'choices' in rolling_window_values:
                    required_windows.update(rolling_window_values['choices'])

            print(f"📊 Найдены rolling windows: {sorted(required_windows)}")

            # Инициализируем глобальный кэш rolling-статистик
            cache_config = {
                'search_space': self.search_space,
                'required_windows': required_windows,
                'backtest': full_config.get('backtest', {}),
                'portfolio': full_config.get('portfolio', {})
            }

            from src.coint2.core.global_rolling_cache import initialize_global_rolling_cache
            print("🔄 Инициализация глобального кэша rolling-статистик...")
            success = initialize_global_rolling_cache(cache_config)
            if success:
                print("✅ Глобальный кэш rolling-статистик успешно инициализирован")
                return True
            else:
                print("❌ Не удалось инициализировать глобальный кэш rolling-статистик")
                return False

        except Exception as e:
            print(f"❌ Ошибка при инициализации глобального кэша: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _load_data_for_step(self, training_start, training_end, testing_start, testing_end):
        """
        Загружает данные для конкретного walk-forward шага с правильным разделением
        на тренировочный и тестовый периоды для предотвращения lookahead bias.
        """

        print(f"📈 Загрузка данных для walk-forward шага:")
        print(f"   Тренировка: {training_start.date()} -> {training_end.date()}")
        print(f"   Тестирование: {testing_start.date()} -> {testing_end.date()}")

        try:
            # ИСПРАВЛЕНО: Загружаем данные точно как в оригинальном walk_forward_orchestrator
            raw_data = load_master_dataset(
                data_path=self.base_config.data_dir,
                start_date=training_start,
                end_date=testing_end
            )

            if raw_data.empty:
                raise ValueError("Не удалось загрузить данные")

            # Преобразуем в формат для бэктестинга точно как в оригинале
            step_df = raw_data.pivot_table(index="timestamp", columns="symbol", values="close")

            # Гарантируем DatetimeIndex
            if not isinstance(step_df.index, pd.DatetimeIndex):
                step_df.index = pd.to_datetime(step_df.index, errors="coerce")
                if getattr(step_df.index, "tz", None) is not None:
                    step_df.index = step_df.index.tz_localize(None)
                step_df = step_df.sort_index()

            # КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ LOOKAHEAD BIAS: Разделяем данные на тренировочные и тестовые
            training_slice = step_df.loc[training_start:training_end]
            testing_slice = step_df.loc[testing_start:testing_end]

            # Проверка на перекрытие данных (защита от lookahead bias)
            if not training_slice.empty and not testing_slice.empty:
                if training_slice.index.max() >= testing_slice.index.min():
                    raise ValueError(
                        f"ОБНАРУЖЕНО ПЕРЕКРЫТИЕ ДАННЫХ! "
                        f"Последняя тренировочная метка: {training_slice.index.max()}, "
                        f"Первая тестовая метка: {testing_slice.index.min()}. "
                        f"Это может привести к lookahead bias!"
                    )

            print(f"✅ Данные загружены и разделены:")
            print(f"   Тренировочный срез: {training_slice.shape}")
            print(f"   Тестовый срез: {testing_slice.shape}")
            print(f"   Временной разрыв: {testing_start - training_end}")

            # ИСПРАВЛЕНО: Возвращаем структуру с правильно разделенными данными
            return {
                'full_data': step_df,
                'training_data': training_slice,
                'testing_data': testing_slice,
                'training_start': training_start,
                'training_end': training_end,
                'testing_start': testing_start,
                'testing_end': testing_end
            }

        except Exception as e:
            print(f"❌ Ошибка загрузки данных: {e}")
            raise
    
    def _suggest_parameters(self, trial: optuna.Trial):
        """Генерирует параметры для Optuna trial на основе search_space.
        
        Args:
            trial: Optuna trial объект
            
        Returns:
            dict: Словарь параметров включая trial_number
        """
        params = {}
        
        # Группа 1: Фильтры отбора пар
        if 'filters' in self.search_space:
            filters = self.search_space['filters']
            if 'ssd_top_n' in filters:
                cfg = filters['ssd_top_n']
                # ИСПРАВЛЕНО: Используем встроенную поддержку логарифмического распределения в Optuna 4
                if cfg.get('step'):
                    params['ssd_top_n'] = trial.suggest_int(
                        "ssd_top_n",
                        cfg['low'],
                        cfg['high'],
                        step=cfg['step']
                    )
                else:
                    # ИСПРАВЛЕНО: Используем trial.suggest_int(..., log=True) вместо ручного log10
                    params['ssd_top_n'] = trial.suggest_int(
                        "ssd_top_n",
                        cfg['low'],
                        cfg['high'],
                        log=True
                    )
            if 'kpss_pvalue_threshold' in filters:
                params['kpss_pvalue_threshold'] = trial.suggest_float(
                    "kpss_pvalue_threshold",
                    filters['kpss_pvalue_threshold']['low'],
                    filters['kpss_pvalue_threshold']['high']
                )
            if 'coint_pvalue_threshold' in filters:
                params['coint_pvalue_threshold'] = trial.suggest_float(
                    "coint_pvalue_threshold",
                    filters['coint_pvalue_threshold']['low'],
                    filters['coint_pvalue_threshold']['high']
                )
            # ИСПРАВЛЕНО: условный sampling для half_life параметров
            if 'min_half_life_days' in filters:
                # ИСПРАВЛЕНО: Убираем log=True для совместимости с разными версиями Optuna
                params['min_half_life_days'] = trial.suggest_float(
                    "min_half_life_days",
                    filters['min_half_life_days']['low'],
                    filters['min_half_life_days']['high']
                )

            if 'max_half_life_days' in filters:
                min_half_life = params.get('min_half_life_days', filters['max_half_life_days']['low'])
                # max_half_life должен быть >= min_half_life
                effective_low = max(filters['max_half_life_days']['low'], min_half_life + 0.1)

                if effective_low <= filters['max_half_life_days']['high']:
                    # ИСПРАВЛЕНО: Убираем log=True для совместимости с разными версиями Optuna
                    params['max_half_life_days'] = trial.suggest_float(
                        "max_half_life_days",
                        effective_low,
                        filters['max_half_life_days']['high']
                    )
                else:
                    # Если диапазон невозможен, используем pruning
                    raise optuna.TrialPruned(f"Невозможный диапазон max_half_life для min_half_life={min_half_life}")
            if 'min_mean_crossings' in filters:
                params['min_mean_crossings'] = trial.suggest_int(
                    "min_mean_crossings",
                    filters['min_mean_crossings']['low'],
                    filters['min_mean_crossings']['high']
                )
        
        # Группа 2: Сигналы и тайминг - ИСПРАВЛЕНО: условный sampling для зависимых параметров
        if 'signals' in self.search_space:
            signals = self.search_space['signals']

            # Сначала семплим zscore_threshold
            if 'zscore_threshold' in signals:
                params['zscore_threshold'] = trial.suggest_float(
                    "zscore_threshold",
                    signals['zscore_threshold']['low'],
                    signals['zscore_threshold']['high']
                )

            # Затем семплим zscore_exit с учетом ограничения
            if 'zscore_exit' in signals and 'zscore_threshold' in params:
                threshold = params['zscore_threshold']
                # zscore_exit должен быть ближе к 0, чем threshold
                max_exit = min(signals['zscore_exit']['high'], threshold - 0.1)
                min_exit = max(signals['zscore_exit']['low'], -threshold + 0.1)

                if min_exit <= max_exit:
                    zscore_exit = trial.suggest_float(
                        "zscore_exit",
                        min_exit,
                        max_exit
                    )
                    params['zscore_exit'] = zscore_exit

                    # BEST PRACTICE: Добавляем анти-чурн проверки
                    gap = threshold - zscore_exit
                    if gap < 0.05:  # Минимальный gap для предотвращения частых сделок
                        raise optuna.TrialPruned(f"Слишком маленький gap между threshold и exit: {gap:.3f} < 0.05")

                    # Логируем hysteresis для отчетности
                    trial.set_user_attr("hysteresis", gap)
                else:
                    # Если диапазон невозможен, используем pruning
                    raise optuna.TrialPruned(f"Невозможный диапазон zscore_exit для threshold={threshold}")
            elif 'zscore_exit' in signals:
                # Fallback если threshold не задан
                params['zscore_exit'] = trial.suggest_float(
                    "zscore_exit",
                    signals['zscore_exit']['low'],
                    signals['zscore_exit']['high']
                )

            if 'rolling_window' in signals:
                cfg = signals['rolling_window']
                if 'step' in cfg:
                    params['rolling_window'] = trial.suggest_int("rolling_window", cfg['low'], cfg['high'], step=cfg['step'])
                else:
                    params['rolling_window'] = trial.suggest_int("rolling_window", cfg['low'], cfg['high'])
        
        # Группа 3: Управление рисками
        if 'risk_management' in self.search_space:
            risk = self.search_space['risk_management']
            if 'stop_loss_multiplier' in risk:
                params['stop_loss_multiplier'] = trial.suggest_float(
                    "stop_loss_multiplier",
                    risk['stop_loss_multiplier']['low'],
                    risk['stop_loss_multiplier']['high']
                )
            if 'time_stop_multiplier' in risk:
                params['time_stop_multiplier'] = trial.suggest_float(
                    "time_stop_multiplier",
                    risk['time_stop_multiplier']['low'],
                    risk['time_stop_multiplier']['high']
                )
            if 'cooldown_hours' in risk:
                cfg = risk['cooldown_hours']
                if 'step' in cfg:
                    params['cooldown_hours'] = trial.suggest_int("cooldown_hours", cfg['low'], cfg['high'], step=cfg['step'])
                else:
                    params['cooldown_hours'] = trial.suggest_int("cooldown_hours", cfg['low'], cfg['high'])
        
        # Группа 4: Портфель
        if 'portfolio' in self.search_space:
            portfolio = self.search_space['portfolio']
            if 'risk_per_position_pct' in portfolio:
                params['risk_per_position_pct'] = trial.suggest_float(
                    "risk_per_position_pct",
                    portfolio['risk_per_position_pct']['low'],
                    portfolio['risk_per_position_pct']['high']
                )
            if 'max_position_size_pct' in portfolio:
                params['max_position_size_pct'] = trial.suggest_float(
                    "max_position_size_pct",
                    portfolio['max_position_size_pct']['low'],
                    portfolio['max_position_size_pct']['high']
                )
            if 'max_active_positions' in portfolio:
                cfg = portfolio['max_active_positions']
                params['max_active_positions'] = trial.suggest_int(
                    "max_active_positions",
                    cfg['low'],
                    cfg['high'],
                    step=cfg.get('step', 1)
                )
        
        # Группа 5: Издержки
        if 'costs' in self.search_space:
            costs = self.search_space['costs']
            if 'commission_pct' in costs:
                if isinstance(costs['commission_pct'], dict):
                    # Диапазон значений
                    params['commission_pct'] = trial.suggest_float(
                        "commission_pct",
                        costs['commission_pct']['low'],
                        costs['commission_pct']['high']
                    )
                else:
                    # Фиксированное значение
                    params['commission_pct'] = costs['commission_pct']
            if 'slippage_pct' in costs:
                if isinstance(costs['slippage_pct'], dict):
                    # Диапазон значений
                    params['slippage_pct'] = trial.suggest_float(
                        "slippage_pct",
                        costs['slippage_pct']['low'],
                        costs['slippage_pct']['high']
                    )
                else:
                    # Фиксированное значение
                    params['slippage_pct'] = costs['slippage_pct']
        
        # Группа 6: Нормализация
        if 'normalization' in self.search_space:
            norm = self.search_space['normalization']
            if 'normalization_method' in norm:
                params['normalization_method'] = trial.suggest_categorical(
                    "normalization_method",
                    norm['normalization_method']
                )
            if 'min_history_ratio' in norm:
                params['min_history_ratio'] = trial.suggest_float(
                    "min_history_ratio",
                    norm['min_history_ratio']['low'],
                    norm['min_history_ratio']['high']
                )
        
        # Добавляем номер trial для логирования
        params['trial_number'] = trial.number
        
        return params

    def _select_pairs_for_step(self, cfg, training_data, step_idx):
        """
        КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ: Отбирает пары для конкретного walk-forward шага
        используя только тренировочные данные этого шага для предотвращения lookahead bias.
        """
        logger = get_logger(f"pair_selection_step_{step_idx}")
        
        print(f"   🔍 Отбор пар для шага {step_idx + 1} на тренировочных данных: {training_data.shape}")
        
        try:
            # Нормализация данных для отбора пар
            min_history_ratio = getattr(cfg.pair_selection, "min_history_ratio", 0.8)
            fill_method = getattr(cfg.pair_selection, "fill_method", "forward")
            norm_method = getattr(cfg.pair_selection, "norm_method", "minmax")
            handle_constant = getattr(cfg.pair_selection, "handle_constant", "drop")
            
            normalized_training, norm_stats = preprocess_and_normalize_data(
                training_data,
                min_history_ratio=min_history_ratio,
                fill_method=fill_method,
                norm_method=norm_method,
                handle_constant=handle_constant
            )
            
            if normalized_training.empty or len(normalized_training.columns) < 2:
                print(f"   ❌ Недостаточно данных для отбора пар в шаге {step_idx + 1}")
                return []
            
            # Сканирование пар
            ssd = calculate_ssd(normalized_training, top_k=None)
            
            # Фильтрация по котировочной валюте (*USDT)
            usdt_ssd = ssd[ssd.index.map(lambda x: x[0].endswith('USDT') and x[1].endswith('USDT'))]
            
            # Берем только top-N пар для дальнейшей фильтрации
            ssd_top_n = cfg.pair_selection.ssd_top_n
            if len(usdt_ssd) > ssd_top_n:
                usdt_ssd = usdt_ssd.sort_values().head(ssd_top_n)
            
            ssd_pairs = [(s1, s2) for s1, s2 in usdt_ssd.index]
            
            if not ssd_pairs:
                print(f"   ❌ Не найдено пар после SSD фильтрации в шаге {step_idx + 1}")
                return []
            
            # Фильтрация пар по коинтеграции и другим критериям
            filtered_pairs = filter_pairs_by_coint_and_half_life(
                ssd_pairs,
                normalized_training,
                min_half_life=getattr(cfg.pair_selection, 'min_half_life_days', 1.0),
                max_half_life=getattr(cfg.pair_selection, 'max_half_life_days', 30.0),
                pvalue_threshold=getattr(cfg.pair_selection, 'coint_pvalue_threshold', 0.05),
                min_beta=0.2,
                max_beta=5.0,
                max_hurst_exponent=getattr(cfg.pair_selection, 'max_hurst_exponent', 0.7),
                min_mean_crossings=getattr(cfg.pair_selection, 'min_mean_crossings', 10),
                kpss_pvalue_threshold=getattr(cfg.pair_selection, 'kpss_pvalue_threshold', 0.05),
            )
            
            if not filtered_pairs:
                print(f"   ❌ Не найдено пар после фильтрации в шаге {step_idx + 1}")
                return []
            
            # Сортируем пары по качеству
            quality_sorted_pairs = sorted(filtered_pairs, key=lambda x: abs(x[4]), reverse=True)
            
            # Топ-M отбор для снижения churn и комиссий
            max_pairs_for_trading = getattr(cfg.pair_selection, 'max_pairs_for_trading', 50)
            active_pairs = quality_sorted_pairs[:max_pairs_for_trading]
            
            # Создаем список пар в формате DataFrame
            pairs_list = []
            for s1, s2, beta, mean, std, metrics in active_pairs:
                pairs_list.append({
                    's1': s1,
                    's2': s2,
                    'beta': beta,
                    'mean': mean,
                    'std': std,
                    'half_life': metrics.get('half_life', 0),
                    'pvalue': metrics.get('pvalue', 0),
                    'hurst': 0,
                    'mean_crossings': metrics.get('mean_crossings', 0)
                })
            
            step_pairs_df = pd.DataFrame(pairs_list)
            
            print(f"   ✅ Шаг {step_idx + 1}: отобрано {len(step_pairs_df)} пар из {len(ssd_pairs)} кандидатов")
            
            return step_pairs_df
            
        except Exception as e:
            print(f"   ❌ Ошибка отбора пар для шага {step_idx + 1}: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def _process_single_walk_forward_step(self, cfg, step_data, step_idx):
        """Обрабатывает один walk-forward шаг с динамическим отбором пар."""
        testing_start = step_data['testing_start']
        testing_end = step_data['testing_end']
        training_data = step_data['training_data']
        step_df = step_data['full_data']

        print(f"   🔄 Обработка шага {step_idx + 1}: {testing_start.strftime('%Y-%m-%d')} -> {testing_end.strftime('%Y-%m-%d')}")

        # УСКОРЕНИЕ: Используем потокобезопасный кэш для отбора пар
        training_start = step_data['training_start']
        training_end = step_data['training_end']
        cache_key = f"{training_start.strftime('%Y-%m-%d')}_{training_end.strftime('%Y-%m-%d')}"

        # 1. Быстрая проверка кэша без блокировки
        if cache_key in self.pair_selection_cache:
            print(f"   🚀 Используем кэшированные пары для периода {cache_key}")
            step_pairs = self.pair_selection_cache[cache_key]
        else:
            # 2. Блокировка для выполнения дорогой операции
            with self._cache_lock:
                # 3. Повторная проверка кэша ВНУТРИ блокировки
                if cache_key in self.pair_selection_cache:
                    print(f"   🚀 Используем кэшированные пары для периода {cache_key} (получены во время ожидания)")
                    step_pairs = self.pair_selection_cache[cache_key]
                else:
                    print(f"   🔍 Отбираем новые пары для периода {cache_key}")
                    step_pairs = self._select_pairs_for_step(cfg, training_data, step_idx)
                    # 4. Сохранение результата в кэш
                    if step_pairs is not None and len(step_pairs) > 0:
                        self.pair_selection_cache[cache_key] = step_pairs
                        print(f"   💾 Сохранили {len(step_pairs)} пар в кэш для периода {cache_key}")

        if step_pairs is None or len(step_pairs) == 0:
            print(f"   ❌ Нет пар для шага {step_idx + 1}")
            return {
                'pnls': [],
                'trades': 0,
                'pairs_checked': 0,
                'pairs_with_data': 0
            }

        step_pnls = []
        step_trades = 0
        pairs_processed = 0
        pairs_with_data = 0

        for _, pair_row in step_pairs.iterrows():
            try:
                # ИСПРАВЛЕНО: Защита от распаковки None
                backtest_output = self._backtest_single_pair(pair_row, cfg, step_df)
                if backtest_output is None:
                    continue  # Пропускаем пару, если бэктест не удался
                pair_result, pair_trades = backtest_output

                if pair_result is not None and len(pair_result) > 0:
                    # Фильтруем результаты по тестовому периоду этого шага
                    step_result = pair_result.loc[testing_start:testing_end]
                    if not step_result.empty:
                        step_pnls.append(step_result)
                        step_trades += pair_trades
                        pairs_with_data += 1

                pairs_processed += 1

            except Exception as e:
                print(f"   ❌ Ошибка при обработке пары в шаге {step_idx + 1}: {e}")
                continue

        print(f"   📊 Шаг {step_idx + 1}: {pairs_with_data}/{pairs_processed} пар, {step_trades} сделок")

        return {
            'pnls': step_pnls,
            'trades': step_trades,
            'pairs_checked': pairs_processed,
            'pairs_with_data': pairs_with_data
        }

    def _run_fast_backtest(self, params):
        """Запускает быстрый бэктест ТОЧНО как в оригинальной системе."""

        print(f"\n🔍 ДЕТАЛЬНАЯ ДИАГНОСТИКА БЭКТЕСТА")
        print(f"📊 Входные параметры:")
        for key, value in params.items():
            print(f"   {key}: {value}")

        # Валидируем параметры перед использованием
        try:
            validated_params = validate_params(params)
            print(f"✅ Параметры валидированы успешно")
        except ValueError as e:
            print(f"❌ Ошибка валидации параметров: {e}")
            return {"sharpe_ratio_abs": None, "total_trades": 0, "error_type": "validation_error", "error_message": str(e)}

        # Создаем временную конфигурацию с новыми параметрами
        cfg = self.base_config.model_copy(deep=True)
        
        # Группа 1: Фильтры отбора пар
        if 'ssd_top_n' in validated_params:
            cfg.pair_selection.ssd_top_n = validated_params['ssd_top_n']
        if 'kpss_pvalue_threshold' in validated_params:
            cfg.pair_selection.kpss_pvalue_threshold = validated_params['kpss_pvalue_threshold']
        if 'coint_pvalue_threshold' in validated_params:
            cfg.pair_selection.coint_pvalue_threshold = validated_params['coint_pvalue_threshold']
        if 'min_half_life_days' in validated_params:
            cfg.pair_selection.min_half_life_days = validated_params['min_half_life_days']
        if 'max_half_life_days' in validated_params:
            cfg.pair_selection.max_half_life_days = validated_params['max_half_life_days']
        if 'min_mean_crossings' in validated_params:
            cfg.pair_selection.min_mean_crossings = validated_params['min_mean_crossings']
        
        # Группа 2: Сигналы и тайминг
        cfg.backtest.zscore_threshold = validated_params.get('zscore_threshold', 2.0)
        cfg.backtest.zscore_entry_threshold = cfg.backtest.zscore_threshold  # Синхронизация с основным пайплайном
        cfg.backtest.zscore_exit = validated_params.get('zscore_exit', 0.0)
        if 'rolling_window' in validated_params:
            cfg.backtest.rolling_window = validated_params['rolling_window']
        
        # Группа 3: Управление рисками
        cfg.backtest.stop_loss_multiplier = validated_params.get('stop_loss_multiplier', 3.0)
        cfg.backtest.time_stop_multiplier = validated_params.get('time_stop_multiplier', 2.0)
        if 'cooldown_hours' in validated_params:
            cfg.backtest.cooldown_hours = validated_params['cooldown_hours']
        
        # Группа 4: Портфель
        if hasattr(cfg, 'portfolio'):
            cfg.portfolio.risk_per_position_pct = validated_params.get('risk_per_position_pct', 0.015)
            if hasattr(cfg.portfolio, 'max_position_size_pct'):
                cfg.portfolio.max_position_size_pct = validated_params.get('max_position_size_pct', 0.1)
            cfg.portfolio.max_active_positions = int(validated_params.get('max_active_positions', 15))
        
        # Группа 5: Издержки
        if 'commission_pct' in validated_params:
            cfg.backtest.commission_pct = validated_params['commission_pct']
        if 'slippage_pct' in validated_params:
            cfg.backtest.slippage_pct = validated_params['slippage_pct']
        
        # Группа 6: Нормализация
        if 'normalization_method' in validated_params:
            if hasattr(cfg.pair_selection, 'norm_method'):
                cfg.pair_selection.norm_method = validated_params['normalization_method']
        if 'min_history_ratio' in validated_params:
            if hasattr(cfg.pair_selection, 'min_history_ratio'):
                cfg.pair_selection.min_history_ratio = validated_params['min_history_ratio']

        # ИСПРАВЛЕНО: Поддержка множественных walk-forward шагов
        start_date = pd.to_datetime(cfg.walk_forward.start_date)
        end_date = pd.to_datetime(getattr(cfg.walk_forward, 'end_date', start_date + pd.Timedelta(days=cfg.walk_forward.testing_period_days)))
        step_size_days = getattr(cfg.walk_forward, 'step_size_days', cfg.walk_forward.testing_period_days)
        bar_minutes = getattr(cfg.pair_selection, "bar_minutes", None) or 15
        bar_delta = pd.Timedelta(minutes=bar_minutes)

        # ДОБАВЛЕНО: КРИТИЧЕСКАЯ ПРОВЕРКА на пересечение тестовых периодов
        # Проверяем только если step_size_days явно задан в конфигурации
        if hasattr(cfg.walk_forward, 'step_size_days') and step_size_days < cfg.walk_forward.testing_period_days:
            raise ValueError(
                f"Критическая ошибка конфигурации: Обнаружено пересечение тестовых периодов! "
                f"step_size_days ({step_size_days}) < testing_period_days ({cfg.walk_forward.testing_period_days}). "
                f"Это приводит к утечке данных. Установите step_size_days >= testing_period_days."
            )

        # Генерируем все walk-forward шаги
        walk_forward_steps = []
        current_test_start = start_date

        while current_test_start < end_date:
            training_start = current_test_start - pd.Timedelta(days=cfg.walk_forward.training_period_days)
            training_end = current_test_start - bar_delta
            testing_start = current_test_start
            testing_end = min(
                testing_start + pd.Timedelta(days=cfg.walk_forward.testing_period_days),
                end_date
            )

            # Проверяем что тестовый период не пустой
            if testing_end > testing_start:
                walk_forward_steps.append({
                    'training_start': training_start,
                    'training_end': training_end,
                    'testing_start': testing_start,
                    'testing_end': testing_end
                })

            # Переходим к следующему шагу
            current_test_start += pd.Timedelta(days=step_size_days)

        print(f"🗓️  МНОЖЕСТВЕННЫЕ WALK-FORWARD ШАГИ ({len(walk_forward_steps)} шагов):")
        for i, step in enumerate(walk_forward_steps):
            print(f"   Шаг {i+1}: Тренировка {step['training_start'].strftime('%Y-%m-%d')} -> {step['training_end'].strftime('%Y-%m-%d')}, "
                  f"Тест {step['testing_start'].strftime('%Y-%m-%d')} -> {step['testing_end'].strftime('%Y-%m-%d')}")

        # Для совместимости с существующим кодом используем первый шаг
        if not walk_forward_steps:
            raise ValueError("Не удалось сгенерировать ни одного walk-forward шага")

        first_step = walk_forward_steps[0]
        training_start = first_step['training_start']
        training_end = first_step['training_end']
        testing_start = first_step['testing_start']
        testing_end = first_step['testing_end']

        # Гарантируем, что все временные метки - Timestamp
        testing_start = pd.to_datetime(testing_start)
        testing_end = pd.to_datetime(testing_end)

        # ИСПРАВЛЕНО: Обрабатываем все walk-forward шаги
        all_step_results = []

        for step_idx, step in enumerate(walk_forward_steps):
            print(f"\n🔄 Обработка walk-forward шага {step_idx + 1}/{len(walk_forward_steps)}")

            # Загружаем данные для этого шага
            step_data = self._load_data_for_step(
                step['training_start'], step['training_end'],
                step['testing_start'], step['testing_end']
            )
            step_df = step_data['full_data']

            if step_df is None:
                print(f"   ❌ Нет данных для шага {step_idx + 1}, пропускаем")
                continue

            # Обрабатываем этот шаг
            step_result = self._process_single_walk_forward_step(
                cfg, step_data, step_idx
            )

            if step_result is not None and step_result['pnls']:
                all_step_results.append(step_result)

        # Проверяем что есть результаты
        if not all_step_results:
            print("❌ Нет результатов ни для одного walk-forward шага")
            return {"sharpe_ratio_abs": None, "total_trades": 0, "error_type": "no_wf_steps", "error_message": "No valid walk-forward steps"}

        # Объединяем результаты всех шагов
        all_pnls = []
        total_trades = 0

        for step_result in all_step_results:
            all_pnls.extend(step_result['pnls'])
            total_trades += step_result['trades']

        print(f"\n📊 АГРЕГИРОВАННЫЕ РЕЗУЛЬТАТЫ ВСЕХ {len(all_step_results)} ШАГОВ:")
        print(f"   📈 Всего PnL серий: {len(all_pnls)}")
        print(f"   🔄 Всего сделок: {total_trades}")

        # Инициализируем портфель для совместимости
        portfolio = Portfolio(
            initial_capital=cfg.portfolio.initial_capital,
            max_active_positions=cfg.portfolio.max_active_positions
        )

        # Проверяем что есть результаты для расчета метрик
        if not all_pnls:
            print(f"🔍 ДИАГНОСТИКА: НЕТ PnL ДАННЫХ - возвращаем невалидный результат")
            return {"sharpe_ratio_abs": None, "total_trades": total_trades, "error_type": "no_pnl_data"}




        # КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ: Полноценная симуляция портфеля с реалистичным управлением позициями
        try:
            if len(all_pnls) == 1:
                combined_pnl = all_pnls[0].fillna(0)
            else:
                # Создаем полноценную симуляцию портфеля
                combined_pnl = self._simulate_realistic_portfolio(all_pnls, cfg)

                print(f"📊 РЕАЛИСТИЧНАЯ СИМУЛЯЦИЯ ПОРТФЕЛЯ:")
                print(f"   • Общий PnL: ${combined_pnl.sum():.2f}")
                print(f"   • Макс. дневной PnL: ${combined_pnl.max():.2f}")
                print(f"   • Мин. дневной PnL: ${combined_pnl.min():.2f}")
                print(f"   • Лимит позиций: {cfg.portfolio.max_active_positions}")

        except Exception as e:
            print(f"Ошибка при агрегации PnL: {e}")
            return {"sharpe_ratio_abs": None, "total_trades": total_trades, "error_type": "aggregation_error", "error_message": str(e)}
        
        # Рассчитываем equity curve
        equity_curve = cfg.portfolio.initial_capital + combined_pnl.cumsum()
        daily_returns = equity_curve.pct_change().dropna()
        
        if len(daily_returns) == 0 or daily_returns.std() == 0:
            return {"sharpe_ratio_abs": None, "total_trades": total_trades, "error_type": "insufficient_data_for_sharpe"}
        
        # Рассчитываем Sharpe ratio
        sharpe = daily_returns.mean() / daily_returns.std() * np.sqrt(cfg.backtest.annualizing_factor)
        
        # Рассчитываем максимальную просадку
        max_dd = ((equity_curve.cummax() - equity_curve) / equity_curve.cummax()).max()
        
        # ИСПРАВЛЕНО: Добавлена диагностика производительности
        avg_trade_size = combined_pnl.abs().mean() if len(combined_pnl) > 0 else 0
        commission_to_pnl_ratio = 0  # Упрощенная версия, можно расширить
        avg_hold_time = len(combined_pnl) / max(total_trades, 1)  # Приблизительная оценка
        micro_trades_pct = 0  # Упрощенная версия
        
        # ИСПРАВЛЕНО: Правильный расчет метрик по сделкам, а не по барам
        win_rate = 0.0
        avg_trade_size = 0.0
        avg_hold_time = 0.0

        if total_trades > 0 and len(all_pnls) > 0:
            # Собираем все результаты бэктестов для анализа сделок
            all_trade_pnls = []
            all_hold_times = []

            # Для каждой пары анализируем её результаты
            for pnl_series in all_pnls:
                if len(pnl_series) == 0:
                    continue

                # Создаем фиктивную позицию на основе PnL (упрощение для быстрого расчета)
                # В реальности нужно получать позицию из бэктестера
                position = (pnl_series != 0).astype(int)

                # Находим сделки по изменениям позиции
                trade_start = (position.shift(fill_value=0) == 0) & (position != 0)
                trade_id = trade_start.cumsum()
                trade_id = trade_id.where(position != 0, None)

                if trade_id.notna().any():
                    # PnL по сделкам
                    trade_pnls = pnl_series.groupby(trade_id).sum().dropna()
                    all_trade_pnls.extend(trade_pnls.tolist())

                    # Длительность сделок в барах
                    hold_bars = position.groupby(trade_id).sum().dropna()
                    all_hold_times.extend(hold_bars.tolist())

            # Рассчитываем метрики по всем сделкам
            if all_trade_pnls:
                win_rate = float(sum(1 for pnl in all_trade_pnls if pnl > 0) / len(all_trade_pnls))
                avg_trade_size = float(sum(abs(pnl) for pnl in all_trade_pnls) / len(all_trade_pnls))

            if all_hold_times:
                avg_hold_time = float(sum(all_hold_times) / len(all_hold_times))
        else:
            # Fallback для случая когда нет сделок - используем старую логику для совместимости
            if len(combined_pnl) > 0:
                winning_bars = sum(1 for pnl in combined_pnl if pnl > 0)
                win_rate = winning_bars / len(combined_pnl)
                avg_trade_size = combined_pnl.abs().mean()
                avg_hold_time = len(combined_pnl) / max(total_trades, 1)
        
        print(f"📊 Диагностика производительности:")
        print(f"   • Всего пар в торговле: {len(all_pnls)}")
        print(f"   • Всего сделок: {total_trades}")
        print(f"   • Средний размер сделки: ${avg_trade_size:.2f}")
        print(f"   • Средний hold-time: {avg_hold_time:.1f} баров")
        print(f"   • Максимальная просадка: {max_dd:.2%}")
        print(f"   • Общий P&L: ${combined_pnl.sum():.2f}")
        
        return {
            "sharpe_ratio_abs": float(sharpe),
            "total_trades": total_trades,
            "max_drawdown": float(max_dd),
            "max_drawdown_on_equity": float(max_dd),  # Для совместимости с objective.py
            "total_pnl": float(combined_pnl.sum()),
            "total_return_pct": float(combined_pnl.sum() / cfg.portfolio.initial_capital),
            "win_rate": float(win_rate),
            "avg_trade_size": float(avg_trade_size),
            "avg_hold_time": float(avg_hold_time)
        }

    def _backtest_single_pair(self, pair_row, cfg, step_df=None):
        """Бэктестирование одной пары - оптимизированная версия с переданными данными."""
        try:
            s1, s2 = pair_row['s1'], pair_row['s2']
            beta, mean, std = pair_row['beta'], pair_row['mean'], pair_row['std']

            # КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ ПРОИЗВОДИТЕЛЬНОСТИ:
            # Если step_df передан, используем его БЕЗ повторной загрузки данных
            if step_df is None:
                print(f"⚠️ FALLBACK: Загрузка данных для пары {s1}-{s2} (step_df не передан)")
                # Определяем периоды точно как в _run_fast_backtest
                start_date = pd.to_datetime(cfg.walk_forward.start_date)
                bar_minutes = getattr(cfg.pair_selection, "bar_minutes", None) or 15
                bar_delta = pd.Timedelta(minutes=bar_minutes)

                current_test_start = start_date
                training_start = current_test_start - pd.Timedelta(days=cfg.walk_forward.training_period_days)
                training_end = current_test_start - bar_delta
                testing_start = current_test_start
                testing_end = testing_start + pd.Timedelta(days=cfg.walk_forward.testing_period_days)

                testing_start = pd.to_datetime(testing_start)
                testing_end = pd.to_datetime(testing_end)

                # Загружаем данные только если они действительно не переданы
                step_data = self._load_data_for_step(training_start, training_end, testing_start, testing_end)
                step_df = step_data['full_data']
            # ИСПРАВЛЕНО: Убираем избыточную загрузку данных когда step_df уже передан

            # Проверяем наличие данных для пары
            if s1 not in step_df.columns or s2 not in step_df.columns:
                return None

            # КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ ПРОИЗВОДИТЕЛЬНОСТИ:
            # Для переданного step_df используем простое разделение по времени вместо повторной загрузки
            if step_df is not None and 'step_data' not in locals():
                # Определяем периоды для разделения данных
                start_date = pd.to_datetime(cfg.walk_forward.start_date)
                bar_minutes = getattr(cfg.pair_selection, "bar_minutes", None) or 15
                bar_delta = pd.Timedelta(minutes=bar_minutes)

                current_test_start = start_date
                training_start = current_test_start - pd.Timedelta(days=cfg.walk_forward.training_period_days)
                training_end = current_test_start - bar_delta
                testing_start = current_test_start
                testing_end = testing_start + pd.Timedelta(days=cfg.walk_forward.testing_period_days)

                # Разделяем данные по времени БЕЗ повторной загрузки с диска
                training_data = step_df.loc[training_start:training_end]
                testing_data = step_df.loc[testing_start:testing_end]
            else:
                # Используем уже загруженные данные из step_data
                training_data = step_data['training_data']
                testing_data = step_data['testing_data']

            # Проверяем наличие данных для пары в обоих периодах
            if s1 not in training_data.columns or s2 not in training_data.columns:
                return None
            if s1 not in testing_data.columns or s2 not in testing_data.columns:
                return None

            training_pair_data = training_data[[s1, s2]].dropna()
            testing_pair_data = testing_data[[s1, s2]].dropna()

            if len(training_pair_data) < cfg.backtest.rolling_window + 10:
                return None
            if len(testing_pair_data) < cfg.backtest.rolling_window + 10:
                return None

            # КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ LOOK-AHEAD BIAS: Вычисляем параметры нормализации только из тренировочных данных
            try:
                # Используем безопасные значения по умолчанию, если секция data_processing не существует
                norm_method = getattr(cfg.data_processing, 'normalization_method', 'minmax') if hasattr(cfg, 'data_processing') else 'minmax'
                fill_method = getattr(cfg.data_processing, 'fill_method', 'linear') if hasattr(cfg, 'data_processing') else 'linear'

                # Шаг 1: Вычисляем параметры нормализации ТОЛЬКО из тренировочных данных
                normalization_params = compute_normalization_params(training_pair_data, norm_method)

                # Шаг 2: Применяем параметры нормализации к тестовым данным
                normalized_data = apply_normalization_with_params(
                    testing_pair_data,
                    normalization_params,
                    norm_method=norm_method,
                    fill_method=fill_method
                )

                if normalized_data.empty:
                    return None
            except Exception as e:
                print(f"Ошибка нормализации для пары {s1}-{s2}: {e}")
                return None

            # Создаем временный портфель для этой пары
            temp_portfolio = Portfolio(
                initial_capital=cfg.portfolio.initial_capital,
                max_active_positions=1
            )

            # КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ: Риск на сделку из лимита позиций, а не "на пару"
            base_capital_per_trade = cfg.portfolio.initial_capital * cfg.portfolio.risk_per_position_pct
            # Эффективный капитал учитывает максимальное количество одновременных позиций
            capital_per_pair = base_capital_per_trade

            # КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ: Конвертируем cooldown_hours в cooldown_periods
            cooldown_hours = getattr(cfg.backtest, 'cooldown_hours', 4)
            bar_minutes = getattr(cfg.pair_selection, "bar_minutes", 15)  # Получаем таймфрейм
            cooldown_periods = convert_hours_to_periods(cooldown_hours, bar_minutes)

            # УСКОРЕНИЕ: Создаем полностью Numba-оптимизированный бэктестер для максимального ускорения
            backtester = PairBacktester(
                pair_data=normalized_data,
                rolling_window=cfg.backtest.rolling_window,
                portfolio=temp_portfolio,
                pair_name=f"{s1}-{s2}",
                z_threshold=cfg.backtest.zscore_threshold,
                z_exit=getattr(cfg.backtest, 'zscore_exit', 0.0),
                commission_pct=getattr(cfg.backtest, 'commission_pct', 0.0),
                slippage_pct=getattr(cfg.backtest, 'slippage_pct', 0.0),
                annualizing_factor=getattr(cfg.backtest, 'annualizing_factor', 365),
                capital_at_risk=capital_per_pair,
                stop_loss_multiplier=getattr(cfg.backtest, 'stop_loss_multiplier', 2.0),
                take_profit_multiplier=getattr(cfg.backtest, 'take_profit_multiplier', None),
                time_stop_multiplier=getattr(cfg.backtest, 'time_stop_multiplier', 2.0),
                cooldown_periods=cooldown_periods  # <--- ИСПОЛЬЗУЕМ ИСПРАВЛЕННЫЙ ПАРАМЕТР
            )

            # Запускаем бэктест (FullNumbaPairBacktester не требует установки имен символов)
            backtester.run()
            results = backtester.get_results()

            # ИСПРАВЛЕНО: Корректная обработка результатов (может быть dict или DataFrame)
            if not isinstance(results, dict) or 'pnl' not in results:
                # Логируем, если результат неожиданного типа, для отладки
                if not isinstance(results, dict):
                    print(f"⚠️ Результат бэктеста для {s1}-{s2} не является словарем, тип: {type(results)}")
                return None, 0

            # Дополнительная проверка на пустоту PnL серии
            pnl_series = results.get('pnl')
            if pnl_series is None or (hasattr(pnl_series, 'empty') and pnl_series.empty):
                return None, 0

            # ИСПРАВЛЕНО: Правильный подсчет сделок
            pair_trades = 0
            if 'trades' in results:
                pair_trades = int(results['trades'].sum())
            else:
                # Альтернативный способ: считаем изменения позиций
                if 'position' in results:
                    position_changes = results['position'].diff().fillna(0)
                    pair_trades = int((position_changes != 0).sum())

            # Возвращаем PnL серию и количество сделок
            # ИСПРАВЛЕНО: Убираем двойное масштабирование PnL - FullNumbaPairBacktester уже возвращает PnL в долларах
            return results['pnl'], pair_trades

        except Exception as e:
            print(f"Ошибка при обработке пары {pair_row.get('s1', 'unknown')}: {e}")
            return None, 0

    def _simulate_realistic_portfolio(self, all_pnls, cfg):
        """
        КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ: Полноценная симуляция портфеля с реалистичным управлением позициями.

        Вместо простого суммирования PnL всех пар, симулируем реальную работу портфеля:
        1. На каждом временном шаге собираем сигналы от всех пар
        2. Применяем лимит max_active_positions
        3. Выбираем лучшие сигналы для торговли
        4. Рассчитываем реалистичный PnL портфеля

        Args:
            all_pnls: Список PnL серий от всех пар
            cfg: Конфигурация с параметрами портфеля

        Returns:
            pd.Series: Реалистичный PnL портфеля с учетом лимитов позиций
        """
        if not all_pnls:
            return pd.Series(dtype=float)

        # Создаем DataFrame со всеми PnL сериями
        pnl_df = pd.concat({f'pair_{i}': pnl.fillna(0) for i, pnl in enumerate(all_pnls)}, axis=1)

        # Создаем DataFrame с сигналами (позиция активна если PnL != 0)
        signals_df = pd.concat({f'pair_{i}': (pnl != 0).astype(int) for i, pnl in enumerate(all_pnls)}, axis=1)

        # Инициализируем портфель
        max_positions = cfg.portfolio.max_active_positions
        portfolio_pnl = pd.Series(0.0, index=pnl_df.index)
        active_positions = {}  # {pair_name: entry_timestamp}

        print(f"🎯 СИМУЛЯЦИЯ ПОРТФЕЛЯ: {len(all_pnls)} пар, лимит {max_positions} позиций")

        # Симулируем торговлю по каждому временному шагу
        for timestamp in pnl_df.index:
            current_signals = signals_df.loc[timestamp]
            current_pnls = pnl_df.loc[timestamp]

            # 1. Закрываем позиции, которые больше не активны
            positions_to_close = []
            for pair_name in list(active_positions.keys()):
                if current_signals[pair_name] == 0:  # Сигнал исчез
                    positions_to_close.append(pair_name)

            for pair_name in positions_to_close:
                del active_positions[pair_name]

            # 2. Ищем новые сигналы для открытия позиций
            new_signals = []
            for pair_name in current_signals.index:
                if current_signals[pair_name] == 1 and pair_name not in active_positions:
                    # Новый сигнал от пары, которая не в портфеле
                    new_signals.append((pair_name, abs(current_pnls[pair_name])))

            # 3. Сортируем новые сигналы по силе (абсолютный PnL)
            new_signals.sort(key=lambda x: x[1], reverse=True)

            # 4. Открываем новые позиции в пределах лимита
            available_slots = max_positions - len(active_positions)
            for i, (pair_name, signal_strength) in enumerate(new_signals):
                if i >= available_slots:
                    break  # Достигли лимита позиций
                active_positions[pair_name] = timestamp

            # 5. Рассчитываем PnL портфеля на этом шаге
            step_pnl = 0.0
            for pair_name in active_positions:
                step_pnl += current_pnls[pair_name]

            portfolio_pnl[timestamp] = step_pnl

        # Диагностика
        total_signals = signals_df.sum(axis=1)
        avg_active_pairs = len([p for p in active_positions]) if active_positions else 0
        max_signals = total_signals.max()
        avg_signals = total_signals.mean()

        print(f"   📈 Макс. одновременных сигналов: {max_signals}")
        print(f"   📊 Средн. сигналов за период: {avg_signals:.1f}")
        print(f"   🎯 Финальных активных позиций: {avg_active_pairs}")

        utilization = (total_signals.clip(upper=max_positions) / max_positions).mean()
        print(f"   ⚡ Утилизация лимита позиций: {utilization:.1%}")

        return portfolio_pnl

    def _run_fast_backtest_with_reports(self, params, trial):
        """Запускает быстрый бэктест с промежуточными отчетами для pruning."""
        # ИСПРАВЛЕНО: Убираем дублированный импорт optuna (уже импортирован в начале файла)

        # Используем ту же логику что и в _run_fast_backtest, но с отчетами
        cfg = self.base_config.model_copy(deep=True)

        # Применяем параметры (сокращенная версия)
        # ИСПРАВЛЕНО: Безопасное применение параметров с проверкой существования секций
        for key, value in params.items():
            if key in ["ssd_top_n", "kpss_pvalue_threshold", "coint_pvalue_threshold",
                      "min_half_life_days", "max_half_life_days", "min_mean_crossings"]:
                if hasattr(cfg, 'pair_selection'):
                    setattr(cfg.pair_selection, key, value)
            elif key in ["zscore_threshold", "zscore_exit", "rolling_window", "stop_loss_multiplier",
                        "time_stop_multiplier", "cooldown_hours", "commission_pct", "slippage_pct"]:
                if hasattr(cfg, 'backtest'):
                    setattr(cfg.backtest, key, value)
            elif key in ["max_active_positions", "risk_per_position_pct", "max_position_size_pct"]:
                if hasattr(cfg, 'portfolio'):
                    setattr(cfg.portfolio, key, value)
            elif key in ["normalization_method", "min_history_ratio"]:
                if hasattr(cfg, 'data_processing'):
                    setattr(cfg.data_processing, key, value)

        # ИСПРАВЛЕНО: Генерируем множественные walk-forward шаги
        start_date = pd.to_datetime(cfg.walk_forward.start_date)
        end_date = pd.to_datetime(getattr(cfg.walk_forward, 'end_date', start_date + pd.Timedelta(days=cfg.walk_forward.testing_period_days)))
        step_size_days = getattr(cfg.walk_forward, 'step_size_days', cfg.walk_forward.testing_period_days)

        # Генерируем все walk-forward шаги
        walk_forward_steps = []
        current_test_start = start_date
        bar_minutes = getattr(cfg.pair_selection, "bar_minutes", None) or 15
        bar_delta = pd.Timedelta(minutes=bar_minutes)

        while current_test_start < end_date:
            training_start = current_test_start - pd.Timedelta(days=cfg.walk_forward.training_period_days)
            training_end = current_test_start - bar_delta
            testing_start = current_test_start
            testing_end = min(testing_start + pd.Timedelta(days=cfg.walk_forward.testing_period_days), end_date)

            walk_forward_steps.append({
                'training_start': training_start,
                'training_end': training_end,
                'testing_start': testing_start,
                'testing_end': testing_end
            })

            current_test_start += pd.Timedelta(days=step_size_days)

        print(f"\n🔄 ГЕНЕРИРОВАНО {len(walk_forward_steps)} WALK-FORWARD ШАГОВ (с отчетами)")

        # ИСПРАВЛЕНО: Обрабатываем все walk-forward шаги
        all_step_results = []

        for step_idx, step in enumerate(walk_forward_steps):
            print(f"\n🔄 Обработка walk-forward шага {step_idx + 1}/{len(walk_forward_steps)} (с отчетами)")

            # Загружаем данные для этого шага
            step_data = self._load_data_for_step(
                step['training_start'], step['training_end'],
                step['testing_start'], step['testing_end']
            )
            step_df = step_data['full_data']

            if step_df is None:
                print(f"   ❌ Нет данных для шага {step_idx + 1}, пропускаем")
                continue

            # Обрабатываем этот шаг
            step_result = self._process_single_walk_forward_step(
                cfg, step_data, step_idx
            )

            if step_result is not None and step_result['pnls']:
                all_step_results.append(step_result)

        # Проверяем что есть результаты
        if not all_step_results:
            print("❌ Нет результатов ни для одного walk-forward шага")
            return {"sharpe_ratio_abs": PENALTY_SOFT, "total_trades": 0, "max_drawdown": 0, "win_rate": 0}

        # Объединяем результаты всех шагов
        all_pnls = []
        total_trades = 0

        for step_result in all_step_results:
            all_pnls.extend(step_result['pnls'])
            total_trades += step_result['trades']

        print(f"\n📊 АГРЕГИРОВАННЫЕ РЕЗУЛЬТАТЫ ВСЕХ {len(all_step_results)} ШАГОВ (с отчетами):")
        print(f"   📈 Всего PnL серий: {len(all_pnls)}")
        print(f"   🔄 Всего сделок: {total_trades}")

        # ИСПРАВЛЕНО: Промежуточные отчеты на накопленных данных (без lookahead bias)
        accumulated_pnls = []

        # Промежуточные отчеты по шагам для pruning
        for step_idx, step_result in enumerate(all_step_results):
            try:
                # Добавляем PnL текущего шага к накопленным данным
                step_pnls = step_result['pnls']
                if step_pnls:
                    accumulated_pnls.extend(step_pnls)

                    # Рассчитываем промежуточную метрику на НАКОПЛЕННЫХ данных до текущего шага
                    if len(accumulated_pnls) > 0:
                        # ИСПРАВЛЕНО: Используем реалистичную симуляцию портфеля вместо простого суммирования
                        if len(accumulated_pnls) == 1:
                            combined_pnl = accumulated_pnls[0].fillna(0)
                        else:
                            combined_pnl = self._simulate_realistic_portfolio(accumulated_pnls, cfg)
                        equity_curve = cfg.portfolio.initial_capital + combined_pnl.cumsum()
                        daily_returns = equity_curve.resample('1D').last().pct_change().dropna()

                        if len(daily_returns) > 0 and daily_returns.std() > 0:
                            intermediate_sharpe = daily_returns.mean() / daily_returns.std() * np.sqrt(cfg.backtest.annualizing_factor)

                            # ИСПРАВЛЕНО: Используем step_idx для консистентности с walk-forward шагами
                            trial.report(float(intermediate_sharpe), step=step_idx)
                            print(f"   📊 Промежуточный отчет шаг {step_idx}: Sharpe={intermediate_sharpe:.4f} (накоплено {len(daily_returns)} дней)")

                            # Проверяем pruning
                            if trial.should_prune():
                                print(f"Trial pruned at walk-forward step {step_idx} (шаг {step_idx + 1}/{len(all_step_results)})")
                                raise optuna.TrialPruned(f"Pruned at step {step_idx}")

            except optuna.TrialPruned:
                raise  # Пробрасываем pruning
            except Exception as e:
                print(f"Ошибка промежуточного отчета для шага {step_idx + 1}: {e}")

        # Финальный расчет (упрощенная версия)
        if not all_pnls:
            return {"sharpe_ratio_abs": PENALTY_SOFT, "total_trades": total_trades, "max_drawdown": 0, "win_rate": 0}

        try:
            # ИСПРАВЛЕНО: Используем реалистичную симуляцию портфеля вместо простого суммирования
            if len(all_pnls) == 1:
                combined_pnl = all_pnls[0].fillna(0)
            else:
                combined_pnl = self._simulate_realistic_portfolio(all_pnls, cfg)
            equity_curve = cfg.portfolio.initial_capital + combined_pnl.cumsum()
            # ИСПРАВЛЕНО: Правильный расчет дневных доходностей для 15-минутных данных
            daily_returns = equity_curve.resample('1D').last().pct_change().dropna()

            if len(daily_returns) == 0 or daily_returns.std() == 0:
                return {"sharpe_ratio_abs": PENALTY_SOFT, "total_trades": total_trades, "max_drawdown": 0, "win_rate": 0}

            sharpe = daily_returns.mean() / daily_returns.std() * np.sqrt(cfg.backtest.annualizing_factor)
            running_max = equity_curve.expanding().max()
            drawdown = (equity_curve - running_max) / running_max
            max_dd = abs(drawdown.min()) if len(drawdown) > 0 else 0

            # ИСПРАВЛЕНО: Рассчитываем win_rate по сделкам, а не по дням
            # Собираем все PnL сделок из всех шагов
            all_trade_pnls = []
            for step_result in all_step_results:
                if 'trade_pnls' in step_result:
                    all_trade_pnls.extend(step_result['trade_pnls'])

            # Рассчитываем win_rate по сделкам
            if all_trade_pnls:
                win_rate = float(sum(1 for pnl in all_trade_pnls if pnl > 0) / len(all_trade_pnls))
            else:
                # Fallback: используем дневные данные если нет информации о сделках
                daily_pnl = combined_pnl.resample('1D').sum()
                win_rate = float((daily_pnl > 0).mean()) if len(daily_pnl) > 0 else 0.0

            return {"sharpe_ratio_abs": sharpe, "total_trades": total_trades, "max_drawdown": max_dd, "win_rate": win_rate}

        except Exception as e:
            print(f"Ошибка финального расчета: {e}")
            return {"sharpe_ratio_abs": PENALTY_HARD, "total_trades": total_trades, "max_drawdown": 0, "win_rate": 0}

    def quick_trial_filter(self, params):
        """
        ОПТИМИЗАЦИЯ: Быстрая предварительная фильтрация заведомо плохих параметров.

        Args:
            params: Словарь параметров trial

        Returns:
            tuple: (is_valid, reason) - валидность и причина отклонения
        """
        # Проверяем логичность соотношений параметров
        zscore_threshold = params.get('zscore_threshold', 1.0)
        zscore_exit = params.get('zscore_exit', 0.3)

        # Проверка 1: zscore_exit должен быть меньше zscore_threshold
        if zscore_exit >= zscore_threshold:
            return False, f"zscore_exit ({zscore_exit:.3f}) >= zscore_threshold ({zscore_threshold:.3f})"

        # Проверка 2: Разумный гистерезис (разница между порогами)
        hysteresis = zscore_threshold - zscore_exit
        if hysteresis < 0.1:
            return False, f"Слишком маленький гистерезис: {hysteresis:.3f}"
        if hysteresis > 1.0:
            return False, f"Слишком большой гистерезис: {hysteresis:.3f}"

        # Проверка 3: Разумные размеры позиций
        risk_per_position = params.get('risk_per_position_pct', 0.02)
        max_position_size = params.get('max_position_size_pct', 0.1)
        max_positions = params.get('max_active_positions', 15)

        # Максимальная экспозиция не должна превышать 100%
        max_exposure = risk_per_position * max_positions
        if max_exposure > 1.0:
            return False, f"Слишком большая экспозиция: {max_exposure:.1%}"

        # Проверка 4: Разумные стоп-лоссы
        stop_loss_mult = params.get('stop_loss_multiplier', 3.0)
        time_stop_mult = params.get('time_stop_multiplier', 5.0)

        if stop_loss_mult < 1.5:
            return False, f"Слишком агрессивный стоп-лосс: {stop_loss_mult}"
        if time_stop_mult < stop_loss_mult:
            return False, f"time_stop_multiplier ({time_stop_mult}) < stop_loss_multiplier ({stop_loss_mult})"

        return True, "OK"

    def _get_cached_data(self, cache_key):
        """
        ОПТИМИЗАЦИЯ: Получает данные из кэша.

        Args:
            cache_key: Ключ кэша

        Returns:
            Данные или None если не найдены
        """
        with self.data_cache_lock:
            return self.data_cache.get(cache_key)

    def _cache_data(self, cache_key, data):
        """
        ОПТИМИЗАЦИЯ: Сохраняет данные в кэш с ограничением размера.

        Args:
            cache_key: Ключ кэша
            data: Данные для сохранения
        """
        with self.data_cache_lock:
            # Ограничиваем размер кэша
            if len(self.data_cache) >= self.max_cache_size:
                # Удаляем самый старый элемент (FIFO)
                oldest_key = next(iter(self.data_cache))
                del self.data_cache[oldest_key]

            self.data_cache[cache_key] = data

    def __call__(self, trial_or_params):
        """Унифицированная функция для совместимости с objective.py.

        Args:
            trial_or_params: optuna.Trial объект или словарь параметров

        Returns:
            float: Значение целевой функции
        """
        # ИСПРАВЛЕНИЕ: Инициализируем глобальный кэш в каждом дочернем процессе
        from src.coint2.core.global_rolling_cache import get_global_rolling_manager
        manager = get_global_rolling_manager()
        if not manager.initialized:
            print(f"🔄 Инициализация глобального кэша в дочернем процессе (PID: {os.getpid()})")
            cache_initialized = self._initialize_global_rolling_cache()
            if cache_initialized:
                print(f"✅ Глобальный кэш инициализирован в процессе {os.getpid()}")
            else:
                print(f"❌ Не удалось инициализировать кэш в процессе {os.getpid()}")

        # Определяем тип входных данных и извлекаем параметры
        if hasattr(trial_or_params, 'suggest_float'):  # Это optuna.Trial
            trial = trial_or_params
            params = self._suggest_parameters(trial)
            trial_number = trial.number
        else:  # Это словарь параметров
            params = trial_or_params
            trial_number = params.get("trial_number", -1)

        # ОПТИМИЗАЦИЯ: Быстрая предварительная фильтрация
        is_valid, reason = self.quick_trial_filter(params)
        if not is_valid:
            logger.info(f"Trial #{trial_number}: Быстро отклонен - {reason}")
            if hasattr(trial_or_params, 'suggest_float') and hasattr(trial_or_params, "set_user_attr"):
                trial_or_params.set_user_attr("quick_filter_reason", reason)
                raise optuna.TrialPruned(f"Quick filter: {reason}")
            return PENALTY_SOFT

        try:
            # ИСПРАВЛЕНО: Правильная обработка ошибок валидации через TrialPruned
            try:
                validated_params = validate_params(params)
            except ValueError as e:
                logger.warning(f"Trial #{trial_number}: Невалидные параметры: {e}")
                if hasattr(trial_or_params, 'suggest_float') and hasattr(trial_or_params, "set_user_attr"):
                    trial_or_params.set_user_attr("error_type", "validation_error")
                    trial_or_params.set_user_attr("validation_message", str(e))
                    trial_or_params.set_user_attr("invalid_params", params)
                    raise optuna.TrialPruned(f"Parameter validation failed: {e}")
                # ИСПРАВЛЕНО: Убираем недостижимый код после raise
                return PENALTY_SOFT
            
            # Запускаем быстрый бэктест с промежуточными отчетами (если это trial)
            if hasattr(trial_or_params, 'suggest_float'):
                metrics = self._run_fast_backtest_with_reports(validated_params, trial)
            else:
                metrics = self._run_fast_backtest(validated_params)
            
            # Используем единую функцию extract_sharpe
            sharpe = extract_sharpe(metrics)
            
            # ИСПРАВЛЕНО: Правильная обработка невалидных Sharpe ratio через TrialPruned
            if sharpe is None or not isinstance(sharpe, (int, float)) or np.isnan(sharpe) or np.isinf(sharpe):
                logger.warning(f"Trial #{trial_number}: Невалидный Sharpe ratio: {sharpe}")
                if hasattr(trial_or_params, 'suggest_float') and hasattr(trial_or_params, "set_user_attr"):
                    trial_or_params.set_user_attr("error_type", "invalid_sharpe")
                    trial_or_params.set_user_attr("sharpe_value", str(sharpe))
                    trial_or_params.set_user_attr("metrics_available", list(metrics.keys()) if metrics else [])
                    raise optuna.TrialPruned(f"Invalid Sharpe ratio: {sharpe}")
                # ИСПРАВЛЕНО: Убираем недостижимый код после raise
                return PENALTY_SOFT
            
            logger.debug(f"Trial #{trial_number}: {metrics.get('total_trades', 0)} сделок, Sharpe: {sharpe:.4f}")
            
            # ИСПРАВЛЕНО: Унифицированы штрафы с objective.py
            max_dd = metrics.get("max_drawdown", 0)

            # ИСПРАВЛЕНО: Используем win_rate вместо positive_days_rate для более корректной оптимизации
            win_rate = metrics.get('win_rate', 0.0)

            # Константы для win_rate бонусов/штрафов
            WIN_RATE_BONUS_THRESHOLD = 0.55  # 55% win rate для бонуса
            WIN_RATE_BONUS_MULTIPLIER = 0.5  # Множитель бонуса
            WIN_RATE_PENALTY_THRESHOLD = 0.40  # 40% win rate для штрафа
            WIN_RATE_PENALTY_MULTIPLIER = 1.0  # Множитель штрафа

            # Бонус за высокий win rate (> 55%)
            win_rate_bonus = 0
            if win_rate > WIN_RATE_BONUS_THRESHOLD:
                win_rate_bonus = (win_rate - WIN_RATE_BONUS_THRESHOLD) * WIN_RATE_BONUS_MULTIPLIER

            # Штраф за низкий win rate (< 40%)
            win_rate_penalty = 0
            if win_rate < WIN_RATE_PENALTY_THRESHOLD:
                win_rate_penalty = (WIN_RATE_PENALTY_THRESHOLD - win_rate) * WIN_RATE_PENALTY_MULTIPLIER
            
            # ИСПРАВЛЕНО: Правильная обработка недостаточного количества сделок через TrialPruned
            total_trades = metrics.get('total_trades', 0)
            win_rate = metrics.get('win_rate', 0.0)  # ИСПРАВЛЕНО: Добавляем получение win_rate из metrics
            logger.debug(f"Trial #{trial_number}: {total_trades} сделок, Sharpe: {sharpe:.4f}")

            # Используем более низкий порог для walk-forward (так как период короткий)
            min_trades_wf = 5  # Для walk-forward достаточно 5 сделок
            if total_trades < min_trades_wf:
                logger.warning(f"Trial #{trial_number}: Недостаточно сделок ({total_trades} < {min_trades_wf})")
                if hasattr(trial_or_params, 'suggest_float') and hasattr(trial_or_params, "set_user_attr"):
                    trial_or_params.set_user_attr("error_type", "insufficient_trades")
                    trial_or_params.set_user_attr("trades_count", total_trades)
                    trial_or_params.set_user_attr("min_required", min_trades_wf)
                    raise optuna.TrialPruned(f"Insufficient trades: {total_trades} < {min_trades_wf}")
                # ИСПРАВЛЕНО: Убираем недостижимый код после raise
                return PENALTY_SOFT

            # ИСПРАВЛЕНО: Единый расчет dd_penalty с использованием констант
            dd_penalty = 0
            if max_dd > MAX_DRAWDOWN_SOFT_THRESHOLD:
                dd_penalty = (max_dd - MAX_DRAWDOWN_SOFT_THRESHOLD) * DD_PENALTY_SOFT_MULTIPLIER

            if max_dd > MAX_DRAWDOWN_HARD_THRESHOLD:
                dd_penalty += (max_dd - MAX_DRAWDOWN_HARD_THRESHOLD) * DD_PENALTY_HARD_MULTIPLIER

            # ИСПРАВЛЕНО: Используем уже рассчитанные выше positive_days_bonus и positive_days_penalty
            # Не пересчитываем их здесь, чтобы избежать дублирования логики

            # BEST PRACTICE: Анти-чурн штраф за частые сделки
            # Получаем настройки из search_space или используем дефолты
            anti_churn_penalty_coeff = 0.02
            max_trades_per_day = 5

            if hasattr(self, 'search_space') and 'metrics' in self.search_space:
                metrics_config = self.search_space['metrics']
                anti_churn_penalty_coeff = metrics_config.get('anti_churn_penalty', 0.02)
                max_trades_per_day = metrics_config.get('max_trades_per_day', 5)

            # ИСПРАВЛЕНО: Правильный расчет торговых дней по фактическим данным
            # Используем консервативную оценку торговых дней
            calendar_days = self.base_config.walk_forward.testing_period_days
            trading_days = max(1, int(calendar_days * 0.7))  # Консервативная оценка (~70% от календарных дней)

            trades_per_day = total_trades / trading_days

            # Штраф за превышение лимита сделок в день
            anti_churn_penalty = anti_churn_penalty_coeff * max(0, trades_per_day - max_trades_per_day)

            # ИСПРАВЛЕНО: Инициализируем переменные для пропущенных пар
            pairs_skipped = 0  # В fast-режиме пары предотобраны, поэтому пропусков нет
            skipped_ratio = 0.0
            skipped_penalty = 0.0

            # ИСПРАВЛЕНО: Используем win_rate вместо positive_days_rate для корректной оптимизации
            final_score = sharpe - dd_penalty + win_rate_bonus - win_rate_penalty - anti_churn_penalty - skipped_penalty

            # Сохраняем детальные метрики в trial (если это Optuna trial)
            if hasattr(trial_or_params, 'suggest_float') and hasattr(trial_or_params, "set_user_attr"):
                # Получаем zscore параметры для логирования
                zscore_threshold = validated_params.get('zscore_threshold', 0)
                zscore_exit = validated_params.get('zscore_exit', 0)
                hysteresis = zscore_threshold - zscore_exit if zscore_threshold > zscore_exit else 0
                rolling_window = validated_params.get('rolling_window', 0)

                trial_or_params.set_user_attr("metrics", {
                    "sharpe": float(sharpe),
                    "max_drawdown": float(max_dd),
                    "win_rate": float(win_rate),
                    "total_trades": int(total_trades),
                    "trades_per_day": float(trades_per_day),
                    "zscore_threshold": float(zscore_threshold),
                    "zscore_exit": float(zscore_exit),
                    "hysteresis": float(hysteresis),
                    "rolling_window": int(rolling_window),
                    "dd_penalty": float(dd_penalty),
                    "win_rate_bonus": float(win_rate_bonus),  # ИСПРАВЛЕНО: используем win_rate_bonus
                    "win_rate_penalty": float(win_rate_penalty),  # ИСПРАВЛЕНО: используем win_rate_penalty
                    "anti_churn_penalty": float(anti_churn_penalty),
                    "skipped_penalty": float(skipped_penalty),  # ИСПРАВЛЕНО: штраф за пропуски
                    "pairs_skipped": int(pairs_skipped),  # ИСПРАВЛЕНО: количество пропущенных пар
                    "skipped_ratio": float(skipped_ratio),  # ИСПРАВЛЕНО: доля пропусков
                    "final_score": float(final_score)
                })

                # Логируем успешный результат (используем уже определенный trial_number)
                logger.info(f"Trial #{trial_number}: SUCCESS - "
                           f"Sharpe={sharpe:.4f}, Trades={total_trades}, DD={max_dd:.2%}, Score={final_score:.4f}")

            return final_score
            
        except optuna.TrialPruned:
            # Пробрасываем TrialPruned без изменений
            raise
        except Exception as e:
            # ИСПРАВЛЕНО: Различаем предсказуемые проблемы данных и системные ошибки
            error_type = type(e).__name__
            error_msg = str(e)

            # ИСПРАВЛЕНО: Более точная категоризация ошибок
            data_related_errors = [
                "ValueError",  # Проблемы с данными, параметрами
                "KeyError",    # Отсутствующие колонки/ключи
                "IndexError",  # Проблемы с индексацией данных
            ]

            # ZeroDivisionError может быть как проблемой данных, так и логической ошибкой
            # Обрабатываем отдельно для лучшей диагностики
            calculation_errors = [
                "ZeroDivisionError",  # Деление на ноль в расчетах
                "FloatingPointError",  # Проблемы с вычислениями
            ]

            if error_type in data_related_errors or "data" in error_msg.lower() or "empty" in error_msg.lower():
                logger.warning(f"Trial #{trial_number}: Предсказуемая проблема данных ({error_type}): {error_msg}")
                if hasattr(trial_or_params, 'suggest_float') and hasattr(trial_or_params, "set_user_attr"):
                    trial_or_params.set_user_attr("error_type", "data_problem")
                    trial_or_params.set_user_attr("exception_type", error_type)
                    trial_or_params.set_user_attr("exception_message", error_msg)
                    raise optuna.TrialPruned(f"Data problem: {error_type} - {error_msg}")
                # ИСПРАВЛЕНО: Убираем недостижимый код после raise
                return PENALTY_SOFT
            elif error_type in calculation_errors:
                # ИСПРАВЛЕНО: Отдельная обработка вычислительных ошибок
                logger.warning(f"Trial #{trial_number}: Вычислительная ошибка ({error_type}): {error_msg}")
                if hasattr(trial_or_params, 'suggest_float') and hasattr(trial_or_params, "set_user_attr"):
                    trial_or_params.set_user_attr("error_type", "calculation_error")
                    trial_or_params.set_user_attr("exception_type", error_type)
                    trial_or_params.set_user_attr("exception_message", error_msg)
                    # Для вычислительных ошибок используем pruning (обычно проблема параметров)
                    raise optuna.TrialPruned(f"Calculation error: {error_type} - {error_msg}")
                return PENALTY_SOFT
            else:
                # Системные ошибки - логируем и возвращаем FAIL через исключение
                logger.error(f"Trial #{trial_number}: Системная ошибка ({error_type}): {error_msg}")
                import traceback
                logger.error(traceback.format_exc())
                if hasattr(trial_or_params, 'suggest_float') and hasattr(trial_or_params, "set_user_attr"):
                    trial_or_params.set_user_attr("error_type", "system_error")
                    trial_or_params.set_user_attr("exception_type", error_type)
                    trial_or_params.set_user_attr("exception_message", error_msg)
                # Для системных ошибок пробрасываем исключение, чтобы trial получил TrialState.FAIL
                raise
