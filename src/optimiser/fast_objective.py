"""
Быстрая целевая функция для Optuna.
Использует предварительно отобранные пары для ускорения оптимизации.
"""

import optuna
import pandas as pd
import numpy as np
from pathlib import Path
import yaml
import optuna

from coint2.utils.config import load_config
from coint2.core.data_loader import DataHandler, load_master_dataset
from coint2.engine.numba_engine import NumbaPairBacktester as PairBacktester
from coint2.core.portfolio import Portfolio
from coint2.core.math_utils import calculate_ssd
from coint2.pipeline.filters import filter_pairs_by_coint_and_half_life
from coint2.core.normalization_improvements import preprocess_and_normalize_data
from coint2.utils.logging_utils import get_logger
from src.optimiser.metric_utils import extract_sharpe, normalize_params, validate_params

# Импортируем константы из единого источника
from .constants import PENALTY, MIN_TRADES_THRESHOLD, MAX_DRAWDOWN_SOFT_THRESHOLD, MAX_DRAWDOWN_HARD_THRESHOLD, \
    WIN_RATE_BONUS_THRESHOLD, WIN_RATE_PENALTY_THRESHOLD, DD_PENALTY_SOFT_MULTIPLIER, DD_PENALTY_HARD_MULTIPLIER, \
    WIN_RATE_BONUS_MULTIPLIER, WIN_RATE_PENALTY_MULTIPLIER, INTERMEDIATE_REPORT_INTERVAL


def convert_hours_to_periods(hours: float, bar_minutes: int) -> int:
    """
    Convert hours to number of periods based on bar timeframe.
    Точно как в walk_forward_orchestrator.py
    """
    if hours <= 0:
        return 0
    return int(hours * 60 / bar_minutes)

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
        
        # Проверяем наличие предварительно отобранных пар
        pairs_file = Path("outputs/preselected_pairs.csv")
        if not pairs_file.exists():
            print("📊 Файл preselected_pairs.csv не найден. Запускаем автоматический отбор пар с новыми периодами...")
            self._run_pair_selection()
        
        # Загружаем предварительно отобранные пары
        self.preselected_pairs = pd.read_csv(pairs_file)
        print(f"✅ Загружено {len(self.preselected_pairs)} предотобранных пар")

        # КРИТИЧЕСКОЕ ПРЕДУПРЕЖДЕНИЕ: Проверяем, что в search space нет параметров фильтрации
        filter_params = ['ssd_top_n', 'kpss_pvalue_threshold', 'coint_pvalue_threshold',
                        'min_half_life_days', 'max_half_life_days', 'min_mean_crossings']

        if 'filters' in self.search_space:
            print("⚠️  КРИТИЧЕСКОЕ ПРЕДУПРЕЖДЕНИЕ: В search space найдена группа 'filters'!")
            print("   Параметры фильтрации НЕ ВЛИЯЮТ на результат в быстрой оптимизации,")
            print("   так как пары уже предотобраны. Используйте search_space_fast.yaml")
            print("   или перенесите отбор пар в __call__() для корректной оптимизации.")

        for group_name, group_params in self.search_space.items():
            if isinstance(group_params, dict):
                for param_name in group_params.keys():
                    if param_name in filter_params:
                        print(f"⚠️  ПРЕДУПРЕЖДЕНИЕ: Параметр '{param_name}' в группе '{group_name}'")
                        print(f"   НЕ ВЛИЯЕТ на результат - пары уже предотобраны!")

        # Данные будут загружаться динамически для каждого шага как в оригинальном бэктесте
    
    def _run_pair_selection(self):
        """Автоматически запускает отбор пар с новыми периодами walk_forward."""
        logger = get_logger("fast_objective_pair_selection")
        
        # ТОЧНО как в оригинальной системе: определяем периоды для первого шага
        start_date = pd.to_datetime(self.base_config.walk_forward.start_date)  # Используем дату из конфигурации
        bar_minutes = getattr(self.base_config.pair_selection, "bar_minutes", None) or 15
        bar_delta = pd.Timedelta(minutes=bar_minutes)
        
        # Первый шаг walk-forward (точно как в walk_forward_orchestrator)
        current_test_start = start_date
        training_start = current_test_start - pd.Timedelta(days=self.base_config.walk_forward.training_period_days)
        training_end = current_test_start - bar_delta
        testing_start = current_test_start
        testing_end = testing_start + pd.Timedelta(days=self.base_config.walk_forward.testing_period_days)
        
        print(f"🗓️  ПЕРВЫЙ WALK-FORWARD ШАГ С НОВЫМИ ПЕРИОДАМИ:")
        print(f"   Тренировка: {training_start.strftime('%Y-%m-%d')} -> {training_end.strftime('%Y-%m-%d')} ({self.base_config.walk_forward.training_period_days} дней)")
        print(f"   Тестирование: {testing_start.strftime('%Y-%m-%d')} -> {testing_end.strftime('%Y-%m-%d')} ({self.base_config.walk_forward.testing_period_days} дней)")
        
        # Загружаем данные ТОЧНО как в preselect_pairs.py
        handler = DataHandler(self.base_config)
        print("📈 Загрузка данных...")
        
        try:
            # Загружаем данные за весь период (тренировка + тестирование) - как в preselect_pairs.py
            full_range_start = training_start
            full_range_end = testing_end

            raw_data = load_master_dataset(
                data_path=self.base_config.data_dir,
                start_date=full_range_start,
                end_date=full_range_end
            )
            
            if raw_data.empty:
                raise ValueError("Не удалось загрузить данные")

            print(f"📊 Загружено {raw_data.shape[0]} записей для {len(raw_data['symbol'].unique())} символов")

            # ТОЧНО как в preselect_pairs.py: преобразуем в pivot table
            step_df = raw_data.pivot_table(index="timestamp", columns="symbol", values="close")
            print(f"📊 Pivot table: {step_df.shape}")
            
            training_slice = step_df.loc[training_start:training_end]
            print(f"📊 Тренировочный срез: {training_slice.shape}")
            
            if training_slice.empty or len(training_slice.columns) < 2:
                raise ValueError("Недостаточно данных для обучения")
            
            # Нормализация данных
            min_history_ratio = getattr(self.base_config.pair_selection, "min_history_ratio", 0.8)
            fill_method = getattr(self.base_config.pair_selection, "fill_method", "forward")
            norm_method = getattr(self.base_config.pair_selection, "norm_method", "minmax")
            handle_constant = getattr(self.base_config.pair_selection, "handle_constant", "drop")
            
            normalized_training, norm_stats = preprocess_and_normalize_data(
                training_slice,
                min_history_ratio=min_history_ratio,
                fill_method=fill_method,
                norm_method=norm_method,
                handle_constant=handle_constant
            )
            
            print(f"📊 Нормализованные данные: {normalized_training.shape}")
            
            # Сканирование пар
            ssd = calculate_ssd(normalized_training, top_k=None)
            print(f"  SSD результат (все пары): {len(ssd)} пар")
            
            # ИСПРАВЛЕНО: Фильтрация по котировочной валюте (*USDT)
            usdt_ssd = ssd[ssd.index.map(lambda x: x[0].endswith('USDT') and x[1].endswith('USDT'))]
            print(f"📊 Фильтрация пар:")
            print(f"   • Исходно после SSD: {len(ssd)} пар")
            print(f"   • После фильтрации по USDT: {len(usdt_ssd)} пар (отсеяно: {len(ssd) - len(usdt_ssd)})")
            
            # Берем только top-N пар для дальнейшей фильтрации
            ssd_top_n = self.base_config.pair_selection.ssd_top_n
            if len(usdt_ssd) > ssd_top_n:
                print(f"   • Ограничиваем до top-{ssd_top_n} пар для дальнейшей обработки (отсеяно: {len(usdt_ssd) - ssd_top_n})")
                usdt_ssd = usdt_ssd.sort_values().head(ssd_top_n)
            else:
                print(f"   • Все {len(usdt_ssd)} пар проходят в дальнейшую обработку")
            
            ssd_pairs = [(s1, s2) for s1, s2 in usdt_ssd.index]
            print(f"📈 Найдено {len(ssd_pairs)} кандидатов по SSD")
            
            # Фильтрация пар
            print("🔬 Фильтрация пар по коинтеграции и другим критериям...")
            
            # ИСПРАВЛЕНО: Усиленные фильтры коинтеграции и контроль хедж-коэффициентов
            filtered_pairs = filter_pairs_by_coint_and_half_life(
                ssd_pairs,
                normalized_training,
                min_half_life=getattr(self.base_config.pair_selection, 'min_half_life_days', 1.0),
                max_half_life=getattr(self.base_config.pair_selection, 'max_half_life_days', 30.0),
                pvalue_threshold=0.05,  # Усиленный фильтр p-value < 0.05
                min_beta=0.2,  # Контроль хедж-коэффициента: abs(beta) >= 0.2
                max_beta=5.0,  # Контроль хедж-коэффициента: abs(beta) <= 5.0
                max_hurst_exponent=getattr(self.base_config.pair_selection, 'max_hurst_exponent', 0.7),
                min_mean_crossings=getattr(self.base_config.pair_selection, 'min_mean_crossings', 10),
                kpss_pvalue_threshold=getattr(self.base_config.pair_selection, 'kpss_pvalue_threshold', 0.05),
            )
            
            print(f"   • После фильтрации по коинтеграции: {len(filtered_pairs)} пар (отсеяно: {len(ssd_pairs) - len(filtered_pairs)})")
            
            if not filtered_pairs:
                raise ValueError("Не найдено ни одной пары после фильтрации.")
            
            print(f"✅ Найдено {len(filtered_pairs)} качественных пар")
            
            # Сортируем пары по качеству (по убыванию стандартного отклонения спреда)
            quality_sorted_pairs = sorted(filtered_pairs, key=lambda x: abs(x[4]), reverse=True)  # x[4] = std
            
            # Топ-M отбор для снижения churn и комиссий
            max_pairs_for_trading = getattr(self.base_config.pair_selection, 'max_pairs_for_trading', 50)
            active_pairs = quality_sorted_pairs[:max_pairs_for_trading]
            
            print(f"   • Топ-M отбор для торговли: {len(active_pairs)} пар (отсеяно: {len(quality_sorted_pairs) - len(active_pairs)})")
            
            # Создаем список пар для сохранения
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
                    'hurst': 0,  # Hurst не возвращается в новой версии
                    'mean_crossings': metrics.get('mean_crossings', 0)
                })
            
            df_pairs = pd.DataFrame(pairs_list)
            
            # Создаем директорию outputs
            Path("outputs").mkdir(exist_ok=True)
            
            # Сохраняем в CSV
            output_path = "outputs/preselected_pairs.csv"
            df_pairs.to_csv(output_path, index=False)
            
            print(f"💾 Отобранные пары сохранены в: {output_path}")
            print(f"📊 Итоговая статистика отобранных пар:")
            print(f"   • Всего пар для торговли: {len(df_pairs)}")
            print(f"   • Средний half-life: {df_pairs['half_life'].mean():.2f} дней")
            print(f"   • Средний p-value: {df_pairs['pvalue'].mean():.4f}")
            print(f"   • Средний Hurst: {df_pairs['hurst'].mean():.3f}")
            print(f"   • Общий процент отсева: {((len(ssd) - len(df_pairs)) / len(ssd) * 100):.1f}%")
            
            print("\n✅ Автоматический отбор пар завершен успешно!")
            print("📊 Использованы НОВЫЕ периоды walk-forward")
            
        except Exception as e:
            print(f"❌ Ошибка при автоматическом отборе пар: {e}")
            import traceback
            traceback.print_exc()
            raise
        
    def _load_data_for_step(self, training_start, training_end, testing_start, testing_end):
        """Загружает данные для конкретного walk-forward шага, точно как в оригинальном бэктесте."""

        print(f"📈 Загрузка данных для walk-forward шага:")
        print(f"   Тренировка: {training_start.date()} -> {training_end.date()}")
        print(f"   Тестирование: {testing_start.date()} -> {testing_end.date()}")

        try:
            # Загружаем данные точно как в оригинальном walk_forward_orchestrator
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

            print(f"✅ Данные загружены: {step_df.shape}")
            return step_df

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
                # ИСПРАВЛЕНО: Используем логарифмическое распределение для больших масштабов
                if cfg.get('step'):
                    params['ssd_top_n'] = trial.suggest_int(
                        "ssd_top_n",
                        cfg['low'],
                        cfg['high'],
                        step=cfg['step']
                    )
                else:
                    # Логарифмическое распределение для лучшего покрытия диапазона
                    log_value = trial.suggest_float(
                        "ssd_top_n_log",
                        np.log10(cfg['low']),
                        np.log10(cfg['high'])
                    )
                    params['ssd_top_n'] = int(10 ** log_value)
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
                    params['zscore_exit'] = trial.suggest_float(
                        "zscore_exit",
                        min_exit,
                        max_exit
                    )
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
                params['commission_pct'] = trial.suggest_float(
                    "commission_pct",
                    costs['commission_pct']['low'],
                    costs['commission_pct']['high']
                )
            if 'slippage_pct' in costs:
                params['slippage_pct'] = trial.suggest_float(
                    "slippage_pct",
                    costs['slippage_pct']['low'],
                    costs['slippage_pct']['high']
                )
        
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
    
    def _run_fast_backtest(self, params):
        """Запускает быстрый бэктест ТОЧНО как в оригинальной системе."""

        # Валидируем параметры перед использованием
        try:
            validated_params = validate_params(params)
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

        # ТОЧНО как в оригинальной системе: определяем периоды
        start_date = pd.to_datetime(cfg.walk_forward.start_date)
        bar_minutes = getattr(cfg.pair_selection, "bar_minutes", None) or 15
        bar_delta = pd.Timedelta(minutes=bar_minutes)

        # Первый шаг walk-forward (точно как в walk_forward_orchestrator)
        current_test_start = start_date
        training_start = current_test_start - pd.Timedelta(days=cfg.walk_forward.training_period_days)
        training_end = current_test_start - bar_delta
        testing_start = current_test_start
        testing_end = testing_start + pd.Timedelta(days=cfg.walk_forward.testing_period_days)

        # Гарантируем, что все временные метки - Timestamp
        testing_start = pd.to_datetime(testing_start)
        testing_end = pd.to_datetime(testing_end)

        # Загружаем данные для этого шага динамически
        step_df = self._load_data_for_step(training_start, training_end, testing_start, testing_end)

        # Инициализируем портфель
        portfolio = Portfolio(
            initial_capital=cfg.portfolio.initial_capital,
            max_active_positions=cfg.portfolio.max_active_positions
        )

        total_trades = 0
        all_pnls = []

        # ДИАГНОСТИКА: Добавляем счетчики
        pairs_checked = 0
        pairs_with_data = 0
        pairs_after_normalization = 0
        pairs_with_enough_data = 0

        print(f"🔍 ДИАГНОСТИКА: Начинаем бэктест для {len(self.preselected_pairs)} пар")
        print(f"   Период тестирования: {testing_start} -> {testing_end}")
        print(f"   Размер step_df: {step_df.shape}")
        print(f"   Колонки в step_df: {len(step_df.columns)}")

        # ТОЧНО как в оригинальной системе: запускаем бэктест для каждой пары
        for _, pair_row in self.preselected_pairs.iterrows():
            s1, s2 = pair_row['s1'], pair_row['s2']
            beta, mean, std = pair_row['beta'], pair_row['mean'], pair_row['std']

            pairs_checked += 1

            # Проверяем наличие данных для пары
            if s1 not in step_df.columns or s2 not in step_df.columns:
                if pairs_checked <= 5:  # Логируем первые 5 пар
                    print(f"   Пара {pairs_checked}: {s1}/{s2} - НЕТ ДАННЫХ в step_df")
                continue

            pairs_with_data += 1

            # ТОЧНО как в оригинальной системе: извлекаем данные тестового периода
            pair_data = step_df.loc[testing_start:testing_end, [s1, s2]].dropna()

            if pair_data.empty:
                if pairs_checked <= 5:
                    print(f"   Пара {pairs_checked}: {s1}/{s2} - ПУСТЫЕ ДАННЫЕ после извлечения")
                continue

            if pairs_checked <= 5:
                print(f"   Пара {pairs_checked}: {s1}/{s2} - {len(pair_data)} точек данных")

            # ИСПРАВЛЕНО: Устранен lookahead bias - используем фиксированную нормализацию
            # Используем предварительно рассчитанные параметры нормализации из pair selection
            # Это исключает динамическое использование будущих данных для нормализации
            
            # Проверяем, есть ли предварительно рассчитанные параметры нормализации
            if 'normalization_base' in pair_row:
                # Используем сохраненную базу нормализации
                normalization_base = pair_row['normalization_base']
                if not np.any(normalization_base == 0):
                    data_values = pair_data.values
                    normalized_values = np.divide(data_values, normalization_base[np.newaxis, :]) * 100
                    pair_data = pd.DataFrame(normalized_values, index=pair_data.index, columns=pair_data.columns)
                else:
                    continue
            else:
                # Fallback: используем первую доступную строку данных (без lookahead)
                # Берем только первую строку тестового периода для нормализации
                if not pair_data.empty:
                    first_row = pair_data.iloc[0].values
                    if not np.any(first_row == 0):
                        data_values = pair_data.values
                        normalized_values = np.divide(data_values, first_row[np.newaxis, :]) * 100
                        pair_data = pd.DataFrame(normalized_values, index=pair_data.index, columns=pair_data.columns)
                    else:
                        continue
                else:
                    continue

            pairs_after_normalization += 1

            if len(pair_data) < cfg.backtest.rolling_window + 10:
                if pairs_checked <= 5:
                    print(f"   Пара {pairs_checked}: {s1}/{s2} - НЕДОСТАТОЧНО ДАННЫХ ({len(pair_data)} < {cfg.backtest.rolling_window + 10})")
                continue

            pairs_with_enough_data += 1

            if pairs_checked <= 5:
                print(f"   Пара {pairs_checked}: {s1}/{s2} - ГОТОВА К БЭКТЕСТУ ({len(pair_data)} точек)")

            try:
                # ИСПРАВЛЕНО: Правильный расчет капитала с num_selected_pairs
                capital_per_pair = portfolio.calculate_position_risk_capital(
                    risk_per_position_pct=cfg.portfolio.risk_per_position_pct,
                    max_position_size_pct=getattr(cfg.portfolio, 'max_position_size_pct', 1.0),
                    num_selected_pairs=len(self.preselected_pairs)
                )
                temp_portfolio = Portfolio(
                    initial_capital=capital_per_pair,
                    max_active_positions=1  # Single pair
                )

                # Извлекаем метрики пары для передачи в бэктестер
                metrics = {}
                if 'half_life' in pair_row:
                    metrics['half_life'] = pair_row['half_life']

                # ТОЧНО как в оригинальной системе: создаем бэктестер со всеми параметрами
                print(f"🔧 DEBUG: Создаем PairBacktester для пары {s1}-{s2}")
                backtester = PairBacktester(
                    pair_data=pair_data,
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
                    cooldown_periods=convert_hours_to_periods(getattr(cfg.backtest, 'cooldown_hours', 0), bar_minutes),
                    wait_for_candle_close=getattr(cfg.backtest, 'wait_for_candle_close', False),
                    max_margin_usage=getattr(cfg.portfolio, 'max_margin_usage', 1.0),
                    half_life=metrics.get('half_life'),
                    time_stop_multiplier=getattr(cfg.backtest, 'time_stop_multiplier', None),
                    # Enhanced risk management parameters
                    use_kelly_sizing=getattr(cfg.backtest, 'use_kelly_sizing', True),
                    max_kelly_fraction=getattr(cfg.backtest, 'max_kelly_fraction', 0.25),
                    volatility_lookback=getattr(cfg.backtest, 'volatility_lookback', 96),
                    adaptive_thresholds=getattr(cfg.backtest, 'adaptive_thresholds', True),
                    var_confidence=getattr(cfg.backtest, 'var_confidence', 0.05),
                    max_var_multiplier=getattr(cfg.backtest, 'max_var_multiplier', 3.0),
                    # Market regime detection parameters
                    market_regime_detection=getattr(cfg.backtest, 'market_regime_detection', True),
                    hurst_window=getattr(cfg.backtest, 'hurst_window', 720),
                    hurst_trending_threshold=getattr(cfg.backtest, 'hurst_trending_threshold', 0.5),
                    variance_ratio_window=getattr(cfg.backtest, 'variance_ratio_window', 480),
                    variance_ratio_trending_min=getattr(cfg.backtest, 'variance_ratio_trending_min', 1.2),
                    variance_ratio_mean_reverting_max=getattr(cfg.backtest, 'variance_ratio_mean_reverting_max', 0.8),
                    # Structural break protection parameters
                    structural_break_protection=getattr(cfg.backtest, 'structural_break_protection', True),
                    cointegration_test_frequency=getattr(cfg.backtest, 'cointegration_test_frequency', 2688),
                    adf_pvalue_threshold=getattr(cfg.backtest, 'adf_pvalue_threshold', 0.05),
                    exclusion_period_days=getattr(cfg.backtest, 'exclusion_period_days', 30),
                    max_half_life_days=getattr(cfg.backtest, 'max_half_life_days', 10),
                    min_correlation_threshold=getattr(cfg.backtest, 'min_correlation_threshold', 0.6),
                    correlation_window=getattr(cfg.backtest, 'correlation_window', 720),
                    # Performance optimization parameters
                    regime_check_frequency=getattr(cfg.backtest, 'regime_check_frequency', 96),
                    use_market_regime_cache=getattr(cfg.backtest, 'use_market_regime_cache', True),
                    adf_check_frequency=getattr(cfg.backtest, 'adf_check_frequency', 5376),
                    lazy_adf_threshold=getattr(cfg.backtest, 'lazy_adf_threshold', 0.1),
                    # EW correlation parameters
                    use_exponential_weighted_correlation=getattr(cfg.backtest, 'use_exponential_weighted_correlation', False),
                    ew_correlation_alpha=getattr(cfg.backtest, 'ew_correlation_alpha', 0.1),
                    hurst_neutral_band=getattr(cfg.backtest, 'hurst_neutral_band', 0.05),
                    vr_neutral_band=getattr(cfg.backtest, 'vr_neutral_band', 0.2),
                    # Volatility-based position sizing parameters (добавлены недостающие параметры)
                    volatility_based_sizing=getattr(cfg.portfolio, 'volatility_based_sizing', False),
                    volatility_lookback_hours=getattr(cfg.portfolio, 'volatility_lookback_hours', 24),
                    min_position_size_pct=getattr(cfg.portfolio, 'min_position_size_pct', 0.005),
                    max_position_size_pct=getattr(cfg.portfolio, 'max_position_size_pct', 0.02),
                    volatility_adjustment_factor=getattr(cfg.portfolio, 'volatility_adjustment_factor', 2.0)
                )

                # Запускаем бэктест
                backtester.run()
                results = backtester.get_results()

                if results is not None and 'pnl' in results:
                    pnl_series = results['pnl']
                    if not pnl_series.empty:
                        all_pnls.append(pnl_series)

                        # ИСПРАВЛЕНО: Правильный подсчет сделок из DataFrame
                        if 'trades' in results:
                            pair_trades = int(results['trades'].sum())
                        else:
                            # Альтернативный способ: считаем изменения позиций
                            position_changes = results['position'].diff().fillna(0)
                            pair_trades = int((position_changes != 0).sum())

                        total_trades += pair_trades

                        if pairs_checked <= 5:
                            print(f"   Пара {pairs_checked}: {s1}/{s2} - {pair_trades} сделок, PnL: {pnl_series.sum():.4f}")
                    else:
                        if pairs_checked <= 5:
                            print(f"   Пара {pairs_checked}: {s1}/{s2} - ПУСТОЙ PnL")
                else:
                    if pairs_checked <= 5:
                        print(f"   Пара {pairs_checked}: {s1}/{s2} - НЕТ РЕЗУЛЬТАТОВ")

            except Exception as e:
                if pairs_checked <= 5:
                    print(f"   Пара {pairs_checked}: {s1}/{s2} - ОШИБКА: {e}")
                continue
        
        # ДИАГНОСТИКА: Итоговая статистика
        print(f"🔍 ДИАГНОСТИКА: Итоги обработки пар:")
        print(f"   Проверено пар: {pairs_checked}")
        print(f"   С данными: {pairs_with_data}")
        print(f"   После нормализации: {pairs_after_normalization}")
        print(f"   С достаточными данными: {pairs_with_enough_data}")
        print(f"   Всего сделок: {total_trades}")
        print(f"   PnL серий: {len(all_pnls)}")

        # ИСПРАВЛЕНО: Стандартизированная обработка ошибок - возвращаем None для невалидных результатов
        if not all_pnls:
            print(f"🔍 ДИАГНОСТИКА: НЕТ PnL ДАННЫХ - возвращаем невалидный результат")
            return {"sharpe_ratio_abs": None, "total_trades": total_trades, "error_type": "no_pnl_data"}

        # Суммируем PnL всех пар (безопасно)
        try:
            if len(all_pnls) == 1:
                combined_pnl = all_pnls[0].fillna(0)
            else:
                combined_pnl = pd.concat(all_pnls, axis=1).sum(axis=1).fillna(0)
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
        
        # Расчет win rate для унификации с objective.py
        win_rate = 0.0
        if total_trades > 0:
            winning_trades = sum(1 for pnl in combined_pnl if pnl > 0)
            win_rate = winning_trades / len(combined_pnl) if len(combined_pnl) > 0 else 0.0
        
        print(f"📊 Диагностика производительности:")
        print(f"   • Всего пар в торговле: {len(all_pnls)} из {len(self.preselected_pairs)} доступных")
        print(f"   • Всего сделок: {total_trades}")
        print(f"   • Средний размер сделки: ${avg_trade_size:.2f}")
        print(f"   • Средний hold-time: {avg_hold_time:.1f} баров")
        print(f"   • Максимальная просадка: {max_dd:.2%}")
        print(f"   • Общий P&L: ${combined_pnl.sum():.2f}")
        print(f"   • Активность пар: {(len(all_pnls) / len(self.preselected_pairs) * 100):.1f}%")
        
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

            # Если данные не переданы, загружаем их (для обратной совместимости)
            if step_df is None:
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

                # Загружаем данные для этого шага
                step_df = self._load_data_for_step(training_start, training_end, testing_start, testing_end)

            # Проверяем наличие данных для пары
            if s1 not in step_df.columns or s2 not in step_df.columns:
                return None

            # Извлекаем данные пары
            pair_data = step_df[[s1, s2]].dropna()
            if len(pair_data) < cfg.backtest.rolling_window + 10:
                return None

            # Нормализация данных
            try:
                normalized_data, _ = preprocess_and_normalize_data(
                    pair_data,
                    method=cfg.data_processing.normalization_method,
                    min_history_ratio=cfg.data_processing.min_history_ratio,
                    handle_constant=cfg.data_processing.handle_constant,
                    fill_method=cfg.data_processing.fill_method
                )
                if normalized_data.empty:
                    return None
            except Exception:
                return None

            # Создаем временный портфель для этой пары
            temp_portfolio = Portfolio(
                initial_capital=cfg.portfolio.initial_capital,
                max_active_positions=1
            )

            capital_per_pair = cfg.portfolio.initial_capital * cfg.portfolio.risk_per_position_pct

            # Создаем бэктестер
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
                cooldown_hours=getattr(cfg.backtest, 'cooldown_hours', 4)
            )

            # Запускаем бэктест
            backtester.run()
            results = backtester.get_results()

            if results is None or results.empty or 'pnl' not in results:
                return None

            # Возвращаем PnL серию
            return results['pnl'] * capital_per_pair

        except Exception as e:
            print(f"Ошибка при обработке пары {pair_row.get('s1', 'unknown')}: {e}")
            return None

    def _run_fast_backtest_with_reports(self, params, trial):
        """Запускает быстрый бэктест с промежуточными отчетами для pruning."""
        import optuna

        # Используем ту же логику что и в _run_fast_backtest, но с отчетами
        cfg = self.base_config.model_copy(deep=True)

        # Применяем параметры (сокращенная версия)
        for key, value in params.items():
            if key in ["ssd_top_n", "kpss_pvalue_threshold", "coint_pvalue_threshold",
                      "min_half_life_days", "max_half_life_days", "min_mean_crossings"]:
                setattr(cfg.pair_selection, key, value)
            elif key in ["zscore_threshold", "zscore_exit", "rolling_window", "stop_loss_multiplier",
                        "time_stop_multiplier", "cooldown_hours", "commission_pct", "slippage_pct"]:
                setattr(cfg.backtest, key, value)
            elif key in ["max_active_positions", "risk_per_position_pct", "max_position_size_pct"]:
                setattr(cfg.portfolio, key, value)
            elif key in ["normalization_method", "min_history_ratio"]:
                setattr(cfg.data_processing, key, value)

        # ИСПРАВЛЕНИЕ: Загружаем данные один раз для всех пар
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

        # Загружаем данные один раз для всех пар
        step_df = self._load_data_for_step(training_start, training_end, testing_start, testing_end)

        # Запускаем бэктест по парам с промежуточными отчетами
        all_pnls = []
        total_trades = 0
        step_idx = 0

        for i, (_, pair_row) in enumerate(self.preselected_pairs.iterrows()):
            try:
                # Передаем данные в метод для избежания повторной загрузки
                pair_result = self._backtest_single_pair(pair_row, cfg, step_df)

                if pair_result is not None and len(pair_result) > 0:
                    all_pnls.append(pair_result)
                    total_trades += len(pair_result)

                # Промежуточный отчет каждые INTERMEDIATE_REPORT_INTERVAL пар
                if (i + 1) % INTERMEDIATE_REPORT_INTERVAL == 0 and all_pnls:
                    try:
                        combined_pnl = pd.concat(all_pnls, axis=1).sum(axis=1).fillna(0) if len(all_pnls) > 1 else all_pnls[0].fillna(0)
                        equity_curve = cfg.portfolio.initial_capital + combined_pnl.cumsum()
                        daily_returns = equity_curve.pct_change().dropna()

                        if len(daily_returns) > 0 and daily_returns.std() > 0:
                            intermediate_sharpe = daily_returns.mean() / daily_returns.std() * np.sqrt(cfg.backtest.annualizing_factor)

                            # Отчет в Optuna
                            trial.report(float(intermediate_sharpe), step=step_idx)

                            # Проверяем pruning
                            if trial.should_prune():
                                print(f"Trial pruned at step {step_idx} (pair {i+1}/{len(self.preselected_pairs)})")
                                raise optuna.TrialPruned(f"Pruned at step {step_idx}")

                        step_idx += 1
                    except Exception as e:
                        print(f"Ошибка промежуточного отчета: {e}")

            except optuna.TrialPruned:
                raise  # Пробрасываем pruning
            except Exception as e:
                print(f"Ошибка при обработке пары {pair_row.get('s1', 'unknown')}: {e}")
                continue

        # Финальный расчет (упрощенная версия)
        if not all_pnls:
            return {"sharpe_ratio_abs": PENALTY, "total_trades": total_trades, "max_drawdown": 0, "win_rate": 0}

        try:
            combined_pnl = pd.concat(all_pnls, axis=1).sum(axis=1).fillna(0) if len(all_pnls) > 1 else all_pnls[0].fillna(0)
            equity_curve = cfg.portfolio.initial_capital + combined_pnl.cumsum()
            daily_returns = equity_curve.pct_change().dropna()

            if len(daily_returns) == 0 or daily_returns.std() == 0:
                return {"sharpe_ratio_abs": PENALTY, "total_trades": total_trades, "max_drawdown": 0, "win_rate": 0}

            sharpe = daily_returns.mean() / daily_returns.std() * np.sqrt(cfg.backtest.annualizing_factor)
            running_max = equity_curve.expanding().max()
            drawdown = (equity_curve - running_max) / running_max
            max_dd = abs(drawdown.min()) if len(drawdown) > 0 else 0
            win_rate = (daily_returns > 0).mean() if len(daily_returns) > 0 else 0

            return {"sharpe_ratio_abs": sharpe, "total_trades": total_trades, "max_drawdown": max_dd, "win_rate": win_rate}

        except Exception as e:
            print(f"Ошибка финального расчета: {e}")
            return {"sharpe_ratio_abs": PENALTY, "total_trades": total_trades, "max_drawdown": 0, "win_rate": 0}

    def __call__(self, trial_or_params):
        """Унифицированная функция для совместимости с objective.py.
        
        Args:
            trial_or_params: optuna.Trial объект или словарь параметров
            
        Returns:
            float: Значение целевой функции
        """
        # Определяем тип входных данных и извлекаем параметры
        if hasattr(trial_or_params, 'suggest_float'):  # Это optuna.Trial
            trial = trial_or_params
            params = self._suggest_parameters(trial)
            trial_number = trial.number
        else:  # Это словарь параметров
            params = trial_or_params
            trial_number = params.get("trial_number", -1)
        
        try:
            # ИСПРАВЛЕНО: Правильная обработка ошибок валидации через TrialPruned
            try:
                validated_params = validate_params(params)
            except ValueError as e:
                print(f"Trial #{trial_number}: Невалидные параметры: {e}")
                if hasattr(trial_or_params, 'suggest_float') and hasattr(trial_or_params, "set_user_attr"):
                    trial_or_params.set_user_attr("error_type", "validation_error")
                    trial_or_params.set_user_attr("validation_message", str(e))
                    trial_or_params.set_user_attr("invalid_params", params)
                    raise optuna.TrialPruned(f"Parameter validation failed: {e}")
                return PENALTY  # Fallback для обратной совместимости
            
            # Запускаем быстрый бэктест с промежуточными отчетами (если это trial)
            if hasattr(trial_or_params, 'suggest_float'):
                metrics = self._run_fast_backtest_with_reports(validated_params, trial)
            else:
                metrics = self._run_fast_backtest(validated_params)
            
            # Используем единую функцию extract_sharpe
            sharpe = extract_sharpe(metrics)
            
            # ИСПРАВЛЕНО: Правильная обработка невалидных Sharpe ratio через TrialPruned
            if sharpe is None or not isinstance(sharpe, (int, float)) or np.isnan(sharpe) or np.isinf(sharpe):
                print(f"Trial #{trial_number}: Невалидный Sharpe ratio: {sharpe}")
                if hasattr(trial_or_params, 'suggest_float') and hasattr(trial_or_params, "set_user_attr"):
                    trial_or_params.set_user_attr("error_type", "invalid_sharpe")
                    trial_or_params.set_user_attr("sharpe_value", str(sharpe))
                    trial_or_params.set_user_attr("metrics_available", list(metrics.keys()) if metrics else [])
                    raise optuna.TrialPruned(f"Invalid Sharpe ratio: {sharpe}")
                return PENALTY  # Fallback для обратной совместимости
            
            print(f"Trial #{trial_number}: {metrics.get('total_trades', 0)} сделок, Sharpe: {sharpe:.4f}")
            
            # ИСПРАВЛЕНО: Унифицированы штрафы с objective.py
            max_dd = metrics.get("max_drawdown", 0)
            win_rate = metrics.get("win_rate", 0.0)
            
            # Базовый штраф за большую просадку (> 25%)
            dd_penalty = 0
            if max_dd > 0.25:
                dd_penalty = (max_dd - 0.25) * 3
                
            # Дополнительный штраф за очень большую просадку (> 50%)
            if max_dd > 0.50:
                dd_penalty += (max_dd - 0.50) * 5
                
            # Бонус за хороший винрейт (> 55%)
            win_rate_bonus = max(0, (win_rate - 0.55) * 0.5) if win_rate > 0.55 else 0
            
            # Штраф за низкий винрейт (< 40%)
            win_rate_penalty = max(0, (0.40 - win_rate) * 1.0) if win_rate < 0.40 else 0
            
            # ИСПРАВЛЕНО: Правильная обработка недостаточного количества сделок через TrialPruned
            total_trades = metrics.get('total_trades', 0)
            if total_trades < MIN_TRADES_THRESHOLD:  # Используем константу
                print(f"Trial #{trial_number}: Недостаточно сделок ({total_trades} < {MIN_TRADES_THRESHOLD})")
                if hasattr(trial_or_params, 'suggest_float') and hasattr(trial_or_params, "set_user_attr"):
                    trial_or_params.set_user_attr("error_type", "insufficient_trades")
                    trial_or_params.set_user_attr("trades_count", total_trades)
                    trial_or_params.set_user_attr("min_required", MIN_TRADES_THRESHOLD)
                    raise optuna.TrialPruned(f"Insufficient trades: {total_trades} < {MIN_TRADES_THRESHOLD}")
                return PENALTY  # Fallback для обратной совместимости

            # Используем константы для расчета штрафов и бонусов
            dd_penalty = 0
            if max_dd > MAX_DRAWDOWN_SOFT_THRESHOLD:
                dd_penalty = (max_dd - MAX_DRAWDOWN_SOFT_THRESHOLD) * DD_PENALTY_SOFT_MULTIPLIER

            if max_dd > MAX_DRAWDOWN_HARD_THRESHOLD:
                dd_penalty += (max_dd - MAX_DRAWDOWN_HARD_THRESHOLD) * DD_PENALTY_HARD_MULTIPLIER

            win_rate_bonus = max(0, (win_rate - WIN_RATE_BONUS_THRESHOLD) * WIN_RATE_BONUS_MULTIPLIER) if win_rate > WIN_RATE_BONUS_THRESHOLD else 0
            win_rate_penalty = max(0, (WIN_RATE_PENALTY_THRESHOLD - win_rate) * WIN_RATE_PENALTY_MULTIPLIER) if win_rate < WIN_RATE_PENALTY_THRESHOLD else 0

            final_score = sharpe - dd_penalty + win_rate_bonus - win_rate_penalty

            # Сохраняем детальные метрики в trial (если это Optuna trial)
            if hasattr(trial_or_params, 'suggest_float') and hasattr(trial_or_params, "set_user_attr"):
                trial_or_params.set_user_attr("metrics", {
                    "sharpe": float(sharpe),
                    "max_drawdown": float(max_dd),
                    "win_rate": float(win_rate),
                    "total_trades": int(total_trades),
                    "dd_penalty": float(dd_penalty),
                    "win_rate_bonus": float(win_rate_bonus),
                    "win_rate_penalty": float(win_rate_penalty),
                    "final_score": float(final_score)
                })

            return final_score
            
        except optuna.TrialPruned:
            # Пробрасываем TrialPruned без изменений
            raise
        except Exception as e:
            print(f"Неожиданная ошибка в fast objective (trial #{trial_number}): {e}")
            import traceback
            traceback.print_exc()
            # Для неожиданных ошибок используем PENALTY + логирование
            if hasattr(trial_or_params, 'suggest_float') and hasattr(trial_or_params, "set_user_attr"):
                trial_or_params.set_user_attr("error_type", "execution_error")
                trial_or_params.set_user_attr("exception_type", type(e).__name__)
                trial_or_params.set_user_attr("exception_message", str(e))
            return PENALTY  # Штраф для неожиданных ошибок
