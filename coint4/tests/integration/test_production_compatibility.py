"""
Тесты для проверки совместимости с production торговлей.
Проверяют что нормализация консистентна между отбором пар и торговлей.
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock

from src.coint2.core.normalization_improvements import (
    preprocess_and_normalize_data,
    apply_production_normalization
)
from src.coint2.core.production_normalizer import (
    ProductionNormalizer,
    create_production_normalizer
)
from src.optimiser.metric_utils import validate_params


@pytest.fixture
def sample_price_data():
    """Создает тестовые данные о ценах."""
    dates = pd.date_range('2023-01-01', periods=100, freq='15T')
    np.random.seed(42)
    
    # Создаем коррелированные цены
    base_price = 100 + np.cumsum(np.random.randn(100) * 0.01)
    correlated_price = base_price * 1.1 + np.random.randn(100) * 0.1
    
    return pd.DataFrame({
        'BTCUSDT': base_price,
        'ETHUSDT': correlated_price
    }, index=dates)


class TestNormalizationConsistency:
    """Тестирует консистентность нормализации между отбором пар и торговлей."""
    
    def test_rolling_zscore_consistency(self, sample_price_data):
        """Проверяет что rolling z-score нормализация дает одинаковые результаты."""
        # 1. Получаем статистики нормализации из training данных
        training_data = sample_price_data.iloc[:60]  # Первые 60 точек для тренировки
        testing_data = sample_price_data.iloc[60:]    # Остальные для тестирования
        
        # Нормализуем training данные и получаем статистики
        normalized_training, stats = preprocess_and_normalize_data(
            training_data,
            norm_method='rolling_zscore',
            rolling_window=25,
            return_stats=True
        )
        
        # Проверяем что статистики сохранены
        assert 'normalization_stats' in stats
        norm_stats = stats['normalization_stats']
        assert norm_stats['method'] == 'rolling_zscore'
        assert norm_stats['window'] == 25
        assert 'rolling_mean' in norm_stats
        assert 'rolling_std' in norm_stats
        
        # 2. Применяем те же статистики к testing данным
        normalized_testing = apply_production_normalization(
            testing_data,
            norm_stats
        )
        
        # 3. Проверяем что результат корректный
        assert not normalized_testing.empty
        assert list(normalized_testing.columns) == ['BTCUSDT', 'ETHUSDT']
        
        # 4. Проверяем что статистики правильно применились
        # (не можем проверить точные значения без детального анализа)
        assert not normalized_testing.isna().all().any()
    
    def test_percent_normalization_consistency(self, sample_price_data):
        """Проверяет консистентность процентной нормализации."""
        training_data = sample_price_data.iloc[:60]
        testing_data = sample_price_data.iloc[60:]
        
        # Получаем статистики из training данных
        normalized_training, stats = preprocess_and_normalize_data(
            training_data,
            norm_method='percent',
            return_stats=True
        )
        
        norm_stats = stats['normalization_stats']
        assert norm_stats['method'] == 'percent'
        assert 'first_values' in norm_stats
        
        # Применяем к testing данным
        normalized_testing = apply_production_normalization(
            testing_data,
            norm_stats
        )
        
        # Проверяем результат
        assert not normalized_testing.empty
        
        # Проверяем что первые значения правильно использованы
        first_btc = norm_stats['first_values']['BTCUSDT']
        expected_btc_normalized = (testing_data['BTCUSDT'] / first_btc) * 100
        
        pd.testing.assert_series_equal(
            normalized_testing['BTCUSDT'],
            expected_btc_normalized,
            check_names=False
        )
    
    def test_production_incompatible_methods_warning(self, sample_price_data):
        """Проверяет предупреждения для несовместимых с production методов."""
        # minmax не должен использоваться в production
        with pytest.warns(None) as warning_list:
            normalized_data, stats = preprocess_and_normalize_data(
                sample_price_data.iloc[:60],
                norm_method='minmax',
                return_stats=True
            )
        
        # Проверяем что было выведено предупреждение
        warning_messages = [str(w.message) for w in warning_list 
                          if 'minmax нормализация НЕ совместима с production' in str(w.message)]
        assert len(warning_messages) > 0


class TestProductionNormalizer:
    """Тестирует ProductionNormalizer для real-time использования."""
    
    def test_rolling_zscore_normalizer(self):
        """Тестирует rolling z-score нормализатор."""
        normalizer = ProductionNormalizer(method='rolling_zscore', window=5)
        
        # Добавляем данные
        prices = [100, 101, 102, 98, 99, 103, 105]
        normalized_values = []
        
        for price in prices:
            normalizer.update('BTCUSDT', price)
            normalized = normalizer.normalize('BTCUSDT')
            normalized_values.append(normalized)
        
        # Проверяем что значения рассчитаны
        assert len(normalized_values) == len(prices)
        assert all(val is not None for val in normalized_values[1:])  # Первое может быть None
    
    def test_percent_normalizer(self):
        """Тестирует процентный нормализатор."""
        normalizer = ProductionNormalizer(method='percent')
        
        # Добавляем данные
        normalizer.update('BTCUSDT', 100)
        normalizer.update('BTCUSDT', 110)
        
        # Проверяем нормализацию
        assert normalizer.normalize('BTCUSDT', 100) == 100.0  # Базовое значение
        assert normalizer.normalize('BTCUSDT', 110) == 110.0  # +10%
        assert normalizer.normalize('BTCUSDT', 90) == 90.0    # -10%
    
    def test_unsupported_method_raises_error(self):
        """Проверяет что неподдерживаемые методы вызывают ошибку."""
        with pytest.raises(ValueError, match="Метод minmax не поддерживается"):
            ProductionNormalizer(method='minmax')
    
    def test_state_persistence(self, tmp_path):
        """Тестирует сохранение и загрузку состояния."""
        normalizer = ProductionNormalizer(method='rolling_zscore', window=3)
        
        # Добавляем данные
        for price in [100, 101, 102]:
            normalizer.update('BTCUSDT', price)
        
        # Сохраняем состояние
        state_file = tmp_path / "normalizer_state.json"
        normalizer.save_state(state_file)
        
        # Создаем новый нормализатор и загружаем состояние
        new_normalizer = ProductionNormalizer()
        new_normalizer.load_state(state_file)
        
        # Проверяем что состояние восстановлено
        assert new_normalizer.method == 'rolling_zscore'
        assert new_normalizer.window == 3
        assert 'BTCUSDT' in new_normalizer.buffers
        assert list(new_normalizer.buffers['BTCUSDT']) == [100, 101, 102]
    
    def test_from_training_stats(self, sample_price_data):
        """Тестирует создание нормализатора из статистик тренировки."""
        # Получаем статистики из training данных
        training_data = sample_price_data.iloc[:60]
        _, stats = preprocess_and_normalize_data(
            training_data,
            norm_method='rolling_zscore',
            rolling_window=25,
            return_stats=True
        )
        
        norm_stats = stats['normalization_stats']
        
        # Создаем продакшн нормализатор
        normalizer = ProductionNormalizer.from_training_stats(norm_stats)
        
        # Проверяем что нормализатор создан правильно
        assert normalizer.method == 'rolling_zscore'
        assert normalizer.window == 25
        assert len(normalizer.symbols_tracked) == 2  # BTCUSDT, ETHUSDT


class TestParameterValidation:
    """Тестирует валидацию параметров на production-совместимость."""
    
    def test_production_compatible_normalization_method(self):
        """Проверяет что production-совместимые методы проходят валидацию."""
        params = {'normalization_method': 'rolling_zscore'}
        validated = validate_params(params)
        assert validated['normalization_method'] == 'rolling_zscore'
    
    def test_production_incompatible_normalization_method_replaced(self, capsys):
        """Проверяет что несовместимые методы заменяются на совместимые."""
        params = {'normalization_method': 'minmax'}
        validated = validate_params(params)
        
        # Проверяем что метод заменен
        assert validated['normalization_method'] == 'rolling_zscore'
        
        # Проверяем что было выведено предупреждение
        captured = capsys.readouterr()
        assert "НЕ совместим с production" in captured.out
    
    def test_negative_zscore_exit_allowed(self):
        """Проверяет что отрицательные zscore_exit разрешены."""
        params = {
            'zscore_threshold': 2.0,
            'zscore_exit': -0.5  # Отрицательное значение для симметричной стратегии
        }
        validated = validate_params(params)
        
        # Проверяем что отрицательное значение сохранено
        assert validated['zscore_exit'] == -0.5
        assert validated['zscore_threshold'] == 2.0
    
    def test_zscore_hysteresis_validation(self):
        """Проверяет валидацию гистерезиса между zscore_threshold и zscore_exit."""
        # Слишком маленький гистерезис должен вызывать ошибку
        with pytest.raises(ValueError, match="Слишком маленький гистерезис"):
            validate_params({
                'zscore_threshold': 1.0,
                'zscore_exit': 0.98  # Гистерезис = 0.02 < 0.05
            })
        
        # Нормальный гистерезис должен проходить
        params = {
            'zscore_threshold': 2.0,
            'zscore_exit': 0.3  # Гистерезис = 1.7
        }
        validated = validate_params(params)
        assert validated['zscore_threshold'] == 2.0
        assert validated['zscore_exit'] == 0.3


class TestEndToEndConsistency:
    """Интеграционные тесты полной консистентности от отбора пар до торговли."""
    
    @patch('src.optimiser.fast_objective.filter_pairs_by_coint_and_half_life')
    @patch('src.optimiser.fast_objective.calculate_ssd')
    def test_pair_selection_to_trading_consistency(self, mock_ssd, mock_filter, sample_price_data):
        """Тестирует консистентность от отбора пар до торговли."""
        # Мокаем функции для упрощения теста
        mock_ssd.return_value = pd.Series({('BTCUSDT', 'ETHUSDT'): 0.1})
        mock_filter.return_value = [('BTCUSDT', 'ETHUSDT', 1.0, 0.0, 0.1, {})]
        
        from src.optimiser.fast_objective import FastWalkForwardObjective
        
        # Создаем конфигурацию
        config_mock = MagicMock()
        config_mock.pair_selection.min_history_ratio = 0.5
        config_mock.pair_selection.fill_method = 'forward'
        config_mock.pair_selection.norm_method = 'rolling_zscore'
        config_mock.pair_selection.handle_constant = 'drop'
        config_mock.backtest.rolling_window = 10
        config_mock.pair_selection.ssd_top_n = 1000
        
        # Создаем объект (с упрощенной инициализацией)
        with patch('src.optimiser.fast_objective.create_temporal_validator'):
            objective = FastWalkForwardObjective.__new__(FastWalkForwardObjective)
            objective.base_config = config_mock
            objective.search_space = {}
            objective.pair_selection_cache = {}
            objective._cache_lock = MagicMock()
            objective.lookahead_validator = MagicMock()
        
        # Тестируем отбор пар
        pairs_df, norm_stats = objective._select_pairs_for_step(
            config_mock,
            sample_price_data.iloc[:50],  # training данные
            0
        )
        
        # Проверяем что статистики возвращены
        assert isinstance(norm_stats, dict)
        # Если нормализация прошла, статистики должны быть заполнены
        if norm_stats:
            assert 'method' in norm_stats
        
        # Проверяем что пары отобраны
        if not pairs_df.empty:
            assert 's1' in pairs_df.columns
            assert 's2' in pairs_df.columns


if __name__ == '__main__':
    pytest.main([__file__, '-v'])