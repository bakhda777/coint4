"""
Адаптер для конфигурации - решает проблемы несоответствия атрибутов.
"""

class ConfigAdapter:
    """
    Адаптер для унификации доступа к конфигурации.
    Решает проблемы типа:
    - config.backtesting vs config.backtest
    - config.signals.zscore_threshold vs config.backtest.zscore_threshold
    """
    
    def __init__(self, config):
        """
        Args:
            config: Оригинальная конфигурация (AppConfig)
        """
        self._config = config
        
    def __getattr__(self, name):
        """Проксирование доступа к атрибутам с адаптацией."""
        
        # Маппинг старых имен на новые
        mappings = {
            'backtesting': 'backtest',
            'risk_management': 'portfolio',
            'signals': 'backtest',  # Параметры сигналов теперь в backtest
        }
        
        # Если запрашивают старое имя, перенаправляем на новое
        if name in mappings:
            return getattr(self._config, mappings[name])
        
        # Иначе возвращаем как есть
        return getattr(self._config, name)
    
    @property
    def signals(self):
        """Специальный адаптер для signals - берем из backtest."""
        class SignalsAdapter:
            def __init__(self, backtest_config):
                self._backtest = backtest_config
                
            @property
            def zscore_threshold(self):
                return self._backtest.zscore_threshold
            
            @property
            def zscore_exit(self):
                return self._backtest.zscore_exit if hasattr(self._backtest, 'zscore_exit') else 0.0
            
            @property
            def rolling_window(self):
                return self._backtest.rolling_window
            
            @property
            def stop_loss_multiplier(self):
                return self._backtest.stop_loss_multiplier
            
            @property
            def time_stop_multiplier(self):
                return self._backtest.time_stop_multiplier if hasattr(self._backtest, 'time_stop_multiplier') else None
            
            def __getattr__(self, name):
                # Пробуем получить из backtest
                return getattr(self._backtest, name)
        
        return SignalsAdapter(self._config.backtest)
    
    @property
    def risk_management(self):
        """Адаптер для risk_management - берем из portfolio."""
        class RiskAdapter:
            def __init__(self, portfolio_config):
                self._portfolio = portfolio_config
            
            @property
            def max_active_positions(self):
                return self._portfolio.max_active_positions
            
            @property
            def risk_per_position_pct(self):
                return self._portfolio.risk_per_position_pct
            
            @property
            def initial_capital(self):
                return self._portfolio.initial_capital
            
            def __getattr__(self, name):
                return getattr(self._portfolio, name)
        
        return RiskAdapter(self._config.portfolio)
    
    @property 
    def backtesting(self):
        """Алиас для backtest."""
        return self._config.backtest
    
    @property
    def walk_forward(self):
        """Адаптер для walk_forward параметров."""
        class WalkForwardAdapter:
            def __init__(self, config):
                self._config = config
                
            @property
            def train_days(self):
                # В старой версии было training_period_days
                if hasattr(self._config.walk_forward, 'training_period_days'):
                    return self._config.walk_forward.training_period_days
                elif hasattr(self._config.walk_forward, 'train_days'):
                    return self._config.walk_forward.train_days
                else:
                    return 60  # Дефолт
            
            @property
            def test_days(self):
                # В старой версии было testing_period_days  
                if hasattr(self._config.walk_forward, 'testing_period_days'):
                    return self._config.walk_forward.testing_period_days
                elif hasattr(self._config.walk_forward, 'test_days'):
                    return self._config.walk_forward.test_days
                else:
                    return 30  # Дефолт
                    
            @property
            def start_date(self):
                return self._config.walk_forward.start_date
            
            @property
            def end_date(self):
                return self._config.walk_forward.end_date
            
            def __getattr__(self, name):
                return getattr(self._config.walk_forward, name)
        
        return WalkForwardAdapter(self._config)
    
    def to_dict(self):
        """Конвертация в словарь для совместимости."""
        return self._config.dict() if hasattr(self._config, 'dict') else self._config.__dict__


def adapt_config(config):
    """
    Функция-хелпер для быстрой адаптации конфигурации.
    
    Args:
        config: Оригинальная конфигурация
        
    Returns:
        ConfigAdapter: Адаптированная конфигурация
    """
    return ConfigAdapter(config)


# Для обратной совместимости
def fix_config_compatibility(config):
    """Легаси функция для совместимости."""
    return adapt_config(config)