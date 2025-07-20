"""Загрузчик конфигурации для бэктест-системы.

Этот модуль обеспечивает:
- Загрузку параметров из YAML файла
- Валидацию конфигурации
- Значения по умолчанию
- Логирование изменений конфигурации
"""

import os
import yaml
import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class WalkForwardConfig:
    """Конфигурация walk-forward анализа."""
    train_window: int = 252
    test_window: int = 63
    rolling_window: int = 50
    step_size: int = 21


@dataclass
class SignalsConfig:
    """Конфигурация торговых сигналов."""
    z_entry: float = 2.0
    z_exit: float = 0.5
    z_stop_loss: float = 3.0
    min_half_life: float = 5.0
    max_half_life: float = 50.0
    min_correlation: float = 0.7


@dataclass
class RiskManagementConfig:
    """Конфигурация управления рисками."""
    capital_at_risk: float = 100000
    max_active_positions: int = 3
    leverage_limit: float = 2.0
    f_max: float = 0.25
    stop_loss_multiplier: float = 3.0
    time_stop_multiplier: float = 2.0
    min_holding_periods: int = 3
    cooldown_periods: int = 5
    take_profit_multiplier: float = 1.5


@dataclass
class CostsConfig:
    """Конфигурация торговых издержек."""
    commission_pct: float = 0.001
    slippage_pct: float = 0.0005
    bid_ask_spread_pct_s1: float = 0.0002
    bid_ask_spread_pct_s2: float = 0.0002
    funding_rate_pct: float = 0.0001


@dataclass
class ModelConfig:
    """Конфигурация модели."""
    lookback_window: int = 252
    half_life_window: int = 50
    volatility_window: int = 20
    min_observations: int = 100
    max_pvalue: float = 0.05
    min_adf_stat: float = -2.5


@dataclass
class LoggingConfig:
    """Конфигурация логирования."""
    level: str = "INFO"
    log_trades: bool = True
    log_signals: bool = False
    log_performance: bool = True
    trade_log_file: str = "logs/trades.csv"
    performance_log_file: str = "logs/performance.csv"
    signal_log_file: str = "logs/signals.csv"


@dataclass
class ReportingSections:
    """Секции отчета."""
    config: bool = True
    metrics_test: bool = True
    turnover: bool = True
    cost_breakdown: bool = True
    drawdown_table: bool = True
    sensitivity_fees: bool = True


@dataclass
class SensitivityConfig:
    """Конфигурация анализа чувствительности."""
    fee_multipliers: list = field(default_factory=lambda: [0.5, 1.0, 1.5, 2.0, 3.0, 5.0])


@dataclass
class ReportingConfig:
    """Конфигурация отчетности."""
    generate_plots: bool = True
    save_results: bool = True
    sections: ReportingSections = field(default_factory=ReportingSections)
    sensitivity: SensitivityConfig = field(default_factory=SensitivityConfig)


@dataclass
class ValidationConfig:
    """Конфигурация валидации."""
    enable_robustness_tests: bool = True
    enable_synthetic_tests: bool = True
    permutation_tests: int = 10
    bootstrap_samples: int = 1000
    max_sharpe_warning: float = 3.0
    min_trades_warning: int = 5
    max_turnover_warning: float = 10.0


@dataclass
class BacktestConfig:
    """Главная конфигурация бэктеста."""
    walk_forward: WalkForwardConfig = field(default_factory=WalkForwardConfig)
    signals: SignalsConfig = field(default_factory=SignalsConfig)
    risk_management: RiskManagementConfig = field(default_factory=RiskManagementConfig)
    costs: CostsConfig = field(default_factory=CostsConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    reporting: ReportingConfig = field(default_factory=ReportingConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    
    # Метаданные
    version: str = "1.0.0"
    random_seed: int = 42


class ConfigLoader:
    """Загрузчик конфигурации из YAML файла."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Инициализация загрузчика.
        
        Args:
            config_path: Путь к файлу конфигурации. Если None, ищет config.yaml в корне проекта.
        """
        self.logger = logging.getLogger(__name__)
        
        if config_path is None:
            # Ищем config.yaml в корне проекта
            current_dir = Path(__file__).parent
            project_root = current_dir.parent.parent.parent  # Поднимаемся к корню проекта
            config_path = project_root / "config.yaml"
        
        self.config_path = Path(config_path)
        self._config_data = None
    
    def load_config(self) -> BacktestConfig:
        """Загружает конфигурацию из файла.
        
        Returns:
            BacktestConfig: Объект конфигурации
            
        Raises:
            FileNotFoundError: Если файл конфигурации не найден
            yaml.YAMLError: Если ошибка парсинга YAML
        """
        if not self.config_path.exists():
            self.logger.warning(f"Файл конфигурации не найден: {self.config_path}")
            self.logger.info("Используются значения по умолчанию")
            return BacktestConfig()
        
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                self._config_data = yaml.safe_load(f)
            
            self.logger.info(f"Конфигурация загружена из: {self.config_path}")
            return self._parse_config(self._config_data)
            
        except yaml.YAMLError as e:
            self.logger.error(f"Ошибка парсинга YAML: {e}")
            raise
        except Exception as e:
            self.logger.error(f"Ошибка загрузки конфигурации: {e}")
            raise
    
    def _parse_config(self, data: Dict[str, Any]) -> BacktestConfig:
        """Парсит данные конфигурации в объекты.
        
        Args:
            data: Словарь с данными конфигурации
            
        Returns:
            BacktestConfig: Объект конфигурации
        """
        config = BacktestConfig()
        
        # Walk-forward конфигурация
        if 'walk_forward' in data:
            wf_data = data['walk_forward']
            config.walk_forward = WalkForwardConfig(
                train_window=wf_data.get('train_window', 252),
                test_window=wf_data.get('test_window', 63),
                rolling_window=wf_data.get('rolling_window', 50),
                step_size=wf_data.get('step_size', 21)
            )
        
        # Сигналы
        if 'signals' in data:
            sig_data = data['signals']
            config.signals = SignalsConfig(
                z_entry=sig_data.get('z_entry', 2.0),
                z_exit=sig_data.get('z_exit', 0.5),
                z_stop_loss=sig_data.get('z_stop_loss', 3.0),
                min_half_life=sig_data.get('min_half_life', 5.0),
                max_half_life=sig_data.get('max_half_life', 50.0),
                min_correlation=sig_data.get('min_correlation', 0.7)
            )
        
        # Управление рисками
        if 'risk_management' in data:
            risk_data = data['risk_management']
            config.risk_management = RiskManagementConfig(
                capital_at_risk=risk_data.get('capital_at_risk', 100000),
                max_active_positions=risk_data.get('max_active_positions', 3),
                leverage_limit=risk_data.get('leverage_limit', 2.0),
                f_max=risk_data.get('f_max', 0.25),
                stop_loss_multiplier=risk_data.get('stop_loss_multiplier', 3.0),
                time_stop_multiplier=risk_data.get('time_stop_multiplier', 2.0),
                min_holding_periods=risk_data.get('min_holding_periods', 3),
                cooldown_periods=risk_data.get('cooldown_periods', 5),
                take_profit_multiplier=risk_data.get('take_profit_multiplier', 1.5)
            )
        
        # Издержки
        if 'costs' in data:
            costs_data = data['costs']
            config.costs = CostsConfig(
                commission_pct=costs_data.get('commission_pct', 0.001),
                slippage_pct=costs_data.get('slippage_pct', 0.0005),
                bid_ask_spread_pct_s1=costs_data.get('bid_ask_spread_pct_s1', 0.0002),
                bid_ask_spread_pct_s2=costs_data.get('bid_ask_spread_pct_s2', 0.0002),
                funding_rate_pct=costs_data.get('funding_rate_pct', 0.0001)
            )
        
        # Модель
        if 'model' in data:
            model_data = data['model']
            config.model = ModelConfig(
                lookback_window=model_data.get('lookback_window', 252),
                half_life_window=model_data.get('half_life_window', 50),
                volatility_window=model_data.get('volatility_window', 20),
                min_observations=model_data.get('min_observations', 100),
                max_pvalue=model_data.get('max_pvalue', 0.05),
                min_adf_stat=model_data.get('min_adf_stat', -2.5)
            )
        
        # Логирование
        if 'logging' in data:
            log_data = data['logging']
            config.logging = LoggingConfig(
                level=log_data.get('level', 'INFO'),
                log_trades=log_data.get('log_trades', True),
                log_signals=log_data.get('log_signals', False),
                log_performance=log_data.get('log_performance', True),
                trade_log_file=log_data.get('trade_log_file', 'logs/trades.csv'),
                performance_log_file=log_data.get('performance_log_file', 'logs/performance.csv'),
                signal_log_file=log_data.get('signal_log_file', 'logs/signals.csv')
            )
        
        # Отчетность
        if 'reporting' in data:
            rep_data = data['reporting']
            
            # Секции отчета
            sections_data = rep_data.get('sections', {})
            sections = ReportingSections(
                config=sections_data.get('config', True),
                metrics_test=sections_data.get('metrics_test', True),
                turnover=sections_data.get('turnover', True),
                cost_breakdown=sections_data.get('cost_breakdown', True),
                drawdown_table=sections_data.get('drawdown_table', True),
                sensitivity_fees=sections_data.get('sensitivity_fees', True)
            )
            
            # Анализ чувствительности
            sens_data = rep_data.get('sensitivity', {})
            sensitivity = SensitivityConfig(
                fee_multipliers=sens_data.get('fee_multipliers', [0.5, 1.0, 1.5, 2.0, 3.0, 5.0])
            )
            
            config.reporting = ReportingConfig(
                generate_plots=rep_data.get('generate_plots', True),
                save_results=rep_data.get('save_results', True),
                sections=sections,
                sensitivity=sensitivity
            )
        
        # Валидация
        if 'validation' in data:
            val_data = data['validation']
            config.validation = ValidationConfig(
                enable_robustness_tests=val_data.get('enable_robustness_tests', True),
                enable_synthetic_tests=val_data.get('enable_synthetic_tests', True),
                permutation_tests=val_data.get('permutation_tests', 10),
                bootstrap_samples=val_data.get('bootstrap_samples', 1000),
                max_sharpe_warning=val_data.get('max_sharpe_warning', 3.0),
                min_trades_warning=val_data.get('min_trades_warning', 5),
                max_turnover_warning=val_data.get('max_turnover_warning', 10.0)
            )
        
        # Метаданные
        if 'metadata' in data:
            meta_data = data['metadata']
            config.version = meta_data.get('version', '1.0.0')
        
        if 'technical' in data:
            tech_data = data['technical']
            config.random_seed = tech_data.get('random_seed', 42)
        
        return config
    
    def validate_config(self, config: BacktestConfig) -> bool:
        """Валидирует конфигурацию.
        
        Args:
            config: Объект конфигурации для валидации
            
        Returns:
            bool: True если конфигурация валидна
            
        Raises:
            ValueError: Если конфигурация невалидна
        """
        errors = []
        
        # Проверка walk-forward параметров
        if config.walk_forward.train_window <= 0:
            errors.append("train_window должен быть положительным")
        
        if config.walk_forward.test_window <= 0:
            errors.append("test_window должен быть положительным")
        
        if config.walk_forward.rolling_window <= 0:
            errors.append("rolling_window должен быть положительным")
        
        # Проверка сигналов
        if config.signals.z_entry <= 0:
            errors.append("z_entry должен быть положительным")
        
        if config.signals.z_exit < 0:
            errors.append("z_exit должен быть неотрицательным")
        
        if config.signals.z_exit >= config.signals.z_entry:
            errors.append("z_exit должен быть меньше z_entry")
        
        # Проверка управления рисками
        if config.risk_management.capital_at_risk <= 0:
            errors.append("capital_at_risk должен быть положительным")
        
        if config.risk_management.max_active_positions <= 0:
            errors.append("max_active_positions должен быть положительным")
        
        if config.risk_management.f_max <= 0 or config.risk_management.f_max > 1:
            errors.append("f_max должен быть в диапазоне (0, 1]")
        
        # Проверка издержек
        if config.costs.commission_pct < 0:
            errors.append("commission_pct не может быть отрицательным")
        
        if config.costs.slippage_pct < 0:
            errors.append("slippage_pct не может быть отрицательным")
        
        if errors:
            error_msg = "Ошибки валидации конфигурации:\n" + "\n".join(f"- {error}" for error in errors)
            raise ValueError(error_msg)
        
        return True
    
    def save_config_snapshot(self, config: BacktestConfig, output_path: str) -> None:
        """Сохраняет снимок конфигурации для воспроизводимости.
        
        Args:
            config: Конфигурация для сохранения
            output_path: Путь для сохранения снимка
        """
        import json
        from datetime import datetime
        
        snapshot = {
            'timestamp': datetime.now().isoformat(),
            'config_version': config.version,
            'config': self._config_to_dict(config)
        }
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(snapshot, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Снимок конфигурации сохранен: {output_path}")
    
    def _config_to_dict(self, config: BacktestConfig) -> Dict[str, Any]:
        """Конвертирует объект конфигурации в словарь.
        
        Args:
            config: Объект конфигурации
            
        Returns:
            Dict[str, Any]: Словарь с конфигурацией
        """
        import dataclasses
        
        def asdict_recursive(obj):
            if dataclasses.is_dataclass(obj):
                return {k: asdict_recursive(v) for k, v in dataclasses.asdict(obj).items()}
            elif isinstance(obj, list):
                return [asdict_recursive(item) for item in obj]
            elif isinstance(obj, dict):
                return {k: asdict_recursive(v) for k, v in obj.items()}
            else:
                return obj
        
        return asdict_recursive(config)


# Глобальный экземпляр загрузчика
_config_loader = None


def get_config(config_path: Optional[str] = None) -> BacktestConfig:
    """Получает конфигурацию (singleton pattern).
    
    Args:
        config_path: Путь к файлу конфигурации
        
    Returns:
        BacktestConfig: Объект конфигурации
    """
    global _config_loader
    
    if _config_loader is None:
        _config_loader = ConfigLoader(config_path)
    
    config = _config_loader.load_config()
    _config_loader.validate_config(config)
    
    return config