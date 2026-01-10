"""
Кастомные исключения для проекта coint2.
Заменяют общие except Exception для лучшей обработки ошибок.
"""


class Coint2Error(Exception):
    """Базовое исключение для всех ошибок проекта."""
    pass


# Ошибки данных
class DataError(Coint2Error):
    """Ошибки связанные с данными."""
    pass


class DataLoadError(DataError):
    """Ошибка загрузки данных."""
    pass


class DataValidationError(DataError):
    """Ошибка валидации данных."""
    pass


class InsufficientDataError(DataError):
    """Недостаточно данных для анализа."""
    pass


# Ошибки конфигурации
class ConfigError(Coint2Error):
    """Ошибки конфигурации."""
    pass


class ConfigValidationError(ConfigError):
    """Ошибка валидации конфигурации."""
    pass


class MissingConfigError(ConfigError):
    """Отсутствует необходимая конфигурация."""
    pass


# Ошибки оптимизации
class OptimizationError(Coint2Error):
    """Ошибки в процессе оптимизации."""
    pass


class ObjectiveError(OptimizationError):
    """Ошибка в objective функции."""
    pass


class PruningError(OptimizationError):
    """Ошибка при pruning."""
    pass


# Ошибки бэктеста
class BacktestError(Coint2Error):
    """Ошибки бэктестинга."""
    pass


class PositionSizingError(BacktestError):
    """Ошибка расчета размера позиции."""
    pass


class TradingSignalError(BacktestError):
    """Ошибка генерации торговых сигналов."""
    pass


# Ошибки статистических тестов
class StatisticalError(Coint2Error):
    """Ошибки статистических тестов."""
    pass


class CointegrationError(StatisticalError):
    """Ошибка теста на коинтеграцию."""
    pass


class StationarityError(StatisticalError):
    """Ошибка теста на стационарность."""
    pass


# Ошибки кэширования
class CacheError(Coint2Error):
    """Ошибки работы с кэшем."""
    pass


class CacheInitError(CacheError):
    """Ошибка инициализации кэша."""
    pass


class CacheAccessError(CacheError):
    """Ошибка доступа к кэшу."""
    pass


# Ошибки валидации
class ValidationError(Coint2Error):
    """Ошибки валидации."""
    pass


class LookaheadBiasError(ValidationError):
    """Обнаружен lookahead bias."""
    pass


class ParameterValidationError(ValidationError):
    """Ошибка валидации параметров."""
    pass


# Ошибки вычислений
class ComputationError(Coint2Error):
    """Ошибки вычислений."""
    pass


class NumericalError(ComputationError):
    """Численная ошибка (деление на ноль, NaN, etc)."""
    pass


class ConvergenceError(ComputationError):
    """Ошибка сходимости алгоритма."""
    pass