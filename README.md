# Cointegration Pairs Trading System

Продвинутая система для торговли коинтегрированными парами с комплексной оптимизацией производительности и анти-leak мерами.

## Основные возможности

### 🚀 Производительность
- **Walk-Forward анализ** с инкрементальной обработкой данных
- **Глобальный кэш** для rolling статистик с оптимизацией памяти
- **Numba-ускоренные** математические вычисления
- **Детекция рыночных режимов** с кэшированием
- **Batch-оптимизация** параметров с grid/random search

### 🛡️ Анти-leak меры
- **Строгий сдвиг сигналов** для предотвращения look-ahead bias
- **Нормализация fit/transform** для корректной обработки данных
- **Онлайновые статистики** без использования будущих данных
- **Валидация временных рядов** на каждом шаге

### 💰 Управление рисками
- **Динамическое управление капиталом** с учетом волатильности
- **Комиссии и проскальзывание** с детальным моделированием
- **Stop-loss и take-profit** с адаптивными уровнями
- **Контроль маржи** и лимиты позиций

### 📊 Аналитика
- **Комплексные метрики** производительности
- **Детальная отчетность** с визуализацией
- **Анализ просадок** и периодов восстановления
- **Статистика сделок** и их характеристики

## Быстрый старт

### Установка зависимостей

```bash
pip install -r requirements.txt
```

### Базовое использование

```python
import pandas as pd
from src.coint2.engine.backtest_engine import PairBacktester

# Загрузка данных
data = pd.read_csv('your_pair_data.csv', index_col=0, parse_dates=True)

# Создание и запуск бэктестера
engine = PairBacktester(
    pair_data=data,
    rolling_window=30,
    z_threshold=2.0,
    z_exit=0.5,
    capital_at_risk=100000.0,
    commission_pct=0.001,
    slippage_pct=0.0005
)

engine.run()
metrics = engine.get_performance_metrics()
print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.3f}")
print(f"Total Return: {metrics['total_return']:.3f}")
```

### Оптимизация параметров

```python
from src.coint2.optimization.batch_optimizer import BatchOptimizer

# Настройка параметров для оптимизации
param_grid = {
    'rolling_window': [20, 30, 40],
    'z_threshold': [1.5, 2.0, 2.5],
    'z_exit': [0.3, 0.5, 0.7]
}

# Запуск оптимизации
optimizer = BatchOptimizer(
    pair_data=data,
    param_grid=param_grid,
    optimization_metric='sharpe_ratio',
    cv_folds=3
)

best_params = optimizer.optimize()
print(f"Лучшие параметры: {best_params}")
```

### Walk-Forward анализ

```python
from src.coint2.validation.walk_forward import WalkForwardValidator

# Настройка walk-forward валидации
validator = WalkForwardValidator(
    pair_data=data,
    train_period_days=252,  # 1 год для обучения
    test_period_days=63,    # 3 месяца для тестирования
    step_days=21           # Шаг 3 недели
)

# Запуск валидации
results = validator.run_validation(
    param_grid=param_grid,
    optimization_metric='sharpe_ratio'
)

print(f"Средний out-of-sample Sharpe: {results['mean_oos_sharpe']:.3f}")
```

## Структура проекта

```
coint4/
├── src/coint2/
│   ├── engine/           # Основные движки бэктестинга
│   ├── core/            # Базовые утилиты и математика
│   ├── optimization/    # Оптимизация параметров
│   ├── validation/      # Walk-forward и валидация
│   ├── reporting/       # Генерация отчетов
│   └── testing/         # Тестовые утилиты
├── tests/               # Unit и интеграционные тесты
├── configs/             # Конфигурационные файлы
└── test_reports/        # Результаты тестирования
```

## Конфигурация

Основные параметры системы настраиваются через файлы в папке `configs/`:

- `system_config.yaml` - общие настройки системы
- `optimization_config.yaml` - параметры оптимизации
- `backtest_config.yaml` - настройки бэктестинга

## Тестирование

### Запуск всех тестов

```bash
python -m pytest tests/ -v
```

### Запуск конкретных групп тестов

```bash
# Тесты робастности
python -m pytest tests/test_backtest_robustness.py -v

# Синтетические тесты
python -m pytest tests/test_synthetic_scenarios.py -v

# Тесты оптимизации
python -m pytest tests/test_optimized_backtest_engine.py -v
```

### Комплексное тестирование

```bash
# Запуск полного тестового набора
python src/coint2/testing/test_runner.py
```

## Производительность

Система оптимизирована для работы с большими объемами данных:

- **Глобальный кэш**: до 10x ускорение для множественных пар
- **Numba JIT**: до 50x ускорение математических вычислений
- **Инкрементальная обработка**: минимальное потребление памяти
- **Векторизованные операции**: эффективное использование NumPy/Pandas

## Метрики качества

Система рассчитывает широкий спектр метрик:

### Доходность
- Total Return
- Annualized Return
- Excess Return
- Risk-adjusted Return

### Риск
- Volatility
- Maximum Drawdown
- Value at Risk (VaR)
- Conditional VaR

### Эффективность
- Sharpe Ratio
- Sortino Ratio
- Calmar Ratio
- Information Ratio

### Торговая активность
- Number of Trades
- Win Rate
- Average Trade Duration
- Turnover

## Лицензия

MIT License - см. файл LICENSE для деталей.

## Поддержка

Для вопросов и предложений создавайте issues в репозитории.