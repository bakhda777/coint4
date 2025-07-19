# Руководство по тестированию

## Обзор

Проект использует pytest для тестирования. После очистки от дублирующихся файлов структура тестов стала более организованной и поддерживаемой.

## Структура тестов

### Корневая папка `tests/`

#### Основные интеграционные тесты
- `test_backtest_integration.py` - интеграционные тесты бэктестинга
- `test_optimized_backtest_engine.py` - тесты оптимизированного движка
- `test_walk_forward_integration.py` - интеграционные тесты Walk-Forward анализа
- `test_global_cache_integration.py` - тесты глобального кэширования
- `test_memory_optimization.py` - тесты оптимизации памяти

#### Тесты оптимизаций
- `test_numba_full_optimization.py` - полная Numba оптимизация
- `test_critical_fixes_comprehensive.py` - комплексные критические исправления

#### Специализированные тесты
- `test_backtest_correctness_with_blas.py` - корректность с BLAS
- `test_pnl_calculation_fix.py` - исправления расчета PnL
- `test_walk_forward_risk_fix.py` - исправления управления рисками

### Подпапки

#### `tests/core/` - Базовая функциональность
- `test_cache.py` - тесты кэширования
- `test_data_loader.py` - загрузка данных
- `test_fast_coint.py` - быстрая коинтеграция
- `test_file_glob.py` - работа с файлами
- `test_intraday_frequency.py` - внутридневные частоты
- `test_math_utils.py` - математические утилиты
- `test_pair_backtester_integration.py` - интеграция парного бэктестера
- `test_performance.py` - тесты производительности

#### `tests/engine/` - Движки бэктестинга
- `test_backtest_engine.py` - основной движок
- `test_backtest_engine_optimization.py` - оптимизации движка
- `test_backtest_fixes.py` - исправления движка
- `test_enhanced_risk_management.py` - улучшенное управление рисками
- `test_lookahead_bias_fix.py` - исправление look-ahead bias
- `test_market_regime_detection.py` - определение рыночных режимов
- `test_market_regime_optimization.py` - оптимизация режимов
- `test_max_positions_increase.py` - увеличение максимальных позиций
- `test_portfolio_position_limits.py` - лимиты позиций портфеля
- `test_volatility_based_sizing.py` - размер позиций на основе волатильности

#### `tests/pipeline/` - Пайплайны обработки
- `test_filters_beta.py` - бета-фильтры
- `test_pair_scanner_integration.py` - интеграция сканера пар
- `test_walk_forward.py` - Walk-Forward анализ

#### `tests/utils/` - Утилиты
- `test_config_loading.py` - загрузка конфигурации
- `test_time_utils.py` - временные утилиты

## Запуск тестов

### Все тесты
```bash
pytest tests/
```

### Конкретная категория
```bash
# Тесты движков
pytest tests/engine/

# Базовые тесты
pytest tests/core/

# Интеграционные тесты
pytest tests/test_*_integration.py
```

### Конкретный файл
```bash
pytest tests/test_backtest_integration.py
```

### С подробным выводом
```bash
pytest tests/ -v
```

### Только неудачные тесты
```bash
pytest tests/ --tb=short
```

## Покрытие кода

```bash
# Запуск с покрытием
pytest tests/ --cov=src/coint2

# HTML отчет
pytest tests/ --cov=src/coint2 --cov-report=html
```

## Ключевые тестовые сценарии

### 1. Корректность бэктестинга
- **Файл:** `test_backtest_integration.py`
- **Цель:** Проверка общей корректности пайплайна
- **Покрывает:** Полный цикл от данных до результатов

### 2. Оптимизации производительности
- **Файлы:** `test_numba_full_optimization.py`, `test_memory_optimization.py`
- **Цель:** Проверка ускорений и экономии памяти
- **Покрывает:** Numba JIT, memory mapping, кэширование

### 3. Критические исправления
- **Файл:** `test_critical_fixes_comprehensive.py`
- **Цель:** Проверка исправления критических ошибок
- **Покрывает:** Look-ahead bias, расчет PnL, управление позициями

### 4. Walk-Forward анализ
- **Файл:** `test_walk_forward_integration.py`
- **Цель:** Проверка временного разделения данных
- **Покрывает:** Обучение, тестирование, валидация

### 5. Глобальное кэширование
- **Файл:** `test_global_cache_integration.py`
- **Цель:** Проверка эффективности кэша
- **Покрывает:** Rolling статистики, memory mapping

## Конфигурация pytest

### `conftest.py`
Содержит общие фикстуры и настройки для всех тестов.

### Маркеры
```python
# Медленные тесты
@pytest.mark.slow

# Интеграционные тесты
@pytest.mark.integration

# Тесты производительности
@pytest.mark.performance
```

### Запуск по маркерам
```bash
# Только быстрые тесты
pytest tests/ -m "not slow"

# Только интеграционные
pytest tests/ -m integration
```

## Лучшие практики

### 1. Именование тестов
- Используйте описательные имена: `test_backtest_handles_missing_data`
- Группируйте по функциональности: `TestBacktestEngine`

### 2. Структура тестов
```python
def test_feature_name():
    # Arrange - подготовка данных
    data = create_test_data()
    
    # Act - выполнение действия
    result = process_data(data)
    
    # Assert - проверка результата
    assert result.is_valid()
```

### 3. Фикстуры
- Используйте фикстуры для общих данных
- Применяйте `scope` для оптимизации

### 4. Моки и патчи
- Мокайте внешние зависимости
- Используйте `unittest.mock` для изоляции

## Отладка тестов

### Запуск с отладчиком
```bash
pytest tests/test_file.py::test_function --pdb
```

### Логирование в тестах
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Временные файлы
```python
import tempfile
with tempfile.TemporaryDirectory() as tmpdir:
    # Тест с временными файлами
    pass
```

## Непрерывная интеграция

Тесты автоматически запускаются при:
- Push в основную ветку
- Создании Pull Request
- Релизе новой версии

### Требования к прохождению
- Все тесты должны проходить
- Покрытие кода > 80%
- Нет критических предупреждений

## Добавление новых тестов

### 1. Определите категорию
- Базовая функциональность → `tests/core/`
- Движки бэктестинга → `tests/engine/`
- Пайплайны → `tests/pipeline/`
- Утилиты → `tests/utils/`
- Интеграционные → корневая папка

### 2. Следуйте конвенциям
- Имя файла: `test_feature_name.py`
- Имя класса: `TestFeatureName`
- Имя метода: `test_specific_behavior`

### 3. Документируйте тесты
```python
def test_complex_feature():
    """Test that complex feature works correctly.
    
    This test verifies that the feature:
    1. Handles edge cases
    2. Returns correct results
    3. Maintains performance
    """
```

## Результат очистки

**До очистки:** 29 файлов в корневой папке tests/  
**После очистки:** 14 файлов в корневой папке tests/  
**Удалено:** 15 дублирующихся файлов

### Преимущества
- ✅ Упрощенная поддержка
- ✅ Ускоренное выполнение CI/CD
- ✅ Лучшая читаемость структуры
- ✅ Отсутствие дублирования функциональности
- ✅ Более четкое разделение ответственности

Все ключевые функции остались покрыты тестами, но теперь структура более организована и поддерживаема.