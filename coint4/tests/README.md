# Структура тестов

## Организация папок

```
tests/
├── ci/                     # Быстрые проверки для CI
├── unit/                   # Быстрые изолированные тесты (<1 сек каждый)
│   ├── core/              # Unit тесты для core модуля
│   ├── engine/            # Unit тесты для engine  
│   ├── optimiser/         # Unit тесты для оптимизатора
│   ├── pipeline/          # Unit тесты для pipeline
│   ├── utils/             # Unit тесты для утилит
│   └── backtest/          # Unit тесты для бэктеста
│
├── integration/           # Интеграционные тесты (могут использовать реальные данные)
│   ├── backtest/         # Тесты полного цикла бэктестирования
│   ├── optimization/     # Тесты оптимизации (Optuna)
│   ├── pipeline/         # Тесты всего pipeline
│   └── walk_forward/     # Walk-forward тесты
│
├── smoke/                 # Smoke/preflight/observability
├── regression/            # Golden traces
├── determinism/           # Повторяемость результатов
├── governance/            # Promotion/rollback пайплайны
├── monitoring/            # Мониторинг и дрейф
├── performance/          # Тесты производительности
│   ├── caching/         # Тесты кэширования и памяти
│   ├── parallel/        # Тесты параллельности и потокобезопасности
│   └── benchmarks/      # Бенчмарки и измерения скорости
│
├── validation/            # Проверки утечек/валидации
├── analytics/             # Аналитические сценарии
├── test_critical_fixes/   # Критические фиксы
├── fixtures/            # Общие фикстуры для всех тестов
└── conftest.py         # Главный конфигурационный файл pytest
```

## Правила именования

- **Unit тесты**: `test_<module>_<feature>.py`
- **Integration тесты**: `test_<system>_integration.py`
- **Performance тесты**: `test_<aspect>_performance.py`

## Маркеры pytest

```python
@pytest.mark.unit         # Быстрые изолированные тесты
@pytest.mark.fast         # Тесты < 5 сек
@pytest.mark.slow         # Тесты > 5 сек  
@pytest.mark.integration  # Тесты с реальными данными/системами
@pytest.mark.serial       # Тесты, которые нельзя параллелить
@pytest.mark.smoke        # Критически важные тесты
```

## Запуск тестов

```bash
# Только unit тесты (самые быстрые)
./.venv/bin/pytest tests/unit -v

# Fast тесты для CI/CD
./.venv/bin/pytest -m "fast or unit"

# Все кроме медленных
./.venv/bin/pytest -m "not slow"

# Только интеграционные
./.venv/bin/pytest tests/integration

# Тесты производительности
./.venv/bin/pytest tests/performance

# Smoke тесты
./.venv/bin/pytest -m smoke

# Полный набор
./.venv/bin/pytest -q
```

## Время выполнения

- **Unit тесты**: < 1 сек каждый, < 30 сек все
- **Fast тесты**: < 5 сек каждый, < 2 мин все
- **Integration тесты**: < 30 сек каждый
- **Slow тесты**: могут занимать минуты

По умолчанию `pytest.ini` исключает `slow` и `serial` (см. `addopts`).

## Соглашения

1. **Один тест - одна проверка**: Каждый тест проверяет только один аспект
2. **Именование**: `test_<что>_when_<условие>_then_<результат>`
3. **Фикстуры**: Используйте минимальные фикстуры (tiny_data, small_data)
4. **Мокирование**: Мокируйте внешние зависимости в unit тестах
5. **Детерминизм**: Используйте фикстуру `rng` для случайных чисел
