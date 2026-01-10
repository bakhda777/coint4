# CI Smoke Tests

Система smoke тестов для CI/CD pipeline проекта коинтеграционной торговой стратегии.

## Компоненты

### 1. `scripts/ci_smoke.py`
Быстрые критичные тесты для проверки основной функциональности системы:

- **Critical imports** - Проверка импорта ключевых модулей
- **Config loading** - Валидация конфигурационных файлов
- **Data pipeline** - Тестирование обработки данных
- **Numba compilation** - JIT компиляция критичных функций
- **Engine parity** - Соответствие Numba и Reference движков
- **Trace system** - Сохранение и загрузка трейсов
- **Optuna basic** - Базовая оптимизация (3 trials)
- **WFA components** - Компоненты Walk-Forward Analysis

**Время выполнения:** < 30 секунд  
**Exit codes:** 0 = успех, 1 = критичная ошибка

#### Использование:
```bash
# Локальный запуск
python scripts/ci_smoke.py

# Через make (если настроен)
make ci-smoke

# В CI/CD окружении (автоматически)
poetry run python scripts/ci_smoke.py
```

### 2. `.github/workflows/ci.yml`
GitHub Actions workflow для автоматического запуска CI тестов:

#### Триггеры:
- Push в ветки: `main`, `develop`, `feature/*`
- Pull Request в ветки: `main`, `develop`
- Ручной запуск (`workflow_dispatch`)

#### Матрица тестирования:
- Python версии: 3.9, 3.10, 3.11
- OS: Ubuntu Latest
- Timeout: 10 минут общий лимит

#### Этапы:
1. **Подготовка**: Checkout кода, установка Python, Poetry
2. **Кэширование**: Зависимости Poetry, виртуальные окружения
3. **Установка**: Зависимости через Poetry
4. **Данные**: Создание минимальной структуры данных для тестов
5. **Lint**: Быстрая проверка критичных модулей
6. **Smoke Tests**: Запуск `ci_smoke.py`
7. **Pytest**: Базовые pytest smoke тесты
8. **Валидация**: Проверка конфигураций
9. **Coverage**: Опциональный отчет покрытия

#### Дополнительные job:
- **fast-validation**: Быстрая валидация для PR
- **notify**: Уведомления о результатах

## Критерии успеха

### Smoke тесты должны:
✅ Завершаться за < 30 секунд  
✅ Тестировать критичную функциональность  
✅ Использовать минимальные datasets  
✅ Проверять engine parity  
✅ Валидировать конфигурации  
✅ Тестировать Numba JIT  
✅ Проверять trace систему  

### CI workflow должен:
✅ Поддерживать multiple Python версии  
✅ Кэшировать зависимости  
✅ Отказоустойчив (continue-on-error где уместно)  
✅ Генерировать понятные отчеты  
✅ Не превышать 10 минут  

## Мониторинг и отладка

### Локальная отладка:
```bash
# Запуск отдельных тестов
python -c "
from scripts.ci_smoke import CISmokeRunner
runner = CISmokeRunner()
result = runner.test_critical_imports()
print(f'Result: {result.passed}, Error: {result.error}')
"

# Проверка конфигурации
python -c "
from src.coint2.utils.config import load_config
config = load_config('configs/main_2024.yaml')
print(f'Config loaded: {hasattr(config, \"backtest\")}')
"
```

### CI/CD мониторинг:
- GitHub Actions logs
- Codecov reports (если настроен)
- Performance timing в логах
- Memory usage tracking

## Настройка проекта

Для корректной работы CI smoke тестов проект должен иметь:

1. **Структуру директорий:**
   ```
   src/
     coint2/
     optimiser/
   configs/
     main_2024.yaml
     search_spaces/fast.yaml
   tests/
   scripts/
   ```

2. **Зависимости в pyproject.toml:**
   - optuna
   - numba 
   - pandas
   - numpy
   - pytest
   - poetry

3. **Конфигурационные файлы:**
   - `configs/main_2024.yaml` - основная конфигурация
   - `configs/search_spaces/fast.yaml` - search space для Optuna
   - `pytest.ini` - маркеры тестов

4. **Минимальные данные:**
   - `data_downloaded/year=2024/month=01/data_part_01.parquet`

## Интеграция с другими системами

### Pre-commit hooks:
```yaml
- repo: local
  hooks:
  - id: ci-smoke-tests
    name: CI Smoke Tests
    entry: python scripts/ci_smoke.py
    language: python
    pass_filenames: false
```

### Makefile:
```makefile
.PHONY: ci-smoke
ci-smoke:
	python scripts/ci_smoke.py

.PHONY: ci-full  
ci-full: ci-smoke
	pytest -m "fast and not serial"
```

### Docker:
```dockerfile
# В Dockerfile для CI
RUN poetry install --only=main,test
RUN python scripts/ci_smoke.py
```

## Обновление и поддержка

### При добавлении новых модулей:
1. Добавить импорт в `test_critical_imports()`
2. Обновить список проверяемых конфигураций
3. Добавить специфичные проверки если нужно

### При изменении архитектуры:
1. Обновить engine parity тесты
2. Проверить trace format совместимость  
3. Обновить WFA component тесты

### Performance tuning:
- Мониторить время выполнения тестов
- Оптимизировать медленные компоненты
- Использовать кэширование где возможно
- Минимизировать размер тестовых данных

## Troubleshooting

### Типичные проблемы:

**Import errors:**
- Проверить PYTHONPATH
- Убедиться что все зависимости установлены
- Проверить структуру src/ директории

**Config validation errors:**
- Проверить актуальность main_2024.yaml
- Убедиться что все требуемые секции присутствуют
- Проверить типы данных параметров

**Numba compilation errors:**  
- Проверить совместимость версий numba
- Убедиться что JIT не отключен (NUMBA_DISABLE_JIT=0)
- Проверить сигнатуры функций

**Timeout errors:**
- Увеличить timeout в CI
- Оптимизировать медленные тесты
- Проверить ресурсы CI runner

**Engine parity failures:**
- Проверить алгоритмические изменения
- Убедиться в детерминизме (seed)
- Проверить численную стабильность