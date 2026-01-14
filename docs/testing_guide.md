# Руководство по тестированию

> Важно: актуальные тесты находятся в `coint4/tests` и запускаются из `coint4/`.
> Описанная ниже структура относится к legacy-набору и будет обновлена.

## Обзор

Проект использует pytest для тестирования. После очистки от дублирующихся файлов структура тестов стала более организованной и поддерживаемой.

## Артефакты и тестовые данные (актуально для `coint4/`)

- `coint4/artifacts/live/` — runtime-артефакты (логи, метрики, snapshots); новые файлы сюда не коммитятся.
- `coint4/artifacts/live/PREFLIGHT_REPORT.md` — отслеживаемый шаблон отчета, временная метка пишется в `PREFLIGHT.log`.
- `coint4/artifacts/baseline_traces/` — golden traces для регрессионных тестов, генерируются на лету и игнорируются.
- `coint4/data_downloaded/` — директория держится через `.gitkeep`, реальные данные игнорируются.
- В тестах использовать `tmp_path`, если не нужен преднамеренно отслеживаемый файл.

## Воспроизводимый прогон (scan → backtest → walk-forward)

Команды запускаются из `coint4/`. После `./.venv/bin/pip install -e .` используйте `./.venv/bin/coint2`.

### Ручной прогон
```bash
./.venv/bin/coint2 scan \
  --config configs/criteria_relaxed.yaml \
  --base-config configs/main_2024.yaml \
  --output-dir bench \
  --symbols ALL

./.venv/bin/coint2 backtest \
  --config configs/main_2024.yaml \
  --pairs-file bench/pairs_universe.yaml \
  --period-start 2023-06-01 \
  --period-end 2023-08-31 \
  --out-dir outputs/fixed_run

./.venv/bin/coint2 walk-forward \
  --config configs/main_2024.yaml
```

### Скрипт для типового прогона
```bash
bash scripts/run_pipeline.sh
```

Переменные окружения для скрипта:
`BASE_CONFIG`, `CRITERIA_CONFIG`, `DATA_ROOT`, `SYMBOLS`, `OUT_DIR`, `PAIRS_FILE`,
`BACKTEST_OUT`, `START_DATE`, `END_DATE`, `PYTHON_BIN`.

Пример запуска с ограничением на пары и период:
```bash
DATA_ROOT=data_downloaded \
SYMBOLS=BTCUSDT,ETHUSDT \
START_DATE=2023-06-01 \
END_DATE=2023-08-31 \
bash scripts/run_pipeline.sh
```

Ожидаемые артефакты:
- `bench/pairs_universe.yaml` и `bench/UNIVERSE_REPORT.md`
- `outputs/fixed_run/metrics.yaml`, `outputs/fixed_run/trades.csv`, `outputs/fixed_run/equity.csv`
- `results/strategy_metrics.csv` (walk-forward), а также `results/daily_pnl.csv` и `results/equity_curve.csv`

## Структура тестов

Актуальная структура (см. также `tests/README.md`):

- `tests/ci/` — быстрые проверки для CI (gates, drawdown).
- `tests/smoke/` — smoke/preflight/observability, быстрые проверки импорта и Optuna.
- `tests/unit/` — юнит-тесты по подсистемам: `core/`, `pipeline/`, `stats/`, `optimiser/`, `portfolio/`, `backtest/`, `execution/`, `utils/`, `engine/`.
- `tests/integration/` — интеграционные сценарии: `backtest/`, `pipeline/`, `walk_forward/`, `optimization/`, `lookahead/`, `wfa/`.
- `tests/regression/` — regression/golden traces.
- `tests/determinism/` — повторяемость результатов.
- `tests/governance/` — promotion/rollback пайплайны.
- `tests/monitoring/` — мониторинг и дрейф.
- `tests/performance/` — бенчмарки, кэширование, параллелизм.
- `tests/analytics/`, `tests/validation/`, `tests/test_critical_fixes/` — аналитика, утечки, критические регрессии.
- Дополнительно присутствуют `tests/core/`, `tests/engine/`, `tests/stats/`, `tests/pipeline/`, `tests/portfolio/` — историческая группировка, всё еще используется в CI.

## Запуск тестов

### Все тесты
```bash
./.venv/bin/pytest -q
```

По умолчанию `pytest.ini` исключает `slow` и `serial` (см. `addopts`).
Чтобы запустить их, используйте:
```bash
./.venv/bin/pytest -m "slow or serial" --override-ini addopts=""
```

### Конкретная категория
```bash
# Unit-тесты
./.venv/bin/pytest tests/unit -q

# Интеграционные
./.venv/bin/pytest tests/integration -q

# Smoke
./.venv/bin/pytest tests/smoke -q
```

### Smoke прогон
```bash
bash scripts/test.sh
```

### Конкретный файл
```bash
./.venv/bin/pytest tests/smoke/test_preflight.py -q
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
- **Файлы:** `tests/integration/backtest/test_backtest_engine.py`, `tests/integration/backtest/test_pair_backtester_integration.py`
- **Цель:** Проверка движка и интеграции парного бэктестера
- **Покрывает:** Базовый торговый цикл и метрики

### 2. Оптимизации производительности
- **Файлы:** `tests/performance/benchmarks/test_performance.py`, `tests/performance/caching/test_optimized_cache_performance.py`
- **Цель:** Проверка ускорений и поведения кэша
- **Покрывает:** Бенчмарки, кэширование, параллелизм

### 3. Критические исправления
- **Файлы:** `tests/test_critical_fixes/test_trade_metrics_correctness.py`, `tests/regression/test_golden_traces.py`
- **Цель:** Защита от регрессий
- **Покрывает:** PnL/метрики и эталонные трассы

### 4. Walk-Forward анализ
- **Файлы:** `tests/integration/walk_forward/test_walk_forward_integration.py`, `tests/integration/pipeline/test_walk_forward.py`
- **Цель:** Проверка временного разделения данных
- **Покрывает:** Обучение, тестирование, агрегацию метрик

### 5. Smoke/Preflight
- **Файлы:** `tests/smoke/test_preflight.py`, `tests/smoke/test_observability.py`
- **Цель:** Быстрая валидация окружения и артефактов
- **Покрывает:** Импорты, конфиги, утилиты наблюдаемости

## Конфигурация pytest

### `conftest.py`
Содержит общие фикстуры и настройки для всех тестов.

### Маркеры
```python
# Быстрые smoke-тесты
@pytest.mark.smoke

# Медленные тесты
@pytest.mark.slow

# Интеграционные тесты
@pytest.mark.integration

# Тесты производительности
@pytest.mark.performance

# Юнит-тесты
@pytest.mark.unit

# Критические фиксы
@pytest.mark.critical_fixes
```

### Запуск по маркерам
```bash
# Только smoke
./.venv/bin/pytest tests/ -m smoke

# Только интеграционные
./.venv/bin/pytest tests/ -m integration

# Только производительность
./.venv/bin/pytest tests/ -m performance
```

## Performance audit

```bash
PYTHONPATH=src ./.venv/bin/python scripts/perf_audit.py --config configs/main_2024.yaml
```

Полезные флаги: `--data-root`, `--lookback-days`, `--end-date`, `--skip-cache`, `--skip-coint`.

## UI/Streamlit preflight

```bash
PYTHONPATH=src ./.venv/bin/python scripts/ui_preflight.py \
  --config configs/main_2024.yaml \
  --search-space configs/search_spaces/web_ui.yaml
```

Если зависимости не установлены, можно получить предупреждения без ошибки:
```bash
PYTHONPATH=src ./.venv/bin/python scripts/ui_preflight.py --allow-missing
```

Запуск UI:
```bash
PYTHONPATH=src ./.venv/bin/streamlit run ui/app.py
```

## Проверка данных (parquet)

Быстрая проверка структуры и качества выгрузок:
```bash
PYTHONPATH=src ./.venv/bin/python scripts/validate_data_dump.py \
  --data-root data_downloaded \
  --mode raw \
  --config configs/main_2024.yaml
```

Для помесячной структуры (`year=YYYY/month=MM/*.parquet`) используйте режим `monthly`
— скрипт также переключается на него автоматически при обнаружении такой структуры:
```bash
PYTHONPATH=src ./.venv/bin/python scripts/validate_data_dump.py \
  --data-root data_downloaded \
  --mode monthly \
  --symbols BTCUSDT \
  --config configs/main_2024.yaml
```
Для строгих порогов качества можно передать `configs/data_quality_strict.yaml`:
```bash
PYTHONPATH=src ./.venv/bin/python scripts/validate_data_dump.py \
  --data-root data_downloaded \
  --mode monthly \
  --symbols BTCUSDT \
  --config configs/data_quality_strict.yaml
```
Для multi-symbol monthly файлов в отчете поле `symbol` будет `ALL` (или список символов, если фильтровали `--symbols`).
Схема выводится через `collect_schema()` (без лишней материализации данных).

Для очищенной структуры:
```bash
PYTHONPATH=src ./.venv/bin/python scripts/validate_data_dump.py \
  --data-root data_clean \
  --mode clean \
  --config configs/main_2024.yaml
```

Сохранение отчета:
```bash
PYTHONPATH=src ./.venv/bin/python scripts/validate_data_dump.py \
  --report outputs/data_quality_report.csv
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
