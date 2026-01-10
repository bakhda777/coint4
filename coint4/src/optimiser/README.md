# Optuna Optimization Module

Минималистичный модуль для оптимизации гиперпараметров стратегии парного трейдинга с использованием Optuna.

## Структура модуля

```
src/optimiser/
├── fast_objective.py      # Быстрая objective функция для Optuna (основная)
├── metric_utils.py        # Утилиты для валидации и нормализации параметров
├── run_optimization.py    # Главный скрипт запуска оптимизации
├── __init__.py           # Инициализация пакета
└── README.md             # Эта документация
```

## Использование

### Быстрая оптимизация (рекомендуется)
```bash
# Быстрая оптимизация с ослабленными параметрами
python src/optimiser/run_optimization.py \
    --n-trials 50 \
    --study-name quick_optimization \
    --search-space configs/search_space_relaxed.yaml \
    --fast

# Полная оптимизация с основными параметрами
python src/optimiser/run_optimization.py \
    --n-trials 200 \
    --study-name full_optimization \
    --search-space configs/search_space.yaml \
    --fast
```

### Параметры командной строки
- `--n-trials` - количество trials (по умолчанию: 200)
- `--study-name` - имя study (по умолчанию: pairs_strategy_v1)
- `--storage-path` - путь к БД (по умолчанию: outputs/studies/pairs_strategy_v1.db)
- `--search-space` - файл пространства поиска (по умолчанию: configs/search_space.yaml)
- `--base-config` - базовая конфигурация (по умолчанию: configs/main_2024.yaml)
- `--fast` - использовать быструю objective функцию (рекомендуется)
- `--n-jobs` - количество параллельных процессов (по умолчанию: -1)
- `--seed` - seed для воспроизводимости (по умолчанию: 42)

## Конфигурация

### Пространства поиска
- **`configs/search_space.yaml`** - Основное пространство поиска (строгие параметры)
- **`configs/search_space_relaxed.yaml`** - Ослабленное пространство поиска (для генерации сделок)

### Базовая конфигурация
- **`configs/main_2024.yaml`** - Базовая конфигурация стратегии

## Оптимизируемые параметры

Optuna подбирает **18 параметров** одновременно:

### Отбор пар (6 параметров)
- `ssd_top_n` - количество пар для анализа
- `kpss_pvalue_threshold` - порог стационарности
- `coint_pvalue_threshold` - порог коинтеграции
- `min_half_life_days` - минимальный полураспад
- `max_half_life_days` - максимальный полураспад
- `min_mean_crossings` - пересечения среднего

### Сигналы (3 параметра)
- `zscore_threshold` - порог входа в позицию
- `zscore_exit` - порог выхода из позиции
- `rolling_window` - размер скользящего окна

### Управление рисками (3 параметра)
- `stop_loss_multiplier` - множитель стоп-лосса
- `time_stop_multiplier` - множитель временного стопа
- `cooldown_hours` - период охлаждения

### Портфель (3 параметра)
- `max_active_positions` - максимум позиций
- `risk_per_position_pct` - риск на позицию
- `max_position_size_pct` - размер позиции

### Издержки (2 параметра)
- `commission_pct` - комиссия за сделку
- `slippage_pct` - проскальзывание

### Нормализация (2 параметра)
- `normalization_method` - метод нормализации
- `min_history_ratio` - требования к истории

## Результаты

Оптимизация максимизирует скорректированный Sharpe ratio:
```
final_score = sharpe_ratio - drawdown_penalty + win_rate_bonus - win_rate_penalty
```

### Выходные файлы
- **`outputs/studies/[study_name].db`** - База данных Optuna с результатами
- **`configs/best_config.yaml`** - Лучшая найденная конфигурация

### Просмотр результатов
```bash
# Запуск Optuna dashboard для анализа
optuna-dashboard sqlite:///outputs/studies/[study_name].db
# Открыть http://127.0.0.1:8080 в браузере
```
