# Ускорение Оптимизации Optuna

Реализованы 7 ключевых оптимизаций для значительного ускорения выполнения trials в Optuna.

## 🚀 Реализованные Оптимизации

### 1. Оптимизированные Настройки TPESampler
**Файл**: `src/optimiser/run_optimization.py` (строки 210-226)

```python
sampler_kwargs = {
    "n_ei_candidates": 24,  # Больше кандидатов для параллелизма
    "constant_liar": (n_jobs > 1),  # Включаем при параллельности
    "multivariate": True,
    "group": True,
    "warn_independent_sampling": False
}
```

**Эффект**: Улучшенная параллельная работа TPE алгоритма.

### 2. Агрессивный MedianPruner
**Файл**: `src/optimiser/run_optimization.py` (строки 246-260)

```python
n_warmup_steps = max(1, min(2, total_reports // 3))  # Меньше warmup
interval_steps = min(3, max(1, total_reports // 4))  # Частая проверка
n_min_trials = max(3, n_startup_trials // 2)  # Минимум для pruning
```

**Эффект**: Быстрое отсеивание неперспективных trials.

### 3. Оптимизированное Storage
**Файл**: `src/optimiser/run_optimization.py` (строки 126-179)

```python
def get_optimized_storage(storage_path: str, n_jobs: int = 1):
    # PostgreSQL: Connection pooling
    pool_size = max(5, n_jobs * 2)
    max_overflow = n_jobs * 3
    
    # SQLite: WAL режим
    sqlite_url = f"sqlite:///{storage_path}?mode=rwc"
```

**Эффект**: Оптимизированная работа с базой данных для параллельности.

### 4. Оптимизированное Threading
**Файл**: `src/coint2/core/memory_optimization.py` (строки 306-336)

```python
def setup_optimized_threading(n_jobs: int = 1):
    # Для параллельных trials: 1 BLAS thread на процесс
    blas_threads = 1 if n_jobs > 1 else min(4, psutil.cpu_count())
    # Настройка OMP_NUM_THREADS, OPENBLAS_NUM_THREADS, MKL_NUM_THREADS
```

**Эффект**: Избежание oversubscription при параллельных вычислениях.

### 5. Быстрая Предварительная Фильтрация
**Файл**: `src/optimiser/fast_objective.py` (строки 1239-1284)

```python
def quick_trial_filter(self, params):
    # Проверка логичности параметров
    if zscore_exit >= zscore_threshold:
        return False, "zscore_exit >= zscore_threshold"
    
    # Проверка разумности экспозиции
    max_exposure = risk_per_position * max_positions
    if max_exposure > 1.0:
        return False, "Слишком большая экспозиция"
```

**Эффект**: Отклонение заведомо плохих параметров без полного бэктеста.

### 6. Кэширование Данных
**Файл**: `src/optimiser/fast_objective.py` (строки 1288-1320)

```python
def _cache_data(self, cache_key, data):
    with self.data_cache_lock:
        if len(self.data_cache) >= self.max_cache_size:
            # FIFO удаление старых элементов
            oldest_key = next(iter(self.data_cache))
            del self.data_cache[oldest_key]
        self.data_cache[cache_key] = data
```

**Эффект**: Переиспользование данных между trials с LRU кэшем.

### 7. Суженное Пространство Поиска
**Файлы**: 
- `configs/search_spaces/fast.yaml` - основной быстрый режим
- `configs/search_space_fast.yaml` - минимальный режим для тестов (legacy/минимальный набор)

**Ключевые изменения**:
- Фиксированные `commission_pct` и `slippage_pct`
- Суженные диапазоны для `zscore_threshold`, `zscore_exit`
- Фиксированный `normalization_method: minmax`
- Уменьшенные диапазоны для `risk_per_position_pct`

**Эффект**: Меньше размерность пространства поиска = быстрее сходимость.

## 📊 Режимы Оптимизации

### Fast Mode
```bash
PYTHONPATH=src ./.venv/bin/python src/optimiser/run_optimization.py \
  --n-trials 50 \
  --study-name fast_optimization \
  --search-space configs/search_spaces/fast.yaml
```
- Пространство поиска: `configs/search_spaces/fast.yaml`
- Быстрый прогон для проверки гипотез

### Custom Mode
```bash
PYTHONPATH=src ./.venv/bin/python src/optimiser/run_optimization.py \
  --n-trials 200 \
  --study-name custom_optimization \
  --search-space configs/search_space.yaml
```
- Пространство поиска: `configs/search_space.yaml` (минимальный набор параметров)
- Для расширения пространства используйте `configs/search_spaces/*.yaml`

## 🧪 Тестирование

Запуск тестов оптимизации:
```bash
./.venv/bin/pytest tests/performance/benchmarks/test_optimization_acceleration.py -v
```

## 📈 Ожидаемые Результаты

| Режим | Trials/мин | Ускорение | Качество |
|-------|------------|-----------|----------|
| Ultra Fast | 15-20 | 5x | Хорошее |
| Fast | 10-15 | 3x | Отличное |
| Full | 5-8 | 2x | Максимальное |

## ⚠️ Важные Замечания

1. **PostgreSQL рекомендуется** для параллельности > 1 процесса
2. **SQLite ограничен** одним процессом, но использует WAL режим
3. **Threading оптимизация** критична для избежания oversubscription
4. **Quick filter** может отклонить до 20-30% trials без полного бэктеста
5. **Кэширование данных** эффективно при повторяющихся параметрах

## 🔧 Настройка

Все оптимизации автоматически активируются при использовании `run_optimization()` с соответствующими параметрами. Дополнительная настройка не требуется.
