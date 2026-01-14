# Очищенная структура данных

> [Legacy] Документ относится к старой per-symbol/day структуре. Для актуальной помесячной схемы см. `docs/data_structure.md`.

## Обзор

Папка `data_clean` содержит оптимизированную версию исходных данных с улучшенной организацией:
- **Один файл на день**: вместо множественных parquet файлов за день
- **Без дубликатов**: удалены повторяющиеся timestamp
- **Консистентные данные**: все записи отсортированы по времени

По умолчанию приложение читает `data_downloaded/` (и `data_optimized/`, если она существует рядом).
Чтобы использовать `data_clean/`, укажите `data_dir: "data_clean"` в `configs/main_2024.yaml`
или сделайте симлинк/переименование `data_clean -> data_optimized`.

## 📁 Структура папки `data_clean`

```
data_clean/
├── ТОРГОВАЯ_ПАРА/                    # Например: BTCUSDT, ETHUSDT
│   ├── year=YYYY/                    # Год (2022-2025)
│   │   ├── month=MM/                 # Месяц (1-12)
│   │   │   ├── day=DD/               # День (1-31)
│   │   │   │   └── data.parquet      # ⭐ ОДИН файл за день
│   │   │   └── day=DD+1/
│   │   │       └── data.parquet
│   │   └── month=MM+1/
│   └── year=YYYY+1/
├── .symbols_cache.json               # Копия кэша торговых пар
├── .gitkeep                         # Файл для git
└── ignore.txt                       # Файл игнорирования
```

## ✨ Преимущества очищенной структуры

| Аспект | Исходная структура | Очищенная структура |
|--------|-------------------|-------------------|
| **Файлов за день** | 2-6 parquet файлов | 1 файл `data.parquet` |
| **Дубликаты** | Есть (до 80% дубликатов) | Отсутствуют |
| **Порядок данных** | Может быть нарушен | Отсортировано по timestamp |
| **Размер** | ~3.2 GB | ~1.6 GB (экономия ~50%) |
| **Скорость загрузки** | Медленная (множественные файлы) | Быстрая (один файл) |

## 📊 Схема данных

Схема остается идентичной исходной:

| Столбец | Тип данных | Описание | Пример |
|---------|------------|----------|---------|
| `timestamp` | int64 | Временная метка в миллисекундах UTC | 1733011200000 |
| `symbol` | object (string) | Символ торговой пары | "BTCUSDT" |
| `open` | float64 | Цена открытия за 15-минутный период | 96484.0 |
| `high` | float64 | Максимальная цена за период | 96687.8 |
| `low` | float64 | Минимальная цена за период | 96412.5 |
| `close` | float64 | Цена закрытия за период | 96633.1 |
| `volume` | float64 | Объем торгов за период | 362.012 |
| `ts_ms` | int64 | Дублирует timestamp | 1733011200000 |

## 🔧 Работа с очищенными данными

### Загрузка одного дня:
```python
import pandas as pd

# Загрузка данных за день (один файл)
df = pd.read_parquet('data_clean/BTCUSDT/year=2024/month=12/day=1/data.parquet')

# Данные уже очищены - дубликатов нет, сортировка есть
print(f"Записей: {len(df)}")
print(f"Уникальных timestamp: {df['timestamp'].nunique()}")
print(f"Данные отсортированы: {df['timestamp'].is_monotonic_increasing}")
```

### Оптимизированная функция загрузки:
```python
import pandas as pd
from pathlib import Path
from typing import Optional
import logging

def load_clean_data(symbol: str, start_date: str, end_date: str, 
                   data_dir: str = "data_clean") -> pd.DataFrame:
    """
    Загружает очищенные данные торговой пары за указанный период.
    
    Args:
        symbol: Символ торговой пары (например, 'BTCUSDT')
        start_date: Начальная дата в формате 'YYYY-MM-DD'
        end_date: Конечная дата в формате 'YYYY-MM-DD'
        data_dir: Путь к папке с очищенными данными
    
    Returns:
        DataFrame с данными за указанный период
    """
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    
    dfs = []
    current = start
    
    while current <= end:
        file_path = (Path(data_dir) / symbol / 
                    f"year={current.year}" / 
                    f"month={current.month}" / 
                    f"day={current.day}" / 
                    "data.parquet")
        
        if file_path.exists():
            try:
                df_day = pd.read_parquet(file_path)
                dfs.append(df_day)
            except Exception as e:
                logging.warning(f"Ошибка при загрузке {file_path}: {e}")
        
        current += pd.Timedelta(days=1)
    
    if dfs:
        return pd.concat(dfs, ignore_index=True)
    return pd.DataFrame()

# Пример использования
btc_data = load_clean_data('BTCUSDT', '2024-12-01', '2024-12-31')
print(f"Загружено {len(btc_data)} записей за декабрь 2024")
```

### Быстрая статистика по данным:
```python
def get_data_stats(symbol: str, data_dir: str = "data_clean") -> dict:
    """Получает статистику по данным торговой пары."""
    symbol_dir = Path(data_dir) / symbol
    
    if not symbol_dir.exists():
        return {"error": f"Пара {symbol} не найдена"}
    
    total_files = 0
    total_size = 0
    date_range = []
    
    for year_dir in symbol_dir.glob("year=*"):
        for month_dir in year_dir.glob("month=*"):
            for day_dir in month_dir.glob("day=*"):
                data_file = day_dir / "data.parquet"
                if data_file.exists():
                    total_files += 1
                    total_size += data_file.stat().st_size
                    
                    # Извлекаем дату
                    year = int(year_dir.name.split('=')[1])
                    month = int(month_dir.name.split('=')[1])
                    day = int(day_dir.name.split('=')[1])
                    date_range.append(pd.Timestamp(year, month, day))
    
    return {
        "symbol": symbol,
        "total_files": total_files,
        "total_size_mb": round(total_size / (1024*1024), 2),
        "date_range": f"{min(date_range).date()} - {max(date_range).date()}" if date_range else "Нет данных",
        "days_available": len(date_range)
    }

# Пример
stats = get_data_stats('BTCUSDT')
print(stats)
```

## 🚀 Производительность

### Сравнение скорости загрузки:
```python
import time

# Загрузка из исходной структуры (с дубликатами)
start_time = time.time()
df_original = pd.read_parquet('data_downloaded/BTCUSDT/year=2024/month=12/day=1/')
df_original = df_original.drop_duplicates(subset=['timestamp']).sort_values('timestamp')
original_time = time.time() - start_time

# Загрузка из очищенной структуры
start_time = time.time()
df_clean = pd.read_parquet('data_clean/BTCUSDT/year=2024/month=12/day=1/data.parquet')
clean_time = time.time() - start_time

print(f"Исходная структура: {original_time:.3f} сек")
print(f"Очищенная структура: {clean_time:.3f} сек")
print(f"Ускорение: {original_time/clean_time:.1f}x")
```

## 🔄 Обновление данных

### Если нужно добавить новые данные:
```python
# Скрипт для обновления отдельных дней
def update_day_data(symbol: str, year: int, month: int, day: int):
    """Обновляет данные за конкретный день."""
    from clean_data_structure import clean_day_data
    
    source_dir = Path("data") / symbol / f"year={year}" / f"month={month}" / f"day={day}"
    target_dir = Path("data_clean") / symbol / f"year={year}" / f"month={month}" / f"day={day}"
    
    if source_dir.exists():
        cleaned_df = clean_day_data(source_dir)
        if not cleaned_df.empty:
            target_dir.mkdir(parents=True, exist_ok=True)
            cleaned_df.to_parquet(target_dir / "data.parquet", index=False)
            print(f"Обновлены данные для {symbol} {year}-{month:02d}-{day:02d}")
```

## 📝 Рекомендации

### Для максимальной производительности:
1. **Используйте очищенную структуру** для всех аналитических задач
2. **Кэшируйте часто используемые данные** в памяти
3. **Фильтруйте на уровне файлов**, а не после загрузки
4. **Используйте векторизованные операции** pandas

### Для экономии места:
- Исходную папку `data` или `data_downloaded` можно архивировать после создания `data_clean`
- Регулярно проверяйте актуальность данных
- Используйте сжатие parquet для долгосрочного хранения

## 🆚 Миграция с исходной структуры

```python
# Быстрая замена в существующем коде
# Было:
# df = pd.read_parquet('data_downloaded/BTCUSDT/year=2024/month=12/day=1/')
# df = df.drop_duplicates(subset=['timestamp']).sort_values('timestamp')

# Стало:
df = pd.read_parquet('data_clean/BTCUSDT/year=2024/month=12/day=1/data.parquet')
```

Очищенная структура данных обеспечивает быструю и удобную работу с историческими данными торгов без необходимости дополнительной обработки. 
