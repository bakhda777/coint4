# Структура данных проекта

## Обзор

Проект содержит исторические данные торгов криптовалютных пар с биржи, организованные в формате Apache Parquet с иерархическим партиционированием по дате.

## 📁 Организация файлов

```
data/
├── ТОРГОВАЯ_ПАРА/                    # Например: BTCUSDT, ETHUSDT, DOGEUSDT
│   ├── year=YYYY/                    # Год (2022-2025)
│   │   ├── month=MM/                 # Месяц (1-12)
│   │   │   ├── day=DD/               # День (1-31)
│   │   │   │   ├── hash1.parquet     # Файлы данных за день
│   │   │   │   ├── hash2.parquet     
│   │   │   │   └── ...
│   │   │   └── day=DD+1/
│   │   └── month=MM+1/
│   └── year=YYYY+1/
├── .symbols_cache.json               # Кэш списка торговых пар
├── .gitkeep                         # Файл для git
└── ignore.txt                       # Файл игнорирования
```

### Принципы организации:
- **Hive-style партиционирование**: `year=YYYY/month=MM/day=DD/`
- **Один уровень = одна торговая пара**: каждая папка верхнего уровня - отдельная пара
- **Множественные файлы за день**: данные за один день могут быть разбиты на несколько parquet файлов

## 📊 Общая статистика

| Параметр | Значение |
|----------|----------|
| Количество торговых пар | 424 |
| Временной период | 2022-2025 годы |
| Общий размер данных | ~3.2 GB |
| Количество parquet файлов | 555,052 |
| Временной интервал | 15-минутные свечи |

## 📈 Схема данных в parquet файлах

Каждый parquet файл содержит следующие столбцы:

| Столбец | Тип данных | Описание | Пример |
|---------|------------|----------|---------|
| `timestamp` | int64 | Временная метка в миллисекундах UTC | 1733011200000 |
| `symbol` | object (string) | Символ торговой пары | "BTCUSDT" |
| `open` | float64 | Цена открытия за 15-минутный период | 96484.0 |
| `high` | float64 | Максимальная цена за период | 96687.8 |
| `low` | float64 | Минимальная цена за период | 96412.5 |
| `close` | float64 | Цена закрытия за период | 96633.1 |
| `volume` | float64 | Объем торгов за период | 362.012 |
| `ts_ms` | int64 | Дублирует timestamp (избыточный) | 1733011200000 |

### Качество данных:
- ✅ **Без пропущенных значений**: все столбцы заполнены
- ⚠️ **Дубликаты timestamp**: данные за день могут дублироваться в разных файлах
- ✅ **Консистентный формат**: все файлы имеют одинаковую схему

## ⏰ Временные характеристики

### Интервалы данных:
- **Базовый интервал**: 15 минут = 900 секунд
- **Записей за день**: 96 (теоретически) до 480+ (с дубликатами)
- **Начало дня**: 00:00:00 UTC
- **Конец дня**: 23:45:00 UTC

### Конвертация времени:
```python
import pandas as pd

# Из timestamp в datetime
datetime_obj = pd.to_datetime(timestamp, unit='ms')

# Из datetime в timestamp  
timestamp = int(datetime_obj.timestamp() * 1000)
```

## 💱 Категории торговых пар

### Основные категории:
- **Bitcoin пары**: BTCUSDT, BTCUSDC, BTCEUR, BTCBRL, BTCBRZ
- **Ethereum пары**: ETHUSDT, ETHUSDC, ETHEUR, ETHBTC
- **Крупные альткоины**: ADAUSDT, BNBUSDT, SOLUSDT, DOGEUSDT
- **DeFi токены**: UNIUSDT, LINKUSDT, AAVEUSDT, CRVUSDT
- **Мемкоины**: SHIBUSDT, FLOKIUSDT, BONKUSDT
- **Стейблкоины**: DAIUSDT, USDCUSDT

### Валюты котировки:
- **USDT** (Tether) - основная
- **USDC** (USD Coin)
- **EUR** (Евро)
- **BTC** (Bitcoin)
- **BRL** (Бразильский реал)
- **TRY** (Турецкая лира)

## 🔧 Работа с данными

### Загрузка одного дня:
```python
import pandas as pd

# Загрузка всех данных за день (все parquet файлы)
df = pd.read_parquet('data/BTCUSDT/year=2024/month=12/day=1/')

# Удаление дубликатов по timestamp
df = df.drop_duplicates(subset=['timestamp']).sort_values('timestamp')
```

### Загрузка диапазона дат:
```python
import pandas as pd
from pathlib import Path

def load_symbol_data(symbol, start_date, end_date):
    """Загружает данные торговой пары за указанный период"""
    dfs = []
    current = start_date
    
    while current <= end_date:
        path = f"data/{symbol}/year={current.year}/month={current.month}/day={current.day}/"
        if Path(path).exists():
            df_day = pd.read_parquet(path)
            dfs.append(df_day)
        current += pd.Timedelta(days=1)
    
    if dfs:
        df = pd.concat(dfs, ignore_index=True)
        return df.drop_duplicates(subset=['timestamp']).sort_values('timestamp')
    return pd.DataFrame()
```

### Пример использования:
```python
# Загрузка данных BTCUSDT за декабрь 2024
start = pd.Timestamp('2024-12-01')
end = pd.Timestamp('2024-12-31')
btc_data = load_symbol_data('BTCUSDT', start, end)

print(f"Загружено {len(btc_data)} записей")
print(f"Период: {pd.to_datetime(btc_data['timestamp'].min(), unit='ms')} - {pd.to_datetime(btc_data['timestamp'].max(), unit='ms')}")
```

## 📝 Примечания для разработчиков

### Важные моменты:
1. **Дубликаты**: Всегда удаляйте дубликаты по timestamp при загрузке дня
2. **Часовой пояс**: Все временные метки в UTC
3. **Размер**: При загрузке больших периодов учитывайте объем данных
4. **Производительность**: Используйте фильтрацию на уровне parquet для оптимизации

### Рекомендации:
- Используйте `pandas.read_parquet()` с параметрами фильтрации
- Для больших объемов данных рассмотрите использование `dask`
- Кэшируйте обработанные данные для повторного использования
- Проверяйте целостность данных после загрузки

### Зависимости:
```python
pandas>=1.5.0
pyarrow>=8.0.0  # Для работы с parquet
```

## 📞 Поддержка

При возникновении вопросов по структуре данных:
1. Проверьте файл `.symbols_cache.json` для списка доступных пар
2. Используйте `pandas.read_parquet().info()` для анализа схемы файла
3. Проверьте наличие данных за конкретные даты перед загрузкой 