import logging
import os

import numpy as np
import threading
import time
from pathlib import Path
import dask.dataframe as dd
import numpy as np
import pandas as pd  # type: ignore
import pyarrow as pa
import pyarrow.dataset as ds

from coint2.utils import empty_ddf, ensure_datetime_index
from coint2.utils.config import AppConfig
from coint2.utils.timing_utils import logged_time, time_block, log_progress

# Настройка логгера
logger = logging.getLogger(__name__)


def _scan_parquet_files(path: Path | str, glob: str = "*.parquet", max_shards: int | None = None) -> ds.Dataset:
    """Build pyarrow dataset from parquet files under ``path``."""
    base = Path(path)
    files = sorted(base.rglob(glob))
    if max_shards is not None:
        files = files[:max_shards]
    if not files:
        return ds.dataset(pa.table({}))
    return ds.dataset([str(f) for f in files], format="parquet", partitioning="hive")


def _dir_mtime_hash(path: Path) -> float:
    """Return hash value based on modification times of ``.parquet`` files."""
    mtimes = [f.stat().st_mtime for f in path.rglob("*.parquet")]
    return max(mtimes) if mtimes else 0.0


class DataHandler:
    """Utility class for loading local parquet price files."""

    def __init__(self, cfg: AppConfig, autorefresh: bool = True) -> None:
        self.data_dir = Path(cfg.data_dir)
        self.cache_dir = self.data_dir.parent / ".cache"  # Добавляем cache_dir
        self.fill_limit_pct = cfg.backtest.fill_limit_pct
        self.max_shards = cfg.max_shards
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.autorefresh = autorefresh
        self._all_data_cache: dict[str, tuple[dd.DataFrame, float]] = {}
        self._freq: str | None = None
        self._lock = threading.Lock()
        self.lookback_days: int = cfg.pair_selection.lookback_days

    @property
    def freq(self) -> str | None:
        """Return detected time step of the loaded data."""
        with self._lock:
            return self._freq

    def clear_cache(self) -> None:
        """Clears the in-memory Dask DataFrame cache."""
        with self._lock:
            self._all_data_cache.clear()
            self._freq = None

    @logged_time("get_all_symbols")
    def get_all_symbols(self) -> list[str]:
        """Return list of symbols based on data in optimized parquet files."""
        # Проверяем наличие оптимизированной директории
        optimized_dir = Path(self.data_dir.parent / "data_optimized")
        
        if optimized_dir.exists():
            # Используем оптимизированную структуру
            logger.info(f"Используем оптимизированную структуру данных: {optimized_dir}")
            
            try:
                # Загружаем один файл для получения списка символов
                # Находим первый доступный файл parquet
                parquet_files = []
                for year_dir in optimized_dir.iterdir():
                    if not year_dir.is_dir() or not year_dir.name.startswith("year="):
                        continue
                    
                    for month_dir in year_dir.iterdir():
                        if not month_dir.is_dir() or not month_dir.name.startswith("month="):
                            continue
                            
                        for file in month_dir.glob("*.parquet"):
                            parquet_files.append(file)
                            break
                        
                        if parquet_files:
                            break
                    
                    if parquet_files:
                        break
                
                if not parquet_files:
                    logger.warning(f"Не найдено parquet файлов в {optimized_dir}")
                    return []
                
                # Читаем первый файл и получаем уникальные символы
                import polars as pl
                df = pl.read_parquet(parquet_files[0])
                symbols = df["symbol"].unique().to_list()
                
                logger.info(f"Найдено {len(symbols)} символов в оптимизированной структуре: {symbols[:10]}{'...' if len(symbols) > 10 else ''}")
                return sorted(symbols)
                
            except Exception as e:
                logger.error(f"Ошибка при чтении оптимизированных данных: {e}")
                # Если произошла ошибка, пробуем использовать старую структуру
        
        # Используем старую структуру, если оптимизированная недоступна
        if not self.data_dir.exists():
            logger.warning(f"Data directory does not exist: {self.data_dir}")
            return []

        symbols = []
        with time_block("scanning symbol directories"):
            for p in self.data_dir.iterdir():
                if not p.is_dir():
                    continue
                if p.name.startswith("symbol="):
                    symbols.append(p.name.replace("symbol=", ""))
                else:
                    symbols.append(p.name)
        
        logger.info(f"Found {len(symbols)} symbols in legacy structure: {symbols[:10]}{'...' if len(symbols) > 10 else ''}")
        return sorted(symbols)

    @logged_time("load_full_dataset")
    def _load_full_dataset(self, start_date: pd.Timestamp = None, end_date: pd.Timestamp = None, symbols: list[str] = None) -> dd.DataFrame:
        """
        Загружает данные из оптимизированной или стандартной структуры parquet.
        
        Parameters
        ----------
        start_date : pd.Timestamp, optional
            Начальная дата для загрузки данных
        end_date : pd.Timestamp, optional
            Конечная дата для загрузки данных
        symbols : list[str], optional
            Список символов для фильтрации
        
        Returns
        -------
        dd.DataFrame
            Dask DataFrame с данными за указанный период
        """
        try:
            # Проверяем наличие оптимизированной директории
            optimized_dir = Path(self.data_dir.parent / "data_optimized")
            data_path = optimized_dir if optimized_dir.exists() else self.data_dir
            
            logger.info(f"Используем директорию данных: {data_path}")
            
            cols_to_load = ["timestamp", "symbol", "close"]
            filters = []
            
            # Добавляем фильтры по дате, если они указаны
            if start_date is not None:
                # Преобразуем в миллисекунды для фильтрации
                start_ts = int(start_date.timestamp() * 1000)
                logger.info(f"Фильтрация от даты: {start_date} (timestamp: {start_ts})")
                filters.append(("timestamp", ">=", start_ts))
                
            if end_date is not None:
                # Преобразуем в миллисекунды для фильтрации
                end_ts = int(end_date.timestamp() * 1000)
                logger.info(f"Фильтрация до даты: {end_date} (timestamp: {end_ts})")
                filters.append(("timestamp", "<=", end_ts))
            
            # Добавляем фильтры по символам, если указаны
            if symbols is not None and len(symbols) > 0:
                logger.info(f"Фильтрация по символам: {symbols[:10]}{'...' if len(symbols) > 10 else ''}")
                # Для оптимизированной структуры используем фильтр по столбцу symbol
                if optimized_dir.exists():
                    filters.append(("symbol", "in", symbols))
            
            logger.info(f"Загрузка данных с фильтрами: {filters if filters else 'без фильтров'}")

            # Используем dd.read_parquet с фильтрами для эффективной загрузки
            ddf = dd.read_parquet(
                data_path,
                engine="pyarrow",
                columns=cols_to_load,
                filters=filters if filters else None,
                gather_statistics=True,  # Важно для отсечения партиций (partition pruning)
                schema_overrides={
                    # Явно указываем типы для предотвращения ошибок типов
                    "timestamp": np.int64,
                    "close": np.float64,
                    "symbol": str
                }
            )

            # Репартиционируем для лучшего параллелизма и кешируем в памяти
            # npartitions можно настроить в зависимости от системы
            num_partitions = os.cpu_count() * 2
            ddf = ddf.repartition(npartitions=num_partitions).persist()

            logger.info(f"Dask dataframe создан и сохранен в памяти: {len(ddf.columns)} колонок, {ddf.npartitions} партиций.")

            return ddf

        except Exception as e:
            logger.error(f"Ошибка при загрузке датасета: {e}")
            # Возвращаем пустой DataFrame в случае ошибки
            return empty_ddf()

    @logged_time("load_all_data_for_period")
    def load_all_data_for_period(self, lookback_days: int | None = None, symbols: list[str] = None) -> pd.DataFrame:
        """Load close prices for all symbols for the specified or configured lookback period.
        
        Parameters
        ----------
        lookback_days : int | None, optional
            Number of days to look back. If None, uses self.lookback_days.
        symbols : list[str], optional
            List of symbols to filter by. If None, loads data for all symbols.
            
        Returns
        -------
        pd.DataFrame
            DataFrame with close prices for all symbols.
        """
        try:
            # Определяем временные рамки для загрузки
            days_back = lookback_days or self.lookback_days
            end_date = pd.Timestamp.now()
            start_date = end_date - pd.Timedelta(days=days_back)
            
            logger.info(f"Загрузка данных за период: {start_date} - {end_date} ({days_back} дней)")
            
            with time_block("loading filtered dataset"):
                # Используем оптимизированный метод с фильтрацией по датам и символам
                ddf = self._load_full_dataset(start_date=start_date, end_date=end_date, symbols=symbols)

            # Проверка на пустой DataFrame
            if not ddf.columns.tolist():
                logger.warning("Empty dataset - no columns found")
                return pd.DataFrame()

            with time_block("computing dataset and preparing data"):
                # Приводим ddf['timestamp'] к datetime если это еще не сделано
                if not pd.api.types.is_datetime64_any_dtype(ddf["timestamp"]):
                    ddf["timestamp"] = dd.to_datetime(ddf["timestamp"], unit="ms")

                # Вычисляем полные данные
                all_data = ddf.compute()
                logger.info(f"Computed dataset: {len(all_data):,} rows, {len(all_data.columns)} columns")
                
                if all_data.empty:
                    # Возвращаем пустой DataFrame с правильными колонками и именем индекса
                    logger.warning("No data in specified time range")
                    return pd.DataFrame(columns=["timestamp", "symbol", "close"], index=pd.DatetimeIndex([], name="timestamp"))

                # Преобразуем timestamp в datetime
                all_data["timestamp"] = pd.to_datetime(all_data["timestamp"], unit="ms")

            # Пивотируем данные
            with time_block("pivoting data"):
                result = all_data.pivot_table(
                    index="timestamp", columns="symbol", values="close"
                )
                logger.info(f"Pivoted data: {len(result)} rows, {len(result.columns)} symbols")

            # Обрабатываем missing data
            with time_block("handling missing data"):
                initial_symbols = len(result.columns)
                
                # Применяем fill_limit_pct
                if hasattr(self, 'fill_limit_pct'):
                    fill_limit = int(len(result) * self.fill_limit_pct)
                    result = result.fillna(method='ffill', limit=fill_limit)
                    result = result.fillna(method='bfill', limit=fill_limit)
                
                # Удаляем столбцы с слишком большим количеством NaN
                nan_threshold = 0.5  # Remove symbols with >50% NaN
                clean_result = result.dropna(axis=1, thresh=int(len(result) * (1 - nan_threshold)))
                
                removed_symbols = initial_symbols - len(clean_result.columns)
                if removed_symbols > 0:
                    logger.info(f"Removed {removed_symbols} symbols with >50% missing data")
                
                logger.info(f"Final result: {len(clean_result)} rows, {len(clean_result.columns)} symbols")

            return clean_result

        except Exception as e:
            logger.error(f"Error in load_all_data_for_period: {e}")
            return pd.DataFrame()

    def load_pair_data(
        self,
        symbol1: str,
        symbol2: str,
        start_date: pd.Timestamp,
        end_date: pd.Timestamp,
    ) -> pd.DataFrame:
        """Load and align data for two symbols within the given date range."""
        # Обеспечиваем, что даты в наивном формате (без timezone)
        if start_date.tzinfo is not None:
            logger.debug(f"Удаляю timezone из start_date: {start_date}")
            start_date = start_date.tz_localize(None)
        if end_date.tzinfo is not None:
            logger.debug(f"Удаляю timezone из end_date: {end_date}")
            end_date = end_date.tz_localize(None)
            
        logger.debug(
            f"Загрузка данных для пары {symbol1}-{symbol2} ({start_date} - {end_date})"
        )

        if not self.data_dir.exists():
            logger.warning(f"Директория данных {self.data_dir} не существует")
            return pd.DataFrame()

        start_ms = int(start_date.timestamp() * 1000)
        end_ms = int(end_date.timestamp() * 1000)
        
        # Сначала пробуем partition-filters (symbol, year, month, day)
        filters = [
            ("symbol", "in", [symbol1, symbol2]),
            ("year", ">=", start_date.year),
            ("year", "<=", end_date.year),
            ("month", ">=", start_date.month if start_date.year == end_date.year else 1),
            ("month", "<=", end_date.month if start_date.year == end_date.year else 12),
            ("timestamp", ">=", start_ms),
            ("timestamp", "<=", end_ms),
        ]

        try:
            logger.debug("Пробуем загрузку с partition-filters")
            ddf = dd.read_parquet(
                self.data_dir,
                engine="pyarrow",
                columns=["timestamp", "close", "symbol"],
                ignore_metadata_file=True,
                calculate_divisions=False,
                filters=filters,
                validate_schema=False,
            )
            pair_pdf = ddf.compute()
        except Exception as e:
            logger.debug(f"Ошибка при загрузке с partition-filters: {str(e)}")
            # Отваливаемся на «сырое» чтение
            try:
                logger.debug("Fallback: загрузка без partition-filters")
                basic_filters = [
                    ("symbol", "in", [symbol1, symbol2]),
                    ("timestamp", ">=", start_ms),
                    ("timestamp", "<=", end_ms),
                ]
                ddf = dd.read_parquet(
                    self.data_dir,
                    engine="pyarrow",
                    columns=["timestamp", "close", "symbol"],
                    ignore_metadata_file=True,
                    calculate_divisions=False,
                    filters=basic_filters,
                    validate_schema=False,
                )
                pair_pdf = ddf.compute()
            except Exception as e2:  # pragma: no cover - fallback rarely used
                logger.debug(f"Error loading pair data via Dask: {str(e2)}")
                try:
                    logger.debug("Fallback to pyarrow dataset scanning in load_pair_data")
                    dataset = _scan_parquet_files(self.data_dir, max_shards=self.max_shards)
                    arrow_filter = (
                        ds.field("symbol").isin([symbol1, symbol2])
                        & (ds.field("timestamp") >= pa.scalar(start_date))
                        & (ds.field("timestamp") <= pa.scalar(end_date))
                    )
                    table = dataset.to_table(
                        columns=["timestamp", "close", "symbol"],
                        filter=arrow_filter,
                    )
                    pair_pdf = table.to_pandas()
                except Exception as e3:  # pragma: no cover - fallback rarely used
                    logger.debug(f"Error loading pair data manually: {str(e3)}")
                    return pd.DataFrame()

        if pair_pdf.empty:
            logger.debug(f"Нет данных для пары {symbol1}-{symbol2}")
            return pd.DataFrame()
        pair_pdf["timestamp"] = pd.to_datetime(pair_pdf["timestamp"], unit="ms").dt.tz_localize(None)

        # Проверка на наличие дубликатов timestamp
        if pair_pdf.duplicated(subset=["timestamp", "symbol"]).any():
            logger.debug(f"Обнаружены дубликаты timestamp для пары {symbol1}-{symbol2}. Удаляем дубликаты.")
            pair_pdf = pair_pdf.drop_duplicates(subset=["timestamp", "symbol"])

        # Преобразуем в широкий формат (timestamp x symbols)
        wide_df = pair_pdf.pivot_table(index="timestamp", columns="symbol", values="close")

        # Проверка на наличие нужных столбцов
        if wide_df.empty or len(wide_df.columns) < 2:
            return pd.DataFrame()

        # Обработка пропущенных значений
        freq_val = pd.infer_freq(wide_df.index)
        with self._lock:
            self._freq = freq_val
        if freq_val:
            wide_df = wide_df.asfreq(freq_val)

        # рассчитываем максимальную длину подряд идущих NA по проценту
        limit = max(1, int(len(wide_df) * self.fill_limit_pct))

        wide_df = wide_df.ffill(limit=limit).bfill(limit=limit)


        # Возвращаем только нужные символы и удаляем строки с NA
        if symbol1 in wide_df.columns and symbol2 in wide_df.columns:
            return wide_df[[symbol1, symbol2]].dropna()
        else:
            return pd.DataFrame()

    def load_and_normalize_data(
        self, start_date: pd.Timestamp, end_date: pd.Timestamp
    ) -> pd.DataFrame:
        """
        Load and normalize data for all symbols within the given date range.
        
        Parameters
        ----------
        start_date : pd.Timestamp
            Начальная дата для загрузки данных
        end_date : pd.Timestamp
            Конечная дата для загрузки данных
            
        Returns
        -------
        pd.DataFrame
            Нормализованный DataFrame с ценами (начало = 100)
        """
        # Обеспечиваем, что даты в наивном формате (без timezone)
        if start_date.tzinfo is not None:
            logger.debug(f"Удаляю timezone из start_date: {start_date}")
            start_date = start_date.tz_localize(None)
        if end_date.tzinfo is not None:
            logger.debug(f"Удаляю timezone из end_date: {end_date}")
            end_date = end_date.tz_localize(None)
            
        logger.debug(f"Загрузка и нормализация данных за период {start_date} - {end_date}")
        
        start_time = time.time()
        data_df = self.preload_all_data(start_date, end_date)
        elapsed_time = time.time() - start_time
        logger.info(f"Данные загружены за {elapsed_time:.2f} секунд. Размер: {data_df.shape}")

        # Нормализация цен
        if not data_df.empty:
            logger.debug(
                f"Количество символов до нормализации: {len(data_df.columns)}"
            )

            numeric_cols = [
                c
                for c in data_df.columns
                if pd.api.types.is_numeric_dtype(data_df[c]) and c != "timestamp"
            ]
            if numeric_cols:
                for col in numeric_cols:
                    series = data_df[col]
                    first_val = series.loc[series.first_valid_index()] if series.first_valid_index() is not None else pd.NA
                    if pd.isna(first_val) or first_val == 0:
                        data_df[col] = 0.0
                    else:
                        data_df[col] = 100 * series / first_val

            logger.debug(
                f"Количество символов после нормализации: {len(data_df.columns)}"
            )
            
            # Проверяем наличие константных серий и серий с большим количеством пропусков
            valid_columns = []
            for column in data_df.columns:
                if pd.api.types.is_numeric_dtype(data_df[column]):
                    # Проверка на константность серии
                    if data_df[column].nunique() > 1:
                        # Проверка на слишком много пропусков (более 50%)
                        na_pct = data_df[column].isna().mean()
                        if na_pct < 0.5:
                            valid_columns.append(column)
                            
            # Оставляем только валидные столбцы
            if valid_columns:
                logger.debug(
                    f"Отфильтровано {len(data_df.columns) - len(valid_columns)} "
                    "константных или разреженных серий"
                )
                data_df = data_df[valid_columns]

        return data_df

    def preload_all_data(
        self, start_date: pd.Timestamp, end_date: pd.Timestamp
    ) -> pd.DataFrame:
        """Loads and pivots all data for a given wide date range."""
        # Обеспечиваем, что даты в наивном формате (без timezone)
        if start_date.tzinfo is not None:
            logger.debug(f"Удаляю timezone из start_date в preload_all_data: {start_date}")
            start_date = start_date.tz_localize(None)
        if end_date.tzinfo is not None:
            logger.debug(f"Удаляю timezone из end_date в preload_all_data: {end_date}")
            end_date = end_date.tz_localize(None)
            
        logger.debug(f"Предзагрузка всех данных за период {start_date} - {end_date}")
            
        if not self.data_dir.exists():
            logger.warning(f"Директория данных {self.data_dir} не существует")
            return pd.DataFrame()

        # Преобразуем даты в timestamp формат для фильтрации
        start_ts = int(start_date.timestamp() * 1000)
        end_ts = int(end_date.timestamp() * 1000)
        logger.debug(f"Фильтрация по timestamp: {start_ts} - {end_ts}")
        
        # Сначала пробуем partition-filters (year, month, day)
        partition_filters = [
            ("year", ">=", start_date.year),
            ("year", "<=", end_date.year),
            ("month", ">=", start_date.month if start_date.year == end_date.year else 1),
            ("month", "<=", end_date.month if start_date.year == end_date.year else 12),
            ("timestamp", ">=", start_ts),
            ("timestamp", "<=", end_ts),
        ]

        try:
            logger.debug("Пробуем загрузку с partition-filters")
            ddf = dd.read_parquet(
                self.data_dir,
                engine="pyarrow",
                columns=["timestamp", "close", "symbol"],
                ignore_metadata_file=True,
                calculate_divisions=False,
                filters=partition_filters,
                validate_schema=False,
            )

            # Преобразуем в pandas для фильтрации
            all_data = ddf.compute()
            logger.debug(f"Загружено записей всего: {len(all_data)}")
        except Exception as e:
            logger.debug(f"Ошибка при загрузке с partition-filters: {str(e)}")
            # Отваливаемся на «сырое» чтение
            try:
                # Для партиционированных данных лучше загружать без фильтров и фильтровать после
                logger.debug("Fallback: загрузка всех данных без фильтров...")
                ddf = dd.read_parquet(
                    self.data_dir,
                    engine="pyarrow",
                    columns=["timestamp", "close", "symbol"],
                    ignore_metadata_file=True,
                    calculate_divisions=False,
                    validate_schema=False,
                )

                # Преобразуем в pandas для фильтрации
                all_data = ddf.compute()
                logger.debug(f"Загружено записей всего: {len(all_data)}")
            except Exception as e2:
                logger.error(f"Error loading data via Dask: {str(e2)}")

                try:
                    logger.debug("Fallback to pyarrow dataset scanning in preload_all_data")
                    dataset = _scan_parquet_files(self.data_dir, max_shards=self.max_shards)
                    table = dataset.to_table(columns=["timestamp", "close", "symbol"])
                    all_data = table.to_pandas()
                    if all_data.empty:
                        return pd.DataFrame()
                except Exception as e3:
                    logger.error(f"Error loading data manually: {str(e3)}")
                    return pd.DataFrame()

        logger.debug(f"Тип all_data['timestamp']: {all_data['timestamp'].dtype}")
        logger.debug(f"Примеры значений timestamp: {all_data['timestamp'].head(5).tolist()}")
        logger.debug(f"Тип timestamp: {all_data['timestamp'].dtype}")
        
        # Проверяем и преобразуем timestamp в числовой тип
        if np.issubdtype(all_data['timestamp'].dtype, np.datetime64):
            logger.debug('Преобразую timestamp из datetime64[ns] в int/ms')
            all_data['timestamp'] = all_data['timestamp'].astype('int64') // 10**6
            logger.debug(f"Преобразовано. Примеры: {all_data['timestamp'].head(5).tolist()}")
        elif all_data['timestamp'].dtype == 'object' or str(all_data['timestamp'].dtype).startswith('string') or 'str' in str(all_data['timestamp'].dtype).lower():
            logger.warning('Обнаружен строковый тип timestamp! Преобразую в int64')
            # Пробуем преобразовать строки в числа
            try:
                all_data['timestamp'] = pd.to_numeric(all_data['timestamp'], errors='coerce')
                # Заполняем NaN значения, если они появились при преобразовании
                if all_data['timestamp'].isna().any():
                    logger.warning(f"При преобразовании timestamp появились NaN значения: {all_data['timestamp'].isna().sum()} из {len(all_data)}")
                    # Удаляем строки с NaN в timestamp, так как они не могут быть корректно обработаны
                    all_data = all_data.dropna(subset=['timestamp'])
                logger.debug(f"Преобразовано в числовой тип. Примеры: {all_data['timestamp'].head(5).tolist()}")
            except Exception as e:
                logger.error(f"Ошибка при преобразовании timestamp из строки в число: {str(e)}")
        
        if all_data.empty:
            logger.warning(f"Нет данных в директории {self.data_dir}")
            return pd.DataFrame()
        
        # Фильтруем по времени в pandas
        mask = (all_data["timestamp"] >= start_ts) & (all_data["timestamp"] <= end_ts)
        filtered_df = all_data[mask]
        logger.debug(f"Записей после фильтрации по времени: {len(filtered_df)}")
        
        if filtered_df.empty:
            logger.warning(f"No data found between {start_date} and {end_date}")
            logger.debug(f"Доступный диапазон времени: {pd.to_datetime(all_data['timestamp'].min(), unit='ms')} - {pd.to_datetime(all_data['timestamp'].max(), unit='ms')}")
            return pd.DataFrame()

        # Преобразуем timestamp в datetime для индекса
        filtered_df["timestamp"] = pd.to_datetime(filtered_df["timestamp"], unit="ms")
        
        if filtered_df.duplicated(subset=["timestamp", "symbol"]).any():
            wide_pdf = filtered_df.pivot_table(
                index="timestamp",
                columns="symbol",
                values="close",
                aggfunc="last",
            )
        else:
            wide_pdf = filtered_df.pivot(
                index="timestamp",
                columns="symbol",
                values="close",
            )

        if wide_pdf.empty:
            logger.debug(f"Пустой pivot после преобразования")
            return pd.DataFrame()

        wide_pdf = ensure_datetime_index(wide_pdf)
        logger.debug(f"Итоговые данны: {wide_pdf.shape}, период: {wide_pdf.index.min()} - {wide_pdf.index.max()}")

        freq_val = pd.infer_freq(wide_pdf.index)
        with self._lock:
            self._freq = freq_val
        if freq_val:
            wide_pdf = wide_pdf.asfreq(freq_val)

        return wide_pdf


# New function for optimized dataset loading
import pandas as pd
import pyarrow.dataset as ds


def load_master_dataset(data_path: str, start_date: pd.Timestamp, end_date: pd.Timestamp) -> pd.DataFrame:
    """
    Загружает данные для указанного диапазона, используя Polars для надежной фильтрации
    и строгого контроля типов данных.
    
    Parameters
    ----------
    data_path : str
        Путь к директории с данными
    start_date : pd.Timestamp
        Начальная дата диапазона
    end_date : pd.Timestamp
        Конечная дата диапазона
        
    Returns
    -------
    pd.DataFrame
        DataFrame с данными за указанный период
    """
    print(f"⚙️  Загрузка данных за период: {start_date.date()} -> {end_date.date()}")

    # Проверяем наличие оптимизированной структуры данных
    import os
    from pathlib import Path
    import polars as pl
    
    data_path_obj = Path(data_path)
    optimized_dir = Path(data_path_obj.parent / "data_optimized")
    
    if optimized_dir.exists() and os.listdir(optimized_dir):
        print(f"ℹ️  Используем оптимизированную структуру данных: {optimized_dir}")
        data_path = str(optimized_dir)
    
    # Преобразуем даты в миллисекунды для фильтрации
    start_ts = int(start_date.timestamp() * 1000)
    end_ts = int(end_date.timestamp() * 1000)
    
    try:
        # Используем Polars для загрузки данных с фильтрацией
        # Сначала найдем все parquet файлы в нужных партициях
        parquet_files = []
        
        # Если используем оптимизированную структуру
        if str(data_path) == str(optimized_dir):
            # Фильтруем по year и month
            for year_dir in Path(data_path).glob("year=*"):
                year = int(year_dir.name.split('=')[1])
                
                # Пропускаем годы, которые точно не входят в диапазон
                if year < start_date.year or year > end_date.year:
                    continue
                    
                for month_dir in year_dir.glob("month=*"):
                    month = int(month_dir.name.split('=')[1])
                    
                    # Проверяем, входит ли месяц в диапазон
                    if year == start_date.year and month < start_date.month:
                        continue
                    if year == end_date.year and month > end_date.month:
                        continue
                        
                    # Добавляем все parquet файлы из этой директории
                    for file in month_dir.glob("*.parquet"):
                        parquet_files.append(str(file))
        else:
            # Для старой структуры используем glob
            for p in Path(data_path).glob("**/*.parquet"):
                parquet_files.append(str(p))
        
        if not parquet_files:
            print("⚠️  Не найдено parquet файлов по указанному пути и фильтрам.")
            return pd.DataFrame()
            
        print(f"📂 Найдено {len(parquet_files)} parquet файлов для обработки.")
        
        # Загружаем и фильтруем данные с помощью Polars
        # Используем LazyFrame для эффективной фильтрации
        ldf = pl.scan_parquet(parquet_files)
        
        # Проверяем схему данных
        print(f"📊 Схема данных: {ldf.schema}")
        
        # Фильтруем по timestamp
        filtered_ldf = ldf.filter(
            (pl.col("timestamp") >= start_ts) & 
            (pl.col("timestamp") <= end_ts)
        )
        
        # Выбираем нужные столбцы и собираем данные
        result = filtered_ldf.select(
            "timestamp", "symbol", "close"
        ).collect()
        
        # Проверяем результаты
        if result.height == 0:
            print("⚠️  После фильтрации данные не найдены.")
            return pd.DataFrame()
            
        print(f"✅ Загружено {result.height} записей с помощью Polars.")
        
        # Преобразуем timestamp в datetime для pandas
        result = result.with_columns(
            pl.col("timestamp").cast(pl.Int64).alias("timestamp_ms"),
            pl.col("timestamp").cast(pl.Datetime).alias("timestamp")
        )
        
        # Преобразуем в pandas DataFrame
        pandas_df = result.to_pandas()
        
        # Проверяем типы данных
        print(f"📊 Типы данных в pandas: {pandas_df.dtypes}")
        
        return pandas_df
        
    except Exception as e:
        print(f"❌ Ошибка при загрузке данных с помощью Polars: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame()
