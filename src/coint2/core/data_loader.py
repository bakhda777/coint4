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
        """Return list of symbols based on partition directory names."""
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
        
        logger.info(f"Found {len(symbols)} symbols: {symbols[:10]}{'...' if len(symbols) > 10 else ''}")
        return sorted(symbols)

    @logged_time("load_full_dataset")
    def _load_full_dataset(self) -> dd.DataFrame:
        """
        Load the dataset for the configured period as a dask dataframe,
        using predicate pushdown for efficiency.
        """
        try:
            # Эти даты должны быть доступны в экземпляре DataHandler
            # и установлены во время инициализации.
            start_date = self.start_date
            end_date = self.end_date

            logger.info(f"Загрузка данных за расширенный период {start_date} - {end_date}")

            # dd.read_parquet лучше всего работает с np.datetime64 для фильтров
            start_ts = np.datetime64(start_date)
            end_ts = np.datetime64(end_date)

            logger.debug(f"Фильтрация по timestamp: {start_ts} - {end_ts}")

            cols_to_load = ["timestamp", "symbol", "close"]

            # Используем dd.read_parquet с фильтрами для эффективной загрузки
            ddf = dd.read_parquet(
                self.data_dir,
                engine="pyarrow",
                columns=cols_to_load,
                filters=[("timestamp", ">=", start_ts), ("timestamp", "<=", end_ts)],
                gather_statistics=True,  # Важно для отсечения партиций (partition pruning)
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
    def load_all_data_for_period(self, lookback_days: int | None = None) -> pd.DataFrame:
        """Load close prices for all symbols for the specified or configured lookback period.
        
        Parameters
        ----------
        lookback_days : int | None, optional
            Number of days to look back. If None, uses self.lookback_days.
            
        Returns
        -------
        pd.DataFrame
            DataFrame with close prices for all symbols.
        """
        try:
            with time_block("loading full dataset"):
                ddf = self._load_full_dataset()

            # Проверка на пустой DataFrame
            if not ddf.columns.tolist():
                logger.warning("Empty dataset - no columns found")
                return pd.DataFrame()

            with time_block("computing dataset and applying filters"):
                # Приводим ddf['timestamp'] к datetime если это еще не сделано
                if not pd.api.types.is_datetime64_any_dtype(ddf["timestamp"]):
                    ddf["timestamp"] = dd.to_datetime(ddf["timestamp"], unit="ms")

                # Вычисляем полные данные один раз
                all_data = ddf.compute()
                logger.info(f"Computed dataset: {len(all_data):,} rows, {len(all_data.columns)} columns")

                # Определяем временные рамки  
                days_back = lookback_days or self.lookback_days
                max_ts = all_data["timestamp"].max()
                start_ts = max_ts - pd.Timedelta(days=days_back)
                
                logger.info(f"Time range: {start_ts} to {max_ts} ({days_back} days)")

            # Фильтруем уже вычисленные данные
            with time_block("filtering data by time range"):
                filtered_df = all_data[all_data["timestamp"] >= start_ts].copy()
                logger.info(f"After time filtering: {len(filtered_df):,} rows")
                
                if filtered_df.empty:
                    # Возвращаем пустой DataFrame с правильными колонками и именем индекса
                    logger.warning("No data in specified time range")
                    return pd.DataFrame(columns=all_data.columns, index=pd.DatetimeIndex([], name="timestamp"))

                # Преобразуем timestamp в datetime
                filtered_df["timestamp"] = pd.to_datetime(filtered_df["timestamp"], unit="ms")

            # Пивотируем данные
            with time_block("pivoting data"):
                result = filtered_df.pivot_table(
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
        
        if np.issubdtype(all_data['timestamp'].dtype, np.datetime64):
            logger.debug('Преобразую timestamp из datetime64[ns] в int/ms')
            all_data['timestamp'] = all_data['timestamp'].astype('int64') // 10**6
            logger.debug(f"Преобразовано. Примеры: {all_data['timestamp'].head(5).tolist()}")
        
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
