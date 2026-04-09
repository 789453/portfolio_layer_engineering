import pandas as pd
import pyarrow.parquet as pq
import pyarrow.dataset as ds
import os
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)

class ParquetDataLoader:
    """
    Efficient data loader utilizing pyarrow dataset for predicate pushdown (filters).
    Massively reduces IO and memory usage when reading large Parquet files.
    """
    def __init__(self, raw_dir: str, feature_dir: str):
        self.raw_dir = raw_dir
        self.feature_dir = feature_dir

    def load_cross_section(self, file_name: str, date: str, is_feature: bool = False, date_col: str = "trade_date") -> pd.DataFrame:
        """
        Loads data for a specific date using dataset filtering.
        """
        base_dir = self.feature_dir if is_feature else self.raw_dir
        file_path = os.path.join(base_dir, file_name)
        
        if not os.path.exists(file_path):
            logger.warning(f"File not found: {file_path}")
            return pd.DataFrame()

        try:
            # Using pyarrow dataset for efficient filtering
            dataset = ds.dataset(file_path, format="parquet")
            table = dataset.to_table(filter=ds.field(date_col) == date)
            df = table.to_pandas()
            return df
        except Exception as e:
            logger.error(f"Error loading {file_name} for date {date}: {e}")
            # Fallback for small files or unsupported types
            try:
                df = pd.read_parquet(file_path)
                if date_col in df.columns:
                    # handle integer dates
                    if pd.api.types.is_numeric_dtype(df[date_col]):
                        df[date_col] = df[date_col].astype(str)
                    return df[df[date_col] == date].copy()
                return df
            except Exception as e2:
                logger.error(f"Fallback loading failed for {file_name}: {e2}")
                return pd.DataFrame()

    def load_time_series(self, file_name: str, start_date: str, end_date: str, is_feature: bool = False, date_col: str = "trade_date") -> pd.DataFrame:
        """
        Loads data for a specific date range.
        """
        base_dir = self.feature_dir if is_feature else self.raw_dir
        file_path = os.path.join(base_dir, file_name)
        
        if not os.path.exists(file_path):
            return pd.DataFrame()

        try:
            dataset = ds.dataset(file_path, format="parquet")
            filter_expr = (ds.field(date_col) >= start_date) & (ds.field(date_col) <= end_date)
            table = dataset.to_table(filter=filter_expr)
            return table.to_pandas()
        except Exception as e:
            # Fallback
            df = pd.read_parquet(file_path)
            if date_col in df.columns:
                if pd.api.types.is_numeric_dtype(df[date_col]):
                    df[date_col] = df[date_col].astype(str)
                return df[(df[date_col] >= start_date) & (df[date_col] <= end_date)].copy()
            return df

    def load_full(self, file_name: str, is_feature: bool = False) -> pd.DataFrame:
        base_dir = self.feature_dir if is_feature else self.raw_dir
        file_path = os.path.join(base_dir, file_name)
        if not os.path.exists(file_path):
            return pd.DataFrame()
        return pd.read_parquet(file_path)
