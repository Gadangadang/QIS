"""
Parquet Storage Handler.
"""

import pandas as pd
import logging
from pathlib import Path
from typing import Union, Optional

logger = logging.getLogger(__name__)

class ParquetHandler:
    """
    Handles reading and writing DataFrames to Parquet files.
    """
    
    def __init__(self, base_path: Union[str, Path]):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)

    def save(self, data: pd.DataFrame, filename: str, partition_cols: Optional[list] = None):
        """
        Save DataFrame to Parquet.
        
        Args:
            data: DataFrame to save.
            filename: Name of the file (e.g., 'prices.parquet').
            partition_cols: Columns to partition by.
        """
        file_path = self.base_path / filename
        try:
            data.to_parquet(file_path, engine='pyarrow', compression='snappy', partition_cols=partition_cols)
            logger.info(f"Saved data to {file_path}")
        except Exception as e:
            logger.error(f"Failed to save data to {file_path}: {e}")
            raise

    def load(self, filename: str) -> pd.DataFrame:
        """
        Load DataFrame from Parquet.
        
        Args:
            filename: Name of the file.
            
        Returns:
            pd.DataFrame: Loaded data.
        """
        file_path = self.base_path / filename
        if not file_path.exists():
            logger.warning(f"File not found: {file_path}")
            return pd.DataFrame()
            
        try:
            return pd.read_parquet(file_path, engine='pyarrow')
        except Exception as e:
            logger.error(f"Failed to load data from {file_path}: {e}")
            raise
