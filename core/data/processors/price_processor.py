"""
Price Data Processor.
Handles cleaning, resampling, and validation of price data.
"""

import pandas as pd
import numpy as np
import logging
from typing import Optional, List
from .base_processor import BaseProcessor

logger = logging.getLogger(__name__)

class PriceProcessor(BaseProcessor):
    """
    Processes raw price data (OHLCV).
    """
    
    def __init__(self, fill_method: str = 'ffill'):
        self.fill_method = fill_method

    def process(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Standardize and clean price data.
        
        Expects yfinance multi-index format: (Ticker, OHLCV) or single level if flattened.
        """
        if data.empty:
            return data

        df = data.copy()
        
        # 1. Handle Missing Values
        if self.fill_method == 'ffill':
            df = df.ffill()
        
        # 2. Ensure Datetime Index
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
            
        # 3. Sort Index
        df = df.sort_index()
        
        return df

    def calculate_returns(self, data: pd.DataFrame, period: int = 1) -> pd.DataFrame:
        """
        Calculate percentage returns.
        """
        return data.pct_change(period)

    def resample(self, data: pd.DataFrame, rule: str = 'W-FRI') -> pd.DataFrame:
        """
        Resample data to a different frequency (e.g., Weekly).
        
        Args:
            data: DataFrame with DatetimeIndex.
            rule: Resampling rule (e.g., 'W-FRI' for weekly ending Friday).
            
        Returns:
            Resampled DataFrame.
        """
        # Logic depends on whether it's a MultiIndex (Ticker, OHLCV) or just OHLCV
        # For simplicity, assuming we are resampling Close prices or similar.
        # If full OHLCV, we need aggregation dict: Open: first, High: max, Low: min, Close: last, Vol: sum
        
        # TODO: Implement robust OHLCV resampling for MultiIndex
        # For now, simple last() for Close prices
        return data.resample(rule).last()
