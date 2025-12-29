"""
Base Data Collector Interface.
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Union
import pandas as pd
from datetime import datetime

class BaseCollector(ABC):
    """
    Abstract base class for all data collectors.
    Enforces a consistent interface for fetching market data.
    """

    @abstractmethod
    def fetch_history(
        self, 
        tickers: List[str], 
        start_date: Union[str, datetime], 
        end_date: Optional[Union[str, datetime]] = None,
        interval: str = "1d"
    ) -> pd.DataFrame:
        """
        Fetch historical data for a list of tickers.

        Args:
            tickers: List of ticker symbols (e.g., ['SPY', 'AAPL'])
            start_date: Start date (YYYY-MM-DD or datetime)
            end_date: End date (YYYY-MM-DD or datetime). Defaults to today.
            interval: Data frequency ('1d', '1wk', '1mo'). Defaults to '1d'.

        Returns:
            pd.DataFrame: Multi-index DataFrame with (Ticker, Date) or similar structure,
                          containing Open, High, Low, Close, Volume.
        """
        pass

    @abstractmethod
    def fetch_latest(self, tickers: List[str]) -> pd.DataFrame:
        """
        Fetch the most recent available data point.

        Args:
            tickers: List of ticker symbols.

        Returns:
            pd.DataFrame: Latest market data.
        """
        pass
