"""
Facset Data Collector.
"""

from typing import List, Optional, Union, Dict
import pandas as pd
import yfinance as yf
from datetime import datetime
import logging
from .base_collector import BaseCollector

# Configure logging
logger = logging.getLogger(__name__)

class YahooCollector(BaseCollector):
    """
    Data collector implementation using yfinance.
    """

    def fetch_history(
        self, 
        tickers: List[str], 
        start_date: Union[str, datetime], 
        end_date: Optional[Union[str, datetime]] = None,
        interval: str = "1d"
    ) -> pd.DataFrame:
        """
        Fetch historical data using yfinance.
        
        Args:
            tickers: List of ticker symbols.
            start_date: Start date.
            end_date: End date.
            interval: Data frequency.
            
        Returns:
            pd.DataFrame: DataFrame with columns MultiIndex (Price, Ticker).
        """
        if not tickers:
            logger.warning("No tickers provided to fetch_history")
            return pd.DataFrame()

        # yfinance expects space-separated string for multiple tickers
        tickers_str = " ".join(tickers)
        
        logger.info(f"Fetching {interval} data for {len(tickers)} tickers from {start_date} to {end_date}")
        
        try:
            # auto_adjust=True handles splits and dividends (Close becomes Adj Close)
            df = yf.download(
                tickers=tickers_str,
                start=start_date,
                end=end_date,
                interval=interval,
                group_by='ticker',
                auto_adjust=True,
                prepost=False,
                threads=True,
                progress=False
            )
            
            if df.empty:
                logger.warning("yfinance returned empty DataFrame")
                return df

            # If single ticker, yfinance doesn't return MultiIndex columns by default if group_by='ticker' isn't forced or handled carefully.
            # However, group_by='ticker' usually returns (Ticker, OHLCV).
            # Let's standardize the output format.
            
            # If we requested multiple tickers, it comes as (Ticker, OHLCV)
            # If we requested one ticker, it might come as (OHLCV) depending on version, 
            # but group_by='ticker' should force Ticker level.
            
            return df

        except Exception as e:
            logger.error(f"Error fetching data from Yahoo Finance: {e}")
            raise

    def fetch_latest(self, tickers: List[str]) -> pd.DataFrame:
        """
        Fetch latest data (1 day).
        """
        # For latest, we can just fetch 5 days and take the last row to be safe
        # or use yf.Ticker().info for real-time, but history is safer for consistency
        end = datetime.now()
        start = end - pd.Timedelta(days=7)
        
        df = self.fetch_history(tickers, start_date=start, end_date=end)
        
        if df.empty:
            return df
            
        return df.iloc[[-1]]
