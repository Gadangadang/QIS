"""
FRED Data Collector.
"""

from typing import List, Optional, Union
import pandas as pd
import pandas_datareader.data as web
from datetime import datetime
import logging
from .base_collector import BaseCollector

logger = logging.getLogger(__name__)

class FredCollector(BaseCollector):
    """
    Data collector implementation for FRED (Federal Reserve Economic Data).
    """

    def fetch_history(
        self, 
        tickers: List[str], 
        start_date: Union[str, datetime], 
        end_date: Optional[Union[str, datetime]] = None,
        interval: str = "1d"
    ) -> pd.DataFrame:
        """
        Fetch historical data from FRED.
        
        Args:
            tickers: List of FRED series IDs (e.g., ['GDP', 'UNRATE']).
            start_date: Start date.
            end_date: End date.
            interval: Not used for FRED (frequency is determined by the series).
            
        Returns:
            pd.DataFrame: DataFrame with columns as Series IDs.
        """
        if not tickers:
            logger.warning("No tickers provided to fetch_history")
            return pd.DataFrame()

        logger.info(f"Fetching FRED data for {len(tickers)} series from {start_date} to {end_date}")
        
        try:
            # pandas_datareader handles multiple tickers by returning a DataFrame with columns=tickers
            df = web.DataReader(tickers, 'fred', start_date, end_date)
            
            if df.empty:
                logger.warning("FRED returned empty DataFrame")
                return df
                
            return df

        except Exception as e:
            logger.error(f"Error fetching data from FRED: {e}")
            raise

    def fetch_latest(self, tickers: List[str]) -> pd.DataFrame:
        """
        Fetch latest data point.
        """
        # FRED data is often monthly/quarterly, so we need a longer lookback to find the "latest"
        end = datetime.now()
        start = end - pd.Timedelta(days=90) # Look back 3 months to be safe
        
        df = self.fetch_history(tickers, start_date=start, end_date=end)
        
        if df.empty:
            return df
            
        return df.iloc[[-1]]
