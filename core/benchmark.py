"""
Benchmark loader for performance comparison.
"""

from pathlib import Path
import pandas as pd
import logging

# Import both collectors
from core.data.collectors import (
    YahooCollector, 
    FactSetCollector, 
    FACTSET_AVAILABLE
)

logger = logging.getLogger(__name__)

class BenchmarkLoader:
    """
    Loads and manages benchmark data for performance comparison.
    """
    
    BENCHMARK_MAP = {
        'SPY': 'S&P 500',
        'VT': 'MSCI World',
        'AGG': 'US Aggregate Bonds',
        'IEF': 'US 7-10Y Treasury',
        'TLT': 'US 20Y+ Treasury',
        '^GSPC': 'S&P 500 Index',
        '^DJI': 'Dow Jones',
        '^IXIC': 'NASDAQ'
    }
    
    def __init__(self, cache_dir: str = 'Dataset', use_factset: bool = False):
        """
        Initialize benchmark loader.
        
        Args:
            cache_dir: Directory to cache benchmark data
            use_factset: If True, use FactSet instead of Yahoo
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True, parents=True)
        
        # Choose data source
        if use_factset and FACTSET_AVAILABLE:
            logger.info("Using FactSet for benchmark data")
            self.collector = FactSetCollector()
            self.use_factset = True
        else:
            if use_factset and not FACTSET_AVAILABLE:
                logger.warning("FactSet requested but not available, using Yahoo")
            logger.info("Using Yahoo Finance for benchmark data")
            self.collector = YahooCollector()
            self.use_factset = False
    
    def load_benchmark(
        self,
        ticker: str,
        start_date: str,
        end_date: str,
        initial_value: float = 100.0
    ) -> pd.DataFrame:
        """
        Load benchmark data and normalize to starting value.
        
        Args:
            ticker: Benchmark ticker (e.g., 'SPY', 'VT')
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            initial_value: Starting value for normalization (default 100)
            
        Returns:
            DataFrame with Date index and TotalValue column (normalized)
        """
        df = self._get_data(ticker, start_date, end_date)
        return self._normalize_data(df, initial_value)

    def _get_data(self, ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Get data from cache or download it."""
        cache_file = self.cache_dir / f"{ticker}_benchmark.csv"
        
        if cache_file.exists():
            try:
                df = pd.read_csv(cache_file, parse_dates=['Date'], index_col='Date')
                start_ts = pd.Timestamp(start_date)
                end_ts = pd.Timestamp(end_date)
                
                # Check if cache covers the requested range
                if not df.empty and df.index.min() <= start_ts and df.index.max() >= end_ts:
                    mask = (df.index >= start_date) & (df.index <= end_date)
                    return df.loc[mask].copy()
            except Exception as e:
                logger.warning(f"Error reading cache for {ticker}: {e}")

        logger.info(f"Downloading {ticker} benchmark data...")
        df = self._fetch_data(ticker, start_date, end_date)
        df.to_csv(cache_file)
        return df

    def _fetch_data(
        self,
        ticker: str,
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """Fetch benchmark data using configured collector."""
        try:
            data = self.collector.fetch_history([ticker], start_date, end_date)
            
            if data.empty:
                raise ValueError(f"No data returned for {ticker}")
            
            # Extract Close price from MultiIndex
            if isinstance(data.columns, pd.MultiIndex):
                if (ticker, 'Close') in data.columns:
                    close_data = data[(ticker, 'Close')]
                else:
                    # Try first ticker in columns
                    first_ticker = data.columns.get_level_values(0)[0]
                    close_data = data[(first_ticker, 'Close')]
            else:
                close_data = data['Close']
            
            # Create DataFrame
            df = pd.DataFrame({
                'Close': close_data
            })
            df.index.name = 'Date'
            
            return df
        
        except Exception as e:
            raise ValueError(f"Failed to download {ticker}: {e}")
    
    def _normalize_data(self, df: pd.DataFrame, initial_value: float) -> pd.DataFrame:
        """Normalize data to initial value."""
        if df.empty:
            return pd.DataFrame(columns=['Close', 'TotalValue'])

        if 'TotalValue' not in df.columns:
            df['TotalValue'] = (df['Close'] / df['Close'].iloc[0]) * initial_value
            
        return df[['Close', 'TotalValue']]
    
    def align_benchmark_to_portfolio(
        self,
        portfolio_equity: pd.DataFrame,
        benchmark_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Align benchmark dates to portfolio dates.
        
        Args:
            portfolio_equity: Portfolio equity curve (Date index, TotalValue column)
            benchmark_df: Benchmark data (Date index, TotalValue column)
            
        Returns:
            Aligned benchmark DataFrame with same dates as portfolio
        """
        # Ensure both have datetime index
        if not isinstance(portfolio_equity.index, pd.DatetimeIndex):
            portfolio_equity.index = pd.to_datetime(portfolio_equity.index)
        if not isinstance(benchmark_df.index, pd.DatetimeIndex):
            benchmark_df.index = pd.to_datetime(benchmark_df.index)
        
        # Reindex benchmark to portfolio dates, forward-fill missing values
        aligned = benchmark_df.reindex(portfolio_equity.index, method='ffill')
        
        # Renormalize to match portfolio starting value
        if not aligned.empty and 'TotalValue' in aligned.columns:
            portfolio_start = portfolio_equity['TotalValue'].iloc[0]
            benchmark_start = aligned['TotalValue'].iloc[0]
            
            if pd.notna(benchmark_start) and benchmark_start != 0:
                aligned['TotalValue'] = (aligned['TotalValue'] / benchmark_start) * portfolio_start
        
        return aligned