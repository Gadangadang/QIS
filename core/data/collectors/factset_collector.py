"""
FactSet Data Collector.
Handles OHLCV data, benchmark total returns, and constituents.
"""

from typing import List, Optional, Union, Dict, Any
import pandas as pd
from datetime import datetime
import logging

# FactSet imports (only available in FactSet environment)
try:
    from fds.fpe.dates import TimeSeries
    from fds.fpe.universe import IdentifierUniverse, UnivLimit, ScreeningExpressionUniverse
    from fds.fpe.screening import Screen
    from fds.fpe.quant.company import Company
    FACTSET_AVAILABLE = True
except ImportError:
    FACTSET_AVAILABLE = False
    UnivLimit = None  # Placeholder when not available

logger = logging.getLogger(__name__)


# Benchmark Dictionary - Maps common names to FactSet identifiers
BENCHMARK_DICT = {
    'S&P 500': {
        'ohlcv': 'SPY-US',           # ETF for OHLCV
        'total_return': 'SP50.R',       # Total return index
        'constituents': UnivLimit.SP500 if FACTSET_AVAILABLE else 'SP50',
        'ticker': 'SPY'
    },
    'NASDAQ 100': {
        'ohlcv': 'QQQ-US',
        'total_return': 'NASDAQ100.R',
        'constituents':  UnivLimit.QQQ,
        'ticker': 'QQQ'
    },
    'Russell 2000': {
        'ohlcv': 'IWM-US',
        'total_return': 'R2000.R',
        'constituents': UnivLimit.R2000 if FACTSET_AVAILABLE else 'R2000',
        'ticker': 'IWM'
    },
    'Dow Jones': {
        'ohlcv': 'DIA-US',
        'total_return': 'DJIA.R',
        'constituents': UnivLimit.DJII_USA,
        'ticker': 'DIA'
    },
    'MSCI World': {
        'ohlcv': 'VT-US',
        'total_return':'990100' ,
        'constituents': UnivLimit.MSCI_WORLD,
        'ticker': 'VT'
    },
    'MSCI EAFE': {
        'ohlcv': 'EFA-US',
        'total_return': '990300',
        'constituents': UnivLimit.MSCI_EAFE,
        'ticker': 'EFA'
    },
}

# Ticker to Security mapping for OHLCV data
TICKER_TO_SECURITY = {
    # Major ETFs
    'SPY': 'SPY-US',
    'QQQ': 'QQQ-US',
    'IWM': 'IWM-US',
    'DIA': 'DIA-US',
    
    # Sector ETFs
    'XLE': 'XLE-US',  # Energy
    'XLF': 'XLF-US',  # Financials
    'XLK': 'XLK-US',  # Technology
    'XLV': 'XLV-US',  # Healthcare
    'XLI': 'XLI-US',  # Industrials
    'XLP': 'XLP-US',  # Consumer Staples
    'XLY': 'XLY-US',  # Consumer Discretionary
    'XLU': 'XLU-US',  # Utilities
    'XLB': 'XLB-US',  # Materials
    'XLRE': 'XLRE-US', # Real Estate
    'XLC': 'XLC-US',  # Communication Services
    
    # International
    'VT': 'VT-US',    # Total World
    'VEA': 'VEA-US',  # Developed Markets
    'VWO': 'VWO-US',  # Emerging Markets
    'EFA': 'EFA-US',  # EAFE
    
    # Fixed Income
    'TLT': 'TLT-US',
    'IEF': 'IEF-US',
    'AGG': 'AGG-US',
    'LQD': 'LQD-US',
    'HYG': 'HYG-US',
    
    # Commodities
    'GLD': 'GLD-US',
    'SLV': 'SLV-US',
    'USO': 'USO-US',
    'UNG': 'UNG-US',
}


class FactSetCollector:
    """
    Data collector implementation using FactSet.
    
    Supports three data types:
    1. 'ohlcv' - Price data for securities
    2. 'total_return' - Benchmark total return indices
    3. 'constituents' - Index constituent lists
    """
    
    def __init__(self):
        """Initialize FactSet collector."""
        if not FACTSET_AVAILABLE:
            raise ImportError(
                "FactSet modules not available. "
                "This collector only works in FactSet environment."
            )
        self.benchmark_dict = BENCHMARK_DICT
    
    def fetch(
        self,
        identifier: Union[str, Any],
        data_type: str,
        start_date: str,
        end_date: str,
        fields: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Core fetch method - handles all data types.
        
        Args:
            identifier: FactSet identifier (varies by data_type)
                - For 'ohlcv': Security ID like 'SPY-US'
                - For 'total_return': Index code like 'SP50.R'
                - For 'constituents': UnivLimit enum or universe code
            data_type: 'ohlcv', 'total_return', or 'constituents'
            start_date: Start date 'YYYY-MM-DD'
            end_date: End date 'YYYY-MM-DD'
            fields: Optional custom fields for constituents
        
        Returns:
            DataFrame (format depends on data_type)
        """
        if data_type == "total_return":
            return self._fetch_total_return(identifier, start_date, end_date)
        
        elif data_type == "constituents":
            return self._fetch_constituents(identifier, start_date, end_date, fields)
        
        elif data_type == "ohlcv":
            # Handle single security or list
            if isinstance(identifier, str):
                identifier = [identifier]
            return self._fetch_ohlcv(identifier, start_date, end_date)
        
        else:
            raise ValueError(f"Unknown data_type: {data_type}. Use 'ohlcv', 'total_return', or 'constituents'.")
    
    def fetch_history(
        self, 
        tickers: List[str], 
        start_date: Union[str, datetime], 
        end_date: Optional[Union[str, datetime]] = None,
        interval: str = "1d"
    ) -> pd.DataFrame:
        """
        Fetch OHLCV data for multiple tickers (main interface for compatibility).
        
        Args:
            tickers: List of ticker symbols (e.g., ['SPY', 'XLE', 'XLF'])
            start_date: '2023-01-01' or datetime object
            end_date: '2024-01-01' or None (defaults to today)
            interval: '1d' (currently only daily supported)
        
        Returns:
            DataFrame with MultiIndex columns (Ticker, PriceType)
        """
        if not tickers:
            logger.warning("No tickers provided to fetch_history")
            return pd.DataFrame()
        
        # Convert datetime to string if needed
        if isinstance(start_date, datetime):
            start_date = start_date.strftime('%Y-%m-%d')
        
        # Set end_date to today if not provided
        if end_date is None:
            end_date = datetime.today().strftime('%Y-%m-%d')
        elif isinstance(end_date, datetime):
            end_date = end_date.strftime('%Y-%m-%d')
        
        logger.info(f"Fetching OHLCV data for {len(tickers)} tickers from {start_date} to {end_date}")
        
        # Convert tickers to FactSet security identifiers
        factset_securities = [
            TICKER_TO_SECURITY.get(t.upper(), f"{t.upper()}-US" if '-' not in t else t)
            for t in tickers
        ]
        
        # Use fetch with ohlcv type
        df = self.fetch(factset_securities, 'ohlcv', start_date, end_date)
        
        # Map FactSet symbols back to original tickers
        if not df.empty and isinstance(df.columns, pd.MultiIndex):
            # Create reverse mapping
            reverse_map = {fs: orig for fs, orig in zip(factset_securities, tickers)}
            
            # Rename columns
            new_columns = []
            for ticker_fs, price_type in df.columns:
                original_ticker = reverse_map.get(ticker_fs, ticker_fs)
                new_columns.append((original_ticker, price_type))
            
            df.columns = pd.MultiIndex.from_tuples(new_columns)
        
        return df
    
    def _fetch_ohlcv(
        self,
        securities: List[str],
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """
        Fetch OHLCV price data for securities.
        
        Returns:
            DataFrame with MultiIndex columns (Security, PriceType)
        """
        try:
            # Create time series (daily frequency)
            ts = TimeSeries(
                start=start_date.replace('-', ''), 
                stop=end_date.replace('-', ''), 
                freq='D'
            )
            
            # Create universe
            univ = IdentifierUniverse(securities, time_series=ts)
            
            # FactSet formulas for split/dividend adjusted OHLCV
            formulas = [
                'P_PRICE_OPEN(0,0,USD)',   # Open
                'P_PRICE_HIGH(0,0,USD)',   # High
                'P_PRICE_LOW(0,0,USD)',    # Low
                'P_PRICE(0,USD)',          # Close
                'P_VOLUME(0)'              # Volume
            ]
            columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            
            # Fetch data
            screen = Screen(universe=univ, formulas=formulas, columns=columns)
            screen.calculate()
            df = screen.data.reset_index()
            
            if df.empty:
                logger.warning("FactSet returned empty DataFrame")
                return df
            
            # Pivot to MultiIndex columns: (Security, PriceType)
            df['date'] = pd.to_datetime(df['date'])
            df = df.set_index('date')
            df = df.pivot_table(index='date', columns='symbol', values=columns)
            
            # Reorder MultiIndex: (Security, PriceType)
            df = df.swaplevel(axis=1).sort_index(axis=1, level=0)
            
            # Remove timezone if present
            if df.index.tz is not None:
                df.index = df.index.tz_localize(None)
            
            return df
            
        except Exception as e:
            logger.error(f"Error fetching OHLCV data from FactSet: {e}")
            raise
    
    def _fetch_total_return(
        self,
        index_code: str,
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """
        Fetch total return data for a benchmark index.
        
        Args:
            index_code: FactSet index code (e.g., 'SP50.R')
            start_date: Start date 'YYYY-MM-DD'
            end_date: End date 'YYYY-MM-DD'
        
        Returns:
            DataFrame with total return series (single column)
        """
        try:
            logger.info(f"Fetching total return for {index_code}")
            
            # Use Company API for total returns
            company = Company(index_code, start=start_date, stop=end_date)
            
            # Get period returns
            returns = company.period_returns[index_code]
            
            if returns is None or returns.empty:
                logger.warning(f"No total return data for {index_code}")
                return pd.DataFrame()
            
            # Convert to DataFrame
            df = pd.DataFrame({index_code: returns})
            
            # Ensure DatetimeIndex and timezone-naive
            df.index = pd.to_datetime(df.index)
            if df.index.tz is not None:
                df.index = df.index.tz_localize(None)
            
            return df
            
        except Exception as e:
            logger.error(f"Error fetching total return for {index_code}: {e}")
            raise
    
    def _fetch_constituents(
        self,
        universe_id: Union[str, Any],
        start_date: str,
        end_date: str,
        fields: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Fetch constituent list for an index.
        
        Args:
            universe_id: UnivLimit enum, universe code string, or universe object
            start_date: Start date 'YYYY-MM-DD'
            end_date: End date 'YYYY-MM-DD'
            fields: Custom FactSet fields to fetch
        
        Returns:
            DataFrame with constituent data
        """
        try:
            # Create time series (monthly for constituent changes)
            ts = TimeSeries(
                start=start_date.replace('-', ''),
                stop=end_date.replace('-', ''),
                freq='M'
            )
            
            # Create universe - handle different input types
            # Check if it's already a universe object
            if hasattr(universe_id, 'calculate') and hasattr(universe_id, 'constituents'):
                # Already a universe object, use directly
                universe = universe_id
            elif isinstance(universe_id, str):
                # String screening expression - wrap it
                universe = ScreeningExpressionUniverse(
                    expression=f'FG_CONSTITUENTS({universe_id},0,CLOSE)=1',
                    time_series=ts
                )
            else:
                # Assume it's a UnivLimit enum or similar - wrap it
                universe = ScreeningExpressionUniverse(
                    expression=universe_id,
                    time_series=ts
                )
            
            # Default fields if not provided
            if fields is None:
                fields = [
                    'P_PRICE(0,USD)',
                    'FREF_MARKET_VALUE_COMPANY(0,USD)',
                    'FG_COMPANY_NAME'
                ]
            
            # Fetch data
            screen = Screen(universe=universe, formulas=fields)
            screen.calculate()
            
            return screen.data
            
        except Exception as e:
            logger.error(f"Error fetching constituents for {universe_id}: {e}")
            raise
    
    def fetch_latest(self, tickers: List[str]) -> pd.DataFrame:
        """
        Fetch latest data (last trading day).
        
        Args:
            tickers: List of ticker symbols
            
        Returns:
            DataFrame with same structure as fetch_history (single row)
        """
        end = datetime.now()
        start = end - pd.Timedelta(days=7)
        
        df = self.fetch_history(
            tickers, 
            start_date=start.strftime('%Y-%m-%d'), 
            end_date=end.strftime('%Y-%m-%d')
        )
        
        if df.empty:
            return df
            
        return df.iloc[[-1]]


# Helper function for benchmark fetching
def fetch_benchmark(
    collector: FactSetCollector,
    benchmark_key: str,
    data_type: str,
    start_date: str,
    end_date: str,
    fields: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Fetch benchmark data using the benchmark dictionary.
    
    Args:
        collector: FactSetCollector instance
        benchmark_key: Key from BENCHMARK_DICT (e.g., 'S&P 500')
        data_type: 'ohlcv', 'total_return', or 'constituents'
        start_date: Start date 'YYYY-MM-DD'
        end_date: End date 'YYYY-MM-DD'
        fields: Optional fields for constituents
    
    Returns:
        DataFrame with requested data
    
    Examples:
        # Get S&P 500 total return
        sp500_tr = fetch_benchmark(collector, 'S&P 500', 'total_return', '2023-01-01', '2023-12-31')
        
        # Get S&P 500 constituents
        sp500_const = fetch_benchmark(collector, 'S&P 500', 'constituents', '2023-01-01', '2023-01-31')
        
        # Get SPY OHLCV data
        spy_ohlcv = fetch_benchmark(collector, 'S&P 500', 'ohlcv', '2023-01-01', '2023-12-31')
    """
    if benchmark_key not in BENCHMARK_DICT:
        raise ValueError(f"Unknown benchmark: {benchmark_key}. Available: {list(BENCHMARK_DICT.keys())}")
    
    info = BENCHMARK_DICT[benchmark_key]
    
    if data_type not in info:
        raise ValueError(f"Data type '{data_type}' not available for {benchmark_key}. Available: {list(info.keys())}")
    
    identifier = info[data_type]
    
    return collector.fetch(identifier, data_type, start_date, end_date, fields)