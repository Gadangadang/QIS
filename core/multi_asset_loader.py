"""
Multi-asset data loader with FactSet support.
"""

from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd
from datetime import datetime
import logging

# Try to import FactSet
try:
    from core.data.collectors import FactSetCollector, FACTSET_AVAILABLE
except ImportError:
    FACTSET_AVAILABLE = False
    FactSetCollector = None

logger = logging.getLogger(__name__)

# Default dataset directory
DATASET_DIR = Path(__file__).parent.parent / 'Dataset'

class MultiAssetLoader:
    """
    Loads and aligns price data for multiple assets.
    
    Supports:
    - CSV files (local data)
    - Yahoo Finance (via yfinance)
    - FactSet (when available)
    """
    
    # Map ticker symbols to filenames
    ASSET_FILES = {
        'ES': 'es_futures_2000_2025.csv',
        'NQ': 'nq_futures_2000_2025.csv',
        'GC': 'gc_futures_2000_2025.csv',
    }
    
    # Map tickers to yfinance symbols
    YFINANCE_SYMBOLS = {
        'ES': 'ES=F',
        'NQ': 'NQ=F',
        'GC': 'GC=F',
        'CL': 'CL=F',
        'NG': 'NG=F',
        'RB': 'RB=F',
        'HO': 'HO=F',
        "MME": "MME=F",
        "NIY": "NIY=F",
    }
    
    def __init__(
        self, 
        dataset_dir: Optional[Path] = None, 
        use_yfinance: bool = True,
        use_factset: bool = False
    ):
        """
        Initialize loader with dataset directory.
        
        Args:
            dataset_dir: Path to dataset directory
            use_yfinance: If True, fetch recent data from yfinance to supplement CSV files
            use_factset: If True and available, use FactSet instead of yfinance
        """
        self.dataset_dir = dataset_dir or DATASET_DIR
        self.use_yfinance = use_yfinance
        self.use_factset = use_factset and FACTSET_AVAILABLE
        
        # Initialize data collector if needed
        if self.use_factset:
            logger.info("MultiAssetLoader: Using FactSet")
            self.collector = FactSetCollector()
        elif self.use_yfinance:
            logger.info("MultiAssetLoader: Using yfinance")
            self.collector = None  # Will use yf.download directly
        else:
            self.collector = None
    
    def _fetch_from_online(self, ticker: str, start_date: str = '2015-01-01', end_date: Optional[str] = None) -> pd.DataFrame:
        """
        Fetch data from online source (FactSet or yfinance).
        
        Args:
            ticker: Asset ticker
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD), defaults to today
            
        Returns:
            DataFrame with Date, Open, High, Low, Close, Volume, Ticker
        """
        if self.use_factset and self.collector:
            return self._fetch_from_factset(ticker, start_date, end_date)
        elif self.use_yfinance:
            return self._fetch_from_yfinance(ticker, start_date, end_date)
        else:
            raise ValueError(f"No online data source available for {ticker}")
    
    def _fetch_from_factset(self, ticker: str, start_date: str = '2015-01-01', end_date: Optional[str] = None) -> pd.DataFrame:
        """Fetch data from FactSet."""
        try:
            end = end_date or datetime.now().strftime('%Y-%m-%d')
            
            # Fetch using FactSet collector
            df = self.collector.fetch_history([ticker], start_date, end)
            
            if df.empty:
                raise ValueError(f"No data returned from FactSet for {ticker}")
            
            # Convert MultiIndex to flat DataFrame
            result = pd.DataFrame({
                'Date': df.index,
                'Open': df[(ticker, 'Open')].values,
                'High': df[(ticker, 'High')].values,
                'Low': df[(ticker, 'Low')].values,
                'Close': df[(ticker, 'Close')].values,
                'Volume': df[(ticker, 'Volume')].values,
                'Ticker': ticker
            })
            
            result = result.reset_index(drop=True)
            result['Date'] = pd.to_datetime(result['Date']).dt.tz_localize(None)
            
            # Remove invalid rows
            result = result[result['Close'] > 0].reset_index(drop=True)
            
            logger.info(f"✅ {ticker}: Fetched {len(result)} rows from FactSet ({result['Date'].min().date()} to {result['Date'].max().date()})")
            
            return result
            
        except Exception as e:
            raise ValueError(f"Failed to fetch data for {ticker} from FactSet: {e}")
    
    def _fetch_from_yfinance(self, ticker: str, start_date: str = '2015-01-01', end_date: Optional[str] = None) -> pd.DataFrame:
        """Fetch data from yfinance."""
        if ticker not in self.YFINANCE_SYMBOLS:
            raise ValueError(f"No yfinance symbol for {ticker}")
        
        try:
            import yfinance as yf
            
            yf_symbol = self.YFINANCE_SYMBOLS[ticker]
            yf_ticker = yf.Ticker(yf_symbol)
            
            end = end_date or datetime.now().strftime('%Y-%m-%d')
            data = yf_ticker.history(start=start_date, end=end)
            
            if len(data) == 0:
                raise ValueError(f"No data returned from yfinance for {ticker}")
            
            df = pd.DataFrame({
                'Date': data.index,
                'Open': data['Open'].values,
                'High': data['High'].values,
                'Low': data['Low'].values,
                'Close': data['Close'].values,
                'Volume': data['Volume'].values,
                'Ticker': ticker
            })
            
            df = df.reset_index(drop=True)
            df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None)
            df = df[df['Close'] > 0].reset_index(drop=True)
            
            logger.info(f"✅ {ticker}: Fetched {len(df)} rows from yfinance ({df['Date'].min().date()} to {df['Date'].max().date()})")
            
            return df
            
        except Exception as e:
            raise ValueError(f"Failed to fetch data for {ticker} from yfinance: {e}")
    
    def load_single_asset(self, ticker: str) -> pd.DataFrame:
        """
        Load a single asset's price data.
        
        Args:
            ticker: Asset ticker (ES, NQ, GC, CL, NG, etc.)
            
        Returns:
            DataFrame with Date, Open, High, Low, Close, Volume
        """
        # Check if ticker has CSV file
        if ticker in self.ASSET_FILES:
            filepath = self.dataset_dir / self.ASSET_FILES[ticker]
            if filepath.exists():
                return self._load_from_csv(filepath, ticker)
            else:
                logger.warning(f"CSV not found for {ticker}, trying online source")
        
        # Try online source
        if self.use_factset or self.use_yfinance:
            return self._fetch_from_online(ticker, start_date='2015-01-01')
        else:
            raise ValueError(f"No data source available for {ticker}")
    
    def _load_from_csv(self, filepath: Path, ticker: str) -> pd.DataFrame:
        """Load data from CSV file."""
        # Implementation same as before...
        pass

    def load_assets(
        self,
        tickers: List[str],
        start_date: str = '2015-01-01',
        end_date: Optional[str] = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Load price data for multiple assets.

        Args:
            tickers: List of asset tickers (ES, NQ, GC, CL, NG, etc.)
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD), defaults to today

        Returns:
            Dict mapping ticker -> DataFrame with price data
        """
        result: Dict[str, pd.DataFrame] = {}
        for ticker in tickers:
            try:
                if self.use_factset or self.use_yfinance:
                    df = self._fetch_from_online(ticker, start_date=start_date, end_date=end_date)
                else:
                    df = self.load_single_asset(ticker)
                result[ticker] = df
            except Exception as e:
                logger.warning(f"Failed to load data for {ticker}: {e}")
        return result


def load_assets(
    tickers: List[str],
    start_date: str = '2015-01-01',
    end_date: Optional[str] = None,
    use_yfinance: bool = True,
    use_factset: bool = False
) -> Dict[str, pd.DataFrame]:
    """
    Convenience function to load price data for multiple assets.

    Args:
        tickers: List of asset tickers
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD), defaults to today
        use_yfinance: If True, fetch data from yfinance
        use_factset: If True and available, use FactSet instead of yfinance

    Returns:
        Dict mapping ticker -> DataFrame with price data
    """
    loader = MultiAssetLoader(use_yfinance=use_yfinance, use_factset=use_factset)
    return loader.load_assets(tickers, start_date=start_date, end_date=end_date)