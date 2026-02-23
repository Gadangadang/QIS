"""
Benchmark loader for performance comparison.
"""

from pathlib import Path
from typing import Dict, Tuple
import pandas as pd
import numpy as np
import logging

try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False
    yf = None

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
    
    def _fetch_from_yfinance(
        self,
        ticker: str,
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """
        Fetch benchmark data directly from yfinance.

        Args:
            ticker: Ticker symbol (e.g., 'SPY').
            start_date: Start date (YYYY-MM-DD).
            end_date: End date (YYYY-MM-DD).

        Returns:
            DataFrame with a Close column and Date index.

        Raises:
            ValueError: If no data is returned for the ticker.
        """
        data = yf.download(ticker, start=start_date, end=end_date, progress=False)

        if data.empty:
            raise ValueError(f"No data returned for {ticker}")

        # Flatten MultiIndex columns — yfinance returns (Price, Ticker) tuples
        # when level 0 contains the price-field names ('Close', 'Open', etc.)
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)

        df = pd.DataFrame({'Close': data['Close']})
        df.index.name = 'Date'
        return df

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


class BenchmarkComparator:
    """
    Compares portfolio performance against a benchmark.

    Provides metrics such as beta, alpha, correlation, and information ratio.
    All public methods are usable either on an instance or as static methods.
    """

    @staticmethod
    def calculate_metrics(
        portfolio_equity: pd.DataFrame,
        benchmark_df: pd.DataFrame,
        risk_free_rate: float = 0.02
    ) -> Dict:
        """
        Calculate benchmark-relative performance metrics.

        Args:
            portfolio_equity: DataFrame with a TotalValue column (Date index).
            benchmark_df: DataFrame with a TotalValue column (Date index).
            risk_free_rate: Annual risk-free rate used for alpha calculation.

        Returns:
            Dict with keys: 'Beta (Full Period)', 'Beta (90-day avg)',
            'Beta (1-year avg)', 'Alpha (Annual)', 'rolling_beta_90d',
            'Correlation', 'Information Ratio'.
            Returns empty dict when data is insufficient.
        """
        if portfolio_equity.empty or benchmark_df.empty:
            return {}

        port_ret = portfolio_equity['TotalValue'].pct_change().dropna()
        bench_ret = benchmark_df['TotalValue'].pct_change().dropna()

        # Align on common dates
        common_idx = port_ret.index.intersection(bench_ret.index)
        if len(common_idx) < 2:
            return {}

        port_ret = port_ret.loc[common_idx]
        bench_ret = bench_ret.loc[common_idx]

        # Full-period beta via OLS (cov / var)
        bench_var = bench_ret.var()
        beta_full = float(port_ret.cov(bench_ret) / bench_var) if bench_var != 0 else float('nan')

        # Rolling betas
        rolling_beta_90d = BenchmarkComparator.calculate_rolling_beta(port_ret, bench_ret, window=90)
        rolling_beta_1y = BenchmarkComparator.calculate_rolling_beta(port_ret, bench_ret, window=252)

        beta_90d_avg = float(rolling_beta_90d.mean()) if not rolling_beta_90d.empty else float('nan')
        beta_1y_avg = float(rolling_beta_1y.mean()) if not rolling_beta_1y.empty else float('nan')

        # Annualised alpha: alpha_daily = mean(port_ret) - beta * mean(bench_ret) - rf_daily
        rf_daily = risk_free_rate / 252
        alpha_daily = port_ret.mean() - beta_full * bench_ret.mean() - rf_daily
        alpha_annual = float(alpha_daily * 252)

        # Correlation
        correlation = float(port_ret.corr(bench_ret))

        # Information ratio: mean(active_return) / std(active_return) * sqrt(252)
        active_ret = port_ret - bench_ret
        ir = float(active_ret.mean() / active_ret.std() * np.sqrt(252)) if active_ret.std() != 0 else float('nan')

        return {
            'Beta (Full Period)': beta_full,
            'Beta (90-day avg)': beta_90d_avg,
            'Beta (1-year avg)': beta_1y_avg,
            'Alpha (Annual)': alpha_annual,
            'rolling_beta_90d': rolling_beta_90d,
            'Correlation': correlation,
            'Information Ratio': ir,
        }

    @staticmethod
    def format_for_base_100(
        portfolio_equity: pd.DataFrame,
        benchmark_df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Normalise both equity curves to start at 100.

        Args:
            portfolio_equity: DataFrame with a TotalValue column.
            benchmark_df: DataFrame with a TotalValue column.

        Returns:
            Tuple of (port_norm, bench_norm) DataFrames, each with TotalValue
            starting at 100.
        """
        port_norm = portfolio_equity.copy()
        bench_norm = benchmark_df.copy()

        if not port_norm.empty and 'TotalValue' in port_norm.columns:
            first = port_norm['TotalValue'].iloc[0]
            if first is not None and first != 0 and not np.isnan(first):
                port_norm['TotalValue'] = port_norm['TotalValue'] / first * 100

        if not bench_norm.empty and 'TotalValue' in bench_norm.columns:
            first = bench_norm['TotalValue'].iloc[0]
            if first is not None and first != 0 and not np.isnan(first):
                bench_norm['TotalValue'] = bench_norm['TotalValue'] / first * 100

        return port_norm, bench_norm

    @staticmethod
    def calculate_rolling_beta(
        portfolio_returns: pd.Series,
        benchmark_returns: pd.Series,
        window: int = 90
    ) -> pd.Series:
        """
        Compute rolling window OLS beta.

        Args:
            portfolio_returns: Series of portfolio period returns.
            benchmark_returns: Series of benchmark period returns.
            window: Rolling window size in periods.

        Returns:
            pd.Series of rolling beta values (length = len(portfolio_returns) - window + 1).
        """
        # Align
        common = portfolio_returns.index.intersection(benchmark_returns.index)
        p = portfolio_returns.loc[common]
        b = benchmark_returns.loc[common]

        rolling_cov = p.rolling(window=window, min_periods=window).cov(b)
        rolling_var = b.rolling(window=window, min_periods=window).var()

        beta_series = rolling_cov / rolling_var
        return beta_series.dropna()