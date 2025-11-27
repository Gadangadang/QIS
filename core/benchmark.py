"""
Benchmark Module

Handles benchmark loading, alignment, and comparison metrics.
Supports various benchmark indices like S&P 500, MSCI World, risk-free rates, etc.
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional
from pathlib import Path
import yfinance as yf


class BenchmarkLoader:
    """
    Loads and manages benchmark data for performance comparison.
    
    Supported benchmarks:
    - SPY: S&P 500 ETF
    - VT: Vanguard Total World Stock ETF (MSCI World proxy)
    - AGG: US Aggregate Bond Index
    - IEF: 7-10 Year Treasury ETF (risk-free proxy)
    - TLT: 20+ Year Treasury ETF
    - Custom tickers
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
    
    def __init__(self, cache_dir: str = 'Dataset'):
        """
        Initialize benchmark loader.
        
        Args:
            cache_dir: Directory to cache benchmark data
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True, parents=True)
    
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
        cache_file = self.cache_dir / f"{ticker}_benchmark.csv"
        
        # Try to load from cache first
        if cache_file.exists():
            df = pd.read_csv(cache_file, parse_dates=['Date'], index_col='Date')
            
            # Check if we need to update
            if df.index.max() < pd.Timestamp(end_date):
                print(f"📥 Updating {ticker} benchmark data...")
                df = self._fetch_from_yfinance(ticker, start_date, end_date)
                df.to_csv(cache_file)
            else:
                # Filter to requested date range
                mask = (df.index >= start_date) & (df.index <= end_date)
                df = df.loc[mask].copy()
        else:
            print(f"📥 Downloading {ticker} benchmark data...")
            df = self._fetch_from_yfinance(ticker, start_date, end_date)
            df.to_csv(cache_file)
        
        # Normalize to initial value
        if len(df) > 0:
            df['TotalValue'] = (df['Close'] / df['Close'].iloc[0]) * initial_value
        
        return df[['Close', 'TotalValue']]
    
    def _fetch_from_yfinance(
        self,
        ticker: str,
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """Fetch benchmark data from yfinance."""
        try:
            data = yf.download(
                ticker,
                start=start_date,
                end=end_date,
                progress=False,
                auto_adjust=True
            )
            
            if data.empty:
                raise ValueError(f"No data returned for {ticker}")
            
            # Flatten multi-index columns if present
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)
            
            # Keep only Close
            df = pd.DataFrame({
                'Close': data['Close']
            })
            df.index.name = 'Date'
            
            return df
        
        except Exception as e:
            raise ValueError(f"Failed to download {ticker}: {e}")
    
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
        
        # Reindex benchmark to portfolio dates, forward fill missing values
        aligned = benchmark_df.reindex(portfolio_equity.index, method='ffill')
        
        # Renormalize to match portfolio starting value
        if len(aligned) > 0 and len(portfolio_equity) > 0:
            initial_portfolio = portfolio_equity['TotalValue'].iloc[0]
            aligned['TotalValue'] = (aligned['TotalValue'] / aligned['TotalValue'].iloc[0]) * initial_portfolio
        
        return aligned


class BenchmarkComparator:
    """
    Calculates benchmark-relative metrics and statistics.
    
    Provides:
    - Relative returns
    - Beta calculation (rolling and full-period)
    - Alpha calculation
    - Tracking error
    - Information ratio
    - Up/down capture ratios
    """
    
    @staticmethod
    def calculate_metrics(
        portfolio_equity: pd.DataFrame,
        benchmark_equity: pd.DataFrame,
        risk_free_rate: float = 0.02
    ) -> Dict:
        """
        Calculate comprehensive benchmark comparison metrics.
        
        Args:
            portfolio_equity: Portfolio equity curve (Date index, TotalValue)
            benchmark_equity: Benchmark equity curve (Date index, TotalValue)
            risk_free_rate: Annual risk-free rate (default 2%)
            
        Returns:
            Dictionary with comparison metrics
        """
        # Calculate returns
        port_returns = portfolio_equity['TotalValue'].pct_change().dropna()
        bench_returns = benchmark_equity['TotalValue'].pct_change().dropna()
        
        # Align returns
        aligned = pd.DataFrame({
            'Portfolio': port_returns,
            'Benchmark': bench_returns
        }).dropna()
        
        if len(aligned) == 0:
            return {}
        
        port_ret = aligned['Portfolio']
        bench_ret = aligned['Benchmark']
        
        # Excess returns
        daily_rf = risk_free_rate / 252
        port_excess = port_ret - daily_rf
        bench_excess = bench_ret - daily_rf
        
        # Total returns
        total_port_return = (portfolio_equity['TotalValue'].iloc[-1] / 
                            portfolio_equity['TotalValue'].iloc[0] - 1)
        total_bench_return = (benchmark_equity['TotalValue'].iloc[-1] / 
                             benchmark_equity['TotalValue'].iloc[0] - 1)
        
        # Beta (full period)
        covariance = np.cov(port_ret, bench_ret)[0, 1]
        bench_variance = np.var(bench_ret)
        beta = covariance / bench_variance if bench_variance > 0 else 0
        
        # Alpha (Jensen's alpha)
        alpha = port_excess.mean() - beta * bench_excess.mean()
        alpha_annual = alpha * 252  # Annualized
        
        # Tracking error (annualized)
        active_returns = port_ret - bench_ret
        tracking_error = active_returns.std() * np.sqrt(252)
        
        # Information ratio
        information_ratio = (active_returns.mean() * 252) / tracking_error if tracking_error > 0 else 0
        
        # Up/Down capture
        up_market = bench_ret > 0
        down_market = bench_ret < 0
        
        up_capture = (port_ret[up_market].mean() / bench_ret[up_market].mean()) if up_market.sum() > 0 else 0
        down_capture = (port_ret[down_market].mean() / bench_ret[down_market].mean()) if down_market.sum() > 0 else 0
        
        # Correlation
        correlation = port_ret.corr(bench_ret)
        
        # Rolling beta (90-day)
        rolling_beta = BenchmarkComparator.calculate_rolling_beta(
            port_ret, bench_ret, window=90
        )
        
        # Rolling beta (252-day / 1 year)
        rolling_beta_1y = BenchmarkComparator.calculate_rolling_beta(
            port_ret, bench_ret, window=252
        )
        
        return {
            'Benchmark Return': total_bench_return,
            'Relative Return': total_port_return - total_bench_return,
            'Beta (Full Period)': beta,
            'Beta (90-day avg)': rolling_beta.mean() if not rolling_beta.empty else beta,
            'Beta (1-year avg)': rolling_beta_1y.mean() if not rolling_beta_1y.empty else beta,
            'Alpha (Annual)': alpha_annual,
            'Tracking Error': tracking_error,
            'Information Ratio': information_ratio,
            'Correlation': correlation,
            'Up Capture Ratio': up_capture,
            'Down Capture Ratio': down_capture,
            'Up/Down Ratio': up_capture / down_capture if down_capture != 0 else 0,
            'rolling_beta_90d': rolling_beta,
            'rolling_beta_1y': rolling_beta_1y
        }
    
    @staticmethod
    def calculate_rolling_beta(
        portfolio_returns: pd.Series,
        benchmark_returns: pd.Series,
        window: int = 90
    ) -> pd.Series:
        """
        Calculate rolling beta over a window.
        
        Args:
            portfolio_returns: Portfolio daily returns
            benchmark_returns: Benchmark daily returns
            window: Rolling window in days
            
        Returns:
            Series with rolling beta values
        """
        aligned = pd.DataFrame({
            'Portfolio': portfolio_returns,
            'Benchmark': benchmark_returns
        }).dropna()
        
        if len(aligned) < window:
            return pd.Series()
        
        rolling_cov = aligned['Portfolio'].rolling(window).cov(aligned['Benchmark'])
        rolling_var = aligned['Benchmark'].rolling(window).var()
        
        rolling_beta = rolling_cov / rolling_var
        return rolling_beta.dropna()
    
    @staticmethod
    def format_for_base_100(
        portfolio_equity: pd.DataFrame,
        benchmark_equity: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Normalize both portfolio and benchmark to base 100.
        
        Args:
            portfolio_equity: Portfolio equity curve
            benchmark_equity: Benchmark equity curve
            
        Returns:
            Tuple of (normalized_portfolio, normalized_benchmark)
        """
        port_norm = portfolio_equity.copy()
        bench_norm = benchmark_equity.copy()
        
        port_norm['TotalValue'] = (port_norm['TotalValue'] / port_norm['TotalValue'].iloc[0]) * 100
        bench_norm['TotalValue'] = (bench_norm['TotalValue'] / bench_norm['TotalValue'].iloc[0]) * 100
        
        return port_norm, bench_norm


if __name__ == "__main__":
    # Test benchmark loader
    print("Testing BenchmarkLoader...")
    
    loader = BenchmarkLoader()
    
    # Load S&P 500
    spy = loader.load_benchmark('SPY', '2020-01-01', '2023-12-31')
    print(f"\nSPY data: {len(spy)} days")
    print(spy.head())
    print(f"Return: {(spy['TotalValue'].iloc[-1] / spy['TotalValue'].iloc[0] - 1) * 100:.2f}%")
    
    # Load MSCI World proxy
    vt = loader.load_benchmark('VT', '2020-01-01', '2023-12-31')
    print(f"\nVT data: {len(vt)} days")
    print(f"Return: {(vt['TotalValue'].iloc[-1] / vt['TotalValue'].iloc[0] - 1) * 100:.2f}%")
