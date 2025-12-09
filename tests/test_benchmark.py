import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch
from pathlib import Path
import shutil
import tempfile
from core.benchmark import BenchmarkLoader, BenchmarkComparator

class TestBenchmarkLoader:
    @pytest.fixture
    def temp_dir(self):
        # Create a temporary directory for testing
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        # Cleanup after test
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def loader(self, temp_dir):
        return BenchmarkLoader(cache_dir=temp_dir)

    @pytest.fixture
    def mock_yf_download(self):
        with patch('yfinance.download') as mock:
            yield mock

    def test_init(self, temp_dir):
        loader = BenchmarkLoader(cache_dir=temp_dir)
        assert Path(temp_dir).exists()

    def test_fetch_from_yfinance_success(self, loader, mock_yf_download):
        # Setup mock return value
        dates = pd.date_range(start='2023-01-01', end='2023-01-10')
        mock_data = pd.DataFrame({
            'Close': np.linspace(100, 110, len(dates))
        }, index=dates)
        mock_yf_download.return_value = mock_data

        df = loader._fetch_from_yfinance('SPY', '2023-01-01', '2023-01-10')
        
        assert isinstance(df, pd.DataFrame)
        assert 'Close' in df.columns
        assert len(df) == 10
        mock_yf_download.assert_called_once()

    def test_fetch_from_yfinance_empty(self, loader, mock_yf_download):
        mock_yf_download.return_value = pd.DataFrame()
        
        with pytest.raises(ValueError, match="No data returned for SPY"):
            loader._fetch_from_yfinance('SPY', '2023-01-01', '2023-01-10')

    def test_fetch_from_yfinance_multiindex(self, loader, mock_yf_download):
        # Simulate yfinance returning MultiIndex columns (common with recent versions)
        dates = pd.date_range(start='2023-01-01', end='2023-01-05')
        columns = pd.MultiIndex.from_product([['Close', 'Open'], ['SPY']])
        mock_data = pd.DataFrame(
            np.random.randn(5, 2),
            index=dates,
            columns=columns
        )
        # Adjust mock to behave like yfinance result where columns might be (Price, Ticker)
        # But the code expects level 0 to be the price type if multiindex
        # Let's match the code's expectation: data.columns.get_level_values(0)
        
        # Actually, yfinance usually returns (Price, Ticker) or just Price if one ticker.
        # The code does: if isinstance(data.columns, pd.MultiIndex): data.columns = data.columns.get_level_values(0)
        # This implies it expects the first level to be 'Close', 'Open' etc.
        
        mock_yf_download.return_value = mock_data
        
        df = loader._fetch_from_yfinance('SPY', '2023-01-01', '2023-01-05')
        assert 'Close' in df.columns
        assert len(df) == 5

    def test_load_benchmark_no_cache(self, loader, mock_yf_download):
        dates = pd.date_range(start='2023-01-01', end='2023-01-10')
        mock_data = pd.DataFrame({
            'Close': np.linspace(100, 110, len(dates))
        }, index=dates)
        mock_yf_download.return_value = mock_data

        df = loader.load_benchmark('SPY', '2023-01-01', '2023-01-10', initial_value=1000)
        
        assert 'TotalValue' in df.columns
        assert df['TotalValue'].iloc[0] == 1000
        # Check if file was created in cache
        assert (Path(loader.cache_dir) / "SPY_benchmark.csv").exists()

    def test_load_benchmark_from_cache(self, loader, mock_yf_download):
        # Create a dummy cache file
        dates = pd.date_range(start='2023-01-01', end='2023-01-20')
        cache_data = pd.DataFrame({
            'Close': np.linspace(100, 120, len(dates))
        }, index=dates)
        cache_data.index.name = 'Date'
        cache_file = Path(loader.cache_dir) / "SPY_benchmark.csv"
        cache_data.to_csv(cache_file)

        # Request a subset that is fully covered by cache
        df = loader.load_benchmark('SPY', '2023-01-05', '2023-01-15')
        
        assert len(df) == 11
        assert df.index.min() == pd.Timestamp('2023-01-05')
        assert df.index.max() == pd.Timestamp('2023-01-15')
        mock_yf_download.assert_not_called()

    def test_load_benchmark_cache_miss_update(self, loader, mock_yf_download):
        # Create a dummy cache file with insufficient range
        dates = pd.date_range(start='2023-01-05', end='2023-01-10')
        cache_data = pd.DataFrame({
            'Close': np.linspace(100, 105, len(dates))
        }, index=dates)
        cache_data.index.name = 'Date'
        cache_file = Path(loader.cache_dir) / "SPY_benchmark.csv"
        cache_data.to_csv(cache_file)

        # Request a range outside cache
        new_dates = pd.date_range(start='2023-01-01', end='2023-01-15')
        mock_data = pd.DataFrame({
            'Close': np.linspace(95, 110, len(new_dates))
        }, index=new_dates)
        mock_yf_download.return_value = mock_data

        df = loader.load_benchmark('SPY', '2023-01-01', '2023-01-15')
        
        assert len(df) == 15
        mock_yf_download.assert_called_once()

    def test_align_benchmark_to_portfolio(self, loader):
        # Portfolio dates
        port_dates = pd.date_range(start='2023-01-01', end='2023-01-05')
        portfolio = pd.DataFrame({
            'TotalValue': [100, 101, 102, 103, 104]
        }, index=port_dates)

        # Benchmark dates (different range/frequency)
        bench_dates = pd.date_range(start='2022-12-31', end='2023-01-06')
        benchmark = pd.DataFrame({
            'TotalValue': [50, 51, 52, 53, 54, 55, 56]
        }, index=bench_dates)

        aligned = loader.align_benchmark_to_portfolio(portfolio, benchmark)
        
        assert len(aligned) == len(portfolio)
        assert aligned.index.equals(portfolio.index)
        # Check renormalization: aligned start should match portfolio start (100)
        assert aligned['TotalValue'].iloc[0] == 100.0


class TestBenchmarkComparator:
    @pytest.fixture
    def sample_data(self):
        dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='B')
        n = len(dates)
        
        # Create synthetic returns
        np.random.seed(42)
        bench_returns = np.random.normal(0.0005, 0.01, n)
        # Portfolio correlated with benchmark + alpha
        port_returns = 1.2 * bench_returns + 0.0001 + np.random.normal(0, 0.005, n)
        
        bench_equity = (1 + bench_returns).cumprod() * 100
        port_equity = (1 + port_returns).cumprod() * 100
        
        bench_df = pd.DataFrame({'TotalValue': bench_equity}, index=dates)
        port_df = pd.DataFrame({'TotalValue': port_equity}, index=dates)
        
        return port_df, bench_df

    def test_calculate_metrics(self, sample_data):
        port_df, bench_df = sample_data
        metrics = BenchmarkComparator.calculate_metrics(port_df, bench_df)
        
        assert 'Beta (Full Period)' in metrics
        assert 'Alpha (Annual)' in metrics
        assert 'Sharpe Ratio' not in metrics # Not calculated here, but good to know
        assert 'Information Ratio' in metrics
        
        # Check basic logic
        assert isinstance(metrics['Beta (Full Period)'], float)
        assert isinstance(metrics['Correlation'], float)

    def test_calculate_metrics_empty(self):
        empty_df = pd.DataFrame({'TotalValue': []})
        metrics = BenchmarkComparator.calculate_metrics(empty_df, empty_df)
        assert metrics == {}

    def test_calculate_rolling_beta(self, sample_data):
        port_df, bench_df = sample_data
        port_ret = port_df['TotalValue'].pct_change().dropna()
        bench_ret = bench_df['TotalValue'].pct_change().dropna()
        
        rolling_beta = BenchmarkComparator.calculate_rolling_beta(port_ret, bench_ret, window=30)
        
        assert len(rolling_beta) == len(port_ret) - 30 + 1
        assert isinstance(rolling_beta, pd.Series)

    def test_format_for_base_100(self, sample_data):
        port_df, bench_df = sample_data
        # Modify start values
        port_df['TotalValue'] *= 2
        bench_df['TotalValue'] *= 0.5
        
        p_norm, b_norm = BenchmarkComparator.format_for_base_100(port_df, bench_df)
        
        assert p_norm['TotalValue'].iloc[0] == 100.0
        assert b_norm['TotalValue'].iloc[0] == 100.0
