"""
Tests for utils.plotter module.
"""

import pytest
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for testing
import matplotlib.pyplot as plt

from utils.plotter import PortfolioPlotter


class MockBacktestResult:
    """Mock BacktestResult for testing."""
    
    def __init__(self, equity_values, dates):
        """Initialize mock backtest result."""
        # Convert to numpy array to ensure proper operations
        equity_values = np.array(equity_values)
        
        self.equity_curve = pd.DataFrame({
            'TotalValue': equity_values,
            'Cash': equity_values * 0.5,
            'Holdings': equity_values * 0.5
        }, index=dates)
        
        # Create some trades
        num_trades = min(5, len(dates))
        self.trades = pd.DataFrame({
            'Date': dates[:num_trades],
            'Action': ['BUY', 'SELL', 'BUY', 'SELL', 'BUY'][:num_trades],
            'Price': [100, 105, 102, 108, 110][:num_trades],
            'Shares': [10, 10, 15, 15, 20][:num_trades]
        })


class TestPortfolioPlotter:
    """Test suite for PortfolioPlotter class."""

    @pytest.fixture
    def sample_dates(self):
        """Create sample date range."""
        return pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')

    @pytest.fixture
    def sample_strategy_results(self, sample_dates):
        """Create sample strategy results."""
        equity_values = 10000 * (1 + np.random.randn(len(sample_dates)).cumsum() * 0.01)
        equity_values = np.maximum(equity_values, 5000)  # Ensure positive values
        
        mock_result = MockBacktestResult(equity_values, sample_dates)
        
        return {
            'Strategy1': {
                'result': mock_result,
                'capital': 10000
            }
        }

    @pytest.fixture
    def multi_strategy_results(self, sample_dates):
        """Create multiple strategy results."""
        results = {}
        
        for i in range(3):
            equity_values = 5000 * (1 + np.random.randn(len(sample_dates)).cumsum() * 0.01)
            equity_values = np.maximum(equity_values, 2500)
            mock_result = MockBacktestResult(equity_values, sample_dates)
            results[f'Strategy{i+1}'] = {
                'result': mock_result,
                'capital': 5000
            }
        
        return results

    @pytest.fixture
    def sample_benchmark_data(self, sample_dates):
        """Create sample benchmark data."""
        close_prices = 100 * (1 + np.random.randn(len(sample_dates)).cumsum() * 0.005)
        close_prices = np.maximum(close_prices, 80)
        
        return pd.DataFrame({
            'Open': close_prices * 0.99,
            'High': close_prices * 1.01,
            'Low': close_prices * 0.98,
            'Close': close_prices,
            'Volume': np.random.randint(1000000, 10000000, len(sample_dates))
        }, index=sample_dates)

    def teardown_method(self, method):
        """Close all matplotlib figures after each test."""
        plt.close('all')

    def test_initialization(self, sample_strategy_results):
        """Test PortfolioPlotter initialization."""
        plotter = PortfolioPlotter(sample_strategy_results)
        
        assert plotter.strategy_results == sample_strategy_results
        assert plotter.benchmark_name == 'SPY'

    def test_initialization_with_benchmark(self, sample_strategy_results, sample_benchmark_data):
        """Test initialization with benchmark data."""
        plotter = PortfolioPlotter(
            sample_strategy_results,
            benchmark_data=sample_benchmark_data,
            benchmark_name='SPY'
        )
        
        assert plotter.benchmark_data is not None
        assert plotter.benchmark_name == 'SPY'

    def test_prepare_combined_data(self, sample_strategy_results):
        """Test combined data preparation."""
        plotter = PortfolioPlotter(sample_strategy_results)
        
        assert hasattr(plotter, 'combined_equity')
        assert hasattr(plotter, 'total_capital')
        assert hasattr(plotter, 'combined_returns')
        assert len(plotter.combined_equity) > 0

    def test_combined_equity_calculation(self, multi_strategy_results):
        """Test combined equity curve calculation."""
        plotter = PortfolioPlotter(multi_strategy_results)
        
        # Combined equity should be sum of all strategies at each point
        first_value = plotter.combined_equity.iloc[0]
        assert first_value > 0

    def test_total_capital_calculation(self, multi_strategy_results):
        """Test total capital calculation."""
        plotter = PortfolioPlotter(multi_strategy_results)
        
        expected_capital = sum(data['capital'] for data in multi_strategy_results.values())
        assert plotter.total_capital == expected_capital

    def test_combined_returns_calculation(self, sample_strategy_results):
        """Test combined returns calculation."""
        plotter = PortfolioPlotter(sample_strategy_results)
        
        # Returns should be calculated
        assert len(plotter.combined_returns) == len(plotter.combined_equity)
        # First return should be 0 (from fillna)
        assert plotter.combined_returns.iloc[0] == 0

    def test_dates_alignment(self, sample_strategy_results):
        """Test that dates are properly aligned."""
        plotter = PortfolioPlotter(sample_strategy_results)
        
        first_result = list(sample_strategy_results.values())[0]['result']
        expected_dates = first_result.equity_curve.index
        
        pd.testing.assert_index_equal(plotter.dates, expected_dates)

    def test_empty_strategy_results_raises_error(self):
        """Test that empty strategy results raises ValueError."""
        with pytest.raises(ValueError):
            PortfolioPlotter({})

    def test_custom_benchmark_name(self, sample_strategy_results, sample_benchmark_data):
        """Test custom benchmark name."""
        plotter = PortfolioPlotter(
            sample_strategy_results,
            benchmark_data=sample_benchmark_data,
            benchmark_name='QQQ'
        )
        
        assert plotter.benchmark_name == 'QQQ'

    def test_plot_equity_curves_basic(self, sample_strategy_results):
        """Test basic equity curve plotting."""
        plotter = PortfolioPlotter(sample_strategy_results)
        
        # Should execute without raising an error
        plotter.plot_equity_curves()

    def test_plot_equity_curves_with_benchmark(self, sample_strategy_results, sample_benchmark_data):
        """Test equity curve plotting with benchmark."""
        plotter = PortfolioPlotter(
            sample_strategy_results,
            benchmark_data=sample_benchmark_data
        )
        
        # Should execute without raising an error
        plotter.plot_equity_curves(show_benchmark=True)

    def test_plot_equity_curves_multi_strategy(self, multi_strategy_results):
        """Test equity curve plotting with multiple strategies."""
        plotter = PortfolioPlotter(multi_strategy_results)
        
        # Should execute without raising an error
        plotter.plot_equity_curves(show_individual=True)

    def test_plotter_handles_positive_equity(self, sample_strategy_results):
        """Test that plotter handles positive equity values."""
        plotter = PortfolioPlotter(sample_strategy_results)
        
        # All equity values should be positive
        assert (plotter.combined_equity > 0).all()

    def test_plotter_with_minimal_data(self):
        """Test plotter with minimal data."""
        # Create minimal valid strategy with just 2 days
        dates = pd.date_range(start='2023-01-01', end='2023-01-02', freq='D')
        equity_values = np.array([10000, 10100])
        
        mock_result = MockBacktestResult(equity_values, dates)
        strategy_results = {
            'Strategy1': {
                'result': mock_result,
                'capital': 10000
            }
        }
        
        plotter = PortfolioPlotter(strategy_results)
        
        assert len(plotter.combined_equity) == 2
        assert plotter.total_capital == 10000

    def test_plot_drawdown_executes(self, sample_strategy_results):
        """Test that plot_drawdown executes without error."""
        plotter = PortfolioPlotter(sample_strategy_results)
        
        # Should execute without raising an error
        plotter.plot_drawdown()

    def test_plot_monthly_returns_executes(self, sample_strategy_results):
        """Test that plot_monthly_returns_heatmap executes without error."""
        plotter = PortfolioPlotter(sample_strategy_results)
        
        # Should execute without raising an error
        plotter.plot_monthly_returns_heatmap()

    def test_plot_rolling_metrics_executes(self, sample_strategy_results):
        """Test that plot_rolling_metrics executes without error."""
        plotter = PortfolioPlotter(sample_strategy_results)
        
        # Should execute without raising an error
        plotter.plot_rolling_metrics()
