"""
Extended unit tests for utils.plotter to maximize coverage.

Focuses on:
- Edge cases: empty data, single point, missing columns
- Different plot types and configurations
- Error handling
- Metric calculations
- Benchmark handling

Run with: pytest tests/test_plotter_utils.py -v --cov=utils/plotter
"""

from __future__ import annotations

import pytest
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for testing
import matplotlib.pyplot as plt
from unittest.mock import Mock, patch, MagicMock

from utils.plotter import PortfolioPlotter, quick_equity_plot, quick_drawdown_plot, quick_heatmap


# ============================================================================
# Mock Classes
# ============================================================================

class MockBacktestResult:
    """Mock BacktestResult for testing."""
    
    def __init__(self, equity_values, dates):
        """
        Initialize mock backtest result.
        
        Args:
            equity_values: List or array of equity values
            dates: DatetimeIndex for the equity curve
        """
        equity_values = np.array(equity_values)
        
        self.equity_curve = pd.DataFrame({
            'TotalValue': equity_values,
            'Cash': equity_values * 0.3,
            'Holdings': equity_values * 0.7
        }, index=dates)
        
        # Create mock trades - ensure all arrays have same length
        num_trades = min(5, len(dates))
        actions = (['BUY', 'SELL'] * (num_trades // 2 + 1))[:num_trades]
        self.trades = pd.DataFrame({
            'Date': dates[:num_trades],
            'Action': actions,
            'Price': [100, 105, 102, 108, 110][:num_trades],
            'Shares': [10, 10, 15, 15, 20][:num_trades]
        })


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def sample_dates():
    """Create sample date range."""
    return pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')


@pytest.fixture
def short_dates():
    """Create short date range for edge case testing."""
    return pd.date_range(start='2023-01-01', periods=5, freq='D')


@pytest.fixture
def single_date():
    """Create single date for edge case testing."""
    return pd.date_range(start='2023-01-01', periods=1, freq='D')


@pytest.fixture
def sample_strategy_results(sample_dates):
    """Create sample strategy results."""
    np.random.seed(42)
    equity_values = 10000 * (1 + np.random.randn(len(sample_dates)).cumsum() * 0.01)
    equity_values = np.maximum(equity_values, 5000)  # Ensure positive
    
    mock_result = MockBacktestResult(equity_values, sample_dates)
    
    return {
        'Strategy1': {
            'result': mock_result,
            'capital': 10000
        }
    }


@pytest.fixture
def multi_strategy_results(sample_dates):
    """Create multiple strategy results."""
    results = {}
    np.random.seed(42)
    
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
def sample_benchmark_data(sample_dates):
    """Create sample benchmark data."""
    np.random.seed(42)
    base_price = 100.0
    returns = np.random.normal(0.0005, 0.01, len(sample_dates))
    close_prices = base_price * (1 + returns).cumprod()
    
    return pd.DataFrame({
        'Open': close_prices * 0.99,
        'High': close_prices * 1.01,
        'Low': close_prices * 0.99,
        'Close': close_prices,
        'Volume': np.random.randint(1000000, 5000000, len(sample_dates))
    }, index=sample_dates)


# ============================================================================
# Test Initialization
# ============================================================================

class TestPortfolioPlotterInitialization:
    """Test PortfolioPlotter initialization."""
    
    def test_initialization_with_valid_data(self, sample_strategy_results):
        """Test initialization with valid strategy results."""
        # Arrange & Act
        plotter = PortfolioPlotter(sample_strategy_results)
        
        # Assert
        assert plotter.strategy_results == sample_strategy_results
        assert plotter.benchmark_data is None
        assert plotter.benchmark_name == 'SPY'
        assert len(plotter.combined_equity) > 0
    
    def test_initialization_with_benchmark(self, sample_strategy_results, sample_benchmark_data):
        """Test initialization with benchmark data."""
        # Arrange & Act
        plotter = PortfolioPlotter(
            sample_strategy_results,
            benchmark_data=sample_benchmark_data,
            benchmark_name='QQQ'
        )
        
        # Assert
        assert plotter.benchmark_data is not None
        assert plotter.benchmark_name == 'QQQ'
    
    def test_initialization_with_empty_results_raises_error(self):
        """Test that empty results raises ValueError."""
        # Arrange & Act & Assert
        with pytest.raises(ValueError, match="No strategy results provided"):
            PortfolioPlotter({})
    
    def test_initialization_calculates_combined_equity(self, multi_strategy_results):
        """Test that combined equity is calculated correctly."""
        # Arrange & Act
        plotter = PortfolioPlotter(multi_strategy_results)
        
        # Assert
        assert len(plotter.combined_equity) > 0
        # Combined should be sum of all strategies
        first_date = plotter.dates[0]
        expected_total = sum(
            data['result'].equity_curve.loc[first_date, 'TotalValue']
            for data in multi_strategy_results.values()
        )
        assert plotter.combined_equity.iloc[0] == pytest.approx(expected_total)


# ============================================================================
# Test Equity Curve Plotting
# ============================================================================

class TestEquityCurvePlotting:
    """Test equity curve plotting functionality."""
    
    def test_plot_equity_curves_basic(self, sample_strategy_results):
        """Test basic equity curve plotting."""
        # Arrange
        plotter = PortfolioPlotter(sample_strategy_results)
        
        # Act
        fig = plotter.plot_equity_curves()
        
        # Assert
        assert fig is not None
        plt.close(fig)
    
    def test_plot_equity_curves_without_individual(self, sample_strategy_results):
        """Test equity curve plotting without individual strategies."""
        # Arrange
        plotter = PortfolioPlotter(sample_strategy_results)
        
        # Act
        fig = plotter.plot_equity_curves(show_individual=False)
        
        # Assert
        assert fig is not None
        plt.close(fig)
    
    def test_plot_equity_curves_without_combined(self, sample_strategy_results):
        """Test equity curve plotting without combined portfolio."""
        # Arrange
        plotter = PortfolioPlotter(sample_strategy_results)
        
        # Act
        fig = plotter.plot_equity_curves(show_combined=False)
        
        # Assert
        assert fig is not None
        plt.close(fig)
    
    def test_plot_equity_curves_with_benchmark(self, sample_strategy_results, sample_benchmark_data):
        """Test equity curve plotting with benchmark."""
        # Arrange
        plotter = PortfolioPlotter(sample_strategy_results, benchmark_data=sample_benchmark_data)
        
        # Act
        fig = plotter.plot_equity_curves(show_benchmark=True)
        
        # Assert
        assert fig is not None
        plt.close(fig)
    
    def test_plot_equity_curves_without_benchmark(self, sample_strategy_results, sample_benchmark_data):
        """Test equity curve plotting without benchmark."""
        # Arrange
        plotter = PortfolioPlotter(sample_strategy_results, benchmark_data=sample_benchmark_data)
        
        # Act
        fig = plotter.plot_equity_curves(show_benchmark=False)
        
        # Assert
        assert fig is not None
        plt.close(fig)
    
    def test_plot_equity_curves_log_scale(self, sample_strategy_results):
        """Test equity curve plotting with log scale."""
        # Arrange
        plotter = PortfolioPlotter(sample_strategy_results)
        
        # Act
        fig = plotter.plot_equity_curves(log_scale=True)
        
        # Assert
        assert fig is not None
        plt.close(fig)
    
    def test_plot_equity_curves_custom_figsize(self, sample_strategy_results):
        """Test equity curve plotting with custom figure size."""
        # Arrange
        plotter = PortfolioPlotter(sample_strategy_results)
        
        # Act
        fig = plotter.plot_equity_curves(figsize=(10, 6))
        
        # Assert
        assert fig is not None
        assert fig.get_figwidth() == 10
        assert fig.get_figheight() == 6
        plt.close(fig)


# ============================================================================
# Test Drawdown Plotting
# ============================================================================

class TestDrawdownPlotting:
    """Test drawdown plotting functionality."""
    
    def test_plot_drawdown_basic(self, sample_strategy_results):
        """Test basic drawdown plotting."""
        # Arrange
        plotter = PortfolioPlotter(sample_strategy_results)
        
        # Act
        fig = plotter.plot_drawdown()
        
        # Assert
        assert fig is not None
        plt.close(fig)
    
    def test_plot_drawdown_without_underwater(self, sample_strategy_results):
        """Test drawdown plotting without underwater chart."""
        # Arrange
        plotter = PortfolioPlotter(sample_strategy_results)
        
        # Act
        fig = plotter.plot_drawdown(show_underwater=False)
        
        # Assert
        assert fig is not None
        plt.close(fig)
    
    def test_plot_drawdown_with_underwater(self, sample_strategy_results):
        """Test drawdown plotting with underwater chart."""
        # Arrange
        plotter = PortfolioPlotter(sample_strategy_results)
        
        # Act
        fig = plotter.plot_drawdown(show_underwater=True)
        
        # Assert
        assert fig is not None
        plt.close(fig)


# ============================================================================
# Test Monthly Returns Heatmap
# ============================================================================

class TestMonthlyReturnsHeatmap:
    """Test monthly returns heatmap functionality."""
    
    def test_plot_monthly_returns_heatmap_combined(self, sample_strategy_results):
        """Test monthly returns heatmap for combined portfolio."""
        # Arrange
        plotter = PortfolioPlotter(sample_strategy_results)
        
        # Act
        fig = plotter.plot_monthly_returns_heatmap()
        
        # Assert
        assert fig is not None
        plt.close(fig)
    
    def test_plot_monthly_returns_heatmap_specific_strategy(self, sample_strategy_results):
        """Test monthly returns heatmap for specific strategy."""
        # Arrange
        plotter = PortfolioPlotter(sample_strategy_results)
        
        # Act
        fig = plotter.plot_monthly_returns_heatmap(strategy_name='Strategy1')
        
        # Assert
        assert fig is not None
        plt.close(fig)
    
    def test_plot_monthly_returns_heatmap_invalid_strategy(self, sample_strategy_results):
        """Test monthly returns heatmap with invalid strategy name."""
        # Arrange
        plotter = PortfolioPlotter(sample_strategy_results)
        
        # Act - should fall back to combined portfolio
        fig = plotter.plot_monthly_returns_heatmap(strategy_name='NonExistent')
        
        # Assert
        assert fig is not None
        plt.close(fig)


# ============================================================================
# Test Rolling Metrics
# ============================================================================

class TestRollingMetrics:
    """Test rolling metrics plotting functionality."""
    
    def test_plot_rolling_metrics_sharpe(self, sample_strategy_results):
        """Test rolling Sharpe ratio plotting."""
        # Arrange
        plotter = PortfolioPlotter(sample_strategy_results)
        
        # Act
        fig = plotter.plot_rolling_metrics(metrics=['sharpe'])
        
        # Assert
        assert fig is not None
        plt.close(fig)
    
    def test_plot_rolling_metrics_volatility(self, sample_strategy_results):
        """Test rolling volatility plotting."""
        # Arrange
        plotter = PortfolioPlotter(sample_strategy_results)
        
        # Act
        fig = plotter.plot_rolling_metrics(metrics=['volatility'])
        
        # Assert
        assert fig is not None
        plt.close(fig)
    
    def test_plot_rolling_metrics_returns(self, sample_strategy_results):
        """Test rolling returns plotting."""
        # Arrange
        plotter = PortfolioPlotter(sample_strategy_results)
        
        # Act
        fig = plotter.plot_rolling_metrics(metrics=['returns'])
        
        # Assert
        assert fig is not None
        plt.close(fig)
    
    def test_plot_rolling_metrics_multiple(self, sample_strategy_results):
        """Test plotting multiple rolling metrics."""
        # Arrange
        plotter = PortfolioPlotter(sample_strategy_results)
        
        # Act
        fig = plotter.plot_rolling_metrics(metrics=['sharpe', 'volatility', 'returns'])
        
        # Assert
        assert fig is not None
        plt.close(fig)
    
    def test_plot_rolling_metrics_custom_window(self, sample_strategy_results):
        """Test rolling metrics with custom window size."""
        # Arrange
        plotter = PortfolioPlotter(sample_strategy_results)
        
        # Act
        fig = plotter.plot_rolling_metrics(window=30, metrics=['sharpe'])
        
        # Assert
        assert fig is not None
        plt.close(fig)


# ============================================================================
# Test Dashboards
# ============================================================================

class TestDashboards:
    """Test dashboard plotting functionality."""
    
    def test_plot_returns_dashboard_in_sample(self, multi_strategy_results):
        """Test returns dashboard for in-sample data."""
        # Arrange
        plotter = PortfolioPlotter(multi_strategy_results)
        
        # Act
        fig = plotter.plot_returns_dashboard(in_sample=True)
        
        # Assert
        assert fig is not None
        plt.close(fig)
    
    def test_plot_returns_dashboard_out_of_sample(self, multi_strategy_results):
        """Test returns dashboard for out-of-sample data."""
        # Arrange
        plotter = PortfolioPlotter(multi_strategy_results)
        
        # Act
        fig = plotter.plot_returns_dashboard(in_sample=False)
        
        # Assert
        assert fig is not None
        plt.close(fig)
    
    def test_plot_risk_dashboard_in_sample(self, multi_strategy_results):
        """Test risk dashboard for in-sample data."""
        # Arrange
        plotter = PortfolioPlotter(multi_strategy_results)
        
        # Act
        fig = plotter.plot_risk_dashboard(in_sample=True)
        
        # Assert
        assert fig is not None
        plt.close(fig)
    
    def test_plot_risk_dashboard_out_of_sample(self, multi_strategy_results):
        """Test risk dashboard for out-of-sample data."""
        # Arrange
        plotter = PortfolioPlotter(multi_strategy_results)
        
        # Act
        fig = plotter.plot_risk_dashboard(in_sample=False)
        
        # Assert
        assert fig is not None
        plt.close(fig)
    
    def test_plot_all_dashboards(self, multi_strategy_results):
        """Test plotting all dashboards."""
        # Arrange
        plotter = PortfolioPlotter(multi_strategy_results)
        
        # Act - should create both dashboards
        plotter.plot_all_dashboards(in_sample=True)
        
        # Assert - no exceptions raised
        plt.close('all')


# ============================================================================
# Test Benchmark Calculations
# ============================================================================

class TestBenchmarkCalculations:
    """Test benchmark equity calculation."""
    
    def test_calculate_benchmark_equity(self, sample_strategy_results, sample_benchmark_data):
        """Test benchmark equity calculation."""
        # Arrange
        plotter = PortfolioPlotter(sample_strategy_results, benchmark_data=sample_benchmark_data)
        
        # Act
        bench_equity = plotter._calculate_benchmark_equity()
        
        # Assert
        assert bench_equity is not None
        assert len(bench_equity) > 0
        assert bench_equity.iloc[0] == pytest.approx(plotter.total_capital, rel=0.01)
    
    def test_calculate_benchmark_equity_none_benchmark(self, sample_strategy_results):
        """Test benchmark equity calculation with no benchmark data."""
        # Arrange
        plotter = PortfolioPlotter(sample_strategy_results, benchmark_data=None)
        
        # Act
        bench_equity = plotter._calculate_benchmark_equity()
        
        # Assert
        assert bench_equity is None


# ============================================================================
# Test Risk Metrics Formatting
# ============================================================================

class TestRiskMetricsFormatting:
    """Test risk metrics table formatting."""
    
    def test_format_risk_metrics_table(self, sample_strategy_results):
        """Test risk metrics table formatting."""
        # Arrange
        plotter = PortfolioPlotter(sample_strategy_results)
        
        # Act
        metrics_text = plotter._format_risk_metrics_table()
        
        # Assert
        assert isinstance(metrics_text, str)
        assert len(metrics_text) > 0
        assert 'Sharpe Ratio' in metrics_text
        assert 'Max Drawdown' in metrics_text
        assert 'VaR' in metrics_text
    
    def test_format_risk_metrics_handles_zero_volatility(self, sample_dates):
        """Test risk metrics with zero volatility (constant returns)."""
        # Arrange
        constant_equity = np.ones(len(sample_dates)) * 10000
        mock_result = MockBacktestResult(constant_equity, sample_dates)
        strategy_results = {'Strategy1': {'result': mock_result, 'capital': 10000}}
        
        plotter = PortfolioPlotter(strategy_results)
        
        # Act
        metrics_text = plotter._format_risk_metrics_table()
        
        # Assert
        assert metrics_text is not None
        assert isinstance(metrics_text, str)


# ============================================================================
# Test Convenience Functions
# ============================================================================

class TestConvenienceFunctions:
    """Test convenience plotting functions."""
    
    def test_quick_equity_plot(self, sample_strategy_results):
        """Test quick_equity_plot convenience function."""
        # Arrange & Act
        fig = quick_equity_plot(sample_strategy_results)
        
        # Assert
        assert fig is not None
        plt.close(fig)
    
    def test_quick_drawdown_plot(self, sample_strategy_results):
        """Test quick_drawdown_plot convenience function."""
        # Arrange & Act
        fig = quick_drawdown_plot(sample_strategy_results)
        
        # Assert
        assert fig is not None
        plt.close(fig)
    
    def test_quick_heatmap(self, sample_strategy_results):
        """Test quick_heatmap convenience function."""
        # Arrange & Act
        fig = quick_heatmap(sample_strategy_results)
        
        # Assert
        assert fig is not None
        plt.close(fig)


# ============================================================================
# Test Edge Cases
# ============================================================================

class TestEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_single_data_point(self, single_date):
        """Test plotting with single data point."""
        # Arrange
        equity_values = [10000]
        mock_result = MockBacktestResult(equity_values, single_date)
        strategy_results = {'Strategy1': {'result': mock_result, 'capital': 10000}}
        
        plotter = PortfolioPlotter(strategy_results)
        
        # Act & Assert - should not raise errors
        fig = plotter.plot_equity_curves()
        assert fig is not None
        plt.close(fig)
    
    def test_short_time_series(self, short_dates):
        """Test plotting with very short time series."""
        # Arrange
        equity_values = [10000, 10100, 10050, 10200, 10150]
        mock_result = MockBacktestResult(equity_values, short_dates)
        strategy_results = {'Strategy1': {'result': mock_result, 'capital': 10000}}
        
        plotter = PortfolioPlotter(strategy_results)
        
        # Act & Assert - should not raise errors
        fig = plotter.plot_equity_curves()
        assert fig is not None
        plt.close(fig)
    
    def test_negative_returns_handling(self, sample_dates):
        """Test plotting with negative returns (losing strategy)."""
        # Arrange
        np.random.seed(42)
        # Declining equity
        equity_values = 10000 * (1 - np.random.uniform(0, 0.01, len(sample_dates))).cumprod()
        mock_result = MockBacktestResult(equity_values, sample_dates)
        strategy_results = {'Strategy1': {'result': mock_result, 'capital': 10000}}
        
        plotter = PortfolioPlotter(strategy_results)
        
        # Act & Assert - should handle negative returns
        fig = plotter.plot_rolling_metrics(metrics=['sharpe', 'returns'])
        assert fig is not None
        plt.close(fig)
    
    def test_zero_drawdown_handling(self, sample_dates):
        """Test drawdown plotting with no drawdown (monotonically increasing)."""
        # Arrange
        equity_values = 10000 + np.arange(len(sample_dates)) * 10
        mock_result = MockBacktestResult(equity_values, sample_dates)
        strategy_results = {'Strategy1': {'result': mock_result, 'capital': 10000}}
        
        plotter = PortfolioPlotter(strategy_results)
        
        # Act
        fig = plotter.plot_drawdown()
        
        # Assert
        assert fig is not None
        plt.close(fig)
    
    def test_large_volatility_handling(self, sample_dates):
        """Test plotting with extremely volatile returns."""
        # Arrange
        np.random.seed(42)
        # Very volatile returns
        returns = np.random.normal(0, 0.1, len(sample_dates))
        equity_values = 10000 * (1 + returns).cumprod()
        equity_values = np.maximum(equity_values, 1000)  # Floor at 1000
        mock_result = MockBacktestResult(equity_values, sample_dates)
        strategy_results = {'Strategy1': {'result': mock_result, 'capital': 10000}}
        
        plotter = PortfolioPlotter(strategy_results)
        
        # Act
        fig = plotter.plot_rolling_metrics(metrics=['volatility'])
        
        # Assert
        assert fig is not None
        plt.close(fig)
    
    def test_monthly_heatmap_with_partial_year(self):
        """Test monthly heatmap with data that doesn't span full years."""
        # Arrange - only 3 months of data
        dates = pd.date_range(start='2023-01-01', periods=90, freq='D')
        equity_values = 10000 * (1 + np.random.randn(len(dates)).cumsum() * 0.01)
        equity_values = np.maximum(equity_values, 5000)
        mock_result = MockBacktestResult(equity_values, dates)
        strategy_results = {'Strategy1': {'result': mock_result, 'capital': 10000}}
        
        plotter = PortfolioPlotter(strategy_results)
        
        # Act
        fig = plotter.plot_monthly_returns_heatmap()
        
        # Assert
        assert fig is not None
        plt.close(fig)
    
    def test_all_negative_profit_contributions(self, sample_dates):
        """Test returns dashboard when all strategies lose money."""
        # Arrange
        results = {}
        np.random.seed(42)
        
        for i in range(3):
            # All declining
            equity_values = 5000 * (1 - np.random.uniform(0, 0.01, len(sample_dates))).cumprod()
            mock_result = MockBacktestResult(equity_values, sample_dates)
            results[f'Strategy{i+1}'] = {'result': mock_result, 'capital': 5000}
        
        plotter = PortfolioPlotter(results)
        
        # Act
        fig = plotter.plot_returns_dashboard()
        
        # Assert
        assert fig is not None
        plt.close(fig)
    
    def test_benchmark_with_missing_dates(self, sample_strategy_results):
        """Test benchmark plotting when benchmark has missing dates."""
        # Arrange - benchmark with fewer dates
        dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
        benchmark_data = pd.DataFrame({
            'Close': 100 + np.arange(100) * 0.1,
            'Open': 100 + np.arange(100) * 0.1,
            'High': 101 + np.arange(100) * 0.1,
            'Low': 99 + np.arange(100) * 0.1
        }, index=dates)
        
        plotter = PortfolioPlotter(sample_strategy_results, benchmark_data=benchmark_data)
        
        # Act
        fig = plotter.plot_equity_curves(show_benchmark=True)
        
        # Assert
        assert fig is not None
        plt.close(fig)
    
    def test_rolling_window_larger_than_data(self, short_dates):
        """Test rolling metrics when window is larger than data."""
        # Arrange
        equity_values = [10000, 10100, 10050, 10200, 10150]
        mock_result = MockBacktestResult(equity_values, short_dates)
        strategy_results = {'Strategy1': {'result': mock_result, 'capital': 10000}}
        
        plotter = PortfolioPlotter(strategy_results)
        
        # Act - window of 100 days but only 5 days of data
        fig = plotter.plot_rolling_metrics(window=100, metrics=['sharpe'])
        
        # Assert - should complete without error (NaN values expected)
        assert fig is not None
        plt.close(fig)
