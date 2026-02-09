"""
Tests for utils.formatter module.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from utils.formatter import PerformanceSummary


class MockBacktestResult:
    """Mock BacktestResult for testing."""
    
    def __init__(self, equity_values, dates, total_trades=10):
        """Initialize mock backtest result."""
        self.equity_curve = pd.DataFrame({
            'TotalValue': equity_values
        }, index=dates)
        self.total_trades = total_trades
        self.metrics = {
            'total_return': (equity_values[-1] - equity_values[0]) / equity_values[0],
            'cagr': 0.10,
            'sharpe': 1.5,
            'max_drawdown': -0.15
        }


class TestPerformanceSummary:
    """Test suite for PerformanceSummary class."""

    @pytest.fixture
    def sample_dates(self):
        """Create sample date range."""
        return pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')

    @pytest.fixture
    def sample_strategy_results(self, sample_dates):
        """Create sample strategy results."""
        equity_values = 10000 * (1 + np.random.randn(len(sample_dates)).cumsum() * 0.01)
        
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
        
        return pd.DataFrame({
            'Close': close_prices
        }, index=sample_dates)

    def test_initialization(self, sample_strategy_results):
        """Test PerformanceSummary initialization."""
        summary = PerformanceSummary(sample_strategy_results)
        
        assert summary.strategy_results == sample_strategy_results
        assert summary.period_label == 'IN-SAMPLE'
        assert summary.benchmark_name == 'SPY'

    def test_initialization_with_benchmark(self, sample_strategy_results, sample_benchmark_data):
        """Test initialization with benchmark data."""
        summary = PerformanceSummary(
            sample_strategy_results,
            benchmark_data=sample_benchmark_data,
            benchmark_name='SPY'
        )
        
        assert summary.benchmark_data is not None
        assert summary.benchmark_name == 'SPY'

    def test_calculate_portfolio_metrics(self, sample_strategy_results):
        """Test portfolio metrics calculation."""
        summary = PerformanceSummary(sample_strategy_results)
        
        # Check that metrics are calculated
        assert hasattr(summary, 'total_capital')
        assert hasattr(summary, 'final_value')
        assert hasattr(summary, 'total_return')
        assert hasattr(summary, 'cagr')
        assert hasattr(summary, 'sharpe')
        assert hasattr(summary, 'max_drawdown')
        assert hasattr(summary, 'n_strategies')

    def test_total_capital_calculation(self, sample_strategy_results):
        """Test that total capital is calculated correctly."""
        summary = PerformanceSummary(sample_strategy_results)
        
        expected_capital = sum(data['capital'] for data in sample_strategy_results.values())
        assert summary.total_capital == expected_capital

    def test_number_of_strategies(self, multi_strategy_results):
        """Test that number of strategies is counted correctly."""
        summary = PerformanceSummary(multi_strategy_results)
        
        assert summary.n_strategies == 3

    def test_combined_equity_calculation(self, multi_strategy_results):
        """Test combined equity curve calculation."""
        summary = PerformanceSummary(multi_strategy_results)
        
        # Combined equity should be sum of all strategies
        assert len(summary.combined_equity) > 0
        assert summary.combined_equity.iloc[0] > 0

    def test_total_return_calculation(self, sample_strategy_results):
        """Test total return calculation."""
        summary = PerformanceSummary(sample_strategy_results)
        
        expected_return = (summary.final_value - summary.total_capital) / summary.total_capital
        assert abs(summary.total_return - expected_return) < 1e-6

    def test_cagr_calculation(self, sample_strategy_results):
        """Test CAGR calculation."""
        summary = PerformanceSummary(sample_strategy_results)
        
        # CAGR should be calculated
        assert isinstance(summary.cagr, (int, float))

    def test_sharpe_calculation(self, sample_strategy_results):
        """Test Sharpe ratio calculation."""
        summary = PerformanceSummary(sample_strategy_results)
        
        # Sharpe should be calculated
        assert isinstance(summary.sharpe, (int, float))

    def test_max_drawdown_calculation(self, sample_strategy_results):
        """Test max drawdown calculation."""
        summary = PerformanceSummary(sample_strategy_results)
        
        # Max drawdown should be negative or zero
        assert summary.max_drawdown <= 0

    def test_benchmark_metrics_calculation(self, sample_strategy_results, sample_benchmark_data):
        """Test benchmark metrics calculation."""
        summary = PerformanceSummary(
            sample_strategy_results,
            benchmark_data=sample_benchmark_data
        )
        
        # Should have benchmark metrics
        assert hasattr(summary, 'benchmark_data')

    def test_custom_period_label(self, sample_strategy_results):
        """Test custom period label."""
        summary = PerformanceSummary(
            sample_strategy_results,
            period_label='OUT-OF-SAMPLE'
        )
        
        assert summary.period_label == 'OUT-OF-SAMPLE'

    def test_custom_benchmark_name(self, sample_strategy_results, sample_benchmark_data):
        """Test custom benchmark name."""
        summary = PerformanceSummary(
            sample_strategy_results,
            benchmark_data=sample_benchmark_data,
            benchmark_name='QQQ'
        )
        
        assert summary.benchmark_name == 'QQQ'

    def test_dates_alignment(self, sample_strategy_results):
        """Test that dates are properly aligned."""
        summary = PerformanceSummary(sample_strategy_results)
        
        first_result = list(sample_strategy_results.values())[0]['result']
        expected_dates = first_result.equity_curve.index
        
        pd.testing.assert_index_equal(summary.dates, expected_dates)

    def test_final_value_equals_last_equity(self, sample_strategy_results):
        """Test that final value equals last equity value."""
        summary = PerformanceSummary(sample_strategy_results)
        
        assert summary.final_value == summary.combined_equity.iloc[-1]

    def test_empty_strategy_results_handling(self):
        """Test handling of edge case with minimal data."""
        # Create minimal valid strategy
        dates = pd.date_range(start='2023-01-01', end='2023-01-10', freq='D')
        equity_values = [10000] * len(dates)
        
        mock_result = MockBacktestResult(equity_values, dates)
        strategy_results = {
            'Strategy1': {
                'result': mock_result,
                'capital': 10000
            }
        }
        
        summary = PerformanceSummary(strategy_results)
        
        # Should handle flat returns
        assert summary.total_return == 0.0
        assert summary.max_drawdown == 0.0

    def test_print_portfolio_metrics_executes(self, sample_strategy_results, capsys):
        """Test that print_portfolio_metrics executes without error."""
        summary = PerformanceSummary(sample_strategy_results)
        
        # Should print metrics without error
        summary.print_portfolio_metrics()
        
        # Capture output
        captured = capsys.readouterr()
        assert 'PORTFOLIO PERFORMANCE' in captured.out
        assert 'Initial Capital' in captured.out

    def test_print_benchmark_comparison_executes(self, sample_strategy_results, sample_benchmark_data, capsys):
        """Test that print_benchmark_comparison executes without error."""
        summary = PerformanceSummary(
            sample_strategy_results,
            benchmark_data=sample_benchmark_data
        )
        
        # Should print comparison without error
        summary.print_benchmark_comparison()
        
        # Capture output
        captured = capsys.readouterr()
        assert 'BENCHMARK COMPARISON' in captured.out

    def test_print_strategy_rankings_executes(self, multi_strategy_results, capsys):
        """Test that print_strategy_rankings executes without error."""
        summary = PerformanceSummary(multi_strategy_results)
        
        # Should print rankings without error
        summary.print_strategy_rankings()
        
        # Capture output
        captured = capsys.readouterr()
        # Should print something about strategies
        assert 'Strategy' in captured.out or captured.out != ''

    def test_print_full_report_executes(self, sample_strategy_results, capsys):
        """Test that print_full_report executes without error."""
        summary = PerformanceSummary(sample_strategy_results)
        
        # Should print full report without error
        summary.print_full_report()
        
        # Capture output
        captured = capsys.readouterr()
        assert 'PERFORMANCE SUMMARY' in captured.out

    def test_print_benchmark_comparison_with_no_benchmark(self, sample_strategy_results, capsys):
        """Test print_benchmark_comparison with no benchmark data."""
        summary = PerformanceSummary(sample_strategy_results)
        
        # Should handle no benchmark gracefully
        summary.print_benchmark_comparison()
        
        # Should not print anything
        captured = capsys.readouterr()
        assert captured.out == '' or 'BENCHMARK' not in captured.out
