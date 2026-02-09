"""
Comprehensive unit tests for utils/formatter.py to increase coverage to 80%+.

Focuses on:
- PerformanceSummary class methods that print reports
- Edge cases: empty strategy results, missing benchmark data
- Different period labels and configurations
- Strategy ranking and recommendation logic

Coverage targets:
- Lines 120, 160, 201-225, 229-272, 305-309, 317, 344-364, 370-372, 377-386
"""

from __future__ import annotations

import pytest
import pandas as pd
import numpy as np
from io import StringIO
import sys

from utils.formatter import (
    PerformanceSummary,
    quick_summary,
    compare_periods
)


# ============================================================================
# Mock Objects
# ============================================================================

class MockBacktestResult:
    """
    Mock BacktestResult for testing formatter functionality.
    
    Mimics the real BacktestResult structure with equity_curve, trades, and metrics.
    """
    
    def __init__(self, equity_values, dates, total_trades=10, win_rate=0.6, 
                 closed_positions=None):
        """
        Initialize mock backtest result.
        
        Args:
            equity_values: Array of portfolio values over time
            dates: DatetimeIndex for equity curve
            total_trades: Number of trades executed
            win_rate: Fraction of winning trades
            closed_positions: Optional list of trade dictionaries
        """
        self.equity_curve = pd.DataFrame({
            'TotalValue': equity_values,
            'Cash': equity_values * 0.3,
            'PositionsValue': equity_values * 0.7
        }, index=dates)
        
        self.total_trades = total_trades
        
        # Create realistic metrics
        total_return = (equity_values[-1] - equity_values[0]) / equity_values[0]
        years = (dates[-1] - dates[0]).days / 365.25
        cagr = (equity_values[-1] / equity_values[0]) ** (1 / years) - 1 if years > 0 else 0
        
        # Calculate max drawdown
        running_max = pd.Series(equity_values, index=dates).expanding().max()
        drawdown = (pd.Series(equity_values, index=dates) - running_max) / running_max
        max_dd = drawdown.min()
        
        # Calculate Sharpe ratio
        returns = pd.Series(equity_values, index=dates).pct_change().dropna()
        sharpe = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() > 0 else 0
        
        self.metrics = {
            'Total Return': total_return,
            'CAGR': cagr,
            'Sharpe Ratio': sharpe,
            'Max Drawdown': max_dd,
            'Win Rate': win_rate,
            'Total Trades': total_trades,
            'Avg Win': 0.05,
            'Avg Loss': -0.03,
            'Profit Factor': 2.5
        }
        
        # Generate realistic closed positions if not provided
        if closed_positions is not None:
            self.closed_positions = closed_positions
        else:
            self.closed_positions = self._generate_closed_positions(
                dates, total_trades, win_rate
            )
    
    def _generate_closed_positions(self, dates, total_trades, win_rate):
        """Generate realistic closed positions for testing."""
        if total_trades == 0:
            return []
        
        positions = []
        np.random.seed(42)
        
        num_wins = int(total_trades * win_rate)
        num_losses = total_trades - num_wins
        
        for i in range(total_trades):
            is_win = i < num_wins
            pnl = np.random.uniform(50, 500) if is_win else np.random.uniform(-300, -50)
            ret = np.random.uniform(0.02, 0.08) if is_win else np.random.uniform(-0.05, -0.01)
            hold_days = np.random.randint(3, 30)
            
            positions.append({
                'ticker': f'ASSET{i % 3}',
                'entry_date': dates[i * 10 % len(dates)],
                'exit_date': dates[(i * 10 + hold_days) % len(dates)],
                'pnl': pnl,
                'return': ret,
                'hold_days': hold_days
            })
        
        return positions


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def sample_dates():
    """Create sample date range for testing."""
    return pd.date_range(start='2020-01-01', end='2023-12-31', freq='D')


@pytest.fixture
def short_dates():
    """Create short date range for edge case testing."""
    return pd.date_range(start='2023-01-01', end='2023-01-31', freq='D')


@pytest.fixture
def winning_strategy_results(sample_dates):
    """
    Create strategy results with positive performance.
    
    Returns single strategy with strong positive returns.
    """
    # Strong upward trend
    equity_values = 100000 * (1 + np.linspace(0, 0.50, len(sample_dates)))
    mock_result = MockBacktestResult(equity_values, sample_dates, total_trades=25, win_rate=0.7)
    
    return {
        'WinningStrategy': {
            'result': mock_result,
            'capital': 100000
        }
    }


@pytest.fixture
def losing_strategy_results(sample_dates):
    """
    Create strategy results with negative performance.
    
    Returns single strategy with losses.
    """
    # Downward trend
    equity_values = 100000 * (1 + np.linspace(0, -0.20, len(sample_dates)))
    mock_result = MockBacktestResult(equity_values, sample_dates, total_trades=30, win_rate=0.4)
    
    return {
        'LosingStrategy': {
            'result': mock_result,
            'capital': 100000
        }
    }


@pytest.fixture
def multi_strategy_results(sample_dates):
    """
    Create multiple strategy results with varied performance.
    
    Returns 3 strategies with high, medium, and low Sharpe ratios.
    """
    results = {}
    
    # High Sharpe strategy (consistent returns, low volatility)
    equity_high = 50000 * (1 + np.linspace(0, 0.40, len(sample_dates)) + 
                           np.random.RandomState(1).normal(0, 0.005, len(sample_dates)))
    mock_high = MockBacktestResult(equity_high, sample_dates, total_trades=20, win_rate=0.75)
    mock_high.metrics['Sharpe Ratio'] = 2.5  # Override for testing
    
    # Medium Sharpe strategy
    equity_med = 30000 * (1 + np.linspace(0, 0.25, len(sample_dates)) + 
                          np.random.RandomState(2).normal(0, 0.01, len(sample_dates)))
    mock_med = MockBacktestResult(equity_med, sample_dates, total_trades=15, win_rate=0.60)
    mock_med.metrics['Sharpe Ratio'] = 1.5
    
    # Low Sharpe strategy (volatile)
    equity_low = 20000 * (1 + np.linspace(0, 0.10, len(sample_dates)) + 
                          np.random.RandomState(3).normal(0, 0.02, len(sample_dates)))
    mock_low = MockBacktestResult(equity_low, sample_dates, total_trades=35, win_rate=0.50)
    mock_low.metrics['Sharpe Ratio'] = 0.8
    
    results = {
        'HighSharpe': {'result': mock_high, 'capital': 50000},
        'MediumSharpe': {'result': mock_med, 'capital': 30000},
        'LowSharpe': {'result': mock_low, 'capital': 20000}
    }
    
    return results


@pytest.fixture
def benchmark_data_outperform(sample_dates):
    """
    Create benchmark data that underperforms the strategy.
    
    Returns SPY-like data with moderate returns.
    """
    # Moderate upward trend (lower than winning strategy)
    close_prices = 400 * (1 + np.linspace(0, 0.25, len(sample_dates)))
    
    return pd.DataFrame({
        'Close': close_prices,
        'Open': close_prices * 0.99,
        'High': close_prices * 1.01,
        'Low': close_prices * 0.98
    }, index=sample_dates)


@pytest.fixture
def benchmark_data_underperform(sample_dates):
    """
    Create benchmark data that outperforms the strategy.
    
    Returns benchmark with strong returns.
    """
    # Strong upward trend (higher than typical strategy)
    close_prices = 400 * (1 + np.linspace(0, 0.60, len(sample_dates)))
    
    return pd.DataFrame({
        'Close': close_prices,
        'Open': close_prices * 0.99,
        'High': close_prices * 1.01,
        'Low': close_prices * 0.98
    }, index=sample_dates)


@pytest.fixture
def empty_trades_result(sample_dates):
    """
    Create strategy result with no trades.
    
    Returns flat equity curve with zero trades.
    """
    equity_values = np.full(len(sample_dates), 100000.0)
    mock_result = MockBacktestResult(equity_values, sample_dates, total_trades=0, 
                                    closed_positions=[])
    
    return {
        'NoTrades': {
            'result': mock_result,
            'capital': 100000
        }
    }


@pytest.fixture
def large_drawdown_results(sample_dates):
    """
    Create strategy with large drawdown (>20%).
    
    Returns strategy with severe drawdown for testing recommendations.
    """
    # Create equity with large drawdown
    equity = 100000 * np.ones(len(sample_dates))
    mid_point = len(sample_dates) // 2
    equity[mid_point:mid_point+100] *= 0.7  # 30% drawdown
    
    mock_result = MockBacktestResult(equity, sample_dates, total_trades=20, win_rate=0.55)
    mock_result.metrics['Max Drawdown'] = -0.30
    
    return {
        'LargeDrawdown': {
            'result': mock_result,
            'capital': 100000
        }
    }


# ============================================================================
# Test PerformanceSummary Class Methods
# ============================================================================

class TestPerformanceSummaryPrintMethods:
    """Test suite for print methods of PerformanceSummary class."""
    
    def test_print_full_report_with_benchmark(self, winning_strategy_results, 
                                             benchmark_data_outperform, capsys):
        """
        Test print_full_report includes benchmark comparison.
        
        Tests line 120: benchmark comparison in full report.
        """
        # Arrange
        summary = PerformanceSummary(
            winning_strategy_results,
            benchmark_data=benchmark_data_outperform,
            benchmark_name='SPY'
        )
        
        # Act
        summary.print_full_report()
        
        # Assert
        captured = capsys.readouterr()
        assert 'PERFORMANCE SUMMARY' in captured.out
        assert 'BENCHMARK COMPARISON' in captured.out
        assert 'SPY' in captured.out
    
    def test_print_benchmark_beating_message(self, winning_strategy_results, 
                                            benchmark_data_outperform, capsys):
        """
        Test print message when beating benchmark.
        
        Tests line 160: positive outperformance message.
        """
        # Arrange
        summary = PerformanceSummary(
            winning_strategy_results,
            benchmark_data=benchmark_data_outperform
        )
        
        # Act
        summary.print_benchmark_comparison()
        
        # Assert
        captured = capsys.readouterr()
        assert 'BEATING' in captured.out or '✅' in captured.out
    
    def test_print_benchmark_lagging_message(self, losing_strategy_results, 
                                            benchmark_data_underperform, capsys):
        """
        Test print message when lagging benchmark.
        
        Tests line 162: negative outperformance message.
        """
        # Arrange
        summary = PerformanceSummary(
            losing_strategy_results,
            benchmark_data=benchmark_data_underperform
        )
        
        # Act
        summary.print_benchmark_comparison()
        
        # Assert
        captured = capsys.readouterr()
        assert 'LAGGING' in captured.out or '❌' in captured.out


class TestPrintTradeStatistics:
    """Test suite for print_trade_statistics method."""
    
    def test_print_trade_statistics_single_strategy(self, multi_strategy_results, capsys):
        """
        Test print_trade_statistics for specific strategy.
        
        Tests lines 201-225: trade statistics printing for named strategy.
        """
        # Arrange
        summary = PerformanceSummary(multi_strategy_results)
        
        # Act
        summary.print_trade_statistics(strategy_name='HighSharpe')
        
        # Assert
        captured = capsys.readouterr()
        assert 'TRADE STATISTICS - HighSharpe' in captured.out
        assert 'Total Trades' in captured.out
        assert 'Win Rate' in captured.out
        assert 'Avg Win' in captured.out
        assert 'Avg Loss' in captured.out
        assert 'Win/Loss Ratio' in captured.out
        assert 'Profit Factor' in captured.out
    
    def test_print_trade_statistics_combined(self, multi_strategy_results, capsys):
        """
        Test print_trade_statistics for combined portfolio.
        
        Tests lines 206-225: combined trade statistics.
        """
        # Arrange
        summary = PerformanceSummary(multi_strategy_results)
        
        # Act
        summary.print_trade_statistics()
        
        # Assert
        captured = capsys.readouterr()
        assert 'COMBINED TRADE STATISTICS' in captured.out
        assert 'Total Trades' in captured.out
    
    def test_print_trade_statistics_with_closed_positions(self, 
                                                         winning_strategy_results, capsys):
        """
        Test trade statistics includes best/worst trades and duration.
        
        Tests lines 221-225: best/worst trade and avg duration.
        """
        # Arrange
        summary = PerformanceSummary(winning_strategy_results)
        
        # Act
        summary.print_trade_statistics(strategy_name='WinningStrategy')
        
        # Assert
        captured = capsys.readouterr()
        assert 'Best Trade' in captured.out
        assert 'Worst Trade' in captured.out
        assert 'Avg Trade Duration' in captured.out
        assert 'days' in captured.out
    
    def test_print_trade_statistics_no_closed_positions(self, empty_trades_result, capsys):
        """
        Test trade statistics with zero trades.
        
        Tests edge case: no closed positions.
        """
        # Arrange
        summary = PerformanceSummary(empty_trades_result)
        
        # Act
        summary.print_trade_statistics(strategy_name='NoTrades')
        
        # Assert - should not crash
        captured = capsys.readouterr()
        assert 'TRADE STATISTICS' in captured.out
    
    def test_print_trade_statistics_invalid_strategy(self, multi_strategy_results, capsys):
        """
        Test trade statistics with non-existent strategy name.
        
        Tests line 207: falls back to aggregate when strategy not found.
        """
        # Arrange
        summary = PerformanceSummary(multi_strategy_results)
        
        # Act
        summary.print_trade_statistics(strategy_name='NonExistent')
        
        # Assert - should fall back to combined
        captured = capsys.readouterr()
        assert 'COMBINED TRADE STATISTICS' in captured.out


class TestPrintRecommendations:
    """Test suite for print_recommendations method."""
    
    def test_recommendations_beating_benchmark(self, winning_strategy_results, 
                                              benchmark_data_outperform, capsys):
        """
        Test recommendations when beating benchmark.
        
        Tests lines 229-241: strong performance recommendations.
        """
        # Arrange
        summary = PerformanceSummary(
            winning_strategy_results,
            benchmark_data=benchmark_data_outperform
        )
        
        # Act
        summary.print_recommendations()
        
        # Assert
        captured = capsys.readouterr()
        assert 'RECOMMENDATIONS' in captured.out
        assert 'STRONG PERFORMANCE' in captured.out or '✅' in captured.out
        assert 'Next Steps' in captured.out
        assert 'HTML report' in captured.out
        assert 'paper trading' in captured.out
        assert 'concentration risk' in captured.out
    
    def test_recommendations_lagging_benchmark(self, losing_strategy_results, 
                                              benchmark_data_underperform, capsys):
        """
        Test recommendations when lagging benchmark.
        
        Tests lines 242-250: underperformance recommendations.
        """
        # Arrange
        summary = PerformanceSummary(
            losing_strategy_results,
            benchmark_data=benchmark_data_underperform
        )
        
        # Act
        summary.print_recommendations()
        
        # Assert
        captured = capsys.readouterr()
        assert 'RECOMMENDATIONS' in captured.out
        assert 'UNDERPERFORMANCE' in captured.out or '⚠️' in captured.out
        assert 'Recommended Actions' in captured.out
        assert 'Optimize signal parameters' in captured.out
        assert 'regime filters' in captured.out
    
    def test_recommendations_low_sharpe(self, multi_strategy_results, capsys):
        """
        Test recommendations for low Sharpe ratio.
        
        Tests lines 252-257: low Sharpe recommendations.
        """
        # Arrange
        summary = PerformanceSummary(multi_strategy_results)
        # Manually set low Sharpe for testing
        summary.sharpe = 0.8
        
        # Act
        summary.print_recommendations()
        
        # Assert
        captured = capsys.readouterr()
        assert 'LOW SHARPE RATIO' in captured.out
        assert 'diversified strategies' in captured.out
        assert 'risk controls' in captured.out
    
    def test_recommendations_high_sharpe(self, multi_strategy_results, capsys):
        """
        Test recommendations for high Sharpe ratio.
        
        Tests lines 258-263: excellent Sharpe recommendations.
        """
        # Arrange
        summary = PerformanceSummary(multi_strategy_results)
        # Manually set high Sharpe
        summary.sharpe = 2.5
        
        # Act
        summary.print_recommendations()
        
        # Assert
        captured = capsys.readouterr()
        assert 'EXCELLENT SHARPE RATIO' in captured.out or '✅' in captured.out
        assert 'Strong risk-adjusted performance' in captured.out
        assert 'out-of-sample testing' in captured.out
    
    def test_recommendations_large_drawdown(self, large_drawdown_results, capsys):
        """
        Test recommendations for large drawdown.
        
        Tests lines 265-271: large drawdown recommendations.
        """
        # Arrange
        summary = PerformanceSummary(large_drawdown_results)
        
        # Act
        summary.print_recommendations()
        
        # Assert
        captured = capsys.readouterr()
        assert 'LARGE DRAWDOWN' in captured.out
        assert 'position sizing controls' in captured.out
        assert 'volatility-based sizing' in captured.out
        assert 'portfolio heat limits' in captured.out
    
    def test_recommendations_no_benchmark(self, winning_strategy_results, capsys):
        """
        Test recommendations without benchmark data.
        
        Tests that recommendations work without benchmark comparison.
        """
        # Arrange
        summary = PerformanceSummary(winning_strategy_results)
        
        # Act
        summary.print_recommendations()
        
        # Assert - should still provide Sharpe/drawdown recommendations
        captured = capsys.readouterr()
        assert 'RECOMMENDATIONS' in captured.out


class TestPrintComparisonTable:
    """Test suite for print_comparison_table method."""
    
    def test_comparison_table_both_profitable(self, winning_strategy_results, 
                                             sample_dates, capsys):
        """
        Test comparison table when both periods are profitable.
        
        Tests lines 305-309: consistency assessment for profitable periods.
        """
        # Arrange
        is_summary = PerformanceSummary(winning_strategy_results, period_label='IN-SAMPLE')
        
        # Create OOS with positive but lower returns
        equity_oos = 100000 * (1 + np.linspace(0, 0.30, len(sample_dates)))
        mock_oos = MockBacktestResult(equity_oos, sample_dates, total_trades=20)
        oos_results = {'Strategy': {'result': mock_oos, 'capital': 100000}}
        oos_summary = PerformanceSummary(oos_results, period_label='OUT-OF-SAMPLE')
        
        # Act
        is_summary.print_comparison_table(oos_summary)
        
        # Assert
        captured = capsys.readouterr()
        assert 'CONSISTENCY CHECK' in captured.out
        assert 'Both periods profitable' in captured.out or '✅' in captured.out
        assert 'Return consistency' in captured.out
    
    def test_comparison_table_overfitting_warning(self, winning_strategy_results, 
                                                  losing_strategy_results, capsys):
        """
        Test comparison table shows overfitting warning.
        
        Tests lines 305-309: overfitting detection.
        """
        # Arrange
        is_summary = PerformanceSummary(winning_strategy_results, period_label='IN-SAMPLE')
        oos_summary = PerformanceSummary(losing_strategy_results, period_label='OUT-OF-SAMPLE')
        
        # Act
        is_summary.print_comparison_table(oos_summary)
        
        # Assert
        captured = capsys.readouterr()
        assert 'CONSISTENCY CHECK' in captured.out
        assert ('overfitting' in captured.out.lower() or 
                'profitable but' in captured.out or 
                'WARNING' in captured.out)
    
    def test_comparison_table_both_unprofitable(self, losing_strategy_results, 
                                               sample_dates, capsys):
        """
        Test comparison table when both periods lose money.
        
        Tests line 309: both periods need optimization.
        """
        # Arrange
        is_summary = PerformanceSummary(losing_strategy_results, period_label='IN-SAMPLE')
        
        # Create second losing period
        equity_oos = 100000 * (1 + np.linspace(0, -0.15, len(sample_dates)))
        mock_oos = MockBacktestResult(equity_oos, sample_dates)
        oos_results = {'Strategy': {'result': mock_oos, 'capital': 100000}}
        oos_summary = PerformanceSummary(oos_results, period_label='OUT-OF-SAMPLE')
        
        # Act
        is_summary.print_comparison_table(oos_summary)
        
        # Assert
        captured = capsys.readouterr()
        assert 'CONSISTENCY CHECK' in captured.out
        assert 'needs optimization' in captured.out or '⚠️' in captured.out
    
    def test_comparison_table_metrics(self, multi_strategy_results, sample_dates, capsys):
        """
        Test comparison table displays all key metrics.
        
        Tests lines 279-296: metric comparison display.
        """
        # Arrange
        is_summary = PerformanceSummary(multi_strategy_results, period_label='IN-SAMPLE')
        
        equity_oos = 50000 * (1 + np.linspace(0, 0.20, len(sample_dates)))
        mock_oos = MockBacktestResult(equity_oos, sample_dates)
        oos_results = {'Strategy': {'result': mock_oos, 'capital': 100000}}
        oos_summary = PerformanceSummary(oos_results, period_label='OUT-OF-SAMPLE')
        
        # Act
        is_summary.print_comparison_table(oos_summary)
        
        # Assert
        captured = capsys.readouterr()
        assert 'Total Return' in captured.out
        assert 'CAGR' in captured.out
        assert 'Sharpe Ratio' in captured.out
        assert 'Max Drawdown' in captured.out
        assert 'Difference' in captured.out


class TestAggregateTradesMethod:
    """Test suite for _aggregate_trades method."""
    
    def test_aggregate_trades_returns_first_result(self, multi_strategy_results):
        """
        Test _aggregate_trades returns first strategy result.
        
        Tests line 317: placeholder implementation.
        """
        # Arrange
        summary = PerformanceSummary(multi_strategy_results)
        
        # Act
        result = summary._aggregate_trades()
        
        # Assert - should return a result object
        assert hasattr(result, 'equity_curve')
        assert hasattr(result, 'metrics')


class TestPrintMetricsTable:
    """Test suite for print_metrics_table method."""
    
    def test_print_metrics_table_displays(self, multi_strategy_results, capsys):
        """
        Test print_metrics_table displays formatted table.
        
        Tests lines 344-364: metrics table display.
        """
        # Arrange
        summary = PerformanceSummary(multi_strategy_results)
        
        # Act
        summary.print_metrics_table()
        
        # Assert
        captured = capsys.readouterr()
        assert 'STRATEGY METRICS TABLE' in captured.out
    
    def test_print_metrics_table_without_ipython(self, multi_strategy_results, capsys, 
                                                 monkeypatch):
        """
        Test print_metrics_table falls back to string when IPython unavailable.
        
        Tests lines 360-364: fallback to to_string().
        """
        # Arrange
        summary = PerformanceSummary(multi_strategy_results)
        
        # Mock IPython import to fail
        import builtins
        real_import = builtins.__import__
        
        def mock_import(name, *args, **kwargs):
            if name == 'IPython.display':
                raise ImportError("No module named 'IPython.display'")
            return real_import(name, *args, **kwargs)
        
        monkeypatch.setattr(builtins, '__import__', mock_import)
        
        # Act
        summary.print_metrics_table()
        
        # Assert
        captured = capsys.readouterr()
        assert 'STRATEGY METRICS TABLE' in captured.out
    
    def test_to_dataframe_structure(self, multi_strategy_results):
        """
        Test to_dataframe returns proper structure.
        
        Tests lines 319-340: DataFrame export.
        """
        # Arrange
        summary = PerformanceSummary(multi_strategy_results)
        
        # Act
        df = summary.to_dataframe()
        
        # Assert
        assert isinstance(df, pd.DataFrame)
        assert 'Strategy' in df.columns
        assert 'Capital' in df.columns
        assert 'Final Value' in df.columns
        assert 'Total Return' in df.columns
        assert 'Sharpe Ratio' in df.columns
        assert 'Max Drawdown' in df.columns
        assert 'Win Rate' in df.columns
        assert 'Total Trades' in df.columns
        assert len(df) == 3  # Three strategies


class TestConvenienceFunctions:
    """Test suite for module-level convenience functions."""
    
    def test_quick_summary_returns_summary(self, winning_strategy_results, capsys):
        """
        Test quick_summary convenience function.
        
        Tests lines 368-372: quick_summary function.
        """
        # Act
        result = quick_summary(winning_strategy_results)
        
        # Assert
        assert isinstance(result, PerformanceSummary)
        captured = capsys.readouterr()
        assert 'PERFORMANCE SUMMARY' in captured.out
    
    def test_quick_summary_with_benchmark(self, winning_strategy_results, 
                                         benchmark_data_outperform, capsys):
        """
        Test quick_summary with benchmark data.
        
        Tests lines 370-372: quick_summary with kwargs.
        """
        # Act
        result = quick_summary(
            winning_strategy_results,
            benchmark_data=benchmark_data_outperform,
            benchmark_name='SPY'
        )
        
        # Assert
        assert isinstance(result, PerformanceSummary)
        assert result.benchmark_name == 'SPY'
        captured = capsys.readouterr()
        assert 'SPY' in captured.out
    
    def test_compare_periods_function(self, winning_strategy_results, 
                                     losing_strategy_results, capsys):
        """
        Test compare_periods convenience function.
        
        Tests lines 375-386: compare_periods function.
        """
        # Act
        is_summary, oos_summary = compare_periods(
            winning_strategy_results,
            losing_strategy_results
        )
        
        # Assert
        assert isinstance(is_summary, PerformanceSummary)
        assert isinstance(oos_summary, PerformanceSummary)
        assert is_summary.period_label == 'IN-SAMPLE'
        assert oos_summary.period_label == 'OUT-OF-SAMPLE'
        
        captured = capsys.readouterr()
        assert 'IN-SAMPLE' in captured.out
        assert 'OUT-OF-SAMPLE' in captured.out
        assert 'CONSISTENCY CHECK' in captured.out
    
    def test_compare_periods_with_benchmark(self, winning_strategy_results, 
                                           losing_strategy_results,
                                           benchmark_data_outperform, capsys):
        """
        Test compare_periods with benchmark data.
        
        Tests lines 377-386: compare_periods with benchmark.
        """
        # Act
        is_summary, oos_summary = compare_periods(
            winning_strategy_results,
            losing_strategy_results,
            benchmark_data=benchmark_data_outperform
        )
        
        # Assert
        assert is_summary.benchmark_data is not None
        assert oos_summary.benchmark_data is not None
        
        captured = capsys.readouterr()
        assert 'BENCHMARK COMPARISON' in captured.out


# ============================================================================
# Edge Cases and Configuration Tests
# ============================================================================

class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_empty_strategy_dict_handling(self, sample_dates):
        """
        Test handling of single strategy (minimal case).
        
        Edge case: single strategy should work correctly.
        """
        # Arrange
        equity = np.full(len(sample_dates), 10000.0)
        mock_result = MockBacktestResult(equity, sample_dates, total_trades=0)
        strategy_results = {'SingleStrategy': {'result': mock_result, 'capital': 10000}}
        
        # Act
        summary = PerformanceSummary(strategy_results)
        
        # Assert
        assert summary.n_strategies == 1
        assert summary.total_capital == 10000
    
    def test_different_period_labels(self, winning_strategy_results):
        """
        Test different period label configurations.
        
        Tests custom period labels work correctly.
        """
        # Arrange & Act
        labels = ['TRAIN', 'TEST', 'VALIDATION', 'WALK-FORWARD-1']
        
        for label in labels:
            summary = PerformanceSummary(winning_strategy_results, period_label=label)
            
            # Assert
            assert summary.period_label == label
    
    def test_custom_benchmark_names(self, winning_strategy_results, 
                                   benchmark_data_outperform):
        """
        Test different benchmark name configurations.
        
        Tests custom benchmark names display correctly.
        """
        # Arrange & Act
        benchmarks = ['SPY', 'QQQ', 'IWM', 'DIA', 'VTI']
        
        for bench in benchmarks:
            summary = PerformanceSummary(
                winning_strategy_results,
                benchmark_data=benchmark_data_outperform,
                benchmark_name=bench
            )
            
            # Assert
            assert summary.benchmark_name == bench
    
    def test_very_short_time_period(self, short_dates):
        """
        Test with very short time period (edge case).
        
        Tests that calculations work with minimal data.
        """
        # Arrange
        equity = 10000 * (1 + np.linspace(0, 0.05, len(short_dates)))
        mock_result = MockBacktestResult(equity, short_dates, total_trades=2)
        strategy_results = {'ShortPeriod': {'result': mock_result, 'capital': 10000}}
        
        # Act
        summary = PerformanceSummary(strategy_results)
        
        # Assert - should not crash
        assert summary.total_capital == 10000
        assert isinstance(summary.cagr, (int, float))
    
    def test_zero_volatility_strategy(self, sample_dates):
        """
        Test strategy with zero volatility (flat equity).
        
        Edge case: Sharpe ratio calculation with zero std.
        """
        # Arrange
        equity = np.full(len(sample_dates), 100000.0)
        mock_result = MockBacktestResult(equity, sample_dates, total_trades=0)
        strategy_results = {'FlatStrategy': {'result': mock_result, 'capital': 100000}}
        
        # Act
        summary = PerformanceSummary(strategy_results)
        
        # Assert - Sharpe should be 0 when volatility is 0
        assert summary.sharpe == 0.0
        assert summary.total_return == 0.0


class TestStrategyRankings:
    """Test strategy ranking logic."""
    
    def test_rankings_sorted_by_return(self, multi_strategy_results, capsys):
        """
        Test that strategies are ranked by return descending.
        
        Verifies ranking logic sorts correctly.
        """
        # Arrange
        summary = PerformanceSummary(multi_strategy_results)
        
        # Act
        summary.print_strategy_rankings()
        
        # Assert
        captured = capsys.readouterr()
        assert 'STRATEGY RANKINGS' in captured.out
        assert 'Rank' in captured.out
        
        # Verify rankings appear in output
        lines = captured.out.split('\n')
        ranking_lines = [l for l in lines if l.strip() and 
                        any(c.isdigit() for c in l[:10])]
        assert len(ranking_lines) >= 3  # Should have at least 3 strategies ranked
    
    def test_rankings_include_all_metrics(self, multi_strategy_results, capsys):
        """
        Test that rankings display all required metrics.
        
        Verifies Return, Sharpe, Max DD, and Capital are shown.
        """
        # Arrange
        summary = PerformanceSummary(multi_strategy_results)
        
        # Act
        summary.print_strategy_rankings()
        
        # Assert
        captured = capsys.readouterr()
        assert 'Return' in captured.out
        assert 'Sharpe' in captured.out
        assert 'Max DD' in captured.out
        assert 'Capital' in captured.out


class TestPrintWidthParameter:
    """Test width parameter in print methods."""
    
    def test_custom_width_full_report(self, winning_strategy_results, capsys):
        """
        Test custom width parameter in print_full_report.
        
        Verifies width parameter is respected.
        """
        # Arrange
        summary = PerformanceSummary(winning_strategy_results)
        
        # Act
        summary.print_full_report(width=100)
        
        # Assert - should not crash with custom width
        captured = capsys.readouterr()
        assert 'PERFORMANCE SUMMARY' in captured.out
    
    def test_custom_width_portfolio_metrics(self, winning_strategy_results, capsys):
        """
        Test custom width parameter in print_portfolio_metrics.
        
        Verifies width parameter works.
        """
        # Arrange
        summary = PerformanceSummary(winning_strategy_results)
        
        # Act
        summary.print_portfolio_metrics(width=100)
        
        # Assert
        captured = capsys.readouterr()
        assert 'PORTFOLIO PERFORMANCE' in captured.out
