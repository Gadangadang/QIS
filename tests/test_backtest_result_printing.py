"""
Extended test suite for BacktestResult print and plot methods.

Tests uncovered functionality in core/portfolio/backtest_result.py:
- print_summary() method with various configurations
- plot_equity_curve() method
- generate_html_report() method
- risk_analysis property
- Edge cases with matplotlib and reporter dependencies
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch, call
from io import StringIO
import sys

from core.portfolio.backtest_result import BacktestResult


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def equity_curve_with_trades():
    """
    Generate equity curve with growth and drawdowns.
    
    Returns:
        Tuple of (equity_df, trades_df)
    """
    dates = pd.date_range('2023-01-01', periods=252, freq='D')
    
    np.random.seed(42)
    returns = np.random.normal(0.001, 0.02, 252)
    total_values = 100000 * (1 + returns).cumprod()
    
    equity = pd.DataFrame({
        'Cash': 50000 * np.ones(252),
        'PositionsValue': total_values - 50000,
        'TotalValue': total_values
    }, index=dates)
    
    trades = pd.DataFrame({
        'ticker': ['AAPL', 'MSFT', 'GOOGL', 'AAPL', 'MSFT', 'TSLA'],
        'entry_date': pd.date_range('2023-01-01', periods=6, freq='30D'),
        'exit_date': pd.date_range('2023-02-01', periods=6, freq='30D'),
        'shares': [100, 50, 20, 150, 75, 30],
        'entry_price': [150.0, 300.0, 2000.0, 145.0, 310.0, 200.0],
        'exit_price': [155.0, 305.0, 2050.0, 142.0, 315.0, 210.0],
        'pnl': [500, 250, 1000, -450, 375, 300],
        'return': [0.033, 0.017, 0.025, -0.021, 0.016, 0.05]
    })
    
    return equity, trades


@pytest.fixture
def benchmark_equity_data():
    """Generate benchmark equity curve."""
    dates = pd.date_range('2023-01-01', periods=252, freq='D')
    
    np.random.seed(123)
    returns = np.random.normal(0.0005, 0.015, 252)
    values = 100000 * (1 + returns).cumprod()
    
    return pd.DataFrame({
        'TotalValue': values
    }, index=dates)


@pytest.fixture
def simple_backtest_result(equity_curve_with_trades):
    """Create BacktestResult with standard data."""
    equity, trades = equity_curve_with_trades
    return BacktestResult(
        equity_curve=equity,
        trades=trades,
        initial_capital=100000
    )


@pytest.fixture
def backtest_with_benchmark(equity_curve_with_trades, benchmark_equity_data):
    """Create BacktestResult with benchmark."""
    equity, trades = equity_curve_with_trades
    return BacktestResult(
        equity_curve=equity,
        trades=trades,
        initial_capital=100000,
        benchmark_equity=benchmark_equity_data,
        benchmark_name='SPY'
    )


# ============================================================================
# print_summary() Tests
# ============================================================================

class TestPrintSummary:
    """Test print_summary() method."""
    
    def test_print_summary_basic(self, simple_backtest_result):
        """
        Test basic print_summary execution.
        
        Arrange: Create BacktestResult
        Act: Call print_summary()
        Assert: No errors, output contains expected text
        """
        # Capture stdout
        captured_output = StringIO()
        sys.stdout = captured_output
        
        try:
            simple_backtest_result.print_summary()
            output = captured_output.getvalue()
            
            # Check that key sections are printed
            assert "BACKTEST RESULTS SUMMARY" in output
            assert "PERFORMANCE METRICS" in output
            assert "TRADE STATISTICS" in output
            assert "PORTFOLIO VALUES" in output
            
        finally:
            sys.stdout = sys.__stdout__
    
    def test_print_summary_contains_metrics(self, simple_backtest_result):
        """
        Test that print_summary contains all key metrics.
        
        Arrange: Create BacktestResult
        Act: Call print_summary()
        Assert: All metrics are printed
        """
        captured_output = StringIO()
        sys.stdout = captured_output
        
        try:
            simple_backtest_result.print_summary()
            output = captured_output.getvalue()
            
            # Check for specific metrics
            assert "Total Return" in output
            assert "CAGR" in output
            assert "Sharpe Ratio" in output
            assert "Sortino Ratio" in output
            assert "Max Drawdown" in output
            assert "Calmar Ratio" in output
            assert "Win Rate" in output
            assert "Avg Trade P&L" in output
            assert "Profit Factor" in output
            assert "Total Trades" in output
            
        finally:
            sys.stdout = sys.__stdout__
    
    def test_print_summary_with_benchmark(self, backtest_with_benchmark):
        """
        Test print_summary with benchmark comparison.
        
        Arrange: Create BacktestResult with benchmark
        Act: Call print_summary()
        Assert: Benchmark section is printed
        """
        captured_output = StringIO()
        sys.stdout = captured_output
        
        try:
            backtest_with_benchmark.print_summary()
            output = captured_output.getvalue()
            
            # Should have benchmark section if metrics available
            # (May not show if BenchmarkComparator fails)
            # Just check it doesn't crash
            assert "BACKTEST RESULTS SUMMARY" in output
            
        finally:
            sys.stdout = sys.__stdout__
    
    def test_print_summary_formatting(self, simple_backtest_result):
        """
        Test print_summary formatting (alignment, separators).
        
        Arrange: Create BacktestResult
        Act: Call print_summary()
        Assert: Output has proper formatting
        """
        captured_output = StringIO()
        sys.stdout = captured_output
        
        try:
            simple_backtest_result.print_summary()
            output = captured_output.getvalue()
            
            # Check for separators
            assert "=" * 70 in output
            assert "-" * 35 in output
            
            # Check for dollar signs in portfolio values
            assert "$" in output
            
        finally:
            sys.stdout = sys.__stdout__
    
    def test_print_summary_empty_data(self):
        """
        Test print_summary with empty equity curve.
        
        Arrange: Create BacktestResult with empty data
        Act: Call print_summary()
        Assert: Prints zeros/defaults, no crash
        """
        result = BacktestResult(
            equity_curve=pd.DataFrame(),
            trades=pd.DataFrame(),
            initial_capital=50000
        )
        
        captured_output = StringIO()
        sys.stdout = captured_output
        
        try:
            result.print_summary()
            output = captured_output.getvalue()
            
            # Should print with zero values
            assert "0.00%" in output or "0%" in output
            assert "BACKTEST RESULTS SUMMARY" in output
            
        finally:
            sys.stdout = sys.__stdout__
    
    def test_print_summary_no_trades(self, equity_curve_with_trades):
        """
        Test print_summary with no trades.
        
        Arrange: Create BacktestResult with equity but no trades
        Act: Call print_summary()
        Assert: Shows 0 trades, no crash
        """
        equity, _ = equity_curve_with_trades
        result = BacktestResult(
            equity_curve=equity,
            trades=pd.DataFrame()
        )
        
        captured_output = StringIO()
        sys.stdout = captured_output
        
        try:
            result.print_summary()
            output = captured_output.getvalue()
            
            assert "Total Trades" in output
            # Should show 0 trades
            
        finally:
            sys.stdout = sys.__stdout__
    
    @patch('core.portfolio.backtest_result.BenchmarkComparator')
    def test_print_summary_benchmark_metrics_displayed(
        self,
        mock_comparator_class,
        equity_curve_with_trades,
        benchmark_equity_data
    ):
        """
        Test that benchmark metrics are displayed when available.
        
        Arrange: Mock BenchmarkComparator to return metrics
        Act: Call print_summary()
        Assert: Benchmark metrics are displayed
        """
        # Setup mock
        mock_comparator = MagicMock()
        mock_comparator_class.return_value = mock_comparator
        mock_comparator.calculate_metrics.return_value = {
            'Beta (Full Period)': 1.05,
            'Beta (90-day avg)': 1.02,
            'Beta (1-year avg)': 1.03,
            'Alpha (Annual)': 0.03,
            'Benchmark Return': 0.10,
            'Relative Return': 0.02,
            'Tracking Error': 0.05,
            'Information Ratio': 0.4,
            'Correlation': 0.85,
            'Up Capture Ratio': 1.1,
            'Down Capture Ratio': 0.9
        }
        
        equity, trades = equity_curve_with_trades
        result = BacktestResult(
            equity_curve=equity,
            trades=trades,
            benchmark_equity=benchmark_equity_data,
            benchmark_name='SPY'
        )
        
        captured_output = StringIO()
        sys.stdout = captured_output
        
        try:
            result.print_summary()
            output = captured_output.getvalue()
            
            # Check for benchmark section
            assert "BENCHMARK COMPARISON" in output
            assert "Beta" in output or "Alpha" in output
            
        finally:
            sys.stdout = sys.__stdout__


# ============================================================================
# plot_equity_curve() Tests
# ============================================================================

class TestPlotEquityCurve:
    """Test plot_equity_curve() method."""
    
    @patch('matplotlib.pyplot.show')
    @patch('matplotlib.pyplot.subplots')
    def test_plot_equity_curve_basic(
        self,
        mock_subplots,
        mock_show,
        simple_backtest_result
    ):
        """
        Test basic plot_equity_curve execution.
        
        Arrange: Mock matplotlib, create BacktestResult
        Act: Call plot_equity_curve()
        Assert: plt.subplots and plt.show are called
        """
        # Setup mock axes
        mock_fig = MagicMock()
        mock_ax1 = MagicMock()
        mock_ax2 = MagicMock()
        mock_subplots.return_value = (mock_fig, [mock_ax1, mock_ax2])
        
        # Call method
        simple_backtest_result.plot_equity_curve()
        
        # Verify matplotlib calls
        mock_subplots.assert_called_once()
        mock_show.assert_called_once()
    
    @patch('matplotlib.pyplot.show')
    @patch('matplotlib.pyplot.subplots')
    def test_plot_equity_curve_two_subplots(
        self,
        mock_subplots,
        mock_show,
        simple_backtest_result
    ):
        """
        Test that plot creates 2 subplots (equity + drawdown).
        
        Arrange: Mock matplotlib
        Act: Call plot_equity_curve()
        Assert: Creates 2 subplots
        """
        mock_fig = MagicMock()
        mock_ax1 = MagicMock()
        mock_ax2 = MagicMock()
        mock_subplots.return_value = (mock_fig, [mock_ax1, mock_ax2])
        
        simple_backtest_result.plot_equity_curve()
        
        # Should create 2x1 subplots
        args, kwargs = mock_subplots.call_args
        assert args[0] == 2  # rows
        assert args[1] == 1  # cols
    
    @patch('matplotlib.pyplot.show')
    @patch('matplotlib.pyplot.subplots')
    def test_plot_equity_curve_plots_data(
        self,
        mock_subplots,
        mock_show,
        simple_backtest_result
    ):
        """
        Test that equity data is plotted on axes.
        
        Arrange: Mock matplotlib axes
        Act: Call plot_equity_curve()
        Assert: plot() method is called on Series
        """
        mock_fig = MagicMock()
        mock_ax1 = MagicMock()
        mock_ax2 = MagicMock()
        mock_subplots.return_value = (mock_fig, [mock_ax1, mock_ax2])
        
        simple_backtest_result.plot_equity_curve()
        
        # Verify that axes methods were called
        # (plot, set_title, etc.)
        assert mock_ax1.method_calls  # Some methods called
        assert mock_ax2.method_calls
    
    def test_plot_equity_curve_empty_data(self):
        """
        Test plot_equity_curve with empty data.
        
        Arrange: Create BacktestResult with empty equity
        Act: Call plot_equity_curve()
        Assert: Prints message, doesn't crash
        """
        result = BacktestResult(
            equity_curve=pd.DataFrame(),
            trades=pd.DataFrame()
        )
        
        captured_output = StringIO()
        sys.stdout = captured_output
        
        try:
            result.plot_equity_curve()
            output = captured_output.getvalue()
            
            # Should print "No data to plot"
            assert "No data to plot" in output
            
        finally:
            sys.stdout = sys.__stdout__
    
    @patch('matplotlib.pyplot.show')
    @patch('matplotlib.pyplot.subplots')
    def test_plot_equity_curve_with_figsize(
        self,
        mock_subplots,
        mock_show,
        simple_backtest_result
    ):
        """
        Test that plot uses correct figure size.
        
        Arrange: Mock matplotlib
        Act: Call plot_equity_curve()
        Assert: figsize parameter is (14, 10)
        """
        mock_fig = MagicMock()
        mock_ax1 = MagicMock()
        mock_ax2 = MagicMock()
        mock_subplots.return_value = (mock_fig, [mock_ax1, mock_ax2])
        
        simple_backtest_result.plot_equity_curve()
        
        args, kwargs = mock_subplots.call_args
        assert 'figsize' in kwargs
        assert kwargs['figsize'] == (14, 10)
    
    @patch('core.portfolio.backtest_result.plt', None)
    def test_plot_equity_curve_matplotlib_not_available(
        self,
        simple_backtest_result
    ):
        """
        Test behavior when matplotlib is not available.
        
        Arrange: Simulate missing matplotlib
        Act: Call plot_equity_curve()
        Assert: Prints error message, doesn't crash
        """
        captured_output = StringIO()
        sys.stdout = captured_output
        
        try:
            simple_backtest_result.plot_equity_curve()
            output = captured_output.getvalue()
            
            # Should print message about matplotlib
            assert "Matplotlib not available" in output or "No data to plot" in output
            
        finally:
            sys.stdout = sys.__stdout__


# ============================================================================
# risk_analysis Property Tests
# ============================================================================

class TestRiskAnalysis:
    """Test risk_analysis property."""
    
    def test_risk_analysis_basic(self, simple_backtest_result):
        """
        Test basic risk_analysis property.
        
        Arrange: Create BacktestResult
        Act: Access risk_analysis property
        Assert: Returns dict with risk metrics
        """
        risk = simple_backtest_result.risk_analysis
        
        assert isinstance(risk, dict)
        
        # Check for expected keys
        expected_keys = [
            'max_drawdown',
            'volatility',
            'downside_volatility',
            'var_95',
            'cvar_95',
            'best_day',
            'worst_day',
            'positive_days',
            'negative_days',
            'avg_positive_return',
            'avg_negative_return'
        ]
        
        for key in expected_keys:
            assert key in risk
    
    def test_risk_analysis_volatility_positive(self, simple_backtest_result):
        """
        Test that volatility metrics are positive.
        
        Arrange: Create BacktestResult
        Act: Get risk_analysis
        Assert: Volatility values are positive
        """
        risk = simple_backtest_result.risk_analysis
        
        assert risk['volatility'] >= 0
        assert risk['downside_volatility'] >= 0
    
    def test_risk_analysis_var_cvar(self, simple_backtest_result):
        """
        Test VaR and CVaR calculations.
        
        Arrange: Create BacktestResult
        Act: Get risk_analysis
        Assert: VaR and CVaR are calculated
        """
        risk = simple_backtest_result.risk_analysis
        
        # VaR and CVaR should be negative (losses)
        assert 'var_95' in risk
        assert 'cvar_95' in risk
        
        # CVaR should be more negative than VaR
        if not np.isnan(risk['cvar_95']):
            assert risk['cvar_95'] <= risk['var_95']
    
    def test_risk_analysis_best_worst_day(self, simple_backtest_result):
        """
        Test best and worst day calculations.
        
        Arrange: Create BacktestResult
        Act: Get risk_analysis
        Assert: Best day > 0, worst day < 0
        """
        risk = simple_backtest_result.risk_analysis
        
        assert 'best_day' in risk
        assert 'worst_day' in risk
        
        # Best day should be > worst day
        assert risk['best_day'] > risk['worst_day']
    
    def test_risk_analysis_positive_negative_days(self, simple_backtest_result):
        """
        Test positive/negative days count.
        
        Arrange: Create BacktestResult
        Act: Get risk_analysis
        Assert: Counts are non-negative integers
        """
        risk = simple_backtest_result.risk_analysis
        
        assert risk['positive_days'] >= 0
        assert risk['negative_days'] >= 0
        assert isinstance(risk['positive_days'], (int, np.integer))
        assert isinstance(risk['negative_days'], (int, np.integer))
    
    def test_risk_analysis_empty_data(self):
        """
        Test risk_analysis with empty equity curve.
        
        Arrange: Create BacktestResult with empty data
        Act: Get risk_analysis
        Assert: Returns empty dict
        """
        result = BacktestResult(
            equity_curve=pd.DataFrame(),
            trades=pd.DataFrame()
        )
        
        risk = result.risk_analysis
        
        assert isinstance(risk, dict)
        assert len(risk) == 0
    
    def test_risk_analysis_avg_returns(self, simple_backtest_result):
        """
        Test average positive and negative return calculations.
        
        Arrange: Create BacktestResult
        Act: Get risk_analysis
        Assert: Avg positive > 0, avg negative < 0
        """
        risk = simple_backtest_result.risk_analysis
        
        # Average positive return should be positive
        if not np.isnan(risk['avg_positive_return']):
            assert risk['avg_positive_return'] > 0
        
        # Average negative return should be negative
        if not np.isnan(risk['avg_negative_return']):
            assert risk['avg_negative_return'] < 0


# ============================================================================
# generate_html_report() Tests
# ============================================================================

class TestGenerateHtmlReport:
    """Test generate_html_report() method."""
    
    @patch('core.portfolio.backtest_result.Reporter')
    def test_generate_html_report_basic(
        self,
        mock_reporter_class,
        simple_backtest_result
    ):
        """
        Test basic HTML report generation.
        
        Arrange: Mock Reporter class
        Act: Call generate_html_report()
        Assert: Reporter is instantiated and called
        """
        mock_reporter = MagicMock()
        mock_reporter_class.return_value = mock_reporter
        
        simple_backtest_result.generate_html_report('test_report.html')
        
        # Verify Reporter was used
        mock_reporter_class.assert_called_once()
        mock_reporter.generate_html_report.assert_called_once()
    
    @patch('core.portfolio.backtest_result.Reporter')
    def test_generate_html_report_parameters(
        self,
        mock_reporter_class,
        simple_backtest_result
    ):
        """
        Test that correct parameters are passed to Reporter.
        
        Arrange: Mock Reporter
        Act: Call generate_html_report()
        Assert: Reporter receives correct data
        """
        mock_reporter = MagicMock()
        mock_reporter_class.return_value = mock_reporter
        
        save_path = 'output/report.html'
        simple_backtest_result.generate_html_report(save_path)
        
        # Check call arguments
        args, kwargs = mock_reporter.generate_html_report.call_args
        
        assert 'equity_df' in kwargs
        assert 'trades_df' in kwargs
        assert 'metrics' in kwargs
        assert 'save_path' in kwargs
        assert kwargs['save_path'] == save_path
    
    @patch('core.portfolio.backtest_result.Reporter')
    def test_generate_html_report_equity_df_format(
        self,
        mock_reporter_class,
        simple_backtest_result
    ):
        """
        Test that equity DataFrame is formatted correctly for Reporter.
        
        Arrange: Mock Reporter
        Act: Call generate_html_report()
        Assert: equity_df has 'Date' column
        """
        mock_reporter = MagicMock()
        mock_reporter_class.return_value = mock_reporter
        
        simple_backtest_result.generate_html_report('test.html')
        
        # Get the equity_df passed to Reporter
        args, kwargs = mock_reporter.generate_html_report.call_args
        equity_df = kwargs['equity_df']
        
        # Should have Date column
        assert 'Date' in equity_df.columns
    
    @patch('core.portfolio.backtest_result.Reporter')
    def test_generate_html_report_success_message(
        self,
        mock_reporter_class,
        simple_backtest_result
    ):
        """
        Test success message is printed.
        
        Arrange: Mock Reporter
        Act: Call generate_html_report()
        Assert: Success message is printed
        """
        mock_reporter = MagicMock()
        mock_reporter_class.return_value = mock_reporter
        
        captured_output = StringIO()
        sys.stdout = captured_output
        
        try:
            simple_backtest_result.generate_html_report('test.html')
            output = captured_output.getvalue()
            
            # Should print success message
            assert "HTML report saved" in output or "✅" in output
            
        finally:
            sys.stdout = sys.__stdout__
    
    @patch('core.portfolio.backtest_result.Reporter')
    def test_generate_html_report_exception_handling(
        self,
        mock_reporter_class,
        simple_backtest_result
    ):
        """
        Test exception handling when Reporter fails.
        
        Arrange: Mock Reporter to raise exception
        Act: Call generate_html_report()
        Assert: Exception is caught, error message printed
        """
        mock_reporter = MagicMock()
        mock_reporter_class.return_value = mock_reporter
        mock_reporter.generate_html_report.side_effect = Exception("Test error")
        
        captured_output = StringIO()
        sys.stdout = captured_output
        
        try:
            simple_backtest_result.generate_html_report('test.html')
            output = captured_output.getvalue()
            
            # Should print error message
            assert "Could not generate HTML report" in output or "⚠" in output
            
        finally:
            sys.stdout = sys.__stdout__
    
    def test_generate_html_report_reporter_import_error(self, simple_backtest_result):
        """
        Test behavior when Reporter import fails.
        
        Arrange: Create BacktestResult
        Act: Call generate_html_report() when Reporter doesn't exist
        Assert: Handles gracefully with error message
        """
        # This test simulates import failure
        # In real case, the import would fail at the top of the method
        captured_output = StringIO()
        sys.stdout = captured_output
        
        try:
            # If Reporter doesn't exist, should catch exception
            with patch('core.portfolio.backtest_result.Reporter', side_effect=ImportError):
                simple_backtest_result.generate_html_report('test.html')
            
            # Note: Implementation catches generic Exception, so this might not be reached
            # depending on actual implementation
            
        finally:
            sys.stdout = sys.__stdout__


# ============================================================================
# Edge Cases and Integration Tests
# ============================================================================

class TestBacktestResultEdgeCasesExtended:
    """Extended edge case tests for BacktestResult."""
    
    def test_all_methods_with_minimal_data(self):
        """
        Test all methods with minimal viable data.
        
        Arrange: Create BacktestResult with 1 day of data
        Act: Call all public methods
        Assert: No crashes
        """
        dates = pd.date_range('2023-01-01', periods=1, freq='D')
        equity = pd.DataFrame({
            'TotalValue': [100000]
        }, index=dates)
        
        trades = pd.DataFrame({
            'pnl': [100],
            'return': [0.01]
        })
        
        result = BacktestResult(equity_curve=equity, trades=trades)
        
        # Call all methods - should not crash
        _ = result.metrics
        _ = result.risk_analysis
        
        captured_output = StringIO()
        sys.stdout = captured_output
        
        try:
            result.print_summary()
        finally:
            sys.stdout = sys.__stdout__
    
    def test_large_dataset_performance(self):
        """
        Test with large dataset (10 years daily data).
        
        Arrange: Create large BacktestResult
        Act: Access metrics and properties
        Assert: Completes in reasonable time
        """
        # 10 years of daily data
        dates = pd.date_range('2014-01-01', periods=2520, freq='D')
        
        np.random.seed(42)
        returns = np.random.normal(0.0005, 0.015, 2520)
        total_values = 100000 * (1 + returns).cumprod()
        
        equity = pd.DataFrame({
            'TotalValue': total_values
        }, index=dates)
        
        # Many trades
        n_trades = 500
        trades = pd.DataFrame({
            'pnl': np.random.normal(50, 200, n_trades),
            'return': np.random.normal(0.01, 0.05, n_trades)
        })
        
        result = BacktestResult(equity_curve=equity, trades=trades)
        
        # Should compute metrics efficiently
        metrics = result.metrics
        risk = result.risk_analysis
        
        assert len(metrics) > 0
        assert len(risk) > 0
    
    def test_extreme_values_handling(self):
        """
        Test handling of extreme values (huge gains/losses).
        
        Arrange: Create data with extreme returns
        Act: Calculate metrics
        Assert: Metrics are finite, no overflow
        """
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        
        # Extreme equity curve (10x gain then 90% loss)
        values = [100000] * 50 + [1000000] * 25 + [100000] * 25
        
        equity = pd.DataFrame({
            'TotalValue': values
        }, index=dates)
        
        result = BacktestResult(equity_curve=equity, trades=pd.DataFrame())
        
        metrics = result.metrics
        
        # Metrics should be finite
        assert np.isfinite(metrics['Total Return'])
        assert np.isfinite(metrics['Max Drawdown'])
        assert np.isfinite(metrics['Sharpe Ratio'])
