"""
Unit tests for utility modules (plotter, formatter).

Tests:
- PortfolioPlotter functionality
- PerformanceSummary calculations
- Formatting utilities

Run with: pytest tests/test_utils.py -v
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

import sys
sys.path.append('..')

from utils.plotter import PortfolioPlotter
from utils.formatter import PerformanceSummary
from core.portfolio.portfolio_manager_v2 import PortfolioManagerV2
from core.portfolio.backtest_result import BacktestResult


@pytest.fixture
def sample_strategy_results():
    """Create sample strategy results for testing."""
    # Create mock equity curve
    dates = pd.date_range(start='2020-01-01', end='2023-12-31', freq='D')
    
    # Strategy 1: Upward trend
    equity1 = pd.DataFrame({
        'TotalValue': 100000 * (1 + np.linspace(0, 0.5, len(dates))),
        'Cash': 50000,
        'PositionValue': 100000 * (1 + np.linspace(0, 0.5, len(dates))) - 50000
    }, index=dates)
    
    # Strategy 2: More volatile
    np.random.seed(42)
    returns2 = np.random.normal(0.001, 0.02, len(dates))
    equity2 = pd.DataFrame({
        'TotalValue': 80000 * (1 + returns2).cumprod(),
        'Cash': 40000,
        'PositionValue': 80000 * (1 + returns2).cumprod() - 40000
    }, index=dates)
    
    # Create mock BacktestResults
    result1 = BacktestResult(
        initial_capital=100000,
        equity_curve=equity1,
        trades=pd.DataFrame({
            'date': [dates[10], dates[50], dates[100]],
            'ticker': ['ES', 'ES', 'ES'],
            'action': ['BUY', 'SELL', 'BUY'],
            'shares': [10, -10, 10],
            'price': [4000, 4200, 4100],
            'pnl': [0, 2000, 0]
        })
    )
    
    result2 = BacktestResult(
        initial_capital=80000,
        equity_curve=equity2,
        trades=pd.DataFrame({
            'date': [dates[20], dates[60]],
            'ticker': ['NQ', 'NQ'],
            'action': ['BUY', 'SELL'],
            'shares': [5, -5],
            'price': [15000, 15500],
            'pnl': [0, 2500]
        })
    )
    
    return {
        'Strategy_1': {
            'result': result1,
            'capital': 100000,
            'assets': ['ES']
        },
        'Strategy_2': {
            'result': result2,
            'capital': 80000,
            'assets': ['NQ']
        }
    }


@pytest.fixture
def sample_benchmark_data():
    """Create sample benchmark data."""
    dates = pd.date_range(start='2020-01-01', end='2023-12-31', freq='D')
    
    # SPY-like benchmark
    np.random.seed(42)
    returns = np.random.normal(0.0005, 0.01, len(dates))
    prices = 300 * (1 + returns).cumprod()
    
    return pd.DataFrame({
        'Close': prices,
        'TotalValue': prices  # Normalized to 100 later
    }, index=dates)


class TestPortfolioPlotter:
    """Test PortfolioPlotter visualization utilities."""
    
    def test_plotter_initialization(self, sample_strategy_results):
        """Test plotter can be initialized with strategy results."""
        plotter = PortfolioPlotter(sample_strategy_results)
        
        assert plotter.strategy_results == sample_strategy_results
        assert plotter.benchmark_data is None
        assert plotter.benchmark_name == 'SPY'  # Default value
    
    def test_plotter_with_benchmark(self, sample_strategy_results, sample_benchmark_data):
        """Test plotter with benchmark data."""
        plotter = PortfolioPlotter(
            sample_strategy_results,
            benchmark_data=sample_benchmark_data,
            benchmark_name='SPY'
        )
        
        assert plotter.benchmark_data is not None
        assert plotter.benchmark_name == 'SPY'
    
    def test_plotter_has_required_methods(self, sample_strategy_results):
        """Test that plotter has expected methods."""
        plotter = PortfolioPlotter(sample_strategy_results)
        
        assert hasattr(plotter, 'plot_equity_curves')
        assert hasattr(plotter, 'plot_risk_dashboard')
        assert hasattr(plotter, 'plot_all_dashboards')
        
        # Check methods are callable
        assert callable(plotter.plot_equity_curves)
        assert callable(plotter.plot_risk_dashboard)


class TestPerformanceSummary:
    """Test PerformanceSummary formatting and calculations."""
    
    def test_summary_initialization(self, sample_strategy_results):
        """Test PerformanceSummary can be initialized."""
        summary = PerformanceSummary(
            sample_strategy_results,
            period_label='TEST'
        )
        
        assert summary.strategy_results == sample_strategy_results
        assert summary.period_label == 'TEST'
    
    def test_summary_with_benchmark(self, sample_strategy_results, sample_benchmark_data):
        """Test summary with benchmark data."""
        summary = PerformanceSummary(
            sample_strategy_results,
            benchmark_data=sample_benchmark_data,
            period_label='IN-SAMPLE'
        )
        
        assert summary.benchmark_data is not None
    
    def test_summary_has_required_methods(self, sample_strategy_results):
        """Test that summary has expected methods."""
        summary = PerformanceSummary(sample_strategy_results)
        
        assert hasattr(summary, 'print_benchmark_comparison')
        assert hasattr(summary, 'print_strategy_rankings')
        assert hasattr(summary, 'print_full_report')
        assert hasattr(summary, 'print_recommendations')
        assert hasattr(summary, 'to_dataframe')
        
        # Check methods are callable
        assert callable(summary.print_benchmark_comparison)
        assert callable(summary.to_dataframe)
    
    def test_to_dataframe(self, sample_strategy_results):
        """Test converting summary to DataFrame."""
        summary = PerformanceSummary(sample_strategy_results)
        
        df = summary.to_dataframe()
        
        assert isinstance(df, pd.DataFrame)
        assert not df.empty
        assert 'Strategy' in df.columns or df.index.name == 'Strategy'
    
    def test_comparison_table(self, sample_strategy_results):
        """Test comparison table generation."""
        summary_is = PerformanceSummary(
            sample_strategy_results, 
            period_label='IN-SAMPLE'
        )
        
        # Create slightly different results for OOS
        summary_oos = PerformanceSummary(
            sample_strategy_results,
            period_label='OUT-OF-SAMPLE'
        )
        
        # Should not raise error
        try:
            summary_is.print_comparison_table(summary_oos)
        except Exception as e:
            pytest.fail(f"Comparison table failed: {e}")


class TestFormatterHelpers:
    """Test helper functions in formatter module."""
    
    def test_formatter_module_imports(self):
        """Test that formatter module can be imported."""
        try:
            from utils.formatter import PerformanceSummary, compare_periods
            assert True
        except ImportError as e:
            pytest.fail(f"Failed to import formatter: {e}")
    
    def test_compare_periods_exists(self):
        """Test that compare_periods function exists."""
        from utils.formatter import compare_periods
        
        assert callable(compare_periods)


class TestPlotterHelpers:
    """Test helper functions in plotter module."""
    
    def test_plotter_module_imports(self):
        """Test that plotter module can be imported."""
        try:
            from utils.plotter import PortfolioPlotter
            assert True
        except ImportError as e:
            pytest.fail(f"Failed to import plotter: {e}")


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
