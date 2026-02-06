"""
Extended test suite for BacktestResult.

Tests uncovered functionality in core/portfolio/backtest_result.py:
- Metrics calculations (Sharpe, Sortino, CAGR, etc.)
- Drawdown analysis
- Trade statistics
- Benchmark comparisons
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from core.portfolio.backtest_result import BacktestResult


@pytest.fixture
def sample_equity_curve():
    """Generate sample equity curve with growth."""
    dates = pd.date_range('2023-01-01', periods=252, freq='D')
    
    # Simulate growing equity curve with volatility
    np.random.seed(42)
    returns = np.random.normal(0.001, 0.02, 252)  # 0.1% daily return, 2% vol
    total_values = 100000 * (1 + returns).cumprod()
    
    return pd.DataFrame({
        'Cash': 50000 * np.ones(252),
        'PositionsValue': total_values - 50000,
        'TotalValue': total_values
    }, index=dates)


@pytest.fixture
def sample_trades():
    """Generate sample trades DataFrame."""
    return pd.DataFrame({
        'ticker': ['AAPL', 'MSFT', 'GOOGL', 'AAPL', 'MSFT'],
        'entry_date': pd.date_range('2023-01-01', periods=5, freq='30D'),
        'exit_date': pd.date_range('2023-02-01', periods=5, freq='30D'),
        'shares': [100, 50, 20, 150, 75],
        'entry_price': [150.0, 300.0, 2000.0, 145.0, 310.0],
        'exit_price': [155.0, 305.0, 2050.0, 142.0, 315.0],
        'pnl': [500, 250, 1000, -450, 375],
        'return': [0.033, 0.017, 0.025, -0.021, 0.016]
    })


@pytest.fixture
def benchmark_equity():
    """Generate benchmark equity curve."""
    dates = pd.date_range('2023-01-01', periods=252, freq='D')
    np.random.seed(123)
    returns = np.random.normal(0.0005, 0.015, 252)  # Lower return than strategy
    values = 100000 * (1 + returns).cumprod()
    
    return pd.DataFrame({
        'TotalValue': values
    }, index=dates)


# ============================================================================
# Basic Properties Tests
# ============================================================================

class TestBacktestResultProperties:
    """Test basic property calculations."""
    
    def test_final_equity(self, sample_equity_curve):
        """Test final equity calculation."""
        result = BacktestResult(
            equity_curve=sample_equity_curve,
            trades=pd.DataFrame()
        )
        
        assert result.final_equity == sample_equity_curve['TotalValue'].iloc[-1]
    
    def test_final_equity_empty(self):
        """Test final equity with empty equity curve."""
        result = BacktestResult(
            equity_curve=pd.DataFrame(),
            trades=pd.DataFrame(),
            initial_capital=50000
        )
        
        assert result.final_equity == 50000
    
    def test_total_return(self, sample_equity_curve):
        """Test total return calculation."""
        result = BacktestResult(
            equity_curve=sample_equity_curve,
            trades=pd.DataFrame()
        )
        
        initial = sample_equity_curve['TotalValue'].iloc[0]
        final = sample_equity_curve['TotalValue'].iloc[-1]
        expected_return = (final / initial - 1)
        
        assert abs(result.total_return - expected_return) < 0.0001
    
    def test_total_return_empty(self):
        """Test total return with empty data."""
        result = BacktestResult(
            equity_curve=pd.DataFrame(),
            trades=pd.DataFrame()
        )
        
        assert result.total_return == 0.0
    
    def test_returns_series(self, sample_equity_curve):
        """Test returns series calculation."""
        result = BacktestResult(
            equity_curve=sample_equity_curve,
            trades=pd.DataFrame()
        )
        
        returns = result.returns
        assert len(returns) == len(sample_equity_curve) - 1  # pct_change drops first
        assert isinstance(returns, pd.Series)


# ============================================================================
# Metrics Calculation Tests
# ============================================================================

class TestBacktestResultMetrics:
    """Test comprehensive metrics calculations."""
    
    def test_metrics_basic(self, sample_equity_curve, sample_trades):
        """Test basic metrics are calculated."""
        result = BacktestResult(
            equity_curve=sample_equity_curve,
            trades=sample_trades
        )
        
        metrics = result.metrics
        
        assert 'Total Return' in metrics
        assert 'CAGR' in metrics
        assert 'Sharpe Ratio' in metrics
        assert 'Sortino Ratio' in metrics
        assert 'Max Drawdown' in metrics
        assert 'Calmar Ratio' in metrics
        assert 'Win Rate' in metrics
        assert 'Total Trades' in metrics
    
    def test_metrics_empty_data(self):
        """Test metrics with empty data."""
        result = BacktestResult(
            equity_curve=pd.DataFrame(),
            trades=pd.DataFrame()
        )
        
        metrics = result.metrics
        
        assert metrics['Total Return'] == 0.0
        assert metrics['CAGR'] == 0.0
        assert metrics['Sharpe Ratio'] == 0.0
        assert metrics['Total Trades'] == 0
    
    def test_win_rate_calculation(self):
        """Test win rate calculation."""
        trades = pd.DataFrame({
            'pnl': [100, -50, 200, -30, 150],  # 3 wins, 2 losses
            'return': [0.1, -0.05, 0.2, -0.03, 0.15]
        })
        
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        equity = pd.DataFrame({
            'Cash': 50000 * np.ones(100),
            'PositionsValue': 50000 * np.ones(100),
            'TotalValue': 100000 * np.ones(100)
        }, index=dates)
        
        result = BacktestResult(
            equity_curve=equity,
            trades=trades
        )
        
        metrics = result.metrics
        assert metrics['Win Rate'] == 0.60  # 3/5 = 60%
    
    def test_profit_factor(self):
        """Test profit factor calculation."""
        trades = pd.DataFrame({
            'pnl': [100, -50, 200, -30],  # Wins: 300, Losses: -80
            'return': [0.1, -0.05, 0.2, -0.03]
        })
        
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        equity = pd.DataFrame({
            'TotalValue': 100000 * np.ones(100)
        }, index=dates)
        
        result = BacktestResult(equity_curve=equity, trades=trades)
        metrics = result.metrics
        
        # Profit factor = gross_wins / abs(gross_losses) = 300 / 80 = 3.75
        assert abs(metrics['Profit Factor'] - 3.75) < 0.01
    
    def test_sharpe_ratio_positive(self, sample_equity_curve):
        """Test Sharpe ratio with positive returns."""
        result = BacktestResult(
            equity_curve=sample_equity_curve,
            trades=pd.DataFrame()
        )
        
        metrics = result.metrics
        # Should have positive Sharpe with positive returns
        assert metrics['Sharpe Ratio'] > 0


# ============================================================================
# Drawdown Analysis Tests
# ============================================================================

class TestDrawdownAnalysis:
    """Test drawdown-related calculations."""
    
    def test_max_drawdown_calculation(self):
        """Test maximum drawdown calculation."""
        # Create equity curve with known drawdown
        dates = pd.date_range('2023-01-01', periods=10, freq='D')
        values = [100000, 105000, 110000, 100000, 95000, 98000, 102000, 105000, 108000, 110000]
        
        equity = pd.DataFrame({
            'TotalValue': values
        }, index=dates)
        
        result = BacktestResult(equity_curve=equity, trades=pd.DataFrame())
        metrics = result.metrics
        
        # Max drawdown is from 110000 (index 2) to 95000 (index 4) = -13.64%
        expected_dd = (95000 / 110000 - 1)
        assert abs(metrics['Max Drawdown'] - expected_dd) < 0.01
    
    def test_no_drawdown(self):
        """Test with monotonically increasing equity."""
        dates = pd.date_range('2023-01-01', periods=10, freq='D')
        values = np.linspace(100000, 120000, 10)
        
        equity = pd.DataFrame({
            'TotalValue': values
        }, index=dates)
        
        result = BacktestResult(equity_curve=equity, trades=pd.DataFrame())
        metrics = result.metrics
        
        # No drawdown in monotonically increasing curve
        assert metrics['Max Drawdown'] >= 0 or abs(metrics['Max Drawdown']) < 0.001


# ============================================================================
# Benchmark Comparison Tests
# ============================================================================

class TestBenchmarkComparison:
    """Test benchmark comparison functionality."""
    
    def test_benchmark_metrics(self, sample_equity_curve, benchmark_equity):
        """Test that benchmark metrics are calculated."""
        result = BacktestResult(
            equity_curve=sample_equity_curve,
            trades=pd.DataFrame(),
            benchmark_equity=benchmark_equity,
            benchmark_name='SPY'
        )
        
        # Metrics should include benchmark comparisons
        assert result.benchmark_equity is not None
        assert result.benchmark_name == 'SPY'
    
    def test_without_benchmark(self, sample_equity_curve):
        """Test metrics without benchmark."""
        result = BacktestResult(
            equity_curve=sample_equity_curve,
            trades=pd.DataFrame()
        )
        
        # Should work without benchmark
        metrics = result.metrics
        assert 'Sharpe Ratio' in metrics
        assert result.benchmark_equity is None


# ============================================================================
# Edge Cases and Error Handling
# ============================================================================

class TestBacktestResultEdgeCases:
    """Test edge cases and error handling."""
    
    def test_single_trade(self):
        """Test with only one trade."""
        trades = pd.DataFrame({
            'pnl': [100],
            'return': [0.10]
        })
        
        dates = pd.date_range('2023-01-01', periods=50, freq='D')
        equity = pd.DataFrame({
            'TotalValue': 100000 * np.ones(50)
        }, index=dates)
        
        result = BacktestResult(equity_curve=equity, trades=trades)
        metrics = result.metrics
        
        assert metrics['Total Trades'] == 1
        assert metrics['Win Rate'] == 1.0
    
    def test_zero_initial_capital(self, sample_equity_curve):
        """Test handling of zero initial capital."""
        result = BacktestResult(
            equity_curve=sample_equity_curve,
            trades=pd.DataFrame(),
            initial_capital=0
        )
        
        # Should not crash
        assert result.final_equity > 0
    
    def test_all_losing_trades(self):
        """Test with all losing trades."""
        trades = pd.DataFrame({
            'pnl': [-100, -50, -200],
            'return': [-0.1, -0.05, -0.2]
        })
        
        dates = pd.date_range('2023-01-01', periods=50, freq='D')
        equity = pd.DataFrame({
            'TotalValue': np.linspace(100000, 80000, 50)
        }, index=dates)
        
        result = BacktestResult(equity_curve=equity, trades=trades)
        metrics = result.metrics
        
        assert metrics['Win Rate'] == 0.0
        assert metrics['Profit Factor'] == 0.0  # No wins
    
    def test_no_trades(self, sample_equity_curve):
        """Test with no trades executed."""
        result = BacktestResult(
            equity_curve=sample_equity_curve,
            trades=pd.DataFrame()
        )
        
        metrics = result.metrics
        assert metrics['Total Trades'] == 0
        assert metrics['Win Rate'] == 0.0


# ============================================================================
# Trade Statistics Tests
# ============================================================================

class TestTradeStatistics:
    """Test trade-level statistics calculations."""
    
    def test_avg_trade_calculation(self):
        """Test average trade calculation."""
        trades = pd.DataFrame({
            'pnl': [100, -50, 200, -30, 150],
            'return': [0.1, -0.05, 0.2, -0.03, 0.15]
        })
        
        dates = pd.date_range('2023-01-01', periods=50, freq='D')
        equity = pd.DataFrame({
            'TotalValue': 100000 * np.ones(50)
        }, index=dates)
        
        result = BacktestResult(equity_curve=equity, trades=trades)
        metrics = result.metrics
        
        # Avg = (100 - 50 + 200 - 30 + 150) / 5 = 370 / 5 = 74
        assert abs(metrics['Avg Trade'] - 74.0) < 0.01
    
    def test_total_trades_count(self, sample_trades):
        """Test total trades counting."""
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        equity = pd.DataFrame({
            'TotalValue': 100000 * np.ones(100)
        }, index=dates)
        
        result = BacktestResult(equity_curve=equity, trades=sample_trades)
        metrics = result.metrics
        
        assert metrics['Total Trades'] == len(sample_trades)


# ============================================================================
# Return-based Metrics Tests
# ============================================================================

class TestReturnMetrics:
    """Test return-based metrics like Sortino, Calmar."""
    
    def test_sortino_ratio(self, sample_equity_curve):
        """Test Sortino ratio calculation."""
        result = BacktestResult(
            equity_curve=sample_equity_curve,
            trades=pd.DataFrame()
        )
        
        metrics = result.metrics
        # Sortino should exist
        assert 'Sortino Ratio' in metrics
        assert isinstance(metrics['Sortino Ratio'], (int, float))
    
    def test_calmar_ratio(self, sample_equity_curve):
        """Test Calmar ratio calculation."""
        result = BacktestResult(
            equity_curve=sample_equity_curve,
            trades=pd.DataFrame()
        )
        
        metrics = result.metrics
        # Calmar = CAGR / |Max Drawdown|
        assert 'Calmar Ratio' in metrics
        assert isinstance(metrics['Calmar Ratio'], (int, float))
    
    def test_cagr_calculation(self):
        """Test CAGR calculation with known values."""
        # 1 year, 20% return
        dates = pd.date_range('2023-01-01', periods=365, freq='D')
        values = np.linspace(100000, 120000, 365)
        
        equity = pd.DataFrame({
            'TotalValue': values
        }, index=dates)
        
        result = BacktestResult(equity_curve=equity, trades=pd.DataFrame())
        metrics = result.metrics
        
        # CAGR should be around 13-20% (linear growth gives lower CAGR than exponential)
        assert 0.10 < metrics['CAGR'] < 0.25
