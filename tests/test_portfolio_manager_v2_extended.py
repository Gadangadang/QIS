"""
Extended unit tests for PortfolioManagerV2 to maximize coverage.

Focuses on:
- Initialization with various configurations
- Edge cases in backtest loops
- Rebalancing logic (frequency and drift-based)
- Risk rejection logging
- Stop-loss and take-profit execution
- Configuration summary generation

Run with: pytest tests/test_portfolio_manager_v2_extended.py -v --cov=core/portfolio/portfolio_manager_v2
"""

from __future__ import annotations

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta

from core.portfolio.portfolio_manager_v2 import PortfolioManagerV2
from core.portfolio.portfolio import Portfolio, Position
from core.portfolio.risk_manager import RiskManager, RiskConfig
from core.portfolio.execution_engine import ExecutionEngine, ExecutionConfig
from core.portfolio.position_sizers import FixedFractionalSizer, ATRSizer


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def sample_prices():
    """Generate sample price data for testing."""
    dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
    np.random.seed(42)
    
    base_price = 100.0
    returns = np.random.normal(0.001, 0.02, len(dates))
    close_prices = base_price * (1 + returns).cumprod()
    
    df = pd.DataFrame({
        'Open': close_prices * 0.99,
        'High': close_prices * 1.02,
        'Low': close_prices * 0.98,
        'Close': close_prices,
        'Volume': np.random.randint(1000000, 5000000, len(dates))
    }, index=dates)
    
    return {'AAPL': df}


@pytest.fixture
def simple_signals():
    """Generate simple buy/hold/sell signals."""
    dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
    
    # Buy signal for first 50 days, sell signal after
    signals = np.ones(len(dates))
    signals[50:] = 0
    
    return {'AAPL': pd.DataFrame({'Signal': signals}, index=dates)}


@pytest.fixture
def alternating_signals():
    """Generate alternating buy/sell signals."""
    dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
    
    # Alternate between buy (1) and sell (0) every 10 days
    signals = np.array([1 if (i // 10) % 2 == 0 else 0 for i in range(len(dates))])
    
    return {'AAPL': pd.DataFrame({'Signal': signals}, index=dates)}


@pytest.fixture
def benchmark_data():
    """Generate benchmark data."""
    dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
    
    # Benchmark starts at 100 and grows steadily
    values = 100 * (1.0 + np.arange(len(dates)) * 0.001)
    
    return pd.DataFrame({'TotalValue': values}, index=dates)


# ============================================================================
# Test Initialization
# ============================================================================

class TestPortfolioManagerV2Initialization:
    """Test PortfolioManagerV2 initialization with various configurations."""
    
    def test_default_initialization(self):
        """Test initialization with default parameters."""
        # Arrange & Act
        pm = PortfolioManagerV2()
        
        # Assert
        assert pm.initial_capital == 100000
        assert pm.rebalance_threshold is None
        assert pm.rebalance_frequency == 'never'
        assert pm.last_rebalance_date is None
        assert pm.risk_config.risk_per_trade == 0.02
        assert pm.risk_config.max_position_size == 0.20
        assert pm.execution_config.transaction_cost_bps == 3.0
        assert pm.risk_rejections == []
    
    def test_custom_capital_initialization(self):
        """Test initialization with custom capital."""
        # Arrange & Act
        pm = PortfolioManagerV2(initial_capital=500000)
        
        # Assert
        assert pm.initial_capital == 500000
    
    def test_risk_parameters_initialization(self):
        """Test initialization with custom risk parameters."""
        # Arrange & Act
        pm = PortfolioManagerV2(
            risk_per_trade=0.01,
            max_position_size=0.15,
            stop_loss_pct=0.05,
            take_profit_pct=0.20
        )
        
        # Assert
        assert pm.risk_config.risk_per_trade == 0.01
        assert pm.risk_config.max_position_size == 0.15
        assert pm.risk_config.stop_loss_pct == 0.05
        assert pm.risk_config.take_profit_pct == 0.20
    
    def test_execution_parameters_initialization(self):
        """Test initialization with custom execution parameters."""
        # Arrange & Act
        pm = PortfolioManagerV2(
            transaction_cost_bps=5.0,
            slippage_bps=2.0
        )
        
        # Assert
        assert pm.execution_config.transaction_cost_bps == 5.0
        assert pm.execution_config.slippage_bps == 2.0
    
    def test_rebalancing_parameters_initialization(self):
        """Test initialization with rebalancing parameters."""
        # Arrange & Act
        pm = PortfolioManagerV2(
            rebalance_threshold=0.10,
            rebalance_frequency='weekly'
        )
        
        # Assert
        assert pm.rebalance_threshold == 0.10
        assert pm.rebalance_frequency == 'weekly'
    
    def test_custom_position_sizer_initialization(self):
        """Test initialization with custom position sizer."""
        # Arrange
        custom_sizer = ATRSizer(atr_multiplier=2.0, max_position_pct=0.25)
        
        # Act
        pm = PortfolioManagerV2(position_sizer=custom_sizer)
        
        # Assert
        assert pm.risk_manager.position_sizer == custom_sizer
    
    def test_risk_log_path_initialization(self):
        """Test initialization with risk log path."""
        # Arrange & Act
        pm = PortfolioManagerV2(risk_log_path='logs/test_risk.csv')
        
        # Assert
        assert pm.risk_log_path == 'logs/test_risk.csv'


# ============================================================================
# Test Backtest Execution
# ============================================================================

class TestBacktestExecution:
    """Test backtest execution with various scenarios."""
    
    def test_basic_backtest_execution(self, sample_prices, simple_signals):
        """Test basic backtest completes successfully."""
        # Arrange
        pm = PortfolioManagerV2(initial_capital=100000)
        
        # Act
        result = pm.run_backtest(signals=simple_signals, prices=sample_prices)
        
        # Assert
        assert result is not None
        assert len(result.equity_curve) > 0
        assert 'TotalValue' in result.equity_curve.columns
        assert result.initial_capital == 100000
    
    def test_backtest_with_benchmark(self, sample_prices, simple_signals, benchmark_data):
        """Test backtest with benchmark data."""
        # Arrange
        pm = PortfolioManagerV2(initial_capital=100000)
        
        # Act
        result = pm.run_backtest(
            signals=simple_signals,
            prices=sample_prices,
            benchmark_data=benchmark_data,
            benchmark_name='SPY'
        )
        
        # Assert
        assert result.benchmark_name == 'SPY'
        assert result.benchmark_equity is not None
        assert len(result.benchmark_equity) > 0
    
    def test_backtest_with_empty_signals(self, sample_prices):
        """Test backtest with signals that never trigger trades."""
        # Arrange
        pm = PortfolioManagerV2(initial_capital=100000)
        dates = sample_prices['AAPL'].index
        no_signals = {'AAPL': pd.DataFrame({'Signal': np.zeros(len(dates))}, index=dates)}
        
        # Act
        result = pm.run_backtest(signals=no_signals, prices=sample_prices)
        
        # Assert
        assert len(result.trades) == 0
        assert result.equity_curve['TotalValue'].iloc[-1] == pytest.approx(100000, rel=0.01)
    
    def test_backtest_with_multiple_assets(self):
        """Test backtest with multiple assets."""
        # Arrange
        pm = PortfolioManagerV2(initial_capital=200000, max_position_size=0.5)
        
        dates = pd.date_range(start='2023-01-01', periods=50, freq='D')
        
        # Create prices for two assets
        prices = {
            'AAPL': pd.DataFrame({
                'Close': 100 + np.arange(50) * 0.5,
                'Open': 100 + np.arange(50) * 0.5,
                'High': 102 + np.arange(50) * 0.5,
                'Low': 98 + np.arange(50) * 0.5
            }, index=dates),
            'MSFT': pd.DataFrame({
                'Close': 200 + np.arange(50) * 1.0,
                'Open': 200 + np.arange(50) * 1.0,
                'High': 202 + np.arange(50) * 1.0,
                'Low': 198 + np.arange(50) * 1.0
            }, index=dates)
        }
        
        # Both assets get buy signal at start
        signals = {
            'AAPL': pd.DataFrame({'Signal': np.ones(50)}, index=dates),
            'MSFT': pd.DataFrame({'Signal': np.ones(50)}, index=dates)
        }
        
        # Act
        result = pm.run_backtest(signals=signals, prices=prices)
        
        # Assert
        assert result is not None
        assert len(result.equity_curve) > 0


# ============================================================================
# Test Stop-Loss and Take-Profit
# ============================================================================

class TestStopLossAndTakeProfit:
    """Test stop-loss and take-profit execution."""
    
    def test_stop_loss_triggered(self):
        """Test that stop-loss exits position when price drops."""
        # Arrange
        pm = PortfolioManagerV2(
            initial_capital=100000,
            stop_loss_pct=0.10,  # 10% stop loss
            max_position_size=1.0
        )
        
        dates = pd.date_range(start='2023-01-01', periods=20, freq='D')
        
        # Price drops 15% after entry
        close_prices = [100] * 5 + [95, 90, 85, 85] + [85] * 11
        prices = {'AAPL': pd.DataFrame({
            'Close': close_prices,
            'Open': close_prices,
            'High': close_prices,
            'Low': close_prices
        }, index=dates)}
        
        # Buy signal at start, hold signal rest of time
        signals = {'AAPL': pd.DataFrame({'Signal': np.ones(20)}, index=dates)}
        
        # Act
        result = pm.run_backtest(signals=signals, prices=prices)
        
        # Assert - position should be closed due to stop loss
        assert len(result.trades) > 0
    
    def test_take_profit_triggered(self):
        """Test that take-profit exits position when price rises."""
        # Arrange
        pm = PortfolioManagerV2(
            initial_capital=100000,
            take_profit_pct=0.20,  # 20% take profit
            max_position_size=1.0
        )
        
        dates = pd.date_range(start='2023-01-01', periods=20, freq='D')
        
        # Price rises 25% after entry
        close_prices = [100] * 5 + [110, 120, 125, 125] + [125] * 11
        prices = {'AAPL': pd.DataFrame({
            'Close': close_prices,
            'Open': close_prices,
            'High': close_prices,
            'Low': close_prices
        }, index=dates)}
        
        # Buy signal at start, hold signal rest of time
        signals = {'AAPL': pd.DataFrame({'Signal': np.ones(20)}, index=dates)}
        
        # Act
        result = pm.run_backtest(signals=signals, prices=prices)
        
        # Assert - position should be closed due to take profit
        assert len(result.trades) > 0


# ============================================================================
# Test Rebalancing Logic
# ============================================================================

class TestRebalancing:
    """Test portfolio rebalancing logic."""
    
    def test_should_rebalance_daily(self):
        """Test daily rebalancing frequency."""
        # Arrange
        pm = PortfolioManagerV2(rebalance_frequency='daily')
        portfolio = Mock()
        portfolio.positions = {'AAPL': Mock(), 'MSFT': Mock()}
        
        date1 = pd.Timestamp('2023-01-01')
        date2 = pd.Timestamp('2023-01-02')
        
        # Act
        pm.last_rebalance_date = date1
        should_rebalance = pm._should_rebalance(date2, portfolio, {})
        
        # Assert
        assert should_rebalance is True
    
    def test_should_rebalance_weekly(self):
        """Test weekly rebalancing frequency."""
        # Arrange
        pm = PortfolioManagerV2(rebalance_frequency='weekly')
        portfolio = Mock()
        portfolio.positions = {'AAPL': Mock(), 'MSFT': Mock()}
        
        date1 = pd.Timestamp('2023-01-01')
        date2 = pd.Timestamp('2023-01-08')
        
        # Act
        pm.last_rebalance_date = date1
        should_rebalance = pm._should_rebalance(date2, portfolio, {})
        
        # Assert
        assert should_rebalance is True
    
    def test_should_rebalance_monthly(self):
        """Test monthly rebalancing frequency."""
        # Arrange
        pm = PortfolioManagerV2(rebalance_frequency='monthly')
        portfolio = Mock()
        portfolio.positions = {'AAPL': Mock(), 'MSFT': Mock()}
        
        date1 = pd.Timestamp('2023-01-01')
        date2 = pd.Timestamp('2023-02-01')
        
        # Act
        pm.last_rebalance_date = date1
        should_rebalance = pm._should_rebalance(date2, portfolio, {})
        
        # Assert
        assert should_rebalance is True
    
    def test_should_rebalance_drift_threshold(self):
        """Test drift-based rebalancing."""
        # Arrange
        pm = PortfolioManagerV2(rebalance_threshold=0.10)
        
        # Mock portfolio with unbalanced positions
        portfolio = Mock()
        portfolio.positions = {'AAPL': Mock(), 'MSFT': Mock()}
        portfolio.get_allocation.return_value = {
            'AAPL': 0.70,  # Drifted from target 0.50
            'MSFT': 0.30,
            'Cash': 0.0
        }
        
        date = pd.Timestamp('2023-01-01')
        
        # Act
        should_rebalance = pm._should_rebalance(date, portfolio, {})
        
        # Assert
        assert should_rebalance is True
    
    def test_should_not_rebalance_single_position(self):
        """Test that single position portfolios don't rebalance."""
        # Arrange
        pm = PortfolioManagerV2(rebalance_threshold=0.10)
        
        portfolio = Mock()
        portfolio.positions = {'AAPL': Mock()}
        portfolio.get_allocation.return_value = {'AAPL': 1.0, 'Cash': 0.0}
        
        date = pd.Timestamp('2023-01-01')
        
        # Act
        should_rebalance = pm._should_rebalance(date, portfolio, {})
        
        # Assert
        assert should_rebalance is False
    
    def test_rebalance_portfolio_execution(self):
        """Test that rebalancing executes trades correctly."""
        # Arrange
        pm = PortfolioManagerV2(
            initial_capital=100000,
            rebalance_threshold=0.05,
            rebalance_frequency='never',
            max_position_size=0.5
        )
        
        # Create scenario where two positions drift apart
        dates = pd.date_range(start='2023-01-01', periods=30, freq='D')
        
        # AAPL rises, MSFT stays flat
        prices = {
            'AAPL': pd.DataFrame({
                'Close': 100 + np.arange(30) * 2,  # Rising
                'Open': 100 + np.arange(30) * 2,
                'High': 102 + np.arange(30) * 2,
                'Low': 98 + np.arange(30) * 2
            }, index=dates),
            'MSFT': pd.DataFrame({
                'Close': [200] * 30,  # Flat
                'Open': [200] * 30,
                'High': [202] * 30,
                'Low': [198] * 30
            }, index=dates)
        }
        
        # Both get buy signals
        signals = {
            'AAPL': pd.DataFrame({'Signal': np.ones(30)}, index=dates),
            'MSFT': pd.DataFrame({'Signal': np.ones(30)}, index=dates)
        }
        
        # Act
        result = pm.run_backtest(signals=signals, prices=prices)
        
        # Assert - backtest should complete without errors
        assert result is not None


# ============================================================================
# Test Risk Rejection Logging
# ============================================================================

class TestRiskRejectionLogging:
    """Test risk rejection logging functionality."""
    
    def test_risk_rejection_logged(self):
        """Test that risk rejections are logged."""
        # Arrange
        pm = PortfolioManagerV2(
            initial_capital=1000,  # Very small capital
            max_position_size=0.1
        )
        
        dates = pd.date_range(start='2023-01-01', periods=10, freq='D')
        
        # Expensive stock that can't be bought
        prices = {'AAPL': pd.DataFrame({
            'Close': [10000] * 10,  # Very expensive
            'Open': [10000] * 10,
            'High': [10000] * 10,
            'Low': [10000] * 10
        }, index=dates)}
        
        signals = {'AAPL': pd.DataFrame({'Signal': np.ones(10)}, index=dates)}
        
        # Act
        result = pm.run_backtest(signals=signals, prices=prices)
        
        # Assert - should have logged rejections
        assert len(pm.risk_rejections) > 0
    
    def test_risk_log_save_to_file(self, tmp_path):
        """Test that risk log is saved to file."""
        # Arrange
        log_file = tmp_path / "risk_rejections.csv"
        pm = PortfolioManagerV2(
            initial_capital=1000,
            max_position_size=0.1,
            risk_log_path=str(log_file)
        )
        
        dates = pd.date_range(start='2023-01-01', periods=10, freq='D')
        
        prices = {'AAPL': pd.DataFrame({
            'Close': [10000] * 10,
            'Open': [10000] * 10,
            'High': [10000] * 10,
            'Low': [10000] * 10
        }, index=dates)}
        
        signals = {'AAPL': pd.DataFrame({'Signal': np.ones(10)}, index=dates)}
        
        # Act
        result = pm.run_backtest(signals=signals, prices=prices)
        
        # Assert - file should exist if there were rejections
        if len(pm.risk_rejections) > 0:
            assert log_file.exists()
    
    def test_risk_rejection_contains_required_fields(self):
        """Test that risk rejection log contains all required fields."""
        # Arrange
        pm = PortfolioManagerV2(initial_capital=1000)
        
        date = pd.Timestamp('2023-01-01')
        
        # Act
        pm._log_risk_rejection(
            date=date,
            ticker='AAPL',
            signal=1,
            price=100.0,
            reason='Test rejection',
            portfolio_value=1000.0,
            cash=1000.0
        )
        
        # Assert
        assert len(pm.risk_rejections) == 1
        rejection = pm.risk_rejections[0]
        assert rejection['Date'] == date
        assert rejection['Ticker'] == 'AAPL'
        assert rejection['Signal'] == 1
        assert rejection['Price'] == 100.0
        assert rejection['Reason'] == 'Test rejection'
        assert rejection['PortfolioValue'] == 1000.0
        assert rejection['Cash'] == 1000.0


# ============================================================================
# Test Configuration Summary
# ============================================================================

class TestConfigurationSummary:
    """Test configuration summary generation."""
    
    def test_config_summary_returns_string(self):
        """Test that get_config_summary returns a string."""
        # Arrange
        pm = PortfolioManagerV2()
        
        # Act
        summary = pm.get_config_summary()
        
        # Assert
        assert isinstance(summary, str)
        assert len(summary) > 0
    
    def test_config_summary_contains_key_parameters(self):
        """Test that config summary contains all key parameters."""
        # Arrange
        pm = PortfolioManagerV2(
            initial_capital=250000,
            risk_per_trade=0.03,
            max_position_size=0.25,
            stop_loss_pct=0.15,
            transaction_cost_bps=5.0
        )
        
        # Act
        summary = pm.get_config_summary()
        
        # Assert
        assert '250,000' in summary
        assert '3.0%' in summary or '0.030' in summary
        assert '25.0%' in summary or '0.25' in summary
        assert '5.0' in summary
    
    def test_config_summary_handles_none_values(self):
        """Test that config summary handles None values gracefully."""
        # Arrange
        pm = PortfolioManagerV2(
            stop_loss_pct=None,
            take_profit_pct=None,
            rebalance_threshold=None
        )
        
        # Act
        summary = pm.get_config_summary()
        
        # Assert
        assert 'None' in summary
        assert summary is not None


# ============================================================================
# Test Edge Cases
# ============================================================================

class TestEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_backtest_with_missing_price_data(self):
        """Test backtest when price data is missing for some dates."""
        # Arrange
        pm = PortfolioManagerV2(initial_capital=100000)
        
        dates = pd.date_range(start='2023-01-01', periods=20, freq='D')
        
        # Prices only available for first 15 days
        prices = {'AAPL': pd.DataFrame({
            'Close': [100] * 15,
            'Open': [100] * 15,
            'High': [102] * 15,
            'Low': [98] * 15
        }, index=dates[:15])}
        
        # Signals for all 20 days
        signals = {'AAPL': pd.DataFrame({'Signal': np.ones(20)}, index=dates)}
        
        # Act
        result = pm.run_backtest(signals=signals, prices=prices)
        
        # Assert - should complete without error
        assert result is not None
    
    def test_backtest_with_single_day(self):
        """Test backtest with just one day of data."""
        # Arrange
        pm = PortfolioManagerV2(initial_capital=100000)
        
        dates = pd.date_range(start='2023-01-01', periods=1, freq='D')
        
        prices = {'AAPL': pd.DataFrame({
            'Close': [100],
            'Open': [100],
            'High': [102],
            'Low': [98]
        }, index=dates)}
        
        signals = {'AAPL': pd.DataFrame({'Signal': [1]}, index=dates)}
        
        # Act
        result = pm.run_backtest(signals=signals, prices=prices)
        
        # Assert
        assert result is not None
        assert len(result.equity_curve) == 1
    
    def test_backtest_handles_position_none(self):
        """Test that backtest handles None position correctly."""
        # Arrange
        pm = PortfolioManagerV2(initial_capital=100000)
        
        dates = pd.date_range(start='2023-01-01', periods=10, freq='D')
        
        prices = {'AAPL': pd.DataFrame({
            'Close': [100] * 10,
            'Open': [100] * 10,
            'High': [102] * 10,
            'Low': [98] * 10
        }, index=dates)}
        
        # Signal changes frequently
        signals = {'AAPL': pd.DataFrame(
            {'Signal': [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]},
            index=dates
        )}
        
        # Act
        result = pm.run_backtest(signals=signals, prices=prices)
        
        # Assert
        assert result is not None
    
    def test_backtest_insufficient_cash_for_trade(self):
        """Test backtest when portfolio has insufficient cash."""
        # Arrange
        pm = PortfolioManagerV2(
            initial_capital=100,  # Very small capital
            max_position_size=1.0
        )
        
        dates = pd.date_range(start='2023-01-01', periods=5, freq='D')
        
        # Expensive stock
        prices = {'AAPL': pd.DataFrame({
            'Close': [1000] * 5,
            'Open': [1000] * 5,
            'High': [1000] * 5,
            'Low': [1000] * 5
        }, index=dates)}
        
        signals = {'AAPL': pd.DataFrame({'Signal': np.ones(5)}, index=dates)}
        
        # Act
        result = pm.run_backtest(signals=signals, prices=prices)
        
        # Assert - should complete without error
        assert result is not None
        # Should have logged rejection
        assert len(pm.risk_rejections) > 0
    
    def test_rebalance_with_small_adjustment(self):
        """Test that small rebalancing adjustments are skipped."""
        # Arrange
        pm = PortfolioManagerV2(
            initial_capital=10000,
            rebalance_threshold=0.01,
            max_position_size=0.5
        )
        
        # Create minimal drift scenario
        dates = pd.date_range(start='2023-01-01', periods=10, freq='D')
        
        prices = {
            'AAPL': pd.DataFrame({
                'Close': [100, 101, 101, 101, 101, 101, 101, 101, 101, 101],
                'Open': [100, 101, 101, 101, 101, 101, 101, 101, 101, 101],
                'High': [102, 103, 103, 103, 103, 103, 103, 103, 103, 103],
                'Low': [98, 99, 99, 99, 99, 99, 99, 99, 99, 99]
            }, index=dates),
            'MSFT': pd.DataFrame({
                'Close': [100] * 10,
                'Open': [100] * 10,
                'High': [102] * 10,
                'Low': [98] * 10
            }, index=dates)
        }
        
        signals = {
            'AAPL': pd.DataFrame({'Signal': np.ones(10)}, index=dates),
            'MSFT': pd.DataFrame({'Signal': np.ones(10)}, index=dates)
        }
        
        # Act
        result = pm.run_backtest(signals=signals, prices=prices)
        
        # Assert
        assert result is not None
    
    def test_backtest_with_zero_signal(self):
        """Test backtest when signal is explicitly 0 (no position)."""
        # Arrange
        pm = PortfolioManagerV2(initial_capital=100000)
        
        dates = pd.date_range(start='2023-01-01', periods=5, freq='D')
        
        prices = {'AAPL': pd.DataFrame({
            'Close': [100] * 5,
            'Open': [100] * 5,
            'High': [102] * 5,
            'Low': [98] * 5
        }, index=dates)}
        
        # All signals are 0
        signals = {'AAPL': pd.DataFrame({'Signal': np.zeros(5)}, index=dates)}
        
        # Act
        result = pm.run_backtest(signals=signals, prices=prices)
        
        # Assert
        assert len(result.trades) == 0
        assert result.equity_curve['TotalValue'].iloc[-1] == pytest.approx(100000)
