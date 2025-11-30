"""
Unit tests for core portfolio functionality.

Tests critical features from:
- risk_manager.py: Risk limits, stop-loss checks, kill switches
- position_sizers.py: Position sizing calculations
- portfolio_manager_v2.py: Backtest orchestration

Run with: pytest tests/test_portfolio_core.py -v
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

import sys
sys.path.append('..')

from core.portfolio.risk_manager import RiskManager, RiskConfig
from core.portfolio.position_sizers import (
    FixedFractionalSizer, KellySizer, ATRSizer, 
    VolatilityScaledSizer, RiskParitySizer
)
from core.portfolio.portfolio import Portfolio, Position
from core.portfolio.portfolio_manager_v2 import PortfolioManagerV2


# ============================================================================
# Fixtures - Shared test data
# ============================================================================

@pytest.fixture
def basic_risk_config():
    """Standard risk configuration for testing."""
    return RiskConfig(
        risk_per_trade=0.02,
        max_position_size=0.20,
        stop_loss_pct=0.10,
        take_profit_pct=0.25,
        max_drawdown_pct=0.15,
        max_daily_loss_pct=0.03,
        min_trade_value=100.0
    )


@pytest.fixture
def sample_portfolio():
    """Create a sample portfolio with initial capital."""
    portfolio = Portfolio(initial_capital=100000)
    return portfolio


@pytest.fixture
def sample_price_data():
    """Generate sample OHLCV price data for testing."""
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
    
    # Generate realistic price movements
    np.random.seed(42)
    base_price = 100
    returns = np.random.normal(0.001, 0.02, len(dates))
    prices = base_price * (1 + returns).cumprod()
    
    data = pd.DataFrame({
        'Date': dates,
        'Open': prices * (1 + np.random.uniform(-0.01, 0.01, len(dates))),
        'High': prices * (1 + np.random.uniform(0, 0.02, len(dates))),
        'Low': prices * (1 - np.random.uniform(0, 0.02, len(dates))),
        'Close': prices,
        'Volume': np.random.randint(1000000, 5000000, len(dates))
    })
    
    data['High'] = data[['Open', 'High', 'Close']].max(axis=1)
    data['Low'] = data[['Open', 'Low', 'Close']].min(axis=1)
    
    return data


# ============================================================================
# Position Sizer Tests
# ============================================================================

class TestFixedFractionalSizer:
    """Test FixedFractionalSizer position sizing logic."""
    
    def test_basic_position_size(self):
        """Test that position size respects max_position_pct."""
        sizer = FixedFractionalSizer(max_position_pct=0.20, risk_per_trade=0.02)
        
        shares = sizer.calculate_size(
            ticker='AAPL',
            signal=1.0,
            current_price=150.0,
            portfolio_value=100000
        )
        
        # With 20% max position: $20,000 / $150 = 133 shares
        position_value = shares * 150.0
        assert position_value <= 20000, "Position exceeds max 20% of portfolio"
        assert shares > 0, "Should return positive shares"
    
    def test_risk_based_sizing(self):
        """Test risk-based sizing with stop-loss."""
        sizer = FixedFractionalSizer(max_position_pct=0.20, risk_per_trade=0.02)
        
        shares = sizer.calculate_size(
            ticker='AAPL',
            signal=1.0,
            current_price=100.0,
            portfolio_value=100000,
            stop_loss_pct=0.10  # 10% stop-loss
        )
        
        # Risk-based: Risk $2,000 (2% of $100k) with 10% stop = $2,000 / $10 = 200 shares
        # Size-based: 20% of $100k = $20,000 / $100 = 200 shares
        # Should take minimum = 200 shares
        assert shares == 200, f"Expected 200 shares, got {shares}"
    
    def test_signal_scaling(self):
        """Test that weaker signals result in smaller positions."""
        sizer = FixedFractionalSizer(max_position_pct=0.20)
        
        shares_full = sizer.calculate_size(
            ticker='AAPL', signal=1.0, current_price=100.0, portfolio_value=100000
        )
        
        shares_half = sizer.calculate_size(
            ticker='AAPL', signal=0.5, current_price=100.0, portfolio_value=100000
        )
        
        assert shares_half == shares_full / 2, "Half signal should give half position"
    
    def test_minimum_trade_value(self):
        """Test that trades below minimum value are rejected."""
        sizer = FixedFractionalSizer(max_position_pct=0.20, min_trade_value=1000.0)
        
        # Small portfolio, expensive asset = tiny position below min
        shares = sizer.calculate_size(
            ticker='BRK.A',
            signal=1.0,
            current_price=500000.0,  # Berkshire Hathaway A shares
            portfolio_value=5000
        )
        
        assert shares == 0, "Should reject trades below minimum value"
    
    def test_edge_cases(self):
        """Test edge cases: zero price, zero portfolio, negative signal."""
        sizer = FixedFractionalSizer(max_position_pct=0.20)
        
        # Zero price
        assert sizer.calculate_size('AAPL', 1.0, 0.0, 100000) == 0
        
        # Zero portfolio
        assert sizer.calculate_size('AAPL', 1.0, 100.0, 0.0) == 0
        
        # Negative signal (short)
        shares_short = sizer.calculate_size('AAPL', -0.5, 100.0, 100000)
        shares_long = sizer.calculate_size('AAPL', 0.5, 100.0, 100000)
        assert shares_short == shares_long, "Negative signal should give same size (abs value)"


class TestKellySizer:
    """Test Kelly Criterion position sizing."""
    
    def test_kelly_calculation(self):
        """Test Kelly formula with known inputs."""
        sizer = KellySizer(max_position_pct=0.30, kelly_fraction=1.0)
        
        # Kelly formula: f* = (p*W - (1-p)) / W
        # p=0.6, W=1.5 (avg_win=0.03, avg_loss=0.02)
        # f* = (0.6*1.5 - 0.4) / 1.5 = 0.333... (33.3%)
        
        shares = sizer.calculate_size(
            ticker='AAPL',
            signal=1.0,
            current_price=100.0,
            portfolio_value=100000,
            win_rate=0.60,
            avg_win=0.03,
            avg_loss=0.02
        )
        
        position_value = shares * 100.0
        kelly_pct = position_value / 100000
        
        # Should be capped at max_position_pct (30%)
        assert kelly_pct <= 0.30, f"Kelly exceeds max: {kelly_pct:.2%}"
        assert kelly_pct > 0, "Kelly should be positive with edge"
    
    def test_half_kelly(self):
        """Test that half-Kelly fraction reduces position size."""
        sizer_full = KellySizer(kelly_fraction=1.0)
        sizer_half = KellySizer(kelly_fraction=0.5)
        
        kwargs = {
            'ticker': 'AAPL', 'signal': 1.0, 'current_price': 100.0,
            'portfolio_value': 100000, 'win_rate': 0.55,
            'avg_win': 0.04, 'avg_loss': 0.02
        }
        
        shares_full = sizer_full.calculate_size(**kwargs)
        shares_half = sizer_half.calculate_size(**kwargs)
        
        assert shares_half < shares_full, "Half Kelly should be smaller"
        assert shares_half >= shares_full * 0.4, "Half Kelly roughly half of full"
    
    def test_no_edge_zero_position(self):
        """Test that negative edge results in zero position."""
        sizer = KellySizer()
        
        # Win rate too low for edge
        shares = sizer.calculate_size(
            ticker='AAPL',
            signal=1.0,
            current_price=100.0,
            portfolio_value=100000,
            win_rate=0.40,  # 40% wins
            avg_win=0.02,
            avg_loss=0.02  # Equal win/loss, no edge
        )
        
        assert shares == 0, "No edge should result in zero position"


class TestATRSizer:
    """Test ATR-based volatility position sizing."""
    
    def test_inverse_volatility_relationship(self):
        """Test that higher ATR (volatility) results in smaller positions."""
        sizer = ATRSizer(risk_per_trade=0.02, atr_multiplier=2.0, max_position_pct=0.20)
        
        # Low volatility (small ATR)
        shares_low_vol = sizer.calculate_size(
            ticker='AAPL', signal=1.0, current_price=100.0,
            portfolio_value=100000, atr=2.0  # Low ATR
        )
        
        # High volatility (large ATR)
        shares_high_vol = sizer.calculate_size(
            ticker='AAPL', signal=1.0, current_price=100.0,
            portfolio_value=100000, atr=10.0  # High ATR (5x larger)
        )
        
        # With larger ATR, stop distance is larger, so we can afford fewer shares
        # risk_amount / stop_distance = fewer shares when stop_distance is larger
        assert shares_high_vol < shares_low_vol, f"Higher ATR should reduce position: low_vol={shares_low_vol}, high_vol={shares_high_vol}"
    
    def test_atr_required(self):
        """Test that ATR is required for calculation."""
        sizer = ATRSizer()
        
        # Should fall back to max_position_pct if no ATR
        shares = sizer.calculate_size(
            ticker='AAPL', signal=1.0, current_price=100.0,
            portfolio_value=100000  # No ATR provided
        )
        
        assert shares > 0, "Should use fallback when ATR missing"


# ============================================================================
# Risk Manager Tests
# ============================================================================

class TestRiskManager:
    """Test RiskManager risk enforcement logic."""
    
    def test_position_size_delegation(self, basic_risk_config):
        """Test that RiskManager correctly delegates to PositionSizer."""
        sizer = FixedFractionalSizer(max_position_pct=0.20)
        risk_mgr = RiskManager(basic_risk_config, position_sizer=sizer)
        
        shares = risk_mgr.calculate_position_size(
            ticker='AAPL',
            signal=1.0,
            current_price=100.0,
            portfolio_value=100000
        )
        
        assert shares > 0, "Should delegate to position sizer"
        assert shares * 100.0 <= 20000, "Should respect max position size"
    
    def test_stop_loss_check(self, basic_risk_config):
        """Test stop-loss violation detection."""
        risk_mgr = RiskManager(basic_risk_config)
        
        # Create a position that's down 15% (beyond 10% stop)
        position = Position(
            ticker='AAPL',
            shares=100,
            entry_price=100.0,
            current_price=85.0,  # Down 15%
            entry_date=pd.Timestamp('2023-01-01')
        )
        
        should_exit = risk_mgr.check_stop_loss(position)
        
        assert should_exit, "Should trigger stop-loss exit at -15%"
    
    def test_take_profit_check(self, basic_risk_config):
        """Test take-profit target detection."""
        risk_mgr = RiskManager(basic_risk_config)
        
        # Position up 30% (beyond 25% take-profit)
        position = Position(
            ticker='AAPL',
            shares=100,
            entry_price=100.0,
            current_price=130.0,  # Up 30%
            entry_date=pd.Timestamp('2023-01-01')
        )
        
        should_exit = risk_mgr.check_take_profit(position)
        
        assert should_exit, "Should trigger take-profit exit at +30%"
    
    def test_concentration_limit(self, basic_risk_config, sample_portfolio):
        """Test concentration limit enforcement."""
        risk_mgr = RiskManager(basic_risk_config)
        
        # Try to open position that would be 25% of portfolio (above 20% limit)
        can_trade = risk_mgr.check_concentration_limit(
            ticker='AAPL',
            new_position_value=25000,  # 25% of $100K
            portfolio_value=sample_portfolio.total_value
        )
        
        assert not can_trade, "Should reject position exceeding 20% concentration limit"
        
        # Try position within limits (15%)
        can_trade_ok = risk_mgr.check_concentration_limit(
            ticker='MSFT',
            new_position_value=15000,  # 15% of $100K
            portfolio_value=sample_portfolio.total_value
        )
        
        assert can_trade_ok, "Should allow position within concentration limits"
    
    def test_var_calculation(self, basic_risk_config):
        """Test Value at Risk (VaR) calculation."""
        risk_mgr = RiskManager(basic_risk_config)
        
        # Create sample returns
        np.random.seed(42)
        returns = pd.Series(np.random.normal(0.001, 0.02, 252))  # 1 year of daily returns
        
        # Calculate 95% VaR
        var_95 = risk_mgr.calculate_var(returns, confidence=0.95)
        
        assert var_95 < 0, "VaR should be negative (represents loss)"
        assert var_95 > returns.min(), "VaR should be less severe than worst loss"
    
    def test_cvar_calculation(self, basic_risk_config):
        """Test Conditional VaR (Expected Shortfall) calculation."""
        risk_mgr = RiskManager(basic_risk_config)
        
        # Create sample returns
        np.random.seed(42)
        returns = pd.Series(np.random.normal(0.001, 0.02, 252))
        
        # Calculate 95% CVaR
        cvar_95 = risk_mgr.calculate_cvar(returns, confidence=0.95)
        var_95 = risk_mgr.calculate_var(returns, confidence=0.95)
        
        assert cvar_95 < var_95, "CVaR should be more severe than VaR (tail risk)"


# ============================================================================
# Portfolio Tests
# ============================================================================

class TestPortfolio:
    """Test Portfolio position management."""
    
    def test_open_position(self, sample_portfolio):
        """Test opening a new position."""
        initial_cash = sample_portfolio.cash
        
        sample_portfolio.open_position(
            ticker='AAPL',
            shares=100,
            price=150.0,
            date=pd.Timestamp('2023-01-01')
        )
        
        assert sample_portfolio.has_position('AAPL'), "Should have AAPL position"
        assert sample_portfolio.cash == initial_cash - 15000, "Cash should decrease"
        
        position = sample_portfolio.get_position('AAPL')
        assert position.shares == 100
        assert position.entry_price == 150.0
    
    def test_close_position(self, sample_portfolio):
        """Test closing a position and realizing P&L."""
        sample_portfolio.open_position('AAPL', 100, 150.0, pd.Timestamp('2023-01-01'))
        
        # Close with profit
        sample_portfolio.close_position('AAPL', 160.0, pd.Timestamp('2023-02-01'))
        
        assert not sample_portfolio.has_position('AAPL'), "Position should be closed"
        assert len(sample_portfolio.closed_positions) == 1, "Should record closed trade"
        
        trade = sample_portfolio.closed_positions[0]
        assert trade['pnl'] == 1000, "P&L should be $1,000"
        assert trade['return'] == 160.0/150.0 - 1, "Return should be 6.67%"
    
    def test_update_prices(self, sample_portfolio):
        """Test updating position values with new prices."""
        sample_portfolio.open_position('AAPL', 100, 150.0, pd.Timestamp('2023-01-01'))
        sample_portfolio.open_position('MSFT', 50, 200.0, pd.Timestamp('2023-01-01'))
        
        # Update prices
        new_prices = {'AAPL': 160.0, 'MSFT': 210.0}
        sample_portfolio.update_prices(new_prices)
        
        aapl_pos = sample_portfolio.get_position('AAPL')
        assert aapl_pos.current_price == 160.0
        assert aapl_pos.pnl == 1000, "AAPL should have $1,000 unrealized profit"
        
        msft_pos = sample_portfolio.get_position('MSFT')
        assert msft_pos.pnl == 500, "MSFT should have $500 unrealized profit"
    
    def test_total_value_calculation(self, sample_portfolio):
        """Test total portfolio value calculation."""
        sample_portfolio.open_position('AAPL', 100, 150.0, pd.Timestamp('2023-01-01'))
        
        initial_total = sample_portfolio.total_value
        
        # Update with price gain
        sample_portfolio.update_prices({'AAPL': 160.0})
        
        new_total = sample_portfolio.total_value
        assert new_total == initial_total + 1000, "Total value should increase by $1,000"


# ============================================================================
# Integration Test
# ============================================================================

class TestPortfolioManagerIntegration:
    """Integration test for PortfolioManagerV2 with real-ish workflow."""
    
    def test_simple_backtest(self, sample_price_data):
        """Test a simple backtest with buy-and-hold signals."""
        # Create simple buy-and-hold signals
        signals = pd.DataFrame({
            'Date': sample_price_data['Date'],
            'Signal': 1  # Always long
        }).set_index('Date')
        
        # Run backtest
        pm = PortfolioManagerV2(
            initial_capital=100000,
            risk_per_trade=0.02,
            max_position_size=1.0,  # 100% allocation
            transaction_cost_bps=5.0,
            stop_loss_pct=0.10
        )
        
        prices_dict = {
            'TEST': sample_price_data.set_index('Date')[['Open', 'High', 'Low', 'Close']]
        }
        signals_dict = {'TEST': signals}
        
        result = pm.run_backtest(signals_dict, prices_dict)
        
        # Basic sanity checks
        assert result.final_equity > 0, "Should have positive final equity"
        assert result.metrics['Total Trades'] >= 0, "Should track trades"
        assert 'Total Return' in result.metrics, "Should calculate return"
        
        # If prices went up and we were long, we should profit
        price_return = (sample_price_data['Close'].iloc[-1] / 
                       sample_price_data['Close'].iloc[0] - 1)
        
        if price_return > 0.05:  # If prices up >5%
            assert result.total_return > 0, "Should profit in uptrend with long signals"


# ============================================================================
# Run Tests
# ============================================================================

if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
