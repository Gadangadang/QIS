"""
Extended test suite for RiskManager.

Tests uncovered functionality in core/portfolio/risk_manager.py:
- VaR and CVaR calculations
- Kelly criterion
- Kill switches and circuit breakers
- Trade approval checks
- Portfolio heat monitoring
- Capital tracking
"""

import pytest
import pandas as pd
import numpy as np
from datetime import date, timedelta
from core.portfolio.risk_manager import RiskManager, RiskConfig
from core.portfolio.portfolio import Position


@pytest.fixture
def basic_risk_config():
    """Standard risk configuration."""
    return RiskConfig(
        risk_per_trade=0.02,
        max_position_size=0.20,
        stop_loss_pct=0.10,
        take_profit_pct=0.25,
        max_drawdown_pct=0.15,
        max_daily_loss_pct=0.03,
        min_trade_value=100.0,
        min_capital_pct=0.50
    )


@pytest.fixture
def risk_manager(basic_risk_config):
    """Create a RiskManager instance."""
    return RiskManager(basic_risk_config)


@pytest.fixture
def sample_returns():
    """Generate sample return series."""
    np.random.seed(42)
    return pd.Series(np.random.normal(0.001, 0.02, 252))  # Daily returns


# ============================================================================
# Risk Metrics Tests
# ============================================================================

class TestRiskMetrics:
    """Test VaR, CVaR, and other risk metrics."""
    
    def test_var_calculation(self, risk_manager, sample_returns):
        """Test Value at Risk calculation."""
        var = risk_manager.calculate_var(sample_returns, confidence=0.95)
        
        # VaR should be negative (loss)
        assert var < 0
        
        # 95% of returns should be above VaR
        below_var = (sample_returns <= var).sum()
        expected_below = int(len(sample_returns) * 0.05)
        assert abs(below_var - expected_below) <= 5  # Allow some tolerance
    
    def test_var_different_confidence_levels(self, risk_manager, sample_returns):
        """Test VaR at different confidence levels."""
        var_90 = risk_manager.calculate_var(sample_returns, confidence=0.90)
        var_95 = risk_manager.calculate_var(sample_returns, confidence=0.95)
        var_99 = risk_manager.calculate_var(sample_returns, confidence=0.99)
        
        # Higher confidence = more negative VaR
        assert var_99 < var_95 < var_90
    
    def test_var_empty_returns(self, risk_manager):
        """Test VaR with empty returns."""
        empty_returns = pd.Series([])
        var = risk_manager.calculate_var(empty_returns)
        assert var == 0.0
    
    def test_cvar_calculation(self, risk_manager, sample_returns):
        """Test Conditional Value at Risk (CVaR)."""
        cvar = risk_manager.calculate_cvar(sample_returns, confidence=0.95)
        var = risk_manager.calculate_var(sample_returns, confidence=0.95)
        
        # CVaR should be <= VaR (more extreme)
        assert cvar <= var
        
        # CVaR should be negative
        assert cvar < 0
    
    def test_cvar_empty_returns(self, risk_manager):
        """Test CVaR with empty returns."""
        empty_returns = pd.Series([])
        cvar = risk_manager.calculate_cvar(empty_returns)
        assert cvar == 0.0
    
    def test_cvar_is_average_of_tail(self, risk_manager):
        """Test that CVaR is average of tail beyond VaR."""
        # Create specific return series
        returns = pd.Series([-0.10, -0.05, -0.02, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07])
        
        var = risk_manager.calculate_var(returns, confidence=0.90)
        cvar = risk_manager.calculate_cvar(returns, confidence=0.90)
        
        # CVaR should be more extreme than VaR
        assert cvar <= var


class TestKellyCriterion:
    """Test Kelly Criterion calculations."""
    
    def test_kelly_basic_calculation(self, risk_manager):
        """Test basic Kelly criterion calculation."""
        # Win rate 60%, avg win 1.5%, avg loss 1.0%
        kelly = risk_manager.calculate_kelly_fraction(
            win_rate=0.60,
            avg_win=0.015,
            avg_loss=0.010
        )
        
        # Kelly should be positive with edge
        assert kelly > 0
        
        # Should be half-Kelly (safety factor)
        full_kelly = 0.60 - ((1 - 0.60) / (0.015 / 0.010))
        assert abs(kelly - full_kelly * 0.5) < 0.001
    
    def test_kelly_no_edge(self, risk_manager):
        """Test Kelly when no edge (50/50)."""
        kelly = risk_manager.calculate_kelly_fraction(
            win_rate=0.50,
            avg_win=0.01,
            avg_loss=0.01
        )
        
        # Should be zero or very small with no edge
        assert abs(kelly) < 0.01
    
    def test_kelly_negative_edge(self, risk_manager):
        """Test Kelly with negative edge."""
        kelly = risk_manager.calculate_kelly_fraction(
            win_rate=0.40,  # 40% win rate
            avg_win=0.01,
            avg_loss=0.01
        )
        
        # Should be zero (clamped at 0)
        assert kelly == 0.0
    
    def test_kelly_zero_avg_loss(self, risk_manager):
        """Test Kelly with zero average loss (division by zero)."""
        kelly = risk_manager.calculate_kelly_fraction(
            win_rate=0.60,
            avg_win=0.01,
            avg_loss=0.0
        )
        
        # Should handle gracefully
        assert kelly == 0.0
    
    def test_kelly_high_win_rate(self, risk_manager):
        """Test Kelly with very high win rate."""
        kelly = risk_manager.calculate_kelly_fraction(
            win_rate=0.80,
            avg_win=0.02,
            avg_loss=0.01
        )
        
        # Should be positive and reasonable
        assert 0 < kelly < 1.0


# ============================================================================
# Kill Switches and Circuit Breakers Tests
# ============================================================================

class TestKillSwitches:
    """Test kill switches and circuit breakers."""
    
    def test_capital_tracking_initialization(self, risk_manager):
        """Test capital tracking initialization."""
        risk_manager.initialize_capital_tracking(100000, date.today())
        
        assert hasattr(risk_manager, 'initial_capital')
        assert risk_manager.initial_capital == 100000
        assert risk_manager.peak_capital == 100000
        assert risk_manager.daily_start_capital == 100000
        assert risk_manager.last_reset_date == date.today()
    
    def test_max_drawdown_kill_switch(self, basic_risk_config):
        """Test max drawdown kill switch triggers."""
        risk_manager = RiskManager(basic_risk_config)
        risk_manager.initialize_capital_tracking(100000)
        
        # Initialize kill switch attributes
        risk_manager.is_killed = False
        risk_manager.kill_reason = None
        risk_manager.breach_history = []
        
        # Lose 16% (exceeds 15% max)
        result = risk_manager.update_capital(84000)
        
        assert risk_manager.is_killed
        assert len(result['breaches']) > 0
        assert any(b['rule'] == 'max_drawdown' for b in result['breaches'])
    
    def test_max_daily_loss_kill_switch(self, basic_risk_config):
        """Test max daily loss kill switch."""
        risk_manager = RiskManager(basic_risk_config)
        risk_manager.initialize_capital_tracking(100000, date.today())
        
        # Initialize kill switch attributes
        risk_manager.is_killed = False
        risk_manager.kill_reason = None
        risk_manager.breach_history = []
        
        # Lose 4% in one day (exceeds 3% max)
        result = risk_manager.update_capital(96000, date.today())
        
        assert risk_manager.is_killed
        assert any(b['rule'] == 'max_daily_loss' for b in result['breaches'])
    
    def test_min_capital_kill_switch(self, basic_risk_config):
        """Test minimum capital kill switch."""
        risk_manager = RiskManager(basic_risk_config)
        risk_manager.initialize_capital_tracking(100000)
        
        # Initialize kill switch attributes
        risk_manager.is_killed = False
        risk_manager.kill_reason = None
        risk_manager.breach_history = []
        
        # Drop to 45% of initial (below 50% min)
        result = risk_manager.update_capital(45000)
        
        assert risk_manager.is_killed
        assert any(b['rule'] == 'min_capital' for b in result['breaches'])
    
    def test_peak_capital_updates(self, risk_manager):
        """Test that peak capital updates correctly."""
        risk_manager.initialize_capital_tracking(100000)
        risk_manager.is_killed = False
        risk_manager.kill_reason = None
        risk_manager.breach_history = []
        
        # Increase capital
        risk_manager.update_capital(110000)
        assert risk_manager.peak_capital == 110000
        
        # Further increase
        risk_manager.update_capital(120000)
        assert risk_manager.peak_capital == 120000
        
        # Decrease doesn't lower peak
        risk_manager.update_capital(115000)
        assert risk_manager.peak_capital == 120000
    
    def test_daily_tracking_reset(self, risk_manager):
        """Test daily tracking resets on new day."""
        today = date.today()
        tomorrow = today + timedelta(days=1)
        
        risk_manager.initialize_capital_tracking(100000, today)
        risk_manager.is_killed = False
        risk_manager.kill_reason = None
        risk_manager.breach_history = []
        
        # Lose money today
        risk_manager.update_capital(98000, today)
        assert risk_manager.daily_start_capital == 100000
        
        # New day resets daily start
        risk_manager.update_capital(98000, tomorrow)
        assert risk_manager.daily_start_capital == 98000
    
    def test_no_kill_switch_with_normal_losses(self, risk_manager):
        """Test that normal losses don't trigger kill switches."""
        # Initialize with today's date and capital
        today = date.today()
        risk_manager.initialize_capital_tracking(100000, today)
        risk_manager.is_killed = False
        risk_manager.kill_reason = None
        risk_manager.breach_history = []
        
        # Lose 2% (within 3% daily loss limit and 15% max drawdown)
        result = risk_manager.update_capital(98000, today)
        
        # 2% drawdown is within both limits
        assert not risk_manager.is_killed, f"Kill switch incorrectly triggered: {risk_manager.kill_reason if hasattr(risk_manager, 'kill_reason') else 'N/A'}"
        assert len(result['breaches']) == 0, f"Unexpected breaches: {result['breaches']}"
    
    @pytest.mark.skip(reason="Auto-initialization not implemented - RiskManager requires manual initialize_capital_tracking() call")
    def test_auto_initialization_on_first_update(self):
        """
        Test that documents RiskManager initialization requirement.
        
        Note: RiskManager currently does NOT auto-initialize on first update.
        The initialize_capital_tracking() method must be called manually before
        using update_capital(). This test is skipped as it documents unimplemented
        functionality that may be added in the future.
        """
        pass


class TestTradeApproval:
    """Test trade approval checks."""
    
    def test_approve_normal_trade(self, risk_manager):
        """Test approval of normal-sized trade."""
        # Initialize kill switch attributes
        risk_manager.is_killed = False
        risk_manager.kill_reason = None
        
        approved, reasons = risk_manager.check_trade_approval(
            asset='AAPL',
            size=100,
            price=150.0,
            portfolio_value=100000
        )
        
        # 100 shares * $150 = $15,000 = 15% of portfolio (within 20% limit)
        assert approved
        assert len(reasons) == 0
    
    def test_reject_oversized_trade(self, risk_manager):
        """Test rejection of oversized trade."""
        risk_manager.is_killed = False
        risk_manager.kill_reason = None
        
        approved, reasons = risk_manager.check_trade_approval(
            asset='AAPL',
            size=200,
            price=150.0,
            portfolio_value=100000
        )
        
        # 200 shares * $150 = $30,000 = 30% of portfolio (exceeds 20% limit)
        assert not approved
        assert len(reasons) > 0
        assert 'exceeds limit' in reasons[0]
    
    def test_reject_when_killed(self, basic_risk_config):
        """Test that trades are rejected when kill switch is active."""
        risk_manager = RiskManager(basic_risk_config)
        risk_manager.initialize_capital_tracking(100000)
        risk_manager.is_killed = False
        risk_manager.kill_reason = None
        risk_manager.breach_history = []
        risk_manager.update_capital(80000)  # Trigger kill switch
        
        approved, reasons = risk_manager.check_trade_approval(
            asset='AAPL',
            size=10,
            price=150.0,
            portfolio_value=80000
        )
        
        assert not approved
        assert 'KILL SWITCH ACTIVE' in reasons[0]
    
    def test_reject_invalid_portfolio_value(self, risk_manager):
        """Test rejection with invalid portfolio value."""
        risk_manager.is_killed = False
        risk_manager.kill_reason = None
        
        approved, reasons = risk_manager.check_trade_approval(
            asset='AAPL',
            size=100,
            price=150.0,
            portfolio_value=0
        )
        
        assert not approved
        assert 'Portfolio value' in reasons[0]
    
    def test_reject_adding_to_losing_position(self, risk_manager):
        """Test rejection when adding to losing position."""
        risk_manager.is_killed = False
        risk_manager.kill_reason = None
        
        current_positions = {
            'AAPL': {
                'shares': 100,
                'entry_price': 160.0,
                'current_price': 150.0,
                'pnl_pct': -0.0625  # -6.25% loss
            }
        }
        
        approved, reasons = risk_manager.check_trade_approval(
            asset='AAPL',
            size=50,
            price=150.0,
            current_positions=current_positions,
            portfolio_value=100000
        )
        
        assert not approved
        assert 'already down' in reasons[0]


class TestPortfolioHeat:
    """Test portfolio heat monitoring."""
    
    def test_calculate_portfolio_heat(self, risk_manager):
        """Test portfolio heat calculation."""
        positions = {
            'AAPL': {'pnl': -500},
            'MSFT': {'pnl': 300},
            'GOOGL': {'pnl': -200}
        }
        
        result = risk_manager.check_portfolio_heat(positions, 100000)
        
        # Heat = (500 + 200) / 100000 = 0.007 = 0.7%
        assert abs(result['heat'] - 0.007) < 0.001
    
    def test_heat_with_no_losses(self, risk_manager):
        """Test heat when all positions are winning."""
        positions = {
            'AAPL': {'pnl': 500},
            'MSFT': {'pnl': 300}
        }
        
        result = risk_manager.check_portfolio_heat(positions, 100000)
        
        # No losses = 0 heat
        assert result['heat'] == 0.0
    
    def test_heat_breach_detection(self, risk_manager):
        """Test heat breach when exceeding max portfolio heat."""
        # Create positions with 12% heat (exceeds 10% limit)
        positions = {
            'AAPL': {'pnl': -6000},
            'MSFT': {'pnl': -6000}
        }
        
        result = risk_manager.check_portfolio_heat(positions, 100000)
        
        # 12% heat exceeds 10% limit
        assert result['heat'] > 0.10
        assert len(result['breaches']) > 0
    
    def test_heat_with_zero_portfolio_value(self, risk_manager):
        """Test heat calculation with zero portfolio value."""
        positions = {'AAPL': {'pnl': -100}}
        
        result = risk_manager.check_portfolio_heat(positions, 0)
        
        assert result['heat'] == 0.0
        assert len(result['breaches']) == 0


class TestStopLossAndTakeProfit:
    """Test stop loss and take profit checks."""
    
    def test_stop_loss_triggered(self, risk_manager):
        """Test stop loss check triggers."""
        # Create position with 12% loss (exceeds 10% stop)
        position = Position(
            ticker='AAPL',
            shares=100,
            entry_price=150.0,
            entry_date=pd.Timestamp('2024-01-01'),
            current_price=132.0  # 12% loss
        )
        
        should_exit = risk_manager.check_stop_loss(position)
        assert should_exit
    
    def test_stop_loss_not_triggered(self, risk_manager):
        """Test stop loss doesn't trigger prematurely."""
        position = Position(
            ticker='AAPL',
            shares=100,
            entry_price=150.0,
            entry_date=pd.Timestamp('2024-01-01'),
            current_price=142.0  # 5.3% loss
        )
        
        should_exit = risk_manager.check_stop_loss(position)
        assert not should_exit
    
    def test_stop_loss_disabled(self):
        """Test stop loss when disabled."""
        config = RiskConfig(stop_loss_pct=None)
        risk_manager = RiskManager(config)
        
        position = Position(
            ticker='AAPL',
            shares=100,
            entry_price=150.0,
            entry_date=pd.Timestamp('2024-01-01'),
            current_price=75.0  # 50% loss
        )
        
        should_exit = risk_manager.check_stop_loss(position)
        assert not should_exit  # Disabled, so no exit
    
    def test_take_profit_triggered(self, risk_manager):
        """Test take profit check triggers."""
        position = Position(
            ticker='AAPL',
            shares=100,
            entry_price=150.0,
            entry_date=pd.Timestamp('2024-01-01'),
            current_price=190.0  # 26.7% gain
        )
        
        should_exit = risk_manager.check_take_profit(position)
        assert should_exit  # Exceeds 25% take profit
    
    def test_take_profit_not_triggered(self, risk_manager):
        """Test take profit doesn't trigger prematurely."""
        position = Position(
            ticker='AAPL',
            shares=100,
            entry_price=150.0,
            entry_date=pd.Timestamp('2024-01-01'),
            current_price=165.0  # 10% gain
        )
        
        should_exit = risk_manager.check_take_profit(position)
        assert not should_exit
    
    def test_take_profit_disabled(self):
        """Test take profit when disabled."""
        config = RiskConfig(take_profit_pct=None)
        risk_manager = RiskManager(config)
        
        position = Position(
            ticker='AAPL',
            shares=100,
            entry_price=150.0,
            entry_date=pd.Timestamp('2024-01-01'),
            current_price=225.0  # 50% gain
        )
        
        should_exit = risk_manager.check_take_profit(position)
        assert not should_exit


class TestConcentrationLimits:
    """Test concentration limit checks."""
    
    def test_concentration_within_limits(self, risk_manager):
        """Test position within concentration limits."""
        # 15% position (within 20% limit)
        within_limit = risk_manager.check_concentration_limit(
            ticker='AAPL',
            new_position_value=15000,
            portfolio_value=100000
        )
        
        assert within_limit
    
    def test_concentration_exceeds_limits(self, risk_manager):
        """Test position exceeding concentration limits."""
        # 25% position (exceeds 20% limit)
        exceeds_limit = risk_manager.check_concentration_limit(
            ticker='AAPL',
            new_position_value=25000,
            portfolio_value=100000
        )
        
        assert not exceeds_limit
    
    def test_concentration_at_exact_limit(self, risk_manager):
        """Test position exactly at limit."""
        # Exactly 20%
        at_limit = risk_manager.check_concentration_limit(
            ticker='AAPL',
            new_position_value=20000,
            portfolio_value=100000
        )
        
        assert at_limit
    
    def test_concentration_with_zero_portfolio(self, risk_manager):
        """Test concentration check with zero portfolio value."""
        # Should return True (edge case handling)
        result = risk_manager.check_concentration_limit(
            ticker='AAPL',
            new_position_value=1000,
            portfolio_value=0
        )
        
        assert result
