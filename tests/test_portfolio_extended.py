"""
Comprehensive unit tests for Portfolio class.
Tests for position management, cash tracking, and portfolio state.
"""

import pytest
import pandas as pd
from core.portfolio.portfolio import Portfolio, Position


class TestPosition:
    """Test suite for Position dataclass."""

    def test_position_creation(self):
        """Test Position object creation with valid data."""
        pos = Position(
            ticker='AAPL',
            shares=100,
            entry_price=150.0,
            entry_date=pd.Timestamp('2023-01-01'),
            current_price=155.0
        )
        
        assert pos.ticker == 'AAPL'
        assert pos.shares == 100
        assert pos.entry_price == 150.0
        assert pos.current_price == 155.0

    def test_market_value_calculation(self):
        """Test market value calculation (shares * current_price)."""
        pos = Position('AAPL', 100, 150.0, pd.Timestamp('2023-01-01'), 155.0)
        assert pos.market_value == 15500.0

    def test_cost_basis_calculation(self):
        """Test cost basis calculation (shares * entry_price)."""
        pos = Position('AAPL', 100, 150.0, pd.Timestamp('2023-01-01'), 155.0)
        assert pos.cost_basis == 15000.0

    def test_pnl_positive(self):
        """Test P&L calculation with profit."""
        pos = Position('AAPL', 100, 150.0, pd.Timestamp('2023-01-01'), 155.0)
        assert pos.pnl == 500.0  # (155-150) * 100

    def test_pnl_negative(self):
        """Test P&L calculation with loss."""
        pos = Position('AAPL', 100, 150.0, pd.Timestamp('2023-01-01'), 145.0)
        assert pos.pnl == -500.0  # (145-150) * 100

    def test_pnl_pct_positive(self):
        """Test percentage P&L with profit."""
        pos = Position('AAPL', 100, 150.0, pd.Timestamp('2023-01-01'), 165.0)
        assert pytest.approx(pos.pnl_pct, rel=1e-4) == 0.10  # 10% gain

    def test_pnl_pct_negative(self):
        """Test percentage P&L with loss."""
        pos = Position('AAPL', 100, 150.0, pd.Timestamp('2023-01-01'), 135.0)
        assert pytest.approx(pos.pnl_pct, rel=1e-4) == -0.10  # 10% loss

    def test_pnl_pct_zero_entry_price(self):
        """Test percentage P&L with zero entry price edge case."""
        pos = Position('AAPL', 100, 0.0, pd.Timestamp('2023-01-01'), 155.0)
        assert pos.pnl_pct == 0.0  # Avoid division by zero

    def test_position_repr(self):
        """Test string representation."""
        pos = Position('AAPL', 100, 150.0, pd.Timestamp('2023-01-01'), 155.0)
        repr_str = repr(pos)
        assert 'AAPL' in repr_str
        assert '100' in repr_str


class TestPortfolioInitialization:
    """Test suite for Portfolio initialization."""

    def test_portfolio_creation(self):
        """Test Portfolio creation with initial capital."""
        portfolio = Portfolio(initial_capital=100000)
        
        assert portfolio.initial_capital == 100000
        assert portfolio.cash == 100000
        assert len(portfolio.positions) == 0
        assert len(portfolio.closed_positions) == 0

    def test_portfolio_total_value_initial(self):
        """Test initial total value equals cash."""
        portfolio = Portfolio(initial_capital=50000)
        assert portfolio.total_value == 50000


class TestPortfolioProperties:
    """Test suite for Portfolio properties."""

    def test_total_value_with_positions(self):
        """Test total value = cash + positions."""
        portfolio = Portfolio(initial_capital=100000)
        portfolio.open_position('AAPL', 100, 150.0, pd.Timestamp('2023-01-01'))
        portfolio.update_prices({'AAPL': 155.0})
        
        # Cash = 100000 - 15000 = 85000
        # Positions = 100 * 155 = 15500
        # Total = 100500
        assert portfolio.total_value == 100500

    def test_positions_value(self):
        """Test positions_value property."""
        portfolio = Portfolio(initial_capital=100000)
        portfolio.open_position('AAPL', 100, 150.0, pd.Timestamp('2023-01-01'))
        portfolio.update_prices({'AAPL': 160.0})
        
        assert portfolio.positions_value == 16000

    def test_invested_value(self):
        """Test invested_value (cost basis of positions)."""
        portfolio = Portfolio(initial_capital=100000)
        portfolio.open_position('AAPL', 100, 150.0, pd.Timestamp('2023-01-01'))
        
        assert portfolio.invested_value == 15000

    def test_unrealized_pnl(self):
        """Test unrealized P&L calculation."""
        portfolio = Portfolio(initial_capital=100000)
        portfolio.open_position('AAPL', 100, 150.0, pd.Timestamp('2023-01-01'))
        portfolio.update_prices({'AAPL': 160.0})
        
        assert portfolio.unrealized_pnl == 1000

    def test_realized_pnl_no_trades(self):
        """Test realized P&L with no closed positions."""
        portfolio = Portfolio(initial_capital=100000)
        assert portfolio.realized_pnl == 0.0

    def test_realized_pnl_with_trades(self):
        """Test realized P&L with closed positions."""
        portfolio = Portfolio(initial_capital=100000)
        portfolio.open_position('AAPL', 100, 150.0, pd.Timestamp('2023-01-01'))
        portfolio.close_position('AAPL', 160.0, pd.Timestamp('2023-02-01'))
        
        assert portfolio.realized_pnl == 1000

    def test_total_pnl(self):
        """Test total P&L (realized + unrealized)."""
        portfolio = Portfolio(initial_capital=100000)
        
        # Open and close AAPL for +1000 realized
        portfolio.open_position('AAPL', 100, 150.0, pd.Timestamp('2023-01-01'))
        portfolio.close_position('AAPL', 160.0, pd.Timestamp('2023-02-01'))
        
        # Open MSFT with +500 unrealized
        portfolio.open_position('MSFT', 50, 200.0, pd.Timestamp('2023-03-01'))
        portfolio.update_prices({'MSFT': 210.0})
        
        assert portfolio.total_pnl == 1500


class TestPortfolioPositionManagement:
    """Test suite for opening/closing positions."""

    def test_open_position_success(self):
        """Test successfully opening a position."""
        portfolio = Portfolio(initial_capital=100000)
        portfolio.open_position('AAPL', 100, 150.0, pd.Timestamp('2023-01-01'))
        
        assert portfolio.has_position('AAPL')
        assert portfolio.cash == 85000
        
        pos = portfolio.get_position('AAPL')
        assert pos.shares == 100
        assert pos.entry_price == 150.0

    def test_open_position_insufficient_cash(self):
        """Test opening position with insufficient cash."""
        portfolio = Portfolio(initial_capital=10000)
        
        with pytest.raises(ValueError, match="Insufficient cash"):
            portfolio.open_position('AAPL', 100, 150.0, pd.Timestamp('2023-01-01'))

    def test_open_position_already_exists(self):
        """Test opening position when one already exists."""
        portfolio = Portfolio(initial_capital=100000)
        portfolio.open_position('AAPL', 100, 150.0, pd.Timestamp('2023-01-01'))
        
        with pytest.raises(ValueError, match="already exists"):
            portfolio.open_position('AAPL', 50, 155.0, pd.Timestamp('2023-01-02'))

    def test_close_position_success(self):
        """Test successfully closing a position."""
        portfolio = Portfolio(initial_capital=100000)
        portfolio.open_position('AAPL', 100, 150.0, pd.Timestamp('2023-01-01'))
        proceeds = portfolio.close_position('AAPL', 160.0, pd.Timestamp('2023-02-01'))
        
        assert proceeds == 16000
        assert not portfolio.has_position('AAPL')
        assert portfolio.cash == 85000 + 16000
        assert len(portfolio.closed_positions) == 1

    def test_close_position_not_exists(self):
        """Test closing position that doesn't exist."""
        portfolio = Portfolio(initial_capital=100000)
        
        with pytest.raises(ValueError, match="No position"):
            portfolio.close_position('AAPL', 160.0, pd.Timestamp('2023-02-01'))

    def test_closed_position_record(self):
        """Test that closed position is properly recorded."""
        portfolio = Portfolio(initial_capital=100000)
        portfolio.open_position('AAPL', 100, 150.0, pd.Timestamp('2023-01-01'))
        portfolio.close_position('AAPL', 160.0, pd.Timestamp('2023-02-15'))
        
        trade = portfolio.closed_positions[0]
        assert trade['ticker'] == 'AAPL'
        assert trade['entry_price'] == 150.0
        assert trade['exit_price'] == 160.0
        assert trade['pnl'] == 1000
        assert trade['return'] == pytest.approx(0.0667, rel=1e-2)
        assert trade['hold_days'] == 45

    def test_has_position(self):
        """Test has_position method."""
        portfolio = Portfolio(initial_capital=100000)
        
        assert not portfolio.has_position('AAPL')
        portfolio.open_position('AAPL', 100, 150.0, pd.Timestamp('2023-01-01'))
        assert portfolio.has_position('AAPL')

    def test_get_position_exists(self):
        """Test get_position when position exists."""
        portfolio = Portfolio(initial_capital=100000)
        portfolio.open_position('AAPL', 100, 150.0, pd.Timestamp('2023-01-01'))
        
        pos = portfolio.get_position('AAPL')
        assert pos is not None
        assert pos.ticker == 'AAPL'

    def test_get_position_not_exists(self):
        """Test get_position when position doesn't exist."""
        portfolio = Portfolio(initial_capital=100000)
        
        pos = portfolio.get_position('AAPL')
        assert pos is None


class TestPortfolioPriceUpdates:
    """Test suite for price update functionality."""

    def test_update_prices_single_position(self):
        """Test updating price for single position."""
        portfolio = Portfolio(initial_capital=100000)
        portfolio.open_position('AAPL', 100, 150.0, pd.Timestamp('2023-01-01'))
        
        portfolio.update_prices({'AAPL': 160.0})
        
        pos = portfolio.get_position('AAPL')
        assert pos.current_price == 160.0

    def test_update_prices_multiple_positions(self):
        """Test updating prices for multiple positions."""
        portfolio = Portfolio(initial_capital=100000)
        portfolio.open_position('AAPL', 100, 150.0, pd.Timestamp('2023-01-01'))
        portfolio.open_position('MSFT', 50, 200.0, pd.Timestamp('2023-01-01'))
        
        portfolio.update_prices({'AAPL': 160.0, 'MSFT': 210.0})
        
        assert portfolio.get_position('AAPL').current_price == 160.0
        assert portfolio.get_position('MSFT').current_price == 210.0

    def test_update_prices_missing_ticker(self):
        """Test updating prices when ticker not in positions."""
        portfolio = Portfolio(initial_capital=100000)
        portfolio.open_position('AAPL', 100, 150.0, pd.Timestamp('2023-01-01'))
        
        # Should not crash if we provide extra tickers
        portfolio.update_prices({'AAPL': 160.0, 'GOOGL': 2800.0})
        
        assert portfolio.get_position('AAPL').current_price == 160.0

    def test_update_prices_partial_update(self):
        """Test updating prices when not all positions provided."""
        portfolio = Portfolio(initial_capital=100000)
        portfolio.open_position('AAPL', 100, 150.0, pd.Timestamp('2023-01-01'))
        portfolio.open_position('MSFT', 50, 200.0, pd.Timestamp('2023-01-01'))
        
        # Only update AAPL
        portfolio.update_prices({'AAPL': 160.0})
        
        assert portfolio.get_position('AAPL').current_price == 160.0
        assert portfolio.get_position('MSFT').current_price == 200.0  # Unchanged


class TestPortfolioEquityCurve:
    """Test suite for equity curve generation."""

    def test_get_equity_curve_point(self):
        """Test getting portfolio snapshot at a point in time."""
        portfolio = Portfolio(initial_capital=100000)
        portfolio.open_position('AAPL', 100, 150.0, pd.Timestamp('2023-01-01'))
        portfolio.update_prices({'AAPL': 160.0})
        
        snapshot = portfolio.get_equity_curve_point(pd.Timestamp('2023-01-15'))
        
        assert 'Cash' in snapshot
        assert 'PositionsValue' in snapshot
        assert 'TotalValue' in snapshot
        assert snapshot['Cash'] == 85000
        assert snapshot['PositionsValue'] == 16000
        assert snapshot['TotalValue'] == 101000


class TestPortfolioEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_multiple_open_close_cycles(self):
        """Test opening and closing positions multiple times."""
        portfolio = Portfolio(initial_capital=100000)
        
        # First cycle
        portfolio.open_position('AAPL', 100, 150.0, pd.Timestamp('2023-01-01'))
        portfolio.close_position('AAPL', 160.0, pd.Timestamp('2023-02-01'))
        
        # Second cycle
        portfolio.open_position('AAPL', 50, 155.0, pd.Timestamp('2023-03-01'))
        portfolio.close_position('AAPL', 165.0, pd.Timestamp('2023-04-01'))
        
        assert len(portfolio.closed_positions) == 2
        assert portfolio.realized_pnl == 1500  # 1000 + 500

    def test_zero_shares_position(self):
        """Test position with zero shares (edge case)."""
        portfolio = Portfolio(initial_capital=100000)
        portfolio.open_position('AAPL', 0, 150.0, pd.Timestamp('2023-01-01'))
        
        pos = portfolio.get_position('AAPL')
        assert pos.market_value == 0
        assert pos.pnl == 0

    def test_negative_price_change(self):
        """Test position losing all value."""
        portfolio = Portfolio(initial_capital=100000)
        portfolio.open_position('AAPL', 100, 150.0, pd.Timestamp('2023-01-01'))
        portfolio.update_prices({'AAPL': 0.0})
        
        assert portfolio.unrealized_pnl == -15000
