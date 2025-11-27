"""
Portfolio class - Tracks positions, cash, and portfolio value.

Responsibilities:
- Maintain current positions
- Track cash balance
- Calculate portfolio value
- Simple state updates (no business logic)
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, List
import pandas as pd


@dataclass
class Position:
    """Represents a single position in a portfolio."""
    ticker: str
    shares: float
    entry_price: float
    entry_date: pd.Timestamp
    current_price: float
    
    @property
    def market_value(self) -> float:
        """Current market value of position."""
        return self.shares * self.current_price
    
    @property
    def cost_basis(self) -> float:
        """Original cost of position."""
        return self.shares * self.entry_price
    
    @property
    def pnl(self) -> float:
        """Profit/loss in dollars."""
        return (self.current_price - self.entry_price) * self.shares
    
    @property
    def pnl_pct(self) -> float:
        """Profit/loss as percentage."""
        if self.entry_price == 0:
            return 0.0
        return (self.current_price / self.entry_price - 1)
    
    def __repr__(self) -> str:
        return f"Position({self.ticker}: {self.shares:.0f} @ ${self.current_price:.2f}, P&L: ${self.pnl:,.2f})"


class Portfolio:
    """
    Tracks portfolio state: positions, cash, equity.
    
    Example:
        portfolio = Portfolio(initial_capital=100000)
        portfolio.open_position('ES', shares=10, price=4500.0, date=pd.Timestamp('2024-01-01'))
        portfolio.update_prices({'ES': 4550.0})
        print(f"Total value: ${portfolio.total_value:,.2f}")
        portfolio.close_position('ES', price=4550.0, date=pd.Timestamp('2024-01-02'))
    """
    
    def __init__(self, initial_capital: float):
        """
        Initialize portfolio with starting capital.
        
        Args:
            initial_capital: Starting cash amount
        """
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.positions: Dict[str, Position] = {}
        self.closed_positions: List[Dict] = []  # Historical trades
        
    @property
    def total_value(self) -> float:
        """Total portfolio value = cash + positions."""
        positions_value = sum(pos.market_value for pos in self.positions.values())
        return self.cash + positions_value
    
    @property
    def positions_value(self) -> float:
        """Total value of all open positions."""
        return sum(pos.market_value for pos in self.positions.values())
    
    @property
    def invested_value(self) -> float:
        """Total cost basis of all positions."""
        return sum(pos.cost_basis for pos in self.positions.values())
    
    @property
    def unrealized_pnl(self) -> float:
        """Total unrealized profit/loss."""
        return sum(pos.pnl for pos in self.positions.values())
    
    @property
    def realized_pnl(self) -> float:
        """Total realized profit/loss from closed trades."""
        return sum(trade['pnl'] for trade in self.closed_positions)
    
    @property
    def total_pnl(self) -> float:
        """Total P&L (realized + unrealized)."""
        return self.realized_pnl + self.unrealized_pnl
    
    def has_position(self, ticker: str) -> bool:
        """Check if we have a position in this ticker."""
        return ticker in self.positions
    
    def get_position(self, ticker: str) -> Optional[Position]:
        """Get position for ticker, or None if not held."""
        return self.positions.get(ticker)
    
    def open_position(self, ticker: str, shares: float, price: float, date: pd.Timestamp):
        """
        Open a new position.
        
        Args:
            ticker: Asset ticker
            shares: Number of shares to buy
            price: Entry price per share
            date: Entry date
            
        Raises:
            ValueError: If insufficient cash or position already exists
        """
        if ticker in self.positions:
            raise ValueError(f"Position in {ticker} already exists. Close it first.")
        
        cost = shares * price
        if cost > self.cash:
            raise ValueError(
                f"Insufficient cash: need ${cost:,.2f}, have ${self.cash:,.2f}"
            )
        
        self.cash -= cost
        self.positions[ticker] = Position(
            ticker=ticker,
            shares=shares,
            entry_price=price,
            entry_date=date,
            current_price=price
        )
    
    def close_position(self, ticker: str, price: float, date: pd.Timestamp) -> float:
        """
        Close an existing position.
        
        Args:
            ticker: Asset ticker
            price: Exit price per share
            date: Exit date
            
        Returns:
            Proceeds from sale (before costs)
            
        Raises:
            ValueError: If no position exists
        """
        if ticker not in self.positions:
            raise ValueError(f"No position in {ticker} to close")
        
        position = self.positions.pop(ticker)
        proceeds = position.shares * price
        self.cash += proceeds
        
        # Record closed trade
        trade = {
            'ticker': ticker,
            'entry_date': position.entry_date,
            'exit_date': date,
            'entry_price': position.entry_price,
            'exit_price': price,
            'shares': position.shares,
            'pnl': proceeds - position.cost_basis,
            'return': price / position.entry_price - 1,
            'hold_days': (date - position.entry_date).days
        }
        self.closed_positions.append(trade)
        
        return proceeds
    
    def update_prices(self, prices: Dict[str, float]):
        """
        Update current prices for all positions.
        
        Args:
            prices: Dict of ticker -> current_price
        """
        for ticker, position in self.positions.items():
            if ticker in prices:
                position.current_price = prices[ticker]
    
    def get_equity_curve_point(self, date: pd.Timestamp) -> Dict:
        """
        Get portfolio state snapshot at this point in time.
        
        Args:
            date: Current date
            
        Returns:
            Dict with Date, Cash, PositionsValue, TotalValue
        """
        return {
            'Date': date,
            'Cash': self.cash,
            'PositionsValue': self.positions_value,
            'TotalValue': self.total_value,
            'UnrealizedPnL': self.unrealized_pnl
        }
    
    def get_allocation(self) -> Dict[str, float]:
        """
        Get portfolio allocation as percentages.
        
        Returns:
            Dict of ticker -> allocation percentage (0.0 to 1.0)
        """
        total = self.total_value
        if total == 0:
            return {}
        
        allocation = {
            ticker: pos.market_value / total 
            for ticker, pos in self.positions.items()
        }
        allocation['Cash'] = self.cash / total
        return allocation
    
    def get_summary(self) -> str:
        """Get human-readable portfolio summary."""
        lines = [
            f"\n{'='*60}",
            f"PORTFOLIO SUMMARY",
            f"{'='*60}",
            f"Total Value:      ${self.total_value:>15,.2f}",
            f"Cash:             ${self.cash:>15,.2f} ({self.cash/self.total_value*100:.1f}%)",
            f"Positions Value:  ${self.positions_value:>15,.2f}",
            f"Unrealized P&L:   ${self.unrealized_pnl:>15,.2f}",
            f"Realized P&L:     ${self.realized_pnl:>15,.2f}",
            f"Total P&L:        ${self.total_pnl:>15,.2f}",
            f"\nOpen Positions: {len(self.positions)}",
        ]
        
        if self.positions:
            lines.append(f"{'Ticker':<8} {'Shares':>8} {'Entry':>10} {'Current':>10} {'P&L':>12} {'Return':>10}")
            lines.append("-" * 60)
            for ticker, pos in self.positions.items():
                lines.append(
                    f"{ticker:<8} {pos.shares:>8.0f} "
                    f"${pos.entry_price:>9,.2f} ${pos.current_price:>9,.2f} "
                    f"${pos.pnl:>11,.2f} {pos.pnl_pct:>9.1%}"
                )
        
        lines.append(f"\nClosed Trades: {len(self.closed_positions)}")
        lines.append(f"{'='*60}\n")
        
        return '\n'.join(lines)
