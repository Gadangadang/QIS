"""
ExecutionEngine class - Simulates realistic order execution.

Responsibilities:
- Apply transaction costs
- Model slippage
- Simulate market impact
- Determine if trades should execute
"""

from dataclasses import dataclass
from typing import Tuple


@dataclass
class ExecutionConfig:
    """Execution cost configuration."""
    transaction_cost_bps: float = 3.0  # 3 basis points (0.03%)
    slippage_bps: float = 2.0  # 2 basis points slippage
    min_trade_value: float = 100.0  # Don't trade less than $100
    market_impact_factor: float = 0.0  # Additional cost for large trades (future)
    

class ExecutionEngine:
    """
    Simulates realistic order execution with costs.
    
    Example:
        config = ExecutionConfig(transaction_cost_bps=3.0, slippage_bps=2.0)
        engine = ExecutionEngine(config)
        
        fill_price, cost = engine.execute_buy('ES', shares=10, market_price=4500.0)
        # Returns slightly higher fill price and transaction cost
    """
    
    def __init__(self, config: ExecutionConfig):
        """
        Initialize execution engine with configuration.
        
        Args:
            config: ExecutionConfig object
        """
        self.config = config
    
    def execute_buy(
        self, 
        ticker: str, 
        shares: float, 
        market_price: float
    ) -> Tuple[float, float]:
        """
        Execute buy order with realistic costs.
        
        Args:
            ticker: Asset ticker
            shares: Number of shares to buy
            market_price: Current market price
            
        Returns:
            Tuple of (fill_price, transaction_cost)
            - fill_price: Actual price paid per share (includes slippage)
            - transaction_cost: Commission/fees in dollars
        """
        # Slippage: we pay slightly more when buying
        slippage = market_price * (self.config.slippage_bps / 10000)
        fill_price = market_price + slippage
        
        # Transaction cost (commission/fees)
        trade_value = shares * fill_price
        transaction_cost = trade_value * (self.config.transaction_cost_bps / 10000)
        
        return fill_price, transaction_cost
    
    def execute_sell(
        self, 
        ticker: str, 
        shares: float, 
        market_price: float
    ) -> Tuple[float, float]:
        """
        Execute sell order with realistic costs.
        
        Args:
            ticker: Asset ticker
            shares: Number of shares to sell
            market_price: Current market price
            
        Returns:
            Tuple of (fill_price, transaction_cost)
            - fill_price: Actual price received per share (includes slippage)
            - transaction_cost: Commission/fees in dollars
        """
        # Slippage: we receive slightly less when selling
        slippage = market_price * (self.config.slippage_bps / 10000)
        fill_price = market_price - slippage
        
        # Transaction cost (commission/fees)
        trade_value = shares * fill_price
        transaction_cost = trade_value * (self.config.transaction_cost_bps / 10000)
        
        return fill_price, transaction_cost
    
    def should_execute(self, shares: float, price: float) -> bool:
        """
        Check if trade is large enough to execute.
        
        Args:
            shares: Number of shares
            price: Price per share
            
        Returns:
            True if trade value exceeds minimum, False otherwise
        """
        trade_value = shares * price
        return trade_value >= self.config.min_trade_value
    
    def calculate_market_impact(
        self, 
        shares: float, 
        price: float,
        avg_daily_volume: float
    ) -> float:
        """
        Calculate additional market impact cost for large trades.
        
        (Future enhancement - currently returns 0)
        
        Args:
            shares: Number of shares
            price: Current price
            avg_daily_volume: Average daily trading volume
            
        Returns:
            Additional cost per share due to market impact
        """
        if avg_daily_volume == 0:
            return 0.0
        
        # Simple square-root model: impact ~ sqrt(trade_size / daily_volume)
        trade_fraction = (shares * price) / (avg_daily_volume * price)
        impact = self.config.market_impact_factor * (trade_fraction ** 0.5) * price
        
        return impact
