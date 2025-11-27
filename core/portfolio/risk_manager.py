"""
RiskManager class - Enforces risk rules and calculates position sizes.

Responsibilities:
- Calculate position sizes based on risk parameters
- Check risk limits (stops, concentration)
- Monitor portfolio-level risk
"""

from dataclasses import dataclass
from typing import Optional
import numpy as np
import pandas as pd


@dataclass
class RiskConfig:
    """Risk management configuration."""
    risk_per_trade: float = 0.02  # 2% risk per trade
    max_position_size: float = 0.20  # Max 20% in one position
    max_portfolio_leverage: float = 1.0  # No leverage by default
    stop_loss_pct: Optional[float] = None  # Stop loss percentage (e.g., 0.10 = 10%)
    take_profit_pct: Optional[float] = None  # Take profit percentage
    max_correlation_exposure: float = 0.50  # Max 50% in correlated assets
    min_trade_value: float = 100.0  # Minimum trade size
    

class RiskManager:
    """
    Enforces risk rules and calculates position sizes.
    
    Example:
        config = RiskConfig(risk_per_trade=0.02, max_position_size=0.20, stop_loss_pct=0.10)
        risk_mgr = RiskManager(config)
        
        shares = risk_mgr.calculate_position_size(
            ticker='ES',
            signal=1.0,
            current_price=4500.0,
            portfolio_value=100000
        )
        
        should_exit = risk_mgr.check_stop_loss(position)
    """
    
    def __init__(self, config: RiskConfig):
        """
        Initialize risk manager with configuration.
        
        Args:
            config: RiskConfig object with risk parameters
        """
        self.config = config
    
    def calculate_position_size(
        self, 
        ticker: str,
        signal: float,
        current_price: float,
        portfolio_value: float,
        volatility: Optional[float] = None
    ) -> float:
        """
        Calculate number of shares to buy based on risk rules.
        
        Uses the smaller of:
        1. Max position size limit (% of portfolio)
        2. Risk-based sizing (based on stop loss if configured)
        
        Args:
            ticker: Asset ticker
            signal: Signal strength (0 to 1, where 1 = full position)
            current_price: Current price per share
            portfolio_value: Current portfolio value
            volatility: Optional volatility measure (annualized std)
            
        Returns:
            Number of shares to buy (integer)
        """
        if portfolio_value <= 0 or current_price <= 0:
            return 0
        
        # Method 1: Fixed percentage of portfolio
        max_position_value = portfolio_value * self.config.max_position_size
        shares_from_size_limit = max_position_value / current_price
        
        # Method 2: Risk-based sizing (if we have stop loss configured)
        if self.config.stop_loss_pct and self.config.stop_loss_pct > 0:
            # Size position based on how much we're willing to lose
            risk_amount = portfolio_value * self.config.risk_per_trade
            loss_per_share = current_price * self.config.stop_loss_pct
            
            if loss_per_share > 0:
                shares_from_risk = risk_amount / loss_per_share
                # Use the smaller of the two
                shares = min(shares_from_size_limit, shares_from_risk)
            else:
                shares = shares_from_size_limit
        else:
            shares = shares_from_size_limit
        
        # Scale by signal strength (if signal is 0.5, use half position)
        shares *= abs(signal)
        
        # Round down to whole shares
        shares = int(shares)
        
        # Check minimum trade value
        if shares * current_price < self.config.min_trade_value:
            return 0
        
        return shares
    
    def check_stop_loss(self, position) -> bool:
        """
        Check if position should be exited due to stop loss.
        
        Args:
            position: Position object
            
        Returns:
            True if should exit, False otherwise
        """
        if self.config.stop_loss_pct is None:
            return False
        return position.pnl_pct <= -self.config.stop_loss_pct
    
    def check_take_profit(self, position) -> bool:
        """
        Check if position should be exited due to take profit.
        
        Args:
            position: Position object
            
        Returns:
            True if should exit, False otherwise
        """
        if self.config.take_profit_pct is None:
            return False
        return position.pnl_pct >= self.config.take_profit_pct
    
    def check_concentration_limit(
        self, 
        ticker: str, 
        new_position_value: float,
        portfolio_value: float
    ) -> bool:
        """
        Check if new position would violate concentration limits.
        
        Args:
            ticker: Asset ticker
            new_position_value: Value of proposed position
            portfolio_value: Current portfolio value
            
        Returns:
            True if within limits, False if would violate
        """
        if portfolio_value == 0:
            return True
        
        concentration = new_position_value / portfolio_value
        return concentration <= self.config.max_position_size
    
    def calculate_var(
        self, 
        returns: pd.Series, 
        confidence: float = 0.95
    ) -> float:
        """
        Calculate Value at Risk (VaR).
        
        Args:
            returns: Series of portfolio returns
            confidence: Confidence level (e.g., 0.95 for 95% VaR)
            
        Returns:
            VaR as a negative percentage
        """
        if len(returns) == 0:
            return 0.0
        return np.percentile(returns, (1 - confidence) * 100)
    
    def calculate_cvar(
        self,
        returns: pd.Series,
        confidence: float = 0.95
    ) -> float:
        """
        Calculate Conditional Value at Risk (CVaR / Expected Shortfall).
        
        Args:
            returns: Series of portfolio returns
            confidence: Confidence level
            
        Returns:
            Average return in worst (1-confidence)% of cases
        """
        if len(returns) == 0:
            return 0.0
        
        var_threshold = self.calculate_var(returns, confidence)
        tail_returns = returns[returns <= var_threshold]
        
        if len(tail_returns) == 0:
            return var_threshold
        
        return tail_returns.mean()
    
    def calculate_kelly_fraction(
        self,
        win_rate: float,
        avg_win: float,
        avg_loss: float
    ) -> float:
        """
        Calculate Kelly Criterion for position sizing.
        
        Args:
            win_rate: Probability of winning trade (0 to 1)
            avg_win: Average winning trade return
            avg_loss: Average losing trade return (positive number)
            
        Returns:
            Kelly fraction (typically use half-Kelly in practice)
        """
        if avg_loss == 0:
            return 0.0
        
        win_loss_ratio = avg_win / avg_loss
        kelly = win_rate - ((1 - win_rate) / win_loss_ratio)
        
        # Return half-Kelly for safety (common practice)
        return max(0, kelly * 0.5)
